"""ZERO-2 stage 4: the trade-off map around full LoRA on Qwen3.5-9B.

Protocol: ZERO2_STAGE4_PROTOCOL_20261120.md. Soup research/zero2-fixes (SOUP_COMMIT below).

Variants (all LoRA on the 10 projection types, NF4, kbit_keep_16bit, no cap):
  full                    all 32 layers, r 16 (the reference)
  top24 / top16 / top12   only the top k layers (faster: the backward stops lower)
  r32                     all layers, r 32 / alpha 64 (more capacity, slower)
  imp                     all layers; layers 0-7 at r 64 / alpha 128, the rest r 16
  dora                    all layers, r 16, DoRA

Part S (speed, GSM8K's first 96 rows, 24 steps, A-B-B-A against full in each block):
  t24, t16, t12, r32, imp, dora (no cap); top16 also at 8 GiB streamed and 4 GiB L2L.
Part Q (quality, 300 steps): gsm8k and arc with seeds 0 and 1, pubmedqa with seed 0; base eval,
then every variant trained and evaluated.

Resumable: an arm whose json already exists is not run again (whatever its status).

    python zero2_stage4.py            # everything, in order
    python zero2_stage4.py S          # one part: S, gsm8k, arc, pubmedqa
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path(os.environ.get("ZERO_OUT", "zero2_stage4_results")).resolve()
WORK = OUT / "work"
REPO = "Qwen/Qwen3.5-9B"
REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL = os.environ.get("ZERO_MODEL_DIR", str(Path.home() / "qwen35_9b"))
SOUP_COMMIT = "b3bca25cda116bf843b4e76aa59f228e4cbbc0ed"
ARM_TIMEOUT_S = 90 * 60
K = 4
MAX_LEN = 1024
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]

#: dataset -> (hf repo, revision, file, train rows)
DATA = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
              "main/train-00000-of-00001.parquet"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453",
            ["ARC-Challenge/train-00000-of-00001.parquet", "ARC-Easy/train-00000-of-00001.parquet"]),
    "pubmedqa": ("qiaojin/PubMedQA", "9001f2853fb87cab8d220904e0de81ac6973b318",
                 "pqa_artificial/train-00000-of-00001.parquet"),
}
#: dataset -> (lm-eval task, limit or None, primary metric, eval batch size)
EVAL = {
    "gsm8k": ("gsm8k", 200, "exact_match,strict-match", 16),
    "arc": ("arc_challenge", None, "acc_norm,none", 16),
    # stage 3: batch 16 x long abstracts x 248k vocabulary OOM'd in lm-eval's log_softmax
    "pubmedqa": ("zero2_pubmedqa", None, "acc,none", 2),
}
SEEDS = {"gsm8k": [0, 1], "arc": [0, 1], "pubmedqa": [0]}
Q_STEPS = 300
Q_ROWS = Q_STEPS * K
S_STEPS = 24
S_ROWS = S_STEPS * K

KEEP16 = ["kbit_keep_16bit: true"]
STREAM8 = ["stream_layers: true", "stream_source: ram", "stream_chunked_ce: true"]
L2L4 = STREAM8 + ["l2l: true"]


def top(k: int) -> list:
    return list(range(32 - k, 32))


#: variant -> (training lines, lora lines; the default lora is r 16 / alpha 32)
VARIANTS = {
    "full": ([], []),
    "top24": (["lora_layers:", "  method: fixed", f"  layers: {top(24)}"], []),
    "top16": (["lora_layers:", "  method: fixed", f"  layers: {top(16)}"], []),
    "top12": (["lora_layers:", "  method: fixed", f"  layers: {top(12)}"], []),
    "r32": ([], ["r: 32", "alpha: 64"]),
    "imp": ([], ["rank_pattern:", "  'layers\\.[0-7]\\..*': 64", "alpha_pattern:",
                 "  'layers\\.[0-7]\\..*': 128"]),
    "dora": ([], ["use_dora: true"]),
}
Q_VARIANTS = list(VARIANTS)
#: part S: (block, cap GiB, base training lines, variant)
S_BLOCKS = [("t24", 0, KEEP16, "top24"), ("t16", 0, KEEP16, "top16"), ("t12", 0, KEEP16, "top12"),
            ("r32", 0, KEEP16, "r32"), ("imp", 0, KEEP16, "imp"), ("dora", 0, KEEP16, "dora"),
            ("s16", 8, STREAM8, "top16"), ("q16", 4, L2L4, "top16")]

PUBMEDQA_TASK = """task: zero2_pubmedqa
dataset_path: qiaojin/PubMedQA
dataset_name: pqa_labeled
dataset_kwargs:
  revision: 9001f2853fb87cab8d220904e0de81ac6973b318
output_type: multiple_choice
test_split: train
doc_to_text: "Abstract: {{context.contexts | join(' ')}}\\nQuestion: {{question}}\\nAnswer:"
doc_to_target: "{{['yes', 'no', 'maybe'].index(final_decision)}}"
doc_to_choice: ["yes", "no", "maybe"]
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""


def write(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    print(f"[z4] wrote {name}.json status={obj.get('status')}", flush=True)


def done(name: str):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def arm_env() -> None:
    if done("env"):
        return
    write("env", {"status": "OK", "python": sys.version,
                  "nvidia": sh("nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv"),
                  "mem": sh("free -b"), "nproc": os.cpu_count(), "cpu": sh("lscpu | grep 'Model name'"),
                  "pip": sh("pip list 2>/dev/null | grep -iE '^(soup-cli|torch|transformers|peft|trl|datasets|"
                            "bitsandbytes|accelerate|flash-linear-attention|fla-core|lm.eval|triton|liger.kernel) '")})


def download() -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download
    res = done("download")
    if res and res.get("status") == "OK":
        return res
    res, t0 = {"status": "RUNNING", "files": {}}, time.time()
    try:
        snapshot_download(REPO, revision=REV, local_dir=MODEL,
                          allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt"], max_workers=8)
        for name, (repo, rev, files) in DATA.items():
            files = files if isinstance(files, list) else [files]
            res["files"][name] = [hf_hub_download(repo, f, repo_type="dataset", revision=rev) for f in files]
        res["status"] = "OK"
    except Exception as e:  # noqa: BLE001
        res.update(status="ERROR", error=f"{type(e).__name__}: {str(e)[:800]}")
    res["wall_s"] = time.time() - t0
    write("download", res)
    return res


def rows_for(name: str, paths: list, n: int) -> list:
    """Alpaca rows in file order (ARC: Challenge then Easy)."""
    import pandas as pd
    out = []
    if name == "gsm8k":
        df = pd.read_parquet(paths[0])
        out = [{"instruction": q, "input": "", "output": a} for q, a in zip(df["question"], df["answer"])]
    elif name == "arc":
        for p in paths:
            df = pd.read_parquet(p)
            for q, ch, key in zip(df["question"], df["choices"], df["answerKey"]):
                labels, texts = list(ch["label"]), list(ch["text"])
                if key in labels:
                    out.append({"instruction": q, "input": "", "output": texts[labels.index(key)]})
    elif name == "pubmedqa":
        df = pd.read_parquet(paths[0])
        for q, ctx, dec in zip(df["question"], df["context"], df["final_decision"]):
            abstract = " ".join(ctx["contexts"])
            out.append({"instruction": f"Abstract: {abstract}\nQuestion: {q}", "input": "",
                        "output": dec})
    elif name == "code":
        data = json.loads(Path(paths[0]).read_text())
        out = [{"instruction": d["instruction"], "input": d.get("input", ""), "output": d["output"]}
               for d in data]
    return out[:n]


def write_rows(tag: str, rows: list) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    dst = WORK / f"{tag}.jsonl"
    with open(dst, "w") as o:
        for r in rows:
            o.write(json.dumps(r) + "\n")
    return dst


def config(data: Path, out: Path, cap: int, extra: list, lora=(), seed: int = 0) -> str:
    lora = list(lora)
    lora_lines = lora if any(x.startswith("r:") for x in lora) else ["r: 16", "alpha: 32"] + lora
    lines = [
        f"base: {MODEL}", "task: sft", "modality: text", "data:", f"  train: {data}",
        "  format: alpaca", "  val_split: 0.0", f"  max_length: {MAX_LEN}", "training:",
        "  epochs: 1", "  lr: 1e-4", "  batch_size: 1", f"  gradient_accumulation_steps: {K}",
        f"  seed: {seed}", "  logging_steps: 1", "  save_steps: 100000",
        "  gradient_checkpointing: true", "  lora:",
    ] + [f"    {x}" for x in lora_lines] + [
        "    dropout: 0.0", f"    target_modules: [{', '.join(TARGETS)}]", "  quantization: 4bit",
    ] + [f"  {x}" for x in extra]
    if cap and any(x.startswith("stream_layers") for x in extra):
        lines += [f"  stream_vram_override: {cap * 10**9}", "  stream_vram_probe: true"]
    lines += [f"output: {out}"]
    return "\n".join(lines) + "\n"


LAUNCH = r'''
import atexit, json, os, sys, time
import torch
cap = float(os.environ.get("ZERO_CAP_GB", "0"))
if cap and torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cap * 2**30 / total, 0)
STEP_T = []
def _dump():
    if torch.cuda.is_available():
        with open(os.environ["ZERO_PEAK_FILE"], "w") as f:
            json.dump({"peak_allocated_gb": torch.cuda.max_memory_allocated(0) / 2**30,
                       "peak_reserved_gb": torch.cuda.max_memory_reserved(0) / 2**30,
                       "step_end_s": STEP_T}, f)
atexit.register(_dump)
import transformers
class _Timer(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        STEP_T.append(time.perf_counter())
_init = transformers.Trainer.__init__
def _patched(self, *a, **k):
    _init(self, *a, **k)
    self.add_callback(_Timer())
transformers.Trainer.__init__ = _patched
from soup_cli.cli import run
sys.argv = ["soup", "--verbose"] + sys.argv[1:]
run()
'''


def parse_log(text: str) -> dict:
    dicts = [dict(re.findall(r"'(\w+)': '?([^',}]*)'?", m)) for m in re.findall(r"\{'loss'[^}]*\}", text)]
    r = {"n_steps": len(dicts), "losses": [float(d["loss"]) for d in dicts if "loss" in d],
         "num_tokens": [float(d["num_tokens"]) for d in dicts if "num_tokens" in d]}
    for key, pat in (("layers_line", r"LoRA layers \([^\n]*"), ("host_line", r"embed_on_host:[^\n]*"),
                     ("liger_line", r"Liger Kernel[^\n]*"), ("l2l_line", r"L2L:[^\n]*")):
        m = re.search(pat, text)
        if m:
            r[key] = m.group(0).strip()[:400]
    if "embedding weight is on" in text:
        r["host_warning"] = True
    errs = re.findall(r"^Error: .*$", text, flags=re.MULTILINE)
    if errs:
        r["error_line"] = errs[-1][:1000]
    r["oom"] = "OutOfMemoryError" in text or "CUDA out of memory" in text
    return r


class Monitor:
    def __init__(self):
        self.peak_mib, self.min_avail_gb = 0, None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()
            if out and out[0].isdigit():
                self.peak_mib = max(self.peak_mib, int(out[0]))
            try:
                with open("/proc/meminfo") as f:
                    kb = int(next(x for x in f if x.startswith("MemAvailable")).split()[1])
                gb = kb / 2**20
                self.min_avail_gb = gb if self.min_avail_gb is None else min(self.min_avail_gb, gb)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)


def arm_train(tag: str, data: Path, cap: int, extra: list, steps: int, keep_out: bool = False,
              lora=(), seed: int = 0) -> dict:
    prev = done(tag)
    if prev:
        return prev
    out = WORK / f"out_{tag}"
    cfg = WORK / f"{tag}.yaml"
    cfg.write_text(config(data, out, cap, extra, lora, seed))
    launcher = WORK / "zero_launch.py"
    launcher.write_text(LAUNCH)
    peak = WORK / f"{tag}.peak.json"
    env = {**os.environ, "ZERO_PEAK_FILE": str(peak), "ZERO_CAP_GB": str(cap),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    env.pop("SOUP_NF4_TRITON", None)
    res = {"arm": tag, "cap_gib": cap, "extra": extra, "lora": list(lora), "seed": seed,
           "status": "RUNNING"}
    t0 = time.time()
    with Monitor() as mon:
        try:
            p = subprocess.run([sys.executable, str(launcher), "train", "--config", str(cfg), "-y"],
                               capture_output=True, text=True, timeout=ARM_TIMEOUT_S, env=env)
            text, code = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            text, code = (e.stdout or "") + (e.stderr or ""), "TIMEOUT"
            text = text.decode() if isinstance(text, bytes) else text
    (OUT / f"{tag}.log").write_text(text[-400_000:])
    res.update(parse_log(text), exit=code, wall_s=time.time() - t0, nvsmi_peak_mib=mon.peak_mib,
               host_min_available_gb=mon.min_avail_gb)
    if peak.exists():
        pk = json.loads(peak.read_text())
        ends = pk.pop("step_end_s", [])
        res["torch_peak"] = pk
        if len(ends) >= 4:
            gaps = [b - a for a, b in zip(ends, ends[1:])]
            res["steady_s_per_step"] = statistics.mean(gaps[2:])
            toks = res["num_tokens"]
            if len(toks) == len(ends):
                res["steady_tok_per_s"] = (toks[-1] - toks[2]) / (ends[-1] - ends[2])
    rec = out / "lora_layers.json"
    if rec.exists():
        res["lora_layers"] = json.loads(rec.read_text())
    res["status"] = ("TIMEOUT" if code == "TIMEOUT" else "OOM" if res["oom"] else
                     "OK" if res["n_steps"] == steps and code == 0 else "ERROR")
    write(tag, res)
    if not keep_out:
        shutil.rmtree(out, ignore_errors=True)
    return res


EVAL_CODE = r'''
import json, os, sys, time, torch, transformers
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
model_dir, adapter, task, limit, include, bs = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
                                                sys.argv[5], int(sys.argv[6]))
q = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = transformers.AutoTokenizer.from_pretrained(model_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                                          quantization_config=q, device_map={"": 0})
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
model.eval()
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
t0 = time.time()
r = simple_evaluate(model=lm, tasks=[task], limit=limit or None,
                    task_manager=TaskManager(include_path=include), confirm_run_unsafe_code=True)
print("ZERO_EVAL " + json.dumps({"results": r["results"][task], "eval_s": time.time() - t0,
                                  "n": r.get("n-samples", {}).get(task)}, default=str))
'''


def arm_eval(tag: str, dataset: str, adapter: str = "") -> dict:
    prev = done(tag)
    if prev:
        return prev
    task, limit, metric, bs = EVAL[dataset]
    include = WORK / "tasks"
    include.mkdir(parents=True, exist_ok=True)
    (include / "zero2_pubmedqa.yaml").write_text(PUBMEDQA_TASK)
    res = {"arm": tag, "dataset": dataset, "task": task, "limit": limit, "metric": metric, "batch": bs,
           "adapter": adapter or None, "status": "RUNNING"}
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", EVAL_CODE, MODEL, adapter, task, str(limit or 0),
                            str(include), str(bs)], capture_output=True, text=True,
                           timeout=ARM_TIMEOUT_S, env={**os.environ, "HF_ALLOW_CODE_EVAL": "1",
                                                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        text = p.stdout + p.stderr
        (OUT / f"{tag}.log").write_text(text[-200_000:])
        m = re.search(r"ZERO_EVAL (.*)", text)
        if m:
            got = json.loads(m.group(1))
            r = got["results"]
            se_key = metric.replace(",", "_stderr,", 1)
            res.update(status="OK", results=r, eval_s=got["eval_s"], n=got.get("n"),
                       score=r.get(metric), stderr=r.get(se_key))
        else:
            res.update(status="ERROR", error_tail=text[-2000:])
    except subprocess.TimeoutExpired:
        res["status"] = "TIMEOUT"
    res["wall_s"] = time.time() - t0
    write(tag, res)
    return res


def part_s(files: dict) -> None:
    data = write_rows("speed_gsm8k", rows_for("gsm8k", files["gsm8k"], S_ROWS))
    for block, cap, base, variant in S_BLOCKS:
        extra_v, lora_v = VARIANTS[variant]
        for tag, extra, lora in ((f"S_{block}_Fa", base, []), (f"S_{block}_Va", base + extra_v, lora_v),
                                 (f"S_{block}_Vb", base + extra_v, lora_v), (f"S_{block}_Fb", base, [])):
            arm_train(tag, data, cap, extra, S_STEPS, lora=lora)


def part_q(dataset: str, files: dict) -> None:
    data = write_rows(f"q_{dataset}", rows_for(dataset, files[dataset], Q_ROWS))
    arm_eval(f"Q_{dataset}_base_eval", dataset)
    for seed in SEEDS[dataset]:
        for variant in Q_VARIANTS:
            extra_v, lora_v = VARIANTS[variant]
            tag = f"Q_{dataset}_{variant}_s{seed}"
            res = arm_train(tag, data, 0, KEEP16 + extra_v, Q_STEPS, keep_out=True, lora=lora_v,
                            seed=seed)
            out = WORK / f"out_{tag}"
            if res.get("status") == "OK" and (out / "adapter_config.json").exists():
                arm_eval(f"{tag}_eval", dataset, str(out))
            shutil.rmtree(out, ignore_errors=True)


def abba(arms: dict, block: str) -> dict:
    sp = {k: arms.get(f"S_{block}_{k}", {}).get("steady_s_per_step") for k in ("Fa", "Va", "Vb", "Fb")}
    v = {"s_per_step": sp}
    if all(x is not None for x in sp.values()):
        F, V = (sp["Fa"], sp["Fb"]), (sp["Va"], sp["Vb"])
        v["speedup"] = sum(F) / sum(V)
        v["verdict"] = ("FASTER" if max(V) < min(F) else "SLOWER" if min(V) > max(F)
                        else "NO_CLEAR_DIFFERENCE")
    else:
        v["verdict"] = "UNKNOWN"
    return v


def decide(arms: dict) -> dict:
    v = {"rule": "ZERO2_STAGE4_PROTOCOL_20261120.md",
         "speed": {block: abba(arms, block) for block, *_ in S_BLOCKS}, "quality": {}}
    for ds, seeds in SEEDS.items():
        row = {"base": arms.get(f"Q_{ds}_base_eval", {}).get("score")}

        def evals(variant, ds=ds, seeds=seeds):
            return [arms.get(f"Q_{ds}_{variant}_s{s}_eval", {}) for s in seeds]

        full = evals("full")
        if not all(e.get("score") is not None for e in full):
            v["quality"][ds] = {**row, "verdict": "UNKNOWN: full missing"}
            continue
        f_mean = sum(e["score"] for e in full) / len(full)
        f_se = math.sqrt(sum((e.get("stderr") or 0) ** 2 for e in full) / len(full))
        row["full"] = {"mean": f_mean, "per_seed": [e["score"] for e in full]}
        for variant in Q_VARIANTS[1:]:
            ev = evals(variant)
            if not all(e.get("score") is not None for e in ev):
                row[variant] = {"verdict": "UNKNOWN"}
                continue
            m = sum(e["score"] for e in ev) / len(ev)
            se_v = math.sqrt(sum((e.get("stderr") or 0) ** 2 for e in ev) / len(ev))
            se = math.sqrt(se_v ** 2 + f_se ** 2)
            per = [e["score"] - f["score"] for e, f in zip(ev, full)]
            d = m - f_mean
            same_sign = all(x > 0 for x in per) or all(x < 0 for x in per)
            verdict = ("BETTER" if d > 2 * se and same_sign else
                       "WORSE" if d < -2 * se and same_sign else "SAME")
            row[variant] = {"mean": m, "per_seed": [e["score"] for e in ev], "delta": d,
                            "delta_per_seed": per, "se": se, "verdict": verdict,
                            "s_per_step": [arms.get(f"Q_{ds}_{variant}_s{s}", {}).get("steady_s_per_step")
                                           for s in seeds]}
        v["quality"][ds] = row
    return v


def main() -> None:
    parts = sys.argv[1:] or ["S", "gsm8k", "arc", "pubmedqa"]
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    arm_env()
    dl = download()
    if dl["status"] == "OK":
        for part in parts:
            if part == "S":
                part_s(dl["files"])
            else:
                part_q(part, dl["files"])
    arms = {p.stem: json.loads(p.read_text()) for p in OUT.glob("*.json")
            if p.stem not in ("summary", "env", "download")}
    summary = {"protocol": "ZERO2_STAGE4_PROTOCOL_20261120.md", "soup_commit": SOUP_COMMIT,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "parts": parts, "arms": arms, "decision": decide(arms),
               "wall_s_this_call": time.time() - t0, "status": "OK"}
    write("summary", summary)
    print(json.dumps(summary["decision"], indent=1))


if __name__ == "__main__":
    main()
