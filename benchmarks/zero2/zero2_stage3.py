"""ZERO-2 stage 3: which Qwen3.5-9B layers to train, on four datasets, and what it costs.

Protocol: ZERO2_STAGE3_PROTOCOL_20261119.md. Soup research/zero2-fixes (SOUP_COMMIT below).

Part S (speed, GSM8K's first 96 rows, 24 steps per arm, A-B-B-A in each block):
  h8                       8 GiB, plain, kbit_keep_16bit + embed_on_host (the stage-2b fix)
  rF rT rT rF              no cap, plain keep16: all 32 layers vs the top 8 (backward cut)
  lK lL lL lK              no cap, plain keep16, all layers: without vs with use_liger
  sF sT sT sF              8 GiB, stream RAM + chunked CE: all vs top 8
  qF qT qT qF              4 GiB, stream RAM + L2L + chunked CE: all vs top 8
Part Q (quality, per dataset: GSM8K, ARC, PubMedQA, code), no cap, plain keep16, 300 steps:
  base eval, then full / top8 / random8 / grad8 / angular8 / lisa8, each trained and evaluated.

Resumable: an arm whose json already exists is not run again (whatever its status).

    python zero2_stage3.py            # everything, in order
    python zero2_stage3.py S          # one part: S, gsm8k, arc, pubmedqa, code
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

OUT = Path(os.environ.get("ZERO_OUT", "zero2_stage3_results")).resolve()
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
TOP8 = list(range(24, 32))

#: dataset -> (hf repo, revision, file, train rows)
DATA = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
              "main/train-00000-of-00001.parquet"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453",
            ["ARC-Challenge/train-00000-of-00001.parquet", "ARC-Easy/train-00000-of-00001.parquet"]),
    "pubmedqa": ("qiaojin/PubMedQA", "9001f2853fb87cab8d220904e0de81ac6973b318",
                 "pqa_artificial/train-00000-of-00001.parquet"),
    "code": ("sahil2801/CodeAlpaca-20k", "152bb5e9a29651266b018106053980070a0521a1",
             "code_alpaca_20k.json"),
}
#: dataset -> (lm-eval task, limit or None, primary metric)
EVAL = {
    "gsm8k": ("gsm8k", 200, "exact_match,strict-match"),
    "arc": ("arc_challenge", None, "acc_norm,none"),
    "pubmedqa": ("zero2_pubmedqa", None, "acc,none"),
    "code": ("humaneval", None, "pass@1,create_test"),
}
Q_STEPS = 300
Q_ROWS = Q_STEPS * K
S_STEPS = 24
S_ROWS = S_STEPS * K

KEEP16 = ["kbit_keep_16bit: true"]
STREAM = ["stream_layers: true", "stream_source: ram", "stream_chunked_ce: true"]
L2L = STREAM + ["l2l: true"]


def layers_block(method: str) -> list:
    if method == "full":
        return []
    if method == "top8":
        return ["lora_layers:", "  method: fixed", f"  layers: [{', '.join(map(str, TOP8))}]"]
    return ["lora_layers:", f"  method: {method}", "  k: 8", "  seed: 0"]


#: part S: (name, cap GiB, extra training lines)
S_ARMS = [("h8", 8, KEEP16 + ["embed_on_host: true"])]
for block, cap, base, other in (("r", 0, KEEP16, layers_block("top8")),
                                ("l", 0, KEEP16, ["use_liger: true"]),
                                ("s", 8, STREAM, layers_block("top8")),
                                ("q", 4, L2L, layers_block("top8"))):
    a, b = (f"{block}F", f"{block}T") if block != "l" else ("lK", "lL")
    for name, extra in ((a, base), (b, base + other), (b, base + other), (a, base)):
        S_ARMS.append((name, cap, extra))
Q_METHODS = ["full", "top8", "random", "grad", "angular", "lisa"]

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
    print(f"[z3] wrote {name}.json status={obj.get('status')}", flush=True)


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


def config(data: Path, out: Path, cap: int, extra: list) -> str:
    lines = [
        f"base: {MODEL}", "task: sft", "modality: text", "data:", f"  train: {data}",
        "  format: alpaca", "  val_split: 0.0", f"  max_length: {MAX_LEN}", "training:",
        "  epochs: 1", "  lr: 1e-4", "  batch_size: 1", f"  gradient_accumulation_steps: {K}",
        "  seed: 0", "  logging_steps: 1", "  save_steps: 100000", "  gradient_checkpointing: true",
        "  lora:", "    r: 16", "    alpha: 32", "    dropout: 0.0",
        f"    target_modules: [{', '.join(TARGETS)}]", "  quantization: 4bit",
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


def arm_train(tag: str, data: Path, cap: int, extra: list, steps: int, keep_out: bool = False) -> dict:
    prev = done(tag)
    if prev:
        return prev
    out = WORK / f"out_{tag}"
    cfg = WORK / f"{tag}.yaml"
    cfg.write_text(config(data, out, cap, extra))
    launcher = WORK / "zero_launch.py"
    launcher.write_text(LAUNCH)
    peak = WORK / f"{tag}.peak.json"
    env = {**os.environ, "ZERO_PEAK_FILE": str(peak), "ZERO_CAP_GB": str(cap),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    env.pop("SOUP_NF4_TRITON", None)
    res = {"arm": tag, "cap_gib": cap, "extra": extra, "status": "RUNNING"}
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
model_dir, adapter, task, limit, include = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
q = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = transformers.AutoTokenizer.from_pretrained(model_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                                          quantization_config=q, device_map={"": 0})
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
model.eval()
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
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
    task, limit, metric = EVAL[dataset]
    include = WORK / "tasks"
    include.mkdir(parents=True, exist_ok=True)
    (include / "zero2_pubmedqa.yaml").write_text(PUBMEDQA_TASK)
    res = {"arm": tag, "dataset": dataset, "task": task, "limit": limit, "metric": metric,
           "adapter": adapter or None, "status": "RUNNING"}
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", EVAL_CODE, MODEL, adapter, task, str(limit or 0),
                            str(include)], capture_output=True, text=True, timeout=ARM_TIMEOUT_S,
                           env={**os.environ, "HF_ALLOW_CODE_EVAL": "1"})
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
    seen = {}
    for name, cap, extra in S_ARMS:
        seen[name] = seen.get(name, 0) + 1
        arm_train(f"S_{name}{'ab'[seen[name] - 1]}", data, cap, extra, S_STEPS)


def part_q(dataset: str, files: dict) -> None:
    data = write_rows(f"q_{dataset}", rows_for(dataset, files[dataset], Q_ROWS))
    arm_eval(f"Q_{dataset}_base_eval", dataset)
    for method in Q_METHODS:
        tag = f"Q_{dataset}_{method}"
        res = arm_train(tag, data, 0, KEEP16 + layers_block(method), Q_STEPS, keep_out=True)
        out = WORK / f"out_{tag}"
        if res.get("status") == "OK" and (out / "adapter_config.json").exists():
            arm_eval(f"{tag}_eval", dataset, str(out))
        shutil.rmtree(out, ignore_errors=True)


def abba(arms: dict, a: str, b: str) -> dict:
    sp = {k: arms.get(k, {}).get("steady_s_per_step") for k in (f"S_{a}a", f"S_{b}a", f"S_{b}b", f"S_{a}b")}
    v = {"s_per_step": sp}
    if all(x is not None for x in sp.values()):
        A, B = (sp[f"S_{a}a"], sp[f"S_{a}b"]), (sp[f"S_{b}a"], sp[f"S_{b}b"])
        v["speedup"] = sum(A) / sum(B)
        v["verdict"] = "FASTER" if max(B) < min(A) else "SLOWER" if min(B) > max(A) else "NO_CLEAR_DIFFERENCE"
    else:
        v["verdict"] = "UNKNOWN"
    return v


def decide(arms: dict) -> dict:
    v = {"rule": "ZERO2_STAGE3_PROTOCOL_20261119.md",
         "fits_8gib_plain_host": arms.get("S_h8a", {}).get("status") == "OK",
         "top8_resident": abba(arms, "rF", "rT"), "liger_resident": abba(arms, "lK", "lL"),
         "top8_stream8": abba(arms, "sF", "sT"), "top8_l2l4": abba(arms, "qF", "qT"), "quality": {}}
    for ds in EVAL:
        full = arms.get(f"Q_{ds}_full_eval", {})
        row = {"base": arms.get(f"Q_{ds}_base_eval", {}).get("score"), "full": full.get("score")}
        for method in Q_METHODS[1:]:
            e = arms.get(f"Q_{ds}_{method}_eval", {})
            if e.get("score") is None or full.get("score") is None:
                row[method] = {"verdict": "UNKNOWN"}
                continue
            d = e["score"] - full["score"]
            se = math.sqrt((e.get("stderr") or 0) ** 2 + (full.get("stderr") or 0) ** 2)
            row[method] = {"score": e["score"], "delta": d, "se": se,
                           "verdict": "SAME" if abs(d) <= 2 * se else "WORSE" if d < 0 else "BETTER",
                           "s_per_step": arms.get(f"Q_{ds}_{method}", {}).get("steady_s_per_step")}
        row["full_s_per_step"] = arms.get(f"Q_{ds}_full", {}).get("steady_s_per_step")
        v["quality"][ds] = row
    return v


def main() -> None:
    parts = sys.argv[1:] or ["S", "gsm8k", "arc", "pubmedqa", "code"]
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
    summary = {"protocol": "ZERO2_STAGE3_PROTOCOL_20261119.md", "soup_commit": SOUP_COMMIT,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "parts": parts, "arms": arms, "decision": decide(arms),
               "wall_s_this_call": time.time() - t0, "status": "OK"}
    write("summary", summary)
    print(json.dumps(summary["decision"], indent=1))


if __name__ == "__main__":
    main()
