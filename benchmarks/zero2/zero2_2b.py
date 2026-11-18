"""ZERO-2 stage 2b: the stage-2 fixes on Qwen3.5-9B, one 24 GB GPU (Vast).

Protocol: ZERO2_2B_PROTOCOL_20261119.md. Soup research/zero2-fixes (SOUP_COMMIT below).

  nf4x   the Triton NF4 dequant against bitsandbytes on the GPU: bit-for-bit on Qwen3.5-9B's
         weight shapes, and the time of one dequant each (CUDA events)
  m24a, t24a, t24b, m24b   A-B-B-A, no cap, plain QLoRA; t = SOUP_NF4_TRITON=1
  k24    no cap, kbit_keep_16bit
  s8     8 GiB, stream RAM + chunked CE (stage 2's m8s again, for the loss comparison)
  m8k    8 GiB, plain, kbit_keep_16bit
  m8h    8 GiB, plain, kbit_keep_16bit + embed_on_host
  e_base, e_m24   lm-eval GSM8K, 50 questions, NF4 model loaded with BitsAndBytesConfig

Every training arm: one `soup train` process, 24 optimizer steps (k = 4, 96 GSM8K rows), LoRA
r 16 on all 32 layers; step times from a TrainerCallback (perf_counter at each step end).

    python zero2_2b.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path(os.environ.get("ZERO_OUT", "zero2_2b_results")).resolve()
WORK = OUT / "work"
REPO = "Qwen/Qwen3.5-9B"
REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL = os.environ.get("ZERO_MODEL_DIR", str(Path.home() / "qwen35_9b"))
SOUP_COMMIT = "6ac802385acba2250f7a09d84d5bf04a81356d6a"
GSM8K = "openai/gsm8k"
GSM8K_REV = "740312add88f781978c0658806c59bc2815b9866"
STEPS = 24
K = 4
ROWS = STEPS * K
MAX_LEN = 1024
ARM_TIMEOUT_S = 60 * 60
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]
#: (name, cap GiB or 0, extra training lines, env)
TRITON = {"SOUP_NF4_TRITON": "1"}
STREAM = ["stream_layers: true", "stream_source: ram", "stream_chunked_ce: true"]
ARMS = [
    ("m24a", 0, [], {}), ("t24a", 0, [], TRITON), ("t24b", 0, [], TRITON), ("m24b", 0, [], {}),
    ("k24", 0, ["kbit_keep_16bit: true"], {}),
    ("s8", 8, STREAM, {}),
    ("m8k", 8, ["kbit_keep_16bit: true"], {}),
    ("m8h", 8, ["kbit_keep_16bit: true", "embed_on_host: true"], {}),
]
EVAL_LIMIT = 50
#: Qwen3.5-9B linear shapes [out, in]: attention q (gated) / kv / o, DeltaNet in/out, MLP.
NF4_SHAPES = [(8192, 4096), (1024, 4096), (4096, 4096), (12288, 4096), (4096, 12288)]


def write(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    print(f"[z2b] wrote {name}.json status={obj.get('status')}", flush=True)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def arm_env() -> None:
    write("env", {"status": "OK", "python": sys.version,
                  "nvidia": sh("nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv"),
                  "mem": sh("free -b"), "nproc": os.cpu_count(), "cpu": sh("lscpu | grep 'Model name'"),
                  "pip": sh("pip list 2>/dev/null | grep -iE '^(soup-cli|torch|transformers|peft|trl|"
                            "datasets|bitsandbytes|accelerate|flash-linear-attention|fla-core|lm.eval|triton) '")})


def arm_download() -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download
    t0 = time.time()
    res = {"status": "RUNNING", "repo": REPO, "rev": REV}
    try:
        snapshot_download(REPO, revision=REV, local_dir=MODEL,
                          allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt"], max_workers=8)
        res["train_parquet"] = hf_hub_download(GSM8K, "main/train-00000-of-00001.parquet",
                                               repo_type="dataset", revision=GSM8K_REV)
        res.update(status="OK")
    except Exception as e:  # noqa: BLE001
        res.update(status="ERROR", error=f"{type(e).__name__}: {str(e)[:800]}")
    res["wall_s"] = time.time() - t0
    write("download", res)
    return res


NF4X = r'''
import json, sys, torch, bitsandbytes.functional as F
from soup_cli.utils import nf4_triton as nt
torch.manual_seed(0)
rows = []
for n, k in json.loads(sys.argv[1]):
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    packed, st = F.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4",
                                 quant_storage=torch.uint8)
    ref = F.dequantize_4bit(packed, st)
    got = nt.dequantize_nf4(packed, st)
    exact = bool(torch.equal(ref.view(torch.int16), got.view(torch.int16)))
    from bitsandbytes.backends.cuda import ops as bops
    x = torch.randn(256, k, device="cuda").to(torch.bfloat16)
    largs = (x, packed.view(-1, 1), st.shape, st.state2.absmax, st.blocksize, "nf4")
    lkw = dict(absmax_8bit=st.absmax, absmax_code=st.state2.code, absmax_offset=st.offset)
    lin_exact = bool(torch.equal(bops._dequant_linear_fallback(*largs, **lkw),
                                 nt.dequant_linear(*largs, **lkw)))
    def t(fn, reps=50):
        for _ in range(5): fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); a.record()
        for _ in range(reps): fn()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / reps
    rows.append({"shape": [n, k], "bit_exact": exact, "linear_exact": lin_exact,
                 "max_abs_diff": float((ref.float() - got.float()).abs().max()),
                 "bnb_ms": t(lambda: F.dequantize_4bit(packed, st)),
                 "triton_ms": t(lambda: nt.dequantize_nf4(packed, st))})
print("ZERO_NF4X " + json.dumps(rows))
'''


def arm_nf4x() -> dict:
    res = {"arm": "nf4x", "status": "RUNNING"}
    p = subprocess.run([sys.executable, "-c", NF4X, json.dumps(NF4_SHAPES)], capture_output=True,
                       text=True, timeout=1800)
    text = p.stdout + p.stderr
    (OUT / "nf4x.log").write_text(text)
    m = re.search(r"ZERO_NF4X (.*)", text)
    if m:
        rows = json.loads(m.group(1))
        res.update(status="OK", rows=rows,
                   all_bit_exact=all(r["bit_exact"] and r["linear_exact"] for r in rows),
                   bnb_ms_total=sum(r["bnb_ms"] for r in rows),
                   triton_ms_total=sum(r["triton_ms"] for r in rows))
    else:
        res.update(status="ERROR", error_tail=text[-2000:])
    write("nf4x", res)
    return res


def prepare_data(parquet: str) -> Path:
    import pandas as pd
    WORK.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(parquet).head(ROWS)
    dst = WORK / "gsm8k_train_head.jsonl"
    with open(dst, "w") as o:
        for q, a in zip(rows["question"], rows["answer"]):
            o.write(json.dumps({"instruction": q, "input": "", "output": a}) + "\n")
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
    r = {"n_steps": len(dicts),
         "losses": [float(d["loss"]) for d in dicts if "loss" in d],
         "grad_norms": [d.get("grad_norm") for d in dicts],
         "num_tokens": [float(d["num_tokens"]) for d in dicts if "num_tokens" in d]}
    m = re.search(r"NF4 Triton calls: (\{[^}]*\})", text)
    if m:
        r["nf4_triton_calls"] = m.group(1)
    for key, pat in (("nf4_line", r"NF4 dequantization:[^\n]*"), ("keep16_line", r"kbit_keep_16bit:[^\n]*"),
                     ("host_line", r"embed_on_host:[^\n]*"), ("stream_ready_line", r"Layer streaming ready:[^\n]*")):
        m = re.search(pat, text)
        if m:
            r[key] = m.group(0).strip()[:300]
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


def arm_train(name, data, cap, extra, extra_env) -> dict:
    out = WORK / f"out_{name}"
    cfg = WORK / f"{name}.yaml"
    cfg.write_text(config(data, out, cap, extra))
    launcher = WORK / "zero_launch.py"
    launcher.write_text(LAUNCH)
    peak = WORK / f"{name}.peak.json"
    env = {**os.environ, "ZERO_PEAK_FILE": str(peak), "ZERO_CAP_GB": str(cap),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **extra_env}
    if "SOUP_NF4_TRITON" not in extra_env:
        env.pop("SOUP_NF4_TRITON", None)
    res = {"arm": name, "cap_gib": cap, "extra": extra, "env": extra_env, "status": "RUNNING"}
    t0 = time.time()
    with Monitor() as mon:
        try:
            p = subprocess.run([sys.executable, str(launcher), "train", "--config", str(cfg), "-y"],
                               capture_output=True, text=True, timeout=ARM_TIMEOUT_S, env=env)
            text, code = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            text, code = (e.stdout or "") + (e.stderr or ""), "TIMEOUT"
            text = text.decode() if isinstance(text, bytes) else text
    (OUT / f"{name}.log").write_text(text)
    res.update(parse_log(text), exit=code, wall_s=time.time() - t0, nvsmi_peak_mib=mon.peak_mib,
               host_min_available_gb=mon.min_avail_gb)
    if peak.exists():
        pk = json.loads(peak.read_text())
        ends = pk.pop("step_end_s", [])
        res["torch_peak"] = pk
        if len(ends) >= 4:
            gaps = [b - a for a, b in zip(ends, ends[1:])]
            res["step_s"] = gaps
            res["steady_s_per_step"] = statistics.mean(gaps[2:])  # steps 4..N
            toks = res["num_tokens"]
            if len(toks) == len(ends):
                res["steady_tok_per_s"] = (toks[-1] - toks[2]) / (ends[-1] - ends[2])
    res["status"] = ("TIMEOUT" if code == "TIMEOUT" else "OOM" if res["oom"] else
                     "OK" if res["n_steps"] == STEPS and code == 0 else "ERROR")
    write(name, res)
    if name != "m24a":
        shutil.rmtree(out, ignore_errors=True)
    return res


EVAL = r'''
import json, sys, time, torch, transformers
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
model_dir, adapter, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
q = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = transformers.AutoTokenizer.from_pretrained(model_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                                          quantization_config=q, device_map={"": 0})
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
model.eval()
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=8)
t0 = time.time()
r = simple_evaluate(model=lm, tasks=["gsm8k"], limit=limit)
print("ZERO_EVAL " + json.dumps({"results": r["results"]["gsm8k"], "eval_s": time.time() - t0,
                                  "model_class": type(model).__name__}, default=str))
'''


def arm_eval(name: str, adapter: str = "") -> dict:
    res = {"arm": name, "adapter": adapter or None, "limit": EVAL_LIMIT, "status": "RUNNING"}
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", EVAL, MODEL, adapter, str(EVAL_LIMIT)],
                           capture_output=True, text=True, timeout=ARM_TIMEOUT_S)
        text = p.stdout + p.stderr
        (OUT / f"{name}.log").write_text(text)
        m = re.search(r"ZERO_EVAL (.*)", text)
        if m:
            got = json.loads(m.group(1))
            res.update(status="OK", results=got["results"], eval_s=got["eval_s"],
                       model_class=got["model_class"], s_per_question=got["eval_s"] / EVAL_LIMIT)
        else:
            res.update(status="ERROR", error_tail=text[-2000:])
    except subprocess.TimeoutExpired:
        res["status"] = "TIMEOUT"
    res["wall_s"] = time.time() - t0
    write(name, res)
    return res


def decide(arms: dict) -> dict:
    v = {"rule": "ZERO2_2B_PROTOCOL_20261119.md"}
    nf = arms.get("nf4x", {})
    v["nf4_bit_exact"] = nf.get("status") == "OK" and bool(nf.get("all_bit_exact"))
    sp = {n: arms.get(n, {}).get("steady_s_per_step") for n in ("m24a", "t24a", "t24b", "m24b")}
    v["abba_s_per_step"] = sp
    if all(x is not None and arms[n]["status"] == "OK" for n, x in sp.items()):
        m, t = (sp["m24a"], sp["m24b"]), (sp["t24a"], sp["t24b"])
        v["nf4_speedup"] = sum(m) / sum(t)
        v["nf4_train_verdict"] = ("FASTER" if max(t) < min(m) else
                                  "SLOWER" if min(t) > max(m) else "NO_CLEAR_DIFFERENCE")
    else:
        v["nf4_train_verdict"] = "UNKNOWN"
    v["fits_8gib_plain_keep16"] = arms.get("m8k", {}).get("status") == "OK"
    v["fits_8gib_plain_keep16_host"] = arms.get("m8h", {}).get("status") == "OK"
    v["eval_ok"] = all(arms.get(n, {}).get("status") == "OK" for n in ("e_base", "e_m24"))
    a, b = arms.get("k24", {}).get("losses") or [], arms.get("s8", {}).get("losses") or []
    if a and len(a) == len(b):
        v["k24_vs_s8_max_rel_loss_diff"] = max(abs(x - y) / abs(y) for x, y in zip(a, b))
    a = arms.get("m24a", {}).get("losses") or []
    if a and len(a) == len(b):
        v["m24_vs_s8_max_rel_loss_diff"] = max(abs(x - y) / abs(y) for x, y in zip(a, b))
    return v


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    arm_env()
    arms = {}
    dl = arm_download()
    if dl["status"] == "OK":
        arms["nf4x"] = arm_nf4x()
        data = prepare_data(dl["train_parquet"])
        for name, cap, extra, env in ARMS:
            arms[name] = arm_train(name, data, cap, extra, env)
        arms["e_base"] = arm_eval("e_base")
        if arms["m24a"]["status"] == "OK":
            arms["e_m24"] = arm_eval("e_m24", str(WORK / "out_m24a"))
    summary = {"protocol": "ZERO2_2B_PROTOCOL_20261119.md", "soup_commit": SOUP_COMMIT,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "arms": arms, "decision": decide(arms), "total_wall_s": time.time() - t0,
               "status": "OK"}
    write("summary", summary)
    print(json.dumps(summary["decision"], indent=1))


if __name__ == "__main__":
    main()
