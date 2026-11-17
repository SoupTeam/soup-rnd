"""ZERO-2 stage 2: Qwen3.5-9B baseline on one 24 GB GPU (Vast) — fit, speed, profile, eval smoke.

Protocol: ZERO2_BASE_PROTOCOL_20261116.md. Soup research/layer-select (SOUP_COMMIT below).

Arms (one `soup train` process each, 12 optimizer steps, LoRA r 16 on all 32 layers, NF4,
batch 1 x k = 4, max_length 1024, GSM8K train rows in file order):
  m24  no VRAM cap, plain QLoRA                      (speed reference)
  m8   8 GiB cap, plain QLoRA
  m8s  8 GiB cap, layer streaming (RAM tier) + chunked CE
  m4   4 GiB cap, layer streaming (RAM tier) + L2L + chunked CE
  p24  m24's config with torch.profiler over optimizer steps 3-4 (top CUDA kernels)
Then an eval smoke: lm-eval GSM8K on 50 test questions for the base model and for m24's
adapter (pipeline + time per question, not quality).

    python zero2_base.py                # on the Vast machine, from an empty directory
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path(os.environ.get("ZERO_OUT", "zero2_base_results")).resolve()
WORK = OUT / "work"
REPO = "Qwen/Qwen3.5-9B"
REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL = os.environ.get("ZERO_MODEL_DIR", str(Path.home() / "qwen35_9b"))
SOUP_COMMIT = "63443f153a9fe5accaed0490df63571a01d3cfd7"
GSM8K = "openai/gsm8k"
GSM8K_REV = "740312add88f781978c0658806c59bc2815b9866"
STEPS = 12
K = 4
ROWS = STEPS * K
MAX_LEN = 1024
ARM_TIMEOUT_S = 60 * 60
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]
#: (name, cap GiB or 0, stream, l2l, profile)
ARMS = [("m24", 0, False, False, False), ("m8", 8, False, False, False),
        ("m8s", 8, True, False, False), ("m4", 4, True, True, False),
        ("p24", 0, False, False, True)]
EVAL_LIMIT = 50


def write(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    print(f"[z2] wrote {name}.json status={obj.get('status')}", flush=True)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def arm_env() -> None:
    write("env", {"status": "OK", "python": sys.version,
                  "nvidia": sh("nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv"),
                  "mem": sh("free -b"), "df": sh("df -B1 / . 2>/dev/null"), "nproc": os.cpu_count(),
                  "cpu": sh("lscpu | grep 'Model name'"),
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
        res.update(status="OK", bytes=sum(p.stat().st_size for p in Path(MODEL).rglob("*.safetensors")))
    except Exception as e:  # noqa: BLE001
        res.update(status="ERROR", error=f"{type(e).__name__}: {str(e)[:800]}")
    res["wall_s"] = time.time() - t0
    write("download", res)
    return res


def prepare_data(parquet: str) -> Path:
    """GSM8K train in file order, first ROWS rows, as alpaca (question -> worked answer)."""
    import pandas as pd
    WORK.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(parquet).head(ROWS)
    dst = WORK / "gsm8k_train_head.jsonl"
    with open(dst, "w") as o:
        for q, a in zip(rows["question"], rows["answer"]):
            o.write(json.dumps({"instruction": q, "input": "", "output": a}) + "\n")
    return dst


def config(data: Path, out: Path, cap: int, stream: bool, l2l: bool) -> str:
    lines = [
        f"base: {MODEL}", "task: sft", "modality: text", "data:", f"  train: {data}",
        "  format: alpaca", "  val_split: 0.0", f"  max_length: {MAX_LEN}", "training:",
        "  epochs: 1", "  lr: 1e-4", "  batch_size: 1", f"  gradient_accumulation_steps: {K}",
        "  seed: 0", "  logging_steps: 1", "  save_steps: 100000", "  gradient_checkpointing: true",
        "  lora:", "    r: 16", "    alpha: 32", "    dropout: 0.0",
        f"    target_modules: [{', '.join(TARGETS)}]", "  quantization: 4bit",
    ]
    if stream:
        # Chunked CE exists on the streamed path only; the plain arms use HF's loss.
        lines += ["  stream_layers: true", "  stream_source: ram", "  stream_chunked_ce: true"]
        if l2l:
            lines += ["  l2l: true"]
        if cap:
            lines += [f"  stream_vram_override: {cap * 10**9}", "  stream_vram_probe: true"]
    lines += [f"output: {out}"]
    return "\n".join(lines) + "\n"


#: soup in-process: optional hard VRAM cap, optional profiler over steps 3-4, peaks at exit.
LAUNCH = r'''
import atexit, json, os, sys
import torch
cap = float(os.environ.get("ZERO_CAP_GB", "0"))
if cap and torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cap * 2**30 / total, 0)
def _dump():
    if torch.cuda.is_available():
        with open(os.environ["ZERO_PEAK_FILE"], "w") as f:
            json.dump({"peak_allocated_gb": torch.cuda.max_memory_allocated(0) / 2**30,
                       "peak_reserved_gb": torch.cuda.max_memory_reserved(0) / 2**30}, f)
atexit.register(_dump)
if os.environ.get("ZERO_PROFILE_FILE"):
    import transformers
    from torch.profiler import ProfilerActivity, profile
    class _Prof(transformers.TrainerCallback):
        prof = None
        def on_step_begin(self, args, state, control, **kw):
            if state.global_step == 2 and self.prof is None:
                self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                self.prof.__enter__()
        def on_step_end(self, args, state, control, **kw):
            if state.global_step == 4 and self.prof is not None:
                torch.cuda.synchronize()
                self.prof.__exit__(None, None, None)
                rows = []
                for e in self.prof.key_averages():
                    t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
                    rows.append({"name": e.key[:160], "self_cuda_ms": t / 1000.0, "calls": e.count,
                                 "cpu_ms": e.self_cpu_time_total / 1000.0})
                rows.sort(key=lambda r: -r["self_cuda_ms"])
                with open(os.environ["ZERO_PROFILE_FILE"], "w") as f:
                    json.dump({"steps": [3, 4], "top": rows[:40],
                               "total_self_cuda_ms": sum(r["self_cuda_ms"] for r in rows)}, f, indent=1)
                self.prof = None
    _init = transformers.Trainer.__init__
    def _patched(self, *a, **k):
        _init(self, *a, **k)
        self.add_callback(_Prof())
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
    el = {}
    for step, mm, ss in re.findall(r"\b(\d+)/\d+ \[(\d+):(\d+)<", text):
        el[int(step)] = int(mm) * 60 + int(ss)
    for step, hh, mm, ss in re.findall(r"\b(\d+)/\d+ \[(\d+):(\d+):(\d+)<", text):
        el[int(step)] = int(hh) * 3600 + int(mm) * 60 + int(ss)
    n = r["n_steps"]
    if n >= 3 and 1 in el and n in el and el[n] > el[1]:
        r["first_step_s"] = el[1]
        r["steady_s_per_step"] = (el[n] - el[1]) / (n - 1)
        toks = r["num_tokens"]
        if len(toks) == n:
            r["steady_tok_per_s"] = (toks[-1] - toks[0]) / (el[n] - el[1])
    for key, pat in (("l2l_line", r"L2L:[^\n]*"), ("stream_ready_line", r"Layer streaming ready:[^\n]*"),
                     ("chunked_line", r"[Cc]hunked[^\n]*")):
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


def arm_train(name, data, cap, stream, l2l, prof) -> dict:
    out = WORK / f"out_{name}"
    cfg = WORK / f"{name}.yaml"
    cfg.write_text(config(data, out, cap, stream, l2l))
    launcher = WORK / "zero_launch.py"
    launcher.write_text(LAUNCH)
    peak = WORK / f"{name}.peak.json"
    profile = OUT / f"{name}.profile.json"
    env = {**os.environ, "ZERO_PEAK_FILE": str(peak), "ZERO_CAP_GB": str(cap),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    env.pop("ZERO_PROFILE_FILE", None)
    if prof:
        env["ZERO_PROFILE_FILE"] = str(profile)
    res = {"arm": name, "cap_gib": cap, "stream": stream, "l2l": l2l, "profile": prof,
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
    (OUT / f"{name}.log").write_text(text)
    res.update(parse_log(text), exit=code, wall_s=time.time() - t0, nvsmi_peak_mib=mon.peak_mib,
               host_min_available_gb=mon.min_avail_gb)
    if peak.exists():
        res["torch_peak"] = json.loads(peak.read_text())
    if profile.exists():
        res["profile_top"] = json.loads(profile.read_text())["top"][:15]
    res["status"] = ("TIMEOUT" if code == "TIMEOUT" else "OOM" if res["oom"] else
                     "OK" if res["n_steps"] == STEPS and code == 0 else "ERROR")
    write(name, res)
    if name != "m24":
        shutil.rmtree(out, ignore_errors=True)
    return res


def arm_eval(name: str, adapter: str = "") -> dict:
    """lm-eval GSM8K (5-shot default of the task), first EVAL_LIMIT questions, NF4 base."""
    t0 = time.time()
    res = {"arm": name, "adapter": adapter or None, "limit": EVAL_LIMIT, "status": "RUNNING"}
    code = f'''
import json, lm_eval, transformers
args = "pretrained={MODEL},dtype=bfloat16,load_in_4bit=True" + (",peft={adapter}" if "{adapter}" else "")
r = lm_eval.simple_evaluate(model="hf", model_args=args, tasks=["gsm8k"], batch_size=8,
                            limit={EVAL_LIMIT}, device="cuda")
print("ZERO_EVAL " + json.dumps(r["results"]["gsm8k"], default=str))
'''
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=ARM_TIMEOUT_S)
        text = p.stdout + p.stderr
        (OUT / f"{name}.log").write_text(text)
        m = re.search(r"ZERO_EVAL (.*)", text)
        res.update(status="OK" if m else "ERROR", results=json.loads(m.group(1)) if m else None,
                   error_tail=None if m else text[-1500:])
    except subprocess.TimeoutExpired:
        res["status"] = "TIMEOUT"
    res["wall_s"] = time.time() - t0
    res["s_per_question"] = res["wall_s"] / EVAL_LIMIT
    write(name, res)
    return res


def decide(arms: dict) -> dict:
    v = {"rule": "ZERO2_BASE_PROTOCOL_20261116.md"}
    v["fits_8gib_plain"] = arms.get("m8", {}).get("status") == "OK"
    v["fits_8gib_streamed"] = arms.get("m8s", {}).get("status") == "OK"
    v["fits_4gib_streamed_l2l"] = arms.get("m4", {}).get("status") == "OK"
    v["s_per_step"] = {n: arms.get(n, {}).get("steady_s_per_step") for n in ("m24", "m8", "m8s", "m4")}
    v["eval_ok"] = all(arms.get(n, {}).get("status") == "OK" for n in ("e_base", "e_m24"))
    return v


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    arm_env()
    arms = {}
    dl = arm_download()
    if dl["status"] == "OK":
        data = prepare_data(dl["train_parquet"])
        for name, cap, stream, l2l, prof in ARMS:
            arms[name] = arm_train(name, data, cap, stream, l2l, prof)
        arms["e_base"] = arm_eval("e_base")
        if arms["m24"]["status"] == "OK":
            arms["e_m24"] = arm_eval("e_m24", str(WORK / "out_m24"))
    summary = {"protocol": "ZERO2_BASE_PROTOCOL_20261116.md", "soup_commit": SOUP_COMMIT,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "arms": arms, "decision": decide(arms), "total_wall_s": time.time() - t0,
               "status": "OK"}
    write("summary", summary)
    print(json.dumps(summary["decision"], indent=1))


if __name__ == "__main__":
    main()
