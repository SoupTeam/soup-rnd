"""Lever arms (C, E, F, G, plus A0 for bits) on a tiny synthetic MoE, one CUDA card, no download.

Run from the repository root on a machine with an sm_80+ GPU, PyTorch 2.14 and the CUDA build
tools (see ``soup_cli.utils.expert_gemm``):

    python benchmarks/harness/lever_tiny_smoke.py [--out DIR] [--skip-speed]

It prints the environment, the bit ladder verdicts of the arms against A0, and the speed table
with the fused-visit counters of the ``lt-fused`` arms (``python_forward`` > 0 means the visit
fell to the Python path). It says nothing about Kanana's absolute numbers; it shows whether the
mechanism (fused against Python path, memory per row count) shows up on a small card.
Writes ``<out>/lever-smoke.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "lever-smoke-out"))
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--vram-limit-gib", default="0", help="allocator limit, 0 = none")
    args = parser.parse_args()
    for sub in ("src", ".", "benchmarks/harness"):
        sys.path.insert(0, str(ROOT / sub))
    import torch

    from soup_cli.utils.expert_gemm import detect_environment

    env = detect_environment()
    print("environment:", env, flush=True)
    if not torch.cuda.is_available():
        print("no CUDA device: nothing to measure")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"gpu {props.name}, {props.total_memory / 2**30:.1f} GiB, sm_{props.major}{props.minor}",
          flush=True)
    from argparse import Namespace

    import b13_moe_fixture as fixture
    import oc_arm_run
    from transformers import AutoTokenizer

    work = Path(tempfile.mkdtemp(prefix="lever-smoke-"))
    vocab = len(AutoTokenizer.from_pretrained("gpt2"))
    weights = fixture.moe_weights_dir(work / "tiny", fixture.moe_config(
        vocab_size=vocab, max_position_embeddings=1024))
    oc_arm_run.prepare(Namespace(source=str(weights), cache=str(work / "cache"), device="cuda",
                                 double_quant=False))
    child_env = dict(os.environ, MBR_VRAM_LIMIT_GIB=args.vram_limit_gib, PYTHONPATH=os.pathsep.join(
        str(ROOT / s) for s in ("src", ".", "benchmarks/harness")))
    out = Path(args.out)
    result = {"environment": str(env), "gpu": props.name, "stages": {}}
    stages = [("lever-bits", ["--seq", "64"])]
    if not args.skip_speed:
        stages.append(("lever-speed", ["--blocks", "1"]))
    for stage, extra in stages:
        done = subprocess.run(
            [sys.executable, str(ROOT / "benchmarks/harness/mbr_measure.py"), stage,
             "--source", str(weights), "--cache", str(work / "cache"), "--out", str(out / stage),
             "--tokenizer", "gpt2", "--steps", "3", *extra],
            capture_output=True, text=True, env=child_env)
        print(f"--- {stage}: exit {done.returncode}")
        print(done.stdout[-1500:])
        rows_file = out / stage / "rows.jsonl"
        rows = [json.loads(x) for x in rows_file.read_text().splitlines()] if rows_file.exists() \
            else []
        result["stages"][stage] = {"returncode": done.returncode, "rows": rows,
                                   "stderr": done.stderr[-1500:]}
        for r in rows:
            print(f"{r['arm']:>3} k={r['k']:<2} S={r['seq']:<3} {r['status']:<5} "
                  f"step={r.get('step_seconds') and round(r['step_seconds'], 2)} "
                  f"nvml={r.get('peak_nvml') and round(r['peak_nvml'] / 2**20)}MiB "
                  f"alloc={r.get('peak_allocated') and round(r['peak_allocated'] / 2**20)}MiB "
                  f"fused={r.get('fused_visits')} {r.get('error', '')[:100]}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "lever-smoke.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
