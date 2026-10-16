"""One synthetic LoRA full-step screen, explicitly requiring FP32 BF16 reduction."""

import argparse
import os
import statistics
import subprocess
import time
from pathlib import Path

from d1_rules import compare_tensors, write_new_json
from rich.console import Console


def run(output):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import d1_fused_nf4 as candidate
    import torch
    from d1_wmma_nf4 import compile_kernel

    torch.manual_seed(1977)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    n = k = 4096
    m, rank = 128, 8
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / k**0.5
    packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
    del weight
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    initial_a = torch.randn(rank, k, device="cuda", dtype=torch.float32) * 0.01
    initial_b = torch.randn(n, rank, device="cuda", dtype=torch.float32) * 0.01
    arms = {}
    for name in ("native", "candidate"):
        a = torch.nn.Parameter(initial_a.clone())
        b = torch.nn.Parameter(initial_b.clone())
        arms[name] = (a, b, torch.optim.SGD([a, b], lr=0.001))

    def step(name):
        a, b, optimizer = arms[name]
        optimizer.zero_grad(set_to_none=True)
        x = source.detach().requires_grad_(True)
        base = (
            bnb.matmul_4bit(x, packed.t(), quant_state=state)
            if name == "native"
            else (candidate.linear(x, packed, state, backend="wmma"))
        )
        result = base + ((x.float() @ a.t()) @ b.t()).to(base.dtype)
        loss = result.float().square().mean()
        loss.backward()
        optimizer.step()
        return {
            "loss": loss.detach(),
            "a": a.detach(),
            "b": b.detach(),
            "grad_a": a.grad,
            "grad_b": b.grad,
            "dx": x.grad,
        }

    correctness = []
    for _ in range(3):
        native = {key: value.clone() for key, value in step("native").items()}
        actual = step("candidate")
        correctness.append(compare_tensors(actual, native))
    write_new_json(output / "step-correctness.json", correctness)
    if not all(r["passed"] for r in correctness):
        write_new_json(
            output / "screen.json", {"status": "CORRECTNESS_FAIL", "correctness": correctness}
        )
        return
    for name in arms:
        for _ in range(10):
            step(name)
    torch.cuda.synchronize()
    telemetry_file = (output / "nvidia-smi.csv").open("w")
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,uuid,clocks.sm,utilization.memory,utilization.gpu,power.limit",
            "--format=csv,noheader,nounits",
            "-lms",
            "100",
        ],
        stdout=telemetry_file,
        stderr=subprocess.STDOUT,
    )
    rows = []
    try:
        for block in range(3):
            order = ("native", "candidate") if block % 2 == 0 else ("candidate", "native")
            for name in order:
                torch.cuda.reset_peak_memory_stats()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                before = time.time_ns()
                start.record()
                for _ in range(200):
                    step(name)
                end.record()
                end.synchronize()
                row = {
                    "arm": name,
                    "block": block,
                    "steps": 200,
                    "start_ns": before,
                    "end_ns": time.time_ns(),
                    "step_cuda_ms": start.elapsed_time(end) / 200,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                rows.append(row)
                Console().print(row)
        # Allocate dense W only after the full-step memory intervals.
        dense = functional.dequantize_4bit(packed, state)
        for _ in range(20):
            torch.nn.functional.linear(source, dense)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.time_ns()
        start.record()
        for _ in range(200):
            torch.nn.functional.linear(source, dense)
        end.record()
        end.synchronize()
        ceiling = {
            "calls": 200,
            "start_ns": before,
            "end_ns": time.time_ns(),
            "gemm_cuda_ms": start.elapsed_time(end) / 200,
        }
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=5)
        telemetry_file.close()
    medians = {
        name: statistics.median(r["step_cuda_ms"] for r in rows if r["arm"] == name)
        for name in arms
    }
    result = {
        "status": "SHORT_SCREEN_ONLY",
        "source_sha": os.environ.get("D1_SOURCE_SHA"),
        "correctness": correctness,
        "rows": rows,
        "median_step_ms": medians,
        "candidate_over_native": medians["candidate"] / medians["native"],
        "dense_gemm": ceiling,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "allow_bf16_reduced_precision_reduction": False,
        "scope": "synthetic single-projection LoRA forward/backward/SGD, weights already on GPU",
        "performance_verdict": "NO_FINAL_SPEED_CLAIM: short windows; raw clocks retained",
        "real_model_training": "NOT_RUN",
        "d1_complete": False,
    }
    _driver, _module, _function, ptx, _log = compile_kernel(torch.bfloat16, 64, 256)
    (output / "candidate.ptx").write_bytes(ptx)
    result["ptx_signed_64bit_divisions"] = ptx.count(b"div.s64")
    write_new_json(output / "screen.json", result)
    Console().print(
        {"median_step_ms": medians, "candidate_over_native": result["candidate_over_native"]}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    run(args.output)
