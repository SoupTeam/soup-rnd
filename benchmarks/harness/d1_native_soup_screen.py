"""Short full-step Soup streaming fixture comparison; no pretrained quality claim."""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from d1_rules import compare_tensors, write_new_json
from rich.console import Console


def run(output: Path) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import d1_fused_nf4 as candidate
    import torch

    root = Path(__file__).parents[2]
    sys.path.insert(0, str(root / "tests"))
    from test_v07202 import _nf4_stream, _randomise_lora_b, _sync_adapters

    from soup_cli.utils import layer_stream_runtime as runtime_module

    torch.manual_seed(1977)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    old_linear = runtime_module.checkpoint_visible_nf4_linear
    old_gate = runtime_module._can_use_checkpoint_visible_nf4_gemm
    arms, runtimes = {}, []
    with tempfile.TemporaryDirectory(prefix="d1-soup-step-") as temporary:
        for name in ("native", "candidate"):
            directory = Path(temporary) / name
            directory.mkdir()
            model, runtime, *_ = _nf4_stream(
                directory, n_layers=4, buffers=2, device="cuda", dtype="bfloat16"
            )
            model.train()
            runtimes.append(runtime)
            _randomise_lora_b(model)
            params = [p for p in model.parameters() if p.requires_grad]
            arms[name] = (model, torch.optim.SGD(params, lr=0.001))
        _sync_adapters(arms["candidate"][0], arms["native"][0])
        inputs = torch.randint(0, 64, (1, 128), device="cuda")

        def step(name: str, capture: bool = False) -> dict[str, Any]:
            runtime_module.checkpoint_visible_nf4_linear = (
                old_linear
                if name == "native"
                else lambda *a, **kw: candidate.linear(*a, **kw, backend="wmma_native")
            )
            runtime_module._can_use_checkpoint_visible_nf4_gemm = (
                old_gate
                if name == "native"
                else lambda x, state: x.dtype == torch.bfloat16 and state.shape[0] % 64 == 0
            )
            model, optimizer = arms[name]
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=inputs, use_cache=False).logits
            loss = logits.float().square().mean()
            loss.backward()
            optimizer.step()
            if not capture:
                return {}
            return {
                "logits": logits.detach().clone(),
                "loss": loss.detach().clone(),
                **{
                    f"parameter:{key}": p.detach().clone()
                    for key, p in model.named_parameters()
                    if p.requires_grad
                },
                **{
                    f"gradient:{key}": p.grad.detach().clone()
                    for key, p in model.named_parameters()
                    if p.requires_grad
                },
            }

        telemetry_file = (output / "nvidia-smi.csv").open("w")
        telemetry = None
        try:
            correctness = [
                compare_tensors(step("candidate", True), step("native", True)) for _ in range(3)
            ]
            write_new_json(output / "correctness.json", correctness)
            if not all(row["passed"] for row in correctness):
                result = {"status": "CORRECTNESS_FAIL", "correctness": correctness}
                write_new_json(output / "screen.json", result)
                return result
            for name in arms:
                for _ in range(5):
                    step(name)
            torch.cuda.synchronize()
            telemetry = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,uuid,clocks.sm,utilization.memory,"
                    "utilization.gpu,power.limit",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "100",
                ],
                stdout=telemetry_file,
                stderr=subprocess.STDOUT,
            )
            rows = []
            for block in range(3):
                order = ("native", "candidate") if block % 2 == 0 else ("candidate", "native")
                for name in order:
                    torch.cuda.reset_peak_memory_stats()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(True)
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
            medians = {
                name: statistics.median(row["step_cuda_ms"] for row in rows if row["arm"] == name)
                for name in arms
            }
            result = {
                "status": "SHORT_SCREEN_ONLY",
                "source_sha": os.environ.get("D1_SOURCE_SHA"),
                "correctness": correctness,
                "rows": rows,
                "median_step_ms": medians,
                "candidate_over_native": medians["candidate"] / medians["native"],
                "scope": "Soup streamed 4-layer Llama, hidden64, seq128, SGD, synthetic weights",
                "reference_reduced_precision": True,
                "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                "d1_complete": False,
                "quality_training": "NOT_RUN",
                "performance_verdict": "NO_FINAL_CLAIM: short fixture and raw clocks retained",
            }
            write_new_json(output / "screen.json", result)
            return result
        finally:
            if telemetry is not None:
                telemetry.terminate()
                telemetry.wait(timeout=5)
            telemetry_file.close()
            runtime_module.checkpoint_visible_nf4_linear = old_linear
            runtime_module._can_use_checkpoint_visible_nf4_gemm = old_gate
            for runtime in runtimes:
                runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.output)
