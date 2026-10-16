"""Bounded arithmetic experiment: calibration is never acceptance evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from d1_rules import compare_tensors, digest, write_new_json
from rich.console import Console

# Fixed before the paid run. Independent partial sums test a different reduction
# tree; tile sizes also change Tensor Core instruction scheduling.
CONFIGS = (
    (16, 32, 16, 4, False),
    (16, 32, 32, 4, False),
    (16, 32, 64, 4, False),
    (16, 32, 128, 4, False),
    (32, 64, 32, 4, False),
    (32, 64, 64, 4, False),
    (32, 64, 128, 4, False),
    (128, 64, 64, 4, False),
    (16, 32, 16, 4, True),
    (16, 32, 32, 4, True),
    (16, 32, 64, 4, True),
    (32, 64, 64, 4, True),
)


def fixture(seed, shape, dtype, nested):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import torch

    torch.manual_seed(seed)
    m, n, k = shape
    precision = getattr(torch, dtype)
    weight = torch.randn(n, k, device="cuda", dtype=precision)
    packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=nested)
    value = torch.randn(m, k, device="cuda", dtype=precision, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=precision)
    native = bnb.matmul_4bit(value, packed.t(), quant_state=state)
    native.backward(dy)
    empty = state.code.new_empty((0,))
    saved = (packed, state.absmax, state.code) + (
        (state.state2.absmax, state.state2.code, state.offset) if nested else (empty, empty, empty)
    )
    metadata = (n, k, state.blocksize, nested, state.state2.blocksize if nested else 0, state.dtype)
    return value.detach(), dy, saved, metadata, native.detach(), value.grad.detach(), state


def evaluate(data, forward_config, dx_config):
    import torch
    from d1_fused_nf4 import _launch

    value, dy, saved, metadata, native, native_dx, _state = data
    result = _launch(value, saved, metadata, False, forward_config)
    dx = _launch(dy, saved, metadata, True, dx_config)
    comparison = compare_tensors(
        {"forward": result, "dx": dx}, {"forward": native, "dx": native_dx}
    )
    counts = {
        "forward": int(torch.count_nonzero(result != native)),
        "dx": int(torch.count_nonzero(dx != native_dx)),
    }
    return {"comparison": comparison, "differing": counts}, result, dx


def run(output: Path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import bitsandbytes as bnb
    import torch
    import triton

    if not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("physical CUDA required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = fixture(842, (128, 64, 128), "float16", False)
    value, dy, saved, _metadata, native, native_dx, state = data
    profile = []
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            x = value.clone().requires_grad_(True)
            bnb.matmul_4bit(x, saved[0].t(), quant_state=state).backward(dy)
            torch.cuda.synchronize()
        profile = sorted(
            {
                event.name
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            }
        )
    except RuntimeError as exc:
        profile = ["PROFILER_UNAVAILABLE: " + str(exc)]
    rows = []
    witnesses = {
        "input": value.cpu(),
        "dy": dy.cpu(),
        "native_forward": native.cpu(),
        "native_dx": native_dx.cpu(),
        "packed": saved[0].cpu(),
    }
    for index, config in enumerate(CONFIGS):
        row, result, dx = evaluate(data, config, config)
        row.update(index=index, config=list(config))
        rows.append(row)
        witnesses[f"config{index}_forward"] = result.cpu()
        witnesses[f"config{index}_dx"] = dx.cpu()
        Console().print({"index": index, "config": config, "differing": row["differing"]})
        # Incremental evidence survives interruption or a later compiler error.
        write_new_json(output.parent / f"calibration-{index:02}.json", row)
    forward = next((r for r in rows if r["differing"]["forward"] == 0), None)
    backward = next((r for r in rows if r["differing"]["dx"] == 0), None)
    validation = []
    if forward is not None and backward is not None:
        for seed in (843, 1977):
            for dtype in ("float16", "bfloat16"):
                for nested in (False, True):
                    for shape in ((128, 64, 128), (133, 67, 96), (256, 256, 512)):
                        sample = fixture(seed, shape, dtype, nested)
                        row, _result, _dx = evaluate(sample, forward["config"], backward["config"])
                        row.update(seed=seed, dtype=dtype, nested=nested, shape=list(shape))
                        validation.append(row)
                        if not row["comparison"]["passed"]:
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
    torch.save(witnesses, output.with_suffix(".pt"))
    result = {
        "status": "VALIDATION_PASS"
        if len(validation) == 24 and all(r["comparison"]["passed"] for r in validation)
        else "ARITHMETIC_GATE_FAIL",
        "source_sha": os.environ.get("D1_SOURCE_SHA"),
        "calibration_seed": 842,
        "configs_predeclared": [list(c) for c in CONFIGS],
        "calibration": rows,
        "selected_forward": forward["config"] if forward else None,
        "selected_dx": backward["config"] if backward else None,
        "validation": validation,
        "native_cuda_kernels": profile,
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "versions": {
            "torch": torch.__version__,
            "bnb": bnb.__version__,
            "triton": triton.__version__,
        },
        "candidate_sha256": digest(Path(__file__).with_name("d1_fused_nf4.py")),
        "screen_sha256": digest(Path(__file__)),
        "witness_sha256": digest(output.with_suffix(".pt")),
        "d1_complete": False,
        "timing_or_training_run": False,
    }
    write_new_json(output, result)
    Console().print({"status": result["status"], "validation_cases": len(validation)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        raise FileExistsError("use a fresh output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.output)
