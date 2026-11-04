"""Explicit aligned-weight scope after WMMA tail failures; no hidden acceptance."""

import argparse
import os
from pathlib import Path

from d1_arithmetic_screen import fixture
from d1_rules import compare_tensors, write_new_json
from rich.console import Console


def run(output):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import bitsandbytes as bnb
    import torch
    from d1_wmma_nf4 import matmul

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    profiles = {}
    for dtype in ("float16", "bfloat16"):
        value, dy, saved, _meta, _out, _dx, state = fixture(1977, (133, 67, 96), dtype, True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            x = value.clone().requires_grad_(True)
            bnb.matmul_4bit(x, saved[0].t(), quant_state=state).backward(dy)
            torch.cuda.synchronize()
        profiles[dtype] = sorted(
            {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}
        )
    write_new_json(output.parent / "tail-profile.json", profiles)
    rows = []
    for dtype in ("float16", "bfloat16"):
        for seed in (843, 1977, 99991):
            for nested in (False, True):
                for shape in (
                    (128, 64, 128),
                    (133, 128, 96),
                    (256, 256, 512),
                    (512, 4096, 4096),
                    (129, 11008, 4096),
                ):
                    data = fixture(seed, shape, dtype, nested)
                    value, dy, saved, meta, native, native_dx, _state = data
                    forward = matmul(value, saved, meta, False)
                    dx = matmul(dy, saved, meta, True)
                    comparison = compare_tensors(
                        {"forward": forward, "dx": dx}, {"forward": native, "dx": native_dx}
                    )
                    row = {
                        "dtype": dtype,
                        "seed": seed,
                        "nested": nested,
                        "shape": list(shape),
                        "comparison": comparison,
                        "differing": {
                            "forward": int(torch.count_nonzero(forward != native)),
                            "dx": int(torch.count_nonzero(dx != native_dx)),
                        },
                    }
                    rows.append(row)
                    write_new_json(output.parent / f"case-{len(rows):02}.json", row)
                    Console().print({k: row[k] for k in ("dtype", "seed", "shape", "differing")})
                    if not comparison["passed"]:
                        break
                else:
                    continue
                break
            else:
                continue
            break
    write_new_json(
        output,
        {
            "source_sha": os.environ.get("D1_SOURCE_SHA"),
            "cases": rows,
            "d1_complete": False,
            "scope": "M>=128, N%64=0, K%32=0",
            "no_timing_or_training": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.output)
