"""Short follow-up: smaller MMA groups and cuBLAS operand orientation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from d1_arithmetic_screen import evaluate, fixture
from d1_rules import digest, write_new_json
from rich.console import Console

CONFIGS = tuple(
    (16, 32, 16, 4, partial, chunk, swap)
    for partial in (False, True)
    for chunk in (4, 8, 16)
    for swap in (False, True)
)


def run(output):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("physical CUDA required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, validations, selections, witnesses = [], [], {}, {}
    for dtype in ("float16", "bfloat16"):
        data = fixture(842, (128, 64, 128), dtype, False)
        witnesses[dtype + "_native_forward"] = data[4].cpu()
        witnesses[dtype + "_native_dx"] = data[5].cpu()
        for index, config in enumerate(CONFIGS):
            row, forward, dx = evaluate(data, config, config)
            row.update(dtype=dtype, index=index, config=list(config))
            rows.append(row)
            witnesses[f"{dtype}_{index}_forward"] = forward.cpu()
            witnesses[f"{dtype}_{index}_dx"] = dx.cpu()
            write_new_json(output.parent / f"{dtype}-calibration-{index:02}.json", row)
            Console().print({"dtype": dtype, "index": index, "differing": row["differing"]})
        eligible = [r for r in rows if r["dtype"] == dtype]
        forward = next((r for r in eligible if r["differing"]["forward"] == 0), None)
        dx = next((r for r in eligible if r["differing"]["dx"] == 0), None)
        selections[dtype] = {
            "forward": forward["config"] if forward else None,
            "dx": dx["config"] if dx else None,
        }
        if forward is None or dx is None:
            continue
        for seed in (843, 1977):
            for nested in (False, True):
                for shape in ((128, 64, 128), (133, 67, 96), (256, 256, 512), (512, 4096, 4096)):
                    sample = fixture(seed, shape, dtype, nested)
                    row, result, gradient = evaluate(sample, forward["config"], dx["config"])
                    row.update(dtype=dtype, seed=seed, nested=nested, shape=list(shape))
                    validations.append(row)
                    key = f"validation-{len(validations):02}"
                    for name, tensor in (
                        ("native_forward", sample[4]),
                        ("native_dx", sample[5]),
                        ("forward", result),
                        ("dx", gradient),
                    ):
                        witnesses[key + "_" + name] = tensor.cpu()
                    write_new_json(output.parent / (key + ".json"), row)
                    Console().print(
                        {
                            "dtype": dtype,
                            "seed": seed,
                            "shape": shape,
                            "differing": row["differing"],
                        }
                    )
                    if not row["comparison"]["passed"]:
                        break
                else:
                    continue
                break
            else:
                continue
            break
    torch.save(witnesses, output.with_suffix(".pt"))
    verdicts = {}
    for dtype in ("float16", "bfloat16"):
        subset = [r for r in validations if r["dtype"] == dtype]
        verdicts[dtype] = (
            "VALIDATION_PASS"
            if len(subset) == 16 and all(r["comparison"]["passed"] for r in subset)
            else "ARITHMETIC_GATE_FAIL"
        )
    result = {
        "source_sha": os.environ.get("D1_SOURCE_SHA"),
        "calibration": rows,
        "configs_predeclared": [list(c) for c in CONFIGS],
        "selection": selections,
        "validation": validations,
        "verdicts": verdicts,
        "witness_sha256": digest(output.with_suffix(".pt")),
        "candidate_sha256": digest(Path(__file__).with_name("d1_fused_nf4.py")),
        "screen_sha256": digest(Path(__file__)),
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "d1_complete": False,
        "timing_or_training_run": False,
    }
    write_new_json(output, result)
    Console().print(verdicts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        raise FileExistsError("use a fresh output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.output)
