"""WMMA hypothesis from the observed native CUTLASS kernel; strict short gate."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from d1_arithmetic_screen import fixture
from d1_rules import compare_tensors, digest, write_new_json
from rich.console import Console


def run(output):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import bitsandbytes.functional as functional
    import torch
    from d1_wmma_nf4 import compile_kernel, matmul

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, verdicts = [], {}
    for dtype in ("float16", "bfloat16"):
        cases = [(842, False, (128, 64, 128))]
        cases += [
            (seed, nested, shape)
            for seed in (843, 1977)
            for nested in (False, True)
            for shape in ((128, 64, 128), (133, 67, 96), (256, 256, 512), (512, 4096, 4096))
        ]
        passed = True
        for index, (seed, nested, shape) in enumerate(cases):
            data = fixture(seed, shape, dtype, nested)
            value, dy, saved, metadata, native, native_dx, state = data
            forward = matmul(value, saved, metadata, False)
            dx = matmul(dy, saved, metadata, True)
            comparison = compare_tensors(
                {"forward": forward, "dx": dx}, {"forward": native, "dx": native_dx}
            )
            row = {
                "dtype": dtype,
                "seed": seed,
                "nested": nested,
                "shape": list(shape),
                "comparison": comparison,
                "index": index,
                "differing": {
                    "forward": int(torch.count_nonzero(forward != native)),
                    "dx": int(torch.count_nonzero(dx != native_dx)),
                },
            }
            if index == 0 or not comparison["passed"]:
                k = shape[2]
                decoded = matmul(
                    torch.eye(k, device="cuda", dtype=value.dtype), saved, metadata, False
                ).t()
                native_decoded = functional.dequantize_4bit(saved[0], state)
                row["decode"] = compare_tensors({"weight": decoded}, {"weight": native_decoded})
                tensors = {
                    "input": value,
                    "dy": dy,
                    "native_forward": native,
                    "native_dx": native_dx,
                    "forward": forward,
                    "dx": dx,
                    "decoded": decoded,
                    "native_decoded": native_decoded,
                    "packed": saved[0],
                    **{
                        "state_" + name: tensor
                        for name, tensor in state.as_dict(packed=True).items()
                    },
                }
                witness = output.parent / f"{dtype}-{index:02}.pt"
                torch.save(
                    {name: tensor.detach().cpu() for name, tensor in tensors.items()}, witness
                )
                row["witness"] = {"file": witness.name, "sha256": digest(witness)}
            rows.append(row)
            write_new_json(output.parent / f"{dtype}-{index:02}.json", row)
            Console().print(
                {"dtype": dtype, "seed": seed, "shape": shape, "differing": row["differing"]}
            )
            if not comparison["passed"] or not row.get("decode", {"passed": True})["passed"]:
                passed = False
                break
        verdicts[dtype] = "VALIDATION_PASS" if passed and index == 16 else "ARITHMETIC_GATE_FAIL"
        _driver, _module, _function, ptx, log = compile_kernel(getattr(torch, dtype))
        (output.parent / (dtype + ".ptx")).write_bytes(ptx)
        (output.parent / (dtype + "-compile.log")).write_text(log)
    result = {
        "source_sha": os.environ.get("D1_SOURCE_SHA"),
        "cases": rows,
        "verdicts": verdicts,
        "d1_complete": False,
        "timing_or_training_run": False,
        "candidate_sha256": digest(Path(__file__).with_name("d1_wmma_nf4.py")),
        "screen_sha256": digest(Path(__file__)),
        "cuda_source_sha256": hashlib.sha256(__import__("d1_wmma_nf4").SOURCE.encode()).hexdigest(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
    }
    write_new_json(output, result)
    Console().print(verdicts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.output)
