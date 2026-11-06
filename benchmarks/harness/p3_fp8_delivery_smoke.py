#!/usr/bin/env python3
"""Exercise the supported production builder, not DeepSeek numerical acceptance.

Run through Kaggle CLI. The tiny dense Llama uses CPU FP32 weights/compute, with
uncompressed and opt-in FP8 storage, a pre-resolved MoE-plan envelope, and C3.
The conservative routing allowance is synthetic; native MoE routing is covered
by the regression suite and the preserved DeepSeek investigation.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from rich.console import Console


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    try:
        from . import p3_bf16_trainer_contract as contract
    except ImportError:
        import p3_bf16_trainer_contract as contract

    torch.set_num_threads(2)
    _, stand, _ = contract._modules()
    stand.set_deterministic_env()
    stand.enable_deterministic()
    report = {
        "schema": "p3_fp8_component_smoke_v1",
        "scope": "supported Llama builder and actual HF Trainer loop; not public B1.4 wiring",
        "device": "cpu", "compute_dtype": "float32", "weight_format": "dense float32",
        "not_nf4_or_full_model_validation": True,
        "numerical_acceptance": "not claimed; existing DeepSeek FP8 failures stand",
        "timing_scope": "three smoke windows per codec; not a controlled speed comparison",
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="p3-component-smoke-") as temporary:
        for codec in ("none", "fp8_e4m3"):
            case = contract._case(
                lambda codec=codec: contract._builder(
                    Path(temporary) / codec, "cpu", codec=codec, batch_size=2,
                )
            )
            report["cases"][codec] = case
    report["status"] = (
        "PASS" if all(row["status"] == "PASS" for row in report["cases"].values()) else "FAIL"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Console().print(f"Component production smoke: {report['status']} ({args.output})")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
