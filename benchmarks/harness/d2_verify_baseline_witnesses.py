"""Replay public loss curves and failure witnesses, not the omitted full tensor matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from rich.console import Console


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    root = Path(os.path.realpath(root))
    manifest = read(root / "compact-manifest.json")
    for name, expected in manifest.items():
        path = os.path.realpath(root / name)
        require(os.path.commonpath([str(root), path]) == str(root), "manifest path escape")
        require(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,
            "payload SHA mismatch: " + name,
        )
    summary = read(root / "summary.json")
    expected_cases = {
        (dtype, backend)
        for dtype in ("float16", "bfloat16")
        for backend in ("peft", "soup", "liger", "unsloth")
    }
    require(
        {(row["dtype"], row["backend"]) for row in summary["table"]} == expected_cases,
        "incomplete case matrix",
    )
    loss_failures = 0
    for row in summary["table"]:
        reference = read(root / f"{row['dtype']}-gate-peft/result.json")["rows"]
        candidate = read(root / f"{row['dtype']}-gate-{row['backend']}/result.json")["rows"]
        require(len(reference) == len(candidate) == 50, "missing loss steps")
        matches = 0
        for index, (left, right) in enumerate(zip(reference, candidate), 1):
            require(left["step"] == right["step"] == index, "wrong loss step order")
            require(math.isfinite(left["loss"]) and math.isfinite(right["loss"]), "nonfinite loss")
            matches += f"{left['loss']:.3f}" == f"{right['loss']:.3f}"
        require(matches == row["loss_matches_3dp"], "loss count mismatch")
        loss_failures += matches < 50
    witnesses = []
    for witness in summary["witnesses"]:
        require(witness["file"] in manifest, "witness is absent from verified manifest")
        tensors = load_file(str(root / witness["file"]))
        require(set(tensors) == {"reference", "candidate"}, "missing witness pair")
        reference, candidate = tensors["reference"], tensors["candidate"]
        require(
            reference.shape == candidate.shape and reference.dtype == candidate.dtype,
            "witness shape/dtype mismatch",
        )
        require(
            torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item(),
            "nonfinite witness",
        )
        relative = (reference.double() - candidate.double()).norm().item() / max(
            reference.double().norm().item(), 1e-12
        )
        require(relative > 0.01, "witness does not reproduce declared failure")
        require(
            math.isclose(relative, witness["relative_l2"], rel_tol=1e-12, abs_tol=1e-12),
            "witness numeric mismatch",
        )
        witnesses.append(
            {
                "backend": witness["backend"],
                "dtype": witness["dtype"],
                "quantity": witness["quantity"],
                "relative_l2": relative,
            }
        )
    require(loss_failures > 0 and witnesses, "no reproducible full-model failure")
    return {
        "payloads_verified": len(manifest),
        "loss_curves_verified": len(summary["table"]),
        "loss_curve_failures": loss_failures,
        "failure_witnesses": witnesses,
        "scope": "failure proof only; omitted tensor pass/fail counts are not replayed here",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print(
        {
            "payloads_verified": result["payloads_verified"],
            "loss_curves_verified": result["loss_curves_verified"],
            "loss_curve_failures": result["loss_curve_failures"],
            "witnesses_verified": len(result["failure_witnesses"]),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
