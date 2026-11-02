"""Replay every stored projection against its unpatched native tensor oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from d1_rules import compare_tensors, digest, evidence_path, write_new_json
from rich.console import Console


def audit(root: Path) -> dict:
    import torch

    rows = []
    for path in sorted(root.rglob("projection.pt")):
        directory = path.parent
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        loaded = {}
        for name in ("tensors", "native_reference_tensors"):
            bundle = record[name]
            tensor_path = evidence_path(directory, bundle["file"])
            if digest(tensor_path) != bundle["sha256"]:
                raise ValueError("projection tensor SHA-256 mismatch")
            loaded[name] = torch.load(tensor_path, map_location="cpu", weights_only=True)
        compared = compare_tensors(loaded["tensors"], loaded["native_reference_tensors"])
        if compared != record["native_comparison"]:
            raise ValueError("stored projection comparison differs from replay")
        allow_dense_difference = record["route"] == "dense" and record["native_fused"]
        if not compared["passed"] and not allow_dense_difference:
            raise ValueError("candidate differs from unpatched native tensors")
        if any(not row["finite"] for row in compared["quantities"]):
            raise ValueError("nonfinite projection witness")
        rows.append(
            {
                "file": path.relative_to(root).as_posix(),
                "route": record["route"],
                "exact_native": compared["passed"],
                "declared_dense_fused_difference": bool(
                    allow_dense_difference and not compared["passed"]
                ),
            }
        )
    if not rows:
        raise ValueError("no projection tensor witnesses")
    return {
        "status": "PASS",
        "arms": len(rows),
        "exact_native_arms": sum(row["exact_native"] for row in rows),
        "dense_fused_differences": sum(row["declared_dense_fused_difference"] for row in rows),
        "timing_acceptance": False,
        "rows": rows,
        "scope": "Stored hash-verified finite tensor comparisons only; prior void timing "
        "stays void, and declared dense/fused arithmetic differences stay different.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    write_new_json(args.output, result)
    Console().print({key: value for key, value in result.items() if key != "rows"})


if __name__ == "__main__":
    main()
