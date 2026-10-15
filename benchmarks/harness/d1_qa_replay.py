"""Offline replay of the fixed D1 QA screen; this is not timing acceptance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from d1_qa import score
from d1_rules import digest, evidence_path, write_new_json
from rich.console import Console


def audit(root: Path, expected_predictions: int = 64) -> dict:
    import torch

    records, tensors = [], []
    for run, arm in enumerate("ABBA"):
        directory = root / f"block-000-run-{run}-{arm}"
        path = directory / "result.json"
        if not path.is_file():
            raise ValueError(f"missing completed arm: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] != "MEASURED" or record["route"] != (
            "native" if arm == "A" else "forced"
        ):
            raise ValueError("unexpected arm status or route")
        losses, predictions = record["losses"], record["predictions"]
        if record["updates"] != 276 or len(losses) != 276 or not all(
            math.isfinite(value) for value in losses
        ):
            raise ValueError("missing or nonfinite training loss/update")
        if len(predictions) != expected_predictions or len(
            {row["id"] for row in predictions}
        ) != expected_predictions:
            raise ValueError("incomplete or duplicate held-out predictions")
        if score(predictions) != record["quality"]:
            raise ValueError("stored metrics differ from rescored predictions")
        loaded = {}
        for name in ("initial", "final"):
            bundle = record[name + "_adapters"]
            tensor_path = evidence_path(directory, bundle["file"])
            if digest(tensor_path) != bundle["sha256"]:
                raise ValueError("adapter SHA-256 mismatch")
            values = torch.load(tensor_path, map_location="cpu", weights_only=True)
            if not values or any(
                "lora_" not in key or not bool(torch.isfinite(value).all())
                for key, value in values.items()
            ):
                raise ValueError("empty, non-LoRA or nonfinite adapters")
            loaded[name] = values
        records.append(record)
        tensors.append(loaded)
    reference, reference_tensors = records[0], tensors[0]
    for record, values in zip(records[1:], tensors[1:]):
        for field in ("source_sha", "gpu_uuid", "settings_sha256", "losses", "predictions"):
            if record[field] != reference[field]:
                raise ValueError(f"unequal {field}")
        for name in ("initial", "final"):
            if values[name].keys() != reference_tensors[name].keys() or any(
                not torch.equal(value, reference_tensors[name][key])
                for key, value in values[name].items()
            ):
                raise ValueError(f"unequal {name} adapters")
    if reference_tensors["initial"].keys() != reference_tensors["final"].keys():
        raise ValueError("initial/final adapter names differ")
    changed = sum(
        not torch.equal(value, reference_tensors["initial"][key])
        for key, value in reference_tensors["final"].items()
    )
    if not changed:
        raise ValueError("no adapter changed during training")
    return {
        "status": "PASS",
        "arms": 4,
        "updates_per_arm": 276,
        "predictions_per_arm": expected_predictions,
        "initial_adapters_equal": True,
        "final_adapters_equal": True,
        "changed_parameters": changed,
        "losses_equal": True,
        "predictions_equal": True,
        "quality": reference["quality"],
        "performance_acceptance": False,
        "scope": "Hash-verified stored CPU tensors, losses and predictions only; "
        "not an independent reviewer, general quality guarantee or timing verdict.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    write_new_json(args.output, result)
    Console().print(result)


if __name__ == "__main__":
    main()
