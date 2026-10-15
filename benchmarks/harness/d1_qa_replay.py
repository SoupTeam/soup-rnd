"""Offline replay of the fixed D1 QA screen; this is not timing acceptance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from d1_qa import load_split, score
from d1_rules import digest, evidence_path, write_new_json
from rich.console import Console


def metric_difference(left: dict, right: dict) -> float:
    if left.keys() != right.keys():
        raise ValueError("stored metric keys differ from rescored predictions")
    differences = []
    for key, value in left.items():
        other = right[key]
        if isinstance(value, dict) and isinstance(other, dict):
            differences.append(metric_difference(value, other))
        elif isinstance(value, float) and isinstance(other, float):
            # Python 3.12 changed float summation; this does not relax tensor equality.
            if not math.isclose(value, other, abs_tol=1e-12, rel_tol=0):
                raise ValueError("stored metrics differ from rescored predictions")
            differences.append(abs(value - other))
        elif type(value) is not type(other) or value != other:
            raise ValueError("stored metric counts/values differ from rescored predictions")
    return max(differences, default=0.0)


def audit(
    root: Path,
    expected_predictions: int = 64,
    gold: list[dict] | None = None,
    block_ids: list[int] | None = None,
) -> dict:
    import torch

    block_ids = [0] if block_ids is None else block_ids
    if not block_ids or len(set(block_ids)) != len(block_ids) or any(
        block < 0 for block in block_ids
    ):
        raise ValueError("block IDs must be unique nonnegative integers")
    records, tensors, metric_differences = [], [], []
    for block, run, arm in (
        (block, run, arm) for block in block_ids for run, arm in enumerate("ABBA")
    ):
        matches = list(root.rglob(f"block-{block:03d}-run-{run}-{arm}/result.json"))
        if len(matches) != 1:
            raise ValueError(f"missing or duplicate completed arm: block {block}, run {run}")
        path = matches[0]
        directory = path.parent
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
        if gold is not None:
            observed_gold = [
                (row["id"], row["answers"], row["unanswerable"]) for row in predictions
            ]
            expected_gold = [
                (row["id"], [answer["text"] for answer in row["answers"]], row["unanswerable"])
                for row in gold
            ]
            if observed_gold != expected_gold:
                raise ValueError("predictions differ from the pinned held-out gold")
        metric_differences.append(metric_difference(score(predictions), record["quality"]))
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
        "arms": 4 * len(block_ids),
        "block_ids": block_ids,
        "updates_per_arm": 276,
        "predictions_per_arm": expected_predictions,
        "initial_adapters_equal": True,
        "final_adapters_equal": True,
        "changed_parameters": changed,
        "losses_equal": True,
        "predictions_equal": True,
        "gold_verified": gold is not None,
        "metric_rounding_max_abs": max(metric_differences),
        "quality": reference["quality"],
        "performance_acceptance": False,
        "scope": "Hash-verified stored CPU tensors, losses and predictions only; "
        "not an independent reviewer, general quality guarantee or timing verdict.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-ids", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    result = audit(args.input, gold=load_split(args.data, "test"), block_ids=args.block_ids)
    write_new_json(args.output, result)
    Console().print(result)


if __name__ == "__main__":
    main()
