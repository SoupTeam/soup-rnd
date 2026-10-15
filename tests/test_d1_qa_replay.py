"""Offline QA evidence must detect altered tensors and unequal training."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


def fixture(tmp_path):
    import torch
    from d1_qa import score

    predictions = [
        {"id": "one", "answers": ["Paris"], "prediction": "Paris", "unanswerable": False},
        {"id": "two", "answers": [], "prediction": "NO_ANSWER", "unanswerable": True},
    ]
    for run, arm in enumerate("ABBA"):
        root = tmp_path / f"block-000-run-{run}-{arm}"
        root.mkdir()
        record = {
            "status": "MEASURED",
            "source_sha": "same-source",
            "gpu_uuid": "same-gpu",
            "settings_sha256": "same-settings",
            "route": "native" if arm == "A" else "forced",
            "updates": 276,
            "losses": [1.0] * 276,
            "predictions": predictions,
            "quality": score(predictions),
        }
        for name, value in (("initial", 0.0), ("final", 1.0)):
            path = root / f"{name}-adapters.pt"
            torch.save({"layer.lora_A.weight": torch.tensor([value])}, path)
            record[f"{name}_adapters"] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (root / "result.json").write_text(json.dumps(record))
    return tmp_path


def test_complete_finite_equal_training_replays(tmp_path):
    from d1_qa_replay import audit

    result = audit(fixture(tmp_path), expected_predictions=2)
    assert result["status"] == "PASS"
    assert result["final_adapters_equal"]
    assert result["changed_parameters"] == 1
    assert result["performance_acceptance"] is False


def test_matching_predictions_cannot_replace_the_held_out_gold(tmp_path):
    from d1_qa import score
    from d1_qa_replay import audit

    root = fixture(tmp_path)
    for path in root.glob("block-*/result.json"):
        record = json.loads(path.read_text())
        record["predictions"][0]["answers"] = ["London"]
        record["predictions"][0]["prediction"] = "London"
        record["quality"] = score(record["predictions"])
        path.write_text(json.dumps(record))
    gold = [
        {"id": "one", "answers": [{"text": "Paris"}], "unanswerable": False},
        {"id": "two", "answers": [], "unanswerable": True},
    ]
    with pytest.raises(ValueError, match="gold"):
        audit(root, expected_predictions=2, gold=gold)


def test_metric_float_rounding_does_not_change_the_tensor_equality_gate(tmp_path):
    from d1_qa_replay import audit

    root = fixture(tmp_path)
    path = root / "block-000-run-1-B/result.json"
    record = json.loads(path.read_text())
    record["quality"]["f1"] -= 1e-14
    path.write_text(json.dumps(record))
    result = audit(root, expected_predictions=2)
    assert result["status"] == "PASS"
    assert 0 < result["metric_rounding_max_abs"] < 1e-12


@pytest.mark.parametrize("change", ["loss", "prediction", "quality", "tensor", "missing"])
def test_audit_rejects_changed_or_incomplete_evidence(tmp_path, change):
    from d1_qa_replay import audit

    root = fixture(tmp_path)
    arm = root / "block-000-run-1-B"
    path = arm / "result.json"
    record = json.loads(path.read_text())
    if change == "tensor":
        with (arm / "final-adapters.pt").open("ab") as handle:
            handle.write(b"tampered")
    elif change == "missing":
        path.unlink()
    else:
        if change == "loss":
            record["losses"][20] = 2.0
        elif change == "prediction":
            record["predictions"] = copy.deepcopy(record["predictions"])
            record["predictions"][0]["prediction"] = "London"
        else:
            record["quality"]["exact_match"] = 0
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit(root, expected_predictions=2)
