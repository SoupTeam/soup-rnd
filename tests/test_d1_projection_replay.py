"""CPU witnesses must detect a false projection equality claim."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


def fixture(root, route, equal=True):
    import torch
    from d1_rules import compare_tensors, digest

    directory = root / "projection-8/block-000-run-0-A"
    directory.mkdir(parents=True)
    candidate = {"output": torch.tensor([1.0]), "grad_x": torch.tensor([2.0])}
    reference = {"output": torch.tensor([1.0 if equal else 3.0]), "grad_x": torch.tensor([2.0])}
    record = {
        "route": route,
        "native_fused": True,
        "native_comparison": compare_tensors(candidate, reference),
    }
    for field, name, values in (
        ("tensors", "projection.pt", candidate),
        ("native_reference_tensors", "native-reference.pt", reference),
    ):
        torch.save(values, directory / name)
        record[field] = {"file": name, "sha256": digest(directory / name)}
    (directory / "result.json").write_text(json.dumps(record))
    return directory


def test_native_equal_and_declared_dense_difference_are_distinct(tmp_path):
    from d1_projection_replay import audit

    fixture(tmp_path, "dense", equal=False)
    result = audit(tmp_path)
    assert result["status"] == "PASS"
    assert result["dense_fused_differences"] == 1
    assert result["exact_native_arms"] == 0
    assert result["timing_acceptance"] is False


@pytest.mark.parametrize("failure", ["native_difference", "tampered", "false_report"])
def test_rejects_changed_or_falsely_reported_projection(tmp_path, failure):
    from d1_projection_replay import audit

    directory = fixture(tmp_path, "native", equal=failure != "native_difference")
    if failure == "tampered":
        with (directory / "projection.pt").open("ab") as handle:
            handle.write(b"changed")
    elif failure == "false_report":
        path = directory / "result.json"
        record = json.loads(path.read_text())
        record["native_comparison"]["passed"] = False
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit(tmp_path)
