"""Fail-closed numerical and artifact controls for the four-arm GPU harness."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.harness.two_read_cache_common import (
    ARMS,
    atomic_json,
    compare_captures,
    fingerprint_files,
    sha256_file,
    validate_capture,
)


def _capture() -> dict:
    import torch

    initial = {"adapter_a": torch.tensor([1.0, 2.0]), "adapter_b": torch.tensor([0.0, 0.0])}
    batches = []
    for _ in range(4):
        batches.append({
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "loss": torch.tensor(1.2341),
            "logits": torch.tensor([0.1, -0.2]),
            "gradients": {name: torch.ones_like(value) for name, value in initial.items()},
            "layer_inputs": {"layer0": torch.ones(1, 3, 2)},
            "router_logits": {"layer0": {
                "values": torch.tensor([0.2, 0.3]), "gradients": torch.tensor([0.01, 0.02]),
            }},
            "routing_weights": {"layer0": {
                "values": torch.tensor([0.4, 0.6]), "gradients": torch.tensor([0.03, 0.04]),
                "indices": torch.tensor([100, 101]),
            }},
        })
    updates = []
    for step in range(1, 3):
        updates.append({
            "adapters": {name: value + step * 0.01 for name, value in initial.items()},
            "optimizer": {
                name: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.ones_like(value) * step * 0.1,
                    "exp_avg_sq": torch.ones_like(value) * step * 0.001,
                }
                for name, value in initial.items()
            },
        })
    return {"initial_adapters": initial, "microbatches": batches, "updates": updates}


def test_complete_capture_and_bitwise_comparison_pass() -> None:
    reference = _capture()
    assert ARMS == ("ordinary", "two_read", "cache", "two_read_cache")
    validation = validate_capture(reference)
    comparison = compare_captures(reference, copy.deepcopy(reference))
    assert validation["passed"] and validation["checks"] > 0
    assert comparison["passed"] and comparison["bitwise"]
    assert comparison["checks"] > 0 and not comparison["failures"]


def test_comparator_allows_nonbitwise_values_inside_unchanged_gate() -> None:
    reference = _capture()
    candidate = copy.deepcopy(reference)
    candidate["microbatches"][0]["logits"][0] += 0.0005
    report = compare_captures(reference, candidate)
    assert report["passed"] and not report["bitwise"]


def test_reference_relative_gate_cannot_be_reversed() -> None:
    """A larger candidate must not enlarge its own 1% tolerance."""
    import torch

    reference = _capture()
    candidate = copy.deepcopy(reference)
    reference["microbatches"][0]["logits"] = torch.tensor([1.0])
    candidate["microbatches"][0]["logits"] = torch.tensor([1.01005])
    assert torch.allclose(
        reference["microbatches"][0]["logits"], candidate["microbatches"][0]["logits"],
        rtol=0.01, atol=1e-5,
    )
    report = compare_captures(reference, candidate)
    assert not report["passed"]
    assert any("reference-relative" in item["reason"] for item in report["failures"])


@pytest.mark.parametrize("value,passes", [(9e-6, True), (1.1e-5, False)])
def test_zero_reference_uses_absolute_tolerance(value: float, passes: bool) -> None:
    import torch

    reference = _capture()
    candidate = copy.deepcopy(reference)
    reference["microbatches"][0]["logits"] = torch.tensor([0.0])
    candidate["microbatches"][0]["logits"] = torch.tensor([value])
    assert compare_captures(reference, candidate)["passed"] is passes


def test_three_decimal_loss_gate_is_additional_to_pointwise_gate() -> None:
    reference = _capture()
    candidate = copy.deepcopy(reference)
    candidate["microbatches"][0]["loss"].fill_(1.2349)
    report = compare_captures(reference, candidate)
    assert not report["passed"]
    assert any("three decimal" in item["reason"] for item in report["failures"])


@pytest.mark.parametrize("field", ["initial_adapters", "input_ids"])
def test_identical_initial_state_and_inputs_are_mandatory(field: str) -> None:
    reference = _capture()
    candidate = copy.deepcopy(reference)
    if field == "initial_adapters":
        candidate[field]["adapter_a"][0] += 1e-4
    else:
        candidate["microbatches"][0][field][0, 0] += 1
    report = compare_captures(reference, candidate)
    assert not report["passed"]
    assert any("bitwise" in item["reason"] for item in report["failures"])


@pytest.mark.parametrize("side", ["reference", "candidate"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_refused_on_both_sides(side: str, bad: float) -> None:
    reference = _capture()
    candidate = copy.deepcopy(reference)
    target = reference if side == "reference" else candidate
    target["microbatches"][0]["gradients"]["adapter_a"][0] = bad
    report = compare_captures(reference, candidate)
    assert not report["passed"]
    assert any("nonfinite" in item["reason"] for item in report["failures"])


@pytest.mark.parametrize("corruption", ["shape", "dtype", "diagnostic_keys", "optimizer_keys"])
def test_structural_capture_mismatch_fails(corruption: str) -> None:
    import torch

    reference = _capture()
    candidate = copy.deepcopy(reference)
    if corruption == "shape":
        candidate["microbatches"][0]["logits"] = torch.ones(2, 1)
    elif corruption == "dtype":
        candidate["microbatches"][0]["logits"] = candidate["microbatches"][0]["logits"].double()
    elif corruption == "diagnostic_keys":
        candidate["microbatches"][0]["layer_inputs"] = {"other": torch.ones(1, 3, 2)}
    else:
        del candidate["updates"][0]["optimizer"]["adapter_a"]["exp_avg"]
    assert not compare_captures(reference, candidate)["passed"]


@pytest.mark.parametrize("missing", [
    "initial_adapters", "microbatches", "updates", "gradients", "layer_inputs",
    "router_logits", "routing_weights",
])
def test_missing_capture_or_gradient_is_never_a_pass(missing: str) -> None:
    capture = _capture()
    if missing in {"initial_adapters", "microbatches", "updates"}:
        capture[missing] = {} if missing == "initial_adapters" else []
    else:
        capture["microbatches"][0][missing] = {}
    assert not validate_capture(capture)["passed"]
    assert not compare_captures(_capture(), capture)["passed"]


def test_some_adapters_changing_does_not_hide_an_unchanged_adapter() -> None:
    capture = _capture()
    capture["updates"][-1]["adapters"]["adapter_b"] = (
        capture["initial_adapters"]["adapter_b"].clone()
    )
    report = validate_capture(capture)
    assert not report["passed"]
    assert any(item["reason"] == "adapter did not change" for item in report["failures"])


@pytest.mark.parametrize("field", ["gradients", "layer_inputs", "router_logits", "routing_weights"])
def test_none_and_empty_tensors_cannot_disguise_missing_captures(field: str) -> None:
    import torch

    for missing in (None, torch.empty(0)):
        capture = _capture()
        key = next(iter(capture["microbatches"][0][field]))
        capture["microbatches"][0][field][key] = missing
        assert not validate_capture(capture)["passed"]


def test_optimizer_step_must_advance_and_each_adapter_must_have_state() -> None:
    capture = _capture()
    capture["updates"][1]["optimizer"]["adapter_b"]["step"].fill_(1)
    assert not validate_capture(capture)["passed"]
    capture = _capture()
    del capture["updates"][1]["optimizer"]["adapter_b"]
    assert not validate_capture(capture)["passed"]


@pytest.mark.parametrize("field", ["router_logits", "routing_weights"])
def test_router_values_without_real_gradients_cannot_qualify(field: str) -> None:
    capture = _capture()
    del capture["microbatches"][0][field]["layer0"]["gradients"]
    report = validate_capture(capture)
    assert not report["passed"]
    assert any("router values and gradients" in item["reason"] for item in report["failures"])


def test_topk_expert_indices_cannot_use_floating_relative_tolerance() -> None:
    reference = _capture()
    candidate = copy.deepcopy(reference)
    candidate["microbatches"][0]["routing_weights"]["layer0"]["indices"][0] += 1
    report = compare_captures(reference, candidate)
    assert not report["passed"]
    assert any("bitwise" in item["reason"] for item in report["failures"])
    del candidate["microbatches"][0]["routing_weights"]["layer0"]["indices"]
    assert not validate_capture(candidate)["passed"]


def test_atomic_json_is_strict_and_preserves_the_last_complete_artifact(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"
    atomic_json(target, {"status": "complete", "value": 1})
    original = target.read_bytes()
    assert json.loads(original)["status"] == "complete"
    with pytest.raises(ValueError, match="JSON compliant"):
        atomic_json(target, {"value": float("nan")})
    assert target.read_bytes() == original
    assert list(target.parent.iterdir()) == [target]


def test_hashes_and_fingerprints_bind_contents_and_refuse_escape(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    first = root / "first.txt"
    first.write_bytes(b"first")
    second = root / "second.txt"
    second.write_bytes(b"second")
    expected = hashlib.sha256(b"first").hexdigest()
    assert sha256_file(first) == expected
    assert fingerprint_files([second, "first.txt"], root) == {
        "first.txt": expected, "second.txt": hashlib.sha256(b"second").hexdigest(),
    }
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes"):
        fingerprint_files([outside], root)
    with pytest.raises(ValueError, match="Duplicate"):
        fingerprint_files([first, root / ".." / "source" / "first.txt"], root)
    with pytest.raises(ValueError, match="not a file"):
        fingerprint_files([root], root)


def test_atomic_json_failed_replace_leaves_no_temporary_file(tmp_path: Path, monkeypatch) -> None:
    import benchmarks.harness.two_read_cache_common as common

    target = tmp_path / "report.json"
    atomic_json(target, {"old": True})

    def fail_replace(*_args):
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr(common.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        atomic_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [target]
