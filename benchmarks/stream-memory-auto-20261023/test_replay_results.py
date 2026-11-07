"""Tiny controls for the exact-byte replay proof; no pretrained traces or GPU."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


@pytest.fixture
def replay():
    namespace = runpy.run_path(str(Path(__file__).with_name("replay_results.py")))
    functions, _verified = namespace["archived_functions"](Path(__file__).parent / "executed")
    return namespace, functions["compare"]


def traces(tmp_path, tensors):
    from safetensors.torch import save_file

    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    save_file(tensors, str(reference / "trace.safetensors"))
    (candidate / "trace.safetensors").write_bytes((reference / "trace.safetensors").read_bytes())
    return reference, candidate


def test_identical_finite_mixed_types_and_scalar_match_original_verdict(tmp_path, replay):
    import torch

    namespace, original = replay
    tensors = {
        "batch/ids": torch.arange(6, dtype=torch.int64).reshape(2, 3),
        "batch/mask": torch.tensor([True, False]),
        "initial/adapter": torch.tensor([[0.125, -0.25]], dtype=torch.float32),
        "gradient/scalar": torch.tensor(0.5, dtype=torch.float64),
        "gradient/matrix": torch.arange(35, dtype=torch.float32).reshape(5, 7),
        "final/bfloat": torch.ones(3, 5, dtype=torch.bfloat16),
        "logits/half": torch.arange(24, dtype=torch.float16).reshape(2, 3, 4),
        "loss": torch.tensor([1.25, 2.5], dtype=torch.float64),
    }
    reference, candidate = traces(tmp_path, tensors)
    expected = original(reference, candidate)
    actual = namespace["compare_traces"](
        reference, candidate, original, max_slice_elements=7
    )
    assert all(actual[key] == value for key, value in expected.items())
    assert actual["passed"]
    assert actual["serialized_trace_proof"]["identical_full_serialized_trace"]
    assert actual["bounded_finiteness"]["finite"]
    assert actual["bounded_finiteness"]["checked_elements"] == sum(
        value.numel() for value in tensors.values()
    )
    assert actual["bounded_finiteness"]["maximum_observed_slice_elements"] <= 7


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_identical_nonfinite_files_fail_like_original(tmp_path, replay, value):
    import torch

    namespace, original = replay
    reference, candidate = traces(tmp_path, {
        "gradient/value": torch.tensor([1.0, value]), "loss": torch.tensor([1.0]),
    })
    expected = original(reference, candidate)
    actual = namespace["compare_traces"](reference, candidate, original, max_slice_elements=1)
    assert not actual["passed"]
    assert not actual["bounded_finiteness"]["finite"]
    assert actual["bounded_finiteness"]["nonfinite_tensors"] == ["gradient/value"]
    assert all(actual[key] == result for key, result in expected.items())


@pytest.mark.parametrize(("changed", "passed"), [(1.0001, True), (1.1, False)])
def test_mutated_values_use_unchanged_tolerances(tmp_path, replay, changed, passed):
    import torch
    from safetensors.torch import save_file

    namespace, original = replay
    reference, candidate = traces(tmp_path, {
        "gradient/value": torch.tensor([1.0]), "loss": torch.tensor([1.0]),
    })
    save_file({"gradient/value": torch.tensor([changed]), "loss": torch.tensor([1.0])},
              str(candidate / "trace.safetensors"))
    called = []

    def unchanged(first, second):
        called.append((first, second))
        return original(first, second)

    actual = namespace["compare_traces"](reference, candidate, unchanged)
    expected = original(reference, candidate)
    assert not actual["serialized_trace_proof"]["identical_full_serialized_trace"]
    assert called == [(reference, candidate)]
    assert actual["passed"] is passed
    assert all(actual[key] == result for key, result in expected.items())


def test_trace_requires_original_loss_key(tmp_path, replay):
    import torch

    namespace, original = replay
    reference, candidate = traces(tmp_path, {"gradient/value": torch.tensor([1.0])})
    with pytest.raises(ValueError, match="required loss key"):
        namespace["compare_traces"](reference, candidate, original)


def test_archive_fallback_keeps_exact_bytes_and_refuses_outside_paths(tmp_path, monkeypatch):
    import hashlib
    import zipfile

    namespace = runpy.run_path(str(Path(__file__).with_name("replay_results.py")))
    report = tmp_path / "report"
    report.mkdir()
    path = report / "executed" / "helper.py.txt"
    path.parent.mkdir()
    payload = b"# exact archival bytes\r\nvalue = 7\r\n"
    path.write_bytes(payload)
    monkeypatch.setitem(namespace["read_bytes"].__globals__, "__file__", str(report / "replay.py"))
    with zipfile.ZipFile(report / "evidence.zip", "w") as handle:
        handle.writestr("executed/helper.py.txt", payload)
    direct = namespace["read_bytes"](path)
    path.unlink()  # Temporary test fixture only; real evidence is never removed here.
    archived = namespace["read_bytes"](path)
    assert archived == direct == payload
    assert hashlib.sha256(archived).hexdigest() == hashlib.sha256(direct).hexdigest()
    with pytest.raises(FileNotFoundError, match="outside the archival report root"):
        namespace["read_bytes"](tmp_path / "outside" / "missing.json")
