"""CPU cache-profile evidence: actual autograd order and measured allocation work."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.harness.expert_cache_cpu_profile import (
    extract_route,
    payload_sha256,
    reconstruct_visits,
    run_experiment,
    staging_entry,
)


def _route(path: Path) -> Path:
    path.write_text(json.dumps({
        "layout": "two_read", "source_capture_sha256": "a" * 64,
        "visits": [[1, 0], [1, 1], [2, 0], [2, 1]] * 2,
    }), encoding="utf-8")
    return path


def test_reconstructed_visits_equal_real_native_fetch_order(tmp_path: Path) -> None:
    """Qualify order against native execution, not a manually repeated source list."""
    import torch

    from benchmarks.harness.two_read_cache_worker import _run_accuracy
    from tests.test_expert_stream import _three_way

    reference, ordinary, two, ids = _three_way(tmp_path, "float32")
    del reference
    batches = torch.stack((ids[:1], ids[1:], ids[1:], ids[:1]))
    try:
        for layout, (model, runtime) in (("ordinary", ordinary), ("two_read", two)):
            actual = []
            original = runtime.expert_fetcher.fetch

            def fetch(reader, expert):
                layer = int(re.search(r"experts_(\d+)\.safetensors$", reader.path).group(1))
                actual.append((layer, expert))
                return original(reader, expert)

            runtime.expert_fetcher.fetch = fetch
            capture = _run_accuracy(model, runtime, batches, {"accuracy_updates": 2,
                                                            "accumulation": 2})
            assert reconstruct_visits(capture, layout) == actual
    finally:
        ordinary[1].close()
        two[1].close()


def test_extraction_binds_source_and_validates_indices(tmp_path: Path) -> None:
    import torch

    capture = {"microbatches": [{"routing_weights": {
        "1": {"indices": torch.tensor([[3, 1], [2, 1]], dtype=torch.int64)},
    }}]}
    source, output = tmp_path / "capture.pt", tmp_path / "route.json"
    torch.save(capture, source)
    result = extract_route(source, "two_read", output)
    assert result["visits"] == [(1, 1), (1, 2), (1, 3), (1, 3), (1, 2), (1, 1)]
    assert len(result["source_capture_sha256"]) == 64
    assert json.loads(output.read_text())["microbatches"] == 1
    capture["microbatches"][0]["routing_weights"]["1"]["indices"] = torch.tensor([1.])
    with pytest.raises(ValueError, match="indices"):
        reconstruct_visits(capture, "ordinary")


def test_real_moonlight_shape_payload_and_reusable_alias() -> None:
    import torch

    from soup_cli.utils.expert_cache import TensorLRU

    owner, entry = staging_entry(2048, 1408)
    assert len(entry) == 8
    assert sum(t.numel() * t.element_size() for t in entry.values()) == 4_462_664
    assert owner.numel() == 4_462_664
    assert all(t.device.type == "cpu" and not t.is_pinned() for t in entry.values())
    for tensor in entry.values():
        tensor.fill_(1)
    cache = TensorLRU(owner.numel(), copy_on_insert=True)
    assert cache.put((1, 0), entry)
    stored = cache.get((1, 0))
    identity = payload_sha256(stored)
    owner.fill_(255)
    assert payload_sha256(stored) == identity
    assert not torch.equal(stored["gate_up"], entry["gate_up"])


@pytest.mark.parametrize("policy", ("lru", "static"))
@pytest.mark.parametrize("mode", ("timing", "profile"))
def test_small_store_profile_measures_real_clones(tmp_path: Path, policy: str, mode: str) -> None:
    _, entry = staging_entry(16, 8)
    cost = sum(t.numel() * t.element_size() for t in entry.values())
    result = run_experiment(
        _route(tmp_path / "visits.json"), tmp_path / "out", policy=policy,
        budget_bytes=cost * 2, hidden=16, intermediate=8, mode=mode,
    )
    admissions = 8 if policy == "lru" else 2
    assert result["counts"]["admissions"] == admissions
    assert result["clone_calls_derived"] == admissions * 8
    assert result["cloned_payload_bytes_derived"] == admissions * cost
    assert result["cache_stats"]["payload_bytes"] <= 2 * cost
    assert result["immutable_after_staging_mutation"]
    assert result["cache_stats"]["hits"] == result["counts"]["hits"] + 1
    if mode == "profile":
        assert result["counts"]["clone_calls_observed"] == admissions * 8
        assert result["counts"]["copy_bytes_observed"] == admissions * cost
        operators = {item["operator"]: item for item in result["operators"]}
        assert operators["aten::clone"]["calls"] == admissions * 8
        assert operators["aten::copy_"]["calls"] >= admissions * 8
        assert (tmp_path / "out/cpu-trace.json").is_file()
    else:
        assert result["operators"] == []


def test_device_hotset_is_only_a_cpu_metadata_filter(tmp_path: Path, monkeypatch) -> None:
    import torch

    def no_cuda(*_args, **_kwargs):
        raise AssertionError("CPU experiment must not use CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", no_cuda)
    result = run_experiment(
        _route(tmp_path / "visits.json"), tmp_path / "out", policy="lru",
        budget_bytes=0, hidden=16, intermediate=8, device_hotset_bytes=3 * 16 * 8 * 2,
        repeats=2,
    )
    assert result["modeled_device_entries"] == 1
    assert result["counts"]["device_hotset_visits"] == 3
    assert result["counts"]["admissions"] == 0
    assert result["clone_calls_derived"] == 0
    assert result["visits"] == 16


def test_profile_ownership_check_rejects_unsafe_reference_store(
    tmp_path: Path, monkeypatch,
) -> None:
    from benchmarks.harness import expert_cache_cpu_profile as profile
    from soup_cli.utils.expert_cache import TensorLRU

    monkeypatch.setattr(profile, "_store", lambda _policy, budget: TensorLRU(
        budget, copy_on_insert=False,
    ))
    with pytest.raises(RuntimeError, match="aliases mutable"):
        run_experiment(
            _route(tmp_path / "visits.json"), tmp_path / "out", policy="lru",
            budget_bytes=1024, hidden=16, intermediate=8,
        )
