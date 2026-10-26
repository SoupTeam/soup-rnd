"""Opt-in route geometry and complete worker trajectories, with unchanged gates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _mask(kind: str) -> Any:
    import torch
    import torch.nn.functional as functional

    pool = [0, 3, 8, 17, 64, 77, 103, 127] if kind == "subset" else list(range(128))
    scores = torch.rand((512, len(pool)), generator=torch.Generator().manual_seed(29))
    indices = torch.tensor(pool)[scores.topk(6, dim=-1).indices]
    if kind == "empty":
        indices = torch.empty((0, 6), dtype=torch.int64)
    elif kind == "duplicates":
        indices = torch.tensor([[3, 3, 17, 17, 17, 127]] * 512, dtype=torch.int64)
    mask = functional.one_hot(indices, num_classes=128).permute(2, 1, 0)
    if kind == "contiguous":
        return mask.contiguous()
    if kind == "strided":
        padded = torch.zeros((128, 6, 1024), dtype=torch.int64)
        padded[..., ::2] = mask
        return padded[..., ::2]
    return mask


@pytest.mark.parametrize("kind", ["contiguous", "permuted", "strided", "subset", "empty",
                                  "duplicates"])
def test_grouped_routes_preserve_exact_legacy_expert_top_slot_and_token_order(kind: str) -> None:
    import torch

    from soup_cli.utils.expert_stream import _grouped_expert_routes, _legacy_expert_routes

    mask = _mask(kind)
    hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    old = list(_legacy_expert_routes(mask, hits))
    new = list(_grouped_expert_routes(mask))
    assert [int(row[0]) for row in old] == [row[0] for row in new]
    assert all(type(row[0]) is int for row in new)
    assert all(torch.equal(left[1], right[1]) and torch.equal(left[2], right[2])
               for left, right in zip(old, new))
    assert sum(row[1].numel() for row in new) == int(mask.sum())
    if kind in {"permuted", "strided", "subset", "duplicates"}:
        assert not mask.is_contiguous()
    if kind == "subset":
        assert [row[0] for row in new] == [0, 3, 8, 17, 64, 77, 103, 127]


def test_grouped_route_views_are_immutable_per_call_and_outlive_the_generator() -> None:
    import gc

    import torch

    from soup_cli.utils.expert_stream import _grouped_expert_routes

    mask = _mask("permuted")
    first = list(_grouped_expert_routes(mask))
    expected = [(row[1].clone(), row[2].clone()) for row in first]
    storage = {row[2].untyped_storage().data_ptr() for row in first}
    second = list(_grouped_expert_routes(mask))
    assert len(storage) == 1
    assert storage.isdisjoint({row[2].untyped_storage().data_ptr() for row in second})
    del second
    gc.collect()
    assert all(torch.equal(row[1], value[0]) and torch.equal(row[2], value[1])
               for row, value in zip(first, expected))


def test_grouped_routes_remove_per_expert_scalar_and_where_cpu_operations() -> None:
    import torch

    from soup_cli.utils.expert_stream import _grouped_expert_routes, _legacy_expert_routes

    mask = _mask("permuted")
    hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    counts, visited = {}, {}
    for name, routes in (("legacy", _legacy_expert_routes(mask, hits)),
                         ("grouped", _grouped_expert_routes(mask))):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            visited[name] = [(int(expert), positions, tokens)
                             for expert, positions, tokens in routes]
        counts[name] = {event.key: event.count for event in profile.key_averages()}
    assert [row[0] for row in visited["legacy"]] == [row[0] for row in visited["grouped"]]
    assert counts["legacy"]["aten::_local_scalar_dense"] == 384
    assert counts["legacy"]["aten::where"] == 128
    assert counts["grouped"].get("aten::_local_scalar_dense", 0) == 0
    assert counts["grouped"].get("aten::where", 0) == 0
    assert counts["grouped"]["aten::nonzero"] == 1


@pytest.mark.parametrize("two_reads", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_actual_streamed_expert_outputs_and_gradients_stay_bitwise(
    tmp_path: Path, two_reads: bool, autocast: bool,
) -> None:
    import torch

    from soup_cli.utils.expert_stream import ExpertFetcher, streamed_experts_class
    from tests.test_expert_stream import _deepseek_v3_experts, _routing, _streamed_copy

    resident, experts, slots = _deepseek_v3_experts(5)
    baseline, reader = _streamed_copy(tmp_path, resident, experts, two_reads)
    candidate = streamed_experts_class()(
        reader, ExpertFetcher(device="cpu", dtype="float32"), resident.act_fn, 1,
        two_reads=two_reads, grouped_expert_routes=True,
    )
    states, indices, weights, grad_out = _routing(24, experts, slots, 32, 9)
    captures = []
    try:
        assert baseline.grouped_expert_routes is False
        assert candidate.grouped_expert_routes is True
        for module in (baseline, candidate):
            local = states.detach().clone().requires_grad_(True)
            routing = weights.detach().clone().requires_grad_(True)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                output = module(local, indices, routing)
            output.backward(grad_out)
            captures.append((output.detach(), local.grad, routing.grad))
        assert all(torch.equal(left, right) for left, right in zip(*captures))
        assert torch.count_nonzero(captures[0][1]).item() > 0
        assert torch.count_nonzero(captures[0][2]).item() > 0
    finally:
        baseline.fetcher.close()
        candidate.fetcher.close()
        reader.close()


@pytest.mark.parametrize("two_reads", [False, True])
def test_grouped_route_storage_survives_two_forwards_and_retained_backwards(
    tmp_path: Path, two_reads: bool,
) -> None:
    import torch

    from soup_cli.utils.expert_stream import ExpertFetcher, streamed_experts_class
    from tests.test_expert_stream import _deepseek_v3_experts, _routing, _streamed_copy

    resident, experts, slots = _deepseek_v3_experts(5)
    baseline, reader = _streamed_copy(tmp_path, resident, experts, two_reads)
    candidate = streamed_experts_class()(
        reader, ExpertFetcher(device="cpu", dtype="float32"), resident.act_fn, 1,
        two_reads=two_reads, grouped_expert_routes=True,
    )
    states, indices, weights, grad_out = _routing(24, experts, slots, 32, 9)
    results = []
    try:
        for module in (baseline, candidate):
            local = states.detach().clone().requires_grad_(True)
            routing = weights.detach().clone().requires_grad_(True)
            first = module(local, indices, routing)
            second = module(local * 1.1, indices, routing)
            first.backward(grad_out, retain_graph=True)
            second.backward(grad_out)
            first.backward(grad_out)
            results.append((local.grad, routing.grad))
        assert all(torch.equal(left, right) for left, right in zip(*results))
    finally:
        baseline.fetcher.close()
        candidate.fetcher.close()
        reader.close()


def test_default_expert_loop_never_calls_the_opt_in_grouped_helper(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    import torch

    from soup_cli.utils import expert_stream
    from tests.test_expert_stream import _deepseek_v3_experts, _routing, _streamed_copy

    resident, experts, slots = _deepseek_v3_experts(5)
    baseline, reader = _streamed_copy(tmp_path, resident, experts)
    monkeypatch.setattr(expert_stream, "_grouped_expert_routes",
                        lambda *_args: pytest.fail("default entered opt-in dispatcher"))
    states, indices, weights, _ = _routing(24, experts, slots, 32, 9)
    try:
        assert baseline.grouped_expert_routes is False
        with torch.no_grad():
            actual = baseline(states, indices, weights)
            assert torch.equal(actual, resident(states, indices, weights))
    finally:
        baseline.fetcher.close()
        reader.close()


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_grouped_route_api_rejects_non_boolean_before_model_build(value: Any) -> None:
    from soup_cli.utils.expert_stream import install_expert_streaming
    from soup_cli.utils.layer_stream_runtime import build_streamed_model, install_streaming

    with pytest.raises(ValueError, match="must be a bool"):
        build_streamed_model(model_id="unused", shard_dir="unused", index=None,
                             lora_config=None, grouped_expert_routes=value)
    with pytest.raises(ValueError, match="must be a bool"):
        install_streaming(None, shard_dir="unused", index=None, grouped_expert_routes=value)
    with pytest.raises(ValueError, match="must be a bool"):
        install_expert_streaming([], {}, None, grouped_expert_routes=value)


def test_grouped_routes_refuse_fused_layout_without_opening_readers(monkeypatch: Any) -> None:
    from soup_cli.utils import expert_stream
    from soup_cli.utils.layer_stream_runtime import _install_expert_streaming

    monkeypatch.setattr(expert_stream, "ExpertShardReader",
                        lambda *_args, **_kwargs: pytest.fail("reader opened"))
    with pytest.raises(ValueError, match="per_expert"):
        _install_expert_streaming([], "unused", SimpleNamespace(expert_layout=""),
                                  device="cpu", codes=None, grouped_expert_routes=True)


def _full_worker_comparison(
    tmp_path: Path, monkeypatch: Any, *, dtype: str, quant: str, device: str,
) -> dict[str, Any]:
    import torch

    from benchmarks.harness.two_read_cache_common import compare_captures
    from soup_cli.utils import layer_stream_runtime
    from tests.test_two_read_cache_full_query import _full_query_trajectories
    from tests.test_two_read_cache_kanana import _kanana_overrides

    reports = {}
    original = layer_stream_runtime.build_streamed_model
    for mode, grouped in (("legacy", False), ("grouped", True)):
        modes = []

        def construct(**kwargs: Any) -> Any:
            model, runtime = original(**kwargs, grouped_expert_routes=grouped)
            actual = runtime.stats()["expert_grouped_routes"]
            assert set(actual) == {"1", "2"} and all(value is grouped for value in actual.values())
            assert all(module.grouped_expert_routes is grouped for module in runtime.expert_modules)
            modes.append(actual)
            return model, runtime

        with monkeypatch.context() as context:
            context.setattr(layer_stream_runtime, "build_streamed_model", construct)
            reports[mode] = _full_query_trajectories(
                tmp_path / mode, dtype=dtype, quant=quant, device=device, bitwise=device == "cpu",
                host_cache_policy="static", config_overrides=_kanana_overrides(),
                evidence_dir=tmp_path / mode / "evidence",
            )
        assert len(modes) == 2
    reference = torch.load(tmp_path / "legacy/evidence/ordinary.pt",
                           map_location="cpu", weights_only=True)
    comparisons = {}
    for arm in ("ordinary", "two_read", "cache", "two_read_cache"):
        capture = torch.load(tmp_path / "grouped/evidence" / (arm + ".pt"),
                             map_location="cpu", weights_only=True)
        assert len(capture["updates"]) == 2 and len(capture["microbatches"]) == 4
        comparisons[arm] = compare_captures(reference, capture)
        assert comparisons[arm]["passed"], (arm, comparisons[arm])
        if device == "cpu":
            assert comparisons[arm]["bitwise"], (arm, comparisons[arm])
        legacy = torch.load(tmp_path / "legacy/evidence" / (arm + ".pt"),
                            map_location="cpu", weights_only=True)
        same_arm = compare_captures(legacy, capture)
        assert same_arm["passed"] and (same_arm["bitwise"] or device != "cpu")
    return {"reports": reports, "grouped_versus_ordinary_legacy": comparisons}


@pytest.mark.parametrize("dtype,quant", [("float32", "none"), ("bfloat16", "none"),
                                        ("float32", "nf4")])
def test_grouped_routes_preserve_full_cpu_gradient_and_adamw_trajectories(
    tmp_path: Path, monkeypatch: Any, dtype: str, quant: str,
) -> None:
    _full_worker_comparison(tmp_path, monkeypatch, dtype=dtype, quant=quant, device="cpu")


@pytest.mark.gpu(reason="Actual grouped routing, BF16 and NF4 trajectories require CUDA")
def test_real_cuda_nf4_bf16_grouped_route_worker_trajectories(
    tmp_path: Path, monkeypatch: Any, record_property: Any,
) -> None:
    import torch

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result = _full_worker_comparison(tmp_path, monkeypatch, dtype="bfloat16", quant="nf4",
                                         device="cuda")
        record_property("grouped_route_worker_trajectories", json.dumps(result, sort_keys=True))
        record_property("grouped_route_evidence_directory", str(tmp_path))
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
