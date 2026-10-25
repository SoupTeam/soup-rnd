"""Local controls for the CUDA worker's capture and failure-retention paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness.two_read_cache_worker import (
    ResourceSampler,
    _adapter_values,
    _adapters,
    _enforce_stage_limits,
    _finite_training_state,
    _numeric_delta,
    _proc_fields,
    _run_accuracy,
    _runtime_snapshot,
    run_worker,
)


@pytest.mark.parametrize("dtype", ("float32", "bfloat16"))
def test_worker_capture_qualifies_all_four_cpu_semantic_arms(tmp_path: Path, dtype: str) -> None:
    """Exercise actual instrumentation, routing replay, gradients and AdamW state."""
    from tests.test_expert_stream import _three_way

    resident, (ordinary, default_runtime), (two, two_runtime), ids = _three_way(
        tmp_path, dtype,
    )
    del resident
    _qualify_captures(ordinary, default_runtime, two, two_runtime, ids)


def test_worker_capture_qualifies_real_nf4_four_arm_cpu_trajectories(tmp_path: Path) -> None:
    """New worker hooks must also preserve actual NF4 materialization and replay."""
    import torch

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes
    from tests.test_expert_stream import _shard, _tiny_deepseek_v3
    from tests.test_stream_expert_cache_nf4 import _build

    weights, resident = _tiny_deepseek_v3(tmp_path)
    del resident  # Reference is the same NF4 base, never the unquantized checkpoint.
    skeleton = build_meta_skeleton(weights, dtype="float32", quant="nf4")
    suffixes = quantised_layer_suffixes(skeleton)
    del skeleton
    index = _shard(
        tmp_path, weights, quant="nf4", expert_layout="per_expert",
        quant_suffixes=suffixes, quant_device="cpu",
    )
    ordinary, default_runtime = _build(
        weights, index, str(tmp_path / "shards"), two_read=False, device="cpu",
    )
    try:
        two, two_runtime = _build(
            weights, index, str(tmp_path / "shards"), two_read=True, device="cpu",
        )
    except BaseException:
        default_runtime.close()
        raise
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(29))
    _qualify_captures(ordinary, default_runtime, two, two_runtime, ids)


def _qualify_captures(
    ordinary: Any, default_runtime: Any, two: Any, two_runtime: Any, ids: Any,
) -> None:
    import torch

    from benchmarks.harness.two_read_cache_common import compare_captures

    spec = {"accuracy_updates": 2, "accumulation": 2}
    batches = torch.stack((ids[:1], ids[1:], ids[1:], ids[:1]))
    initial = _adapter_values(ordinary)
    reference = None
    try:
        for name, model, runtime, cache in (
            ("ordinary", ordinary, default_runtime, False),
            ("two_read", two, two_runtime, False),
            ("cache", ordinary, default_runtime, True),
            ("two_read_cache", two, two_runtime, True),
        ):
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                adapters = _adapters(model)
                assert adapters.keys() == initial.keys()
                for key, parameter in adapters.items():
                    parameter.copy_(initial[key])
            runtime.expert_fetcher.configure_cache(
                host_cache_bytes=2**20 if cache else 0,
                device_cache_bytes=2**15 if cache else 0,
            )
            capture = _run_accuracy(model, runtime, batches, spec)
            assert len(capture["microbatches"]) == 4
            for microbatch in capture["microbatches"]:
                assert set(microbatch["layer_inputs"]) == {"1", "2"}
                for kind in ("router_logits", "routing_weights"):
                    assert set(microbatch[kind]) == {"1", "2"}
                    expected = {"values", "gradients"}
                    if kind == "routing_weights":
                        expected.add("indices")
                    assert all(set(item) == expected
                               for item in microbatch[kind].values())
            if reference is None:
                reference = capture
            result = compare_captures(reference, capture)
            assert result["passed"], (name, result)
            assert result["bitwise"], (name, result)
            snapshot = _runtime_snapshot(runtime)
            assert len(snapshot["expert_reader_direct_io"]) == 2
            if cache:
                assert snapshot["expert_cache"]["host"]["hits"] > 0
                assert snapshot["expert_cache"]["device"]["hits"] > 0
    finally:
        default_runtime.close()
        two_runtime.close()


def test_worker_retains_cuda_refusal_as_failed_result(tmp_path: Path, monkeypatch) -> None:
    import torch

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"dtype": "bfloat16", "quant": "nf4"}), encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "failed"
    assert run_worker(spec, "ordinary", "accuracy", output) == 1
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "real BF16-capable CUDA" in result["error"]
    assert "Traceback" in result["traceback"]
    assert result["spec_sha256"]
    assert not (output / "capture.pt").exists()
    with pytest.raises(FileExistsError):
        run_worker(spec, "ordinary", "accuracy", output)


def test_process_metrics_keep_missing_unavailable_and_payload_separate(tmp_path: Path) -> None:
    fixture = tmp_path / "status"
    fixture.write_text("VmRSS:\t12 kB\nread_bytes: 150\nname: ignored\n", encoding="utf-8")
    assert _proc_fields(fixture) == {"VmRSS": 12 * 1024, "read_bytes": 150}
    assert _proc_fields(tmp_path / "absent") == {}
    delta = _numeric_delta(
        {"logical_fetches": 12, "storage_bytes_read": 100, "host": {"hits": 3}},
        {"logical_fetches": 20, "storage_bytes_read": 110, "host": {"hits": 8}},
    )
    assert delta == {"logical_fetches": 8, "storage_bytes_read": 10, "host": {"hits": 5}}


def test_memory_sampler_preserves_stage_boundaries(monkeypatch) -> None:
    from benchmarks.harness import two_read_cache_worker as worker

    snapshots = iter([
        {"rss_bytes": 100, "process_swap_bytes": None, "process_io": {"read_bytes": 10}},
        {"rss_bytes": 130, "process_swap_bytes": None, "process_io": {"read_bytes": 12}},
        {"rss_bytes": 120, "process_swap_bytes": None, "process_io": {"read_bytes": 13}},
        {"rss_bytes": 110, "process_swap_bytes": None, "process_io": {"read_bytes": 30}},
    ])
    monkeypatch.setattr(worker, "_resources", lambda: next(snapshots))
    sampler = ResourceSampler()
    sampler._sample()
    result = sampler.finish()
    assert result["sampled_peak_rss_bytes"] == 130
    assert result["sampled_peak_process_swap_bytes"] is None
    assert result["process_io_delta"]["read_bytes"] == 20
    assert result["samples"] == 2


@pytest.mark.parametrize("corrupt", ("early_loss", "gradient", "adapter", "moment", "missing"))
def test_post_update_finite_gate_checks_every_training_component(corrupt: str) -> None:
    import torch

    model = torch.nn.Module()
    model.lora_A = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    model.lora_A.grad = torch.ones(3)
    optimizer.step()
    losses = [torch.tensor(1.), torch.tensor(1.)]
    _finite_training_state(model, optimizer, losses)
    if corrupt == "early_loss":
        losses[0].fill_(float("nan"))
    elif corrupt == "gradient":
        model.lora_A.grad[0] = float("nan")
    elif corrupt == "adapter":
        with torch.no_grad():
            model.lora_A[0] = float("inf")
    elif corrupt == "moment":
        optimizer.state[model.lora_A]["exp_avg_sq"][0] = float("nan")
    else:
        model.lora_A.grad = None
    with pytest.raises(RuntimeError, match="non-finite|no gradient"):
        _finite_training_state(model, optimizer, losses)


@pytest.mark.parametrize(
    "field,value,error",
    (("peak_allocated_bytes", 101, "VRAM"),
     ("peak_reserved_bytes", 101, "VRAM"),
     ("sampled_peak_rss_bytes", 201, "RAM"),
     ("sampled_peak_rss_bytes", None, "unavailable"),
     ("sampled_peak_process_swap_bytes", 1, "swap"),
     ("sampled_peak_cgroup_swap_bytes", 1, "swap")),
)
def test_stage_limits_fail_closed(field: str, value: int | None, error: str) -> None:
    spec = {"device_cap_bytes": 100, "max_host_bytes": 200}
    metrics = {
        "cuda": {"peak_allocated_bytes": 90, "peak_reserved_bytes": 100},
        "resources": {
            "sampled_peak_rss_bytes": 200, "sampled_peak_process_swap_bytes": 0,
            "sampled_peak_cgroup_swap_bytes": 0,
        },
    }
    _enforce_stage_limits(spec, metrics)
    metrics["cuda" if field.startswith("peak_") else "resources"][field] = value
    with pytest.raises(RuntimeError, match=error):
        _enforce_stage_limits(spec, metrics)
