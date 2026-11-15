"""B1.4 ticket 09: the counters and the measurement of the B1 path report.

Two seams (``benchmarks/harness/b14_path_report.py``): ``OpCounter`` and ``measure_path``.
Both run here on CPU and on the tiny synthetic checkpoint of ``b13_moe_fixture``. The CUDA
numbers (kernel launches, VRAM) come only from the run on the rented or Kaggle GPU; these
tests do not claim them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness.b14_path_report import OpCounter, measure_path
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR


def test_counts_each_aten_op_dispatched_inside_the_block() -> None:
    import torch

    a, b = torch.ones(2, 2), torch.ones(2, 2)
    with OpCounter() as counter:
        torch.mm(a, b)
        torch.mm(a, b)
        torch.add(a, b)

    assert counter.by_op == {"aten.mm.default": 2, "aten.add.Tensor": 1}
    assert counter.aten_ops == 3


def test_ops_outside_the_block_are_not_counted() -> None:
    import torch

    a = torch.ones(2, 2)
    torch.mm(a, a)
    with OpCounter() as counter:
        pass
    torch.mm(a, a)

    assert counter.aten_ops == 0


def test_backward_ops_inside_the_block_are_counted() -> None:
    import torch

    weight = torch.ones(3, 3, requires_grad=True)
    inputs = torch.ones(2, 3)
    forward_only = OpCounter()
    with forward_only:
        loss = torch.mm(inputs, weight).sum()
    with OpCounter() as backward:
        loss.backward()

    # d(loss)/d(weight) = inputs.T @ ones: one mm in backward.
    assert backward.by_op.get("aten.mm.default") == 1
    assert backward.aten_ops > 0
    assert forward_only.by_op["aten.mm.default"] == 1


def test_a_cpu_run_has_no_cuda_launch_count() -> None:
    import torch

    with OpCounter() as counter:
        torch.mm(torch.ones(2, 2), torch.ones(2, 2))

    assert counter.cuda_kernel_launches is None
    assert counter.cuda_memcpy_events is None


def test_profiling_a_nonexistent_cuda_device_is_refused() -> None:
    from unittest.mock import patch

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA"):
            OpCounter(profile_cuda=True).__enter__()


# --------------------------------------------------------------------------
# measure_path on the tiny synthetic checkpoint, CPU.
# --------------------------------------------------------------------------

#: 3 projections x hidden 32 x moe_intermediate 16 x 4 bytes (float32), from ``moe_config``.
EXPERT_BYTES = 3 * 32 * 16 * 4


#: Words that make a field name a duration or a speed. Compared whole, per ``_`` part, so a name
#: such as ``runtime`` or ``secondary`` is not one.
_DURATION_WORDS = {"time", "second", "seconds", "sec", "duration", "tok", "latency"}


def _duration_fields(value: object, path: str = "") -> list[str]:
    """Paths of the dict keys, at any depth, whose name says duration or speed."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            here = f"{path}.{key}" if path else str(key)
            if _DURATION_WORDS & set(str(key).lower().split("_")):
                found.append(here)
            found += _duration_fields(inner, here)
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            found += _duration_fields(inner, f"{path}[{index}]")
    return found


def test_duration_fields_are_found_by_name_not_by_substring() -> None:
    record = {
        "runtime": 1, "secondary": 2, "passes": [{"wall_time_s": 3, "tok_s": 4}], "p": "/x/time",
    }

    assert _duration_fields(record) == ["passes[0].wall_time_s", "passes[0].tok_s"]


@pytest.fixture(scope="module")
def tiny_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    weights = fixture.moe_weights_dir(tmp_path_factory.mktemp("b14-path-weights"))
    work = tmp_path_factory.mktemp("b14-path-work")
    return measure_path(weights, work, device="cpu", dtype="float32", seqs=(6, 12))


@_REQUIRES_TRAINING_FLOOR
def test_a_cpu_pass_reports_what_the_expert_pool_read(tiny_result: dict) -> None:
    one = tiny_result["passes"][0]

    assert one["status"] == "ok"
    # Two MoE layers, the router hits at least two experts in each.
    assert one["experts_visited"] >= 4
    # B1.3 report: a streamed layer is recomputed in backward, so each visited expert is read
    # three times: forward, the recompute, and the backward reload.
    assert one["expert_fetches"] == 3 * one["experts_visited"]
    assert one["expert_bytes_read"] == one["expert_fetches"] * EXPERT_BYTES
    assert one["grads_nonzero"] > 0


@_REQUIRES_TRAINING_FLOOR
def test_operation_counts_split_forward_backward_and_divide_by_visited_experts(
    tiny_result: dict,
) -> None:
    one = tiny_result["passes"][0]

    assert one["aten_ops_forward"] > 0 and one["aten_ops_backward"] > 0
    assert one["aten_ops_total"] == one["aten_ops_forward"] + one["aten_ops_backward"]
    assert one["aten_ops_per_expert_visit"] == pytest.approx(
        one["aten_ops_total"] / one["experts_visited"]
    )
    assert one["cuda"] is None  # no GPU on this host: no kernel count is invented
    # Operations inside the routed-expert modules' forward: a part of the forward total, and
    # at least one operation per visited expert (the gate/up/down products).
    inside = one["aten_ops_in_experts_forward"]
    assert 3 * one["experts_visited"] <= inside < one["aten_ops_forward"]
    assert one["aten_ops_per_expert_visit_forward"] == pytest.approx(
        inside / one["experts_visited"]
    )


@_REQUIRES_TRAINING_FLOOR
def test_a_repeated_pass_gives_the_same_counts(tiny_result: dict) -> None:
    repeat = tiny_result["repeat"]

    assert repeat["identical"] is True
    assert repeat["seq"] == tiny_result["passes"][0]["seq"]


@_REQUIRES_TRAINING_FLOOR
def test_memory_is_recorded_idle_and_peak_and_no_wall_time_is_recorded(
    tiny_result: dict,
) -> None:
    import json

    one = tiny_result["passes"][0]

    assert tiny_result["idle"]["rss_bytes"] > 0
    assert one["rss_peak_sampled_bytes"] >= tiny_result["idle"]["rss_bytes"]
    assert tiny_result["idle"]["cuda_allocated_bytes"] is None
    assert [p["seq"] for p in tiny_result["passes"]] == [6, 12]
    # Capacity report: no field of the result is a duration.
    assert _duration_fields(tiny_result) == []
    assert json.loads(json.dumps(tiny_result))["passes"][0]["seq"] == 6


# --------------------------------------------------------------------------
# The synthetic checkpoint writer: a layer of any size without instantiating the model.
# --------------------------------------------------------------------------
@_REQUIRES_TRAINING_FLOOR
def test_a_written_checkpoint_streams_per_expert_with_a_layer_above_the_limit(
    tmp_path: Path,
) -> None:
    from benchmarks.harness.b14_synthetic_checkpoint import expert_layer_bytes, write_checkpoint

    config = fixture.moe_config(n_routed_experts=16, n_group=4, topk_group=2)
    weights = write_checkpoint(tmp_path / "ckpt", config, dtype="float32", seed=5)
    # 16 experts x 3 x 32 x 16 x 4 bytes = 98 304 per layer: the independent expectation.
    layer_bytes = expert_layer_bytes(config, dtype="float32")
    assert layer_bytes == 16 * EXPERT_BYTES == 98_304
    limit = layer_bytes // 4

    result = measure_path(weights, tmp_path / "work", device="cpu", dtype="float32", seqs=(8,))

    one = result["passes"][0]
    assert one["status"] == "ok" and one["grads_nonzero"] > 0
    assert one["expert_fetches"] == 3 * one["experts_visited"]
    assert one["expert_bytes_read"] == one["expert_fetches"] * EXPERT_BYTES
    # The bounded slot is one expert, well under a limit that is itself under the layer.
    assert result["stream_stats"]["expert_buffer_bytes"] <= limit < layer_bytes
