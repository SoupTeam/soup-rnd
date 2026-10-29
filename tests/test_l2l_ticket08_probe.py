"""CPU tests for the A5 ticket 08 probe: the gradient comparison and the dtype summary."""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def probe():
    return _load("l2l_ticket08_gpu_probe")


def _grads(value=1.0):
    import torch

    return {
        "layers.0.q.lora_A": torch.full((2, 2), value),
        "layers.0.q.lora_B": torch.full((2, 2), 3.0),
    }


def test_equal_gradients_report_zero_difference_at_every_step(probe):
    report = probe.compare_gradients([_grads(), _grads()], [_grads(), _grads()])
    assert [row["equal"] for row in report] == [True, True]
    assert [row["max_abs_diff"] for row in report] == [0.0, 0.0]


def test_the_first_unequal_step_names_the_worst_parameter(probe):
    cand = _grads()
    cand["layers.0.q.lora_B"][0, 0] = 3.5
    report = probe.compare_gradients([_grads(), _grads()], [_grads(), cand])
    assert [row["equal"] for row in report] == [True, False]
    assert report[1]["worst_param"] == "layers.0.q.lora_B"
    assert report[1]["max_abs_diff"] == pytest.approx(0.5)
    assert report[1]["max_abs_ref"] == pytest.approx(3.0)


def test_parameters_missing_on_one_side_are_reported_not_skipped(probe):
    cand = _grads()
    cand.pop("layers.0.q.lora_A")
    report = probe.compare_gradients([_grads()], [cand])
    assert report[0]["equal"] is False
    assert report[0]["missing"] == ["layers.0.q.lora_A"]


def test_a_shorter_run_compares_only_the_steps_both_have(probe):
    report = probe.compare_gradients([_grads(), _grads()], [_grads()])
    assert len(report) == 1


def test_dtype_summary_flags_a_store_that_rounds_the_residual_stream(probe):
    import torch

    seen = probe.DtypeLog()
    seen.note_store(torch.bfloat16)
    seen.note_hidden(torch.float32)
    assert seen.summary()["store_rounds_hidden"] is True
    ok = probe.DtypeLog()
    ok.note_store(torch.bfloat16)
    ok.note_hidden(torch.bfloat16)
    assert ok.summary()["store_rounds_hidden"] is False


def test_gradient_hash_accepts_bf16_and_tells_values_apart(probe):
    import torch

    left = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    same = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    other = torch.tensor([1.0, 2.015625], dtype=torch.bfloat16)
    assert probe.tensor_sha(left) == probe.tensor_sha(same)
    assert probe.tensor_sha(left) != probe.tensor_sha(other)
