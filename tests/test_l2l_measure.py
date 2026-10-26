"""CPU tests for the A5 measurement record (l2l_measure.py).

One record per step and per run holds the useful speed (tokens without padding per
second), the GPU and RAM peaks and the I/O counters together. A source that is
missing gives an explicit null with its reason, never a zero.
"""

from __future__ import annotations

import json

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def measure():
    return _load("l2l_measure")


def _step(measure, **override):
    fields = dict(
        step_s=2.0,
        real_tokens=1000,
        layer_loads=64,
        nf4_restores=128,
        spill_written_bytes=0,
        spill_read_bytes=0,
        peak_vram_allocated_bytes=5_000_000_000,
        peak_vram_reserved_bytes=6_000_000_000,
        peak_rss_bytes=20_000_000_000,
        pinned_store_bytes=8_000_000_000,
        weight_read_bytes=14_000_000_000,
    )
    fields.update(override)
    return measure.step_record(**fields)


def test_real_tokens_leave_out_the_padding(measure):
    mask = [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
    assert measure.count_real_tokens(mask) == 5


def test_a_step_reports_tokens_without_padding_per_second(measure):
    record = _step(measure, step_s=2.0, real_tokens=1000)
    assert record["tokens_per_s"] == 500.0
    assert record["step_s"] == 2.0


def test_a_step_record_names_the_unit_of_every_byte_field(measure):
    record = _step(measure)
    assert record["peak_vram_allocated_bytes"] == 5_000_000_000
    assert record["peak_vram_reserved_bytes"] == 6_000_000_000
    assert record["peak_rss_bytes"] == 20_000_000_000
    assert record["pinned_store_bytes"] == 8_000_000_000
    assert record["weight_read_bytes"] == 14_000_000_000
    assert record["layer_loads"] == 64
    assert record["nf4_restores"] == 128


def test_a_missing_source_is_null_with_its_reason_not_zero(measure):
    record = _step(
        measure,
        peak_vram_allocated_bytes=None,
        peak_vram_reserved_bytes=None,
        unavailable={
            "peak_vram_allocated_bytes": "no CUDA device",
            "peak_vram_reserved_bytes": "no CUDA device",
        },
    )
    assert record["peak_vram_allocated_bytes"] is None
    assert record["unavailable"]["peak_vram_allocated_bytes"] == "no CUDA device"
    assert record["peak_rss_bytes"] == 20_000_000_000
    assert "peak_rss_bytes" not in record["unavailable"]


def test_a_null_without_a_reason_is_refused(measure):
    with pytest.raises(ValueError, match="peak_rss_bytes"):
        _step(measure, peak_rss_bytes=None)


def test_a_reason_for_a_field_that_has_a_value_is_refused(measure):
    with pytest.raises(ValueError, match="peak_rss_bytes"):
        _step(measure, unavailable={"peak_rss_bytes": "stale"})


@pytest.mark.parametrize("step_s,tokens", [(0.0, 10), (-1.0, 10), (1.0, 0), (1.0, -3)])
def test_a_step_with_no_time_or_no_tokens_is_refused(measure, step_s, tokens):
    with pytest.raises(ValueError):
        _step(measure, step_s=step_s, real_tokens=tokens)


def test_a_run_sums_tokens_and_time_and_takes_the_peaks(measure):
    steps = [
        _step(measure, step_s=2.0, real_tokens=1000, peak_rss_bytes=10, layer_loads=64,
              spill_written_bytes=100, spill_read_bytes=300, weight_read_bytes=7),
        _step(measure, step_s=4.0, real_tokens=1000, peak_rss_bytes=30, layer_loads=64,
              spill_written_bytes=100, spill_read_bytes=300, weight_read_bytes=5),
    ]
    run = measure.run_record(steps)
    assert run["steps"] == 2
    assert run["tokens_per_s"] == pytest.approx(2000 / 6.0)
    assert run["step_s_mean"] == 3.0
    assert run["peak_rss_bytes"] == 30
    assert run["layer_loads"] == 128
    assert run["spill_written_bytes"] == 200
    assert run["spill_read_bytes"] == 600
    assert run["weight_read_bytes"] == 12


def test_a_run_is_null_when_any_step_lacks_the_source(measure):
    steps = [
        _step(measure),
        _step(measure, peak_rss_bytes=None, unavailable={"peak_rss_bytes": "psutil missing"}),
    ]
    run = measure.run_record(steps)
    assert run["peak_rss_bytes"] is None
    assert "psutil missing" in run["unavailable"]["peak_rss_bytes"]


def test_a_run_of_no_steps_is_refused(measure):
    with pytest.raises(ValueError):
        measure.run_record([])


def test_the_records_are_plain_json_with_no_nan(measure):
    run = measure.run_record([_step(measure)])
    json.dumps(run, allow_nan=False)


def test_a_tiny_cpu_step_gives_every_field_a_value_or_a_reason(measure):
    import torch

    model = torch.nn.Linear(8, 8)
    ids = torch.ones(2, 8)
    mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3])

    def body():
        model(ids).sum().backward()

    record = measure.measure_step(
        body,
        attention_mask=mask,
        device="cpu",
        counters=lambda: {"layer_loads": 0, "nf4_restores": 0, "spill_written_bytes": 0,
                          "spill_read_bytes": 0, "pinned_store_bytes": 0},
    )
    assert record["real_tokens"] == 13
    assert record["tokens_per_s"] == pytest.approx(13 / record["step_s"])
    assert record["peak_vram_allocated_bytes"] is None
    assert record["unavailable"]["peak_vram_allocated_bytes"] == "no CUDA device"
    assert record["weight_read_bytes"] is None
    assert record["unavailable"]["weight_read_bytes"]
    for name in measure.MEASURED_FIELDS:
        assert (record[name] is None) == (name in record["unavailable"])
    json.dumps(record, allow_nan=False)


def test_counters_count_the_step_only(measure):
    state = {"loads": 10}

    def counters():
        return {"layer_loads": state["loads"], "nf4_restores": 0, "spill_written_bytes": 0,
                "spill_read_bytes": 0, "pinned_store_bytes": 7}

    def body():
        state["loads"] += 64

    record = measure.measure_step(
        body, attention_mask=[[1, 1]], device="cpu", counters=counters,
        read_bytes=lambda: 0,
    )
    assert record["layer_loads"] == 64
    assert record["pinned_store_bytes"] == 7
    assert record["weight_read_bytes"] == 0
    assert "weight_read_bytes" not in record["unavailable"]


def _counters():
    return {"layer_loads": 0, "nf4_restores": 0, "spill_written_bytes": 0,
            "spill_read_bytes": 0, "pinned_store_bytes": 0}


def test_the_rss_peak_is_the_steps_own_when_the_peak_can_be_reset(measure, monkeypatch):
    import a4_box

    calls = []
    monkeypatch.setattr(a4_box, "reset_peak_rss", lambda: calls.append("reset") or True)
    monkeypatch.setattr(a4_box, "process_memory", lambda: {"peak_rss": 123})
    record = measure.measure_step(lambda: calls.append("body"), attention_mask=[[1]],
                                  device="cpu", counters=_counters)
    assert calls == ["reset", "body"]
    assert record["peak_rss_bytes"] == 123


def test_the_rss_peak_is_null_when_it_cannot_be_reset(measure, monkeypatch):
    import a4_box

    monkeypatch.setattr(a4_box, "reset_peak_rss", lambda: False)
    monkeypatch.setattr(a4_box, "process_memory", lambda: {"peak_rss": 123})
    record = measure.measure_step(lambda: None, attention_mask=[[1]], device="cpu",
                                  counters=_counters)
    assert record["peak_rss_bytes"] is None
    assert "process-lifetime" in record["unavailable"]["peak_rss_bytes"]


def test_a_counter_the_trainer_does_not_have_is_null_with_a_reason(measure):
    def counters():
        found = _counters()
        del found["nf4_restores"]
        return found

    record = measure.measure_step(lambda: None, attention_mask=[[1]], device="cpu",
                                  counters=counters)
    assert record["nf4_restores"] is None
    assert "nf4_restores" in record["unavailable"]["nf4_restores"]


def test_a_read_counter_that_goes_backwards_is_null_with_a_reason(measure):
    readings = iter([100, 40])
    record = measure.measure_step(lambda: None, attention_mask=[[1]], device="cpu",
                                  counters=_counters, read_bytes=lambda: next(readings))
    assert record["weight_read_bytes"] is None
    assert "backwards" in record["unavailable"]["weight_read_bytes"]
