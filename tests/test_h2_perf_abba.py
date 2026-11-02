"""H2-close stage P runner: block order, run validity, pair verdicts and the series driver."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "harness"))

import h2_perf_abba as abba  # noqa: E402

RECEIPT = json.loads((ROOT / "benchmarks/h2/perf_workload_receipt.json").read_text())
STEPS = RECEIPT["steps"]["step_tokens"]


def report(rate=1000.0, steps=None, valid=True, measured=20, memory=8e9):
    return {
        "valid": valid,
        "failures": [] if valid else [{"check": "grad_norm", "message": "zero"}],
        "steps_measured": measured,
        "throughput": {"useful_tokens_per_second": rate},
        "timing": {"median_seconds": 1.0},
        "memory": {"max_memory_allocated_bytes": memory, "max_memory_reserved_bytes": memory + 1},
        "h2": {"step_tokens": list(STEPS if steps is None else steps)},
    }


# --------------------------------------------------------------------------- order and validity


def test_block_order_for_two_and_three_arms():
    assert abba.block_order(["A", "L", "U"]) == ("A", "L", "U", "U", "L", "A")
    assert abba.block_order(["A", "L"]) == ("A", "L", "L", "A")
    assert abba.block_order(["A", "U"]) == ("A", "U", "U", "A")
    for bad in (["A"], ["L", "U"], ["A", "X"]):
        with pytest.raises(ValueError):
            abba.block_order(bad)


def test_receipt_pins_this_runner_and_twenty_steps():
    runner = (ROOT / "benchmarks/harness/h2_perf_abba.py").read_bytes()
    assert RECEIPT["steps"]["runner_sha256"] == hashlib.sha256(runner).hexdigest()
    assert len(STEPS) == 20 == abba.STEPS and RECEIPT["steps"]["warmup"] == abba.WARMUP == 4
    counted = STEPS[abba.WARMUP :]
    assert sum(s["useful"] for s in counted) == RECEIPT["steps"]["counted_useful"]
    assert sum(s["total"] for s in counted) == RECEIPT["steps"]["counted_total"]


def test_run_problems_accepts_a_valid_run_and_names_each_defect():
    assert abba.run_problems(report(), STEPS) == []
    assert "bench validity" in abba.run_problems(report(valid=False), STEPS)[0]
    assert any("steps measured" in p for p in abba.run_problems(report(measured=19), STEPS))
    other = [dict(s) for s in STEPS]
    other[7]["useful"] += 1
    assert any("token counts" in p for p in abba.run_problems(report(steps=other), STEPS))
    for rate in (0.0, -1.0, float("nan"), None):
        assert any("tokens/s" in p for p in abba.run_problems(report(rate=rate), STEPS))


def test_unsloth_arm_is_refused_until_stage_s():
    with pytest.raises(NotImplementedError):
        abba.run_arm("U", {}, device="cpu")


# --------------------------------------------------------------------------- analysis


def records(values_per_block, order=("A", "L", "L", "A"), void_blocks=()):
    out = []
    for block, values in enumerate(values_per_block):
        for position, (arm, value) in enumerate(zip(order, values)):
            outcome = "void" if block in void_blocks and position == 0 else "ok"
            out.append(
                abba.RunRecord(
                    arm,
                    block,
                    position,
                    outcome,
                    value,
                    {"max_memory_allocated_bytes": 8e9 if arm == "A" else 7e9},
                )
            )
    return out


def test_clear_speedup_is_faster_with_a_ratio_and_interval():
    rows = [[1000 + i, 1100 + i, 1102 + i, 1001 + i] for i in range(5)]
    result = abba.analyze(records(rows), ("A", "L", "L", "A"))
    pair = result["pairs"]["L"]
    assert pair["label"] == "FASTER" and pair["blocks_used"] == 5
    assert pair["ratio"] == pytest.approx(1.1, abs=0.005)
    assert pair["ratio_95"][0] > 1 and pair["p_value"] < 0.05
    assert result["memory"]["L"]["max_memory_allocated_bytes"]["median"] == 7e9


def test_slower_and_no_detectable_difference():
    slow = [[1000, 900 + i, 901 + i, 1000] for i in range(5)]
    assert abba.analyze(records(slow), ("A", "L", "L", "A"))["pairs"]["L"]["label"] == "SLOWER"
    noisy = [[1000, 1000 + s, 1000 + s, 1000] for s in (5, -5, 4, -4, 1)]
    pair = abba.analyze(records(noisy), ("A", "L", "L", "A"))["pairs"]["L"]
    assert pair["label"] == "NO_DETECTABLE_DIFFERENCE"
    assert pair["difference_95"][0] < 0 < pair["difference_95"][1]


def test_void_block_is_not_counted_and_too_few_blocks_give_no_verdict():
    rows = [[1000, 1100, 1100, 1000]] * 5
    pair = abba.analyze(records(rows, void_blocks=(2,)), ("A", "L", "L", "A"))["pairs"]["L"]
    assert pair["valid_blocks"] == 4 and pair["void_blocks"] == 1
    assert pair["label"] == "NO_VERDICT_INSUFFICIENT_REPETITIONS"


def test_n_above_the_cap_gives_no_verdict():
    rows = [[1000, 1000 + s, 1000 + s, 1000 - s] for s in (300, -280, 260, -310, 290)]
    rows += [[1000, 1000 + s, 1000 + s, 1000 - s] for s in (250, -250, 240, -240, 230, -230, 220)]
    pair = abba.analyze(records(rows), ("A", "L", "L", "A"))["pairs"]["L"]
    assert pair["blocks_needed"] > abba.BLOCK_CAP
    assert pair["label"] == "NO_VERDICT_INSUFFICIENT_REPETITIONS"


def test_each_pair_uses_only_its_own_first_n_blocks():
    order = ("A", "L", "U", "U", "L", "A")
    rows = [[1000, 1100, 1000 + s, 1000 - s, 1100, 1000] for s in (90, -80, 70, -60, 50)]
    rows += [[1000, 1300, 1000, 1000, 1300, 1000]] * 3  # later blocks, L would look faster still
    result = abba.analyze(records(rows, order=order), order)
    assert result["pairs"]["L"]["blocks_used"] == 5
    assert result["pairs"]["L"]["ratio"] == pytest.approx(1.1)
    assert result["pairs"]["U"]["blocks_needed"] >= 5


# --------------------------------------------------------------------------- series driver


class FakeWatch:
    outcomes: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.record = {"outcome": FakeWatch.outcomes.pop(0) if FakeWatch.outcomes else "ok"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def make_series(tmp_path, rates, admitted=("A", "L"), deadline=1e12, gpu=None, fail_at=None):
    calls = []

    def launch(arm, out):
        calls.append(arm)
        out.mkdir(parents=True)
        if fail_at is not None and len(calls) == fail_at:
            (out / "failure.json").write_text("{}")
            return 1
        (out / "run.json").write_text(json.dumps(report(rate=rates(arm, len(calls)))))
        return 0

    series = abba.Series(
        out=tmp_path / "series",
        admitted=admitted,
        expected_steps=STEPS,
        deadline_unix_s=deadline,
        model_path="/data/h2/model_snapshot",
        label="test",
        launch=launch,
        watch=FakeWatch,
        gpu_processes=gpu or (lambda: []),
        clock=lambda: 0.0,
    )
    return series, calls


def test_quiet_series_stops_after_the_pilot(tmp_path):
    FakeWatch.outcomes = []
    series, calls = make_series(
        tmp_path, lambda arm, i: 1100.0 + i % 3 if arm == "L" else 1000.0 + i % 3
    )
    result = series.run(block_seconds_estimate=60)
    assert result["status"] == "COMPLETE" and len(calls) == 20
    assert result["analysis"]["pairs"]["L"]["label"] == "FASTER"
    saved = json.loads((tmp_path / "series" / "series.json").read_text())
    assert saved["status"] == "COMPLETE" and len(saved["runs"]) == 20


def test_f1_void_block_is_replaced(tmp_path):
    FakeWatch.outcomes = ["ok", "void"]
    series, calls = make_series(tmp_path, lambda arm, i: 1000.0 + i % 3)
    result = series.run(block_seconds_estimate=60)
    assert result["status"] == "COMPLETE" and len(calls) == 24
    assert result["analysis"]["pairs"]["L"]["void_blocks"] == 1


def test_gpu_not_exclusive_voids_the_run(tmp_path):
    FakeWatch.outcomes = []
    seen = iter([[{"pid": 1}]] + [[]] * 200)
    series, calls = make_series(tmp_path, lambda arm, i: 1000.0 + i % 3, gpu=lambda: next(seen))
    result = series.run(block_seconds_estimate=60)
    assert result["runs"][0]["outcome"] == "void" and len(calls) == 24


def test_a_failed_run_not_explained_by_f1_stops_the_series(tmp_path):
    FakeWatch.outcomes = []
    series, calls = make_series(tmp_path, lambda arm, i: 1000.0, fail_at=3)
    result = series.run(block_seconds_estimate=60)
    assert result["status"] == "FAILED_RUN" and len(calls) == 3
    assert "child exit 1" in result["reason"]


def test_deadline_and_host_noise_stop_the_series(tmp_path):
    FakeWatch.outcomes = []
    series, calls = make_series(tmp_path, lambda arm, i: 1000.0, deadline=30)
    assert series.run(block_seconds_estimate=60)["status"] == "INCOMPLETE_DEADLINE"
    assert calls == []
    FakeWatch.outcomes = ["void", "ok", "ok", "ok"] * 6
    series, calls = make_series(tmp_path / "noise", lambda arm, i: 1000.0)
    result = series.run(block_seconds_estimate=60)
    assert result["status"] == "NO_VERDICT_HOST_NOISE" and len(calls) == 20


def test_noisy_series_stops_at_the_cap(tmp_path):
    FakeWatch.outcomes = []
    swings = [300, -280, 260, -310, 290, -250, 240, -240, 230, -230, 220, -220, 210]
    series, calls = make_series(
        tmp_path, lambda arm, i: 1000.0 + (swings[(i - 1) // 4] if arm == "L" else 0.0)
    )
    result = series.run(block_seconds_estimate=60)
    assert result["status"] == "COMPLETE_AT_CAP" and len(calls) == 4 * abba.BLOCK_CAP
    assert result["analysis"]["pairs"]["L"]["label"] == "NO_VERDICT_INSUFFICIENT_REPETITIONS"


def test_block_difference_uses_both_runs_of_each_arm():
    rows = [[1000, 1100, 1300 + i, 1000] for i in range(5)]
    pair = abba.analyze(records(rows), ("A", "L", "L", "A"))["pairs"]["L"]
    assert pair["mean_difference"] == pytest.approx(201.0)


def test_a_void_run_of_the_other_arm_voids_the_block_for_every_pair():
    order = ("A", "L", "U", "U", "L", "A")
    rows = [[1000, 1100, 1000, 1000, 1100, 1000]] * 5
    recs = records(rows, order=order)
    for record in recs:
        if record.block == 1 and record.arm == "U" and record.position == 2:
            record.outcome = "void"
    pair = abba.analyze(recs, order)["pairs"]["L"]
    assert pair["void_blocks"] == 1 and pair["valid_blocks"] == 4


def test_n_above_the_cap_gives_no_verdict_even_with_enough_blocks_on_record():
    swings = [300, -280, 260, -310, 290] + [250, -250, 240, -240, 230, -230, 220, -220] * 5
    rows = [[1000, 1000 + s, 1000 + s, 1000] for s in swings]
    pair = abba.analyze(records(rows), ("A", "L", "L", "A"))["pairs"]["L"]
    assert abba.BLOCK_CAP < pair["blocks_needed"] <= pair["valid_blocks"]
    assert pair["label"] == "NO_VERDICT_INSUFFICIENT_REPETITIONS"
