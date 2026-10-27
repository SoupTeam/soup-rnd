"""CPU tests for the pure parts of the ticket-10 harness (benchmarks/harness)."""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load


def _prof():
    return _load("l2l_ticket10_profile")


def test_buckets_sum_each_name_and_start_again_after_take():
    prof = _prof()
    ticks = iter([0.0, 1.0, 1.0, 3.0, 3.0, 3.5])
    buckets = prof.Buckets(clock=lambda: next(ticks))
    with buckets.timed("layer_wait"):
        pass
    with buckets.timed("layer_wait"):
        pass
    with buckets.timed("optimizer"):
        pass
    out = buckets.take()
    assert out["layer_wait"] == 3.0 and out["optimizer"] == 0.5
    assert out["activation_put"] == 0.0
    assert buckets.take()["layer_wait"] == 0.0


def test_buckets_sync_around_the_block():
    prof = _prof()
    calls = []
    buckets = prof.Buckets(clock=lambda: 0.0, sync=lambda: calls.append("s"))
    with buckets.timed("optimizer"):
        calls.append("body")
    assert calls == ["s", "body", "s"]


def test_decompose_leaves_compute_as_the_rest_of_the_step():
    prof = _prof()
    out = prof.decompose(10.0, {"activation_put": 1.0, "activation_get": 2.0,
                                "layer_wait": 0.5, "optimizer": 0.5})
    assert out["compute"] == 6.0 and not out["overhead_negative"]


def test_decompose_flags_parts_that_exceed_the_step():
    prof = _prof()
    assert prof.decompose(1.0, {"layer_wait": 2.0})["overhead_negative"]


def test_cpu_fractions_from_two_proc_stat_readings():
    prof = _prof()
    a = prof.parse_proc_stat("cpu  100 0 100 700 100 0 0 0\ncpu0 1 1 1 1 1 1 1 1\n")
    b = prof.parse_proc_stat("cpu  200 0 100 750 100 0 0 50\n")
    out = prof.cpu_fractions(a, b)
    assert out["steal"] == pytest.approx(0.25) and out["busy"] == pytest.approx(0.75)


def test_parse_proc_stat_without_a_cpu_line_is_none():
    assert _prof().parse_proc_stat("intr 1 2 3\n") is None


def test_parse_gpu_query_reads_clock_and_throttle_hex():
    prof = _prof()
    out = prof.parse_gpu_query("1395, 1950, 61, 212.5, 0x0000000000000004")
    assert out["sm_mhz"] == 1395 and out["max_sm_mhz"] == 1950 and out["throttle"] == 4
    assert prof.parse_gpu_query("1395, 1950") is None
    assert prof.parse_gpu_query("1395, 1950, 61, [N/A], [N/A]")["throttle"] is None


def _sample(steal=0.0, busy=0.2, sm=1900, throttle=0):
    return {"cpu": {"steal": steal, "busy": busy, "iowait": 0.0},
            "gpu": {"sm_mhz": sm, "max_sm_mhz": 1950, "throttle": throttle}}


def test_judge_run_passes_a_quiet_host():
    out = _prof().judge_run([_sample()] * 5, own_cpu_share=0.15)
    assert out["sensed"] and not out["disturbed"]


@pytest.mark.parametrize("sample,text", [
    (_sample(steal=0.2), "steal"),
    (_sample(sm=1000), "sm clock"),
    (_sample(throttle=0x20), "throttle"),
    (_sample(throttle=0x40), "throttle"),
])
def test_judge_run_flags_each_disturbance(sample, text):
    out = _prof().judge_run([sample] * 5, own_cpu_share=0.15)
    assert out["disturbed"] and any(text in f for f in out["flags"])


def test_judge_run_flags_outside_cpu_load_only_when_the_own_share_is_known():
    prof = _prof()
    busy = [_sample(busy=0.9)] * 3
    assert prof.judge_run(busy, own_cpu_share=0.2)["disturbed"]
    assert not prof.judge_run(busy)["disturbed"]


def test_judge_run_ignores_the_idle_throttle_bits():
    assert not _prof().judge_run([_sample(throttle=0x1)] * 3)["disturbed"]


def test_judge_run_without_samples_is_not_sensed():
    out = _prof().judge_run([])
    assert not out["sensed"] and not out["disturbed"]


def test_per_layer_seconds_and_spread():
    prof = _prof()
    assert prof.per_layer_seconds({"layer_wait": 6.0, "layer_loads": 60}, 32) == pytest.approx(0.1)
    assert prof.per_layer_seconds({"layer_wait": 6.0, "layer_loads": 0}, 32) is None
    assert prof.spread([72.3, 184.8])["ratio"] == pytest.approx(2.556, abs=1e-3)


def test_disturbed_runs_lists_block_arm_and_flags():
    camp = _load("l2l_ticket10_campaign")

    def entry(arm, flags):
        return {"arm": arm, "measurement": {"sensors": {"disturbed": bool(flags), "flags": flags}}}

    blocks = [[entry("A", []), entry("B", ["cpu steal 12.0%"])], [entry("A", [])]]
    assert camp.disturbed_runs(blocks) == [{"block": 1, "arm": "B", "flags": ["cpu steal 12.0%"]}]


def test_campaign_hands_the_ticket_10_arm_script_and_profile_flag_to_the_subprocess(tmp_path):
    camp = _load("l2l_ticket10_campaign")
    seen = []

    class Done:
        returncode = 0

    def fake(cmd, **kwargs):
        seen.append(cmd)
        return Done()

    real = camp.camp09.subprocess.run
    camp.camp09.subprocess.run = fake
    try:
        work, out = tmp_path / "w", tmp_path / "o"
        work.mkdir()
        out.mkdir()
        c = camp.Campaign(work, out, "24gb", {"before": str(tmp_path), "after": str(tmp_path)},
                          "ga", "after")
        c.profile = True
        c.run_arm("s", camp.camp09.ARMS["ga"])
    finally:
        camp.camp09.subprocess.run = real
    assert seen[0][1].endswith("l2l_ticket10_arm.py") and seen[0][-1] == "--profile"


def test_judge_run_skips_samples_where_the_gpu_is_idle():
    idle = _sample(sm=210, throttle=0x1)
    out = _prof().judge_run([idle] * 10 + [_sample()] * 5)
    assert not out["disturbed"]


def test_cpu_parts_are_reported_and_not_taken_off_the_step():
    prof = _prof()
    out = prof.decompose(10.0, {"layer_wait": 1.0, "cpu_rebind": 4.0, "cpu_load_issue": 2.0})
    assert out["compute"] == 9.0 and out["cpu_rebind"] == 4.0 and out["cpu_load_issue"] == 2.0
    assert prof.mean_parts([out, out])["cpu_rebind"] == 4.0


def test_buckets_add_seconds_measured_elsewhere():
    buckets = _prof().Buckets()
    buckets.add("layer_wait", 0.25)
    buckets.add("layer_wait", 0.5)
    assert buckets.take()["layer_wait"] == 0.75


def test_analyze_run_row_rejudges_the_saved_samples(tmp_path):
    an = _load("l2l_ticket10_analyze")
    idle = {"cpu": {"busy": 0.0, "steal": 0.0, "iowait": 0.0},
            "gpu": {"sm_mhz": 210, "max_sm_mhz": 2100, "power_w": 40.0, "throttle": 1}}
    busy_ = {"cpu": {"busy": 0.1, "steal": 0.0, "iowait": 0.0},
             "gpu": {"sm_mhz": 1900, "max_sm_mhz": 2100, "power_w": 200.0, "throttle": 0}}
    path = tmp_path / "b1-p0-A.json"
    path.write_text(__import__("json").dumps({
        "run": {"tokens_per_s": 130.0, "layer_loads_per_step": 244.0},
        "sensor_samples": [idle, idle, busy_, busy_],
    }))
    row = an.run_row(path)
    assert row["arm"] == "GA" and row["flags"] == []
    assert row["power_w_mean"] == 200.0 and row["sm_clock_ratio"] == 1900 / 2100


def test_judge_run_reports_the_power_cap_without_flagging_it():
    out = _prof().judge_run([_sample(throttle=0x4)] * 4)
    assert not out["disturbed"] and out["sensors"]["power_cap_share"] == 1.0
