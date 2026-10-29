"""CPU tests for the A3 stage-2 stand (benchmarks/harness/a3_spill_stand.py).

The GPU session cannot run here. These tests pin what the stand decides on its own: the tight
limit that makes the product's real RAM budget spill about half the layers, the order and the
scopes of the planned children, when a signal is sent, the CPU comparison of the unpublished
tensors, and one real child pair on a tiny model through the product's trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.test_l2l_probe_harness import _load

MIB = 1 << 20
GIB = 1 << 30
MISTRAL_CHUNK = 512 * 4096 * 2  # one micro-batch of one layer at seq 512, bf16


@pytest.fixture()
def stand():
    return _load("a3_spill_stand")


@pytest.fixture()
def rule():
    return _load("a3_spill_rule")


def _mistral_plan():
    """The product's own plan dict for Mistral-7B at k 4, seq 512, bf16, pinned."""
    from soup_cli.utils import l2l_spill

    return l2l_spill.plan_spill(None, n_layers=32, k=4, chunk_bytes=MISTRAL_CHUNK,
                                ring=l2l_spill.TRAINER_RING, pin=True).as_dict()


def _scope_tree(tmp_path, *, limit, current, inactive, peak=None, swap="0", oom_kill=0,
                active=0, slab=0, dirty=0, writeback=0, shmem=0):
    """A fake /proc/self/cgroup and cgroup v2 tree with one limited scope."""
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("0::/system.slice/a3.scope\n")
    group = tmp_path / "cgroup" / "system.slice" / "a3.scope"
    group.mkdir(parents=True)
    anon = current - inactive - active - slab - shmem
    files = {
        "memory.max": str(limit), "memory.swap.max": swap, "memory.current": str(current),
        "memory.stat": (f"anon {anon}\nfile {inactive + active}\nshmem {shmem}\n"
                        f"file_dirty {dirty}\nfile_writeback {writeback}\n"
                        f"active_file {active}\ninactive_file {inactive}\n"
                        f"slab_reclaimable {slab}\nslab_unreclaimable 0\n"),
        "memory.events": f"low 0\nhigh 0\nmax 2\noom 0\noom_kill {oom_kill}\n",
    }
    if peak is not None:
        files["memory.peak"] = str(peak)
    for name, text in files.items():
        (group / name).write_text(text)
    return str(proc), str(tmp_path / "cgroup")


# --------------------------------------------------------------------------
# the tight limit
# --------------------------------------------------------------------------
@pytest.mark.parametrize("working", [4 * GIB, 9 * GIB + 123 * MIB + 4567, 15 * GIB])
def test_the_tight_limit_spills_16_of_mistrals_32_layers(stand, rule, working):
    out = stand.tight_limit(working, _mistral_plan())
    assert out["predicted_spill_layers"] == 16
    low, high = rule.SPILL_BAND
    assert low <= out["predicted_spill_layers"] <= high
    assert out["limit_bytes"] % MIB == 0 and out["limit_bytes"] > working


@pytest.mark.parametrize("drift_mib, layers", [
    (-7, 16), (8, 16),  # the protocol's "exactly 16" edges
    (-8, 15), (9, 17),  # one MiB past them
    (-71, 12), (72, 20),  # the band's edges
    (-72, 11), (73, 21),  # one MiB past the band
])
def test_a_working_set_that_moves_between_children_moves_the_spill_as_documented(
    stand, drift_mib, layers
):
    """Below 1.79 GB available the budget is available - 512 MiB, so a MiB of working set
    moves the budget by a MiB (not 0.7 of one, as in v1): the band is +-72 MiB wide."""
    plan = _mistral_plan()
    working = 9 * GIB
    limit = stand.tight_limit(working, plan)["limit_bytes"]
    assert limit - working == 840 * MIB  # (320 + 8) MiB budget + the 512 MiB reserve
    assert stand.predicted_spill(limit, working + drift_mib * MIB, plan) == layers


def test_the_calibrated_limit_makes_the_products_own_budget_spill_16_layers(
    stand, tmp_path, monkeypatch
):
    """Product numbers -> the stand's working set -> its limit -> the product's real
    ram_budget and plan_spill under a scope at that limit."""
    from soup_cli.utils import l2l_spill, layer_stream

    monkeypatch.setattr(layer_stream, "free_ram_bytes", lambda: 200 * GIB)
    plan = _mistral_plan()
    working, inactive = 9 * GIB + 77 * MIB, 2 * GIB
    generous = 24 * GIB
    proc, root = _scope_tree(tmp_path / "generous", limit=generous,
                             current=working + inactive, inactive=inactive)
    seen = l2l_spill.ram_budget(proc_root=proc, v2_root=root, v1_root=str(tmp_path / "v1"))
    record = {"spill_plan": {**plan, "limiting_level": seen.limiting_level,
                             "limiting_numbers": seen.limiting_numbers},
              "cgroup_at_first_step": stand.cgroup_readback(proc_root=proc, v2_root=root)}
    measured, source = stand.working_set(record)
    assert measured == working and "product" in source

    limit = stand.tight_limit(measured, plan)["limit_bytes"]
    proc, root = _scope_tree(tmp_path / "tight", limit=limit, current=working + inactive,
                             inactive=inactive)
    budget = l2l_spill.ram_budget(proc_root=proc, v2_root=root, v1_root=str(tmp_path / "v1"))
    assert budget.limiting_level.endswith("a3.scope")
    spilled = l2l_spill.plan_spill(budget.budget_bytes, n_layers=32, k=4,
                                   chunk_bytes=MISTRAL_CHUNK, ring=l2l_spill.TRAINER_RING,
                                   pin=True)
    assert spilled.spill_layers == 16


def test_the_working_set_falls_back_to_the_readback_when_psutil_limited(stand):
    plan = {**_mistral_plan(), "limiting_level": "psutil",
            "limiting_numbers": {"psutil_available": 50 * GIB}}
    record = {"spill_plan": plan, "cgroup_at_first_step": {"path": "/x", "working_set": 7 * GIB}}
    assert stand.working_set(record) == (7 * GIB, "the stand's readback at the first "
                                                  "training_step entry")
    assert stand.working_set({"spill_plan": plan})[0] is None


def test_calibration_refuses_a_generous_limit_that_already_spills(stand):
    record = {"exit": "normal", "spill_plan": {**_mistral_plan(), "spill_layers": 3}}
    with pytest.raises(stand.SessionStopError, match="--generous-limit"):
        stand.calibration_limit(record)


def test_calibration_refuses_a_child_that_planned_no_store(stand):
    with pytest.raises(stand.SessionStopError, match="did not plan a store"):
        stand.calibration_limit({"exit": "error", "error": "OOM"})


def _check(layers, working=9 * GIB):
    plan = {**_mistral_plan(), "spill_layers": layers, "limiting_level": "/s",
            "limiting_numbers": {"used": working + GIB, "inactive_file": GIB}}
    return {"exit": "normal", "spill_plan": plan, "cgroup_at_first_step": {"path": "/s"}}


def test_a_check_in_the_band_keeps_its_limit(stand):
    assert stand.next_limit(_check(14), 123 * MIB, 1) == (
        "done", 123 * MIB, {"spill_layers": 14, "in_band": True})


@pytest.mark.parametrize("layers", [4, 27])
def test_a_check_outside_the_band_recomputes_once_from_its_own_numbers(stand, layers):
    action, limit, why = stand.next_limit(_check(layers, working=10 * GIB), 9 * GIB, 1)
    assert action == "again"
    assert why["recomputed"]["working_set_bytes"] == 10 * GIB
    assert limit == stand.tight_limit(10 * GIB, _mistral_plan())["limit_bytes"]
    assert stand.predicted_spill(limit, 10 * GIB, _mistral_plan()) == 16


def test_after_the_last_check_any_spill_is_kept_and_none_stops(stand):
    last = stand.MAX_CALIBRATION_CHECKS
    assert stand.next_limit(_check(25), 9 * GIB, last)[:2] == ("done", 9 * GIB)
    assert stand.next_limit(_check(0), 9 * GIB, last)[0] == "stop"
    assert stand.next_limit({"exit": "error"}, 9 * GIB, 1)[0] == "stop"


@pytest.mark.parametrize("text, value", [("24G", 24 * GIB), ("512M", 512 * MIB),
                                         ("1.5G", 1536 * MIB), ("25769803776", 24 * GIB),
                                         (str(MIB + 1), MIB)])
def test_limits_are_binary_and_whole_mib(stand, text, value):
    assert stand.parse_size(text) == value


def test_a_limit_below_one_mib_is_refused(stand):
    with pytest.raises(ValueError, match="below 1 MiB"):
        stand.parse_size("100K")


# --------------------------------------------------------------------------
# the scope readback (V3)
# --------------------------------------------------------------------------
def test_the_readback_reads_the_own_scope(stand, tmp_path):
    proc, root = _scope_tree(tmp_path, limit=9 * GIB, current=5 * GIB, inactive=GIB,
                             peak=6 * GIB, oom_kill=1)
    out = stand.cgroup_readback(proc_root=proc, v2_root=root)
    assert out["version"] == 2 and out["path"].endswith("a3.scope")
    assert (out["memory_max"], out["swap_max"], out["peak"]) == (9 * GIB, 0, 6 * GIB)
    assert out["working_set"] == 4 * GIB
    assert out["events"]["oom_kill"] == 1


def test_unreclaimable_is_current_less_clean_file_cache_and_reclaimable_slab(stand, tmp_path):
    proc, root = _scope_tree(tmp_path, limit=9 * GIB, current=10 * GIB, inactive=2 * GIB,
                             active=3 * GIB, slab=GIB // 2, dirty=200 * MIB,
                             writeback=56 * MIB, shmem=GIB)
    out = stand.cgroup_readback(proc_root=proc, v2_root=root)
    # dirty and writeback file pages cannot be dropped at once: they stay unreclaimable
    assert out["unreclaimable"] == 10 * GIB - (5 * GIB - 256 * MIB) - GIB // 2
    assert out["anon_shmem"] == 10 * GIB - 5 * GIB - GIB // 2
    assert out["working_set"] == 8 * GIB  # the product's own count keeps the active cache


def test_without_the_file_counters_unreclaimable_is_unknown(stand):
    assert stand.unreclaimable_bytes(5 * GIB, {"anon": GIB, "inactive_file": GIB}) is None
    assert stand.unreclaimable_bytes("max", {}) is None


def test_the_process_peak_rss_is_read_from_proc_status(stand, tmp_path):
    (tmp_path / "self").mkdir()
    (tmp_path / "self" / "status").write_text("Name:\tpython\nVmHWM:\t 5379072 kB\n"
                                              "VmRSS:\t 5117952 kB\n")
    assert stand.process_memory(proc_root=str(tmp_path)) == {"rss": 5117952 * 1024,
                                                             "hwm": 5379072 * 1024}
    assert stand.process_memory(proc_root=str(tmp_path / "none")) == {"rss": None, "hwm": None}


# --------------------------------------------------------------------------
# the OOM margin of the tight limit
# --------------------------------------------------------------------------
def _sampler(stand, readings):
    """A sampler fed with (unreclaimable, pinned) per call, in call order."""
    feed = iter(readings)
    current = {}

    def read():
        current["u"], current["p"] = next(feed)
        return {"unreclaimable": current["u"], "anon_shmem": current["u"] - GIB,
                "current": current["u"] + 3 * GIB, "peak": current["u"] + 4 * GIB}

    return stand.MemorySampler(read=read, pinned=lambda: current["p"],
                               process=lambda: {"rss": None, "hwm": 6 * GIB})


def _calibration_transient(stand, *, build_peak, step_extra, plan_host=512 * MIB):
    """Readings of a calibration child: build end, a background reading inside the build,
    the first training_step entry at 9 GiB, then a step that allocates the whole store
    (pinned) plus ``step_extra`` unreclaimable bytes the plan does not account for."""
    base, pinned = 9 * GIB, 64 * MIB
    sampler = _sampler(stand, [(base + build_peak, pinned), (base, pinned), (base, pinned),
                               (base + plan_host + step_extra, pinned + plan_host)])
    sampler.samples.append(sampler._row()[1])  # a 50 ms reading at the build's peak
    sampler.sample("build_end")
    sampler.first_step()
    sampler.sample("step_end")
    return stand.transients(sampler.samples, plan_host)


def test_the_transient_takes_out_the_store_but_not_pinned_bytes_beyond_it(stand):
    out = _calibration_transient(stand, build_peak=300 * MIB, step_extra=100 * MIB)
    assert out["unreclaimable_at_first_step"] == 9 * GIB
    assert out["build_transient_bytes"] == 300 * MIB
    assert out["train_transient_bytes"] == 100 * MIB
    assert out["transient_bytes"] == 300 * MIB
    assert out["vm_hwm_bytes"] == 6 * GIB
    # memory.peak: the largest over the named readings, the step end with the store in it
    assert out["memory_peak_bytes"] == 9 * GIB + 512 * MIB + 100 * MIB + 4 * GIB
    assert out["anon_shmem_at_first_step"] == 8 * GIB


def test_pinned_bytes_beyond_the_planned_store_count_as_transient(stand):
    # the step pins 200 MiB more than the plan's 512 MiB: only the plan is taken out
    base, pinned = 9 * GIB, 64 * MIB
    sampler = _sampler(stand, [(base, pinned), (base + 712 * MIB, pinned + 712 * MIB)])
    sampler.first_step()
    sampler.sample("step_end")
    assert stand.transients(sampler.samples, 512 * MIB)["train_transient_bytes"] == 200 * MIB


def test_without_a_first_step_reading_the_transient_is_unknown(stand):
    sampler = _sampler(stand, [(9 * GIB, 0)])
    sampler.sample("build_end")
    assert stand.transients(sampler.samples, 0)["transient_bytes"] is None


def _guard(stand, transient, limit_over_base, host=320 * MIB):
    status = {"limits": {"oom_margin": []}}
    calibration = {"memory_transient": transient}
    limit = transient["unreclaimable_at_first_step"] + limit_over_base
    return status, lambda: stand.guard_oom(status, calibration, limit, host, 9 * GIB)


def test_margins_above_both_bounds_proceed_without_a_warning(stand):
    transient = _calibration_transient(stand, build_peak=100 * MIB, step_extra=50 * MIB)
    # 1 GiB over the baseline: build margin 1024 - 100, training margin 1024 - 50 - 320 MiB
    status, guard = _guard(stand, transient, 1 * GIB)
    guard()
    margin = status["limits"]["oom_margin"][0]
    assert margin["stop"] is False and margin["warnings"] == []
    assert margin["build_margin_bytes"] == GIB - 100 * MIB
    assert margin["train_margin_bytes"] == GIB - 50 * MIB - 320 * MIB
    assert margin["margin_over_working_bytes"] == 9 * GIB + GIB - (9 * GIB + 100 * MIB
                                                                   + 320 * MIB)


def test_a_build_margin_below_0_stops_with_the_numbers_and_never_raises_the_limit(stand):
    # 840 MiB over the baseline (the stand's limit with nothing reclaimable), a build
    # transient of 841 MiB: build margin -1 MiB
    transient = _calibration_transient(stand, build_peak=841 * MIB, step_extra=0)
    status, guard = _guard(stand, transient, 840 * MIB)
    with pytest.raises(stand.OomRiskError) as caught:
        guard()
    margin = status["limits"]["oom_margin"][0]
    assert margin["stop"] is True and margin["build_margin_bytes"] == -MIB
    assert margin["limit_bytes"] == 9 * GIB + 840 * MIB  # the limit tried, not raised
    text = str(caught.value)
    assert f"build margin {-MIB:,} B < 0" in text and "not raised" in text
    assert caught.value.state == "stopped_calibration_oom_risk"


def test_a_build_margin_of_exactly_0_proceeds(stand):
    transient = _calibration_transient(stand, build_peak=469 * MIB, step_extra=0)
    status, guard = _guard(stand, transient, 469 * MIB)
    guard()
    assert status["limits"]["oom_margin"][0]["build_margin_bytes"] == 0


def test_a_training_margin_below_0_proceeds_with_a_warning(stand, capsys):
    """The product promises its own headroom: an OOM in training is a product finding, so a
    negative training margin warns and never stops."""
    transient = _calibration_transient(stand, build_peak=0, step_extra=300 * MIB)
    status, guard = _guard(stand, transient, 469 * MIB)
    guard()
    margin = status["limits"]["oom_margin"][0]
    assert margin["stop"] is False
    assert margin["train_margin_bytes"] == 469 * MIB - 300 * MIB - 320 * MIB < 0
    assert [w.split(" margin")[0] for w in margin["warnings"]] == ["training"]
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.parametrize("build_over, warned", [(256 * MIB - 1, ["build"]), (256 * MIB, [])])
def test_256_mib_only_warns_and_decides_nothing(stand, build_over, warned):
    # build margin = build_over; training margin = build_over + 300 MiB, never warned
    transient = _calibration_transient(stand, build_peak=0, step_extra=0)
    status, guard = _guard(stand, transient, build_over, host=-300 * MIB)
    guard()
    margin = status["limits"]["oom_margin"][0]
    assert margin["stop"] is False and margin["build_margin_bytes"] == build_over
    assert [w.split(" margin")[0] for w in margin["warnings"]] == warned


def test_an_unmeasured_transient_stops_rather_than_guessing(stand):
    status = {"limits": {"oom_margin": []}}
    with pytest.raises(stand.OomRiskError, match="build margin is unknown"):
        stand.guard_oom(status, {"memory_transient": {"transient_bytes": None}}, 10 * GIB,
                        320 * MIB, 9 * GIB)


def test_the_calibration_record_carries_its_own_margin(stand):
    plan = {**_mistral_plan(), "limiting_level": "/scope",
            "limiting_numbers": {"used": 10 * GIB, "inactive_file": GIB}}
    transient = _calibration_transient(stand, build_peak=100 * MIB, step_extra=0)
    margin = stand.calibration_margin({"exit": "normal", "spill_plan": plan,
                                       "cgroup_at_first_step": {"path": "/scope"},
                                       "memory_transient": transient})
    assert margin["limit_bytes"] == stand.tight_limit(9 * GIB, plan)["limit_bytes"]
    assert margin["build_margin_bytes"] == 840 * MIB - 100 * MIB
    # session 1's v1 limit left 149 MiB here (a warning); the reserve leaves 520 MiB
    assert margin["train_margin_bytes"] == 840 * MIB - 320 * MIB
    assert margin["warnings"] == []


def test_the_background_sampler_sees_a_transient_between_named_readings(stand):
    peak = threading.Event()

    def read():
        value = 9 * GIB + (700 * MIB if not peak.is_set() else 0)
        return {"unreclaimable": value, "anon_shmem": value, "current": value, "peak": None}

    with stand.MemorySampler(0.005, read=read, process=lambda: {"rss": None, "hwm": None}) as s:
        started = time.monotonic()
        while not [r for r in s.samples if "label" not in r]:
            assert time.monotonic() - started < 2.0
            time.sleep(0.005)
        peak.set()
        s.sample("build_end")
        s.first_step()
    out = stand.transients(s.samples, 0)
    assert out["build_transient_bytes"] == 700 * MIB


def test_an_unlimited_scope_reads_back_max_not_a_number(stand, rule, tmp_path):
    proc, root = _scope_tree(tmp_path, limit="max", current=5 * GIB, inactive=GIB, swap="max")
    out = stand.cgroup_readback(proc_root=proc, v2_root=root)
    assert out["memory_max"] == "max" and out["swap_max"] == "max"
    rec = {"device": "cuda", "cgroup_start": out, "expected_memory_max": 9 * GIB}
    assert rule.child_problems(rec, {"memory_max": 9 * GIB})


def test_no_cgroup_v2_line_reads_back_no_version(stand, tmp_path):
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_text("4:memory:/user.slice\n")
    out = stand.cgroup_readback(proc_root=str(tmp_path / "proc"), v2_root=str(tmp_path))
    assert out["version"] is None


# --------------------------------------------------------------------------
# rows and digests
# --------------------------------------------------------------------------
def test_equal_digests_mean_equal_bytes_dtype_and_shape(stand):
    import torch

    base = torch.tensor([1.0, 2.0, 3.0])
    assert stand.grad_digest(base) == stand.grad_digest(base.clone())
    bumped = base.clone()
    bumped[1] = torch.nextafter(bumped[1], torch.tensor(3.0))
    assert stand.grad_digest(bumped)["sha256"] != stand.grad_digest(base)["sha256"]
    assert stand.grad_digest(base.double())["dtype"] == "float64"
    assert stand.grad_digest(base.reshape(3, 1))["shape"] == [3, 1]
    assert stand.grad_digest(torch.zeros(2))["nonzero"] is False
    assert stand.grad_digest(torch.tensor([float("inf")]))["finite"] is False


def test_the_rows_fingerprint_sees_a_masked_label(stand):
    one = stand.make_rows("correctness", 64, k=4, seq=64, rows=10)
    two = stand.make_rows("correctness", 64, k=4, seq=64, rows=10)
    assert stand.rows_fingerprint(one) == stand.rows_fingerprint(two)
    two[0]["labels"][0] = -100  # row 0 masks nothing in make_rows
    assert stand.rows_fingerprint(one) != stand.rows_fingerprint(two)


def test_cost_rows_are_full_length_unmasked_and_cover_warmup_and_timed_steps(stand):
    rows = stand.make_rows("cost", 64, k=4, seq=512, rows=10)
    assert len(rows) == 4 * (stand.COST_WARMUP + stand.COST_STEPS)
    assert all(len(rows[i]["input_ids"]) == 512 for i in range(len(rows)))
    assert all(bool((rows[i]["labels"] != -100).all()) for i in range(len(rows)))
    assert len(stand.make_rows("calibrate", 64, k=4, seq=512, rows=10)) == 4


def test_correctness_rows_have_a_short_last_window(stand):
    rows = stand.make_rows("correctness", 64, k=4, seq=512, rows=stand.ROWS)
    lengths = stand.rows_fingerprint(rows)["lengths"]
    assert len(set(lengths)) > 1 and len(rows) % 4 != 0


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------
def _session(tmp_path, *extra):
    return ["session", "--weights", str(tmp_path / "w"), "--shards", str(tmp_path / "s"),
            "--spill-dir", str(tmp_path / "spill"), "--out", str(tmp_path / "out"), *extra]


def _printed(text):
    names, commands = [], {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# "):
            name = line[2:].split(":")[0]
            names.append(name)
            commands[name] = lines[i + 1]
    return names, commands


def test_plan_mode_prints_every_child_in_the_rule_order_and_writes_nothing(
    stand, tmp_path, capsys
):
    assert stand.main(_session(tmp_path)) == 0
    names, commands = _printed(capsys.readouterr().out)
    assert names == [
        "calibration", "calibration_check_1", "calibration_check_2",
        "l2l_nospill_1", "l2l_nospill_2", "l2l_spill", "ga_1", "ga_2", "ga_floor",
        "compare", "sigint", "kill9", "kill9_next",
        "cost/b00_p0_A", "cost/b00_p1_B", "cost/b00_p2_B", "cost/b00_p3_A",
    ]
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "out.tensors").exists()
    generous = f"MemoryMax={24 * GIB}"
    for name in ("calibration", "l2l_nospill_1", "l2l_nospill_2", "ga_1", "ga_2", "ga_floor",
                 "cost/b00_p0_A", "cost/b00_p3_A"):
        assert generous in commands[name], name
    for name in ("calibration_check_1", "l2l_spill", "sigint", "kill9", "kill9_next",
                 "cost/b00_p1_B", "cost/b00_p2_B"):
        assert "MemoryMax=<TIGHT>" in commands[name], name
    for name, command in commands.items():
        if name != "compare":
            assert "MemorySwapMax=0" in command and "--scope" in command, name


def test_the_kill9_pair_shares_one_spill_directory_and_others_do_not(stand, tmp_path, capsys):
    stand.main(_session(tmp_path))
    _names, commands = _printed(capsys.readouterr().out)

    def spill_dir(name):
        parts = commands[name].split()
        return parts[parts.index("--spill-dir") + 1]

    assert spill_dir("kill9") == spill_dir("kill9_next")
    assert len({spill_dir(n) for n in ("l2l_spill", "sigint", "kill9", "l2l_nospill_1")}) == 4
    assert "--marker" in commands["sigint"] and "--marker" in commands["kill9"]
    for name in ("l2l_spill", "kill9_next", "calibration", "cost/b00_p1_B"):
        assert "--marker" in commands[name], name  # first-step marker: build or training
    markers = {commands[n].split("--marker ")[1].split()[0] for n in commands if n != "compare"}
    assert len(markers) == len(commands) - 1  # one file per child, cost children included
    tensors = str(tmp_path / "out") + ".tensors"
    for name in ("l2l_nospill_1", "l2l_spill", "ga_floor"):
        assert f"--tensors-dir {tensors}" in commands[name]
    assert "--tensors-dir" not in commands["cost/b00_p1_B"]


def test_plan_mode_refuses_an_existing_out(stand, tmp_path, capsys):
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit) as caught:
        stand.main(_session(tmp_path))
    assert caught.value.code == 2
    assert "exists" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [["--k", "2"], ["--seq", "1024"]])
def test_the_session_refuses_another_k_or_seq(stand, tmp_path, extra):
    with pytest.raises(SystemExit):
        stand.main(_session(tmp_path, *extra))


def test_execute_without_systemd_run_refuses_and_writes_nothing(stand, tmp_path, capsys):
    with pytest.raises(SystemExit):
        stand.main(_session(tmp_path, "--execute", "--systemd-run",
                            str(tmp_path / "no-such-systemd-run")))
    assert "systemd-run" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def _fake_session(stand, tmp_path, monkeypatch, *, build_peak, step_extra, active_cache,
                  killed=None, killed_after_marker=True):
    """``session --execute`` with every child faked: the calibration child reports Mistral's
    plan with nothing spilled, the product's working set (used - inactive_file) at 9 GiB +
    ``active_cache``, of which ``active_cache`` is reclaimable active file cache, and the
    given transients; a check child reports 16 spilled layers. ``killed`` names a child the
    kernel OOM-kills after its first-step marker (no record, exit -9, the kernel's lines).
    Everything the session decides runs for real."""
    used = 9 * GIB + active_cache + GIB
    plan = {**_mistral_plan(), "limiting_level": "/scope", "spill_path": None,
            "limiting_numbers": {"used": used, "inactive_file": GIB}}
    transient = _calibration_transient(stand, build_peak=build_peak, step_extra=step_extra)
    ran = []

    def run_child(entry, out, env, deadline):
        ran.append(entry["name"])
        if entry["name"] == killed:
            limit = entry["memory_max"]
            marker = {"pid": 4204, "steps_done": 0, "in_step": True, "device": "cuda",
                      "expected_memory_max": limit,
                      "cgroup_start": {"version": 2, "memory_max": limit, "swap_max": 0}}
            return {**entry, "pid": 4200, "returncode": -9,
                    "marker_after_exit": marker if killed_after_marker else None,
                    "kernel_oom": {"read": True, "killed": True, "pids": [4200, 4204],
                                   "lines": [JOURNAL_KILL]}}
        record = {"exit": "normal", "device": "cuda", "spill_plan": dict(plan),
                  "cgroup_at_first_step": {"path": "/scope"}, "memory_transient": transient}
        if entry["name"].startswith("calibration_check"):
            record["spill_plan"]["spill_layers"] = 16
        stand.write_json(Path(entry["output"]), record)
        return {**entry, "returncode": 0}

    monkeypatch.setattr(stand, "run_child", run_child)
    monkeypatch.setattr(stand.sys, "platform", "linux")
    monkeypatch.setattr(stand.shutil, "which", lambda name: "/usr/bin/systemd-run")
    monkeypatch.setattr(stand.arm_validity, "ram_available", lambda: 64 * GIB)
    code = stand.main(_session(tmp_path, "--execute"))
    session = json.loads((tmp_path / "out" / "session.json").read_text())
    return code, session, ran


def test_a_build_margin_below_0_stops_the_session_before_any_child_at_the_tight_limit(
    stand, tmp_path, monkeypatch, capsys
):
    code, session, ran = _fake_session(stand, tmp_path, monkeypatch, build_peak=900 * MIB,
                                       step_extra=0, active_cache=0)
    assert code == 3
    assert session["state"] == "stopped_calibration_oom_risk"
    assert ran == ["calibration"]  # no check child, no correctness arm
    margin = session["limits"]["oom_margin"][0]
    assert margin["stop"] is True and margin["build_margin_bytes"] == 840 * MIB - 900 * MIB
    assert margin["limit_bytes"] == session["limits"]["calibration"]["limit_bytes"]
    assert f"{margin['build_margin_bytes']:,} B" in session["reason"]
    assert "STOPPED (stopped_calibration_oom_risk)" in capsys.readouterr().out
    verdict = json.loads((tmp_path / "out" / "verdict.json").read_text())
    assert verdict["verdict"].startswith("NO VERDICT")
    assert verdict["oom_margin"]["stop"] is True


def test_a_negative_training_margin_runs_the_correctness_arms_with_the_warning_recorded(
    stand, tmp_path, monkeypatch
):
    # nothing reclaimable beyond inactive_file, a 700 MiB training transient: training
    # margin 840 - 700 - 320 < 0 (a product finding if it OOMs), build margin 840 - 100 >= 0
    _code, session, ran = _fake_session(stand, tmp_path, monkeypatch, build_peak=100 * MIB,
                                        step_extra=700 * MIB, active_cache=0)
    margin = session["limits"]["oom_margin"][0]
    assert margin["stop"] is False
    assert margin["build_margin_bytes"] == 840 * MIB - 100 * MIB
    assert margin["train_margin_bytes"] == 840 * MIB - 700 * MIB - 320 * MIB
    assert [w.split(" margin")[0] for w in margin["warnings"]] == ["training"]
    assert ran[:3] == ["calibration", "calibration_check_1", "l2l_nospill_1"]
    assert session["limits"]["tight"] == session["limits"]["calibration"]["limit_bytes"]
    assert session["state"] != "stopped_calibration_oom_risk"
    verdict = json.loads((tmp_path / "out" / "verdict.json").read_text())
    assert verdict["oom_margin"]["warnings"] == margin["warnings"]


def test_reclaimable_active_cache_counts_toward_the_build_margin(stand, tmp_path, monkeypatch):
    # a 1000 MiB build transient fits only because 400 MiB of active file cache is reclaimable
    _code, session, ran = _fake_session(stand, tmp_path, monkeypatch, build_peak=1000 * MIB,
                                        step_extra=0, active_cache=400 * MIB)
    margin = session["limits"]["oom_margin"][0]
    assert margin["build_margin_bytes"] == 400 * MIB + 840 * MIB - 1000 * MIB
    assert margin["stop"] is False and "l2l_nospill_1" in ran


# --------------------------------------------------------------------------
# the kernel's word on an OOM kill (protocol §3.6, rule v2)
# --------------------------------------------------------------------------
#: Session 1's lines (kernel journal, 2026-10-27 07:40:11), reduced to the two that matter.
JOURNAL_KILL = ("1793086811.000000 ubuntu kernel: Memory cgroup out of memory: Killed process "
                "4204 (python) total-vm:17949912kB, anon-rss:1288440kB, file-rss:480312kB, "
                "shmem-rss:4507584kB, UID:0 pgtables:14252kB oom_score_adj:0")
JOURNAL_MEMCG = ("1793086811.000000 ubuntu kernel: oom-kill:constraint=CONSTRAINT_MEMCG,"
                 "nodemask=(null),cpuset=run-r1.scope,mems_allowed=0,oom_memcg=/system.slice/"
                 "run-r1.scope,task_memcg=/system.slice/run-r1.scope,task=python,pid=4204,uid=0")


def test_the_kernel_lines_name_the_pid_exactly(stand):
    text = "\n".join(["boot noise", JOURNAL_MEMCG, JOURNAL_KILL,
                      JOURNAL_KILL.replace("process 4204 (", "process 42045 (")])
    assert stand.oom_kill_lines(text, [4204]) == [JOURNAL_MEMCG, JOURNAL_KILL]
    assert stand.oom_kill_lines(text, [420]) == []


def test_a_child_killed_before_its_marker_is_found_by_its_scope_unit(stand):
    """systemd-run --scope runs the child under a pid Popen never sees; before the marker
    exists only the scope's unit name ties the kernel's lines to the child."""
    unit = stand.scope_unit(Path("/x/out"), "calibration_check_1")
    memcg = JOURNAL_MEMCG.replace("run-r1.scope", unit)
    text = "\n".join([memcg, JOURNAL_KILL, JOURNAL_KILL.replace("4204", "999")])
    assert stand.oom_kill_lines(text, [77], [unit]) == [memcg, JOURNAL_KILL]
    assert stand.oom_kill_lines(text, [77], ["a3-other.scope"]) == []
    args = argparse.Namespace(spill_dir="/s", weights="w", shards="h", k=4, seq=512,
                              tensors_dir="/t", systemd_run="systemd-run", user_scope=False)
    entry = stand.child_entry(args, Path("/x/out"), "calibration_check_1", {"tight": 7 * GIB})
    assert entry["scope_unit"] == unit and f"--unit={unit}" in entry["command"]


def test_a_kill_shows_up_on_the_second_read_of_the_kernel_log(stand):
    """journald reads /dev/kmsg on its own schedule: the first read may predate the line."""
    reads = iter([("nothing yet\n", None), (JOURNAL_KILL + "\n", None)])
    slept = []
    out = stand.kernel_oom([4200, 4204], 10.0, 20.0, read=lambda a, b: next(reads),
                           sleep=slept.append)
    assert out["killed"] is True and out["lines"] == [JOURNAL_KILL]
    assert slept == [stand.KERNEL_LOG_RETRY_S]


def test_a_sigkill_from_the_parent_is_not_an_oom(stand):
    """kill9's SIGKILL also exits -9; the kernel logs no OOM line for it."""
    out = stand.kernel_oom([4242], 10.0, 20.0, read=lambda a, b: ("some line\n", None),
                           sleep=lambda s: None)
    assert out == {"source": "journalctl -k", "pids": [4242], "scopes": [], "read": True,
                   "error": None, "killed": False, "lines": []}


def test_an_unreadable_kernel_log_is_reported_not_guessed(stand):
    out = stand.kernel_oom([7], 1.0, 2.0, read=lambda a, b: (None, "journalctl exited 1"),
                           sleep=lambda s: None)
    assert out["read"] is False and out["killed"] is False
    assert out["error"] == "journalctl exited 1"


def test_an_oom_in_the_check_child_stops_the_session_and_fails_ram(stand, tmp_path,
                                                                  monkeypatch, capsys):
    """Session 1's event under v2: the check child at the tight limit is OOM-killed after its
    first training_step. v1 said NO VERDICT; v2 fails criterion 2 and stops."""
    code, session, ran = _fake_session(stand, tmp_path, monkeypatch, build_peak=100 * MIB,
                                       step_extra=0, active_cache=0,
                                       killed="calibration_check_1")
    assert code == 3 and session["state"] == "stopped_oom_in_training"
    assert ran == ["calibration", "calibration_check_1"]
    assert "calibration_check_1: the kernel OOM-killed pid 4200" in session["reason"]
    verdict = json.loads((tmp_path / "out" / "verdict.json").read_text())
    assert verdict["verdict"] == "SPILL NOT SAFE IN THE TRAINER: criterion 2 failed"
    failed = [p for p in verdict["criteria"]["2"]["parts"] if p["status"] == "FAIL"]
    assert [p["detail"].split(":")[0] for p in failed] == ["calibration_check_1"]
    assert "after first training_step (marker at 0 optimizer steps done)" in failed[0]["detail"]


def test_an_oom_at_the_generous_limit_does_not_stop_the_session(stand, tmp_path, monkeypatch):
    """Only the tight limit tests the product's reserve; a GA arm's OOM at 24 GiB is not
    criterion 2's (its record is missing, so criterion 1b has no verdict)."""
    code, session, ran = _fake_session(stand, tmp_path, monkeypatch, build_peak=100 * MIB,
                                       step_extra=0, active_cache=0, killed="ga_1")
    assert session["state"] != "stopped_oom_in_training"
    assert "ga_2" in ran


def test_an_oom_in_the_build_at_the_tight_limit_is_not_called_training(stand, tmp_path,
                                                                     monkeypatch):
    """No first-step marker: the stand's limit, not the product's reserve. The check child
    has failed, so the calibration stops as before (exit 3, stopped), with no criterion 2
    FAIL."""
    code, session, ran = _fake_session(stand, tmp_path, monkeypatch, build_peak=100 * MIB,
                                       step_extra=0, active_cache=0,
                                       killed="calibration_check_1", killed_after_marker=False)
    assert code == 3 and session["state"] == "stopped"
    verdict = json.loads((tmp_path / "out" / "verdict.json").read_text())
    assert verdict["criteria"]["2"]["status"] == "NO VERDICT"
    assert not verdict["verdict"].startswith("SPILL NOT SAFE")


@pytest.mark.parametrize("limit, marker, stops", [
    ("tight", {"steps_done": 0}, True),
    ("tight", None, False),  # the build: the stand's limit
    ("generous", {"steps_done": 1}, False),  # not the product's reserve under test
])
def test_only_a_tight_child_killed_after_its_marker_stops_the_session(stand, limit, marker,
                                                                     stops):
    result = {"name": "x", "pid": 1, "memory_max": 5 * GIB, "limit": limit,
              "marker_after_exit": marker, "kernel_oom": {"killed": True}}
    assert (stand.oom_in_training(result) is not None) is stops
    assert stand.oom_in_training({**result, "kernel_oom": {"killed": False}}) is None


def test_the_memcg_line_alone_is_not_a_kill(stand):
    """The kernel names the victim before it kills it; only ``Killed process`` is the kill."""
    out = stand.kernel_oom([4204], 10.0, 20.0, read=lambda a, b: (JOURNAL_MEMCG + "\n", None),
                           sleep=lambda s: None)
    assert out["lines"] == [JOURNAL_MEMCG] and out["killed"] is False


def test_a_child_refuses_to_overwrite_its_output(stand, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    out = tmp_path / "arm.json"
    out.write_text("{}")
    assert stand.main(["child", "--arm", "l2l_spill", "--weights", "w", "--device", "cpu",
                       "--out", str(out)]) == 2
    assert out.read_text() == "{}"
    assert "REFUSED" in capsys.readouterr().out


def test_cost_blocks_count_only_complete_blocks_and_void_ones_are_replaced(stand, tmp_path):
    import arm_validity

    def arm(block, position, outcome="ok", value=1.0):
        role = arm_validity.BLOCK_ORDER[position]
        stand.write_json(tmp_path / "cost" / f"b{block:02d}_p{position}_{role}.json",
                         {"block": block, "position": position, "value": value,
                          "outcome": outcome})

    for position in range(4):
        arm(0, position)
        arm(1, position, outcome="void" if position == 2 else "ok")
    for position in range(3):
        arm(2, position)
    runs = stand.cost_runs(tmp_path)
    assert len(runs) == 2 and runs[1][2].outcome == "void"
    assert arm_validity.blocks_to_add(runs) == arm_validity.PILOT_BLOCKS - 1


# --------------------------------------------------------------------------
# the signal
# --------------------------------------------------------------------------
class FakeChild:
    pid = 77

    def __init__(self, code=None):
        self.code = code
        self.waits = []

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.code is None:
            raise subprocess.TimeoutExpired("child", timeout)
        return self.code

    def kill(self):
        self.code = -9


SPILL = "/spill/sigint/soup-l2l-spill-4242-0badcafe.bin"


def _drive(stand, tmp_path, child, *, send, phases=("enter", "enter", "hold"),
            clock=lambda: 0.0, deadline=float("inf")):
    marker = tmp_path / "marker.json"
    events, files, steps = [], [SPILL], iter(phases)

    def sleep(seconds):
        events.append(("sleep", seconds))
        stand.write_json(marker, {"pid": 4242, "steps_done": 1, "phase": next(steps),
                                  "spill_path": SPILL})

    def sender(pid, sig):
        events.append(("send", pid, sig))
        send(files)

    entry = {"marker": str(marker), "signal_name": "SIGINT", "spill_dir": "/spill/sigint"}
    out = stand.drive_signal(child, entry, deadline, list_files=lambda d: list(files),
                             send=sender, read_status=lambda pid: {"pid": pid}, sleep=sleep,
                             clock=clock, exit_timeout=0.01)
    return out, events


def test_one_signal_goes_to_the_markers_pid_once_it_holds(stand, tmp_path):
    child = FakeChild()

    def exits_clean(files):
        files.clear()
        child.code = 130

    out, events = _drive(stand, tmp_path, child, send=exits_clean)
    # Entry markers of step 2 are not enough: the signal waits for the hold marker.
    assert [e[0] for e in events] == ["sleep", "sleep", "sleep", "send"]
    assert events[-1] == ("send", 4242, int(signal.SIGINT))
    assert out["files_before_signal"] == [SPILL] and out["files_after_exit"] == []
    assert out["target_at_send"] == {"pid": 4242}
    assert out["exited"] is True and out["returncode"] == 130
    assert out["marker"]["phase"] == "hold" and out["sent_unix"] is not None


def test_a_child_that_does_not_exit_after_the_signal_is_killed_and_reported(stand, tmp_path):
    child = FakeChild()
    out, _events = _drive(stand, tmp_path, child, send=lambda files: None)
    assert out["exited"] is False and out["returncode"] == -9
    assert out["files_after_exit"] == [SPILL]


def test_no_signal_when_the_child_ends_before_it_holds(stand, tmp_path):
    out, events = _drive(stand, tmp_path, FakeChild(code=1), send=lambda files: None)
    assert out["sent_unix"] is None and "before it held" in out["why"]
    assert not [e for e in events if e[0] == "send"]


def test_no_signal_after_the_session_deadline(stand, tmp_path):
    out, events = _drive(stand, tmp_path, FakeChild(), send=lambda files: None,
                         phases=("enter",) * 3, clock=lambda: 10.0, deadline=5.0)
    assert out["sent_unix"] is None and out["why"] == "session deadline"
    assert not [e for e in events if e[0] == "send"]


def _held_signal_entry(stand, tmp_path):
    marker = tmp_path / "marker.json"
    stand.write_json(marker, {"pid": 4242, "steps_done": 1, "phase": "hold",
                              "spill_path": SPILL})
    return {"marker": str(marker), "signal_name": "SIGINT", "spill_dir": "/spill/sigint"}


@pytest.mark.parametrize("now", [5.0, 6.0])
def test_an_already_written_hold_cannot_bypass_the_deadline(stand, tmp_path, now):
    entry = _held_signal_entry(stand, tmp_path)
    sent = []
    out = stand.drive_signal(FakeChild(), entry, 5.0, list_files=lambda d: [SPILL],
                             send=lambda pid, sig: sent.append((pid, sig)),
                             read_status=lambda pid: {"pid": pid}, clock=lambda: now)
    assert out["sent_unix"] is None and out["why"] == "session deadline"
    assert sent == []


@pytest.mark.parametrize("slow_read", ["listing", "status"])
def test_pre_send_observations_cannot_spend_the_last_deadline_budget(
    stand, tmp_path, slow_read
):
    entry = _held_signal_entry(stand, tmp_path)
    now, sent = [0.0], []

    def files(directory):
        if slow_read == "listing":
            now[0] = 5.0
        return [SPILL]

    def status(pid):
        if slow_read == "status":
            now[0] = 5.0
        return {"pid": pid}

    out = stand.drive_signal(FakeChild(), entry, 5.0, list_files=files,
                             send=lambda pid, sig: sent.append((pid, sig)),
                             read_status=status, clock=lambda: now[0])
    assert out["sent_unix"] is None and out["why"] == "session deadline"
    assert sent == []


def test_marker_poll_sleep_is_limited_to_the_remaining_deadline(stand, tmp_path):
    now, slept = [4.99], []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    entry = {"marker": str(tmp_path / "missing.json"), "signal_name": "SIGINT",
             "spill_dir": "/spill/sigint"}
    out = stand.drive_signal(FakeChild(), entry, 5.0, list_files=lambda d: [],
                             send=lambda pid, sig: pytest.fail("expired parent sent a signal"),
                             sleep=sleep, clock=lambda: now[0])
    assert out["sent_unix"] is None
    assert slept == [pytest.approx(0.01)]


@pytest.mark.parametrize("pid_gone", [False, True])
@pytest.mark.parametrize("exit_timeout, expected_wait", [(600.0, 2.0), (1.0, 1.0)])
def test_signal_and_disappearing_target_waits_use_only_the_remaining_deadline(
    stand, tmp_path, pid_gone, exit_timeout, expected_wait
):
    entry = _held_signal_entry(stand, tmp_path)
    now = [0.0]

    class LiveLauncher(FakeChild):
        def wait(self, timeout=None):
            if self.code is None:
                assert timeout is not None, "live launcher wait must be bounded"
            return super().wait(timeout=timeout)

    child = LiveLauncher()

    def files(directory):
        now[0] += 1.0
        return [SPILL]

    def status(pid):
        now[0] += 1.0
        return {"pid": pid}

    def send(pid, sig):
        now[0] += 1.0
        if pid_gone:
            raise ProcessLookupError(pid)

    out = stand.drive_signal(child, entry, 5.0, list_files=files, send=send,
                             read_status=status, clock=lambda: now[0],
                             exit_timeout=exit_timeout)
    assert child.waits == [expected_wait, None]  # SIGKILL is followed by mandatory reap
    assert child.code == out["returncode"] == -9 and out["exited"] is False
    assert (out["sent_unix"] is None) is pid_gone


@pytest.mark.parametrize("pid_gone", [False, True])
def test_a_send_that_uses_the_last_budget_cannot_extend_the_exit_wait(
    stand, tmp_path, pid_gone
):
    entry = _held_signal_entry(stand, tmp_path)
    now, child = [0.0], FakeChild()

    def send(pid, sig):
        now[0] = 5.0
        if pid_gone:
            raise ProcessLookupError(pid)

    out = stand.drive_signal(child, entry, 5.0, list_files=lambda d: [SPILL], send=send,
                             read_status=lambda pid: {"pid": pid}, clock=lambda: now[0])
    assert child.waits == [0.0, None]
    assert out["returncode"] == -9 and out["exited"] is False


def test_a_disappearing_signal_target_can_still_reap_an_exited_launcher(stand, tmp_path):
    entry = _held_signal_entry(stand, tmp_path)
    child = FakeChild()

    def send(pid, sig):
        child.code = 0
        raise ProcessLookupError(pid)

    out = stand.drive_signal(child, entry, 5.0, list_files=lambda d: [], send=send,
                             read_status=lambda pid: None, clock=lambda: 3.0)
    assert child.waits == [2.0]
    assert out["sent_unix"] is None and out["returncode"] == 0 and out["exited"] is True


def _preflight_records(stand, tmp_path):
    path = "/spill/kill9/soup-l2l-spill-4242-0badcafe.bin"
    sigint = {"exit": "interrupted", "at_interrupt": {
        "in_step": True, "steps_done": 1, "hold": {"released": "interrupt", "pass": "backward"}},
        "spill_files_after_train": [],
        "sigint_handling": {"installed": {"handler": "default_int_handler"}}}
    rates = [0.001, 0.0005, 0.0]
    restart = {"arm": "kill9_next", "kind": "cleanup", "exit": "normal", "k": 4,
               "steps": 3, "expected_steps": 3, "stale_removed": [path],
               "spill_files_after_train": [], "rates": rates,
               "rows": {"count": 10, "row_sha256": ["a" * 64] * 10},
               "logs": [{"loss": 2.0 - index * 0.1, "learning_rate": rate}
                        for index, rate in enumerate(rates)],
               "grads": [{
                   f"base_model.model.model.layers.{layer}.self_attn."
                   f"{projection}_proj.lora_{matrix}.default.weight": {
                       "sha256": "b" * 64, "dtype": "float32",
                       "shape": [4, 64] if matrix == "A" else [64, 4],
                       "finite": True, "nonzero": True}
                   for layer in range(8) for projection in ("q", "k", "v", "o")
                   for matrix in ("A", "B")} for _ in range(3)],
               "spill_plan": {"spill_layers": 4},
               "store_io": {"bytes_written": 65_536, "bytes_read": 65_536}}
    results = {
        "sigint": {"signal": {"sent_unix": 1.0, "marker": {
            "phase": "hold", "steps_done": 1, "spill_path": SPILL},
            "files_before_signal": [SPILL], "files_after_exit": [],
            "exited": True, "returncode": 130}},
        "kill9": {"signal": {"sent_unix": 2.0, "marker": {
            "phase": "hold", "steps_done": 1, "spill_path": path},
            "files_before_signal": [path], "files_after_exit": [path], "returncode": -9}},
        "kill9_next": {"returncode": 0, "spill_files_after_exit": []},
    }
    stand.write_json(tmp_path / "cleanup" / "sigint.json", sigint)
    return restart, results


def test_preflight_requires_a_real_three_step_spill_restart(stand, tmp_path):
    preflight = _load("a3_signal_preflight")
    restart, results = _preflight_records(stand, tmp_path)
    stand.write_json(tmp_path / "cleanup" / "kill9_next.json", restart)
    assert all(check["pass"] for check in preflight.judge(tmp_path, results, scoped=False))


@pytest.mark.parametrize("observation", ["after_train", "after_exit"])
@pytest.mark.parametrize("missing", [None, "absent"])
def test_preflight_does_not_infer_cleanup_from_a_missing_listing(
    stand, tmp_path, observation, missing
):
    preflight = _load("a3_signal_preflight")
    restart, results = _preflight_records(stand, tmp_path)
    record, key = ((restart, "spill_files_after_train") if observation == "after_train"
                   else (results["kill9_next"], "spill_files_after_exit"))
    if missing == "absent":
        del record[key]
    else:
        record[key] = missing
    stand.write_json(tmp_path / "cleanup" / "kill9_next.json", restart)
    assert not preflight.judge(tmp_path, results, scoped=False)[-1]["pass"]


@pytest.mark.parametrize("damage", ["missing_steps", "zero_steps", "one_step", "no_grads",
                                    "zero_grads", "nonfinite_loss", "no_spill", "no_write",
                                    "no_read"])
def test_preflight_rejects_vacuous_invalid_or_nonspilling_restart(stand, tmp_path, damage):
    preflight = _load("a3_signal_preflight")
    restart, results = _preflight_records(stand, tmp_path)
    if damage == "missing_steps":
        del restart["steps"], restart["expected_steps"]
    elif damage in ("zero_steps", "one_step"):
        steps = 0 if damage == "zero_steps" else 1
        restart.update(steps=steps, expected_steps=steps, grads=restart["grads"][:steps],
                       logs=restart["logs"][:steps], rates=restart["rates"][:steps])
    elif damage == "no_grads":
        restart["grads"] = []
    elif damage == "zero_grads":
        for digest in restart["grads"][0].values():
            digest["nonzero"] = False
    elif damage == "nonfinite_loss":
        restart["logs"][0]["loss"] = float("nan")
    elif damage == "no_spill":
        restart["spill_plan"]["spill_layers"] = 0
    else:
        key = "bytes_written" if damage == "no_write" else "bytes_read"
        restart["store_io"][key] = 0
    stand.write_json(tmp_path / "cleanup" / "kill9_next.json", restart)
    assert not preflight.judge(tmp_path, results, scoped=False)[-1]["pass"]


@pytest.mark.parametrize("field", ["spill_layers", "bytes_written", "bytes_read"])
@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1, 0])
def test_preflight_requires_positive_observed_spill_participation(
    stand, tmp_path, field, value
):
    preflight = _load("a3_signal_preflight")
    restart, results = _preflight_records(stand, tmp_path)
    record = restart["spill_plan"] if field == "spill_layers" else restart["store_io"]
    record[field] = value
    stand.write_json(tmp_path / "cleanup" / "kill9_next.json", restart)
    assert not preflight.judge(tmp_path, results, scoped=False)[-1]["pass"]


def test_preflight_rejects_a_restart_missing_one_lora_parameter_at_every_step(stand, tmp_path):
    preflight = _load("a3_signal_preflight")
    restart, results = _preflight_records(stand, tmp_path)
    name = "base_model.model.model.layers.7.self_attn.o_proj.lora_B.default.weight"
    for step in restart["grads"]:
        del step[name]
    stand.write_json(tmp_path / "cleanup" / "kill9_next.json", restart)
    assert not preflight.judge(tmp_path, results, scoped=False)[-1]["pass"]


# --------------------------------------------------------------------------
# compare: the unpublished tensors
# --------------------------------------------------------------------------
def _grads(scale=1.0, steps=3):
    import torch

    gen = torch.Generator().manual_seed(11)
    return [{"lora_A": torch.randn(4, 8, generator=gen) * scale,
             "lora_B": torch.randn(8, 4, generator=gen) * scale} for _ in range(steps)]


def _write_arm(stand, session, tensors, name, grads):
    import torch

    folder = tensors / name
    folder.mkdir(parents=True)
    for step, grad in enumerate(grads):
        torch.save(grad, folder / f"step_{step:04d}.pt")
    steps = len(grads)
    stand.write_json(session / "arms" / f"{name}.json", {
        "arm": name, "exit": "normal", "dtype": "float32", "l2l": name.startswith("l2l"),
        "steps": steps, "expected_steps": steps,
        "rates": [0.001 * (steps - s) for s in range(steps)],
        "logs": [{"loss": 2.0 - 0.1 * s, "learning_rate": 0.001 * (steps - s)}
                 for s in range(steps)],
        "grads": [{n: stand.grad_digest(t) for n, t in sorted(g.items())} for g in grads],
    })


def _compare(stand, tmp_path, tamper=None):
    session, tensors = tmp_path / "session", tmp_path / "tensors"
    for name in ("l2l_nospill_1", "l2l_nospill_2", "l2l_spill", "ga_1", "ga_2", "ga_floor"):
        _write_arm(stand, session, tensors, name, _grads())
    if tamper:
        tamper(tensors)
    out = tmp_path / "compare.json"
    assert stand.compare_main(argparse.Namespace(session=str(session),
                                                 tensors_dir=str(tensors),
                                                 out=str(out))) == 0
    return json.loads(out.read_text())


def test_compare_checks_the_tensors_against_their_records_and_judges_ga(stand, tmp_path):
    out = _compare(stand, tmp_path)
    assert set(out["hashes_match"]) == {"l2l_nospill_1", "l2l_nospill_2", "l2l_spill",
                                        "ga_1", "ga_2", "ga_floor"}
    assert all(out["hashes_match"].values())
    assert out["nospill_repeat"]["equal"] and out["spill_vs_nospill"]["equal"]
    assert out["mode"] == "fp32" and out["tolerance"]["status"] == "within_tolerance"


def test_a_tensor_file_that_does_not_match_its_record_is_flagged_with_max_abs(stand, tmp_path):
    import torch

    def tamper(tensors):
        path = tensors / "l2l_spill" / "step_0001.pt"
        grad = torch.load(path, weights_only=True)
        grad["lora_A"][0, 0] += 0.25
        torch.save(grad, path)

    out = _compare(stand, tmp_path, tamper)
    assert out["hashes_match"]["l2l_spill"] is False
    assert out["hashes_match"]["l2l_nospill_1"] is True
    assert out["spill_vs_nospill"]["equal"] is False
    assert out["spill_vs_nospill"]["max_abs"] == pytest.approx(0.25)


# --------------------------------------------------------------------------
# a real child pair on a tiny model (CPU)
# --------------------------------------------------------------------------
#: A process id no OS hands out: above Linux's PID_MAX_LIMIT, and odd (Windows ids are
#: multiples of 4). A spill file with it is a dead run's file.
DEAD_PID = 4_206_649
#: Free RAM whose budget (240,000 B, what the 512 MiB reserve leaves) spills most of 8 layers
#: at k 4, seq 64 (as tests/test_l2l_spill_*).
SMALL_FREE = (512 << 20) + 240_000


@pytest.fixture()
def tiny(tmp_path, monkeypatch):
    """A tiny 8-layer Llama checkpoint; the product's budget sees no cgroup and the free RAM
    the test sets. Strict determinism is switched off again afterwards."""
    pytest.importorskip("peft")
    import torch

    from soup_cli.utils import l2l_spill, layer_stream
    from tests.test_v07204 import _tiny_llama_dir

    empty = tmp_path / "no-cgroup"
    empty.mkdir()
    for name in ("PROC_ROOT", "CGROUP_V2_ROOT", "CGROUP_V1_ROOT"):
        monkeypatch.setattr(l2l_spill, name, str(empty))
    probe = tmp_path / "probe"
    probe.mkdir()
    problem = l2l_spill.filesystem_refusal(os.path.realpath(probe))
    problem = problem or l2l_spill._round_trip(str(probe))
    if problem:
        pytest.skip(f"the temp directory cannot hold an unbuffered spill: {problem}")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    weights, _model, _config = _tiny_llama_dir(tmp_path / "ckpt", n_layers=8, tie=False)

    def set_free(nbytes):
        monkeypatch.setattr(layer_stream, "free_ram_bytes", lambda: nbytes)

    try:
        yield weights, set_free
    finally:
        torch.use_deterministic_algorithms(False)


def _child(stand, tmp_path, weights, arm):
    out = tmp_path / "session" / "arms" / f"{arm}.json"
    code = stand.main([
        "child", "--arm", arm, "--weights", weights, "--shards", str(tmp_path / "shards"),
        "--quant", "none", "--device", "cpu", "--spill-dir", str(tmp_path / "spill" / arm),
        "--k", "4", "--seq", "64", "--rows", "10", "--tensors-dir", str(tmp_path / "tensors"),
        "--out", str(out),
    ])
    return code, json.loads(out.read_text())


def test_a_spill_child_equals_a_no_spill_child_and_cleans_up(stand, rule, tmp_path, tiny):
    weights, set_free = tiny
    set_free(10**12)
    code, nospill = _child(stand, tmp_path, weights, "l2l_nospill_1")
    assert code == 0, nospill.get("traceback_tail")

    stale = tmp_path / "spill" / "l2l_spill" / f"soup-l2l-spill-{DEAD_PID}-0badcafe.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"\0" * 4096)
    set_free(SMALL_FREE)
    code, spill = _child(stand, tmp_path, weights, "l2l_spill")
    assert code == 0, spill.get("traceback_tail")

    assert nospill["spill_plan"]["spill_layers"] == 0
    assert 0 < spill["spill_plan"]["spill_layers"] < 8
    assert spill["store_io"]["bytes_written"] > 0 and spill["store_io"]["bytes_read"] > 0
    assert spill["spill_files_after_train"] == [] and not stale.exists()
    assert [os.path.basename(p) for p in spill["stale_removed"]] == [stale.name]
    assert spill["steps"] == spill["expected_steps"] == 3  # windows 4, 4, 2
    assert spill["deterministic"]["algorithms"] is True
    assert spill["deterministic"]["sdpa"] == "MATH"
    assert rule.training_problems(spill, n_layers=8) == []
    assert rule.training_problems(nospill, n_layers=8) == []
    assert rule.spill_problems(spill, spilled=True) == []
    assert rule.spill_problems(nospill, spilled=False) == []
    assert rule.fingerprint(spill) == rule.fingerprint(nospill)
    assert rule.compare_records(nospill, spill, n_layers=8)["equal"] is True

    out = tmp_path / "compare.json"
    stand.compare_main(argparse.Namespace(session=str(tmp_path / "session"),
                                          tensors_dir=str(tmp_path / "tensors"), out=str(out)))
    compared = json.loads(out.read_text())
    assert compared["hashes_match"] == {"l2l_nospill_1": True, "l2l_spill": True}
    assert compared["spill_vs_nospill"]["equal"] is True


# --------------------------------------------------------------------------
# the signal through real processes: an inherited SIG_IGN and the cleanup children (CPU)
# --------------------------------------------------------------------------
def _ignore_sigint_and_exec(argv, env):
    """Run ``argv`` with SIGINT ignored, the way a non-interactive shell's ``cmd &`` starts
    it (session 2's launch): the child inherits SIG_IGN across exec."""
    return subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))


_PROBE = """
import os, signal, sys, time
sys.path.insert(0, sys.argv[1])
import a3_spill_stand as stand
if sys.argv[2] == "restore":
    stand.restore_sigint()
print("ready", flush=True)
try:
    time.sleep(20)
    print("exit normal", flush=True)
except KeyboardInterrupt:
    print("exit interrupted", flush=True)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX signal inheritance")
@pytest.mark.parametrize("mode", ["restore", "inherit"])
def test_restore_sigint_undoes_an_inherited_ignore(stand, mode):
    """A real child started with SIGINT ignored, a real SIGINT: only the stand's restore makes
    it raise KeyboardInterrupt (``inherit`` is what session 2's sigint child did)."""
    harness = str(Path(stand.__file__).resolve().parent)
    child = _ignore_sigint_and_exec([sys.executable, "-c", _PROBE, harness, mode],
                                    {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert child.stdout.readline().strip() == b"ready"
    os.kill(child.pid, signal.SIGINT)
    if mode == "inherit":
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=2)
        child.kill()
    child.wait(timeout=30)
    expected = b"exit interrupted" if mode == "restore" else b""
    assert child.stdout.read().strip() == expected


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc and POSIX signals")
def test_the_cleanup_children_through_the_stand_with_sigint_ignored(tmp_path):
    """The preflight's three stand children of a tiny L2L spill model, started by a launcher
    whose SIGINT is ignored as in session 2: one SIGINT at the hold in optimizer step 2 raises
    KeyboardInterrupt there and the product removes its file; SIGKILL there leaves it; the
    next start removes it and trains."""
    pytest.importorskip("peft")
    from soup_cli.utils import l2l_spill

    problem = l2l_spill.filesystem_refusal(os.path.realpath(tmp_path))
    if problem:
        pytest.skip(f"the temp directory cannot hold an unbuffered spill: {problem}")
    preflight = Path(__file__).resolve().parents[1] / "benchmarks/harness/a3_signal_preflight.py"
    work = tmp_path / "preflight"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    launcher = _ignore_sigint_and_exec([sys.executable, str(preflight), "run", "--work",
                                        str(work), "--no-scope", "--hold-timeout", "20"], env)
    output = launcher.communicate(timeout=600)[0].decode(errors="replace")
    report = json.loads((work / "preflight.json").read_text())
    assert report["launcher"]["proc"]["ignored"] is True, report["launcher"]
    failed = [check for check in report["checks"] if not check["pass"]]
    assert failed == [] and launcher.returncode == 0, output
    inherited = report["children"]["sigint"]
    sigint = json.loads((work / "out/cleanup/sigint.json").read_text())
    assert sigint["sigint_handling"]["inherited"]["handler"] == "SIG_IGN"
    assert sigint["sigint_handling"]["inherited"]["proc"]["ignored"] is True
    assert inherited["signal"]["target_at_send"]["ignored"] is False
    assert inherited["signal"]["target_at_send"]["caught"] is True
