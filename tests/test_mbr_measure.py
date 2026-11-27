"""Measurement runner for the exact-MBR campaign: modes, OOM rows, MBPP data hash (pure CPU)."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from benchmarks.harness import mbr_measure as mm


def test_windows_are_the_four_agreed_pairs() -> None:
    assert mm.WINDOWS == ((8, 64), (8, 512), (64, 64), (64, 512))
    assert mm.BIT_LADDER == (1, 2, 4, 8)


def test_speed_in_deterministic_mode_is_refused() -> None:
    with pytest.raises(mm.MixedModeError, match="speed"):
        mm.check_mode("speed", deterministic=True)


def test_bits_in_normal_mode_are_refused() -> None:
    with pytest.raises(mm.MixedModeError, match="bits"):
        mm.check_mode("bits", deterministic=False)


def test_allowed_modes_return_their_label() -> None:
    assert mm.check_mode("speed", deterministic=False) == "normal"
    assert mm.check_mode("bits", deterministic=True) == "deterministic"


def test_a0_is_not_planned_above_k8() -> None:
    assert mm.arms_for(k=8) == ("A0", "B0", "C")
    assert mm.arms_for(k=64) == ("B0", "C")


def test_bits_reference_is_a0_through_k8_and_c_above_it() -> None:
    assert [mm.reference_arm(k) for k in (1, 2, 4, 8)] == ["A0"] * 4
    assert mm.reference_arm(64) == "C"
    assert mm.reference_arm(16) == "C"


def test_default_bit_plan_is_the_a0_b0_c_ladder_at_every_k() -> None:
    rows = mm.bit_plan(seq=64)
    assert [(r["k"], r["arm"]) for r in rows] == [
        (1, "A0"), (1, "B0"), (1, "C"), (2, "A0"), (2, "B0"), (2, "C"),
        (4, "A0"), (4, "B0"), (4, "C"), (8, "A0"), (8, "B0"), (8, "C")]
    assert all(r["seq"] == 64 and r["block"] == 0 and r["kind"] == "bits"
               and r["deterministic"] is True for r in rows)


def test_bit_plan_with_arms_gives_those_arms_at_each_k() -> None:
    rows = mm.bit_plan(seq=64, ks=(1, 2, 4, 8), arms=("A0", "O"))
    assert [(r["k"], r["arm"]) for r in rows] == [
        (1, "A0"), (1, "O"), (2, "A0"), (2, "O"), (4, "A0"), (4, "O"), (8, "A0"), (8, "O")]


def test_bit_plan_with_arms_drops_a0_above_k8_and_keeps_o() -> None:
    rows = mm.bit_plan(seq=64, ks=(8, 64), arms=("A0", "O"))
    assert [(r["k"], r["arm"]) for r in rows] == [(8, "A0"), (8, "O"), (64, "O")]


def test_dropped_arms_names_a0_above_k8_only() -> None:
    assert mm.dropped_arms(ks=(1, 8, 64), arms=("A0", "O")) == [(64, "A0")]
    assert mm.dropped_arms(ks=(1, 2, 4, 8), arms=("A0", "O")) == []
    assert mm.dropped_arms(ks=(64,), arms=("O",)) == []


def test_lever_bits_with_arms_a0_and_o_plans_o_rows(tmp_path, monkeypatch, capsys) -> None:
    planned = []

    def fake_sweep(args, plan):
        planned.extend(plan)
        return []

    monkeypatch.setattr(mm, "sweep", fake_sweep)
    mm.main(["lever-bits", "--source", "unused", "--cache", str(tmp_path / "cache"),
             "--out", str(tmp_path), "--arms", "A0,O", "--ladder-ks", "1,2,4,8,64"])
    assert [(item["k"], item["arm"]) for item in planned] == [
        (1, "A0"), (1, "O"), (2, "A0"), (2, "O"), (4, "A0"), (4, "O"),
        (8, "A0"), (8, "O"), (64, "O")]
    assert "k=64" in capsys.readouterr().err


def test_every_report_row_carries_the_mode() -> None:
    row = mm.make_row(arm="C", k=8, seq=64, kind="speed", deterministic=False,
                      result={"step_seconds": 1.5, "peak_allocated": 1, "peak_reserved": 2,
                              "peak_nvml": 3, "peak_ram": 4, "grad_sha256": "ab"},
                      dataset_sha256="d")
    assert row["mode"] == "normal" and row["status"] == "ok" and row["dataset_sha256"] == "d"
    assert row["peak_nvml"] == 3 and row["grad_sha256"] == "ab"


def test_oom_becomes_a_row_with_the_peak_not_an_exception() -> None:
    class FakeOomError(RuntimeError):
        pass

    def run() -> dict:
        raise FakeOomError("CUDA out of memory. Tried to allocate 2 GiB")

    peaks = {"peak_allocated": 7, "peak_reserved": 8, "peak_nvml": 9, "peak_ram": 10}
    out = mm.run_guarded(run, peaks=lambda: peaks, is_oom=lambda e: "out of memory" in str(e))
    assert out["status"] == "OOM"
    assert out["peak_allocated"] == 7 and out["peak_nvml"] == 9 and out["peak_ram"] == 10
    row = mm.make_row(arm="C", k=64, seq=512, kind="speed", deterministic=False, result=out,
                      dataset_sha256="d")
    assert row["status"] == "OOM" and row["mode"] == "normal"


def test_other_errors_are_not_swallowed() -> None:
    def run() -> dict:
        raise ValueError("bug")

    with pytest.raises(ValueError):
        mm.run_guarded(run, peaks=lambda: {}, is_oom=lambda e: False)


def test_a_successful_run_keeps_its_result_and_peaks() -> None:
    out = mm.run_guarded(lambda: {"step_seconds": 2.0}, peaks=lambda: {"peak_ram": 5},
                         is_oom=lambda e: False)
    assert out == {"step_seconds": 2.0, "peak_ram": 5, "status": "ok"}


def test_mbpp_records_use_real_text_and_the_dataset_hash_is_stable() -> None:
    rows = [{"text": "Write a function.", "code": "def f():\n    return 1", "task_id": 1},
            {"text": "Another task.", "code": "def g():\n    return 2", "task_id": 2}]

    class Tok:
        eos_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
            return {"input_ids": [ord(c) % 250 + 1 for c in text]}

    first = mm.mbpp_records(rows, Tok(), seq=64)
    again = mm.mbpp_records(list(rows), Tok(), seq=64)
    assert first["sha256"] == again["sha256"]
    assert all(len(r) == 64 for r in first["input_ids"])
    edited = [dict(rows[0], code="def f():\n    return 9"), rows[1]]
    changed = mm.mbpp_records(edited, Tok(), seq=64)
    assert changed["sha256"] != first["sha256"]


def test_mbpp_records_fill_k_rows_by_cycling_the_dataset() -> None:
    rows = [{"text": "t" * 5, "code": "c" * 5, "task_id": i} for i in range(3)]

    class Tok:
        eos_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
            return {"input_ids": [3] * len(text)}

    out = mm.mbpp_records(rows, Tok(), seq=16)
    window = mm.take_window(out["input_ids"], k=8)
    assert len(window) == 8 and window[3] == window[0]


def test_block_median_of_three_after_warmup() -> None:
    assert mm.median_after_warmup([9.0, 2.0, 1.0, 3.0]) == 2.0
    with pytest.raises(ValueError):
        mm.median_after_warmup([1.0])


def test_arm_options_are_a0_plain_l2l_b0_route_preserving_c_ordered() -> None:
    assert mm.arm_options("A0", k=8) == {}
    assert mm.arm_options("B0", k=8) == {"route_preserving_mbr": True, "mbr_group_size": 8}
    assert mm.arm_options("C", k=64) == {"route_preserving_mbr": True, "mbr_group_size": 64,
                                         "mbr_ordered_backward": True}
    with pytest.raises(ValueError):
        mm.arm_options("A0", k=64)
    with pytest.raises(ValueError):
        mm.arm_options("Z", k=8)


def test_speed_plan_is_a_b_c_c_b_a_blocks_over_the_arms_of_each_window() -> None:
    plan = mm.speed_plan(blocks=2)
    first = [p for p in plan if (p["k"], p["seq"]) == (8, 64)]
    assert [p["arm"] for p in first[:6]] == ["A0", "B0", "C", "C", "B0", "A0"]
    assert len(first) == 12
    big = [p for p in plan if p["k"] == 64 and p["seq"] == 64]
    assert [p["arm"] for p in big[:4]] == ["B0", "C", "C", "B0"] and len(big) == 8
    assert all(p["kind"] == "speed" and p["deterministic"] is False for p in plan)


def test_bit_plan_is_the_ladder_against_a0_in_deterministic_mode() -> None:
    plan = mm.bit_plan(seq=64)
    assert [(p["k"], p["arm"]) for p in plan] == [
        (k, arm) for k in mm.BIT_LADDER for arm in ("A0", "B0", "C")]
    assert all(p["kind"] == "bits" and p["deterministic"] is True for p in plan)


def test_grad_digest_changes_with_any_bit_and_ignores_dict_order() -> None:
    a = {"x": [1.0, 2.0], "y": [3.0]}
    assert mm.grad_digest(a) == mm.grad_digest({"y": [3.0], "x": [1.0, 2.0]})
    assert mm.grad_digest(a) != mm.grad_digest({"x": [1.0, 2.0000001], "y": [3.0]})


def test_ladder_verdict_compares_each_arm_digest_to_a0_and_skips_missing_a0() -> None:
    rows = [{"k": 1, "arm": "A0", "grad_sha256": "a", "loss": 1.0, "status": "ok"},
            {"k": 1, "arm": "B0", "grad_sha256": "a", "loss": 1.0, "status": "ok"},
            {"k": 1, "arm": "C", "grad_sha256": "b", "loss": 1.0, "status": "ok"},
            {"k": 2, "arm": "C", "grad_sha256": "c", "loss": 1.0, "status": "ok"}]
    out = mm.ladder_verdict(rows)
    assert out[(1, "B0")] == "equal" and out[(1, "C")] == "differs"
    assert out[(2, "C")] == "no A0"


def test_ladder_verdict_at_k64_compares_the_levers_to_c() -> None:
    rows = [{"k": 64, "arm": "C", "grad_sha256": "c", "loss": 1.0, "status": "ok"},
            {"k": 64, "arm": "E", "grad_sha256": "c", "loss": 1.0, "status": "ok"},
            {"k": 64, "arm": "F", "grad_sha256": "d", "loss": 1.0, "status": "ok"}]
    assert mm.ladder_verdict(rows) == {(64, "E"): "equal", (64, "F"): "differs"}
    assert mm.ladder_verdict([rows[1]])[(64, "E")] == "no C"
    assert mm.ladder_verdict([dict(rows[0], status="OOM"), rows[1]])[(64, "E")] == "no C"


def test_nvml_peak_parses_the_largest_used_memory_sample() -> None:
    assert mm.parse_nvml_used(["512 MiB", "7011 MiB", "6900 MiB"]) == 7011 * 2**20


def test_amdahl_share_needed_for_a_target_gain() -> None:
    # +10% total with a 2x local speedup needs 18.2% of the path; 1.5x needs 27.3%
    got = mm.amdahl_min_share(target_gain=0.10, local_speedup=2.0)
    assert got == pytest.approx(0.182, abs=1e-3)
    got = mm.amdahl_min_share(target_gain=0.10, local_speedup=1.5)
    assert got == pytest.approx(0.273, abs=1e-3)
    with pytest.raises(ValueError):
        mm.amdahl_min_share(target_gain=0.10, local_speedup=1.0)


def test_phase_shares_sum_to_one_and_name_the_largest_phase() -> None:
    summary = {"e_bwd": {"seconds": 6.0, "calls": 3}, "r1": {"seconds": 3.0, "calls": 3},
               "fwd": {"seconds": 1.0, "calls": 3}}
    shares = mm.phase_shares(summary)
    assert shares["e_bwd"] == pytest.approx(0.6) and sum(shares.values()) == pytest.approx(1.0)
    assert mm.phase_shares({}) == {}


def test_environment_record_keeps_versions_and_marks_missing_tools_unknown() -> None:
    def lookup(name: str) -> str | None:
        return {"torch": "2.14.0", "triton": "3.5.0"}.get(name)

    out = mm.environment_record(lookup, git_sha="abc", ptxas=None)
    assert out["versions"]["torch"] == "2.14.0" and out["versions"]["triton"] == "3.5.0"
    assert out["versions"]["bitsandbytes"] == "unknown"
    assert out["git_sha"] == "abc" and out["ptxas"] == "unknown"


def test_environment_record_carries_the_allocator_config_of_the_run() -> None:
    """Ticket 16 runs every hand with expandable segments: the row must say so."""
    def lookup(name: str) -> str | None:
        return None

    out = mm.environment_record(lookup, git_sha="abc", ptxas=None,
                                alloc_conf="expandable_segments:True")
    assert out["alloc_conf"] == "expandable_segments:True"
    assert mm.environment_record(lookup, git_sha="abc", ptxas=None)["alloc_conf"] == "default"


def test_direct_io_flag_is_recorded_and_a_silent_fallback_is_called_out() -> None:
    assert mm.direct_io_record(requested=True, actual=True) == {"requested": True,
                                                                 "actual": True, "fallback": False}
    assert mm.direct_io_record(requested=True, actual=False)["fallback"] is True


def test_profile_plan_is_one_normal_mode_run_per_arm_and_window_flagged_for_profiling() -> None:
    plan = mm.profile_plan()
    assert len(plan) == 3 + 3 + 2 + 2
    assert all(p["profile"] and p["kind"] == "speed" and p["deterministic"] is False
               for p in plan)
    assert not any(p["arm"] == "A0" and p["k"] > 8 for p in plan)


def test_speed_plan_is_block_major_so_a_cut_run_keeps_every_window() -> None:
    plan = mm.speed_plan(blocks=2)
    first_block = [p for p in plan if p["block"] == 0]
    assert {(p["k"], p["seq"]) for p in first_block} == set(mm.WINDOWS)
    assert plan[:len(first_block)] == first_block


def test_summarize_rows_takes_median_step_and_max_peaks_per_arm_and_window() -> None:
    def row(arm, step, nvml, status="ok"):
        return {"arm": arm, "k": 8, "seq": 64, "kind": "speed", "mode": "normal",
                "status": status, "step_seconds": step, "peak_allocated": 10, "peak_reserved": 20,
                "peak_nvml": nvml, "peak_ram": 5}

    rows = [row("C", 10.0, 100), row("C", 14.0, 120), row("C", 12.0, 110),
            row("B0", 20.0, 90), {"arm": "C", "k": 64, "seq": 512, "kind": "speed",
                                  "mode": "normal", "status": "OOM", "peak_nvml": 8000}]
    table = mm.summarize_rows(rows)
    cell = table[("C", 8, 64)]
    assert cell["step_seconds"] == 12.0 and cell["peak_nvml"] == 120 and cell["runs"] == 3
    oom = table[("C", 64, 512)]
    assert oom["status"] == "OOM" and oom["step_seconds"] is None


def test_markdown_table_has_one_line_per_arm_and_window_and_shows_oom() -> None:
    table = {("C", 8, 64): {"step_seconds": 12.0, "peak_allocated": 2**30,
                            "peak_reserved": 2**30, "peak_nvml": 2**31, "peak_ram": 2**32,
                            "runs": 3, "status": "ok"},
             ("C", 64, 512): {"step_seconds": None, "peak_allocated": None, "peak_reserved": None,
                              "peak_nvml": 8000, "peak_ram": None, "runs": 1, "status": "OOM"}}
    text = mm.markdown_table(table)
    assert "| C | 8 | 64 | 12.0 |" in text and "OOM" in text
    assert text.count("\n") >= 3


# ---- lever arms (ticket 02): E = C + lt-fused, F = E + async_lease, G = F + fast_row_gather ----
def test_lever_arms_stack_on_the_ordered_arm() -> None:
    c = mm.arm_options("C", k=8)
    for arm in ("E", "F", "G"):
        assert mm.arm_options(arm, k=8) == c  # the step options are C's; the levers sit outside
    assert mm.arm_runtime("C") == {"expert_gemm": "native", "async_lease": False,
                                   "fast_row_gather": False, "expert_loader": "async_lease"}
    assert mm.arm_runtime("E") == {"expert_gemm": "lt-fused", "async_lease": False,
                                   "fast_row_gather": False, "expert_loader": "async_lease"}
    assert mm.arm_runtime("F")["async_lease"] and not mm.arm_runtime("F")["fast_row_gather"]
    assert mm.arm_runtime("G") == {"expert_gemm": "lt-fused", "async_lease": True,
                                   "fast_row_gather": True, "expert_loader": "async_lease"}
    assert mm.arm_runtime("A0") == mm.arm_runtime("B0") == mm.arm_runtime("C")


def test_lever_speed_plan_is_symmetric_and_normal_mode() -> None:
    plan = mm.lever_speed_plan(blocks=1)
    assert len(plan) == 8 * len(mm.WINDOWS)
    first = [p["arm"] for p in plan[:8]]
    assert first == ["C", "E", "F", "G", "G", "F", "E", "C"]
    assert all(p["kind"] == "speed" and not p["deterministic"] for p in plan)


def test_lever_bit_plan_covers_a0_and_every_lever_in_deterministic_mode() -> None:
    plan = mm.lever_bit_plan(seq=64)
    assert {p["arm"] for p in plan} == {"A0", "C", "D", "E", "F", "G", "H"}
    assert {p["k"] for p in plan} == set(mm.BIT_LADDER)
    assert all(p["kind"] == "bits" and p["deterministic"] for p in plan)


def test_k64_ladder_uses_c_as_reference_and_plans_no_a0() -> None:
    assert mm.ladder_arms(8) == ("A0", "C", "E", "F", "G", "H", "D")
    assert mm.ladder_arms(64) == ("C", "E", "F", "G", "H", "D")
    lever = mm.lever_bit_plan(seq=64, ks=(64,))
    assert [p["arm"] for p in lever] == ["C", "E", "F", "G", "H", "D"]
    plain = mm.bit_plan(seq=64, ks=(64,))
    assert [p["arm"] for p in plain] == ["B0", "C"]
    assert all(p["k"] == 64 and p["kind"] == "bits" and p["deterministic"]
               for p in lever + plain)


def test_ladder_flags_parse_k_values_and_known_arm_names() -> None:
    assert mm.parse_k_list("1,2,4,8") == (1, 2, 4, 8)
    assert mm.parse_k_list("64") == (64,)
    assert mm.parse_arm_list("C, E") == ("C", "E")
    with pytest.raises(argparse.ArgumentTypeError):
        mm.parse_k_list("0")
    with pytest.raises(argparse.ArgumentTypeError):
        mm.parse_arm_list("Z")


def _speed(arm: str, k: int, seq: int, step: float, nvml: int = 100, block: int = 0) -> dict:
    return {"arm": arm, "k": k, "seq": seq, "kind": "speed", "status": "ok", "block": block,
            "step_seconds": step, "peak_allocated": nvml, "peak_reserved": nvml,
            "peak_nvml": nvml}


def _gate_rows(gain: float, nvml_lever: int = 100, windows: int = 4) -> list:
    rows = []
    for k, seq in mm.WINDOWS[:windows]:
        rows += [_speed("C", k, seq, 100.0), _speed("E", k, seq, 100.0 * (1 - gain), nvml_lever)]
    return rows


def test_a_lever_with_a_three_percent_gain_in_three_windows_passes_speed() -> None:
    rows = _gate_rows(0.04, windows=3) + [_speed("C", 64, 512, 100.0), _speed("E", 64, 512, 101.0)]
    gate = mm.lever_gate(rows, lever="E", previous="C")
    assert gate["speed"]["windows_won"] == 3 and gate["speed"]["pass"] is True
    assert gate["speed"]["t_test"] == "not computed (fewer than 5 blocks)"


def test_two_winning_windows_fail_speed_unless_vram_needs_the_lever() -> None:
    rows = _gate_rows(0.04, windows=2) + [_speed("C", 64, 64, 100.0), _speed("E", 64, 64, 100.0),
                                          _speed("C", 64, 512, 100.0), _speed("E", 64, 512, 100.0)]
    assert mm.lever_gate(rows, lever="E", previous="C")["speed"]["pass"] is False


def test_vram_worse_in_one_window_fails_and_an_oom_relief_counts() -> None:
    worse = _gate_rows(0.05, nvml_lever=101)
    assert mm.lever_gate(worse, lever="E", previous="C")["vram"]["pass"] is False
    rows = _gate_rows(0.05)
    rows[-2] = {"arm": "C", "k": 64, "seq": 512, "kind": "speed", "status": "OOM",
                "peak_allocated": 8, "peak_reserved": 8, "peak_nvml": 9, "block": 0}
    gate = mm.lever_gate(rows, lever="E", previous="C")
    assert gate["vram"]["pass"] is True and gate["vram"]["oom_relief"] == [(64, 512)]


def test_bits_gate_follows_the_ladder_against_a0() -> None:
    def bit(arm: str, k: int, sha: str) -> dict:
        return {"arm": arm, "k": k, "seq": 64, "kind": "bits", "status": "ok",
                "grad_sha256": sha, "loss": 1.0}

    ok = [bit("A0", k, "x") for k in mm.BIT_LADDER] + [bit("E", k, "x") for k in mm.BIT_LADDER]
    assert mm.lever_gate(ok, lever="E", previous="C")["bits"]["pass"] is True
    broken = ok[:-1] + [bit("E", 8, "y")]
    gate = mm.lever_gate(broken, lever="E", previous="C")
    assert gate["bits"]["pass"] is False and gate["bits"]["differs_at"] == [8]


def test_bits_gate_counts_a_k64_verdict_against_the_c_reference() -> None:
    def bit(arm: str, k: int, sha: str) -> dict:
        return {"arm": arm, "k": k, "seq": 64, "kind": "bits", "status": "ok",
                "grad_sha256": sha, "loss": 1.0}

    rows = [bit("A0", k, "x") for k in mm.BIT_LADDER]
    rows += [bit("E", k, "x") for k in mm.BIT_LADDER]
    rows += [bit("C", 64, "y"), bit("E", 64, "y")]
    assert mm.lever_gate(rows, lever="E", previous="C")["bits"]["pass"] is True
    rows[-1] = bit("E", 64, "z")
    gate = mm.lever_gate(rows, lever="E", previous="C")
    assert gate["bits"]["pass"] is False and gate["bits"]["differs_at"] == [64]


def test_sweep_child_run_forwards_the_model_options_to_every_window() -> None:
    args = argparse.Namespace(source="SRC", cache="CACHE", steps=4, out="OUT", device="auto",
                              tokenizer=None, lora_r=8,
                              lora_targets=("q_proj", "kv_a_proj_with_mqa"))
    item = {"arm": "C", "k": 8, "seq": 64, "kind": "bits", "deterministic": True,
            "profile": False}
    argv = mm.child_run_args(args, item)
    assert argv[0] == "run"
    assert argv[argv.index("--lora-targets") + 1] == "q_proj,kv_a_proj_with_mqa"
    assert argv[argv.index("--lora-r") + 1] == "8"
    assert "--deterministic" in argv and "--profile" not in argv


def test_lora_targets_default_is_the_kanana_list_and_the_flag_overrides_it() -> None:
    assert mm.parse_lora_targets(None) == ("q_proj", "kv_b_proj", "o_proj")
    assert mm.parse_lora_targets("q_proj, kv_a_proj_with_mqa ,o_proj") == (
        "q_proj", "kv_a_proj_with_mqa", "o_proj")
    with pytest.raises(argparse.ArgumentTypeError, match="at least one module name"):
        mm.parse_lora_targets(" , ")


def test_abba_plan_runs_a0_c_c_a0_per_block_on_the_k8_windows_only() -> None:
    plan = mm.abba_plan(blocks=2)
    assert {(p["k"], p["seq"]) for p in plan} == {(8, 64), (8, 512)}
    first_window_block0 = [p["arm"] for p in plan if p["block"] == 0 and p["seq"] == 64]
    assert first_window_block0 == ["A0", "C", "C", "A0"]
    assert len(plan) == 2 * 2 * 4
    assert all(p["kind"] == "speed" and p["deterministic"] is False for p in plan)


def test_arm_watch_checks_every_run_and_watches_the_model_folder() -> None:
    import argparse

    args = argparse.Namespace(source="/models/moon", label="RTX 3050 Laptop, real")
    kwargs = mm.arm_watch_kwargs(args, arm="C", round_number=1, run=3)
    assert kwargs == {"arm": "C", "round": 1, "run": 3, "label": "RTX 3050 Laptop, real",
                      "required": ("suspend", "power", "foreign_reads"),
                      "model_path": "/models/moon"}


def test_abba_plan_can_resume_at_a_later_block_with_global_run_indexes() -> None:
    plan = mm.abba_plan(blocks=1, start_block=1)
    assert {p["block"] for p in plan} == {1}
    assert [p["index"] for p in plan] == list(range(8, 16))
    assert [p["arm"] for p in plan[:4]] == ["A0", "C", "C", "A0"]


def test_default_label_is_the_one_the_speed_rule_names() -> None:
    assert mm.DEFAULT_LABEL == "RTX 3050 Laptop, real"


def test_run_number_follows_the_plan_index_and_falls_back_to_position() -> None:
    assert mm.run_number({"index": 9}, position=0) == 9
    assert mm.run_number({}, position=4) == 4


def test_append_jsonl_adds_one_line_per_call(tmp_path) -> None:
    target = tmp_path / "rows.jsonl"
    mm.append_jsonl(target, {"arm": "A0"})
    mm.append_jsonl(target, {"arm": "C"})
    assert target.read_text().splitlines() == ['{"arm": "A0"}', '{"arm": "C"}']


def test_run_fit_refuses_to_run_over_existing_rows(tmp_path) -> None:
    import subprocess

    script = (mm.__file__.rsplit("benchmarks", 1)[0]
              + "benchmarks/mbr-moonlight-fit-20261121/run_fit.sh")
    (tmp_path / "fit.jsonl").write_text("{}\n")
    env = {"HOME": str(tmp_path), "OUT": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "refusing to run" in result.stderr



def test_tensor_file_name_is_arm_k_and_seq() -> None:
    assert mm.tensor_file_name("E", 8, 512) == "E-k8-s512.pt"


def test_bit_tensor_dir_is_the_tensors_folder_under_out(tmp_path) -> None:
    assert mm.bit_tensor_dir(tmp_path) == tmp_path / "tensors"


def test_save_bit_tensors_round_trips_loss_and_grads_exactly(tmp_path) -> None:
    import torch

    grad = torch.tensor([1.0, -2.0])
    path = tmp_path / "A0-k1-s64.pt"
    mm.save_bit_tensors(path, losses=[1.5, 0.25], grads={"w": grad})
    loaded = torch.load(path, weights_only=True)
    assert loaded["loss"].dtype == torch.float64
    assert loaded["loss"].tolist() == [1.5, 0.25]
    assert torch.equal(loaded["grads"]["w"], grad.detach())


def test_tensors_equal_is_true_for_identical_loss_and_grads() -> None:
    import torch

    first = {"loss": torch.tensor([1.5], dtype=torch.float64),
             "grads": {"w": torch.tensor([1.0, -2.0])}}
    second = {"loss": torch.tensor([1.5], dtype=torch.float64),
              "grads": {"w": torch.tensor([1.0, -2.0])}}
    assert mm.tensors_equal(first, second)


def test_tensors_equal_flags_a_one_ulp_change_in_one_grad() -> None:
    import torch

    weights = torch.tensor([1.0, -2.0])
    first = {"loss": torch.tensor([1.5], dtype=torch.float64), "grads": {"w": weights}}
    nudged = weights.clone()
    nudged[1] = torch.nextafter(nudged[1], torch.tensor(0.0))
    second = {"loss": torch.tensor([1.5], dtype=torch.float64), "grads": {"w": nudged}}
    assert not mm.tensors_equal(first, second)


def test_tensors_equal_needs_the_same_grad_names_and_the_same_loss() -> None:
    import torch

    base = {"loss": torch.tensor([1.5], dtype=torch.float64), "grads": {"w": torch.ones(2)}}
    missing = {"loss": base["loss"], "grads": {"v": torch.ones(2)}}
    other_loss = {"loss": torch.tensor([1.5000001], dtype=torch.float64),
                  "grads": {"w": torch.ones(2)}}
    assert not mm.tensors_equal(base, missing)
    assert not mm.tensors_equal(base, other_loss)


def test_a_nan_grad_counts_as_different_even_against_itself() -> None:
    import torch

    nan_run = {"loss": torch.tensor([1.5], dtype=torch.float64),
               "grads": {"w": torch.tensor([float("nan")])}}
    assert not mm.tensors_equal(nan_run, nan_run)


def _saved_bits_row(tmp_path, arm: str, k: int, grad: list, *, seq: int = 64) -> dict:
    """A bits row bound to its saved payload: loss total, digest of the saved grad and manifest."""
    import torch

    name = mm.tensor_file_name(arm, k, seq)
    payload = {"loss": torch.tensor([2.0] + [0.0] * (k - 1), dtype=torch.float64),
               "grads": {"w": torch.tensor(grad)}}
    torch.save(payload, tmp_path / name)
    flat = {"w": payload["grads"]["w"].float().flatten().tolist()}
    return {"arm": arm, "k": k, "seq": seq, "status": "ok", "tensor_file": name, "loss": 2.0,
            "grad_sha256": mm.grad_digest(flat), "grad_shapes": {"w": [len(grad)]}}


def test_tensor_ladder_verdict_reads_saved_tensors_and_compares_each_arm_to_a0(tmp_path) -> None:
    def save(arm: str, k: int, grad: list) -> dict:
        return _saved_bits_row(tmp_path, arm, k, grad)

    rows = [save("A0", 1, [1.0, 2.0]), save("C", 1, [1.0, 2.0]),
            save("E", 1, [1.0, 2.0000002]), {"arm": "F", "k": 1, "seq": 64, "status": "OOM"},
            save("C", 2, [0.5, 0.5])]
    out = mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path, manifest={"w": [2]})
    assert out[(1, "C")] == "equal"
    assert out[(1, "E")] == "differs"
    assert out[(1, "F")] == "OOM"
    assert out[(2, "C")] == "no A0"


def test_a_failed_a0_row_leaves_its_k_without_a_reference(tmp_path) -> None:
    import torch

    name = mm.tensor_file_name("C", 1, 64)
    torch.save({"loss": torch.tensor([2.0], dtype=torch.float64),
                "grads": {"w": torch.tensor([1.0])}}, tmp_path / name)
    rows = [{"arm": "A0", "k": 1, "seq": 64, "status": "OOM"},
            {"arm": "C", "k": 1, "seq": 64, "status": "ok", "tensor_file": name}]
    assert mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path,
                                    manifest={"w": [1]})[(1, "C")] == "no A0"


def test_tensor_ladder_verdict_at_k64_reads_the_saved_c_tensors(tmp_path) -> None:
    def save(arm: str, grad: list) -> dict:
        return _saved_bits_row(tmp_path, arm, 64, grad)

    rows = [save("C", [1.0, 2.0]), save("E", [1.0, 2.0])]
    manifest = {"w": [2]}
    assert mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path,
                                    manifest=manifest) == {(64, "E"): "equal"}
    assert mm.tensor_ladder_verdict(rows[:1] + [
        dict(rows[1], status="OOM")], tensor_dir=tmp_path, manifest=manifest)[(64, "E")] == "OOM"


def _archived_bits_row(tmp_path, arm: str, k: int, grad: list, *, seq: int = 64) -> dict:
    """A bits row as the archived campaigns wrote it: no ``grad_shapes``, only the digest."""
    row = _saved_bits_row(tmp_path, arm, k, grad, seq=seq)
    return {name: value for name, value in row.items() if name != "grad_shapes"}


MANIFEST_2 = {"w": [2]}


def test_an_explicit_manifest_gives_verdicts_for_archived_rows_bound_to_their_digests(
        tmp_path) -> None:
    rows = [_archived_bits_row(tmp_path, "A0", 1, [1.0, 2.0]),
            _archived_bits_row(tmp_path, "C", 1, [1.0, 2.0]),
            _archived_bits_row(tmp_path, "E", 1, [1.0, 2.0000002])]
    out = mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path, manifest=MANIFEST_2)
    assert out == {(1, "C"): "equal", (1, "E"): "differs"}


def test_an_explicit_manifest_still_refuses_a_payload_that_fails_its_row_digest(tmp_path) -> None:
    base = _archived_bits_row(tmp_path, "A0", 1, [1.0, 2.0])
    other = _archived_bits_row(tmp_path, "E", 1, [1.0, 3.0])
    other["grad_sha256"] = base["grad_sha256"]
    out = mm.tensor_ladder_verdict([base, other], tensor_dir=tmp_path, manifest=MANIFEST_2)
    assert out == {(1, "E"): "no verdict: row:digest_mismatch"}


def test_a_mismatched_manifest_gives_no_verdict(tmp_path) -> None:
    rows = [_archived_bits_row(tmp_path, "A0", 1, [1.0, 2.0]),
            _archived_bits_row(tmp_path, "C", 1, [1.0, 2.0])]
    wrong_shape = mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path, manifest={"w": [3]})
    missing_name = mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path,
                                            manifest={"w": [2], "v": [2]})
    no_manifest = mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path, manifest={})
    reason = "no verdict: base:invalid_gradient_coverage"
    assert wrong_shape[(1, "C")].startswith(reason), wrong_shape
    assert missing_name[(1, "C")].startswith(reason), missing_name
    assert no_manifest[(1, "C")] == "no verdict: base:no_expected_grads, row:no_expected_grads"


def test_a_row_cannot_borrow_a_payload_under_another_arms_file_name(tmp_path) -> None:
    good = _saved_bits_row(tmp_path, "A0", 1, [1.0, 2.0])
    borrowed = dict(good, arm="E", grad_sha256="0" * 64)
    out = mm.tensor_ladder_verdict([good, borrowed], tensor_dir=tmp_path, manifest={"w": [2]})
    assert out == {(1, "E"): "no verdict: row:tensor_file_name"}, out


def test_each_row_is_checked_against_its_own_digest_not_a_cached_one(tmp_path) -> None:
    base = _saved_bits_row(tmp_path, "A0", 1, [1.0, 2.0])
    first = _saved_bits_row(tmp_path, "E", 1, [1.0, 2.0])
    second = dict(first, grad_sha256="0" * 64)
    out = mm.tensor_ladder_verdict([base, first, second], tensor_dir=tmp_path,
                                   manifest={"w": [2]})
    assert out == {(1, "E"): "no verdict: row:digest_mismatch"}, out


def test_a_tensor_path_that_leaves_the_folder_is_never_read(tmp_path) -> None:
    import shutil

    folder = tmp_path / "tensors"
    folder.mkdir()
    base = _saved_bits_row(folder, "A0", 1, [1.0, 2.0])
    shutil.copy(folder / base["tensor_file"], tmp_path / "escape.pt")
    escaped = dict(base, arm="E", tensor_file="../escape.pt")
    out = mm.tensor_ladder_verdict([base, escaped], tensor_dir=folder, manifest={"w": [2]})
    assert out == {(1, "E"): "no verdict: row:missing_tensor_file"}, out
    assert mm.contained_tensor(folder, "../escape.pt") is None
    inside = mm.contained_tensor(folder, base["tensor_file"])
    assert inside == (folder / base["tensor_file"]).resolve()


def test_two_missing_digests_or_losses_never_read_as_equal() -> None:
    row = {"arm": "A0", "k": 1, "seq": 64, "status": "ok", "grad_sha256": "", "loss": 1.0}
    empty = mm.ladder_verdict([row, dict(row, arm="E")])
    assert empty == {(1, "E"): "no verdict: missing_digest"}, empty
    no_loss = {"arm": "E", "k": 1, "seq": 64, "status": "ok", "grad_sha256": "x"}
    assert mm.ladder_verdict([dict(row, grad_sha256="x", loss=None), no_loss]) == {
        (1, "E"): "no verdict: missing_loss"}
    no_digest = {"arm": "E", "k": 1, "seq": 64, "status": "ok", "loss": 1.0}
    assert mm.ladder_verdict([row, no_digest]) == {(1, "E"): "no verdict: missing_digest"}


def test_the_sweep_reports_through_a_rich_console_not_print() -> None:
    import ast
    import inspect

    calls = [node.lineno for node in ast.walk(ast.parse(inspect.getsource(mm.sweep)))
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"]
    assert calls == []


def test_an_incomplete_deadline_stop_fails_a_required_sweep(tmp_path, monkeypatch, capsys):
    import time

    clock = iter([0.0] + [1e9] * 100)
    monkeypatch.setattr(time, "time", lambda: next(clock))
    argv = ["ceg-speed", "--source", "s", "--cache", "c", "--out", str(tmp_path),
            "--deadline-hours", "1", "--require-complete"]
    assert mm.main(argv) == 1
    assert "deadline reached, 0 of" in capsys.readouterr().out
    assert not (tmp_path / "rows.jsonl").exists()


@pytest.mark.parametrize("command", ["bits", "speed"])
def test_an_incomplete_deadline_stop_fails_a_bits_or_speed_sweep_without_the_flag(
        tmp_path, monkeypatch, capsys, command):
    import time

    clock = iter([0.0] + [1e9] * 100)
    monkeypatch.setattr(time, "time", lambda: next(clock))
    argv = [command, "--source", "s", "--cache", "c", "--out", str(tmp_path),
            "--deadline-hours", "1"]
    assert mm.main(argv) == 1
    assert "deadline reached, 0 of" in capsys.readouterr().out
    assert not (tmp_path / "bits-verdict.json").exists()


def test_a_required_bits_sweep_without_tensors_exits_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mm, "sweep", lambda args, plan: [
        dict(item, status="ok", outcome="ok", grad_sha256="same", loss=1.0, step_seconds=1.0,
             tensor_file=None) for item in plan])
    argv = ["bits", "--source", "s", "--cache", "c", "--out", str(tmp_path), "--require-complete"]
    assert mm.main(argv) == 1
    assert "Bit run unsuccessful or missing tensor" in capsys.readouterr().err
    assert not (tmp_path / "bits-verdict.json").exists()


@pytest.mark.parametrize("command", ["bits", "lever-bits", "ds-bits"])
def test_an_incomplete_bits_sweep_fails_without_the_flag(tmp_path, monkeypatch, capsys, command):
    monkeypatch.setattr(mm, "sweep", lambda args, plan: [])
    argv = [command, "--source", "s", "--cache", "c", "--out", str(tmp_path), "--ladder-ks", "1"]
    assert mm.main(argv) == 1
    assert "Incomplete sweep: 0 of" in capsys.readouterr().err


def test_a_void_bits_row_without_the_flag_gets_no_equal(tmp_path, monkeypatch):
    import json

    import torch

    payload = {"loss": torch.tensor([2.0], dtype=torch.float64),
               "grads": {"w": torch.tensor([1.0, 2.0])}}
    digest = mm.grad_digest({"w": [1.0, 2.0]})
    monkeypatch.setattr(mm, "MOONLIGHT_GRAD_SHAPES", {"w": [2]})

    def fake_sweep(args, plan):
        tensors = mm.bit_tensor_dir(args.out)
        tensors.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in plan:
            name = mm.tensor_file_name(item["arm"], item["k"], item["seq"])
            torch.save(payload, tensors / name)
            rows.append(dict(item, status="ok", loss=2.0, grad_sha256=digest, step_seconds=1.0,
                             tensor_file=name,
                             outcome="void" if item["arm"] == "C" else "ok"))
        return rows

    monkeypatch.setattr(mm, "sweep", fake_sweep)
    out = tmp_path / "out"
    argv = ["bits", "--source", "s", "--cache", "c", "--out", str(out), "--ladder-ks", "1"]
    assert mm.main(argv) == 1
    verdicts = json.loads((out / "bits-verdict.json").read_text(encoding="utf-8"))
    assert verdicts["k1-C"] == "void", verdicts
    assert verdicts["k1-B0"] == "equal", verdicts


def test_a_void_reference_row_in_a_reference_only_bits_run_fails_without_the_flag(
        tmp_path, monkeypatch, capsys):
    import torch

    payload = {"loss": torch.tensor([2.0], dtype=torch.float64),
               "grads": {"w": torch.tensor([1.0, 2.0])}}
    monkeypatch.setattr(mm, "MOONLIGHT_GRAD_SHAPES", {"w": [2]})

    def fake_sweep(args, plan):
        tensors = mm.bit_tensor_dir(args.out)
        tensors.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in plan:
            name = mm.tensor_file_name(item["arm"], item["k"], item["seq"])
            torch.save(payload, tensors / name)
            rows.append(dict(item, status="ok", loss=2.0, grad_sha256=mm.grad_digest(
                {"w": [1.0, 2.0]}), step_seconds=1.0, tensor_file=name, outcome="void"))
        return rows

    monkeypatch.setattr(mm, "sweep", fake_sweep)
    out = tmp_path / "out"
    argv = ["bits", "--source", "s", "--cache", "c", "--out", str(out), "--ladder-ks", "1",
            "--arms", "A0"]
    assert mm.main(argv) == 1
    assert "k1-A0" in capsys.readouterr().err


def test_the_moonlight_manifest_is_the_post_arena_grad_shapes() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "mbr-moonlight-post-arena-20261125" / "analyze.py"
    spec = importlib.util.spec_from_file_location("mbr_post_arena_analyze", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert len(mm.MOONLIGHT_GRAD_SHAPES) == 216
    assert mm.MOONLIGHT_GRAD_SHAPES == {name: list(shape)
                                        for name, shape in module.GRAD_SHAPES.items()}


def test_load_grad_manifest_accepts_only_the_exact_authoritative_manifest(tmp_path) -> None:
    import json

    good = tmp_path / "good.json"
    good.write_text(json.dumps(mm.MOONLIGHT_GRAD_SHAPES), encoding="utf-8")
    assert mm.load_grad_manifest(good) == mm.MOONLIGHT_GRAD_SHAPES
    changed = dict(mm.MOONLIGHT_GRAD_SHAPES)
    changed[next(iter(changed))] = [1, 1]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(changed), encoding="utf-8")
    assert mm.load_grad_manifest(bad) is None
    assert mm.load_grad_manifest(tmp_path / "missing.json") is None


def test_digest_ladder_verdict_and_tensor_verdict_share_the_a0_rule() -> None:
    rows = [{"arm": "A0", "k": 2, "seq": 64, "status": "ok", "grad_sha256": "a", "loss": 1.0},
            {"arm": "E", "k": 2, "seq": 64, "status": "ok", "grad_sha256": "a", "loss": 1.0},
            {"arm": "C", "k": 2, "seq": 64, "status": "OOM"}]
    assert mm.ladder_verdict(rows) == {(2, "E"): "equal", (2, "C"): "OOM"}


# ---- ticket 05: H = G + the cpp_async_2slot loader, and the loader speed plan ----
def test_the_cpp_loader_is_the_h_arm_stacked_on_g() -> None:
    assert mm.arm_runtime("H") == {"expert_gemm": "lt-fused", "async_lease": True,
                                   "fast_row_gather": True, "expert_loader": "cpp_async_2slot"}
    assert mm.arm_runtime("G")["expert_loader"] == "async_lease"
    assert mm.LEVER_PREVIOUS["H"] == "G"
    assert mm.parse_arm_list("H") == ("H",)


def test_every_arm_runtime_names_its_loader() -> None:
    for arm in ("A0", "B0", "C", "E", "F", "G", "H"):
        assert mm.arm_runtime(arm)["expert_loader"] in ("async_lease", "cpp_async_2slot")


def test_the_ladder_carries_h_at_every_k_and_keeps_a0_below_it() -> None:
    assert "H" in mm.ladder_arms(8) and "A0" in mm.ladder_arms(8)
    assert "H" in mm.ladder_arms(64) and "A0" not in mm.ladder_arms(64)
    assert mm.arm_options("H", k=4) == mm.arm_options("C", k=4)


# ---- ticket 14 tail: S = C + the segmented native expert GEMM, the price of the exact path ----
def test_the_segmented_arm_keeps_c_step_options_and_switches_the_gemm() -> None:
    assert mm.SEGMENTED_ARM == "S"
    assert mm.arm_options("S", k=8) == mm.arm_options("C", k=8)
    assert mm.arm_options("S", k=64) == mm.arm_options("C", k=64)
    assert mm.arm_runtime("S") == {"expert_gemm": "segmented", "async_lease": False,
                                   "fast_row_gather": False, "expert_loader": "async_lease"}
    assert mm.parse_arm_list("S") == ("S",)


def test_the_segmented_speed_plan_is_a_b_c_c_b_a_and_drops_a0_above_k8() -> None:
    plan = mm.segmented_speed_plan(blocks=1)
    assert {(p["k"], p["seq"]) for p in plan} == set(mm.SEGMENTED_WINDOWS)
    assert [p["arm"] for p in plan if (p["k"], p["seq"]) == (8, 64)] == \
        ["A0", "C", "S", "S", "C", "A0"]
    assert [p["arm"] for p in plan if (p["k"], p["seq"]) == (64, 64)] == ["C", "S", "S", "C"]
    assert len(plan) == 16
    assert all(p["kind"] == "speed" and p["deterministic"] is False for p in plan)
    five = mm.segmented_speed_plan(blocks=5)
    assert len(five) == 80
    assert {p["block"] for p in five} == set(range(5))


# ---- ticket 16: C-S-E-E-S-C, the three candidate hands in one campaign ----
def test_the_esc_plan_is_c_s_e_e_s_c_on_the_three_fitting_windows() -> None:
    plan = mm.esc_speed_plan(blocks=1)
    assert {(p["k"], p["seq"]) for p in plan} == set(mm.ESC_WINDOWS)
    for window in mm.ESC_WINDOWS:
        assert [p["arm"] for p in plan if (p["k"], p["seq"]) == window] == \
            ["C", "S", "E", "E", "S", "C"]
    assert len(plan) == 6 * len(mm.ESC_WINDOWS)
    assert all(p["kind"] == "speed" and p["deterministic"] is False for p in plan)


def test_the_esc_plan_runs_five_blocks_of_ninety_runs_and_keeps_every_hand_centred() -> None:
    """Every hand sits at the block centre (mean position 8.5 of 18), so host drift cancels."""
    plan = mm.esc_speed_plan(blocks=5)
    assert len(plan) == 90
    assert {p["block"] for p in plan} == set(range(5))
    first_block = plan[:6 * len(mm.ESC_WINDOWS)]
    assert {p["block"] for p in first_block} == {0}
    positions: dict = {}
    for position, item in enumerate(first_block):
        positions.setdefault(item["arm"], []).append(position)
    assert {arm: sum(at) / len(at) for arm, at in positions.items()} == \
        {"C": 8.5, "S": 8.5, "E": 8.5}


def test_the_esc_plan_numbers_a_resumed_block_after_the_blocks_before_it() -> None:
    plan = mm.esc_speed_plan(blocks=1, start_block=4)
    assert {p["block"] for p in plan} == {4}
    assert [p["index"] for p in plan][:2] == [72, 73]


def test_the_esc_plan_needs_no_a0_at_k64_and_keeps_the_same_six_hands() -> None:
    """A0 is not planned above k=8; C, S and E all exist at k=64, so the palindrome is whole."""
    plan = mm.esc_speed_plan(blocks=1)
    assert "A0" not in {p["arm"] for p in plan}
    assert [p["arm"] for p in plan if (p["k"], p["seq"]) == (64, 64)] == \
        ["C", "S", "E", "E", "S", "C"]


# ---- the per-run gate of ticket 13, checked before every arm ----
def test_a_refused_gate_stops_the_sweep_before_any_run(tmp_path) -> None:
    import argparse

    args = argparse.Namespace(
        source="s", cache="c", out=str(tmp_path), steps=4, device="cpu", lora_r=16,
        lora_targets=("q_proj",), tokenizer=None, label="x", deadline_hours=0,
        gate=[__import__("sys").executable, "-c", "raise SystemExit(1)"])
    plan = [{"arm": "C", "k": 8, "seq": 64, "block": 0, "kind": "speed",
             "deterministic": False}]
    assert mm.sweep(args, plan) == []
    assert not (tmp_path / "rows.jsonl").exists()


def test_a_refused_gate_on_a_plain_sweep_exits_nonzero(tmp_path, capsys) -> None:
    import sys

    refuse = tmp_path / "refuse_gate.py"
    refuse.write_text("raise SystemExit(1)\n", encoding="utf-8")
    out = tmp_path / "speed"
    argv = ["speed", "--source", "s", "--cache", "c", "--out", str(out),
            "--gate", sys.executable, str(refuse)]
    assert mm.main(argv) == 1
    assert "gate refused run 1 of" in capsys.readouterr().out
    assert not (out / "rows.jsonl").exists()


def test_the_segmented_arm_stays_out_of_the_bits_ladders() -> None:
    """The bits of S against A0 are ticket 14's ladder; this arm is speed and memory only."""
    assert mm.SEGMENTED_ARM not in mm.ladder_arms(8)
    assert mm.SEGMENTED_ARM not in mm.LEVER_ARMS
    assert mm.SEGMENTED_ARM not in mm.arms_for(k=8)


def test_loader_speed_plan_is_g_h_h_g_over_the_three_fitting_windows() -> None:
    plan = mm.loader_speed_plan(blocks=5)
    assert {(p["k"], p["seq"]) for p in plan} == set(mm.LOADER_WINDOWS)
    per_window = [p for p in plan if (p["k"], p["seq"]) == (8, 64)]
    assert [p["arm"] for p in per_window[:4]] == ["G", "H", "H", "G"]
    assert len(plan) == 5 * len(mm.LOADER_WINDOWS) * 4
    # Block-major: one block covers every window before the next block starts.
    assert {p["block"] for p in plan[:len(mm.LOADER_WINDOWS) * 4]} == {0}
    assert {p["block"] for p in plan} == set(range(5))
    assert all(p["kind"] == "speed" and p["deterministic"] is False for p in plan)


def test_the_loader_gate_reads_the_three_fitting_windows_and_two_wins() -> None:
    rows = [{"arm": "A0", "k": k, "seq": 64, "kind": "bits", "status": "ok",
             "grad_sha256": "x", "loss": 1.0} for k in mm.BIT_LADDER]
    rows += [{"arm": "H", "k": k, "seq": 64, "kind": "bits", "status": "ok",
              "grad_sha256": "x", "loss": 1.0} for k in mm.BIT_LADDER]
    for k, seq in mm.LOADER_WINDOWS:
        gain = 0.0 if (k, seq) == (64, 64) else 0.04
        for block in range(5):
            rows += [_speed("G", k, seq, 100.0, block=block),
                     _speed("H", k, seq, 100.0 * (1 - gain), block=block)]
    gate = mm.lever_gate(rows, lever="H", previous="G", windows=mm.LOADER_WINDOWS,
                         min_windows=mm.LOADER_MIN_WINDOWS)
    assert gate["speed"]["windows_won"] == 2 and gate["speed"]["pass"] is True
    assert gate["speed"]["t_test"] == "computed"
    assert gate["bits"]["pass"] is True


def test_the_loader_gate_needs_two_of_the_three_windows() -> None:
    rows = [{"arm": "A0", "k": k, "seq": 64, "kind": "bits", "status": "ok",
             "grad_sha256": "x", "loss": 1.0} for k in mm.BIT_LADDER]
    rows += [{"arm": "H", "k": k, "seq": 64, "kind": "bits", "status": "ok",
              "grad_sha256": "x", "loss": 1.0} for k in mm.BIT_LADDER]
    for k, seq in mm.LOADER_WINDOWS:
        gain = 0.04 if (k, seq) == (8, 64) else 0.0
        for block in range(5):
            rows += [_speed("G", k, seq, 100.0, block=block),
                     _speed("H", k, seq, 100.0 * (1 - gain), block=block)]
    gate = mm.lever_gate(rows, lever="H", previous="G", windows=mm.LOADER_WINDOWS,
                         min_windows=mm.LOADER_MIN_WINDOWS)
    assert gate["speed"]["windows_won"] == 1 and gate["speed"]["pass"] is False


def test_a_window_that_clears_three_percent_but_not_the_paired_t_does_not_win() -> None:
    def bit(arm: str, k: int) -> dict:
        return {"arm": arm, "k": k, "seq": 64, "kind": "bits", "status": "ok",
                "grad_sha256": "x", "loss": 1.0}

    rows = [bit("A0", k) for k in mm.BIT_LADDER] + [bit("H", k) for k in mm.BIT_LADDER]
    # Median gain 3%, mean 5%, spread 5.57 pt: t = 2.0, below the 2.78 critical value.
    for k, seq in mm.LOADER_WINDOWS:
        for block, gain in enumerate((0.03, 0.03, 0.03, 0.0, 0.16)):
            rows += [_speed("G", k, seq, 100.0, block=block),
                     _speed("H", k, seq, 100.0 * (1 - gain), block=block)]
    gate = mm.lever_gate(rows, lever="H", previous="G", windows=mm.LOADER_WINDOWS,
                         min_windows=mm.LOADER_MIN_WINDOWS)
    assert gate["speed"]["windows"][(8, 64)]["gain"] == pytest.approx(0.03)
    assert gate["speed"]["windows"][(8, 64)]["t"] < mm.LEVER_T_CRITICAL
    assert gate["speed"]["windows_won"] == 0 and gate["speed"]["pass"] is False


def test_the_loader_gate_fails_when_h_differs_from_a0_at_k2() -> None:
    def bit(arm: str, k: int, sha: str) -> dict:
        return {"arm": arm, "k": k, "seq": 64, "kind": "bits", "status": "ok",
                "grad_sha256": sha, "loss": 1.0}

    rows = [bit("A0", k, "x") for k in mm.BIT_LADDER]
    rows += [bit("H", k, "x" if k != 2 else "y") for k in mm.BIT_LADDER]
    gate = mm.lever_gate(rows, lever="H", previous="G", windows=mm.LOADER_WINDOWS,
                         min_windows=mm.LOADER_MIN_WINDOWS)
    assert gate["bits"]["pass"] is False and gate["bits"]["differs_at"] == [2]


# ---- ticket 06: D = decoupled MBR, C's sibling, and the large-window switch ----
def test_the_decoupled_arm_is_c_sibling_with_the_subwindow_flag() -> None:
    assert mm.DECOUPLED_ARM == "D" and mm.DECOUPLED_PREVIOUS == "C"
    assert mm.arm_options("D", k=8) == {"route_preserving_mbr": True, "mbr_group_size": 8,
                                        "mbr_decoupled_subwindow": True}
    assert "mbr_ordered_backward" not in mm.arm_options("D", k=64)
    # Decoupled runs its own expert backward, so it needs the native GEMM like C.
    assert mm.arm_runtime("D") == mm.arm_runtime("C")
    assert mm.parse_arm_list("D") == ("D",)
    assert "D" in mm.ladder_arms(8) and "D" in mm.ladder_arms(64)


def test_the_large_window_switch_only_opens_when_c_does_not_fit() -> None:
    assert mm.large_window_enabled(previous_fits=True) is False
    assert mm.large_window_enabled(previous_fits=False) is True


def test_decoupled_speed_plan_is_c_d_d_c_over_the_three_fitting_windows() -> None:
    plan = mm.decoupled_speed_plan(blocks=5)
    assert {(p["k"], p["seq"]) for p in plan} == set(mm.DECOUPLED_WINDOWS)
    assert mm.DECOUPLED_WINDOWS == ((8, 64), (8, 512), (64, 64))
    per_window = [p for p in plan if (p["k"], p["seq"]) == (8, 64)]
    assert [p["arm"] for p in per_window[:4]] == ["C", "D", "D", "C"]
    assert len(plan) == 5 * len(mm.DECOUPLED_WINDOWS) * 4
    assert {p["block"] for p in plan} == set(range(5))
    assert all(p["kind"] == "speed" and p["deterministic"] is False for p in plan)


def test_sweep_child_forwards_math_attention_only_when_asked() -> None:
    args = argparse.Namespace(source="s", cache="c", steps=3, out="o", device="auto", lora_r=16,
                              lora_targets=("q_proj",), tokenizer=None, math_attention=True)
    item = {"arm": "C", "k": 2, "seq": 512, "kind": "bits", "deterministic": True}
    assert "--math-attention" in mm.child_run_args(args, item)
    args.math_attention = False
    assert "--math-attention" not in mm.child_run_args(args, item)


def test_a_segmented_expert_gemm_plan_has_no_algorithm_table() -> None:
    from soup_cli.utils.expert_gemm import SegmentState

    assert mm.gemm_plan_record(None) == {}
    state = SegmentState(linear_rows=96, mul_rows=96, passthrough_calls=2)
    assert mm.gemm_plan_record(state) == {
        "fused_visits": {}, "linear_rows": 96, "mul_rows": 96, "passthrough_calls": 2,
    }

    class Table:
        stats = {"merged_calls": 3}

    class Fused:
        direct_stats = {"fused_forward": 7}
        table = Table()

    assert mm.gemm_plan_record(Fused()) == {
        "fused_visits": {"fused_forward": 7}, "algo_table": {"merged_calls": 3},
    }


def test_sweep_child_forwards_trace_only_when_the_item_asks() -> None:
    args = argparse.Namespace(source="s", cache="c", steps=3, out="o", device="auto", lora_r=16,
                              lora_targets=("q_proj",), tokenizer=None)
    item = {"arm": "C", "k": 8, "seq": 64, "kind": "speed", "deterministic": False}
    assert "--trace" not in mm.child_run_args(args, item)
    assert "--trace" in mm.child_run_args(args, dict(item, trace=True))


def test_the_run_command_takes_trace_and_hands_it_to_run_one(monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture) -> None:
    seen = {}

    def fake_run_one(args: argparse.Namespace) -> dict:
        seen["trace"] = args.trace
        return {"trace": args.trace, "arm": args.arm, "k": args.k, "seq": args.seq}

    monkeypatch.setattr(mm, "run_one", fake_run_one)
    base = ["run", "--arm", "S", "--k", "8", "--seq", "64", "--kind", "speed",
            "--source", "s", "--cache", "c", "--out", "o"]
    assert mm.main(base + ["--trace"]) == 0
    assert seen["trace"] is True
    assert '"trace": true' in capsys.readouterr().out
    assert mm.main(base) == 0
    assert seen["trace"] is False


def test_the_run_child_forwards_route_trace_only_when_asked() -> None:
    args = argparse.Namespace(source="s", cache="c", steps=3, out="o", device="auto", lora_r=16,
                              lora_targets=("q_proj",), tokenizer=None)
    item = {"arm": "G", "k": 8, "seq": 64, "kind": "speed", "deterministic": False}
    assert "--route-trace" not in mm.child_run_args(args, item)
    args.route_trace = True
    assert "--route-trace" in mm.child_run_args(args, item)


def test_the_run_command_takes_route_trace_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_run_one(args: argparse.Namespace) -> dict:
        seen["route_trace"] = args.route_trace
        return {"arm": args.arm, "k": args.k, "seq": args.seq}

    monkeypatch.setattr(mm, "run_one", fake_run_one)
    base = ["run", "--arm", "G", "--k", "8", "--seq", "64", "--kind", "speed",
            "--source", "s", "--cache", "c", "--out", "o"]
    assert mm.main(base) == 0
    assert seen["route_trace"] is False
    assert mm.main(base + ["--route-trace"]) == 0
    assert seen["route_trace"] is True


def test_route_visits_drain_one_row_per_nonzero_expert_per_step() -> None:
    log = {(3, 5): 2, (0, 1): 4, (0, 7): 0}
    rows = mm.drain_route_visits(log, step=2)
    assert rows == [[2, 0, 1, 4], [2, 3, 5, 2]]
    assert log == {}


def test_a_row_without_route_trace_has_no_route_visits_field() -> None:
    row = mm.make_row(arm="G", k=8, seq=64, kind="speed", deterministic=False,
                      result={"fused_visits": {}}, dataset_sha256="d")
    assert "route_visits" not in row


def test_the_run_child_forwards_count_only_when_asked() -> None:
    args = argparse.Namespace(source="s", cache="c", steps=3, out="o", device="auto", lora_r=16,
                              lora_targets=("q_proj",), tokenizer=None)
    item = {"arm": "C", "k": 8, "seq": 64, "kind": "speed", "deterministic": False}
    assert "--count" not in mm.child_run_args(args, item)
    args.count = True
    assert "--count" in mm.child_run_args(args, item)


def test_the_run_command_takes_count_and_hands_it_to_run_one(monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture) -> None:
    seen = {}

    def fake_run_one(args: argparse.Namespace) -> dict:
        seen["count"] = args.count
        return {"count": args.count}

    monkeypatch.setattr(mm, "run_one", fake_run_one)
    base = ["run", "--arm", "C", "--k", "8", "--seq", "64", "--kind", "speed",
            "--source", "s", "--cache", "c", "--out", "o"]
    assert mm.main(base) == 0
    assert seen["count"] is False
    assert mm.main(base + ["--count"]) == 0
    assert seen["count"] is True
    assert '"count": true' in capsys.readouterr().out


def test_the_run_command_host_cache_defaults_to_zero_and_hands_it_to_run_one(
        monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_run_one(args: argparse.Namespace) -> dict:
        seen["host_cache_bytes"] = args.host_cache_bytes
        return {}

    monkeypatch.setattr(mm, "run_one", fake_run_one)
    base = ["run", "--arm", "G", "--k", "8", "--seq", "64", "--kind", "speed",
            "--source", "s", "--cache", "c", "--out", "o"]
    assert mm.main(base) == 0
    assert seen["host_cache_bytes"] == 0
    assert mm.main(base + ["--host-cache-bytes", "1048576"]) == 0
    assert seen["host_cache_bytes"] == 1048576


def test_host_cache_kwargs_reach_build_streamed_model_only_when_the_budget_is_positive() -> None:
    assert mm.host_cache_kwargs(0) == {}
    assert mm.host_cache_kwargs(1048576) == {"expert_host_cache_bytes": 1048576}


def test_a_negative_or_fractional_host_cache_is_refused_at_parse_time(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(mm, "run_one", lambda args: pytest.fail("run_one must not start"))
    base = ["run", "--arm", "G", "--k", "8", "--seq", "64", "--kind", "speed",
            "--source", "s", "--cache", "c", "--out", "o"]
    for bad in ("-1", "1.5", "abc"):
        with pytest.raises(SystemExit) as refused:
            mm.main(base + ["--host-cache-bytes", bad])
        assert refused.value.code == 2
        assert "non-negative integer" in capsys.readouterr().err


def test_host_cache_stats_reach_the_row_only_when_the_cache_is_on() -> None:
    class FakePool:
        def stats(self) -> dict:
            return {"fetches": 9, "host_cache": {"budget_bytes": 1048576, "payload_bytes": 4096,
                                                  "entries": 2, "hits": 5, "misses": 3,
                                                  "rejected": 1}}

    class FakeRuntime:
        expert_pool = FakePool()

    assert mm.host_cache_record(0, FakeRuntime()) == {}
    assert mm.host_cache_record(1048576, FakeRuntime()) == {"host_cache_stats": {
        "budget_bytes": 1048576, "payload_bytes": 4096, "entries": 2, "hits": 5, "misses": 3,
        "rejected": 1}}


def test_lever_bits_passes_host_cache_bytes_to_every_child_run(tmp_path, monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(mm, "sweep", lambda args, plan: seen.update(args=args, plan=list(plan))
                        or [])
    base = ["lever-bits", "--source", "s", "--cache", "c", "--out", str(tmp_path),
            "--ladder-ks", "1"]
    mm.main(base + ["--host-cache-bytes", "1048576"])
    assert seen["plan"], "the sweep must get a plan"
    for item in seen["plan"]:
        argv = mm.child_run_args(seen["args"], item)
        assert argv[argv.index("--host-cache-bytes") + 1] == "1048576"


def test_lever_bits_without_the_flag_keeps_the_plan_and_the_child_argv(tmp_path,
                                                                      monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(mm, "sweep", lambda args, plan: seen.update(args=args, plan=list(plan))
                        or [])
    mm.main(["lever-bits", "--source", "s", "--cache", "c", "--out", str(tmp_path),
             "--ladder-ks", "1"])
    assert seen["plan"] == mm.lever_bit_plan(seq=64, ks=(1,))
    argv = mm.child_run_args(seen["args"], seen["plan"][0])
    assert argv == ["run", "--arm", "A0", "--k", "1", "--seq", "64", "--kind", "bits",
                    "--source", "s", "--cache", "c", "--steps", "4", "--out", str(tmp_path),
                    "--device", "auto", "--lora-r", "16",
                    "--lora-targets", "q_proj,kv_b_proj,o_proj", "--deterministic"]


def test_lever_bits_refuses_a_negative_host_cache_at_parse_time(
        tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(mm, "sweep", lambda args, plan: pytest.fail("sweep must not start"))
    base = ["lever-bits", "--source", "s", "--cache", "c", "--out", str(tmp_path)]
    for bad in ("-1", "1.5", "abc"):
        with pytest.raises(SystemExit) as refused:
            mm.main(base + ["--host-cache-bytes", bad])
        assert refused.value.code == 2
        assert "non-negative integer" in capsys.readouterr().err


# ---- ticket 17: G against E and C after the arena, and the D+segmented diagnostic cell ----
def test_the_ceg_speed_plan_is_c_e_g_g_e_c_on_the_two_k8_windows() -> None:
    assert mm.CEG_WINDOWS == ((8, 64), (8, 512))
    plan = mm.ceg_speed_plan(blocks=1)
    for window in mm.CEG_WINDOWS:
        assert [p["arm"] for p in plan if (p["k"], p["seq"]) == window] == \
            ["C", "E", "G", "G", "E", "C"]
    assert len(plan) == 6 * len(mm.CEG_WINDOWS)
    assert all(p["kind"] == "speed" and p["deterministic"] is False for p in plan)


def test_the_ceg_plan_runs_five_blocks_and_keeps_every_hand_centred() -> None:
    """Every hand sits at the block centre (mean position 5.5 of 12), so host drift cancels."""
    plan = mm.ceg_speed_plan(blocks=5)
    assert len(plan) == 60
    assert {p["block"] for p in plan} == set(range(5))
    first_block = plan[:6 * len(mm.CEG_WINDOWS)]
    positions: dict = {}
    for position, item in enumerate(first_block):
        positions.setdefault(item["arm"], []).append(position)
    assert {arm: sum(at) / len(at) for arm, at in positions.items()} == \
        {"C": 5.5, "E": 5.5, "G": 5.5}


def test_the_ceg_speed_plan_takes_a_windows_keyword_and_keeps_the_default() -> None:
    plan = mm.ceg_speed_plan(blocks=5, windows=((64, 64),))
    assert len(plan) == 30
    assert {p["block"] for p in plan} == set(range(5))
    assert {(p["k"], p["seq"]) for p in plan} == {(64, 64)}
    for block in range(5):
        arms = [p["arm"] for p in plan if p["block"] == block]
        assert arms == ["C", "E", "G", "G", "E", "C"]
    default = mm.ceg_speed_plan(blocks=5)
    assert len(default) == 60
    assert {(p["k"], p["seq"]) for p in default} == {(8, 64), (8, 512)}


def test_the_decoupled_segmented_arm_keeps_d_step_options_and_switches_the_gemm() -> None:
    assert mm.DECOUPLED_SEGMENTED_ARM == "DS"
    assert mm.arm_options("DS", k=2) == mm.arm_options("D", k=2)
    assert mm.arm_runtime("DS") == {"expert_gemm": "segmented", "async_lease": False,
                                    "fast_row_gather": False, "expert_loader": "async_lease"}
    assert mm.parse_arm_list("DS") == ("DS",)
    assert "DS" not in mm.ladder_arms(2)


def test_the_ds_bit_plan_is_a0_against_decoupled_segmented_in_deterministic_mode() -> None:
    plan = mm.decoupled_segmented_bit_plan(seq=64, ks=(2,))
    assert [p["arm"] for p in plan] == ["A0", "DS"]
    assert all(p["k"] == 2 and p["seq"] == 64 and p["kind"] == "bits"
               and p["deterministic"] for p in plan)
    assert mm.decoupled_segmented_bit_plan(seq=64) == \
        mm.decoupled_segmented_bit_plan(seq=64, ks=(2,))


def test_the_ceg_and_ds_commands_reach_their_plans(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path) -> None:
    seen = {}

    def fake_sweep(args: argparse.Namespace, plan: list) -> list:
        seen["command"] = args.command
        seen["arms"] = [item["arm"] for item in plan]
        seen["kind"] = {item["kind"] for item in plan}
        return []

    monkeypatch.setattr(mm, "sweep", fake_sweep)
    base = ["--source", "s", "--cache", "c", "--out", str(tmp_path)]
    assert mm.main(["ceg-speed", "--blocks", "1", *base]) == 0
    assert seen["command"] == "ceg-speed"
    assert seen["arms"][:6] == ["C", "E", "G", "G", "E", "C"]
    assert seen["kind"] == {"speed"}
    assert mm.main(["ds-bits", *base]) == 1
    assert seen["command"] == "ds-bits"
    assert seen["arms"] == ["A0", "DS"]
    assert seen["kind"] == {"bits"}
    assert mm.main(["ds-bits", "--arms", "DS", *base]) == 1
    assert seen["arms"] == ["DS"]


# ---- arm GO: G's lever settings, C's step options, and the routing Oracle of arm O ----
def test_go_takes_the_lever_settings_of_g_exactly() -> None:
    assert mm.arm_runtime("GO") == mm.arm_runtime("G")
    assert mm.arm_runtime("GO") == {"expert_gemm": "lt-fused", "async_lease": True,
                                    "fast_row_gather": True, "expert_loader": "async_lease"}


def test_go_has_c_step_options_and_the_oracle_marker_at_every_k() -> None:
    for k in (8, 64):
        assert mm.arm_options("GO", k=k) == {**mm.arm_options("C", k=k),
                                             mm.ROUTING_ORACLE_KEY: True}
    assert mm.arm_options("GO", k=8) == {"route_preserving_mbr": True, "mbr_group_size": 8,
                                         "mbr_ordered_backward": True, "routing_oracle": True}


def test_g_stays_without_the_oracle_marker_and_makes_no_oracle() -> None:
    assert mm.ROUTING_ORACLE_KEY not in mm.arm_options("G", k=8)
    assert mm.arm_options("G", k=8) == mm.arm_options("C", k=8)
    made = []
    assert mm.make_routing_oracle(mm.arm_options("G", k=8), lambda: made.append(1)) is None
    assert made == []


def test_o_keeps_its_step_options_marker_and_native_levers() -> None:
    assert mm.arm_options("O", k=8) == {"route_preserving_mbr": True, "mbr_group_size": 8,
                                        "mbr_ordered_backward": True, "routing_oracle": True}
    assert mm.arm_runtime("O") == mm.arm_runtime("C")


def test_parse_arm_list_accepts_go() -> None:
    assert mm.parse_arm_list("GO") == ("GO",)
    assert mm.parse_arm_list("G, GO") == ("G", "GO")


def test_go_is_in_no_lever_table_ladder_or_plan() -> None:
    assert "GO" not in mm.LEVER_ARMS and "GO" not in mm.LEVER_PREVIOUS
    assert "GO" not in mm.ladder_arms(8) and "GO" not in mm.ladder_arms(64)
    assert "GO" not in mm.arms_for(k=8)
    assert "GO" not in {item["arm"] for item in mm.lever_bit_plan(seq=64)}


def test_go_makes_its_oracle_with_the_zeroed_aux_config_and_c_step_kwargs() -> None:
    options = mm.arm_options("GO", k=8)
    made = []

    def make() -> str:
        made.append(True)
        return "oracle"

    oracle = mm.make_routing_oracle(options, make)
    assert oracle == "oracle" and made == [True]
    config = SimpleNamespace(aux_loss_alpha=0.01, router_aux_loss_coef=0.5)
    safe = mm.oracle_config_for_arm(options, config)
    assert safe is not config and safe.aux_loss_alpha == 0 and safe.router_aux_loss_coef is None
    assert config.aux_loss_alpha == 0.01
    step = mm.routing_step_options(options, oracle)
    assert step == {**mm.arm_options("C", k=8), mm.ROUTING_ORACLE_KEY: "oracle"}


def test_traj_plan_is_a0_then_g_for_each_step_at_fixed_k_and_seq() -> None:
    plan = mm.traj_plan(steps=3, k=2, seq=64)
    assert len(plan) == 2 * 3
    assert [(p["arm"], p["step"]) for p in plan] == [
        ("A0", 0), ("A0", 1), ("A0", 2), ("G", 0), ("G", 1), ("G", 2)]
    assert all(p["k"] == 2 and p["seq"] == 64 and p["kind"] == "traj"
               and p["deterministic"] for p in plan)


DATASET = "ab" * 32
IDENTITY = mm.campaign_identity(source="s", cache="c", tokenizer=None, lora_r=16,
                                lora_targets=mm.DEFAULT_LORA_TARGETS, steps=3)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _record(step: int) -> dict:
    loss = float(step) + 0.5
    return {"step": step, "loss": loss, "loss_hex": struct.pack("<d", loss).hex(),
            "weight_sha256": _digest(f"w{step}")}


def _traj_rows(steps: int = 5, *, identity: str = "id", dataset: str = DATASET) -> list:
    return [{**_record(step), "arm": arm, "k": 2, "seq": 64, "status": "ok", "outcome": "ok",
             "campaign_id": identity, "dataset_sha256": dataset}
            for arm in ("A0", "G") for step in range(steps)]


def test_traj_plan_maps_a_nonstandard_window_step_by_step() -> None:
    plan = mm.traj_plan(steps=5, k=3, seq=32)
    assert [(p["arm"], p["step"], p["k"], p["seq"]) for p in plan] == (
        [("A0", step, 3, 32) for step in range(5)] + [("G", step, 3, 32) for step in range(5)])


def test_traj_verdict_equal_when_every_step_matches() -> None:
    report = mm.traj_verdict(_traj_rows())
    assert report["verdict"] == "equal"
    assert report["first_difference"] is None
    assert report["steps"] == {str(step): "equal" for step in range(5)}


def test_traj_verdict_reports_the_first_differing_step() -> None:
    rows = _traj_rows()
    rows[-2]["weight_sha256"] = _digest("other")  # G at step 3
    report = mm.traj_verdict(rows)
    assert report["first_difference"] == 3
    assert report["steps"] == {"0": "equal", "1": "equal", "2": "equal", "3": "differs",
                               "4": "equal"}


def test_traj_verdict_names_the_first_step_for_a_loss_only_change_and_keeps_later_ones() -> None:
    rows = _traj_rows()
    loss_only = next(r for r in rows if r["arm"] == "G" and r["step"] == 1)
    loss_only.update(loss=9.0, loss_hex=struct.pack("<d", 9.0).hex())
    weights_only = next(r for r in rows if r["arm"] == "G" and r["step"] == 3)
    weights_only["weight_sha256"] = _digest("other")
    report = mm.traj_verdict(rows)
    assert report["first_difference"] == 1
    assert report["steps"] == {"0": "equal", "1": "differs", "2": "equal", "3": "differs",
                               "4": "equal"}


def test_traj_verdict_needs_campaign_id_on_every_row() -> None:
    rows = _traj_rows()
    del rows[0]["campaign_id"]
    report = mm.traj_verdict(rows)
    assert report["verdict"].startswith("no verdict")
    assert report["first_difference"] is None


def test_traj_verdict_refuses_rows_from_two_campaigns() -> None:
    rows = _traj_rows()
    rows[-1]["campaign_id"] = "other"
    assert mm.traj_verdict(rows)["verdict"].startswith("no verdict")


def test_traj_verdict_refuses_rows_with_two_dataset_hashes() -> None:
    rows = _traj_rows()
    rows[0]["dataset_sha256"] = _digest("other")
    assert mm.traj_verdict(rows)["verdict"].startswith("no verdict")


def test_traj_verdict_refuses_a_conflicting_duplicate_even_before_a_matching_row() -> None:
    rows = _traj_rows()
    rows.insert(0, {**rows[0], "weight_sha256": _digest("other")})
    report = mm.traj_verdict(rows)
    assert report["verdict"].startswith("no verdict")
    assert report["first_difference"] is None


def test_traj_verdict_refuses_a_row_that_is_not_ok() -> None:
    rows = _traj_rows()
    rows[0]["status"] = "OOM"
    assert mm.traj_verdict(rows)["verdict"].startswith("no verdict")


def test_traj_verdict_refuses_a_nan_loss_even_when_both_arms_agree() -> None:
    rows = _traj_rows()
    for row in rows:
        row.update(loss=float("nan"), loss_hex=struct.pack("<d", float("nan")).hex())
    report = mm.traj_verdict(rows)
    assert report["verdict"].startswith("no verdict")
    assert report["steps"] == {}


def test_traj_verdict_refuses_malformed_digests() -> None:
    rows = _traj_rows()
    for row in rows:
        if row["step"] == 0:
            row["weight_sha256"] = "w0"
    assert mm.traj_verdict(rows)["verdict"].startswith("no verdict")


def test_traj_verdict_needs_exactly_the_planned_rows() -> None:
    plan = mm.traj_plan(steps=3, k=2, seq=64)
    assert mm.traj_verdict(_traj_rows(steps=3), plan)["verdict"] == "equal"
    assert mm.traj_verdict(_traj_rows(steps=3)[:-1], plan)["verdict"].startswith("no verdict")


def test_traj_record_refuses_a_non_finite_loss() -> None:
    with pytest.raises(ValueError, match="non-finite loss"):
        mm.traj_record(0, [float("nan")], {})


def test_traj_record_refuses_a_non_finite_weight() -> None:
    torch = pytest.importorskip("torch")
    params = {"w": torch.tensor([1.0, float("nan")])}
    with pytest.raises(ValueError, match="non-finite weight"):
        mm.traj_record(0, [0.5], params)


def test_trajectory_step_records_the_weights_after_one_cpu_adamw_update() -> None:
    torch = pytest.importorskip("torch")
    weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    weight.grad = torch.tensor([1.0], dtype=torch.float64)
    optimizer = torch.optim.AdamW([weight], lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)
    before = mm.traj_record(0, [0.5], {"w": weight})["weight_sha256"]
    record = mm.trajectory_step(optimizer, 0, [0.5], {"w": weight})
    assert record["step"] == 0 and record["loss"] == 0.5
    assert weight.item() == pytest.approx(1.0 - 1e-4, abs=1e-9)
    assert record["weight_sha256"] != before


def test_traj_runs_under_strict_determinism_and_other_kinds_keep_warn_only() -> None:
    assert mm.determinism_warn_only("traj") is False
    assert mm.determinism_warn_only("bits") is True


@pytest.mark.parametrize("flags", [["--steps", "1"], ["--steps", "0"], ["--k", "9"],
                                   ["--k", "0"], ["--seq", "0"]])
def test_traj_refuses_dimensions_it_cannot_measure(monkeypatch, tmp_path, flags) -> None:
    launched = []
    monkeypatch.setattr(mm, "traj_sweep", lambda args, plan: launched.append(plan) or [])
    with pytest.raises(SystemExit):
        mm.main(_traj_argv(tmp_path, *flags))
    assert launched == []


def test_traj_accepts_two_steps_and_the_a0_limit_of_k(monkeypatch, tmp_path) -> None:
    launched = []
    monkeypatch.setattr(mm, "traj_sweep", lambda args, plan: launched.append(plan) or [])
    assert mm.main(_traj_argv(tmp_path, "--steps", "2", "--k", "8")) == 1  # no rows: no verdict
    assert len(launched) == 1


def test_traj_takes_the_math_attention_flag_into_every_child(monkeypatch, tmp_path) -> None:
    seen = {}
    monkeypatch.setattr(mm, "traj_sweep",
                        lambda args, plan: seen.update(math=args.math_attention) or [])
    mm.main(_traj_argv(tmp_path, "--math-attention"))
    assert seen["math"] is True
    args = argparse.Namespace(source="s", cache="c", tokenizer=None, lora_r=16, steps=2,
                              lora_targets=mm.DEFAULT_LORA_TARGETS, out=str(tmp_path),
                              device="auto", math_attention=True, count=False)
    assert "--math-attention" in mm.child_run_args(args, mm.traj_plan(steps=2, k=2, seq=64)[0])


# Sweep of ``traj`` with stub children: ``subprocess.run`` answers each child, ``ArmWatch`` is
# replaced by a record with the outcome.
def _traj_argv(out, *extra) -> list:
    return ["traj", "--source", "s", "--cache", "c", "--out", str(out), "--steps", "3",
            "--k", "2", "--seq", "64", *extra]


@pytest.fixture
def stub_watch(monkeypatch):
    class FakeWatch:
        def __init__(self, **kwargs) -> None:
            self.record = {"outcome": "ok", "arm": kwargs["arm"]}

        def __enter__(self) -> "FakeWatch":
            return self

        def __exit__(self, *exc) -> bool:
            return False

    monkeypatch.setattr("benchmarks.harness.arm_validity.ArmWatch", FakeWatch)


def _stub_children(monkeypatch, reply, *, gate_code: int = 0) -> list:
    """``reply(command)`` gives (exit code, stdout, stderr) of each child; the gate command
    returns ``gate_code``. Returns the arms of the children that were launched."""
    launched = []

    def fake_run(command, capture_output=True, text=True, **_):
        if command[0] != sys.executable:
            return subprocess.CompletedProcess(command, gate_code,
                                               "", "AC off" if gate_code else "")
        launched.append(command[command.index("--arm") + 1])
        code, out, err = reply(command)
        return subprocess.CompletedProcess(command, code, out, err)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return launched


def _child(command, *, trajectory: list, status: str = "ok", campaign: str = IDENTITY,
           error: str = "", drop: tuple = (), **fields) -> str:
    row = {"arm": command[command.index("--arm") + 1], "k": 2, "seq": 64, "kind": "traj",
           "mode": "deterministic", "status": status, "campaign_id": campaign,
           "dataset_sha256": DATASET, "trajectory": trajectory, **fields}
    if error:
        row["error"] = error
    for key in drop:
        row.pop(key)
    return json.dumps(row) + "\n"


def _good_trajectory(steps: int = 3) -> list:
    return [_record(step) for step in range(steps)]


def _read_rows(folder) -> list:
    path = folder / "rows.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_traj_sweep_with_matching_stub_children_says_equal(monkeypatch, tmp_path,
                                                           stub_watch) -> None:
    _stub_children(monkeypatch, lambda cmd: (0, _child(cmd, trajectory=_good_trajectory()), ""))
    assert mm.main(_traj_argv(tmp_path)) == 0
    assert json.loads((tmp_path / "traj-verdict.json").read_text())["verdict"] == "equal"
    assert len(_read_rows(tmp_path)) == 6


def test_traj_sweep_does_not_trust_a_child_that_exits_nonzero(monkeypatch, tmp_path,
                                                              stub_watch) -> None:
    _stub_children(monkeypatch,
                   lambda cmd: (1, _child(cmd, trajectory=_good_trajectory()), "boom"))
    assert mm.main(_traj_argv(tmp_path)) == 1
    assert json.loads((tmp_path / "traj-verdict.json").read_text())["verdict"].startswith(
        "no verdict")
    assert {r["status"] for r in _read_rows(tmp_path)} == {"CRASH"}


def test_traj_sweep_refuses_a_child_without_campaign_id(monkeypatch, tmp_path,
                                                        stub_watch) -> None:
    _stub_children(monkeypatch, lambda cmd: (0, _child(cmd, trajectory=_good_trajectory(),
                                                      drop=("campaign_id",)), ""))
    assert mm.main(_traj_argv(tmp_path)) == 1
    assert {r["status"] for r in _read_rows(tmp_path)} == {"CRASH"}


def test_traj_sweep_refuses_a_child_from_another_campaign(monkeypatch, tmp_path,
                                                          stub_watch) -> None:
    _stub_children(monkeypatch, lambda cmd: (0, _child(cmd, trajectory=_good_trajectory(),
                                                      campaign="other"), ""))
    assert mm.main(_traj_argv(tmp_path)) == 1
    assert {r["status"] for r in _read_rows(tmp_path)} == {"CRASH"}


def test_traj_sweep_keeps_the_oom_status_of_a_child(monkeypatch, tmp_path, stub_watch) -> None:
    _stub_children(monkeypatch, lambda cmd: (0, _child(cmd, trajectory=[], status="OOM",
                                                      error="CUDA out of memory"), ""))
    assert mm.main(_traj_argv(tmp_path)) == 1
    rows = _read_rows(tmp_path)
    assert {r["status"] for r in rows} == {"OOM"}
    assert all("out of memory" in r["error"] for r in rows)


def test_traj_sweep_maps_steps_by_their_step_field_not_their_position(
        monkeypatch, tmp_path, stub_watch) -> None:
    def reply(cmd):
        arm_a0 = cmd[cmd.index("--arm") + 1] == "A0"
        trajectory = list(reversed(_good_trajectory())) if arm_a0 else _good_trajectory()
        return 0, _child(cmd, trajectory=trajectory), ""

    _stub_children(monkeypatch, reply)
    assert mm.main(_traj_argv(tmp_path)) == 0
    assert json.loads((tmp_path / "traj-verdict.json").read_text())["verdict"] == "equal"


def test_traj_sweep_refuses_a_trajectory_with_a_repeated_step(monkeypatch, tmp_path,
                                                              stub_watch) -> None:
    repeated = [_record(0), _record(0), _record(2)]
    _stub_children(monkeypatch, lambda cmd: (0, _child(cmd, trajectory=repeated), ""))
    assert mm.main(_traj_argv(tmp_path)) == 1


def test_traj_refuses_every_arm_when_the_gate_fails(monkeypatch, tmp_path, stub_watch) -> None:
    launched = _stub_children(
        monkeypatch, lambda cmd: (0, _child(cmd, trajectory=_good_trajectory()), ""),
        gate_code=1)
    assert mm.main(_traj_argv(tmp_path, "--gate", "check")) == 1
    assert launched == []
    assert _read_rows(tmp_path) == []


def test_traj_stops_before_the_arm_that_starts_past_the_deadline(monkeypatch, tmp_path,
                                                                 stub_watch) -> None:
    ticks = [0.0, 0.0]  # the deadline is set from the first read, A0 starts on the second
    monkeypatch.setattr(time, "time", lambda: ticks.pop(0) if ticks else 1e9)
    launched = _stub_children(
        monkeypatch, lambda cmd: (0, _child(cmd, trajectory=_good_trajectory()), ""))
    assert mm.main(_traj_argv(tmp_path, "--deadline-hours", "1")) == 1
    assert launched == ["A0"]
    rows = _read_rows(tmp_path)
    assert len(rows) == 3 and {r["arm"] for r in rows} == {"A0"}


@pytest.mark.parametrize("field, wrong, crash", [("arm", "G", True), ("k", 4, True),
                                                 ("dataset_sha256", "cd" * 32, False)])
def test_traj_sweep_refuses_a_child_whose_arm_k_or_dataset_differs_from_the_plan(
        monkeypatch, tmp_path, stub_watch, field, wrong, crash) -> None:
    def reply(cmd):
        off_plan = cmd[cmd.index("--arm") + 1] == "A0"
        extra = {field: wrong} if off_plan else {}
        return 0, _child(cmd, trajectory=_good_trajectory(), **extra), ""

    _stub_children(monkeypatch, reply)
    assert mm.main(_traj_argv(tmp_path)) == 1
    verdict = json.loads((tmp_path / "traj-verdict.json").read_text())["verdict"]
    assert verdict.startswith("no verdict") and verdict != "equal"
    if crash:
        assert "CRASH" in {r["status"] for r in _read_rows(tmp_path)}


def test_traj_sweep_gives_crash_rows_when_a_child_returns_fewer_steps_than_asked(
        monkeypatch, tmp_path, stub_watch) -> None:
    _stub_children(monkeypatch,
                   lambda cmd: (0, _child(cmd, trajectory=_good_trajectory(steps=2)), ""))
    assert mm.main(_traj_argv(tmp_path, "--steps", "3")) == 1
    rows = _read_rows(tmp_path)
    assert len(rows) == 6 and {r["status"] for r in rows} == {"CRASH"}
    verdict = json.loads((tmp_path / "traj-verdict.json").read_text())["verdict"]
    assert verdict.startswith("no verdict")


def test_run_refuses_traj_dimensions_it_cannot_measure_before_running(
        monkeypatch, tmp_path) -> None:
    called: list = []
    monkeypatch.setattr(mm, "run_one", lambda args: called.append(args) or {})
    argv = ["run", "--arm", "A0", "--k", "9", "--seq", "64", "--kind", "traj", "--steps", "1",
            "--source", "s", "--cache", "c", "--out", str(tmp_path)]
    with pytest.raises(SystemExit) as refused:
        mm.main(argv)
    assert refused.value.code == 2
    assert called == []


class _FakeOracle:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_arm_o_is_the_c_options_plus_the_routing_oracle_marker() -> None:
    c_options = mm.arm_options("C", k=8)
    assert mm.arm_options("O", k=8) == {"route_preserving_mbr": True, "mbr_group_size": 8,
                                        "mbr_ordered_backward": True, "routing_oracle": True}
    assert mm.arm_options("O", k=8) == {**c_options, "routing_oracle": True}
    assert mm.arm_options("O", k=64) == {"route_preserving_mbr": True, "mbr_group_size": 64,
                                         "mbr_ordered_backward": True, "routing_oracle": True}


def test_arm_o_runs_with_the_runtime_levers_of_c() -> None:
    assert mm.arm_runtime("O") == mm.arm_runtime("C")
    assert "O" not in mm.LEVER_ARMS


def test_arms_for_and_the_ladder_do_not_name_arm_o() -> None:
    assert mm.arms_for(k=8) == ("A0", "B0", "C")
    assert mm.arms_for(k=64) == ("B0", "C")
    assert mm.ladder_arms(8) == ("A0", "C", "E", "F", "G", "H", "D")
    assert mm.ladder_arms(64) == ("C", "E", "F", "G", "H", "D")


def test_no_existing_plan_names_arm_o() -> None:
    plans = [
        mm.speed_plan(), mm.profile_plan(), mm.bit_plan(seq=64), mm.lever_bit_plan(seq=64),
        mm.lever_speed_plan(), mm.traj_plan(), mm.loader_speed_plan(), mm.decoupled_speed_plan(),
        mm.segmented_speed_plan(), mm.esc_speed_plan(), mm.ceg_speed_plan(),
        mm.decoupled_segmented_bit_plan(),
    ]
    arms = {row["arm"] for plan in plans for row in plan}
    assert "O" not in arms and {"A0", "B0", "C"} <= arms


def test_the_cli_accepts_arm_o_next_to_c() -> None:
    assert mm.parse_arm_list("C,O") == ("C", "O")


def test_arm_o_creates_one_oracle_and_its_step_options_carry_it() -> None:
    made: list = []

    def make() -> _FakeOracle:
        made.append(_FakeOracle())
        return made[-1]

    options = mm.arm_options("O", k=8)
    oracle = mm.make_routing_oracle(options, make)
    assert oracle is made[0] and len(made) == 1
    step = mm.routing_step_options(options, oracle)
    assert step["routing_oracle"] is oracle
    assert "routing_oracle" in step and step["mbr_ordered_backward"] is True
    assert step == {"route_preserving_mbr": True, "mbr_group_size": 8,
                    "mbr_ordered_backward": True, "routing_oracle": oracle}
    mm.close_routing_oracle(oracle)
    assert oracle.closed == 1


@pytest.mark.parametrize("arm", ["A0", "B0", "C", "S", "D", "DS", "E", "H"])
def test_other_arms_never_create_an_oracle_and_pass_no_oracle(arm: str) -> None:
    made: list = []

    def make() -> _FakeOracle:
        made.append(_FakeOracle())
        return made[-1]

    options = mm.arm_options(arm, k=8)
    assert mm.make_routing_oracle(options, make) is None and made == []
    assert mm.routing_step_options(options, None) == options
    assert "routing_oracle" not in mm.routing_step_options(options, None)
    mm.close_routing_oracle(None)


def test_oracle_safe_config_zeroes_aux_loss_alpha_and_router_coef() -> None:
    config = SimpleNamespace(aux_loss_alpha=0.001, router_aux_loss_coef=0.01,
                             model_type="deepseek_v3")
    safe = mm.oracle_safe_config(config)
    assert safe.aux_loss_alpha == 0
    assert safe.router_aux_loss_coef is None
    assert safe.model_type == "deepseek_v3"


def test_oracle_safe_config_does_not_mutate_its_input() -> None:
    config = SimpleNamespace(aux_loss_alpha=0.001, router_aux_loss_coef=0.01)
    safe = mm.oracle_safe_config(config)
    assert safe is not config
    assert config.aux_loss_alpha == 0.001 and config.router_aux_loss_coef == 0.01


def test_oracle_safe_config_does_not_add_absent_fields() -> None:
    config = SimpleNamespace(aux_loss_alpha=0.001)
    safe = mm.oracle_safe_config(config)
    assert safe.aux_loss_alpha == 0
    assert not hasattr(safe, "router_aux_loss_coef")


def test_arm_o_gets_the_oracle_safe_config() -> None:
    config = SimpleNamespace(aux_loss_alpha=0.001, router_aux_loss_coef=None)
    out = mm.oracle_config_for_arm(mm.arm_options("O", k=8), config)
    assert out is not config
    assert out.aux_loss_alpha == 0
    assert config.aux_loss_alpha == 0.001


@pytest.mark.parametrize("arm", ["A0", "C", "B0", "S", "D", "DS", "E", "H"])
def test_other_arms_keep_the_original_config_object(arm: str) -> None:
    config = SimpleNamespace(aux_loss_alpha=0.001, router_aux_loss_coef=0.01)
    out = mm.oracle_config_for_arm(mm.arm_options(arm, k=8), config)
    assert out is config
    assert out.aux_loss_alpha == 0.001 and out.router_aux_loss_coef == 0.01


def _trace_event(name: str, device: str, start: float, end: float, self_us: float = 0.0,
                 annotation: bool = False):
    return SimpleNamespace(
        name=name, device_type=f"DeviceType.{device}",
        time_range=SimpleNamespace(start=start, end=end), self_cpu_time_total=self_us,
        is_user_annotation=annotation)


def test_cpu_self_time_is_summed_by_name_inside_the_step_window() -> None:
    events = [
        _trace_event(mm.STEP_RANGE, "CPU", 0, 100),
        _trace_event("aten::mm", "CPU", 0, 20, 20),
        _trace_event("aten::add", "CPU", 20, 30, 10),
        _trace_event("aten::mm", "CPU", 30, 50, 15),
        _trace_event("aten::copy_", "CPU", 200, 300, 100),
        _trace_event("aten::mm", "CUDA", 5, 18),
    ]
    out = mm.cpu_gpu_summary(events)
    assert out["cpu_self_top"] == [
        {"name": "aten::mm", "calls": 2, "self_seconds": pytest.approx(35e-6)},
        {"name": "aten::add", "calls": 1, "self_seconds": pytest.approx(10e-6)},
    ]


def test_a_gpu_gap_is_attributed_to_the_cpu_op_running_before_it() -> None:
    events = [
        _trace_event(mm.STEP_RANGE, "CPU", 0, 100),
        _trace_event("aten::mm", "CPU", 0, 20, 20),
        _trace_event("Python: forward", "CPU", 50, 90, 5),
        _trace_event("aten::mm", "CUDA", 5, 18),
        _trace_event("aten::mm", "CUDA", 40, 60),
        _trace_event("aten::mm", "CUDA", 62, 70),
    ]
    out = mm.cpu_gpu_summary(events)
    assert out["gaps"] == {"count": 2, "seconds": pytest.approx(24e-6)}
    assert out["gpu_busy_seconds"] == pytest.approx(41e-6)
    assert out["gap_top"] == [
        {"op": "aten::mm", "gaps": 1, "gap_seconds": pytest.approx(22e-6)},
        {"op": "Python: forward", "gaps": 1, "gap_seconds": pytest.approx(2e-6)},
    ]


def test_cuda_user_annotations_and_host_runtime_calls_are_not_gpu_busy() -> None:
    events = [
        _trace_event(mm.STEP_RANGE, "CPU", 0, 100),
        _trace_event(mm.STEP_RANGE, "CUDA", 0, 100, annotation=True),
        _trace_event("cudaLaunchKernel", "CPU", 5, 40),
        _trace_event("aten::mm", "CUDA", 10, 30),
        _trace_event("aten::mm", "CUDA", 50, 60),
    ]
    out = mm.cpu_gpu_summary(events)
    assert out["gpu_busy_seconds"] == pytest.approx(30e-6)
    assert out["gaps"] == {"count": 1, "seconds": pytest.approx(20e-6)}


def test_an_empty_gap_list_gives_no_gap_rows() -> None:
    events = [
        _trace_event(mm.STEP_RANGE, "CPU", 0, 100),
        _trace_event("aten::mm", "CPU", 0, 100, 10),
        _trace_event("aten::mm", "CUDA", 0, 100),
    ]
    out = mm.cpu_gpu_summary(events)
    assert out["gaps"] == {"count": 0, "seconds": 0.0}
    assert out["gap_top"] == []


def test_a_gap_with_no_cpu_op_running_is_labelled_not_dropped() -> None:
    events = [
        _trace_event(mm.STEP_RANGE, "CPU", 0, 100),
        _trace_event("aten::mm", "CUDA", 0, 10),
        _trace_event("aten::mm", "CUDA", 30, 40),
    ]
    out = mm.cpu_gpu_summary(events)
    assert out["gap_top"] == [
        {"op": mm.NO_CPU_OP, "gaps": 1, "gap_seconds": pytest.approx(20e-6)}]


def test_top_lists_are_capped_at_fifteen_entries() -> None:
    events = [_trace_event(mm.STEP_RANGE, "CPU", 0, 1000)]
    for index in range(20):
        start = 10 + index * 40
        events.append(_trace_event(f"aten::op{index:02d}", "CPU", start, start + 20, 1 + index))
        events.append(_trace_event("aten::mm", "CUDA", start, start + 10))
    out = mm.cpu_gpu_summary(events)
    assert len(out["cpu_self_top"]) == 15
    assert len(out["gap_top"]) == 15
    assert out["cpu_self_top"][0]["name"] == "aten::op19"


def test_a_trace_without_the_step_range_is_refused() -> None:
    with pytest.raises(ValueError, match=mm.STEP_RANGE):
        mm.cpu_gpu_summary([_trace_event("aten::mm", "CPU", 0, 10, 10)])


# ---- lease timers: opt-in seconds per phase of the expert lease (expert_stream) ----
@pytest.fixture
def fake_clock(monkeypatch):
    from soup_cli.utils import expert_stream

    ticks: list = []
    monkeypatch.setattr(expert_stream, "time", SimpleNamespace(perf_counter=lambda: ticks.pop(0)))
    return ticks


def test_lease_timers_are_off_by_default_and_never_read_the_clock(monkeypatch) -> None:
    from soup_cli.utils import expert_stream

    def no_clock() -> float:
        raise AssertionError("clock read while the lease timers are off")

    monkeypatch.setattr(expert_stream, "time", SimpleNamespace(perf_counter=no_clock))
    with expert_stream.lease_phase("disk_read"):
        pass
    assert expert_stream.take_lease_timers() == dict.fromkeys(expert_stream.LEASE_PHASES, 0.0)


def test_enabled_timers_add_each_phase_its_own_wall_clock(fake_clock) -> None:
    from soup_cli.utils import expert_stream as es

    fake_clock.extend([10.0, 10.5, 20.0, 23.0, 30.0, 31.25])
    with es.lease_timers():
        with es.lease_phase("disk_read"):
            pass
        with es.lease_phase("dequant"):
            pass
        with es.lease_phase("disk_read"):
            pass
    totals = es.take_lease_timers()
    assert totals["disk_read"] == pytest.approx(0.5 + 1.25)
    assert totals["dequant"] == pytest.approx(3.0)
    assert totals["host_copy"] == totals["wait"] == totals["lease"] == 0.0


def test_timers_switch_off_again_when_their_block_ends(monkeypatch) -> None:
    from soup_cli.utils import expert_stream as es

    with es.lease_timers():
        pass

    def no_clock() -> float:
        raise AssertionError("clock read after the lease timers were switched off")

    monkeypatch.setattr(es, "time", SimpleNamespace(perf_counter=no_clock))
    with es.lease_phase("wait"):
        pass
    assert es.take_lease_timers()["wait"] == 0.0


def test_take_returns_the_totals_and_resets_them(fake_clock) -> None:
    from soup_cli.utils import expert_stream as es

    fake_clock.extend([0.0, 2.0])
    with es.lease_timers():
        with es.lease_phase("wait"):
            pass
    assert es.take_lease_timers()["wait"] == 2.0
    assert es.take_lease_timers() == dict.fromkeys(es.LEASE_PHASES, 0.0)


def test_an_unknown_lease_phase_is_refused() -> None:
    from soup_cli.utils import expert_stream as es

    with pytest.raises(ValueError, match="unknown lease phase"):
        with es.lease_phase("gemm"):
            pass


def test_the_lease_timer_step_enables_timers_around_its_one_step_only(fake_clock) -> None:
    from soup_cli.utils import expert_stream as es

    class Chunk:
        shape = (1, 5)

        def clone(self) -> "Chunk":
            return self

    class Stepper:
        def run(self, ids, store, n_items, labels, **options) -> list:
            with es.lease_phase("disk_read"):
                pass
            return []

    fake_clock.extend([1.0, 4.0])
    totals = mm._lease_timer_step(Stepper(), None, [Chunk()], {})
    assert totals["disk_read"] == pytest.approx(3.0)
    assert totals["dequant"] == 0.0


def test_the_run_command_takes_lease_timers_and_writes_them_into_the_row(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    seen = {}

    def fake_run_one(args: argparse.Namespace) -> dict:
        seen["lease_timers"] = args.lease_timers
        row = {"arm": args.arm}
        if args.lease_timers:
            row["lease_timers"] = {"disk_read": 0.5}
        return row

    monkeypatch.setattr(mm, "run_one", fake_run_one)
    base = ["run", "--arm", "C", "--k", "8", "--seq", "64", "--kind", "speed",
            "--source", "s", "--cache", "c", "--out", "o"]
    assert mm.main(base) == 0
    assert seen["lease_timers"] is False
    assert "lease_timers" not in capsys.readouterr().out
    assert mm.main(base + ["--lease-timers"]) == 0
    assert seen["lease_timers"] is True
    assert '"lease_timers": {"disk_read": 0.5}' in capsys.readouterr().out


# ---- arm GP: G's lever settings plus expert_prefetch=True, C's step options ----
def test_gp_takes_the_lever_settings_of_g_plus_expert_prefetch() -> None:
    assert mm.arm_runtime("GP") == {**mm.arm_runtime("G"), "expert_prefetch": True}
    assert mm.arm_runtime("GP") == {"expert_gemm": "lt-fused", "async_lease": True,
                                    "fast_row_gather": True, "expert_loader": "async_lease",
                                    "expert_prefetch": True}


def test_g_has_no_expert_prefetch() -> None:
    assert "expert_prefetch" not in mm.arm_runtime("G")
    assert mm.arm_runtime("G").get("expert_prefetch", False) is False


def test_gp_has_g_step_options_at_every_k() -> None:
    for k in (8, 64):
        assert mm.arm_options("GP", k=k) == mm.arm_options("G", k=k)


def test_parse_arm_list_accepts_gp() -> None:
    assert mm.parse_arm_list("GP") == ("GP",)
    assert mm.parse_arm_list("G,GP") == ("G", "GP")


def test_no_plan_ladder_or_lever_table_lists_gp() -> None:
    plans = [
        mm.speed_plan(), mm.profile_plan(), mm.bit_plan(seq=64), mm.lever_bit_plan(seq=64),
        mm.lever_speed_plan(), mm.traj_plan(), mm.loader_speed_plan(), mm.decoupled_speed_plan(),
        mm.segmented_speed_plan(), mm.esc_speed_plan(), mm.ceg_speed_plan(),
        mm.decoupled_segmented_bit_plan(), mm.abba_plan(blocks=1),
    ]
    arms = {row["arm"] for plan in plans for row in plan}
    assert "GP" not in arms
    assert "GP" not in mm.LEVER_ARMS and "GP" not in mm.LEVER_PREVIOUS
    assert "GP" not in mm.ladder_arms(8) and "GP" not in mm.ladder_arms(64)
    assert "GP" not in mm.arms_for(k=8)


def test_run_one_passes_the_runtime_expert_prefetch_flag_to_the_streamed_model() -> None:
    import ast
    import inspect

    calls = [node for node in ast.walk(ast.parse(inspect.getsource(mm.run_one)))
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "build_streamed_model"]
    assert len(calls) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert keywords["expert_prefetch"] == "runtime_levers.get('expert_prefetch', False)"
