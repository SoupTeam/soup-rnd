"""Measurement runner for the exact-MBR campaign: modes, OOM rows, MBPP data hash (pure CPU)."""

from __future__ import annotations

import argparse

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
            {"k": 16, "arm": "C", "grad_sha256": "c", "loss": 1.0, "status": "ok"}]
    out = mm.ladder_verdict(rows)
    assert out[(1, "B0")] == "equal" and out[(1, "C")] == "differs"
    assert out[(16, "C")] == "no A0"


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
                                   "fast_row_gather": False}
    assert mm.arm_runtime("E") == {"expert_gemm": "lt-fused", "async_lease": False,
                                   "fast_row_gather": False}
    assert mm.arm_runtime("F")["async_lease"] and not mm.arm_runtime("F")["fast_row_gather"]
    assert mm.arm_runtime("G") == {"expert_gemm": "lt-fused", "async_lease": True,
                                   "fast_row_gather": True}
    assert mm.arm_runtime("A0") == mm.arm_runtime("B0") == mm.arm_runtime("C")


def test_lever_speed_plan_is_symmetric_and_normal_mode() -> None:
    plan = mm.lever_speed_plan(blocks=1)
    assert len(plan) == 8 * len(mm.WINDOWS)
    first = [p["arm"] for p in plan[:8]]
    assert first == ["C", "E", "F", "G", "G", "F", "E", "C"]
    assert all(p["kind"] == "speed" and not p["deterministic"] for p in plan)


def test_lever_bit_plan_covers_a0_and_every_lever_in_deterministic_mode() -> None:
    plan = mm.lever_bit_plan(seq=64)
    assert {p["arm"] for p in plan} == {"A0", "C", "E", "F", "G"}
    assert {p["k"] for p in plan} == set(mm.BIT_LADDER)
    assert all(p["kind"] == "bits" and p["deterministic"] for p in plan)


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


def test_tensor_ladder_verdict_reads_saved_tensors_and_compares_each_arm_to_a0(tmp_path) -> None:
    import torch

    def save(arm: str, k: int, grad: list) -> dict:
        name = mm.tensor_file_name(arm, k, 64)
        torch.save({"loss": torch.tensor([2.0], dtype=torch.float64),
                    "grads": {"w": torch.tensor(grad)}}, tmp_path / name)
        return {"arm": arm, "k": k, "seq": 64, "status": "ok", "tensor_file": name}

    rows = [save("A0", 1, [1.0, 2.0]), save("C", 1, [1.0, 2.0]),
            save("E", 1, [1.0, 2.0000002]), {"arm": "F", "k": 1, "seq": 64, "status": "OOM"},
            save("C", 2, [0.5, 0.5])]
    out = mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path)
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
    assert mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path)[(1, "C")] == "no A0"


def test_digest_ladder_verdict_and_tensor_verdict_share_the_a0_rule() -> None:
    rows = [{"arm": "A0", "k": 2, "seq": 64, "status": "ok", "grad_sha256": "a", "loss": 1.0},
            {"arm": "E", "k": 2, "seq": 64, "status": "ok", "grad_sha256": "a", "loss": 1.0},
            {"arm": "C", "k": 2, "seq": 64, "status": "OOM"}]
    assert mm.ladder_verdict(rows) == {(2, "E"): "equal", (2, "C"): "OOM"}


def test_sweep_child_forwards_math_attention_only_when_asked() -> None:
    args = argparse.Namespace(source="s", cache="c", steps=3, out="o", device="auto", lora_r=16,
                              lora_targets=("q_proj",), tokenizer=None, math_attention=True)
    item = {"arm": "C", "k": 2, "seq": 512, "kind": "bits", "deterministic": True}
    assert "--math-attention" in mm.child_run_args(args, item)
    args.math_attention = False
    assert "--math-attention" not in mm.child_run_args(args, item)
