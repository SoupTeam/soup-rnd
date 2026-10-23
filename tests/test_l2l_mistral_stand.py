"""CPU tests for the A3-04 stand script (benchmarks/harness/l2l_mistral_stand.py).

The 7B NF4 run needs a GPU and cannot be tested here. These pin what the script decides
and writes: how two training runs are compared, what an A-against-A failure means, the
RAM plan and its refusal, the determinism setup, the gemm_4bit path counter and the
report. One end-to-end test drives the whole stand on the tiny CPU model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, streamed  # noqa: F401

GIB = 1024**3


@pytest.fixture()
def stand():
    _load("l2l_schedule")
    return _load("l2l_mistral_stand")


def _run(stand, *, grads=None, losses=(1.5, 1.25), rates=(0.5, 0.25), steps=2):
    import torch

    grads = grads or [{"lora_A": torch.tensor([1.0, 2.0])} for _ in range(steps)]
    logs = [{"loss": loss, "learning_rate": rate} for loss, rate in zip(losses, rates)]
    return stand.RunRecord(logs=logs, rates=list(rates), grads=grads, steps=steps)


# --------------------------------------------------------------------------
# comparison of two runs
# --------------------------------------------------------------------------
def test_identical_runs_are_equal(stand):
    out = stand.compare_runs(_run(stand), _run(stand))
    assert out["equal"] is True
    assert out["max_abs"] == 0.0
    assert out["steps"] == (2, 2)


def test_one_ulp_in_one_gradient_is_unequal_and_reported(stand):
    import torch

    bumped = torch.tensor([1.0, 2.0]).clone()
    bumped[1] = torch.nextafter(bumped[1], torch.tensor(3.0))
    cand = _run(stand, grads=[{"lora_A": torch.tensor([1.0, 2.0])}, {"lora_A": bumped}])
    out = stand.compare_runs(_run(stand), cand)
    assert out["grads_equal"] is False
    assert out["equal"] is False
    assert 0.0 < out["max_abs"] < 1e-6  # the real gap, not a rounded-off zero
    assert out["worst"][0] == {"step": 1, "name": "lora_A", "max_abs": out["max_abs"]}


@pytest.mark.parametrize(
    "change, flag",
    [
        ({"losses": (1.5, 1.2500001)}, "losses_equal"),
        ({"rates": (0.5, 0.2)}, "rates_equal"),
        ({"steps": 3}, "steps_equal"),
    ],
)
def test_each_difference_trips_its_own_flag(stand, change, flag):
    cand = _run(stand, **change)
    if change.get("steps") == 3:
        cand.grads.append(cand.grads[0])
    out = stand.compare_runs(_run(stand), cand)
    assert out[flag] is False
    assert out["equal"] is False


# --------------------------------------------------------------------------
# verdict: the A-against-A control comes first, then a tolerance
# --------------------------------------------------------------------------
def _scaled(stand, factor, steps=3, base=(3.0, 4.0)):
    """A run whose gradients are the base vector times ``factor`` (relative error |factor-1|)."""
    import torch

    grads = [{"lora_A": torch.tensor(base) * factor} for _ in range(steps)]
    logs = [{"loss": 2.0 * factor, "learning_rate": 0.5} for _ in range(steps)]
    return stand.RunRecord(logs=logs, rates=[0.5] * steps, grads=grads, steps=steps)


def test_relative_error_is_the_l2_norm_of_the_gap_over_the_norm_of_the_reference(stand):
    errors = stand.rel_l2_by_step(_scaled(stand, 1.0), _scaled(stand, 1.02))
    assert errors == pytest.approx([0.02, 0.02, 0.02])


def test_fp32_passes_below_1e_minus_4_and_fails_above(stand):
    ok = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0), _scaled(stand, 1 + 5e-5),
                      mode="fp32")
    assert ok["pass"] is True and ok["status"] == "within_tolerance"
    bad = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0), _scaled(stand, 1 + 3e-4),
                       mode="fp32")
    assert bad["pass"] is False and bad["status"] == "l2l_out_of_tolerance"
    assert bad["tolerance"]["limit_fp32"] == 1e-4


def test_bf16_limit_is_one_and_a_half_times_the_floor_at_every_step(stand):
    floor = _scaled(stand, 1.04)  # floor error 4% at each step
    inside = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0), _scaled(stand, 1.059),
                          mode="bf16", floor=floor)
    assert inside["pass"] is True
    assert inside["tolerance"]["grad_limit"] == pytest.approx([0.06, 0.06, 0.06])
    outside = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0), _scaled(stand, 1.062),
                           mode="bf16", floor=floor)
    assert outside["pass"] is False and outside["status"] == "l2l_out_of_tolerance"
    assert outside["tolerance"]["factor"] == 1.5


def test_one_bad_step_fails_even_when_the_others_are_inside(stand):
    import torch

    cand = _scaled(stand, 1.0)
    cand.grads[2] = {"lora_A": torch.tensor((3.0, 4.0)) * 1.5}
    out = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0), cand, mode="bf16",
                       floor=_scaled(stand, 1.04))
    assert out["pass"] is False
    assert [row["step"] for row in out["tolerance"]["failing"]] == [2]


def test_bf16_without_a_floor_run_is_not_judged_and_fails(stand):
    out = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0), _scaled(stand, 1.0), mode="bf16")
    assert out["pass"] is False and out["status"] == "no_floor"


def test_a_different_step_count_or_lr_still_fails_under_a_tolerance(stand):
    short = _scaled(stand, 1.0, steps=2)
    out = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0), short, mode="fp32")
    assert out["pass"] is False and out["status"] == "l2l_differs"
    cand = _scaled(stand, 1.0)
    cand.rates = [0.5, 0.5, 0.25]
    out = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0), cand, mode="fp32")
    assert out["pass"] is False and "rates_equal" in out["note"]


def test_losses_are_held_to_the_same_tolerance(stand):
    cand = _scaled(stand, 1.0)
    cand.logs[1] = {"loss": 2.0 * 1.01, "learning_rate": 0.5}
    out = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0), cand, mode="fp32")
    assert out["pass"] is False
    assert out["tolerance"]["failing"][0]["what"] == "loss"


def test_unequal_a_vs_a_stops_the_run_and_does_not_judge_l2l(stand):
    out = stand.decide(_scaled(stand, 1.0), _scaled(stand, 1.0 + 1e-7), _scaled(stand, 1.0),
                       mode="fp32")
    assert out["pass"] is False
    assert out["status"] == "a_vs_a_failed"
    assert "stops" in out["note"]
    assert out["l2l_vs_ga"] is None  # not judged against a reference that does not repeat


def test_all_zero_gradients_are_never_a_pass(stand):
    import torch

    zero = [{"lora_A": torch.zeros(2)} for _ in range(2)]
    out = stand.decide(_run(stand, grads=zero), _run(stand, grads=zero), _run(stand, grads=zero),
                       mode="fp32")
    assert out["pass"] is False
    assert out["status"] == "vacuous"


# --------------------------------------------------------------------------
# RAM plan
# --------------------------------------------------------------------------
def test_ram_plan_matches_the_worked_examples(stand):
    plan = stand.ram_plan(n_layers=32, hidden=4096, k=4, batch=1, seq=1024)
    assert plan["chunk_bytes"] == 8 * 1024**2
    assert plan["cache_bytes"] == 1 * GIB
    assert plan["need_free_for_cache"] == pytest.approx(1 * GIB / 0.7)
    assert plan["weights_bytes"] == pytest.approx(4.1e9)
    assert plan["workspace_bytes"] == 1_000_000_000
    assert plan["need_total"] == pytest.approx(4.1e9 + 1e9 + GIB / 0.7)
    big = stand.ram_plan(n_layers=32, hidden=4096, k=4, batch=4, seq=2048)
    assert big["cache_bytes"] == 8 * GIB


def test_ram_plan_rounds_a_pinned_block_up_to_a_power_of_two(stand):
    plan = stand.ram_plan(n_layers=32, hidden=4096, k=4, batch=1, seq=1500)
    assert plan["chunk_bytes"] == 1500 * 4096 * 2
    assert plan["block_bytes"] == 16_777_216
    assert plan["cache_bytes"] == 32 * 4 * 16_777_216


def test_check_ram_refuses_below_the_total_and_allows_above(stand):
    plan = stand.ram_plan(n_layers=32, hidden=4096, k=4, batch=1, seq=1024)
    with pytest.raises(stand.StandRefusedError, match="free RAM"):
        stand.check_ram(plan, free_ram=int(plan["need_total"]) - 1)
    stand.check_ram(plan, free_ram=int(plan["need_total"]) + 1)


def test_check_ram_refuses_when_free_ram_is_unknown(stand):
    plan = stand.ram_plan(n_layers=32, hidden=4096, k=4, batch=1, seq=1024)
    with pytest.raises(stand.StandRefusedError, match="unknown"):
        stand.check_ram(plan, free_ram=None)


# --------------------------------------------------------------------------
# determinism setup
# --------------------------------------------------------------------------
def test_cublas_workspace_is_set_before_cuda_starts(stand):
    environ = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    stand.set_deterministic_env(environ, cuda_initialized=lambda: False)
    assert environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_setting_it_after_cuda_started_is_refused(stand):
    with pytest.raises(stand.StandRefusedError, match="CUBLAS_WORKSPACE_CONFIG"):
        stand.set_deterministic_env({}, cuda_initialized=lambda: True)


def test_strict_deterministic_mode_has_no_warn_only(stand):
    import torch

    try:
        stand.enable_deterministic()
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# SDPA backend
# --------------------------------------------------------------------------
def test_sdpa_probe_checks_every_backend_and_math_works(stand):
    out = stand.probe_sdpa_backends(device="cpu", dtype="float32")
    assert set(out) == {"MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"}
    assert out["MATH"]["ok"] is True
    assert out["MATH"]["backward_repeatable"] is True
    assert stand.PINNED_SDPA_BACKEND == "MATH"


def test_the_pinned_sdpa_context_still_runs_attention(stand):
    import torch

    q = torch.randn(1, 2, 4, 8)
    with stand.pinned_sdpa():
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert out.shape == q.shape


# --------------------------------------------------------------------------
# gemm_4bit path counter
# --------------------------------------------------------------------------
def _fake_runtime(custom_up_to=128):
    """A stand-in for ``layer_stream_runtime``: the dispatch decision the patched forward asks."""
    return SimpleNamespace(
        _can_use_checkpoint_visible_nf4_gemm=lambda x, state: x.shape[0] <= custom_up_to
    )


def _mat(rows):
    return SimpleNamespace(numel=lambda: rows * 8, shape=(rows, 8))


def test_path_counter_counts_every_m_including_above_128_and_restores_the_hook(stand):
    runtime = _fake_runtime()
    original = runtime._can_use_checkpoint_visible_nf4_gemm
    with stand.GemmPathCounter(runtime) as counter:
        assert runtime._can_use_checkpoint_visible_nf4_gemm(_mat(64), None) is True
        runtime._can_use_checkpoint_visible_nf4_gemm(_mat(2048), None)
        runtime._can_use_checkpoint_visible_nf4_gemm(_mat(2048), None)
    assert counter.counts == {64: {"custom": 1, "dequant": 0}, 2048: {"custom": 0, "dequant": 2}}
    assert runtime._can_use_checkpoint_visible_nf4_gemm is original


def test_paths_match_compares_the_kinds_of_path_not_the_m_values(stand):
    one = {2048: {"custom": 0, "dequant": 5}}
    assert stand.paths_match(one, {1024: {"custom": 0, "dequant": 9}}) is True
    assert stand.paths_match(one, {2048: {"custom": 3, "dequant": 0}}) is False
    assert stand.paths_match(one, {2048: {"custom": 3, "dequant": 5}}) is False
    assert stand.paths_match(one, None) is None


def test_an_empty_counter_is_not_a_match(stand):
    # 05 recorded ``l2l: {}`` and compared it with a non-empty GA: two empties must not agree.
    assert stand.paths_match({}, {2048: {"custom": 0, "dequant": 1}}) is None
    assert stand.paths_match({}, {}) is None


def test_path_counter_without_a_runtime_reports_not_counted(stand):
    assert stand.GemmPathCounter(None).counts is None


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _payload(stand, **over):
    base = {
        "data_kind": "synthetic",
        "model": "mistralai/Mistral-7B-v0.1",
        "command": ["python", "l2l_mistral_stand.py", "--out", "runs/a"],
        "env": {"packages": {"torch": "2.14.1", "bitsandbytes": "0.50.2"}, "gpu": "RTX 3060"},
        "deterministic": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "algorithms": "strict"},
        "sdpa": {"pinned": "MATH", "probe": {"MATH": {"ok": True}}},
        "gemm_paths": {"ga1": {2048: {"custom": 0, "dequant": 4}},
                       "ga2": {2048: {"custom": 0, "dequant": 4}},
                       "l2l": {2048: {"custom": 0, "dequant": 4}}, "match": True},
        "ram_plan": {"cache_bytes": GIB, "need_total": 7e9},
        "verdict": {"pass": True, "status": "within_tolerance", "note": "",
                    "a_vs_a": {"equal": True},
                    "l2l_vs_ga": {"equal": False, "max_abs": 0.25, "steps": (3, 3)},
                    "tolerance": {"mode": "bf16", "factor": 1.5, "limit_fp32": 1e-4,
                                  "grad_rel": [0.018, 0.016, 0.053],
                                  "grad_limit": [0.021, 0.0225, 0.0675],
                                  "floor_grad": [0.014, 0.015, 0.045],
                                  "loss_rel": [0.0, 0.0, 0.0], "loss_limit": [0.0, 0.0, 0.0],
                                  "floor_loss": [0.0, 0.0, 0.0], "failing": [], "pass": True}},
    }
    base.update(over)
    return base


def test_report_has_every_required_fact(stand):
    text = stand.render_report(_payload(stand))
    for needle in (
        "python l2l_mistral_stand.py --out runs/a", "torch 2.14.1", "bitsandbytes 0.50.2",
        "RTX 3060", ":4096:8", "MATH", "gemm_4bit", "PASS", "A-vs-A",
    ):
        assert needle in text, needle
    assert "ga1: {2048: {'custom': 0, 'dequant': 4}}" in text
    assert "not counted" not in text


def test_report_prints_the_tolerance_and_its_values(stand):
    text = stand.render_report(_payload(stand))
    assert "## Tolerance" in text
    assert "1.5" in text and "1e-04" in text.replace("0.0001", "1e-04")
    assert "0.053" in text and "0.0675" in text  # error and limit of the worst step


def test_an_empty_gemm_counter_gets_a_warning(stand):
    text = stand.render_report(_payload(stand, gemm_paths={"ga1": {}, "l2l": {}, "match": None}))
    assert "WARNING: gemm_4bit paths were not counted" in text


def test_uncounted_gemm_paths_get_a_warning(stand):
    text = stand.render_report(_payload(stand, gemm_paths={"match": None}))
    assert "WARNING: gemm_4bit paths were not counted" in text


def test_synthetic_and_real_results_are_labelled_apart(stand):
    synthetic = stand.render_report(_payload(stand, data_kind="synthetic"))
    real = stand.render_report(_payload(stand, data_kind="real"))
    assert "Synthetic data" in synthetic and "Real data: not run" in synthetic
    assert "Real data" in real and "Synthetic data: not run" in real
    assert "Synthetic data: not run" not in synthetic


def test_a_failed_verdict_is_printed_with_its_note(stand):
    text = stand.render_report(_payload(stand, verdict={
        "pass": False, "status": "a_vs_a_failed", "note": "move the criterion to a tolerance",
        "a_vs_a": {"equal": False, "max_abs": 1e-7}, "l2l_vs_ga": None,
    }))
    assert "FAIL" in text and "move the criterion to a tolerance" in text


def test_write_outputs_keeps_the_report_the_records_and_the_raw_logs(stand, tmp_path):
    stand.write_outputs(tmp_path / "out", _payload(stand), {"ga1.log": "a\n", "l2l.log": "b\n"})
    out = tmp_path / "out"
    assert (out / "report.md").read_text().startswith("#")
    assert json.loads((out / "records.json").read_text())["data_kind"] == "synthetic"
    assert (out / "raw" / "ga1.log").read_text() == "a\n"
    assert (out / "raw" / "l2l.log").read_text() == "b\n"


def test_write_outputs_never_overwrites_an_attempt(stand, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(stand.StandRefusedError, match="exists"):
        stand.write_outputs(tmp_path / "out", _payload(stand), {})


def test_environment_facts_name_versions_and_the_command(stand):
    facts = stand.environment_facts(["torch", "no_such_package_xyz"])
    assert facts["packages"]["torch"]
    assert facts["packages"]["no_such_package_xyz"] is None
    assert "python" in facts and "platform" in facts


# --------------------------------------------------------------------------
# the whole stand on the tiny CPU model
# --------------------------------------------------------------------------
def test_the_stand_passes_end_to_end_on_the_tiny_model(streamed, stand, tmp_path):  # noqa: F811
    model, runtime = streamed
    rows = stand.SyntheticRows(
        lengths=[SEQ] * 7, masked_spans=[[(0, i % 3)] if i % 3 else [] for i in range(7)],
        vocab=64,
    )
    result = stand.run_stand(
        model, runtime, rows, k=3, work_dir=tmp_path,
        trainer_options={"use_cpu": True}, dtype=None, mode="fp32",
    )
    verdict = result.verdict
    assert verdict["status"] == "within_tolerance" and verdict["pass"] is True
    assert verdict["tolerance"]["grad_rel"] == pytest.approx([0.0] * 3, abs=1e-4)
    assert verdict["a_vs_a"]["equal"] is True
    assert verdict["l2l_vs_ga"]["steps"] == (3, 3)  # windows of 3, 3 and 1 row
    assert set(result.logs) == {"ga1.log", "ga2.log", "l2l.log"}
    assert all(result.logs.values())


def test_the_stand_reports_a_failure_when_l2l_is_broken(streamed, stand, tmp_path, monkeypatch):  # noqa: F811
    model, runtime = streamed
    rows = stand.SyntheticRows(lengths=[SEQ] * 3, masked_spans=[[], [], []], vocab=64)
    real = stand.train_arm

    def broken(*args, **kwargs):
        record = real(*args, **kwargs)
        if record.l2l:  # an L2L whose gradients are 1% off
            record.grads = [{n: g * 1.01 for n, g in step.items()} for step in record.grads]
        return record

    monkeypatch.setattr(stand, "train_arm", broken)
    result = stand.run_stand(
        model, runtime, rows, k=3, work_dir=tmp_path,
        trainer_options={"use_cpu": True}, dtype=None, mode="fp32",
    )
    assert result.verdict["status"] == "l2l_out_of_tolerance"
    assert result.verdict["pass"] is False


def test_the_stand_runs_a_floor_arm_when_given_a_second_backend(streamed, stand, tmp_path):  # noqa: F811
    model, runtime = streamed
    rows = stand.SyntheticRows(lengths=[SEQ] * 3, masked_spans=[[], [], []], vocab=64)
    result = stand.run_stand(
        model, runtime, rows, k=3, work_dir=tmp_path, trainer_options={"use_cpu": True},
        dtype=None, mode="bf16", floor_sdpa=stand.pinned_sdpa,
    )
    assert "ga_floor.log" in result.logs
    assert result.verdict["tolerance"]["floor_grad"] is not None


# --------------------------------------------------------------------------
# weights estimate and the command line (no GPU)
# --------------------------------------------------------------------------
def test_weights_estimate_matches_the_mistral_figure(stand):
    config = SimpleNamespace(
        hidden_size=4096, num_hidden_layers=32, intermediate_size=14336,
        num_attention_heads=32, num_key_value_heads=8, vocab_size=32000,
        tie_word_embeddings=False,
    )
    assert stand.weights_bytes_estimate(config, "nf4") == pytest.approx(4.1e9, rel=0.02)
    assert stand.weights_bytes_estimate(config, "none") == pytest.approx(14.5e9, rel=0.02)


def test_main_refuses_a_row_count_that_fills_every_window(stand, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    code = stand.main(["--rows", "8", "--k", "4", "--out", str(tmp_path / "run")])
    assert code == 2
    assert "incomplete last window" in capsys.readouterr().out


def test_main_refuses_an_existing_output_directory(stand, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    code = stand.main(["--rows", "10", "--k", "4", "--out", str(tmp_path)])
    assert code == 2
    assert "never overwritten" in capsys.readouterr().out


def test_main_sets_the_cublas_workspace_before_anything_else(stand, tmp_path, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    stand.main(["--rows", "8", "--k", "4", "--out", str(tmp_path / "run")])
    import os

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")


# --------------------------------------------------------------------------
# bnb #2034: a recycled slot buffer must not change the gradients
# --------------------------------------------------------------------------
def test_aliasing_is_active_when_the_shipped_path_differs(stand):
    out = stand.judge_aliasing(shipped_max_abs=1e-3, raw_max_abs=0.5)
    assert out["status"] == "active" and out["pass"] is False
    assert "2034" in out["note"]


def test_aliasing_is_neutralized_when_only_the_raw_path_differs(stand):
    out = stand.judge_aliasing(shipped_max_abs=0.0, raw_max_abs=0.5)
    assert out["status"] == "neutralized" and out["pass"] is True


def test_aliasing_not_reproduced_still_passes_but_says_the_probe_is_blind(stand):
    out = stand.judge_aliasing(shipped_max_abs=0.0, raw_max_abs=0.0)
    assert out["status"] == "not_reproduced" and out["pass"] is True
    assert "cannot show" in out["note"]


def test_the_probe_passes_on_the_shipped_path_for_each_m(stand):
    pytest.importorskip("bitsandbytes")
    probe = stand.run_aliasing_probe(
        device="cpu", dtype="float32", m_values=[8, 24], dim=64, n_layers=4
    )
    assert probe["pass"] is True
    assert sorted(probe["by_m"]) == [8, 24]
    for record in probe["by_m"].values():
        assert record["shipped_max_abs"] == 0.0
        assert record["raw_max_abs"] >= 0.0


def test_the_probe_stops_the_stand_when_the_shipped_path_aliases(stand, monkeypatch):
    pytest.importorskip("bitsandbytes")
    import soup_cli.utils.layer_stream_runtime as runtime

    # Without the patch the layers go through bitsandbytes' MatMul4Bit itself.
    monkeypatch.setattr(runtime, "install_dequant_forward", lambda module: 0)
    probe = stand.run_aliasing_probe(
        device="cpu", dtype="float32", m_values=[8], dim=64, n_layers=4
    )
    if probe["by_m"][8]["raw_max_abs"] == 0.0:
        pytest.skip("this bitsandbytes build does not reproduce #2034 on CPU")
    assert probe["pass"] is False
    verdict = stand.aliasing_block(probe)
    assert verdict["pass"] is False and verdict["status"] == "aliasing_active"
    assert stand.aliasing_block({"pass": True, "by_m": {}}) is None


def test_the_report_has_an_aliasing_section(stand):
    probe = {"pass": True, "status": "neutralized", "by_m": {8: {"shipped_max_abs": 0.0,
             "raw_max_abs": 0.25, "status": "neutralized"}}}
    text = stand.render_report(_payload(stand, aliasing=probe))
    assert "bitsandbytes #2034" in text and "neutralized" in text
    assert "not run" in stand.render_report(_payload(stand)).split("#2034")[1].split("##")[0]
