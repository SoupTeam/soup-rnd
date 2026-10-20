"""E4 part 2: the harness matches the calibration rule where it was found not to.

1 W: a changed frozen base fails calibration (reported, no rerun), 2 P: a non-finite loss is a
void, 3 F: a recipe mismatch fails F, 4 the report projects the main plan's wall time, 5 W: an
fp32 reference that cannot run with the pinned experts implementation is a void, 6 the
attention implementation is pinned and recorded.
"""
# ruff: noqa: I001 (the support import must run before the harness imports)

import json

import pytest

from tests.e4_part2_support import tiny_model, write_legacy_checkpoint
from tests.test_e4_part2_calibration import E, GOOD_PROV, SMALL, synthetic_examples

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")

import e4_part2_calibration as cal  # noqa: E402
import e4_part2_common as common  # noqa: E402
import e4_part2_quant as quant  # noqa: E402
import e4_part2_wref as wref  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(common.library_versions()["mismatches"]),
    reason="the E4 part 2 harness is pinned to torch/transformers/bitsandbytes/peft of the rule",
)


@pytest.fixture(autouse=True)
def _own_gpu_context(monkeypatch):
    """execute_attempt identifies its own GPU context; there is no GPU here."""
    import os

    monkeypatch.setattr(common, "identify_own_gpu_pids_cuda", lambda gpu_index=0, **kw: {
        "own_pids": [os.getpid()], "method": "stub"})


@pytest.fixture(scope="module")
def tiny_world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    model = tiny_model(seed=6, dtype=torch.bfloat16)
    source = write_legacy_checkpoint(model, root / "source")
    data = root / "data"
    quant.run(source, data, root / "manifest.json", None, 2, E, log=lambda *_: None)
    return {"source": source, "data": data, "model": model}


def run_w_on_tiny(world, **kwargs):
    rows = [r["input_ids"] for r in synthetic_examples(4, seed=8)]
    return cal.run_w({"A_off": world["data"] / "writeback-dqoff",
                      "A_on": world["data"] / "writeback-dqon"},
                     rows, SMALL, "cpu", layers=(0, 1), rows=4,
                     experts_implementation="eager", **kwargs)


# --- 1 W: frozen base changed = calibration fails ---------------------------------------------


class TestWFrozenBaseFails:
    @pytest.fixture
    def changed(self, tiny_world, monkeypatch):
        counter = iter(range(10_000))
        monkeypatch.setattr(cal, "hash_base", lambda _experts: str(next(counter)))
        return run_w_on_tiny(tiny_world)

    def test_a_changed_base_is_a_failed_result_not_a_void_reason(self, changed):
        assert changed["failed"] is True
        assert changed["frozen_base_changed"]
        assert not any("frozen base" in r for r in changed["void_reasons"])

    def test_a_failed_w_gives_no_tolerance(self, changed):
        assert changed["tol_W"] == {}

    def test_an_unchanged_base_is_not_failed(self, tiny_world):
        out = run_w_on_tiny(tiny_world)
        assert out["failed"] is False and out["tol_W"]

    def test_the_attempt_is_failed_and_may_not_be_rerun(self, changed, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        record = cal.execute_attempt(
            journal, "W", "W", tmp_path / "runs", lambda: dict(changed), require_gpu=False,
            stamp=lambda *a, **k: {"unix_s": 1.0},
            prov=lambda: {"soup_commit": "a", "harness_commit": "b", "harness_tree": "c",
                          "harness_dirty": False})
        assert record["outcome"] == common.OUTCOME_FAILED
        with pytest.raises(RuntimeError, match="not rerun"):
            journal.assert_may_start("W")

    def test_the_driver_stops_calibration_on_a_failed_stage(self, tmp_path, monkeypatch):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")

        def failing_child(j, run_id, _argv):
            attempt = j.start(run_id, "W")
            j.end(attempt, common.OUTCOME_FAILED)

        monkeypatch.setattr(cal, "drive_child", failing_child)
        with pytest.raises(common.CalibrationStopped, match="failed"):
            cal.drive_run(journal, "W", [], stop_on_failed=True)

    def test_the_driver_still_returns_a_failed_f_run(self, tmp_path, monkeypatch):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")

        def failing_child(j, run_id, _argv):
            attempt = j.start(run_id, "F")
            j.end(attempt, common.OUTCOME_FAILED)

        monkeypatch.setattr(cal, "drive_child", failing_child)
        assert cal.drive_run(journal, "F_A_on", [])["outcome"] == common.OUTCOME_FAILED


# --- 2 P: a non-finite loss is a void ---------------------------------------------------------


class TestPNonFiniteIsVoid:
    CLEAN = {"all_finite": True, "smoke_nll": {"finite": True}, "frozen_base_unchanged": True,
             "failed": False}

    def test_a_non_finite_update_loss_voids_p(self):
        out = cal.finalize_p_result({**self.CLEAN, "all_finite": False, "failed": True})
        assert out["void_reasons"] and out["void_reasons"][0].startswith("finiteness: P")
        assert out["failed"] is False

    def test_a_non_finite_smoke_nll_voids_p(self):
        out = cal.finalize_p_result({**self.CLEAN, "smoke_nll": {"finite": False}, "failed": True})
        assert out["void_reasons"] and out["failed"] is False

    def test_a_changed_frozen_base_still_fails_p(self):
        out = cal.finalize_p_result({**self.CLEAN, "frozen_base_unchanged": False,
                                     "failed": True})
        assert out["failed"] is True and out["void_reasons"] == []

    def test_a_clean_run_has_neither(self):
        out = cal.finalize_p_result(dict(self.CLEAN))
        assert out["failed"] is False and out["void_reasons"] == []

    def test_the_attempt_is_a_void_that_may_be_rerun_once(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        bad = cal.finalize_p_result({**self.CLEAN, "all_finite": False, "failed": True})
        record = cal.execute_attempt(
            journal, "P", "P", tmp_path / "runs", lambda: dict(bad), require_gpu=False,
            stamp=lambda *a, **k: {"unix_s": 1.0},
            prov=lambda: {"soup_commit": "a", "harness_commit": "b", "harness_tree": "c",
                          "harness_dirty": False})
        assert record["outcome"] == common.OUTCOME_VOID
        journal.assert_may_start("P")  # one rerun is allowed


# --- 3 F: trainable count, fp32 factors, placement --------------------------------------------


class TestFRecipeMismatchFails:
    @pytest.fixture
    def directory(self, tmp_path):
        return write_legacy_checkpoint(tiny_model(seed=4, dtype=torch.bfloat16), tmp_path / "m")

    def run_f(self, directory, tmp_path):
        return cal.run_f(directory, "A_bf16", synthetic_examples(12), synthetic_examples(4, 9),
                         SMALL, "cpu", 1, None, tmp_path / "adapter",
                         experts_implementation="eager")

    def test_a_correct_model_passes(self, directory, tmp_path):
        res = self.run_f(directory, tmp_path)
        assert res["failed"] is False and res["recipe_mismatch"] == []
        assert res["placement"]["matches"]

    def test_a_wrong_trainable_count_fails_f(self, directory, tmp_path, monkeypatch):
        real = cal.trainable_report
        monkeypatch.setattr(cal, "trainable_report", lambda *a, **k: {
            **real(*a, **k), "matches": False})
        res = self.run_f(directory, tmp_path)
        assert res["failed"] is True
        assert any("trainable" in m for m in res["recipe_mismatch"])

    def test_a_non_fp32_factor_fails_f(self, directory, tmp_path, monkeypatch):
        real = cal.trainable_report
        monkeypatch.setattr(cal, "trainable_report", lambda *a, **k: {
            **real(*a, **k), "non_fp32_factors": ["x.lora_A"]})
        res = self.run_f(directory, tmp_path)
        assert res["failed"] is True
        assert any("fp32" in m for m in res["recipe_mismatch"])

    def test_wrong_placement_fails_f(self, directory, tmp_path, monkeypatch):
        from peft import LoraConfig, get_peft_model

        def attention_only(model, recipe=cal.RECIPE, gradient_checkpointing=True):
            peft_model = get_peft_model(model, LoraConfig(
                r=recipe.r, lora_alpha=recipe.alpha, lora_dropout=0.0, bias="none",
                target_modules=list(cal.ATTENTION_KINDS)))
            peft_model.train()
            return peft_model

        monkeypatch.setattr(cal, "build_lora_model", attention_only)
        res = self.run_f(directory, tmp_path)
        assert res["failed"] is True
        assert any("placement" in m for m in res["recipe_mismatch"])

    def test_a_mismatch_makes_the_final_card_decision_f_fails(self):
        ok = {"oom": False, "failed": False, "max_memory_reserved_bytes": 1}
        bad = {"oom": False, "failed": True, "max_memory_reserved_bytes": 1,
               "recipe_mismatch": ["trainable count"]}
        out = cal.decide_final_card({"A_off": ok, "A_on": bad, "A_bf16": ok})
        assert out["decision"] == "F_FAILS" and "A_on" in out["reason"]


class TestPlacementReport:
    def test_the_rule_placement_matches(self):
        pm = cal.build_lora_model(tiny_model(seed=4, dtype=torch.bfloat16), SMALL,
                                  gradient_checkpointing=False)
        report = cal.placement_report(pm, pm.base_model.model.config)
        assert report["matches"] and report["missing"] == [] and report["unexpected"] == []

    def test_attention_only_is_missing_the_expert_adapters(self):
        from peft import LoraConfig, get_peft_model

        pm = get_peft_model(tiny_model(seed=4, dtype=torch.bfloat16), LoraConfig(
            r=16, lora_alpha=32, target_modules=list(cal.ATTENTION_KINDS)))
        report = cal.placement_report(pm, pm.base_model.model.config)
        assert not report["matches"] and any("experts" in m for m in report["missing"])

    def test_an_extra_trainable_tensor_is_reported(self):
        pm = cal.build_lora_model(tiny_model(seed=4, dtype=torch.bfloat16), SMALL,
                                  gradient_checkpointing=False)
        pm.base_model.model.lm_head.weight.requires_grad_(True)
        report = cal.placement_report(pm, pm.base_model.model.config)
        assert not report["matches"] and any("lm_head" in m for m in report["unexpected"])


# --- 4 the projected wall time of the main plan ------------------------------------------------


class TestProjectedWallTime:
    def test_runs_are_nine_plus_one_rerun_of_a_on_over_256_updates(self):
        medians = {"A_off": 2.0, "A_on": 3.0, "A_bf16": 5.0}
        out = cal.project_main_plan(medians)
        # 3 seeds x (off + on + bf16) + one on/42 rerun, 256 updates each
        assert out["total_seconds"] == pytest.approx(256 * (3 * (2.0 + 3.0 + 5.0) + 3.0))
        assert out["runs"] == 9 and out["reruns"] == 1 and out["updates_per_run"] == 256
        assert out["total_hours"] == pytest.approx(out["total_seconds"] / 3600)

    def test_a_missing_arm_gives_no_projection(self):
        assert cal.project_main_plan({"A_off": 2.0, "A_on": 3.0}) is None
        assert cal.project_main_plan({"A_off": 2.0, "A_on": None, "A_bf16": 1.0}) is None

    def test_the_report_carries_it_from_the_f_records(self, tmp_path):
        out = tmp_path
        journal = common.AttemptsJournal(out / "attempts.jsonl")
        (out / "runs").mkdir()
        for arm, median in (("A_off", 2.0), ("A_on", 3.0), ("A_bf16", 5.0)):
            attempt = journal.start(f"F_{arm}", "F")
            record = {"run_id": f"F_{arm}", "result": {
                "arm": arm, "oom": False, "failed": False, "median_seconds_per_update": median,
                "max_memory_reserved_bytes": 1}}
            (out / "runs" / f"{attempt}.json").write_text(json.dumps(record))
            journal.end(attempt, common.OUTCOME_OK)
        report = cal.assemble_report(out)
        assert report["projected_main_plan"]["total_seconds"] == pytest.approx(
            256 * (3 * 10.0 + 3.0))

    def test_the_report_lists_failed_runs(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        (tmp_path / "runs").mkdir()
        attempt = journal.start("W", "W")
        (tmp_path / "runs" / f"{attempt}.json").write_text(json.dumps({"result": {
            "tol_W": {}, "failed": True, "frozen_base_changed": ["A_on layer 0 call 1"]}}))
        journal.end(attempt, common.OUTCOME_FAILED)
        report = cal.assemble_report(tmp_path)
        assert [f["run_id"] for f in report["failed_runs"]] == ["W"]


# --- 5 W: an fp32 reference that cannot run is a void -----------------------------------------


class TestWFp32CannotRunIsVoid:
    def test_a_failing_fp32_reference_voids_w_and_never_falls_back(self, tiny_world,
                                                                    monkeypatch):
        real = wref.run_reference
        impls = []

        def flaky(experts, base, routed, state, g, precision, kwargs, impl=None):
            impls.append((precision, impl))
            if precision == "fp32":
                raise RuntimeError("grouped_mm: unsupported dtype float32")
            return real(experts, base, routed, state, g, precision, kwargs, impl=impl)

        monkeypatch.setattr(wref, "run_reference", flaky)
        out = run_w_on_tiny(tiny_world)
        assert any("fp32 reference" in r and "grouped_mm" in r for r in out["void_reasons"])
        assert out["tol_W"] == {}
        assert ("fp32", "eager") not in impls
        assert "fp32_fallback_to_eager" not in out

    def test_it_stops_at_the_first_failure(self, tiny_world, monkeypatch):
        real = wref.run_reference
        calls = []

        def flaky(experts, base, routed, state, g, precision, kwargs, impl=None):
            calls.append(precision)
            if precision == "fp32":
                raise RuntimeError("boom")
            return real(experts, base, routed, state, g, precision, kwargs, impl=impl)

        monkeypatch.setattr(wref, "run_reference", flaky)
        run_w_on_tiny(tiny_world)
        assert calls == ["bf16", "fp32"]

    def test_the_void_attempt_may_be_rerun_once(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        record = cal.execute_attempt(
            journal, "W", "W", tmp_path / "runs",
            lambda: {"void_reasons": ["fp32 reference: RuntimeError: boom"]}, require_gpu=False,
            stamp=lambda *a, **k: {"unix_s": 1.0},
            prov=lambda: {"soup_commit": "a", "harness_commit": "b", "harness_tree": "c",
                          "harness_dirty": False})
        assert record["outcome"] == common.OUTCOME_VOID
        journal.assert_may_start("W")


# --- 6 the attention implementation is pinned and recorded ------------------------------------


class TestAttentionImplementationPinned:
    def test_the_pin_is_a_constant_of_the_harness(self):
        assert common.ATTENTION_IMPLEMENTATION == "sdpa"

    def test_load_base_passes_it_to_the_model(self, tmp_path, monkeypatch):
        directory = write_legacy_checkpoint(tiny_model(seed=4, dtype=torch.bfloat16),
                                            tmp_path / "m")
        from transformers import AutoModelForCausalLM

        seen = {}
        real = AutoModelForCausalLM.from_pretrained

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", spy)
        model = cal.load_base(directory, "cpu", "eager")
        assert seen["attn_implementation"] == common.ATTENTION_IMPLEMENTATION
        assert model.config._attn_implementation == "sdpa"

    def test_run_f_records_both_resolved_implementations(self, tmp_path):
        directory = write_legacy_checkpoint(tiny_model(seed=4, dtype=torch.bfloat16),
                                            tmp_path / "m")
        res = cal.run_f(directory, "A_bf16", synthetic_examples(12), synthetic_examples(4, 9),
                        SMALL, "cpu", 1, None, tmp_path / "adapter",
                        experts_implementation="eager")
        assert res["attention_implementation_resolved"] == "sdpa"
        assert res["experts_implementation_resolved"] == "eager"

    def test_w_and_t_record_it_too(self, tiny_world):
        out = run_w_on_tiny(tiny_world)
        assert out["attention_implementation_resolved"] == ["sdpa"]

    def test_the_report_carries_it_per_arm(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        (tmp_path / "runs").mkdir()
        attempt = journal.start("F_A_on", "F")
        (tmp_path / "runs" / f"{attempt}.json").write_text(json.dumps({"result": {
            "arm": "A_on", "attention_implementation_resolved": "sdpa"}}))
        journal.end(attempt, common.OUTCOME_OK)
        report = cal.assemble_report(tmp_path)
        assert report["f"]["A_on"]["attention_implementation_resolved"] == "sdpa"



# --- 7 review: the rest stage is bound to the final F card ------------------------------------


def write_decision(out, decision):
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps(decision))


INSTANCE = {"gpu_uuid": "GPU-chosen", "instance_id": "777"}


def drive_args(out, f_prefix, stage="rest", device="cpu", gpu_index=0):
    import types

    return types.SimpleNamespace(
        out=out, f_prefix=f_prefix, stage=stage, data_root=out / "data", split_dir=out / "split",
        manifest=out / "manifest.json", device=device, gpu_index=gpu_index, instance_id=None)


class TestFPrefixFollowsTheDecision:
    @pytest.mark.parametrize("decision,prefix", [("FINAL_A100_40GB", "F"),
                                                 ("FINAL_A100_80GB", "F80")])
    def test_the_prefix_is_derived_from_decision_json(self, tmp_path, decision, prefix):
        write_decision(tmp_path, {"decision": decision})
        assert cal.resolve_f_prefix(tmp_path, None) == prefix
        assert cal.resolve_f_prefix(tmp_path, prefix) == prefix

    def test_an_explicit_prefix_that_disagrees_is_an_error(self, tmp_path):
        write_decision(tmp_path, {"decision": "FINAL_A100_80GB"})
        with pytest.raises(common.CalibrationStopped, match="F80"):
            cal.resolve_f_prefix(tmp_path, "F")
        write_decision(tmp_path, {"decision": "FINAL_A100_40GB"})
        with pytest.raises(common.CalibrationStopped, match="F"):
            cal.resolve_f_prefix(tmp_path, "F80")

    @pytest.mark.parametrize("decision", ["REPEAT_F_ON_A100_80GB", "INCOMPLETE", "F_FAILS",
                                          "STOP_RECIPE_TO_OWNER"])
    def test_a_decision_that_is_not_a_final_card_refuses(self, tmp_path, decision):
        write_decision(tmp_path, {"decision": decision})
        with pytest.raises(common.CalibrationStopped, match=decision):
            cal.resolve_f_prefix(tmp_path, None)

    def test_no_decision_file_refuses(self, tmp_path):
        with pytest.raises(common.CalibrationStopped, match="decision.json"):
            cal.resolve_f_prefix(tmp_path, None)

    def test_the_rest_stage_hands_the_derived_prefix_to_every_child(self, tmp_path, monkeypatch):
        write_decision(tmp_path, {"decision": "FINAL_A100_80GB", "final_instance": INSTANCE})
        argvs = []
        monkeypatch.setattr(cal, "drive_run", lambda j, run_id, argv, **kw: argvs.append(argv))
        monkeypatch.setattr(cal, "_cmd_report", lambda a: 0)
        assert cal._cmd_drive(drive_args(tmp_path, None)) == 0
        assert len(argvs) == 2 + cal.N_REPEATS + 1
        for argv in argvs:
            assert argv[argv.index("--f-prefix") + 1] == "F80"

    def test_the_rest_stage_with_the_wrong_prefix_drives_nothing(self, tmp_path, monkeypatch):
        write_decision(tmp_path, {"decision": "FINAL_A100_80GB"})

        def must_not_drive(*a, **k):
            raise AssertionError("drove a run with the wrong F prefix")

        monkeypatch.setattr(cal, "drive_run", must_not_drive)
        with pytest.raises(common.CalibrationStopped):
            cal._cmd_drive(drive_args(tmp_path, "F"))

    def test_the_cli_has_no_silent_default_for_the_rest_stage(self, tmp_path, monkeypatch):
        def must_not_drive(*a, **k):
            raise AssertionError("drove a run without a decision")

        monkeypatch.setattr(cal, "drive_run", must_not_drive)
        with pytest.raises(common.CalibrationStopped):
            cal.main(["drive", "--stage", "rest", "--out", str(tmp_path), "--data-root", "d",
                      "--split-dir", "s", "--manifest", "m"])

    def test_the_f_stage_needs_an_explicit_prefix(self, tmp_path, monkeypatch):
        def must_not_drive(*a, **k):
            raise AssertionError("drove an F run without a prefix")

        monkeypatch.setattr(cal, "drive_run", must_not_drive)
        with pytest.raises(SystemExit):
            cal.main(["drive", "--stage", "F", "--out", str(tmp_path), "--data-root", "d",
                      "--split-dir", "s", "--manifest", "m"])

    def test_a_single_run_of_n_or_p_with_the_wrong_prefix_is_refused(self, tmp_path):
        write_decision(tmp_path, {"decision": "FINAL_A100_80GB"})
        args = drive_args(tmp_path, "F", stage="N")
        args.arm, args.repeat, args.run_id = None, 1, None
        with pytest.raises(common.CalibrationStopped):
            cal._cmd_run(args)

    def test_the_report_uses_the_derived_prefix(self, tmp_path, monkeypatch):
        write_decision(tmp_path, {"decision": "FINAL_A100_80GB"})
        seen = []
        monkeypatch.setattr(cal, "assemble_report",
                            lambda out, prefix="F": seen.append(prefix) or {"voids": [],
                                                                          "provenance_gaps": []})
        monkeypatch.setattr(cal, "write_sha256sums", lambda out: 0)
        cal._cmd_report(drive_args(tmp_path, None))
        assert seen == ["F80"]


# --- 8 review: the model runs on the requested GPU --------------------------------------------


class TestDeviceFollowsGpuIndex:
    @pytest.mark.parametrize("device,index,want", [
        ("cuda", 0, "cuda:0"), ("cuda", 1, "cuda:1"), ("cuda:1", 1, "cuda:1"),
        ("cpu", 0, "cpu")])
    def test_a_bare_cuda_is_the_requested_index(self, device, index, want):
        assert cal.resolve_device(device, index) == want

    @pytest.mark.parametrize("device,index", [("cuda:0", 1), ("cuda:1", 0), ("cpu", 1),
                                              ("mps", 0)])
    def test_a_device_that_disagrees_with_the_index_is_rejected(self, device, index):
        with pytest.raises(ValueError):
            cal.resolve_device(device, index)

    def test_run_gives_the_body_the_mapped_device_and_selects_it(self, tmp_path, monkeypatch):
        import transformers

        write_decision(tmp_path, {"decision": "FINAL_A100_40GB", "final_instance": INSTANCE})
        (tmp_path / "manifest.json").write_text("{}")
        seen = {}
        selected = []
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                            lambda *a, **k: object())
        monkeypatch.setattr(torch.cuda, "set_device", lambda index: selected.append(index))

        def fake_execute(journal, run_id, kind, out_dir, body, **kw):
            seen["kwargs"] = kw
            body()
            return {"outcome": common.OUTCOME_OK, "void_reasons": []}

        def fake_body(args, run_id, tokenizer, data_root, device, out, hint):
            seen["device"] = device

        monkeypatch.setattr(cal, "execute_attempt", fake_execute)
        monkeypatch.setattr(cal, "_body", fake_body)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        args = drive_args(tmp_path, None, stage="W", device="cuda", gpu_index=0)
        args.arm, args.repeat, args.run_id = None, 1, None
        assert cal._cmd_run(args) == 0
        assert seen["device"] == "cuda:0"
        assert seen["kwargs"]["gpu_index"] == 0 and seen["kwargs"]["require_gpu"] is True
        assert selected == [0]

    def test_a_model_on_the_wrong_device_is_refused(self):
        model = tiny_model(seed=4, dtype=torch.bfloat16)
        assert cal.assert_on_device(model, "cpu") == "cpu"
        with pytest.raises(RuntimeError, match="cuda:1"):
            cal.assert_on_device(model, "cuda:1")

    def test_the_device_actually_used_is_recorded(self, tmp_path, tiny_world):
        directory = write_legacy_checkpoint(tiny_model(seed=4, dtype=torch.bfloat16),
                                            tmp_path / "m")
        res = cal.run_f(directory, "A_bf16", synthetic_examples(12), synthetic_examples(4, 9),
                        SMALL, "cpu", 1, None, tmp_path / "adapter",
                        experts_implementation="eager")
        assert res["device_used"] == "cpu"
        assert run_w_on_tiny(tiny_world)["device_used"] == ["cpu"]
        rows = [r["input_ids"] for r in synthetic_examples(40, seed=3)]
        bf16 = cal.load_base(tiny_world["source"], "cpu", "eager")
        common.set_fp32_reference_mode()
        out = cal.run_t(quant.enumerate_units(2, E), cal.UnitFiles(tiny_world["data"]), bf16,
                        rows, SMALL, "cpu", None, per_kind=1, calls_per_unit=1, min_units=1)
        assert out["device_used"] == "cpu"


# --- 9 review: one GPU per run ----------------------------------------------------------------


class TestOneGpuPerRun:
    """arm_validity names the workload GPU from CUDA_VISIBLE_DEVICES and takes no index, so the
    harness runs on exactly one visible GPU, addressed as index 0."""

    def pick(self, index=0, visible="unset", count=None):
        environ = {} if visible == "unset" else {"CUDA_VISIBLE_DEVICES": visible}
        return common.require_single_gpu(index, environ=environ, count_gpus=lambda: count)

    @pytest.mark.parametrize("visible,want", [("2", "2"), ("GPU-0a1b", "GPU-0a1b")])
    def test_one_visible_gpu_is_the_physical_gpu(self, visible, want):
        assert self.pick(visible=visible) == want

    def test_an_unset_variable_is_fine_on_a_one_gpu_machine_only(self):
        assert self.pick(count=1) == "0"
        for count in (2, 8, None):
            with pytest.raises(common.GpuSelectionError, match="exactly one"):
                self.pick(count=count)

    @pytest.mark.parametrize("visible", ["0,1", "GPU-a,GPU-b", "", "NoDevFiles", "-1"])
    def test_a_variable_that_exposes_none_or_several_gpus_is_an_error(self, visible):
        with pytest.raises(common.GpuSelectionError, match="exactly one"):
            self.pick(visible=visible)

    @pytest.mark.parametrize("index", [1, 2])
    def test_a_gpu_index_other_than_zero_is_an_error(self, index):
        with pytest.raises(common.GpuSelectionError, match="--gpu-index 0"):
            self.pick(index=index, visible="1")

    def test_run_refuses_before_starting_an_attempt(self, tmp_path, monkeypatch):
        import transformers

        write_decision(tmp_path, {"decision": "FINAL_A100_40GB", "final_instance": INSTANCE})
        (tmp_path / "manifest.json").write_text("{}")
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                            lambda *a, **k: object())

        def must_not_run(*a, **k):
            raise AssertionError("an attempt started on an unusable GPU selection")

        monkeypatch.setattr(cal, "execute_attempt", must_not_run)
        for index, visible in ((1, "1"), (0, "0,1")):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
            args = drive_args(tmp_path, None, stage="W", device="cuda", gpu_index=index)
            args.arm, args.repeat, args.run_id = None, 1, None
            with pytest.raises(common.GpuSelectionError):
                cal._cmd_run(args)

    def test_drive_refuses_before_any_child_starts(self, tmp_path, monkeypatch):
        def must_not_drive(*a, **k):
            raise AssertionError("drove a child on an unusable GPU selection")

        monkeypatch.setattr(cal, "drive_run", must_not_drive)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        write_decision(tmp_path, {"decision": "FINAL_A100_40GB", "final_instance": INSTANCE})
        with pytest.raises(common.GpuSelectionError):
            cal._cmd_drive(drive_args(tmp_path, None, stage="rest", device="cuda"))
        with pytest.raises(common.GpuSelectionError):
            cal._cmd_drive(drive_args(tmp_path, "F", stage="F", device="cuda"))

    def test_the_peers_selftest_checks_the_selection_too(self, monkeypatch):
        import types

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        with pytest.raises(common.GpuSelectionError):
            cal._cmd_peers_selftest(types.SimpleNamespace(gpu_index=0))

    def test_the_box_stamp_queries_the_workload_gpu_not_the_index(self, monkeypatch):
        import types

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        seen = []

        def fake_run(argv, **kw):
            seen.append(argv)
            return types.SimpleNamespace(returncode=0, stdout="x\n", stderr="")

        monkeypatch.setattr(common.subprocess, "run", fake_run)
        sensors = {name: (lambda: None) for name in (
            "ram_available_bytes", "commit_charge", "gpu_processes", "gpu_clock",
            "gpu_memory_used_mib")}
        common.box_stamp(0, "42", readers=sensors)
        assert seen and all("--id=2" in argv for argv in seen), seen

    @pytest.mark.parametrize("gpu,method,arg", [("2", "ByIndex", 2),
                                                 ("GPU-0a1b", "ByUUID", "GPU-0a1b")])
    def test_nvml_sampling_reads_the_workload_gpu(self, monkeypatch, gpu, method, arg):
        import sys
        import types

        calls = []
        fake = types.SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda i: calls.append(("ByIndex", i)) or "h",
            nvmlDeviceGetHandleByUUID=lambda u: calls.append(("ByUUID", u)) or "h",
            nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(used=1, free=2))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
        assert cal.nvml_reader()() == {"used": 1, "free": 2}
        assert calls == [(method, arg)]


# --- 10 review: the input check is bound to the pinned manifest -------------------------------


class TestInputsBoundToThePinnedManifest:
    def instance(self, tmp_path, monkeypatch):
        from tests.test_e4_part2_review_fixes import fake_download, synthetic_instance
        import types
        import e4_part2_transfer as tr

        root, manifest = synthetic_instance(tmp_path)
        fake_download(monkeypatch)
        tr.pull(root, "r", types.SimpleNamespace(whoami=lambda: {"name": "me"}),
                log=lambda *_: None)
        return root, manifest, common.sha256_file(root / tr.MANIFEST_NAME)

    def test_the_pin_is_the_manifest_of_the_rule(self):
        assert common.QUANT_MANIFEST_SHA256 == (
            "1f3b550d707aa18c098851a1718f1712cdb1162adc28552c5fbbcce8611f0ccb")
        committed = (common.REPO_ROOT / "benchmarks" / "results" / "e4-part2-quant"
                     / "quant-manifest.json")
        assert common.sha256_file(committed) == common.QUANT_MANIFEST_SHA256

    def test_a_supplied_manifest_that_matches_record_and_pin_passes(self, tmp_path, monkeypatch):
        root, manifest, sha = self.instance(tmp_path, monkeypatch)
        assert cal.run_input_check(root, manifest, "F", "A_off", sha, pinned_sha256=sha) == []

    def test_a_supplied_manifest_other_than_the_pin_voids_the_run(self, tmp_path, monkeypatch):
        root, manifest, sha = self.instance(tmp_path, monkeypatch)
        reasons = cal.run_input_check(root, manifest, "F", "A_off", sha, pinned_sha256="0" * 64)
        assert any("pinned" in r for r in reasons)

    def test_a_supplied_manifest_other_than_the_record_voids_the_run(self, tmp_path,
                                                                     monkeypatch):
        root, manifest, sha = self.instance(tmp_path, monkeypatch)
        other = "e" * 64
        reasons = cal.run_input_check(root, manifest, "F", "A_off", other, pinned_sha256=other)
        assert any("verification record" in r for r in reasons)

    def test_a_manifest_outside_the_data_root_cannot_pick_the_files_to_hash(self, tmp_path,
                                                                            monkeypatch):
        root, manifest, sha = self.instance(tmp_path, monkeypatch)
        (root / "writeback-dqoff" / "config.json").write_bytes(b"tampered")
        thin = dict(manifest, files={}, source_files={})  # a manifest that names nothing
        thin_sha = common.sha256_bytes(json.dumps(thin).encode())
        reasons = cal.run_input_check(root, thin, "F", "A_off", thin_sha, pinned_sha256=sha)
        assert reasons  # not the pinned manifest: void, whatever files it names

    def test_run_hashes_the_bytes_it_parsed(self, tmp_path, monkeypatch):
        import transformers

        write_decision(tmp_path, {"decision": "FINAL_A100_40GB", "final_instance": INSTANCE})
        (tmp_path / "manifest.json").write_text('{"files": {}, "source_files": {}}')
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                            lambda *a, **k: object())
        seen = {}

        def fake_execute(journal, run_id, kind, out_dir, body, input_check=None, **kw):
            seen["reasons"] = input_check()
            return {"outcome": common.OUTCOME_VOID, "void_reasons": seen["reasons"]}

        monkeypatch.setattr(cal, "execute_attempt", fake_execute)
        args = drive_args(tmp_path, None, stage="W", device="cpu")
        args.arm, args.repeat, args.run_id = None, 1, None
        cal._cmd_run(args)
        assert any("pinned" in r for r in seen["reasons"])


# --- 11 review: rest stages are bound to the instance of the chosen F -------------------------


def f_stamp(uuid="GPU-chosen", instance="777"):
    return {"unix_s": 1.0, "gpu_uuid": uuid, "instance_id": instance, "gpu_processes": [],
            "unknown": {}}


class TestRestStagesStayOnTheChosenInstance:
    def write_f(self, out, prefix, stamps, reserved=1):
        journal = common.AttemptsJournal(out / "attempts.jsonl")
        for arm, stamp in zip(cal.ARMS, stamps):
            cal.execute_attempt(
                journal, f"{prefix}_{arm}", "F", out / "runs",
                lambda: {"oom": False, "failed": False, "max_memory_reserved_bytes": reserved},
                stamp=lambda *_a, s=stamp: dict(s), prov=lambda: GOOD_PROV,
                identify_own=lambda: {"own_pids": [], "method": "stub"})

    def test_decision_records_the_gpu_and_instance_of_the_chosen_f(self, tmp_path):
        self.write_f(tmp_path, "F", [f_stamp()] * 3)
        assert cal.main(["decide", "--out", str(tmp_path)]) == 0
        decision = json.loads((tmp_path / "decision.json").read_text())
        assert decision["decision"] == "FINAL_A100_40GB"
        assert decision["final_instance"] == INSTANCE

    def test_the_80gb_decision_records_the_f80_instance(self, tmp_path):
        self.write_f(tmp_path, "F", [f_stamp("GPU-40", "1")] * 3, reserved=39e9)
        self.write_f(tmp_path, "F80", [f_stamp("GPU-80", "2")] * 3, reserved=39e9)
        (tmp_path / "x").mkdir()
        assert cal.main(["decide", "--out", str(tmp_path)]) == 0
        decision = json.loads((tmp_path / "decision.json").read_text())
        assert decision["decision"] == "FINAL_A100_80GB"
        assert decision["final_instance"] == {"gpu_uuid": "GPU-80", "instance_id": "2"}

    def test_f_arms_on_different_instances_stop_calibration(self, tmp_path):
        self.write_f(tmp_path, "F", [f_stamp(), f_stamp(instance="778"), f_stamp()])
        with pytest.raises(common.CalibrationStopped, match="different"):
            cal.final_instance(tmp_path, "F")

    def test_a_decision_without_an_instance_stops_the_rest_stages(self, tmp_path):
        write_decision(tmp_path, {"decision": "FINAL_A100_40GB"})
        with pytest.raises(common.CalibrationStopped, match="final_instance"):
            cal.resolve_final_instance(tmp_path)

    def test_the_rest_stage_reads_the_instance_from_the_decision(self, tmp_path):
        write_decision(tmp_path, {"decision": "FINAL_A100_80GB", "final_instance": INSTANCE})
        assert cal.resolve_final_instance(tmp_path) == INSTANCE

    def run(self, tmp_path, before, after, expected=INSTANCE):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        stamps = iter([before, after])
        return cal.execute_attempt(
            journal, "T", "T", tmp_path / "runs", lambda: {"x": 1},
            stamp=lambda *_a: next(stamps), prov=lambda: GOOD_PROV,
            identify_own=lambda: {"own_pids": [], "method": "stub"}, expected_instance=expected)

    def test_the_same_instance_is_ok(self, tmp_path):
        assert self.run(tmp_path, f_stamp(), f_stamp())["outcome"] == "ok"

    @pytest.mark.parametrize("before,after,word", [
        (f_stamp(instance="778"), f_stamp(), "instance_id"),
        (f_stamp(uuid="GPU-other"), f_stamp(), "gpu_uuid"),
        (f_stamp(), f_stamp(instance="778"), "instance_id")])
    def test_a_run_on_another_instance_or_gpu_is_void_and_keeps_its_record(
            self, tmp_path, before, after, word):
        record = self.run(tmp_path, before, after)
        assert record["outcome"] == "void"
        assert any("chosen F" in r and word in r for r in record["void_reasons"])
        assert (tmp_path / "runs" / f"{record['attempt_id']}.json").exists()

    def test_run_passes_the_decided_instance_to_the_attempt(self, tmp_path, monkeypatch):
        import transformers

        write_decision(tmp_path, {"decision": "FINAL_A100_40GB", "final_instance": INSTANCE})
        (tmp_path / "manifest.json").write_text("{}")
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                            lambda *a, **k: object())
        seen = {}

        def fake_execute(journal, run_id, kind, out_dir, body, **kw):
            seen.update(kw)
            return {"outcome": common.OUTCOME_OK, "void_reasons": []}

        monkeypatch.setattr(cal, "execute_attempt", fake_execute)
        args = drive_args(tmp_path, None, stage="T", device="cpu")
        args.arm, args.repeat, args.run_id = None, 1, None
        cal._cmd_run(args)
        assert seen["expected_instance"] == INSTANCE

    def test_the_f_stage_has_no_instance_to_match(self, tmp_path, monkeypatch):
        import transformers

        (tmp_path / "manifest.json").write_text("{}")
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                            lambda *a, **k: object())
        seen = {}

        def fake_execute(journal, run_id, kind, out_dir, body, **kw):
            seen.update(kw)
            return {"outcome": common.OUTCOME_OK, "void_reasons": []}

        monkeypatch.setattr(cal, "execute_attempt", fake_execute)
        args = drive_args(tmp_path, "F", stage="F", device="cpu")
        args.arm, args.repeat, args.run_id = "A_off", 1, None
        cal._cmd_run(args)
        assert seen["expected_instance"] is None


