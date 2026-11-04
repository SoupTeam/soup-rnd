"""E4 part 2 calibration harness on CPU with a tiny random OLMoE.

Covers: the W reference factor layout against peft's own, label masks through the trainer,
fp32 setting asserts, the attempts journal and void behaviour, the final-card rule, tau/floor
arithmetic, the gemm_4bit probe (against a stub of the CUDA dispatch), and a CPU end-to-end of
T and W on the Mac-side quantization of the tiny model.
"""

import json
import types

import pytest

from tests.e4_part2_support import tiny_model, write_legacy_checkpoint

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")

import e4_part2_calibration as cal  # noqa: E402
import e4_part2_common as _common  # noqa: E402
import e4_part2_common as common  # noqa: E402
import e4_part2_gemm_probe as gp  # noqa: E402
import e4_part2_quant as quant  # noqa: E402
import e4_part2_wref as wref  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(_common.library_versions()["mismatches"]),
    reason="the E4 part 2 harness is pinned to torch/transformers/bitsandbytes/peft of the rule",
)

E = 4  # experts in the tiny model


@pytest.fixture(autouse=True)
def _own_gpu_context(monkeypatch):
    """execute_attempt identifies its own GPU context; there is no GPU here."""
    import os

    monkeypatch.setattr(common, "identify_own_gpu_pids_cuda", lambda gpu_index=0, **kw: {
        "own_pids": [os.getpid()], "method": "stub"})


def peft_wrapped(model, r=16):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))


@pytest.fixture(scope="module")
def wrapped():
    return peft_wrapped(tiny_model(seed=1, dtype=torch.bfloat16))


class TestWReferenceLayoutMatchesPeft:
    def _wrappers(self, peft_model):
        from peft.tuners.lora.layer import ParamWrapper

        found = {}
        for name, module in peft_model.named_modules():
            if isinstance(module, ParamWrapper):
                found.setdefault(name, module)
        return found

    def test_the_tiny_model_gets_one_wrapper_per_fused_parameter_and_layer(self, wrapped):
        names = list(self._wrappers(wrapped))
        assert len(names) == 4  # 2 layers x (gate_up_proj, down_proj)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_factors_and_folded_weight_equal_peft_get_delta_factors(self, wrapped, seed):
        gen = torch.Generator().manual_seed(seed)
        for name, wrapper in self._wrappers(wrapped).items():
            param = wrapper.get_param()
            a, b = wref.init_factors(param.shape[2], param.shape[1], E, gen)
            wrapper.lora_A["default"].weight.data.copy_(a)
            wrapper.lora_B["default"].weight.data.copy_(b)
            lhs_p, rhs_p, scaling = wrapper.get_delta_factors("default")
            lhs, rhs = wref.peft_factor_layout(a, b, E)
            assert torch.equal(lhs.to(torch.bfloat16), lhs_p), name
            assert torch.equal(rhs.to(torch.bfloat16), rhs_p), name
            assert scaling == wref.SCALING
            mine = wref.effective_weight(param.data, a, b, E, bf16_factors=True)
            with torch.autocast(device_type="cpu", enabled=False):
                theirs = torch.baddbmm(param.data, lhs_p, rhs_p, alpha=scaling)
            assert torch.equal(mine, theirs), name

    def test_layout_is_experts_out_in_for_both_fused_parameters(self, wrapped):
        for name, wrapper in self._wrappers(wrapped).items():
            param = wrapper.get_param()
            lhs, rhs, _ = wrapper.get_delta_factors("default")
            assert lhs.shape == (E, param.shape[1], 16), name
            assert rhs.shape == (E, 16, param.shape[2]), name

    def test_peft_default_a_init_has_the_shape_and_bound_of_the_reference_init(self, wrapped):
        wrapper = next(iter(self._wrappers(wrapped).values()))
        a_peft = wrapper.lora_A["default"].weight
        a_ref, b_ref = wref.init_factors(a_peft.shape[1], wrapper.lora_B["default"].weight.shape[
            0], E, torch.Generator().manual_seed(0))
        assert a_ref.shape == a_peft.shape
        bound = 1 / (a_peft.shape[1] ** 0.5)  # kaiming_uniform(a=sqrt 5) on fan_in
        assert float(a_ref.abs().max()) <= bound + 1e-6
        assert float(a_peft.detach().abs().max()) <= bound + 1e-6
        assert b_ref.shape == wrapper.lora_B["default"].weight.shape
        assert abs(float(b_ref.std()) - wref.B_STD) < 0.002

    def test_reference_output_is_bit_identical_to_peft_experts_on_the_same_factors(self):
        torch.manual_seed(0)
        model = tiny_model(seed=2, dtype=torch.bfloat16)
        experts = model.model.layers[0].mlp.experts
        hidden = torch.randn(7, 64).to(torch.bfloat16)
        idx = torch.stack([torch.randperm(E)[:2] for _ in range(7)])
        w = torch.softmax(torch.randn(7, 2), -1).to(torch.bfloat16)
        base = {"gate_up_proj": experts.gate_up_proj.data.clone(),
                "down_proj": experts.down_proj.data.clone()}
        state = wref.make_adapter_state(experts, torch.Generator().manual_seed(5))
        ref = wref.run_reference(experts, base, {"hidden": hidden, "top_k_index": idx,
                                                 "top_k_weights": w}, state,
                                 torch.ones(7, 64), "bf16", cal.RECIPE.adamw_kwargs())
        peft_model = peft_wrapped(model)
        wrappers = {n: m for n, m in peft_model.named_modules()
                    if type(m).__name__ == "ParamWrapper" and ".layers.0." in n}
        for n, m in wrappers.items():
            key = "gate_up_proj" if m.parameter_name == "gate_up_proj" else "down_proj"
            m.lora_A["default"].weight.data.copy_(state[key]["A"])
            m.lora_B["default"].weight.data.copy_(state[key]["B"])
        peft_experts = peft_model.base_model.model.model.layers[0].mlp.experts
        with torch.no_grad():
            theirs = peft_experts(hidden, idx, w)
        assert torch.equal(ref["output"], theirs)

    def test_bf16_and_fp32_references_differ_and_tol_w_doubles_the_max(self):
        per_call = [{q: v for q in wref.W_QUANTITIES} for v in (0.1, 0.3, 0.2)]
        assert wref.tol_w(per_call) == {q: 0.6 for q in wref.W_QUANTITIES}

    def test_frozen_base_is_untouched_by_the_reference_step(self):
        torch.manual_seed(0)
        model = tiny_model(seed=3, dtype=torch.bfloat16)
        experts = model.model.layers[0].mlp.experts
        before = cal.hash_base(experts)
        hidden = torch.randn(5, 64).to(torch.bfloat16)
        idx = torch.stack([torch.randperm(E)[:2] for _ in range(5)])
        routed = {"hidden": hidden, "top_k_index": idx,
                  "top_k_weights": torch.full((5, 2), 0.5, dtype=torch.bfloat16)}
        base = {"gate_up_proj": experts.gate_up_proj.data, "down_proj": experts.down_proj.data}
        state = wref.make_adapter_state(experts, torch.Generator().manual_seed(5))
        common.set_fp32_reference_mode()
        for precision in ("bf16", "fp32"):
            wref.run_reference(experts, base, routed, state, torch.ones(5, 64), precision,
                               cal.RECIPE.adamw_kwargs())
        assert cal.hash_base(experts) == before

    def test_fp32_reference_asserts_its_output_is_fp32(self):
        torch.manual_seed(0)
        model = tiny_model(seed=3, dtype=torch.bfloat16)
        experts = model.model.layers[0].mlp.experts
        routed = {"hidden": torch.randn(5, 64).to(torch.bfloat16),
                  "top_k_index": torch.stack([torch.randperm(E)[:2] for _ in range(5)]),
                  "top_k_weights": torch.full((5, 2), 0.5, dtype=torch.bfloat16)}
        base = {"gate_up_proj": experts.gate_up_proj.data, "down_proj": experts.down_proj.data}
        state = wref.make_adapter_state(experts, torch.Generator().manual_seed(5))
        common.set_fp32_reference_mode()
        out = wref.run_reference(experts, base, routed, state, torch.ones(5, 64), "fp32",
                                 cal.RECIPE.adamw_kwargs())
        assert out["output"].dtype == torch.float32
        assert out["gate_up_proj.A_grad"].dtype == torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            with pytest.raises(common.Fp32SettingError, match="allow_tf32"):
                wref.run_reference(experts, base, routed, state, torch.ones(5, 64), "fp32",
                                   cal.RECIPE.adamw_kwargs())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False


class TestFp32Settings:
    def test_set_then_assert_passes_and_records_flags(self):
        flags = common.set_fp32_reference_mode()
        assert flags["float32_matmul_precision"] == "highest"
        assert not flags["cuda_matmul_allow_tf32"] and not flags["cudnn_allow_tf32"]
        got = common.assert_fp32_reference(torch.zeros(2), where="t")
        assert got == flags

    def test_tf32_on_cudnn_raises(self):
        common.set_fp32_reference_mode()
        torch.backends.cudnn.allow_tf32 = True
        try:
            with pytest.raises(common.Fp32SettingError, match="cudnn"):
                common.assert_fp32_reference(torch.zeros(2))
        finally:
            torch.backends.cudnn.allow_tf32 = False

    def test_precision_not_highest_raises(self):
        common.set_fp32_reference_mode()
        torch.set_float32_matmul_precision("high")
        try:
            with pytest.raises(common.Fp32SettingError, match="highest"):
                common.assert_fp32_reference(torch.zeros(2))
        finally:
            torch.set_float32_matmul_precision("highest")

    def test_autocast_enabled_raises(self):
        common.set_fp32_reference_mode()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with pytest.raises(common.Fp32SettingError, match="autocast"):
                common.assert_fp32_reference(torch.zeros(2))

    def test_bf16_output_raises(self):
        common.set_fp32_reference_mode()
        with pytest.raises(common.Fp32SettingError, match="bfloat16"):
            common.assert_fp32_reference(torch.zeros(2, dtype=torch.bfloat16))


class TestJournal:
    def j(self, tmp_path):
        return common.AttemptsJournal(tmp_path / "attempts.jsonl")

    def test_start_end_ok_is_recorded_and_not_rerunnable(self, tmp_path):
        j = self.j(tmp_path)
        a = j.start("F_A_on", "F")
        j.end(a, common.OUTCOME_OK)
        assert [x["outcome"] for x in j.attempts("F_A_on")] == ["ok"]
        with pytest.raises(RuntimeError, match="already has a completed"):
            j.start("F_A_on", "F")

    def test_a_void_run_is_rerun_once_and_keeps_its_record(self, tmp_path):
        j = self.j(tmp_path)
        a = j.start("T", "T")
        j.end(a, common.OUTCOME_VOID, ["peers: foreign GPU process"])
        b = j.start("T", "T")
        j.end(b, common.OUTCOME_OK)
        outcomes = [x["outcome"] for x in j.attempts("T")]
        assert outcomes == ["void", "ok"]
        assert j.attempts("T")[0]["reasons"] == ["peers: foreign GPU process"]

    def test_a_second_void_stops_calibration(self, tmp_path):
        j = self.j(tmp_path)
        for _ in range(2):
            a = j.start("W", "W")
            j.end(a, common.OUTCOME_VOID, ["fp32 settings"])
        with pytest.raises(common.CalibrationStopped, match="W"):
            j.start("W", "W")

    def test_an_attempt_with_no_end_becomes_a_void_on_the_next_start(self, tmp_path):
        j = self.j(tmp_path)
        j.start("N_1", "N")  # the process died here
        j.start("N_1", "N")
        first, second = j.attempts("N_1")
        assert first["outcome"] == "void" and "interrupted" in first["reasons"][0]
        assert second["outcome"] is None

    def test_two_interruptions_stop_the_run(self, tmp_path):
        j = self.j(tmp_path)
        j.start("P", "P")
        j.start("P", "P")
        with pytest.raises(common.CalibrationStopped):
            j.start("P", "P")

    def test_the_journal_is_append_only_jsonl(self, tmp_path):
        j = self.j(tmp_path)
        a = j.start("x", "F")
        j.end(a, common.OUTCOME_OK)
        lines = (tmp_path / "attempts.jsonl").read_text().splitlines()
        assert [json.loads(line)["event"] for line in lines] == ["start", "end"]

    def test_a_void_needs_a_reason_and_outcomes_are_checked(self, tmp_path):
        j = self.j(tmp_path)
        a = j.start("x", "F")
        with pytest.raises(ValueError):
            j.end(a, common.OUTCOME_VOID, [])
        with pytest.raises(ValueError):
            j.end(a, "maybe")

    def test_provenance_gap_lists_runs_without_a_journal_entry_or_an_end(self, tmp_path):
        j = self.j(tmp_path)
        a = j.start("F_A_off", "F")
        j.end(a, common.OUTCOME_OK)
        j.start("F_A_on", "F")
        gaps = j.provenance_gap(["F_A_off", "F_A_on", "F_A_bf16"])
        assert gaps == ["F_A_on: attempt without an end", "F_A_bf16: no journal entry"]


GOOD_STAMP = {"unix_s": 1.0, "gpu_uuid": "GPU-1", "instance_id": "123",
              "gpu_processes": [], "unknown": {}}


def stamp_factory(before, after):
    calls = iter([before, after])

    def stamp(_gpu_index, _instance):
        return next(calls)

    return stamp


GOOD_PROV = {"soup_commit": "a" * 40, "harness_commit": "b" * 40, "harness_tree": "c" * 40,
             "harness_dirty": False}


class TestExecuteAttempt:
    def run(self, tmp_path, body, before=GOOD_STAMP, after=GOOD_STAMP, prov=GOOD_PROV, **kw):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        record = cal.execute_attempt(journal, "T", "T", tmp_path / "runs", body,
                                     stamp=stamp_factory(before, after), prov=lambda: prov,
                                     **kw)
        return journal, record

    def test_clean_run_is_ok_and_its_record_is_on_disk(self, tmp_path):
        journal, rec = self.run(tmp_path, lambda: {"x": 1})
        assert rec["outcome"] == "ok" and rec["result"] == {"x": 1}
        assert (tmp_path / "runs" / f"{rec['attempt_id']}.json").exists()
        assert rec["box_stamp_before"]["unix_s"] == 1.0
        assert journal.attempts("T")[0]["outcome"] == "ok"

    def test_a_foreign_gpu_process_voids_the_run_but_keeps_the_record(self, tmp_path):
        peer = dict(GOOD_STAMP, gpu_processes=[{"pid": 999999, "type": "C", "name": "python"}])
        journal, rec = self.run(tmp_path, lambda: {"x": 1}, after=peer)
        assert rec["outcome"] == "void"
        assert any(r.startswith("peers:") for r in rec["void_reasons"])
        assert (tmp_path / "runs" / f"{rec['attempt_id']}.json").exists()
        assert journal.void_count("T") == 1

    def test_own_process_on_the_gpu_is_not_a_peer(self, tmp_path):
        import os

        own = dict(GOOD_STAMP, gpu_processes=[{"pid": os.getpid(), "type": "C", "name": "py"}])
        _, rec = self.run(tmp_path, lambda: {}, before=own, after=own)
        assert rec["outcome"] == "ok"

    def test_dirty_harness_is_a_provenance_void_and_the_body_never_runs(self, tmp_path):
        ran = []
        _, rec = self.run(tmp_path, lambda: ran.append(1) or {},
                          prov=dict(GOOD_PROV, harness_dirty=True))
        assert rec["outcome"] == "void" and not ran
        assert "uncommitted" in rec["void_reasons"][0]

    def test_missing_gpu_uuid_or_instance_id_is_a_provenance_void(self, tmp_path):
        bad = dict(GOOD_STAMP, gpu_uuid=None, instance_id=None)
        _, rec = self.run(tmp_path, lambda: {}, before=bad)
        assert rec["outcome"] == "void"
        assert sum("missing" in r for r in rec["void_reasons"]) == 2

    def test_unavailable_process_list_is_a_void_not_a_pass(self, tmp_path):
        blind = dict(GOOD_STAMP, gpu_processes=None)
        _, rec = self.run(tmp_path, lambda: {}, before=blind)
        assert rec["outcome"] == "void"

    def test_input_hash_mismatch_voids(self, tmp_path):
        _, rec = self.run(tmp_path, lambda: {}, input_check=lambda: ["inputs: x differs"])
        assert rec["outcome"] == "void" and rec["void_reasons"] == ["inputs: x differs"]

    def test_body_void_reasons_are_applied(self, tmp_path):
        _, rec = self.run(tmp_path, lambda: {"void_reasons": ["fp32 settings: autocast on"]})
        assert rec["outcome"] == "void"

    def test_failed_result_is_failed_not_void(self, tmp_path):
        journal, rec = self.run(tmp_path, lambda: {"failed": True})
        assert rec["outcome"] == "failed"
        assert journal.void_count("T") == 0

    def test_a_crash_is_journaled_as_a_void_with_its_record_and_reraised(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")

        def boom():
            raise MemoryError("out of host memory")

        with pytest.raises(MemoryError):
            cal.execute_attempt(journal, "T", "T", tmp_path / "runs", boom,
                                stamp=stamp_factory(GOOD_STAMP, GOOD_STAMP),
                                prov=lambda: GOOD_PROV)
        [attempt] = journal.attempts("T")
        assert attempt["outcome"] == "void" and "MemoryError" in attempt["reasons"][0]
        [record_file] = (tmp_path / "runs").glob("*.json")
        assert "crash" in json.loads(record_file.read_text())

    def test_two_voids_of_the_same_run_stop_the_next_attempt(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        for _ in range(2):
            cal.execute_attempt(journal, "T", "T", tmp_path / "runs", lambda: {},
                                stamp=stamp_factory(GOOD_STAMP, GOOD_STAMP),
                                prov=lambda: dict(GOOD_PROV, harness_dirty=True))
        with pytest.raises(common.CalibrationStopped):
            cal.execute_attempt(journal, "T", "T", tmp_path / "runs", lambda: {},
                                stamp=stamp_factory(GOOD_STAMP, GOOD_STAMP),
                                prov=lambda: GOOD_PROV)

    def test_cpu_runs_do_not_require_a_gpu_uuid(self, tmp_path):
        cpu = {"unix_s": 1.0, "gpu_uuid": None, "instance_id": None, "gpu_processes": None,
               "unknown": {}}
        _, rec = self.run(tmp_path, lambda: {}, before=cpu, after=cpu, require_gpu=False)
        assert rec["outcome"] == "ok"


class TestFinalCard:
    def ok(self, reserved):
        return {"oom": False, "failed": False, "max_memory_reserved_bytes": reserved}

    def test_all_arms_under_36_gb_choose_the_40_gb_card(self):
        res = cal.decide_final_card({a: self.ok(20e9) for a in cal.ARMS})
        assert res["decision"] == "FINAL_A100_40GB"

    def test_exactly_36_gb_is_still_40_gb_but_one_byte_over_is_not(self):
        at = {a: self.ok(36_000_000_000) for a in cal.ARMS}
        assert cal.decide_final_card(at)["decision"] == "FINAL_A100_40GB"
        over = dict(at, A_on=self.ok(36_000_000_001))
        assert cal.decide_final_card(over)["decision"] == "REPEAT_F_ON_A100_80GB"

    def test_an_oom_arm_sends_f_to_the_80_gb_card(self):
        f40 = {"A_off": self.ok(20e9), "A_on": {"oom": True}, "A_bf16": self.ok(20e9)}
        assert cal.decide_final_card(f40)["decision"] == "REPEAT_F_ON_A100_80GB"
        f80 = {a: self.ok(45e9) for a in cal.ARMS}
        assert cal.decide_final_card(f40, f80)["decision"] == "FINAL_A100_80GB"

    def test_oom_on_the_80_gb_card_stops_and_goes_to_the_owner(self):
        f40 = {a: {"oom": True} for a in cal.ARMS}
        f80 = {a: {"oom": True} for a in cal.ARMS}
        assert cal.decide_final_card(f40, f80)["decision"] == "STOP_RECIPE_TO_OWNER"

    def test_a_non_finite_loss_fails_f_on_either_card(self):
        bad = {a: {"oom": False, "failed": a == "A_off", "max_memory_reserved_bytes": 1}
               for a in cal.ARMS}
        assert cal.decide_final_card(bad)["decision"] == "F_FAILS"
        f40 = {a: {"oom": True} for a in cal.ARMS}
        assert cal.decide_final_card(f40, bad)["decision"] == "F_FAILS"


class TestArithmetic:
    def test_tau_is_twice_the_max_over_calls_and_both_settings_of_both_paths(self):
        recs = [
            {"cell": "q_proj/needs_grad=False/kernel=fused", "quantity": "output",
             "e_F": 0.1, "e_R": 0.3, "dq": "off"},
            {"cell": "q_proj/needs_grad=False/kernel=fused", "quantity": "output",
             "e_F": 0.4, "e_R": 0.2, "dq": "on"},
            {"cell": "q_proj/needs_grad=True/kernel=fallback", "quantity": "output",
             "e_F": 0.05, "e_R": 0.01, "dq": "on"},
            {"cell": "q_proj/backward", "quantity": "input_grad", "e_F": 0.2, "e_R": 0.25,
             "dq": "off"},
        ]
        tau = cal.compute_tau(recs)
        assert tau["q_proj/needs_grad=False/kernel=fused"]["output"] == pytest.approx(0.8)
        assert tau["q_proj/needs_grad=True/kernel=fallback"]["output"] == pytest.approx(0.1)
        assert tau["q_proj/backward"]["input_grad"] == pytest.approx(0.5)
        assert "k_proj/needs_grad=False/kernel=fused" not in tau  # a cell that never occurs

    def test_floor_and_epsilon_use_strict_resolution_floor(self):
        runs = [{"scores": {"s": v, "t": 0.5}, "n_items": {"s": 40, "t": 24}}
                for v in (0.50, 0.55, 0.50, 0.525, 0.50)]
        out = cal.aggregate_n(runs)
        assert out["s"]["floor"] == pytest.approx(0.05)
        assert out["s"]["epsilon"] == pytest.approx(0.10)  # 2 x floor beats 1/40
        assert out["t"]["floor"] == 0.0
        assert out["t"]["epsilon"] == pytest.approx(1 / 24)  # one lost answer is not a flag
        assert out["t"]["comparison"] == "strict >"

    def test_n_needs_exactly_five_scorings(self):
        with pytest.raises(ValueError, match="5"):
            cal.aggregate_n([{"scores": {"s": 1.0}, "n_items": {"s": 10}}] * 4)

    def test_repeatability_compares_nll_and_bit_identity(self):
        a = {"smoke_nll": {"nll": 2.0}, "adapter_tensor_hashes": {"x": "1"}}
        b = {"smoke_nll": {"nll": 2.25}, "adapter_tensor_hashes": {"x": "1"}}
        out = cal.compare_repeat(a, b)
        assert out["difference"] == pytest.approx(0.25) and out["adapters_bit_identical"]
        c = {"smoke_nll": {"nll": 2.0}, "adapter_tensor_hashes": {"x": "2"}}
        assert not cal.compare_repeat(a, c)["adapters_bit_identical"]

    def test_lr_schedule_warms_up_over_16_updates_and_reaches_zero_at_256(self):
        f = lambda u: cal.lr_factor(u, 16, 256)  # noqa: E731
        assert f(0) == 0.0 and f(8) == 0.5 and f(16) == 1.0
        assert f(136) == pytest.approx(0.5) and f(256) == 0.0 and f(300) == 0.0

    def test_shuffle_is_a_seeded_permutation_identical_across_calls(self):
        a, b = cal.shuffled_order(100, 42), cal.shuffled_order(100, 42)
        assert a == b and sorted(a) == list(range(100))
        assert a != cal.shuffled_order(100, 7)


class TestSelection:
    def test_eight_units_per_kind_from_one_seeded_generator_and_stable(self):
        units = quant.enumerate_units()
        first = cal.select_units(units)
        again = cal.select_units(units)
        assert first == again
        assert set(first) == set(cal.KINDS)
        assert all(len(v) == 8 and len(set(u.key for u in v)) == 8 for v in first.values())
        assert all(u.kind == k for k, v in first.items() for u in v)


# --- gemm_4bit probe against a stub of the CUDA dispatch -------------------------------------


def stub_modules():
    calls = {"fused": 0, "fallback": 0}

    def kernel_impl(*_a, **_k):
        calls["fused"] += 1
        return "fused-result"

    def fallback(*_a, **_k):
        calls["fallback"] += 1
        return "fallback-result"

    def gemm_4bit(a, b, shape_b, absmax, blocksize, quant_type):
        # the structure of backends/cuda/ops.py: a heuristic picks the branch at call time
        use_custom = a.shape[0] <= 4
        impl = ops._gemm_4bit_kernel_impl if use_custom else ops._dequant_linear_fallback
        return impl(a, b, shape_b, absmax, blocksize, quant_type)

    ops = types.SimpleNamespace(_gemm_4bit_kernel_impl=kernel_impl,
                                _dequant_linear_fallback=fallback, gemm_4bit=gemm_4bit)

    class MatMul4Bit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 1

        @staticmethod
        def backward(ctx, g):
            return g

    autograd = types.SimpleNamespace(MatMul4Bit=MatMul4Bit)
    return ops, autograd, calls


class TestGemmProbe:
    def test_records_the_branch_that_ran_and_the_shape(self):
        ops, autograd, _ = stub_modules()
        with gp.GemmProbe(ops, autograd) as probe:
            ops.gemm_4bit(torch.zeros(2, 8), None, (16, 8), None, 64, "nf4")
            assert probe.take(1)["kernels"] == [gp.FUSED]
            ops.gemm_4bit(torch.zeros(9, 8), None, (16, 8), None, 64, "nf4")
            info = probe.take(1)
            assert info["kernels"] == [gp.FALLBACK] and info["shapes"] == [(9, 16, 8)]

    def test_observes_the_grad_path_independently(self):
        ops, autograd, _ = stub_modules()
        with gp.GemmProbe(ops, autograd) as probe:
            assert probe.take(0)["needs_grad_observed"] is False
            autograd.MatMul4Bit.apply(torch.ones(1))
            assert probe.take(0)["needs_grad_observed"] is True

    def test_wrong_call_count_is_an_error_not_a_guess(self):
        ops, autograd, _ = stub_modules()
        with gp.GemmProbe(ops, autograd) as probe:
            with pytest.raises(gp.ProbeError, match="expected 1"):
                probe.take(1)

    def test_restores_everything_on_exit(self):
        ops, autograd, calls = stub_modules()
        originals = (ops._gemm_4bit_kernel_impl, ops._dequant_linear_fallback)
        with gp.GemmProbe(ops, autograd):
            pass
        assert (ops._gemm_4bit_kernel_impl, ops._dequant_linear_fallback) == originals
        assert "apply" not in vars(autograd.MatMul4Bit)
        ops.gemm_4bit(torch.zeros(2, 8), None, (16, 8), None, 64, "nf4")
        assert calls["fused"] == 1

    def test_refuses_to_attach_when_the_dispatch_it_instruments_is_gone(self):
        ops, autograd, _ = stub_modules()
        del ops._dequant_linear_fallback
        with pytest.raises(gp.ProbeError, match="_dequant_linear_fallback"):
            with gp.GemmProbe(ops, autograd):
                pass

    def test_needs_grad_condition_includes_grad_mode(self):
        assert gp.expected_needs_grad(True, False, True) is True
        assert gp.expected_needs_grad(True, False, False) is False
        assert gp.expected_needs_grad(False, True, True) is True
        assert gp.expected_needs_grad(False, False, True) is False

    def test_the_real_cuda_dispatch_names_exist_in_the_installed_bitsandbytes_source(self):
        import importlib.metadata
        import pathlib

        dist = importlib.metadata.distribution("bitsandbytes")
        source = pathlib.Path(dist.locate_file("bitsandbytes/backends/cuda/ops.py")).read_text()
        for name in gp.KERNEL_ATTRS.values():
            assert f"def {name}(" in source
        assert '@register_kernel("bitsandbytes::gemm_4bit", "cuda")' in source
        assert "_dequant_linear_fallback(" in source and "_gemm_4bit_kernel_impl(" in source
        auto = pathlib.Path(dist.locate_file("bitsandbytes/autograd/_functions.py")).read_text()
        assert "class MatMul4Bit" in auto and "MatMul4Bit.apply(" in auto


# --- training, labels, T and W end to end on CPU ---------------------------------------------


def synthetic_examples(n, seed=0, vocab=128, eos=2):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        n_prompt = int(torch.randint(4, 9, (1,), generator=gen))
        n_resp = int(torch.randint(3, 8, (1,), generator=gen))
        prompt = torch.randint(3, vocab, (n_prompt,), generator=gen).tolist()
        resp = torch.randint(3, vocab, (n_resp,), generator=gen).tolist() + [eos]
        out.append({"prompt_ids": prompt, "response_ids": resp, "input_ids": prompt + resp,
                    "labels": [-100] * n_prompt + resp})
    return out


SMALL = cal.Recipe(accum=2, f_updates=3, warmup_updates=2, total_updates=8)


@pytest.fixture(scope="module")
def peft_model():
    model = tiny_model(seed=4, dtype=torch.bfloat16)
    return cal.build_lora_model(model, SMALL, gradient_checkpointing=False)


class TestTraining:
    def test_adapted_modules_and_trainable_count_match_the_main_rule_placement(self, peft_model):
        report = cal.trainable_report(peft_model, peft_model.base_model.model.config, 16)
        assert report["matches"], report
        assert report["non_fp32_factors"] == []

    def test_expected_trainable_count_for_the_real_model_is_the_rule_number(self):
        from transformers import OlmoeConfig

        real = OlmoeConfig(vocab_size=50304, hidden_size=2048, intermediate_size=1024,
                           num_hidden_layers=16, num_attention_heads=16, num_key_value_heads=16,
                           num_experts=64)
        got = cal.expected_trainable_params(real, 16)
        assert got == {"attention": 4_194_304, "experts": 117_440_512, "total": 121_634_816}

    def test_loss_counts_only_response_targets_and_matches_a_manual_sum(self):
        logits = torch.randn(1, 6, 10)
        labels = torch.tensor([[-100, -100, 3, 4, 5, 2]])
        got = cal.response_loss_sum(logits, labels)
        manual = sum(-torch.log_softmax(logits[0, t - 1].float(), -1)[labels[0, t]]
                     for t in (2, 3, 4, 5))
        assert float(got) == pytest.approx(float(manual), rel=1e-6)
        assert cal.n_targets({"labels": labels[0].tolist()}) == 4

    def test_updates_are_finite_frozen_base_unchanged_adapters_move(self):
        model = tiny_model(seed=4, dtype=torch.bfloat16)
        peft_model = cal.build_lora_model(model, SMALL, gradient_checkpointing=False)
        base_hash = {n: common.sha256_tensor(p) for n, p in peft_model.named_parameters()
                     if not p.requires_grad}
        before = cal.adapter_state(peft_model)
        res = cal.train_updates(peft_model, synthetic_examples(12), SMALL, 3, 42, "cpu", 1)
        assert res["all_finite"] and len(res["updates"]) == 3
        assert res["updates"][0]["lr"] == 0.0 and res["updates"][2]["lr"] == pytest.approx(1e-4)
        after = cal.adapter_state(peft_model)
        assert not cal.adapters_bit_identical(before, after)
        assert base_hash == {n: common.sha256_tensor(p) for n, p in peft_model.named_parameters()
                             if not p.requires_grad}
        assert all(p.grad is None for p in peft_model.parameters())

    def test_same_seed_gives_identical_data_order_and_adapters_on_cpu(self):
        results = []
        for _ in range(2):
            model = tiny_model(seed=4, dtype=torch.bfloat16)
            pm = cal.build_lora_model(model, SMALL, gradient_checkpointing=False)
            torch.manual_seed(0)
            cal.train_updates(pm, synthetic_examples(12), SMALL, 2, 42, "cpu", 1)
            results.append(cal.adapter_state(pm))
        assert cal.adapters_bit_identical(*results)

    def test_nan_loss_is_reported_as_not_finite(self):
        model = tiny_model(seed=4, dtype=torch.bfloat16)
        pm = cal.build_lora_model(model, SMALL, gradient_checkpointing=False)
        for p in pm.parameters():
            if p.requires_grad:
                p.data.fill_(float("nan"))
        res = cal.train_updates(pm, synthetic_examples(4), SMALL, 1, 42, "cpu", 1)
        assert not res["all_finite"]

    def test_smoke_nll_is_token_weighted(self):
        model = tiny_model(seed=4, dtype=torch.bfloat16)
        ex = synthetic_examples(5)
        out = cal.smoke_nll(model, ex, "cpu")
        assert out["targets"] == sum(cal.n_targets(e) for e in ex) and out["finite"]
        assert 0 < out["nll"] < 20

    def test_run_f_on_cpu_end_to_end_records_the_feasibility_fields(self, tmp_path):
        model = tiny_model(seed=4, dtype=torch.bfloat16)
        directory = write_legacy_checkpoint(model, tmp_path / "m")
        res = cal.run_f(directory, "A_bf16", synthetic_examples(12), synthetic_examples(4, 9),
                        SMALL, "cpu", 1, None, tmp_path / "adapter", experts_implementation="eager")
        assert res["oom"] is False and res["failed"] is False
        assert res["trainable"]["matches"]
        assert len(res["seconds_per_update"]) == 3 and res["median_seconds_per_update"] > 0
        assert res["experts_implementation_resolved"] == "eager"
        assert res["smoke_nll"]["finite"] and (tmp_path / "adapter").exists()
        assert res["adapter_tensor_hashes"]


class TestMemorySampler:
    def test_samples_once_per_interval_and_summarizes(self):
        import time

        readings = iter([{"used": 5, "free": 95}, {"used": 9, "free": 91}, {"used": 7, "free": 93}]
                        + [{"used": 7, "free": 93}] * 100)
        with cal.DeviceMemorySampler(lambda: next(readings), interval_s=0.01) as sampler:
            time.sleep(0.1)
        summary = sampler.summary()
        assert summary["max_used_bytes"] == 9 and summary["min_free_bytes"] == 91
        assert summary["n"] >= 3


# --- T and W on the tiny model, with real bitsandbytes CPU for R -----------------------------


@pytest.fixture(scope="module")
def tiny_world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    model = tiny_model(seed=6, dtype=torch.bfloat16)
    source = write_legacy_checkpoint(model, root / "source")
    data = root / "data"
    code = quant.run(source, data, root / "manifest.json", None, 2, E, log=lambda *_: None)
    assert code == 0 or code == 2  # U2 not compared here, so it is reported as not passed
    return {"source": source, "data": data, "model": model}


class TestTOnCpu:
    def test_run_t_produces_cells_taus_and_e0_for_every_chosen_unit(self, tiny_world):
        bf16 = cal.load_base(tiny_world["source"], "cpu", "eager")
        rows = [r["input_ids"] for r in synthetic_examples(40, seed=3, vocab=128)]
        common.set_fp32_reference_mode()
        out = cal.run_t(quant.enumerate_units(2, E), cal.UnitFiles(tiny_world["data"]), bf16,
                        rows, SMALL, "cpu", None, per_kind=2, calls_per_unit=2, min_units=1)
        assert out["records"] and out["tau"]
        assert all(r["fp32_flags"]["float32_matmul_precision"] == "highest"
                   for r in out["records"] if "fp32_flags" in r)
        kinds = {r["kind"] for r in out["records"]}
        assert kinds == set(cal.KINDS)
        quantities = {r["quantity"] for r in out["records"]}
        assert quantities == {"output", "input_grad", "A_grad", "B_grad", "update"}
        assert out["e0"] and all(e["identical"] for e in out["e0"])
        for cell, qs in out["tau"].items():
            assert all(v >= 0 for v in qs.values())
        assert any(c.endswith("/backward") for c in out["tau"])

    def test_input_gradient_cells_are_keyed_by_kind_independent_of_forward_kernel(self, tiny_world):
        bf16 = cal.load_base(tiny_world["source"], "cpu", "eager")
        rows = [r["input_ids"] for r in synthetic_examples(40, seed=3)]
        out = cal.run_t(quant.enumerate_units(2, E), cal.UnitFiles(tiny_world["data"]), bf16,
                        rows, SMALL, "cpu", None, per_kind=2, calls_per_unit=1, min_units=1)
        back = [r for r in out["records"] if r["quantity"] == "input_grad"]
        assert back and all(r["cell"] == f"{r['kind']}/backward" for r in back)
        assert all("forward_cell" in r for r in back)

    def test_captured_expert_inputs_are_the_routed_tokens(self, tiny_world):
        bf16 = cal.load_base(tiny_world["source"], "cpu", "eager")
        units = quant.enumerate_units(2, E)
        chosen = {k: [u for u in units if u.kind == k][:1] for k in cal.KINDS}
        rows = [r["input_ids"] for r in synthetic_examples(20, seed=5)]
        got = cal.capture_unit_inputs(bf16, chosen, rows, 2, "cpu")
        attn = chosen["q_proj"][0]
        assert all(x.shape[1] == 64 for x in got[attn.key]) and len(got[attn.key]) == 2
        down = chosen["down_proj"][0]
        assert all(x.shape[1] == bf16.config.intermediate_size for x in got[down.key])
        gate = chosen["gate_proj"][0]
        assert all(x.shape[1] == 64 for x in got[gate.key])
        assert len(got[gate.key]) == len(got[down.key])  # same routing, same calls

    def test_unit_files_read_back_what_the_quantization_wrote(self, tiny_world):
        from bitsandbytes.functional import QuantState, dequantize_4bit

        files = cal.UnitFiles(tiny_world["data"])
        unit = quant.enumerate_units(2, E)[5]
        w = files.weight("on", unit)
        packed, qs = files.codes("on", unit)
        back = dequantize_4bit(packed, QuantState.from_dict(qs, device="cpu"))
        assert quant.bitwise_equal(back, w)


class TestWOnCpu:
    def test_run_w_reports_tolerances_and_a_clean_frozen_base(self, tiny_world):
        rows = [r["input_ids"] for r in synthetic_examples(6, seed=8)]
        out = cal.run_w({"A_off": tiny_world["data"] / "writeback-dqoff",
                         "A_on": tiny_world["data"] / "writeback-dqon"},
                        rows, SMALL, "cpu", layers=(0, 1), rows=6,
                        experts_implementation="eager")
        assert out["frozen_base_changed"] == []
        assert len(out["per_call"]) == 2 * 2 * 6
        assert set(out["tol_W"]) == set(wref.W_QUANTITIES)
        assert all(v > 0 for v in out["tol_W"].values())
        assert out["fp32_flags"]["float32_matmul_precision"] == "highest"
        assert out["experts_implementations_bf16_fp32"] == [("eager", "eager")]


# --- driver, input hashes, report -------------------------------------------------------------

CHILD = """
import sys
sys.path.insert(0, {harness!r})
import e4_part2_common as c
j = c.AttemptsJournal({journal!r})
a = j.start({run_id!r}, "T")
mode = {mode!r}
if mode == "ok":
    j.end(a, c.OUTCOME_OK)
elif mode == "void":
    j.end(a, c.OUTCOME_VOID, ["peers: foreign"])
# mode == "die": no end, like a killed process
"""


def child_argv(tmp_path, run_id, mode):
    code = CHILD.format(harness=str(cal.common.HARNESS_DIR),
                        journal=str(tmp_path / "attempts.jsonl"),
                        run_id=run_id, mode=mode)
    return ["-c", code]


class TestDriver:
    def test_a_void_is_rerun_once_in_a_fresh_process_then_accepted(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        calls = iter(["void", "ok"])

        def fake_child(_j, run_id, _argv):
            mode = next(calls)
            subprocess_run = __import__("subprocess").run
            subprocess_run([__import__("sys").executable, *child_argv(tmp_path, run_id, mode)],
                           check=True)

        original = cal.drive_child
        cal.drive_child = fake_child
        try:
            last = cal.drive_run(journal, "T", [])
        finally:
            cal.drive_child = original
        assert last["outcome"] == "ok"
        assert [a["outcome"] for a in journal.attempts("T")] == ["void", "ok"]

    def test_a_killed_child_is_a_void_and_two_of_them_stop_calibration(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        with pytest.raises(common.CalibrationStopped):
            cal.drive_run(journal, "W", child_argv(tmp_path, "W", "die"))
        outcomes = [a["outcome"] for a in journal.attempts("W")]
        assert outcomes == ["void", "void"]
        assert all("interrupted" in a["reasons"][0] for a in journal.attempts("W"))

    def test_every_attempt_is_a_new_process(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        cal.drive_run(journal, "T", child_argv(tmp_path, "T", "ok"))
        starts = [e for e in journal.events() if e["event"] == "start"]
        import os

        assert starts[0]["pid"] != os.getpid()


class TestInputHashes:
    def test_a_changed_or_missing_file_is_a_void_reason(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"one")
        (tmp_path / "b.bin").write_bytes(b"two")
        expected = {"a.bin": {"sha256": common.sha256_bytes(b"one")},
                    "b.bin": {"sha256": common.sha256_bytes(b"two")},
                    "c.bin": {"sha256": common.sha256_bytes(b"three")}}
        assert cal.verify_files(tmp_path, expected, ["a.bin", "b.bin"]) == []
        (tmp_path / "b.bin").write_bytes(b"tampered")
        reasons = cal.verify_files(tmp_path, expected, ["a.bin", "b.bin", "c.bin", "d.bin"])
        assert any("b.bin" in r and "differs" in r for r in reasons)
        assert any("c.bin" in r and "missing" in r for r in reasons)
        assert any("d.bin" in r and "not in the manifest" in r for r in reasons)

    def test_manifest_hash_mismatch_is_an_error(self, tmp_path):
        path = tmp_path / "smoke.manifest.jsonl"
        path.write_bytes(b'{"row_id":1}\n')
        assert cal.read_manifest(path, common.sha256_bytes(path.read_bytes())) == [{"row_id": 1}]
        with pytest.raises(ValueError, match="SHA-256"):
            cal.read_manifest(path, "0" * 64)

    def test_encoding_rows_checks_identity_and_token_counts(self):
        import e4_part2_split as split

        from tests.test_e4_part2_split import TOK, row

        raw = [row(0), row(1)]
        good = []
        for i, r in enumerate(raw):
            f = [split.normalize_field(r[k]) for k in ("instruction", "context", "response")]
            enc = split.encode_example(TOK, *f)
            good.append({"row_id": i, "identity": split.identity(*f),
                         "prompt_tokens": len(enc["prompt_ids"]),
                         "response_tokens": len(enc["response_ids"])})
        assert [e["row_id"] for e in cal.encode_manifest_rows(good, raw, TOK)] == [0, 1]
        with pytest.raises(ValueError, match="identity"):
            cal.encode_manifest_rows([dict(good[0], identity="0" * 64)], raw, TOK)
        with pytest.raises(ValueError, match="token counts"):
            cal.encode_manifest_rows([dict(good[0], prompt_tokens=1)], raw, TOK)


class TestReport:
    def _run(self, tmp_path, run_id, result, stamp_extra=None, fail_first=False):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        stamp = dict(GOOD_STAMP, gpu_name="NVIDIA A100-SXM4-40GB", gpu_memory_total_mib="40960",
                     gpu_compute_cap="8.0", **(stamp_extra or {}))
        if fail_first:
            cal.execute_attempt(journal, run_id, run_id[0], tmp_path / "runs", lambda: {},
                                stamp=stamp_factory(stamp, stamp),
                                prov=lambda: dict(GOOD_PROV, harness_dirty=True))
        cal.execute_attempt(journal, run_id, run_id[0], tmp_path / "runs", lambda: result,
                            stamp=stamp_factory(stamp, stamp), prov=lambda: GOOD_PROV)

    def test_report_collects_the_values_the_main_rule_copies_and_keeps_voids(self, tmp_path):
        for arm in cal.ARMS:
            self._run(tmp_path, f"F_{arm}", {"oom": False, "failed": False,
                                              "max_memory_reserved_bytes": 21e9,
                                              "median_seconds_per_update": 12.5})
        (tmp_path / "decision.json").write_text(json.dumps(
            cal.decide_final_card(cal.f_results(tmp_path))))
        self._run(tmp_path, "T", {"tau": {"q_proj/backward": {"input_grad": 0.5}},
                                  "calls_per_unit": 4}, fail_first=True)
        self._run(tmp_path, "W", {"tol_W": {"output": 0.25}})
        for i in range(1, 6):
            self._run(tmp_path, f"N_{i}", {"scores": {"s": 0.5 + 0.01 * (i % 2)},
                                           "n_items": {"s": 40}})
        self._run(tmp_path, "P", {"comparison": {"difference": 0.0}})
        report = cal.assemble_report(tmp_path)
        assert report["final_card_decision"]["decision"] == "FINAL_A100_40GB"
        assert report["final_card"]["gpu_uuid"] == "GPU-1"
        assert report["tau"] == {"q_proj/backward": {"input_grad": 0.5}}
        assert report["tol_W"] == {"output": 0.25}
        assert report["noise_floor"]["s"]["floor"] == pytest.approx(0.01)
        assert report["repeatability"] == {"difference": 0.0}
        assert [v["run_id"] for v in report["voids"]] == ["T"]
        assert report["provenance_gaps"] == []

    def test_report_names_runs_with_no_journal_entry_as_gaps(self, tmp_path):
        self._run(tmp_path, "F_A_off", {"oom": False})
        report = cal.assemble_report(tmp_path)
        assert "T: no journal entry" in report["provenance_gaps"]
        assert "noise_floor" not in report

    def test_sha256sums_cover_every_file_but_themselves(self, tmp_path):
        (tmp_path / "x.json").write_text("{}")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "y.json").write_text("[]")
        assert cal.write_sha256sums(tmp_path) == 2
        lines = (tmp_path / "SHA256SUMS").read_text().splitlines()
        assert lines[0].endswith("  sub/y.json") and lines[1].endswith("  x.json")

    def test_decide_command_writes_the_decision(self, tmp_path):
        for arm in cal.ARMS:
            self._run(tmp_path, f"F_{arm}", {"oom": arm == "A_on", "failed": False,
                                              "max_memory_reserved_bytes": 1})
        code = cal.main(["decide", "--out", str(tmp_path)])
        assert code == 4
        assert json.loads((tmp_path / "decision.json").read_text())["decision"] == (
            "REPEAT_F_ON_A100_80GB")


class TestFrozenBaseHashInF:
    def test_frozen_hash_changes_when_a_frozen_weight_changes_and_not_when_adapters_move(self):
        model = tiny_model(seed=4, dtype=torch.bfloat16)
        pm = cal.build_lora_model(model, SMALL, gradient_checkpointing=False)
        before = cal.frozen_base_hash(pm)
        for p in pm.parameters():
            if p.requires_grad:
                p.data.add_(1.0)
        assert cal.frozen_base_hash(pm) == before
        next(p for p in pm.parameters() if not p.requires_grad).data.add_(1.0)
        assert cal.frozen_base_hash(pm) != before


class TestNonFinite:
    def test_non_finite_numbers_void_t_w_and_n(self):
        assert cal.nonfinite_reasons("T", {"a": 1.0, "b": float("nan")}) == [
            "finiteness: T b is not finite"]
        assert cal.nonfinite_reasons("W", {"a": float("inf")})
        assert cal.nonfinite_reasons("N", {"a": 0.5}) == []
