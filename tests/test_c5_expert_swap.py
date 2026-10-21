"""C5 ticket 02: the expert-only codec swap (benchmarks/harness/c5_expert_swap.py).

Spec: .scratch/c5-q2-0-experts/issues/02-expert-only-swap.md. Every model here
is a tiny random-init MoE built from a config, so nothing is downloaded.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("bitsandbytes")

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def harness():
    original_path = sys.path[:]
    names = ("_c5_expert_swap_test_module", "q2_0_codec")
    original_modules = {name: sys.modules.get(name) for name in names}
    spec = importlib.util.spec_from_file_location(names[0], HARNESS / "c5_expert_swap.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[names[0]] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        for name, previous in original_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _granite_experts(layers: int) -> list[str]:
    """The fused expert tensors of transformers' granitemoe, written out by hand."""
    return [
        f"model.layers.{i}.block_sparse_moe.experts.{proj}"
        for i in range(layers)
        for proj in ("gate_up_proj", "down_proj")
    ]


def _granite_arm(harness, codec_name: str, **fixture):
    return harness.run_arm(harness.granite_fixture(seed=0, **fixture), codec_name)


def _patch_swap(harness, monkeypatch, *, before=None, skip=None):
    """Wrap ``swap_experts``: run ``before(model)`` first, or leave expert ``skip`` alone."""
    real_swap = harness.swap_experts

    def patched(model, codec_name, names):
        if before is not None:
            with torch.no_grad():
                before(model)
        return real_swap(model, codec_name, [name for name in names if name != skip])

    monkeypatch.setattr(harness, "swap_experts", patched)


def _rel_mse(restored, original) -> float:
    restored, original = restored.float(), original.float()
    return ((restored - original).pow(2).sum() / original.pow(2).sum()).item()


class TestNonExpertsUntouched:
    def test_the_swap_changes_only_the_experts(self, harness):
        record = _granite_arm(harness, "q2_0", layers=2)
        assert sorted(record["changed"]) == sorted(_granite_experts(2))
        assert record["checks"]["non_expert_untouched"] is True

    def test_a_codec_that_touches_a_norm_fails_the_check(self, harness, monkeypatch):
        _patch_swap(harness, monkeypatch, before=lambda model: model.model.norm.weight.add_(1.0))
        record = _granite_arm(harness, "q2_0", layers=2)
        assert record["checks"]["non_expert_untouched"] is False
        assert "model.norm.weight" in record["changed"]


class TestArmsAgree:
    def test_nf4_and_q2_0_leave_the_same_non_expert_weights(self, harness):
        nf4 = _granite_arm(harness, "nf4", layers=2)
        q2 = _granite_arm(harness, "q2_0", layers=2)
        assert "model.norm.weight" in nf4["non_expert_fingerprints"]
        assert "model.layers.1.block_sparse_moe.router.weight" in nf4["non_expert_fingerprints"]
        assert harness.non_expert_mismatch(nf4, q2) == []

    def test_a_differing_router_is_reported(self, harness):
        nf4 = _granite_arm(harness, "nf4", layers=2)
        other = harness.granite_fixture(seed=0, layers=2)
        with torch.no_grad():
            other.model.layers[0].block_sparse_moe.router.weight.mul_(2.0)
        q2 = harness.run_arm(other, "q2_0")
        assert harness.non_expert_mismatch(nf4, q2) == [
            "model.layers.0.block_sparse_moe.router.weight"
        ]


class TestErrors:
    def test_q2_0_records_search_and_absmax_reference_for_every_expert(self, harness):
        record = _granite_arm(harness, "q2_0", layers=2)
        assert sorted(record["tensors"]) == sorted(_granite_experts(2))
        for entry in record["tensors"].values():
            search = entry["error"]["rel_mse"]
            reference = entry["reference_q2_0_absmax"]["rel_mse"]
            assert 0 < search < reference

    def test_nf4_has_no_absmax_reference(self, harness):
        record = _granite_arm(harness, "nf4", layers=2)
        for entry in record["tensors"].values():
            assert entry["error"]["rel_mse"] > 0
            assert "reference_q2_0_absmax" not in entry

    @pytest.mark.parametrize("codec_name", ["q2_0", "nf4"])
    def test_in_model_error_is_the_weight_the_model_now_holds(self, harness, codec_name):
        original = harness.granite_fixture(seed=0, layers=1)
        swapped = harness.granite_fixture(seed=0, layers=1)
        record = harness.run_arm(swapped, codec_name)
        name = "model.layers.0.block_sparse_moe.experts.down_proj"
        now, then = swapped.get_parameter(name).data, original.get_parameter(name).data
        entry = record["tensors"][name]
        assert entry["error_in_model"]["rel_mse"] == pytest.approx(
            _rel_mse(now, then), rel=1e-5
        )
        assert len(entry["per_expert"]) == 4
        assert entry["per_expert"][2]["error_in_model"]["rel_mse"] == pytest.approx(
            _rel_mse(now[2], then[2]), rel=1e-5
        )

    def test_rows_not_a_multiple_of_64_are_recorded(self, harness):
        record = _granite_arm(harness, "q2_0", layers=1, intermediate_size=96)
        down = "model.layers.0.block_sparse_moe.experts.down_proj"
        gate_up = "model.layers.0.block_sparse_moe.experts.gate_up_proj"
        assert record["tensors"][down]["row_size"] == 96
        assert record["tensors"][down]["rows_multiple_of_64"] is False
        assert record["tensors"][gate_up]["rows_multiple_of_64"] is True
        assert record["rows_not_multiple_of_64"] == [down]


class TestMain:
    def test_granite_demo_writes_both_arms_labelled_synthetic(self, harness, tmp_path, capsys):
        assert harness.main(["--fixture", "granite", "--out-dir", str(tmp_path)]) == 0
        assert "SYNTHETIC" in capsys.readouterr().out
        records = {
            codec_name: json.loads((tmp_path / f"{codec_name}.json").read_text("utf-8"))
            for codec_name in ("nf4", "q2_0")
        }
        for codec_name, record in records.items():
            assert record["codec"] == codec_name
            assert record["model"]["synthetic"] is True
            assert "SYNTHETIC" in record["model"]["note"]
            assert record["model"]["model_type"] == "granitemoe"
            assert "revision" in record["model"]
            assert record["versions"]["torch"] == importlib.metadata.version("torch")
            assert record["versions"]["transformers"]
            assert len(record["harness"]["sha256"]) == 64
            assert record["harness"]["codec_sha256"] == harness.codec.codec_sha256()
            assert record["checks"]["non_expert_match_other_arm"] is True
            assert all(record["checks"].values())
            assert record["non_expert_mismatch"] == []
        assert (
            records["nf4"]["non_expert_fingerprints"]
            == records["q2_0"]["non_expert_fingerprints"]
        )

    def test_a_leaky_codec_fails_the_run(self, harness, tmp_path, monkeypatch):
        _patch_swap(harness, monkeypatch, before=lambda model: model.lm_head.weight.add_(1.0))
        assert harness.main(["--fixture", "granite", "--out-dir", str(tmp_path)]) == 1
        record = json.loads((tmp_path / "q2_0.json").read_text("utf-8"))
        assert record["checks"]["non_expert_untouched"] is False

    def test_a_real_model_needs_a_pinned_revision(self, harness, tmp_path):
        with pytest.raises(SystemExit):
            harness.main(["--model", "allenai/OLMoE-1B-7B-0924", "--out-dir", str(tmp_path)])


class TestExpertList:
    def test_quantized_tensors_are_exactly_the_granite_experts(self, harness):
        record = _granite_arm(harness, "q2_0", layers=2)
        assert sorted(record["quantized"]) == sorted(_granite_experts(2))
        assert record["checks"]["expert_rules_agree"] is True
        assert record["checks"]["every_expert_changed"] is True

    def test_a_skipped_expert_fails_the_check(self, harness, monkeypatch):
        skipped = "model.layers.1.block_sparse_moe.experts.down_proj"
        _patch_swap(harness, monkeypatch, skip=skipped)
        record = _granite_arm(harness, "q2_0", layers=2)
        assert record["checks"]["every_expert_changed"] is False

    def test_a_model_without_experts_fails_the_check(self, harness):
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=4,
        )
        record = harness.run_arm(LlamaForCausalLM(config), "q2_0")
        assert record["experts"] == []
        assert record["checks"]["expert_rules_agree"] is False

    def test_olmoe_experts_are_found_under_mlp(self, harness):
        from transformers import OlmoeConfig, OlmoeForCausalLM

        config = OlmoeConfig(
            vocab_size=128, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, num_experts=4,
            num_experts_per_tok=2, pad_token_id=0, eos_token_id=1, bos_token_id=None,
        )
        record = harness.run_arm(OlmoeForCausalLM(config), "q2_0")
        assert sorted(record["quantized"]) == sorted(
            f"model.layers.{i}.mlp.experts.{proj}"
            for i in range(2)
            for proj in ("gate_up_proj", "down_proj")
        )
        assert record["checks"]["expert_rules_agree"] is True

    def test_shared_expert_is_not_quantized(self, harness):
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

        config = Qwen2MoeConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=64,
            shared_expert_intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=4, num_experts=4,
            num_experts_per_tok=2,
        )
        record = harness.run_arm(Qwen2MoeForCausalLM(config), "q2_0")
        assert sorted(record["quantized"]) == [
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.experts.gate_up_proj",
        ]
        assert record["checks"]["expert_rules_agree"] is True
