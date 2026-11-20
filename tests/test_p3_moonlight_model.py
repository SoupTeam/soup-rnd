"""Moonlight contract tests and separately qualified native NF4 session integration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.harness import p3_deepseek_quality_model as quality
from benchmarks.harness.b13_moe_fixture import moe_config, moe_weights_dir

PROFILE = "moonlight-16b-a3b"


def test_native_moonlight_normalization_preserves_source_and_shared_initial_adapter(
    tmp_path, moonlight_native_cpu_stack,
):
    torch = moonlight_native_cpu_stack
    from benchmarks.harness.p3_deepseek_oracle import DeepseekOracleProbe

    source = moe_weights_dir(tmp_path, moe_config(
        hidden_size=128, intermediate_size=128, moe_intermediate_size=64,
        num_hidden_layers=2, q_lora_rank=None, kv_lora_rank=32,
        n_routed_experts=64, num_experts_per_tok=6, n_group=1, topk_group=1,
        scoring_func="sigmoid", topk_method="noaux_tc", routed_scaling_factor=2.446,
        aux_loss_alpha=0.001, seq_aux=True, attention_dropout=0.0,
    ), seed=23)
    original = (source / "config.json").read_bytes()
    original_config = json.loads(original)
    cache, initial = tmp_path / "cache", tmp_path / "initial.safetensors"
    with quality.open_session(source, cache, device="cpu", prepare=True,
                              profile_id=PROFILE) as first:
        runtime_dir = Path(first.runtime_config_directory.name)
        normalized = json.loads((runtime_dir / "config.json").read_bytes())
        assert normalized == {key: value for key, value in original_config.items()
                              if key not in {"aux_loss_alpha", "seq_aux"}}
        assert {"aux_loss_alpha", "seq_aux"}.isdisjoint(vars(first.config))
        assert (source / "config.json").read_bytes() == original
        changes = first.metadata["source"]["runtime_config_normalization"]
        assert [(row["field"], row["source_value"], row["runtime"]) for row in changes] == [
            ("aux_loss_alpha", 0.001, "absent"), ("seq_aux", True, "absent"),
        ]
        params = quality.trainable_parameters(first.model)
        targets = {name.split(".self_attn.", 1)[1].split(".", 1)[0] for name in params}
        assert targets == {"q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"}
        assert len(params) == 16 and all(value.dtype == torch.float32 for value in params.values())
        guard = object.__new__(DeepseekOracleProbe)
        guard._configs = [first.config]
        guard._check_config()
        rejected_config = copy.deepcopy(first.config)
        rejected_config.aux_loss_alpha = 0.001
        rejected_config.seq_aux = True
        guard._configs = [rejected_config]
        with pytest.raises(RuntimeError, match="auxiliary"):
            guard._check_config()
        receipt = quality.write_initial_adapter(first.model, initial, profile_id=PROFILE)
        expected_digest = quality.adapter_digest(params)[0]
        metadata = first.metadata
    assert not runtime_dir.exists()
    with quality.open_session(source, cache, device="cpu", profile_id=PROFILE) as second:
        assert second.metadata == metadata
        assert quality.load_initial_adapter(second.model, initial, profile_id=PROFILE) == (
            receipt["sha256"])
        restored_digest = quality.adapter_digest(quality.trainable_parameters(second.model))[0]
        assert restored_digest == expected_digest
        with pytest.raises(ValueError, match="profile"):
            quality.load_initial_adapter(second.model, initial)
    assert (source / "config.json").read_bytes() == original


@pytest.mark.parametrize("declared,reason", [
    ({"aux_loss_alpha": 0.002, "seq_aux": True}, "aux value differs.*aux_loss_alpha"),
    ({"aux_loss_alpha": 0.001, "seq_aux": False}, "aux value differs.*seq_aux"),
    ({"aux_loss_alpha": 0.001}, "only part of the admitted aux field pair"),
])
def test_normalization_rejects_values_not_covered_by_the_proof(
    declared, reason, moonlight_aux_admission,
):
    with pytest.raises(ValueError, match=reason):
        quality._source_aux_metadata(declared, profile_id=PROFILE)


@pytest.mark.parametrize("damage,reason", [
    ("missing_bias", "source keys differ"),
    ("scale_key", "source keys differ"),
    ("wrong_shape", "tensor shape"),
    ("wrong_rotary_shape", "tensor shape"),
    ("wrong_dtype", "source dtype"),
    ("header_index_disagreement", "header/index"),
])
def test_checkpoint_header_admission_refuses_incompatible_source(tmp_path, damage, reason):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from benchmarks.harness.p3_deepseek_quality_profiles import get_profile

    config = dict(get_profile(PROFILE).shape)
    config.update(num_hidden_layers=2, hidden_size=8, vocab_size=16,
                  num_attention_heads=2, kv_lora_rank=4, qk_rope_head_dim=2,
                  qk_nope_head_dim=2, v_head_dim=2, intermediate_size=16,
                  moe_intermediate_size=8, n_routed_experts=2, n_shared_experts=1)
    shapes = quality._moonlight_source_shapes(config)
    tensors = {name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in shapes.items()}
    weight_map = dict.fromkeys(shapes, "tiny.safetensors")
    bias = "model.layers.1.mlp.gate.e_score_correction_bias"
    if damage == "missing_bias":
        del weight_map[bias]
        del tensors[bias]
    elif damage == "scale_key":
        weight_map["model.layers.1.mlp.experts.0.weight_scale_inv"] = "tiny.safetensors"
        tensors["model.layers.1.mlp.experts.0.weight_scale_inv"] = torch.ones(1)
    elif damage == "wrong_shape":
        tensors["model.embed_tokens.weight"] = torch.zeros((17, 8), dtype=torch.bfloat16)
    elif damage == "wrong_rotary_shape":
        tensors["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.zeros(
            config["qk_rope_head_dim"] // 2, dtype=torch.bfloat16)
    elif damage == "wrong_dtype":
        tensors["model.embed_tokens.weight"] = tensors["model.embed_tokens.weight"].float()
    else:
        del tensors[bias]
    save_file(tensors, str(tmp_path / "tiny.safetensors"))
    with pytest.raises(ValueError, match=reason):
        quality._validate_moonlight_headers(tmp_path, config, weight_map)


def _aux_repository_evidence(root: Path) -> tuple[dict, dict]:
    """Use isolated source bytes, not a snapshot of the current Soup implementation."""
    relative = "src/soup_cli/utils/peft_wiring.py"
    payload = b"def audited_source():\n    return 1\n"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    sources = {f"repository/{relative}": {"sha256": digest}}
    continuity = {
        "aux_report_sha256": quality._AUX_REPORT_SHA256,
        "unchanged_sources": {
            relative: {"source_sha256": digest, "normalized_text_sha256": digest},
        },
        "reviewed_wrapper_deltas": [],
    }
    return sources, continuity


def test_aux_repository_audit_refuses_changed_protected_source(tmp_path):
    sources, continuity = _aux_repository_evidence(tmp_path)
    quality._audit_aux_repository_sources(tmp_path, sources, continuity)
    modified = tmp_path / "src/soup_cli/utils/peft_wiring.py"
    with modified.open("ab") as stream:
        stream.write(b"\n# unreviewed execution-source change\n")
    with pytest.raises(ValueError, match="audited repository source differs.*peft_wiring"):
        quality._audit_aux_repository_sources(tmp_path, sources, continuity)


def test_aux_repository_audit_normalizes_only_line_endings(tmp_path):
    sources, continuity = _aux_repository_evidence(tmp_path)
    source = tmp_path / "src/soup_cli/utils/peft_wiring.py"
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    quality._audit_aux_repository_sources(tmp_path, sources, continuity)
    source.write_bytes(source.read_bytes().replace(b"return 1", b"return 2"))
    with pytest.raises(ValueError, match="audited repository source differs"):
        quality._audit_aux_repository_sources(tmp_path, sources, continuity)


@pytest.mark.parametrize("damage", ["missing_source", "unlisted_source", "overlap"])
def test_aux_repository_audit_requires_exact_continuity_coverage(tmp_path, damage):
    sources, continuity = _aux_repository_evidence(tmp_path)
    relative = "src/soup_cli/utils/peft_wiring.py"
    if damage == "missing_source":
        sources.clear()
    elif damage == "unlisted_source":
        sources["repository/src/soup_cli/utils/unlisted.py"] = {"sha256": "a" * 64}
    else:
        continuity["reviewed_wrapper_deltas"] = [relative]
    with pytest.raises(ValueError, match="source continuity coverage differs"):
        quality._audit_aux_repository_sources(tmp_path, sources, continuity)


@pytest.mark.parametrize("field", ["source_sha256", "normalized_text_sha256"])
def test_aux_repository_audit_checks_both_historical_and_normalized_hashes(tmp_path, field):
    sources, continuity = _aux_repository_evidence(tmp_path)
    continuity["unchanged_sources"]["src/soup_cli/utils/peft_wiring.py"][field] = "0" * 64
    with pytest.raises(ValueError, match="audited repository source differs"):
        quality._audit_aux_repository_sources(tmp_path, sources, continuity)


def test_aux_repository_audit_keeps_explicit_wrapper_delta_partition(tmp_path):
    sources, continuity = _aux_repository_evidence(tmp_path)
    relative = "benchmarks/harness/p3_deepseek_quality_model.py"
    sources[f"repository/{relative}"] = {"sha256": "a" * 64}
    continuity["reviewed_wrapper_deltas"] = [relative]
    quality._audit_aux_repository_sources(tmp_path, sources, continuity)
    continuity["aux_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source continuity coverage differs"):
        quality._audit_aux_repository_sources(tmp_path, sources, continuity)


@pytest.mark.parametrize("package", [
    "huggingface_hub", "numpy", "peft", "rich", "torch", "transformers",
])
def test_native_aux_admission_still_refuses_each_unqualified_package(monkeypatch, package):
    root = Path(quality.__file__).parents[2]
    report = json.loads((root / quality._AUX_REPORT).read_bytes())
    expected = report["packages"][package]["version"]

    def installed_version(name):
        return ("unqualified-test-version" if name == package
                else report["packages"][name]["version"])

    monkeypatch.setattr(quality, "version", installed_version)
    monkeypatch.setattr(quality, "distribution", lambda _name: object())
    with pytest.raises(ValueError, match=rf"requires {package}=={expected}"):
        quality.native_aux_admission(PROFILE)
