"""``moe_lora: true`` must keep the architecture's own attention projections.

``get_moe_target_modules`` named ``q_proj``/``k_proj``/``v_proj``/``o_proj`` for
every MoE model. DeepSeek-V3's MLA attention has none of the first three, so under
``moe_lora`` only ``o_proj`` and the routed experts were adapted, while
``target_modules: auto`` adapted all five MLA projections. B2 ticket 13 follow-up.
"""

from __future__ import annotations

import re
import types

import pytest

from soup_cli.utils.moe import STANDARD_TARGET_MODULES, get_moe_target_modules

_EXPERT_NAMES = ["down_proj", "gate_proj", "up_proj"]
_MLA = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]


def _tiny_deepseek_v3(**overrides):
    transformers = pytest.importorskip("transformers")
    fields = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    fields.update(overrides)
    config = transformers.DeepseekV3Config(**fields)
    return transformers.AutoModelForCausalLM.from_config(config)


def _fake_moe(model_type):
    """A model that is MoE by config and has no named modules to scan."""
    config = types.SimpleNamespace(model_type=model_type, num_experts=4, text_config=None)
    return types.SimpleNamespace(config=config, named_modules=lambda: [])


class TestAttentionNamesFollowTheArchitecture:
    def test_deepseek_v3_gets_its_mla_projections(self):
        targets = get_moe_target_modules(_tiny_deepseek_v3())

        assert targets == _MLA + _EXPERT_NAMES

    def test_deepseek_v3_without_q_lora_gets_q_proj(self):
        targets = get_moe_target_modules(_tiny_deepseek_v3(q_lora_rank=None))

        assert targets == ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"] + _EXPERT_NAMES

    def test_an_architecture_with_standard_names_is_unchanged(self):
        targets = get_moe_target_modules(_fake_moe("qwen3_moe"))

        assert targets == STANDARD_TARGET_MODULES + ["gate_proj", "up_proj", "down_proj"]

    @pytest.mark.parametrize(
        ("model_type", "attention"),
        [
            ("deepseek_v4", ["q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj"]),
            ("glm_moe_dsa", _MLA),
            ("kimi_k2", _MLA),
            ("kimi_k25", _MLA),
        ],
    )
    def test_other_table_entries_with_their_own_names(self, model_type, attention):
        targets = get_moe_target_modules(_fake_moe(model_type))

        assert targets == attention + ["gate_proj", "up_proj", "down_proj"]

    def test_a_model_type_the_table_lacks_keeps_the_standard_names(self):
        targets = get_moe_target_modules(_fake_moe("mixtral"))

        assert targets[: len(STANDARD_TARGET_MODULES)] == STANDARD_TARGET_MODULES

    def test_a_regex_table_entry_keeps_the_standard_names(self):
        """``minimax_m3_vl`` is a regex scoped to its language tower, which cannot
        be merged into a suffix list; it keeps today's behaviour."""
        targets = get_moe_target_modules(_fake_moe("minimax_m3_vl"))

        assert targets[: len(STANDARD_TARGET_MODULES)] == STANDARD_TARGET_MODULES


class TestEveryMlaProjectionGetsAnAdapter:
    def test_peft_adapts_the_mla_projections_and_the_experts(self):
        from peft import LoraConfig, TaskType, get_peft_model

        model = _tiny_deepseek_v3()
        config = LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=get_moe_target_modules(model),
            task_type=TaskType.CAUSAL_LM,
        )

        peft_model = get_peft_model(model, config)

        adapted = {
            re.sub(r"\.\d+\.", ".N.", name.split(".lora_")[0])
            for name, _ in peft_model.named_parameters()
            if "lora_" in name
        }
        for projection in _MLA:
            assert f"base_model.model.model.layers.N.self_attn.{projection}" in adapted
        assert "base_model.model.model.layers.N.mlp.experts" in adapted
