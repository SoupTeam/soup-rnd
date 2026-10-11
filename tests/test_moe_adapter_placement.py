"""Tiny-model checks for benchmarks/harness/moe_adapter_placement.py (E1).

The harness builds its own expert LoRA instead of going through peft, so these
pin what the record leans on: a zero adapter is the base model, the per-expert
delta is the one peft's ParamWrapper would add, only the selected experts carry
parameters, and the shared-expert arm refuses a model that has none.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "moe_adapter_placement.py"


@pytest.fixture(scope="module")
def harness():
    original_path = sys.path[:]
    spec = importlib.util.spec_from_file_location("_moe_adapter_placement", HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        sys.modules.pop(spec.name, None)


def _olmoe():
    from transformers import OlmoeConfig, OlmoeForCausalLM

    config = OlmoeConfig(vocab_size=96, hidden_size=32, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         num_experts=8, num_experts_per_tok=2, pad_token_id=0, bos_token_id=1,
                         eos_token_id=2)
    torch.manual_seed(0)
    return OlmoeForCausalLM(config).eval()


def _qwen35():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    config = Qwen3_5MoeTextConfig(
        vocab_size=96, hidden_size=32, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, num_experts=8, num_experts_per_tok=2,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=["linear_attention", "full_attention"],
        pad_token_id=0, bos_token_id=1, eos_token_id=2)
    torch.manual_seed(0)
    return Qwen3_5MoeForCausalLM(config).eval()


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


IDS = torch.tensor([[1, 5, 9, 14, 22, 31, 40, 7, 3, 60, 61, 62]])


def test_adapter_classes_are_cached(harness):
    assert harness.adapter_classes() is harness.adapter_classes()


@pytest.mark.parametrize("make", [_olmoe, _qwen35])
@pytest.mark.parametrize("arm", ["A", "AT", "AE"])
def test_zero_adapter_is_the_base_model(harness, make, arm):
    model = make()
    before = _logits(model, IDS)
    hot = [[0, 3], [1, 5]] if arm == "AT" else None
    harness.attach_adapters(model, arm, hot=hot, seed=0)
    assert torch.allclose(_logits(model, IDS), before, atol=1e-5, rtol=0)


def test_shared_expert_arm_is_not_applicable_without_one(harness):
    with pytest.raises(harness.NotApplicableError):
        harness.attach_adapters(_olmoe(), "AS")


def test_shared_expert_arm_adapts_exactly_attention_and_shared_expert(harness):
    targets = harness.attach_adapters(_qwen35(), "AS")
    assert sorted(t.rsplit(".", 1)[-1] for t in targets if ".shared_expert." in t) == \
        sorted(["down_proj", "gate_proj", "up_proj"] * 2)
    assert not any("experts[" in t for t in targets)


def test_only_selected_experts_carry_parameters(harness):
    model = _olmoe()
    harness.attach_adapters(model, "AT", hot=[[0, 3], [1, 5]], seed=0)
    full = _olmoe()
    harness.attach_adapters(full, "AE", seed=0)
    count = sum(p.numel() for p in harness.adapter_parameters(model))
    count_full = sum(p.numel() for p in harness.adapter_parameters(full))
    attention = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and "self_attn" in n)
    # Per selected expert: gate_up (r*32 + 32*r) + down (r*16 + 32*r) at r=16.
    per_expert = 16 * (32 + 32) + 16 * (16 + 32)
    assert count - attention == 2 * 2 * per_expert
    assert count_full - attention == 2 * 8 * per_expert


def _trained_expert_delta(harness, seed=0):
    model = _olmoe()
    harness.attach_adapters(model, "AT", hot=[[2, 6], [0, 7]], seed=seed)
    generator = torch.Generator().manual_seed(1)
    _, expert_cls = harness.adapter_classes()
    for module in model.modules():
        if isinstance(module, expert_cls):
            module.gate_up_B.data = torch.randn(module.gate_up_B.shape, generator=generator)
            module.down_B.data = torch.randn(module.down_B.shape, generator=generator)
    return model


def test_expert_delta_matches_a_merged_dense_reference(harness):
    """Merging ``scale * B_e A_e`` into the selected experts reproduces the adapted output."""
    model = _trained_expert_delta(harness)
    adapted = _logits(model, IDS)
    _, expert_cls = harness.adapter_classes()
    for module in list(model.modules()):
        if isinstance(module, expert_cls):
            base = module.base
            for slot, expert in enumerate(module.selected.tolist()):
                base.gate_up_proj.data[expert] += module.scale * (
                    module.gate_up_B[slot] @ module.gate_up_A[slot])
                base.down_proj.data[expert] += module.scale * (
                    module.down_B[slot] @ module.down_A[slot])
            module.enabled = False
    assert not torch.allclose(_logits(_olmoe(), IDS), adapted, atol=1e-4)
    assert torch.allclose(_logits(model, IDS), adapted, atol=1e-4, rtol=1e-4)


def test_expert_adapter_shapes_match_peft_param_wrapper(harness):
    """Same per-expert factor shapes peft 0.21's ParamWrapper builds on these parameters."""
    peft = pytest.importorskip("peft")
    wrapped = peft.get_peft_model(_olmoe(), peft.LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, target_modules=[],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))
    shapes = sorted(tuple(p.shape) for n, p in wrapped.named_parameters()
                    if "layers.0.mlp.experts" in n and "lora_" in n)
    ours = _olmoe()
    harness.attach_adapters(ours, "AE")
    expert = ours.model.layers[0].mlp.experts
    experts, rank = 8, 16

    def stacked_a(a):  # peft keeps A as (r * E, in)
        return tuple(a.reshape(experts * rank, -1).shape)

    def stacked_b(b):  # and B as (out, r * E)
        return tuple(b.permute(1, 0, 2).reshape(b.shape[1], -1).shape)

    assert sorted([stacked_a(expert.gate_up_A), stacked_b(expert.gate_up_B),
                   stacked_a(expert.down_A), stacked_b(expert.down_B)]) == shapes


def test_gradients_reach_adapters_only(harness):
    model = _trained_expert_delta(harness)
    loss = model(input_ids=IDS, labels=IDS).loss
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert grads and all(".lora_" in n or "_A" in n or "_B" in n for n in grads)
    assert any("gate_up_A" in n for n in grads) and any("q_proj.lora_B" in n for n in grads)


def test_hot_set_takes_busiest_quarter_ties_to_lower_index(harness):
    assert harness.hot_experts([5, 9, 9, 1, 0, 9, 2, 3]) == [1, 2]
    assert harness.hot_experts([0] * 8) == [0, 1]


def test_reference_answer_reads_the_metamath_marker(harness):
    assert harness.reference_answer("Work... The answer is: 18") == "18"
    assert harness.reference_answer("no marker here 18") is None


def _row(question, query):
    return {"original_question": question, "query": query, "response": "x The answer is: 1"}


def test_selection_takes_one_row_per_source_question(harness):
    rows = [_row("q1", "a"), _row("q1", "b"), _row("q2", "c"), _row(" q2 ", "d"),
            _row("q3", "e"), _row("q4", "f")]
    accepted, dropped = harness.select_rows(rows, range(len(rows)), 3, lambda row: None)
    assert accepted == [0, 2, 4]
    assert dropped == {"same_group": 2}


def test_a_rejected_row_does_not_claim_its_group(harness):
    rows = [_row("q1", "too long"), _row("q1", "fits"), _row("q2", "fits")]
    accepted, dropped = harness.select_rows(
        rows, range(3), 2, lambda row: "too_long" if row["query"] == "too long" else None)
    assert accepted == [1, 2] and dropped["too_long"] == 1


def test_split_sharing_a_source_question_across_splits_is_refused(harness):
    rows = [_row("q1", "a"), _row("q2", "b"), _row("q1", "c")]
    with pytest.raises(ValueError, match="group-disjoint"):
        harness.check_group_disjoint(rows, {"calibration": [], "train": [0, 1], "eval": [2]})
    harness.check_group_disjoint(rows, {"calibration": [], "train": [0], "eval": [1]})
