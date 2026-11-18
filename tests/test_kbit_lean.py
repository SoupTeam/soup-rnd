"""ZERO-2: training.kbit_keep_16bit and training.embed_on_host (soup_cli.trainer.kbit_lean)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from soup_cli.config.schema import SoupConfig  # noqa: E402
from soup_cli.trainer.kbit_lean import HOST_EMBED_ATTR, embedding_on_host, prepare_kbit  # noqa: E402


def _tiny(tied: bool = False):
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, tie_word_embeddings=tied)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).to(torch.bfloat16)
    model.is_loaded_in_4bit = True  # what PEFT checks before it enables checkpointing
    return model


def test_peft_default_upcasts_and_keep_16bit_does_not():
    up = prepare_kbit(_tiny(), use_gradient_checkpointing=False, keep_16bit=False)
    assert {p.dtype for p in up.parameters()} == {torch.float32}
    keep = prepare_kbit(_tiny(), use_gradient_checkpointing=False, keep_16bit=True)
    assert {p.dtype for p in keep.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad for p in keep.parameters())
    ref = _tiny()
    for (n, a), (_, b) in zip(keep.named_parameters(), ref.named_parameters()):
        assert torch.equal(a, b), n  # the values are put back untouched


def test_keep_16bit_still_enables_input_grads_with_checkpointing():
    model = prepare_kbit(_tiny(), use_gradient_checkpointing=True, keep_16bit=True)
    assert model.is_gradient_checkpointing
    out = model.get_input_embeddings()(torch.tensor([[1, 2, 3]]))
    assert out.requires_grad


def test_embedding_on_host_same_rows_and_hooks_fire():
    model = prepare_kbit(_tiny(), use_gradient_checkpointing=True, keep_16bit=True)
    ids = torch.tensor([[5, 0, 63, 7]])
    before = model.get_input_embeddings()(ids).detach().clone()
    logits_before = model(input_ids=ids).logits.detach().clone()
    embedding_on_host(model)
    emb = model.get_input_embeddings()
    assert getattr(emb, HOST_EMBED_ATTR) and emb.weight.device.type == "cpu"
    after = emb(ids)
    assert after.requires_grad  # PEFT's enable_input_require_grads hook still fires
    assert torch.equal(after.detach(), before)
    assert torch.equal(model(input_ids=ids).logits, logits_before)
    assert embedding_on_host(model) == 0  # idempotent
    # a later model.to(...) (the Trainer's device placement) must leave the host weight alone
    weight = emb.weight
    model.to(torch.float32)
    assert emb.weight is weight and weight.dtype == torch.bfloat16 and weight.device.type == "cpu"
    assert model.lm_head.weight.dtype == torch.float32


def test_embedding_on_host_refuses_tied_and_trainable():
    tied = prepare_kbit(_tiny(tied=True), use_gradient_checkpointing=False, keep_16bit=True)
    with pytest.raises(ValueError, match="untied"):
        embedding_on_host(tied)
    model = _tiny()
    model.get_input_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        embedding_on_host(model)


def _cfg(**training):
    return {"base": "some/model", "task": "sft", "data": {"train": "x.jsonl"},
            "training": {"quantization": "4bit", **training}}


@pytest.mark.parametrize("field", ["kbit_keep_16bit", "embed_on_host"])
def test_config_gates(field):
    SoupConfig(**_cfg(**{field: True}))
    with pytest.raises(ValueError, match="quantization"):
        SoupConfig(**_cfg(**{field: True, "quantization": "none"}))
    with pytest.raises(ValueError, match="resident path"):
        SoupConfig(**_cfg(**{field: True, "stream_layers": True}))
    with pytest.raises(ValueError, match="task='sft'"):
        SoupConfig(**{**_cfg(**{field: True}), "task": "dpo"})
