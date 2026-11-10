"""ZERO: training.stream_chunked_ce -- chunked causal-LM loss for streamed SFT."""
from __future__ import annotations

import pytest
import torch

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.stream_chunked_ce import (
    IGNORE_INDEX,
    chunked_causal_lm_loss,
    make_chunked_ce_trainer_class,
)


def _tiny_qwen35(vocab: int = 300):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3_5_text", vocab_size=vocab, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, full_attention_interval=4, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).double()


def _labels(ids: torch.Tensor) -> torch.Tensor:
    labels = ids.clone()
    labels[:, :9] = IGNORE_INDEX  # a prompt the loss must skip
    labels[1, -5:] = IGNORE_INDEX  # right padding on the second row
    return labels


@pytest.mark.parametrize("chunk", [1, 7, 64, 4096])
@pytest.mark.parametrize("with_items", [False, True])
def test_matches_transformers_loss_and_grads(chunk: int, with_items: bool) -> None:
    model = _tiny_qwen35()
    ids = torch.randint(0, 300, (2, 50), generator=torch.Generator().manual_seed(1))
    labels = _labels(ids)
    n_items = (labels[:, 1:] != IGNORE_INDEX).sum() + 3 if with_items else None

    kwargs = {"num_items_in_batch": n_items} if with_items else {}
    ref = model(input_ids=ids, labels=labels, **kwargs).loss
    ref.backward()
    ref_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    hidden = model.model(input_ids=ids).last_hidden_state
    loss = chunked_causal_lm_loss(hidden, model.lm_head, labels, n_items, chunk=chunk)
    loss.backward()

    # Both paths upcast the logits to fp32 (as transformers does), so they agree to
    # fp32 summation order, not to float64.
    assert torch.allclose(loss, ref, rtol=1e-6, atol=0)
    for name, g in ref_grads.items():
        got = dict(model.named_parameters())[name].grad
        assert torch.allclose(got, g, rtol=1e-5, atol=1e-9), name


def test_all_ignored_chunk_is_skipped_and_loss_is_finite() -> None:
    model = _tiny_qwen35()
    ids = torch.randint(0, 300, (1, 40))
    labels = ids.clone()
    labels[:, :30] = IGNORE_INDEX
    hidden = model.model(input_ids=ids).last_hidden_state
    loss = chunked_causal_lm_loss(hidden, model.lm_head, labels, chunk=8)
    ref = model(input_ids=ids, labels=labels).loss
    assert torch.allclose(loss, ref, rtol=1e-6, atol=0)


def test_trainer_subclass_keeps_base_and_name() -> None:
    class Base:
        pass

    cls = make_chunked_ce_trainer_class(Base)
    assert issubclass(cls, Base)
    assert cls.__name__ == "ChunkedCEBase"
    assert cls._soup_chunked_ce is True


_BASE = """
base: ./m
task: {task}
modality: text
data:
  train: ./d.jsonl
  format: alpaca
  max_length: 512
training:
  quantization: none
  batch_size: 1
  stream_layers: {stream}
  stream_chunked_ce: true
  l2l: {l2l}
{extra}
"""


def _cfg(task: str = "sft", stream: str = "true", l2l: str = "false", extra: str = ""):
    return load_config_from_string(_BASE.format(task=task, stream=stream, l2l=l2l, extra=extra))


def test_schema_accepts_streamed_sft() -> None:
    assert _cfg().training.stream_chunked_ce is True


def test_schema_refuses_without_streaming() -> None:
    with pytest.raises(Exception, match="stream_chunked_ce"):
        _cfg(stream="false")


def test_schema_refuses_l2l() -> None:
    with pytest.raises(Exception, match="stream_chunked_ce"):
        _cfg(l2l="true")


def test_schema_default_off() -> None:
    cfg = load_config_from_string(_BASE.format(task="sft", stream="true", l2l="false", extra="")
                                  .replace("  stream_chunked_ce: true\n", ""))
    assert cfg.training.stream_chunked_ce is False
