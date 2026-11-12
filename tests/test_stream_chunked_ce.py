"""ZERO: training.stream_chunked_ce -- chunked causal-LM loss for streamed SFT."""
from __future__ import annotations

import pytest
import torch

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.stream_chunked_ce import (
    IGNORE_INDEX,
    chunked_causal_lm_loss,
    chunked_model_loss,
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


def test_probe_loss_matches_model_loss() -> None:
    """The VRAM probe's loss_fn computes the same loss as model(labels=ids)."""
    model = _tiny_qwen35()
    ids = torch.randint(0, 300, (2, 40), generator=torch.Generator().manual_seed(3))
    assert torch.allclose(chunked_model_loss(model, ids), model(input_ids=ids, labels=ids).loss,
                          rtol=1e-6, atol=0)


def test_probe_accepts_loss_fn_and_is_cpu_noop() -> None:
    from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

    model = _tiny_qwen35()
    assert measure_step_peak_bytes(model, rows=1, seq_len=8, vocab_size=300, device="cpu",
                                   loss_fn=chunked_model_loss) is None


# --- streamed Cut Cross-Entropy (training.use_cut_ce with stream_layers) ---------------


def _fake_cce(monkeypatch):
    """A cut_cross_entropy stand-in with the same signature, exact in torch (CCE needs CUDA)."""
    import sys
    import types

    from torch.nn import functional as F  # noqa: N812

    calls = []

    def linear_cross_entropy(e, c, targets, ignore_index=-100, softcap=None,
                             reduction="mean", shift=False, filter_eps="high", impl="cce"):
        calls.append({"dtype": (e.dtype, c.dtype), "reduction": reduction, "shift": shift})
        logits = (e @ c.t()).float()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=ignore_index, reduction=reduction)

    mod = types.ModuleType("cut_cross_entropy")
    mod.linear_cross_entropy = linear_cross_entropy
    monkeypatch.setitem(sys.modules, "cut_cross_entropy", mod)
    return calls


@pytest.mark.parametrize("with_items", [False, True])
def test_cce_path_shift_and_normalisation(monkeypatch, with_items: bool) -> None:
    from soup_cli.utils.stream_chunked_ce import cce_causal_lm_loss

    calls = _fake_cce(monkeypatch)
    model = _tiny_qwen35()
    ids = torch.randint(0, 300, (2, 50), generator=torch.Generator().manual_seed(4))
    labels = _labels(ids)
    n_items = (labels[:, 1:] != IGNORE_INDEX).sum() + 3 if with_items else None
    kwargs = {"num_items_in_batch": n_items} if with_items else {}
    ref = model(input_ids=ids, labels=labels, **kwargs).loss
    hidden = model.model(input_ids=ids).last_hidden_state
    got = cce_causal_lm_loss(hidden, model.lm_head, labels, n_items)
    assert torch.allclose(got, ref, rtol=1e-6, atol=0)
    assert calls and calls[0]["reduction"] == "sum" and calls[0]["shift"] is False


def test_head_weight_reads_a_streamed_slot() -> None:
    from soup_cli.utils.stream_chunked_ce import _head_weight

    w = torch.randn(5, 3)

    class Pool:
        def wait(self, key):
            assert key == "lm_head"
            return w

    class Streamed:
        pool, key = Pool(), "lm_head"

    assert _head_weight(Streamed()) is w
    lin = torch.nn.Linear(3, 5, bias=False)
    assert _head_weight(lin) is lin.weight


def test_stream_loss_impl_selection() -> None:
    from types import SimpleNamespace

    from soup_cli.utils.stream_chunked_ce import stream_loss_impl

    def impl(stream: bool, cce: bool, chunked: bool):
        return stream_loss_impl(
            SimpleNamespace(stream_layers=stream, use_cut_ce=cce, stream_chunked_ce=chunked)
        )

    assert impl(False, True, False) is None
    assert impl(True, True, False) == "cce"
    assert impl(True, False, True) == "chunked"
    assert impl(True, False, False) is None


def test_schema_refuses_both_losses() -> None:
    with pytest.raises(Exception, match="pick one"):
        _cfg(extra="  use_cut_ce: true")


def test_schema_streamed_cce_refuses_l2l() -> None:
    cfg_text = _BASE.format(
        task="sft", stream="true", l2l="true", extra="  use_cut_ce: true"
    ).replace("  stream_chunked_ce: true\n", "")
    with pytest.raises(Exception, match="use_cut_ce"):
        load_config_from_string(cfg_text)


def test_trainer_class_for_cce_names_itself() -> None:
    class Base:
        pass

    cls = make_chunked_ce_trainer_class(Base, "cce")
    assert cls.__name__ == "CutCEBase" and cls._soup_loss_impl == "cce"
