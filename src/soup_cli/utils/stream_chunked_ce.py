"""Chunked causal-LM cross-entropy for streamed SFT (ZERO, ``training.stream_chunked_ce``).

The transformers causal-LM loss materialises ``logits = lm_head(hidden)`` for the
whole batch x seq x vocab and upcasts it to fp32. On a 248k-vocabulary model
(Qwen3.5 / Qwen3.8) at seq 2048 that is ~2 GB per copy and the largest VRAM term
of a streamed run (Soup's pre-flight charges it at ~7 GB with retention). Here the
loss is summed over chunks of positions; each chunk's logits are built inside a
non-reentrant checkpoint, so backward recomputes them instead of keeping them.

Semantics match ``transformers.loss.loss_utils.ForCausalLMLoss`` for the SFT path:
labels shifted left by one with ``-100`` appended, ``ignore_index=-100``, fp32
logits, ``sum / num_items_in_batch`` when the trainer passes it, else the mean over
non-ignored positions.
"""
from __future__ import annotations

from typing import Any, Optional

#: Positions per chunk. 256 x 248 320 x 4 B = 254 MB of fp32 logits live at a time.
CE_CHUNK = 256
IGNORE_INDEX = -100


def chunked_causal_lm_loss(
    hidden: Any,
    lm_head: Any,
    labels: Any,
    num_items_in_batch: Optional[Any] = None,
    *,
    chunk: int = CE_CHUNK,
) -> Any:
    """Causal-LM loss from final hidden states without the full logits tensor."""
    import torch
    from torch.nn import functional as F  # noqa: N812
    from torch.utils.checkpoint import checkpoint

    if chunk <= 0:
        raise ValueError("chunk must be positive")
    shift = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].to(hidden.device)

    def piece(h: Any, y: Any) -> Any:
        logits = lm_head(h).float()
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum",
        )

    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for s in range(0, hidden.shape[1], chunk):
        y = shift[:, s:s + chunk]
        if bool((y == IGNORE_INDEX).all()):
            continue
        total = total + checkpoint(piece, hidden[:, s:s + chunk], y, use_reentrant=False)
    if num_items_in_batch is None:
        denom = (shift != IGNORE_INDEX).sum()
    else:
        denom = num_items_in_batch
    if torch.is_tensor(denom):
        denom = denom.to(total.device)
    return total / denom


def _head_weight(head: Any) -> Any:
    """The output projection's weight, also when the head is a streamed large layer.

    A streamed ``lm_head`` (``StreamedLargeLayer``) holds no parameter; its weight is a
    view of the shared device slot, returned by ``pool.wait(key)`` once the copy landed.
    """
    pool, key = getattr(head, "pool", None), getattr(head, "key", None)
    if pool is not None and key is not None:
        return pool.wait(key)
    return head.weight


def cce_causal_lm_loss(
    hidden: Any,
    lm_head: Any,
    labels: Any,
    num_items_in_batch: Optional[Any] = None,
) -> Any:
    """Causal-LM loss through Apple's Cut Cross-Entropy kernel (``cut_cross_entropy``).

    Same shift, ``-100`` ignore and normalisation as :func:`chunked_causal_lm_loss`; the
    logits are never materialised. CCE's default gradient filtering (``filter_eps="high"``)
    is kept, so the gradient is CCE's, not bit-equal to the full-logits one.
    """
    import torch
    from cut_cross_entropy import linear_cross_entropy
    from torch.nn import functional as F  # noqa: N812

    weight = _head_weight(lm_head)
    shift = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].to(hidden.device)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        total = linear_cross_entropy(
            hidden.to(weight.dtype), weight, shift,
            ignore_index=IGNORE_INDEX, reduction="sum",
        ).float()
    denom = (shift != IGNORE_INDEX).sum() if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(denom):
        denom = denom.to(total.device)
    return total / denom


#: Streamed-SFT loss implementations: ``training.stream_chunked_ce`` -> "chunked",
#: ``training.use_cut_ce`` with ``stream_layers`` -> "cce".
LOSS_IMPLS = {"chunked": chunked_causal_lm_loss, "cce": cce_causal_lm_loss}


def stream_loss_impl(tcfg: Any) -> Optional[str]:
    """Which streamed-SFT loss a training config selects, or None for the model's own."""
    if not getattr(tcfg, "stream_layers", False):
        return None
    if getattr(tcfg, "use_cut_ce", False):
        return "cce"
    if getattr(tcfg, "stream_chunked_ce", False):
        return "chunked"
    return None


def _causal_parts(model: Any) -> tuple[Any, Any]:
    """(decoder, lm_head) of a possibly PEFT-wrapped causal LM."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(base, "model", None)
    head = getattr(base, "lm_head", None)
    if decoder is None or head is None:
        raise ValueError(
            "training.stream_chunked_ce needs a causal LM with `.model` and `.lm_head` "
            f"(got {type(base).__name__})"
        )
    return decoder, head


_DECODER_KEYS = ("input_ids", "attention_mask", "position_ids")


def model_loss_fn(impl: str = "chunked") -> Any:
    """``(model, input_ids) -> loss`` through ``impl``: the VRAM probe's step shape."""
    loss_impl = LOSS_IMPLS[impl]

    def fn(model: Any, input_ids: Any) -> Any:
        decoder, head = _causal_parts(model)
        hidden = decoder(input_ids=input_ids, use_cache=False).last_hidden_state
        return loss_impl(hidden, head, input_ids)

    return fn


def chunked_model_loss(model: Any, input_ids: Any) -> Any:
    """Labels-equal-inputs causal-LM loss of ``model`` through the chunked path."""
    return model_loss_fn("chunked")(model, input_ids)


def _hidden_through_wrapper(model: Any, decoder: Any, kwargs: dict) -> Any:
    """The decoder's last hidden state, from a forward through ``model``'s own wrapper.

    A wrapped model (DDP) must be entered through its ``forward``: calling the unwrapped
    decoder directly skips DDP's forward bookkeeping, so the ranks' adapter gradients are not
    averaged. The wrapper runs the full causal LM with ``logits_to_keep=1`` (one position of
    logits, not the whole vocabulary row per token); a hook on the decoder hands back its
    output, still in the graph.
    """
    captured: dict = {}

    def keep(_module: Any, _args: Any, output: Any) -> None:
        captured["hidden"] = getattr(output, "last_hidden_state", None)
        if captured["hidden"] is None:
            captured["hidden"] = output[0]

    handle = decoder.register_forward_hook(keep)
    try:
        model(**kwargs, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("training.stream_chunked_ce: the wrapped forward never ran the decoder")
    return captured["hidden"]


def make_chunked_ce_trainer_class(base_cls: type, impl: str = "chunked") -> type:
    """Subclass ``base_cls`` (TRL's SFTTrainer or a subclass) whose ``compute_loss`` uses ``impl``.

    Keeps TRL's ``num_tokens`` metric; skips ``entropy`` and ``mean_token_accuracy``,
    which need the full logits.
    """

    loss_impl = LOSS_IMPLS[impl]

    class ChunkedCETrainer(base_cls):  # type: ignore[misc, valid-type]
        _soup_chunked_ce = True
        _soup_loss_impl = impl

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            mode = "train" if self.model.training else "eval"
            inputs.pop("_prediction_loss_only", None)
            labels = inputs["labels"]
            unwrapped = self.accelerator.unwrap_model(model)
            decoder, head = _causal_parts(unwrapped)
            kwargs = {k: inputs[k] for k in _DECODER_KEYS if k in inputs}
            if model is unwrapped:
                hidden = decoder(**kwargs, use_cache=False).last_hidden_state
            else:
                hidden = _hidden_through_wrapper(model, decoder, kwargs)
            loss = loss_impl(hidden, head, labels, num_items_in_batch)
            if (
                num_items_in_batch is not None
                and getattr(self.args, "average_tokens_across_devices", False)
                and self.accelerator.num_processes > 1
            ):
                loss = loss * self.accelerator.num_processes
            if mode == "train" and hasattr(self, "_total_train_tokens"):
                if "attention_mask" in inputs:
                    n = inputs["attention_mask"].sum()
                else:
                    n = inputs["input_ids"].new_tensor(inputs["input_ids"].numel())
                self._total_train_tokens += self.accelerator.gather_for_metrics(n).sum().item()
                self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
            return (loss, None) if return_outputs else loss

    prefix = "ChunkedCE" if impl == "chunked" else "CutCE"
    ChunkedCETrainer.__name__ = f"{prefix}{base_cls.__name__}"
    ChunkedCETrainer.__qualname__ = ChunkedCETrainer.__name__
    return ChunkedCETrainer
