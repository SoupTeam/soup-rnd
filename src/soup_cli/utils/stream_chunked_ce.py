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
    import torch.nn.functional as F
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


def chunked_model_loss(model: Any, input_ids: Any) -> Any:
    """Labels-equal-inputs causal-LM loss of ``model`` through the chunked path.

    The shape of the VRAM probe's step (``measure_step_peak_bytes(loss_fn=...)``).
    """
    decoder, head = _causal_parts(model)
    hidden = decoder(input_ids=input_ids, use_cache=False).last_hidden_state
    return chunked_causal_lm_loss(hidden, head, input_ids)


def make_chunked_ce_trainer_class(base_cls: type) -> type:
    """Subclass ``base_cls`` (TRL's SFTTrainer or a subclass) with a chunked-CE ``compute_loss``.

    Keeps TRL's ``num_tokens`` metric; skips ``entropy`` and ``mean_token_accuracy``,
    which need the full logits.
    """

    class ChunkedCETrainer(base_cls):  # type: ignore[misc, valid-type]
        _soup_chunked_ce = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            mode = "train" if self.model.training else "eval"
            inputs.pop("_prediction_loss_only", None)
            labels = inputs["labels"]
            decoder, head = _causal_parts(self.accelerator.unwrap_model(model))
            kwargs = {k: inputs[k] for k in _DECODER_KEYS if k in inputs}
            hidden = decoder(**kwargs, use_cache=False).last_hidden_state
            loss = chunked_causal_lm_loss(hidden, head, labels, num_items_in_batch)
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

    ChunkedCETrainer.__name__ = f"ChunkedCE{base_cls.__name__}"
    ChunkedCETrainer.__qualname__ = ChunkedCETrainer.__name__
    return ChunkedCETrainer
