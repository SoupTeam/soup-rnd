"""Strict CPU adapter reload and token-weighted held-out causal CE mechanics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from consumer import batch_for_piece, validate_row
from prepare import token_accounting


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_adapter(model: object, path: Path, expected_sha: str,
                 expected_frozen_sha: str) -> dict:
    """Validate everything before copying; caller must pin base and adapter recipe."""
    import torch
    from check_consumer import state_hash
    from safetensors.torch import load_file

    if sha256(path) != expected_sha:
        raise ValueError("Adapter SHA mismatch")
    if state_hash(model, frozen_only=True) != expected_frozen_sha:
        raise ValueError("Frozen base mismatch")
    tensors = load_file(str(path), device="cpu")
    parameters = {key: value for key, value in model.named_parameters() if value.requires_grad}
    if set(tensors) != set(parameters):
        raise ValueError("Adapter names mismatch")
    for key, value in tensors.items():
        target = parameters[key]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError("Adapter shape/dtype mismatch")
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite adapter")
    with torch.no_grad():
        for key, value in tensors.items():
            parameters[key].copy_(value)
    if state_hash(model, frozen_only=True) != expected_frozen_sha:
        raise ValueError("Frozen base changed during reload")
    return {"status": "PASS_STRICT_ADAPTER_RELOAD", "adapter_sha256": expected_sha,
            "frozen_base_sha256": expected_frozen_sha}


def evaluate_cpu(model: object, rows: list[dict], *, expected_targets: int,
                 pad_multiple: int = 8) -> dict:
    """All-token CE; no generation/task/safety claims. No optimizer or gradients."""
    import torch
    import torch.nn.functional as functional

    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("CPU reference only")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate eval rows")
    total, targets, processed, padding = 0.0, 0, 0, 0
    logits_hash = hashlib.sha256()
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for row in rows:
                validate_row(row)
                piece = {"input_ids": row["input_ids"], "target_start": 1,
                         "target_stop": len(row["input_ids"])}
                batch = batch_for_piece(piece, pad_multiple=pad_multiple)
                count = token_accounting(**batch)
                ids = torch.tensor(batch["input_ids"])
                mask = torch.tensor(batch["attention_mask"])
                labels = torch.tensor(batch["labels"])
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                active = logits[:, :len(row["input_ids"])].contiguous()
                logits_hash.update(active.view(torch.uint8).numpy().tobytes())
                loss_logits = logits if logits.dtype == torch.float64 else logits.float()
                loss = functional.cross_entropy(
                    loss_logits[:, :-1].reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite eval loss")
                total += loss.item()
                targets += count["useful_loss_tokens"]
                processed += count["processed_input_slots"]
                padding += count["padded_input_slots"]
    finally:
        model.train(was_training)
    if targets != expected_targets:
        raise ValueError("Eval target count mismatch")
    mean = total / targets
    return {"sum_nll": total, "useful_targets": targets, "mean_ce": mean,
            "perplexity": math.exp(mean), "processed_slots": processed,
            "padding_slots": padding, "active_logits_sha256": logits_hash.hexdigest(),
            "rows": len(rows), "scope": "CPU causal CE mechanics only"}
