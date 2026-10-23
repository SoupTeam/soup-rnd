"""Common ordered consumer for CPU conformance and prospective CUDA AMP pilot.

CUDA code is prepared, not validated by any CPU result. No retries or resume.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Iterable

from consumer import batch_for_piece, fingerprint
from prepare import token_accounting


def train(
    model: object,
    optimizer: object,
    planned: Iterable[dict],
    *,
    budget: int,
    base_lr: float,
    trace: Path,
    telemetry: Path,
    device: str,
    guard: object,
    clip: float = 1.0,
    lr_budget: int | None = None,
) -> dict:
    import torch
    import torch.nn.functional as functional

    if device not in {"cpu", "cuda:0"}:
        raise ValueError("Only CPU reference or a single cuda:0 device is supported")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(str(parameter.device) != device for parameter in model.parameters()):
        raise ValueError("Model placement differs from declared single device")
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
        raise ValueError("Dropout must be zero")
    if budget <= 0 or base_lr <= 0 or clip <= 0:
        raise ValueError("Positive training bounds required")
    horizon = budget if lr_budget is None else lr_budget
    if horizon < budget:
        raise ValueError("LR horizon must cover the run budget")
    cuda = device == "cuda:0"
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=128.0, growth_interval=2147483647, enabled=cuda
    )
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    consumed = slots = padded = updates = 0
    model.train()
    with trace.open("x") as log, telemetry.open("x") as sensors:
        for window in planned:
            guard()
            sensors.write(
                json.dumps(
                    {
                        "event": "update_attempt",
                        "update": window["update"],
                        "targets_before": consumed,
                    }
                )
                + "\n"
            )
            sensors.flush()
            if window["tokens_before"] != consumed or window["update"] != updates + 1:
                raise ValueError("Progress does not match planned order")
            optimizer.zero_grad(set_to_none=True)
            lr = base_lr * (1 - consumed / horizon)
            for group in optimizer.param_groups:
                group["lr"] = lr
            actual, sum_loss = 0, 0.0
            for piece in window["pieces"]:
                guard()
                batch = batch_for_piece(piece)
                counts = token_accounting(**batch)
                tensors = {
                    name: torch.tensor(value, dtype=torch.long, device=device)
                    for name, value in batch.items()
                }
                amp = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if cuda
                    else contextlib.nullcontext()
                )
                with amp:
                    logits = model(
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        use_cache=False,
                    ).logits
                    loss_logits = logits if logits.dtype == torch.float64 else logits.float()
                    loss_sum = functional.cross_entropy(
                        loss_logits[:, :-1].reshape(-1, logits.shape[-1]),
                        tensors["labels"][:, 1:].reshape(-1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                if not torch.isfinite(loss_sum):
                    raise ValueError("Nonfinite loss; no automatic retry")
                scaler.scale(loss_sum / window["useful_tokens"]).backward()
                actual += counts["useful_loss_tokens"]
                slots += counts["processed_input_slots"]
                padded += counts["padded_input_slots"]
                sum_loss += loss_sum.detach().item()
                log.write(
                    json.dumps(
                        {
                            "event": "forward_backward",
                            "update": window["update"],
                            "row_id": piece["row_id"],
                            "row_index": piece["row_index"],
                            "target_start": piece["target_start"],
                            "target_stop": piece["target_stop"],
                            "actual_batch_sha256": fingerprint(batch),
                            "tokens": counts,
                        }
                    )
                    + "\n"
                )
                log.flush()
            if actual != window["useful_tokens"] or consumed + actual > budget:
                raise ValueError("Actual target count differs from plan")
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
            if norm.item() == 0:
                raise ValueError("Zero gradient; refusing a fake training step")
            scale = scaler.get_scale()
            # unscale + finite norm must succeed before this step: overflow is fatal, never skipped.
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() != scale:
                raise ValueError("AMP scale changed; invalid exposure, no retry")
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise ValueError("Nonfinite trainable parameter")
            consumed += actual
            updates += 1
            log.write(
                json.dumps(
                    {
                        "event": "optimizer_step",
                        "update": window["update"],
                        "actual_useful_tokens": actual,
                        "consumed_useful_tokens": consumed,
                        "lr": lr,
                        "gradient_norm_before_clip": norm.item(),
                        "mean_loss": sum_loss / actual,
                    }
                )
                + "\n"
            )
            log.flush()
            if cuda:
                torch.cuda.synchronize()
            sensors.write(
                json.dumps(
                    {
                        "applied_update": updates,
                        "attempted_update": updates,
                        "amp_scale": scale,
                        "amp_skipped_steps": 0,
                        "elapsed_s": time.monotonic() - started,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                        if cuda
                        else None,
                        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()
                        if cuda
                        else None,
                    }
                )
                + "\n"
            )
            sensors.flush()
    if consumed != budget:
        raise ValueError("Incomplete consumption; no completion claim")
    return {
        "status": "CONSUMPTION_COMPLETED",
        "actual_useful_tokens": consumed,
        "applied_optimizer_updates": updates,
        "processed_input_slots": slots,
        "padded_input_slots": padded,
        "AMP_exercised": cuda,
        "elapsed_s": time.monotonic() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "no_resume_or_automatic_retry": True,
    }


def evaluate(
    model: object,
    rows: Iterable[dict],
    *,
    expected_targets: int,
    trace: Path,
    device: str,
    guard: object,
) -> dict:
    import torch
    import torch.nn.functional as functional

    total, targets, count, slots, padding = 0.0, 0, 0, 0, 0
    was_training = model.training
    model.eval()
    cuda = device == "cuda:0"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    seen = set()
    try:
        with trace.open("x") as log, torch.inference_mode():
            for row in rows:
                guard()
                from consumer import validate_row

                validate_row(row)
                if row["id"] in seen:
                    raise ValueError("Duplicate eval identity")
                seen.add(row["id"])
                batch = batch_for_piece(
                    {
                        "input_ids": row["input_ids"],
                        "target_start": 1,
                        "target_stop": len(row["input_ids"]),
                    }
                )
                counts = token_accounting(**batch)
                tensors = {
                    name: torch.tensor(value, dtype=torch.long, device=device)
                    for name, value in batch.items()
                }
                amp = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if cuda
                    else contextlib.nullcontext()
                )
                with amp:
                    logits = model(
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        use_cache=False,
                    ).logits
                    loss_logits = logits if logits.dtype == torch.float64 else logits.float()
                    nll = functional.cross_entropy(
                        loss_logits[:, :-1].reshape(-1, logits.shape[-1]),
                        tensors["labels"][:, 1:].reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                if not torch.isfinite(nll):
                    raise ValueError("Nonfinite eval loss")
                value = nll.item()
                total += value
                targets += counts["useful_loss_tokens"]
                slots += counts["processed_input_slots"]
                padding += counts["padded_input_slots"]
                count += 1
                log.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "actual_batch_sha256": fingerprint(batch),
                            "sum_nll": value,
                            "tokens": counts,
                        }
                    )
                    + "\n"
                )
                log.flush()
    finally:
        model.train(was_training)
    if targets != expected_targets or not targets:
        raise ValueError("Eval coverage differs from frozen contract")
    if cuda:
        torch.cuda.synchronize()
    return {
        "mean_ce": total / targets,
        "sum_nll": total,
        "useful_targets": targets,
        "rows": count,
        "processed_slots": slots,
        "padding_slots": padding,
        "elapsed_s": time.monotonic() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
    }


def audit_runtime(
    trace: Path, planned: Iterable[dict], budget: int, base_lr: float, lr_budget: int
) -> dict:
    """Independent persisted row/mask and applied-update audit, including pilot LR horizon."""
    import math

    consumed = updates = 0
    with trace.open() as handle:
        records = iter(json.loads(line) for line in handle)
        for window in planned:
            for piece in window["pieces"]:
                record = next(records, None)
                expected = {
                    "event": "forward_backward",
                    "update": window["update"],
                    "row_id": piece["row_id"],
                    "row_index": piece["row_index"],
                    "target_start": piece["target_start"],
                    "target_stop": piece["target_stop"],
                    "actual_batch_sha256": fingerprint(batch_for_piece(piece)),
                    "tokens": token_accounting(**batch_for_piece(piece)),
                }
                if record != expected:
                    raise ValueError("Actual consumed row/mask differs from frozen plan")
            record = next(records, None)
            consumed += window["useful_tokens"]
            updates += 1
            if (
                not record
                or record.get("event") != "optimizer_step"
                or record["update"] != updates
                or record["actual_useful_tokens"] != window["useful_tokens"]
                or record["consumed_useful_tokens"] != consumed
                or record["lr"] != base_lr * (1 - window["tokens_before"] / lr_budget)
                or not math.isfinite(record["mean_loss"])
                or not math.isfinite(record["gradient_norm_before_clip"])
                or record["gradient_norm_before_clip"] <= 0
            ):
                raise ValueError("Invalid optimizer/loss/gradient/LR trace")
        if next(records, None) is not None or consumed != budget:
            raise ValueError("Incomplete or extra runtime records")
    return {
        "status": "PASS_CONSUMED_ORDER_MASKS_AND_UPDATE_AUDIT",
        "useful_targets": consumed,
        "applied_updates": updates,
        "lr_horizon": lr_budget,
    }
