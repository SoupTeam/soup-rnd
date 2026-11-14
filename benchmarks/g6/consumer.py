"""Ordered token-window planning and a CPU-only training reference for G6.

No CUDA runner is exposed. No model weights are downloaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Iterator


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_row(row: dict) -> None:
    ids = row.get("input_ids")
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("Missing row identity")
    if not isinstance(ids, list) or not 2 <= len(ids) <= 2048:
        raise ValueError("Sequence length must be in2..2048")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Invalid token id")


def inspect_source(root: Path, arm: str, manifest_sha256: str) -> dict:
    manifest_path = root / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
        raise ValueError("Manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    # Arm is checked against fixed names before using it as a path component.
    allowed = {"baseline", "coherence_filter", "length_curriculum",
               "reverse_length_curriculum"}
    if arm not in allowed:
        raise ValueError("Unknown training arm")
    path = root / f"{arm}.jsonl"
    receipt = manifest["outputs"][arm]
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("Training input SHA mismatch")
    count = useful = 0
    seen = set()
    previous_length = None
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            validate_row(row)
            if row["id"] in seen:
                raise ValueError("Repeated source identity")
            seen.add(row["id"])
            length = len(row["input_ids"])
            if previous_length is not None:
                if arm == "length_curriculum" and length < previous_length:
                    raise ValueError("Curriculum order was changed")
                if arm == "reverse_length_curriculum" and length > previous_length:
                    raise ValueError("Reverse curriculum order was changed")
            previous_length = length
            count += 1
            useful += length - 1
    if count != receipt["rows"] or useful != receipt["useful_tokens"]:
        raise ValueError("Source counters differ from manifest")
    if useful != manifest["budget"]:
        raise ValueError("Source does not meet exact useful-token budget")
    return {"path": str(path), "sha256": receipt["sha256"], "rows": count,
            "useful_tokens": useful}


def source_rows(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def windows(rows: Iterable[dict], budget: int, targets_per_update: int) -> Iterator[dict]:
    """Preserve row/target order, with each causal target supervised exactly once.

    A row crossing an update boundary is forwarded again with its full causal
    prefix and disjoint label masks. This repeats context compute, not targets.
    """
    if budget <= 0 or targets_per_update <= 0:
        raise ValueError("Positive budget and update size required")
    pieces, used, consumed, update = [], 0, 0, 0
    seen = set()
    for index, row in enumerate(rows):
        validate_row(row)
        if row["id"] in seen:
            raise ValueError("Repeated source identity")
        seen.add(row["id"])
        ids = row["input_ids"]
        start = 1
        while start < len(ids):
            take = min(len(ids) - start, targets_per_update - used)
            if consumed + used + take > budget:
                raise ValueError("Input exceeds budget; implicit truncation is forbidden")
            stop = start + take
            pieces.append({"row_id": row["id"], "row_index": index,
                           "target_start": start, "target_stop": stop,
                           "source_length": len(ids), "input_ids": ids[:stop],
                           "source_input_sha256": fingerprint(ids)})
            start, used = stop, used + take
            if used == targets_per_update:
                update += 1
                yield {"update": update, "tokens_before": consumed,
                       "useful_tokens": used, "pieces": pieces}
                consumed += used
                pieces, used = [], 0
    if consumed + used != budget:
        raise ValueError("Insufficient input: no cycling or automatic replay")
    if used:
        yield {"update": update + 1, "tokens_before": consumed,
               "useful_tokens": used, "pieces": pieces}


def batch_for_piece(piece: dict, pad_multiple: int = 8) -> dict:
    """Use the SAME causal prefix, right-pad, ignore BOS/context/pad labels."""
    if pad_multiple <= 0:
        raise ValueError("Padding multiple must be positive")
    ids = piece["input_ids"]
    start, stop = piece["target_start"], piece["target_stop"]
    if not 1 <= start < stop == len(ids):
        raise ValueError("Invalid supervised target span")
    width = math.ceil(len(ids) / pad_multiple) * pad_multiple
    padding = width - len(ids)
    labels = [-100] * start + ids[start:stop] + [-100] * padding
    return {"input_ids": [ids + [0] * padding], "labels": [labels],
            "attention_mask": [[1] * len(ids) + [0] * padding]}


def audit_plan(planned: Iterable[dict], budget: int, targets_per_update: int) -> dict:
    """Recount disjoint spans, source order, masks and planned input overhead."""
    from prepare import token_accounting

    cursors, source_lengths, order = {}, {}, []
    trace_hash = hashlib.sha256()
    totals = {"useful_loss_tokens": 0, "processed_input_slots": 0,
              "nonpadding_input_tokens": 0, "padded_input_slots": 0}
    updates = pieces = last_update_tokens = 0
    for window in planned:
        updates += 1
        expected = min(targets_per_update, budget - totals["useful_loss_tokens"])
        if window["update"] != updates or window["useful_tokens"] != expected:
            raise ValueError("Optimizer window sizes/order disagree")
        if window["tokens_before"] != totals["useful_loss_tokens"]:
            raise ValueError("Token progress differs from audited labels")
        actual = 0
        for piece in window["pieces"]:
            identity = piece["row_id"]
            if identity not in cursors:
                if piece["row_index"] != len(order):
                    raise ValueError("Source order differs")
                order.append(identity)
            elif order[-1] != identity:
                raise ValueError("A finished row was replayed")
            if piece["target_start"] != cursors.get(identity, 1):
                raise ValueError("Duplicate or missing target span")
            if piece["source_length"] != source_lengths.get(identity, piece["source_length"]):
                raise ValueError("Source length changed")
            cursors[identity] = piece["target_stop"]
            source_lengths[identity] = piece["source_length"]
            counts = token_accounting(**batch_for_piece(piece))
            actual += counts["useful_loss_tokens"]
            for name, value in counts.items():
                totals[name] += value
            trace_hash.update(json.dumps(piece, sort_keys=True).encode())
            pieces += 1
        if actual != window["useful_tokens"]:
            raise ValueError("Loss masks differ from window budget")
        last_update_tokens = actual
    if totals["useful_loss_tokens"] != budget or cursors != source_lengths:
        raise ValueError("Target coverage is incomplete")
    return {"status": "PASS_PLANNED_CONSUMPTION_ONLY", "rows": len(order),
            "optimizer_updates": updates, "last_update_useful_tokens": last_update_tokens,
            "forward_pieces": pieces, "tokens": totals,
            "source_order_sha256": fingerprint(order),
            "planned_piece_stream_sha256": trace_hash.hexdigest(),
            "runtime_training_not_proven": True}


def cpu_train(model: object, optimizer: object, planned: Iterable[dict], *,
              budget: int, base_lr: float, trace: Path, clip: float = 1.0) -> dict:
    """CPU-only numerical reference. CUDA AMP requires a separate future gate."""
    import torch
    import torch.nn.functional as functional
    from prepare import token_accounting

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("CPU reference only; CUDA/AMP is not validated")
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
        raise ValueError("CPU reference requires dropout zero")
    if budget <= 0 or base_lr <= 0 or clip <= 0:
        raise ValueError("Positive training bounds required")
    consumed = slots = padded = updates = 0
    model.train()
    with trace.open("x") as log:
        for window in planned:
            if window["tokens_before"] != consumed:
                raise ValueError("Progress does not match planned order")
            optimizer.zero_grad(set_to_none=True)
            lr = base_lr * (1 - consumed / budget)
            for group in optimizer.param_groups:
                group["lr"] = lr
            actual, sum_loss = 0, 0.0
            for piece in window["pieces"]:
                batch = batch_for_piece(piece)
                counts = token_accounting(**batch)
                tensors = {name: torch.tensor(value, dtype=torch.long)
                           for name, value in batch.items()}
                logits = model(input_ids=tensors["input_ids"],
                               attention_mask=tensors["attention_mask"], use_cache=False).logits
                # float32 is the real recipe; preserve float64 for numerical reference tests.
                if logits.dtype != torch.float64:
                    logits = logits.float()
                loss_sum = functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    tensors["labels"][:, 1:].reshape(-1), reduction="sum", ignore_index=-100)
                if not torch.isfinite(loss_sum):
                    raise ValueError("Nonfinite loss; no step/retry is accepted")
                (loss_sum / window["useful_tokens"]).backward()
                actual += counts["useful_loss_tokens"]
                slots += counts["processed_input_slots"]
                padded += counts["padded_input_slots"]
                sum_loss += loss_sum.detach().item()
                log.write(json.dumps({"event": "forward_backward", "update": window["update"],
                    "row_id": piece["row_id"], "row_index": piece["row_index"],
                    "target_start": piece["target_start"], "target_stop": piece["target_stop"],
                    "actual_batch_sha256": fingerprint(batch), "tokens": counts}) + "\n")
            if actual != window["useful_tokens"] or consumed + actual > budget:
                raise ValueError("Actual tokens differ from plan")
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
            if norm.item() == 0:
                raise ValueError("Zero trainable gradient; refusing a fake training step")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise ValueError("Nonfinite parameters after optimizer step")
            consumed += actual
            updates += 1
            log.write(json.dumps({"event": "optimizer_step", "update": window["update"],
                "actual_useful_tokens": actual, "consumed_useful_tokens": consumed,
                "lr": lr, "gradient_norm_before_clip": norm.item(),
                "mean_loss": sum_loss / actual}) + "\n")
            log.flush()
    if consumed != budget:
        raise ValueError("Incomplete consumption; no completion claim")
    return {"status": "CPU_REFERENCE_COMPLETED", "actual_useful_tokens": consumed,
            "applied_optimizer_updates": updates, "processed_input_slots": slots,
            "padded_input_slots": padded, "AMP_exercised": False,
            "quality_or_Mistral7B_training_claim": False}


def audit_trace(trace: Path, planned: Iterable[dict], budget: int, base_lr: float) -> dict:
    """Verify persisted actual-batch traces against the ordered plan, fail closed."""
    from prepare import token_accounting

    records = iter(json.loads(line) for line in trace.read_text().splitlines())
    consumed = updates = 0
    for window in planned:
        for piece in window["pieces"]:
            actual = next(records, None)
            expected = {"event": "forward_backward", "update": window["update"],
                        "row_id": piece["row_id"], "row_index": piece["row_index"],
                        "target_start": piece["target_start"],
                        "target_stop": piece["target_stop"],
                        "actual_batch_sha256": fingerprint(batch_for_piece(piece)),
                        "tokens": token_accounting(**batch_for_piece(piece))}
            if actual != expected:
                raise ValueError("Actual row/target/mask trace differs from plan")
        record = next(records, None)
        consumed += window["useful_tokens"]
        expected_lr = base_lr * (1 - window["tokens_before"] / budget)
        if not record or record.get("event") != "optimizer_step":
            raise ValueError("Missing applied optimizer step")
        if (record["update"] != window["update"]
                or record["actual_useful_tokens"] != window["useful_tokens"]
                or record["consumed_useful_tokens"] != consumed
                or record["lr"] != expected_lr):
            raise ValueError("Actual optimizer exposure/LR differs from plan")
        if (not math.isfinite(record["mean_loss"])
                or not math.isfinite(record["gradient_norm_before_clip"])
                or record["gradient_norm_before_clip"] <= 0):
            raise ValueError("Invalid loss/gradient evidence")
        updates += 1
    if next(records, None) is not None or consumed != budget:
        raise ValueError("Unexpected extra/missing training records")
    return {"status": "PASS_CPU_RUNTIME_TRACE", "actual_useful_tokens": consumed,
            "applied_optimizer_updates": updates, "source_order_and_masks_match": True,
            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}
