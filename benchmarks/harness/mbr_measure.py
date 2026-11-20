"""Measurement runner pieces for the exact-MBR campaign on Kanana (harness only, not shipped).

Pure logic that does not need a GPU: the four windows, the bit ladder, the mode guard (speed only
in normal mode, bits only in deterministic mode), OOM turned into a table row with its peaks,
MBPP-train records with a dataset hash, and the median of timed steps after warm-up. The GPU
driver calls these around ``L2LStep.run``.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any, Callable, Dict, List, Sequence, Tuple

WINDOWS: Tuple[Tuple[int, int], ...] = ((8, 64), (8, 512), (64, 64), (64, 512))
BIT_LADDER: Tuple[int, ...] = (1, 2, 4, 8)
A0_MAX_K = 8
PEAK_KEYS = ("peak_allocated", "peak_reserved", "peak_nvml", "peak_ram")


class MixedModeError(RuntimeError):
    """Raised when a measurement kind is requested in the wrong torch mode."""


def check_mode(kind: str, *, deterministic: bool) -> str:
    """The mode label for ``kind`` ("speed" or "bits"); a mixed mode is refused."""
    if kind == "speed" and deterministic:
        raise MixedModeError("speed must be measured in normal mode, not deterministic")
    if kind == "bits" and not deterministic:
        raise MixedModeError("bits must be measured in deterministic mode, not normal")
    if kind not in ("speed", "bits"):
        raise ValueError(f"unknown measurement kind {kind!r}")
    return "deterministic" if deterministic else "normal"


def arms_for(*, k: int) -> Tuple[str, ...]:
    """A0 only exists for k <= 8 (its micro-batches no longer fit above that)."""
    return ("A0", "B0", "C") if k <= A0_MAX_K else ("B0", "C")


def run_guarded(run: Callable[[], Dict[str, Any]], *, peaks: Callable[[], Dict[str, Any]],
                is_oom: Callable[[BaseException], bool]) -> Dict[str, Any]:
    """Run one window; an OOM becomes ``status: OOM`` with the peaks seen, other errors raise."""
    try:
        out = dict(run())
        status = "ok"
    except Exception as error:  # noqa: BLE001 - re-raised unless it is an OOM
        if not is_oom(error):
            raise
        out, status = {"error": str(error).splitlines()[0][:200]}, "OOM"
    out.update(peaks())
    out["status"] = status
    return out


def make_row(*, arm: str, k: int, seq: int, kind: str, deterministic: bool,
             result: Dict[str, Any], dataset_sha256: str) -> Dict[str, Any]:
    """One report row; the mode is always written and checked against the kind."""
    row = {"arm": arm, "k": k, "seq": seq, "kind": kind,
           "mode": check_mode(kind, deterministic=deterministic),
           "dataset_sha256": dataset_sha256}
    row.update(result)
    row.setdefault("status", "ok")
    return row


def median_after_warmup(step_seconds: Sequence[float]) -> float:
    """Median of the steps after the first (warm) one."""
    if len(step_seconds) < 2:
        raise ValueError("need a warm-up step and at least one timed step")
    return statistics.median(step_seconds[1:])


def mbpp_records(rows: Sequence[Dict[str, Any]], tokenizer: Any, *, seq: int) -> Dict[str, Any]:
    """Right-truncated, padded token rows of MBPP text + code, and the sha256 of all of them.

    The hash covers the final padded ids and the sequence length, so any change of the text,
    the tokenizer or ``seq`` changes it."""
    pad = int(tokenizer.eos_token_id or 0)
    input_ids: List[List[int]] = []
    for row in rows:
        text = f"{str(row['text']).strip()}\n{str(row['code']).strip()}"
        ids = list(tokenizer(text, add_special_tokens=True)["input_ids"])[:seq]
        input_ids.append(ids + [pad] * (seq - len(ids)))
    digest = hashlib.sha256(json.dumps({"seq": seq, "ids": input_ids}).encode()).hexdigest()
    return {"input_ids": input_ids, "sha256": digest, "seq": seq}


def take_window(input_ids: Sequence[Sequence[int]], *, k: int) -> List[List[int]]:
    """``k`` rows, cycling through the dataset when it has fewer than ``k``."""
    return [list(input_ids[i % len(input_ids)]) for i in range(k)]
