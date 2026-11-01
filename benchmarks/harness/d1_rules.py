"""Offline, fail-closed evidence checks for the D1 NF4 study (not shipped)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

MIN_BLOCKS = 5
CLOCK_SPREAD = 0.05


def canonical_gpu_uuid(value: str) -> str:
    return "GPU-" + str(uuid.UUID(value.removeprefix("GPU-")))


def all_block_reads_check(before: dict, after: dict) -> dict:
    """Conservative upper bound: all visible block reads, no own-I/O subtraction."""
    evidence = {
        "before": before,
        "after": after,
        "limit_bytes": 1_000_000_000,
        "scope": "all entries in /proc/diskstats, including stacked devices and "
        "partitions; own I/O is included, not subtracted",
    }
    if (
        not before
        or before.keys() != after.keys()
        or any(
            not isinstance(before[name], int)
            or not isinstance(after[name], int)
            or after[name] < before[name]
            for name in before
        )
    ):
        return {
            "outcome": "unknown",
            "reason": "missing/changing block counters",
            "evidence": evidence,
        }
    delta = sum(after[name] - before[name] for name in before)
    evidence["all_block_read_bytes"] = delta
    return {
        "outcome": "ok" if delta < 1_000_000_000 else "void",
        "reason": f"all block-device reads, own reads included: {delta} bytes",
        "evidence": evidence,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings_digest(settings: dict) -> str:
    raw = json.dumps(settings, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    """Never replace a failure with a later successful attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def evidence_path(root: Path, name: str) -> Path:
    """Containment uses realpath/commonpath, including on Windows."""
    base = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(base, name))
    try:
        contained = os.path.commonpath([base, candidate]) == base
    except ValueError:
        contained = False
    if not contained or candidate == base or ":" in name or name.startswith(("/", "\\")):
        raise ValueError(f"evidence containment failed: {name!r}")
    return Path(candidate)


def compare_tensors(candidate: dict, reference: dict) -> dict:
    import torch

    missing = sorted(reference.keys() - candidate.keys())
    extra = sorted(candidate.keys() - reference.keys())
    quantities = []
    for name in sorted(reference.keys() & candidate.keys()):
        left, right = candidate[name], reference[name]
        same_shape = left.shape == right.shape and left.dtype == right.dtype
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        exact = same_shape and finite and torch.equal(left, right)
        difference = None
        if same_shape and finite and left.numel():
            difference = float((left.double() - right.double()).abs().max())
        quantities.append({"name": name, "exact": exact, "finite": finite, "max_abs": difference})
    return {
        "passed": bool(reference)
        and not missing
        and not extra
        and all(row["exact"] for row in quantities),
        "missing": missing,
        "extra": extra,
        "quantities": quantities,
    }


def negative_control(candidate: dict, reference: dict) -> bool:
    result = compare_tensors(candidate, reference)
    return (
        bool(result["quantities"])
        and not result["missing"]
        and not result["extra"]
        and all(row["finite"] for row in result["quantities"])
        and any(not row["exact"] for row in result["quantities"])
    )


def _positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def peer_check(before: dict, during: list[dict], after: dict, gpu_uuid: str) -> bool:
    """Infer the sole new CUDA PID only from a clean full process lifecycle."""
    samples = [before, *during, after]
    if not during or not gpu_uuid:
        return False
    if any(not row.get("available") or row.get("uuid") != gpu_uuid for row in samples):
        return False
    if before["compute"] or after["compute"] or any(row["graphics"] for row in samples):
        return False
    active = [row["compute"] for row in during if row["compute"]]
    return (
        bool(active)
        and all(len(pids) == 1 for pids in active)
        and len({pids[0] for pids in active}) == 1
    )


def _comparison(case: str, rows: list[dict]) -> dict:
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["block"]].append(row)
    identities = ("gpu_uuid", "source_sha", "rule_sha256", "settings_sha256")
    mixed = [
        field
        for field in identities
        if len({row.get(field) for row in rows}) != 1 or any(not row.get(field) for row in rows)
    ]
    references = [row["dense_gemm_reference"] for row in rows if "dense_gemm_reference" in row]
    windows = [*rows, *references]
    clocks = [sample for row in windows for sample in row.get("clock_sm_mhz", [])]
    clocks_ok = (
        all(row.get("clock_sm_mhz") for row in rows)
        and all(_positive(sample) for sample in clocks)
        and max(clocks, default=0) <= min(clocks, default=0) * (1 + CLOCK_SPREAD)
    )
    limits = [sample for row in windows for sample in row.get("power_limit_mw", [])]
    power_ok = (
        all(row.get("power_limit_mw") for row in rows)
        and all(_positive(value) for value in limits)
        and len(set(limits)) == 1
    )
    accepted, rejected = [], []
    for number, block in sorted(blocks.items()):
        block = sorted(block, key=lambda row: row["run"])
        reasons = []
        if [row["run"] for row in block] != list(range(4)) or [row["arm"] for row in block] != list(
            "ABBA"
        ):
            reasons.append("incomplete or reordered ABBA")
        if any(row.get("validity", {}).get("outcome") != "ok" for row in block):
            reasons.append("void or unknown required validity")
        if any(not _positive(row.get("median_step_ms")) for row in block):
            reasons.append("nonfinite or nonpositive timing")
        if mixed:
            reasons.append("mixed identity: " + ", ".join(mixed))
        if not clocks_ok:
            reasons.append("missing clock or >5% clock spread")
        if any(row.get("peers_clean") is not True for row in block):
            reasons.append("GPU peer visibility missing or contaminated")
        if not power_ok:
            reasons.append("missing or changed GPU power limit")
        if any(not row.get("memory_util_pct") for row in block):
            reasons.append("missing memory-controller utilization")
        for row in block:
            if ("reference_harness_sha256" in row and row.get("shape")) or (
                "dense_gemm_reference" in row
            ):
                from d1_reference import validate_reference

                validation = validate_reference(row, row.get("dense_gemm_reference", {}))
                if not validation["passed"]:
                    reasons.append("dense GEMM reference: " + "; ".join(validation["reasons"]))
        if reasons:
            rejected.append({"block": number, "reasons": reasons})
        else:
            accepted.append(block)
    result = {
        "case": case,
        "accepted_blocks": len(accepted),
        "accepted_block_ids": [block[0]["block"] for block in accepted],
        "rejected_blocks": rejected,
        "status": "NO_VERDICT",
        "ratio_a_over_b": None,
        "speedup_claim": False,
    }
    if len(accepted) < MIN_BLOCKS:
        return result
    from arm_validity import Run, blocks_to_add

    required_extra = blocks_to_add(
        [[Run(row["median_step_ms"], "ok") for row in block] for block in accepted]
    )
    result["blocks_to_add_for_10_percent_effect"] = required_extra
    if required_extra:
        return result
    a_values = [row["median_step_ms"] for block in accepted for row in block if row["arm"] == "A"]
    b_values = [row["median_step_ms"] for block in accepted for row in block if row["arm"] == "B"]
    paths = {
        arm: sorted({row["path"] for block in accepted for row in block if row["arm"] == arm})
        for arm in "AB"
    }
    result.update(
        status="SAME_PATH" if paths["A"] == paths["B"] else "MEASURED",
        ratio_a_over_b=statistics.median(a_values) / statistics.median(b_values),
        paths=paths,
        median_ms={"A": statistics.median(a_values), "B": statistics.median(b_values)},
        ranges_ms={"A": [min(a_values), max(a_values)], "B": [min(b_values), max(b_values)]},
        peak_allocated_bytes={
            arm: max(
                row["peak_allocated_bytes"]
                for block in accepted
                for row in block
                if row["arm"] == arm
            )
            for arm in "AB"
        },
        peak_reserved_bytes={
            arm: max(
                row["peak_reserved_bytes"]
                for block in accepted
                for row in block
                if row["arm"] == arm
            )
            for arm in "AB"
        },
    )
    # Measured ratios are descriptive. F1's sample sizing targets 10%, not a 1-4% claim.
    return result


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row["case"], row["block"], row["run"])
        if identity in seen:
            raise ValueError(f"duplicate run: {identity}")
        seen.add(identity)
        groups[row["case"]].append(row)
    comparisons = [_comparison(case, values) for case, values in sorted(groups.items())]
    return {
        "status": "MEASURED_SUBSET"
        if any(row["status"] in ("MEASURED", "SAME_PATH") for row in comparisons)
        else "NO_VERDICT"
        if rows
        else "NOT_RUN",
        "d1_complete": False,
        "comparisons": comparisons,
        "scope": "Only the recorded cases; D1 closure also needs correctness, large-alias "
        "controls, pretrained real-data training and independent acceptance.",
    }
