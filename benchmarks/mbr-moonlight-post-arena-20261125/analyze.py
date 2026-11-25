#!/usr/bin/env python3
"""Ticket 17: E's bits after the arena, G's bits and speed, and the D+segmented cell.

Reads ``bits/<label>/`` (the deterministic sweeps of ``run_bits.sh``) and ``speed/`` (the
C-E-G-G-E-C blocks of ``run_speed.sh``) and writes:

- ``analysis.json``: the bit cells of E, G, D and DS against their same-run A0, the CPU cross
  of the fresh A0/E/G/D tensors against the stored ticket 03 and ticket 06 tensors, and the
  speed of C, E and G per window with the three paired deltas, their t over the five blocks,
  the memory peaks and the repeats-rule block count;
- ``rows.md``: result lines with model, card, window, arm, mode and torch version.

The per-block number is the mean of an arm's two runs; the window number is the mean over
blocks, as ``speed-rule.md`` says. Benchmark code, not shipped.

    python analyze.py --root benchmarks/mbr-moonlight-post-arena-20261125
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "benchmarks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.harness import mbr_measure as mm  # noqa: E402
from benchmarks.harness.arm_validity import PILOT_BLOCKS, Run, repeats  # noqa: E402

MODEL = "Moonlight-16B-A3B"
CARD = "RTX 3050 Laptop sm_86 4 GiB"
SPEED_FOLDER = "speed"
ATTEMPTS_FILE = "attempts.jsonl"
MIN_SPEED_CLOCK_MHZ = 1000  # committed speed-rule.md, not ticket 13's pre-run load gate
EXPECTED_ALLOC_CONF = "expandable_segments:True"  # run_speed.sh exports it for every arm
ARMS = ("C", "E", "G")
ARM_LABELS = {"C": "ordered backward", "E": "C + lt-fused",
              "G": "E + async lease + fast row gather", "A0": "plain L2L",
              "D": "decoupled MBR", "DS": "D + segmented"}
DELTAS = (("G", "C"), ("E", "C"), ("G", "E"))
PEAK_KEYS = ("peak_allocated", "peak_reserved", "peak_nvml", "peak_ram")
GRAD_TOTAL = 216
# The one Moonlight LoRA manifest, shared with the launcher verdict (mbr_measure.py).
GRAD_SHAPES = mm.MOONLIGHT_GRAD_SHAPES
# The stored tensors the fresh runs are crossed against, and the cell each one covers.
BIT_FOLDERS = {
    "s64": ("A0", "E"),
    "s512-math": ("A0", "E"),
    "g-k2-s64": ("A0", "G"),
    "d-k2-s64": ("A0", "D"),
    "ds-k2-s64": ("A0", "DS"),
}
TICKET03_ROWS = Path(__file__).resolve().parent.parent / "mbr-moonlight-bits-20261122"
TICKET06_D = (Path(__file__).resolve().parent.parent / "mbr-moonlight-decoupled-20261124"
              / "s64" / "tensors")
# Passed as --lora-targets by run_speed.sh and run_bits.sh.
LORA_TARGETS = "q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"


def expected_campaign_id(source: str, cache: str) -> str:
    """The ``campaign_id`` the launchers write. They pass no ``--steps``, ``--lora-r`` or
    ``--tokenizer``, so the harness defaults apply."""
    return mm.campaign_identity(source=source, cache=cache, tokenizer=None, lora_r=16,
                                lora_targets=mm.parse_lora_targets(LORA_TARGETS), steps=4)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError(f"{path.name}: the last record has no newline")
    return [json.loads(line) for line in text.splitlines() if line]


def _torch(rows: Sequence[Dict[str, Any]]) -> str:
    versions = (_object(_object(row.get("environment")).get("versions")).get("torch")
                for row in rows)
    return ", ".join(sorted({v if isinstance(v, str) and v else "unknown"
                             for v in versions})) or "unknown"


def _gib(value: Optional[int]) -> str:
    return "-" if value is None else f"{value / 2**30:.2f}"


def compare_tensors(left: Path, right: Path, *, k: int, left_row: Row,
                    right_row: Row) -> Dict[str, Any]:
    """Compare two saved bits payloads. Each must be complete against the model manifest and
    hash to its own row's digest (``mm.payload_report``); otherwise there is no verdict, and
    ``reasons`` names each side's problem."""
    import pickle

    import torch

    def load(path: Path, row: Row) -> tuple[dict, dict]:
        issue = None
        try:
            payload = _object(torch.load(path, weights_only=True, map_location="cpu"))
        except FileNotFoundError:
            issue, payload = "missing_tensor_file", {}
        except (RuntimeError, OSError, EOFError, ValueError, pickle.UnpicklingError):
            issue, payload = "unreadable_tensor_file", {}
        report = mm.payload_report(payload, shapes=GRAD_SHAPES, k=k, row=row)
        if issue is not None:
            report["reasons"].insert(0, issue)
        return payload, report

    first, left_report = load(left, left_row)
    second, right_report = load(right, right_row)
    reasons = [f"left:{reason}" for reason in left_report["reasons"]]
    reasons += [f"right:{reason}" for reason in right_report["reasons"]]
    result: Dict[str, Any] = {
        "verdict": None, "loss_equal": None, "grad_total": GRAD_TOTAL,
        "grads_differing": None, "max_abs_diff": None,
        "coverage": {"left": left_report, "right": right_report}, "reasons": reasons,
    }
    if reasons:
        return result
    loss_equal = torch.equal(first["loss"], second["loss"])
    differing = [name for name in GRAD_SHAPES
                 if not torch.equal(first["grads"][name], second["grads"][name])]
    if differing:
        distances = [(first["grads"][name] - second["grads"][name]).abs().max().item()
                     for name in differing]
        max_abs = max(distances) if all(math.isfinite(delta) for delta in distances) else None
    else:
        max_abs = 0.0
    return {**result, "verdict": "equal" if loss_equal and not differing else "differs",
            "loss_equal": loss_equal, "grads_differing": len(differing), "max_abs_diff": max_abs}


def _stored_cell(folder: str, arm: str, k: int, stored_root: Path) -> Optional[Tuple[Path, str]]:
    """Resolve earlier evidence, keeping its physical path separate from the report ID."""
    series = "s512-math" if folder == "s512-math" else "s64"
    name = mm.tensor_file_name(arm, k, 512 if series == "s512-math" else 64)
    if arm in ("A0", "E", "G"):
        return stored_root / series / "tensors" / name, f"ticket-03/{series}/{name}"
    if arm == "D":
        return TICKET06_D / name, f"ticket-06/s64/{name}"
    return None


def _stored_row(folder: str, arm: str, k: int) -> Row:
    """The bits row that wrote the stored tensor of ``arm`` at ``k``, from the in-repo rows; {}
    when no single row matches, so the cross then has no digest to bind to."""
    if arm == "D":
        source, seq = TICKET06_D.parent, 64
    else:
        source = TICKET03_ROWS / ("s512-math" if folder == "s512-math" else "s64")
        seq = 512 if folder == "s512-math" else 64
    matches = [row for row in _evidence(source, "rows.jsonl", []) if row.get("arm") == arm
               and row.get("k") == k and row.get("seq") == seq and row.get("kind") == "bits"]
    return matches[0] if len(matches) == 1 else {}


def expected_dataset_shas() -> Dict[int, str]:
    """The dataset hash ticket 03 recorded per sequence length, from the in-repo rows."""
    found: Dict[int, str] = {}
    for name in ("s64", "s512-math"):
        seq = 512 if name == "s512-math" else 64
        rows = _evidence(TICKET03_ROWS / name, "rows.jsonl", [])
        shas = {row["dataset_sha256"] for row in rows if row.get("seq") == seq
                and isinstance(row.get("dataset_sha256"), str) and row["dataset_sha256"]}
        if len(shas) == 1:
            found[seq] = shas.pop()
    return found


def _bit_row_errors(row: dict, record: dict, *, arm: str, k: int, seq: int, run: int,
                    expected_sha: str | None, campaign_id: str) -> list[str]:
    errors = _watch_errors(record, speed=False)
    if (row.get("arm"), row.get("k"), row.get("seq"), row.get("block"),
        row.get("kind"), row.get("mode"), row.get("attention")) != (
        arm, k, seq, 0, "bits", "deterministic", "math" if seq == 512 else "default",
    ) or row.get("run", run) != run:
        errors.append("row_identity")
    # Archived bits rows carry no campaign_id, so they never equal the expected one.
    if row.get("campaign_id") != campaign_id:
        errors.append("campaign_id")
    if (record.get("arm"), record.get("run"), record.get("round")) != (arm, run, (run + 1) // 2):
        errors.append("record_identity")
    if row.get("status") != "ok" or row.get("outcome") != "ok":
        errors.append("row_status_or_outcome")
    if row.get("tensor_file") != mm.tensor_file_name(arm, k, seq):
        errors.append("tensor_file")
    if not expected_sha or row.get("dataset_sha256") != expected_sha:
        errors.append("dataset_sha256")
    if _torch([row]) == "unknown":
        errors.append("environment.torch")
    if not _number(row.get("loss")):
        errors.append("loss")
    return [f"{arm}:{error}" for error in errors]


def bits_report(root: Path, stored_root: Path, campaign_id: str) -> Dict[str, Any]:
    """Validate the 11 expected cells before comparing fresh or historical tensors."""
    cells, crosses, missing, errors = [], [], [], []
    fresh: dict[tuple[str, str], tuple[Path, Row]] = {}
    expected = expected_dataset_shas()
    for folder, (base_arm, arm) in BIT_FOLDERS.items():
        path = root / "bits" / folder
        folder_errors: list[str] = []
        rows = _evidence(path, "rows.jsonl", folder_errors)
        records = _evidence(path, "arm_records.jsonl", folder_errors)
        if "missing:rows.jsonl" in folder_errors:
            missing.append(folder)
        seq = 512 if folder == "s512-math" else 64
        ks = (1, 2, 4, 8) if folder in ("s64", "s512-math") else (2,)
        if len(rows) != 2 * len(ks) or len(records) != 2 * len(ks):
            folder_errors.append(f"cell_count:rows={len(rows)},records={len(records)}")
        errors.extend(f"{folder}:{reason}" for reason in folder_errors)
        for pos, k in enumerate(ks):
            reasons = list(folder_errors)
            pair, watched = rows[2 * pos:2 * pos + 2], records[2 * pos:2 * pos + 2]
            version = _torch(pair)
            if len(pair) == len(watched) == 2:
                for offset, (side, row, record) in enumerate(zip((base_arm, arm), pair, watched)):
                    reasons.extend(_bit_row_errors(row, record, arm=side, k=k, seq=seq,
                                                   run=2 * pos + offset + 1,
                                                   expected_sha=expected.get(seq),
                                                   campaign_id=campaign_id))
                if _torch(pair[:1]) != _torch(pair[1:]):
                    reasons.append("torch_version_mismatch")
            cell = {"folder": folder, "k": k, "seq": seq, "arm": arm, "torch": version,
                    "dataset_sha256": expected.get(seq) if not reasons else None,
                    "verdict": None, "reasons": reasons}
            cells.append(cell)
            if reasons:
                continue
            files = [path / "tensors" / mm.tensor_file_name(side, k, seq)
                     for side in (base_arm, arm)]
            comparison = compare_tensors(*files, k=k, left_row=pair[0], right_row=pair[1])
            reasons.extend(comparison["reasons"])
            cell.update({key: value for key, value in comparison.items() if key != "reasons"})
            if cell["verdict"] is None:
                reasons.append("invalid_tensor_evidence")
                continue
            if k == 2:
                fresh[folder, arm] = files[1], pair[1]
            for side, tensor, row in zip((base_arm, arm), files, pair):
                stored = _stored_cell(folder, side, k, stored_root)
                if stored is not None:
                    stored_path, source = stored
                    crosses.append({"folder": folder, "k": k, "arm": side, "stored": source,
                                    **compare_tensors(tensor, stored_path, k=k, left_row=row,
                                                      right_row=_stored_row(folder, side, k))})
    ds = fresh.get(("ds-k2-s64", "DS"))
    if ds is not None:
        ds_file, ds_row = ds
        controls = [(TICKET06_D / "D-k2-s64.pt", _stored_row("d-k2-s64", "D", 2),
                     "ticket-06/s64/D-k2-s64.pt", "DS vs stored D (ticket 06)")]
        same_day = fresh.get(("d-k2-s64", "D"))
        if same_day is not None:
            controls.append((same_day[0], same_day[1], "ticket-17/d-k2-s64/D-k2-s64.pt",
                             "DS vs D (same day)"))
        for tensor, row, source, label in controls:
            crosses.append({"folder": "ds-k2-s64", "k": 2, "arm": label, "stored": source,
                            **compare_tensors(ds_file, tensor, k=2, left_row=ds_row,
                                              right_row=row)})
    return {"cells": cells, "crosses": crosses, "missing_folders": missing,
            "errors": errors, "dataset_sha_expected": expected,
            "expected_grad_total": GRAD_TOTAL}


def _paired_t(gains: Sequence[float]) -> Optional[float]:
    if len(gains) < 2:
        return None
    spread = statistics.stdev(gains)
    if spread == 0:
        return None
    return statistics.mean(gains) / (spread / len(gains) ** 0.5)


Window = Tuple[int, int]
Row = Dict[str, Any]


def _object(value: Any) -> Row:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _evidence(folder: Path, name: str, errors: list[str], *, optional: bool = False) -> list[Row]:
    try:
        rows = _read_jsonl(folder / name)
    except FileNotFoundError:
        if not optional:
            errors.append(f"missing:{name}")
        return []
    except (OSError, ValueError):
        errors.append(f"unreadable:{name}")
        return []
    if not all(isinstance(row, dict) for row in rows):
        errors.append(f"invalid_object:{name}")
        return []
    return rows


def _max_clock(record: Row) -> Any:
    return _object(_object(record.get("gpu_clock")).get("sm_clock_mhz")).get("max")


def _watch_errors(record: Row, *, speed: bool) -> list[str]:
    errors = []
    if record.get("outcome") != "ok":
        errors.append("record_outcome")
    required = record.get("required")
    checks = _object(record.get("checks"))
    for check in mm.ARM_CHECKS:
        if not isinstance(required, list) or check not in required \
                or _object(checks.get(check)).get("outcome") != "ok":
            errors.append(f"required_check:{check}")
    if speed:
        clock = _max_clock(record)
        if (_object(record.get("gpu_clock")).get("outcome") != "ok"
                or not _number(clock) or clock < MIN_SPEED_CLOCK_MHZ):
            errors.append("gpu_clock_max")
    return errors


def _block_plan(window: Window, block: int) -> list[Row]:
    return [item for item in mm.ceg_speed_plan(blocks=1, start_block=block)
            if (item["k"], item["seq"]) == window]


def _environment_key(row: Row) -> str:
    environment = _object(row.get("environment"))
    return json.dumps({"versions": _object(environment.get("versions")),
                       "alloc_conf": environment.get("alloc_conf"),
                       "git_sha": environment.get("git_sha"),
                       "ptxas": environment.get("ptxas")}, sort_keys=True)


def _known_version(value: Any) -> bool:
    return isinstance(value, str) and value not in ("", "unknown")


def _nonnegative(value: Any) -> bool:
    return _number(value) and value >= 0


def _speed_row_errors(row: Row, item: Row, expected_sha: Optional[str]) -> list[str]:
    """Metadata a speed row needs before it can be timed evidence: its dataset, its campaign,
    the levers of its arm, every tracked tool version, the allocator config and the peaks."""
    environment = _object(row.get("environment"))
    versions = _object(environment.get("versions"))
    errors = []
    if not expected_sha or row.get("dataset_sha256") != expected_sha:
        errors.append("dataset_sha256")
    if not isinstance(row.get("campaign_id"), str) or not row["campaign_id"]:
        errors.append("campaign_id")
    if row.get("levers") != mm.arm_runtime(item["arm"]):
        errors.append("levers")
    if not all(_known_version(versions.get(name)) for name in mm._TRACKED):
        errors.append("environment_versions")
    if environment.get("alloc_conf") != EXPECTED_ALLOC_CONF:
        errors.append("allocator")
    errors.extend(name for name in ("peak_allocated", "peak_reserved", "peak_nvml")
                  if not _nonnegative(row.get(name)))
    return errors


def _block_errors(plan: list[Row], rows: list[tuple[int, Row]],
                  records: list[tuple[int, Row]], *, shas: Dict[int, str],
                  expected_id: Optional[str]) -> list[str]:
    errors = []
    if len(rows) != len(plan) or len(records) != len(plan):
        errors.append(f"block_count:rows={len(rows)},records={len(records)},expected=6")
    for label, entries in (("row", rows), ("record", records)):
        positions = [position for position, _ in entries]
        if positions and positions != list(range(positions[0], positions[0] + len(positions))):
            errors.append(f"nonconsecutive:{label}")
    for item, (_, row), (_, record) in zip(plan, rows, records):
        run = item["index"] + 1
        if (row.get("arm"), row.get("kind"), row.get("mode")) != (item["arm"], "speed", "normal"):
            errors.append(f"row_identity:{run}")
        if row.get("run") != run:
            errors.append(f"row_run_id:{run}")
        if (record.get("arm"), record.get("run"), record.get("round")) != (
            item["arm"], run, (run - 1) // 2 + 1,
        ):
            errors.append(f"record_identity:{run}")
        if expected_id is not None and record.get("campaign_id") != expected_id:
            errors.append(f"record_campaign:{run}")
        if row.get("status") != "ok" or row.get("outcome") != "ok":
            errors.append(f"row_status_or_outcome:{run}")
        seconds = row.get("step_seconds")
        if not _number(seconds) or seconds <= 0:
            errors.append(f"invalid_step_seconds:{run}")
        errors.extend(f"{reason}:{run}" for reason in _speed_row_errors(
            row, item, shas.get(item["seq"])))
        errors.extend(f"{reason}:{run}" for reason in _watch_errors(record, speed=True))
    return errors


@dataclass
class SpeedEvidence:
    accepted: dict[Window, dict[int, tuple[int, list[Row]]]]
    reserved_slots: dict[Window, set[int]]
    invalid: dict[Window, list[Row]]
    errors: list[str]
    next_block: int
    rows: list[Row]
    records: list[Row]


def read_speed(folder: Path, expected_id: Optional[str] = None) -> SpeedEvidence:
    """Associate row/record identities before filtering. Each slot has a chain of attempts.
    ``expected_id`` is the campaign the evidence must belong to; None checks only consistency."""
    errors: list[str] = []
    rows = _evidence(folder, "rows.jsonl", errors)
    records = _evidence(folder, "arm_records.jsonl", errors)
    journal = _evidence(folder, ATTEMPTS_FILE, errors, optional=True)
    # Old slot-patching exclusions are never evidence of a complete replacement.
    if (folder / "excluded_blocks.json").exists():
        errors.append("unsupported_legacy_exclusions")
    if rows:
        campaigns = {row.get("campaign_id") for row in rows}
        if (len(campaigns) != 1 or not all(isinstance(c, str) and c for c in campaigns)
                or (expected_id is not None and campaigns != {expected_id})):
            errors.append("campaign_identity")
        if len({_environment_key(row) for row in rows}) != 1:
            errors.append("environment_mixed")
    shas = expected_dataset_shas()
    attempts = {window: {slot: [slot] for slot in range(PILOT_BLOCKS)}
                for window in mm.CEG_WINDOWS}
    last_id = PILOT_BLOCKS - 1
    for entry in journal:
        window = (entry.get("k"), entry.get("seq"))
        slot, block = entry.get("slot"), entry.get("block")
        if (not all(type(part) is int for part in (*window, slot, block))
                or window not in attempts or slot < 0 or block <= last_id):
            errors.append("invalid_attempt_reservation")
            continue
        last_id = block
        attempts[window].setdefault(slot, []).append(block)
    plans = {(window, block): _block_plan(window, block)
             for window, slots in attempts.items() for chain in slots.values() for block in chain}
    run_keys = {item["index"] + 1: key for key, plan in plans.items() for item in plan}
    by_row: dict[tuple, list[tuple[int, Row]]] = defaultdict(list)
    by_record: dict[tuple, list[tuple[int, Row]]] = defaultdict(list)
    for position, row in enumerate(rows):
        key = ((row.get("k"), row.get("seq")), row.get("block"))
        if not all(type(part) is int for part in (*key[0], key[1])) or key not in plans:
            errors.append(f"unexpected_row:{position + 1}")
            continue
        by_row[key].append((position, row))
    for position, record in enumerate(records):
        run = record.get("run")
        if type(run) is not int or run not in run_keys:
            errors.append(f"unexpected_record:{position + 1}")
            continue
        by_record[run_keys[run]].append((position, record))
    accepted: dict[Window, dict[int, tuple[int, list[Row]]]] = defaultdict(dict)
    invalid: dict[Window, list[Row]] = defaultdict(list)
    for window, slots in attempts.items():
        for slot, chain in slots.items():
            for index, block in enumerate(chain):
                key = window, block
                reasons = _block_errors(plans[key], by_row[key], by_record[key],
                                        shas=shas, expected_id=expected_id)
                superseded = index != len(chain) - 1
                if superseded and not reasons:
                    errors.append(f"replaced_valid_block:{window}:{block}")
                if reasons:
                    invalid[window].append({"slot": slot, "block": block, "reasons": reasons,
                                            "superseded": superseded})
                elif not superseded:
                    accepted[window][slot] = (block, [row for _, row in by_row[key]])
    return SpeedEvidence(accepted, {window: set(slots) for window, slots in attempts.items()},
                         invalid, errors, last_id + 1, rows, records)


def _window_name(window: Window) -> str:
    return f"({window[0]},{window[1]})"


def _repeat_counts(evidence: SpeedEvidence, count: int) -> dict[str, dict[str, int]]:
    counts = {}
    for window in mm.CEG_WINDOWS:
        slots = evidence.accepted[window]
        if not all(slot in slots for slot in range(count)):
            continue
        blocks = [slots[slot][1] for slot in range(count)]
        counts[_window_name(window)] = {
            f"{now}_vs_{before}": repeats([
                [Run(row["step_seconds"], "ok") for row in block
                 if row["arm"] in (before, now)] for block in blocks
            ]).blocks_needed for now, before in DELTAS
        }
    return counts


def speed_progress(evidence: SpeedEvidence) -> dict[str, Any]:
    """Pilot, one recomputation at N and at most one top-up, shared by both windows."""
    def target(counts: dict, floor: int) -> int:
        return max(floor, *(value for pairs in counts.values() for value in pairs.values()))

    def missing(count: int) -> list[tuple[Window, int]]:
        return [(window, slot) for window in mm.CEG_WINDOWS for slot in range(count)
                if slot not in evidence.accepted[window]]

    pilot = _repeat_counts(evidence, PILOT_BLOCKS)
    pilot_target = target(pilot, PILOT_BLOCKS) if pilot else PILOT_BLOCKS
    initial_target = pilot_target if len(pilot) == len(mm.CEG_WINDOWS) else PILOT_BLOCKS
    topup = _repeat_counts(evidence, initial_target) if not missing(initial_target) else {}
    final_target = target(topup, initial_target) if topup else initial_target
    pending = missing(final_target)
    extra = {_window_name(window): sorted(slots - set(range(final_target)))
             for window, slots in evidence.reserved_slots.items()} if not pending else {}
    return {"pilot": pilot, "pilot_target": pilot_target,
            "pilot_blocks_to_add": pilot_target - PILOT_BLOCKS,
            "recomputed_at_pilot_target": topup, "final_target": final_target,
            "topup_blocks_to_add": final_target - initial_target,
            "pending": pending, "unexpected_slots": extra}


def _window_stats(blocks: list[tuple[int, list[Row]]]) -> Row:
    per_block = {str(block): {arm: statistics.mean(
        row["step_seconds"] for row in rows if row["arm"] == arm
    ) for arm in ARMS} for block, rows in blocks}
    means = {arm: statistics.mean(steps[arm] for steps in per_block.values()) for arm in ARMS}
    deltas = {}
    for now, before in DELTAS:
        paired = [steps[before] - steps[now] for steps in per_block.values()]
        reductions = [100 * (1 - steps[now] / steps[before]) for steps in per_block.values()]
        ratio = means[before] / means[now]
        deltas[f"{now}_vs_{before}"] = {
            "ratio_of_arm_means": ratio, "ratio_speedup_pct": 100 * (ratio - 1),
            "mean_paired_difference_s": statistics.mean(paired),
            "per_block_time_reduction_pct": reductions,
            "mean_per_block_time_reduction_pct": statistics.mean(reductions),
            "paired_t_seconds": _paired_t(paired), "paired_t_time_reduction": _paired_t(reductions),
        }
    by_arm = {arm: [row for _, rows in blocks for row in rows if row["arm"] == arm]
              for arm in ARMS}
    return {**{f"{arm}_step_s": means[arm] for arm in ARMS}, "per_block": per_block,
            "deltas": deltas, "torch": {arm: _torch(rows) for arm, rows in by_arm.items()},
            "memory": {arm: {name: max((row[name] for row in rows if _number(row.get(name))),
                                       default=None) for name in PEAK_KEYS}
                       for arm, rows in by_arm.items()}}


def speed_report(folder: Path, expected_id: Optional[str] = None) -> Row:
    evidence = read_speed(folder, expected_id)
    progress = speed_progress(evidence)
    target = progress["final_target"]
    windows = {}
    for window in mm.CEG_WINDOWS:
        pending = [slot for key, slot in progress["pending"] if key == window]
        slots = evidence.accepted[window]
        unexpected = progress["unexpected_slots"].get(_window_name(window), [])
        verdict = not evidence.errors and not progress["pending"] and not unexpected
        windows[_window_name(window)] = {
            "verdict": "available" if verdict else None,
            "invalid_blocks": evidence.invalid[window], "missing_slots": pending,
            "accepted_blocks": len(slots),
            "unexpected_slots": unexpected,
            **(_window_stats([slots[slot] for slot in range(target)]) if verdict else {}),
        }
    clocks = [_max_clock(record) for record in evidence.records]
    return {"windows": windows, "errors": evidence.errors, "blocks_needed": progress["pilot"],
            "repeats": progress, "raw_runs": len(evidence.rows), "records": len(evidence.records),
            "runs": len(mm.CEG_HANDS) * sum(map(len, evidence.accepted.values())),
            "outcomes": dict(Counter(str(r.get("outcome", "missing")) for r in evidence.records)),
            "gpu_clock_max_min": min((clock for clock in clocks if _number(clock)), default=None)}


def _speed_result_lines(speed: Row) -> list[str]:
    lines = []
    for window, stats in speed["windows"].items():
        if stats["verdict"] is None:
            lines.append(f"- Window {window}, speed: no verdict (see `analysis.json`).")
            continue
        for arm in ARMS:
            text = (f"- {MODEL}, {CARD}, window {window}, arm {arm} ({ARM_LABELS[arm]}), "
                    f"mode speed (normal), torch {stats['torch'][arm]}, "
                    f"step {stats[f'{arm}_step_s']:.2f} s")
            for name, label in zip(PEAK_KEYS, ("allocated", "reserved", "NVML", "RAM")):
                text += f", {label} {_gib(stats['memory'][arm][name])} GiB"
            lines.append(text + f", runs {2 * stats['accepted_blocks']}")
    return lines


def _bit_result_lines(cells: Sequence[Dict[str, Any]]) -> List[str]:
    lines = []
    for cell in cells:
        verdict = cell["verdict"]
        line = (
            f"- {MODEL}, {CARD}, window ({cell['k']},{cell['seq']}), arm {cell['arm']} "
            f"({ARM_LABELS.get(cell['arm'], cell['arm'])}) against A0, mode bits "
            f"(deterministic warn-only), torch {cell['torch']}"
        )
        if verdict is None:
            lines.append(line + ": no verdict (see `analysis.json`).")
            continue
        max_abs = cell["max_abs_diff"]
        delta = f"{max_abs:.2e}" if max_abs is not None else "nonfinite"
        lines.append(line + f": {verdict}, loss equal {cell['loss_equal']}, "
                     f"grads differing {cell['grads_differing']} / {cell['grad_total']}, "
                     f"max abs {delta}, dataset_sha {cell['dataset_sha256'][:8]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    moonlight = os.environ.get("MOONLIGHT_ROOT") or str(Path.home() / "moonlight-local")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stored-root", type=Path,
                        default=Path(os.environ.get("MOONLIGHT_ROOT",
                                                    Path.home() / "moonlight-local"))
                        / "results" / "bits",
                        help="ticket 03 tensor root (default $MOONLIGHT_ROOT/results/bits)")
    parser.add_argument("--source", default=f"{moonlight}/source",
                        help="speed campaign model source (default $MOONLIGHT_ROOT/source)")
    parser.add_argument("--cache", default=f"{moonlight}/cache",
                        help="speed campaign cache (default $MOONLIGHT_ROOT/cache)")
    args = parser.parse_args(argv)

    campaign_id = expected_campaign_id(args.source, args.cache)
    bits = bits_report(args.root, args.stored_root, campaign_id)
    speed = speed_report(args.root / SPEED_FOLDER, campaign_id)
    analysis = {"model": MODEL, "card": CARD, "bits": bits, "speed": speed}
    (args.root / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    lines = ["# Result lines (ticket 17, post-arena E, G and D+segmented)", ""]
    lines += _speed_result_lines(speed)
    lines += _bit_result_lines(bits["cells"])
    (args.root / "rows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Console().print_json(data={
        "bits": dict(Counter(c["verdict"] or "no verdict" for c in bits["cells"])),
        "bit_errors": bits["errors"],
        "stored_crosses": dict(Counter(c["verdict"] or "no verdict" for c in bits["crosses"])),
        "speed": {window: stats["verdict"] for window, stats in speed["windows"].items()},
        "speed_errors": speed["errors"],
    })
    return int(bool(bits["errors"]) or any(c["verdict"] is None for c in bits["cells"])
               or any(c["verdict"] is None for c in bits["crosses"])
               or any(stats["verdict"] is None for stats in speed["windows"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
