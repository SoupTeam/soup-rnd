#!/usr/bin/env python3
"""The G2 verdict and break-even analysis from a session recorded under the current rule.

The rule is section 3 of benchmarks/probe-g2-recompute-vs-spill.md; ``RULE`` below
is that text in machine form, written into the session's protocol.json. Rule v2
was chosen after session 1 and before session 2. Evaluation requires an exact
protocol.rule match. Historical v1 needs its pinned evaluator, not reinterpretation
under v2. Usage::

    python benchmarks/harness/g2_rule.py <session dir> [--out verdict.json]

Files read (written by ``g2_session.py`` / ``g2_probe.py`` / ``g2_disk.py``):

* ``protocol.json`` (the recorded rule must match this evaluator);
* ``correctness_k<k>.json`` and, when its references were not reproducible,
  ``correctness_k<k>_det.json``;
* ``profile_ram.json`` and ``profile_disk.json`` (recompute time, bytes, store overheads);
* ``arms/<comparison>/b<block>_p<position>_<A|B>.json``;
* ``disk/rates.json`` (``g2_disk.export_rates`` of the sustained I/O session).

A store that is not bit-exact has no speed: G3/G4 read nothing unless G1 passed.
A CPU or synthetic session never gets a verdict. Memory savings are not speed.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arm_validity  # noqa: E402

PASS, FAIL, NONE = "pass", "fail", "no verdict"
#: The store must cut the mean step time by at least this fraction to pass.
MIN_GAIN = 0.05
CORRECTNESS_KS = (1, 2, 4)
CORRECTNESS_STORES = ("g2_ram", "g2_spill")
RAM_COMPARISON = "ram_store_vs_recompute"
DISK_COMPARISON = "disk_spill_vs_recompute"
COMPARISONS = (RAM_COMPARISON, DISK_COMPARISON)
#: Theoretical step-FLOP ratio with vs without the checkpoint recompute (6N / 4N).
COMPUTE_CEILING = 1.5
#: Sustained I/O inputs must come from at least this long a measured window.
MIN_SUSTAINED_S = 1800.0

RULE = {
    "profile": "g2-recompute-vs-spill-v2",
    "G1": "for k in 1,2,4 and both stores (pinned RAM, direct NVMe spill): per-micro-batch "
          "losses (forward) and nonzero LoRA grads (backward) torch.equal to GA AND to the "
          "recompute L2L step; references reproducible (GA vs GA), else the --deterministic "
          "rerun; the controls must hold: identity saved-tensor hooks without a reload must "
          "raise or give different grads, a corrupted record must change the grads and leave "
          "the losses equal; and the records hold activation bytes only (0 weight bytes)",
    "G2": "bytes: every record holds exactly its activation items (payload == sum of the "
          "activation storages saved) and no streamed weight; the spilled bytes written and "
          "read per step equal spill_layers x k x record_capacity",
    "G3": f"{RAM_COMPARISON} (weights in RAM, records in pinned RAM): per-block step-time "
          f"ratio B (store) / A (recompute), two-sided 95% t interval: pass if its upper bound "
          f"< {1 - MIN_GAIN:.2f}, fail if its lower bound >= {1 - MIN_GAIN:.2f}, else no verdict",
    "G4": f"{DISK_COMPARISON} (weights streamed from disk, records spilled to the weights' "
          f"drive): the same ratio and bounds as G3",
    "V1": "device cuda; a CPU or synthetic session never gets a verdict",
    "V2": "required F1 checks (suspend, power, foreign_reads) ok on every arm a gate reads; "
          "a block with a void arm is replaced, an unknown one removes the gate",
    "V3": "every spilling store and the sustained I/O arms are direct (unbuffered)",
    "V4": "A and B of a block train on the same token rows (ids fingerprint)",
    "V5": "finite losses on every timed step",
    "V6": "every B arm wrote and read exactly its planned record bytes; A arms wrote none",
    "V7": "the F1 repeats rule (pilot 5 blocks, N = ceil(7.85 x (spread / 10%)^2), one "
          "top-up) has no block left to add, with the spread of the A runs and of the B runs "
          "each taken relative to its own arm's mean and pooled (rule v2; v1 pooled them in "
          "seconds against 10% of A's mean, so a B five times A's length counted its own "
          "relative noise 25 times)",
    "V8": "disk-tier arms: the weight source reads with direct I/O",
    "V10": "disk-tier arms: the OS placement records (resolved path, mount, whole physical "
           "disk) show the records' spill directory and the weight shards on one physical "
           "disk; unknown placement removes the gate (direct-I/O flags are no evidence)",
    "V9": f"the break-even inputs are sustained: g2_disk arms with status candidate, a "
          f"fixture larger than RAM and >= {MIN_SUSTAINED_S:.0f} s each; else the break-even "
          f"is reported as an estimate only",
    "V11": "the sustained rates are bound to the disk-tier profile: every valid g2_disk arm "
           "and the profile name one machine id, and the arms' spill / weight disks are the "
           "profile's; a mismatch or an unknown identity makes the break-even an estimate",
    "V12": "label only, no gate status depends on it: a disk-tier gate and the break-even "
           "name the disk kind from the placement records (virtio vendor 0x1af4, a "
           "hypervisor's model string, a vd*/xvd* device); on a VIRTUAL disk they describe "
           "that virtual disk at that time, not a physical NVMe",
    "breakeven": "informational: saved recompute seconds per step against write + read + "
                 "weight-read contention + store overhead, serial and overlapped; the store "
                 "overhead is packing and host copies with the measured spill I/O waits "
                 "taken out, so the drive is charged once, from its sustained rates; the 1.5x "
                 "ceiling is a FLOP ratio, never a measured speed-up",
    "order": "A-B-B-A blocks, fresh process per arm",
}

#: Student t quantiles for a two-sided 95% interval, by degrees of freedom.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df: int) -> float:
    if df < 1:
        raise ValueError("an interval needs at least two blocks")
    if df in _T975:
        return _T975[df]
    known = [d for d in _T975 if d <= df]
    return _T975[max(known)] if df <= 30 else 1.960


def interval(values: Sequence[float]) -> Dict[str, Any]:
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "low": None, "high": None}
    mean = statistics.fmean(values)
    if n < 2:
        return {"n": n, "mean": mean, "low": None, "high": None}
    half = t975(n - 1) * statistics.stdev(values) / math.sqrt(n)
    return {"n": n, "mean": mean, "low": mean - half, "high": mean + half}


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rows(payload: Optional[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [row for row in (payload or {}).get("records", []) if row.get("kind") == kind]


def _gate(status: str, detail: str, **extra: Any) -> Dict[str, Any]:
    return {"status": status, "detail": detail, **extra}


# ---------------------------------------------------------------------------
# V7, rule v2: the F1 repeats rule with each arm's spread relative to its own mean
# ---------------------------------------------------------------------------
def relative_spread(blocks: Sequence[Sequence[arm_validity.Run]]) -> float:
    """sqrt((CV_A^2 + CV_B^2) / 2): the run-to-run spread of each arm over its own mean,
    pooled. Equal to F1's sigma / mean_A when A and B run equally long."""
    squares = []
    for arm in ("A", "B"):
        values = [run.value for block in blocks
                  for run, name in zip(block, arm_validity.BLOCK_ORDER) if name == arm]
        mean = statistics.fmean(values)
        if mean <= 0:
            raise ValueError(f"the mean of {arm} must be positive, got {mean}")
        squares.append(statistics.variance(values) / mean ** 2)
    return math.sqrt(sum(squares) / 2)


def blocks_needed(blocks: Sequence[Sequence[arm_validity.Run]]) -> int:
    """Blocks a 10% effect needs: ceil(7.85 x (relative spread / 10%)^2), never below
    F1's minimum. Blocks with a void run do not count."""
    blocks = arm_validity.valid_blocks(blocks)
    if len(blocks) < arm_validity.MIN_BLOCKS:
        raise ValueError(f"the spread needs at least {arm_validity.MIN_BLOCKS} valid blocks")
    ratio = relative_spread(blocks) / arm_validity.EFFECT_FRACTION
    return max(arm_validity.MIN_BLOCKS,
               math.ceil(arm_validity.SAMPLE_SIZE_FACTOR * ratio ** 2))


def blocks_to_add(blocks: Sequence[Sequence[arm_validity.Run]]) -> int:
    """F1's schedule (``arm_validity.blocks_to_add``) with ``blocks_needed`` above: the
    pilot, then up to N from the pilot, then one top-up from the first N blocks."""
    blocks = arm_validity.valid_blocks(blocks)
    if len(blocks) < arm_validity.PILOT_BLOCKS:
        return arm_validity.PILOT_BLOCKS - len(blocks)
    first_n = max(arm_validity.PILOT_BLOCKS, blocks_needed(blocks[:arm_validity.PILOT_BLOCKS]))
    if len(blocks) < first_n:
        return first_n - len(blocks)
    final_n = max(first_n, blocks_needed(blocks[:first_n]))
    return max(0, final_n - len(blocks))


def _combine(parts: Sequence[Dict[str, Any]]) -> str:
    statuses = [part["status"] for part in parts]
    return FAIL if FAIL in statuses else (NONE if NONE in statuses or not parts else PASS)


# ---------------------------------------------------------------------------
# G1: bit-exactness, G2: activation-only bytes
# ---------------------------------------------------------------------------
def _exact(pair: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(pair, dict) or not isinstance(pair.get("backward"), dict):
        return False
    backward = pair["backward"]
    tensors = backward.get("tensors")
    return (isinstance(tensors, int) and not isinstance(tensors, bool) and tensors > 0
            and backward.get("equal") is True and pair.get("forward_equal") is True
            and backward.get("all_zero") == [])


def _reproducible(row: Dict[str, Any]) -> bool:
    return _exact(row.get("ga_vs_ga"))


def _only_row(payload: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    rows = _rows(payload, "correctness")
    if len(rows) != 1:
        return None, "no record" if not rows else f"{len(rows)} correctness records, not one"
    row = rows[0]
    if row.get("skipped") or row.get("error"):
        return None, str(row.get("skipped") or row.get("error"))
    return row, ""


def select_correctness(session: Path, k: int) -> Dict[str, Any]:
    """The ONE correctness record of ``k`` that G1 and G2 both read.

    The default run when its GA reference is reproducible; else the predeclared
    ``--deterministic`` rerun, when that is reproducible and says it ran
    deterministically. Exactness and bytes come from the same run, never one each.
    """
    default, why = _only_row(_load(session / f"correctness_k{k}.json"))
    if default is None:
        return {"row": None, "source": None, "why": f"k={k}: {why}"}
    if _reproducible(default):
        return {"row": default, "source": f"correctness_k{k}.json", "why": "reproducible"}
    det, det_why = _only_row(_load(session / f"correctness_k{k}_det.json"))
    if det is None or not det.get("deterministic") or not _reproducible(det):
        detail = det_why or ("not a --deterministic run" if det is not None
                             and not det.get("deterministic") else "not reproducible")
        return {"row": None, "source": None,
                "why": f"k={k}: GA is not reproducible, and the --deterministic rerun "
                       f"cannot replace it ({detail})"}
    return {"row": det, "source": f"correctness_k{k}_det.json",
            "why": "default GA not reproducible; deterministic rerun"}


def _g1_k(selected: Dict[str, Any], k: int) -> Dict[str, Any]:
    row = selected["row"]
    if row is None:
        return _gate(NONE, selected["why"])
    source = {"source": selected["source"]}
    controls = row.get("controls") or {}
    naive = controls.get("naive_identity") or {}
    if not (naive.get("raised") or (naive.get("backward") and not naive["backward"]["equal"])):
        return _gate(NONE, f"k={k}: the identity-hook control neither raised nor differed, so "
                           "the fixture cannot show the weight-slot hazard", **source)
    corrupt = controls.get("corrupt") or {}
    if not corrupt.get("backward") or corrupt["backward"]["equal"]:
        return _gate(NONE, f"k={k}: the corrupted-record control did not change the grads",
                     **source)
    if not corrupt.get("forward_equal"):
        return _gate(FAIL, f"k={k}: a corrupted record changed the forward losses, so the "
                           "forward reads the store", **source)
    if not _exact(row.get("l2l_vs_ga")):
        return _gate(NONE, f"k={k}: the recompute L2L reference is not exact to GA", **source)
    failures = []
    for store in CORRECTNESS_STORES:
        for ref in ("ga", "l2l"):
            if not _exact(row.get(f"{store}_vs_{ref}")):
                failures.append(f"{store} vs {ref}")
    if failures:
        return _gate(FAIL, f"k={k}: not bit-exact: {', '.join(failures)}", **source)
    stored = [(row.get(f"{store}_bytes") or {}).get("weight_bytes_stored")
              for store in CORRECTNESS_STORES]
    if any(value is None for value in stored):
        return _gate(NONE, f"k={k}: a store has no weight-alias accounting", **source)
    if any(stored):
        return _gate(FAIL, f"k={k}: records hold {sum(stored)} streamed-weight bytes", **source)
    return _gate(PASS, f"k={k}: both stores exact to GA and to recompute; controls differ "
                       f"({selected['source']})",
                 deterministic=bool(row.get("deterministic")), **source)


def gate_g1(session: Path) -> Dict[str, Any]:
    parts = {k: _g1_k(select_correctness(session, k), k) for k in CORRECTNESS_KS}
    status = _combine(list(parts.values()))
    return _gate(status, "; ".join(part["detail"] for part in parts.values()), parts=parts)


def gate_g2(session: Path) -> Dict[str, Any]:
    """Activation-only records, and the store moved exactly what it planned, read from
    the same record per k that G1 judged.

    ``weight_bytes_stored`` is an address-overlap test of every stored item
    against the streamed weight buffers, independent of the classification.
    """
    problems, missing, checked = [], [], 0
    for k in CORRECTNESS_KS:
        selected = select_correctness(session, k)
        row = selected["row"]
        if row is None:
            missing.append(selected["why"])
            continue
        tag = f"k={k} ({selected['source']})"
        for store in CORRECTNESS_STORES:
            bytes_ = row.get(f"{store}_bytes")
            if not bytes_:
                problems.append(f"{tag} {store}: no byte accounting")
                continue
            checked += 1
            if bytes_.get("weight_bytes_stored") != 0:
                problems.append(f"{tag} {store}: {bytes_.get('weight_bytes_stored')} "
                                "streamed-weight bytes in the records")
            capacity = bytes_.get("record_capacity_bytes")
            if bytes_.get("store_chunk_bytes") != capacity:
                problems.append(f"{tag} {store}: store chunk {bytes_.get('store_chunk_bytes')} "
                                f"!= record capacity {capacity}")
            if (bytes_.get("payload_bytes_per_step") or 0) > (capacity or 0) * bytes_.get(
                    "records", 0):
                problems.append(f"{tag} {store}: payload exceeds the records' capacity")
            planned = bytes_.get("planned_spill_bytes")
            for direction in ("written", "read"):
                if bytes_.get(f"spill_{direction}") != planned:
                    problems.append(f"{tag} {store}: spill {direction} "
                                    f"{bytes_.get(f'spill_{direction}')} != plan {planned}")
            if store == "g2_spill" and not planned:
                problems.append(f"{tag} {store}: nothing spilled")
            if planned and bytes_.get("spill_direct") is not True:
                problems.append(f"{tag} {store}: spill backend not direct")
            if bytes_.get("left_behind"):
                problems.append(f"{tag} {store}: spill files left behind")
    if problems:
        return _gate(FAIL, "; ".join(problems))
    if missing:
        return _gate(NONE, "; ".join(missing))
    if not checked:
        return _gate(NONE, "no correctness record carries byte accounting")
    return _gate(PASS, f"{checked} stores: activation bytes only, spill bytes as planned")


# ---------------------------------------------------------------------------
# G3/G4: store vs recompute step time
# ---------------------------------------------------------------------------
def _arm_files(session: Path, name: str) -> Tuple[Dict[int, Dict[int, Dict[str, Any]]],
                                                List[str]]:
    blocks: Dict[int, Dict[int, Dict[str, Any]]] = {}
    problems = []
    for path in sorted((session / "arms" / name).glob("b*_p*_*.json")):
        stem = path.stem.split("_")
        try:
            block, position = int(stem[0][1:]), int(stem[1][1:])
        except (IndexError, ValueError):
            problems.append(f"{path.name}: invalid arm filename")
            continue
        if len(stem) != 3 or position not in range(4) or \
                stem[2] != arm_validity.BLOCK_ORDER[position]:
            problems.append(f"{path.name}: filename does not match A-B-B-A")
        payload = _load(path)
        if not isinstance(payload, dict):
            problems.append(f"{path.name}: unreadable arm")
            continue
        arms = blocks.setdefault(block, {})
        if position in arms:
            problems.append(f"block {block} p{position}: multiple arm files")
        else:
            arms[position] = payload
    return blocks, problems


def _arm_record(arm: Dict[str, Any]) -> Dict[str, Any]:
    rows = arm.get("records")
    return rows[0] if isinstance(rows, list) and len(rows) == 1 and \
        isinstance(rows[0], dict) else {}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _arm_problems(arm: Dict[str, Any], *, tier: str) -> List[str]:
    problems = []
    rec = _arm_record(arm)
    if rec.get("kind") != "arm":
        return ["expected exactly one arm record"]
    if rec.get("error") or rec.get("skipped"):
        return [f"arm error: {rec.get('error') or rec.get('skipped')}"]
    meta = arm.get("meta") or {}
    role = rec.get("variant_role")
    comparison = RAM_COMPARISON if tier == "ram" else DISK_COMPARISON
    store = "ram" if tier == "ram" else "spill"
    if role not in ("A", "B") or rec.get("arm") != role or \
            rec.get("variant") != ("recompute" if role == "A" else "store"):
        problems.append("arm role does not match its recompute/store variant")
    if rec.get("comparison") != comparison or rec.get("tier") != tier or \
            rec.get("store") != store:
        problems.append("arm comparison/tier/store does not match the gate")
    if meta.get("mode") != "arm" or any(meta.get(key) != rec.get(key) for key in
            ("comparison", "arm", "block", "position", "tier", "store", "k")):
        problems.append("arm metadata does not match its record")
    steps = rec.get("steps") or []
    k = rec.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        problems.append("arm has no positive micro-batch count")
    if not isinstance(meta.get("steps"), int) or isinstance(meta.get("steps"), bool) or \
            not steps or len(steps) != meta["steps"] or _median_step(arm) is None:
        problems.append("timed step seconds must all be positive and finite; count must match")
    losses = rec.get("losses") or []
    if not losses or any(not _finite_number(loss) for loss in losses) or any(
            not step.get("losses") or len(step["losses"]) != k or
            any(not _finite_number(loss) for loss in step["losses"]) for step in steps):
        problems.append("V5: non-finite or missing loss on a timed step")
    expected = rec.get("planned_spill_bytes_per_step")
    spills = tier == "disk" and role == "B"
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0 or \
            (expected <= 0 if spills else expected != 0):
        problems.append("V6: spill plan does not match the arm and tier")
    for step in steps:
        if step.get("spill_written") != expected or step.get("spill_read") != expected:
            problems.append(f"V6: spill bytes {step.get('spill_written')}/"
                            f"{step.get('spill_read')} != plan {expected}")
            break
    if expected and rec.get("spill_direct") is not True:
        problems.append("V3: spilling store is not direct")
    if tier == "disk" and meta.get("direct_io") is not True:
        problems.append("V8: weight source without direct I/O")
    if spills:
        same, why = same_physical_disk(rec.get("placement"))
        if same is not True:
            problems.append(f"V10: spill and weights not shown on one physical disk ({why})")
    return problems


def same_physical_disk(placement: Optional[Dict[str, Any]]) -> Tuple[Optional[bool], str]:
    """From the arm's own OS readings, recomputed here: True/False when both whole disks
    are named, None when either is unknown (never read as "same")."""
    place = placement or {}
    disks = []
    for side in ("spill", "weights"):
        disk = ((place.get(side) or {}).get("disk") or {})
        if not disk.get("id"):
            return None, f"{side} disk unknown: {disk.get('problem') or 'no placement record'}"
        disks.append(disk["id"])
    return (disks[0] == disks[1]), " and ".join(dict.fromkeys(disks))


def disk_scope(placement: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Which kind of disk a disk-tier measurement describes, from its placement record:
    ``virtual`` True when either side is a hypervisor's disk (then G4 and the break-even
    describe that virtual disk at that time, not a physical NVMe), False when both are
    named non-virtual, None when unknown. A label only: no gate status depends on it."""
    place = placement or {}
    flags = {side: ((place.get(side) or {}).get("disk") or {}).get("virtual")
             for side in ("spill", "weights")}
    reasons = {side: ((place.get(side) or {}).get("disk") or {}).get("virtual_reason")
               for side in ("spill", "weights")}
    if True in flags.values():
        virtual: Optional[bool] = True
    elif all(flag is False for flag in flags.values()):
        virtual = False
    else:
        virtual = None
    return {"virtual": virtual, "reasons": reasons}


def _f1_outcome(arm: Dict[str, Any]) -> str:
    rec = _arm_record(arm)
    record = rec.get("F1") or {}
    checks = record.get("checks") or {}
    required = record.get("required")
    if not isinstance(required, list) or set(required) != set(arm_validity.CHECKS) or any(
            record.get(key) != rec.get(field) for key, field in
            (("arm", "arm"), ("round", "block"), ("run", "position"))):
        return arm_validity.UNKNOWN
    outcomes = [(checks.get(name) or {}).get("outcome") for name in arm_validity.CHECKS]
    if any(outcome not in (arm_validity.OK, arm_validity.VOID, arm_validity.UNKNOWN)
           for outcome in outcomes):
        return arm_validity.UNKNOWN
    outcome = (arm_validity.VOID if arm_validity.VOID in outcomes else
               arm_validity.UNKNOWN if arm_validity.UNKNOWN in outcomes else arm_validity.OK)
    if record.get("outcome") != outcome or rec.get("outcome") != outcome:
        return arm_validity.UNKNOWN
    return outcome


def _median_step(arm: Dict[str, Any]) -> Optional[float]:
    steps = _arm_record(arm).get("steps") or []
    times = [step.get("step_s") for step in steps]
    if not times or any(not _finite_number(value) or value <= 0 for value in times):
        return None
    return statistics.median(times)


def gate_speed(session: Path, name: str, *, tier: str) -> Dict[str, Any]:
    if {RAM_COMPARISON: "ram", DISK_COMPARISON: "disk"}.get(name) != tier:
        return _gate(NONE, f"{name}: comparison does not match tier {tier}")
    blocks, problems = _arm_files(session, name)
    if not blocks:
        return _gate(NONE, f"{name}: {problems[0] if problems else 'no arms'}")
    ratios, runs, unknown = [], [], False
    for block, arms in sorted(blocks.items()):
        if sorted(arms) != [0, 1, 2, 3]:
            problems.append(f"block {block}: {len(arms)} of 4 arms")
            continue
        order = [_arm_record(arms[p]).get("variant_role") for p in range(4)]
        if order != list(arm_validity.BLOCK_ORDER):
            problems.append(f"block {block}: order {order}, not A-B-B-A")
            continue
        if any(_arm_record(arms[p]).get("block") != block or
               _arm_record(arms[p]).get("position") != p for p in range(4)):
            problems.append(f"block {block}: record identity does not match filename")
            continue
        fingerprints = {_arm_record(arms[p]).get("ids_sha256") for p in range(4)}
        if len(fingerprints) != 1 or not all(fingerprints):
            problems.append(f"block {block}: V4 token rows differ")
            continue
        block_problems = [f"block {block} p{p}: {issue}" for p in range(4)
                          for issue in _arm_problems(arms[p], tier=tier)]
        if block_problems:
            problems.extend(block_problems)
            continue
        outcomes = [_f1_outcome(arms[p]) for p in range(4)]
        unknown = unknown or arm_validity.UNKNOWN in outcomes
        if arm_validity.VOID in outcomes:
            problems.append(f"block {block}: void arm, replace the block")
            continue
        medians = [_median_step(arms[p]) for p in range(4)]
        if None in medians:
            problems.append(f"block {block}: an arm has no timed step")
            continue
        run = [arm_validity.Run(value=medians[p], outcome=outcomes[p]) for p in range(4)]
        runs.append(run)
        a = (medians[0] + medians[3]) / 2
        b = (medians[1] + medians[2]) / 2
        ratios.append(b / a)
    ci = interval(ratios)
    scope = disk_scope(next((_arm_record(arm).get("placement")
                             for arms in blocks.values() for arm in arms.values()
                             if _arm_record(arm).get("planned_spill_bytes_per_step")),
                            None)) if tier == "disk" else None
    extra = {"ratios": ratios, "interval": ci, "problems": problems}
    if scope is not None:
        extra["disk_scope"] = scope
    suffix = ("; on a VIRTUAL disk: describes that disk at that time, not a physical NVMe"
              if scope and scope["virtual"] else "")
    # A void block is replaced (METHOD.md) and only costs a block; anything else
    # (missing arm, wrong order, other rows, bytes off plan) removes the gate.
    hard = [problem for problem in problems if "replace the block" not in problem]
    if hard:
        return _gate(NONE, f"{name}: {hard[0]}{suffix}", **extra)
    if unknown:
        return _gate(NONE, f"{name}: an F1 check is unknown on a read arm{suffix}", **extra)
    if len(runs) < arm_validity.MIN_BLOCKS:
        return _gate(NONE, f"{name}: {len(runs)} valid blocks{suffix}", **extra)
    to_add = blocks_to_add(runs)
    extra["blocks_to_add"] = to_add
    if to_add:
        return _gate(NONE, f"{name}: V7 repeats rule asks for {to_add} more blocks{suffix}",
                     **extra)
    limit = 1 - MIN_GAIN
    if ci["high"] is not None and ci["high"] < limit:
        return _gate(PASS, f"{name}: ratio CI [{ci['low']:.3f}, {ci['high']:.3f}] < {limit}"
                           f"{suffix}", **extra)
    if ci["low"] is not None and ci["low"] >= limit:
        return _gate(FAIL, f"{name}: ratio CI [{ci['low']:.3f}, {ci['high']:.3f}] >= {limit}"
                           f"{suffix}", **extra)
    return _gate(NONE, f"{name}: ratio CI straddles {limit}{suffix}", **extra)


# ---------------------------------------------------------------------------
# break-even
# ---------------------------------------------------------------------------
def break_even(*, saved_s: float, spilled_bytes: int, write_Bps: Optional[float],  # noqa: N803
               read_Bps: Optional[float], overhead_s: float = 0.0,  # noqa: N803
               weight_bytes: int = 0, weight_ref_Bps: Optional[float] = None,  # noqa: N803
               weight_with_spill_Bps: Optional[float] = None,  # noqa: N803
               forward_compute_s: Optional[float] = None,
               backward_compute_s: Optional[float] = None) -> Dict[str, Any]:
    """Seconds per step the store saves against what it costs.

    ``saved_s`` is the recompute the store removes; ``spilled_bytes`` is written
    once in the forward and read once in the backward; ``overhead_s`` is the
    store's own work (packing, host copies) measured with weights in RAM.
    Contention is the extra time the step's weight reads take while the spill
    shares the drive: ``weight_bytes / with - weight_bytes / ref``.

    Serial: nothing overlaps. Overlapped: writes hide behind the forward
    compute and reads behind the backward compute, only the excess counts.
    The break-even bandwidth is the symmetric write = read rate at which the
    serial margin is zero.
    """
    def rate_s(nbytes: float, rate: Optional[float]) -> Optional[float]:
        if not nbytes:
            return 0.0
        return None if not rate or rate <= 0 else nbytes / rate

    write_s = rate_s(spilled_bytes, write_Bps)
    read_s = rate_s(spilled_bytes, read_Bps)
    contention_s: Optional[float] = 0.0
    if weight_bytes and spilled_bytes:
        if weight_ref_Bps and weight_with_spill_Bps:
            contention_s = weight_bytes / weight_with_spill_Bps - weight_bytes / weight_ref_Bps
        else:
            contention_s = None
    out: Dict[str, Any] = {
        "saved_s": saved_s, "overhead_s": overhead_s, "spilled_bytes": spilled_bytes,
        "write_s": write_s, "read_s": read_s, "contention_s": contention_s,
        "compute_ceiling": COMPUTE_CEILING,
        "compute_ceiling_note": "6N vs 4N FLOP per token; a potential ceiling, not measured",
    }
    if write_s is None or read_s is None or contention_s is None:
        out.update(serial_margin_s=None, overlapped_margin_s=None, breakeven_Bps=None,
                   note="a rate is missing")
        return out
    serial_cost = write_s + read_s + contention_s + overhead_s
    out["serial_cost_s"] = serial_cost
    out["serial_margin_s"] = saved_s - serial_cost
    budget = saved_s - contention_s - overhead_s
    out["breakeven_Bps"] = (2 * spilled_bytes / budget) if spilled_bytes and budget > 0 else (
        0.0 if not spilled_bytes and budget > 0 else None)
    if forward_compute_s is not None and backward_compute_s is not None:
        exposed = (max(0.0, write_s - forward_compute_s) + max(0.0, read_s - backward_compute_s))
        out["overlapped_cost_s"] = exposed + contention_s + overhead_s
        out["overlapped_margin_s"] = saved_s - out["overlapped_cost_s"]
    else:
        out["overlapped_margin_s"] = None
    return out


def _sustained_ok(rates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not rates:
        return {"ok": False, "reason": "no disk/rates.json"}
    problems = []
    for arm in rates.get("arms", []):
        if arm.get("status") != "candidate":
            problems.append(f"{arm.get('mode')}: status {arm.get('status')}")
        if not arm.get("direct"):
            problems.append(f"{arm.get('mode')}: buffered")
        if (arm.get("measured_s") or 0) < MIN_SUSTAINED_S:
            problems.append(f"{arm.get('mode')}: {arm.get('measured_s')} s measured")
        if not arm.get("fixture_exceeds_ram"):
            problems.append(f"{arm.get('mode')}: fixture not larger than RAM")
    if not rates.get("arms"):
        problems.append("no arm summaries")
    return {"ok": not problems, "reason": "; ".join(problems) or "sustained"}


def _profile_row(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = [row for row in _rows(payload, "profile") if not row.get("error")]
    return rows[0] if rows else None


def rates_match_profile(rates: Optional[Dict[str, Any]],
                        row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """V11: the sustained rates and the disk profile were measured on one host (machine
    id) and on the same spill and weight disks. Unknown on either side is not a match."""
    problems: List[str] = []
    arms = [arm for arm in (rates or {}).get("arms", []) if arm.get("status") != "invalid"]
    if not arms:
        return {"ok": False, "reason": "no valid sustained arm to bind"}
    if not row:
        return {"ok": False, "reason": "no disk-tier profile"}
    host = (row.get("host") or {}).get("machine_id")
    place = row.get("placement") or {}
    spill = ((place.get("spill") or {}).get("disk") or {}).get("id")
    weights = ((place.get("weights") or {}).get("disk") or {}).get("id")
    if not host:
        problems.append("the profile names no machine id")
    if not spill or not weights:
        problems.append("the profile names no spill/weight disk")
    for arm in arms:
        arm_host = (arm.get("host") or {}).get("machine_id")
        where = f"{arm.get('mode')} ({arm.get('source')})"
        if not arm_host:
            problems.append(f"{where}: no machine id")
        elif host and arm_host != host:
            problems.append(f"{where}: another machine ({arm_host[:12]} vs {host[:12]})")
        for side, profile_disk in (("spill", spill), ("weight", weights)):
            arm_disk = arm.get(f"{side}_disk")
            uses = arm.get("mode") in (("write", "read", "cycle", "cycle+weights")
                                       if side == "spill" else ("weights", "cycle+weights"))
            if not uses:
                continue
            if not arm_disk:
                problems.append(f"{where}: {side} disk unknown")
            elif profile_disk and arm_disk != profile_disk:
                problems.append(f"{where}: {side} disk {arm_disk}, profile {profile_disk}")
    return {"ok": not problems, "reason": "; ".join(problems) or
            f"rates and profile on machine {host[:12]}, spill {spill}, weights {weights}"}


def analyse_breakeven(session: Path) -> Dict[str, Any]:
    """Both tiers from their own profiles: weights in RAM with records in RAM (nothing
    on the drive), and weights streamed from disk with every record spilled to it.

    ``measured`` needs V9 (sustained rates) AND V11 (those rates come from the same host
    and disks as the disk profile); otherwise the whole analysis is an ``estimate``."""
    rates = _load(session / "disk" / "rates.json")
    sustained = _sustained_ok(rates)
    rates = rates or {}
    profiles = {"ram_tier_records_in_ram": _profile_row(_load(session / "profile_ram.json")),
                "disk_tier_records_spilled": _profile_row(_load(session / "profile_disk.json"))}
    bound = rates_match_profile(rates, profiles["disk_tier_records_spilled"])
    if not any(profiles.values()):
        return {"status": "missing", "detail": "no profile_ram.json / profile_disk.json",
                "V9": sustained, "V11": bound}
    cases: Dict[str, Any] = {}
    for label, row in profiles.items():
        if row is None:
            cases[label] = None
            continue
        spilled = int(row["bytes"]["planned_spill_bytes"]) if "bytes" in row else 0
        if spilled and "store_io_wait_s_per_step" not in row:
            # A profile whose overhead still contains the spill waits would charge the
            # drive twice (once inside the overhead, once from the rates).
            cases[label] = {"serial_margin_s": None, "overlapped_margin_s": None,
                            "breakeven_Bps": None,
                            "note": "the profile's store overhead includes spill I/O waits; "
                                    "rerun the profile with this harness"}
            continue
        cases[label] = break_even(
            saved_s=row["recompute_s_per_step"], spilled_bytes=spilled,
            write_Bps=rates.get("write_Bps"), read_Bps=rates.get("read_Bps"),
            overhead_s=row["store_overhead_s_per_step"],
            weight_bytes=int(row.get("weight_bytes_per_step") or 0),
            weight_ref_Bps=rates.get("weight_ref_Bps"),
            weight_with_spill_Bps=rates.get("weight_with_spill_Bps"),
            forward_compute_s=row.get("forward_compute_s_per_step"),
            backward_compute_s=row.get("backward_compute_s_per_step"))
        cases[label]["tokens_per_step"] = row.get("tokens_per_step")
        cases[label]["record_bytes_per_step"] = (row["record_capacity_bytes"]
                                                 * row["records_per_step"])
    measured = sustained["ok"] and bound["ok"]
    disk_row = profiles["disk_tier_records_spilled"]
    return {"status": "measured" if measured else "estimate", "V9": sustained, "V11": bound,
            "disk_scope": disk_scope((disk_row or {}).get("placement")),
            "cases": cases, "rates": rates}


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
def _session_scope(session: Path) -> Dict[str, Any]:
    devices, synthetic, problems = set(), False, []
    # Protocol metadata is a plan, not evidence that a measured child ran on CUDA.
    # shapes.json and sustained disk records have no training-device requirement.
    paths = [*session.glob("correctness_k*.json"), *session.glob("profile_*.json"),
             *(session / "arms").rglob("*.json")]
    for path in sorted(paths):
        payload = _load(path) or {}
        meta = payload.get("meta") or {}
        if meta.get("device"):
            devices.add(meta["device"])
        synthetic = synthetic or bool(meta.get("synthetic"))
        if meta.get("device") != "cuda" or meta.get("synthetic") is not False:
            problems.append(f"{path.relative_to(session)}: device {meta.get('device')}, "
                            f"synthetic {meta.get('synthetic')}; explicit CUDA metadata required")
    return {"devices": sorted(devices), "synthetic": synthetic, "problems": problems,
            "ok": devices == {"cuda"} and not synthetic and not problems}


def _protocol_scope(session: Path) -> Dict[str, Any]:
    protocol = _load(session / "protocol.json") or {}
    recorded = protocol.get("rule")
    if not isinstance(recorded, dict):
        return {"ok": False, "detail": "protocol.json has no readable rule"}
    profile = recorded.get("profile")
    if profile != RULE["profile"]:
        return {"ok": False, "detail": f"protocol declares {profile!r}; this evaluator supports "
                f"only {RULE['profile']!r}; use the pinned historical evaluator for replay"}
    changed = sorted(key for key in set(recorded) | set(RULE)
                     if recorded.get(key) != RULE.get(key) or (key in recorded) != (key in RULE))
    if changed:
        return {"ok": False, "detail": f"protocol.rule differs from the current rule: "
                f"{', '.join(changed)}"}
    return {"ok": True, "detail": "protocol.rule matches the current evaluator"}


def evaluate(session: Path) -> Dict[str, Any]:
    session = Path(session)
    protocol = _protocol_scope(session)
    if not protocol["ok"]:
        gates = {name: _gate(NONE, protocol["detail"]) for name in ("G1", "G2", "G3", "G4")}
        return {"rule": RULE, "validity": {"protocol": protocol}, "gates": gates,
                "measured_gates": {},
                "breakeven": {"status": "not evaluated", "detail": protocol["detail"]},
                "verdict": "NO VERDICT: PROTOCOL MISMATCH"}
    scope = _session_scope(session)
    gates = {"G1": gate_g1(session), "G2": gate_g2(session)}
    if gates["G1"]["status"] != PASS:
        reason = f"G1 is {gates['G1']['status']}: a path that is not exact has no speed"
        gates["G3"] = _gate(NONE, reason)
        gates["G4"] = _gate(NONE, reason)
    else:
        gates["G3"] = gate_speed(session, RAM_COMPARISON, tier="ram")
        gates["G4"] = gate_speed(session, DISK_COMPARISON, tier="disk")
    measured = {name: dict(gate) for name, gate in gates.items()}
    validity = {"protocol": protocol, "V1": scope}
    if not scope["ok"]:
        reason = scope["problems"][0] if scope["problems"] else (
            f"devices {scope['devices']}, synthetic {scope['synthetic']}")
        for name in gates:
            gates[name] = _gate(NONE, f"V1: {reason}; measured: {measured[name]['status']}")
    statuses = [gates[name]["status"] for name in ("G1", "G2", "G3", "G4")]
    if not scope["ok"]:
        verdict = "NO VERDICT: CPU/SYNTHETIC PILOT"
    elif FAIL in statuses[:2]:
        verdict = "DO NOT BUILD: the store is not exact or not activation-only"
    elif PASS in statuses[2:] and statuses[:2] == [PASS, PASS]:
        tiers = [tier for tier, status in zip(("RAM", "disk"), statuses[2:]) if status == PASS]
        verdict = f"BUILD: storing beats recompute on the {' and '.join(tiers)} tier"
    elif statuses[2:] == [FAIL, FAIL]:
        verdict = "DO NOT BUILD: recompute is as fast or faster on both tiers"
    else:
        verdict = "NO VERDICT"
    return {"rule": RULE, "validity": validity, "gates": gates, "measured_gates": measured,
            "breakeven": analyse_breakeven(session), "verdict": verdict}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    result = evaluate(Path(args.session))
    text = json.dumps(result, indent=2, allow_nan=False, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    for name, gate in result["gates"].items():
        sys.stdout.write(f"{name}: {gate['status']}: {gate['detail']}\n")
    sys.stdout.write(f"verdict: {result['verdict']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
