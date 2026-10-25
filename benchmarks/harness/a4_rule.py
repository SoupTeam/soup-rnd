#!/usr/bin/env python3
"""The A4 verdict, a pure function of the raw JSON of one session directory.

The rule is §3 of benchmarks/probe-a4-l2l-ram-budget-spill.md, fixed before the GPU run
and to be committed before the first GPU arm; ``RULE`` below is that text in machine form
and is written into every session's protocol.json. Usage::

    python benchmarks/harness/a4_rule.py <session dir>  [--out verdict.json]

Files read: ``preflight.json``, ``drive.json``, ``correctness_k<k>.json`` (and
``correctness_k<k>_det.json`` when the default run was not reproducible),
``cleanup.json``, ``arms/<comparison>/b<block>_p<position>_<A|B>.json``.
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

import a4_activation_budget as budget  # noqa: E402
import arm_validity  # noqa: E402

PASS, FAIL, NONE = "pass", "fail", "no verdict"
COST_LIMIT = 0.10
HIDDEN_LIMIT = 0.05
INTERFERENCE_FLOOR = 0.90
CORRECTNESS_KS = (2, 3, 4)
COST_GATE_COMPARISONS = ("ram_k2_half", "ram_k4_half")
INTERFERENCE_COMPARISON = "disk_k4_same_drive"
SPILL_SCOPES = ("local", "virtual")

#: v1 judged the 2026-10-25 GPU session; v2 is fixed before the next one. Only G2 differs.
RULE_VERSION = "v2"

RULE = {
    "profile": "a4-ram-budget-spill-v2",
    "G1": "for k in 2,3,4: every planned budget (0.5, 0.25, min) gives per-micro-batch losses "
          "(forward) and nonzero LoRA grads (backward) torch.equal to the all-pinned L2L AND to "
          "GA; references reproducible (GA vs GA, pinned vs pinned), else the --deterministic "
          "rerun; the corrupted-spill-read control's grads must DIFFER, else no verdict",
    "G2": "every arm: the store holds exactly the plan's bytes (power-of-two pinned blocks, "
          "rings at chunk+4096) and <= its budget (cap.ok). excess = (CUDA host allocator peak "
          "during the timed steps - its baseline before the store) - the store's held bytes. "
          "Reference per block: both A arms (positions 0 and 3) present, not void, spilling "
          "no layer, with counters, and their excesses EQUAL; else that block's B arms get no "
          "verdict. A B arm fails if its excess > the reference. A missing counter is no "
          "verdict. With a declared host total, the process peak RSS <= that total",
    "G3": "cleanup: Ctrl-C mid-backward, exception mid-forward and ENOSPC mid-step each raise "
          "their named error, leave no spill file, no spill thread, and return the host "
          "allocator to its baseline; ENOSPC names SPILL_ENOSPC with the arithmetic; and no "
          "arm leaves a spill file",
    "G4": f"ram_k2_half and ram_k4_half: mean per-block step-time overhead of B (spill) over A "
          f"(all pinned), two-sided 95% t interval: pass if its upper bound < "
          f"{COST_LIMIT:.0%}, fail if its lower bound >= {COST_LIMIT:.0%}, else no verdict",
    "G5": f"disk_k4_same_drive: weight read GB/s while busy, B (spill on the weights' drive) "
          f"over A (no spill), per-block ratio, 95% t interval: pass (no interference past "
          f"10%) if lower bound >= {INTERFERENCE_FLOOR}, fail if upper bound < "
          f"{INTERFERENCE_FLOOR}, else no verdict",
    "V1": "required F1 checks (suspend, power, foreign_reads) ok on every arm a gate reads; a "
          "block with a void arm is replaced (METHOD.md), an unknown one removes the gate",
    "V2": "spill volume scope local or virtual, never tmpfs/overlay/network; unbuffered round "
          "trip ok; every spilling arm's store is direct",
    "V3": "device cuda; a CPU pilot never gets a verdict",
    "V4": "A and B of a block train on the same token rows (ids fingerprint)",
    "V5": "finite losses on every timed step",
    "V6": "the plan happened: budget '1' spills nothing; any other budget spills > 0 layers "
          "and writes and reads exactly spill_file_bytes per step",
    "V7": "blocks: the F1 repeats rule has no block left to add (blocks_to_add == 0)",
    "V8": "disk-tier arms: the weight source reads with direct I/O",
    "V9": "disk_k4_same_drive: every spilling arm's OS placement records (resolved path, "
          "mount, whole physical disk) show the spill file and the weight shards on one "
          "physical disk; unknown placement removes G5",
    "order": "A-B-B-A blocks, fresh process per arm, comparisons in the order listed",
    "hidden": f"informational: overhead upper bound <= {HIDDEN_LIMIT:.0%} is 'hidden'; the "
              f"hidden share of spill I/O is 1 - (put+get wait)/(write+read busy)",
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
    mean = statistics.fmean(values)
    if n < 2:
        return {"n": n, "mean": mean, "low": None, "high": None}
    half = t975(n - 1) * statistics.stdev(values) / math.sqrt(n)
    return {"n": n, "mean": mean, "low": mean - half, "high": mean + half}


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: Optional[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [row for row in (payload or {}).get("records", []) if row.get("kind") == kind]


def _gate(status: str, detail: str, **extra: Any) -> Dict[str, Any]:
    return {"status": status, "detail": detail, **extra}


def _combine(parts: Sequence[Dict[str, Any]]) -> str:
    statuses = [part["status"] for part in parts]
    return FAIL if FAIL in statuses else (NONE if NONE in statuses or not parts else PASS)


# ---------------------------------------------------------------------------
# G1: bit-exactness
# ---------------------------------------------------------------------------
def _exact(pair: Dict[str, Any]) -> bool:
    return bool(pair["backward"]["equal"] and pair["forward_equal"]
                and not pair["backward"]["all_zero"])


def _g1_k(default: Optional[Dict[str, Any]], det: Optional[Dict[str, Any]], k: int
          ) -> Dict[str, Any]:
    chosen = _rows(default, "correctness")
    row = chosen[0] if chosen else None
    reproducible = row is not None and _exact(row["ga_vs_ga"]) and _exact(row["pinned_vs_pinned"])
    if not reproducible:
        det_rows = _rows(det, "correctness")
        row = det_rows[0] if det_rows else None
        if row is None:
            return _gate(NONE, f"k={k}: references not reproducible and no deterministic rerun")
        if not (_exact(row["ga_vs_ga"]) and _exact(row["pinned_vs_pinned"])):
            return _gate(NONE, f"k={k}: references not reproducible even with --deterministic")
    control = row.get("control") or {}
    if "vs_pinned" not in control:
        return _gate(NONE, f"k={k}: no expected-to-differ control ({control.get('skipped')})")
    if control["vs_pinned"]["backward"]["equal"]:
        return _gate(NONE, f"k={k}: the corrupted-read control did not differ; the "
                     f"comparison cannot fail, so it proves nothing")
    if not _exact(row["pinned_vs_ga"]):
        return _gate(FAIL, f"k={k}: all-pinned L2L differs from GA")
    planned = {spec: entry for spec, entry in row["budgets"].items() if "plan" in entry}
    if not any(entry["plan"]["spill_layers"] for entry in planned.values()):
        return _gate(NONE, f"k={k}: no budget spilled anything")
    bad = []
    for spec, entry in sorted(planned.items()):
        for ref in ("vs_pinned", "vs_ga"):
            pair = entry[ref]
            if not pair["forward_equal"]:
                bad.append(f"budget {spec} {ref}: forward (losses) differ")
            if not pair["backward"]["equal"] or pair["backward"]["all_zero"]:
                bad.append(f"budget {spec} {ref}: backward (grads) differ, max_abs "
                           f"{pair['backward']['max_abs']}")
    if bad:
        return _gate(FAIL, f"k={k}: " + "; ".join(bad))
    spilled = {spec: entry["plan"]["spill_layers"] for spec, entry in planned.items()}
    return _gate(PASS, f"k={k}: forward and backward exact at spill layers {spilled}; "
                 f"control differs", deterministic=bool(row.get("deterministic")))


def gate_g1(root: Path) -> Dict[str, Any]:
    parts = [_g1_k(_load(root / f"correctness_k{k}.json"),
                   _load(root / f"correctness_k{k}_det.json"), k) for k in CORRECTNESS_KS]
    return _gate(_combine(parts), "; ".join(p["detail"] for p in parts), parts=parts)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def load_arms(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Every arm row, by comparison, with its file's meta attached as ``_meta``.

    ``_file`` is ``<comparison>/<basename>``: basenames such as ``b00_p1_B.json`` repeat
    in every comparison, so a bare basename would let one comparison's arm overwrite
    another's finding wherever arms are keyed by file (G2's budget check)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    arms_dir = root / "arms"
    if not arms_dir.is_dir():
        return out
    for directory in sorted(p for p in arms_dir.iterdir() if p.is_dir()):
        for path in sorted(directory.glob("b*_p*_*.json")):
            payload = _load(path) or {}
            for index, row in enumerate(_rows(payload, "arm")):
                name = f"{directory.name}/{path.name}"
                out.setdefault(directory.name, []).append(
                    {**row, "_meta": payload.get("meta", {}),
                     "_file": name if index == 0 else f"{name}#{index}"})
    return out


def arm_problems(row: Dict[str, Any]) -> List[str]:
    """Why an arm's numbers cannot enter a gate (V2, V3, V5, V6, V8); empty if they can."""
    if row.get("skipped"):
        return [f"skipped: {row['skipped']}"]
    why = []
    meta = row["_meta"]
    if meta.get("device") != "cuda":
        why.append("V3: not a CUDA arm")
    if not row.get("finite"):
        why.append("V5: a non-finite loss")
    plan = row["plan"]
    if row["budget"] == "1":
        if plan["spill_layers"]:
            why.append("V6: the all-pinned arm spilled")
    else:
        if not plan["spill_layers"]:
            why.append("V6: the spill arm spilled nothing")
        expected = plan["spill_file_bytes"]
        moved = {(rec["spill_written"], rec["spill_read"]) for rec in row["records"]}
        if moved != {(expected, expected)}:
            why.append(f"V6: spill bytes per step {sorted(moved)} != {expected}")
        if row["store"].get("direct") is not True:
            why.append("V2: the spill backend is not unbuffered")
        volume = row.get("volume") or {}
        if volume.get("scope") not in SPILL_SCOPES or volume.get("direct_io") is not True:
            why.append(f"V2: spill volume scope {volume.get('scope')}, "
                       f"direct_io {volume.get('direct_io')}")
    if row["tier"] == "disk" and meta.get("direct_io") is not True:
        why.append("V8: the weight source is not on direct I/O")
    return why


def blocks_of(rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    blocks: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for row in rows:
        if row.get("skipped"):
            continue
        blocks.setdefault(int(row["block"]), {})[int(row["position"])] = row
    return {b: arms for b, arms in blocks.items() if len(arms) == len(arm_validity.BLOCK_ORDER)}


def comparison(rows: Sequence[Dict[str, Any]], metric: str = "value",
               ratio: bool = False) -> Dict[str, Any]:
    """Validity, the repeats rule and the per-block effect of one A-B-B-A comparison.

    The effect per block is (mean B - mean A) / mean A, or mean B / mean A with
    ``ratio``. Blocks with a void arm are dropped (METHOD.md: replaced)."""
    blocks = blocks_of(rows)
    why: List[str] = [f"{row['_file']}: {p}" for row in rows for p in arm_problems(row)]
    ordered = [[blocks[b][p] for p in range(4)] for b in sorted(blocks)]
    runs = [[arm_validity.Run(float(arm["value"]), str(arm["outcome"])) for arm in block]
            for block in ordered]
    valid = [block for block, run in zip(ordered, runs)
             if all(r.outcome != arm_validity.VOID for r in run)]
    unknown = [arm["_file"] for block in valid for arm in block if arm["outcome"] != "ok"]
    if unknown:
        why.append(f"V1: unknown F1 outcome on {unknown}")
    for block in valid:
        ids = {json.dumps(arm["ids"]["row_sha256"]) for arm in block}
        if len(ids) != 1:
            why.append(f"V4: block {block[0]['block']} trains on different rows")
    due = arm_validity.blocks_to_add(runs) if runs else arm_validity.PILOT_BLOCKS
    if due:
        why.append(f"V7: the repeats rule wants {due} more block(s)")
    effects = []
    for block in valid:
        a = [arm[metric] for arm, name in zip(block, arm_validity.BLOCK_ORDER) if name == "A"]
        b = [arm[metric] for arm, name in zip(block, arm_validity.BLOCK_ORDER) if name == "B"]
        if None in a or None in b:
            why.append(f"block {block[0]['block']}: {metric} missing")
            continue
        mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
        effects.append(mean_b / mean_a if ratio else (mean_b - mean_a) / mean_a)
    result: Dict[str, Any] = {"blocks": len(ordered), "valid_blocks": len(valid),
                              "void_blocks": len(ordered) - len(valid), "problems": why,
                              "effects": effects}
    result["interval"] = interval(effects) if len(effects) >= 2 else None
    if len(effects) < 2:
        why.append("fewer than two valid blocks")
    spill = [arm for block in valid for arm in block if arm["budget"] != "1"]
    if spill:
        result["hidden_share_median"] = statistics.median(
            arm["hidden_share"] for arm in spill if arm["hidden_share"] is not None)
        result["spill_write_gb_per_s_median"] = statistics.median(
            arm["spill_write_gb_per_s"] for arm in spill if arm["spill_write_gb_per_s"])
        result["spill_read_gb_per_s_median"] = statistics.median(
            arm["spill_read_gb_per_s"] for arm in spill if arm["spill_read_gb_per_s"])
        result["spill_layers"] = spill[0]["plan"]["spill_layers"]
    return result


def gate_g4(arms: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    parts = []
    for name in COST_GATE_COMPARISONS:
        rows = arms.get(name, [])
        if not rows:
            parts.append(_gate(NONE, f"{name}: no arms"))
            continue
        result = comparison(rows)
        ci = result["interval"]
        if result["problems"] or ci is None:
            parts.append(_gate(NONE, f"{name}: " + "; ".join(result["problems"]), **result))
        elif ci["high"] < COST_LIMIT:
            parts.append(_gate(PASS, f"{name}: overhead {ci['mean']:+.2%} "
                               f"[{ci['low']:+.2%}, {ci['high']:+.2%}] < {COST_LIMIT:.0%}"
                               + (" (hidden)" if ci["high"] <= HIDDEN_LIMIT else ""), **result))
        elif ci["low"] >= COST_LIMIT:
            parts.append(_gate(FAIL, f"{name}: overhead {ci['mean']:+.2%} "
                               f"[{ci['low']:+.2%}, {ci['high']:+.2%}] >= {COST_LIMIT:.0%}",
                               **result))
        else:
            parts.append(_gate(NONE, f"{name}: overhead {ci['mean']:+.2%} "
                               f"[{ci['low']:+.2%}, {ci['high']:+.2%}] straddles "
                               f"{COST_LIMIT:.0%}", **result))
    return _gate(_combine(parts), "; ".join(p["detail"] for p in parts), parts=parts)


def placement_problems(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """V9: every spilling arm of the same-drive comparison shows, from the OS's own
    placement records, that its spill file and the weights it reads share one physical
    disk. Unknown placement is not evidence of the same drive (nor of another one)."""
    import a4_box

    why = []
    for row in rows:
        if row.get("skipped") or not (row.get("plan") or {}).get("spill_layers"):
            continue
        place = row.get("placement") or {}
        same, text = a4_box.same_physical_disk(place.get("spill"), place.get("weights"))
        if same is not True:
            why.append(f"V9: {row['_file']}: spill and weights not shown on one disk ({text})")
    return why


def gate_g5(arms: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    rows = arms.get(INTERFERENCE_COMPARISON, [])
    if not rows:
        return _gate(NONE, f"{INTERFERENCE_COMPARISON}: no arms")
    result = comparison(rows, metric="weight_read_gb_per_s", ratio=True)
    ci = result["interval"]
    step = comparison(rows)
    result["step_overhead"] = step["interval"]
    result["problems"] = list(result["problems"]) + placement_problems(rows)
    if result["problems"] or ci is None:
        return _gate(NONE, "; ".join(result["problems"]), **result)
    text = (f"weight reads at {ci['mean']:.3f}x [{ci['low']:.3f}, {ci['high']:.3f}] of the "
            f"no-spill rate with spill writes on the same drive")
    if ci["low"] >= INTERFERENCE_FLOOR:
        return _gate(PASS, text + f": no interference past {1 - INTERFERENCE_FLOOR:.0%}",
                     **result)
    if ci["high"] < INTERFERENCE_FLOOR:
        return _gate(FAIL, text + ": the spill slows the weight reads", **result)
    return _gate(NONE, text + f": straddles {INTERFERENCE_FLOOR}", **result)


# ---------------------------------------------------------------------------
# G2, G3
# ---------------------------------------------------------------------------
def arm_excess(row: Dict[str, Any]) -> Optional[int]:
    """Host-allocator growth during the timed steps that the store does not hold:
    (allocator peak - allocator baseline) - the store's held bytes. None without a counter."""
    pinned = row.get("pinned") or {}
    base, peak = pinned.get("allocator_baseline_bytes"), pinned.get("allocator_peak_bytes")
    held = (row.get("cap") or {}).get("held_bytes")
    if base is None or peak is None or held is None:
        return None
    return int(peak) - int(base) - int(held)


def block_reference(arms_of_block: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """G2's reference for one block: the excess both A arms share.

    Both A arms (positions 0 and 3) must be present, not skipped, not void, spill no layer
    and carry the counters, and their excesses must be EQUAL; any other case leaves the
    block's B arms without a reference (no verdict). Nothing is averaged or chosen."""
    a_rows = [arms_of_block.get(p) for p in (0, 3)]
    why = []
    for position, row in zip((0, 3), a_rows):
        if row is None or row.get("skipped"):
            why.append(f"A at position {position} missing")
        elif row.get("arm") != "A":
            why.append(f"position {position} is not an A arm")
        elif row.get("outcome") == arm_validity.VOID:
            why.append(f"A at position {position} void")
        elif (row.get("plan") or {}).get("spill_layers") != 0:
            why.append(f"A at position {position} spilled")
        elif arm_excess(row) is None:
            why.append(f"A at position {position} has no allocator counter")
    if why:
        return {"bytes": None, "why": "; ".join(why)}
    first, second = (arm_excess(row) for row in a_rows)
    if first != second:
        return {"bytes": None, "why": f"the A arms' excesses differ ({first:,} B vs "
                                      f"{second:,} B): not reproducible"}
    return {"bytes": first, "why": None}


def _cap_problems(row: Dict[str, Any], reference: Optional[Dict[str, Any]]
                  ) -> List[Tuple[str, str]]:
    """(FAIL or NONE, text) per finding for one arm; ``reference`` only for a B arm."""
    why: List[Tuple[str, str]] = []
    cap = row.get("cap") or {}
    if not cap.get("ok"):
        why.append((FAIL, f"store: {cap.get('reason')}"))
    excess = arm_excess(row)
    if excess is None:
        why.append((NONE, "no CUDA host allocator counters or no held bytes"))
    elif reference is not None:
        if reference["bytes"] is None:
            why.append((NONE, f"no reference: {reference['why']}"))
        elif excess > reference["bytes"]:
            why.append((FAIL, f"allocated {excess:,} B outside its store during the timed "
                              f"steps, over the {reference['bytes']:,} B the block's A arms "
                              f"allocated outside theirs"))
    split = row.get("split")
    rss = (row.get("process_memory") or {}).get("peak_rss")
    if split is not None:
        if rss is None:
            why.append((NONE, "no peak RSS for the declared host total"))
        elif rss > split["total_bytes"]:
            why.append((FAIL, f"peak RSS {rss:,} B over the declared "
                              f"{split['total_bytes']:,} B"))
    return why


def gate_g2(arms: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    rows = [(name, row) for name, group in arms.items() for row in group
            if not row.get("skipped")]
    if not rows:
        return _gate(NONE, "no arm ran")
    by_block: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = {}
    for name, row in rows:
        by_block.setdefault((name, int(row["block"])), {})[int(row["position"])] = row
    references = {key: block_reference(block) for key, block in by_block.items()}
    problems: Dict[str, List[Tuple[str, str]]] = {}
    excess: Dict[str, Optional[int]] = {}
    for name, row in rows:
        if row["_file"] in problems:  # never let one arm's finding replace another's
            raise ValueError(f"two arm rows share the id {row['_file']!r}")
        reference = (references[(name, int(row["block"]))] if row.get("arm") == "B"
                     else None)
        problems[row["_file"]] = _cap_problems(row, reference)
        excess[row["_file"]] = arm_excess(row)
    failed = {name: [text for kind, text in why if kind == FAIL]
              for name, why in problems.items() if any(kind == FAIL for kind, _ in why)}
    unknown = {name: [text for _, text in why] for name, why in problems.items()
               if why and name not in failed}
    extra = {"references": {f"{name}/b{block:02d}": ref
                            for (name, block), ref in sorted(references.items())},
             "excess_bytes": excess}
    if failed:
        return _gate(FAIL, f"{len(failed)} arm(s) over: {failed}", **extra)
    if unknown:
        return _gate(NONE, f"{len(unknown)} arm(s) without a G2 verdict: {unknown}", **extra)
    counts = sorted({ref["bytes"] for ref in references.values() if ref["bytes"] is not None})
    return _gate(PASS, f"{len(rows)} arms: held == plan <= budget; every B arm's excess "
                 f"outside its store <= its block's A excess (A excesses seen: {counts} B)",
                 **extra)


def gate_g3(root: Path, arms: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    rows = _rows(_load(root / "cleanup.json"), "cleanup")
    if not rows:
        return _gate(NONE, "no cleanup record")
    record = rows[0]
    bad = [s["scenario"] for s in record["scenarios"] if not s["ok"]]
    names = {s["scenario"] for s in record["scenarios"]}
    missing = {"ctrl_c_mid_backward", "exception_mid_forward",
               "enospc_injected_mid_step"} - names
    enospc = [s for s in record["scenarios"] if s["scenario"].startswith("enospc")]
    unnamed = [s["scenario"] for s in enospc
               if not (s["message"] or "").startswith("SPILL_ENOSPC")
               or "layers x k=" not in (s["message"] or "")]
    leaked = [row["_file"] for group in arms.values() for row in group
              if not row.get("skipped")
              and (row.get("spill_file_exists_after") or row.get("leftover_spill_files"))]
    if missing:
        return _gate(NONE, f"scenarios not run: {sorted(missing)}")
    if bad or unnamed or leaked:
        return _gate(FAIL, f"failed {bad}; ENOSPC without arithmetic {unnamed}; arms that "
                     f"left a spill file {leaked}")
    unmeasured = [s["scenario"] for s in record["scenarios"]
                  if s.get("allocator_baseline_bytes") is None
                  or s.get("allocator_after_bytes") is None]
    if unmeasured:
        return _gate(NONE, f"{sorted(names)}: named, no file, no thread; the return of "
                     f"page-locked memory was not measured on {unmeasured} (no CUDA host "
                     f"allocator counters)")
    return _gate(PASS, f"{sorted(names)}: named, no file, no thread, pinned memory returned")


# ---------------------------------------------------------------------------
# arithmetic and the verdict
# ---------------------------------------------------------------------------
def tokens_arithmetic(arms: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Tokens per step each measured budget reaches with and without spill (arithmetic)."""
    seen, out = set(), []
    for group in arms.values():
        for row in group:
            if row.get("skipped"):
                continue
            plan, meta = row["plan"], row["_meta"]
            key = (plan["budget_bytes"], plan["chunk_bytes"], plan["ring"], plan["n_layers"])
            if key in seen:
                continue
            seen.add(key)
            free = (row.get("volume") or {}).get("free_bytes")
            out.append({"budget": row["budget"], "k_measured": row["k"], **budget.tokens_per_step(
                plan["budget_bytes"], n_layers=plan["n_layers"], seq=int(meta.get("seq", 512)),
                chunk_bytes=plan["chunk_bytes"], ring=plan["ring"], pin=plan["pin"],
                spill_free_bytes=free, k_limit=4096)})
    return out


def verdict_of(gates: Dict[str, Dict[str, Any]], cpu: bool) -> str:
    if cpu:
        return "NO VERDICT: SYNTHETIC/CPU PILOT"
    safety = [gates[g]["status"] for g in ("G1", "G2", "G3")]
    if FAIL in safety:
        return "SPILL NOT SAFE: A3 must not ship the spill path as built"
    if NONE in safety:
        return "NO VERDICT"
    cost = gates["G4"]["status"]
    if cost == NONE:
        return "SPILL SAFE, COST NO VERDICT"
    tail = ""
    if gates["G5"]["status"] == FAIL:
        tail = "; on the disk tier spill to a drive the weights are not read from"
    if cost == PASS:
        return "SPILL READY FOR A3: automatic fallback past the RAM budget" + tail
    return "SPILL SAFE BUT COSTLY: an explicit capacity tier, not an automatic fallback" + tail


def evaluate(results_dir: str) -> Dict[str, Any]:
    root = Path(results_dir)
    arms = load_arms(root)
    gates = {"G1": gate_g1(root), "G2": gate_g2(arms), "G3": gate_g3(root, arms),
             "G4": gate_g4(arms), "G5": gate_g5(arms)}
    devices = {row["_meta"].get("device") for group in arms.values() for row in group}
    for k in CORRECTNESS_KS:
        payload = _load(root / f"correctness_k{k}.json")
        if payload:
            devices.add(payload.get("meta", {}).get("device"))
    cpu = "cpu" in devices
    informative = {name: comparison(rows) for name, rows in arms.items()
                   if name not in COST_GATE_COMPARISONS + (INTERFERENCE_COMPARISON,)}
    drive = _rows(_load(root / "drive.json"), "drive")
    preflight = _rows(_load(root / "preflight.json"), "preflight")
    verdict = verdict_of(gates, cpu)
    return {
        "rule_version": RULE_VERSION, "rule": RULE, "verdict": verdict, "gates": gates,
        "informative": informative,
        "drive": drive[0] if drive else None,
        "preflight_ok": preflight[0]["volume"]["ok"] if preflight else None,
        "tokens_per_step": tokens_arithmetic(arms),
        "summary": {"verdict": verdict, "rule_version": RULE_VERSION,
                    **{g: v["status"] for g, v in gates.items()}, "cpu_pilot": cpu},
    }


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    result = evaluate(args.results_dir)
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        # LF on every OS, so a re-analysis on Windows is byte-identical to the box's file.
        Path(args.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["summary"], indent=2))
    for name, gate in result["gates"].items():
        print(f"{name} {gate['status']:<10} {gate['detail'][:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
