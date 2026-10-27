#!/usr/bin/env python3
"""A3 stage 2: the verdict of the GPU check of activation spill in ``training.l2l``.

A pure function of one session directory written by ``a3_spill_stand.py session``. The
criteria are §3 of benchmarks/a3-spill-gpu-check.md, fixed before any rental; ``RULE`` is
that text in machine form and goes into every session's protocol.json. Usage::

    python benchmarks/harness/a3_spill_rule.py <session dir> [--out verdict.json]

Files read: ``session.json`` (each child's return code, the kernel's OOM-kill lines the
parent read for it, and the parent's own observations of the spill directory and of the
signals it sent), ``calibration.json``, ``arms/<arm>.json``, ``compare.json``,
``cleanup/<arm>.json`` and ``cost/b<NN>_p<P>_<A|B>.json``. Statuses are PASS, FAIL and NO
VERDICT per criterion, each with its arithmetic. Standard library only (``arm_validity`` is
too).

Rule v2 (protocol §3.6, after session 1): an OOM kill the kernel logs for any child at the
tight limit after its first ``training_step`` entry, calibration checks included, fails
criterion 2. Rule v1 is this file at ``bda4a00a``; session 1's ``verdict.json`` is its output.
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

PASS, FAIL, NONE = "PASS", "FAIL", "NO VERDICT"
RULE_VERSION = "v2"
COST_LIMIT = 0.10
MAX_BLOCKS = 12
#: Spilled layers the calibration aims at (of Mistral-7B's 32). Reported, not a criterion.
SPILL_BAND = (12, 20)
#: Description only (protocol §4): a build or training margin below this is warned about and
#: recorded; it decides nothing. The session stops only on a build margin below 0.
OOM_MARGIN_WARN_BYTES = 256 << 20
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

NOSPILL = ("l2l_nospill_1", "l2l_nospill_2")
SPILL = "l2l_spill"
GA = ("ga_1", "ga_2")
FLOOR = "ga_floor"
CORRECTNESS_ARMS = (*NOSPILL, SPILL, *GA, FLOOR)
#: The SDPA backend each correctness arm must have pinned (V2).
SDPA = {**{arm: "MATH" for arm in (*NOSPILL, SPILL, *GA)}, FLOOR: "EFFICIENT_ATTENTION"}
CLEANUP_ARMS = ("sigint", "kill9", "kill9_next")

RULE = {
    "profile": "a3-spill-gpu-check-v2",
    "1": "l2l_spill (tight limit; spill_layers > 0) equals l2l_nospill_1 (generous limit; "
         "nothing spilled) bit for bit: SHA-256 of the raw bytes, dtype and shape of every "
         "LoRA gradient at every optimizer step, the logged losses, the number of optimizer "
         "steps and the lr at every step. l2l_nospill_1 and l2l_nospill_2 must be equal the "
         "same way, else NO VERDICT (not FAIL). Unequal spill: FAIL",
    "1b": "reported, the A3 stand's own criterion: l2l_mistral_stand.decide on (ga_1, ga_2, "
          "l2l_spill) with ga_floor (GA on EFFICIENT_ATTENTION) in bf16, from the unpublished "
          "gradient tensors (compare.json). It does not change the spill verdict",
    "2": "l2l_spill, l2l_nospill_1, l2l_nospill_2: child exit 0, no oom_kill increment in its "
         "scope's memory.events, scope memory.peak <= its MemoryMax; excess = CUDA host "
         "allocator allocated_bytes.current at the last optimizer step - at train begin - the "
         "plan's host_bytes; excess(l2l_spill) <= excess(l2l_nospill_1). A kill or an oom_kill "
         "is a FAIL wherever it happens; the detail says 'before first training_step (build)' "
         "(no first-step marker, or oom_kill already raised there) or 'after first "
         "training_step' (the product's own headroom did not cover training). v2: every other "
         "child at the tight limit (calibration checks, sigint, kill9, kill9_next, cost B) "
         "fails criterion 2 when the kernel log the parent read after its exit shows the "
         "memory cgroup OOM-killing its pid after its first-step marker; a return code of -9 "
         "alone is not that evidence, and the SIGKILL the parent sends kill9 is not an OOM. An "
         "OOM kill before the marker is the stand's limit (build): no verdict. A child killed "
         "with no kernel log to tell: no verdict",
    "3": "normal: after train() and after the process exit of l2l_spill no spill file is left "
         "(spill_files); sigint: one SIGINT while optimizer step 2 runs, the child exits (any "
         "code) and no spill file is left; kill9: SIGKILL while step 2 runs leaves the spill "
         "file, the next child in the same directory removes it at start (stale_removed holds "
         "it, it is gone) and trains its optimizer steps. All three must hold",
    "4": f"A-B-B-A blocks of fresh processes, k 4, seq 512, rows all 512 tokens, 2 warm-up and "
         f"5 timed optimizer steps; A = L2L at the generous limit (no spill), B = L2L at the "
         f"tight limit (spill); value = median timed step; per-block overhead (mean B - mean "
         f"A) / mean A; two-sided 95% t interval. Upper < {COST_LIMIT:.0%}: PASS (below "
         f"{COST_LIMIT:.0%} confirmed on a faster machine); lower >= {COST_LIMIT:.0%}: FAIL "
         f"(the fallback stays automatic; the console message and the docs state the measured "
         f"overhead); else NO VERDICT. Blocks: arm_validity.blocks_to_add, at most "
         f"{MAX_BLOCKS}",
    "V1": "device cuda in every child; a CPU session never gets a verdict",
    "V2": "correctness arms: CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA, strict deterministic "
          "algorithms (warn_only off), SDPA pinned (MATH; ga_floor EFFICIENT_ATTENTION)",
    "V3": "every child reads back from /proc/self/cgroup that its scope's memory.max is the "
          "limit the parent planned for it and memory.swap.max is 0 (cgroup v2); a mismatch "
          "removes every criterion that reads that child",
    "V4": "spill arms spilled (spill_layers > 0, store bytes written > 0 and read > 0); "
          "no-spill arms spilled nothing",
    "V5": "compared arms trained on the same rows (SHA-256 per row of ids and labels)",
    "V6": "cost arms: F1 ArmWatch with suspend, power and foreign_reads required; a block with "
          "a void arm is replaced (METHOD.md), an unknown outcome removes criterion 4",
    "V7": "cost: the repeats rule has no block left to add",
    "V8": "trained arms: normal exit, finite losses and gradients, a nonzero LoRA gradient at "
          "every step, the expected number of optimizer steps",
    "stop": "if criterion 1 FAILS the parent runs no cost block: no new rentals, find the "
            "cause. The cost blocks run only when criterion 1 PASSES. v2: a kernel OOM kill "
            "of a child at the tight limit after its first training_step stops the session "
            "(stopped_oom_in_training)",
    "calibration": "a child at the generous limit reads its scope's memory.current - "
                   "inactive_file before the first step (the product plans the store then) "
                   "and sets limit = working set + the least available memory whose product "
                   "budget (l2l_spill.budget_of: min(0.7 x available, available - "
                   "RAM_RESERVE_BYTES)) holds the target host_bytes plus half a layer, the "
                   "target being 16 of 32 layers spilled; one check child at that limit reads "
                   "the product's own plan, and if spill_layers is outside [12, 20] the limit "
                   "is recomputed once from the check's own plan-time numbers. Before any child "
                   "runs at a tight limit: U = memory.current - (active_file + inactive_file - "
                   "file_dirty - file_writeback) - slab_reclaimable (what reclaim cannot free, "
                   "swap 0) of the calibration child, sampled every 50 ms and at build end, the "
                   "first training_step entry and each step end; U0 at that entry; build "
                   "transient = max(U before the entry) - U0; training transient = max(U after "
                   "it - the store's pinned growth, at most the plan's host_bytes) - U0. Build "
                   "margin = limit - U0 - build transient: below 0 (or unknown) the session "
                   "stops (stopped_calibration_oom_risk). Training margin = limit - U0 - "
                   "training transient - target host_bytes: description only. Either margin "
                   f"below {OOM_MARGIN_WARN_BYTES >> 20} MiB: a warning, recorded, deciding "
                   "nothing. The limit is never raised",
    "order": "calibration, calibration_check, l2l_nospill_1, l2l_nospill_2, l2l_spill, ga_1, "
             "ga_2, ga_floor, compare, sigint, kill9, kill9_next, then the cost blocks",
}

#: Student t quantiles for a two-sided 95% interval, by degrees of freedom (as a4_rule).
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


def _crit(status: str, detail: str, **extra: Any) -> Dict[str, Any]:
    return {"status": status, "detail": detail, **extra}


def _combine(parts: Sequence[Dict[str, Any]]) -> str:
    statuses = [part["status"] for part in parts]
    return FAIL if FAIL in statuses else (NONE if NONE in statuses or not parts else PASS)


def _named(name: str, problems: Sequence[str]) -> List[str]:
    return [f"{name}: {problem}" for problem in problems]


# ---------------------------------------------------------------------------
# V-checks of one child
# ---------------------------------------------------------------------------
def child_problems(rec: Dict[str, Any], entry: Optional[Dict[str, Any]]) -> List[str]:
    """V1 and V3: a CUDA run inside a scope that carries exactly the limit the parent planned
    for it (``session.json``), with swap 0. The child's own claim is not the plan."""
    why = []
    if rec.get("device") != "cuda":
        why.append(f"V1: device {rec.get('device')!r}, not cuda")
    planned = (entry or {}).get("memory_max")
    cgroup = rec.get("cgroup_start") or {}
    if not isinstance(planned, int) or isinstance(planned, bool):
        why.append(f"V3: no planned limit in session.json ({planned!r})")
    elif cgroup.get("version") != 2:
        why.append(f"V3: no cgroup v2 readback ({cgroup.get('version')!r})")
    elif cgroup.get("memory_max") != planned or rec.get("expected_memory_max") != planned:
        why.append(f"V3: memory.max read back {cgroup.get('memory_max')!r}, planned {planned}")
    elif cgroup.get("swap_max") != 0:
        why.append(f"V3: memory.swap.max read back {cgroup.get('swap_max')!r}, not 0")
    return why


def determinism_problems(rec: Dict[str, Any], arm: str) -> List[str]:
    """V2 for a correctness arm."""
    det = rec.get("deterministic") or {}
    expected = {"CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG, "algorithms": True,
                "warn_only": False, "sdpa": SDPA[arm]}
    wrong = {key: det.get(key) for key, value in expected.items() if det.get(key) != value}
    return [f"V2: deterministic setup {wrong} differs from {expected}"] if wrong else []


def training_problems(rec: Dict[str, Any]) -> List[str]:
    """V8: the arm trained, every number finite, no all-zero step, every planned step."""
    why = []
    if rec.get("exit") != "normal":
        why.append(f"V8: exit {rec.get('exit')!r} ({rec.get('error')})")
    grads = rec.get("grads") or []
    if rec.get("steps") != rec.get("expected_steps") or len(grads) != rec.get("steps"):
        why.append(f"V8: {rec.get('steps')} optimizer steps and {len(grads)} gradient records, "
                   f"{rec.get('expected_steps')} expected")
    losses = [row.get("loss") for row in rec.get("logs") or []]
    if not losses or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses):
        why.append(f"V8: losses {losses} are missing or not finite")
    if not all(digest.get("finite") for step in grads for digest in step.values()):
        why.append("V8: a non-finite LoRA gradient")
    if not all(any(digest.get("nonzero") for digest in step.values()) for step in grads):
        why.append("V8: a step with all-zero LoRA gradients")
    return why


def spill_problems(rec: Dict[str, Any], *, spilled: bool) -> List[str]:
    """V4: the spill happened where planned, and nowhere else."""
    layers = (rec.get("spill_plan") or {}).get("spill_layers")
    if not spilled:
        return [] if layers == 0 else [f"V4: the no-spill arm planned spill_layers {layers!r}"]
    io = rec.get("store_io") or {}
    if not layers:
        return [f"V4: spill_layers {layers!r}: the spill did not happen, the arm is invalid"]
    if not io.get("bytes_written") or not io.get("bytes_read"):
        return [f"V4: spill_layers {layers} but the store moved {io.get('bytes_written')!r} B "
                f"out and {io.get('bytes_read')!r} B back: the spill did not happen"]
    return []


def fingerprint(rec: Dict[str, Any]) -> Optional[str]:
    rows = (rec.get("rows") or {}).get("row_sha256")
    return None if rows is None else json.dumps(rows)


def compare_records(ref: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    """Bit equality of two runs from their JSON: per-step gradient digests (SHA-256 of the
    raw bytes, dtype, shape), losses, the step count and the lr at every step."""

    def key(digest: Dict[str, Any]) -> Tuple[Any, ...]:
        return digest.get("sha256"), digest.get("dtype"), tuple(digest.get("shape") or ())

    left, right = ref.get("grads") or [], cand.get("grads") or []
    first = None
    for step, (a, b) in enumerate(zip(left, right)):
        names = sorted(set(a) | set(b))
        bad = [n for n in names if n not in a or n not in b or key(a[n]) != key(b[n])]
        if bad:
            first = {"step": step, "names": bad[:5], "count": len(bad)}
            break
    grads_equal = first is None and len(left) == len(right)
    losses_equal = ([r.get("loss") for r in ref.get("logs") or []]
                    == [r.get("loss") for r in cand.get("logs") or []])
    rates_equal = (ref.get("rates") == cand.get("rates")
                   and [r.get("learning_rate") for r in ref.get("logs") or []]
                   == [r.get("learning_rate") for r in cand.get("logs") or []])
    steps_equal = ref.get("steps") == cand.get("steps") and len(left) == len(right)
    return {"grads_equal": grads_equal, "losses_equal": losses_equal,
            "rates_equal": rates_equal, "steps_equal": steps_equal,
            "steps": [ref.get("steps"), cand.get("steps")], "first_difference": first,
            "equal": bool(grads_equal and losses_equal and rates_equal and steps_equal)}


def _flags(result: Dict[str, Any]) -> str:
    return ", ".join(k[:-6] for k in ("grads_equal", "losses_equal", "rates_equal",
                                      "steps_equal") if not result[k])


# ---------------------------------------------------------------------------
# criterion 1: spill == no spill
# ---------------------------------------------------------------------------
def criterion_1(arms: Dict[str, Optional[Dict[str, Any]]], entries: Dict[str, Dict[str, Any]],
                compare: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    names = (*NOSPILL, SPILL)
    missing = [name for name in names if arms.get(name) is None]
    if missing:
        return _crit(NONE, f"no record from {missing}")
    recs = {name: arms[name] for name in names}
    why: List[str] = []
    for name, rec in recs.items():
        why += _named(name, child_problems(rec, entries.get(name)))
        why += _named(name, determinism_problems(rec, name))
        why += _named(name, training_problems(rec))
        why += _named(name, spill_problems(rec, spilled=name == SPILL))
    if len({fingerprint(rec) for rec in recs.values()}) != 1 or fingerprint(recs[SPILL]) is None:
        why.append("V5: the arms trained on different rows")
    if why:
        return _crit(NONE, "; ".join(why))
    described = {key: (compare or {}).get(key) for key in ("nospill_repeat", "spill_vs_nospill")}
    repeat = compare_records(recs[NOSPILL[0]], recs[NOSPILL[1]])
    if not repeat["equal"]:
        return _crit(NONE, f"l2l_nospill_1 and l2l_nospill_2 differ ({_flags(repeat)}): the "
                     f"reference does not repeat, so spill against it proves nothing",
                     nospill_repeat=repeat, tensors=described)
    result = compare_records(recs[NOSPILL[0]], recs[SPILL])
    layers = recs[SPILL]["spill_plan"]["spill_layers"]
    if not result["equal"]:
        return _crit(FAIL, f"l2l_spill ({layers} layers spilled) differs from l2l_nospill_1 in "
                     f"{_flags(result)}; first difference {result['first_difference']}",
                     nospill_repeat=repeat, spill_vs_nospill=result, tensors=described)
    return _crit(PASS, f"l2l_spill ({layers} of {recs[SPILL]['spill_plan']['n_layers']} layers "
                 f"spilled) is bit-equal to l2l_nospill_1 over {result['steps'][0]} optimizer "
                 f"steps; the no-spill reference repeats", nospill_repeat=repeat,
                 spill_vs_nospill=result, tensors=described)


def criterion_1b(arms: Dict[str, Optional[Dict[str, Any]]], entries: Dict[str, Dict[str, Any]],
                 compare: Optional[Dict[str, Any]],
                 calibration: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    names = (*GA, FLOOR, SPILL)
    missing = [name for name in names if arms.get(name) is None]
    if missing:
        return _crit(NONE, f"no record from {missing}")
    why: List[str] = []
    for name in names:
        rec = arms[name]
        why += _named(name, child_problems(rec, entries.get(name)))
        why += _named(name, determinism_problems(rec, name))
        why += _named(name, training_problems(rec))
    if len({fingerprint(arms[name]) for name in names}) != 1:
        why.append("V5: GA and L2L trained on different rows")
    aliasing = (calibration or {}).get("aliasing")
    if not aliasing or not aliasing.get("pass"):
        why.append(f"bnb #2034 probe {(aliasing or {}).get('status', 'not run')}: an L2L/GA gap "
                   "could come from the aliasing")
    if compare is None or compare.get("tolerance") is None:
        why.append("no tolerance in compare.json")
    else:
        unmatched = [n for n in names if not (compare.get("hashes_match") or {}).get(n)]
        if unmatched:
            why.append(f"the tensors of {unmatched} do not hash to their arm records")
    if why:
        return _crit(NONE, "; ".join(why))
    tol = compare["tolerance"]
    status = PASS if tol.get("pass") else (
        FAIL if tol.get("status") in ("l2l_out_of_tolerance", "l2l_differs") else NONE)
    return _crit(status, f"A3 stand decide(): {tol.get('status')} {tol.get('note') or ''}".strip(),
                 tolerance=tol)


# ---------------------------------------------------------------------------
# criterion 2: RAM
# ---------------------------------------------------------------------------
def excess_bytes(rec: Dict[str, Any]) -> Optional[int]:
    """Host-allocator growth over training that the store's plan does not account for."""
    host = rec.get("host_allocator") or {}
    base, end = host.get("baseline_bytes"), host.get("at_last_step_bytes")
    held = (rec.get("spill_plan") or {}).get("host_bytes")
    if base is None or end is None or held is None:
        return None
    return int(end) - int(base) - int(held)


def _oom_kills(readback: Optional[Dict[str, Any]]) -> Optional[int]:
    value = ((readback or {}).get("events") or {}).get("oom_kill")
    return None if value is None else int(value)


BUILD_PHASE = "before first training_step (build)"
TRAIN_PHASE = "after first training_step"


def oom_phase(rec: Optional[Dict[str, Any]], entry: Optional[Dict[str, Any]]) -> str:
    """When a kill or an oom_kill came: before the child's first training_step entry (the
    build and setup) or after it. A killed child leaves no record, so its first-step marker
    (read by the parent after the exit) decides; a surviving child's own readback at that
    entry does."""
    events = [_oom_kills((rec or {}).get(key))
              for key in ("cgroup_start", "cgroup_at_first_step", "cgroup_end")]
    if rec is not None and None not in events:
        start, first, end = events
        if first > start:
            return BUILD_PHASE
        if end > first:
            return TRAIN_PHASE
    if entry is None or "marker" not in entry:
        return "unknown (no first-step marker was planned)"
    marker = entry.get("marker_after_exit")
    if not marker:
        return f"{BUILD_PHASE}: no first-step marker"
    return f"{TRAIN_PHASE} (marker at {marker.get('steps_done')} optimizer steps done)"


def _ram_part(name: str, rec: Optional[Dict[str, Any]],
              entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if entry is None:
        return _crit(NONE, f"{name}: not run")
    if entry.get("returncode") != 0:
        phase = oom_phase(rec, entry)
        return _crit(FAIL, f"{name}: the child exited {entry.get('returncode')} (no OOM needs "
                     f"exit 0), {phase}", oom_phase=phase)
    if rec is None:
        return _crit(NONE, f"{name}: no record")
    problems = child_problems(rec, entry)
    if problems:
        return _crit(NONE, "; ".join(_named(name, problems)))
    start, end = _oom_kills(rec.get("cgroup_start")), _oom_kills(rec.get("cgroup_end"))
    if start is None or end is None:
        return _crit(NONE, f"{name}: memory.events oom_kill not read")
    if end > start:
        phase = oom_phase(rec, entry)
        return _crit(FAIL, f"{name}: memory.events oom_kill rose {start} -> {end}, {phase}",
                     oom_phase=phase)
    peak, limit = (rec.get("cgroup_end") or {}).get("peak"), entry["memory_max"]
    if peak is None:
        return _crit(NONE, f"{name}: no memory.peak (cgroup v2, kernel >= 5.19)")
    if peak > limit:
        return _crit(FAIL, f"{name}: memory.peak {peak:,} B > MemoryMax {limit:,} B")
    return _crit(PASS, f"{name}: exit 0, oom_kill {end}, memory.peak {peak:,} B <= "
                 f"{limit:,} B", peak_bytes=peak, limit_bytes=limit)


def _tight_oom_part(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rule v2: a child at the tight limit, other than the three scored arms. A FAIL when the
    kernel log the parent read shows the memory cgroup OOM-killing it after its first-step
    marker; NO VERDICT when it was killed and the evidence cannot tell; None otherwise (no
    kill, or the SIGKILL the parent sent kill9, which the kernel does not log as an OOM)."""
    name, code = entry.get("name"), entry.get("returncode")
    if code in (0, None):
        return None
    oom = entry.get("kernel_oom")
    if not isinstance(oom, dict) or not oom.get("read"):
        if code == -9 and not entry.get("signal_name"):
            why = (oom or {}).get("error") if isinstance(oom, dict) else "no kernel log read"
            return _crit(NONE, f"{name}: killed (exit -9) and the kernel log cannot tell "
                         f"whether it was an OOM kill ({why})")
        return None
    if not oom.get("killed"):
        return None
    marker = entry.get("marker_after_exit")
    if not marker:
        return _crit(NONE, f"{name}: the kernel OOM-killed it {BUILD_PHASE}: no first-step "
                     f"marker, so the stand's limit, not the product", oom_phase=BUILD_PHASE)
    problems = child_problems(marker, entry)
    if problems:
        return _crit(NONE, "; ".join(_named(name, problems)))
    phase = f"{TRAIN_PHASE} (marker at {marker.get('steps_done')} optimizer steps done)"
    return _crit(FAIL, f"{name}: the kernel OOM-killed pid {marker.get('pid')} in its scope at "
                 f"MemoryMax {entry.get('memory_max'):,} B, {phase}: the product's reserve did "
                 f"not cover training", oom_phase=phase, kernel_lines=oom.get("lines"))


def criterion_2(arms: Dict[str, Optional[Dict[str, Any]]],
                entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    parts = [_ram_part(name, arms.get(name), entries.get(name)) for name in (SPILL, *NOSPILL)]
    for entry in entries.values():
        if entry.get("limit") == "tight" and entry.get("name") != SPILL:
            part = _tight_oom_part(entry)
            if part is not None:
                parts.append(part)
    spill, ref = arms.get(SPILL), arms.get(NOSPILL[0])
    excess = {name: excess_bytes(arms[name]) if arms.get(name) else None
              for name in (SPILL, *NOSPILL)}
    if spill is None or ref is None:
        parts.append(_crit(NONE, "excess: l2l_spill or l2l_nospill_1 has no record"))
    elif spill_problems(spill, spilled=True) or spill_problems(ref, spilled=False):
        parts.append(_crit(NONE, "excess: " + "; ".join(
            spill_problems(spill, spilled=True) + spill_problems(ref, spilled=False))))
    elif excess[SPILL] is None or excess[NOSPILL[0]] is None:
        parts.append(_crit(NONE, "excess: no CUDA host allocator counter or no host_bytes"))
    elif excess[SPILL] > excess[NOSPILL[0]]:
        parts.append(_crit(FAIL, f"excess: l2l_spill allocated {excess[SPILL]:,} B outside its "
                           f"store, over l2l_nospill_1's {excess[NOSPILL[0]]:,} B"))
    else:
        parts.append(_crit(PASS, f"excess: l2l_spill {excess[SPILL]:,} B <= l2l_nospill_1 "
                           f"{excess[NOSPILL[0]]:,} B (l2l_nospill_2 "
                           f"{excess[NOSPILL[1]]})"))
    return _crit(_combine(parts), "; ".join(p["detail"] for p in parts), parts=parts,
                 excess_bytes=excess)


# ---------------------------------------------------------------------------
# criterion 3: cleanup
# ---------------------------------------------------------------------------
def _normal_cleanup(rec: Optional[Dict[str, Any]],
                    entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if rec is None or entry is None:
        return _crit(NONE, "normal: l2l_spill did not run or left no record")
    problems = child_problems(rec, entry) + spill_problems(rec, spilled=True)
    if rec.get("exit") != "normal":
        problems.append(f"exit {rec.get('exit')!r}")
    if problems:
        return _crit(NONE, "normal: " + "; ".join(problems))
    after_train = rec.get("spill_files_after_train")
    after_exit = entry.get("spill_files_after_exit")
    if after_train is None or after_exit is None:
        return _crit(NONE, "normal: the spill directory was not listed after the run")
    if after_train or after_exit:
        return _crit(FAIL, f"normal: spill files left after train() {after_train}, after the "
                     f"process exit {after_exit}")
    return _crit(PASS, "normal: no spill file after train() nor after the process exit")


def _sigint(rec: Optional[Dict[str, Any]], entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sent = (entry or {}).get("signal")
    if not sent:
        return _crit(NONE, "sigint: not run")
    if rec is not None:
        problems = child_problems(rec, entry)
        if problems:
            return _crit(NONE, "sigint: " + "; ".join(problems))
    if sent.get("sent_unix") is None:
        return _crit(NONE, f"sigint: no signal sent ({sent.get('why')})")
    if not sent.get("exited"):
        return _crit(FAIL, "sigint: the child did not exit after one SIGINT")
    before, after = sent.get("files_before_signal") or [], sent.get("files_after_exit")
    if after is None:
        return _crit(NONE, "sigint: the spill directory was not listed after the exit")
    if before and after:
        return _crit(FAIL, f"sigint: spill files left after Ctrl-C: {after}")
    at = (rec or {}).get("at_interrupt") or {}
    if not before or (rec or {}).get("exit") != "interrupted" or not at.get("in_step"):
        return _crit(NONE, f"sigint: the scenario was not reached (spill file before the signal "
                     f"{before}, child exit {(rec or {}).get('exit')!r}, inside a step "
                     f"{at.get('in_step')})")
    return _crit(PASS, f"sigint: one SIGINT inside optimizer step {at.get('steps_done', 0) + 1}, "
                 f"exit {sent.get('returncode')}, {before} gone")


def _kill9(entry: Optional[Dict[str, Any]], next_rec: Optional[Dict[str, Any]],
           next_entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sent = (entry or {}).get("signal")
    if not sent:
        return _crit(NONE, "kill9: not run")
    marker = sent.get("marker") or {}
    path = marker.get("spill_path")
    before = sent.get("files_before_signal") or []
    if sent.get("sent_unix") is None or not path or path not in before:
        return _crit(NONE, f"kill9: the scenario was not reached (spill file {path!r}, files "
                     f"before SIGKILL {before}, {sent.get('why')})")
    # A killed child writes no record; its marker carries the device and the cgroup readback.
    problems = child_problems(marker, entry)
    if problems:
        return _crit(NONE, "kill9: " + "; ".join(problems))
    if path not in (sent.get("files_after_exit") or []):
        return _crit(FAIL, f"kill9: {path} is gone after SIGKILL; the documented behaviour is "
                     "that it stays")
    if next_entry is None:
        return _crit(NONE, "kill9: the next child did not run")
    if next_entry.get("returncode") != 0:
        return _crit(FAIL, f"kill9: the next child exited {next_entry.get('returncode')}")
    if next_rec is None:
        return _crit(NONE, "kill9: the next child left no record")
    problems = child_problems(next_rec, next_entry)
    if problems:
        return _crit(NONE, "kill9: next child " + "; ".join(problems))
    removed = next_rec.get("stale_removed") or []
    left = [p for p in (next_rec.get("spill_files_after_train") or [],
                        next_entry.get("spill_files_after_exit") or []) if path in p]
    if path not in removed or left:
        return _crit(FAIL, f"kill9: the next start did not remove {path} (stale_removed "
                     f"{removed}; still listed {bool(left)})")
    trained = training_problems(next_rec)
    if trained:
        return _crit(FAIL, "kill9: the next child did not train normally: " + "; ".join(trained))
    return _crit(PASS, f"kill9: {path} left by SIGKILL, removed at the next start, which trained "
                 f"{next_rec.get('steps')} optimizer steps")


def criterion_3(arms: Dict[str, Optional[Dict[str, Any]]],
                cleanup: Dict[str, Optional[Dict[str, Any]]],
                entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    parts = [
        _normal_cleanup(arms.get(SPILL), entries.get(SPILL)),
        _sigint(cleanup.get("sigint"), entries.get("sigint")),
        _kill9(entries.get("kill9"), cleanup.get("kill9_next"), entries.get("kill9_next")),
    ]
    return _crit(_combine(parts), "; ".join(p["detail"] for p in parts), parts=parts)


# ---------------------------------------------------------------------------
# criterion 4: cost
# ---------------------------------------------------------------------------
def load_cost(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((root / "cost").glob("b*_p*_*.json")):
        rec = _load(path) or {}
        rows.append({**rec, "_name": f"cost/{path.stem}"})
    return rows


def _cost_problems(rec: Dict[str, Any], entry: Optional[Dict[str, Any]]) -> List[str]:
    why = child_problems(rec, entry)
    if rec.get("exit") != "normal" or rec.get("value") is None:
        why.append(f"exit {rec.get('exit')!r}, value {rec.get('value')!r}")
    why += spill_problems(rec, spilled=rec.get("role") == "B")
    if not all(isinstance(row.get("loss"), (int, float)) and math.isfinite(row["loss"])
               for row in rec.get("logs") or [{}]):
        why.append("a non-finite loss")
    return why


def criterion_4(rows: Sequence[Dict[str, Any]], entries: Dict[str, Dict[str, Any]],
                c1: Dict[str, Any]) -> Dict[str, Any]:
    if c1["status"] == FAIL:
        return _crit(NONE, "refused: criterion 1 failed (spill changes the numbers), so no "
                     "cost block runs. No new rentals: find the cause", refused=True)
    if c1["status"] != PASS:
        return _crit(NONE, "refused: the cost blocks run only after criterion 1 passes, and it "
                     "has no verdict", refused=True)
    if not rows:
        return _crit(NONE, "no cost block ran")
    why: List[str] = []
    for rec in rows:
        why += _named(rec["_name"], _cost_problems(rec, entries.get(rec["_name"])))
    blocks: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for rec in rows:
        if rec.get("value") is not None:
            blocks.setdefault(int(rec["block"]), {})[int(rec["position"])] = rec
    ordered = [[arms[p] for p in range(4)] for _, arms in sorted(blocks.items())
               if set(arms) == {0, 1, 2, 3}
               and [arms[p].get("role") for p in range(4)] == list(arm_validity.BLOCK_ORDER)]
    runs = [[arm_validity.Run(float(arm["value"]), str(arm.get("outcome"))) for arm in block]
            for block in ordered]
    valid = [block for block, run in zip(ordered, runs)
             if all(r.outcome != arm_validity.VOID for r in run)]
    unknown = [arm["_name"] for block in valid for arm in block
               if arm.get("outcome") != arm_validity.OK]
    if unknown:
        why.append(f"V6: F1 outcome not ok on {unknown}")
    for block in valid:
        if len({fingerprint(arm) for arm in block}) != 1:
            why.append(f"V5: block {block[0]['block']} trains on different rows")
    due = arm_validity.blocks_to_add(runs) if runs else arm_validity.PILOT_BLOCKS
    if due:
        why.append(f"V7: the repeats rule wants {due} more block(s) after {len(ordered)} "
                   f"(cap {MAX_BLOCKS})")
    effects = []
    for block in valid:
        a = [arm["value"] for arm in block if arm["role"] == "A"]
        b = [arm["value"] for arm in block if arm["role"] == "B"]
        effects.append((statistics.fmean(b) - statistics.fmean(a)) / statistics.fmean(a))
    if len(effects) < 2:
        why.append("fewer than two valid blocks")
    extra = {"blocks": len(ordered), "valid_blocks": len(valid),
             "void_blocks": len(ordered) - len(valid), "effects": effects,
             "interval": interval(effects) if len(effects) >= 2 else None,
             "spill_layers": sorted({(arm.get("spill_plan") or {}).get("spill_layers")
                                     for block in valid for arm in block
                                     if arm.get("role") == "B"})}
    if why:
        return _crit(NONE, "; ".join(why), **extra)
    ci = extra["interval"]
    text = f"overhead {ci['mean']:+.2%} [{ci['low']:+.2%}, {ci['high']:+.2%}] over {ci['n']} blocks"
    if ci["high"] < COST_LIMIT:
        return _crit(PASS, f"{text} < {COST_LIMIT:.0%}: below {COST_LIMIT:.0%} confirmed on a "
                     f"faster machine", **extra)
    if ci["low"] >= COST_LIMIT:
        return _crit(FAIL, f"{text} >= {COST_LIMIT:.0%}: the fallback stays automatic; the "
                     f"console message and the docs state the measured overhead", **extra)
    return _crit(NONE, f"{text} straddles {COST_LIMIT:.0%}: no verdict", **extra)


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
def verdict_of(criteria: Dict[str, Dict[str, Any]], cpu: bool) -> str:
    if cpu:
        return "NO VERDICT: CPU session (V1)"
    if criteria["1"]["status"] == FAIL:
        return "SPILL CHANGES THE NUMBERS: no new rentals, find the cause"
    safety = {c: criteria[c]["status"] for c in ("1", "2", "3")}
    note = ""
    if criteria["1b"]["status"] == FAIL:
        note = "; L2L is outside the A3 stand's GA tolerance (criterion 1b, not a spill finding)"
    if FAIL in safety.values():
        failed = [c for c, s in safety.items() if s == FAIL]
        return f"SPILL NOT SAFE IN THE TRAINER: criterion {', '.join(failed)} failed" + note
    if NONE in safety.values():
        open_ = [c for c, s in safety.items() if s == NONE]
        return f"NO VERDICT: criterion {', '.join(open_)} has no verdict" + note
    cost = criteria["4"]["status"]
    if cost == PASS:
        return ("SPILL CONFIRMED: equal, within its RAM, cleaned up; overhead below 10% "
                "confirmed on a faster machine" + note)
    if cost == FAIL:
        return ("SPILL CONFIRMED, COSTLY: overhead >= 10%; the fallback stays automatic and the "
                "console message and the docs state the measured overhead" + note)
    return "SPILL CONFIRMED, COST NO VERDICT: equal, within its RAM, cleaned up" + note


def _inputs(root: Path) -> Dict[str, Any]:
    session = _load(root / "session.json") or {}
    return {
        "session": session,
        "entries": {child["name"]: child for child in session.get("children", [])},
        "arms": {name: _load(root / "arms" / f"{name}.json") for name in CORRECTNESS_ARMS},
        "cleanup": {name: _load(root / "cleanup" / f"{name}.json") for name in CLEANUP_ARMS},
        "calibration": _load(root / "calibration.json"),
        "checks": [_load(path) for path in sorted(root.glob("calibration_check_*.json"))],
        "compare": _load(root / "compare.json"),
        "cost": load_cost(root),
    }


def cost_gate(results_dir: str) -> Tuple[bool, str]:
    """Whether the parent may run the cost blocks: only after criterion 1 passes."""
    data = _inputs(Path(results_dir))
    c1 = criterion_1(data["arms"], data["entries"], data["compare"])
    if c1["status"] == PASS:
        return True, c1["detail"]
    if c1["status"] == FAIL:
        return False, ("STOP: criterion 1 failed (spill != no spill), so no cost block runs. "
                       "No new rentals: find the cause. " + c1["detail"])
    return False, "criterion 1 has no verdict, so no cost block runs: " + c1["detail"]


def evaluate(results_dir: str) -> Dict[str, Any]:
    data = _inputs(Path(results_dir))
    arms, entries = data["arms"], data["entries"]
    c1 = criterion_1(arms, entries, data["compare"])
    criteria = {
        "1": c1,
        "1b": criterion_1b(arms, entries, data["compare"], data["calibration"]),
        "2": criterion_2(arms, entries),
        "3": criterion_3(arms, data["cleanup"], entries),
        "4": criterion_4(data["cost"], entries, c1),
    }
    records = [rec for rec in (*arms.values(), *data["cleanup"].values(), *data["cost"],
                               data["calibration"], *data["checks"]) if rec]
    cpu = any(rec.get("device") != "cuda" for rec in records)
    layers = (arms.get(SPILL) or {}).get("spill_plan", {}) or {}
    verdict = verdict_of(criteria, cpu)
    return {
        "rule_version": RULE_VERSION, "rule": RULE, "verdict": verdict, "criteria": criteria,
        "session_state": data["session"].get("state"), "limits": data["session"].get("limits"),
        "spill_layers": layers.get("spill_layers"),
        "spill_layers_in_target": (None if layers.get("spill_layers") is None
                                   else SPILL_BAND[0] <= layers["spill_layers"] <= SPILL_BAND[1]),
        "oom_margin": oom_margin_summary(data["session"]),
        "summary": {"verdict": verdict, "rule_version": RULE_VERSION, "cpu_session": cpu,
                    **{f"criterion_{c}": v["status"] for c, v in criteria.items()}},
    }


def oom_margin_summary(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Description only: the build and training margins of the tight limit the session ran
    at (the last one it tried), with their warnings. None when no limit was tried."""
    tried = ((session.get("limits") or {}).get("oom_margin") or [])
    if not tried:
        return None
    last = tried[-1]
    return {key: last.get(key) for key in ("limit_bytes", "build_margin_bytes",
                                           "train_margin_bytes", "warn_below_bytes",
                                           "warnings", "stop")}


def write_verdict(path: Path, verdict: Dict[str, Any]) -> None:
    """LF on every OS, so a re-analysis is byte-identical to the session's own file."""
    Path(path).write_text(json.dumps(verdict, indent=2, default=str) + "\n", encoding="utf-8",
                          newline="\n")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    result = evaluate(args.results_dir)
    if args.out:
        write_verdict(Path(args.out), result)
    print(json.dumps(result["summary"], indent=2))
    for name, crit in result["criteria"].items():
        print(f"{name:<3} {crit['status']:<10} {crit['detail'][:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
