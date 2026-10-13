"""E1 confirmation verdicts on the set C (probe-moe-adapter-placement.md section 11).

Pre-registered before any confirmation result exists. E1 chose its learning rates on the
evaluation set E it then scored; C is 256 group-disjoint MetaMathQA questions E1 never
used, scored on NEW reruns of every counted E1 configuration at E1's frozen rates
(``FROZEN_RATES``). The main answer is the verdicts and gaps on C compared with E on the
SAME new adapters; the published E1 records are read only for a reproduction check and
are labelled historical.

The rule reuses ``moe_adapter_rule`` (``summarise``, ``material_gap``, ``leg2_failures``,
``paired_clustered``, ``lr_sweep``, ``provenance_problem``) and never changes its
thresholds or ``GATE``. V3 is not re-decided: the rates are frozen by E1.

``(E - C)`` of an arm minus ``(E - C)`` of the base is the "base-corrected E/C
generalisation contrast". Its interval reflects question-sampling noise only and says
nothing about a cause; an LR-sweep winner on C that differs from E1's is a diagnostic.

Pure Python, deterministic, no torch.

2026-10-13: the RAM-tier parity gate of §11.3 implemented after the parity pilot's result
and before any RAM-tier score result was read; no threshold changed.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import pathlib
import random
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import moe_adapter_rule as rule  # noqa: E402

FROZEN_RATES = {"olmoe": {"A": 2e-4, "AT": 1e-4, "AE": 1e-4},
                "qwen35": {"A": 2e-4, "AS": 2e-4, "AT": 1e-4, "AE": 1e-4}}
PER_ROW_RELATIVE_TOLERANCE = 1e-5
MEMORY_TOLERANCE_BYTES = round(13.2 * 2**20)  # E1's largest paired residual
TASK_SETS = ("E", "C")
ARM_ORDER = ("A", "AS", "AT", "AE")
RAM_MODEL = "qwen35"
RAM_ARMS = ("A", "AS")
RAM_FULLEST = "AE"
CONTRAST_NAME = "base-corrected E/C generalisation contrast"
CONTRAST_NOTE = ("(E - C) of the arm minus (E - C) of the base; the 95% interval reflects "
                 "question-sampling noise only (E and C rows resampled independently), "
                 "not a cause")
V3_NOTE = "V3 not re-decided: learning rates are frozen at E1's section 8 winners"
SWEEP_NOTE = "a C winner that differs from E1's rate is a diagnostic, not proof of a cause"
VS_RESIDENT_NOTE = ("not attributable to streaming or NF4: it mixes the NF4 base, warmup, "
                    "data order and initialisation")
HISTORICAL_NOTE = "historical: published E1 records scored on E, from rule.analyse"


def _ok(record: Dict[str, Any]) -> bool:
    return record.get("status") == "ok"


def _trained(record: Dict[str, Any]) -> bool:
    return record["arm"] != "base"


def task_view(record: Dict[str, Any], task_set: str) -> Dict[str, Any]:
    """The record as ``rule`` reads it on ``task_set``: E unchanged, C with C's eval/task.

    An ``ok`` record without a ``confirm`` block is refused. A non-ok record without one
    keeps no E evaluation in its C view.
    """
    if task_set not in TASK_SETS:
        raise ValueError(f"unknown task set {task_set!r}")
    view = dict(record)
    if task_set == "E":
        return view
    confirm = record.get("confirm")
    if confirm is None:
        if _ok(record):
            raise ValueError(f"{record.get('_path')}: status ok without a confirm block")
        view["eval"] = {}
        view["task"] = {}
        return view
    view["eval"] = confirm["eval"]
    view["task"] = confirm["task"]
    return view


def check_per_row(record: Dict[str, Any]) -> None:
    """Per-row E and C data must add up to the set's ``eval`` (data integrity)."""
    confirm = record.get("confirm")
    if confirm is None:
        raise ValueError(f"{record.get('_path')}: no confirm block")
    for task_set, evaluated in (("E", record["eval"]), ("C", confirm["eval"])):
        rows = confirm["per_row"][task_set]
        tokens = sum(rows["tokens"])
        if len(rows["tokens"]) != len(rows["nll_sum"]):
            raise ValueError(f"{record.get('_path')}: {task_set} per-row lengths differ")
        if tokens != evaluated["tokens"]:
            raise ValueError(f"{record.get('_path')}: {task_set} per-row tokens {tokens} != "
                             f"eval tokens {evaluated['tokens']}")
        nll = sum(rows["nll_sum"]) / tokens
        if abs(nll - evaluated["nll"]) > PER_ROW_RELATIVE_TOLERANCE * evaluated["nll"]:
            raise ValueError(f"{record.get('_path')}: {task_set} per-row NLL {nll!r} != "
                             f"eval NLL {evaluated['nll']!r}")


def frozen_partition(records: Sequence[Dict[str, Any]], model: str) -> Dict[str, List]:
    """``counted``: the base + trained records at the arm's frozen rate; the rest uncounted."""
    rates = FROZEN_RATES.get(model, {})
    counted: List[Dict[str, Any]] = []
    uncounted: List[Dict[str, Any]] = []
    for r in records:
        if r["model"] != model:
            continue
        if not _trained(r) or (r["arm"] in rates and r.get("lr") == rates[r["arm"]]):
            counted.append(r)
        else:
            uncounted.append(r)
    return {"counted": counted, "uncounted": uncounted}


def verdicts(records: Sequence[Dict[str, Any]], model: str, task_set: str) -> Dict[str, Any]:
    """``rule.summarise`` on ``task_set`` over the frozen-rate partition of ``model``."""
    part = frozen_partition(records, model)
    plan = {model: rule.PLAN[model]} if model in rule.PLAN else {}
    result = rule.summarise([task_view(r, task_set) for r in part["counted"]], edge_lrs={},
                            plan=plan,
                            uncounted=[task_view(r, task_set) for r in part["uncounted"]])
    out = result[model]
    out["v3_note"] = V3_NOTE
    return out


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _seed_gaps(row: Dict[str, Any], full: Dict[str, Any]) -> Dict[int, float]:
    full_nll = dict(zip(full["seeds"], full["nll"]))
    return {s: v - full_nll[s] for s, v in zip(row["seeds"], row["nll"]) if s in full_nll}


def _ordering(result: Dict[str, Any]) -> List[str]:
    arms = result["arms"]
    return sorted(arms, key=lambda a: (arms[a]["nll_mean"], a))


def compare(e: Dict[str, Any], c: Dict[str, Any]) -> Dict[str, Any]:
    """Per arm in both: does the E verdict and E task gap hold on C, on the same adapters."""
    if e["verdict"] != "MEASURED" or c["verdict"] != "MEASURED":
        return {"verdict": "NO VERDICT",
                "E": {"verdict": e["verdict"], "reason": e.get("reason"),
                      "missing": e.get("missing", []), "validity": e.get("validity")},
                "C": {"verdict": c["verdict"], "reason": c.get("reason"),
                      "missing": c.get("missing", []), "validity": c.get("validity")}}
    arms: Dict[str, Any] = {}
    full_e, full_c = e["arms"][e["fullest"]], c["arms"][c["fullest"]]
    for arm in sorted(set(e["arms"]) & set(c["arms"])):
        re_, rc = e["arms"][arm], c["arms"][arm]
        ge, gc = re_["task_gap"], rc["task_gap"]
        seeds_e, seeds_c = _seed_gaps(re_, full_e), _seed_gaps(rc, full_c)
        arms[arm] = {
            "verdict_E": re_["verdict"], "verdict_C": rc["verdict"],
            "verdict_holds": re_["verdict"] == rc["verdict"],
            "task_gap_E": ge, "task_gap_C": gc,
            "gap_holds": _sign(ge["gap"]) == _sign(gc["gap"]) and ge["material"] == gc["material"],
            "seed_gaps_E": {str(s): v for s, v in sorted(seeds_e.items())},
            "seed_gaps_C": {str(s): v for s, v in sorted(seeds_c.items())},
            "seed_sign_agrees": {str(s): _sign(seeds_e[s]) == _sign(seeds_c[s])
                                 for s in sorted(set(seeds_e) & set(seeds_c))},
            "paired_clustered_E": re_["paired_clustered"],
            "paired_clustered_C": rc["paired_clustered"],
            "accuracy_E": re_["accuracy"], "accuracy_C": rc["accuracy"],
        }
    return {"verdict": "MEASURED", "fullest_E": e["fullest"], "fullest_C": c["fullest"],
            "arms": arms, "ordering_E": _ordering(e), "ordering_C": _ordering(c)}


def _picker(idx: Optional[List[int]]):
    """Values at ``idx`` (all values when ``idx`` is None), always as a sequence."""
    if idx is None:
        return lambda values: values
    if len(idx) == 1:
        return lambda values: (values[idx[0]],)
    return operator.itemgetter(*idx)


def _percentile_interval(values: List[float]) -> List[float]:
    values = sorted(values)
    n = len(values)
    return [values[int(0.025 * n)], values[int(0.975 * n) - 1]]


def _averaged_rows(runs: Sequence[Dict[str, Any]], task_set: str) -> Dict[str, List]:
    """Seeds averaged per question: nll_sum and correctness; tokens identical across seeds."""
    blocks = [r["confirm"]["per_row"][task_set] for r in runs]
    correct = [r["task"]["correct"] if task_set == "E" else r["confirm"]["task"]["correct"]
               for r in runs]
    tokens = blocks[0]["tokens"]
    if any(b["tokens"] != tokens for b in blocks):
        raise ValueError(f"{task_set} per-row tokens differ across seeds")
    if any(len(c) != len(tokens) for c in correct):
        raise ValueError(f"{task_set} correctness and per-row lengths differ")
    k = len(runs)
    return {"tokens": list(tokens),
            "nll_sum": [sum(col) / k for col in zip(*(b["nll_sum"] for b in blocks))],
            "correct": [sum(int(v) for v in col) / k for col in zip(*correct)]}


def contrast(runs: Sequence[Dict[str, Any]], *, draws: int = rule.BOOTSTRAP_DRAWS,
             seed: int = 0) -> Dict[str, Any]:
    """Base-corrected E/C generalisation contrast and gap differences, with 95% intervals.

    ``runs``: ok runs of one model (the base and trained arms, frozen rates). Per arm, seeds
    are averaged per question. Each bootstrap draw resamples E and C rows independently,
    with the same indices for every arm and the base.
    """
    base = [r for r in runs if r["arm"] == "base"]
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        if r["arm"] != "base":
            by_arm.setdefault(r["arm"], []).append(r)
    if len(base) != 1 or not by_arm:
        return {"missing": f"needs one ok base and an ok trained arm "
                           f"(base records: {len(base)}, arms: {sorted(by_arm)})"}
    fullest = next(a for a in rule.FULLEST_ORDER if a in by_arm)
    units = {"base": base, **{a: sorted(rs, key=lambda r: r["seed"]) for a, rs in by_arm.items()}}
    rows = {s: {u: _averaged_rows(rs, s) for u, rs in units.items()} for s in TASK_SETS}
    for s in TASK_SETS:
        if any(rows[s][u]["tokens"] != rows[s]["base"]["tokens"] for u in units):
            raise ValueError(f"{s} per-row tokens differ between the base and an arm")
    n = {s: len(rows[s]["base"]["tokens"]) for s in TASK_SETS}

    def stats(idx: Dict[str, Optional[List[int]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        out: Dict[str, Dict[str, Dict[str, float]]] = {u: {} for u in units}
        for s in TASK_SETS:
            pick = _picker(idx[s])
            tokens = sum(pick(rows[s]["base"]["tokens"]))
            for u in units:
                out[u][f"nll_{s}"] = sum(pick(rows[s][u]["nll_sum"])) / tokens
                out[u][f"accuracy_{s}"] = sum(pick(rows[s][u]["correct"])) / n[s]
        return out

    def derived(st: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for metric in ("nll", "accuracy"):
            base_diff = st["base"][f"{metric}_E"] - st["base"][f"{metric}_C"]
            for arm in by_arm:
                d[f"{arm}:contrast_{metric}"] = (
                    st[arm][f"{metric}_E"] - st[arm][f"{metric}_C"] - base_diff)
                if arm != fullest:
                    d[f"{arm}:gap_difference_{metric}"] = (
                        (st[arm][f"{metric}_E"] - st[fullest][f"{metric}_E"])
                        - (st[arm][f"{metric}_C"] - st[fullest][f"{metric}_C"]))
        return d

    point_stats = stats({"E": None, "C": None})
    point = derived(point_stats)
    rng = random.Random(seed)
    samples: Dict[str, List[float]] = {k: [] for k in point}
    for _ in range(draws):
        idx = {s: [rng.randrange(n[s]) for _ in range(n[s])] for s in TASK_SETS}
        for k, v in derived(stats(idx)).items():
            samples[k].append(v)
    intervals = {k: _percentile_interval(v) for k, v in samples.items()}

    def unit_row(u: str) -> Dict[str, Any]:
        st = point_stats[u]
        return {"nll_E": st["nll_E"], "nll_C": st["nll_C"],
                "e_minus_c_nll": st["nll_E"] - st["nll_C"],
                "accuracy_E": st["accuracy_E"], "accuracy_C": st["accuracy_C"],
                "e_minus_c_accuracy": st["accuracy_E"] - st["accuracy_C"]}

    arms_out: Dict[str, Any] = {}
    for arm in sorted(by_arm):
        row = unit_row(arm)
        row["seeds"] = [r["seed"] for r in units[arm]]
        for key in ("contrast_nll", "contrast_accuracy", "gap_difference_nll",
                    "gap_difference_accuracy"):
            if f"{arm}:{key}" in point:
                row[key] = point[f"{arm}:{key}"]
                row[f"{key}_ci95"] = intervals[f"{arm}:{key}"]
        arms_out[arm] = row
    return {"name": CONTRAST_NAME, "note": CONTRAST_NOTE, "fullest": fullest,
            "draws": draws, "seed": seed, "rows": n, "base": unit_row("base"),
            "arms": arms_out}


def _strip_seconds(log: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in entry.items() if k != "seconds"} for entry in log]


REPRODUCED_FIELDS = {
    "train.log": lambda r: _strip_seconds(r.get("train", {}).get("log", [])),
    "eval.nll": lambda r: r.get("eval", {}).get("nll"),
    "eval.tokens": lambda r: r.get("eval", {}).get("tokens"),
    "task.correct": lambda r: r.get("task", {}).get("correct"),
    "task.outputs": lambda r: r.get("task", {}).get("outputs"),
    "gate.scores": lambda r: r.get("gate", {}).get("scores"),
    "gate.outputs": lambda r: r.get("gate", {}).get("outputs"),
    "v4": lambda r: r.get("v4"),
    "trainable_parameters": lambda r: r.get("trainable_parameters"),
}


def _key(record: Dict[str, Any]) -> tuple:
    if record["arm"] == "base":
        return (record["model"], "base")
    return (record["model"], record["arm"], record.get("lr"), record.get("seed"))


def reproduce(new: Dict[str, Any], published: Dict[str, Any]) -> Dict[str, Any]:
    """Exact comparison of a new run with its published counterpart (``seconds`` ignored)."""
    differ = [name for name, get in REPRODUCED_FIELDS.items() if get(new) != get(published)]
    return {"reproduced": not differ, "differ": differ,
            "peak_cuda_bytes_equal": new.get("peak_cuda_bytes") == published.get(
                "peak_cuda_bytes"),
            "source": new.get("_path"), "published": published.get("_path")}


def reproduction(counted: Sequence[Dict[str, Any]],
                 published: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every new counted record against the published record of the same configuration."""
    index: Dict[tuple, Dict[str, Any]] = {}
    for r in published:
        k = _key(r)
        if k in index:
            raise ValueError(f"two published records for {k}: {index[k].get('_path')}, "
                             f"{r.get('_path')}")
        index[k] = r
    out = []
    for r in counted:
        k = _key(r)
        entry: Dict[str, Any] = {"model": r["model"], "arm": r["arm"], "lr": r.get("lr"),
                                 "seed": r.get("seed")}
        if k in index:
            entry.update(reproduce(r, index[k]))
        else:
            entry.update(reproduced=None, published=None, source=r.get("_path"),
                         note="no published counterpart")
        out.append(entry)
    return out


def sweeps(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """``rule.lr_sweep`` on the new records' E and C views, next to E1's frozen rates."""
    e, c = (rule.lr_sweep([task_view(r, s) for r in records]) for s in TASK_SETS)
    arms: Dict[str, Any] = {}
    for arm in sorted(set(e) | set(c)):
        wc = c.get(arm, {}).get("winner")
        arms[arm] = {"frozen": FROZEN_RATES[rule.SWEEP_MODEL].get(arm), "E": e.get(arm),
                     "C": c.get(arm),
                     "c_winner_differs": None if wc is None else
                     wc != FROZEN_RATES[rule.SWEEP_MODEL].get(arm)}
    return {"model": rule.SWEEP_MODEL, "note": SWEEP_NOTE, "arms": arms}


BOX_FIELDS = {
    "task.outputs": lambda r: r.get("task", {}).get("outputs"),
    "gate.outputs": lambda r: r.get("gate", {}).get("outputs"),
    "confirm.task.outputs": lambda r: r.get("confirm", {}).get("task", {}).get("outputs"),
    "eval.nll": lambda r: r.get("eval", {}).get("nll"),
    "confirm.eval.nll": lambda r: r.get("confirm", {}).get("eval", {}).get("nll"),
}


def box_checks(checks: Sequence[Dict[str, Any]],
               bases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each second-box base against its model's counted base (never in a verdict)."""
    out = []
    for check in checks:
        base = bases.get(check["model"])
        entry: Dict[str, Any] = {"model": check["model"], "source": check.get("_path")}
        if base is None:
            entry.update(identical=None, differ=[], reason="no counted ok base")
        else:
            differ = [f for f, get in BOX_FIELDS.items() if get(check) != get(base)]
            entry.update(identical=not differ, differ=differ, base=base.get("_path"))
        out.append(entry)
    return out


def _by_seed(runs: Sequence[Dict[str, Any]], what: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in runs:
        if r["seed"] in out:
            raise ValueError(f"two ok {what} records for seed {r['seed']}")
        out[r["seed"]] = r
    return out


def ram_verdicts(ram: Sequence[Dict[str, Any]], resident: Sequence[Dict[str, Any]],
                 task_set: str) -> Dict[str, Any]:
    """RAM-tier A and AS against the resident fullest arm AE at its frozen rate, on one set.

    ``resident``: the counted (frozen-rate) resident records of RAM_MODEL, the base included.
    """
    ram_v = [task_view(r, task_set) for r in ram]
    res_v = [task_view(r, task_set) for r in resident if r["model"] == RAM_MODEL and _ok(r)]
    seeds = rule.PLAN[RAM_MODEL]["seeds"]
    rate = FROZEN_RATES[RAM_MODEL][RAM_FULLEST]
    full = _by_seed(sorted((r for r in res_v if r["arm"] == RAM_FULLEST and r["lr"] == rate
                            and r["seed"] in seeds), key=lambda r: r["seed"]), "resident AE")
    base = [r for r in res_v if r["arm"] == "base"]
    out: Dict[str, Any] = {"fullest": RAM_FULLEST, "fullest_seeds": sorted(full)}
    missing = [f"resident {RAM_FULLEST} seed {s}" for s in seeds if s not in full]
    if len(base) != 1:
        missing.insert(0, "resident base")
    ok_ram = [r for r in ram_v if _ok(r)]
    out["void"] = [{"arm": r["arm"], "seed": r.get("seed"), "status": r.get("status")}
                   for r in ram_v if not _ok(r)]
    by_arm = {a: sorted((r for r in ok_ram if r["arm"] == a), key=lambda r: r["seed"])
              for a in RAM_ARMS}
    missing += [f"RAM-tier {a}" for a in RAM_ARMS if not by_arm[a]]
    for a in RAM_ARMS:
        _by_seed(by_arm[a], f"RAM-tier {a}")
    vs: Dict[str, Any] = {}
    for a in RAM_ARMS:
        res = _by_seed([r for r in res_v if r["arm"] == a
                        and r["lr"] == FROZEN_RATES[RAM_MODEL][a]], f"resident {a}")
        if not by_arm[a] or not res:
            continue
        vs[a] = {"nll_ram": [rule._eval_nll(r) for r in by_arm[a]],
                 "nll_resident": [rule._eval_nll(r) for _, r in sorted(res.items())],
                 "accuracy_ram": [r["task"]["accuracy"] for r in by_arm[a]],
                 "accuracy_resident": [r["task"]["accuracy"] for _, r in sorted(res.items())],
                 "seeds_ram": [r["seed"] for r in by_arm[a]], "seeds_resident": sorted(res)}
        vs[a]["nll_difference"] = (statistics.fmean(vs[a]["nll_ram"])
                                   - statistics.fmean(vs[a]["nll_resident"]))
        vs[a]["accuracy_difference"] = (statistics.fmean(vs[a]["accuracy_ram"])
                                        - statistics.fmean(vs[a]["accuracy_resident"]))
    out["vs_resident"] = {"note": VS_RESIDENT_NOTE, "arms": vs}
    if missing:
        out.update(verdict="NO VERDICT", missing=missing)
        return out
    base_nll = rule._eval_nll(base[0])
    base_gate = base[0]["gate"]["scores"]
    validity = {
        "V1": all(base_nll - rule._eval_nll(r) >= rule.MIN_LEARNED_NAT - rule.TOLERANCE
                  for r in full.values()),
        "V2": not any(rule._nonfinite(r) for r in ram_v),
        "V4": all(r.get("v4", {}).get("passed", False) for r in ram_v),
    }
    out["validity"] = validity
    if not all(validity.values()):
        out["verdict"] = "NO VERDICT"
        return out
    full_nll = [rule._eval_nll(r) for _, r in sorted(full.items())]
    arms: Dict[str, Any] = {}
    for a in RAM_ARMS:
        runs = by_arm[a]
        nll = [rule._eval_nll(r) for r in runs]
        gap = rule.material_gap(nll, full_nll)
        failed = rule.leg2_failures(base_gate, [r["gate"]["scores"] for r in runs])
        if failed:
            verdict = "REGRESSES"
        elif gap["material"]:
            verdict = "COSTS QUALITY"
        else:
            verdict = "ACCEPTABLE"
        paired = [r for r in runs if r["seed"] in full]
        arms[a] = {"seeds": [r["seed"] for r in runs], "nll": nll,
                   "nll_mean": statistics.fmean(nll), "task_gap": gap, "leg2_failed": failed,
                   "accuracy": [r["task"]["accuracy"] for r in runs], "verdict": verdict,
                   "paired_clustered": rule.paired_clustered(paired, full) if paired else None}
    out.update(verdict="MEASURED", arms=arms)
    return out


def ram_memory(ram: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The adapter-state increment AS - A on the RAM tier against two byte rules."""
    runs = {a: _by_seed([r for r in ram if r["arm"] == a and _ok(r)], f"RAM-tier {a}")
            for a in RAM_ARMS}
    seeds = sorted(set(runs["A"]) & set(runs["AS"]))
    if not seeds:
        return {"missing": "no seed has both RAM-tier A and AS"}

    def one(arm: str, field: str) -> int:
        values = {runs[arm][s]["ram_tier"][field] for s in seeds}
        if len(values) != 1:
            raise ValueError(f"RAM-tier {arm} {field} differs across seeds: {sorted(values)}")
        return values.pop()

    params = {a: one(a, "trainable_parameters") for a in RAM_ARMS}
    bpp = {a: one(a, "bytes_per_parameter") for a in RAM_ARMS}
    if bpp["A"] != bpp["AS"]:
        raise ValueError(f"bytes_per_parameter differs between A and AS: {bpp}")
    extra = params["AS"] - params["A"]
    predicted = {"e1": 16 * extra, "dtype": bpp["A"] * extra}
    increments = {s: runs["AS"][s]["ram_tier"]["peak_cuda_bytes"]
                  - runs["A"][s]["ram_tier"]["peak_cuda_bytes"] for s in seeds}
    max_increment = (max(runs["AS"][s]["ram_tier"]["peak_cuda_bytes"] for s in seeds)
                     - max(runs["A"][s]["ram_tier"]["peak_cuda_bytes"] for s in seeds))
    out: Dict[str, Any] = {"seeds": seeds, "extra_parameters": extra,
                           "bytes_per_parameter": bpp["A"],
                           "increment": {str(s): v for s, v in increments.items()},
                           "increment_max_over_seeds": max_increment,
                           "tolerance_bytes": MEMORY_TOLERANCE_BYTES}
    for name, value in predicted.items():
        residuals = {str(s): v - value for s, v in increments.items()}
        residual_max = max_increment - value
        out[f"predicted_{name}"] = value
        out[f"residual_{name}"] = residuals
        out[f"residual_{name}_mib"] = {s: v / 2**20 for s, v in residuals.items()}
        out[f"residual_{name}_max_over_seeds"] = residual_max
        out[f"residual_{name}_max_over_seeds_mib"] = residual_max / 2**20
        out[f"holds_{name}_rule"] = all(
            abs(v) <= MEMORY_TOLERANCE_BYTES for v in [*residuals.values(), residual_max])
    return out


def confirm_problem(record: Dict[str, Any], expect: Dict[str, Any]) -> Optional[str]:
    """``rule.provenance_problem`` plus the confirmation split and wrapper hashes."""
    problem = rule.provenance_problem(record, harness_sha256=expect["harness_sha256"],
                                      split_sha256=expect["split_sha256"],
                                      hot_sha256=expect["hot_sha256"])
    return problem or _confirm_hash_problem(record, expect)


def _confirm_hash_problem(record: Dict[str, Any], expect: Dict[str, Any]) -> Optional[str]:
    confirm = record.get("confirm")
    if confirm is None:
        return None
    if confirm.get("split_sha256") != expect["confirm_split_sha256"]:
        return f"confirm split {confirm.get('split_sha256')}"
    if expect.get("wrapper_sha256") and confirm.get("wrapper_sha256") != expect["wrapper_sha256"]:
        return f"confirm wrapper {confirm.get('wrapper_sha256')}"
    return None


def _ram_problem(record: Dict[str, Any], expect: Dict[str, Any]) -> Optional[str]:
    if record.get("split_sha256") != expect["split_sha256"]:
        return f"split {record.get('split_sha256')}"
    return _confirm_hash_problem(record, expect)


def _filter(records, check, expect, rejected):
    kept = []
    for r in records:
        problem = check(r, expect)
        if problem is None:
            kept.append(r)
        else:
            rejected.append({"source": r.get("_path"), "problem": problem})
    return kept


def _historical(result: Dict[str, Any]) -> Dict[str, Any]:
    return {"note": HISTORICAL_NOTE, "verdict": result["verdict"],
            "reason": result.get("reason"), "fullest": result.get("fullest"),
            "arms": {a: {"verdict": row["verdict"], "task_gap": row["task_gap"]}
                     for a, row in result.get("arms", {}).items()}}


MEMORY_PARITY_NOTE = ("memory is a property of the regime and is reported whether or not the "
                      "parity pilot passed; §11.3's strict reading counts nothing from these runs")


PARITY_MIN_STEPS = 4


def _parity_problem(parity: Dict[str, Any]) -> Optional[str]:
    """Why ``parity`` is not a passed NF4 RAM-tier CUDA pilot of four steps, or None."""
    if parity.get("status") != "ok" or parity.get("passed") is not True:
        return (f"parity pilot did not pass (status {parity.get('status')}, passed "
                f"{parity.get('passed')})")
    if parity.get("quant") != "nf4":
        return f"parity quant {parity.get('quant')!r}, not 'nf4'"
    if parity.get("tier") != "ram":
        return f"parity tier {parity.get('tier')!r}, not 'ram'"
    if not str(parity.get("device") or "").startswith("cuda"):
        return f"parity device {parity.get('device')!r}, not cuda"
    if len(parity.get("steps") or []) < PARITY_MIN_STEPS:
        return (f"parity steps {len(parity.get('steps') or [])}, fewer than "
                f"{PARITY_MIN_STEPS}")
    sha = parity.get("harness_sha256")
    if not (isinstance(sha, str) and len(sha) == 64
            and all(ch in "0123456789abcdef" for ch in sha.lower())):
        return f"parity harness_sha256 {sha!r} missing or not a SHA-256"
    return None


def parity_gate(parity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 11.3: RAM-tier runs count only if the NF4 RAM-tier parity pilot passed."""
    if parity is None:
        return {"counted": False, "reason": "no parity record"}
    reason = _parity_problem(parity)
    counted = reason is None
    grad = parity.get("gradient_step0") or {}
    return {"counted": counted, "reason": reason,
            "bit_exact": parity.get("bit_exact"),
            "steps": [{k: s.get(k) for k in ("step", "streamed", "resident", "relative")}
                      for s in parity.get("steps", [])],
            "gradient_step0": {k: grad.get(k)
                               for k in ("global_relative_l2", "min_cosine", "max_abs")},
            "rule": parity.get("rule"), "_path": parity.get("_path")}


def apply_parity_gate(ram_tier: Dict[str, Any], parity: Optional[Dict[str, Any]]) -> None:
    """Withhold the RAM-tier verdicts (descriptive fields kept) unless the pilot passed."""
    gate = parity_gate(parity)
    ram_tier["parity"] = gate
    if not gate["counted"]:
        for s in TASK_SETS:
            ram_tier[s]["verdict_if_counted"] = ram_tier[s]["verdict"]
            ram_tier[s]["verdict"] = "NOT COUNTED"
            ram_tier[s]["gate_reason"] = gate["reason"]
    ram_tier["memory"]["parity_counted"] = gate["counted"]
    ram_tier["memory"]["note"] = MEMORY_PARITY_NOTE


def analyse(confirm_records: Sequence[Dict[str, Any]],
            published_records: Sequence[Dict[str, Any]],
            ram_records: Sequence[Dict[str, Any]] = (), *, expect: Dict[str, Any],
            checks: Sequence[Dict[str, Any]] = (), parity: Optional[Dict[str, Any]] = None,
            draws: int = rule.BOOTSTRAP_DRAWS) -> Dict[str, Any]:
    """Everything the confirmation reports, as one JSON-able dict.

    ``expect``: ``harness_sha256``, ``split_sha256``, ``hot_sha256`` (E1's, also used to
    filter the published records), ``confirm_split_sha256`` and optional
    ``wrapper_sha256``. ``checks``: second-box base records (``confirmcheck-*``).
    ``parity``: the RAM-tier parity pilot record (``parity-*``), or None; see
    ``apply_parity_gate``. Per-row checks run on every ok record before provenance.
    """
    for r in [*confirm_records, *checks, *ram_records]:
        if _ok(r):
            check_per_row(r)
    rejected: List[Dict[str, Any]] = []
    records = _filter(confirm_records, confirm_problem, expect, rejected)
    checks = _filter(checks, confirm_problem, expect, rejected)
    ram = _filter(ram_records, _ram_problem, expect, rejected)
    e1_expect = {k: expect[k] for k in ("harness_sha256", "split_sha256", "hot_sha256")}
    published = rule.analyse(published_records, expect=e1_expect)
    published_kept = [r for r in published_records
                      if rule.provenance_problem(r, **e1_expect) is None]
    models: Dict[str, Any] = {}
    counted_all: List[Dict[str, Any]] = []
    bases: Dict[str, Dict[str, Any]] = {}
    for model in sorted({r["model"] for r in records} | set(rule.PLAN)):
        part = frozen_partition(records, model)
        counted_all += part["counted"]
        ok_counted = [r for r in part["counted"] if _ok(r)]
        base = [r for r in ok_counted if r["arm"] == "base"]
        if len(base) == 1:
            bases[model] = base[0]
        result = {s: verdicts(records, model, s) for s in TASK_SETS}
        result["comparison"] = compare(result["E"], result["C"])
        result["contrast"] = contrast(ok_counted, draws=draws)
        if model in published["models"]:
            result["published"] = _historical(published["models"][model])
        models[model] = result
    out: Dict[str, Any] = {
        "gate": rule.GATE, "frozen_rates": FROZEN_RATES, "v3_note": V3_NOTE,
        "rejected": rejected, "models": models,
        "published_rejected": published["rejected"],
        "reproduction": reproduction(counted_all, published_kept),
        "sweeps": sweeps(records), "box_checks": box_checks(checks, bases)}
    if ram:
        resident = frozen_partition(records, RAM_MODEL)["counted"]
        out["ram_tier"] = {s: ram_verdicts(ram, resident, s) for s in TASK_SETS}
        out["ram_tier"]["memory"] = ram_memory(ram)
        apply_parity_gate(out["ram_tier"], parity)
    return out


def _f(value: Optional[float], digits: int, signed: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(values: Optional[List[float]], digits: int) -> str:
    return "-" if values is None else f"[{values[0]:+.{digits}f}, {values[1]:+.{digits}f}]"


def _gap(gap: Dict[str, Any]) -> str:
    return f"{gap['gap']:+.4f} ({'material' if gap['material'] else 'not material'})"


def _mib(value: float) -> str:
    return f"{value / 2**20:+.1f} MiB"


def _verdict_lines(model: str, result: Dict[str, Any]) -> List[str]:
    lines = []
    for s in TASK_SETS:
        v = result[s]
        line = f"- {s}: **{v['verdict']}**"
        if v.get("reason"):
            line += f"; reason: {v['reason']}"
        if v.get("missing"):
            line += "; missing: " + ", ".join(f"`{m}`" for m in v["missing"])
        if v.get("validity"):
            line += "; validity: " + ", ".join(f"{k} {'pass' if ok else 'FAIL'}"
                                               for k, ok in v["validity"].items())
        lines.append(line)
    cmp = result["comparison"]
    if cmp["verdict"] != "MEASURED":
        lines += ["", "No comparison: E or C is NO VERDICT (no `holds` given).", ""]
        return lines
    lines += ["", f"Arm order by mean NLL: E {' < '.join(cmp['ordering_E'])}; "
                  f"C {' < '.join(cmp['ordering_C'])}.", "",
              "| Arm | E verdict | C verdict | Holds | E gap | C gap | Gap holds "
              "| Per-seed gaps E / C | Accuracy E / C |",
              "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        if arm not in cmp["arms"]:
            continue
        a = cmp["arms"][arm]
        seeds = "; ".join(f"s{s}: {a['seed_gaps_E'][s]:+.4f} / {a['seed_gaps_C'].get(s, 0):+.4f}"
                          for s in a["seed_gaps_E"] if s in a["seed_gaps_C"])
        acc = (", ".join(f"{x:.3f}" for x in a["accuracy_E"]) + " / "
               + ", ".join(f"{x:.3f}" for x in a["accuracy_C"]))
        lines.append(f"| `{arm}` | {a['verdict_E']} | {a['verdict_C']} "
                     f"| {'yes' if a['verdict_holds'] else 'NO'} | {_gap(a['task_gap_E'])} "
                     f"| {_gap(a['task_gap_C'])} | {'yes' if a['gap_holds'] else 'NO'} "
                     f"| {seeds or '-'} | {acc} |")
    return lines + [""]


def _contrast_lines(con: Dict[str, Any]) -> List[str]:
    if "missing" in con:
        return [f"Contrast: not computed ({con['missing']}).", ""]
    lines = [f"{CONTRAST_NAME[0].upper()}{CONTRAST_NAME[1:]}: {CONTRAST_NOTE}. "
             f"Bootstrap {con['draws']} draws, seed {con['seed']}; E rows {con['rows']['E']}, "
             f"C rows {con['rows']['C']}. Gap difference = (X - {con['fullest']}) on E minus "
             f"(X - {con['fullest']}) on C.", "",
             "| Arm | NLL E | NLL C | E-C | Contrast [95% CI] | Accuracy E | Accuracy C "
             "| Contrast [95% CI] | Gap difference NLL [95% CI] |",
             "|---|---:|---:|---:|---|---:|---:|---|---|"]
    b = con["base"]
    lines.append(f"| base | {b['nll_E']:.4f} | {b['nll_C']:.4f} | {b['e_minus_c_nll']:+.4f} "
                 f"| - | {b['accuracy_E']:.3f} | {b['accuracy_C']:.3f} | - | - |")
    for arm in ARM_ORDER:
        if arm not in con["arms"]:
            continue
        r = con["arms"][arm]
        gd = ("-" if "gap_difference_nll" not in r else
              f"{r['gap_difference_nll']:+.4f} {_ci(r['gap_difference_nll_ci95'], 4)}")
        lines.append(
            f"| `{arm}` | {r['nll_E']:.4f} | {r['nll_C']:.4f} | {r['e_minus_c_nll']:+.4f} "
            f"| {r['contrast_nll']:+.4f} {_ci(r['contrast_nll_ci95'], 4)} "
            f"| {r['accuracy_E']:.3f} | {r['accuracy_C']:.3f} "
            f"| {r['contrast_accuracy']:+.3f} {_ci(r['contrast_accuracy_ci95'], 3)} | {gd} |")
    return lines + [""]


def _published_lines(pub: Optional[Dict[str, Any]]) -> List[str]:
    if pub is None:
        return []
    lines = [f"Published E1 verdict ({pub['note']}): **{pub['verdict']}**"
             + (f"; reason: {pub['reason']}" if pub.get("reason") else ""), ""]
    if pub["arms"]:
        lines += ["| Arm | Published verdict | Published E gap |", "|---|---|---|"]
        lines += [f"| `{a}` | {pub['arms'][a]['verdict']} | {_gap(pub['arms'][a]['task_gap'])} |"
                  for a in ARM_ORDER if a in pub["arms"]]
        lines.append("")
    return lines


def _parity_lines(p: Dict[str, Any]) -> List[str]:
    head = f"Parity pilot: {'PASSED' if p['counted'] else 'NOT PASSED'}"
    if p["reason"]:
        head += f" ({p['reason']})"
    if p.get("_path"):
        head += f", `{p['_path']}`"
    lines = [head + ".", ""]
    limits = p.get("rule") or {}
    if p.get("steps"):
        lines += ["| Step | Streamed loss | Resident loss | Relative |", "|---:|---:|---:|---:|"]
        lines += [f"| {s.get('step')} | {_f(s.get('streamed'), 6)} | {_f(s.get('resident'), 6)} "
                  f"| {_f(s.get('relative'), 6)} |" for s in p["steps"]]
        lines += ["", f"Loss rule: relative <= {limits.get('loss_relative', '-')}.", ""]
    grad = p.get("gradient_step0")
    if grad:
        lines += [f"Step-0 gradient: global relative L2 {_f(grad.get('global_relative_l2'), 6)} "
                  f"(rule <= {limits.get('grad_relative_l2', '-')}), min cosine "
                  f"{_f(grad.get('min_cosine'), 6)} (rule >= {limits.get('grad_cosine', '-')}), "
                  f"max abs {_f(grad.get('max_abs'), 6)}; bit exact {p.get('bit_exact')}.", ""]
    return lines


def _ram_lines(ram: Dict[str, Any]) -> List[str]:
    lines = [f"## RAM tier ({RAM_MODEL})", ""] + _parity_lines(ram["parity"])
    counted = ram["parity"]["counted"]
    for s in TASK_SETS:
        t = ram[s]
        if counted:
            line = f"{s}: **{t['verdict']}**"
        else:
            line = (f"{s}: **{t['verdict']}** ({t['gate_reason']}); descriptive verdict if "
                    f"counted: {t['verdict_if_counted']}")
        if t.get("missing"):
            line += "; missing: " + ", ".join(f"`{m}`" for m in t["missing"])
        if t.get("validity"):
            line += "; validity: " + ", ".join(f"{k} {'pass' if ok else 'FAIL'}"
                                               for k, ok in t["validity"].items())
        lines += [line, ""]
        if t.get("arms"):
            if not counted:
                lines += ["Arms (descriptive, not a verdict):", ""]
            lines += ["| Arm | Seeds | NLL mean | Gap vs resident AE | Leg 2 failed "
                      "| Clustered accuracy vs AE | Verdict |", "|---|---|---:|---|---|---|---|"]
            for a, r in t["arms"].items():
                c = r["paired_clustered"]
                pc = "-" if c is None else f"{c['difference'] * 100:+.1f} pp, p={c['p']:.3g}"
                lines.append(f"| `{a}` | {r['seeds']} | {r['nll_mean']:.4f} "
                             f"| {_gap(r['task_gap'])} | {', '.join(r['leg2_failed']) or '-'} "
                             f"| {pc} | **{r['verdict']}** |")
            lines.append("")
        vs = t["vs_resident"]
        if vs["arms"]:
            lines += [f"vs resident ({vs['note']}):", "",
                      "| Arm | NLL RAM - resident | Accuracy RAM - resident |", "|---|---:|---:|"]
            lines += [f"| `{a}` | {r['nll_difference']:+.4f} | {r['accuracy_difference']:+.3f} |"
                      for a, r in vs["arms"].items()]
            lines.append("")
    mem = ram["memory"]
    if "missing" in mem:
        return lines + [f"Memory: {mem['missing']}.", ""]
    lines += [f"Memory ({mem['note']}; AS - A peak, seeds {mem['seeds']}): extra parameters "
              f"{mem['extra_parameters']}, bytes per parameter {mem['bytes_per_parameter']}, "
              f"tolerance {_mib(mem['tolerance_bytes'])}.", "",
              "| Rule | Predicted | Residual per seed | Residual max-over-seeds | Holds |",
              "|---|---:|---|---:|---|"]
    for name in ("e1", "dtype"):
        per = ", ".join(f"s{s}: {_mib(v)}" for s, v in mem[f"residual_{name}"].items())
        lines.append(f"| {name} | {mem[f'predicted_{name}'] / 2**20:.1f} MiB | {per} "
                     f"| {_mib(mem[f'residual_{name}_max_over_seeds'])} "
                     f"| {'yes' if mem[f'holds_{name}_rule'] else 'NO'} |")
    return lines + [""]


def tables(analysis: Dict[str, Any]) -> str:
    gate = analysis["gate"]
    lines = ["**Quality gate.** "
             + ", ".join(f"{k} {v:g}" for k, v in gate["thresholds"].items())
             + f". Status: {gate['status']}. Scope: {gate['scope']}.", "",
             "Frozen rates: " + "; ".join(
                 f"{m} " + ", ".join(f"`{a}` {lr:g}" for a, lr in rates.items())
                 for m, rates in analysis["frozen_rates"].items()) + f". {V3_NOTE}.", ""]
    for model, result in analysis["models"].items():
        lines += [f"## {model}", ""]
        if result["E"].get("scope"):
            lines += [f"Scope: {result['E']['scope']}.", ""]
        lines += _verdict_lines(model, result)
        lines += _contrast_lines(result["contrast"])
        lines += _published_lines(result.get("published"))
    lines += ["## Reproduction of the published E1 records (historical check)", ""]
    if analysis["reproduction"]:
        lines += ["| Model | Arm | lr | Seed | Reproduced | Differs | Peak CUDA equal |",
                  "|---|---|---:|---:|---|---|---|"]
        for r in analysis["reproduction"]:
            rep = {True: "yes", False: "NO", None: "no published counterpart"}[r["reproduced"]]
            lines.append(f"| {r['model']} | `{r['arm']}` | {r['lr']:g} | {r['seed']} | {rep} "
                         f"| {', '.join(r.get('differ', [])) or '-'} "
                         f"| {r.get('peak_cuda_bytes_equal', '-')} |")
    else:
        lines.append("No counted confirmation records.")
    lines.append("")
    sw = analysis["sweeps"]
    lines += [f"## Learning-rate sweep on the new records ({sw['model']}, seed 0)", "",
              f"Note: {sw['note']}.", ""]
    if sw["arms"]:
        rates = sorted({float(lr) for a in sw["arms"].values() for s in TASK_SETS
                        if a[s] for lr in a[s]["nll"]})
        lines += ["| Arm | Set | " + " | ".join(f"{lr:g}" for lr in rates)
                  + " | Winner | Status | Frozen | C winner differs |",
                  "|---|---|" + "---:|" * len(rates) + "---:|---|---:|---|"]
        for arm, a in sw["arms"].items():
            for s in TASK_SETS:
                row = a[s] or {"nll": {}, "winner": None, "status": "missing"}
                cells = [_f(row["nll"].get(f"{lr:g}"), 4) for lr in rates]
                win = "-" if row["winner"] is None else f"{row['winner']:g}"
                frozen = "-" if a["frozen"] is None else f"{a['frozen']:g}"
                lines.append(f"| `{arm}` | {s} | " + " | ".join(cells)
                             + f" | {win} | {row['status']} | {frozen} "
                             f"| {a['c_winner_differs'] if s == 'C' else ''} |")
    else:
        lines.append("No seed-0 sweep records.")
    lines += ["", "## Box checks (second-box base, never counted)", ""]
    if analysis["box_checks"]:
        lines += [f"- `{b['source']}` ({b['model']}): identical {b['identical']}"
                  + (f"; differs: {', '.join(b['differ'])}" if b["differ"] else "")
                  + (f"; {b['reason']}" if b.get("reason") else "")
                  for b in analysis["box_checks"]]
    else:
        lines.append("None.")
    lines.append("")
    if "ram_tier" in analysis:
        lines += _ram_lines(analysis["ram_tier"])
    for title, key in (("Rejected confirmation records", "rejected"),
                       ("Rejected published records", "published_rejected")):
        if analysis.get(key):
            lines += [f"## {title}", ""] + [f"- `{r['source']}`: {r['problem']}"
                                           for r in analysis[key]] + [""]
    return "\n".join(lines).rstrip("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("confirm_dir")
    parser.add_argument("published_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--ram-tier-dir")
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--confirm-split-sha256", required=True)
    parser.add_argument("--hot", action="append", required=True, metavar="MODEL=SHA256")
    parser.add_argument("--wrapper-sha256")
    parser.add_argument("--draws", type=int, default=rule.BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)
    expect = {"harness_sha256": args.harness_sha256, "split_sha256": args.split_sha256,
              "hot_sha256": dict(item.split("=", 1) for item in args.hot),
              "confirm_split_sha256": args.confirm_split_sha256,
              "wrapper_sha256": args.wrapper_sha256}
    ram = rule.load_runs(args.ram_tier_dir, "ramtier-*.json") if args.ram_tier_dir else []
    parity = None
    if args.ram_tier_dir:
        pilots = rule.load_runs(args.ram_tier_dir, "parity-*.json")
        if len(pilots) > 1:
            raise ValueError("more than one parity record: "
                             + ", ".join(p["_path"] for p in pilots))
        parity = pilots[0] if pilots else None
    analysis = analyse(rule.load_runs(args.confirm_dir, "confirm-*.json"),
                       rule.load_runs(args.published_dir), ram, expect=expect,
                       checks=rule.load_runs(args.confirm_dir, "confirmcheck-*.json"),
                       parity=parity, draws=args.draws)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(analysis, indent=1) + "\n", encoding="utf-8",
                                      newline="\n")
    (out / "tables.md").write_text(tables(analysis) + "\n", encoding="utf-8", newline="\n")
    console = Console()
    for model, result in analysis["models"].items():
        pub = result.get("published", {}).get("verdict", "-")
        console.print(f"{model}: E {result['E']['verdict']}, C {result['C']['verdict']}, "
                      f"published (historical) {pub}; missing on C: "
                      f"{len(result['C'].get('missing', []))}")
    reproduced = sum(r["reproduced"] is True for r in analysis["reproduction"])
    console.print(f"reproduced {reproduced}/{len(analysis['reproduction'])}; rejected "
                  f"{len(analysis['rejected'])}; wrote {out / 'summary.json'} and "
                  f"{out / 'tables.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
