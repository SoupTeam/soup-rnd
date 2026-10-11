"""Verdicts for probe-moe-adapter-placement.md section 5, from the run JSONs alone.

Pure Python on purpose: the rule is checked without torch, and a reader can
rerun it over the published JSONs. ``analyse`` is the only entry point the record
uses (through ``moe_adapter_report.py``): it picks each arm's rate from the sweep,
then ``summarise`` gives each model its verdicts on the runs at those rates.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence

MIN_LEARNED_NAT = 0.05  # V1
MATERIAL_NAT = 0.02
SD_MULTIPLE = 2.0
FORGETTING_THRESHOLD = 0.05  # soup ship leg 2 (DEFAULT_FORGETTING_THRESHOLD)
FULLEST_ORDER = ("AE", "AT", "AS", "A")
# The committed plan (sections 2 and 3): every counted (arm, seed) at its arm's chosen rate.
# A planned run without an ok record withholds the model's verdict.
PLAN = {"olmoe": {"arms": ("A", "AT", "AE"), "seeds": (0, 1, 2)},
        "qwen35": {"arms": ("A", "AS", "AT", "AE"), "seeds": (0, 1)}}
# Section 6: AE "does not fit" only when a run of it died of a CUDA out-of-memory error.
# The caching allocator's "memory allocation failed with OOM" warning is a retry, not this.
OOM_MARKERS = ("torch.OutOfMemoryError: CUDA out of memory",
               "torch.cuda.OutOfMemoryError: CUDA out of memory")
BOOTSTRAP_DRAWS = 10_000
GRID = (1e-4, 2e-4, 4e-4)  # section 3: swept per arm on SWEEP_MODEL at seed 0
EXTENSION = {1e-4: 5e-5, 4e-4: 8e-4}  # the one extra rate past a grid-edge winner
SWEEP_MODEL = "olmoe"
RATE_FROM = {"AS": "A"}  # section 6 addendum: Qwen3.5 AS reuses OLMoE A's rate


def sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def material_gap(arm: Sequence[float], full: Sequence[float]) -> Dict[str, Any]:
    """``g = mean(arm) - mean(full)``; material when > 0.02 nat AND > 2 x the larger SD."""
    gap = statistics.fmean(arm) - statistics.fmean(full)
    noise = SD_MULTIPLE * max(sample_sd(arm), sample_sd(full))
    return {"gap": gap, "noise": noise, "material": gap > MATERIAL_NAT and gap > noise}


def leg2_failures(base: Dict[str, float], runs: Sequence[Dict[str, float]]) -> List[str]:
    """Suites where ANY seed dropped more than the threshold below the base."""
    failed = []
    for suite, base_score in sorted(base.items()):
        if any(base_score - run[suite] > FORGETTING_THRESHOLD + 1e-12 for run in runs):
            failed.append(suite)
    return failed


def bootstrap_interval(correct: Sequence[bool], seed: int = 0) -> List[float]:
    rng = random.Random(seed)
    n = len(correct)
    means = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(BOOTSTRAP_DRAWS))
    return [means[int(0.025 * BOOTSTRAP_DRAWS)], means[int(0.975 * BOOTSTRAP_DRAWS) - 1]]


def mcnemar_exact(x_only: int, f_only: int) -> float:
    """Two-sided exact McNemar p: binomial(n = x_only + f_only, 1/2) on the discordant pairs."""
    n = x_only + f_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(x_only, f_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_accuracy(run: Dict[str, Any], full: Dict[str, Any]) -> Dict[str, Any]:
    """Seed-matched X against F on the same evaluation questions (reporting only)."""
    x, f = run["task"]["correct"], full["task"]["correct"]
    if len(x) != len(f):
        raise ValueError("paired runs scored different evaluation rows")
    x_only = sum(a and not b for a, b in zip(x, f))
    f_only = sum(b and not a for a, b in zip(x, f))
    return {"seed": run["seed"], "x_only": x_only, "f_only": f_only,
            "difference": (x_only - f_only) / len(x), "p": mcnemar_exact(x_only, f_only)}


def oom_evidence(log: Optional[str]) -> Optional[str]:
    """The log line proving a CUDA out-of-memory death, or None."""
    for line in (log or "").splitlines():
        if any(marker in line for marker in OOM_MARKERS):
            return line.strip()
    return None


def _ok(record: Dict[str, Any]) -> bool:
    return record.get("status") == "ok"


def _completeness(model: str, arms: Dict[str, List[Dict[str, Any]]],
                  plan: Dict[str, Dict[str, Sequence]], ae_log: Optional[str]
                  ) -> Dict[str, Any]:
    """Planned runs that have no ok record, and whether AE left the plan on OOM proof."""
    if model not in plan:
        return {"missing": [], "reason": f"no committed plan for {model}"}
    arms_planned, seeds = plan[model]["arms"], plan[model]["seeds"]
    have = {}
    for arm, runs in arms.items():
        for r in runs:
            key = (arm, r["seed"])
            if key in have:
                raise ValueError(f"two ok runs for {arm} seed {r['seed']} on {model}")
            have[key] = r
    out: Dict[str, Any] = {"missing": [], "reason": None, "ae_did_not_fit": None}
    if "AE" in arms_planned and not arms.get("AE"):
        proof = oom_evidence(ae_log)
        if proof is not None:
            out["ae_did_not_fit"] = proof
            arms_planned = tuple(a for a in arms_planned if a != "AE")
        elif ae_log is not None:
            out["reason"] = "AE log has no CUDA out-of-memory error"
    elif "AE" in arms_planned and ae_log is not None and oom_evidence(ae_log):
        out["reason"] = "an AE run completed, so an AE out-of-memory log does not drop AE"
    out["missing"] = [f"{arm} seed {seed}" for arm in arms_planned for seed in seeds
                      if (arm, seed) not in have]
    return out


def summarise(records: Sequence[Dict[str, Any]], *, edge_lrs: Optional[Dict[str, bool]] = None,
              plan: Optional[Dict[str, Dict[str, Sequence]]] = None,
              ae_oom_logs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Per model: validity rows and one verdict per applicable arm.

    ``records`` are the final-rate runs (sweep runs belong in ``edge_lrs``: arm ->
    True when its winning rate sat at the edge of the extended grid, V3). Every run
    in ``plan`` (default: the committed PLAN) must have an ok record, or the model is
    NO VERDICT with the ``missing`` runs listed. AE leaves the plan, and the fullest
    arm falls to the next one, only when ``ae_oom_logs[model]`` proves an AE run died
    of a CUDA out-of-memory error and no AE run completed.
    """
    plan = PLAN if plan is None else plan
    out: Dict[str, Any] = {}
    for model in sorted({r["model"] for r in records}):
        mine = [r for r in records if r["model"] == model]
        base = [r for r in mine if r["arm"] == "base" and _ok(r)]
        arms: Dict[str, List[Dict[str, Any]]] = {}
        not_applicable = sorted({r["arm"] for r in mine if r.get("status") == "not_applicable"})
        void = [{"arm": r["arm"], "seed": r.get("seed"), "status": r.get("status")}
                for r in mine if r["arm"] != "base" and r.get("status")
                not in ("ok", "not_applicable")]
        for r in mine:
            if r["arm"] != "base" and _ok(r):
                arms.setdefault(r["arm"], []).append(r)
        result: Dict[str, Any] = {"not_applicable": not_applicable, "void": void}
        complete = _completeness(model, arms, plan, (ae_oom_logs or {}).get(model))
        if complete.get("ae_did_not_fit"):
            result["ae_did_not_fit"] = complete["ae_did_not_fit"]
        if complete["missing"]:
            result["missing"] = complete["missing"]
        if len(base) != 1:
            result["verdict"] = "NO VERDICT"
            result["reason"] = f"expected one base record, found {len(base)}"
            out[model] = result
            continue
        base_nll = base[0]["eval"]["nll"]
        base_gate = base[0]["gate"]["scores"]
        fullest = next((arm for arm in FULLEST_ORDER if arm in arms), None)
        validity = {
            "V1": fullest is not None and all(base_nll - r["eval"]["nll"] >= MIN_LEARNED_NAT
                                              for r in arms[fullest]),
            "V2": not any(r.get("status") == "void_nonfinite" for r in mine),
            "V3": not any((edge_lrs or {}).values()),
            "V4": all(r.get("v4", {}).get("passed", False) for r in mine
                      if r["arm"] != "base" and r.get("status") != "not_applicable"),
        }
        result.update(fullest=fullest, validity=validity, base_nll=base_nll,
                      base_gate=base_gate, base_source=base[0].get("_path"),
                      base_accuracy=base[0]["task"]["accuracy"])
        if not all(validity.values()):
            result["verdict"] = "NO VERDICT"
            out[model] = result
            continue
        if complete["missing"] or complete["reason"]:
            result["verdict"] = "NO VERDICT"
            if complete["reason"]:
                result["reason"] = complete["reason"]
            out[model] = result
            continue
        full_by_seed = {r["seed"]: r for r in arms[fullest]}
        full_nll = [r["eval"]["nll"] for r in arms[fullest]]
        rows = {}
        for arm, runs in sorted(arms.items()):
            runs = sorted(runs, key=lambda r: r["seed"])
            nll = [r["eval"]["nll"] for r in runs]
            gap = material_gap(nll, full_nll)
            failures = leg2_failures(base_gate, [r["gate"]["scores"] for r in runs])
            if failures:
                verdict = "REGRESSES"
            elif gap["material"] and arm != fullest:
                verdict = "COSTS QUALITY"
            else:
                verdict = "ACCEPTABLE"
            pooled = [c for r in runs for c in r["task"]["correct"]]
            paired = None if arm == fullest else [
                paired_accuracy(r, full_by_seed[r["seed"]]) for r in runs]
            rows[arm] = {
                "seeds": [r["seed"] for r in runs], "nll": nll,
                "nll_mean": statistics.fmean(nll), "nll_sd": sample_sd(nll),
                "task_gap": gap, "leg2_failed": failures, "verdict": verdict,
                "accuracy": [r["task"]["accuracy"] for r in runs],
                "accuracy_pooled_ci95": bootstrap_interval(pooled),
                "trainable_parameters": runs[0]["trainable_parameters"],
                "adapter_bytes_adamw_fp32": runs[0]["adapter_bytes_adamw_fp32"],
                "peak_cuda_bytes": max(r.get("peak_cuda_bytes", 0) for r in runs),
                "seconds_per_step": statistics.fmean(
                    e["seconds"] for r in runs for e in r["train"]["log"][1:]),
                "gate": {s: [r["gate"]["scores"][s] for r in runs] for s in base_gate},
                "sources": [r.get("_path") for r in runs],
                "paired_vs_fullest": paired,
            }
        result["arms"] = rows
        result["verdict"] = "MEASURED"
        out[model] = result
    return out


def rate_source(model: str, arm: str) -> str:
    """The sweep arm whose winning rate ``arm`` trains at on ``model``."""
    return arm if model == SWEEP_MODEL else RATE_FROM.get(arm, arm)


def lr_sweep(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per trained arm of SWEEP_MODEL: seed-0 evaluation NLL per rate and the winner.

    ``status`` is ``ok`` (the winner is inside the extended grid), ``edge`` (the
    extension won, V3 fails), ``needs_extension`` (a grid-edge rate won and its
    extension has not run) or ``incomplete`` (a grid rate has no ok seed-0 run).
    """
    seed0 = [r for r in records if r["model"] == SWEEP_MODEL and r["arm"] != "base"
             and r.get("seed") == 0 and _ok(r)]
    out: Dict[str, Dict[str, Any]] = {}
    for arm in sorted({r["arm"] for r in seed0}):
        nll: Dict[float, float] = {}
        for r in seed0:
            if r["arm"] != arm:
                continue
            if r["lr"] in nll:
                raise ValueError(f"two ok seed-0 {SWEEP_MODEL} {arm} runs at lr {r['lr']:g}")
            nll[r["lr"]] = r["eval"]["nll"]
        row: Dict[str, Any] = {"nll": {f"{lr:g}": v for lr, v in sorted(nll.items())},
                               "winner": None}
        out[arm] = row
        if any(lr not in nll for lr in GRID):
            row["status"] = "incomplete"
            continue
        winner = min(GRID, key=nll.__getitem__)
        extension = EXTENSION.get(winner)
        if extension is None:
            row.update(status="ok", winner=winner)
        elif extension not in nll:
            row.update(status="needs_extension", extension=extension)
        else:
            winner = min((*GRID, extension), key=nll.__getitem__)
            row.update(status="edge" if winner == extension else "ok", winner=winner,
                       extension=extension)
    return out


def provenance_problem(record: Dict[str, Any], *, harness_sha256: str, split_sha256: str,
                       hot_sha256: Dict[str, str]) -> Optional[str]:
    """Why ``record`` was not made by the bundled harness on the committed inputs."""
    if record.get("harness_sha256") != harness_sha256:
        return f"harness {record.get('harness_sha256')}"
    if record.get("split_sha256") != split_sha256:
        return f"split {record.get('split_sha256')}"
    if record["arm"] == "AT" and record.get("status") != "not_applicable" \
            and record.get("hot_sha256") != hot_sha256.get(record["model"]):
        return f"hot set {record.get('hot_sha256')}"
    return None


def analyse(records: Sequence[Dict[str, Any]], *, expect: Optional[Dict[str, Any]] = None,
            plan: Optional[Dict[str, Dict[str, Sequence]]] = None,
            ae_oom_logs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Section 5 from every run record of the plan: rates first, then verdicts.

    ``expect`` (keys of ``provenance_problem``) rejects records made by another
    harness, split or hot set before anything else reads them. A trained run
    counts only at its arm's winning rate (``rate_source``); the others are listed,
    not scored. Runs that did not end ``ok`` always reach ``summarise``, so V2 and
    V4 see them at any rate. An arm whose rate is not settled (``incomplete`` or
    ``needs_extension``) leaves its model at NO VERDICT; V3 fails only on ``edge``.
    ``plan`` and ``ae_oom_logs`` go to ``summarise`` (completeness of the plan).
    """
    rejected: List[Dict[str, Any]] = []
    if expect is not None:
        kept = []
        for r in records:
            problem = provenance_problem(r, **expect)
            if problem is None:
                kept.append(r)
            else:
                rejected.append({"source": r.get("_path"), "problem": problem})
        records = kept
    sweep = lr_sweep(records)
    counted: List[Dict[str, Any]] = []
    not_counted: List[Dict[str, Any]] = []
    for r in records:
        trained = r["arm"] != "base" and r.get("status") != "not_applicable"
        winner = sweep.get(rate_source(r["model"], r["arm"]), {}).get("winner")
        if not trained or not _ok(r) or r["lr"] == winner:
            counted.append(r)
        else:
            not_counted.append(r)
    models: Dict[str, Any] = {}
    for model in sorted({r["model"] for r in records}):
        arms = {r["arm"] for r in records if r["model"] == model and r["arm"] != "base"
                and r.get("status") != "not_applicable"}
        status = {arm: sweep.get(rate_source(model, arm), {}).get("status", "missing")
                  for arm in arms}
        result = summarise([r for r in counted if r["model"] == model],
                           edge_lrs={arm: s == "edge" for arm, s in status.items()},
                           plan=plan, ae_oom_logs=ae_oom_logs).get(
            model, {"verdict": "NO VERDICT", "reason": "no counted runs"})
        unsettled = sorted(arm for arm, s in status.items() if s not in ("ok", "edge"))
        if unsettled:
            result.pop("arms", None)
            result.update(verdict="NO VERDICT", unsettled_rates=unsettled)
        models[model] = result
    return {"sweep": sweep, "models": models, "rejected": rejected,
            "not_counted": [{"model": r["model"], "arm": r["arm"], "lr": r["lr"],
                             "seed": r.get("seed"), "source": r.get("_path")}
                            for r in not_counted]}


def load_runs(directory: str, pattern: str = "run-*.json") -> List[Dict[str, Any]]:
    """Every ``pattern`` file under ``directory`` at any depth, with its relative ``_path``."""
    root = pathlib.Path(directory)
    records = []
    for path in sorted(root.rglob(pattern)):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = path.relative_to(root).as_posix()
        records.append(record)
    return records
