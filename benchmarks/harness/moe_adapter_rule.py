"""Verdicts for probe-moe-adapter-placement.md section 5, from the run JSONs alone.

Pure Python on purpose: the rule is checked without torch, and a reader can
rerun it over the published JSONs. ``analyse`` is the only entry point the record
uses (through ``moe_adapter_report.py``): it picks each arm's rate from the sweep,
then ``summarise`` gives each model its verdicts on the runs at those rates.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence

MIN_LEARNED_NAT = 0.05  # V1
MATERIAL_NAT = 0.02
SD_MULTIPLE = 2.0
FORGETTING_THRESHOLD = 0.05  # soup ship leg 2 (DEFAULT_FORGETTING_THRESHOLD)
FULLEST_ORDER = ("AE", "AT", "AS", "A")
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


def _ok(record: Dict[str, Any]) -> bool:
    return record.get("status") == "ok"


def summarise(records: Sequence[Dict[str, Any]], *, edge_lrs: Optional[Dict[str, bool]] = None
              ) -> Dict[str, Any]:
    """Per model: validity rows and one verdict per applicable arm.

    ``records`` are the final-rate runs (sweep runs belong in ``edge_lrs``: arm ->
    True when its winning rate sat at the edge of the extended grid, V3).
    """
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


def analyse(records: Sequence[Dict[str, Any]], *, expect: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
    """Section 5 from every run record of the plan: rates first, then verdicts.

    ``expect`` (keys of ``provenance_problem``) rejects records made by another
    harness, split or hot set before anything else reads them. A trained run
    counts only at its arm's winning rate (``rate_source``); the others are listed,
    not scored. Runs that did not end ``ok`` always reach ``summarise``, so V2 and
    V4 see them at any rate. An arm whose rate is not settled (``incomplete`` or
    ``needs_extension``) leaves its model at NO VERDICT; V3 fails only on ``edge``.
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
                           edge_lrs={arm: s == "edge" for arm, s in status.items()}).get(
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
