"""Verdicts for probe-moe-adapter-placement.md section 5, from the run JSONs alone.

Pure Python on purpose: the rule is checked without torch, and a reader can
rerun it over the published JSONs. ``summarise`` is the only entry point the
record uses; ``python moe_adapter_rule.py <runs-dir>`` prints its JSON.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

MIN_LEARNED_NAT = 0.05  # V1
MATERIAL_NAT = 0.02
SD_MULTIPLE = 2.0
FORGETTING_THRESHOLD = 0.05  # soup ship leg 2 (DEFAULT_FORGETTING_THRESHOLD)
FULLEST_ORDER = ("AE", "AT", "AS", "A")
BOOTSTRAP_DRAWS = 10_000


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
            "V4": all(r.get("v4", {}).get("passed", False) for rs in arms.values() for r in rs),
        }
        result.update(fullest=fullest, validity=validity, base_nll=base_nll,
                      base_gate=base_gate,
                      base_accuracy=base[0]["task"]["accuracy"])
        if not all(validity.values()):
            result["verdict"] = "NO VERDICT"
            out[model] = result
            continue
        full_nll = [r["eval"]["nll"] for r in arms[fullest]]
        rows = {}
        for arm, runs in sorted(arms.items()):
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
                "seeds": sorted(r["seed"] for r in runs), "nll": nll,
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
            }
        result["arms"] = rows
        result["verdict"] = "MEASURED"
        out[model] = result
    return out


def load_runs(directory: str) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(pathlib.Path(directory).glob("run-*.json"))]


if __name__ == "__main__":
    print(json.dumps(summarise(load_runs(sys.argv[1])), indent=1))  # noqa: T201
