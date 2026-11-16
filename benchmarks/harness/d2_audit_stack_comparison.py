"""Independent full-tensor and telemetry replay of the explicit stack-v2 experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from rich.console import Console

try:
    from . import d2_audit_training as audit
    from . import d2_operator_probes as operators
except ImportError:
    import d2_audit_training as audit
    import d2_operator_probes as operators


def replay(root: Path, proofs: Path) -> dict:
    import torch

    torch.set_num_threads(1)
    arguments = audit.read(root / "arguments.json")
    if not arguments.get("stack_comparison") or arguments.get("steps") != 50:
        raise ValueError("Wrong stack experiment declaration")
    cycles = arguments["cycles"]
    stable_clocks = not arguments.get("default_boost", False)
    if not isinstance(cycles, int) or cycles < 3:
        raise ValueError("Missing ABBA repetitions")
    candidates = ("liger", "unsloth") if arguments.get("external_only", False) else (
        "peft", "liger", "unsloth")
    result = {"gates": {}, "arms": {}, "ratios": {}, "complete": False,
              "timing_candidates": candidates, "cycles": cycles,
              "clock_policy": "strict <=5%" if stable_clocks else "observed default GPU Boost",
              "scope": "configured training-stack speed; numerical drift reported separately"}
    operator_results = {}
    for backend in ("liger", "unsloth"):
        operator_results[backend] = operators.verify(proofs / backend, backend)
    for dtype in ("float16", "bfloat16"):
        peft = root / f"{dtype}-gate-peft"
        for backend in ("peft", "soup", "liger", "unsloth"):
            folder = root / f"{dtype}-gate-{backend}"
            checks = audit.replay(peft, folder)
            quantities = {row["quantity"] for row in checks["tensor_checks"]}
            allowed_drift = quantities | {"loss .3f mismatch"}
            structural = set(checks["failures"]) - allowed_drift
            ready = audit.read(folder / "ready.json")
            if ready["execution"].get("loss_normalization") != "token-sum":
                structural.add("wrong loss normalization")
            if backend == "unsloth" and not ready["execution"].get("fused_loss_scaler_attached"):
                structural.add("missing fused loss scaler")
            if backend in operator_results:
                for package, value in operator_results[backend]["packages"].items():
                    if ready["environment"]["packages"].get(package) != value:
                        structural.add("operator/model package mismatch: " + package)
            label = folder.name
            checks["lifecycle"] = audit.lifecycle(
                audit.read(root / f"{label}-telemetry.json"), ready,
                audit.read(root / f"{label}-process.json")["exit_code"],
                stable_clocks=stable_clocks)
            checks["admitted"] = not structural and checks["lifecycle"]["passed"] and (
                checks["passed"] or backend in ("liger", "unsloth"))
            checks["structural_failures"] = sorted(structural)
            result["gates"][f"{dtype}/{backend}"] = checks
        for candidate in candidates:
            ratios = []
            for cycle in range(cycles):
                durations = {"soup": [], candidate: []}
                clocks = []
                for position, backend in enumerate(("soup", candidate, candidate, "soup")):
                    label = f"{dtype}-{candidate}-{cycle}-{position}-{backend}"
                    folder = root / label
                    if not (folder / "snapshots.safetensors").exists():
                        result["arms"][label] = {"passed": False, "missing": True}
                        continue
                    checks = audit.replay(root / f"{dtype}-gate-{backend}", folder)
                    telemetry, ready = [audit.read(path) for path in (
                        root / f"{label}-telemetry.json", folder / "ready.json")]
                    checks["lifecycle"] = audit.lifecycle(
                        telemetry, ready, audit.read(root / f"{label}-process.json")["exit_code"],
                        stable_clocks=stable_clocks)
                    checks["passed"] &= checks["lifecycle"]["passed"]
                    if ready["execution"].get("loss_normalization") != "token-sum":
                        checks["passed"] = False
                    if backend == "unsloth" and not ready["execution"].get(
                        "fused_loss_scaler_attached"
                    ):
                        checks["passed"] = False
                    result["arms"][label] = checks
                    times = [row["wall_ms"] for row in audit.read(folder / "result.json")["rows"]]
                    if not all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0
                               for t in times):
                        checks["passed"] = False
                        checks["failures"].append("invalid wall timings")
                    if checks["passed"]:
                        durations[backend].append(sum(times))
                        clocks.extend(telemetry["live"])
                clocks_admitted = (audit.clocks(clocks) if stable_clocks
                                   else audit.clock_observations(clocks))
                if all(len(times) == 2 for times in durations.values()) and clocks_admitted:
                    ratios.append(statistics.mean(durations[candidate]) /
                                  statistics.mean(durations["soup"]))
                else:
                    for position, backend in enumerate(("soup", candidate, candidate, "soup")):
                        label = f"{dtype}-{candidate}-{cycle}-{position}-{backend}"
                        result["arms"][label]["passed"] = False
                result["ratios"][f"{dtype}/{candidate}"] = {
                    "candidate_over_soup": ratios,
                    "range": [min(ratios), max(ratios)] if ratios else None,
                    "median": statistics.median(ratios) if ratios else None}
    result["complete"] = all(row["admitted"] for row in result["gates"].values()) and all(
        row["passed"] for row in result["arms"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--operator-proofs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.root, args.operator_proofs)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print({"complete": result["complete"], "ratios": result["ratios"]})
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
