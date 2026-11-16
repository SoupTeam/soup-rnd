"""Offline replay of isolated D2 arms and the predeclared 5% SM-clock rule."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from rich.console import Console

try:
    from . import fast_lora_probe as probe
except ImportError:
    import fast_lora_probe as probe


def clock_verdict(samples: list[dict]) -> dict:
    clocks = [row.get("clock_mhz") for row in samples]
    valid = len(clocks) >= 3 and all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in clocks
    )
    spread = max(clocks) / min(clocks) - 1 if valid else None
    return {"passed": valid and spread <= 0.05, "samples": len(clocks), "spread": spread}


def audit(root: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    summary = json.loads((root / "summary.json").read_text())
    result = {
        "numerical_comparisons": 0,
        "numerical_failures": [],
        "arms": [],
        "cases": {},
        "scope": "synthetic dense single-layer; PEFT comparison only",
    }
    for arm in summary["arms"]:
        case = root / arm["case"]
        folder = case / arm["arm"]
        config = json.loads((case / "config.json").read_text())
        reference = load_file(str(case / "correctness.safetensors"))
        actual = load_file(str(folder / "measured.safetensors"))
        required = probe._expected_quantities(config["path"], config["dx"])
        probe._require(set(actual) == required, "missing measured tensors")
        probe._require(
            set(reference)
            == {f"{label}/{key}" for label in ("fast", "PEFT", "float64") for key in required},
            "missing reference tensors",
        )
        exact = 0
        for key in sorted(required):
            value, expected, oracle = (
                actual[key],
                reference[f"PEFT/{key}"],
                reference[f"float64/{key}"],
            )
            passed = value.shape == expected.shape == oracle.shape and value.dtype == expected.dtype
            passed = passed and all(
                torch.isfinite(item).all().item() for item in (value, expected, oracle)
            )
            fast_error = (value.double() - oracle.double()).abs().max().item()
            peft_error = (expected.double() - oracle.double()).abs().max().item()
            passed = passed and fast_error <= 2 * peft_error + 1e-8
            result["numerical_comparisons"] += 1
            if not passed:
                result["numerical_failures"].append(
                    {"case": arm["case"], "arm": arm["arm"], "quantity": key}
                )
            exact += int(torch.equal(value, expected))
        telemetry = json.loads((folder / "telemetry.json").read_text())
        ready = json.loads((folder / "ready.json").read_text())
        timed = arm["measurement"]
        live = telemetry["live"]
        identities = {tuple(row["processes"]) for row in live}
        owner = (
            next(iter(identities))[0]
            if len(identities) == 1 and len(next(iter(identities))) == 1
            else None
        )
        lifecycle = owner is not None and arm["exit_code"] == 0
        lifecycle = lifecycle and all(
            row.get("available") is True and not row["graphics"]
            for rows in telemetry.values()
            for row in rows
        )
        lifecycle = lifecycle and all(
            not row["processes"] for row in telemetry["before"] + telemetry["after"]
        )
        lifecycle = lifecycle and all(
            row["processes"] in ([], [owner]) for row in telemetry["context_birth"]
        )
        def normalize(value: str) -> str:
            return value.lower().removeprefix("gpu-")
        lifecycle = lifecycle and all(
            normalize(row["uuid"]) == normalize(ready["uuid"]) for row in live
        )
        selected = [
            row for row in live if timed["started_unix"] <= row["unix"] <= timed["ended_unix"]
        ]
        clocks = clock_verdict(selected)
        validity = (
            lifecycle
            and clocks["passed"]
            and all(
                row["pstate"] in (0, 2)
                and row["throttle"] & ~0x5 == 0
                and math.isfinite(row["power_mw"])
                for row in selected
            )
        )
        result["arms"].append(
            {
                "case": arm["case"],
                "arm": arm["arm"],
                "cycle": arm["cycle"],
                "implementation": arm["implementation"],
                "passed": validity,
                "lifecycle": lifecycle,
                "clock": clocks,
                "clock_values": [row["clock_mhz"] for row in selected],
                "bit_exact": exact,
                "quantities": len(required),
            }
        )
    for name in sorted({row["case"] for row in summary["arms"]}):
        cycles = []
        for cycle in sorted({row["cycle"] for row in summary["arms"] if row["case"] == name}):
            raw = [row for row in summary["arms"] if row["case"] == name and row["cycle"] == cycle]
            checks = [
                row for row in result["arms"] if row["case"] == name and row["cycle"] == cycle
            ]
            across = clock_verdict(
                [{"clock_mhz": value} for row in checks for value in row["clock_values"]]
            )
            valid = len(raw) == 4 and all(row["passed"] for row in checks) and across["passed"]
            record = {"cycle": cycle, "accepted": valid, "clock": across}
            if valid:
                for phase in ("forward_ms", "backward_ms", "total_ms"):
                    values = {
                        label: [
                            sample[phase]
                            if phase != "total_ms"
                            else sample["forward_ms"] + sample["backward_ms"]
                            for row in raw
                            if row["implementation"] == label
                            for sample in row["measurement"]["rows"]
                        ]
                        for label in ("PEFT", "fast")
                    }
                    medians = {label: statistics.median(times) for label, times in values.items()}
                    record[phase] = {**medians, "PEFT_over_fast": medians["PEFT"] / medians["fast"]}
                record["peak_allocated_bytes"] = {
                    label: max(
                        row["measurement"]["peak_bytes"]["allocated"]
                        for row in raw
                        if row["implementation"] == label
                    )
                    for label in ("PEFT", "fast")
                }
            cycles.append(record)
        accepted = [row for row in cycles if row["accepted"]]
        result["cases"][name] = {
            "cycles": cycles,
            "accepted_cycles": len(accepted),
            "complete_five_cycles": len(accepted) == 5,
        }
        if accepted:
            result["cases"][name]["median_ratios"] = {
                phase: statistics.median(row[phase]["PEFT_over_fast"] for row in accepted)
                for phase in ("forward_ms", "backward_ms", "total_ms")
            }
    result["verdict"] = (
        "VALID_SCOPED_MEASUREMENTS"
        if len(result["cases"]) == 12
        and all(case["complete_five_cycles"] for case in result["cases"].values())
        and not result["numerical_failures"]
        else "NO VERDICT / INCOMPLETE MATRIX"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    Console().print(
        result["verdict"],
        result["numerical_comparisons"],
        "comparisons; failures:",
        len(result["numerical_failures"]),
    )
    return int(bool(result["numerical_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
