"""CPU replay of retained full-step baseline tensors and GPU lifecycle records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rich.console import Console


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def adapter_schema() -> set[str]:
    return {
        f"base_model.model.model.layers.{layer}.{block}.{projection}.lora_{matrix}.default.weight"
        for layer in range(22)
        for block, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
        for matrix in ("A", "B")
    }


def clock_observations(rows: list[dict]) -> bool:
    values = [row.get("clock_mhz") for row in rows]
    return (
        len(values) >= 3
        and all(
            isinstance(value, (float, int)) and math.isfinite(value) and value > 0
            for value in values
        )
    )


def clocks(rows: list[dict]) -> bool:
    return clock_observations(rows) and (
        max(row["clock_mhz"] for row in rows) / min(row["clock_mhz"] for row in rows) <= 1.05
    )


def replay(reference: Path, candidate: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    left, right = read(reference / "result.json"), read(candidate / "result.json")
    ready_left, ready_right = read(reference / "ready.json"), read(candidate / "ready.json")
    failures = []
    if left["steps"] != 50 or right["steps"] != 50:
        failures.append("wrong step count")
    if ready_left["fixture_sha256"] != ready_right["fixture_sha256"]:
        failures.append("different fixtures")
    if ready_left["harness_sha256"] != ready_right["harness_sha256"]:
        failures.append("different harness source")
    for field in ("deterministic_algorithms", "cublas_workspace_config"):
        left_execution, right_execution = ready_left.get("execution", {}), ready_right.get(
            "execution", {}
        )
        if field not in left_execution or field not in right_execution:
            failures.append("missing execution setting: " + field)
        elif left_execution[field] != right_execution[field]:
            failures.append("different execution setting: " + field)
    adapter_names = ready_left["adapter_keys"]
    if set(adapter_names) != adapter_schema() or adapter_names != ready_right["adapter_keys"]:
        failures.append("incomplete/different adapter list")
    for package in ("torch", "transformers", "peft", "triton", "safetensors"):
        if (
            ready_left["environment"]["packages"][package]
            != ready_right["environment"]["packages"][package]
        ):
            failures.append("different version: " + package)
    required = {
        f"step-{step}/{kind}/{name}"
        for step in (1, 50)
        for kind in ("weight", "gradient")
        for name in adapter_names
    }
    expected, actual = [
        load_file(str(folder / "snapshots.safetensors")) for folder in (reference, candidate)
    ]
    if set(expected) != required or set(actual) != required:
        failures.append("missing/extra tensor quantities")
    checks = []
    for key in sorted(required & set(expected) & set(actual)):
        value, other = expected[key], actual[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            failures.append(key + ": different shape/dtype")
            continue
        if not torch.isfinite(value).all().item() or not torch.isfinite(other).all().item():
            failures.append(key + ": nonfinite")
            continue
        error = (value.double() - other.double()).norm().item()
        denominator = max(value.double().norm().item(), 1e-12)
        relative = error / denominator
        passed = relative <= 0.01
        checks.append(
            {
                "quantity": key,
                "relative_l2": relative,
                "passed": passed,
                "exact": torch.equal(value, other),
            }
        )
        if not passed:
            failures.append(key)
    mismatch = []
    if len(left["rows"]) != 50 or len(right["rows"]) != 50:
        failures.append("missing loss rows")
    for index, (a, b) in enumerate(zip(left["rows"], right["rows"]), 1):
        if a["step"] != index or b["step"] != index:
            failures.append("invalid loss step sequence")
        if not all(math.isfinite(row["loss"]) for row in (a, b)):
            failures.append("nonfinite loss")
        valid = (
            a["step"] == b["step"] == index
            and all(math.isfinite(row["loss"]) for row in (a, b))
            and f"{a['loss']:.3f}" == f"{b['loss']:.3f}"
        )
        if not valid:
            mismatch.append({"step": index, "peft_loss": a["loss"], "candidate_loss": b["loss"]})
    if mismatch:
        failures.append("loss .3f mismatch")
    for folder in (reference, candidate):
        if read(folder / "validation.json").get("base_unchanged") is not True:
            failures.append("frozen base not verified")
    return {
        "passed": not failures,
        "failures": failures,
        "tensor_checks": checks,
        "loss_mismatches": mismatch,
    }


def lifecycle(telemetry: dict, ready: dict, exit_code: int, *, stable_clocks: bool = True) -> dict:
    before, birth, live, after = [
        telemetry[key] for key in ("before", "context_birth", "live", "after")
    ]
    failures = []
    identities = {tuple(row.get("processes", [])) for row in live}
    owner = (
        next(iter(identities))[0]
        if len(identities) == 1 and len(next(iter(identities))) == 1
        else None
    )
    if not before or not birth or not live or not after or owner is None or exit_code != 0:
        failures.append("incomplete lifecycle/owner/exit")
    if any(row.get("processes") for row in before + after):
        failures.append("GPU occupied before/after")
    if any(
        not row.get("available") or row.get("graphics") for row in before + birth + live + after
    ):
        failures.append("missing telemetry/graphics peer")
    if any(row.get("processes") not in ([], [owner]) for row in birth):
        failures.append("context birth peer")
    normalize = lambda value: str(value).lower().removeprefix("gpu-")  # noqa: E731
    if any(normalize(row.get("uuid")) != normalize(ready.get("gpu_uuid")) for row in live):
        failures.append("wrong GPU UUID")
    if not (clocks(live) if stable_clocks else clock_observations(live)):
        failures.append("clock drift/missing clocks")
    if any(row.get("pstate") not in (0, 2) or row.get("throttle", 0) & ~0x5 for row in live):
        failures.append("power state/disallowed throttle")
    if any(
        not isinstance(row.get("power_mw"), (float, int))
        or not math.isfinite(row["power_mw"])
        or row["power_mw"] <= 0
        for row in live
    ):
        failures.append("missing power")
    return {"passed": not failures, "failures": failures,
            "strict_clock_diagnostic": clocks(live), "stable_clock_gate_applied": stable_clocks}


def audit(root: Path) -> dict:
    import torch

    torch.set_num_threads(1)
    output = {
        "scope": "TinyLlama 1.1B dense fixed-token complete LoRA step",
        "gates": {},
        "arms": {},
        "complete": False,
    }
    for dtype in ("float16", "bfloat16"):
        reference = root / f"{dtype}-gate-peft"
        for backend in ("peft", "soup", "liger", "unsloth"):
            folder = root / f"{dtype}-gate-{backend}"
            if (
                not (reference / "snapshots.safetensors").exists()
                or not (folder / "snapshots.safetensors").exists()
            ):
                output["gates"][f"{dtype}/{backend}"] = {"passed": False, "missing": True}
            else:
                output["gates"][f"{dtype}/{backend}"] = replay(reference, folder)
        for candidate in ("soup", "liger", "unsloth"):
            for cycle in range(3):
                cycle_clocks = []
                for position, backend in enumerate(("peft", candidate, candidate, "peft")):
                    label = f"{dtype}-{candidate}-{cycle}-{position}-{backend}"
                    folder = root / label
                    if not (folder / "snapshots.safetensors").exists():
                        output["arms"][label] = {"passed": False, "missing": True}
                        continue
                    checks = replay(reference, folder)
                    telemetry = read(root / f"{label}-telemetry.json")
                    checks["lifecycle"] = lifecycle(
                        telemetry,
                        read(folder / "ready.json"),
                        read(root / f"{label}-process.json")["exit_code"],
                    )
                    cycle_clocks.extend(telemetry["live"])
                    checks["passed"] &= checks["lifecycle"]["passed"]
                    output["arms"][label] = checks
                if cycle_clocks and not clocks(cycle_clocks):
                    for position, backend in enumerate(("peft", candidate, candidate, "peft")):
                        label = f"{dtype}-{candidate}-{cycle}-{position}-{backend}"
                        output["arms"][label]["passed"] = False
                        output["arms"][label]["cycle_clock_failure"] = True
    output["complete"] = all(row["passed"] for row in output["gates"].values()) and all(
        row["passed"] for row in output["arms"].values()
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print(
        {"complete": result["complete"], "gates": len(result["gates"]), "arms": len(result["arms"])}
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
