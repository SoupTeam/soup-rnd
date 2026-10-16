"""Replay public stack-v3 timing records; omitted tensors require the full raw bundle."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from rich.console import Console

try:
    from . import d2_audit_training as audit
except ImportError:
    import d2_audit_training as audit


def replay(root: Path) -> dict:
    arguments = audit.read(root / "arguments.json")
    if not (
        arguments.get("default_boost") is True
        and arguments.get("stack_comparison") is True
        and arguments.get("external_only") is True
        and arguments.get("steps") == 50
        and arguments.get("cycles") == 3
        and arguments.get("warmup", 0) >= 100
        and arguments.get("deterministic") is True
        and arguments.get("token_sum_normalization") is True
    ):
        raise ValueError("Different experiment declaration")
    output = {"timing_records_valid": False, "ratios": {}, "arms": 0,
              "clock_policy": "observed default GPU Boost",
              "tensor_replay": "not performed; requires complete raw snapshots"}
    sources, fixtures = set(), {}
    for dtype in ("float16", "bfloat16"):
        for candidate in ("liger", "unsloth"):
            ratios, cycle_clocks = [], []
            for cycle in range(3):
                durations = {"soup": [], candidate: []}
                frequencies = []
                for position, backend in enumerate(("soup", candidate, candidate, "soup")):
                    label = f"{dtype}-{candidate}-{cycle}-{position}-{backend}"
                    folder = root / label
                    ready, result = [audit.read(folder / name) for name in (
                        "ready.json", "result.json")]
                    gate = root / f"{dtype}-gate-{backend}"
                    gate_ready, gate_result = [audit.read(gate / name) for name in (
                        "ready.json", "result.json")]
                    for key in ("fixture_sha256", "harness_sha256", "adapter_keys", "execution"):
                        if ready[key] != gate_ready[key]:
                            raise ValueError(label + ": different gate settings: " + key)
                    if ready["environment"]["packages"] != gate_ready["environment"]["packages"]:
                        raise ValueError(label + ": different packages")
                    if set(ready["adapter_keys"]) != audit.adapter_schema():
                        raise ValueError(label + ": incomplete adapter schema")
                    if ready["warmup_iterations"] < 100 or result["phase"] != "timing":
                        raise ValueError(label + ": different phase/warmup")
                    if (result["backend"], result["dtype"], result["steps"],
                        result["input_tokens_per_step"], result["predicted_tokens_per_step"]) != (
                        backend, dtype, 50, 512, 511
                    ):
                        raise ValueError(label + ": different workload")
                    if audit.read(folder / "validation.json").get("base_unchanged") is not True:
                        raise ValueError(label + ": missing base validation")
                    execution = ready["execution"]
                    if not execution.get("deterministic_algorithms") or (
                        execution.get("loss_normalization") != "token-sum"
                        or execution.get("cublas_workspace_config") != ":4096:8"
                    ):
                        raise ValueError(label + ": wrong execution")
                    events = execution["autograd_events"]
                    if backend == "liger" and not all(
                        any(marker in event for event in events)
                        for marker in ("LigerFusedLinearCrossEntropy", "LigerRMSNorm",
                                       "LigerRope", "LigerSiLUMul")
                    ):
                        raise ValueError(label + ": missing native Liger paths")
                    if backend == "unsloth" and not all(
                        any(marker in event for event in events)
                        for marker in ("LoRA_MLP", "LoRA_QKV", "LoRA_W")
                    ):
                        raise ValueError(label + ": missing native Unsloth paths")
                    if backend == "soup":
                        for path, count, grad_fn in (
                            ("single", 22, "_FastLoraSingleProjectionBackward"),
                            ("qkv", 66, "_FastLoraQKVBackward"),
                            ("mlp", 22, "_FastLoraSwiGLUBackward"),
                        ):
                            observed = execution["observed_custom_paths"].get(path, {})
                            if len(observed) != count or set(observed.values()) != {grad_fn}:
                                raise ValueError(label + ": missing Soup paths")
                    if backend == "unsloth" and not execution.get("fused_loss_scaler_attached"):
                        raise ValueError(label + ": missing scaler hook")
                    if len(result["rows"]) != 50 or len(gate_result["rows"]) != 50:
                        raise ValueError(label + ": missing steps")
                    for index, (row, reference) in enumerate(
                        zip(result["rows"], gate_result["rows"]), 1
                    ):
                        if row["step"] != index or reference["step"] != index:
                            raise ValueError(label + ": wrong step sequence")
                        if not all(math.isfinite(r["loss"]) for r in (row, reference)) or (
                            f"{row['loss']:.3f}" != f"{reference['loss']:.3f}"
                        ):
                            raise ValueError(label + ": loss repeatability failed")
                        if not all(math.isfinite(row[key]) and row[key] > 0
                                   for key in ("wall_ms", "cuda_ms")):
                            raise ValueError(label + ": invalid timing")
                    telemetry = audit.read(root / f"{label}-telemetry.json")
                    lifecycle = audit.lifecycle(
                        telemetry, ready, audit.read(root / f"{label}-process.json")["exit_code"],
                        stable_clocks=False)
                    if not lifecycle["passed"]:
                        raise ValueError(label + ": " + str(lifecycle["failures"]))
                    sources.add(ready["harness_sha256"])
                    fixtures.setdefault(dtype, set()).add(ready["fixture_sha256"])
                    durations[backend].append(sum(row["wall_ms"] for row in result["rows"]))
                    frequencies.extend(row["clock_mhz"] for row in telemetry["live"])
                    output["arms"] += 1
                ratios.append(statistics.mean(durations[candidate]) /
                              statistics.mean(durations["soup"]))
                cycle_clocks.append([min(frequencies), max(frequencies)])
            output["ratios"][f"{dtype}/{candidate}"] = {
                "candidate_over_soup": ratios, "median": statistics.median(ratios),
                "range": [min(ratios), max(ratios)], "cycle_clock_range_mhz": cycle_clocks}
    if len(sources) != 1 or any(len(values) != 1 for values in fixtures.values()):
        raise ValueError("Mixed sources/fixtures")
    output["timing_records_valid"] = True
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    Console().print(replay(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
