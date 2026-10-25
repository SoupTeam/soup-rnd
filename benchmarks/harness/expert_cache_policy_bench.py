"""Accuracy-first three-arm follow-up for the experimental static expert hotset.

Prepare a local Moonlight checkpoint with ``two_read_cache.py prepare`` using
2 GiB host / .25 GiB device ceilings. This runner reuses its frozen inputs and
native worker. It never rents hardware, downloads data or changes system limits.
The prepared spec retains the four-arm runner's valid rounds field; this
follow-up separately schedules all six three-arm permutations. Explicit
``--large-device-cache-gib 1.5`` adds a fourth static arm with a larger GPU
ceiling and uses four balanced Latin-order rounds instead. This ceiling still
must pass the unchanged native whole-model admission probes.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.harness import two_read_cache as native  # noqa: E402
from benchmarks.harness.two_read_cache_common import (  # noqa: E402
    atomic_json,
    compare_captures,
    sha256_file,
    validate_capture,
)

ARMS = ("two_read", "two_read_lru", "two_read_static")
ORDERS = tuple(itertools.permutations(ARMS))
LARGE_ARM = "two_read_static_large_gpu"
GIB = 2**30


def protocol(large_device_cache_gib: float | None = None) -> tuple[tuple, tuple]:
    if large_device_cache_gib is None:
        return ARMS, ORDERS
    if type(large_device_cache_gib) not in (int, float) or large_device_cache_gib != 1.5:
        raise ValueError("the optional large-device cohort fixes an explicit 1.5 GiB ceiling")
    arms = (*ARMS, LARGE_ARM)
    return arms, tuple(arms[shift:] + arms[:shift] for shift in range(4))


def arm_spec(
    base: dict[str, Any], arm: str, large_device_cache_gib: float | None = None,
) -> tuple[dict[str, Any], str]:
    """Vary host policy, or explicitly the fourth arm's GPU ceiling.

    Cache disablement is the native worker arm. The baseline input spec's
    matched 2/.25 GiB ceilings are required even for the optional large arm.
    """
    if arm not in protocol(large_device_cache_gib)[0]:
        raise ValueError("unknown follow-up arm")
    native.validate_spec(base)
    if base["host_cache_bytes"] != 2 * GIB or base["device_cache_bytes"] != GIB // 4:
        raise ValueError("the follow-up fixes matched 2 GiB RAM / .25 GiB GPU cache ceilings")
    result = copy.deepcopy(base)
    result["host_cache_policy"] = "static" if arm.startswith("two_read_static") else "lru"
    if arm == LARGE_ARM:
        result["device_cache_bytes"] = int(large_device_cache_gib * GIB)
    return result, "two_read" if arm == "two_read" else "two_read_cache"


def cache_behavior(
    accuracy: dict[str, Any], large_device_cache_gib: float | None = None,
) -> dict[str, Any]:
    deltas = {}
    for arm in protocol(large_device_cache_gib)[0]:
        stage = accuracy[arm]["worker"]["accuracy"]
        before = stage["runtime_before"]["expert_cache"]
        after = stage["runtime_after"]["expert_cache"]
        row = {}
        for key in ("logical_fetches", "storage_reads", "storage_bytes_read", "materializations"):
            row[key] = after[key] - before[key]
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"{arm}: invalid expert counter delta {key}")
        for tier in ("host", "device"):
            row[tier + "_hits"] = after[tier]["hits"] - before[tier]["hits"]
            if type(row[tier + "_hits"]) is not int or row[tier + "_hits"] < 0:
                raise ValueError(f"{arm}: invalid {tier} hit counter")
            ceiling = 0 if arm == "two_read" else (2 * GIB if tier == "host" else GIB // 4)
            if tier == "device" and arm == LARGE_ARM:
                ceiling = int(large_device_cache_gib * GIB)
            for snapshot in (before, after):
                if (snapshot[tier]["budget_bytes"] != ceiling or
                        not 0 <= snapshot[tier]["payload_bytes"] <= ceiling):
                    raise ValueError(f"{arm}: {tier} payload or ceiling differs from the protocol")
        if (not 0 < row["storage_reads"] <= row["logical_fetches"] or
                row["storage_bytes_read"] <= 0 or
                row["storage_reads"] + row["host_hits"] + row["device_hits"] !=
                row["logical_fetches"] or
                row["materializations"] != row["logical_fetches"] - row["device_hits"]):
            raise ValueError(f"{arm}: expert visits, hits, reads and materializations disagree")
        if arm == "two_read":
            if row["host_hits"] or row["device_hits"]:
                raise ValueError("uncached control unexpectedly used a cache")
        else:
            policy = "static" if arm.startswith("two_read_static") else "lru"
            if (before.get("host_cache_policy") != policy or
                    after.get("host_cache_policy") != policy):
                raise ValueError(f"{arm}: worker did not execute the requested host policy")
            if min(row["host_hits"], row["device_hits"]) <= 0:
                raise ValueError(f"{arm}: both cache tiers must actually be exercised")
            if policy == "static" and after["host"]["evictions"] != before["host"]["evictions"]:
                raise ValueError("static hotset unexpectedly evicted an admitted expert")
        deltas[arm] = row
    if len({row["logical_fetches"] for row in deltas.values()}) != 1:
        raise ValueError("the three two-read arms followed different logical expert visits")
    return {"expert_counter_deltas": deltas,
            "storage_bytes_meaning": "expert file payload, not physical SSD traffic",
            "cache_bytes_meaning": "owned tensor payload, not total RSS/VRAM"}


def timing_summary(
    blocks: list[dict[str, Any]], rounds: int | None = None,
    large_device_cache_gib: float | None = None,
) -> dict[str, Any]:
    arms, orders = protocol(large_device_cache_gib)
    rounds = len(orders) if rounds is None else rounds
    if type(rounds) is not int or rounds < len(orders) or rounds % len(orders):
        raise ValueError("rounds must be a positive multiple of the complete balanced order")
    by_round: dict[int, dict[str, float]] = {}
    for block in blocks:
        number, arm, result = block["round"], block["arm"], block["result"]
        if (type(number) is not int or not 0 <= number < rounds or arm not in arms
                or block.get("position") != orders[number % len(orders)].index(arm)):
            raise ValueError("timing block differs from the balanced schedule")
        if result.get("status") != "passed":
            raise ValueError("a timing worker did not complete successfully")
        times = result["timing"]["update_seconds"]
        if len(times) < 6 or any(type(value) not in (int, float) or
                                 not math.isfinite(value) or value <= 0 for value in times):
            raise ValueError("each timing block needs >=6 finite, positive full optimizer updates")
        row = by_round.setdefault(number, {})
        if arm in row:
            raise ValueError("duplicate arm in a timing round")
        row[arm] = statistics.median(times)
    if (set(by_round) != set(range(rounds)) or
            any(set(row) != set(arms) for row in by_round.values())):
        raise ValueError("incomplete balanced timing matrix")

    def compare(arm: str, reference: str) -> dict[str, Any]:
        logs = [math.log(by_round[number][arm] / by_round[number][reference])
                for number in range(rounds)]
        rng = random.Random(29)
        distribution = sorted(
            100 * (1 - math.exp(statistics.mean(rng.choices(logs, k=rounds))))
            for _ in range(5000)
        )
        lower, upper = distribution[124], distribution[4874]
        return {"reference": reference, "paired_rounds": rounds,
                "step_time_reduction_percent": 100 * (1 - math.exp(statistics.mean(logs))),
                "paired_bootstrap_95_percent": [lower, upper],
                "passes_practical_2_percent_gate": lower >= 2.0}

    absolute = {}
    for arm in arms:
        results = [block["result"] for block in blocks if block["arm"] == arm]
        tokens = {result["timing"].get("tokens_per_update") for result in results}
        if len(tokens) != 1 or type(next(iter(tokens))) is not int or next(iter(tokens)) < 1:
            raise ValueError("tokens per update must be a stable positive integer")
        median = statistics.median([value for result in results
                                    for value in result["timing"]["update_seconds"]])

        def maximum(section: str, key: str) -> int | None:
            values = [result["timing"].get(section, {}).get(key) for result in results]
            valid = all(type(value) is int and value >= 0 for value in values)
            return max(values) if valid else None

        payloads = {}
        for tier in ("host", "device"):
            values = [result["timing"].get("runtime_after", {}).get("expert_cache", {}).get(
                tier, {},
            ).get("payload_bytes") for result in results]
            payloads[tier] = max(values) if all(type(value) is int for value in values) else None
        absolute[arm] = {
            "median_update_seconds": median, "tokens_per_second": next(iter(tokens)) / median,
            "max_step_allocated_bytes": maximum("cuda", "peak_allocated_bytes"),
            "max_step_reserved_bytes": maximum("cuda", "peak_reserved_bytes"),
            "max_sampled_rss_bytes": maximum("resources", "sampled_peak_rss_bytes"),
            "max_observed_cache_payload_bytes": payloads,
            "setup_seconds": [result.get("setup", {}).get("seconds") for result in results],
            "entry_wall_seconds": [result.get("fresh_process_wall_seconds") for result in results],
        }
    result = {
        "method": "paired round-median geometric ratios; 5000 round bootstrap draws, seed29",
        "comparisons": {arm + "_vs_two_read": compare(arm, "two_read") for arm in arms[1:]},
        "static_vs_lru": compare("two_read_static", "two_read_lru"), "arms": absolute,
        "caution": "Fixed-host numerical/throughput experiment, not task quality or full training. "
        "Observed cache payload is distinct from whole-process RSS and allocator VRAM.",
    }
    if large_device_cache_gib is not None:
        result["large_device_vs_static"] = compare(LARGE_ARM, "two_read_static")
    return result


def run(
    spec_path: Path, out: Path, *, rounds: int | None = None, accuracy_only: bool = False,
    large_device_cache_gib: float | None = None,
) -> dict[str, Any]:
    import torch

    arms, orders = protocol(large_device_cache_gib)
    rounds = len(orders) if rounds is None else rounds
    if type(rounds) is not int or rounds < len(orders) or rounds % len(orders):
        raise ValueError("follow-up rounds must preserve the complete balanced schedule")
    spec_path, out = Path(os.path.realpath(spec_path)), Path(os.path.realpath(out))
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a new/empty output directory; partial runs remain retained")
    out.mkdir(parents=True, exist_ok=True)
    original_hash, controller_hash = sha256_file(spec_path), sha256_file(__file__)
    report: dict[str, Any] = {
        "status": "running", "base_spec_sha256": original_hash,
        "controller_sha256": controller_hash, "rounds": rounds,
        "orders": orders, "arms": arms, "large_device_cache_gib": large_device_cache_gib,
        "accuracy": {}, "timing_blocks": [],
        "gpu_rented_by_runner": False, "arm_specs": {},
    }
    started = time.perf_counter()
    try:
        base = json.loads(spec_path.read_text(encoding="utf-8"))
        native.verify_spec(base, content=True)
        specs = {}
        for arm in arms:
            child, worker_arm = arm_spec(base, arm, large_device_cache_gib)
            child_path = out / "specs" / (arm + ".json")
            atomic_json(child_path, child)
            specs[arm] = child_path, child, worker_arm
            report["arm_specs"][arm] = {"sha256": sha256_file(child_path),
                                        "worker_arm": worker_arm,
                                        "host_cache_policy": child["host_cache_policy"],
                                        "effective_host_payload_ceiling_bytes": (
                                            0 if arm == "two_read" else child["host_cache_bytes"]),
                                        "effective_device_payload_ceiling_bytes": (
                                            0 if arm == "two_read"
                                            else child["device_cache_bytes"])}

        def launch(arm: str, phase: str, directory: Path) -> dict[str, Any]:
            child_path, child, worker_arm = specs[arm]
            if (sha256_file(spec_path) != original_hash or sha256_file(__file__) != controller_hash
                    or sha256_file(child_path) != report["arm_specs"][arm]["sha256"]):
                raise ValueError("frozen controller/base/arm spec changed during the run")
            result = native.launch_worker(child_path, child, worker_arm, phase, directory)
            if (sha256_file(spec_path) != original_hash or sha256_file(__file__) != controller_hash
                    or sha256_file(child_path) != report["arm_specs"][arm]["sha256"]):
                raise ValueError("frozen controller/base/arm spec changed during a worker")
            if (result.get("arm") != worker_arm or result.get("phase") != phase
                    or result.get("spec_sha256") != report["arm_specs"][arm]["sha256"]):
                raise ValueError("worker metadata differs from its frozen arm specification")
            plan = result.get("setup", {}).get("cache_plan")
            if arm == "two_read":
                if plan is not None:
                    raise ValueError("uncached worker unexpectedly admitted a cache plan")
            elif not isinstance(plan, dict) or any(plan.get(key) != child[key] for key in (
                "host_cache_policy", "host_cache_bytes", "device_cache_bytes",
            )):
                raise ValueError("worker cache admission did not use the requested policy/ceilings")
            return result

        for arm in arms:
            result = launch(arm, "accuracy", out / "accuracy" / arm)
            report["accuracy"][arm] = {"worker": result}
            atomic_json(out / "run.json", report)
        reference = None
        for arm in arms:
            path = out / "accuracy" / arm / "capture.pt"
            identity = sha256_file(path)
            if identity != report["accuracy"][arm]["worker"]["accuracy"]["capture_sha256"]:
                raise ValueError(f"{arm}: capture does not match its worker evidence")
            capture = torch.load(path, map_location="cpu", weights_only=True)
            expected_microbatches = base["accuracy_updates"] * base["accumulation"]
            if (len(capture["updates"]) != base["accuracy_updates"] or
                    len(capture["microbatches"]) != expected_microbatches):
                raise ValueError(f"{arm}: capture omitted required accumulated updates")
            if arm == "two_read":
                reference = capture
                gate = validate_capture(reference)
            else:
                gate = compare_captures(reference, capture)
            report["accuracy"][arm].update(gate=gate, capture_sha256=identity)
            if arm != "two_read":
                del capture
        del reference
        atomic_json(out / "run.json", report)
        if not all(report["accuracy"][arm]["gate"]["passed"] for arm in arms):
            raise ValueError("accuracy failed; timing was not started")
        report["cache_behavior"] = cache_behavior(report["accuracy"], large_device_cache_gib)
        report["accuracy_passed"] = True
        native.verify_spec(base, content=True)
        if not accuracy_only:
            for number in range(rounds):
                for position, arm in enumerate(orders[number % len(orders)]):
                    Console().print(f"Policy timing round {number + 1}/{rounds}: {arm}")
                    result = launch(arm, "timing", out / "timing" / f"round-{number:02d}" / arm)
                    if len(result["timing"]["update_seconds"]) != base["timed_updates"]:
                        raise ValueError("timing worker omitted required updates")
                    report["timing_blocks"].append({
                        "round": number, "position": position, "arm": arm, "result": result,
                    })
                    atomic_json(out / "run.json", report)
            native.verify_spec(base, content=True)
            report["timing_summary"] = timing_summary(
                report["timing_blocks"], rounds, large_device_cache_gib,
            )
        report["status"] = "accuracy_passed" if accuracy_only else "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        atomic_json(out / "run.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rounds", type=int, help="default: 6; optional large-device cohort: 4")
    parser.add_argument("--large-device-cache-gib", type=float, choices=(1.5,))
    parser.add_argument("--accuracy-only", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        result = run(args.spec, args.out, rounds=args.rounds, accuracy_only=args.accuracy_only,
                     large_device_cache_gib=args.large_device_cache_gib)
    except Exception as exc:
        Console(stderr=True).print(f"Policy qualification failed: {exc}")
        return 1
    Console().print({"status": result["status"], "out": str(args.out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
