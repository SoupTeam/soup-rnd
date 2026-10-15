#!/usr/bin/env python3
"""E2 A/B performance orchestrator; retains raw records, never issues PASS.

Place in benchmarks/e2/ of the soup-rnd repository.
By default, dry-run only; --execute launches real model training.

This harness does NOT independently time cold-cache construction or full epoch
wall-clock. BenchCollector's epoch_timing is the sum of measured step durations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=25)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}


def machine_state() -> dict:
    info = {
        "utc": now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "gpu": capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,"
                "temperature.gpu,clocks.sm,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "gpu_identity": capture(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_smi": capture(["nvidia-smi"]),
    }
    if hasattr(os, "sysconf"):
        try:
            info["physical_ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            info["physical_ram_bytes"] = None
    return info


def save_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def config_and_lines(root: Path, baseline: Path, cached: Path) -> tuple[dict, dict, Path, int]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required: run python -m pip install pyyaml inside the benchmark environment"
        ) from exc

    def read(path: Path) -> dict:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Invalid YAML document: {path}")
        return config

    baseline_config, cached_config = read(baseline), read(cached)
    if baseline_config.get("training", {}).get("frozen_prefix_cache") is not False:
        raise ValueError("A must explicitly disable frozen_prefix_cache")
    if cached_config.get("training", {}).get("frozen_prefix_cache") is not True:
        raise ValueError("B must explicitly enable frozen_prefix_cache")

    left, right = copy.deepcopy(baseline_config), copy.deepcopy(cached_config)
    left.pop("output", None)
    right.pop("output", None)
    left["training"].pop("frozen_prefix_cache")
    right["training"].pop("frozen_prefix_cache")
    if left != right:
        raise ValueError("Baseline and cached YAML differ in more than output/frozen_prefix_cache")

    training = baseline_config["training"]
    dataset_config = baseline_config["data"]
    required = (
        ("epochs", 2),
        ("batch_size", 1),
        ("gradient_accumulation_steps", 1),
    )
    for key, expected in required:
        if training.get(key) != expected:
            raise ValueError(f"Expected training.{key}={expected}; got {training.get(key)!r}")
    for key in ("packing", "multipack"):
        if training.get(key, False):
            raise ValueError(f"training.{key} must be false for fixed epoch boundaries")
    if dataset_config.get("val_split") != 0.0:
        raise ValueError("Use train-only benchmark data with data.val_split=0")

    train_path = Path(dataset_config["train"])
    if not train_path.is_absolute():
        train_path = root / train_path
    if not train_path.is_file():
        raise ValueError(f"Training data missing: {train_path}")
    with train_path.open(encoding="utf-8") as stream:
        examples = sum(bool(line.strip()) for line in stream)
    if not examples:
        raise ValueError("Training dataset is empty")
    return baseline_config, cached_config, train_path, examples


def validate_report(report: dict, steps: int, mode: str, examples: int) -> None:
    if report.get("valid") is not True:
        raise ValueError(f"Invalid benchmark report: {report.get('failures')}")
    if report.get("steps_measured") != steps:
        raise ValueError(
            f"Unexpected steps_measured={report.get('steps_measured')}; expected {steps}"
        )
    epochs = report.get("epoch_timing")
    if not isinstance(epochs, list) or len(epochs) != 2:
        count = len(epochs) if isinstance(epochs, list) else None
        raise ValueError(f"Expected exactly 2 epoch records; got {count}")
    for index, epoch in enumerate(epochs, start=1):
        measured = epoch.get("end_step", 0) - epoch.get("start_step", 0)
        if epoch.get("epoch") != index or measured != examples:
            raise ValueError(f"Incorrect epoch {index} boundaries: {epoch}")
        if epoch.get("useful_tokens", 0) <= 0:
            raise ValueError(f"Epoch {index} has no supervised tokens")
    if mode == "B":
        if not (epochs[0].get("cache_misses") or 0) > 0:
            raise ValueError("No cold-cache misses observed in epoch 1")
        if not (epochs[1].get("cache_hits") or 0) > 0:
            raise ValueError("No warm-cache hits observed in epoch 2")


def validate_precision_provenance(
    report: dict, expected_mode: str
) -> dict:
    """Fail closed when actual training precision differs from protocol."""
    expected_flags = {
        "bf16": (True, False),
        "fp16": (False, True),
    }

    if expected_mode not in expected_flags:
        raise ValueError(f"Unsupported precision: {expected_mode!r}")

    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Missing precision provenance")

    expected_dtype = {
        "bf16": "torch.bfloat16",
        "fp16": "torch.float16",
    }[expected_mode]

    actual_dtype = provenance.get("dtype")
    if actual_dtype != expected_dtype:
        raise ValueError(
            f"Precision dtype mismatch: "
            f"{actual_dtype!r} != {expected_dtype!r}"
        )

    flags = provenance.get("training_precision")
    if not isinstance(flags, dict):
        raise ValueError("Missing actual training precision flags")

    bf16 = flags.get("bf16")
    fp16 = flags.get("fp16")

    if type(bf16) is not bool or type(fp16) is not bool:
        raise ValueError("Training precision flags must be booleans")

    if (bf16, fp16) != expected_flags[expected_mode]:
        raise ValueError(
            f"Training precision mismatch: "
            f"bf16={bf16}, fp16={fp16}, "
            f"expected={expected_mode}"
        )

    return {
        "mode": expected_mode,
        "reported_dtype": actual_dtype,
        "training_precision": {"bf16": bf16, "fp16": fp16},
    }


def _median_epoch(records: list[dict], index: int) -> float:
    return statistics.median(
        item["report"]["epoch_timing"][index]["duration_seconds"] for item in records
    )


def _median_command(records: list[dict]) -> float:
    return statistics.median(item["elapsed_train_command_seconds"] for item in records)


def summary(results: list[dict]) -> dict:
    """Aggregate the APPROVED rule: median(A/B) within each cycle, then median."""
    if len(results) != 12:
        raise ValueError("Three complete ABBA cycles (12 runs) required")
    if [item["mode"] for item in results] != ["A", "B", "B", "A"] * 3:
        raise ValueError("Unexpected ABBA order")

    cycle_rows = []
    for cycle in range(3):
        group = results[4 * cycle : 4 * (cycle + 1)]
        a_rows = [item for item in group if item["mode"] == "A"]
        b_rows = [item for item in group if item["mode"] == "B"]
        a_first, b_first = _median_epoch(a_rows, 0), _median_epoch(b_rows, 0)
        a_second, b_second = _median_epoch(a_rows, 1), _median_epoch(b_rows, 1)
        a_command, b_command = _median_command(a_rows), _median_command(b_rows)
        if b_second <= 0 or b_command <= 0:
            raise ValueError(f"Cycle {cycle + 1}: nonpositive measured time")

        # The first-epoch difference is NOT an independent cache-build timer.
        # It includes other effects of running E2 during the first epoch.
        cold_overhead_proxy = b_first - a_first
        warm_saving = a_second - b_second
        break_even = max(0.0, cold_overhead_proxy) / warm_saving if warm_saving > 0 else None
        cycle_rows.append(
            {
                "cycle": cycle + 1,
                "A_epoch1_step_sum_median_s": a_first,
                "B_epoch1_step_sum_median_s": b_first,
                "A_epoch2_step_sum_median_s": a_second,
                "B_epoch2_step_sum_median_s": b_second,
                "warm_epoch_ratio_A_over_B": a_second / b_second,
                "A_command_elapsed_median_s": a_command,
                "B_command_elapsed_median_s": b_command,
                "end_to_end_ratio_A_over_B": a_command / b_command,
                "end_to_end_saved_seconds_A_minus_B": a_command - b_command,
                "cold_epoch_extra_step_time_proxy_seconds_B_minus_A": (cold_overhead_proxy),
                "warm_epoch_saved_step_time_seconds_A_minus_B": warm_saving,
                "payback_warm_epochs_estimate_FROM_PROXY": break_even,
            }
        )

    ratios = [row["warm_epoch_ratio_A_over_B"] for row in cycle_rows]
    median_ratio = statistics.median(ratios)
    e2e_ratios = [row["end_to_end_ratio_A_over_B"] for row in cycle_rows]
    all_cycles_nonslow = all(value >= 1.00 for value in ratios)
    threshold_met = median_ratio >= 1.10 and all_cycles_nonslow
    return {
        "rule": (
            "For each of 3 ABBA cycles: median(A epoch2 step-sums) / "
            "median(B epoch2 step-sums). Final = median of 3 cycle ratios. "
            "PASS threshold: final >= 1.10 and every valid cycle >= 1.00."
        ),
        "cycle_metrics": cycle_rows,
        "warm_epoch_ratios": ratios,
        "median_of_three_warm_epoch_ratios": median_ratio,
        "all_cycles_at_least_1_00": all_cycles_nonslow,
        "approved_warm_epoch_threshold_met": threshold_met,
        "end_to_end_ratios": e2e_ratios,
        "median_of_three_end_to_end_ratios": statistics.median(e2e_ratios),
        "cold_cache_build_seconds": None,
        "measurement_limits": [
            "Cold-cache compute/write/load not independently instrumented.",
            "First-epoch differences are proxies, not isolated cache build cost.",
            "Epoch durations are sums of measured step wall times; "
            "the end-to-end command timer separately includes model setup.",
            "Proxy payback estimates do not establish realized amortization.",
            "Final decisions also require review of validity and GPU records.",
        ],
        "performance_verdict": "NO VERDICT pending independent cold-cache timing/review",
        "research_acceptance": "NO VERDICT: quality gate and Small-MoE pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="experiments/e2_mistral7b_smoke.yaml")
    parser.add_argument("--cached", default="experiments/e2_mistral7b_cached_smoke.yaml")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--amp-init-scale", type=float, default=None)
    parser.add_argument(
        "--precision",
        choices=("fp16", "bf16"),
        default=None,
        help="Expected CUDA precision; required for final experiments",
    )
    parser.add_argument(
        "--run-dir",
        help="New destination; defaults to timestamped benchmarks/e2/full_abba_*",
    )
    parser.add_argument(
        "--decision-rule-file",
        help="Approved predeclared rule; saved verbatim with its SHA256",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Run model training (default: dry run)"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Exploratory run: allows small dataset/fewer cycles; no final verdict",
    )
    args = parser.parse_args()
    if args.amp_init_scale is not None:
        import math

        if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
            parser.error("--amp-init-scale must be finite and >0")
    if args.execute and not args.pilot and args.precision is None:
        parser.error("Final execution requires --precision fp16 or bf16")

    if args.precision == "bf16" and args.amp_init_scale is not None:
        parser.error("BF16 must not use --amp-init-scale")

    if args.execute and args.precision == "fp16" and args.amp_init_scale is None:
        parser.error("FP16 execution requires --amp-init-scale")

    if args.execute and os.environ.get("SOUP_E2_PROFILE_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        parser.error(
            "Disable SOUP_E2_PROFILE_CACHE before running ABBA. Profiling changes benchmark timing."
        )

    root = Path(__file__).resolve().parents[2]
    baseline = root / args.baseline
    cached = root / args.cached
    for path in (baseline, cached):
        if not path.is_file():
            parser.error(f"Config missing: {path}")
    try:
        _a, _b, dataset, examples = config_and_lines(root, baseline, cached)
    except ValueError as exc:
        parser.error(str(exc))
    detected_precision = None

    if args.execute:
        from soup_cli.utils.gpu import bf16_fp16_flags

        if _a["training"].get("auto_mixed_precision", False):
            parser.error("ABBA requires auto_mixed_precision=False")

        use_bf16, use_fp16 = bf16_fp16_flags("cuda")

        if (use_bf16, use_fp16) == (True, False):
            detected_precision = "bf16"
        elif (use_bf16, use_fp16) == (False, True):
            detected_precision = "fp16"
        else:
            parser.error("Could not establish CUDA precision mode")

        if args.precision is not None and args.precision != detected_precision:
            parser.error(
                f"Requested {args.precision}, "
                f"but Soup selects {detected_precision}"
            )

        if args.amp_init_scale is not None and detected_precision != "fp16":
            parser.error("GradScaler requested for non-FP16 training")

    precision_mode = args.precision or detected_precision
    steps = examples * 2
    if args.cycles < 1 or not 0 <= args.warmup < steps:
        parser.error("Require cycles >= 1 and 0 <= warmup < 2 * dataset rows")
    if args.execute and not args.pilot and (args.cycles != 3 or examples < 128):
        parser.error("Final run requires 3 cycles and >=128 examples; use --pilot for exploration")
    if args.execute and not args.pilot and not args.decision_rule_file:
        parser.error(
            "Final run requires --decision-rule-file approved before measuring; "
            "otherwise use --pilot"
        )
    if args.execute and not args.pilot and not machine_state()["gpu_identity"]["stdout"]:
        parser.error("No NVIDIA GPU detected; refusing a final performance experiment")

    rule = None
    if args.decision_rule_file:
        rule_path = Path(args.decision_rule_file).resolve()
        if not rule_path.is_file():
            parser.error(f"Decision rule file missing: {rule_path}")
        rule = {
            "file": str(rule_path),
            "sha256": sha256(rule_path),
            "content": rule_path.read_text(encoding="utf-8"),
        }

    order = ["A", "B", "B", "A"] * args.cycles
    print(f"Repository: {root}")
    print(f"Dataset: {dataset} ({examples} nonblank rows, sha256={sha256(dataset)})")
    print(f"Expected training: 2 epochs x {examples} steps = {steps} optimizer steps")
    print(f"Run order: {'-'.join(order)}")
    print(f"Warmup steps excluded from step summary: {args.warmup}")
    print(
        "Each subprocess uses soup bench train, which creates a fresh "
        "TemporaryDirectory for its cache."
    )
    print("NOTE: epoch timing sums step times; it is not full epoch wall-clock time.")
    print("NOTE: cold-cache build time is not separately instrumented; NO VERDICT.")
    print(f"Decision rule: {rule['sha256'] if rule else 'NOT APPROVED / NOT PROVIDED'}")
    scale_label = "default" if args.amp_init_scale is None else str(args.amp_init_scale)
    print(f"AMP initial scale: {scale_label}")
    if not args.execute:
        print("DRY RUN ONLY. No GPU work performed. Use --execute to start.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else root / "benchmarks" / "e2" / f"full_abba_{stamp}"
    )
    if directory.exists():
        parser.error(f"Refusing to overwrite existing directory: {directory}")
    directory.mkdir(parents=True)
    shutil.copy2(baseline, directory / "A_config.yaml")
    shutil.copy2(cached, directory / "B_config.yaml")
    manifest = {
        "schema": "e2-abba-verified-v1",
        "created_at": now(),
        "status": "IN_PROGRESS",
        "protocol": "benchmarks/e2/experiment_protocol.md",
        "pilot": args.pilot,
        "expected_run_order": order,
        "train_examples": examples,
        "steps": steps,
        "warmup": args.warmup,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "configs": {
            "A": {"source": str(baseline), "sha256": sha256(baseline)},
            "B": {"source": str(cached), "sha256": sha256(cached)},
        },
        "decision_rule": rule,
        "runner_sha256": sha256(root / "benchmarks/e2/run_e2_abba_verified.py"),
        "amp_initial_scale_requested": args.amp_init_scale,
        "precision_contract": {
            "requested": args.precision,
            "detected": detected_precision,
            "effective": precision_mode,
            "verification": "report.provenance.dtype",
        },
        "amp_wrapper_sha256": sha256(root / "benchmarks/e2/bench_train_amp.py")
        if args.amp_init_scale is not None
        else None,
        "git": capture(["git", "rev-parse", "HEAD"]),
        "environment_start": machine_state(),
        "runs": [],
        "measurement_limits": [
            "No independent cold-cache timer: epoch 1 includes more than cache build",
            "Epoch durations sum step wall seconds, not full epoch wall time",
            "This harness never issues performance PASS/FAIL automatically",
        ],
    }
    save_json(directory / "manifest.json", manifest)

    try:
        for index, mode in enumerate(order, start=1):
            name = f"{index:02d}_{mode}"
            report_path = directory / f"{name}.json"
            log_path = directory / f"{name}.log"
            command = [sys.executable]
            if args.amp_init_scale is not None:
                command.extend(
                    [
                        str(root / "benchmarks/e2/bench_train_amp.py"),
                        "--amp-init-scale",
                        str(args.amp_init_scale),
                    ]
                )
            else:
                command.extend(["-m", "soup_cli"])
            command.extend(
                [
                    "bench",
                    "train",
                    "--config",
                    str(baseline if mode == "A" else cached),
                    "--steps",
                    str(steps),
                    "--warmup",
                    str(args.warmup),
                    "--output",
                    str(report_path),
                ]
            )
            env = os.environ.copy()
            env.pop("SOUP_E2_PROFILE_CACHE", None)
            env["PYTHONPATH"] = str(root / "src")
            before = machine_state()
            start = time.perf_counter()
            print(f"[{index}/{len(order)}] {mode} started; log: {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as stream:
                with subprocess.Popen(
                    command,
                    cwd=root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                ) as process:
                    returncode = process.wait()
            elapsed = time.perf_counter() - start
            after = machine_state()
            record = {
                "index": index,
                "mode": mode,
                "started_at": before["utc"],
                "finished_at": after["utc"],
                "returncode": returncode,
                "elapsed_train_command_seconds": elapsed,
                "machine_before": before,
                "machine_after": after,
                "command": command,
                "raw_report": str(report_path),
                "log": str(log_path),
            }
            manifest["runs"].append(record)
            if args.amp_init_scale is not None:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                created = [
                    line for line in lines if line.startswith("E2_AMP_SCALER_CREATED init_scale=")
                ]
                counts = [line for line in lines if line.startswith("E2_AMP_SCALERS_CREATED: ")]
                finals = [line for line in lines if line.startswith("E2_AMP_FINAL_SCALE: ")]
                if (len(created), len(counts), len(finals)) != (1, 1, 1):
                    raise RuntimeError(f"Missing/duplicate AMP proof for {name}: {log_path}")
                actual = float(created[0].split("init_scale=", 1)[1])
                if actual != args.amp_init_scale or counts[0] != "E2_AMP_SCALERS_CREATED: 1":
                    raise RuntimeError(f"AMP scale/count mismatch for {name}: {log_path}")
                record["amp"] = {
                    "initial_scale": actual,
                    "final_scale": float(finals[0].split(": ", 1)[1]),
                    "scalers_created": 1,
                }
                save_json(directory / "manifest.json", manifest)
            if returncode:
                raise RuntimeError(f"Run {name} failed with exit code {returncode}; see {log_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            record["report"] = report
            validate_report(report, steps, mode, examples)

            if precision_mode is not None:
                record["precision"] = validate_precision_provenance(
                    report, precision_mode
                )

            record["report_sha256"] = sha256(report_path)
            save_json(directory / "manifest.json", manifest)
            first_epoch_seconds = report["epoch_timing"][0]["duration_seconds"]
            second_epoch_seconds = report["epoch_timing"][1]["duration_seconds"]
            print(
                f"  valid: epoch1 {first_epoch_seconds:.3f}s of step time; "
                f"epoch2 {second_epoch_seconds:.3f}s of step time",
                flush=True,
            )

        tokens = [entry["report"]["tokens"]["useful"] for entry in manifest["runs"]]
        if len(set(tokens)) != 1:
            raise RuntimeError(f"Supervised token counts differ across A/B: {tokens}")
        uuids = [entry["machine_before"]["gpu_identity"]["stdout"] for entry in manifest["runs"]]
        if not args.pilot and not all(value and value == uuids[0] for value in uuids):
            raise RuntimeError(
                "GPU snapshots unavailable or inconsistent; comparability not established"
            )
        if len(manifest["runs"]) == 12:
            manifest["summary"] = summary(manifest["runs"])
        manifest["status"] = "COMPLETE_PRELIMINARY" if args.pilot else "COMPLETE_AWAITING_REVIEW"
        manifest["environment_end"] = machine_state()
        save_json(directory / "manifest.json", manifest)
        print(f"Complete; review results at {directory / 'manifest.json'}")
        if "summary" in manifest:
            print(json.dumps(manifest["summary"], indent=2))
    except BaseException as exc:
        manifest["status"] = "INCOMPLETE_OR_INVALID"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["environment_end"] = machine_state()
        save_json(directory / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
