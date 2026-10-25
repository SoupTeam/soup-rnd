"""Qualify two-read and expert caches before measuring four fresh-process arms.

This runner never leases hardware, downloads weights, sets resource limits or
deletes a store. Use a materialized local checkpoint and an externally bounded
Linux CUDA environment. See two_read_cache.md for the complete protocol.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console

# A direct `python benchmarks/harness/two_read_cache.py` invocation also works.
ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.harness.two_read_cache_common import (  # noqa: E402
    ARMS,
    atomic_json,
    compare_captures,
    fingerprint_files,
    sha256_file,
    validate_capture,
)

GIB = 2**30
# Each arm occupies every position once. Every block uses a fresh process.
ORDERS = tuple(ARMS[shift:] + ARMS[:shift] for shift in range(4))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def source_fingerprint() -> dict[str, str]:
    paths = list((ROOT / "src" / "soup_cli").rglob("*.py"))
    paths += sorted((ROOT / "benchmarks" / "harness").glob("two_read_cache*.py"))
    paths.append(ROOT / "pyproject.toml")
    return fingerprint_files(paths, ROOT)


def _model_files(weights: Path) -> list[Path]:
    files = sorted(weights.glob("*.safetensors"))
    if not files:
        raise ValueError("A local safetensors checkpoint is required; downloads are not automatic")
    return [weights / "config.json", *files, *sorted(weights.glob("*.safetensors.index.json"))]


def _store_files(store: Path) -> list[Path]:
    return [store / "index.json", *sorted(store.glob("*.safetensors"))]


def _cgroup_limits() -> dict[str, Any]:
    """Collect v2 limits through parents; never silently call an unlimited host bounded."""
    from soup_cli.utils.stream_ram_cache import _cgroup_paths

    groups = _cgroup_paths(Path("/proc/self/cgroup"), Path("/proc/self/mountinfo"))
    if any(version != 2 for version, _directory in groups):
        raise ValueError("This protocol requires cgroup v2 for RAM/swap qualification")
    directories = {directory for _version, directory in groups}
    report: dict[str, Any] = {"paths": {}}
    for directory in sorted(directories):
        row: dict[str, Any] = {}
        for filename in ("memory.max", "memory.swap.max", "memory.swap.current"):
            try:
                value = (directory / filename).read_text().strip()
                row[filename] = None if value == "max" else int(value)
            except (OSError, ValueError):
                row[filename] = None
        if any(value is not None for value in row.values()):
            report["paths"][str(directory)] = row
    for name in ("memory.max", "memory.swap.max"):
        values = [row[name] for row in report["paths"].values() if row[name] is not None]
        report[name] = min(values) if values else None
    return report


def host_boundary(max_host_bytes: int) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise ValueError("GPU qualification requires Linux process I/O and RAM/swap evidence")
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    groups = _cgroup_limits()
    physical = values["MemTotal"]
    limits = [physical]
    if groups["memory.max"] is not None:
        limits.append(groups["memory.max"])
    effective = min(limits)
    if effective > max_host_bytes:
        raise ValueError("Set a real RAM limit externally before running this benchmark")
    no_swap = values.get("SwapTotal", 0) == 0 or groups["memory.swap.max"] == 0
    if not no_swap or any(
        row["memory.swap.current"] not in (None, 0) for row in groups["paths"].values()
    ):
        raise ValueError("Disable swap for this workload externally before GPU qualification")
    return {"physical_bytes": physical, "effective_limit_bytes": effective,
            "requested_max_bytes": max_host_bytes, "cgroup": groups, "swap_disabled": True}


def _hardware_in_process() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is required; a CPU-only run is not GPU qualification")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    if (torch.cuda.get_device_capability(device)[0] < 8
            or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError("Native BF16 support is required")
    total = int(properties.total_memory)
    if not int(7.5 * GIB) <= total <= int(8.5 * GIB):
        raise ValueError("This protocol requires a physical 8 GiB card, not an allocator cap")
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("nvidia-smi evidence is required") from exc
    uuid = str(getattr(properties, "uuid", ""))
    if not uuid or uuid == "None":
        raise ValueError("Physical GPU UUID is required for run identity")
    return {"name": properties.name, "physical_vram_bytes": total,
            "uuid": uuid,
            "capability": list(torch.cuda.get_device_capability(device)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "nvidia_smi": smi}


def hardware() -> dict[str, Any]:
    # The controller must not keep its own CUDA context in the measured 8 GiB.
    code = ("from benchmarks.harness.two_read_cache import _hardware_in_process; "
            "from rich.console import Console; import json; "
            "Console().print_json(json.dumps(_hardware_in_process()))")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT)
               + os.pathsep + os.environ.get("PYTHONPATH", ""))
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=60)
    if process.returncode:
        raise ValueError("CUDA hardware qualification failed: " + process.stderr[-3000:])
    return json.loads(process.stdout)


def environment() -> dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "bitsandbytes", "trl", "safetensors")
    }}


def attention_targets(config: Any) -> list[str]:
    """Freeze the actual DeepSeek query projection before preparing any arm.

    Moonlight has no query low-rank factor: its attention owns ``q_proj``.
    DeepSeek configurations with ``q_lora_rank`` instead own ``q_b_proj``.
    The same resolved list is retained in the spec for all four arms.
    """
    query = "q_proj" if config.q_lora_rank is None else "q_b_proj"
    return [query, "kv_b_proj", "o_proj"]


def validate_attention_targets(skeleton: Any, targets: list[str]) -> None:
    """Refuse an unmatched LoRA projection before expensive checkpoint sharding."""
    available = {name.rsplit(".", 1)[-1] for name, _module in skeleton.named_modules()}
    missing = sorted(set(targets) - available)
    if missing:
        raise ValueError(f"Requested attention targets are absent from the skeleton: {missing}")


def validate_spec(spec: dict[str, Any]) -> None:
    for field, minimum in (("seq_len", 2), ("batch_size", 1), ("accumulation", 2),
                           ("accuracy_updates", 2), ("warmup_updates", 2), ("timed_updates", 6),
                           ("rounds", 4), ("timeout_seconds", 1), ("lora_r", 1),
                           ("lora_alpha", 1)):
        if type(spec.get(field)) is not int or spec[field] < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    if spec["rounds"] % 4:
        raise ValueError("rounds must be a multiple of four for balanced arm order")
    for field in ("host_cache_bytes", "device_cache_bytes"):
        if type(spec.get(field)) is not int or spec[field] < 1:
            raise ValueError(f"{field} must be positive to qualify both cache tiers")
    for field in ("device_cap_bytes", "max_host_bytes"):
        if type(spec.get(field)) is not int or spec[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if spec["device_cap_bytes"] != 8 * GIB or spec["max_host_bytes"] > 32 * GIB:
        raise ValueError("This protocol fixes an 8 GiB VRAM cap and at most 32 GiB RAM")
    if spec.get("dtype") != "bfloat16" or spec.get("quant") != "nf4":
        raise ValueError("This protocol fixes BF16 compute and NF4 base storage")
    targets = spec.get("lora_targets")
    if (not isinstance(targets, list) or not targets or
            any(not isinstance(item, str) or not item or "expert" in item for item in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError("Use explicit non-expert LoRA projections supported by two-read")


def _identity_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    # Driver/visible-index can differ in formatting; UUID, architecture and capacity cannot.
    keys = ("name", "uuid", "capability", "physical_vram_bytes")
    return all(actual[key] == expected[key] for key in keys)


def verify_spec(spec: dict[str, Any], *, content: bool) -> None:
    validate_spec(spec)
    if os.path.realpath(spec["index_path"]) != os.path.realpath(
        os.path.join(spec["shard_dir"], "index.json")
    ):
        raise ValueError("Only the fingerprinted shard_dir/index.json may be executed")
    if source_fingerprint() != spec["source_fingerprint"]:
        raise ValueError("Executed source changed after preparation")
    if sha256_file(spec["training_data_path"]) != spec["training_data_sha256"]:
        raise ValueError("Frozen token input changed")
    if environment() != spec["environment"]:
        raise ValueError("Dependency environment changed after preparation")
    if not _identity_matches(hardware(), spec["hardware"]):
        raise ValueError("Physical GPU changed after preparation")
    host_boundary(spec["max_host_bytes"])
    for label, root, files in (
        ("model", Path(spec["weights"]), _model_files(Path(spec["weights"]))),
        ("store", Path(spec["shard_dir"]), _store_files(Path(spec["shard_dir"]))),
    ):
        actual = fingerprint_files(files, root) if content else {
            path.relative_to(root).as_posix(): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in files
        }
        expected = spec[f"{label}_fingerprint" if content else f"{label}_stat"]
        if actual != expected:
            raise ValueError(f"{label} identity changed after preparation")


def prepare(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    for field in ("host_cache_gib", "device_cache_gib", "max_host_gib"):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be finite and positive")
    validate_spec({**vars(args), "dtype": "bfloat16", "quant": "nf4", "lora_r": 4,
                   "lora_alpha": 8, "lora_targets": ["q_b_proj", "kv_b_proj", "o_proj"],
                   "host_cache_bytes": int(args.host_cache_gib * GIB),
                   "device_cache_bytes": int(args.device_cache_gib * GIB),
                   "device_cap_bytes": 8 * GIB, "max_host_bytes": int(args.max_host_gib * GIB)})
    output = Path(os.path.realpath(args.out))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation output must be a new or empty directory")
    weights = Path(os.path.realpath(args.weights))
    store = Path(os.path.realpath(args.shards))
    _model_files(weights)
    gpu = hardware()
    host = host_boundary(int(args.max_host_gib * GIB))
    import torch
    from transformers import AutoConfig

    from soup_cli.utils.layer_shard import read_shard_index, shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    config = AutoConfig.from_pretrained(
        str(weights), local_files_only=True, trust_remote_code=False,
    )
    if config.model_type != "deepseek_v3" or not getattr(config, "n_routed_experts", 0):
        raise ValueError("Two-read requires a supported DeepSeek-V3 MoE checkpoint")
    if args.seq_len > int(config.max_position_embeddings):
        raise ValueError("Sequence length exceeds the checkpoint context limit")
    targets = attention_targets(config)
    before = fingerprint_files(_model_files(weights), weights)
    skeleton = build_meta_skeleton(str(weights), dtype="bfloat16", quant="nf4")
    validate_attention_targets(skeleton, targets)
    suffixes = quantised_layer_suffixes(skeleton)
    del skeleton
    shard_started = time.perf_counter()
    shard_checkpoint(
        str(weights), str(store), dtype="bfloat16", arch="deepseek_v3", quant="nf4",
        quant_suffixes=suffixes, double_quant=True, quant_device="cuda",
        num_hidden_layers=config.num_hidden_layers, expert_layout="per_expert",
        notify=Console().print,
    )
    shard_seconds = time.perf_counter() - shard_started
    index = read_shard_index(str(store))
    if index.stripe_roots or index.external_tensors:
        raise ValueError("This first protocol requires an unstriped, self-contained expert store")
    if (index.arch != "deepseek_v3" or index.quant != "nf4" or index.dtype != "bfloat16"
            or index.expert_layout != "per_expert" or not index.expert_layers
            or index.quant_device != "cuda" or not index.double_quant):
        raise ValueError("Shard index does not match the fixed CUDA NF4 per-expert protocol")
    if before != fingerprint_files(_model_files(weights), weights):
        raise ValueError("Checkpoint changed during sharding")
    output.mkdir(parents=True, exist_ok=True)
    token_path = output / "inputs.pt"
    generator = torch.Generator().manual_seed(args.input_seed)
    tokens = torch.randint(0, config.vocab_size, (
        args.accuracy_updates * args.accumulation, args.batch_size, args.seq_len,
    ), generator=generator)
    torch.save(tokens, token_path)
    spec = {
        "schema_version": 1, "weights": str(weights), "shard_dir": str(store),
        "index_path": str(store / "index.json"), "dtype": "bfloat16", "quant": "nf4",
        "seed": args.seed, "input_seed": args.input_seed, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "accumulation": args.accumulation,
        "accuracy_updates": args.accuracy_updates, "warmup_updates": args.warmup_updates,
        "timed_updates": args.timed_updates, "rounds": args.rounds,
        "timeout_seconds": args.timeout_seconds, "lora_r": 4, "lora_alpha": 8,
        "lora_targets": targets,
        "host_cache_bytes": int(args.host_cache_gib * GIB),
        "device_cache_bytes": int(args.device_cache_gib * GIB),
        "device_cap_bytes": 8 * GIB, "max_host_bytes": int(args.max_host_gib * GIB),
        "training_data_path": str(token_path), "training_data_sha256": sha256_file(token_path),
        "input_kind": "frozen_random_token_ids; numerical/throughput only, not task quality",
        "source_fingerprint": source_fingerprint(), "model_fingerprint": before,
        "store_fingerprint": fingerprint_files(_store_files(store), store),
        "hardware": gpu, "host_boundary": host, "environment": environment(),
        "shard_seconds": shard_seconds, "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
    }
    for label, root, files in (("model", weights, _model_files(weights)),
                               ("store", store, _store_files(store))):
        spec[f"{label}_stat"] = {
            path.relative_to(root).as_posix(): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in files
        }
    validate_spec(spec)
    spec["prepare_seconds"] = time.perf_counter() - started
    atomic_json(output / "spec.json", spec)
    return output / "spec.json"


def launch_worker(spec_path: Path, spec: dict[str, Any], arm: str, phase: str, out: Path) -> dict:
    verify_spec(spec, content=False)
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "benchmarks.harness.two_read_cache_worker",
               "--spec", str(spec_path), "--arm", arm, "--phase", phase, "--out", str(out)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep +
               str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.perf_counter()
    with (out / "worker.log").open("wb") as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=spec["timeout_seconds"])
    if process.returncode:
        raise RuntimeError(f"{phase}/{arm} failed ({process.returncode}); see {out / 'worker.log'}")
    result = _read_json(out / "result.json")
    if (result.get("status") != "passed" or result.get("phase") != phase
            or result.get("arm") != arm or result.get("spec_sha256") != sha256_file(spec_path)):
        raise ValueError(f"Worker result is incomplete or belongs to another run: {out}")
    result["fresh_process_wall_seconds"] = time.perf_counter() - started
    atomic_json(out / "controller-result.json", result)
    return result


def timing_summary(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired block bootstrap; >=2% gate is fixed, not a configurable tolerance."""
    by_round: dict[int, dict[str, float]] = {}
    for block in blocks:
        times = block["result"]["timing"]["update_seconds"]
        if len(times) < 6 or any(type(value) not in (int, float) or
                                 not math.isfinite(value) or value <= 0 for value in times):
            raise ValueError("Every timing block needs >=6 complete, finite optimizer updates")
        row = by_round.setdefault(block["round"], {})
        if block["arm"] in row or block["arm"] not in ARMS:
            raise ValueError("Duplicate or unknown arm in a timing round")
        row[block["arm"]] = statistics.median(times)
    if len(by_round) < 4 or any(set(row) != set(ARMS) for row in by_round.values()):
        raise ValueError("At least four complete paired four-arm rounds are required")
    if len(by_round) % 4 or set(by_round) != set(range(len(by_round))):
        raise ValueError("Timing rounds must be contiguous and a multiple of four")
    for block in blocks:
        if block.get("position") != ORDERS[block["round"] % 4].index(block["arm"]):
            raise ValueError("Timing arm position differs from the balanced schedule")
    def comparison(arm: str, reference: str) -> dict[str, Any]:
        logs = [math.log(row[arm] / row[reference]) for row in by_round.values()]
        generator = random.Random(29)
        bootstrap = sorted(
            100 * (1 - math.exp(statistics.mean(generator.choices(logs, k=len(logs)))))
            for _ in range(5000)
        )
        reduction = 100 * (1 - math.exp(statistics.mean(logs)))
        lower, upper = bootstrap[124], bootstrap[4874]
        return {
            "step_time_reduction_percent": reduction, "paired_bootstrap_95_percent": [lower, upper],
            "passes_practical_2_percent_gate": lower >= 2.0,
            "reference": reference, "paired_rounds": len(logs),
        }
    comparisons = {arm: comparison(arm, "ordinary") for arm in ARMS[1:]}
    incremental = {"two_read_cache_vs_" + reference: comparison("two_read_cache", reference)
                   for reference in ("two_read", "cache")}
    absolute = {}
    for arm in ARMS:
        results = [block["result"] for block in blocks if block["arm"] == arm]
        seconds = statistics.median([
            value for result in results for value in result["timing"]["update_seconds"]
        ])
        tokens = {result["timing"].get("tokens_per_update") for result in results}
        if len(tokens) != 1:
            raise ValueError("Tokens per optimizer update changed between timing blocks")
        token_count = next(iter(tokens))
        def peak(section: str, key: str) -> int | None:
            values = [result["timing"].get(section, {}).get(key) for result in results]
            return max(values) if all(value is not None for value in values) else None
        absolute[arm] = {
            "median_update_seconds": seconds,
            "tokens_per_second": token_count / seconds if token_count is not None else None,
            "max_step_allocated_bytes": peak("cuda", "peak_allocated_bytes"),
            "max_step_reserved_bytes": peak("cuda", "peak_reserved_bytes"),
            "max_sampled_rss_bytes": peak("resources", "sampled_peak_rss_bytes"),
            "setup_seconds": [result.get("setup", {}).get("seconds") for result in results],
            "entry_wall_seconds": [result.get("fresh_process_wall_seconds") for result in results],
        }
    return {"method": "geometric mean paired block-median ratio; 5000 round bootstrap samples",
            "comparisons": comparisons, "incremental_comparisons": incremental, "arms": absolute,
            "caution": "Small-sample fixed-host result; not full training or task-quality evidence"}


def validate_cache_behavior(accuracy: dict[str, Any]) -> dict[str, Any]:
    """Separate logical visits, file payload reads and actual coverage of both cache tiers."""
    deltas = {}
    for arm in ARMS:
        stage = accuracy[arm]["worker"]["accuracy"]
        before = stage["runtime_before"]["expert_cache"]
        after = stage["runtime_after"]["expert_cache"]
        row = {key: after[key] - before[key] for key in (
            "logical_fetches", "storage_reads", "storage_bytes_read", "materializations",
        )}
        for tier in ("host", "device"):
            row[tier + "_hits"] = after[tier]["hits"] - before[tier]["hits"]
        if not 0 < row["storage_reads"] <= row["logical_fetches"]:
            raise ValueError(f"{arm}: missing or inconsistent expert-fetch evidence")
        if arm in ("ordinary", "two_read"):
            if row["storage_reads"] != row["logical_fetches"] or any(
                row[tier + "_hits"] for tier in ("host", "device")
            ):
                raise ValueError(f"{arm}: uncached control unexpectedly used a cache")
        elif any(row[tier + "_hits"] <= 0 for tier in ("host", "device")):
            raise ValueError(f"{arm}: both RAM and GPU cache hits must actually be exercised")
        deltas[arm] = row
    visits = {arm: row["logical_fetches"] for arm, row in deltas.items()}
    if (visits["ordinary"] * 2 != visits["two_read"] * 3 or
            visits["cache"] != visits["ordinary"] or
            visits["two_read_cache"] != visits["two_read"]):
        raise ValueError("Observed visits do not support the expected three/two-read controls")
    return {"expert_counter_deltas": deltas,
            "storage_bytes_read_meaning": "expert file payload; not physical SSD traffic"}


def run(spec_path: Path, out: Path, *, accuracy_only: bool = False) -> dict[str, Any]:
    import torch

    spec_path = Path(os.path.realpath(spec_path))
    out = Path(os.path.realpath(out))
    if out.exists() and any(out.iterdir()):
        raise ValueError("Run output must be a new or empty directory; partial runs are retained")
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "running", "spec_sha256": sha256_file(spec_path),
                              "accuracy": {}, "timing_blocks": [], "gpu_rented_by_runner": False}
    started = time.perf_counter()
    try:
        spec = _read_json(spec_path)
        verify_spec(spec, content=True)
        for arm in ARMS:
            result = launch_worker(spec_path, spec, arm, "accuracy", out / "accuracy" / arm)
            report["accuracy"][arm] = {"worker": result}
            atomic_json(out / "run.json", report)
        reference_path = out / "accuracy" / "ordinary" / "capture.pt"
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        def check_counts(capture: dict[str, Any]) -> None:
            microbatches = spec["accuracy_updates"] * spec["accumulation"]
            if (len(capture["updates"]) != spec["accuracy_updates"] or
                    len(capture["microbatches"]) != microbatches):
                raise ValueError("Capture omitted required updates or accumulated microbatches")

        check_counts(reference)
        report["accuracy"]["ordinary"]["gate"] = validate_capture(reference)
        for arm in ARMS[1:]:
            path = out / "accuracy" / arm / "capture.pt"
            candidate = torch.load(path, map_location="cpu", weights_only=True)
            check_counts(candidate)
            report["accuracy"][arm]["gate"] = compare_captures(reference, candidate)
            del candidate
        del reference
        for arm in ARMS:
            path = out / "accuracy" / arm / "capture.pt"
            report["accuracy"][arm]["capture_sha256"] = sha256_file(path)
            if report["accuracy"][arm]["worker"]["accuracy"]["capture_sha256"] != sha256_file(path):
                raise ValueError(f"{arm} capture does not match worker evidence")
        atomic_json(out / "run.json", report)
        if not all(report["accuracy"][arm]["gate"]["passed"] for arm in ARMS):
            raise ValueError("Accuracy gate failed; timing was not started")
        report["cache_behavior"] = validate_cache_behavior(report["accuracy"])
        verify_spec(spec, content=True)
        report["accuracy_passed"] = True
        if not accuracy_only:
            for number in range(spec["rounds"]):
                for position, arm in enumerate(ORDERS[number % 4]):
                    Console().print(f"Timing round {number + 1}/{spec['rounds']}: {arm}")
                    directory = out / "timing" / f"round-{number:02d}" / arm
                    result = launch_worker(spec_path, spec, arm, "timing", directory)
                    report["timing_blocks"].append({
                        "round": number, "position": position, "arm": arm, "result": result,
                    })
                    atomic_json(out / "run.json", report)
            verify_spec(spec, content=True)
            report["timing_summary"] = timing_summary(report["timing_blocks"])
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
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("prepare", help="CUDA: shard local real weights and freeze run identity")
    for name in ("weights", "shards", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.add_argument("--host-cache-gib", type=float, required=True)
    stage.add_argument("--device-cache-gib", type=float, required=True)
    stage.add_argument("--max-host-gib", type=float, default=32)
    for name, default in (("seq-len", 512), ("batch-size", 1), ("accumulation", 2),
                          ("accuracy-updates", 2), ("warmup-updates", 2), ("timed-updates", 6),
                          ("rounds", 4), ("seed", 3), ("input-seed", 29),
                          ("timeout-seconds", 1800)):
        stage.add_argument(f"--{name}", type=int, default=default)
    execute = sub.add_parser("run", help="CUDA: accuracy qualification, then balanced timing")
    execute.add_argument("--spec", type=Path, required=True)
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--accuracy-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            Console().print({"prepared_spec": str(prepare(args))})
        else:
            result = run(args.spec, args.out, accuracy_only=args.accuracy_only)
            Console().print({"status": result["status"], "results": str(args.out)})
        return 0
    except Exception as exc:
        Console(stderr=True).print(f"Qualification refused/failed: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
