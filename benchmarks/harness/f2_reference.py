"""Prepare, freeze, run and report F2 through the ordinary Soup CLI.

No download, rental or system changes. Preparation never trains. Freeze and
run require an approved matrix, a reserved window, a calibrated F1 watch,
NVMe evidence and a physical 8 GB GPU. See benchmarks/f2-14b-reference.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

from rich.console import Console

LENGTHS = (128, 384, 1024)
MODES = ("ram", "nvme", "refusal")
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
STEPS = 12
WARMUP = 4
ROOT = Path(__file__).parents[2]
console = Console()


def write_json(path: Path, value: object) -> None:
    def finite(item: object) -> object:
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        if isinstance(item, dict):
            return {key: finite(part) for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(part) for part in item]
        return item

    path.write_text(json.dumps(finite(value), indent=2, allow_nan=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def schedule() -> list[tuple[str, int]]:
    """Five F1 ABBA blocks per length, refusal after each block; no cherry picking."""
    return [
        (mode, length)
        for _ in range(5)
        for length in LENGTHS
        for mode in ("ram", "nvme", "nvme", "ram", "refusal")
    ]


def prepare(directory: Path, checkpoint: Path, cache: Path) -> None:
    directory = Path(os.path.realpath(directory))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "configs").mkdir()
    # Fixed, human-readable text for throughput, not a quality benchmark.
    text = (
        "Layer streaming keeps frozen weights in host memory. The adapter learns "
        "from supervised tokens while layers are transferred to the GPU. "
    ) * 180
    data = directory / "train.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Explain layer streaming, example {index}."},
                {"role": "assistant", "content": text},
            ]
        }
        for index in range(STEPS)
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cells = []
    for mode in MODES:
        for length in LENGTHS:
            cell = f"{mode}-{length}"
            config = {
                "base": os.path.realpath(checkpoint),
                "task": "sft",
                "backend": "transformers",
                "output": str(directory / "adapters" / cell),
                "data": {
                    "train": str(data),
                    "format": "chatml",
                    "val_split": 0,
                    "max_length": length,
                },
                "training": {
                    "epochs": 1,
                    "lr": 2e-5,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "seed": 42,
                    "data_seed": 42,
                    "quantization": "4bit",
                    "stream_layers": True,
                    "stream_source": "ram" if mode == "ram" else "disk",
                    "stream_buffers": 2,
                    "stream_read_ahead": 2,
                    "packing": False,
                    "stream_vram_probe": False,
                    "logging_steps": 1,
                    "save_steps": STEPS + 1,
                    "warmup_ratio": 0,
                    "weight_decay": 0.01,
                    "optimizer": "adamw_torch",
                    "lora": {
                        "r": 16,
                        "alpha": 16,
                        "dropout": 0,
                        "target_modules": ["q_proj", "v_proj"],
                    },
                },
            }
            if mode == "refusal":
                config["training"]["stream_disk_kind"] = "hdd"
            elif mode == "nvme":
                # On the recorded overlayfs rental, physical NVMe is verified
                # by O_DIRECT + diskstats; namespace-based auto detection is not.
                config["training"]["stream_disk_kind"] = "nvme"
            write_json(directory / "configs" / f"{cell}.json", config)
            cells.append({"cell": cell, "status": "NOT_RUN", "reason": "not measured"})
    write_json(
        directory / "plan.json",
        {
            "schema": 1,
            "model": MODEL,
            "checkpoint_revision": REVISION,
            "checkpoint": os.path.realpath(checkpoint),
            "cache": os.path.realpath(cache),
            "source_commit_at_preparation": git("rev-parse", "HEAD"),
            "seed": 42,
            "data_sha256": digest(data),
            "steps": STEPS,
            "warmup": WARMUP,
            "matrix_approved": False,
            "matrix_approval_reference": "",
            "resource_window": "",
            "nvme_evidence": "",
            "f1_clean_record": "",
            "f1_foreign_record": "",
            "window_ends_unix": None,
            "cells": cells,
            "note": "PROPOSAL: forced NVMe is not automatic RAM overflow; HDD is a policy control",
        },
    )
    report(directory)


def preflight(directory: Path) -> dict:
    plan = read_json(directory / "plan.json")
    tool = shutil.which("nvidia-smi")
    gpu = None
    gpu_error = "nvidia-smi unavailable"
    if tool:
        result = subprocess.run(
            [
                tool,
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        gpu_error = result.stderr or result.stdout
        if result.returncode == 0 and len(result.stdout.strip().splitlines()) == 1:
            uuid, name, memory, driver = [part.strip() for part in result.stdout.split(",")]
            gpu = {
                "uuid": uuid,
                "name": name,
                "memory_bytes": int(memory) * 1024**2,
                "driver": driver,
            }
            gpu_error = None
    versions = {}
    for name in (
        "soup-cli",
        "torch",
        "transformers",
        "trl",
        "peft",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "psutil",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    import psutil

    result = {
        "unix_s": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": versions,
        "gpu": gpu,
        "gpu_error": gpu_error,
        "host_ram_bytes": psutil.virtual_memory().total,
        "free_disk_bytes": shutil.disk_usage(directory).free,
        "checkpoint_present": Path(plan["checkpoint"]).is_dir(),
        "commit": git("rev-parse", "HEAD"),
    }
    write_json(directory / "preflight.json", result)
    return result


def freeze(directory: Path) -> None:
    """Pin resolved schema defaults and input hashes before the first measured arm."""
    plan = read_json(directory / "plan.json")
    if not plan["matrix_approved"] or not plan["matrix_approval_reference"]:
        raise ValueError("matrix approval is missing")
    if not all(plan[key] for key in ("resource_window", "nvme_evidence", "f1_clean_record")):
        raise ValueError("reserved window, physical NVMe evidence and clean F1 record required")
    if (plan.get("window_ends_unix") or 0) <= time.time():
        raise ValueError("resource window expired or missing")
    clean = read_json(Path(plan["f1_clean_record"]))
    if clean.get("outcome") != "ok" or any(
        clean.get("checks", {}).get(key, {}).get("outcome") != "ok"
        for key in ("suspend", "power", "foreign_reads")
    ):
        raise ValueError("F1 clean calibration is not ok on all required checks")
    if plan.get("physical_disk"):
        foreign_path = plan.get("f1_foreign_record")
        if not foreign_path:
            raise ValueError("overlay mapping requires a controlled foreign-read calibration")
        foreign = read_json(Path(foreign_path))
        if foreign.get("checks", {}).get("foreign_reads", {}).get("outcome") != "void":
            raise ValueError("foreign-read negative control did not trigger the F1 gate")
        if foreign.get("physical_disk_mapping") != plan["physical_disk"]:
            raise ValueError("foreign-read control used a different physical disk mapping")
    # The rule and observer must be committed; a dirty tree cannot freeze.
    git("ls-files", "--error-unmatch", "benchmarks/harness/f2_reference.py")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit the protocol and source before freezing")
    stamp = preflight(directory)
    if not stamp["gpu"] or not 7.5 * 1024**3 <= stamp["gpu"]["memory_bytes"] <= 8.5 * 1024**3:
        raise ValueError("a physical 8 GB GPU is required (24 GB limits do not qualify)")
    if not 28e9 <= stamp["host_ram_bytes"] <= 36e9:
        raise ValueError("the 32 GB host requirement is not met")
    if stamp["free_disk_bytes"] < 15e9:
        raise ValueError("less than 15 GB scratch headroom")
    cache = Path(os.path.realpath(plan["cache"]))
    if os.path.commonpath([str(cache), os.path.realpath(ROOT)]) != os.path.realpath(ROOT):
        raise ValueError("layer cache must be contained in the measured checkout")
    cache_files = sorted(
        path
        for path in cache.rglob("*")
        if path.is_file() and path.suffix in (".json", ".safetensors")
    )
    if not any(path.suffix == ".safetensors" for path in cache_files):
        raise ValueError("build and validate the complete NF4 cache before freezing")
    checkpoint = Path(plan["checkpoint"])
    config = read_json(checkpoint / "config.json")
    source = read_json(checkpoint / "f2-source.json")
    if source.get("model") != MODEL or source.get("revision") != REVISION:
        raise ValueError("checkpoint provenance does not match the pinned Hub revision")
    if (config.get("model_type"), config.get("num_hidden_layers"), config.get("hidden_size")) != (
        "qwen2",
        48,
        5120,
    ):
        raise ValueError("checkpoint is not the declared Qwen2.5-14B architecture")
    files = sorted(path for path in checkpoint.iterdir() if path.is_file())
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("real checkpoint weights are missing")
    index = read_json(checkpoint / "model.safetensors.index.json")
    if any(not (checkpoint / shard).is_file() for shard in set(index["weight_map"].values())):
        raise ValueError("checkpoint is missing a required safetensors shard")
    from soup_cli.config.schema import SoupConfig

    for path in sorted((directory / "configs").glob("*.json")):
        cfg = SoupConfig.model_validate(read_json(path))
        write_json(path, cfg.model_dump(mode="json"))
    inputs = [
        directory / "plan.json",
        directory / "train.jsonl",
        *sorted((directory / "configs").glob("*.json")),
    ]
    frozen = {
        "unix_s": time.time(),
        "commit": git("rev-parse", "HEAD"),
        "preflight": stamp,
        "inputs": {str(p): digest(p) for p in inputs},
        "checkpoint_files": {str(p): digest(p) for p in files},
        "cache_files": {str(p): digest(p) for p in cache_files},
        "nvme_evidence_sha256": digest(Path(plan["nvme_evidence"])),
        "f1_clean_sha256": digest(Path(plan["f1_clean_record"])),
    }
    with (directory / "frozen.json").open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2)


def judge(evidence: dict, kind: str, mode: str) -> tuple[str, str]:
    if evidence.get("validity", {}).get("outcome") != "ok":
        return "NO_VERDICT", "required F1 validity check void or unknown"
    audit = evidence.get("audit", {})
    physical = audit.get("physical_vram_bytes") or 0
    if not 7.5 * 1024**3 <= physical <= 8.5 * 1024**3:
        return "NO_VERDICT", "physical GPU is not 8 GB or not measured"
    if mode == "refusal":
        if evidence.get("returncode") != 0 and "needs NVMe or more RAM" in evidence.get("log", ""):
            return "EXPECTED_REFUSAL", "explicit HDD policy control, not a physical HDD result"
        return "FAIL", "expected policy refusal absent (unrelated errors do not count)"
    if evidence.get("returncode") != 0:
        return "FAIL", "CLI error/OOM; see retained log"
    if not audit.get("complete"):
        return "NO_VERDICT", "training observation incomplete"
    first, last = audit.get("fingerprint_first"), audit.get("fingerprint_last")
    if not first or not last or first == last:
        return "FAIL", "no trainable parameter update proved"
    counted_first = audit.get("fingerprint_counted_first")
    if not counted_first or counted_first == last:
        return "FAIL", "no parameter update in the counted window proved"
    steps = audit.get("steps", [])
    if len(steps) != STEPS or any((s.get("useful_tokens") or 0) <= 0 for s in steps[WARMUP:]):
        return "FAIL", "wrong step count or no supervised tokens"
    norms = [step.get("grad_norm") for step in steps[WARMUP:]]
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
        for value in norms
    ):
        return "FAIL", "missing, zero or nonfinite gradient norm"
    if not all(evidence.get(key) for key in ("telemetry_complete", "tier_confirmed")):
        return "NO_VERDICT", "missing counted-window telemetry or storage tier confirmation"
    if kind == "train" and not evidence.get("adapter_saved"):
        return "NO_VERDICT", "saved adapter missing"
    if kind == "bench" and not (evidence.get("bench") or {}).get("valid"):
        return "FAIL", "native bench train report is invalid or missing"
    peaks = [audit.get("peak_allocated_bytes"), audit.get("peak_reserved_bytes")]
    if any(value is None for value in peaks):
        return "NO_VERDICT", "allocator peaks missing"
    peaks.extend(
        audit[key]
        for key in ("setup_peak_allocated_bytes", "setup_peak_reserved_bytes")
        if key in audit
    )
    if max(peaks) > physical:
        return "FAIL", "allocator exceeds physical VRAM (spill is not a fit)"
    return "PASS", "training update, gradients, fit and arm validity demonstrated"


def trace_summary(rows: list[dict], audit: dict) -> dict:
    start, end = audit.get("counted_window_started"), audit.get("counted_window_ended")
    counted = [
        row
        for row in rows
        if start is not None and end is not None and start <= row["perf_counter"] <= end
    ]
    parsed = []
    for row in counted:
        try:
            values = [part.strip() for part in row["gpu"].split(",")]
            numeric = [float(values[index]) for index in (0, 1, 2, 3, 4, 6)]
            if not all(math.isfinite(value) for value in numeric) or "N/A" in values[5]:
                continue
            if any(row.get(key) is None for key in ("rss_bytes", "read_bytes", "write_bytes")):
                continue
            parsed.append((row, numeric))
        except (ValueError, IndexError, KeyError):
            continue
    if len(parsed) < 2:
        return {"complete": False, "reason": "no complete sample in counted training window"}
    stamps = [row["perf_counter"] for row, _ in parsed]
    gaps = [
        stamps[0] - start,
        end - stamps[-1],
        *[right - left for left, right in zip(stamps, stamps[1:])],
    ]
    return {
        "complete": len(parsed) == len(counted) and max(gaps) <= 4,
        "samples": len(parsed),
        "max_gap_seconds": max(gaps),
        "mean_gpu_util_percent": statistics.mean(value[0] for _, value in parsed),
        "median_sm_clock_mhz": statistics.median(value[1] for _, value in parsed),
        "sampled_peak_gpu_used_mib": max(value[2] for _, value in parsed),
        "sampled_peak_process_tree_rss_bytes": max(row["rss_bytes"] for row, _ in parsed),
        "sampled_read_bytes_delta": max(
            0, parsed[-1][0]["read_bytes"] - parsed[0][0]["read_bytes"]
        ),
        "sampled_write_bytes_delta": max(
            0, parsed[-1][0]["write_bytes"] - parsed[0][0]["write_bytes"]
        ),
        "sampled_span_seconds": stamps[-1] - stamps[0],
        "median_power_watts": statistics.median(value[3] for _, value in parsed),
        "power_limits_watts": sorted(set(value[4] for _, value in parsed)),
        "power_states": sorted(set(row["gpu"].split(",")[5].strip() for row, _ in parsed)),
        "note": "RSS sums processes; shared pages may be counted twice; not allocator VRAM",
    }


def tier_confirmed(mode: str, log: str) -> bool:
    normalized = " ".join(log.split())
    prefix = r"Layer streaming ready: \d+ layers, "
    if mode == "ram":
        return bool(re.search(prefix + r"[\d.]+ GB pinned RAM store", normalized))
    return bool(
        re.search(prefix + r"streamed from DISK .*?NVMe volume.*?pinned host staging", normalized)
    )


class Trace:
    """One-second GPU and child-process samples, saved before interpretation."""

    def __init__(self, path: Path, gpu: str) -> None:
        self.path, self.gpu = path, gpu
        self.rows: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.process = None

    def _sample(self) -> None:
        import psutil

        with self.path.open("x", encoding="utf-8") as stream:
            while not self.stop.is_set():
                before = time.perf_counter()
                result = subprocess.run(
                    [
                        shutil.which("nvidia-smi"),
                        "--id=" + self.gpu,
                        "--query-gpu=utilization.gpu,clocks.sm,memory.used,power.draw,power.limit,"
                        "pstate,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                row = {
                    "perf_counter": (before + time.perf_counter()) / 2,
                    "unix_s": time.time(),
                    "gpu": result.stdout.strip(),
                    "gpu_error": result.stderr.strip(),
                    "rss_bytes": None,
                    "read_bytes": None,
                    "write_bytes": None,
                    "vanished_child_pids": [],
                }
                try:
                    process = psutil.Process(self.process.pid)
                    tree = [process, *process.children(recursive=True)]
                    sampled = []
                    for member in tree:
                        try:
                            rss = member.memory_info().rss
                            io = member.io_counters()
                        except psutil.NoSuchProcess:
                            if member.pid == process.pid:
                                raise
                            # A child may finish between census and sensor reads.
                            # Keep the live tree; an inaccessible live process or
                            # a missing root still makes the entire sample unknown.
                            row["vanished_child_pids"].append(member.pid)
                            continue
                        sampled.append((rss, io.read_bytes, io.write_bytes))
                    row.update(
                        rss_bytes=sum(value[0] for value in sampled),
                        read_bytes=sum(value[1] for value in sampled),
                        write_bytes=sum(value[2] for value in sampled),
                    )
                except (psutil.Error, AttributeError):
                    pass  # Missing sensors are incomplete, not zero.
                self.rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                self.stop.wait(max(0, 1 - (time.perf_counter() - before)))


def run_one(
    directory: Path,
    cell: str,
    kind: str,
    index: int,
    gpu: str,
    *,
    deadline: float,
    physical_disk: str | None = None,
) -> dict:
    """Fresh CLI process inside F1; an error stays on disk and in the matrix."""
    from arm_validity import ArmWatch

    plan = read_json(directory / "plan.json")
    destination = directory / "runs" / f"{index:03d}-{cell}-{kind}"
    destination.mkdir(parents=True, exist_ok=False)
    config_path = directory / "configs" / f"{cell}.json"
    config = read_json(config_path)
    cli = ["train", "--config", str(config_path), "--yes"]
    bench_path, audit_path = destination / "bench.json", destination / "audit.json"
    if kind == "bench":
        cli = [
            "bench",
            "train",
            "--config",
            str(config_path),
            "--steps",
            str(STEPS),
            "--warmup",
            str(WARMUP),
            "--output",
            str(bench_path),
        ]
    command = [
        sys.executable,
        str(ROOT / "benchmarks/harness/f2_cli_observe.py"),
        "--audit",
        str(audit_path),
        "--",
        *cli,
    ]
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        CUDA_VISIBLE_DEVICES=gpu,
        SOUP_LAYER_STREAM_CACHE_DIR=plan["cache"],
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    write_json(
        destination / "command.json",
        {
            "argv": command,
            "native_cli_argv": ["soup", "--verbose", *cli],
            "cwd": str(ROOT),
            "environment_overrides": {
                key: env[key]
                for key in (
                    "PYTHONPATH",
                    "CUDA_VISIBLE_DEVICES",
                    "SOUP_LAYER_STREAM_CACHE_DIR",
                    "HF_HUB_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                    "HF_HUB_DISABLE_TELEMETRY",
                )
            },
        },
    )
    kwargs = {}
    if physical_disk:
        # Explicit overlay-to-physical mapping, established in the NVMe evidence.
        # This changes only device resolution; all host-wide F1 counters remain live.
        kwargs["disk_device"] = lambda path: physical_disk
    started = time.time()
    trace = Trace(destination / "trace.jsonl", gpu)
    timeout = False
    with ArmWatch(
        arm=cell, round=index, run=index, label=f"F2 {gpu}", model_path=plan["cache"], **kwargs
    ) as watch:
        with (destination / "console.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            trace.process = process
            trace.thread.start()
            try:
                code = process.wait(timeout=max(1, min(1800, deadline - time.time())))
            except subprocess.TimeoutExpired:
                timeout = True
                process.terminate()  # This arm's child only, never other GPU processes.
                try:
                    code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
            finally:
                trace.stop.set()
                trace.thread.join(timeout=15)
    write_json(destination / "validity.json", watch.record)
    audit = read_json(audit_path) if audit_path.exists() else {}
    log = (destination / "console.log").read_text(encoding="utf-8")
    mode = cell.split("-")[0]
    telemetry = trace_summary(trace.rows, audit)
    adapters = list(Path(config["output"]).glob("adapter_model.*"))
    evidence = {
        "cell": cell,
        "kind": kind,
        "returncode": code,
        "elapsed_seconds": time.time() - started,
        "timeout": timeout,
        "audit": audit,
        "validity": watch.record,
        "bench": read_json(bench_path) if bench_path.exists() else None,
        "telemetry": telemetry,
        "telemetry_complete": telemetry["complete"],
        "tier_confirmed": tier_confirmed(mode, log),
        "adapter_saved": any(p.stat().st_mtime >= started for p in adapters),
        "log": log,
    }
    evidence["status"], evidence["reason"] = judge(evidence, kind, mode)
    # Raw log has its own file; keep just its hash in the summary.
    evidence.pop("log")
    evidence["log_sha256"] = digest(destination / "console.log")
    write_json(destination / "result.json", evidence)
    report(directory)
    return evidence


def run(directory: Path) -> None:
    directory = Path(os.path.realpath(directory))
    frozen = read_json(directory / "frozen.json")
    plan = read_json(directory / "plan.json")
    if frozen["commit"] != git("rev-parse", "HEAD") or git(
        "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("source changed since freeze")
    for path, expected in frozen["inputs"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"frozen input changed: {path}")
    # Recheck checkpoint once before the campaign, never hash 30 GB inside an arm.
    for path, expected in (frozen["checkpoint_files"] | frozen["cache_files"]).items():
        if digest(Path(path)) != expected:
            raise ValueError(f"checkpoint changed: {path}")
    gpu = frozen["preflight"]["gpu"]["uuid"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # F1 and workload observe the same GPU.
    current = preflight(directory)
    if current["gpu"] != frozen["preflight"]["gpu"]:
        raise ValueError("GPU identity, driver or physical VRAM changed since freeze")
    # No automatic restart: replacing partial ABBA blocks needs an explicit new campaign.
    if (directory / "runs").exists():
        raise ValueError("existing arms retained; prepare a new campaign to rerun")
    for pair, (mode, length) in enumerate(schedule()):
        for offset, kind in enumerate(("train", "bench") if pair % 2 == 0 else ("bench", "train")):
            if time.time() >= plan["window_ends_unix"]:
                console.print("Reserved window ended; unexecuted cells remain NOT_RUN.")
                return
            peers = subprocess.check_output(
                [
                    shutil.which("nvidia-smi"),
                    f"--id={gpu}",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            if peers:
                raise ValueError("GPU compute peers present; leave them running and stop F2")
            result = run_one(
                directory,
                f"{mode}-{length}",
                kind,
                pair * 2 + offset + 1,
                gpu,
                deadline=plan["window_ends_unix"],
                physical_disk=plan.get("physical_disk"),
            )
            console.print(f"{result['cell']} {kind}: {result['status']} ({result['reason']})")


def report(directory: Path) -> None:
    plan = read_json(directory / "plan.json")
    results = [read_json(path) for path in sorted((directory / "runs").glob("*/result.json"))]
    lines = [
        "# F2 matrix (new hardware only)",
        "",
        "NOT_RUN is not a measurement or an OOM.",
        "",
        "| Condition | Length | Outcomes | supervised tok/s | step s | "
        "VRAM alloc/res GiB | sampled RAM GiB | sampled read MB/s | GPU % | SM MHz |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in plan["cells"]:
        matching = [r for r in results if r["cell"] == cell["cell"]]
        mode, length = cell["cell"].split("-")
        counts = Counter(f"{r['kind']}:{r['status']}" for r in matching)
        outcomes = "; ".join(f"{key}={count}" for key, count in sorted(counts.items())) or "NOT_RUN"
        valid = [r for r in matching if r["status"] == "PASS" and r["kind"] == "bench"]
        rates = [r["bench"]["throughput"]["useful_tokens_per_second"] for r in valid]
        metrics = ["—"] * 7
        if rates:
            peak_rss = max(r["telemetry"]["sampled_peak_process_tree_rss_bytes"] for r in valid)
            read_rates = [
                r["telemetry"]["sampled_read_bytes_delta"]
                / max(1e-9, r["telemetry"]["sampled_span_seconds"])
                / 1e6
                for r in valid
            ]
            metrics = [
                f"{statistics.median(rates):.3f} (n={len(rates)})",
                f"{statistics.median(r['bench']['timing']['median_seconds'] for r in valid):.4f}",
                "/".join(
                    f"{max(r['audit'][key] for r in valid) / 1024**3:.3f}"
                    for key in ("peak_allocated_bytes", "peak_reserved_bytes")
                ),
                f"{peak_rss / 1024**3:.3f}",
                f"{statistics.median(read_rates):.3f}",
                f"{statistics.mean(r['telemetry']['mean_gpu_util_percent'] for r in valid):.2f}",
                f"{statistics.median(r['telemetry']['median_sm_clock_mhz'] for r in valid):.0f}",
            ]
        lines.append(f"| {mode} | {length} | {outcomes} | " + " | ".join(metrics) + " |")
    lines += ["", "Descriptive only. No 10% effect claim without the F1 repeats rule."]
    (directory / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "freeze", "run", "report"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.checkpoint or not args.cache:
            parser.error("prepare requires --checkpoint and --cache")
        prepare(args.directory, args.checkpoint, args.cache)
    elif args.action == "preflight":
        console.print(preflight(args.directory))
    elif args.action == "freeze":
        freeze(args.directory)
    elif args.action == "report":
        report(args.directory)
    else:
        run(args.directory)


if __name__ == "__main__":
    main()
