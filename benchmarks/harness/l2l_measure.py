#!/usr/bin/env python3
"""A5 measurement record: speed, GPU and RAM, and I/O in one record per step and per run.

``step_record`` holds one step: tokens without padding per second, the step time,
the VRAM peaks (allocated and reserved), the process RSS peak, the pinned bytes of
the activation store, the bytes read from the weights' disk, the bytes the spill
wrote and read, the layer loads and the NF4 restores. ``run_record`` folds the
steps of one run into the same fields.

A value whose source is missing is ``None`` with its reason under ``unavailable``,
never zero. A ``None`` without a reason, or a reason for a value that is there, is
refused, so a record cannot hide a gap.

Benchmark harness code, not shipped. Standard library only.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a4_box  # noqa: E402
import arm_validity  # noqa: E402

#: Fields that hold a peak: a run takes the largest step value.
PEAK_FIELDS = (
    "peak_vram_allocated_bytes",
    "peak_vram_reserved_bytes",
    "peak_rss_bytes",
    "pinned_store_bytes",
)

#: Fields that count work or bytes moved: a run sums its steps.
COUNT_FIELDS = (
    "weight_read_bytes",
    "os_read_bytes",
    "os_write_bytes",
    "device_read_bytes",
    "spill_written_bytes",
    "spill_read_bytes",
    "layer_loads",
    "nf4_restores",
)

MEASURED_FIELDS = PEAK_FIELDS + COUNT_FIELDS


def count_real_tokens(attention_mask: Any) -> int:
    """Tokens that are not padding: the ones in an attention mask (rows of 0 and 1)."""
    total = getattr(attention_mask, "sum", None)
    if callable(total):
        return int(total())
    return sum(int(value) for row in attention_mask for value in row)


def _check_reasons(values: Mapping[str, Optional[int]], unavailable: Mapping[str, str]) -> None:
    for name, value in values.items():
        if value is None and name not in unavailable:
            raise ValueError(f"{name} is null without a reason in 'unavailable'")
        if value is not None and name in unavailable:
            raise ValueError(f"{name} has a value and a reason for being unavailable")
    unknown = sorted(set(unavailable) - set(values))
    if unknown:
        raise ValueError(f"no such measured fields: {', '.join(unknown)}")


def step_record(
    *,
    step_s: float,
    real_tokens: int,
    layer_loads: Optional[int],
    nf4_restores: Optional[int],
    spill_written_bytes: Optional[int],
    spill_read_bytes: Optional[int],
    peak_vram_allocated_bytes: Optional[int],
    peak_vram_reserved_bytes: Optional[int],
    peak_rss_bytes: Optional[int],
    pinned_store_bytes: Optional[int],
    weight_read_bytes: Optional[int],
    os_read_bytes: Optional[int],
    os_write_bytes: Optional[int],
    device_read_bytes: Optional[int],
    unavailable: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """One step. Tokens are the ones without padding; bytes are bytes, time is seconds."""
    if not step_s > 0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    if real_tokens <= 0:
        raise ValueError(f"real_tokens must be positive, got {real_tokens}")
    values = {
        "peak_vram_allocated_bytes": peak_vram_allocated_bytes,
        "peak_vram_reserved_bytes": peak_vram_reserved_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "pinned_store_bytes": pinned_store_bytes,
        "weight_read_bytes": weight_read_bytes,
        "os_read_bytes": os_read_bytes,
        "os_write_bytes": os_write_bytes,
        "device_read_bytes": device_read_bytes,
        "spill_written_bytes": spill_written_bytes,
        "spill_read_bytes": spill_read_bytes,
        "layer_loads": layer_loads,
        "nf4_restores": nf4_restores,
    }
    reasons = dict(unavailable or {})
    _check_reasons(values, reasons)
    return {
        "step_s": float(step_s),
        "real_tokens": int(real_tokens),
        "tokens_per_s": real_tokens / step_s,
        **values,
        "unavailable": reasons,
    }


def run_record(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """One run from its steps: tokens over time, the largest peaks, summed counters.

    A field is ``None`` when any step lacks it, with the steps' reasons joined.
    """
    if not steps:
        raise ValueError("a run needs at least one step")
    total_s = sum(step["step_s"] for step in steps)
    total_tokens = sum(step["real_tokens"] for step in steps)
    record: Dict[str, Any] = {
        "steps": len(steps),
        "step_s_mean": total_s / len(steps),
        "real_tokens": total_tokens,
        "tokens_per_s": total_tokens / total_s,
    }
    reasons: Dict[str, str] = {}
    for name in MEASURED_FIELDS:
        column: List[Optional[int]] = [step[name] for step in steps]
        if any(value is None for value in column):
            record[name] = None
            found = (step["unavailable"][name] for step in steps if name in step["unavailable"])
            reasons[name] = "; ".join(dict.fromkeys(found))
        elif name in PEAK_FIELDS:
            record[name] = max(value for value in column if value is not None)
        else:
            record[name] = sum(value for value in column if value is not None)
    record["unavailable"] = reasons
    return record


#: ``counters()`` fields that only grow: a step reports the difference.
CUMULATIVE_COUNTERS = ("layer_loads", "nf4_restores", "spill_written_bytes", "spill_read_bytes")

NO_OS_COUNTER = "no operating-system counter was supplied"
NO_DEVICE_COUNTER = "no device read counter was supplied, or the path has no block device"


# ==========================================================================
# counters read from outside the training code
# ==========================================================================
def process_io(
    *,
    proc_root: str = "/proc",
    platform: str = sys.platform,
    psutil_io: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Storage bytes this process has read and written since it started, from the kernel.

    Linux reads ``read_bytes`` and ``write_bytes`` of ``/proc/self/io``: bytes that reached
    the storage layer, not the ``rchar`` of every read call. Elsewhere psutil's
    ``io_counters`` (Windows counts every read the process makes, from any device).
    A counter that cannot be read is ``None``, with ``reason``.
    """
    none = {"read_bytes": None, "write_bytes": None}
    if platform.startswith("linux"):
        path = f"{proc_root}/self/io"
        try:
            with open(path, encoding="ascii") as handle:
                found = dict(line.split(":", 1) for line in handle if ":" in line)
            return {
                "read_bytes": int(found["read_bytes"]),
                "write_bytes": int(found["write_bytes"]),
                "reason": None,
            }
        except (OSError, KeyError, ValueError) as exc:
            return {**none, "reason": f"cannot read {path}: {type(exc).__name__}: {exc}"}
    try:
        if psutil_io is None:
            import psutil

            def psutil_io() -> Any:
                return psutil.Process().io_counters()

        counters = psutil_io()
        return {"read_bytes": int(counters.read_bytes), "write_bytes": int(counters.write_bytes),
                "reason": None}
    except Exception as exc:  # psutil raises its own errors, AttributeError where absent
        return {**none, "reason": f"io_counters unavailable: {type(exc).__name__}: {exc}"}


def process_read_counter() -> Optional[int]:
    return process_io()["read_bytes"]


def process_write_counter() -> Optional[int]:
    return process_io()["write_bytes"]


def device_read_counter(
    path: str,
    *,
    disk_device: Callable[[str], Optional[str]] = arm_validity.disk_device,
    disk_reads: Callable[[str], Optional[int]] = arm_validity.disk_reads,
) -> Callable[[], Optional[int]]:
    """A counter of every byte the drive that holds ``path`` has read, by any process."""

    def counter() -> Optional[int]:
        device = disk_device(path)
        return None if device is None else disk_reads(device)

    return counter


def _total(source: Optional[Callable[[], Optional[int]]]) -> Optional[int]:
    return None if source is None else source()


def _delta(
    name: str,
    before: Optional[int],
    after: Optional[int],
    reasons: Dict[str, str],
    missing: str,
) -> Optional[int]:
    if before is None or after is None:
        reasons[name] = missing
        return None
    if after < before:
        reasons[name] = "the counter went backwards"
        return None
    return after - before


def _on_cuda(device: str) -> bool:
    import torch

    return device == "cuda" and torch.cuda.is_available()


def _counter_delta(
    name: str, before: Mapping[str, int], after: Mapping[str, int], reasons: Dict[str, str]
) -> Optional[int]:
    if name not in before or name not in after:
        reasons[name] = f"the trainer's counters have no {name}"
        return None
    return after[name] - before[name]


def _weight_reads(
    os_read: Optional[int], spill_read: Optional[int], reasons: Dict[str, str]
) -> Optional[int]:
    """Weight bytes from disk: the process's storage reads minus what the spill read back."""
    if os_read is None:
        reasons["weight_read_bytes"] = f"os_read_bytes unavailable: {reasons['os_read_bytes']}"
        return None
    if spill_read is None:
        reasons["weight_read_bytes"] = "spill_read_bytes unavailable, so it cannot be subtracted"
        return None
    if spill_read > os_read:
        reasons["weight_read_bytes"] = (
            f"the spill read {spill_read} bytes, more than the {os_read} the OS counted"
        )
        return None
    return os_read - spill_read


def measure_step(
    body: Callable[[], Any],
    *,
    attention_mask: Any,
    device: str,
    counters: Callable[[], Mapping[str, int]],
    read_bytes: Optional[Callable[[], Optional[int]]] = None,
    write_bytes: Optional[Callable[[], Optional[int]]] = None,
    device_read_bytes: Optional[Callable[[], Optional[int]]] = None,
) -> Dict[str, Any]:
    """Run one step and return its ``step_record``.

    ``counters()`` returns ``layer_loads``, ``nf4_restores``, ``spill_written_bytes`` and
    ``spill_read_bytes`` as running totals, and ``pinned_store_bytes`` as a level. A key it
    lacks gives a null with a reason. ``read_bytes()`` and ``write_bytes()`` are the
    operating system's running totals for this process (``process_read_counter``,
    ``process_write_counter``), ``device_read_bytes()`` is the drive's
    (``device_read_counter``). The weights' bytes are the process's reads minus the spill's
    reads. That subtraction holds for the direct-I/O spill, whose reads the kernel counts; a
    buffered spill reads from the page cache, so only use the figure with ``direct`` spill.
    Page-cache hits are not storage reads either, so a warm run undercounts the weights.
    A source that is missing, or goes backwards, gives null with a reason. VRAM peaks
    are null off CUDA. The RSS peak is the step's own only where the platform can reset it.
    """
    import torch

    on_cuda = _on_cuda(device)
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rss_reset = a4_box.reset_peak_rss()
    before = dict(counters())
    os_before = (_total(read_bytes), _total(write_bytes), _total(device_read_bytes))
    started = time.perf_counter()
    body()
    if on_cuda:
        torch.cuda.synchronize()
    step_s = time.perf_counter() - started
    after = dict(counters())
    os_after = (_total(read_bytes), _total(write_bytes), _total(device_read_bytes))

    reasons: Dict[str, str] = {}
    allocated = reserved = None
    if on_cuda:
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
    else:
        reasons["peak_vram_allocated_bytes"] = reasons["peak_vram_reserved_bytes"] = (
            "no CUDA device"
        )
    peak_rss = a4_box.process_memory()["peak_rss"] if rss_reset else None
    if not rss_reset:
        reasons["peak_rss_bytes"] = "peak RSS is process-lifetime: VmHWM could not be reset"
    elif peak_rss is None:
        reasons["peak_rss_bytes"] = "the platform gives no peak RSS"
    os_read = _delta("os_read_bytes", os_before[0], os_after[0], reasons, NO_OS_COUNTER)
    os_write = _delta("os_write_bytes", os_before[1], os_after[1], reasons, NO_OS_COUNTER)
    device_read = _delta(
        "device_read_bytes", os_before[2], os_after[2], reasons, NO_DEVICE_COUNTER
    )
    delta = {name: _counter_delta(name, before, after, reasons) for name in CUMULATIVE_COUNTERS}
    weight_read = _weight_reads(os_read, delta["spill_read_bytes"], reasons)
    pinned = after.get("pinned_store_bytes")
    if pinned is None:
        reasons["pinned_store_bytes"] = "the trainer's counters have no pinned_store_bytes"
    return step_record(
        step_s=step_s,
        real_tokens=count_real_tokens(attention_mask),
        pinned_store_bytes=pinned,
        peak_vram_allocated_bytes=allocated,
        peak_vram_reserved_bytes=reserved,
        peak_rss_bytes=peak_rss,
        weight_read_bytes=weight_read,
        os_read_bytes=os_read,
        os_write_bytes=os_write,
        device_read_bytes=device_read,
        unavailable=reasons,
        **delta,
    )


# ==========================================================================
# the validity gate and the summary
# ==========================================================================
#: t quantile for a two-sided 95% interval, by degrees of freedom 1..30.
T_95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip

#: The normal quantile, used past 30 degrees of freedom.
Z_95 = 1.96


def run_entry(
    measurement: Mapping[str, Any],
    arm_record: Mapping[str, Any],
    *,
    arm: str,
    round: int,
    run: int,
) -> Dict[str, Any]:
    """One run of a block: its ``run_record`` joined to its F1 arm record.

    A run whose arm record is void is ``spoiled`` and names the checks that voided it and
    why; the summary leaves its whole block out. An unknown outcome counts, as in F1, and
    is marked ``unverified``.
    """
    outcome = str(arm_record["outcome"])
    checks = arm_record.get("checks") or {}
    return {
        "arm": arm,
        "round": round,
        "run": run,
        "outcome": outcome,
        "spoiled": outcome == arm_validity.VOID,
        "unverified": outcome == arm_validity.UNKNOWN,
        "void_reasons": {
            name: check.get("reason")
            for name, check in checks.items()
            if check.get("outcome") == arm_validity.VOID
        },
        "measurement": dict(measurement),
        "arm_record": dict(arm_record),
    }


def _interval(diffs: List[float]) -> Optional[List[float]]:
    if len(diffs) < 2:
        return None
    mean = statistics.fmean(diffs)
    sem = statistics.stdev(diffs) / math.sqrt(len(diffs))
    degrees = len(diffs) - 1
    quantile = T_95[degrees - 1] if degrees <= len(T_95) else Z_95
    return [mean - quantile * sem, mean + quantile * sem]


def summarize(blocks: Sequence[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """The summary of blocks of runs in the order A, B, B, A, by the F1 repeats rule.

    Blocks with a spoiled run stay out. Speed is tokens without padding per second. The
    difference of A and B is the mean of the per-block differences with a 95% interval from
    the t distribution; it is null with a reason when fewer than 2 blocks are valid.
    """
    order = list(arm_validity.BLOCK_ORDER)
    for block in blocks:
        if [entry["arm"] for entry in block] != order:
            raise ValueError("a block is four runs in the order A, B, B, A")
    runs = [
        [
            arm_validity.Run(
                entry["measurement"]["tokens_per_s"],
                arm_validity.VOID if entry["spoiled"] else entry["outcome"],
            )
            for entry in block
        ]
        for block in blocks
    ]
    valid = {
        i for i, spans in enumerate(runs) if all(run.outcome != arm_validity.VOID for run in spans)
    }
    kept = [block for i, block in enumerate(blocks) if i in valid]
    summary: Dict[str, Any] = {
        "blocks": len(blocks),
        "valid_blocks": len(kept),
        "spoiled_blocks": [i for i in range(1, len(blocks) + 1) if i - 1 not in valid],
        "unverified_runs": sum(entry["unverified"] for block in kept for entry in block),
        "blocks_to_add": arm_validity.blocks_to_add(runs),
    }
    reasons: Dict[str, str] = {}
    values = [[entry["measurement"]["tokens_per_s"] for entry in block] for block in kept]
    diffs = [(v[0] + v[3]) / 2 - (v[1] + v[2]) / 2 for v in values]
    summary["mean_a_tokens_per_s"] = (
        statistics.fmean(v[i] for v in values for i in (0, 3)) if values else None
    )
    summary["mean_b_tokens_per_s"] = (
        statistics.fmean(v[i] for v in values for i in (1, 2)) if values else None
    )
    summary["difference_a_minus_b"] = {
        "mean": statistics.fmean(diffs) if len(diffs) >= arm_validity.MIN_BLOCKS else None,
        "interval_95": _interval(diffs),
    }
    if len(kept) < arm_validity.MIN_BLOCKS:
        why = f"needs at least {arm_validity.MIN_BLOCKS} valid blocks, got {len(kept)}"
        for name in ("difference_a_minus_b", "sigma_tokens_per_s", "blocks_needed"):
            reasons[name] = why
        summary["sigma_tokens_per_s"] = summary["blocks_needed"] = None
    else:
        rule = arm_validity.repeats(runs)
        summary["sigma_tokens_per_s"] = rule.sigma
        summary["blocks_needed"] = rule.blocks_needed
    if not values:
        reasons["mean_a_tokens_per_s"] = reasons["mean_b_tokens_per_s"] = "no valid block"
    summary["unavailable"] = reasons
    return summary


# ==========================================================================
# the launch passport
# ==========================================================================
PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors",
            "pydantic", "numpy", "psutil", "rich")
NVIDIA_SMI_TIMEOUT_S = 20.0
REPO_ROOT = Path(__file__).resolve().parents[2]


def package_versions(names: Sequence[str]) -> Dict[str, Optional[str]]:
    """Installed version of each package, ``None`` for one that is not installed."""
    found: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def gpu_facts() -> Dict[str, Any]:
    """Name, memory and driver of the GPU the workload sees, from nvidia-smi."""
    query = "name,memory.total,driver_version"
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    workload = arm_validity.workload_gpu()
    if workload["id"] is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES leaves the workload no GPU")
    command.append(f"--id={workload['id']}")
    done = subprocess.run(command, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S)
    if done.returncode != 0:
        raise RuntimeError(f"nvidia-smi exited {done.returncode}: {done.stderr.strip()}")
    name, memory, driver = (part.strip() for part in done.stdout.strip().split(","))
    return {"name": name, "memory_total_mib": int(memory), "driver": driver}


def git_commit(root: Path = REPO_ROOT) -> Optional[Dict[str, Any]]:
    """The commit hash of ``root`` and whether tracked files differ from it."""

    def git(*args: str) -> Optional[str]:
        try:
            done = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    if head is None or status is None:
        return None
    return {"hash": head.strip(), "dirty": bool(status.strip())}


def launch_passport(
    *,
    argv: Sequence[str],
    package_names: Sequence[str] = PACKAGES,
    packages: Callable[[Sequence[str]], Mapping[str, Optional[str]]] = package_versions,
    gpu: Callable[[], Optional[Mapping[str, Any]]] = gpu_facts,
    container_ram: Callable[[], Optional[Mapping[str, Any]]] = a4_box.ram_limit,
    commit: Callable[[], Optional[Mapping[str, Any]]] = git_commit,
) -> Dict[str, Any]:
    """What the run was started with: the whole command, package versions, GPU, container
    RAM and commit. A source that fails or finds nothing is ``None`` with its reason."""
    reasons: Dict[str, str] = {}

    def read(name: str, source: Callable[[], Any], empty: str) -> Any:
        try:
            found = source()
        except Exception as exc:  # a missing tool or a refused read must not end the run
            reasons[name] = f"{type(exc).__name__}: {exc}"
            return None
        if found is None:
            reasons[name] = empty
        return found

    versions = read("packages", lambda: dict(packages(package_names)), "no package list")
    if versions is not None and any(version is None for version in versions.values()):
        missing = sorted(name for name, version in versions.items() if version is None)
        reasons["packages"] = f"not installed: {', '.join(missing)}"
    ram = read("container_ram", container_ram, "no container RAM reading")
    if ram is not None and ram.get("limit_bytes") is None:
        reasons["container_ram"] = (
            "container_ram: no cgroup memory limit found (unlimited or unknown)"
        )
    return {
        "command": shlex.join(argv),
        "argv": list(argv),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": versions,
        "gpu": read("gpu", gpu, "nvidia-smi gave no GPU"),
        "container_ram": ram,
        "commit": read("commit", commit, "not a git checkout"),
        "unavailable": reasons,
    }


def write_report(
    directory: Path,
    *,
    passport: Mapping[str, Any],
    blocks: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, Path]:
    """Write ``record.json`` (passport and every run) and ``summary.json`` beside it.

    The directory may exist but neither file may: a record is never overwritten.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"record": directory / "record.json", "summary": directory / "summary.json"}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"{path} exists; a record is never overwritten")
    documents = {
        "record": {"passport": passport, "blocks": blocks},
        "summary": summarize(blocks),
    }
    for name, path in paths.items():
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(documents[name], indent=2, allow_nan=False) + "\n")
    return paths
