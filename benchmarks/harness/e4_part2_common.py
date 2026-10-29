"""E4 part 2 - shared pieces of the calibration harness and the Mac-side quantization.

Rules: ``e4-part2-calibration-rule.md`` (v4) and ``e4-rule-part2-draft.md`` (v4.2).
Nothing here touches a GPU at import time; heavy libraries are imported inside functions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parents[1]
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

MODEL_REPO = "allenai/OLMoE-1B-7B-0924"
MODEL_REV = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
LIBRARIES = {
    "torch": "2.14.1",
    "transformers": "5.19.0",
    "bitsandbytes": "0.50.2",
    "peft": "0.21.2",
}
PYTHON_SERIES = "3.12"

#: SHA-256 of ``benchmarks/results/e4-part2-quant/quant-manifest.json``, as pinned in section 1
#: of the calibration rule. Every run's input check is bound to it.
QUANT_MANIFEST_SHA256 = "1f3b550d707aa18c098851a1718f1712cdb1162adc28552c5fbbcce8611f0ccb"

#: One seed for everything the calibration rule draws (rule sections 4, 5).
CAL_SEED = 20261015
EXPERTS_IMPLEMENTATION = "grouped_mm"
#: Same in every arm and every stage; the resolved value is recorded like the experts one.
ATTENTION_IMPLEMENTATION = "sdpa"

# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor) -> str:
    """SHA-256 of a tensor's raw bytes (dtype and shape are hashed too)."""
    import torch

    flat = tensor.detach().contiguous().cpu()
    header = f"{flat.dtype}|{tuple(flat.shape)}|".encode()
    raw = flat.reshape(-1).view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    return sha256_bytes(header + raw)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# provenance (calibration rule section 1: commits and the harness tree hash in every record)
# --------------------------------------------------------------------------


HARNESS_REL = "benchmarks/harness"


def _git(*args: str, repo: Path = REPO_ROOT, strip: bool = True) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        return ""
    return out.stdout.strip() if strip else out.stdout


def provenance(repo: Path = REPO_ROOT) -> Dict[str, Any]:
    """Soup commit, harness commit, harness tree hash and whether the repo tree is clean.

    ``harness_tree`` is the git tree object of ``benchmarks/harness`` at HEAD. The dirty check
    covers the whole repository tree (staged, modified and untracked files, both rule files,
    tests/ and the split manifests alike) except paths git ignores; ``dirty_paths`` lists them
    and ``validate_provenance`` turns a dirty tree into a void.
    """
    status = _git("status", "--porcelain", "--untracked-files=all", repo=repo, strip=False)
    dirty_paths = sorted({line[3:].split(" -> ")[-1].strip('"')
                          for line in status.splitlines() if line.strip()})
    return {
        "soup_commit": _git("rev-parse", "HEAD", repo=repo),
        "harness_commit": _git("log", "-1", "--format=%H", "--", HARNESS_REL, repo=repo),
        "harness_tree": _git("rev-parse", f"HEAD:{HARNESS_REL}", repo=repo),
        "harness_dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def validate_provenance(record: Dict[str, Any]) -> List[str]:
    """Void reasons for a provenance record (rule section 8, row Provenance)."""
    reasons = []
    for key in ("soup_commit", "harness_commit", "harness_tree"):
        if not record.get(key):
            reasons.append(f"provenance: {key} missing")
    if record.get("harness_dirty"):
        paths = record.get("dirty_paths") or []
        shown = ", ".join(paths[:5]) + (f" and {len(paths) - 5} more" if len(paths) > 5 else "")
        reasons.append(f"provenance: the repository tree has uncommitted changes: {shown}")
    return reasons


# --------------------------------------------------------------------------
# fp32 references (calibration rule section 1)
# --------------------------------------------------------------------------


def set_fp32_reference_mode() -> Dict[str, Any]:
    """TF32 off everywhere, highest matmul precision. Returns the recorded flags."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return fp32_mode_record()


def fp32_mode_record() -> Dict[str, Any]:
    import torch

    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_cuda_enabled": bool(torch.is_autocast_enabled("cuda")),
        "autocast_cpu_enabled": bool(torch.is_autocast_enabled("cpu")),
    }


class Fp32SettingError(AssertionError):
    """An fp32 reference ran with a setting the rule forbids (rule section 8: T or W void)."""


def assert_fp32_reference(*tensors, where: str = "") -> Dict[str, Any]:
    """Assert autocast off, TF32 off, highest precision and fp32 outputs. Returns the flags."""
    flags = fp32_mode_record()
    wrong = []
    if flags["cuda_matmul_allow_tf32"]:
        wrong.append("torch.backends.cuda.matmul.allow_tf32 is True")
    if flags["cudnn_allow_tf32"]:
        wrong.append("torch.backends.cudnn.allow_tf32 is True")
    if flags["float32_matmul_precision"] != "highest":
        wrong.append(f"matmul precision is {flags['float32_matmul_precision']!r}, not 'highest'")
    if flags["autocast_cuda_enabled"] or flags["autocast_cpu_enabled"]:
        wrong.append("autocast is enabled")
    for index, tensor in enumerate(tensors):
        if tensor.dtype.is_floating_point and str(tensor.dtype) != "torch.float32":
            wrong.append(f"output {index} has dtype {tensor.dtype}, not torch.float32")
    if wrong:
        raise Fp32SettingError(f"fp32 reference {where}: " + "; ".join(wrong))
    return flags


# --------------------------------------------------------------------------
# attempts journal (calibration rule section 1, section 8)
# --------------------------------------------------------------------------

OUTCOME_OK = "ok"
OUTCOME_VOID = "void"
OUTCOME_FAILED = "failed"  # a completed run whose result is a rule failure (e.g. non-finite F)
OUTCOMES = (OUTCOME_OK, OUTCOME_VOID, OUTCOME_FAILED)


class CalibrationStopped(RuntimeError):  # noqa: N818 - named for the rule's wording
    """A second void of the same run: calibration stops and is reported (rule section 8)."""


class AttemptsJournal:
    """Append-only ``attempts.jsonl``. Every attempt is journaled, voids included.

    An attempt that has a ``start`` and no ``end`` (crash, kill, OOM of the whole process)
    is closed as a void the next time the journal is opened for that run, and the closing
    record says so. Nothing is ever rewritten or deleted.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def events(self) -> List[dict]:
        return read_jsonl(self.path)

    def start(self, run_id: str, kind: str, **meta: Any) -> str:
        self.close_interrupted(run_id)
        self.assert_may_start(run_id)
        attempt_id = uuid.uuid4().hex
        append_jsonl(self.path, {
            "event": "start", "attempt_id": attempt_id, "run_id": run_id, "kind": kind,
            "ts": now_utc(), "unix_s": time.time(), "pid": os.getpid(), "meta": meta,
        })
        return attempt_id

    def end(self, attempt_id: str, outcome: str, reasons: Optional[List[str]] = None,
            **meta: Any) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        if outcome == OUTCOME_VOID and not reasons:
            raise ValueError("a void attempt needs at least one reason")
        append_jsonl(self.path, {
            "event": "end", "attempt_id": attempt_id, "outcome": outcome,
            "reasons": reasons or [], "ts": now_utc(), "meta": meta,
        })

    def attempts(self, run_id: str) -> List[dict]:
        """Attempts of one run in order: {attempt_id, outcome or None, reasons}."""
        by_id: Dict[str, dict] = {}
        order: List[str] = []
        for event in self.events():
            if event["event"] == "start" and event["run_id"] == run_id:
                by_id[event["attempt_id"]] = {
                    "attempt_id": event["attempt_id"], "outcome": None, "reasons": [],
                }
                order.append(event["attempt_id"])
            elif event["event"] == "end" and event["attempt_id"] in by_id:
                by_id[event["attempt_id"]]["outcome"] = event["outcome"]
                by_id[event["attempt_id"]]["reasons"] = event["reasons"]
        return [by_id[attempt_id] for attempt_id in order]

    def close_interrupted(self, run_id: str) -> List[str]:
        closed = []
        for attempt in self.attempts(run_id):
            if attempt["outcome"] is None:
                self.end(attempt["attempt_id"], OUTCOME_VOID,
                         ["interrupted: attempt has a start and no end (closed by the journal)"])
                closed.append(attempt["attempt_id"])
        return closed

    def void_count(self, run_id: str) -> int:
        return sum(a["outcome"] == OUTCOME_VOID for a in self.attempts(run_id))

    def assert_may_start(self, run_id: str) -> None:
        """A void run is rerun once; a second void of the same run stops calibration."""
        voids = self.void_count(run_id)
        if voids >= 2:
            raise CalibrationStopped(
                f"{run_id}: {voids} void attempts; calibration stops and is reported"
            )
        if any(a["outcome"] in (OUTCOME_OK, OUTCOME_FAILED) for a in self.attempts(run_id)):
            raise RuntimeError(f"{run_id}: already has a completed attempt; not rerun")

    def provenance_gap(self, expected_run_ids: List[str]) -> List[str]:
        """Runs with a result but no journal entry, or an unclosed attempt (rule section 8)."""
        gaps = []
        for run_id in expected_run_ids:
            attempts = self.attempts(run_id)
            if not attempts:
                gaps.append(f"{run_id}: no journal entry")
            elif any(a["outcome"] is None for a in attempts):
                gaps.append(f"{run_id}: attempt without an end")
        return gaps


# --------------------------------------------------------------------------
# box stamps (calibration rule section 1: per-run provenance of the machine)
# --------------------------------------------------------------------------

#: Environment variables that carry a vast.ai instance id, first one set wins.
VAST_ID_VARIABLES = ("VAST_CONTAINERLABEL", "CONTAINER_ID", "VAST_CONTAINER_ID")


class GpuSelectionError(ValueError):
    """The run cannot name exactly one GPU for every sensor and the model."""


def _count_gpus() -> Optional[int]:
    out = _nvidia_query("uuid", None)
    return len(out.splitlines()) if out else None


def require_single_gpu(gpu_index: int, environ: Optional[Dict[str, str]] = None,
                       count_gpus: Callable[[], Optional[int]] = _count_gpus) -> str:
    """The physical GPU id of a run that sees exactly one GPU, addressed as ``cuda:0``.

    ``arm_validity`` names the workload GPU from the first entry of CUDA_VISIBLE_DEVICES and
    takes no index, so the process list, clocks and memory sensors can only follow that
    variable. A run is therefore one GPU: ``--gpu-index 0`` and a variable that exposes exactly
    one GPU (an unset variable counts only on a machine with a single GPU). Returns the id the
    sensors and ``nvidia-smi --id`` use, an index or a UUID.
    """
    env = os.environ if environ is None else environ
    if gpu_index != 0:
        raise GpuSelectionError(
            f"--gpu-index {gpu_index}: a run is one GPU, so use --gpu-index 0 and select the "
            "GPU with CUDA_VISIBLE_DEVICES")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        if count_gpus() != 1:
            raise GpuSelectionError(
                "CUDA_VISIBLE_DEVICES is unset and this machine does not have exactly one GPU; "
                "set it to exactly one GPU")
        return "0"
    entries = [e.strip() for e in visible.split(",")]
    if len(entries) != 1 or entries[0] in ("", "NoDevFiles") or entries[0].startswith("-"):
        raise GpuSelectionError(
            f"CUDA_VISIBLE_DEVICES={visible!r} must expose exactly one GPU")
    return entries[0]


def workload_gpu_id() -> Optional[str]:
    """The physical GPU the sensors read (``nvidia-smi --id``), None if no GPU is visible."""
    import arm_validity as av

    return av.workload_gpu()["id"]


def _nvidia_query(fields: str, gpu_id: Optional[str] = "0") -> Optional[str]:
    """``nvidia-smi --query-gpu``; ``gpu_id`` None queries every GPU."""
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    if gpu_id is not None:
        command.insert(1, f"--id={gpu_id}")
    try:
        out = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def instance_id(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    for name in VAST_ID_VARIABLES:
        if os.environ.get(name):
            return os.environ[name]
    return None


def driver_cuda_version(gpu_id: Optional[str] = "0") -> Optional[str]:
    """The highest CUDA version the driver supports, from the ``nvidia-smi`` header (the
    ``--query-gpu`` fields do not offer it). Asked of the workload GPU like every other field.
    None when it cannot be read."""
    command = ["nvidia-smi"] + ([f"--id={gpu_id}"] if gpu_id is not None else [])
    try:
        out = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", out.stdout) if out.returncode == 0 else None
    return match.group(1) if match else None


def box_stamp(gpu_index: int = 0, explicit_instance: Optional[str] = None,
              readers: Optional[Dict[str, Callable[[], Any]]] = None) -> Dict[str, Any]:
    """Machine state at one moment. A reading that is unavailable is None with a reason.

    ``readers`` replaces the sensors in tests. AC or battery is recorded as N/A (hosted).
    The GPU named in the stamp is the one the sensors read (CUDA_VISIBLE_DEVICES, see
    ``require_single_gpu``); ``gpu_index`` is 0 in every run and does not choose it.
    """
    import arm_validity as av

    sensors = {
        "ram_available_bytes": av.ram_available,
        "commit_charge": av.commit_charge,
        "gpu_processes": av.gpu_processes,
        "gpu_clock": av.gpu_clock,
        "gpu_memory_used_mib": av.gpu_memory_used,
    }
    sensors.update(readers or {})
    stamp: Dict[str, Any] = {"unix_s": time.time(), "power_source": "N/A (hosted instance)"}
    unknown: Dict[str, str] = {}
    for name, sensor in sensors.items():
        try:
            stamp[name] = sensor()
        except Exception as exc:  # noqa: BLE001 - a missing sensor is an unknown, not a crash
            stamp[name] = None
            unknown[name] = f"{type(exc).__name__}: {exc}"
    stamp["own_pid"] = os.getpid()
    stamp["instance_id"] = instance_id(explicit_instance)
    gpu_id = workload_gpu_id()
    stamp["gpu_name"] = _nvidia_query("name", gpu_id) if gpu_id is not None else None
    stamp["gpu_uuid"] = _nvidia_query("uuid", gpu_id) if gpu_id is not None else None
    stamp["driver_version"] = (_nvidia_query("driver_version", gpu_id)
                               if gpu_id is not None else None)
    stamp["gpu_memory_total_mib"] = (_nvidia_query("memory.total", gpu_id)
                                     if gpu_id is not None else None)
    stamp["gpu_compute_cap"] = (_nvidia_query("compute_cap", gpu_id)
                                if gpu_id is not None else None)
    stamp["driver_cuda_version"] = driver_cuda_version(gpu_id) if gpu_id is not None else None
    try:
        import torch

        stamp["torch_cuda"] = torch.version.cuda
        stamp["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        stamp["torch_cuda"] = None
        unknown["torch"] = f"{type(exc).__name__}: {exc}"
    stamp["unknown"] = unknown
    return stamp


# These bounds define the rule's card names ("A100 40 GB" and "A100 80 GB", calibration rule
# sections 1 and 3) as values `nvidia-smi` reports for the workload GPU. They change no
# threshold: the 36,000,000,000-byte limit and every other number stay as the rule has them.
CARD_NAME_MARKER = "A100"  # SXM4 and PCIe variants both qualify
CARD_COMPUTE_CAP = 8.0
CARD_MEMORY_MIB = {"F": (40_000, 42_000), "F80": (80_000, 82_000)}  # inclusive, per F prefix


def card_class_problems(stamp: Dict[str, Any], prefix: str) -> List[str]:
    """``field=value ...`` for each field of a box stamp outside the card class of ``prefix``."""
    problems = []
    name = stamp.get("gpu_name")
    if not isinstance(name, str) or CARD_NAME_MARKER not in name:
        problems.append(f"gpu_name={name} does not name an {CARD_NAME_MARKER}")
    cap = stamp.get("gpu_compute_cap")
    try:
        cap_ok = float(cap) == CARD_COMPUTE_CAP
    except (TypeError, ValueError):
        cap_ok = False
    if not cap_ok:
        problems.append(f"gpu_compute_cap={cap} is not {CARD_COMPUTE_CAP}")
    low, high = CARD_MEMORY_MIB[prefix]
    mem = stamp.get("gpu_memory_total_mib")
    try:
        mem_ok = low <= float(mem) <= high
    except (TypeError, ValueError):
        mem_ok = False
    if not mem_ok:
        problems.append(f"gpu_memory_total_mib={mem} is outside {low}-{high} MiB")
    return problems


def foreign_gpu_processes(stamp: Dict[str, Any], own_pids: List[int]) -> List[dict]:
    """GPU processes in a stamp that are not this run's own (rule section 8, Peers)."""
    return [p for p in (stamp.get("gpu_processes") or []) if p.get("pid") not in own_pids]


def validate_run_stamps(before: Dict[str, Any], after: Dict[str, Any],
                        own_pids: List[int], require_gpu: bool = True) -> List[str]:
    """Void reasons for a run's pair of box stamps (Provenance and Peers rows)."""
    reasons = []
    if require_gpu:
        for key in ("gpu_uuid", "instance_id"):
            if not before.get(key):
                reasons.append(f"provenance: {key} missing from box_stamp_before")
        if before.get("gpu_processes") is None:
            reasons.append("peers: GPU process list unavailable before the run")
        if after.get("gpu_processes") is None:
            reasons.append("peers: GPU process list unavailable after the run")
    for label, stamp in (("before", before), ("after", after)):
        foreign = foreign_gpu_processes(stamp, own_pids)
        if foreign:
            reasons.append(f"peers: foreign GPU process {label} the run: {foreign}")
    if "unix_s" not in before:
        reasons.append("provenance: box_stamp_before.unix_s missing")
    return reasons


class PeersUnidentifiableError(RuntimeError):
    """This process's own GPU context could not be told apart from other processes'."""


def identify_own_gpu_pids(read, allocate, release, sizes_mib=(320, 704), tolerance_mib=2
                          ) -> Dict[str, Any]:
    """Find this process among the GPU processes without trusting PIDs.

    Inside a container NVML and ``nvidia-smi`` list host PIDs, which ``os.getpid()`` cannot
    match. So the process proves which entry is its own by memory: for each probe size it
    allocates that much device memory, re-reads the process list, and keeps the PIDs whose
    used memory grew by that amount; the memory is then released. The PIDs that grew by the
    probe size in *every* probe are this process. Exactly one must remain; none, several, or
    a driver that hides per-process memory raises ``PeersUnidentifiableError`` (the run is a
    void, never a guess).

    ``read()`` returns [{"pid", "used_mib"}, ...]; ``allocate(mib)`` returns a handle for
    ``release``. The CUDA context must exist before the first probe.
    """
    def usage() -> Dict[int, int]:
        out = {}
        for proc in read():
            if proc.get("used_mib") is None:
                raise PeersUnidentifiableError(
                    "the driver reports no per-process GPU memory, so the own context cannot "
                    "be identified")
            out[int(proc["pid"])] = int(proc["used_mib"])
        return out

    probes = []
    surviving = None
    for size in sizes_mib:
        before = usage()
        handle = allocate(size)
        try:
            after = usage()
        finally:
            release(handle)
        grew = sorted(pid for pid, used in after.items()
                      if abs((used - before.get(pid, 0)) - size) <= tolerance_mib)
        probes.append({"size_mib": size, "pids_that_grew_by_that_much": grew})
        surviving = set(grew) if surviving is None else surviving & set(grew)
    if not surviving:
        raise PeersUnidentifiableError(
            "no process in the GPU process list grew by every probe size; this process's "
            f"context is not visible there (probes: {probes})")
    if len(surviving) > 1:
        raise PeersUnidentifiableError(
            f"more than one process grew by every probe size: {sorted(surviving)}")
    return {
        "own_pids": sorted(surviving),
        "method": (f"allocation probe: allocated {list(sizes_mib)} MiB in turn and took the "
                   "PID whose per-process GPU memory grew by the same amount each time "
                   f"(tolerance {tolerance_mib} MiB); needed because NVML lists host PIDs "
                   "inside a container"),
        "probes": probes,
    }


def identify_own_gpu_pids_cuda(gpu_index: int = 0, **kw) -> Dict[str, Any]:
    """``identify_own_gpu_pids`` on the real device: torch allocations, ``nvidia-smi`` list."""
    import arm_validity as av
    import torch

    torch.cuda.init()
    torch.zeros(1, device=f"cuda:{gpu_index}")  # create the context before the baseline
    held: List[Any] = []

    def allocate(mib: int):
        held.append(torch.empty(mib * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{gpu_index}"))
        torch.cuda.synchronize()
        return len(held) - 1

    def release(_handle) -> None:
        held.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def read():
        return av.gpu_processes() or []

    return identify_own_gpu_pids(read, allocate, release, **kw)


def library_versions() -> Dict[str, Any]:
    """Installed versions of the pinned stack, and which ones differ from the rule."""
    import importlib.metadata as md

    found = {}
    for name in LIBRARIES:
        try:
            found[name] = md.version(name)
        except md.PackageNotFoundError:
            found[name] = None
    found["python"] = ".".join(map(str, sys.version_info[:3]))
    mismatches = [n for n, v in LIBRARIES.items() if found[n] is None or not found[n].startswith(v)]
    if not found["python"].startswith(PYTHON_SERIES):
        mismatches.append("python")
    return {"versions": found, "mismatches": mismatches}


#: Fields every attempt record carries in ``identity`` (the task's list): GPU name, compute
#: capability, memory.total, driver, CUDA, torch, bitsandbytes, transformers, commit, dataset
#: revision and instance id. ``IDENTITY_OPTIONAL`` may be None without voiding the attempt.
IDENTITY_REQUIRED = (
    "gpu_name", "compute_capability", "memory_total_mib", "driver_version", "torch_cuda",
    "torch", "bitsandbytes", "transformers", "soup_commit", "dataset_revision", "instance_id",
)
IDENTITY_OPTIONAL = ("gpu_uuid", "driver_cuda_version", "peft", "python", "harness_commit",
                     "harness_tree")


def run_identity(stamp: Dict[str, Any], stack: Dict[str, Any], prov: Dict[str, Any],
                 dataset_revision: Optional[str]) -> Dict[str, Any]:
    """The flat identity of one attempt, from its first box stamp, the library versions, the
    provenance and the dataset revision. A value that is unknown is None."""
    versions = stack.get("versions", {})
    return {
        "gpu_name": stamp.get("gpu_name"),
        "compute_capability": stamp.get("gpu_compute_cap"),
        "memory_total_mib": stamp.get("gpu_memory_total_mib"),
        "driver_version": stamp.get("driver_version"),
        "driver_cuda_version": stamp.get("driver_cuda_version"),
        "torch_cuda": stamp.get("torch_cuda"),
        "torch": versions.get("torch"),
        "bitsandbytes": versions.get("bitsandbytes"),
        "transformers": versions.get("transformers"),
        "peft": versions.get("peft"),
        "python": versions.get("python"),
        "soup_commit": prov.get("soup_commit"),
        "harness_commit": prov.get("harness_commit"),
        "harness_tree": prov.get("harness_tree"),
        "dataset_revision": dataset_revision,
        "gpu_uuid": stamp.get("gpu_uuid"),
        "instance_id": stamp.get("instance_id"),
    }


def identity_problems(identity: Dict[str, Any]) -> List[str]:
    """Void reasons for an identity with a required field missing."""
    return [f"provenance: identity.{key} missing" for key in IDENTITY_REQUIRED
            if identity.get(key) in (None, "")]


def dry_run_mix_problem(out: Path | str, dry_run: bool) -> Optional[str]:
    """A directory holding both dry-run and real records is not a calibration record.

    Returns a description when ``out`` holds records of the other kind than ``dry_run``.
    """
    runs = Path(out) / "runs"
    if not runs.is_dir():
        return None
    for path in sorted(runs.glob("*.json")):
        found = bool(json.loads(path.read_text()).get("dry_run"))
        if found != dry_run:
            return (f"{path.name} is a {'dry-run' if found else 'real'} record but this is "
                    f"a {'dry run' if dry_run else 'real run'}; use another --out")
    return None


def finite_max(values) -> float:
    """``max`` that refuses NaN and infinity instead of letting argument order decide."""
    values = list(values)
    bad = [v for v in values if not math.isfinite(v)]
    if bad:
        raise ValueError(f"{len(bad)} value(s) are not finite: {bad[:3]}")
    return max(values)


def max_or_nan(values) -> float:
    """``max`` that is NaN when any value is not finite, whatever the order (a void marker)."""
    values = list(values)
    if not all(math.isfinite(v) for v in values):
        return float("nan")
    return max(values)
