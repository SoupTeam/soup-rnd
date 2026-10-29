#!/usr/bin/env python3
"""A3 stage 2: the GPU check of activation spill in ``training.l2l``, through the real trainer.

One session (``session``) orchestrates fresh child processes (``child``), each in its own
transient scope ``systemd-run --scope -p MemoryMax=<limit> -p MemorySwapMax=0``, so that every
child's ``memory.peak`` and ``memory.events`` are its own. Planning is the default: without
``--execute`` the session prints every child command in order and writes nothing.

Order: ``calibration`` (generous limit) and up to two ``calibration_check`` children find the
tight limit at which the product's own planner spills about half of Mistral-7B's 32 layers,
and the session stops when that limit leaves the build less room than its transient needs
(the build margin of ``oom_margin`` below 0); then ``l2l_nospill_1``, ``l2l_nospill_2``
(generous), ``l2l_spill`` (tight), ``ga_1``, ``ga_2``, ``ga_floor``; ``compare`` (CPU, the
unpublished gradient tensors); the cleanup children ``sigint``, ``kill9`` and ``kill9_next``
(tight); and, only when criterion 1 of ``a3_spill_rule`` passes, the A-B-B-A cost blocks.
Every L2L child trains through ``make_l2l_trainer_class(transformers.Trainer)`` +
``attach_l2l_state``; spill turns on only through the product's own RAM check under the
scope's limit. Nothing in the product is told the target. Every child writes a marker at its
first ``training_step`` entry, so a kill before it (the build) is told apart from one after it.

The rule is ``a3_spill_rule.py``; the protocol is benchmarks/a3-spill-gpu-check.md. Benchmark
code, not shipped; torch is imported inside functions, after the environment is set.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a3_spill_rule as rule  # noqa: E402
import arm_validity  # noqa: E402
import l2l_mistral_stand as base  # noqa: E402
import l2l_session  # noqa: E402

STAND = str(Path(__file__).resolve())
ROOT = l2l_session.ROOT
HARNESS = Path(__file__).resolve().parent
MIB = 1 << 20

K = 4
SEQ = 512
#: Correctness rows: mixed lengths, masked labels, an incomplete last window at k = 4.
ROWS = 10
COST_WARMUP = 2
COST_STEPS = 5
#: Of Mistral-7B's 32 layers, the calibration aims at this many spilled.
TARGET_SPILL = 16
MAX_CALIBRATION_CHECKS = 2
DEFAULT_GENEROUS = "24G"
#: Protocol v3 (§3.7): the signal children hold inside optimizer step 2, at the backward
#: recompute of the highest spilled layer (its first micro-batch), until a signal arrives or
#: HOLD_TIMEOUT_S passes; the parent signals once it reads the hold marker. v2 instead waited
#: 0.5 s after the step-2 entry marker, which did not show where the step was.
HOLD_STEPS_DONE = 1
HOLD_TIMEOUT_S = 120.0
HOLD_POLL_S = 0.05
SIGNAL_EXIT_TIMEOUT_S = 600.0
MARKER_POLL_S = 0.05
#: The calibration and check children sample their scope's memory this often (s).
SAMPLE_S = 0.05
REQUIRED_CHECKS = ("suspend", "power", "foreign_reads")
TIGHT = "<TIGHT>"

#: name -> what the child does. ``limit``: the scope it runs in; ``group``: the subdirectory
#: of --spill-dir it spills to (kill9 and kill9_next share one); ``folder``: where its JSON goes.
ARMS: Dict[str, Dict[str, Any]] = {
    "calibration": {"kind": "calibrate", "l2l": True, "limit": "generous",
                    "group": "calibration", "folder": ""},
    **{f"calibration_check_{i}": {"kind": "check", "l2l": True, "limit": "tight",
                                  "group": "calibration", "folder": ""}
       for i in range(1, MAX_CALIBRATION_CHECKS + 1)},
    "l2l_nospill_1": {"kind": "correctness", "l2l": True, "limit": "generous"},
    "l2l_nospill_2": {"kind": "correctness", "l2l": True, "limit": "generous"},
    "l2l_spill": {"kind": "correctness", "l2l": True, "limit": "tight"},
    "ga_1": {"kind": "correctness", "l2l": False, "limit": "generous"},
    "ga_2": {"kind": "correctness", "l2l": False, "limit": "generous"},
    "ga_floor": {"kind": "correctness", "l2l": False, "limit": "generous",
                 "sdpa": base.FLOOR_SDPA_BACKEND},
    "sigint": {"kind": "cleanup", "l2l": True, "limit": "tight", "signal": "SIGINT"},
    "kill9": {"kind": "cleanup", "l2l": True, "limit": "tight", "signal": "SIGKILL"},
    "kill9_next": {"kind": "cleanup", "l2l": True, "limit": "tight", "group": "kill9"},
    "A": {"kind": "cost", "l2l": True, "limit": "generous", "group": "cost"},
    "B": {"kind": "cost", "l2l": True, "limit": "tight", "group": "cost"},
}
FOLDERS = {"correctness": "arms", "cleanup": "cleanup", "cost": "cost"}
CORRECTNESS_ORDER = ("l2l_nospill_1", "l2l_nospill_2", "l2l_spill", "ga_1", "ga_2", "ga_floor")
CLEANUP_ORDER = ("sigint", "kill9", "kill9_next")


class SessionStopError(RuntimeError):
    """The session cannot go on; the message says why."""


class OomRiskError(SessionStopError):
    """The tight limit leaves the build less room than its transient needs (build margin
    below 0, protocol §4); the session stops before any child at that limit and the limit is
    not raised."""

    state = "stopped_calibration_oom_risk"


class OomInTrainingError(SessionStopError):
    """The kernel OOM-killed a child at the tight limit after its first ``training_step``
    entry (protocol §3.6, rule v2): criterion 2 fails and the session stops."""

    state = "stopped_oom_in_training"


def spec_of(arm: str) -> Dict[str, Any]:
    spec = dict(ARMS[arm])
    spec.setdefault("sdpa", base.PINNED_SDPA_BACKEND)
    spec.setdefault("group", arm)
    spec.setdefault("folder", FOLDERS.get(spec["kind"], ""))
    return spec


# ==========================================================================
# small pure helpers
# ==========================================================================
def parse_size(text: str) -> int:
    """Bytes of ``24G``, ``512M``, ``1T`` (binary, as systemd reads them) or a plain integer,
    rounded down to a whole MiB so that the kernel's page rounding of ``memory.max`` never
    changes the number read back."""
    raw = str(text).strip().upper().removesuffix("B")
    scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}.get(raw[-1:], 1)
    number = raw[:-1] if scale != 1 else raw
    value = int(float(number) * scale)
    if value < MIB:
        raise ValueError(f"a memory limit of {text!r} is below 1 MiB")
    return value // MIB * MIB


def jsonable(value: Any) -> Any:
    """JSON-safe copy: non-finite floats become their names ('nan', 'inf'), tuples lists."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, data: Any) -> None:
    """LF, no NaN, written whole then moved into place (a reader never sees half a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(jsonable(data), indent=1, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    os.replace(temp, path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ==========================================================================
# cgroup v2 readback of this process's own scope (V3)
# ==========================================================================
def _text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _number(raw: Optional[str]) -> Any:
    """An integer, the literal ``max``, or None when unreadable."""
    if raw is None:
        return None
    return int(raw) if raw.isdigit() else raw


#: memory.stat counters the readback keeps.
STAT_KEYS = ("anon", "shmem", "active_file", "inactive_file", "slab_reclaimable",
             "file_dirty", "file_writeback")


def unreclaimable_bytes(current: Any, stat: Dict[str, int]) -> Optional[int]:
    """What memcg reclaim cannot free with swap off: ``memory.current`` minus the clean file
    cache (active and inactive file pages less the dirty and writeback ones, which cannot be
    dropped at once) and minus reclaimable slab. Anon, shmem and pinned pages stay in it."""
    needed = ("active_file", "inactive_file", "slab_reclaimable", "file_dirty", "file_writeback")
    if not isinstance(current, int) or any(key not in stat for key in needed):
        return None
    clean = max(0, stat["active_file"] + stat["inactive_file"] - stat["file_dirty"]
                - stat["file_writeback"])
    return max(0, current - clean - stat["slab_reclaimable"])


def cgroup_readback(*, proc_root: str = "/proc", v2_root: str = "/sys/fs/cgroup"
                    ) -> Dict[str, Any]:
    """The own cgroup v2 group of this process (``0::/path`` in /proc/self/cgroup): its
    ``memory.max``, ``memory.swap.max``, ``memory.current``, ``memory.peak``, the
    ``memory.stat`` counters in ``STAT_KEYS``, ``memory.events``, the working set as the
    product counts it (current - inactive_file), anon + shmem, and the unreclaimable bytes.
    ``version`` None: no cgroup v2 line (not under a scope)."""
    text = _text(f"{proc_root}/self/cgroup") or ""
    path = next((line[3:] for line in text.splitlines() if line.startswith("0::")), None)
    if path is None:
        return {"version": None, "path": None}
    directory = v2_root + ("" if path == "/" else path.rstrip("/"))
    events: Dict[str, int] = {}
    for line in (_text(f"{directory}/memory.events") or "").splitlines():
        key, _, value = line.partition(" ")
        if value.strip().isdigit():
            events[key] = int(value)
    stat: Dict[str, int] = {}
    for line in (_text(f"{directory}/memory.stat") or "").splitlines():
        key, _, value = line.partition(" ")
        if key in STAT_KEYS and value.strip().isdigit():
            stat[key] = int(value)
    current = _number(_text(f"{directory}/memory.current"))
    inactive = stat.get("inactive_file")
    return {
        "version": 2, "path": directory,
        "memory_max": _number(_text(f"{directory}/memory.max")),
        "swap_max": _number(_text(f"{directory}/memory.swap.max")),
        "current": current, "peak": _number(_text(f"{directory}/memory.peak")),
        "inactive_file": inactive, "stat": stat, "events": events,
        "working_set": current - inactive if isinstance(current, int) and inactive is not None
        else None,
        "anon_shmem": stat["anon"] + stat["shmem"] if "anon" in stat and "shmem" in stat
        else None,
        "unreclaimable": unreclaimable_bytes(current, stat),
    }


def process_memory(*, proc_root: str = "/proc") -> Dict[str, Optional[int]]:
    """This process's resident set (VmRSS) and its peak (VmHWM), in bytes; None where
    /proc/self/status does not give them."""
    out: Dict[str, Optional[int]] = {"rss": None, "hwm": None}
    names = {"VmRSS": "rss", "VmHWM": "hwm"}
    for line in (_text(f"{proc_root}/self/status") or "").splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if key in names and len(parts) == 2 and parts[0].isdigit() and parts[1] == "kB":
            out[names[key]] = int(parts[0]) * 1024
    return out


class MemorySampler:
    """The scope's memory over one child, for the OOM margin (protocol §4).

    ``sample(label)`` takes a named reading (``build_end``, ``first_step``, ``step_end``): the
    full cgroup readback, VmRSS/VmHWM and torch's pinned host allocator. With ``interval`` a
    daemon thread also takes a compact reading every ``interval`` seconds, so a transient
    between the named readings is seen. Each reading carries its phase: ``build`` until the
    first ``training_step`` entry, ``train`` from it on."""

    def __init__(self, interval: Optional[float] = None, *,
                 read: Callable[[], Dict[str, Any]] = cgroup_readback,
                 pinned: Callable[[], Optional[int]] = lambda: None,
                 process: Callable[[], Dict[str, Optional[int]]] = process_memory):
        self.interval = interval
        self.read, self.pinned, self.process = read, pinned, process
        self.phase = "build"
        self.samples: List[Dict[str, Any]] = []
        self.failed = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _row(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        phase = self.phase
        cgroup = self.read()
        return cgroup, {"t": time.monotonic(), "phase": phase,
                        "unreclaimable": cgroup.get("unreclaimable"),
                        "anon_shmem": cgroup.get("anon_shmem"), "current": cgroup.get("current"),
                        "pinned": self.pinned()}

    def sample(self, label: str) -> Dict[str, Any]:
        cgroup, row = self._row()
        row.update(label=label, cgroup=cgroup, process=self.process())
        self.samples.append(row)
        return row

    def first_step(self) -> Dict[str, Any]:
        self.phase = "train"
        return self.sample("first_step")

    def labeled(self) -> List[Dict[str, Any]]:
        return [row for row in self.samples if "label" in row]

    def compact(self) -> List[Dict[str, Any]]:
        return [{key: value for key, value in row.items() if key not in ("cgroup", "process")}
                for row in self.samples]

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.samples.append(self._row()[1])
            except Exception:  # noqa: BLE001 - a reading must never stop the child
                self.failed += 1

    def __enter__(self) -> "MemorySampler":
        if self.interval:
            self._thread = threading.Thread(target=self._loop, name="a3-memory-sampler",
                                            daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


# ==========================================================================
# calibration of the tight limit (pure)
# ==========================================================================
def working_set(record: Dict[str, Any]) -> Tuple[Optional[int], str]:
    """The scope's working set when the product planned the store: ``used - inactive_file``
    from the product's own ``limiting_numbers`` when its limiting level is this child's own
    scope, else the stand's readback at the first ``training_step`` entry (``memory.current -
    inactive_file``), read just before the product plans."""
    plan = record.get("spill_plan") or {}
    numbers = plan.get("limiting_numbers") or {}
    at_step = record.get("cgroup_at_first_step") or {}
    if (plan.get("limiting_level") not in (None, "psutil")
            and plan.get("limiting_level") == at_step.get("path")
            and isinstance(numbers.get("used"), int)):
        return (numbers["used"] - int(numbers.get("inactive_file") or 0),
                f"the product's plan-time numbers of {plan['limiting_level']}")
    if isinstance(at_step.get("working_set"), int):
        return at_step["working_set"], "the stand's readback at the first training_step entry"
    return None, "neither the product nor the readback gives a working set"


def _layout(plan: Dict[str, Any]) -> Any:
    """The product's own all-resident plan for the store this record planned."""
    from soup_cli.utils.l2l_spill import plan_spill

    return plan_spill(None, n_layers=int(plan["n_layers"]), k=int(plan["k"]),
                      chunk_bytes=int(plan["chunk_bytes"]), ring=int(plan["ring"]),
                      pin=bool(plan["pin"]))


def predicted_spill(limit: int, working: int, plan: Dict[str, Any]) -> int:
    """The spilled layers the product's planner gives at ``limit``, with the scope's
    remainder ``limit - working`` as the available memory, as ``ram_budget`` counts it."""
    from soup_cli.utils.l2l_spill import budget_of, plan_spill

    layout = _layout(plan)
    budget = budget_of(max(0, limit - working))
    return plan_spill(budget, n_layers=layout.n_layers, k=layout.k,
                      chunk_bytes=layout.chunk_bytes, ring=layout.ring, pin=layout.pin
                      ).spill_layers


def available_for(budget: int) -> int:
    """The least available memory whose product budget (``l2l_spill.budget_of``) is at least
    ``budget``: the larger of budget / 0.7 and budget + the reserve."""
    from soup_cli.utils.l2l_spill import RAM_RESERVE_BYTES
    from soup_cli.utils.layer_stream import RAM_TIER_HEADROOM

    return max(math.ceil(budget / RAM_TIER_HEADROOM), int(budget) + RAM_RESERVE_BYTES)


def tight_limit(working: int, plan: Dict[str, Any], *, target: int = TARGET_SPILL
                ) -> Dict[str, Any]:
    """limit = working set + the available memory whose product budget is target host_bytes
    + half a layer, rounded up to a MiB (``available_for``).

    target host_bytes is what the product's plan holds with ``target`` layers spilled: the
    resident layers' pinned blocks plus the two staging rings. Half a layer's cost puts the
    budget in the middle of the interval that gives exactly ``target`` spilled layers, so a
    working set that moves by a few MiB between children does not cross a layer boundary."""
    layout = _layout(plan)
    per_layer = layout.all_resident_bytes // layout.n_layers
    host = (layout.n_layers - target) * per_layer + layout.minimum_bytes
    budget = host + per_layer // 2
    limit = -(-(int(working) + available_for(budget)) // MIB) * MIB
    return {"working_set_bytes": int(working), "per_layer_bytes": per_layer,
            "minimum_bytes": layout.minimum_bytes, "target_spill_layers": target,
            "target_host_bytes": host, "target_budget_bytes": budget, "limit_bytes": limit,
            "predicted_spill_layers": predicted_spill(limit, working, plan)}


def calibration_limit(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The first tight limit, from the calibration child (generous limit, nothing spilled)."""
    if not record or record.get("exit") != "normal" or not record.get("spill_plan"):
        raise SessionStopError(f"the calibration child did not plan a store: "
                               f"{(record or {}).get('exit')} {(record or {}).get('error')}")
    layers = record["spill_plan"]["spill_layers"]
    if layers:
        raise SessionStopError(f"the generous limit already spills {layers} layers; raise "
                               f"--generous-limit so the no-spill arms spill nothing")
    working, source = working_set(record)
    if working is None:
        raise SessionStopError(f"calibration: {source}")
    return {**tight_limit(working, record["spill_plan"]), "source": source}


def next_limit(record: Optional[Dict[str, Any]], limit: int, checks_done: int
               ) -> Tuple[str, Optional[int], Dict[str, Any]]:
    """After a check child at ``limit``: ``('done', limit, why)`` when its product plan spills
    a number in the band (or any number > 0 once the checks are spent), ``('again', new,
    why)`` with the limit recomputed from its own plan-time numbers, or ``('stop', None,
    why)``."""
    if not record or record.get("exit") != "normal" or not record.get("spill_plan"):
        return "stop", None, {"why": f"the check child at {limit:,} B failed: "
                                     f"{(record or {}).get('exit')} {(record or {}).get('error')}"}
    layers = int(record["spill_plan"]["spill_layers"])
    low, high = rule.SPILL_BAND
    if low <= layers <= high:
        return "done", limit, {"spill_layers": layers, "in_band": True}
    if checks_done >= MAX_CALIBRATION_CHECKS:
        if layers > 0:
            return "done", limit, {"spill_layers": layers, "in_band": False}
        return "stop", None, {"why": f"{checks_done} checks and the product spills nothing at "
                                     f"{limit:,} B"}
    working, source = working_set(record)
    if working is None:
        return "stop", None, {"why": source}
    again = tight_limit(working, record["spill_plan"])
    return "again", again["limit_bytes"], {"spill_layers": layers, "in_band": False,
                                           "recomputed": {**again, "source": source}}


def _rise(samples: Sequence[Dict[str, Any]], key: str, entry: Dict[str, Any],
          plan_host: int) -> Tuple[int, int]:
    """``(build, training)``: the largest rise of ``key`` over its value at ``entry``, before
    and after it, floored at 0. After it the store's pinned bytes are taken out: torch's
    pinned host allocator growth since the entry, at most the plan's ``host_bytes``."""
    base, pinned0 = int(entry[key]), entry.get("pinned")
    build, train = [0], [0]
    for row in samples:
        value = row.get(key)
        if value is None:
            continue
        if row["phase"] == "build":
            build.append(value - base)
            continue
        store = (0 if pinned0 is None or row.get("pinned") is None
                 else min(int(plan_host), max(0, row["pinned"] - pinned0)))
        train.append(value - base - store)
    return max(build), max(train)


def transients(samples: Sequence[Dict[str, Any]], plan_host: int) -> Dict[str, Any]:
    """The calibration child's rise above its first-``training_step`` baseline. It is decided
    on in bytes reclaim cannot free (``unreclaimable``); anon + shmem, VmHWM and the scope's
    ``memory.peak`` are recorded beside it. Build: every reading before that entry. Training:
    every reading after it, less the store being allocated (``_rise``), so pinned bytes
    beyond the plan still count."""
    entry = next((row for row in samples if row.get("label") == "first_step"), None)
    if entry is None or entry.get("unreclaimable") is None:
        return {"transient_bytes": None,
                "why": "no unreclaimable reading at the first training_step entry"}
    build, train = _rise(samples, "unreclaimable", entry, plan_host)
    labeled = [row for row in samples if "label" in row]
    out: Dict[str, Any] = {
        "unreclaimable_at_first_step": int(entry["unreclaimable"]),
        "build_transient_bytes": build, "train_transient_bytes": train,
        "transient_bytes": max(build, train), "plan_host_bytes": int(plan_host),
        "store_taken_out": ("torch pinned host allocator growth since the first step, up to "
                            "the plan's host_bytes") if entry.get("pinned") is not None
        else "nothing: no pinned host allocator counter, so the store counts as transient",
        "anon_shmem_at_first_step": entry.get("anon_shmem"),
        "anon_shmem_transient_bytes": (max(_rise(samples, "anon_shmem", entry, plan_host))
                                       if entry.get("anon_shmem") is not None else None),
        "vm_hwm_bytes": max((row["process"]["hwm"] for row in labeled
                             if row["process"].get("hwm") is not None), default=None),
        "memory_peak_bytes": max((row["cgroup"]["peak"] for row in labeled
                                  if isinstance(row["cgroup"].get("peak"), int)), default=None),
        "readings": len(samples),
    }
    return out


def oom_margin(transient: Optional[Dict[str, Any]], *, limit: int, target_host: int,
               working: int) -> Dict[str, Any]:
    """The room ``limit`` leaves for what reclaim cannot free (protocol §4), from the
    calibration child at the generous limit (U0 = its unreclaimable bytes at the first
    ``training_step`` entry).

    The build margin, limit - (U0 + build transient), decides: below 0 the session stops.
    The stand, not the product, puts this limit close to the steady state, and the product
    plans only after the build, so an OOM in the build would come from the stand's setup.
    The training margin, limit - (U0 + training transient + the store planned at this
    limit), is description only. The product promises its own reserve (``budget_of``), so an
    OOM in training is a product finding (criterion 2). Either margin below
    ``rule.OOM_MARGIN_WARN_BYTES`` adds a warning that decides nothing. An unknown baseline
    stops: unknown is not clean. The limit is never raised. ``margin_over_working_bytes``
    counts from the product's working set, which holds reclaimable active file cache:
    description only."""
    warn = rule.OOM_MARGIN_WARN_BYTES
    out: Dict[str, Any] = {"limit_bytes": int(limit), "target_host_bytes": int(target_host),
                           "warn_below_bytes": warn, "transient": transient}
    base = (transient or {}).get("unreclaimable_at_first_step")
    if base is None:
        return {**out, "build_margin_bytes": None, "train_margin_bytes": None, "stop": True,
                "warnings": [], "why": "the calibration measured no unreclaimable baseline: "
                                       "the build margin is unknown"}
    build = int(limit) - (base + transient["build_transient_bytes"])
    train = int(limit) - (base + transient["train_transient_bytes"] + int(target_host))
    warnings = [f"{name} margin {value:,} B < {warn:,} B (description only)"
                for name, value in (("build", build), ("training", train)) if value < warn]
    return {**out, "build_margin_bytes": build, "train_margin_bytes": train,
            "stop": build < 0, "warnings": warnings,
            "margin_over_working_bytes": int(limit) - (int(working)
                                                       + transient["transient_bytes"]
                                                       + int(target_host))}


def calibration_margin(record: Dict[str, Any]) -> Dict[str, Any]:
    """The OOM margin of the first tight limit, as the calibration child records it in its
    own JSON (the session computes the same for every limit it tries)."""
    try:
        limit = calibration_limit(record)
    except SessionStopError as exc:
        return {"why": str(exc)}
    return oom_margin(record.get("memory_transient"), limit=limit["limit_bytes"],
                      target_host=limit["target_host_bytes"], working=limit["working_set_bytes"])


def guard_oom(status: Dict[str, Any], calibration: Optional[Dict[str, Any]], limit: int,
              target_host: int, working: int) -> None:
    """Record the OOM margin of ``limit`` in ``status``; print its warnings; stop the session
    when the build margin is below 0 or unknown."""
    margin = oom_margin((calibration or {}).get("memory_transient"), limit=limit,
                        target_host=target_host, working=working)
    status["limits"]["oom_margin"].append(margin)
    for line in margin["warnings"]:
        print(f"WARNING at the tight limit {limit:,} B: {line}")
    if not margin["stop"]:
        return
    if margin["build_margin_bytes"] is None:
        raise OomRiskError(f"calibration: {margin['why']}")
    t = margin["transient"]
    raise OomRiskError(
        f"calibration: build margin {margin['build_margin_bytes']:,} B < 0 at the tight limit "
        f"{limit:,} B (unreclaimable at the first step {t['unreclaimable_at_first_step']:,} B, "
        f"build transient {t['build_transient_bytes']:,} B; training transient "
        f"{t['train_transient_bytes']:,} B, store {target_host:,} B, training margin "
        f"{margin['train_margin_bytes']:,} B). An OOM in the build would come from the stand's "
        f"limit, not the product. The limit is not raised: the owner decides")


# ==========================================================================
# the kernel's word on an OOM kill (protocol §3.6)
# ==========================================================================
#: After a killed child exits, the kernel log is read again after this long when it does not
#: show the kill yet (journald reads /dev/kmsg asynchronously).
KERNEL_LOG_RETRY_S = 3.0


def oom_kill_lines(text: str, pids: Sequence[int], scopes: Sequence[str] = ()) -> List[str]:
    """The kernel's memory-cgroup OOM lines about any of ``pids``, or about a task in one of
    ``scopes`` (cgroup paths): the ``oom-kill:constraint=CONSTRAINT_MEMCG`` line names the
    killed pid and its cgroup, the ``Memory cgroup out of memory: Killed process <pid>`` line
    is the kill. A pid a matching ``CONSTRAINT_MEMCG`` line names counts as one of ``pids``."""
    wanted = {int(pid) for pid in pids}
    names = {scope.rstrip("/").rsplit("/", 1)[-1] for scope in scopes if scope}
    lines = text.splitlines()
    for line in lines:
        if "CONSTRAINT_MEMCG" in line and any("task_memcg=" in line and f"/{name}," in line
                                              for name in names):
            pid = line.split(",pid=", 1)[-1].split(",", 1)[0]
            if pid.isdigit():
                wanted.add(int(pid))
    return [line.strip() for line in lines
            if any(f"Memory cgroup out of memory: Killed process {pid} (" in line
                   or ("CONSTRAINT_MEMCG" in line and f",pid={pid}," in line)
                   for pid in wanted)]


def read_kernel_log(since: float, until: float) -> Tuple[Optional[str], Optional[str]]:
    """``(text, None)`` of the kernel log between two unix times, or ``(None, why)``."""
    command = ["journalctl", "-k", "--no-pager", "-o", "short-unix",
               "--since", f"@{int(since) - 1}", "--until", f"@{int(until) + 5}"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if done.returncode != 0:
        return None, f"journalctl exited {done.returncode}: {done.stderr.strip()[:300]}"
    return done.stdout, None


def kernel_oom(pids: Sequence[int], since: float, until: float, *, scopes: Sequence[str] = (),
               read: Callable[[float, float], Tuple[Optional[str], Optional[str]]]
               = read_kernel_log,
               sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """Whether the kernel logged a memory-cgroup OOM kill of any of ``pids``, or of a task in
    one of ``scopes``, while the child ran: ``killed`` from the ``Killed process`` line, never
    from a return code. Under ``systemd-run --scope`` the child's pid is not the one
    ``Popen`` sees; the child's marker carries its own pid and cgroup path."""
    pids = sorted({int(pid) for pid in pids if pid})
    scopes = sorted({scope for scope in scopes if scope})
    text, error = read(since, until)
    lines = oom_kill_lines(text or "", pids, scopes)
    if text is not None and not any("Killed process" in line for line in lines):
        sleep(KERNEL_LOG_RETRY_S)
        again, error_again = read(since, until + KERNEL_LOG_RETRY_S)
        if again is not None:
            text, lines = again, oom_kill_lines(again, pids, scopes)
        error = error or error_again
    return {"source": "journalctl -k", "pids": pids, "scopes": scopes,
            "read": text is not None, "error": error,
            "killed": any("Killed process" in line for line in lines), "lines": lines}


# ==========================================================================
# the child: one arm in one process
# ==========================================================================
def grad_digest(tensor: Any) -> Dict[str, Any]:
    """SHA-256 of a tensor's raw bytes, with its dtype and shape. Equal digests mean equal
    bytes, which is torch.equal for finite tensors except that it also tells -0.0 from 0.0."""
    import torch

    flat = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    return {"sha256": hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
            "dtype": str(tensor.dtype).replace("torch.", ""), "shape": list(tensor.shape),
            "finite": bool(torch.isfinite(tensor).all()),
            "nonzero": bool(tensor.count_nonzero() > 0)}


def rows_fingerprint(rows: Any) -> Dict[str, Any]:
    """SHA-256 per row of its token ids and labels (V5)."""
    digests, lengths = [], []
    for i in range(len(rows)):
        row = rows[i]
        ids, labels = row["input_ids"].contiguous(), row["labels"].contiguous()
        digests.append(hashlib.sha256(ids.numpy().tobytes() + b"|" + labels.numpy().tobytes())
                       .hexdigest())
        lengths.append(int(ids.numel()))
    return {"count": len(digests), "lengths": lengths, "row_sha256": digests}


def make_rows(kind: str, vocab: int, *, k: int, seq: int, rows: int) -> Any:
    """Correctness and cleanup: the A3 stand's mixed rows. Calibration: one window of full
    rows. Cost: every row ``seq`` tokens, no masked labels, whole windows only."""
    if kind in ("correctness", "cleanup"):
        return base.make_rows(rows, seq, vocab)
    count = k if kind in ("calibrate", "check") else k * (COST_WARMUP + COST_STEPS)
    return base.SyntheticRows([seq] * count, [[] for _ in range(count)], vocab)


def host_allocator(key: str = "current") -> Optional[int]:
    """torch's caching host allocator, ``allocated_bytes.<key>``; None without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    stats = torch.cuda.host_memory_stats()
    for name in (f"allocated_bytes.{key}", f"allocated_bytes.all.{key}"):
        if name in stats:
            return int(stats[name])
    return None


def sigint_status(pid: Any = "self", *, proc_root: str = "/proc") -> Optional[Dict[str, Any]]:
    """SIGINT in ``/proc/<pid>/status``: pending (SigPnd or ShdPnd), blocked (SigBlk), ignored
    (SigIgn), caught by a handler (SigCgt), and the process State. None where /proc does not
    give them (not Linux, or the process is gone)."""
    bit = 1 << (int(signal.SIGINT) - 1)
    fields: Dict[str, str] = {}
    try:
        with open(os.path.join(proc_root, str(pid), "status"), encoding="ascii",
                  errors="replace") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                fields[key] = value.strip()
        masks = {key: int(fields[key], 16)
                 for key in ("SigPnd", "ShdPnd", "SigBlk", "SigIgn", "SigCgt")}
    except (OSError, KeyError, ValueError):
        return None
    return {"pending": bool((masks["SigPnd"] | masks["ShdPnd"]) & bit),
            "blocked": bool(masks["SigBlk"] & bit), "ignored": bool(masks["SigIgn"] & bit),
            "caught": bool(masks["SigCgt"] & bit), "state": fields.get("State")}


def _handler_name(handler: Any) -> str:
    if handler is signal.default_int_handler:
        return "default_int_handler"
    if isinstance(handler, signal.Handlers):
        return handler.name
    return repr(handler)


def restore_sigint() -> Dict[str, Any]:
    """Give this child Python's own SIGINT handler (KeyboardInterrupt), whatever it inherited,
    and say what it inherited. A command a non-interactive shell starts in the background
    (``&``) has SIGINT ignored; exec keeps an ignored signal ignored, neither Popen nor
    ``systemd-run --scope`` resets it, and Python leaves an inherited SIG_IGN in place. Session
    2's sigint child was started so and never saw its SIGINT (protocol §3.7)."""
    inherited = {"handler": _handler_name(signal.getsignal(signal.SIGINT)),
                 "proc": sigint_status()}
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return {"inherited": inherited,
            "installed": {"handler": _handler_name(signal.getsignal(signal.SIGINT)),
                          "proc": sigint_status()}}


class Hooks:
    """What the stand's trainer subclass and its callback share: the L2L state, the marker
    the parent reads (at the first ``training_step`` entry; for the signal children at each
    entry and exit, and at the hold), the memory sampler, and whether a KeyboardInterrupt came
    inside a training_step."""

    def __init__(self, marker: Optional[str], info: Dict[str, Any], *, every: bool = False,
                 sampler: Optional[MemorySampler] = None,
                 hold_timeout: float = HOLD_TIMEOUT_S):
        self.marker = marker
        self.info = info
        self.every = every
        self.sampler = sampler or MemorySampler()
        self.state: Any = None
        self.trainer: Any = None
        self.calls = 0
        self.interrupted_in_step = False
        self.cgroup_at_first_step: Optional[Dict[str, Any]] = None
        self.hold_timeout = hold_timeout
        #: Where and how the signal child held (None until it does).
        self.hold: Optional[Dict[str, Any]] = None

    def _write(self, trainer: Any, **fields: Any) -> None:
        plan = (self.state.spill_plan if self.state is not None else None) or {}
        write_json(Path(self.marker), {
            **self.info, "pid": os.getpid(), "unix": time.time(), "calls": self.calls,
            "steps_done": int(trainer.state.global_step), "spill_path": plan.get("spill_path"),
            **fields,
        })

    def enter(self, trainer: Any) -> None:
        self.calls += 1
        self.trainer = trainer
        if self.calls == 1:
            self.cgroup_at_first_step = self.sampler.first_step()["cgroup"]
        if self.marker and (self.every or self.calls == 1):
            self._write(trainer, in_step=True, phase="enter")

    def leave(self, trainer: Any) -> None:
        if self.marker and self.every:
            self._write(trainer, in_step=False, phase="leave")

    def install_hold(self, model: Any) -> None:
        """A forward pre-hook on every decoder layer; only ``_maybe_hold`` decides."""
        from soup_cli.utils.l2l_step import model_parts

        _causal, owner = model_parts(model)
        for index, layer in enumerate(owner.layers):
            layer.register_forward_pre_hook(functools.partial(self._maybe_hold, index))

    def _maybe_hold(self, index: int, _module: Any, _args: Any) -> None:
        """Hold at the first call of the highest spilled layer with grad on (the L2L step's
        backward recompute: its forward and the shape capture run without grad) in optimizer
        step 2. The spilled layers below it are still to be read back, so the store's reader
        has the file open. A KeyboardInterrupt in the wait is raised here, inside the step."""
        import torch

        if self.hold is not None or self.trainer is None or not torch.is_grad_enabled():
            return
        if int(self.trainer.state.global_step) != HOLD_STEPS_DONE:
            return
        plan = (self.state.spill_plan if self.state is not None else None) or {}
        spilled = int(plan.get("spill_layers") or 0)
        if spilled < 1 or index != spilled - 1:
            return
        store = getattr(self.state, "store", None)
        stats = getattr(store, "stats", None)
        self.hold = {"layer": index, "micro_batch": 0, "pass": "backward",
                     "spill_layers": spilled, "unix": time.time(),
                     "store_io": None if stats is None else {
                         "bytes_written": int(stats.bytes_written),
                         "bytes_read": int(stats.bytes_read)}}
        if self.marker:
            self._write(self.trainer, in_step=True, phase="hold", hold=self.hold,
                        sigint=sigint_status())
        started = time.monotonic()
        try:
            while time.monotonic() - started < self.hold_timeout:
                time.sleep(HOLD_POLL_S)
        except KeyboardInterrupt:
            self.hold.update(released="interrupt", waited_s=time.monotonic() - started)
            raise
        self.hold.update(released="timeout", waited_s=time.monotonic() - started)


def trainer_class(l2l: bool, hooks: Hooks) -> type:
    """The product's L2L trainer class (or the plain Trainer), with a marker at each
    ``training_step`` entry. Nothing of the product is replaced; the subclass only observes."""
    from transformers import Trainer

    from soup_cli.trainer.l2l_trainer import make_l2l_trainer_class

    parent = make_l2l_trainer_class(Trainer) if l2l else Trainer

    class StandTrainer(parent):  # type: ignore[misc, valid-type]
        def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
            hooks.enter(self)
            try:
                loss = super().training_step(model, inputs, num_items_in_batch)
            except KeyboardInterrupt:
                hooks.interrupted_in_step = True
                raise
            hooks.leave(self)
            return loss

    return StandTrainer


def recorder(record: Dict[str, Any], hooks: Hooks, *, device: str, record_grads: bool,
             tensor_dir: Optional[Path]) -> Any:
    """Per optimizer step: LoRA gradient digests, lr, step time, the store's spill I/O and the
    host allocator. With ``tensor_dir`` each step's gradients also go to
    ``<tensor_dir>/step_<NNNN>.pt`` at once, so the child never holds more than one step."""
    import torch
    from transformers import TrainerCallback

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    class Recorder(TrainerCallback):
        started = 0.0

        def on_train_begin(self, args, state, control, **kw):
            sync()
            record["host_allocator"]["baseline_bytes"] = host_allocator()
            if device == "cuda" and hasattr(torch.cuda, "reset_peak_host_memory_stats"):
                torch.cuda.reset_peak_host_memory_stats()

        def on_step_begin(self, args, state, control, **kw):
            sync()
            self.started = time.perf_counter()

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            if not record_grads:
                return
            grads = {n: p.grad.detach().to("cpu", copy=True)
                     for n, p in model.named_parameters() if "lora_" in n and p.grad is not None}
            record["grads"].append({n: grad_digest(g) for n, g in sorted(grads.items())})
            if tensor_dir is not None:
                tensor_dir.mkdir(parents=True, exist_ok=True)
                torch.save(grads, tensor_dir / f"step_{len(record['grads']) - 1:04d}.pt")

        def on_step_end(self, args, state, control, optimizer=None, **kw):
            record["rates"].append(optimizer.param_groups[0]["lr"])
            sync()
            record["step_s"].append(time.perf_counter() - self.started)
            store = getattr(hooks.state, "store", None)
            if store is not None:
                io = {"bytes_written": int(store.stats.bytes_written),
                      "bytes_read": int(store.stats.bytes_read)}
                record["store_io"] = io
                record["store_io_by_step"].append(io)
            record["host_allocator"]["at_last_step_bytes"] = host_allocator()
            record["host_allocator"]["peak_bytes"] = host_allocator("peak")
            hooks.sampler.sample("step_end")

    return Recorder()


def train_recorded(record: Dict[str, Any], model: Any, runtime: Any, rows: Any, *, arm: str,
                   l2l: bool, sdpa: str, k: int, seq: int, dtype: Any, device: str,
                   work_dir: Path, spill_dir: Optional[str], shard_dir: Optional[str],
                   seed: int = 1, record_grads: bool = True, tensor_dir: Optional[Path] = None,
                   marker: Optional[str] = None, marker_every: bool = False,
                   hold_timeout: Optional[float] = None, sampler: Optional[MemorySampler] = None,
                   watch: Any = None, trainer_options: Optional[Dict[str, Any]] = None) -> None:
    """One run through the HF trainer, as the A3 stand's ``train_arm``, filling ``record``.

    A KeyboardInterrupt (the sigint child) is recorded, not re-raised; the spill directory is
    listed after ``train()`` returns or raises, i.e. after the product's own cleanup. With
    ``hold_timeout`` the run waits up to that long for the parent's signal inside optimizer
    step 2 (``Hooks``)."""
    import torch
    from transformers import TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state
    from soup_cli.utils.l2l_spill import spill_files

    record.update(logs=[], rates=[], grads=[], step_s=[], store_io=None, store_io_by_step=[],
                  host_allocator={"baseline_bytes": None, "at_last_step_bytes": None,
                                  "peak_bytes": None},
                  spill_plan=None, stale_removed=[], exit="running")
    hooks = Hooks(marker, {key: record.get(key) for key in
                           ("arm", "device", "expected_memory_max", "cgroup_start")},
                  every=marker_every, sampler=sampler,
                  hold_timeout=HOLD_TIMEOUT_S if hold_timeout is None else hold_timeout)
    options: Dict[str, Any] = dict(
        output_dir=str(Path(work_dir) / arm.replace("/", "_")), per_device_train_batch_size=1,
        gradient_accumulation_steps=k, learning_rate=1e-3, optim="sgd",
        lr_scheduler_type="linear", warmup_steps=1, num_train_epochs=1, logging_steps=1,
        save_strategy="no", report_to=[], seed=seed, data_seed=seed, max_grad_norm=0.0,
        disable_tqdm=True,
    )
    options.update(trainer_options or {})
    torch.manual_seed(seed)
    trainer = trainer_class(l2l, hooks)(
        model=model, args=TrainingArguments(**options), train_dataset=rows,
        callbacks=[recorder(record, hooks, device=device, record_grads=record_grads,
                            tensor_dir=tensor_dir)],
    )
    if l2l:
        hooks.state = attach_l2l_state(
            trainer, runtime, dtype=dtype or torch.float32, spill_dir=spill_dir,
            weight_dirs=(shard_dir,) if shard_dir else (), max_seq=seq,
        )
    if hold_timeout is not None:
        hooks.install_hold(model)
    record["spill_files_before"] = spill_files(spill_dir) if spill_dir else []
    try:
        with base.pinned_sdpa(sdpa), (watch if watch is not None else contextlib.nullcontext()):
            trainer.train()
        record["exit"] = "normal"
    except KeyboardInterrupt:
        record["exit"] = "interrupted"
        record["at_interrupt"] = {"in_step": hooks.interrupted_in_step,
                                  "steps_done": int(trainer.state.global_step),
                                  "hold": hooks.hold}
    finally:
        record["spill_files_after_train"] = spill_files(spill_dir) if spill_dir else []
        record["steps"] = int(trainer.state.global_step)
        record["log_history"] = list(trainer.state.log_history)
        record["logs"] = [row for row in trainer.state.log_history if "loss" in row]
        record["cgroup_at_first_step"] = hooks.cgroup_at_first_step
        record["hold"] = hooks.hold
        state = hooks.state
        if state is not None:
            record["spill_plan"] = state.spill_plan
            record["stale_removed"] = list(state.stale_removed)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    """Mistral-7B NF4 on the RAM tier through ``stream_probe.build`` (the A3 stand's path)."""
    import l2l_probe
    import stream_probe

    from soup_cli.utils.layer_stream import resolve_stream_dtype

    cuda = args.device == "cuda"
    dtype = resolve_stream_dtype(args.device) if cuda else "float32"
    probe = ["--mode", "correctness", "--weights", args.weights,
             "--quant", args.quant if cuda else "none", "--seq", str(args.seq), "--tier", "ram",
             "--out", str(Path(args.out).with_suffix(".unused.json"))]
    if args.shards:
        probe += ["--shards", args.shards]
    parsed = l2l_probe.parse_args(probe)
    parsed.no_pin = not cuda  # page-locked memory needs CUDA
    started = time.perf_counter()
    model, runtime, config, _index, weights_dir, shard_dir, *_ = stream_probe.build(
        parsed, args.device, dtype)
    return {"model": model, "runtime": runtime, "config": config, "dtype": dtype,
            "shard_dir": str(shard_dir), "weights_dir": str(weights_dir),
            "build_seconds": time.perf_counter() - started}


def deterministic_facts(sdpa: str) -> Dict[str, Any]:
    import torch

    return {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "sdpa": sdpa}


def child_main(args: argparse.Namespace) -> int:
    sigint = restore_sigint()  # first: the session's launch may have left SIGINT ignored
    base.set_deterministic_env()  # before torch and CUDA start
    spec = spec_of(args.arm)
    out = Path(args.out)
    if out.exists():
        print(f"REFUSED: {out} exists; attempts are never overwritten")
        return 2
    cost = spec["kind"] == "cost"
    name = f"cost/b{args.block:02d}_p{args.position}_{args.arm}" if cost else args.arm
    record: Dict[str, Any] = {
        "arm": name, "kind": spec["kind"], "l2l": spec["l2l"], "sdpa": spec["sdpa"],
        "argv": [sys.executable, *sys.argv], "device": args.device,
        "expected_memory_max": args.memory_max, "pid": os.getpid(), "k": args.k,
        "seq": args.seq, "spill_dir": args.spill_dir, "started_unix": time.time(),
        "cgroup_start": cgroup_readback(), "exit": "starting", "sigint_handling": sigint,
    }
    if cost:
        record.update(role=args.arm, block=args.block, position=args.position)
    built: Optional[Dict[str, Any]] = None
    code = 0
    # The calibration and check children sample all along (the OOM margin, protocol §4);
    # every child takes the named readings (build end, first training_step, each step end).
    sampled = spec["kind"] in ("calibrate", "check")
    sampler = MemorySampler(SAMPLE_S if sampled else None,
                            pinned=host_allocator if args.device == "cuda" else (lambda: None))
    try:
        import l2l_schedule
        import stream_probe
        import torch

        if args.device == "cuda" and not stream_probe.cuda_available():
            record.update(exit="error", error="CUDA is not available")
            return 3
        base.enable_deterministic()
        record["deterministic"] = deterministic_facts(spec["sdpa"])
        record["env"] = base.environment_facts()
        if args.device == "cuda":
            record["env"]["gpu_facts"] = stream_probe.gpu_facts(args.device)
        if spec["l2l"] and args.spill_dir:
            os.makedirs(args.spill_dir, exist_ok=True)
        with sampler:
            built = build(args)
            record.update(dtype=built["dtype"], build_seconds=built["build_seconds"],
                          shard_dir=built["shard_dir"],
                          cgroup_after_build=sampler.sample("build_end")["cgroup"])
            model, config = built["model"], built["config"]
            l2l_schedule.make_non_vacuous(model)
            rows = make_rows(spec["kind"], int(config.vocab_size), k=args.k, seq=args.seq,
                             rows=args.rows)
            record["rows"] = rows_fingerprint(rows)
            record["expected_steps"] = -(-len(rows) // args.k)
            tensor_dir = (Path(args.tensors_dir) / args.arm
                          if args.tensors_dir and spec["kind"] == "correctness" else None)
            watch = None
            if cost:
                watch = arm_validity.ArmWatch(arm=args.arm, round=args.block,
                                              run=args.position, required=REQUIRED_CHECKS,
                                              model_path=args.spill_dir)
            import tempfile

            with tempfile.TemporaryDirectory() as work:
                train_recorded(
                    record, model, built["runtime"], rows, arm=name, l2l=spec["l2l"],
                    sdpa=spec["sdpa"], k=args.k, seq=args.seq,
                    dtype=getattr(torch, built["dtype"]), device=args.device,
                    work_dir=Path(work), spill_dir=args.spill_dir if spec["l2l"] else None,
                    shard_dir=built["shard_dir"], record_grads=not cost, tensor_dir=tensor_dir,
                    marker=args.marker, marker_every=bool(spec.get("signal")),
                    hold_timeout=args.hold_timeout if spec.get("signal") else None,
                    sampler=sampler, watch=watch,
                    trainer_options={"bf16": built["dtype"] == "bfloat16",
                                     "use_cpu": args.device == "cpu"},
                )
        if sampled:
            record["memory_transient"] = transients(
                sampler.samples, int((record.get("spill_plan") or {}).get("host_bytes") or 0))
        if spec["kind"] == "calibrate" and record["exit"] == "normal":
            record["oom_margin"] = calibration_margin(record)
        if watch is not None:
            record.update(watch=watch.record, outcome=watch.record["outcome"])
        if cost and record["exit"] == "normal":
            timed = record["step_s"][COST_WARMUP:]
            record.update(value=statistics.median(timed) if timed else None, timed_step_s=timed)
        if spec["kind"] == "calibrate" and record["exit"] == "normal":
            record.update(probes(config, built["dtype"], args))
        if record["exit"] == "interrupted":
            code = 130
    except Exception as exc:
        import traceback

        record.update(exit="error", error=f"{type(exc).__name__}: {exc}",
                      traceback_tail="".join(traceback.format_exc())[-4000:])
        code = 1
    finally:
        sampler.stop()
        record["memory"] = {"interval_s": sampler.interval, "labeled": sampler.labeled(),
                            "series": sampler.compact() if sampled else [],
                            "failed_readings": sampler.failed}
        if built is not None:
            built["runtime"].close()
        record.update(cgroup_end=cgroup_readback(), ended_unix=time.time())
        write_json(out, record)
    print(f"{name}: exit {record['exit']}, spill_layers "
          f"{(record.get('spill_plan') or {}).get('spill_layers')}, steps {record.get('steps')}")
    return code


def probes(config: Any, dtype: str, args: argparse.Namespace) -> Dict[str, Any]:
    """The A3 stand's bnb #2034 probe and SDPA probe, after training (not inside the
    working set the calibration measures)."""
    heads = int(config.num_attention_heads)
    padded = -(-args.seq // 64) * 64
    lengths = base.make_rows(args.rows, args.seq, int(config.vocab_size))
    sizes = [len(lengths[i]["input_ids"]) for i in range(len(lengths))]
    return {
        "aliasing": base.run_aliasing_probe(device=args.device, dtype=dtype,
                                            dim=int(config.hidden_size),
                                            m_values={min(sizes), max(sizes), padded}),
        "sdpa_probe": base.probe_sdpa_backends(device=args.device, dtype=dtype, heads=heads,
                                               seq=padded,
                                               head_dim=int(config.hidden_size) // heads),
    }


# ==========================================================================
# compare: the unpublished tensors of the correctness arms (CPU)
# ==========================================================================
def _digests_only(steps: Sequence[Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [{n: {k: d.get(k) for k in ("sha256", "dtype", "shape")} for n, d in step.items()}
            for step in steps]


def load_run(record: Dict[str, Any], tensor_dir: Path) -> Tuple[Any, bool]:
    """The A3 stand's ``RunRecord`` of one arm from its JSON (logs, lr, steps) and its
    per-step gradient files, and whether those files hash to the digests the JSON carries."""
    import torch

    paths = sorted(tensor_dir.glob("step_*.pt"))
    grads = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    match = _digests_only([{n: grad_digest(g) for n, g in sorted(step.items())}
                           for step in grads]) == _digests_only(record.get("grads") or [])
    run = base.RunRecord(logs=record.get("logs") or [], rates=record.get("rates") or [],
                         grads=grads, steps=int(record.get("steps") or 0),
                         l2l=bool(record.get("l2l")))
    return run, match


def compare_main(args: argparse.Namespace) -> int:
    session, tensors = Path(args.session), Path(args.tensors_dir)
    runs: Dict[str, Any] = {}
    out: Dict[str, Any] = {"device": "cpu (analysis)", "hashes_match": {}}
    dtypes = set()
    for name in rule.CORRECTNESS_ARMS:
        rec = read_json(session / "arms" / f"{name}.json")
        if rec is None or rec.get("exit") != "normal" or not (tensors / name).is_dir():
            continue
        dtypes.add(rec.get("dtype"))
        runs[name], out["hashes_match"][name] = load_run(rec, tensors / name)
    for key, (left, right) in {"nospill_repeat": ("l2l_nospill_1", "l2l_nospill_2"),
                               "spill_vs_nospill": ("l2l_nospill_1", "l2l_spill")}.items():
        if left in runs and right in runs:
            out[key] = base.compare_runs(runs[left], runs[right])
    mode = "bf16" if "bfloat16" in dtypes else "fp32"
    out["mode"] = mode
    if all(name in runs for name in ("ga_1", "ga_2", "l2l_spill")):
        verdict = base.decide(runs["ga_1"], runs["ga_2"], runs["l2l_spill"], mode=mode,
                              floor=runs.get("ga_floor") if mode == "bf16" else None)
        out["tolerance"] = {"status": verdict["status"], "pass": verdict["pass"],
                            "note": verdict["note"], "a_vs_a": verdict["a_vs_a"],
                            "l2l_vs_ga": verdict["l2l_vs_ga"], "tolerance": verdict["tolerance"]}
    else:
        out["tolerance"] = None
    write_json(Path(args.out), out)
    print(f"compare: {sorted(runs)}; tolerance "
          f"{(out['tolerance'] or {}).get('status')}")
    return 0


# ==========================================================================
# the session (parent)
# ==========================================================================
def _scope(args: argparse.Namespace, memory_max: Any, unit: str) -> List[str]:
    return [args.systemd_run, *(["--user"] if args.user_scope else []), "--scope", "--quiet",
            "--collect", f"--unit={unit}", "-p", f"MemoryMax={memory_max}", "-p",
            "MemorySwapMax=0", "--"]


def scope_unit(out: Path, name: str) -> str:
    """The child's own scope unit: unique per session and child, so the kernel's OOM lines
    (``task_memcg=.../<unit>``) name it even before the child writes its marker."""
    session = hashlib.sha256(str(out).encode()).hexdigest()[:8]
    return f"a3-{session}-{name.replace('/', '-').replace('_', '-')}.scope"


def output_path(out: Path, arm: str, block: int = 0, position: int = 0) -> Path:
    spec = spec_of(arm)
    if spec["kind"] == "cost":
        return out / "cost" / f"b{block:02d}_p{position}_{arm}.json"
    return out / spec["folder"] / f"{arm}.json" if spec["folder"] else out / f"{arm}.json"


def child_entry(args: argparse.Namespace, out: Path, arm: str, limits: Dict[str, Any],
                block: int = 0, position: int = 0) -> Dict[str, Any]:
    """The command of one child, inside its own scope at its planned limit."""
    spec = spec_of(arm)
    memory_max = limits[spec["limit"]]
    name = f"cost/b{block:02d}_p{position}_{arm}" if spec["kind"] == "cost" else arm
    target = output_path(out, arm, block, position)
    spill_dir = os.path.join(args.spill_dir, spec["group"])
    arguments = [sys.executable, STAND, "child", "--arm", arm, "--weights", args.weights,
                 "--shards", args.shards, "--spill-dir", spill_dir,
                 "--memory-max", str(memory_max), "--k", str(args.k), "--seq", str(args.seq),
                 "--rows", str(ROWS), "--out", str(target)]
    entry: Dict[str, Any] = {"name": name, "output": str(target), "spill_dir": spill_dir,
                             "memory_max": memory_max, "limit": spec["limit"],
                             "scope_unit": scope_unit(out, name)}
    if spec["kind"] == "cost":
        arguments += ["--block", str(block), "--position", str(position)]
    if spec["kind"] == "correctness":
        arguments += ["--tensors-dir", args.tensors_dir]
    # Every child writes a marker at its first training_step entry (a kill before it is the
    # build's); the signal children rewrite it at each entry, and the parent signals from it.
    entry["marker"] = str(out / "markers" / (name.replace("/", "__") + ".json"))
    arguments += ["--marker", entry["marker"]]
    if spec.get("signal"):
        entry["signal_name"] = spec["signal"]
    entry["command"] = _scope(args, memory_max, entry["scope_unit"]) + arguments
    return entry


def compare_entry(args: argparse.Namespace, out: Path) -> Dict[str, Any]:
    target = out / "compare.json"
    return {"name": "compare", "output": str(target), "memory_max": None, "limit": None,
            "command": [sys.executable, STAND, "compare", "--session", str(out),
                        "--tensors-dir", args.tensors_dir, "--out", str(target)]}


def fixed_plan(args: argparse.Namespace, out: Path, limits: Dict[str, Any]
               ) -> List[Dict[str, Any]]:
    """Every child in run order, cost block 0 last. ``limits['tight']`` is the placeholder
    ``<TIGHT>`` until the calibration has run."""
    plan = [child_entry(args, out, "calibration", limits)]
    plan += [child_entry(args, out, f"calibration_check_{i}", limits)
             for i in range(1, MAX_CALIBRATION_CHECKS + 1)]
    plan += [child_entry(args, out, arm, limits) for arm in CORRECTNESS_ORDER]
    plan.append(compare_entry(args, out))
    plan += [child_entry(args, out, arm, limits) for arm in CLEANUP_ORDER]
    plan += [child_entry(args, out, arm, limits, 0, position)
             for position, arm in enumerate(arm_validity.BLOCK_ORDER)]
    return plan


NOTES = {
    "calibration": "generous limit: the product's plan inputs, the working set, the limit",
    "calibration_check_2": "only if check 1 is outside [12, 20] spilled layers",
    "compare": "CPU: max |d| and the A3 stand's GA tolerance from the unpublished tensors",
    "sigint": "ONE SIGINT while it holds inside optimizer step 2 (backward recompute of the "
              "highest spilled layer); no spill file may be left",
    "kill9": "SIGKILL while it holds inside optimizer step 2; the spill file must stay",
    "kill9_next": "same spill dir: must remove kill9's file at start and train",
}


def print_plan(plan: Sequence[Dict[str, Any]], limits: Dict[str, Any]) -> None:
    print(f"limits: generous {limits['generous']:,} B (MemoryMax), tight {limits['tight']} "
          f"(from the calibration)")
    for entry in plan:
        note = NOTES.get(entry["name"], "")
        if entry["name"].startswith("cost/"):
            note = "cost block 0 of >= 5, <= 12; only if criterion 1 passes"
        print(f"# {entry['name']}" + (f": {note}" if note else ""))
        print(" ".join(str(part) for part in entry["command"]))
    print(f"plan only: {len(plan) - 4} fixed children, then A-B-B-A cost blocks "
          f"(>= {arm_validity.PILOT_BLOCKS}, <= {rule.MAX_BLOCKS}) only if criterion 1 passes; "
          f"--execute runs them")


def drive_signal(child: Any, entry: Dict[str, Any], deadline: float, *,
                 list_files: Callable[[str], List[str]], send: Callable[[int, int], None],
                 read_status: Callable[[int], Optional[Dict[str, Any]]] = sigint_status,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 exit_timeout: float = SIGNAL_EXIT_TIMEOUT_S) -> Dict[str, Any]:
    """Wait for the step-2 hold, list the spill directory, read the target's SIGINT status,
    then send ONE signal. Polling and the exit wait share the session's monotonic deadline;
    timeout cleanup still kills and reaps the launcher as ``run_child`` does."""
    marker_path = Path(entry["marker"])
    marker = None
    while True:
        code = child.poll()
        if code is not None:
            return {"signal": entry["signal_name"], "sent_unix": None, "returncode": code,
                    "why": f"the child exited {code} before it held inside optimizer step 2",
                    "marker": marker}
        marker = read_json(marker_path)
        remaining = deadline - clock()
        if remaining <= 0:
            return {"signal": entry["signal_name"], "sent_unix": None, "why": "session deadline",
                    "marker": marker}
        if marker and marker.get("phase") == "hold":
            break
        sleep(min(MARKER_POLL_S, remaining))
    before = list_files(entry["spill_dir"])
    pid = int(marker["pid"])
    target = read_status(pid)
    if deadline - clock() <= 0:
        return {"signal": entry["signal_name"], "sent_unix": None, "why": "session deadline",
                "marker": marker, "files_before_signal": before, "target_at_send": target}
    why = None
    try:
        send(pid, int(getattr(signal, entry["signal_name"])))
    except ProcessLookupError:
        sent = None
        why = "the child was gone when the signal was sent"
    else:
        sent = time.time()
    try:
        code, exited = child.wait(timeout=max(0.0, min(exit_timeout, deadline - clock()))), True
    except subprocess.TimeoutExpired:
        child.kill()
        code, exited = child.wait(), False
    result = {"signal": entry["signal_name"], "sent_unix": sent, "marker": marker,
              "child_pid": getattr(child, "pid", None), "files_before_signal": before,
              "target_at_send": target, "files_after_exit": list_files(entry["spill_dir"]),
              "exited": exited, "returncode": code}
    if why is not None:
        result["why"] = why
    return result


def run_child(entry: Dict[str, Any], out: Path, env: Dict[str, str], deadline: float, *,
              oom_reader: Callable[..., Dict[str, Any]] = kernel_oom) -> Dict[str, Any]:
    """Run one child to its exit. A child that exits other than 0 gets the kernel's OOM-kill
    lines about its pid (and the pid in its marker) read into ``kernel_oom``."""
    from soup_cli.utils.l2l_spill import spill_files

    target = Path(entry["output"])
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    log_path = out / "logs" / (entry["name"].replace("/", "__") + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = {key: value for key, value in entry.items() if key != "command"}
    result.update(command=entry["command"], started_unix=time.time())
    with log_path.open("xb") as log:
        child = subprocess.Popen(entry["command"], cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        result["pid"] = child.pid
        try:
            if entry.get("signal_name"):
                result["signal"] = drive_signal(child, entry, deadline, list_files=spill_files,
                                                send=os.kill)
            code = child.wait(timeout=max(0.01, deadline - time.monotonic()))
            if entry.get("marker"):
                result["marker_after_exit"] = read_json(Path(entry["marker"]))
        except BaseException:
            if child.poll() is None:
                child.kill()
                child.wait()
            raise
    result.update(returncode=code, ended_unix=time.time())
    if code != 0:
        marker = result.get("marker_after_exit") or {}
        result["kernel_oom"] = oom_reader(
            [p for p in (child.pid, marker.get("pid")) if p], result["started_unix"],
            result["ended_unix"],
            scopes=[entry.get("scope_unit"), (marker.get("cgroup_start") or {}).get("path")])
    if entry.get("spill_dir"):
        result["spill_files_after_exit"] = spill_files(entry["spill_dir"])
    print(f"{entry['name']:<26} exit {code}"
          + (" (kernel: memory cgroup OOM kill)"
             if (result.get("kernel_oom") or {}).get("killed") else ""))
    return result


def oom_in_training(result: Dict[str, Any]) -> Optional[str]:
    """Why the session stops after this child (protocol §3.6), or None: the kernel OOM-killed
    a child at the tight limit after its first-step marker."""
    if not (result.get("kernel_oom") or {}).get("killed") or result.get("limit") != "tight":
        return None
    marker = result.get("marker_after_exit")
    if not marker:
        return None
    return (f"{result['name']}: the kernel OOM-killed pid {result.get('pid')} at "
            f"{result.get('memory_max'):,} B after its first training_step (marker at "
            f"{marker.get('steps_done')} optimizer steps done): criterion 2 fails")


def cost_runs(out: Path) -> List[List[arm_validity.Run]]:
    blocks: Dict[int, Dict[int, arm_validity.Run]] = {}
    for path in sorted((out / "cost").glob("b*_p*_*.json")):
        rec = read_json(path) or {}
        if rec.get("value") is None:
            continue
        blocks.setdefault(int(rec["block"]), {})[int(rec["position"])] = arm_validity.Run(
            float(rec["value"]), str(rec.get("outcome")))
    return [[runs[p] for p in range(4)] for _, runs in sorted(blocks.items()) if len(runs) == 4]


def session_parser(sub: Any) -> None:
    parser = sub.add_parser("session", help="plan (default) or run the whole session")
    parser.add_argument("--weights", required=True, help="local Mistral-7B checkpoint")
    parser.add_argument("--shards", required=True, help="its NF4 layer cache")
    parser.add_argument("--spill-dir", required=True,
                        help="a local NVMe directory (not tmpfs, not overlay); one subdirectory "
                             "per child group")
    parser.add_argument("--out", required=True, help="a new directory")
    parser.add_argument("--tensors-dir", default=None,
                        help="unpublished gradient tensors (default: <out>.tensors, beside it)")
    parser.add_argument("--generous-limit", default=DEFAULT_GENEROUS,
                        help="MemoryMax of the no-spill children (nothing may spill there)")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--seq", type=int, default=SEQ)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--systemd-run", default="systemd-run")
    parser.add_argument("--user-scope", action="store_true",
                        help="systemd-run --user (a VM with a user session, not root)")
    parser.add_argument("--execute", action="store_true")


def session_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    for field in ("weights", "shards", "spill_dir", "out", "tensors_dir"):
        if getattr(args, field):
            setattr(args, field, os.path.abspath(getattr(args, field)))
    out = Path(args.out)
    if out.exists():
        parser.error(f"{out} exists; every session writes a new directory")
    if args.tensors_dir is None:
        args.tensors_dir = str(out) + ".tensors"
    if args.k != K or args.seq != SEQ:
        parser.error(f"the rule fixes k {K} and seq {SEQ}")
    try:
        generous = parse_size(args.generous_limit)
    except ValueError as exc:
        parser.error(str(exc))
    limits: Dict[str, Any] = {"generous": generous, "tight": TIGHT}
    plan = fixed_plan(args, out, limits)
    if not args.execute:
        print_plan(plan, limits)
        return 0
    if not sys.platform.startswith("linux") or shutil.which(args.systemd_run) is None:
        parser.error("--execute needs Linux with systemd-run and cgroup v2")
    if Path(args.tensors_dir).exists():
        parser.error(f"{args.tensors_dir} exists; tensors are never overwritten")
    available = arm_validity.ram_available()
    if available is None or available < generous:
        parser.error(f"MemAvailable {available} B is below the generous limit {generous:,} B")
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    hashes = l2l_session.source_hashes()
    l2l_session.write_json(out / "protocol.json", jsonable({
        "rule": rule.RULE, "rule_version": rule.RULE_VERSION, "args": vars(args),
        "plan": plan, "base_commit": l2l_session.git_head(),
        "versions": l2l_session.versions(), "source_sha256": hashes,
        "python": sys.executable, "platform": platform.platform(), "planned_unix": time.time(),
    }))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(HARNESS)]),
           "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "CUBLAS_WORKSPACE_CONFIG": base.CUBLAS_WORKSPACE_CONFIG}
    deadline = time.monotonic() + args.max_hours * 3600
    status: Dict[str, Any] = {"state": "running", "children": [],
                              "limits": {"generous": generous, "tight": None,
                                         "calibration": None, "checks": [],
                                         "oom_margin": []}}

    def run(entry: Dict[str, Any]) -> Dict[str, Any]:
        result = run_child(entry, out, env, deadline)
        status["children"].append(result)
        write_json(out / "session.json", status)
        why = oom_in_training(result)
        if why:
            raise OomInTrainingError(why)
        return result

    try:
        run(child_entry(args, out, "calibration", limits))
        measured = read_json(out / "calibration.json")
        calibration = calibration_limit(measured)
        status["limits"]["calibration"] = calibration
        tight, sizing = calibration["limit_bytes"], calibration
        for index in range(1, MAX_CALIBRATION_CHECKS + 1):
            # Before any child runs at a tight limit: does it leave room for what reclaim
            # cannot free? The limit is never raised; a short margin stops the session.
            guard_oom(status, measured, tight, sizing["target_host_bytes"],
                      sizing["working_set_bytes"])
            name = f"calibration_check_{index}"
            run(child_entry(args, out, name, {**limits, "tight": tight}))
            action, value, why = next_limit(read_json(out / f"{name}.json"), tight, index)
            status["limits"]["checks"].append({"limit_bytes": tight, "action": action, **why})
            if action == "stop":
                raise SessionStopError(f"calibration: {why['why']}")
            if action == "done":
                break
            tight, sizing = value, why["recomputed"]
        limits["tight"] = status["limits"]["tight"] = tight
        print(f"tight limit {tight:,} B; generous {generous:,} B")
        for arm in CORRECTNESS_ORDER:
            run(child_entry(args, out, arm, limits))
        run(compare_entry(args, out))
        for arm in CLEANUP_ORDER:
            run(child_entry(args, out, arm, limits))
        allowed, why = rule.cost_gate(str(out))
        if not allowed:
            status.update(state="stopped_before_cost", reason=why)
            print(why)
        else:
            block = 0
            while block < rule.MAX_BLOCKS and arm_validity.blocks_to_add(cost_runs(out)) > 0:
                for position, arm in enumerate(arm_validity.BLOCK_ORDER):
                    result = run(child_entry(args, out, arm, limits, block, position))
                    if result["returncode"] != 0:
                        raise SessionStopError(f"{result['name']} exited {result['returncode']}")
                block += 1
        if l2l_session.source_hashes() != hashes:
            raise SessionStopError("source files changed during the session")
        if status["state"] == "running":
            status["state"] = "finished"
    except (Exception, KeyboardInterrupt) as exc:
        status.update(state=getattr(exc, "state", "stopped"),
                      reason=f"{type(exc).__name__}: {exc}")
        print(f"STOPPED ({status['state']}): {status['reason']}")
    finally:
        write_json(out / "session.json", status)
        try:
            verdict = rule.evaluate(str(out))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            verdict = {"verdict": "NO VERDICT: incomplete or malformed raw data",
                       "analysis_error": f"{type(exc).__name__}: {exc}",
                       "summary": {"verdict": "NO VERDICT", "analysis_error": str(exc)}}
        rule.write_verdict(out / "verdict.json", verdict)
        print(json.dumps(verdict["summary"], indent=2))
    return {"finished": 0, "stopped_before_cost": 4}.get(status["state"], 3)


# ==========================================================================
# command line
# ==========================================================================
def child_parser(sub: Any) -> None:
    parser = sub.add_parser("child", help="one arm in this process (the session runs these)")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--shards", default=None)
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--spill-dir", default=None)
    parser.add_argument("--memory-max", type=int, default=None,
                        help="the MemoryMax the parent gave this child's scope (V3 readback)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--seq", type=int, default=SEQ)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--marker", default=None)
    parser.add_argument("--hold-timeout", type=float, default=HOLD_TIMEOUT_S,
                        help="signal children: how long to hold inside optimizer step 2 for "
                             "the parent's signal")
    parser.add_argument("--tensors-dir", default=None)
    parser.add_argument("--out", required=True)


def compare_parser(sub: Any) -> None:
    parser = sub.add_parser("compare", help="compare the correctness arms' tensors (CPU)")
    parser.add_argument("--session", required=True)
    parser.add_argument("--tensors-dir", required=True)
    parser.add_argument("--out", required=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    session_parser(sub)
    child_parser(sub)
    compare_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "session":
        return session_main(args, sub.choices["session"])
    if args.command == "child":
        return child_main(args)
    return compare_main(args)


if __name__ == "__main__":
    sys.exit(main())
