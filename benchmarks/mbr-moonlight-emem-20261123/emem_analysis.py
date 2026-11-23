"""Analyze Moonlight memory measurements without importing PyTorch."""

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

MODEL = "Moonlight-16B-A3B"
MOONLIGHT_TARGETS = ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
WINDOWS = ((8, 64), (8, 512), (64, 64))
CLASSES = ("cublaslt_workspace", "algo_discovery", "staging", "fused_scratch", "other")
_PEAKS = {"alloc": "peak_allocated", "reserved": "peak_reserved", "nvml": "peak_nvml"}
_EXT = "a0_exact_lt_ext.cpp"
_NATIVE_FILES = (_EXT, "a0_exact_lt_kernels.cu")


def window_label(k: int, seq: int) -> str:
    """Format a benchmark window for the canonical row."""
    return f"k={k},S={seq}"


def canonical_row(
    *,
    model: str,
    model_revision: str,
    card: str,
    window: str,
    arm: str,
    mode: str,
    torch_version: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Merge measurement data without replacing the run's identity."""
    row = {
        "model": model, "model_revision": model_revision, "card": card, "window": window,
        "arm": arm, "mode": mode, "torch_version": torch_version,
    }
    collisions = row.keys() & extra.keys()
    if collisions:
        raise ValueError(f"extra fields collide with row metadata: {', '.join(sorted(collisions))}")
    return {**row, **extra}


def mem_premium(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare the latest successful normal-mode measurements in each window."""
    index, ooms = {}, set()
    for row in rows:
        if row.get("mode") != "normal":
            continue
        key = (row.get("arm"), row.get("k"), row.get("seq"))
        if row.get("status") == "ok":
            index[key] = row
        elif row.get("status") == "OOM":
            ooms.add(key)
    out = []
    for k, seq in WINDOWS:
        missing = [
            f"{arm} (OOM)" if (arm, k, seq) in ooms else arm
            for arm in ("C", "E") if (arm, k, seq) not in index
        ]
        pair = {
            arm: {
                f"{metric}_bytes": index.get((arm, k, seq), {}).get(key)
                for metric, key in _PEAKS.items()
            }
            for arm in ("C", "E")
        }
        deltas = {}
        for metric in _PEAKS:
            c_bytes, e_bytes = (pair[arm][f"{metric}_bytes"] for arm in ("C", "E"))
            deltas[f"delta_{metric}"] = (
                e_bytes - c_bytes if c_bytes is not None and e_bytes is not None else None
            )
        out.append({
            "k": k, "seq": seq, "window": window_label(k, seq),
            "status": f"missing {' and '.join(missing)}" if missing else "ok",
            **pair, **deltas,
        })
    return out


def _names_function(name: Any, candidates: tuple[str, ...]) -> bool:
    """Match a bare or demangled symbol on the function name, not on substrings.

    The profiler writes demangled C++ signatures such as
    ``(anonymous namespace)::scratch(int, long, at::Tensor const&)``. A plain
    substring test would also accept ``seg_matmul_raw`` for ``matmul``.
    """
    symbol = str(name or "")
    if symbol in candidates:
        return True
    return any(
        re.search(rf"(?:^|::){re.escape(candidate)}\(", symbol) for candidate in candidates
    )


def classify_frames(frames: Sequence[dict[str, Any]]) -> str:
    """Use the highest-priority matching class anywhere in an allocation stack."""
    stack = [
        (frame.get("filename", "").replace("\\", "/").rsplit("/", 1)[-1], frame.get("name"))
        for frame in frames
    ]
    if any(file == _EXT
           and _names_function(name, ("matmul", "shared_workspace", "workspace_ptr", "lt_raw"))
           for file, name in stack):
        return "cublaslt_workspace"
    if any(file == "a0_exact_lt.py"
           and _names_function(name, ("lookup", "_reproduces", "native_matmul", "_sample",
                                      "ensure"))
           for file, name in stack):
        return "algo_discovery"
    if any(
        file in _NATIVE_FILES
        and _names_function(name, ("scratch", "seg_matmul_raw", "merged", "copy_segments"))
        for file, name in stack
    ):
        return "staging"
    if any(file in _NATIVE_FILES for file, _ in stack):
        return "fused_scratch"
    return "other"


def address_frames(snapshot: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Index the latest recorded allocation stack for each address."""
    return {
        event["addr"]: event.get("frames", [])
        for trace in snapshot.get("device_traces", [])
        for event in trace
        if event["action"] == "alloc"
    }


def _block_frames(
    block: dict[str, Any], frames_by_address: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return block.get("frames") or frames_by_address.get(block.get("address"), [])


def holdings(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten allocator blocks, keeping cached blocks separate from live allocations."""
    frames_by_address = address_frames(snapshot)
    out = []
    for segment in snapshot.get("segments", []):
        for block in segment["blocks"]:
            frames = _block_frames(block, frames_by_address)
            out.append({
                "size": block["size"], "live": block["state"] == "active_allocated",
                "klass": classify_frames(frames), "frames": frames,
            })
    return out


def segment_traps(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Separate fully cached segments from free space in pinned segments."""
    frames_by_address = address_frames(snapshot)
    releasable = trapped = 0
    pinned = []
    for segment in snapshot.get("segments", []):
        total, active = segment["total_size"], segment["active_size"]
        if active == 0:
            releasable += total
        elif total > active:
            free = total - active
            trapped += free
            holders = {
                classify_frames(_block_frames(block, frames_by_address))
                for block in segment["blocks"] if block["state"] == "active_allocated"
            }
            pinned.append({
                "total_size": total, "active_bytes": active, "trapped_bytes": free,
                "holders": sorted(holders),
            })
    pinned.sort(key=lambda segment: -segment["trapped_bytes"])
    return {"releasable_bytes": releasable, "trapped_bytes": trapped, "pinned": pinned}


def holdings_diff(c_snapshot: dict[str, Any], e_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compare allocator totals, classes, segment traps, and blocks that only E holds."""
    by_class = {
        klass: {
            arm: {"live_bytes": 0, "reserved_bytes": 0} for arm in ("C", "E", "delta")
        }
        for klass in CLASSES
    }
    totals = {arm: {"live_bytes": 0, "reserved_bytes": 0} for arm in ("C", "E", "delta")}
    blocks = {"C": holdings(c_snapshot), "E": holdings(e_snapshot)}
    for arm, arm_blocks in blocks.items():
        for block in arm_blocks:
            for counts in (totals, by_class[block["klass"]]):
                counts[arm]["reserved_bytes"] += block["size"]
                if block["live"]:
                    counts[arm]["live_bytes"] += block["size"]
    for counts in (totals, *by_class.values()):
        for metric in counts["delta"]:
            counts["delta"][metric] = counts["E"][metric] - counts["C"][metric]

    def signature(block: dict[str, Any]) -> tuple[str, bool, int]:
        return block["klass"], block["live"], block["size"]

    remaining = Counter(signature(block) for block in blocks["C"])
    surplus = {}
    for block in blocks["E"]:
        key = signature(block)
        if remaining[key]:
            remaining[key] -= 1
        else:
            entry = surplus.setdefault(key, {**block, "count": 0})
            entry["count"] += 1
    extra = sorted(surplus.values(), key=lambda block: -block["size"])
    main_source = max(sorted(CLASSES), key=lambda klass: by_class[klass]["delta"]["reserved_bytes"])
    if by_class[main_source]["delta"]["reserved_bytes"] <= 0:
        main_source = None

    traps = {"C": segment_traps(c_snapshot), "E": segment_traps(e_snapshot)}
    trapped_by_holder = dict.fromkeys(CLASSES, 0)
    for arm, sign in (("C", -1), ("E", 1)):
        for segment in traps[arm]["pinned"]:
            # A segment's free space counts once for each distinct live holder class.
            for klass in segment["holders"]:
                trapped_by_holder[klass] += sign * segment["trapped_bytes"]
    traps["delta"] = {
        metric: traps["E"][metric] - traps["C"][metric]
        for metric in ("releasable_bytes", "trapped_bytes")
    }
    return {
        "totals": totals, "by_class": by_class, "extra": extra, "main_source": main_source,
        "traps": traps, "trapped_by_holder": trapped_by_holder,
    }


def describe_diff(diff: dict[str, Any]) -> str:
    """Summarize E minus C, segment free space, and the largest surplus holdings."""
    mib = 2**20
    source = diff["main_source"] or "none"
    traps = diff["traps"]
    trapped_delta = traps["delta"]["trapped_bytes"]
    if source == "other" and trapped_delta > 0:
        by_holder = diff["trapped_by_holder"]
        largest = max(by_holder.values())
        holders = sorted(klass for klass, delta in by_holder.items() if delta == largest > 0)
        names = ", ".join(holders) or "unknown holders"
        source += (
            f" = {trapped_delta / mib:.1f} MiB of free space trapped in segments pinned by {names}"
        )
    lines = [f"Main source: {source}", "E minus C:"]
    for name, counts in (("total", diff["totals"]), *diff["by_class"].items()):
        delta = counts["delta"]
        live, reserved = delta["live_bytes"] / mib, delta["reserved_bytes"] / mib
        lines.append(f"- {name}: live {live:+.1f} MiB, reserved {reserved:+.1f} MiB")
    lines.append("Segment free space:")
    for arm in ("C", "E"):
        releasable = traps[arm]["releasable_bytes"] / mib
        trapped = traps[arm]["trapped_bytes"] / mib
        lines.append(f"- {arm}: releasable {releasable:.1f} MiB, trapped {trapped:.1f} MiB")
    if diff["extra"]:
        lines.append("Largest surplus holdings:")
        for block in diff["extra"][:5]:
            state = "live" if block["live"] else "not live"
            size = block["size"] / mib
            lines.append(f"- {block['count']} x {size:.1f} MiB {block['klass']} ({state})")
    return "\n".join(lines)
