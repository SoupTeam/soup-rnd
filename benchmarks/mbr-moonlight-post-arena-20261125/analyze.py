#!/usr/bin/env python3
"""Ticket 17: E's bits after the arena, G's bits and speed, and the D+segmented cell.

Reads ``bits/<label>/`` (the deterministic sweeps of ``run_bits.sh``) and ``speed/`` (the
C-E-G-G-E-C blocks of ``run_speed.sh``) and writes:

- ``analysis.json``: the bit cells of E, G, D and DS against their same-run A0, the CPU cross
  of the fresh A0/E/G/D tensors against the stored ticket 03 and ticket 06 tensors, and the
  speed of C, E and G per window with the three paired deltas, their t over the five blocks,
  the memory peaks and the repeats-rule block count;
- ``rows.md``: one result line per measured cell, each naming model, card, window, arm, mode and
  torch version.

The per-block number is the median of an arm's two runs in that block; the window number is the
mean of the per-block medians, as ``speed-rule.md`` says. Benchmark code, not shipped.

    python analyze.py --root benchmarks/mbr-moonlight-post-arena-20261125
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "benchmarks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.harness import mbr_measure as mm  # noqa: E402

MODEL = "Moonlight-16B-A3B"
CARD = "RTX 3050 Laptop sm_86 4 GiB"
SPEED_FOLDER = "speed"
ARMS = ("C", "E", "G")
ARM_LABELS = {"C": "ordered backward", "E": "C + lt-fused",
              "G": "E + async lease + fast row gather", "A0": "plain L2L",
              "D": "decoupled MBR", "DS": "D + segmented"}
DELTAS = (("G", "C"), ("E", "C"), ("G", "E"))
PEAK_KEYS = ("peak_allocated", "peak_reserved", "peak_nvml", "peak_ram")
GRAD_TOTAL = 216
# The stored tensors the fresh runs are crossed against, and the cell each one covers.
BIT_FOLDERS = {
    "s64": ("A0", "E"),
    "s512-math": ("A0", "E"),
    "g-k2-s64": ("A0", "G"),
    "d-k2-s64": ("A0", "D"),
    "ds-k2-s64": ("A0", "DS"),
}
TICKET03_ROOT = Path.home() / "moonlight-local" / "results" / "bits"
TICKET03_ROWS = Path(__file__).resolve().parent.parent / "mbr-moonlight-bits-20261122"
TICKET06_D = (Path(__file__).resolve().parent.parent / "mbr-moonlight-decoupled-20261124"
              / "s64" / "tensors")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _torch(rows: Sequence[Dict[str, Any]]) -> str:
    versions = [r.get("environment", {}).get("versions", {}).get("torch")
                for r in rows if r.get("status") == "ok"]
    return next((v for v in versions if v), "unknown")


def _gib(value: Optional[int]) -> str:
    return "-" if value is None else f"{value / 2**30:.2f}"


def compare_tensors(left: Path, right: Path) -> Dict[str, Any]:
    """``torch.equal`` on the loss and every LoRA gradient of two stored cells."""
    import torch

    first = torch.load(left, weights_only=True)
    second = torch.load(right, weights_only=True)
    loss_equal = bool(tuple(first["loss"].shape) == tuple(second["loss"].shape)
                      and torch.equal(first["loss"], second["loss"]))
    names = sorted(set(first["grads"]) | set(second["grads"]))
    differing = [name for name in names
                 if name not in first["grads"] or name not in second["grads"]
                 or not torch.equal(first["grads"][name], second["grads"][name])]
    max_abs = 0.0
    for name in differing:
        if name in first["grads"] and name in second["grads"]:
            delta = (first["grads"][name].float() - second["grads"][name].float()).abs().max()
            max_abs = max(max_abs, delta.item())
    return {"loss_equal": loss_equal, "grad_total": len(names),
            "grads_differing": len(differing), "max_abs_diff": max_abs}


def _stored_cell(folder: str, arm: str, k: int, stored_root: Path) -> Optional[Path]:
    """The stored tensor of one cell: ticket 03 for A0/E/G, ticket 06 for D, none for DS."""
    seq = 512 if folder == "s512-math" else 64
    series = "s512-math" if seq == 512 else "s64"
    if arm in ("A0", "E"):
        path = stored_root / series / "tensors" / f"{arm}-k{k}-s{seq}.pt"
    elif arm == "G":
        path = stored_root / "s64" / "tensors" / f"G-k{k}-s64.pt"
    elif arm == "D":
        path = TICKET06_D / f"D-k{k}-s64.pt"
    else:
        return None
    return path if path.exists() else None


def expected_dataset_shas() -> Dict[int, str]:
    """The dataset hash ticket 03 recorded per sequence length, from the in-repo rows."""
    found: Dict[int, str] = {}
    for name in ("s64", "s512-math"):
        path = TICKET03_ROWS / name / "rows.jsonl"
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            if row.get("dataset_sha256"):
                found.setdefault(int(row["seq"]), row["dataset_sha256"])
    return found


def bits_report(root: Path, stored_root: Path) -> Dict[str, Any]:
    """Each fresh cell against its same-run A0, the stored-tensor crosses, the dataset hashes.

    A void run (an ArmWatch check failed) is never read: its bit row is as invalid as a timed
    one. A folder whose sweep was refused or never ran is named in ``missing_folders`` instead
    of aborting the report.
    """
    cells: List[Dict[str, Any]] = []
    crosses: List[Dict[str, Any]] = []
    missing: List[str] = []
    shas: Dict[str, Dict[str, List[str]]] = {}
    fresh: Dict[Tuple[str, str, int], Path] = {}
    for folder, arms in BIT_FOLDERS.items():
        path = root / "bits" / folder
        if not (path / "rows.jsonl").exists():
            missing.append(folder)
            continue
        rows = [row for row in _read_jsonl(path / "rows.jsonl")
                if row.get("outcome", "ok") == "ok"]
        seq = 512 if folder.startswith("s512") else 64
        shas[folder] = {}
        for row in rows:
            if row.get("dataset_sha256"):
                values = shas[folder].setdefault(str(row["k"]), [])
                if row["dataset_sha256"] not in values:
                    values.append(row["dataset_sha256"])
            if row.get("status") == "ok" and row.get("tensor_file"):
                fresh[(folder, row["arm"], row["k"])] = path / "tensors" / row["tensor_file"]
        a0 = {r["k"]: r for r in rows if r["arm"] == "A0" and r.get("status") == "ok"}
        for arm in arms:
            if arm == "A0":
                continue
            for row in rows:
                if row["arm"] != arm or row.get("status") != "ok":
                    continue
                base = a0.get(row["k"])
                if base is None or (folder, arm, row["k"]) not in fresh \
                        or (folder, "A0", row["k"]) not in fresh:
                    continue
                verdict = compare_tensors(fresh[(folder, "A0", row["k"])],
                                          fresh[(folder, arm, row["k"])])
                verdict.update({"folder": folder, "k": row["k"], "seq": seq, "arm": arm,
                                "dataset_sha256": row["dataset_sha256"],
                                "loss": row["loss"], "step_seconds": row["step_seconds"]})
                cells.append(verdict)
                for arm_for_cross in ("A0", arm):
                    stored = _stored_cell(folder, arm_for_cross, row["k"], stored_root)
                    if stored is None:
                        continue
                    cross = compare_tensors(fresh[(folder, arm_for_cross, row["k"])], stored)
                    cross.update({"folder": folder, "k": row["k"], "arm": arm_for_cross,
                                  "stored": str(stored)})
                    if cross not in crosses:
                        crosses.append(cross)
    # The same-day D control against DS, and the stored plain-D tensor against the fresh DS.
    if ("d-k2-s64", "D", 2) in fresh and ("ds-k2-s64", "DS", 2) in fresh:
        cross = compare_tensors(fresh[("d-k2-s64", "D", 2)], fresh[("ds-k2-s64", "DS", 2)])
        cross.update({"folder": "ds-k2-s64", "k": 2, "arm": "D vs DS (same day)",
                      "stored": str(fresh[("d-k2-s64", "D", 2)])})
        crosses.append(cross)
    stored_d = TICKET06_D / "D-k2-s64.pt"
    if stored_d.exists() and ("ds-k2-s64", "DS", 2) in fresh:
        cross = compare_tensors(fresh[("ds-k2-s64", "DS", 2)], stored_d)
        cross.update({"folder": "ds-k2-s64", "k": 2, "arm": "DS vs stored D (ticket 06)",
                      "stored": str(stored_d)})
        crosses.append(cross)
    expected = expected_dataset_shas()
    agrees = {folder: {key: (int(key) in expected and values == [expected[int(key)]])
                       for key, values in per.items()}
              for folder, per in shas.items()}
    return {"cells": cells, "crosses": crosses, "missing_folders": missing,
            "dataset_shas": shas, "dataset_sha_expected": {str(k): v for k, v in expected.items()},
            "dataset_sha_agrees": agrees}


def _paired_t(gains: Sequence[float]) -> Optional[float]:
    if len(gains) < 2:
        return None
    spread = statistics.stdev(gains)
    if spread == 0:
        return float("inf") if statistics.mean(gains) > 0 else 0.0
    return statistics.mean(gains) / (spread / len(gains) ** 0.5)


def _arm_steps(rows: Sequence[Dict[str, Any]], *, k: int, seq: int,
               arm: str) -> Dict[int, float]:
    per_block: Dict[int, List[float]] = {}
    for row in rows:
        if (row["k"], row["seq"], row["arm"]) == (k, seq, arm) \
                and row.get("status") == "ok" and row.get("step_seconds") is not None:
            per_block.setdefault(row.get("block", 0), []).append(row["step_seconds"])
    return {block: statistics.median(values) for block, values in per_block.items()}


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def window_stats(rows: Sequence[Dict[str, Any]], *, k: int, seq: int) -> Dict[str, Any]:
    """The three hands' step means, the three paired deltas and the memory peaks."""
    steps = {arm: _arm_steps(rows, k=k, seq=seq, arm=arm) for arm in ARMS}
    means = {arm: _mean(list(values.values())) for arm, values in steps.items()}

    def gains(now: str, before: str) -> Optional[Dict[str, Any]]:
        pairs = [1 - steps[now][block] / steps[before][block]
                 for block in sorted(steps[now])
                 if block in steps[before] and steps[before][block]]
        if not pairs:
            return None
        return {"gains_pct": [round(gain * 100, 2) for gain in pairs],
                "mean_gain_pct": round(statistics.mean(pairs) * 100, 2),
                "t": None if (t := _paired_t(pairs)) is None else round(t, 2)}

    table = mm.summarize_rows(rows)
    per_block = {str(block): {arm: round(steps[arm][block], 3)
                              for arm in ARMS if block in steps[arm]}
                 for block in sorted(set().union(*steps.values()))}
    return {"C_step_s": None if means["C"] is None else round(means["C"], 2),
            "E_step_s": None if means["E"] is None else round(means["E"], 2),
            "G_step_s": None if means["G"] is None else round(means["G"], 2),
            "per_block": per_block,
            "deltas": {f"{now}_vs_{before}": gains(now, before)
                       for now, before in DELTAS},
            "memory": {arm: {name: table.get((arm, k, seq), {}).get(name) for name in PEAK_KEYS}
                       for arm in ARMS}}


def blocks_needed_for(rows: Sequence[Dict[str, Any]], *, k: int, seq: int, now: str,
                      before: str) -> Optional[int]:
    from benchmarks.harness.arm_validity import Run, blocks_needed

    left = _arm_steps(rows, k=k, seq=seq, arm=before)
    right = _arm_steps(rows, k=k, seq=seq, arm=now)
    blocks = [[Run(left[block], "ok"), Run(right[block], "ok"),
               Run(right[block], "ok"), Run(left[block], "ok")]
              for block in sorted(set(left) & set(right))]
    if len(blocks) < 5:
        return None
    return blocks_needed(blocks)


def _valid_runs(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("outcome", "ok") == "ok"]


def _excluded_blocks(folder: Path) -> set:
    path = folder / "excluded_blocks.json"
    if not path.exists():
        return set()
    return {(item["k"], item["seq"], item["block"])
            for item in json.loads(path.read_text(encoding="utf-8"))}


def _speed_rows(folder: Path) -> List[Dict[str, Any]]:
    excluded = _excluded_blocks(folder)
    return [row for row in _valid_runs(_read_jsonl(folder / "rows.jsonl"))
            if (row["k"], row["seq"], row.get("block", 0)) not in excluded]


def speed_report(folder: Path) -> Dict[str, Any]:
    if not (folder / "rows.jsonl").exists():
        return {"folder": str(folder), "missing": True, "windows": {}, "blocks_needed": {},
                "blocks_needed_max": None, "outcomes": {}, "check_failures": {},
                "gpu_clock_max_min": None, "blocks": 0, "runs": 0, "records": 0}
    rows = _speed_rows(folder)
    records = _read_jsonl(folder / "arm_records.jsonl")
    windows = {f"({k},{seq})": window_stats(rows, k=k, seq=seq) for k, seq in mm.CEG_WINDOWS}
    needed = {f"({k},{seq})": {f"{now}_vs_{before}": blocks_needed_for(
        rows, k=k, seq=seq, now=now, before=before) for now, before in DELTAS}
        for k, seq in mm.CEG_WINDOWS}
    outcomes = Counter(record["outcome"] for record in records)
    clock_max = [record["gpu_clock"]["sm_clock_mhz"]["max"] for record in records]
    failed = Counter(name for record in records for name, check in record["checks"].items()
                     if check["outcome"] != "ok")
    return {"folder": str(folder), "windows": windows, "blocks_needed": needed,
            "blocks_needed_max": max((value for per_pair in needed.values()
                                      for value in per_pair.values()
                                      if value is not None), default=None),
            "outcomes": dict(outcomes), "check_failures": dict(failed),
            "gpu_clock_max_min": min((clock for clock in clock_max if clock is not None),
                                     default=None),
            "blocks": max((row.get("block", 0) for row in rows), default=0) + 1,
            "runs": len(rows), "records": len(records)}


def _speed_result_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    cells: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["arm"], row["k"], row["seq"]), []).append(row)
    torch = _torch(rows)
    order = {arm: position for position, arm in enumerate(ARMS)}
    lines = []
    for (arm, k, seq), group in sorted(cells.items(),
                                       key=lambda kv: (kv[0][1], kv[0][2],
                                                       order.get(kv[0][0], len(ARMS)))):
        steps = [row["step_seconds"] for row in group
                 if row.get("status") == "ok" and row.get("step_seconds") is not None]
        text = (f"- {MODEL}, {CARD}, window ({k},{seq}), arm {arm} "
                f"({ARM_LABELS.get(arm, arm)}), mode speed (normal), torch {torch}")
        if steps:
            text += f", step {statistics.median(steps):.2f} s"
        for name, label in zip(PEAK_KEYS, ("allocated", "reserved", "NVML", "RAM")):
            values = [row[name] for row in group if row.get(name) is not None]
            text += f", {label} {_gib(max(values) if values else None)} GiB"
        lines.append(text + f", runs {len(group)}")
    return lines


def _bit_result_lines(cells: Sequence[Dict[str, Any]]) -> List[str]:
    lines = []
    for cell in cells:
        verdict = "equal" if not cell["grads_differing"] and cell["loss_equal"] else "differs"
        lines.append(
            f"- {MODEL}, {CARD}, window ({cell['k']},{cell['seq']}), arm {cell['arm']} "
            f"({ARM_LABELS.get(cell['arm'], cell['arm'])}) against A0, mode bits "
            f"(deterministic): {verdict}, loss equal {cell['loss_equal']}, "
            f"grads differing {cell['grads_differing']} / {cell['grad_total']}, "
            f"max abs {cell['max_abs_diff']:.2e}, dataset_sha {cell['dataset_sha256'][:8]}")


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stored-root", type=Path,
                        default=Path(os.environ.get("MOONLIGHT_ROOT",
                                                    Path.home() / "moonlight-local"))
                        / "results" / "bits",
                        help="ticket 03 tensor root (default $MOONLIGHT_ROOT/results/bits)")
    args = parser.parse_args(argv)

    bits = bits_report(args.root, args.stored_root)
    speed = speed_report(args.root / SPEED_FOLDER)
    analysis = {"model": MODEL, "card": CARD, "bits": bits, "speed": speed}
    (args.root / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str),
                                             encoding="utf-8")
    lines = ["# Result lines (ticket 17, post-arena E, G and D+segmented)", ""]
    if not speed.get("missing"):
        lines += _speed_result_lines(_speed_rows(args.root / SPEED_FOLDER))
    lines += _bit_result_lines(bits["cells"])
    (args.root / "rows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"missing_bit_folders": bits["missing_folders"],
                      "cells": len(bits["cells"]),
                      "bit_failures": [f"{c['folder']} {c['arm']} k{c['k']}"
                                       for c in bits["cells"]
                                       if c["grads_differing"] or not c["loss_equal"]],
                      "speed": {window: {name: stats[f"{name}_step_s"]
                                         for name in ARMS}
                                for window, stats in speed["windows"].items()},
                      "deltas": {window: {name: (value or {}).get("mean_gain_pct")
                                         for name, value in stats["deltas"].items()}
                                 for window, stats in speed["windows"].items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
