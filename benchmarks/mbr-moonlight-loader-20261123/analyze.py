#!/usr/bin/env python3
"""Ticket 05, part 2: tables from the bits ladders and the loader speed blocks.

Reads the sweeps written by ``run_bits.sh`` and ``run_speed.sh`` (``rows.jsonl``,
``arm_records.jsonl`` and ``tensors/`` under ``--root``) and writes:

- ``analysis.json``: the bits verdicts with the diff detail, the H-against-G speed gate, per
  block gains, the memory cells and the ArmWatch check summary;
- ``rows.md``: one result line per measured cell, each naming model, card, window, arm, mode and
  torch version (the reporting rule of the map).

``torch.equal`` on the saved loss and every LoRA gradient decides the bits verdicts; the
reference arm is A0 at k <= 8 and C above it, as in the harness. The LoRA names and shapes come
from ``grad_manifest.json`` in the campaign folder, which must equal ``mm.MOONLIGHT_GRAD_SHAPES``;
without it no bits cell has a verdict. Benchmark code, not shipped.

    python analyze.py --root benchmarks/mbr-moonlight-loader-20261123

The raw data (``rows.jsonl``, ``arm_records.jsonl``, ``tensors/``) lives inside this campaign
folder in the repository, never outside it.
"""

from __future__ import annotations

import argparse
import json
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
BITS_FOLDERS = ("s64", "s512-math", "k64")
SPEED_FOLDER = "speed"
GRAD_MANIFEST = "grad_manifest.json"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _torch(rows: Sequence[Dict[str, Any]]) -> str:
    versions = [r.get("environment", {}).get("versions", {}).get("torch")
                for r in rows if r.get("status") == "ok"]
    return next((v for v in versions if v), "unknown")


def _gib(value: Optional[int]) -> str:
    return "-" if value is None else f"{value / 2**30:.2f}"


def _peaks(rows: Sequence[Dict[str, Any]], name: str) -> Optional[int]:
    values = [r[name] for r in rows if r.get(name) is not None]
    return max(values) if values else None


def _result_lines(rows: Sequence[Dict[str, Any]], *, kind: str) -> List[str]:
    """One line per (arm, k, seq) cell, naming model, card, window, arm, mode and torch."""
    cells: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["arm"], row["k"], row["seq"]), []).append(row)
    torch = _torch(rows)
    lines = []
    ordered = sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]))
    for (arm, k, seq), group in ordered:
        status = next((r.get("status") for r in group if r.get("status") != "ok"), "ok")
        losses = [r["loss"] for r in group if r.get("status") == "ok"]
        clocks = [r.get("levers", {}).get("expert_loader") for r in group]
        loader = next((c for c in clocks if c), "async_lease")
        mode = "bits (deterministic)" if kind == "bits" else "speed (normal)"
        text = (f"- {MODEL}, {CARD}, window ({k},{seq}), arm {arm} ({loader} loader), mode "
                f"{mode}, torch {torch}: {status}")
        if losses:
            text += f", loss {losses[-1]:.6f}"
        text += (f", allocated {_gib(_peaks(group, 'peak_allocated'))} GiB"
                 f", reserved {_gib(_peaks(group, 'peak_reserved'))} GiB"
                 f", NVML {_gib(_peaks(group, 'peak_nvml'))} GiB"
                 f", RAM {_gib(_peaks(group, 'peak_ram'))} GiB")
        if len(group) > 1:
            text += f", runs {len(group)}"
        lines.append(text)
    return lines


def _diff_detail(tensor_dir: Path, base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    ref_path = mm.contained_tensor(tensor_dir, base["tensor_file"])
    row_path = mm.contained_tensor(tensor_dir, other["tensor_file"])
    if ref_path is None or row_path is None:
        raise ValueError(f"tensor outside {tensor_dir} or missing: {base['tensor_file']!r}, "
                         f"{other['tensor_file']!r}")
    ref = torch.load(ref_path, weights_only=True)
    row = torch.load(row_path, weights_only=True)
    differing = [name for name in ref["grads"]
                 if not torch.equal(ref["grads"][name], row["grads"][name])]
    max_diff = max((float((ref["grads"][name] - row["grads"][name]).abs().max())
                    for name in differing), default=0.0)
    max_grad = max(float(value.abs().max()) for value in ref["grads"].values())
    return {"loss_equal": bool(torch.equal(ref["loss"], row["loss"])),
            "grads_differing": len(differing), "grads_total": len(ref["grads"]),
            "max_abs_diff": max_diff, "max_abs_grad": max_grad}


def _grad_manifest(root: Path) -> Dict[str, List[int]]:
    """The campaign's LoRA manifest, or {} (no verdict) when the file is missing or differs."""
    return mm.load_grad_manifest(root / GRAD_MANIFEST) or {}


RECORD_FIELDS = ("arm", "k", "seq", "run")


def _binds(row: Dict[str, Any], record: Dict[str, Any]) -> bool:
    """A row binds to its arm record when the record is ok, names the row's arm, and agrees on
    every field of ``RECORD_FIELDS`` that both carry."""
    return (record.get("outcome") == "ok" and "arm" in record
            and all(record[name] == row.get(name) for name in RECORD_FIELDS
                    if name in record and name in row))


def _bound_rows(folder: Path) -> List[Dict[str, Any]]:
    """Rows a bits verdict may read. A row whose outcome is not ok, and a row without an outcome,
    reads as status ``void``, so its pair gets no verdict. When ``arm_records.jsonl`` exists, a
    row must also bind to the record at its position (``_binds``). A folder without records
    binds nothing beyond its own outcome; ``README.md`` names that limit."""
    rows = _read_jsonl(folder / "rows.jsonl")
    path = folder / "arm_records.jsonl"
    records = _read_jsonl(path) if path.exists() else None
    bound: List[Dict[str, Any]] = []
    for position, row in enumerate(rows):
        ok = row.get("outcome") == "ok"
        if records is not None:
            record = records[position] if position < len(records) else {}
            ok = ok and _binds(row, record)
        bound.append(row if ok else dict(row, status="void"))
    return bound


def _bits(folder: Path, grad_manifest: Dict[str, List[int]]) -> Dict[str, Any]:
    rows = _bound_rows(folder)
    verdicts = mm.tensor_ladder_verdict(rows, tensor_dir=folder / "tensors",
                                        manifest=grad_manifest)
    detail: Dict[str, Any] = {}
    for (k, arm), verdict in sorted(verdicts.items()):
        if verdict != "differs":
            continue
        base = next((r for r in reversed(rows) if r["k"] == k and r["arm"] == mm.reference_arm(k)
                     and r.get("status") == "ok"), None)
        other = next(r for r in reversed(rows) if r["k"] == k and r["arm"] == arm)
        if base is not None:
            detail[f"k{k}-{arm}"] = _diff_detail(folder / "tensors", base, other)
    return {"folder": str(folder),
            "verdicts": {f"k{k}-{arm}": v for (k, arm), v in sorted(verdicts.items())},
            "diff_detail": detail,
            "h_equal": {f"k{k}": verdicts.get((k, "H")) for k in sorted({r["k"] for r in rows})}}


def _valid_runs(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows a verdict may read: a void run is never counted and is replaced (``speed-rule.md``,
    ``METHOD.md``); its record stays in ``arm_records.jsonl`` and its raw row in ``rows.jsonl``."""
    return [r for r in rows if r.get("outcome", "ok") == "ok"]


def _excluded_blocks(folder: Path) -> set:
    """Blocks a void run disqualifies. ``speed-rule.md``: a block with a void run is not counted
    and is replaced; the replacement pass runs under its own block number, so the excluded block's
    surviving ``ok`` arm never joins a pair. The file names each exclusion and its reason."""
    path = folder / "excluded_blocks.json"
    if not path.exists():
        return set()
    return {(item["k"], item["seq"], item["block"])
            for item in json.loads(path.read_text(encoding="utf-8"))}


def _speed_rows(folder: Path) -> List[Dict[str, Any]]:
    """Rows a verdict may read: valid runs of blocks no void run disqualified."""
    excluded = _excluded_blocks(folder)
    return [row for row in _valid_runs(_read_jsonl(folder / "rows.jsonl"))
            if (row["k"], row["seq"], row.get("block", 0)) not in excluded]


def _speed(folder: Path) -> Dict[str, Any]:
    rows = _speed_rows(folder)
    records = _read_jsonl(folder / "arm_records.jsonl")
    table = mm.summarize_rows(rows)
    windows: Dict[str, Any] = {}
    for k, seq in mm.LOADER_WINDOWS:
        per_block: Dict[int, Dict[str, List[float]]] = {}
        for row in rows:
            if (row["k"], row["seq"]) == (k, seq) and row.get("status") == "ok":
                per_block.setdefault(row["block"], {}).setdefault(row["arm"], []).append(
                    row["step_seconds"])
        gains, g_steps, h_steps = [], [], []
        for block in sorted(per_block):
            arms = per_block[block]
            if "G" not in arms or "H" not in arms:
                continue
            g, h = statistics.median(arms["G"]), statistics.median(arms["H"])
            g_steps.append(g)
            h_steps.append(h)
            gains.append(1 - h / g)
        if len(gains) >= 2:
            spread = statistics.stdev(gains)
            t_value = (statistics.mean(gains) / (spread / len(gains) ** 0.5)
                       if spread else float("inf") if statistics.mean(gains) > 0 else 0.0)
        else:
            t_value = None
        windows[f"({k},{seq})"] = {
            "G_step_s": round(statistics.mean(g_steps), 2) if g_steps else None,
            "H_step_s": round(statistics.mean(h_steps), 2) if h_steps else None,
            "gains_pct": [round(g * 100, 2) for g in gains],
            "mean_gain_pct": round(statistics.mean(gains) * 100, 2) if gains else None,
            "t": None if t_value is None else round(t_value, 2),
            "mem_G": {name: table.get(("G", k, seq), {}).get(name)
                      for name in ("peak_allocated", "peak_reserved", "peak_nvml", "peak_ram")},
            "mem_H": {name: table.get(("H", k, seq), {}).get(name)
                      for name in ("peak_allocated", "peak_reserved", "peak_nvml", "peak_ram")},
        }
    outcomes = Counter(record["outcome"] for record in records)
    clock_max = [record["gpu_clock"]["sm_clock_mhz"]["max"] for record in records]
    failed_checks = Counter(
        name for record in records for name, check in record["checks"].items()
        if check["outcome"] != "ok")
    return {"folder": str(folder),
            "windows": windows,
            "outcomes": dict(outcomes),
            "check_failures": dict(failed_checks),
            "gpu_clock_max_min": min(c for c in clock_max if c is not None),
            "blocks": max((r.get("block", 0) for r in rows), default=0) + 1}


def _window(k: int, seq: int) -> str:
    return f"({k},{seq})"


def _gate_json(gate: Dict[str, Any]) -> Dict[str, Any]:
    """The harness gate with its windows named as JSON strings. ``lever_gate`` keys its windows
    by ``(k, seq)`` tuples and lists the pairs of its VRAM cells; JSON has no tuple key, so the
    writer names them the way ``speed["windows"]`` and the result lines do."""
    speed = dict(gate["speed"])
    speed["windows"] = {_window(*pair): detail for pair, detail in speed["windows"].items()}
    vram = dict(gate["vram"])
    vram["worse_in"] = [_window(*pair) for pair in vram["worse_in"]]
    vram["oom_relief"] = [_window(*pair) for pair in vram["oom_relief"]]
    return dict(gate, speed=speed, vram=vram)


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)

    bits_rows: List[Dict[str, Any]] = []
    bits: Dict[str, Any] = {}
    grad_manifest = _grad_manifest(args.root)
    for name in BITS_FOLDERS:
        folder = args.root / name
        bits[name] = _bits(folder, grad_manifest)
        bits_rows += _read_jsonl(folder / "rows.jsonl")
    speed_rows = _speed_rows(args.root / SPEED_FOLDER)
    speed = _speed(args.root / SPEED_FOLDER)
    gate = mm.lever_gate(bits_rows + speed_rows, lever="H", previous="G",
                         windows=mm.LOADER_WINDOWS, min_windows=mm.LOADER_MIN_WINDOWS)
    analysis = {"bits": bits, "speed": speed, "gate": _gate_json(gate)}
    (args.root / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    lines = ["# Result lines (ticket 05, part 2)", ""]
    lines += _result_lines(bits_rows, kind="bits")
    lines.append("")
    lines += _result_lines(speed_rows, kind="speed")
    (args.root / "rows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": analysis["gate"], "speed": speed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
