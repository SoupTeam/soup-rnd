"""Ticket 14, B1: A0 plain (ticket 03) against C with the segmented expert GEMM (the ladder).

The ladder compared C-seg with A0+seg in every window; only k=2, S=64 had a direct control
against plain A0. This script closes that gap on the CPU from stored tensors:

- A0 plain: ``<bits>/<s64|s512-math>/tensors/A0-k<K>-s<SEQ>.pt`` of ticket 03's campaign
  (20261122), still in ``~/moonlight-local/results/bits``;
- A0+seg and C-seg: the ticket 14 ladder tensors (20261124);
- the recorded A0+seg against C-seg verdict: ``ladder/k<K>-s<SEQ>/bits.json``.

Each pair is ``torch.equal`` on the loss and all 216 LoRA gradients. A cell is covered only when
the C tensor on disk is the ladder's C-seg artifact and the direct comparison passes; a tensor
that differs from A0+seg before the direct step is reported as not covered, never silently
counted. Writes ``bits.json`` and ``bits.md``; no GPU is used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import compare_bits
from rich.console import Console

CELLS: tuple[tuple[int, int], ...] = (
    (1, 64), (2, 64), (4, 64), (8, 64), (1, 512), (2, 512), (4, 512), (8, 512),
)
_WINDOWS = {64: "s64", 512: "s512-math"}


def window_dir(seq: int) -> str:
    """The ticket 03 result folder of the window: ``s64`` or the math-backend ``s512-math``."""
    return _WINDOWS[seq]


def compare_paths(paths: tuple[Path, Path], labels: tuple[str, str]) -> dict[str, Any]:
    """``torch.equal`` on the loss and every gradient of two saved runs."""
    return compare_bits.compare(paths, labels)


def equal(report: dict[str, Any]) -> bool:
    """A pair is bitwise equal only when the loss and all gradients are."""
    return bool(report["loss_bitwise_equal"]) and not report["grad_differing"]


def cell_verdict(plain: Path, segmented: Path, candidate: Path) -> str:
    """Whether a cell has a direct A0-plain against C-seg comparison from the tensors on disk.

    ``direct`` is the only covered verdict. ``not-the-ladder-artifact`` means the C tensor differs
    from A0+seg, so it cannot be the ladder's C-seg capture (the ladder recorded that pair equal);
    the cell is reported as uncovered instead of counting a different run as C-seg."""
    if not all(path.exists() for path in (plain, segmented, candidate)):
        return "missing-tensor"
    seg_against_c = compare_paths((segmented, candidate), ("A0+seg", "C-seg"))
    direct = compare_paths((plain, candidate), ("A0 plain", "C-seg"))
    if not equal(seg_against_c):
        return "inconsistent" if equal(direct) else "not-the-ladder-artifact"
    return "direct"


def dataset_sha(rows_path: Path) -> str:
    """The single dataset hash of the bits rows in one window; rows that disagree are an error."""
    hashes = {
        row["dataset_sha256"]
        for row in (json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines())
        if row.get("kind") == "bits" and row.get("dataset_sha256")
    }
    if len(hashes) != 1:
        raise ValueError(f"{rows_path}: expected one dataset hash, found {sorted(hashes)}")
    return hashes.pop()


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    """The verdict and its size, without the 216 per-gradient records."""
    return {key: report[key] for key in (
        "loss_bitwise_equal", "grad_total", "grad_differing", "max_abs_diff", "grad_rel_l2",
        "min_cos", "differing_names",
    )}


def _ladder(ladder_dir: Path, k: int, seq: int) -> dict[str, Any] | None:
    path = ladder_dir / f"k{k}-s{seq}" / "bits.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"loss_bitwise_equal": payload["loss_bitwise_equal"],
            "grad_differing": payload["grad_differing"]}


def collect(bits: Path, campaign: Path) -> list[dict[str, Any]]:
    """One record per ladder cell, with the three comparisons and its coverage."""
    tensors = campaign / "tensors"
    ladder_dir = campaign / "ladder"
    cells = []
    for k, seq in CELLS:
        plain = bits / window_dir(seq) / "tensors" / f"A0-k{k}-s{seq}.pt"
        segmented = tensors / f"A0-k{k}-s{seq}.pt"
        candidate = tensors / f"C-k{k}-s{seq}.pt"
        verdict = cell_verdict(plain, segmented, candidate)
        direct = compare_paths((plain, candidate), ("A0 plain", "C-seg"))
        plain_against_seg = compare_paths((plain, segmented), ("A0 plain", "A0+seg"))
        cells.append({
            "k": k, "seq": seq, "verdict": verdict,
            "dataset_sha256": dataset_sha(bits / window_dir(seq) / "rows.jsonl"),
            "plain_against_seg": _summary(plain_against_seg),
            "ladder_seg_against_c": _ladder(ladder_dir, k, seq),
            "plain_against_c": _summary(direct),
        })
    return cells


def _table(cells: Iterable[dict[str, Any]]) -> str:
    lines = [
        "| k, S | A0 plain vs A0+seg | ladder A0+seg vs C-seg | A0 plain vs C-seg | covered |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        direct = cell["plain_against_c"]
        ladder = cell["ladder_seg_against_c"]
        plain = cell["plain_against_seg"]
        ladder_text = (
            "missing" if ladder is None
            else f"{'equal' if ladder['loss_bitwise_equal'] else 'differs'}, "
                 f"{ladder['grad_differing']} / 216"
        )
        lines.append(
            f"| k={cell['k']}, S={cell['seq']} | {_pair(plain)} | {ladder_text} | "
            f"{_pair(direct)} | {'yes' if cell['verdict'] == 'direct' else cell['verdict']} |"
        )
    return "\n".join(lines) + "\n"


def _pair(report: dict[str, Any]) -> str:
    loss = "equal" if report["loss_bitwise_equal"] else "differs"
    return f"{loss}, {report['grad_differing']} / {report['grad_total']}"


def main(argv: list[str] | None = None) -> int:
    """Compare the stored A0 plain and ladder C-seg tensors of the eight cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=Path,
                        default=Path(os.path.expanduser("~/moonlight-local/results/bits")),
                        help="ticket 03 result root with s64/ and s512-math/")
    parser.add_argument("--campaign", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <campaign>/cross-day")
    args = parser.parse_args(argv)
    out = args.out or args.campaign / "cross-day"
    cells = collect(args.bits, args.campaign)
    payload = {
        "description": "ticket 14 B1: A0 plain (20261122) against ladder C-seg (20261124)",
        "cells": cells,
        "covered": sum(cell["verdict"] == "direct" for cell in cells),
        "total": len(cells),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "bits.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "bits.md").write_text(_table(cells), encoding="utf-8")
    console = Console(soft_wrap=True, markup=False, highlight=False)
    console.print(_table(cells))
    console.print(f"covered {payload['covered']} of {payload['total']} cells")
    return 0 if payload["covered"] == payload["total"] else 1


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo), str(Path(__file__).resolve().parent))
    raise SystemExit(main())
