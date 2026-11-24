#!/usr/bin/env python3
"""Ticket 16: the CPU cross of the stored bits, E against S (and both against A0).

The exact paths were proven against A0 in two separate ladders: E (``lt-fused``) at k = 1,2,4,8
in ticket 03 (``benchmarks/mbr-moonlight-bits-20261122/``) and S (``segmented``) in ticket 14
(``benchmarks/mbr-moonlight-cdiag-20261124/``). This script closes the transitive gap directly
on the CPU: ``torch.equal`` on the loss and all 216 LoRA gradients of each cell, plus the
recorded ``dataset_sha256`` of each side.

The tensors are not committed (about 28 MB each). The E side lives under the ticket 03 results
folder in ``~/moonlight-local``; the S side under the ticket 14 tensor folder, where the
``C-k2-s64.pt`` file was overwritten by a plain-C reproduction run (ticket 14 README), so that
one cell is read from the direct proof folder instead.

    python bits_cross.py \
        --e-root $HOME/moonlight-local/results/bits \
        --s-dir benchmarks/mbr-moonlight-cdiag-20261124/tensors \
        --s-a0-dir benchmarks/mbr-moonlight-cdiag-20261124/tensors \
        --s-cell 2,64=benchmarks/mbr-moonlight-cdiag-20261124/direct-proof/tensors/C-k2-s64.pt \
        --e-rows benchmarks/mbr-moonlight-bits-20261122/s64/rows.jsonl \
                 benchmarks/mbr-moonlight-bits-20261122/s512-math/rows.jsonl \
        --s-sha benchmarks/mbr-moonlight-cdiag-20261124/cross-day/bits.json \
        --out benchmarks/mbr-moonlight-esc-speed-20261125/bits-cross.json

Benchmark code, not shipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CELLS: Tuple[Tuple[int, int], ...] = tuple((k, seq) for seq in (64, 512) for k in (1, 2, 4, 8))
E_SUBDIR = {64: Path("s64") / "tensors", 512: Path("s512-math") / "tensors"}
GRAD_TOTAL = 216


def cross_cell(e_path: Path, s_path: Path) -> Dict[str, Any]:
    """``torch.equal`` on the loss and every LoRA gradient of one cell; NaN never equals NaN."""
    import torch

    e = torch.load(e_path, weights_only=True)
    s = torch.load(s_path, weights_only=True)
    e_loss, s_loss = e["loss"], s["loss"]
    loss_equal = bool(tuple(e_loss.shape) == tuple(s_loss.shape)
                      and torch.equal(e_loss, s_loss))
    e_grads, s_grads = e["grads"], s["grads"]
    names = sorted(set(e_grads) | set(s_grads))
    differing = [name for name in names
                 if name not in e_grads or name not in s_grads
                 or not torch.equal(e_grads[name], s_grads[name])]
    max_abs = 0.0
    for name in differing:
        if name in e_grads and name in s_grads:
            delta = (e_grads[name].float() - s_grads[name].float()).abs().max().item()
            max_abs = max(max_abs, delta)
    return {"e_file": str(e_path), "s_file": str(s_path), "loss_equal": loss_equal,
            "grad_total": len(names), "grads_differing": len(differing),
            "differing_names": differing, "max_abs_diff": max_abs}


def sha_by_seq(rows_paths: Iterable[Path]) -> Dict[int, set]:
    """The recorded ``dataset_sha256`` values of each sequence length, from the rows of a side."""
    found: Dict[int, set] = {}
    for path in rows_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("dataset_sha256"):
                found.setdefault(int(row["seq"]), set()).add(row["dataset_sha256"])
    return found


def _shas(path: Optional[Path]) -> Dict[int, set]:
    """The S side stored its hashes once for the campaign (``cross-day/bits.json``)."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    found: Dict[int, set] = {}
    for cell in data.get("cells", []):
        if cell.get("dataset_sha256"):
            found.setdefault(int(cell["seq"]), set()).add(cell["dataset_sha256"])
    return found


def _override(values: Sequence[str]) -> Dict[Tuple[int, int], Path]:
    parsed: Dict[Tuple[int, int], Path] = {}
    for value in values:
        key, _, path = value.partition("=")
        k, _, seq = key.partition(",")
        parsed[(int(k), int(seq))] = Path(path)
    return parsed


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e-root", required=True, type=Path,
                        help="ticket 03 results folder (s64/tensors, s512-math/tensors)")
    parser.add_argument("--s-dir", required=True, type=Path,
                        help="ticket 14 tensor folder (C-k*-s*.pt, the segmented runs)")
    parser.add_argument("--a0-dir", type=Path, default=None,
                        help="A0 tensors of the same windows, for the transitive column")
    parser.add_argument("--s-cell", action="append", default=[], metavar="K,SEQ=PATH",
                        help="read one S cell from another file (the k=2,S=64 overwrite)")
    parser.add_argument("--e-rows", nargs="+", type=Path, default=[],
                        help="rows.jsonl files of the E side, for the dataset hashes")
    parser.add_argument("--s-sha", type=Path, default=None,
                        help="cross-day/bits.json of the S side, for the dataset hashes")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    overrides = _override(args.s_cell)
    e_shas = sha_by_seq(args.e_rows)
    s_shas = _shas(args.s_sha)
    expected = {seq: sorted(values)[0] for seq, values in e_shas.items() if len(values) == 1}
    sha_match = {seq: values == {expected[seq]}
                 for seq, values in s_shas.items() if seq in expected}

    cells: List[Dict[str, Any]] = []
    for k, seq in CELLS:
        e_path = args.e_root / E_SUBDIR[seq] / f"E-k{k}-s{seq}.pt"
        s_path = overrides.get((k, seq), args.s_dir / f"C-k{k}-s{seq}.pt")
        cell = {"k": k, "seq": seq, "dataset_sha256": expected.get(seq)}
        try:
            cell.update(cross_cell(e_path, s_path))
            cell["equal"] = cell["loss_equal"] and cell["grads_differing"] == 0
            if args.a0_dir is not None:
                a0 = cross_cell(args.a0_dir / f"A0-k{k}-s{seq}.pt", e_path)
                cell["e_vs_a0_equal"] = a0["loss_equal"] and a0["grads_differing"] == 0
                a0s = cross_cell(args.a0_dir / f"A0-k{k}-s{seq}.pt", s_path)
                cell["s_vs_a0_equal"] = a0s["loss_equal"] and a0s["grads_differing"] == 0
        except FileNotFoundError as missing:
            cell.update({"equal": None, "error": str(missing)})
        cells.append(cell)

    report = {"cells": cells,
              "expected_dataset_sha": expected,
              "e_dataset_sha": {str(seq): sorted(values) for seq, values in e_shas.items()},
              "s_dataset_sha": {str(seq): sorted(values) for seq, values in s_shas.items()},
              "dataset_sha_agrees": sha_match,
              "equal_cells": sum(1 for cell in cells if cell.get("equal")),
              "total_cells": len(cells)}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Bits cross E against S (ticket 16)", "",
             f"{report['equal_cells']} of {report['total_cells']} cells are bit-equal "
             "on the loss and all 216 LoRA gradients.", "",
             "| k | S | loss equal | grads differing | max abs | E==A0 | S==A0 | dataset sha |",
             "|---|---|---|---|---|---|---|---|"]
    for cell in cells:
        if cell.get("equal") is None:
            lines.append(f"| {cell['k']} | {cell['seq']} | missing | - | - | - | - | - |")
            continue
        lines.append(
            f"| {cell['k']} | {cell['seq']} | {'yes' if cell['loss_equal'] else 'no'} | "
            f"{cell['grads_differing']} / {cell['grad_total']} | "
            f"{cell['max_abs_diff']:.1e} | "
            f"{cell.get('e_vs_a0_equal', '-')} | {cell.get('s_vs_a0_equal', '-')} | "
            f"{'agrees' if sha_match.get(cell['seq'], True) else 'differs'} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"equal_cells": report["equal_cells"], "total": report["total_cells"],
                      "dataset_sha_agrees": sha_match}))
    return 0 if report["equal_cells"] == report["total_cells"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
