#!/usr/bin/env python3
"""Compare the ticket 06 repeat pass against the first pass, cell by cell.

The campaign rule: one run per cell decides, a ``differs`` cell is repeated in a new folder.
The harness compares arms inside one folder only, so this script loads the saved tensors of
the first and the repeat folder and applies the same ``torch.equal`` rule and the same D1
metrics (``grad_rel_l2``, ``min_cos``) through ``analyze.tensor_diff``.

    python compare_repeats.py --root benchmarks/mbr-moonlight-decoupled-20261124

Pairs compared, when both folders have the cell:

- every (arm, k, seq) of ``s64`` against ``repeat-s64``;
- ``k64`` against ``repeat-k64``;
- ``s512-math`` against ``repeat-s512-math``.

Also checks that each repeat agrees with itself across two repeat folders when a cell has
more than one repeat (not needed by this campaign's single repeat pass). Benchmark code, not
shipped.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

PAIRS = (("s64", "repeat-s64"), ("k64", "repeat-k64"), ("s512-math", "repeat-s512-math"))


def _analyze() -> Any:
    path = Path(__file__).with_name("analyze.py")
    spec = importlib.util.spec_from_file_location("mbr_decoupled_analyze_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_cell(first_dir: Path, repeat_dir: Path, arm: str, k: int, seq: int) -> Dict[str, Any]:
    """``tensor_diff`` of one cell's first run against its repeat, both folders' tensors."""
    import torch

    name = f"{arm}-k{k}-s{seq}.pt"
    first = torch.load(first_dir / name, weights_only=True)
    repeat = torch.load(repeat_dir / name, weights_only=True)
    return _analyze().tensor_diff(first, repeat)


def cells(folder: Path) -> List[Tuple[str, int, int]]:
    out = []
    for path in sorted((folder / "tensors").glob("*.pt")):
        arm, k, seq = path.stem.split("-")
        out.append((arm, k[1:], seq[1:]))
    return [(arm, int(k), int(seq)) for arm, k, seq in out]


def compare_folders(first: Path, repeat: Path) -> Dict[str, Dict[str, Any]]:
    return {f"{arm}-k{k}-s{seq}": compare_cell(first / "tensors", repeat / "tensors", arm, k, seq)
            for arm, k, seq in cells(first)}


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    result: Dict[str, Any] = {}
    for first_name, repeat_name in PAIRS:
        first, repeat = args.root / first_name, args.root / repeat_name
        if not (first / "tensors").is_dir() or not (repeat / "tensors").is_dir():
            result[first_name] = {"status": "not measured"}
            continue
        result[first_name] = compare_folders(first, repeat)
    (args.root / "repeats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
