"""Collect every result row of the ticket 14 campaign into one labeled Markdown list.

Each line names the model and revision, the card, the window, the arm, the mode, the expert GEMM
mode and the torch version, as the campaign's rows require.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _line(row: dict[str, Any]) -> str:
    def number(key: str, scale: float, digits: int) -> str:
        value = row.get(key)
        return "n/a" if value is None else f"{value / scale:.{digits}f}"

    loss = row.get("loss")
    loss_text = "loss n/a" if loss is None else f"loss {loss:.6f}"
    return (
        f"- {row['model']} rev {row['model_revision']}, {row['card']}, window {row['window']}, "
        f"arm {row['arm']}, mode {row['mode']}, expert_gemm {row['expert_gemm']}, "
        f"torch {row['torch_version']}: {row.get('status')}, {loss_text}, "
        f"step {number('step_seconds', 1.0, 2)} s, "
        f"alloc {number('peak_allocated', 2**30, 2)} / reserved "
        f"{number('peak_reserved', 2**30, 2)} / NVML {number('peak_nvml', 2**30, 2)} GiB, "
        f"RAM {number('peak_ram', 2**30, 2)} GiB"
    )


def collect(paths: list[Path]) -> list[str]:
    lines = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lines.append(_line(json.loads(line)))
    return lines


def main(argv: list[str] | None = None) -> int:
    """Write rows.md from the campaign's rows.jsonl files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    lines = collect(args.paths)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
