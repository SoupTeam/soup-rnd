"""Compare C and E allocator snapshots. Only load trusted local pickle artifacts."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from emem_analysis import describe_diff, holdings_diff
from rich.console import Console


def _load_snapshot(path: Path) -> dict[str, Any]:
    # _dump_snapshot writes a plain pickle, not a torch.save archive.
    with path.open("rb") as handle:
        return pickle.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--e", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)

    diff = holdings_diff(_load_snapshot(args.c), _load_snapshot(args.e))
    summary = describe_diff(diff)
    Console(soft_wrap=True, markup=False, highlight=False).print(summary)
    if args.json is not None:
        args.json.write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(summary + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys

    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo))
    raise SystemExit(main())
