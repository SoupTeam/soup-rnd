"""One phase-2 memory cell (ticket 13): the phase-1 runner plus a lever label and the env.

The identity fields stay the phase-1 ones (model, card, window, arm, mode, torch); ``lever``,
``alloc_conf`` and ``env`` say which experiment produced the row. Every row is appended to
``rows_canonical.jsonl`` in ``--out``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PHASE1 = Path(__file__).resolve().parents[1] / "mbr-moonlight-emem-20261123"
sys.path[:0] = [str(PHASE1)]

from emem_analysis import canonical_row  # noqa: E402
from rich.console import Console  # noqa: E402
from run_mem import measurement_identity, measurement_parser, power_state, run_measurement  # noqa: E402

_ENV_KEYS = (
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUBLASLT_WORKSPACE_SIZE",
    "A0_LT_WORKSPACE_BYTES",
    "A0_SCRATCH_PREALLOC_ROWS",
    "A0_SCRATCH_PREALLOC_HID",
    "A0_SCRATCH_PREALLOC_INTER",
    "A0_SCRATCH_WHEN",
    "A0_SCRATCH_PREALLOC_GATHER",
    "A0_SCRATCH_MODE",
)


def env_snapshot() -> dict[str, str]:
    return {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}


def main(argv: list[str] | None = None) -> int:
    from benchmarks.harness.mbr_measure import append_jsonl

    parser = measurement_parser(__doc__, steps=3)
    parser.add_argument("--lever", required=True, help="experiment name of this row")
    parser.add_argument("--note", default="", help="free-form note stored with the row")
    args = parser.parse_args(argv)
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = measurement_identity(args, torch)
    power = power_state()

    def write_canonical(row: dict[str, Any]) -> None:
        extra = {key: value for key, value in row.items() if key not in identity}
        canonical = canonical_row(**identity, extra={
            **extra,
            "k": args.k,
            "seq": args.seq,
            "lever": args.lever,
            "note": args.note,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "env": env_snapshot(),
            "power_ac_online": power,
        })
        append_jsonl(out / "rows_canonical.jsonl", canonical)
        Console(soft_wrap=True, markup=False, highlight=False).print(json.dumps(canonical))

    try:
        raw = run_measurement(args)
    except Exception as error:
        write_canonical({"status": "CRASH", "error": f"{type(error).__name__}: {error}"})
        raise
    append_jsonl(out / "rows.jsonl", raw)
    write_canonical(raw)
    return 0 if raw["status"] == "ok" else 1


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(repo / "src"), str(repo)]
    raise SystemExit(main())
