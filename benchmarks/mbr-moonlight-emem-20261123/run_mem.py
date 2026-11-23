"""Measure one Moonlight C or E window in normal mode."""

import argparse
import json
from pathlib import Path
from typing import Any

from emem_analysis import MODEL, MOONLIGHT_TARGETS, canonical_row, window_label
from rich.console import Console

_AC_ONLINE = Path("/sys/class/power_supply/ACAD/online")


def measurement_parser(description: str | None, *, steps: int) -> argparse.ArgumentParser:
    from benchmarks.harness.mbr_measure import parse_lora_targets

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("C", "E"))
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--seq", required=True, type=int)
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-targets", type=parse_lora_targets, default=MOONLIGHT_TARGETS)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--model-revision", help="Defaults to config.json _commit_hash or unknown")
    return parser


def validate_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.steps < 2 or min(args.k, args.seq, args.lora_r) < 1:
        parser.error("--steps must be at least 2; --k, --seq, and --lora-r must be positive")


def power_state() -> int | None:
    try:
        return int(_AC_ONLINE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def measurement_identity(args: argparse.Namespace, torch: Any) -> dict[str, str]:
    revision = args.model_revision
    if not revision:
        try:
            path = Path(args.source).expanduser() / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
        revision = config.get("_commit_hash") if isinstance(config, dict) else None
    card = (
        torch.cuda.get_device_name()
        if args.device != "cpu" and torch.cuda.is_available()
        else "CPU"
    )
    return {
        "model": MODEL, "model_revision": revision or "unknown", "card": card,
        "window": window_label(args.k, args.seq), "arm": args.arm, "mode": "normal",
        "torch_version": str(torch.__version__),
    }


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    """Call the shared run without changing the model or its normal-mode settings."""
    from benchmarks.harness.mbr_measure import run_one

    return run_one(argparse.Namespace(
        source=args.source, cache=args.cache, out=args.out, arm=args.arm, k=args.k, seq=args.seq,
        steps=args.steps, lora_r=args.lora_r, lora_targets=args.lora_targets, device=args.device,
        kind="speed", deterministic=False, math_attention=False, tokenizer=None, profile=False,
    ))


def main(argv: list[str] | None = None) -> int:
    from benchmarks.harness.mbr_measure import append_jsonl

    parser = measurement_parser(__doc__, steps=3)
    args = parser.parse_args(argv)
    validate_run(parser, args)
    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    identity = measurement_identity(args, torch)
    power = power_state()

    def write_canonical(row: dict[str, Any]) -> None:
        extra = {key: value for key, value in row.items() if key not in identity}
        canonical = canonical_row(**identity, extra={
            **extra, "k": args.k, "seq": args.seq, "power_ac_online": power,
        })
        append_jsonl(args.out / "rows_canonical.jsonl", canonical)
        Console(soft_wrap=True, markup=False, highlight=False).print(json.dumps(canonical))

    try:
        raw = run_measurement(args)
    except Exception as error:
        write_canonical({"status": "CRASH", "error": f"{type(error).__name__}: {error}"})
        raise
    append_jsonl(args.out / "rows.jsonl", raw)
    write_canonical(raw)
    return 0 if raw["status"] == "ok" else 1


if __name__ == "__main__":
    import sys

    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo))
    raise SystemExit(main())
