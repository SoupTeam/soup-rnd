"""MBR invariant minus MBR native per seed, from the stage-2 per-example eval rows.

Uses the same paired bootstrap as the evaluated contrasts (``_bootstrap_mean_delta``), after
checking that all four progress files of a seed score the same examples in the same order.

    python benchmarks/harness/p3_a2_stage2_contrast.py \
        benchmarks/results/p3-a2-stage2-20261118/vast-rtx4060ti-quality/eval
"""

import json
import sys
from pathlib import Path

from p3_a2_quality_common import _bootstrap_mean_delta
from rich.console import Console


def rows(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def contrast(eval_dir: Path, seed: int) -> dict:
    files = {
        (arm, side): eval_dir / f"{arm}-minus-control_native-{seed}.{side}.progress.json"
        for arm in ("mbr_native", "mbr_invariant")
        for side in ("control", "mbr")
    }
    loaded = {key: rows(path) for key, path in files.items()}
    identity = {
        key: [(r["group_sha256"], r["type"], r["target_tokens"]) for r in value]
        for key, value in loaded.items()
    }
    if len({json.dumps(v) for v in identity.values()}) != 1:
        raise ValueError(f"seed {seed}: eval rows differ in identity or order")
    invariant = [r["nll_token_mean"] for r in loaded[("mbr_invariant", "mbr")]]
    native = [r["nll_token_mean"] for r in loaded[("mbr_native", "mbr")]]
    # Same argument order as the evaluated contrasts: (control, candidate) -> candidate - control.
    return _bootstrap_mean_delta(native, invariant)


def main() -> None:
    eval_dir = Path(sys.argv[1])
    result = {str(seed): contrast(eval_dir, seed) for seed in (42, 43, 44)}
    Console().print_json(json.dumps(result))


if __name__ == "__main__":
    main()
