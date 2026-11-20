"""Tables for the PR from two or three ``train_steps.json`` files (harness only, no GPU needed).

    python p72_train_report.py --a a/train_steps.json --b b/train_steps.json --c c/train_steps.json

Prints the performance table (training time, logical expert fetches, logical expert bytes), the
adapter-equality statement for A and C, and, when both runs hold held-out NLL, the quality table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

HARNESS = Path(__file__).resolve().parent
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from p72_train_compare import first_difference, nll_summary, paired_bootstrap_delta  # noqa: E402
from rich.console import Console  # noqa: E402

console = Console()


def performance_table(runs: Dict[str, Dict[str, Any]]) -> str:
    names = list(runs)
    rows = [
        "| Metric | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
        "| Training time | " + " | ".join(f"{runs[n]['training_seconds']:.2f} s" for n in names)
        + " |",
        "| Logical expert fetches | "
        + " | ".join(f"{runs[n]['total_fetches']:,}" for n in names) + " |",
        "| Logged logical expert bytes | "
        + " | ".join(f"{runs[n]['total_logical_bytes'] / 1e9:.1f} GB" for n in names) + " |",
    ]
    return "\n".join(rows)


def equality_text(a: Dict[str, Any], c: Dict[str, Any]) -> str:
    """The adapter-equality statement. Equality is only a claim about the exact MBR when both runs
    were deterministic (without it two runs of the same arm differ, atomics add in any order), so
    the text says which it was."""
    result = first_difference(a["steps"], c["steps"])
    steps = result["steps_compared"]
    modes = [run.get("deterministic") for run in (a, c)]
    if modes == [True, True]:
        mode = " in torch deterministic mode"
        caveat = ""
    elif None in modes:
        mode = ""
        caveat = " (determinism was not recorded for these runs)"
    else:
        mode = " not in deterministic mode"
        caveat = (" This does not support a claim of bit equality: without deterministic mode "
                  "two runs of the same arm differ from each other.")
    if result["equal"]:
        return (f"The adapter (every LoRA tensor, SHA-256 of the bytes) of the L2L control and of "
                f"the exact MBR matched after {steps} of {steps} steps, bit-for-bit{mode}."
                f"{caveat}")
    return (f"The adapters of the L2L control and of the exact MBR first differ at step "
            f"{result['first_difference_step']} of {steps}{mode}.{caveat}")


def quality_table(a: Dict[str, Any], c: Dict[str, Any], *, seed: int) -> str:
    left, right = a["heldout"]["per_example"], c["heldout"]["per_example"]
    if [(x["group_sha256"], x["target_tokens"]) for x in left] != [
            (x["group_sha256"], x["target_tokens"]) for x in right]:
        raise ValueError("held-out rows of the two runs differ")
    sa, sc = nll_summary(left), nll_summary(right)
    paired = paired_bootstrap_delta([x["nll_token_mean"] for x in left],
                                    [x["nll_token_mean"] for x in right], seed=seed)
    return "\n".join([
        "| Metric | L2L control | Exact MBR |",
        "| --- | ---: | ---: |",
        f"| Mean NLL per example | {sa['mean_per_example_nll']:.6f} "
        f"| {sc['mean_per_example_nll']:.6f} |",
        f"| Token-weighted NLL | {sa['token_weighted_nll']:.6f} | {sc['token_weighted_nll']:.6f} |",
        "",
        f"Mean paired difference, exact MBR minus control: {paired['mean_delta']:+.6f} NLL per "
        f"example, 95% paired bootstrap interval [{paired['ci95'][0]:+.6f}, "
        f"{paired['ci95'][1]:+.6f}] ({paired['draws']} draws, seed {paired['seed']}).",
    ])


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="L2L A0")
    parser.add_argument("--b", type=Path, help="B0")
    parser.add_argument("--c", type=Path, required=True, help="exact MBR")
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args(argv)
    a, c = (json.loads(p.read_text(encoding="utf-8")) for p in (args.a, args.c))
    runs = {"L2L A0": a}
    if args.b:
        runs["MBR B0"] = json.loads(args.b.read_text(encoding="utf-8"))
    runs["Exact MBR"] = c
    console.out(performance_table(runs), end="\n\n")
    console.out(equality_text(a, c), end="\n\n")
    if "heldout" in a and "heldout" in c:
        console.out(quality_table(a, c, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
