#!/usr/bin/env python3
"""Driver for G7: the whole measurement in the committed order (ticket 07).

Run on the RTX 3050 laptop on mains after a reboot, with the fixture from
``visit_profile.py fixture``, and only after the commit that holds the rule::

    python benchmarks/harness/visit_measure.py run --root /var/tmp/g7 \\
        --results-dir benchmarks/results/probe-rtx3050-short-step-visit
    python benchmarks/harness/visit_measure.py dry-run

The profile is two rounds with one run per arm per round. Round 1 runs, for each of 16, 32, 64
and 128 tokens, the bare, timer and profile arms, and after the three arms at 16 tokens the
control arm. Round 2 runs the same list reversed. Every run is a fresh process under the arm
watch (``visit_profile.run_arm_process``). The experiment (``tie_experiment``) runs after the
profile, always. Then ``visit_verdicts.recommend`` reads the 16-token rows, the experiment
verdict and the prototype result (``prototype.json`` in the results folder, if it exists).

``<results-dir>/profile`` holds every run JSON and arm record, ``<results-dir>/experiment``
holds the gate, every block and the verdict, and ``summary.json`` holds the profile rows, the
experiment verdict and the recommendation. ``dry-run`` prints the order and starts nothing.

Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import visit_control
import visit_verdicts
from rich.console import Console

#: Tokens per step of the profile.
TOKEN_COUNTS = (16, 32, 64, 128)
#: The arms of each token count, in round 1.
ROUND_ARMS = ("bare", "timer", "profile")
#: The control runs once per round, at this token count, after that token count's own arms.
CONTROL_TOKENS = visit_control.CONTROL_TOKENS
ROUNDS = (1, 2)
#: The rows of every token count. The control and the threshold read 16 tokens only.
ROWS_EVERY_TOKEN_COUNT = ("timer_is_free", "profile_is_readable")

console = Console()

#: Runs one arm: ``(arm=, seq=, round=, run=, tag=, out_dir=)`` to ``{"run", "arm"}``.
RunArm = Callable[..., Dict[str, Any]]
#: Runs the experiment and returns ``{"gate", "blocks", "verdict"}``.
RunExperiment = Callable[[], Mapping[str, Any]]

PROFILE_DIR = "profile"
EXPERIMENT_DIR = "experiment"
PROTOTYPE_FILE = "prototype.json"


def round_plan(round: int) -> List[Tuple[str, int]]:
    """The arms of one round, in order. Round 2 is round 1 reversed."""
    plan: List[Tuple[str, int]] = []
    for seq in TOKEN_COUNTS:
        plan += [(arm, seq) for arm in ROUND_ARMS]
        if seq == CONTROL_TOKENS:
            plan.append(("control", seq))
    return plan if round == 1 else list(reversed(plan))


def run_profile(run_arm: RunArm, out_dir: Path) -> List[Dict[Tuple[str, int], Dict[str, Any]]]:
    """Run both rounds. Returns one ``{(arm, seq): {"run", "arm"}}`` per round."""
    rounds: List[Dict[Tuple[str, int], Dict[str, Any]]] = []
    run = 0
    for round in ROUNDS:
        results: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for arm, seq in round_plan(round):
            run += 1
            console.print(f"[dim]run {run}: round {round}, {arm} at {seq} tokens[/]")
            results[(arm, seq)] = run_arm(
                arm=arm, seq=seq, round=round, run=run, tag=f"round{round}", out_dir=out_dir
            )
        rounds.append(results)
    return rounds


def evaluate_profile(
    rounds: Sequence[Mapping[Tuple[str, int], Mapping[str, Any]]],
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """The rows of every token count, the two rounds combined. At 16 tokens that is all four
    rows. At the other token counts it is the two instrument rows."""
    rows: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for seq in TOKEN_COUNTS:
        per_round = []
        for results in rounds:
            evaluated = visit_verdicts.evaluate(
                bare=results.get(("bare", seq)),
                timer=results.get(("timer", seq)),
                profile=results.get(("profile", seq)),
                control=results.get(("control", seq)),
            )
            if seq != CONTROL_TOKENS:
                evaluated = {name: evaluated[name] for name in ROWS_EVERY_TOKEN_COUNT}
            per_round.append(evaluated)
        rows[seq] = visit_verdicts.combine_rounds(per_round)
    return rows


def run_measurement(
    run_arm: RunArm,
    run_experiment: RunExperiment,
    results_dir: Path,
    prototype: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """The profile, then the experiment, then the recommendation. Writes ``summary.json``."""
    rounds = run_profile(run_arm, results_dir / PROFILE_DIR)
    rows = evaluate_profile(rounds)
    experiment = run_experiment()
    summary: Dict[str, Any] = {
        "profile_rows": {str(seq): seq_rows for seq, seq_rows in rows.items()},
        "experiment": {"gate": experiment["gate"], "verdict": experiment["verdict"]},
        "prototype": prototype,
        **visit_verdicts.recommend(rows[CONTROL_TOKENS], experiment["verdict"], prototype),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    return summary


def dry_run() -> List[str]:
    """The order of the measurement as lines of text."""
    lines = [
        f"profile round {round}: "
        + ", ".join(f"{arm} {seq}" for arm, seq in round_plan(round))
        for round in ROUNDS
    ]
    lines.append(
        "experiment: the bit-exact gate, then A-B-B-A blocks at 16 tokens by the repeats rule "
        "(5-block pilot), void blocks replaced"
    )
    return lines


def _load_prototype(results_dir: Path) -> Optional[Dict[str, Any]]:
    path = results_dir / PROTOTYPE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="the profile, the experiment and the recommendation")
    run.add_argument("--root", type=Path, required=True, help="the fixture root")
    run.add_argument("--results-dir", type=Path, required=True)
    sub.add_parser("dry-run", help="print the order and start nothing")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        for line in dry_run():
            console.print(line)
        return 0
    import tie_experiment
    import visit_profile

    root = args.root.resolve()
    results_dir = args.results_dir.resolve()
    fixture = visit_profile.fixture_paths(root)
    experiment_dir = results_dir / EXPERIMENT_DIR
    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_arm = partial(_run_arm_process, fixture=fixture, root=root)

    def run_experiment() -> Mapping[str, Any]:
        result = tie_experiment.run_experiment(
            lambda: tie_experiment.gate_in_subprocess(
                fixture.weights, fixture.shards, experiment_dir
            ),
            tie_experiment.arm_runner(fixture=fixture, root=root, out_dir=experiment_dir),
        )
        (experiment_dir / "experiment.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result

    summary = run_measurement(run_arm, run_experiment, results_dir, _load_prototype(results_dir))
    console.print(f"recommendation: {summary['recommendation']}")
    for name, branch in summary["branches"].items():
        console.print(f"  {name}: {branch['verdict']}. {branch['reason']}")
    return 0


def _run_arm_process(**kwargs: Any) -> Dict[str, Any]:
    import visit_profile

    return visit_profile.run_arm_process(**kwargs)


if __name__ == "__main__":
    sys.exit(main())
