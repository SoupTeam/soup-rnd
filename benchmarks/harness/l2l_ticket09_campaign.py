#!/usr/bin/env python3
"""A5 ticket 09: A-B-B-A campaigns of ``soup train`` for the layer-call capture, on one card.

Arms (a name, a source tree, a config, an optional bucket):

``ga``             gradient accumulation, tree ``after``
``before``         L2L, tree ``before`` (the capture is a full forward)
``after``          L2L, tree ``after`` (the capture stops at layer 0)
``before-bucket``  L2L, tree ``before``, rows padded to a multiple of 64 (option b)

A campaign is one pair, ``--a`` against ``--b``, in blocks of A, B, B, A watched by F1's
``ArmWatch`` until the repeats rule asks for no more blocks. The harness scripts always come
from this tree; each arm's ``PYTHONPATH`` is its own tree's ``src``. Benchmark harness code,
not shipped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

import arm_validity  # noqa: E402
import l2l_measure  # noqa: E402
import l2l_ticket04_arm as arm04  # noqa: E402
import l2l_ticket04_campaign as camp04  # noqa: E402

CONFIGS = HARNESS / "l2l_ticket04"
ARM_SCRIPT = HARNESS / "l2l_ticket09_arm.py"
LAYERS = 32
RUN_TIMEOUT_S = 3600
ARMS = {
    "ga": {"tree": "after", "config": "ga.yaml", "bucket": 0},
    "before": {"tree": "before", "config": "l2l.yaml", "bucket": 0},
    "after": {"tree": "after", "config": "l2l.yaml", "bucket": 0},
    "before-bucket": {"tree": "before", "config": "l2l.yaml", "bucket": 64},
}


def consistency(steps: Sequence[Dict[str, Any]], layers: int = LAYERS) -> Dict[str, Any]:
    """Whether each warm step's loads are one base plus ``layers`` per capture.

    ``residual`` = layer loads - ``layers`` x captures. If the cause is the capture forward,
    the residual is the same on every step whatever the number of new shapes.
    """
    rows = [
        (s["step"], s["l2l_window"]["new_shapes"], s["l2l_window"]["captures"],
         s["layer_loads"] - layers * s["l2l_window"]["captures"])
        for s in steps[1:] if "l2l_window" in s
    ]
    residuals = sorted({r[3] for r in rows})
    return {
        "steps": len(rows),
        "residuals": residuals,
        "agrees": len(residuals) == 1,
        "captures_equal_new_shapes": all(r[1] == r[2] for r in rows),
        "per_step": [{"step": r[0], "new_shapes": r[1], "captures": r[2], "residual": r[3]}
                     for r in rows],
    }


def _run(key: str):
    return lambda m: m["run"].get(key)


CAPTURE_METRICS = {
    "new_shapes_per_step": _run("new_shapes_per_step"),
    "captures_per_step": _run("captures_per_step"),
    "capture_layer_loads_per_step": _run("capture_layer_loads_per_step"),
    "capture_seconds_per_step": _run("capture_seconds_per_step"),
    "step_s_mean": _run("step_s_mean"),
}


class Campaign:
    def __init__(self, work: Path, out: Path, card: str, trees: Dict[str, str], a: str, b: str):
        self.work, self.out, self.card, self.trees, self.a, self.b = work, out, card, trees, a, b
        self.arms = {"A": ARMS[a], "B": ARMS[b]}
        self.runs = out / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.blocks: List[List[Dict[str, Any]]] = []
        self.spans: List[List[arm_validity.Run]] = []
        for name in ("ga.yaml", "l2l.yaml"):
            shutil.copy(CONFIGS / name, work / name)
        self.model_path = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    def run_arm(self, stem: str, spec: Dict[str, Any]) -> "subprocess.CompletedProcess[bytes]":
        shutil.rmtree(self.work / "out", ignore_errors=True)
        env = dict(os.environ, PYTHONPATH=str(Path(self.trees[spec["tree"]]) / "src"),
                   TOKENIZERS_PARALLELISM="false")
        cmd = [sys.executable, str(ARM_SCRIPT), "--config", spec["config"],
               "--out", str(self.runs / f"{stem}.json")]
        if spec["bucket"]:
            cmd += ["--bucket", str(spec["bucket"])]
        with (self.runs / f"{stem}.log").open("w") as log:
            return subprocess.run(cmd, cwd=self.work, stdout=log, stderr=subprocess.STDOUT,
                                  env=env, timeout=RUN_TIMEOUT_S, check=False)

    def block(self) -> None:
        round_no = len(self.blocks) + 1
        entries, spans = [], []
        for position, name in enumerate(arm_validity.BLOCK_ORDER):
            stem = f"b{round_no}-p{position}-{name}"
            with arm_validity.ArmWatch(
                arm=name, round=round_no, run=position,
                label=f"A5-09 {self.card} {self.a} vs {self.b}", model_path=self.model_path,
            ) as watch:
                done = self.run_arm(stem, self.arms[name])
            if done.returncode:
                raise RuntimeError(f"{stem} exited {done.returncode}; see runs/{stem}.log")
            data = json.loads((self.runs / f"{stem}.json").read_text())
            entry = l2l_measure.run_entry(
                data["run"], watch.record, arm=name, round=round_no, run=position
            )
            entry["measurement"] = data
            entries.append(entry)
            spans.append(arm_validity.Run(
                data["run"]["tokens_per_s"],
                arm_validity.VOID if entry["spoiled"] else entry["outcome"],
            ))
            print(stem, f"{data['run']['tokens_per_s']:.1f} tokens/s", entry["outcome"], flush=True)
        self.blocks.append(entries)
        self.spans.append(spans)
        (self.out / "blocks.json").write_text(json.dumps(self.blocks, indent=1, default=str))
        self.write_summary()

    def abba(self, max_blocks: int) -> None:
        while True:
            if len(self.blocks) >= arm_validity.MIN_BLOCKS and not arm_validity.blocks_to_add(
                self.spans
            ):
                break
            if len(self.blocks) >= max_blocks:
                print("block cap reached; the repeats rule is not satisfied")
                break
            self.block()

    def write_summary(self) -> None:
        kept = [b for b in self.blocks if not any(e["spoiled"] for e in b)]
        both_l2l = "ga" not in (self.a, self.b)
        getters = dict(camp04.METRICS)
        if both_l2l:
            getters.update(CAPTURE_METRICS)
        summary: Dict[str, Any] = {
            "card": self.card, "a": self.a, "b": self.b, "blocks": len(self.blocks),
            "kept_blocks": len(kept),
            "speed": l2l_measure.summarize([
                [{**e, "measurement": e["measurement"]["run"]} for e in block]
                for block in self.blocks
            ]) if self.blocks else None,
            "metrics": {n: camp04.metric_summary(kept, g) for n, g in getters.items()},
        }
        a_curves = [camp04.losses_of(e["measurement"]) for b in kept for e in b if e["arm"] == "A"]
        b_curves = [camp04.losses_of(e["measurement"]) for b in kept for e in b if e["arm"] == "B"]
        if a_curves:
            ref = a_curves[0]
            summary["loss"] = {
                "reference": "first A run of the first valid block",
                "a_repeat_max_rel_error": [
                    max(arm04.relative_errors(ref, c)) for c in a_curves[1:]
                ],
                "b_max_rel_error": [max(arm04.relative_errors(ref, c)) for c in b_curves],
            }
        summary["consistency"] = {
            e["arm"] + str(i): consistency(e["measurement"]["steps"])
            for i, e in enumerate(x for b in self.blocks for x in b)
            if self.arms[e["arm"]]["config"] == "l2l.yaml"
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--before", required=True, help="source tree before the change")
    parser.add_argument("--after", required=True, help="source tree after the change")
    parser.add_argument("--a", required=True, choices=sorted(ARMS))
    parser.add_argument("--b", required=True, choices=sorted(ARMS))
    parser.add_argument("--max-blocks", type=int, default=4)
    parsed = parser.parse_args()
    campaign = Campaign(Path(parsed.work), Path(parsed.out), parsed.card,
                        {"before": parsed.before, "after": parsed.after}, parsed.a, parsed.b)
    campaign.abba(parsed.max_blocks)
    print(json.dumps(json.loads((campaign.out / "summary.json").read_text()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
