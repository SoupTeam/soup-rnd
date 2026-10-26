#!/usr/bin/env python3
"""A5 ticket 04: the A-B-B-A campaign of ``soup train``, GA (A) against L2L (B), on one card.

Stages, run in this order and each writable on its own with ``--stage``:

``abba``     blocks of A, B, B, A, each run a fresh process of ``l2l_ticket04_arm.py`` watched
             by F1's ``ArmWatch``. Blocks are added by the repeats rule (``blocks_to_add``)
             until it stops asking or ``--max-blocks`` is reached; a block with a void run is
             kept on record and replaced.
``floor``    one GA run on the second SDPA backend, the noise floor of A3-07.
``overflow`` one L2L run whose activation window is larger than the free host RAM, to see
             what the trainer does with it (the A4 spill is not wired into the trainer).

``summary`` is written after every stage that changes it. The working directory ``--work``
holds ``data/`` (from ``prepare_data.py``) and the config files are copied there, so each run
reads the same relative paths. Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

import arm_validity  # noqa: E402
import l2l_measure  # noqa: E402
import l2l_ticket04_arm as arm_module  # noqa: E402

CONFIGS = HARNESS / "l2l_ticket04"
ARM_SCRIPT = HARNESS / "l2l_ticket04_arm.py"
ARM_CONFIG = {"A": "ga.yaml", "B": "l2l.yaml"}
RUN_TIMEOUT_S = 3600

#: One decoder layer's activation chunk for a window of ``batch`` rows, bytes (Mistral-7B,
#: hidden 4096, bf16), rounded up to the power of two the pinned allocator takes.
LAYERS = 32
HIDDEN = 4096
ITEMSIZE = 2


def next_power_of_two(value: int) -> int:
    return 1 << max(0, (int(value) - 1).bit_length())


def overflow_k(free_ram: int, *, batch: int, seq: int, factor: float = 1.3) -> int:
    """Accumulation steps ``k`` that make the pinned activation store ``factor`` x ``free_ram``.

    The trainer refuses a store above 70% of the free RAM, so any ``factor`` over 0.7 is a run
    it must refuse; the default 1.3 leaves a margin. One micro-batch is ``batch`` rows of
    ``seq`` tokens.
    """
    chunk = next_power_of_two(batch * seq * HIDDEN * ITEMSIZE)
    return max(2, math.ceil(factor * free_ram / (LAYERS * chunk)))


def interval(diffs: Sequence[float]) -> Optional[List[float]]:
    return l2l_measure._interval(list(diffs))


def metric_summary(
    blocks: Sequence[Sequence[Dict[str, Any]]], getter: Callable[[Dict[str, Any]], Any]
) -> Dict[str, Any]:
    """Mean of A, mean of B and the 95% interval of A minus B over the valid blocks.

    A value that is ``None`` in any run (a source was missing) makes the metric null for the
    whole table, with the reason left to the run records.
    """
    values = [[getter(entry["measurement"]) for entry in block] for block in blocks]
    if not values or any(value is None for block in values for value in block):
        return {"a": None, "b": None, "a_minus_b": None, "interval_95": None,
                "reason": "no valid block, or a run lacks this value"}
    diffs = [(v[0] + v[3]) / 2 - (v[1] + v[2]) / 2 for v in values]
    return {
        "a": statistics.fmean(v[i] for v in values for i in (0, 3)),
        "b": statistics.fmean(v[i] for v in values for i in (1, 2)),
        "a_minus_b": statistics.fmean(diffs),
        "interval_95": interval(diffs),
        "reason": None,
    }


METRICS = {
    "tokens_per_s": lambda m: m["run"]["tokens_per_s"],
    "peak_vram_allocated_bytes": lambda m: m["run"]["peak_vram_allocated_bytes"],
    "peak_vram_reserved_bytes": lambda m: m["run"]["peak_vram_reserved_bytes"],
    "peak_rss_bytes_step": lambda m: m["run"]["peak_rss_bytes"],
    "process_peak_rss_bytes": lambda m: m["process_peak_rss_bytes"],
    "process_peak_vram_allocated_bytes": lambda m: m["process_peak_vram_allocated_bytes"],
    "load_read_bytes": lambda m: m["load"]["read_bytes_since_process_start"],
    "step_read_bytes": lambda m: m["run"]["os_read_bytes"],
    "layer_loads_per_step": lambda m: m["run"]["layer_loads_per_step"],
    "nf4_restores": lambda m: m["run"]["nf4_restores"],
    "pinned_store_bytes": lambda m: m["run"]["pinned_store_bytes"],
}


def loss_verdicts(curves: Dict[str, List[List[Optional[float]]]]) -> Dict[str, Any]:
    """The A3-07 rule on every L2L curve against the first GA curve and the noise floor.

    ``curves`` holds ``ga`` (the reference first), ``l2l`` and ``floor`` (one curve or none).
    Also reports how far the other GA runs sit from the reference, the same-arm repeat error.
    """
    ga, l2l, floor = curves["ga"], curves["l2l"], curves["floor"]
    out: Dict[str, Any] = {"reference": "first GA run of the first valid block"}
    if not ga:
        out["problem"] = "no GA curve"
        return out
    ref = ga[0]
    out["ga_repeat_error"] = [arm_module.relative_errors(ref, other) for other in ga[1:]]
    out["l2l"] = [
        arm_module.judge_loss_curve(ref, curve, floor[0] if floor else None) for curve in l2l
    ]
    out["floor_error"] = arm_module.relative_errors(ref, floor[0]) if floor else None
    out["pass"] = bool(out["l2l"]) and all(item["pass"] for item in out["l2l"])
    return out


def losses_of(record: Dict[str, Any]) -> List[Optional[float]]:
    return [item["loss"] for item in record["losses"]]


class Campaign:
    def __init__(self, work: Path, out: Path, card: str):
        self.work, self.out, self.card = work, out, card
        self.runs = out / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.blocks: List[List[Dict[str, Any]]] = []
        self.floor_records: List[Dict[str, Any]] = []
        self.overflow: Optional[Dict[str, Any]] = None
        for name in ARM_CONFIG.values():
            shutil.copy(CONFIGS / name, work / name)
        self.model_path = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    def run_arm(
        self, stem: str, config: str, *, sdpa: str = "MATH"
    ) -> "subprocess.CompletedProcess[bytes]":
        shutil.rmtree(self.work / "out", ignore_errors=True)
        env = dict(os.environ, PYTHONPATH=str(HARNESS.parents[1] / "src"),
                   TOKENIZERS_PARALLELISM="false")
        with (self.runs / f"{stem}.log").open("w") as log:
            return subprocess.run(
                [sys.executable, str(ARM_SCRIPT), "--config", config, "--sdpa", sdpa,
                 "--out", str(self.runs / f"{stem}.json")],
                cwd=self.work, stdout=log, stderr=subprocess.STDOUT, env=env,
                timeout=RUN_TIMEOUT_S, check=False,
            )

    def load(self, stem: str) -> Dict[str, Any]:
        return json.loads((self.runs / f"{stem}.json").read_text())

    def block(self) -> None:
        round_no = len(self.blocks) + 1
        entries, spans = [], []
        for position, name in enumerate(arm_validity.BLOCK_ORDER):
            stem = f"b{round_no}-p{position}-{name}"
            with arm_validity.ArmWatch(
                arm=name, round=round_no, run=position, label=f"A5-04 {self.card}",
                model_path=self.model_path,
            ) as watch:
                done = self.run_arm(stem, ARM_CONFIG[name])
            if done.returncode:
                raise RuntimeError(f"{stem} exited {done.returncode}; see runs/{stem}.log")
            data = self.load(stem)
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
        self.spans = getattr(self, "spans", []) + [spans]
        (self.out / "blocks.json").write_text(json.dumps(self.blocks, indent=1, default=str))

    def abba(self, max_blocks: int) -> None:
        self.spans: List[List[arm_validity.Run]] = []
        while True:
            if len(self.blocks) >= arm_validity.MIN_BLOCKS and not arm_validity.blocks_to_add(
                self.spans
            ):
                break
            if len(self.blocks) >= max_blocks:
                print("block cap reached; the repeats rule is not satisfied")
                break
            self.block()
            self.write_summary()

    def floor(self) -> None:
        stem = "floor-ga-efficient"
        done = self.run_arm(stem, ARM_CONFIG["A"], sdpa="EFFICIENT_ATTENTION")
        if done.returncode:
            raise RuntimeError(f"{stem} exited {done.returncode}; see runs/{stem}.log")
        self.floor_records.append(self.load(stem))
        self.write_summary()

    def overflow_run(self, free_ram: int, batch: int = 4, seq: int = 256) -> None:
        """One L2L run with the window larger than the free RAM; records what happens."""
        k = overflow_k(free_ram, batch=batch, seq=seq)
        text = (CONFIGS / ARM_CONFIG["B"]).read_text()
        text = text.replace("gradient_accumulation_steps: 4", f"gradient_accumulation_steps: {k}")
        text = text.replace("batch_size: 1", f"batch_size: {batch}")
        text = text.replace("data/alpaca64.jsonl", "data/alpaca_large.jsonl")
        (self.work / "overflow.yaml").write_text(text)
        done = self.run_arm("overflow-l2l", "overflow.yaml")
        log = (self.runs / "overflow-l2l.log").read_text()
        self.overflow = {
            "free_ram_bytes": free_ram, "batch_size": batch, "k": k,
            "returncode": done.returncode, "refused_for_ram": "host RAM" in log,
            "log": "runs/overflow-l2l.log",
        }
        self.write_summary()

    def write_summary(self) -> None:
        kept = [
            block for block in self.blocks
            if not any(entry["spoiled"] for entry in block)
        ]
        summary: Dict[str, Any] = {"card": self.card, "speed": None}
        if self.blocks:
            summary["speed"] = l2l_measure.summarize([
                [{**e, "measurement": e["measurement"]["run"]} for e in block]
                for block in self.blocks
            ])
            summary["metrics"] = {
                name: metric_summary(kept, getter) for name, getter in METRICS.items()
            }
        curves = {
            "ga": [losses_of(e["measurement"]) for b in kept for e in b if e["arm"] == "A"],
            "l2l": [losses_of(e["measurement"]) for b in kept for e in b if e["arm"] == "B"],
            "floor": [losses_of(r) for r in self.floor_records],
        }
        summary["loss"] = loss_verdicts(curves)
        summary["overflow"] = self.overflow
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="directory with data/ from prepare_data.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--max-blocks", type=int, default=7)
    parser.add_argument("--stage", action="append", choices=["abba", "floor", "overflow"])
    parsed = parser.parse_args()
    campaign = Campaign(Path(parsed.work), Path(parsed.out), parsed.card)
    stages = parsed.stage or ["abba", "floor", "overflow"]
    if "abba" in stages:
        campaign.abba(parsed.max_blocks)
    if "floor" in stages:
        campaign.floor()
    if "overflow" in stages:
        import psutil

        campaign.overflow_run(int(psutil.virtual_memory().available))
    print(json.dumps(json.loads((campaign.out / "summary.json").read_text()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
