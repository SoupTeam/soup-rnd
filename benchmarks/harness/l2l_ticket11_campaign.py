#!/usr/bin/env python3
"""A5 ticket 11: GA (A), L2L (B), ordinary accumulation (C1) and the largest ordinary batch (C2).

Stages: ``probe`` (C1 fits? then the C2 batch boundary by doubling), ``abba`` (blocks in the
order A-B-C1-C2-C2-C1-B-A, written to ``order.json`` before the first run, added by the F1 repeats
rule applied to each pair (A, X)), ``floor`` (A on the second SDPA backend, the A3-07 noise
floor for C1 against A). Benchmark harness code, not shipped.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

CONFIGS_04 = HARNESS / "l2l_ticket04"
CONFIGS = HARNESS / "l2l_ticket11"
ROWS = 64


def epochs_for(batch: int, *, rows: int = ROWS, warm_steps: int = 8) -> int:
    """Epochs that give ``warm_steps`` steps after step 0, which the speed leaves out."""
    per_epoch = max(1, rows // batch)
    return max(1, math.ceil((warm_steps + 1) / per_epoch))


def c2_config(c1_text: str, batch: int, epochs: int) -> str:
    """The C1 config with ``batch_size`` = ``batch``, no accumulation and ``epochs`` epochs."""
    text = c1_text
    for pattern, new in (
        (r"^  batch_size: \d+$", f"  batch_size: {batch}"),
        (r"^  gradient_accumulation_steps: \d+$", "  gradient_accumulation_steps: 1"),
        (r"^  epochs: \d+$", f"  epochs: {epochs}"),
    ):
        text, count = re.subn(pattern, new, text, flags=re.M)
        if count != 1:
            raise ValueError(f"c1 config: {pattern!r} must match once, matched {count}")
    return text.replace(
        "# A5 ticket 11, arm C1: ordinary QLoRA with gradient accumulation (k = 4)",
        f"# A5 ticket 11, arm C2: ordinary batch {batch}, {epochs} epochs",
    )


def find_boundary(run_batch, *, start: int = 2, max_batch: int = ROWS) -> dict:
    """Double the batch from ``start`` until ``run_batch`` fails or ``max_batch`` is reached.

    ``run_batch(batch)`` returns ``{"ok", "peak_allocated", "peak_reserved"}``. Returns the
    largest batch that ran, the first that failed (None if none did), the peaks of both and
    every try.
    """
    tried, largest, failed, peaks = [], None, None, {}
    batch = start
    while batch <= max_batch:
        result = run_batch(batch)
        tried.append({"batch_size": batch, **result})
        peaks[batch] = {k: result[k] for k in ("peak_allocated", "peak_reserved")}
        if not result["ok"]:
            failed = batch
            break
        largest = batch
        batch *= 2
    return {
        "largest_ok": largest, "failed_at": failed, "tried": tried,
        "peak_at_largest_ok": peaks.get(largest), "peak_at_failure": peaks.get(failed),
    }


def block_order(arms):
    """Run order of one block: the arms, then the arms reversed (A-B-C1-C2-C2-C1-B-A)."""
    return (*arms, *reversed(arms))


def pair_block(runs, order, arm):
    """The block as an A, X, X, A block of F1 for the pair (A, ``arm``).

    A is first and last in the order, ``arm`` sits at mirrored places. One void run in the
    whole block voids all four runs: the block is replaced for every pair.
    """
    import arm_validity

    last = len(order) - 1
    at = order.index(arm)
    picked = [runs[0], runs[at], runs[last - at], runs[last]]
    if any(run.outcome == arm_validity.VOID for run in runs):
        return [arm_validity.Run(run.value, arm_validity.VOID) for run in picked]
    return picked


def blocks_to_add(blocks, order):
    """Blocks the F1 rule still asks for: the largest need over the pairs (A, X)."""
    import arm_validity

    arms = [arm for arm in dict.fromkeys(order) if arm != "A"]
    return max(
        arm_validity.blocks_to_add([pair_block(b, order, arm) for b in blocks]) for arm in arms
    )


def speed_summary(blocks, order):
    """Mean tokens/s of each arm and, per arm, mean (arm minus A) per block with a 95% interval.

    ``blocks`` hold one speed per run, in ``order``. Callers pass only valid blocks.
    """
    import statistics

    import l2l_measure

    def arm_mean(block, arm):
        return statistics.fmean(v for v, name in zip(block, order) if name == arm)

    arms = list(dict.fromkeys(order))
    out = {"blocks": len(blocks), "mean": {}, "vs_a": {}}
    for arm in arms:
        out["mean"][arm] = statistics.fmean(arm_mean(b, arm) for b in blocks) if blocks else None
    for arm in arms:
        if arm == "A":
            continue
        diffs = [arm_mean(b, arm) - arm_mean(b, "A") for b in blocks]
        out["vs_a"][arm] = {
            "mean_diff": statistics.fmean(diffs) if diffs else None,
            "interval_95": l2l_measure._interval(diffs),
        }
    return out


# ==========================================================================
# the process side (run on the rented instance; checked by the run, not by tests)
# ==========================================================================
ARM_CONFIG = {"A": "ga.yaml", "B": "l2l.yaml", "C1": "c1.yaml", "C2": "c2.yaml"}
RUN_TIMEOUT_S = 3600
OOM_MARKERS = ("out of memory", "OutOfMemoryError")


class Campaign:
    def __init__(self, work, out, card):
        import shutil

        self.work, self.out, self.card = Path(work), Path(out), card
        self.runs = self.out / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        for name, source in (("ga.yaml", CONFIGS_04), ("l2l.yaml", CONFIGS_04),
                             ("c1.yaml", CONFIGS)):
            shutil.copy(source / name, self.work / name)
        self.blocks, self.spans, self.floor_records = [], [], []
        self.arms, self.order, self.boundary = ["A", "B", "C1", "C2"], None, None

    def run_arm(self, stem, config, *, sdpa="MATH"):
        import os
        import shutil
        import subprocess

        shutil.rmtree(self.work / "out", ignore_errors=True)
        env = dict(os.environ, PYTHONPATH=str(HARNESS.parents[1] / "src"),
                   TOKENIZERS_PARALLELISM="false")
        with (self.runs / f"{stem}.log").open("w") as log:
            return subprocess.run(
                [sys.executable, str(HARNESS / "l2l_ticket11_arm.py"), "--config", config,
                 "--sdpa", sdpa, "--out", str(self.runs / f"{stem}.json")],
                cwd=self.work, stdout=log, stderr=subprocess.STDOUT, env=env,
                timeout=RUN_TIMEOUT_S, check=False)

    def load(self, stem):
        import json

        return json.loads((self.runs / f"{stem}.json").read_text())

    def try_run(self, stem, config):
        """One run whose failure is a result: ``{"ok", "peak_allocated", "peak_reserved"}``."""
        done = self.run_arm(stem, config)
        log = (self.runs / f"{stem}.log").read_text(errors="replace")
        record = self.load(stem) if (self.runs / f"{stem}.json").exists() else {}
        run = record.get("run")
        ok = done.returncode == 0 and run is not None
        end = record.get("end_peaks") or {}
        return {
            "ok": ok,
            "oom": any(m in log or m in str(record.get("failure")) for m in OOM_MARKERS),
            "returncode": done.returncode, "failure": record.get("failure"),
            "peak_allocated": run["peak_vram_allocated_bytes"] if ok else end.get("allocated"),
            "peak_reserved": run["peak_vram_reserved_bytes"] if ok else end.get("reserved"),
            "log": f"runs/{stem}.log",
        }

    def probe(self):
        """C1 fit, then the C2 boundary. Writes ``boundary.json`` and ``c2.yaml``."""
        import json

        c1 = self.try_run("fit-c1", "c1.yaml")
        boundary = {"c1": c1, "c2": None}
        if c1["ok"]:
            c1_text = (CONFIGS / "c1.yaml").read_text()

            def run_batch(batch):
                epochs = epochs_for(batch)
                (self.work / "c2.yaml").write_text(c2_config(c1_text, batch, epochs))
                return self.try_run(f"boundary-c2-b{batch}", "c2.yaml")

            boundary["c2"] = find_boundary(run_batch)
            best = boundary["c2"]["largest_ok"]
            if best:
                text = c2_config(c1_text, best, epochs_for(best))
                (self.work / "c2.yaml").write_text(text)
                (self.out / "c2.yaml").write_text(text)
        self.boundary = boundary
        (self.out / "boundary.json").write_text(json.dumps(boundary, indent=1, default=str))
        self.arms = ["A", "B"] + (["C1"] if c1["ok"] else []) + (
            ["C2"] if boundary["c2"] and boundary["c2"]["largest_ok"] else [])
        return boundary

    def block(self):
        import json

        import arm_validity
        import l2l_measure

        round_no = len(self.blocks) + 1
        entries, spans = [], []
        for position, name in enumerate(self.order):
            stem = f"b{round_no}-p{position}-{name}"
            with arm_validity.ArmWatch(
                arm=name, round=round_no, run=position,
                label=f"A5-11 {self.card}", model_path=self.model_path(),
            ) as watch:
                done = self.run_arm(stem, ARM_CONFIG[name])
            if done.returncode:
                raise RuntimeError(f"{stem} exited {done.returncode}; see runs/{stem}.log")
            data = self.load(stem)
            entry = l2l_measure.run_entry(data["run"], watch.record, arm=name,
                                          round=round_no, run=position)
            entry["measurement"] = data
            entries.append(entry)
            spans.append(arm_validity.Run(
                data["run"]["tokens_per_s"],
                arm_validity.VOID if entry["spoiled"] else entry["outcome"]))
            print(stem, f"{data['run']['tokens_per_s']:.1f} tokens/s", entry["outcome"],
                  flush=True)
        self.blocks.append(entries)
        self.spans.append(spans)
        (self.out / "blocks.json").write_text(json.dumps(self.blocks, indent=1, default=str))
        self.write_summary()

    @staticmethod
    def model_path():
        import os

        return os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    def abba(self, max_blocks):
        import json

        import arm_validity

        self.order = block_order(self.arms)
        (self.out / "order.json").write_text(json.dumps(
            {"order": self.order, "written_before_the_first_run": True}))
        while True:
            if len(self.blocks) >= arm_validity.MIN_BLOCKS and not blocks_to_add(
                self.spans, self.order
            ):
                break
            if len(self.blocks) >= max_blocks:
                print("block cap reached; the repeats rule is not satisfied")
                break
            self.block()

    def floor(self):
        stem = "floor-ga-efficient"
        done = self.run_arm(stem, ARM_CONFIG["A"], sdpa="EFFICIENT_ATTENTION")
        if done.returncode:
            raise RuntimeError(f"{stem} exited {done.returncode}; see runs/{stem}.log")
        self.floor_records.append(self.load(stem))
        self.write_summary()

    def write_summary(self):
        import json
        import statistics

        import l2l_ticket04_arm as arm04
        import l2l_ticket04_campaign as camp04

        kept = [b for b in self.blocks if not any(e["spoiled"] for e in b)]
        summary = {"card": self.card, "arms": self.arms, "order": self.order,
                   "blocks": len(self.blocks), "kept_blocks": len(kept),
                   "boundary": self.boundary}
        summary["speed"] = speed_summary(
            [[e["measurement"]["run"]["tokens_per_s"] for e in b] for b in kept], self.order)
        metrics = {}
        for name, getter in camp04.METRICS.items():
            per_arm = {}
            for arm in dict.fromkeys(self.order):
                values = [getter(e["measurement"]) for b in kept for e in b if e["arm"] == arm]
                per_arm[arm] = None if not values or None in values else statistics.fmean(values)
            metrics[name] = per_arm
        summary["metrics"] = metrics
        summary["step_s_mean"] = {
            arm: statistics.fmean(e["measurement"]["run"]["step_s_mean"]
                                  for b in kept for e in b if e["arm"] == arm)
            for arm in dict.fromkeys(self.order)
        } if kept else None
        curves = {arm: [camp04.losses_of(e["measurement"]) for b in kept for e in b
                        if e["arm"] == arm] for arm in dict.fromkeys(self.order)}
        floor = [camp04.losses_of(r) for r in self.floor_records]
        if curves.get("A") and curves.get("C1"):
            ref = curves["A"][0]
            summary["loss_c1_vs_a"] = [
                arm04.judge_loss_curve(ref, c, floor[0] if floor else None)
                for c in curves["C1"]]
        summary["loss_c2"] = "not compared: other row order and other number of steps"
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--max-blocks", type=int, default=7)
    parser.add_argument("--stage", action="append", choices=["probe", "abba", "floor"])
    parsed = parser.parse_args()
    campaign = Campaign(parsed.work, parsed.out, parsed.card)
    stages = parsed.stage or ["probe", "abba", "floor"]
    if "probe" in stages:
        campaign.probe()
    else:
        saved = Path(parsed.out) / "boundary.json"
        if saved.exists():
            campaign.boundary = json.loads(saved.read_text())
    if "abba" in stages:
        campaign.abba(parsed.max_blocks)
    if "floor" in stages:
        campaign.floor()
    print(json.dumps(json.loads((campaign.out / "summary.json").read_text()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
