"""Stage 1 driver of the ordinary against complete Kanana plan: ABBA blocks inside ``ArmWatch``.

Plan: ``benchmarks/streaming-ordinary-vs-complete-l2l-plan-20261114.md``. Rule and protocol:
``benchmarks/streaming-ordinary-vs-complete-l2l-protocol-20261114.md`` (sections 3, 4 and 6). It
launches ``oc_arm_run.py run`` once per arm run, in a fresh process, inside the ``ArmWatch`` of
``arm_validity``, and keeps every run, void ones included, in ``<root>/runs.jsonl``. Benchmark
harness code, not shipped.

Commands (all take ``--root`` for the output folder, ``--source``, ``--cache`` and ``--device``).

``capture``  The accuracy capture of protocol 3.3: the five arms at each ``--k``, then
             ``oc_arm_run.py judge``. It also checks reads per hit: ``l2l_complete`` must read 3,
             any other number stops the stage (``StageStopError``).
``blocks``   ABBA blocks of one comparison (C1 or C2) at one ``--k``: the pilot of 5 blocks, then
             ``blocks_to_add`` of METHOD.md. A void block is replaced. A run whose process fails
             stops the stage: a crash is not a void, it is not replaced.
``single``   Each of the four arms once at ``--k``, no ABBA, as a characterisation (plan option A
             for k = 8). No verdict comes from it.
``verdict``  Reads ``runs.jsonl`` and the accuracy files and writes ``verdicts.json``: WIN, SMALL,
             NO DIFFERENCE, LOSS or NO VERDICT per comparison and k (protocol 3.2).
``stage1``   ``capture`` at k = 2 and 8, ``blocks`` for C1 and C2 at each ``--full-k``,
             ``single`` at each ``--single-k``, then ``verdict``. Option A of the plan:
             ``--full-k 1 2 4 --single-k 8``.

Ceiling. ``stage.json`` in the root records the start time, the price per hour and the ceilings
(``--ceiling-hours``, ``--ceiling-usd``). A block is not started when it would end past either
one, by the length of the last block of that comparison and k. The first block of a pair has no
history and always starts.

The rule's required checks are the constant ``REQUIRED``, a copy of the committed rule (ticket
06 commits it before the first run). Unknown in one of them gives NO VERDICT.
Ticket 08: ``--two-read`` includes arm B in captures and runs only C2 k=2, then
A/B k=1 and k=2 in ``stage1``. ``blocks --comparison AB`` requires the B accuracy
pairs first. ``--foreign-read-fraction 0.05`` opts into the proposed relative
validity limit and writes both verdicts, with a sensitivity flag. The ticket-08
profile selects the approved 5% rule; legacy commands keep the old rule.

"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rich.console import Console

try:
    from benchmarks.harness import arm_validity
except ImportError:  # run as a script from benchmarks/harness
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import arm_validity  # type: ignore[no-redef]

console = Console()

#: Copy of the committed rule's required checks (plan section 4, protocol 3.1).
REQUIRED = ("suspend", "power", "foreign_reads")
#: comparison -> (arm A, arm B). A is ordinary, B is complete.
COMPARISONS = {"C1": ("ga_ordinary", "ga_complete"),
               "C2": ("l2l_ordinary", "l2l_complete"),
               "AB": ("l2l_complete", "l2l_two_read")}
LEGACY_COMPARISONS = ("C1", "C2")
TWO_READ_SCHEDULE = (("C2", 2), ("AB", 1), ("AB", 2))
#: The accuracy pairs each comparison needs (protocol 3.3).
ACCURACY_PAIRS = {
    "AB": ("l2l_two_read_vs_l2l_complete", "l2l_two_read_vs_l2l_ordinary"),
    "C1": ("ga_complete_vs_ga_ordinary",),
    "C2": ("l2l_complete_vs_l2l_ordinary", "l2l_ordinary_vs_ga_ordinary"),
}
CAPTURE_KS = (2, 8)
CAPTURE_ARMS = ("ga_ordinary", "ga_complete", "l2l_ordinary", "l2l_complete", "ga_floor")
EFFECT_THRESHOLD = 0.10
ALPHA = 0.05
L2L_COMPLETE_READS_PER_HIT = 3.0
GIB = 1 << 30


class StageStopError(RuntimeError):
    """The stage must stop: a precondition of the rule failed. The message is the reason."""


# --------------------------------------------------------------------------
# Statistics: two-sided paired t-test without scipy
# --------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta (Numerical Recipes, betacf)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    numerator, denominator = 1.0, 1.0 - qab * x / qap
    denominator = 1.0 / (denominator if abs(denominator) > tiny else tiny)
    h = denominator
    for iteration in range(1, 300):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        denominator = 1.0 + aa * denominator
        denominator = 1.0 / (denominator if abs(denominator) > tiny else tiny)
        numerator = 1.0 + aa / numerator
        numerator = numerator if abs(numerator) > tiny else tiny
        h *= denominator * numerator
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        denominator = 1.0 + aa * denominator
        denominator = 1.0 / (denominator if abs(denominator) > tiny else tiny)
        numerator = 1.0 + aa / numerator
        numerator = numerator if abs(numerator) > tiny else tiny
        delta = denominator * numerator
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_test_p(diffs: Sequence[float]) -> Tuple[float, float]:
    """``(t, two-sided p)`` of the mean of ``diffs`` against zero. All-equal diffs give p = 1
    for zero and p = 0 for a non-zero constant."""
    n = len(diffs)
    if n < 2:
        raise ValueError("a t-test needs at least two blocks")
    mean = statistics.fmean(diffs)
    spread = statistics.stdev(diffs)
    if spread == 0.0:
        return (0.0, 1.0) if mean == 0.0 else (math.copysign(math.inf, mean), 0.0)
    t = mean / (spread / math.sqrt(n))
    df = n - 1
    return t, _betainc(df / 2.0, 0.5, df / (df + t * t))


# --------------------------------------------------------------------------
# Rows, state and budget
# --------------------------------------------------------------------------
def _subprocess_runner(command: List[str]) -> int:
    return subprocess.run(command).returncode


def _arm_watch(**kwargs: Any) -> Any:
    return arm_validity.ArmWatch(**kwargs)


@dataclass
class Stage:
    root: Path
    source: Path
    cache: Path
    model_path: Path
    label: str
    device: str = "cuda"
    revision: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    ram_limit_gib: Optional[float] = None
    two_read: bool = False
    foreign_read_fraction: float = 0.0
    #: Accuracy capture k values. The ticket-08 series uses (2,) by the user's decision of
    #: 2026-11-15; every other run keeps CAPTURE_KS.
    capture_ks: Tuple[int, ...] = CAPTURE_KS
    #: ``runner(command) -> returncode``; the default launches a subprocess.
    runner: Callable[[List[str]], int] = field(default_factory=lambda: _subprocess_runner)
    watch_factory: Callable[..., Any] = field(default_factory=lambda: _arm_watch)
    clock: Callable[[], float] = field(default_factory=lambda: time.time)

    @property
    def rows_path(self) -> Path:
        return self.root / "runs.jsonl"


def read_rows(stage: Stage) -> List[Dict[str, Any]]:
    if not stage.rows_path.is_file():
        return []
    return [json.loads(line) for line in stage.rows_path.read_text("utf-8").splitlines() if line]


def append_row(stage: Stage, row: Dict[str, Any]) -> None:
    stage.root.mkdir(parents=True, exist_ok=True)
    with stage.rows_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False, default=str) + "\n")


def ensure_budget(stage: Stage, *, rate: float, hours: float, usd: float) -> Dict[str, float]:
    """Read ``stage.json``, or create it with the start time now. The file wins over the flags."""
    path = stage.root / "stage.json"
    if path.is_file():
        return json.loads(path.read_text("utf-8"))
    stage.root.mkdir(parents=True, exist_ok=True)
    state = {"started_unix": stage.clock(), "rate_usd_per_hour": rate,
             "ceiling_hours": hours, "ceiling_usd": usd}
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def check_ceiling(stage: Stage, budget: Dict[str, float], estimate_seconds: float) -> None:
    """Refuse a block that would end past the hours ceiling or the dollar ceiling."""
    end_hours = (stage.clock() - budget["started_unix"] + estimate_seconds) / 3600.0
    if end_hours > budget["ceiling_hours"]:
        raise StageStopError(f"ceiling: the next block would end at {end_hours:.2f} h, past "
                        f"{budget['ceiling_hours']} h")
    spend = end_hours * budget["rate_usd_per_hour"]
    if spend > budget["ceiling_usd"]:
        raise StageStopError(
            f"ceiling: the next block would bring the spend to about {spend:.2f} USD, "
            f"past {budget['ceiling_usd']} USD")


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------
def run_command(stage: Stage, arm: str, k: int, out: Path, *, capture: bool) -> List[str]:
    command = [sys.executable, str(Path(__file__).resolve().parent / "oc_arm_run.py"), "run",
               "--arm", arm, "--k", str(k), "--source", str(stage.source),
               "--cache", str(stage.cache), "--out", str(out), "--device", stage.device]
    if stage.revision:
        command += ["--revision", stage.revision]
    if capture:
        command += ["--capture", "--hits-probe"]
    command += stage.extra_args
    if stage.ram_limit_gib is not None:
        # Same limit as the rule's host: a cgroup scope, swap off (protocol section 5).
        command = ["systemd-run", "--scope", "-q", "-p", f"MemoryMax={stage.ram_limit_gib}G",
                   "-p", "MemorySwapMax=0", *command]
    return command


def watched_run(
    stage: Stage, *, comparison: str, k: int, block: int, position: int, side: str, arm: str,
    capture: bool = False, folder: str = "runs",
) -> Dict[str, Any]:
    """Run one arm inside ``ArmWatch`` and append its row. A failed process stops the stage."""
    out = stage.root / folder / f"{comparison}-k{k}" / f"b{block:02d}p{position}-{arm}"
    round_, run = 2 * block + 1 + position // 2, 4 * block + position + 1
    started = stage.clock()
    watch_options = ({"foreign_read_fraction": stage.foreign_read_fraction}
                     if stage.foreign_read_fraction else {})
    with stage.watch_factory(arm=f"{comparison}:{arm}", round=round_, run=run, label=stage.label,
                             required=REQUIRED, model_path=str(stage.model_path),
                             **watch_options) as watch:
        code = stage.runner(run_command(stage, arm, k, out, capture=capture))
    wall = stage.clock() - started
    (out.parent).mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "watch.json").write_text(json.dumps(watch.record, indent=2, default=str) + "\n",
                                    encoding="utf-8")
    record_path = out / "record.json"
    ok = code == 0 and record_path.is_file()
    record = json.loads(record_path.read_text("utf-8")) if ok else {}
    row = {
        "comparison": comparison, "k": k, "block": block, "position": position, "side": side,
        "arm": arm, "value": record.get("median_step_seconds"),
        "outcome": watch.record["outcome"], "process_ok": ok, "returncode": code,
        "wall_seconds": wall, "setup_seconds": record.get("setup_seconds"),
        "run_seconds": record.get("run_seconds"), "dir": str(out),
        "reads_per_hit": record.get("reads_per_hit"), "capture": capture,
        "vram_peak_allocated_bytes": record.get("vram_peak_allocated_bytes"),
        "vram_peak_reserved_bytes": record.get("vram_peak_reserved_bytes"),
        "rss_after_last": (record.get("rss_bytes") or {}).get("after_last"),
        "expert_fetches_per_window": record.get("expert_fetches_per_window"),
        "host_cache_bytes": record.get("host_cache_bytes"),
    }
    if stage.foreign_read_fraction:
        evidence = watch.record.get("checks", {}).get("foreign_reads", {}).get("evidence", {})
        own = evidence.get("own_read_bytes")
        foreign = evidence.get("foreign_read_bytes_min")
        row["foreign_reads_above_five_percent"] = (
            own is not None and foreign is not None and foreign > 0.05 * own)
    append_row(stage, row)
    if not ok:
        raise StageStopError(f"the run {arm} k={k} failed (return code {code}); see {out}. A crash "
                        "is not a void: the block is not replaced and the stage stops")
    if stage.foreign_read_fraction and sum(
        bool(r.get("foreign_reads_above_five_percent")) for r in read_rows(stage)
    ) >= 2:
        raise StageStopError("noisy host: multiple runs exceeded 5% foreign reads; discuss "
                             "the host with the user before continuing")
    return row


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------
def completed_blocks(rows: Sequence[Dict[str, Any]], comparison: str, k: int) -> List[List[Any]]:
    """The blocks of a pair with all four runs ok, as ``arm_validity.Run`` lists, in order."""
    by_block: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        if row["comparison"] == comparison and row["k"] == k and not row["capture"]:
            by_block.setdefault(row["block"], []).append(row)
    blocks = []
    for number in sorted(by_block):
        runs = sorted(by_block[number], key=lambda r: r["position"])
        if len(runs) == len(arm_validity.BLOCK_ORDER) and all(r["process_ok"] for r in runs):
            blocks.append([arm_validity.Run(r["value"], r["outcome"]) for r in runs])
    return blocks


def _next_block(rows: Sequence[Dict[str, Any]], comparison: str, k: int) -> int:
    seen = [r["block"] for r in rows if r["comparison"] == comparison and r["k"] == k]
    return max(seen) + 1 if seen else 0


def _last_block_seconds(rows: Sequence[Dict[str, Any]], comparison: str, k: int) -> float:
    last: Dict[int, float] = {}
    for row in rows:
        if row["comparison"] == comparison and row["k"] == k and not row["capture"]:
            last[row["block"]] = last.get(row["block"], 0.0) + row["wall_seconds"]
    return last[max(last)] if last else 0.0


def run_blocks(stage: Stage, budget: Dict[str, float], comparison: str, k: int) -> int:
    """ABBA blocks of one comparison at one k until ``blocks_to_add`` is zero. Returns the
    number of blocks run in this call."""
    if comparison == "AB" or stage.two_read:
        ok, reason = accuracy_status(stage, comparison)
        if not ok:
            raise StageStopError(f"accuracy: {reason}")
    arm_a, arm_b = COMPARISONS[comparison]
    ran = 0
    while arm_validity.blocks_to_add(completed_blocks(read_rows(stage), comparison, k)) > 0:
        rows = read_rows(stage)
        check_ceiling(stage, budget, _last_block_seconds(rows, comparison, k))
        block = _next_block(rows, comparison, k)
        for position, side in enumerate(arm_validity.block_order(1)[0]):
            watched_run(stage, comparison=comparison, k=k, block=block, position=position,
                        side=side, arm=arm_a if side == "A" else arm_b)
        ran += 1
    return ran


def run_single(stage: Stage, budget: Dict[str, float], k: int) -> None:
    """Each arm once at ``k``, no ABBA. The rows are tagged ``single`` and give no verdict."""
    rows = read_rows(stage)
    if any(r["comparison"] == "single" and r["k"] == k for r in rows):
        raise StageStopError(f"single runs at k={k} already exist in {stage.rows_path}")
    arms = [a for name in LEGACY_COMPARISONS for a in COMPARISONS[name]]
    for position, arm in enumerate(arms):
        check_ceiling(stage, budget, 0.0)
        watched_run(stage, comparison="single", k=k, block=0, position=position,
                    side=arm, arm=arm)


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------
def capture_arms(stage: Stage) -> Tuple[str, ...]:
    """The arms to capture. On CPU ``ga_floor`` is left out: the CPU has no second SDPA backend,
    and the fp32 tolerance of the judge (1e-4) needs no noise floor."""
    arms: Tuple[str, ...] = CAPTURE_ARMS
    if stage.two_read:
        arms = tuple(arm for arm in arms if arm != "ga_complete") + ("l2l_two_read",)
    return tuple(a for a in arms if not (stage.device == "cpu" and a == "ga_floor"))


def run_capture(stage: Stage, budget: Dict[str, float], k: int) -> Dict[str, Any]:
    """The five accuracy captures at ``k``, the reads check and the judge."""
    rows: Dict[str, Dict[str, Any]] = {}
    for position, arm in enumerate(capture_arms(stage)):
        check_ceiling(stage, budget, 0.0)
        rows[arm] = watched_run(stage, comparison="capture", k=k, block=0, position=position,
                                side=arm, arm=arm, capture=True, folder="capture")
    reads = rows["l2l_complete"]["reads_per_hit"]
    if reads != L2L_COMPLETE_READS_PER_HIT:
        raise StageStopError(
            f"l2l_complete read {reads} experts per hit at k={k}, the rule expects "
            f"{L2L_COMPLETE_READS_PER_HIT}: the stage stops")
    if stage.two_read and rows["l2l_two_read"]["reads_per_hit"] != 2.0:
        raise StageStopError("l2l_two_read must read two experts per hit: the stage stops")
    judge_command = [sys.executable, str(Path(__file__).resolve().parent / "oc_arm_run.py"),
                     "judge", "--runs", str(stage.root / "capture"), "--k", str(k)]
    if stage.two_read:
        judge_command.append("--two-read")
    # The captures sit in <root>/capture/capture-k<k>/b00p<i>-<arm>; judge wants <arm>-k<k>.
    link_captures(stage, k)
    stage.runner(judge_command)
    path = stage.root / "capture" / f"accuracy-k{k}.json"
    return json.loads(path.read_text("utf-8")) if path.is_file() else {}


def link_captures(stage: Stage, k: int) -> None:
    """Give ``judge`` the ``<arm>-k<k>`` folders it reads, as links to the run folders."""
    base = stage.root / "capture"
    for position, arm in enumerate(capture_arms(stage)):
        source = base / f"capture-k{k}" / f"b00p{position}-{arm}"
        target = base / f"{arm}-k{k}"
        if source.is_dir() and not target.exists():
            target.symlink_to(source.resolve(), target_is_directory=True)


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------
def accuracy_status(stage: Stage, comparison: str) -> Tuple[bool, str]:
    """Whether the accuracy pairs of a comparison passed at every capture k, and why not."""
    found = []
    for k in stage.capture_ks:
        path = stage.root / "capture" / f"accuracy-k{k}.json"
        if not path.is_file():
            return False, f"no accuracy capture at k={k}"
        found.append((k, json.loads(path.read_text("utf-8"))))
    for k, data in found:
        for pair in ACCURACY_PAIRS[comparison]:
            result = data["pairs"].get(pair)
            if not result or not result.get("pass"):
                return False, f"accuracy pair {pair} did not pass at k={k}"
    return True, ""


def original_threshold_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reclassify copies using recorded disk bounds and ADR 0002; never rewrite evidence."""
    classified = []
    for row in rows:
        path = Path(row["dir"]) / "watch.json"
        watch = json.loads(path.read_text("utf-8")) if path.is_file() else {}
        checks = watch.get("checks", {})
        outcomes = {name: {"outcome": checks.get(name, {}).get("outcome", arm_validity.UNKNOWN)}
                    for name in REQUIRED}
        foreign = checks.get("foreign_reads", {})
        evidence = foreign.get("evidence", {})
        least = evidence.get("foreign_read_bytes_min")
        most = evidence.get("foreign_read_bytes_max")
        outcome = arm_validity.UNKNOWN
        if least is not None and most is not None:
            if least >= arm_validity.FOREIGN_READ_LIMIT_BYTES:
                outcome = arm_validity.VOID
            elif most < arm_validity.FOREIGN_READ_LIMIT_BYTES:
                outcome = arm_validity.OK
        outcomes["foreign_reads"] = {"outcome": outcome}
        classified.append({**row, "outcome": arm_validity.arm_outcome(outcomes, REQUIRED)})
    return classified


def verdict_for(
    stage: Stage, comparison: str, k: int, *, original_threshold: bool = False,
) -> Dict[str, Any]:
    """The verdict of one comparison at one k (protocol 3.2). NO VERDICT carries its reason."""
    rows = read_rows(stage)
    if original_threshold:
        rows = original_threshold_rows(rows)
    blocks = completed_blocks(rows, comparison, k)
    valid = arm_validity.valid_blocks(blocks) if blocks else []
    out: Dict[str, Any] = {"comparison": comparison, "k": k, "blocks": len(valid),
                           "blocks_run": len(blocks), "outcome": None, "reason": None}
    ok, why = accuracy_status(stage, comparison)
    if not ok:
        return {**out, "outcome": "NO VERDICT", "reason": why}
    if not valid or arm_validity.blocks_to_add(blocks) > 0:
        return {**out, "outcome": "NO VERDICT", "reason": "the blocks of the rule are not done"}
    unknown = sorted({r.outcome for b in valid for r in b} - {arm_validity.OK})
    if unknown:
        return {**out, "outcome": "NO VERDICT",
                "reason": f"a required check was {', '.join(unknown)} in a counted run"}
    mean_a = [statistics.fmean([b[0].value, b[3].value]) for b in valid]
    mean_b = [statistics.fmean([b[1].value, b[2].value]) for b in valid]
    diffs = [a - b for a, b in zip(mean_a, mean_b)]
    statistic, p_value = t_test_p(diffs)
    effect = statistics.fmean(diffs) / statistics.fmean(mean_a)
    if p_value >= ALPHA:
        outcome = "NO DIFFERENCE"
    elif effect >= EFFECT_THRESHOLD:
        outcome = "WIN"
    elif effect > 0:
        outcome = "SMALL"
    else:
        outcome = "LOSS"
    return {**out, "outcome": outcome, "t": statistic, "p": p_value, "reduction": effect,
            "mean_a_seconds": statistics.fmean(mean_a), "mean_b_seconds": statistics.fmean(mean_b)}


def write_verdicts(stage: Stage, ks: Sequence[int]) -> Dict[str, Any]:
    pairs = TWO_READ_SCHEDULE if stage.two_read else tuple(
        (cmp_, k) for k in ks for cmp_ in LEGACY_COMPARISONS)
    results = []
    for comparison, k in pairs:
        result = verdict_for(stage, comparison, k)
        if stage.foreign_read_fraction:
            original = verdict_for(stage, comparison, k, original_threshold=True)
            result["original_threshold"] = original
            result["sensitive_to_foreign_reads"] = result["outcome"] != original["outcome"]
        results.append(result)
    path = stage.root / "verdicts.json"
    path.write_text(json.dumps(results, indent=2, allow_nan=False, default=str) + "\n",
                    encoding="utf-8")
    for result in results:
        console.print({key: result[key] for key in ("comparison", "k", "outcome", "reason")
                       if key in result})
    return {"verdicts": results, "path": str(path)}


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {name: sub.add_parser(name) for name in
                ("capture", "blocks", "single", "verdict", "stage1")}
    for cmd in commands.values():
        cmd.add_argument("--root", required=True)
        cmd.add_argument("--source", required=True)
        cmd.add_argument("--cache", required=True)
        cmd.add_argument("--model-path", help="the path foreign_reads watches (default: --cache)")
        cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        cmd.add_argument("--label", default="oc-stage1")
        cmd.add_argument("--revision")
        cmd.add_argument("--two-read", action="store_true",
                         help="ticket-08 captures and series (C2 k=2, then AB k=1,2)")
        cmd.add_argument("--foreign-read-fraction", type=float,
                         help="Linux fraction (ticket-08 default: 0.05; legacy default: 0)")
        cmd.add_argument("--ram-limit-gib", type=float)
        cmd.add_argument("--rate-usd-per-hour", type=float, default=0.0)
        cmd.add_argument("--ceiling-hours", type=float, default=10.0)
        cmd.add_argument("--ceiling-usd", type=float, default=5.0)
        cmd.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                         help="everything after --extra goes to oc_arm_run.py run")
    for name in ("capture", "blocks", "single"):
        commands[name].add_argument("--k", type=int, required=(name != "capture"))
    commands["capture"].add_argument("--ks", type=int, nargs="+", default=list(CAPTURE_KS))
    commands["blocks"].add_argument("--comparison", choices=sorted(COMPARISONS), required=True)
    commands["verdict"].add_argument("--ks", type=int, nargs="+", required=True)
    commands["stage1"].add_argument("--full-k", type=int, nargs="+", default=[1, 2, 4])
    commands["stage1"].add_argument("--single-k", type=int, nargs="+", default=[8])
    for name in ("capture", "stage1", "verdict"):
        commands[name].add_argument("--capture-ks", type=int, nargs="+",
                                    default=list(CAPTURE_KS),
                                    help="accuracy capture k values (ticket 08 series: 2)")
    return parser


def stage_from(args: argparse.Namespace) -> Stage:
    return Stage(root=Path(args.root), source=Path(args.source), cache=Path(args.cache),
                 model_path=Path(args.model_path or args.cache), label=args.label,
                 device=args.device, revision=args.revision, extra_args=list(args.extra),
                 ram_limit_gib=args.ram_limit_gib, two_read=args.two_read,
                 foreign_read_fraction=(args.foreign_read_fraction
                                        if args.foreign_read_fraction is not None
                                        else (0.05 if args.two_read else 0.0)),
                 capture_ks=tuple(getattr(args, "capture_ks", None) or CAPTURE_KS))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stage = stage_from(args)
    budget = ensure_budget(stage, rate=args.rate_usd_per_hour, hours=args.ceiling_hours,
                           usd=args.ceiling_usd)
    try:
        if args.command == "capture":
            for k in ([args.k] if args.k else args.ks):
                run_capture(stage, budget, k)
        elif args.command == "blocks":
            run_blocks(stage, budget, args.comparison, args.k)
        elif args.command == "single":
            run_single(stage, budget, args.k)
        elif args.command == "verdict":
            write_verdicts(stage, args.ks)
        else:
            for k in stage.capture_ks:
                run_capture(stage, budget, k)
            if stage.two_read:
                for comparison, k in TWO_READ_SCHEDULE:
                    run_blocks(stage, budget, comparison, k)
            else:
                for k in args.full_k:
                    for comparison in LEGACY_COMPARISONS:
                        run_blocks(stage, budget, comparison, k)
                for k in args.single_k:
                    run_single(stage, budget, k)
            write_verdicts(stage, args.full_k)
    except StageStopError as stop:
        console.print(f"[red]STAGE STOP:[/] {stop}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
