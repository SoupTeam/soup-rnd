"""The stage 1 driver ``oc_stage1_drive.py``: blocks, voids, ceiling, capture gate and verdicts.

The arm process and ``ArmWatch`` are replaced by fakes, so no model, GPU or Linux sensor is
needed. What is checked is the driver's own logic against the rules of the plan: ABBA order,
the pilot of five blocks, void blocks replaced and kept, a crash that stops the stage, the
ceiling, the reads check of ``l2l_complete`` and the five verdict outcomes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from benchmarks.harness import arm_validity
from benchmarks.harness import oc_stage1_drive as drive


class FakeWatch:
    def __init__(self, outcome: str) -> None:
        self.record = {"outcome": outcome, "checks": {}}

    def __enter__(self) -> "FakeWatch":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def parse(command: List[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class Fakes:
    """A runner that writes ``record.json`` and a watch factory with scripted outcomes."""

    def __init__(
        self,
        value: Callable[[str, int], float] = lambda arm, block: 10.0,
        outcome: Callable[[int, int], str] = lambda block, position: "ok",
        reads: float = 3.0,
        fail: Optional[Callable[[str, int], bool]] = None,
    ) -> None:
        self.value, self.outcome, self.reads, self.fail = value, outcome, reads, fail
        self.commands: List[List[str]] = []
        self.watch_calls: List[Dict[str, Any]] = []
        self.block_of: Dict[str, int] = {}

    def watch(self, **kwargs: Any) -> FakeWatch:
        self.watch_calls.append(kwargs)
        block = (kwargs["run"] - 1) // 4
        position = (kwargs["run"] - 1) % 4
        return FakeWatch(self.outcome(block, position))

    def runner(self, command: List[str]) -> int:
        self.commands.append(command)
        if "judge" in command:
            return self.judge(command)
        arm, out = parse(command, "--arm"), Path(parse(command, "--out"))
        block = int(out.name[1:3])
        if self.fail and self.fail(arm, block):
            return 1
        out.mkdir(parents=True)
        (out / "record.json").write_text(json.dumps({
            "median_step_seconds": self.value(arm, block), "setup_seconds": 5.0,
            "run_seconds": 60.0, "reads_per_hit": (self.reads if arm == "l2l_complete" else 3.0),
            "rss_bytes": {"after_last": 123}, "expert_fetches_per_window": [9, 9],
        }), encoding="utf-8")
        return 0

    def judge(self, command: List[str]) -> int:
        return 0


@pytest.fixture()
def clock() -> Dict[str, float]:
    return {"now": 1000.0}


def make_stage(tmp_path: Path, fakes: Fakes, clock: Dict[str, float]) -> drive.Stage:
    def tick() -> float:
        clock["now"] += 1.0
        return clock["now"]

    return drive.Stage(
        root=tmp_path / "root", source=tmp_path / "src", cache=tmp_path / "cache",
        model_path=tmp_path / "cache", label="test", device="cpu", revision="rev",
        runner=fakes.runner, watch_factory=fakes.watch, clock=tick,
    )


def budget(stage: drive.Stage, **kw: float) -> Dict[str, float]:
    values = dict(rate=0.2, hours=10.0, usd=5.0)
    values.update(kw)
    return drive.ensure_budget(stage, **values)


def write_accuracy(stage: drive.Stage, *, passing: bool = True) -> None:
    folder = stage.root / "capture"
    folder.mkdir(parents=True, exist_ok=True)
    pairs = {name: {"pass": passing, "bit_equal": True} for name in (
        "ga_complete_vs_ga_ordinary", "l2l_complete_vs_l2l_ordinary",
        "l2l_ordinary_vs_ga_ordinary")}
    for k in drive.CAPTURE_KS:
        (folder / f"accuracy-k{k}.json").write_text(json.dumps({"pairs": pairs}), "utf-8")


# ---------------------------------------------------------------- t-test
def test_t_test_matches_a_known_value() -> None:
    t, p = drive.t_test_p([1.0, 2.0, 3.0, 4.0, 5.0])
    assert t == pytest.approx(4.2426, abs=1e-3)
    assert p == pytest.approx(0.01324, abs=2e-4)


def test_t_test_agrees_with_scipy_when_it_is_installed() -> None:
    stats = pytest.importorskip("scipy.stats")
    diffs = [0.4, 1.9, -0.3, 2.2, 1.1, 0.8, 1.5]
    t, p = drive.t_test_p(diffs)
    want = stats.ttest_1samp(diffs, 0.0)
    assert t == pytest.approx(want.statistic, rel=1e-9)
    assert p == pytest.approx(want.pvalue, rel=1e-6)


def test_t_test_of_a_constant_difference() -> None:
    assert drive.t_test_p([0.0, 0.0, 0.0]) == (0.0, 1.0)
    t, p = drive.t_test_p([2.0, 2.0])
    assert math.isinf(t) and p == 0.0


def test_t_test_needs_two_blocks() -> None:
    with pytest.raises(ValueError):
        drive.t_test_p([1.0])


# ---------------------------------------------------------------- blocks
def test_a_pilot_runs_five_blocks_in_abba_order(tmp_path: Path, clock: Dict[str, float]) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    ran = drive.run_blocks(stage, budget(stage), "C1", 2)
    rows = drive.read_rows(stage)
    assert ran == arm_validity.PILOT_BLOCKS and len(rows) == 4 * ran
    first = [r for r in rows if r["block"] == 0]
    assert [r["arm"] for r in first] == ["ga_ordinary", "ga_complete", "ga_complete", "ga_ordinary"]
    assert [r["side"] for r in first] == ["A", "B", "B", "A"]
    assert [r["position"] for r in first] == [0, 1, 2, 3]
    assert all(r["k"] == 2 and r["comparison"] == "C1" for r in rows)


def test_c2_uses_the_l2l_arms(tmp_path: Path, clock: Dict[str, float]) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    drive.run_blocks(stage, budget(stage), "C2", 1)
    arms = {r["arm"] for r in drive.read_rows(stage)}
    assert arms == {"l2l_ordinary", "l2l_complete"}


def test_every_run_is_inside_a_watch_with_the_required_checks(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    drive.run_blocks(stage, budget(stage), "C1", 2)
    assert len(fakes.watch_calls) == 20
    assert all(call["required"] == ("suspend", "power", "foreign_reads")
               for call in fakes.watch_calls)
    assert all(call["model_path"] == str(tmp_path / "cache") for call in fakes.watch_calls)
    assert [c["run"] for c in fakes.watch_calls] == list(range(1, 21))
    assert [c["round"] for c in fakes.watch_calls[:4]] == [1, 1, 2, 2]


def test_a_second_call_adds_no_block(tmp_path: Path, clock: Dict[str, float]) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    drive.run_blocks(stage, budget(stage), "C1", 2)
    assert drive.run_blocks(stage, budget(stage), "C1", 2) == 0
    assert len(drive.read_rows(stage)) == 20


def test_a_void_block_is_replaced_and_its_rows_are_kept(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    fakes = Fakes(outcome=lambda block, position: "void" if (block, position) == (1, 2) else "ok")
    stage = make_stage(tmp_path, fakes, clock)
    ran = drive.run_blocks(stage, budget(stage), "C1", 2)
    rows = drive.read_rows(stage)
    assert ran == arm_validity.PILOT_BLOCKS + 1
    assert len(rows) == 4 * ran
    assert [r["outcome"] for r in rows].count("void") == 1  # the void row is on file


def test_a_crashed_run_stops_the_stage_and_is_not_replaced(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    fakes = Fakes(fail=lambda arm, block: block == 1 and arm == "ga_complete")
    stage = make_stage(tmp_path, fakes, clock)
    with pytest.raises(drive.StageStopError, match="not a void"):
        drive.run_blocks(stage, budget(stage), "C1", 2)
    rows = drive.read_rows(stage)
    assert rows[-1]["process_ok"] is False and rows[-1]["value"] is None
    assert len(drive.completed_blocks(rows, "C1", 2)) == 1  # block 0 only


def test_the_next_call_after_a_crash_starts_a_new_block_number(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    flag = {"fail": True}
    fakes = Fakes(fail=lambda arm, block: flag["fail"] and block == 1 and arm == "ga_complete")
    stage = make_stage(tmp_path, fakes, clock)
    with pytest.raises(drive.StageStopError):
        drive.run_blocks(stage, budget(stage), "C1", 2)
    flag["fail"] = False
    drive.run_blocks(stage, budget(stage), "C1", 2)
    numbers = {r["block"] for r in drive.read_rows(stage)}
    assert 1 in numbers and max(numbers) == arm_validity.PILOT_BLOCKS  # block 1 abandoned


def test_the_ceiling_refuses_a_block_that_would_end_too_late(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    state = budget(stage, hours=0.5)  # 1800 s
    clock["now"] = state["started_unix"] + 1790.0
    # No history: the first block starts. The second would end past the ceiling.
    with pytest.raises(drive.StageStopError, match="ceiling"):
        drive.run_blocks(stage, state, "C1", 2)
    assert len(drive.completed_blocks(drive.read_rows(stage), "C1", 2)) == 1


def test_the_dollar_ceiling_counts_the_rate(tmp_path: Path, clock: Dict[str, float]) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    state = budget(stage, rate=1.0, usd=0.5, hours=100.0)  # 0.5 USD at 1 USD/h is 1800 s
    clock["now"] = state["started_unix"] + 1790.0
    with pytest.raises(drive.StageStopError, match="USD"):
        drive.run_blocks(stage, state, "C1", 2)
    assert len(drive.completed_blocks(drive.read_rows(stage), "C1", 2)) == 1


def test_stage_json_wins_over_later_flags(tmp_path: Path, clock: Dict[str, float]) -> None:
    stage = make_stage(tmp_path, Fakes(), clock)
    first = drive.ensure_budget(stage, rate=0.2, hours=10.0, usd=5.0)
    again = drive.ensure_budget(stage, rate=9.0, hours=99.0, usd=99.0)
    assert again == first


# ---------------------------------------------------------------- single, capture, command
def test_single_runs_each_arm_once_and_refuses_a_repeat(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    drive.run_single(stage, budget(stage), 8)
    rows = drive.read_rows(stage)
    assert [r["arm"] for r in rows] == ["ga_ordinary", "ga_complete", "l2l_ordinary",
                                        "l2l_complete"]
    assert all(r["comparison"] == "single" and r["k"] == 8 for r in rows)
    with pytest.raises(drive.StageStopError, match="already exist"):
        drive.run_single(stage, budget(stage), 8)


def test_capture_runs_five_arms_with_capture_flags_and_links_them(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    drive.run_capture(stage, budget(stage), 2)
    runs = [c for c in fakes.commands if "run" in c]
    assert [parse(c, "--arm") for c in runs] == list(drive.capture_arms(stage))
    assert all("--capture" in c and "--hits-probe" in c for c in runs)
    for arm in drive.capture_arms(stage):
        assert (stage.root / "capture" / f"{arm}-k2" / "record.json").is_file()
    assert any("judge" in c for c in fakes.commands)


def test_capture_on_a_card_includes_the_noise_floor_arm_and_cpu_leaves_it_out(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    stage = make_stage(tmp_path, Fakes(), clock)
    assert "ga_floor" not in drive.capture_arms(stage)  # the fixture stage is CPU
    stage.device = "cuda"
    assert drive.capture_arms(stage) == drive.CAPTURE_ARMS


def test_capture_stops_when_l2l_complete_does_not_read_three_per_hit(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    fakes = Fakes(reads=2.0)
    stage = make_stage(tmp_path, fakes, clock)
    with pytest.raises(drive.StageStopError, match="2.0 experts per hit"):
        drive.run_capture(stage, budget(stage), 2)
    assert not any("judge" in c for c in fakes.commands)


def test_the_run_command_carries_the_stage_settings(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    stage = make_stage(tmp_path, Fakes(), clock)
    stage.extra_args = ["--updates", "4"]
    command = drive.run_command(stage, "ga_complete", 4, tmp_path / "out", capture=False)
    assert parse(command, "--arm") == "ga_complete" and parse(command, "--k") == "4"
    assert parse(command, "--revision") == "rev"
    assert command[-2:] == ["--updates", "4"]
    assert "--capture" not in command


def test_a_ram_limit_wraps_the_command_in_a_cgroup_scope(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    stage = make_stage(tmp_path, Fakes(), clock)
    stage.ram_limit_gib = 29.756
    command = drive.run_command(stage, "ga_ordinary", 1, tmp_path / "out", capture=False)
    assert command[:7] == ["systemd-run", "--scope", "-q", "-p", "MemoryMax=29.756G", "-p",
                           "MemorySwapMax=0"]


# ---------------------------------------------------------------- verdict
def noisy(a: float, b: float) -> Callable[[str, int], float]:
    """A runs near ``a``, B runs near ``b``, with a block-dependent wobble."""
    wobble = [0.00, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01, 0.015]

    def value(arm: str, block: int) -> float:
        base = a if arm.endswith("ordinary") else b
        return base * (1 + wobble[block % len(wobble)])

    return value


def run_c1(tmp_path: Path, clock: Dict[str, float], fakes: Fakes, *, accuracy: bool = True,
           ) -> drive.Stage:
    stage = make_stage(tmp_path, fakes, clock)
    if accuracy:
        write_accuracy(stage)
    drive.run_blocks(stage, budget(stage), "C1", 2)
    return stage


def test_verdict_win_when_complete_is_twenty_percent_faster(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    stage = run_c1(tmp_path, clock, Fakes(value=noisy(10.0, 8.0)))
    result = drive.verdict_for(stage, "C1", 2)
    assert result["outcome"] == "WIN"
    assert result["reduction"] == pytest.approx(0.20, abs=0.02)


def test_verdict_small_for_a_five_percent_gain(tmp_path: Path, clock: Dict[str, float]) -> None:
    stage = run_c1(tmp_path, clock, Fakes(value=noisy(10.0, 9.5)))
    result = drive.verdict_for(stage, "C1", 2)
    assert result["outcome"] == "SMALL" and 0 < result["reduction"] < 0.10


def test_verdict_loss_when_complete_is_slower(tmp_path: Path, clock: Dict[str, float]) -> None:
    stage = run_c1(tmp_path, clock, Fakes(value=noisy(10.0, 12.0)))
    assert drive.verdict_for(stage, "C1", 2)["outcome"] == "LOSS"


def test_verdict_no_difference_when_the_arms_are_the_same(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    # Same mean, run-to-run noise that does not follow the arm.
    noise = [0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.0, 0.15]

    def value(arm: str, block: int) -> float:
        return 10.0 + noise[(block * 3 + (arm == "ga_complete")) % 8]

    stage = run_c1(tmp_path, clock, Fakes(value=value))
    assert drive.verdict_for(stage, "C1", 2)["outcome"] == "NO DIFFERENCE"


def test_verdict_no_verdict_when_a_required_check_was_unknown(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    fakes = Fakes(
        value=noisy(10.0, 8.0),
        outcome=lambda block, position: "unknown" if (block, position) == (2, 1) else "ok")
    stage = run_c1(tmp_path, clock, fakes)
    result = drive.verdict_for(stage, "C1", 2)
    assert result["outcome"] == "NO VERDICT" and "unknown" in result["reason"]


def test_verdict_no_verdict_without_the_accuracy_capture(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    stage = run_c1(tmp_path, clock, Fakes(value=noisy(10.0, 8.0)), accuracy=False)
    result = drive.verdict_for(stage, "C1", 2)
    assert result["outcome"] == "NO VERDICT" and "accuracy" in result["reason"]


def test_verdict_no_verdict_when_an_accuracy_pair_failed(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    stage = run_c1(tmp_path, clock, Fakes(value=noisy(10.0, 8.0)), accuracy=False)
    write_accuracy(stage, passing=False)
    result = drive.verdict_for(stage, "C1", 2)
    assert result["outcome"] == "NO VERDICT" and "did not pass" in result["reason"]


def test_verdict_no_verdict_before_the_blocks_are_done(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    stage = make_stage(tmp_path, Fakes(value=noisy(10.0, 8.0)), clock)
    write_accuracy(stage)
    assert drive.verdict_for(stage, "C1", 2)["outcome"] == "NO VERDICT"


def test_write_verdicts_lists_every_comparison_and_k(
    tmp_path: Path, clock: Dict[str, float]
) -> None:
    stage = run_c1(tmp_path, clock, Fakes(value=noisy(10.0, 8.0)))
    out = drive.write_verdicts(stage, [2])
    written = json.loads((stage.root / "verdicts.json").read_text("utf-8"))
    assert [(r["comparison"], r["k"]) for r in written] == [("C1", 2), ("C2", 2)]
    assert written[0]["outcome"] == "WIN" and written[1]["outcome"] == "NO VERDICT"
    assert out["path"].endswith("verdicts.json")


def test_two_read_blocks_require_accuracy_and_keep_abba_order(
    tmp_path: Path, clock: Dict[str, float],
) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    with pytest.raises(drive.StageStopError, match="accuracy"):
        drive.run_blocks(stage, budget(stage), "AB", 1)
    folder = stage.root / "capture"
    folder.mkdir(exist_ok=True)
    for k in drive.CAPTURE_KS:
        (folder / f"accuracy-k{k}.json").write_text(json.dumps({"pairs": {
            "l2l_two_read_vs_l2l_complete": {"pass": True},
            "l2l_two_read_vs_l2l_ordinary": {"pass": True},
        }}), encoding="utf-8")
    drive.run_blocks(stage, budget(stage), "AB", 1)
    arms = [parse(cmd, "--arm") for cmd in fakes.commands]
    assert arms == ["l2l_complete", "l2l_two_read", "l2l_two_read", "l2l_complete"] * 5


def test_two_read_capture_requires_two_reads_and_passes_flag_to_judge(
    tmp_path: Path, clock: Dict[str, float],
) -> None:
    fakes = Fakes()
    original_runner = fakes.runner

    def runner(command: List[str]) -> int:
        code = original_runner(command)
        if "--arm" in command and parse(command, "--arm") == "l2l_two_read":
            path = Path(parse(command, "--out")) / "record.json"
            record = json.loads(path.read_text("utf-8"))
            record["reads_per_hit"] = 2.0
            path.write_text(json.dumps(record), encoding="utf-8")
        return code

    stage = make_stage(tmp_path, fakes, clock)
    stage.two_read = True
    stage.runner = runner
    drive.run_capture(stage, budget(stage), 2)
    arms = [parse(cmd, "--arm") for cmd in fakes.commands if "--arm" in cmd]
    assert "l2l_two_read" in arms
    assert "ga_complete" not in arms
    assert "--two-read" in fakes.commands[-1]


def test_relative_threshold_report_keeps_both_verdicts_without_changing_rows(
    tmp_path: Path, clock: Dict[str, float],
) -> None:
    stage = run_c1(tmp_path, clock, Fakes(value=noisy(10.0, 8.0)))
    stage.foreign_read_fraction = 0.05
    original_rows = stage.rows_path.read_bytes()
    for row in drive.read_rows(stage):
        watch = {"checks": {
            "suspend": {"outcome": "ok"}, "power": {"outcome": "ok"},
            "foreign_reads": {"outcome": "ok", "evidence": {
                "foreign_read_bytes_min": 2_000_000_000,
                "foreign_read_bytes_max": 2_000_000_000,
            }},
        }}
        (Path(row["dir"]) / "watch.json").write_text(json.dumps(watch), encoding="utf-8")
    result = drive.write_verdicts(stage, [2])["verdicts"][0]
    assert result["outcome"] == "WIN"
    assert result["original_threshold"]["outcome"] == "NO VERDICT"
    assert result["sensitive_to_foreign_reads"] is True
    assert stage.rows_path.read_bytes() == original_rows


def test_two_read_series_runs_only_the_new_pairs_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(drive, "run_capture", lambda _s, _b, k: calls.append(("capture", k)))
    monkeypatch.setattr(drive, "accuracy_status", lambda _s, _c: (True, ""))
    monkeypatch.setattr(drive, "run_blocks", lambda _s, _b, c, k: calls.append((c, k)))
    monkeypatch.setattr(drive, "write_verdicts", lambda _s, _ks: {})
    result = drive.main([
        "stage1", "--root", str(tmp_path), "--source", "src", "--cache", "cache", "--two-read",
    ])
    assert result == 0
    assert calls == [("capture", 2), ("capture", 8), ("C2", 2), ("AB", 1), ("AB", 2)]


def test_repeated_foreign_reads_above_five_percent_stop_the_series(
    tmp_path: Path, clock: Dict[str, float],
) -> None:
    fakes = Fakes()
    stage = make_stage(tmp_path, fakes, clock)
    stage.foreign_read_fraction = 0.05

    def watch(**_kwargs: Any) -> FakeWatch:
        instance = FakeWatch("void")
        instance.record["checks"] = {"foreign_reads": {"evidence": {
            "own_read_bytes": 100_000_000_000, "foreign_read_bytes_min": 6_000_000_000,
        }}}
        return instance

    stage.watch_factory = watch
    drive.watched_run(stage, comparison="AB", k=1, block=0, position=0,
                      side="A", arm="l2l_complete")
    with pytest.raises(drive.StageStopError, match="noisy host"):
        drive.watched_run(stage, comparison="AB", k=1, block=0, position=1,
                          side="B", arm="l2l_two_read")
    assert len(drive.read_rows(stage)) == 2


def test_two_read_profile_selects_the_approved_relative_threshold(tmp_path: Path) -> None:
    parser = drive.build_parser()
    common = ["verdict", "--root", str(tmp_path), "--source", "src", "--cache", "cache",
              "--ks", "1", "2"]
    assert drive.stage_from(parser.parse_args(common)).foreign_read_fraction == 0.0
    profile = drive.stage_from(parser.parse_args(common + ["--two-read"]))
    assert profile.foreign_read_fraction == 0.05
