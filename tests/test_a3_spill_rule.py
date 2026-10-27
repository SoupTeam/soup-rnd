"""CPU tests for the A3 stage-2 rule (benchmarks/harness/a3_spill_rule.py).

Every test writes a synthetic session directory in the shape ``a3_spill_stand.py session``
leaves, with the numbers a passing GPU run would have, then breaks one thing and checks the
criterion it belongs to. The rule reads JSON only, so nothing here needs torch or a GPU.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tests.test_l2l_probe_harness import _load

GENEROUS = 24 << 30
TIGHT = 9_000 << 20
SPILL_DIR = "/mnt/nvme/a3-spill"
KILLED_FILE = f"{SPILL_DIR}/kill9/soup-l2l-spill-4242-0badcafe.bin"
SIGINT_FILE = f"{SPILL_DIR}/sigint/soup-l2l-spill-4343-0badf00d.bin"


@pytest.fixture()
def rule():
    return _load("a3_spill_rule")


# --------------------------------------------------------------------------
# a passing synthetic session
# --------------------------------------------------------------------------
def _digest(seed: str, step: int) -> Dict[str, Any]:
    return {"sha256": f"{seed}-{step}", "dtype": "float32", "shape": [8, 4096],
            "finite": True, "nonzero": True}


def _cgroup(limit: int, *, oom: int = 0, peak: Optional[int] = None) -> Dict[str, Any]:
    return {"version": 2, "path": "/sys/fs/cgroup/system.slice/run-x.scope",
            "memory_max": limit, "swap_max": 0, "current": 7 << 30, "peak": peak,
            "inactive_file": 1 << 30, "events": {"oom_kill": oom, "oom": oom}}


def _plan(layers: int, spill_dir: str) -> Dict[str, Any]:
    chunk = 512 * 4096 * 2
    return {"spill_layers": layers, "n_layers": 32, "k": 4, "chunk_bytes": chunk,
            "slot_bytes": chunk, "ring": 4, "pin": True,
            "host_bytes": (32 - layers) * 4 * chunk + (8 * (8 << 20) if layers else 0),
            "all_resident_bytes": 32 * 4 * chunk, "budget_bytes": 1 << 30,
            "available_bytes": 2 << 30, "limiting_level": "/sys/fs/cgroup/run-x.scope",
            "limiting_numbers": {}, "spill_dir": spill_dir,
            "spill_path": f"{spill_dir}/soup-l2l-spill-1-00000000.bin" if layers else None,
            "spill_file_bytes": layers * 4 * chunk, "explicit_dir": True}


def _arm(name: str, limit: int, *, layers: Optional[int], sdpa: str = "MATH",
         grads: str = "same", steps: int = 3) -> Dict[str, Any]:
    spill_dir = f"{SPILL_DIR}/{name}"
    rec: Dict[str, Any] = {
        "arm": name, "device": "cuda", "dtype": "bfloat16", "expected_memory_max": limit,
        "cgroup_start": _cgroup(limit), "cgroup_at_first_step": _cgroup(limit),
        "cgroup_end": _cgroup(limit, peak=limit - (1 << 30)),
        "deterministic": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "algorithms": True,
                          "warn_only": False, "sdpa": sdpa},
        "exit": "normal", "steps": steps, "expected_steps": steps,
        "grads": [{"lora_A": _digest(grads, s), "lora_B": _digest(grads + "B", s)}
                  for s in range(steps)],
        "logs": [{"loss": 2.5 - 0.1 * s, "learning_rate": 0.001 * (steps - s)}
                 for s in range(steps)],
        "rates": [0.001 * (steps - s) for s in range(steps)],
        "rows": {"row_sha256": ["r0", "r1", "r2"]},
        "spill_files_after_train": [],
    }
    if layers is not None:
        rec["spill_plan"] = _plan(layers, spill_dir)
        rec["store_io"] = ({"bytes_written": 3 << 30, "bytes_read": 3 << 30} if layers
                           else {"bytes_written": 0, "bytes_read": 0})
        held = rec["spill_plan"]["host_bytes"]
        rec["host_allocator"] = {"baseline_bytes": 4 << 30,
                                 "at_last_step_bytes": (4 << 30) + held + (64 << 20)}
        rec["stale_removed"] = []
    return rec


def _entry(name: str, limit: Optional[int], **extra: Any) -> Dict[str, Any]:
    return {"name": name, "memory_max": limit, "returncode": 0,
            "spill_files_after_exit": [], **extra}


def _cost(block: int, position: int, role: str, value: float) -> Dict[str, Any]:
    limit = GENEROUS if role == "A" else TIGHT
    rec = _arm(f"cost/b{block:02d}_p{position}_{role}", limit,
               layers=16 if role == "B" else 0, steps=7)
    rec.update(role=role, block=block, position=position, value=value, outcome="ok")
    return rec


class Session:
    """A passing session in memory; ``write`` puts it on disk the way the stand does."""

    def __init__(self, overheads: Optional[List[float]] = None):
        self.arms = {
            "l2l_nospill_1": _arm("l2l_nospill_1", GENEROUS, layers=0),
            "l2l_nospill_2": _arm("l2l_nospill_2", GENEROUS, layers=0),
            "l2l_spill": _arm("l2l_spill", TIGHT, layers=16),
            "ga_1": _arm("ga_1", GENEROUS, layers=None, grads="ga"),
            "ga_2": _arm("ga_2", GENEROUS, layers=None, grads="ga"),
            "ga_floor": _arm("ga_floor", GENEROUS, layers=None, grads="floor",
                             sdpa="EFFICIENT_ATTENTION"),
        }
        self.entries: Dict[str, Dict[str, Any]] = {
            name: _entry(name, rec["expected_memory_max"]) for name, rec in self.arms.items()}
        self.compare = {"hashes_match": {name: True for name in self.arms},
                        "tolerance": {"status": "within_tolerance", "pass": True, "note": ""}}
        self.calibration = {"device": "cuda", "aliasing": {"pass": True,
                                                           "status": "neutralized"}}
        sigint = _arm("sigint", TIGHT, layers=16)
        sigint.update(exit="interrupted", at_interrupt={"in_step": True, "steps_done": 1},
                      steps=1)
        kill9_next = _arm("kill9_next", TIGHT, layers=16)
        kill9_next["stale_removed"] = [KILLED_FILE]
        self.cleanup = {"sigint": sigint, "kill9_next": kill9_next}
        marker = {"pid": 4242, "steps_done": 1, "in_step": True, "spill_path": KILLED_FILE,
                  "device": "cuda", "expected_memory_max": TIGHT, "cgroup_start": _cgroup(TIGHT)}
        self.entries["sigint"] = _entry("sigint", TIGHT, returncode=130, signal={
            "signal": "SIGINT", "sent_unix": 1.0, "exited": True, "returncode": 130,
            "files_before_signal": [SIGINT_FILE], "files_after_exit": [],
            "marker": {**marker, "pid": 4343, "spill_path": SIGINT_FILE}})
        self.entries["kill9"] = _entry("kill9", TIGHT, returncode=-9, signal={
            "signal": "SIGKILL", "sent_unix": 1.0, "exited": True, "returncode": -9,
            "files_before_signal": [KILLED_FILE], "files_after_exit": [KILLED_FILE],
            "marker": marker})
        self.entries["kill9_next"] = _entry("kill9_next", TIGHT)
        self.limits: Dict[str, Any] = {"generous": GENEROUS, "tight": TIGHT, "oom_margin": []}
        self.cost: List[Dict[str, Any]] = []
        for block, overhead in enumerate(overheads if overheads is not None
                                         else [0.03, 0.04, 0.02, 0.05, 0.03]):
            for position, role in enumerate(("A", "B", "B", "A")):
                self.add_cost(_cost(block, position, role, 1.0 + (overhead if role == "B"
                                                                   else 0.0)))

    def add_cost(self, rec: Dict[str, Any]) -> None:
        self.cost.append(rec)
        name = rec["arm"]
        self.entries[name] = _entry(name, rec["expected_memory_max"])

    def write(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        children = list(self.entries.values())
        (root / "session.json").write_text(json.dumps({"state": "finished",
                                                       "children": children,
                                                       "limits": self.limits}))
        for folder, records in (("arms", self.arms), ("cleanup", self.cleanup)):
            (root / folder).mkdir(exist_ok=True)
            for name, rec in records.items():
                (root / folder / f"{name}.json").write_text(json.dumps(rec))
        (root / "cost").mkdir(exist_ok=True)
        for rec in self.cost:
            (root / f"{rec['arm']}.json").write_text(json.dumps(rec))
        (root / "compare.json").write_text(json.dumps(self.compare))
        (root / "calibration.json").write_text(json.dumps(self.calibration))
        return root


def _judge(rule, session: Session, tmp_path: Path) -> Dict[str, Any]:
    return rule.evaluate(str(session.write(tmp_path / "session")))


def _part(result: Dict[str, Any], criterion: str, prefix: str) -> Dict[str, Any]:
    return next(p for p in result["criteria"][criterion]["parts"]
                if p["detail"].startswith(prefix))


def test_a_clean_gpu_session_passes_every_criterion(rule, tmp_path):
    result = _judge(rule, Session(), tmp_path)
    statuses = {c: v["status"] for c, v in result["criteria"].items()}
    assert statuses == {"1": "PASS", "1b": "PASS", "2": "PASS", "3": "PASS", "4": "PASS"}
    assert result["verdict"].startswith("SPILL CONFIRMED: ")
    assert result["spill_layers"] == 16 and result["spill_layers_in_target"] is True
    assert result["summary"]["cpu_session"] is False
    assert result["oom_margin"] is None  # no tight limit recorded in this synthetic session


def test_a_training_margin_warning_is_reported_and_changes_nothing(rule, tmp_path):
    session = Session()
    warning = f"training margin {-(151 << 20):,} B < {256 << 20:,} B (description only)"
    session.limits["oom_margin"] = [{"limit_bytes": TIGHT, "build_margin_bytes": 369 << 20,
                                     "train_margin_bytes": -(151 << 20),
                                     "warn_below_bytes": 256 << 20, "warnings": [warning],
                                     "stop": False, "transient": {"readings": 900}}]
    result = _judge(rule, session, tmp_path)
    assert result["oom_margin"] == {"limit_bytes": TIGHT, "build_margin_bytes": 369 << 20,
                                    "train_margin_bytes": -(151 << 20),
                                    "warn_below_bytes": 256 << 20, "warnings": [warning],
                                    "stop": False}
    assert result["verdict"].startswith("SPILL CONFIRMED: ")
    assert {c: v["status"] for c, v in result["criteria"].items()} == {
        "1": "PASS", "1b": "PASS", "2": "PASS", "3": "PASS", "4": "PASS"}


# --------------------------------------------------------------------------
# criterion 1
# --------------------------------------------------------------------------
def test_unequal_no_spill_runs_give_no_verdict_not_fail(rule, tmp_path):
    session = Session()
    session.arms["l2l_nospill_2"]["grads"][1]["lora_A"]["sha256"] = "drift"
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert "does not repeat" in result["criteria"]["1"]["detail"]
    assert result["criteria"]["4"]["status"] == "NO VERDICT"
    assert result["criteria"]["4"]["refused"] is True


@pytest.mark.parametrize("damage, word", [
    (lambda rec: rec["grads"][2]["lora_B"].update(sha256="other"), "grads"),
    (lambda rec: rec["grads"][0]["lora_A"].update(dtype="bfloat16"), "grads"),
    (lambda rec: rec["grads"][0]["lora_A"].update(shape=[4096, 8]), "grads"),
    (lambda rec: rec["logs"][1].update(loss=2.4000001), "losses"),
    (lambda rec: rec["rates"].__setitem__(2, 0.5), "rates"),
    (lambda rec: rec.update(steps=4, expected_steps=4,
                            grads=rec["grads"] + [rec["grads"][0]]), "steps"),
])
def test_spill_unequal_to_no_spill_fails_and_refuses_the_cost_blocks(rule, tmp_path, damage,
                                                                      word):
    session = Session()
    damage(session.arms["l2l_spill"])
    root = session.write(tmp_path / "session")
    result = rule.evaluate(str(root))
    assert result["criteria"]["1"]["status"] == "FAIL"
    assert word in result["criteria"]["1"]["detail"]
    assert result["criteria"]["4"]["status"] == "NO VERDICT"
    assert result["criteria"]["4"]["refused"] is True
    assert "No new rentals" in result["criteria"]["4"]["detail"]
    assert result["verdict"].startswith("SPILL CHANGES THE NUMBERS")
    allowed, why = rule.cost_gate(str(root))
    assert allowed is False and why.startswith("STOP") and "No new rentals" in why


def test_the_cost_gate_opens_only_on_a_criterion_1_pass(rule, tmp_path):
    allowed, _why = rule.cost_gate(str(Session().write(tmp_path / "ok")))
    assert allowed is True
    session = Session()
    session.arms["l2l_nospill_2"]["logs"][0]["loss"] = 9.0
    allowed, why = rule.cost_gate(str(session.write(tmp_path / "open")))
    assert allowed is False and "no verdict" in why


def test_a_spill_arm_that_spilled_nothing_is_invalid(rule, tmp_path):
    session = Session()
    spill = session.arms["l2l_spill"]
    spill["spill_plan"] = _plan(0, f"{SPILL_DIR}/l2l_spill")
    # bytes still moved: only the spill_layers check can void the arm
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert "spill_layers 0: the spill did not happen" in result["criteria"]["1"]["detail"]
    assert "spill_layers 0: the spill did not happen" in result["criteria"]["3"]["detail"]
    assert result["criteria"]["2"]["status"] == "NO VERDICT"


def test_a_spill_plan_without_bytes_moved_is_invalid(rule, tmp_path):
    session = Session()
    session.arms["l2l_spill"]["store_io"] = {"bytes_written": 0, "bytes_read": 3 << 30}
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert "moved 0 B out" in result["criteria"]["1"]["detail"]


def test_a_no_spill_reference_that_spilled_is_invalid(rule, tmp_path):
    session = Session()
    session.arms["l2l_nospill_1"]["spill_plan"]["spill_layers"] = 3
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert "no-spill arm planned spill_layers 3" in result["criteria"]["1"]["detail"]


def test_different_rows_give_no_verdict(rule, tmp_path):
    session = Session()
    session.arms["l2l_spill"]["rows"]["row_sha256"][2] = "other"
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert "V5" in result["criteria"]["1"]["detail"]


@pytest.mark.parametrize("field, value, word", [
    ("CUBLAS_WORKSPACE_CONFIG", None, "V2"),
    ("algorithms", False, "V2"),
    ("warn_only", True, "V2"),
    ("sdpa", "FLASH_ATTENTION", "V2"),
])
def test_a_spill_arm_without_the_deterministic_setup_gives_no_verdict(rule, tmp_path, field,
                                                                      value, word):
    session = Session()
    session.arms["l2l_spill"]["deterministic"][field] = value
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert word in result["criteria"]["1"]["detail"]


@pytest.mark.parametrize("damage, word", [
    (lambda rec, entry: rec["cgroup_start"].update(memory_max=TIGHT + (1 << 20)), "memory.max"),
    (lambda rec, entry: rec["cgroup_start"].update(memory_max="max"), "memory.max"),
    (lambda rec, entry: rec["cgroup_start"].update(swap_max="max"), "swap.max"),
    (lambda rec, entry: rec.update(cgroup_start={"version": None}), "no cgroup v2"),
    (lambda rec, entry: entry.update(memory_max=None), "no planned limit"),
    (lambda rec, entry: rec.update(expected_memory_max=GENEROUS), "memory.max"),
])
def test_a_cgroup_readback_unequal_to_the_plan_voids_the_arm(rule, tmp_path, damage, word):
    session = Session()
    damage(session.arms["l2l_spill"], session.entries["l2l_spill"])
    result = _judge(rule, session, tmp_path)
    for criterion in ("1", "2"):
        assert result["criteria"][criterion]["status"] == "NO VERDICT"
        assert "V3" in result["criteria"][criterion]["detail"]
        assert word in result["criteria"][criterion]["detail"]


@pytest.mark.parametrize("damage, word", [
    (lambda rec: rec.update(exit="error", error="boom"), "exit"),
    (lambda rec: rec["logs"][0].update(loss=float("nan")), "not finite"),
    (lambda rec: [d.update(nonzero=False) for d in rec["grads"][1].values()], "all-zero"),
    (lambda rec: rec["grads"][0]["lora_A"].update(finite=False), "non-finite"),
    (lambda rec: rec.update(expected_steps=4), "expected"),
])
def test_an_arm_that_did_not_train_properly_gives_no_verdict(rule, tmp_path, damage, word):
    session = Session()
    for name in ("l2l_nospill_1", "l2l_nospill_2", "l2l_spill"):
        damage(session.arms[name])
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert word in result["criteria"]["1"]["detail"]


def test_a_missing_arm_gives_no_verdict(rule, tmp_path):
    session = Session()
    del session.arms["l2l_nospill_2"]
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert "no record" in result["criteria"]["1"]["detail"]


# --------------------------------------------------------------------------
# criterion 1b (reported)
# --------------------------------------------------------------------------
def test_the_ga_tolerance_is_reported_and_does_not_decide_the_spill(rule, tmp_path):
    session = Session()
    session.compare["tolerance"] = {"status": "l2l_out_of_tolerance", "pass": False, "note": "x"}
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1b"]["status"] == "FAIL"
    assert result["criteria"]["1"]["status"] == "PASS"
    assert result["verdict"].startswith("SPILL CONFIRMED: ")
    assert "criterion 1b" in result["verdict"]


@pytest.mark.parametrize("damage", [
    lambda s: s.compare["hashes_match"].update(ga_floor=False),
    lambda s: s.calibration.update(aliasing={"pass": False, "status": "active"}),
    lambda s: s.compare.update(tolerance=None),
    lambda s: s.arms["ga_floor"]["deterministic"].update(sdpa="MATH"),
])
def test_the_ga_tolerance_without_its_preconditions_gives_no_verdict(rule, tmp_path, damage):
    session = Session()
    damage(session)
    assert _judge(rule, session, tmp_path)["criteria"]["1b"]["status"] == "NO VERDICT"


# --------------------------------------------------------------------------
# criterion 2
# --------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ["l2l_spill", "l2l_nospill_1", "l2l_nospill_2"])
def test_an_oom_kill_fails_ram(rule, tmp_path, arm):
    session = Session()
    session.arms[arm]["cgroup_end"]["events"]["oom_kill"] = 1
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "FAIL"
    assert "oom_kill rose 0 -> 1" in _part(result, "2", arm)["detail"]


def test_a_child_that_did_not_exit_0_fails_ram(rule, tmp_path):
    session = Session()
    session.entries["l2l_spill"]["returncode"] = -9
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "FAIL"
    assert "exited -9" in _part(result, "2", "l2l_spill")["detail"]


def _killed(session: Session, marker: Optional[Dict[str, Any]]) -> None:
    """l2l_spill OOM-killed: no record left; the parent read its first-step marker (or not)."""
    del session.arms["l2l_spill"]
    entry = session.entries["l2l_spill"]
    entry.update(returncode=-9, marker="markers/l2l_spill.json", marker_after_exit=marker)


@pytest.mark.parametrize("marker, phase", [
    (None, "before first training_step (build): no first-step marker"),
    ({"steps_done": 0, "pid": 7}, "after first training_step (marker at 0 optimizer steps "
                                  "done)"),
])
def test_a_killed_spill_child_fails_ram_and_says_build_or_training(rule, tmp_path, marker,
                                                                   phase):
    session = Session()
    _killed(session, marker)
    result = _judge(rule, session, tmp_path)
    part = _part(result, "2", "l2l_spill")
    assert part["status"] == "FAIL" and result["criteria"]["2"]["status"] == "FAIL"
    assert part["oom_phase"] == phase and phase in part["detail"]
    assert result["verdict"] == "SPILL NOT SAFE IN THE TRAINER: criterion 2 failed"


@pytest.mark.parametrize("first, end, phase", [
    (1, 1, "before first training_step (build)"),
    (0, 1, "after first training_step"),
])
def test_an_oom_kill_in_a_surviving_child_says_build_or_training(rule, tmp_path, first, end,
                                                                 phase):
    session = Session()
    spill = session.arms["l2l_spill"]
    spill["cgroup_at_first_step"]["events"]["oom_kill"] = first
    spill["cgroup_end"]["events"]["oom_kill"] = end
    part = _part(_judge(rule, session, tmp_path), "2", "l2l_spill")
    assert part["status"] == "FAIL" and part["oom_phase"] == phase
    assert part["detail"] == f"l2l_spill: memory.events oom_kill rose 0 -> 1, {phase}"


def test_a_kill_without_a_planned_marker_says_the_phase_is_unknown(rule, tmp_path):
    session = Session()
    del session.arms["l2l_spill"]
    session.entries["l2l_spill"]["returncode"] = -9
    part = _part(_judge(rule, session, tmp_path), "2", "l2l_spill")
    assert part["status"] == "FAIL" and part["oom_phase"].startswith("unknown")


def test_a_peak_above_the_limit_fails_ram(rule, tmp_path):
    session = Session()
    session.arms["l2l_spill"]["cgroup_end"]["peak"] = TIGHT + 4096
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "FAIL"
    assert "memory.peak" in _part(result, "2", "l2l_spill")["detail"]


def test_a_peak_exactly_at_the_limit_passes(rule, tmp_path):
    session = Session()
    session.arms["l2l_spill"]["cgroup_end"]["peak"] = TIGHT
    assert _judge(rule, session, tmp_path)["criteria"]["2"]["status"] == "PASS"


def test_no_peak_or_events_gives_no_verdict(rule, tmp_path):
    session = Session()
    session.arms["l2l_spill"]["cgroup_end"]["peak"] = None
    session.arms["l2l_nospill_1"]["cgroup_end"]["events"] = {}
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "NO VERDICT"
    assert "no memory.peak" in _part(result, "2", "l2l_spill")["detail"]
    assert "oom_kill not read" in _part(result, "2", "l2l_nospill_1")["detail"]


def test_more_host_memory_outside_the_store_with_spill_fails_ram(rule, tmp_path):
    session = Session()
    session.arms["l2l_spill"]["host_allocator"]["at_last_step_bytes"] += 1
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "FAIL"
    excess = result["criteria"]["2"]["excess_bytes"]
    assert excess["l2l_spill"] == excess["l2l_nospill_1"] + 1
    assert "outside its store" in _part(result, "2", "excess")["detail"]


def test_equal_excess_passes_and_both_are_recorded(rule, tmp_path):
    result = _judge(rule, Session(), tmp_path)
    excess = result["criteria"]["2"]["excess_bytes"]
    assert excess["l2l_spill"] == excess["l2l_nospill_1"] == 64 << 20
    assert _part(result, "2", "excess")["status"] == "PASS"


def test_a_missing_allocator_counter_gives_no_verdict(rule, tmp_path):
    session = Session()
    session.arms["l2l_spill"]["host_allocator"]["baseline_bytes"] = None
    result = _judge(rule, session, tmp_path)
    assert _part(result, "2", "excess")["status"] == "NO VERDICT"
    assert result["criteria"]["2"]["status"] == "NO VERDICT"


# --------------------------------------------------------------------------
# criterion 3
# --------------------------------------------------------------------------
@pytest.mark.parametrize("where", ["after_train", "after_exit"])
def test_a_file_left_after_a_normal_end_fails_cleanup(rule, tmp_path, where):
    session = Session()
    left = [f"{SPILL_DIR}/l2l_spill/soup-l2l-spill-1-00000000.bin"]
    if where == "after_train":
        session.arms["l2l_spill"]["spill_files_after_train"] = left
    else:
        session.entries["l2l_spill"]["spill_files_after_exit"] = left
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["3"]["status"] == "FAIL"
    assert _part(result, "3", "normal")["status"] == "FAIL"


def test_a_file_left_after_sigint_fails_cleanup(rule, tmp_path):
    session = Session()
    session.entries["sigint"]["signal"]["files_after_exit"] = [SIGINT_FILE]
    result = _judge(rule, session, tmp_path)
    assert _part(result, "3", "sigint")["status"] == "FAIL"
    assert result["criteria"]["3"]["status"] == "FAIL"


def test_a_child_that_ignores_sigint_fails_cleanup(rule, tmp_path):
    session = Session()
    session.entries["sigint"]["signal"]["exited"] = False
    assert _part(_judge(rule, session, tmp_path), "3", "sigint")["status"] == "FAIL"


@pytest.mark.parametrize("damage", [
    lambda s: s.entries["sigint"]["signal"].update(files_before_signal=[]),
    lambda s: s.cleanup["sigint"].update(at_interrupt={"in_step": False, "steps_done": 1}),
    lambda s: s.cleanup["sigint"].update(exit="normal"),
    lambda s: s.entries["sigint"]["signal"].update(sent_unix=None, why="exited early"),
])
def test_a_sigint_that_missed_the_scenario_gives_no_verdict(rule, tmp_path, damage):
    session = Session()
    damage(session)
    result = _judge(rule, session, tmp_path)
    assert _part(result, "3", "sigint")["status"] == "NO VERDICT"
    assert result["criteria"]["3"]["status"] == "NO VERDICT"


def test_kill9_fails_when_the_killed_run_removed_its_own_file(rule, tmp_path):
    session = Session()
    session.entries["kill9"]["signal"]["files_after_exit"] = []
    session.cleanup["kill9_next"]["stale_removed"] = []
    result = _judge(rule, session, tmp_path)
    assert _part(result, "3", "kill9")["status"] == "FAIL"
    assert "gone after SIGKILL" in _part(result, "3", "kill9")["detail"]
    assert result["criteria"]["3"]["status"] == "FAIL"


def test_kill9_fails_when_the_left_file_is_not_removed_at_the_next_start(rule, tmp_path):
    session = Session()
    session.cleanup["kill9_next"]["stale_removed"] = []
    session.entries["kill9_next"]["spill_files_after_exit"] = [KILLED_FILE]
    result = _judge(rule, session, tmp_path)
    assert _part(result, "3", "kill9")["status"] == "FAIL"
    assert "did not remove" in _part(result, "3", "kill9")["detail"]


def test_kill9_fails_when_the_file_is_claimed_removed_but_still_listed(rule, tmp_path):
    session = Session()
    session.cleanup["kill9_next"]["spill_files_after_train"] = [KILLED_FILE]
    assert _part(_judge(rule, session, tmp_path), "3", "kill9")["status"] == "FAIL"


def test_kill9_fails_when_the_next_run_does_not_train(rule, tmp_path):
    session = Session()
    session.cleanup["kill9_next"].update(steps=1)
    result = _judge(rule, session, tmp_path)
    assert _part(result, "3", "kill9")["status"] == "FAIL"
    assert "did not train normally" in _part(result, "3", "kill9")["detail"]


def test_kill9_fails_when_the_next_child_crashes(rule, tmp_path):
    session = Session()
    session.entries["kill9_next"]["returncode"] = 1
    assert _part(_judge(rule, session, tmp_path), "3", "kill9")["status"] == "FAIL"


def test_kill9_before_any_spill_file_existed_gives_no_verdict(rule, tmp_path):
    session = Session()
    session.entries["kill9"]["signal"]["files_before_signal"] = []
    assert _part(_judge(rule, session, tmp_path), "3", "kill9")["status"] == "NO VERDICT"


def test_kill9_outside_its_planned_scope_gives_no_verdict(rule, tmp_path):
    session = Session()
    session.entries["kill9"]["signal"]["marker"]["cgroup_start"]["memory_max"] = GENEROUS
    result = _judge(rule, session, tmp_path)
    assert _part(result, "3", "kill9")["status"] == "NO VERDICT"
    assert "V3" in _part(result, "3", "kill9")["detail"]


def test_cleanup_passes_only_when_all_three_hold(rule, tmp_path):
    result = _judge(rule, Session(), tmp_path)
    assert [p["status"] for p in result["criteria"]["3"]["parts"]] == ["PASS"] * 3


# --------------------------------------------------------------------------
# criterion 4
# --------------------------------------------------------------------------
def test_an_upper_bound_below_10_percent_passes_cost(rule, tmp_path):
    result = _judge(rule, Session([0.03, 0.04, 0.02, 0.05, 0.03]), tmp_path)
    cost = result["criteria"]["4"]
    assert cost["status"] == "PASS" and cost["interval"]["high"] < 0.10
    assert "confirmed on a faster machine" in cost["detail"]
    assert cost["effects"] == pytest.approx([0.03, 0.04, 0.02, 0.05, 0.03])


def test_a_lower_bound_at_or_above_10_percent_fails_cost(rule, tmp_path):
    result = _judge(rule, Session([0.20, 0.21, 0.19, 0.22, 0.20]), tmp_path)
    cost = result["criteria"]["4"]
    assert cost["status"] == "FAIL" and cost["interval"]["low"] >= 0.10
    assert "docs state the measured overhead" in cost["detail"]
    assert result["verdict"].startswith("SPILL CONFIRMED, COSTLY")


def test_an_interval_that_straddles_10_percent_gives_no_verdict(rule, tmp_path):
    result = _judge(rule, Session([0.06, 0.14, 0.09, 0.12, 0.08]), tmp_path)
    cost = result["criteria"]["4"]
    assert cost["interval"]["low"] < 0.10 <= cost["interval"]["high"]
    assert cost["status"] == "NO VERDICT" and "straddles" in cost["detail"]
    assert result["verdict"].startswith("SPILL CONFIRMED, COST NO VERDICT")


@pytest.mark.parametrize("effects, status", [
    ([0.093, 0.095, 0.094, 0.096, 0.092], "PASS"),  # upper bound just below 10%
    ([0.104, 0.106, 0.105, 0.107, 0.103], "FAIL"),  # lower bound just above 10%
])
def test_the_cost_outcome_holds_close_to_10_percent(rule, tmp_path, effects, status):
    cost = _judge(rule, Session(effects), tmp_path)["criteria"]["4"]
    assert cost["status"] == status
    bound = cost["interval"]["high"] if status == "PASS" else cost["interval"]["low"]
    assert abs(bound - 0.10) < 0.01



def test_the_interval_is_the_two_sided_95_percent_t_interval(rule):
    out = rule.interval([0.03, 0.04, 0.02, 0.05, 0.03])
    assert out["mean"] == pytest.approx(0.034)
    half = 2.776 * 0.011401754 / 5 ** 0.5
    assert out["low"] == pytest.approx(0.034 - half) and out["high"] == pytest.approx(0.034 + half)


def test_the_repeats_rule_not_satisfied_gives_no_verdict(rule, tmp_path):
    noisy = [0.30, -0.25, 0.28, -0.20, 0.33]  # sigma wants far more than 5 blocks
    result = _judge(rule, Session(noisy), tmp_path)
    cost = result["criteria"]["4"]
    assert cost["status"] == "NO VERDICT" and "V7" in cost["detail"]


def test_fewer_than_the_pilot_blocks_gives_no_verdict(rule, tmp_path):
    result = _judge(rule, Session([0.03, 0.04, 0.02]), tmp_path)
    assert result["criteria"]["4"]["status"] == "NO VERDICT"
    assert "V7: the repeats rule wants 2 more" in result["criteria"]["4"]["detail"]


def test_a_void_block_is_replaced_not_counted(rule, tmp_path):
    session = Session([0.03, 0.04, 0.02, 0.05, 0.03, 0.04])
    session.cost[1]["outcome"] = "void"  # block 0, the first B
    cost = _judge(rule, session, tmp_path)["criteria"]["4"]
    assert cost["status"] == "PASS"
    assert cost["blocks"] == 6 and cost["void_blocks"] == 1 and cost["valid_blocks"] == 5


def test_an_unknown_f1_outcome_gives_no_verdict(rule, tmp_path):
    session = Session()
    session.cost[0]["outcome"] = "unknown"
    cost = _judge(rule, session, tmp_path)["criteria"]["4"]
    assert cost["status"] == "NO VERDICT" and "V6" in cost["detail"]


def test_a_cost_b_arm_that_did_not_spill_gives_no_verdict(rule, tmp_path):
    session = Session()
    session.cost[1]["spill_plan"]["spill_layers"] = 0
    cost = _judge(rule, session, tmp_path)["criteria"]["4"]
    assert cost["status"] == "NO VERDICT" and "spill did not happen" in cost["detail"]


def test_no_cost_block_gives_no_verdict(rule, tmp_path):
    result = _judge(rule, Session([]), tmp_path)
    assert result["criteria"]["4"]["status"] == "NO VERDICT"
    assert result["criteria"]["4"]["detail"] == "no cost block ran"


# --------------------------------------------------------------------------
# V1 and the whole verdict
# --------------------------------------------------------------------------
@pytest.mark.parametrize("where", ["arm", "cleanup", "cost", "calibration"])
def test_a_cpu_session_never_gets_a_verdict(rule, tmp_path, where):
    session = Session()
    if where == "arm":
        session.arms["ga_2"]["device"] = "cpu"
    elif where == "cleanup":
        session.cleanup["kill9_next"]["device"] = "cpu"
    elif where == "cost":
        session.cost[5]["device"] = "cpu"
    else:
        session.calibration["device"] = "cpu"
    result = _judge(rule, session, tmp_path)
    assert result["verdict"] == "NO VERDICT: CPU session (V1)"
    assert result["summary"]["cpu_session"] is True


def test_every_arm_on_cpu_gives_no_verdict_even_with_passing_numbers(rule, tmp_path):
    session = Session()
    for rec in [*session.arms.values(), *session.cleanup.values(), *session.cost]:
        rec["device"] = "cpu"
    result = _judge(rule, session, tmp_path)
    assert result["verdict"] == "NO VERDICT: CPU session (V1)"
    assert result["criteria"]["1"]["status"] == "NO VERDICT"


def test_a_safety_failure_names_its_criterion(rule, tmp_path):
    session = Session()
    session.arms["l2l_spill"]["cgroup_end"]["peak"] = TIGHT + 1
    assert _judge(rule, session, tmp_path)["verdict"] == (
        "SPILL NOT SAFE IN THE TRAINER: criterion 2 failed")


def test_an_open_safety_criterion_gives_no_verdict(rule, tmp_path):
    session = Session()
    del session.entries["sigint"]
    result = _judge(rule, session, tmp_path)
    assert result["verdict"] == "NO VERDICT: criterion 3 has no verdict"


def test_the_verdict_file_is_lf_and_rerunnable(rule, tmp_path):
    root = Session().write(tmp_path / "session")
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    assert rule.main([str(root), "--out", str(first)]) == 0
    rule.write_verdict(second, rule.evaluate(str(root)))
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()


def test_the_rule_never_mutates_its_inputs(rule, tmp_path):
    session = Session()
    before = copy.deepcopy(session.__dict__)
    root = session.write(tmp_path / "session")
    snapshot = {p: p.read_bytes() for p in root.rglob("*.json")}
    rule.evaluate(str(root))
    assert {p: p.read_bytes() for p in root.rglob("*.json")} == snapshot
    assert session.__dict__ == before
