"""CPU tests for the A3 stage-2 rule (benchmarks/harness/a3_spill_rule.py).

Every test writes a synthetic session directory in the shape ``a3_spill_stand.py session``
leaves, with the numbers a passing GPU run would have, then breaks one thing and checks the
criterion it belongs to. The rule reads JSON only, so nothing here needs torch or a GPU.
"""

from __future__ import annotations

import copy
import hashlib
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
GRAD_A = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
GRAD_B = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"


@pytest.fixture()
def rule():
    return _load("a3_spill_rule")


# --------------------------------------------------------------------------
# a passing synthetic session
# --------------------------------------------------------------------------
def _digest(seed: str, step: int) -> Dict[str, Any]:
    return {"sha256": hashlib.sha256(f"{seed}-{step}".encode()).hexdigest(),
            "dtype": "float32", "shape": [8, 4096],
            "finite": True, "nonzero": True}


def _gradients(seed: str, step: int) -> Dict[str, Any]:
    # The fixed Mistral fixture: 32 layers, q/k/v/o adapters, both LoRA matrices.
    return {
        f"base_model.model.model.layers.{layer}.self_attn.{projection}_proj."
        f"lora_{matrix}.default.weight": _digest(f"{seed}-{layer}-{projection}-{matrix}", step)
        for layer in range(32) for projection in ("q", "k", "v", "o") for matrix in ("A", "B")
    }


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
        "grads": [] if name.startswith("cost/") else [_gradients(grads, s) for s in range(steps)],
        "logs": [{"loss": 2.5 - 0.1 * s, "learning_rate": 0.001 * (steps - s),
                  "grad_norm": 1.0 + s}
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
    rec.update(role=role, block=block, position=position, value=value, outcome="ok",
               grads=[], step_s=[value] * 7, timed_step_s=[value] * 5)
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
        hold = {"layer": 15, "micro_batch": 0, "pass": "backward", "released": "interrupt"}
        sigint.update(exit="interrupted", steps=1, hold=hold,
                      at_interrupt={"in_step": True, "steps_done": 1, "hold": hold})
        kill9_next = _arm("kill9_next", TIGHT, layers=16)
        kill9_next["stale_removed"] = [KILLED_FILE]
        self.cleanup = {"sigint": sigint, "kill9_next": kill9_next}
        marker = {"pid": 4242, "steps_done": 1, "in_step": True, "phase": "hold",
                  "spill_path": KILLED_FILE, "device": "cuda", "expected_memory_max": TIGHT,
                  "cgroup_start": _cgroup(TIGHT)}
        self.entries["sigint"] = _entry("sigint", TIGHT, returncode=130, signal={
            "signal": "SIGINT", "sent_unix": 1.0, "exited": True, "returncode": 130,
            "files_before_signal": [SIGINT_FILE], "files_after_exit": [],
            "target_at_send": {"pending": False, "blocked": False, "ignored": False,
                               "caught": True, "state": "S (sleeping)"},
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
    session.arms["l2l_nospill_2"]["grads"][1][GRAD_A]["sha256"] = _digest("drift", 1)["sha256"]
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert "does not repeat" in result["criteria"]["1"]["detail"]
    assert result["criteria"]["4"]["status"] == "NO VERDICT"
    assert result["criteria"]["4"]["refused"] is True


@pytest.mark.parametrize("damage, word", [
    (lambda rec: rec["grads"][2][GRAD_B].update(sha256=_digest("other", 2)["sha256"]), "grads"),
    (lambda rec: rec["grads"][0][GRAD_A].update(dtype="bfloat16"), "grads"),
    (lambda rec: rec["grads"][0][GRAD_A].update(shape=[4096, 8]), "grads"),
    (lambda rec: rec["logs"][1].update(loss=2.4000001), "losses"),
    (lambda rec: rec["rates"].__setitem__(2, 0.5), "rates"),
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
    (lambda rec: rec["grads"][0][GRAD_A].update(finite=False), "non-finite"),
    (lambda rec: rec.update(expected_steps=4), "expected"),
])
def test_an_arm_that_did_not_train_properly_gives_no_verdict(rule, tmp_path, damage, word):
    session = Session()
    for name in ("l2l_nospill_1", "l2l_nospill_2", "l2l_spill"):
        damage(session.arms[name])
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert word in result["criteria"]["1"]["detail"]


@pytest.mark.parametrize("damage", [
    lambda rec: [d.pop("sha256") for step in rec["grads"] for d in step.values()],
    lambda rec: [d.pop("dtype") for step in rec["grads"] for d in step.values()],
    lambda rec: [d.pop("shape") for step in rec["grads"] for d in step.values()],
    lambda rec: [d.pop("finite") for step in rec["grads"] for d in step.values()],
    lambda rec: [d.pop("nonzero") for step in rec["grads"] for d in step.values()],
    lambda rec: [d.update(sha256="not-a-sha256") for step in rec["grads"]
                 for d in step.values()],
    lambda rec: [d.update(dtype="") for step in rec["grads"] for d in step.values()],
    lambda rec: [d.update(dtype="not-a-floating-dtype") for step in rec["grads"]
                 for d in step.values()],
    lambda rec: [d.update(shape=[]) for step in rec["grads"] for d in step.values()],
    lambda rec: [d.update(shape=[8, -1]) for step in rec["grads"] for d in step.values()],
    lambda rec: rec.update(grads=[{} for _ in range(3)]),
    lambda rec: rec.pop("rates"),
    lambda rec: rec.update(rates=[0.001]),
    lambda rec: rec["rates"].__setitem__(1, float("inf")),
    lambda rec: [row.pop("learning_rate") for row in rec["logs"]],
    lambda rec: rec["logs"][0].update(learning_rate=float("nan")),
    lambda rec: rec.update(logs=rec["logs"][:1]),
    lambda rec: rec["logs"][0].update(loss=True),
    lambda rec: [row.pop("loss") for row in rec["logs"]],
], ids=["missing-sha256", "missing-dtype", "missing-shape", "missing-finite",
        "missing-nonzero", "invalid-sha256", "empty-dtype", "invalid-dtype", "empty-shape",
        "invalid-dimension", "empty-gradient-steps", "missing-rates", "short-rates",
        "infinite-rate", "missing-logged-rate", "nan-logged-rate", "short-logs",
        "boolean-loss", "missing-loss"])
def test_matching_missing_numerical_evidence_cannot_prove_equality(rule, tmp_path, damage):
    session = Session()
    for name in ("l2l_nospill_1", "l2l_nospill_2", "l2l_spill"):
        damage(session.arms[name])
    root = session.write(tmp_path / "session")
    result = rule.evaluate(str(root))
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert result["criteria"]["4"]["refused"] is True
    assert rule.cost_gate(str(root))[0] is False
    compared = rule.compare_records(session.arms["l2l_nospill_1"], session.arms["l2l_spill"])
    assert compared["equal"] is False


@pytest.mark.parametrize("omit", ["one-parameter", "all-lora-B", "last-layer"])
def test_matching_whole_parameter_omissions_cannot_prove_equality(rule, tmp_path, omit):
    session = Session()
    for name in ("l2l_nospill_1", "l2l_nospill_2", "l2l_spill"):
        for step in session.arms[name]["grads"]:
            removed = [name for name in step if (name == GRAD_B if omit == "one-parameter"
                       else ".lora_B." in name if omit == "all-lora-B"
                       else ".layers.31." in name)]
            for name in removed:
                del step[name]
    root = session.write(tmp_path / "session")
    assert rule.evaluate(str(root))["criteria"]["1"]["status"] == "NO VERDICT"
    assert rule.cost_gate(str(root))[0] is False
    assert not rule.compare_records(
        session.arms["l2l_nospill_1"], session.arms["l2l_spill"]
    )["equal"]


@pytest.mark.parametrize("steps", [0, 1, 4])
def test_matching_self_reported_step_counts_do_not_replace_the_three_step_fixture(
        rule, tmp_path, steps):
    session = Session()
    for name in ("l2l_nospill_1", "l2l_nospill_2", "l2l_spill"):
        rec = session.arms[name]
        rec.update(steps=steps, expected_steps=steps,
                   grads=[copy.deepcopy(rec["grads"][0]) for _ in range(steps)],
                   logs=[copy.deepcopy(rec["logs"][0]) for _ in range(max(1, steps))],
                   rates=[0.001] * steps)
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"
    assert result["criteria"]["4"]["refused"] is True


def test_training_evidence_validation_is_shared_with_cpu_preflight(rule):
    rec = _arm("kill9_next", TIGHT, layers=16)
    rec["device"] = "cpu"
    assert rule.training_problems(rec) == []
    rec.update(steps=0, expected_steps=0, grads=[])
    assert rule.training_problems(rec)


def test_incidental_record_metadata_does_not_change_numerical_validity(rule, tmp_path):
    session = Session()
    for rec in session.arms.values():
        rec.update(argv=None, pid="not-used-by-numerical-comparison", log_history=None)
        for row in rec["logs"]:
            row.update(epoch="not-used", step="not-used")
    assert _judge(rule, session, tmp_path)["criteria"]["1"]["status"] == "PASS"


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



# --------------------------------------------------------------------------
# criterion 2, rule v2: any child at the tight limit, by the kernel's word
# --------------------------------------------------------------------------
def _tight_child(name: str, *, returncode: int, kernel: Optional[Dict[str, Any]],
                 marker: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    entry = _entry(name, TIGHT, returncode=returncode)
    entry["limit"] = "tight"  # the scope, as child_entry names it
    if kernel is not None:
        entry["kernel_oom"] = kernel
    if marker is not None:
        entry["marker_after_exit"] = marker
    return entry


KILLED_BY_KERNEL = {"read": True, "killed": True, "pids": [4204],
                    "lines": ["Memory cgroup out of memory: Killed process 4204 (python)"]}
FIRST_STEP = {"pid": 4204, "steps_done": 0, "in_step": True, "device": "cuda",
              "expected_memory_max": TIGHT,
              "cgroup_start": {"version": 2, "memory_max": TIGHT, "swap_max": 0}}


def test_the_rule_is_v3(rule):
    assert rule.RULE_VERSION == "v3" and rule.RULE["profile"] == "a3-spill-gpu-check-v3"


@pytest.mark.parametrize("name", ["calibration_check_1", "sigint", "kill9_next",
                                  "cost/b00_p1_B"])
def test_a_kernel_oom_after_the_first_step_of_any_tight_child_fails_ram(rule, tmp_path, name):
    session = Session()
    session.entries[name] = _tight_child(name, returncode=-9, kernel=KILLED_BY_KERNEL,
                                         marker=FIRST_STEP)
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "FAIL"
    part = _part(result, "2", name)
    assert "after first training_step (marker at 0 optimizer steps done)" in part["detail"]
    assert part["kernel_lines"] == KILLED_BY_KERNEL["lines"]
    assert result["verdict"].startswith("SPILL NOT SAFE IN THE TRAINER: criterion 2")


def test_session_1_shape_under_v2_is_spill_not_safe(rule, tmp_path):
    """Only the calibration and its check ran; the check was OOM-killed in its first step."""
    root = tmp_path / "s1"
    root.mkdir()
    children = [{**_entry("calibration", GENEROUS), "limit": "generous"},
                _tight_child("calibration_check_1", returncode=-9, kernel=KILLED_BY_KERNEL,
                             marker=FIRST_STEP)]
    (root / "session.json").write_text(json.dumps({"state": "stopped_oom_in_training",
                                                   "children": children, "limits": {}}))
    result = rule.evaluate(str(root))
    assert result["verdict"] == "SPILL NOT SAFE IN THE TRAINER: criterion 2 failed"
    assert result["criteria"]["1"]["status"] == "NO VERDICT"


def test_the_parents_sigkill_of_kill9_is_not_an_oom(rule, tmp_path):
    session = Session()
    session.entries["kill9"].update(limit="tight", signal_name="SIGKILL",
                                    kernel_oom={"read": True, "killed": False, "lines": []})
    assert _judge(rule, session, tmp_path)["criteria"]["2"]["status"] == "PASS"


@pytest.mark.parametrize("signal_name, sent_signal, sent_unix, name", [
    ("SIGINT", "SIGINT", 1.0, "sigint"),
    ("SIGKILL", "SIGKILL", None, "kill9"),
    ("SIGKILL", "SIGINT", 1.0, "kill9"),
    ("SIGKILL", "SIGKILL", 1.0, "sigint"),
    ("SIGKILL", "SIGKILL", 0.0, "kill9"),
    ("SIGKILL", "SIGKILL", float("nan"), "kill9"),
], ids=["sigint", "unsent-kill9", "wrong-sent-signal", "wrong-child", "zero-timestamp",
        "nan-timestamp"])
def test_only_an_actually_sent_planned_kill9_exempts_an_unexplained_kill(
        rule, tmp_path, signal_name, sent_signal, sent_unix, name):
    session = Session()
    entry = session.entries[name]
    entry.update(limit="tight", returncode=-9, signal_name=signal_name,
                 kernel_oom={"read": False, "error": "kernel log unavailable"})
    entry["signal"].update(signal=sent_signal, sent_unix=sent_unix)
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "NO VERDICT"
    assert "cannot tell" in _part(result, "2", name)["detail"]


def test_an_actually_sent_planned_kill9_does_not_need_a_kernel_log(rule, tmp_path):
    session = Session()
    session.entries["kill9"].update(limit="tight", signal_name="SIGKILL",
                                    kernel_oom={"read": False, "error": "unavailable"})
    assert _judge(rule, session, tmp_path)["criteria"]["2"]["status"] == "PASS"


@pytest.mark.parametrize("damage", [
    lambda sent: sent["marker"].update(phase="enter"),
    lambda sent: sent["marker"].update(steps_done=0),
    lambda sent: sent.update(files_before_signal=[]),
], ids=["not-held", "wrong-step", "no-spill-file"])
def test_a_kill9_label_without_the_planned_spill_hold_cannot_explain_a_kill(
        rule, tmp_path, damage):
    session = Session()
    entry = session.entries["kill9"]
    entry.update(limit="tight", signal_name="SIGKILL", kernel_oom={"read": False})
    damage(entry["signal"])
    assert _judge(rule, session, tmp_path)["criteria"]["2"]["status"] == "NO VERDICT"


@pytest.mark.parametrize("path", [None, ""])
def test_a_missing_spill_path_cannot_exempt_an_unexplained_kill(rule, tmp_path, path):
    session = Session()
    entry = session.entries["kill9"]
    entry.update(limit="tight", signal_name="SIGKILL", kernel_oom={"read": False})
    entry["signal"]["marker"]["spill_path"] = path
    entry["signal"]["files_before_signal"] = [path]
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "NO VERDICT"
    assert "cannot tell" in _part(result, "2", "kill9")["detail"]


@pytest.mark.parametrize("container", [1, True, "not-a-container", []])
@pytest.mark.parametrize("field", ["signal", "marker", "files_before_signal"])
@pytest.mark.parametrize("known_oom", [False, True])
def test_malformed_kill_exemption_evidence_cannot_hide_unknown_or_known_oom(
        rule, tmp_path, container, field, known_oom):
    session = Session()
    entry = session.entries["kill9"]
    entry.update(limit="tight", signal_name="SIGKILL", kernel_oom={"read": False})
    if field == "signal":
        entry[field] = container
    else:
        entry["signal"][field] = container
    if known_oom:
        session.entries["calibration_check_1"] = _tight_child(
            "calibration_check_1", returncode=-9, kernel=KILLED_BY_KERNEL, marker=FIRST_STEP)
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == ("FAIL" if known_oom else "NO VERDICT")
    assert "cannot tell" in _part(result, "2", "kill9")["detail"]


@pytest.mark.parametrize("field", ["loss", "learning_rate", "rate", "bytes_read"])
def test_oversized_json_integer_evidence_does_not_abort_the_rule(rule, tmp_path, field):
    session = Session()
    rec = session.arms["l2l_spill"]
    if field == "bytes_read":
        rec["store_io"][field] = 10**400
    elif field == "rate":
        rec["rates"][0] = 10**400
    else:
        rec["logs"][0][field] = 10**400
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["1"]["status"] == "NO VERDICT"


def test_a_known_kernel_oom_still_fails_even_for_the_intentional_kill9(rule, tmp_path):
    session = Session()
    session.entries["kill9"].update(limit="tight", signal_name="SIGKILL",
                                    kernel_oom=KILLED_BY_KERNEL, marker_after_exit=FIRST_STEP)
    # Unknown evidence in another child must not mask this known OOM failure.
    session.entries["sigint"].update(limit="tight", returncode=-9, signal_name="SIGINT",
                                     kernel_oom={"read": False})
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "FAIL"
    assert _part(result, "2", "kill9")["status"] == "FAIL"
    assert result["verdict"] == "SPILL NOT SAFE IN THE TRAINER: criterion 2 failed"


def test_a_kernel_oom_before_the_first_step_is_the_stands_limit_not_a_fail(rule, tmp_path):
    session = Session()
    session.entries["sigint"] = _tight_child("sigint", returncode=-9, kernel=KILLED_BY_KERNEL)
    result = _judge(rule, session, tmp_path)
    part = _part(result, "2", "sigint")
    assert part["status"] == "NO VERDICT" and part["oom_phase"] == rule.BUILD_PHASE
    assert result["criteria"]["2"]["status"] == "NO VERDICT"


@pytest.mark.parametrize("kernel", [None, {"read": False, "killed": False,
                                           "error": "journalctl exited 1"}])
def test_a_minus_9_without_the_kernels_word_is_no_verdict_not_fail(rule, tmp_path, kernel):
    session = Session()
    session.entries["kill9_next"] = _tight_child("kill9_next", returncode=-9, kernel=kernel,
                                                 marker=FIRST_STEP)
    result = _judge(rule, session, tmp_path)
    part = _part(result, "2", "kill9_next")
    assert part["status"] == "NO VERDICT" and "cannot tell" in part["detail"]


def test_a_tight_child_that_exited_0_adds_nothing(rule, tmp_path):
    session = Session()
    session.entries["calibration_check_1"] = _tight_child("calibration_check_1", returncode=0,
                                                          kernel=None)
    result = _judge(rule, session, tmp_path)
    assert result["criteria"]["2"]["status"] == "PASS"
    assert len(result["criteria"]["2"]["parts"]) == 4

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


@pytest.mark.parametrize("where", ["after_train", "after_exit"])
def test_normal_cleanup_observed_file_fails_despite_the_other_missing_listing(
        rule, tmp_path, where):
    session = Session()
    if where == "after_train":
        session.arms["l2l_spill"]["spill_files_after_train"] = ["/spill/left.bin"]
        session.entries["l2l_spill"].pop("spill_files_after_exit")
    else:
        session.arms["l2l_spill"].pop("spill_files_after_train")
        session.entries["l2l_spill"]["spill_files_after_exit"] = ["/spill/left.bin"]
    assert _part(_judge(rule, session, tmp_path), "3", "normal")["status"] == "FAIL"


@pytest.mark.parametrize("where", ["after_train", "after_exit"])
@pytest.mark.parametrize("observation", [None, False, 0, "", {}])
def test_normal_cleanup_requires_actual_directory_lists(rule, tmp_path, where, observation):
    session = Session()
    if where == "after_train":
        session.arms["l2l_spill"]["spill_files_after_train"] = observation
    else:
        session.entries["l2l_spill"]["spill_files_after_exit"] = observation
    assert _part(_judge(rule, session, tmp_path), "3", "normal")["status"] == "NO VERDICT"


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


@pytest.mark.parametrize("damage, why", [
    (lambda s: s.entries["sigint"]["signal"].update(files_before_signal=[]),
     "no spill file before the signal"),
    (lambda s: s.entries["sigint"]["signal"].update(sent_unix=None, why="exited early"),
     "no signal sent (exited early)"),
    (lambda s: s.entries["sigint"]["signal"]["marker"].update(phase="enter"),
     "not sent at the hold"),
    (lambda s: s.cleanup["sigint"]["at_interrupt"].update(in_step=False),
     "did not come from the hold"),
    (lambda s: s.cleanup["sigint"]["at_interrupt"].update(steps_done=2),
     "did not come from the hold"),
    (lambda s: s.cleanup["sigint"]["at_interrupt"]["hold"].update(released="timeout"),
     "did not come from the hold"),
])
def test_a_sigint_that_missed_the_scenario_gives_no_verdict(rule, tmp_path, damage, why):
    session = Session()
    damage(session)
    result = _judge(rule, session, tmp_path)
    part = _part(result, "3", "sigint")
    assert part["status"] == "NO VERDICT" and why in part["detail"]
    assert result["criteria"]["3"]["status"] == "NO VERDICT"


@pytest.mark.parametrize("target, hold, why", [
    ({"ignored": True, "blocked": False}, {"released": "timeout"},
     "SIGINT was ignored in the child when it was sent (SigIgn)"),
    ({"ignored": False, "blocked": True}, {"released": "timeout"},
     "SIGINT was blocked in the child when it was sent (SigBlk)"),
    ({"ignored": False, "blocked": False}, {"released": "timeout", "waited_s": 120.0},
     "the hold ended by its timeout"),
])
def test_a_sigint_without_keyboardinterrupt_names_the_cause(rule, tmp_path, target, hold, why):
    """Session 2's case (exit normal after the signal) and its look-alikes: no verdict, and
    the detail tells an ignored or blocked signal from a hold that timed out."""
    session = Session()
    session.cleanup["sigint"].update(exit="normal", at_interrupt=None, hold=hold, steps=3)
    session.entries["sigint"]["signal"].update(returncode=0, target_at_send=target)
    part = _part(_judge(rule, session, tmp_path), "3", "sigint")
    assert part["status"] == "NO VERDICT" and why in part["detail"]


def test_kill9_away_from_the_hold_gives_no_verdict(rule, tmp_path):
    session = Session()
    session.entries["kill9"]["signal"]["marker"]["phase"] = "enter"
    part = _part(_judge(rule, session, tmp_path), "3", "kill9")
    assert part["status"] == "NO VERDICT" and "not sent at the hold" in part["detail"]


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


@pytest.mark.parametrize("where", ["after_train", "after_exit", "both"])
@pytest.mark.parametrize("missing", [None, "absent"])
def test_restart_cleanup_needs_explicit_directory_observations(rule, tmp_path, where, missing):
    session = Session()
    if where in ("after_train", "both"):
        if missing == "absent":
            session.cleanup["kill9_next"].pop("spill_files_after_train")
        else:
            session.cleanup["kill9_next"]["spill_files_after_train"] = None
    if where in ("after_exit", "both"):
        if missing == "absent":
            session.entries["kill9_next"].pop("spill_files_after_exit")
        else:
            session.entries["kill9_next"]["spill_files_after_exit"] = None
    result = _judge(rule, session, tmp_path)
    assert _part(result, "3", "kill9")["status"] == "NO VERDICT"
    assert result["criteria"]["3"]["status"] == "NO VERDICT"


@pytest.mark.parametrize("where", ["after_train", "after_exit"])
def test_a_real_restart_file_fails_even_if_the_other_listing_is_missing(rule, tmp_path, where):
    session = Session()
    if where == "after_train":
        session.cleanup["kill9_next"]["spill_files_after_train"] = [KILLED_FILE]
        session.entries["kill9_next"].pop("spill_files_after_exit")
    else:
        session.cleanup["kill9_next"].pop("spill_files_after_train")
        session.entries["kill9_next"]["spill_files_after_exit"] = [KILLED_FILE]
    assert _part(_judge(rule, session, tmp_path), "3", "kill9")["status"] == "FAIL"


@pytest.mark.parametrize("where", ["after_train", "after_exit"])
def test_any_restart_spill_file_left_fails_even_when_the_killed_file_is_gone(rule, tmp_path, where):
    session = Session()
    left = [f"{SPILL_DIR}/kill9/soup-l2l-spill-99-new.bin"]
    if where == "after_train":
        session.cleanup["kill9_next"]["spill_files_after_train"] = left
    else:
        session.entries["kill9_next"]["spill_files_after_exit"] = left
    assert _part(_judge(rule, session, tmp_path), "3", "kill9")["status"] == "FAIL"


@pytest.mark.parametrize("steps", [0, 1])
def test_a_restart_cannot_self_report_a_shorter_successful_fixture(rule, tmp_path, steps):
    session = Session()
    rec = session.cleanup["kill9_next"]
    rec.update(steps=steps, expected_steps=steps, grads=rec["grads"][:steps],
               logs=rec["logs"][:max(1, steps)], rates=rec["rates"][:steps])
    assert _part(_judge(rule, session, tmp_path), "3", "kill9")["status"] == "FAIL"


def test_restart_training_without_spill_participation_has_no_verdict(rule, tmp_path):
    session = Session()
    session.cleanup["kill9_next"]["store_io"]["bytes_read"] = 0
    assert _part(_judge(rule, session, tmp_path), "3", "kill9")["status"] == "NO VERDICT"


@pytest.mark.parametrize("field", ["spill_layers", "bytes_written", "bytes_read"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_restart_spill_participation_needs_positive_finite_measurements(
        rule, tmp_path, field, value):
    session = Session()
    rec = session.cleanup["kill9_next"]
    (rec["spill_plan"] if field == "spill_layers" else rec["store_io"])[field] = value
    assert _part(_judge(rule, session, tmp_path), "3", "kill9")["status"] == "NO VERDICT"


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


@pytest.mark.parametrize("steps", [0, 1, 6])
def test_a_truncated_cost_child_cannot_supply_a_block_value(rule, tmp_path, steps):
    session = Session()
    rec = session.cost[0]
    rec.update(steps=steps, expected_steps=steps, grads=[],
               logs=rec["logs"][:max(1, steps)], rates=rec["rates"][:steps],
               step_s=rec["step_s"][:steps], timed_step_s=rec["timed_step_s"][:max(1, steps - 2)])
    assert _judge(rule, session, tmp_path)["criteria"]["4"]["status"] == "NO VERDICT"


@pytest.mark.parametrize("damage", [
    lambda rec: rec.pop("timed_step_s"),
    lambda rec: rec.update(timed_step_s=[1.0] * 4),
    lambda rec: rec["timed_step_s"].__setitem__(0, 0.0),
    lambda rec: rec["timed_step_s"].__setitem__(0, float("nan")),
    lambda rec: rec["timed_step_s"].__setitem__(0, float("inf")),
    lambda rec: rec["timed_step_s"].__setitem__(0, True),
    lambda rec: rec.update(value=1.001),
    lambda rec: rec.update(value=0.0),
    lambda rec: rec.update(value=float("nan")),
    lambda rec: rec.update(logs=rec["logs"][:1]),
    lambda rec: rec.pop("rates"),
    lambda rec: rec["logs"][0].pop("learning_rate"),
    lambda rec: rec["logs"][0].update(learning_rate=float("inf")),
    lambda rec: rec["logs"][0].pop("grad_norm"),
    lambda rec: rec["logs"][0].update(grad_norm=0.0),
    lambda rec: rec["logs"][0].update(grad_norm=float("inf")),
    lambda rec: rec["logs"][0].update(grad_norm=True),
    lambda rec: rec.update(steps=6),
    lambda rec: rec.update(expected_steps=6),
    lambda rec: rec.pop("step_s"),
    lambda rec: rec.update(step_s=rec["step_s"][:6]),
    lambda rec: rec["step_s"].__setitem__(0, 0.0),
    lambda rec: rec["rates"].__setitem__(0, float("nan")),
    lambda rec: rec["logs"][0].update(loss=float("inf")),
    lambda rec: rec.update(step_s=[1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.1]),
], ids=["missing-timings", "four-timings", "zero-timing", "nan-timing", "infinite-timing",
        "boolean-timing", "not-median", "zero-value", "nan-value", "short-logs",
        "missing-rates", "missing-logged-rate", "infinite-logged-rate", "missing-grad-norm",
        "zero-grad-norm", "infinite-grad-norm", "boolean-grad-norm", "six-steps",
        "wrong-expected-steps", "missing-step-times", "short-step-times", "zero-warmup-time",
        "nan-rate", "infinite-loss", "unrelated-timed-suffix"])
def test_cost_requires_complete_timing_and_lightweight_training_evidence(rule, tmp_path, damage):
    session = Session()
    damage(session.cost[0])
    assert _judge(rule, session, tmp_path)["criteria"]["4"]["status"] == "NO VERDICT"


def test_cost_accepts_the_median_not_the_mean_of_five_timed_steps(rule, tmp_path):
    session = Session()
    rec = session.cost[0]
    rec.update(step_s=[3.0, 2.0, 0.8, 1.0, 0.9, 1.2, 1.5],
               timed_step_s=[0.8, 1.0, 0.9, 1.2, 1.5], value=1.0)
    assert rec["grads"] == []  # cost deliberately avoids full gradient copies
    assert _judge(rule, session, tmp_path)["criteria"]["4"]["status"] == "PASS"


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
