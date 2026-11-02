"""The G2 verdict and break-even: pure functions of a session directory."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "harness"


@pytest.fixture()
def rule():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("g2_rule", HARNESS / "g2_rule.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["g2_rule"] = module
    spec.loader.exec_module(module)
    return module


def _cmp(equal=True, zero=()):
    return {"tensors": 4, "unequal": [] if equal else ["x"], "max_abs": 0.0 if equal else 0.1,
            "worst": [], "all_zero": list(zero), "equal": equal}


def _pair(equal=True, forward=True):
    return {"backward": _cmp(equal), "forward_equal": forward}


def _bytes(spill, *, weight=0, written=None):
    capacity = 8192
    planned = 3 * 2 * capacity if spill else 0
    return {"weight_bytes_stored": weight, "record_capacity_bytes": capacity,
            "store_chunk_bytes": capacity, "records": 6, "payload_bytes_per_step": 6 * 8000,
            "planned_spill_bytes": planned,
            "spill_written": planned if written is None else written, "spill_read": planned,
            "spill_direct": True if spill else None, "left_behind": []}


def _correctness(k, device="cuda", **over):
    row = {"kind": "correctness", "k": k, "deterministic": False, "ga_vs_ga": _pair(),
           "l2l_vs_ga": _pair(), "g2_ram_vs_ga": _pair(), "g2_ram_vs_l2l": _pair(),
           "g2_spill_vs_ga": _pair(), "g2_spill_vs_l2l": _pair(),
           "g2_ram_bytes": _bytes(False), "g2_spill_bytes": _bytes(True),
           "controls": {"naive_identity": {"raised": None, **_pair(False)},
                        "corrupt": {"record": [1, 0], **_pair(False, True)}}}
    row.update(over)
    return {"meta": {"device": device, "synthetic": device != "cuda"}, "records": [row]}


def _placement(spill_disk="nvme0n1", weight_disk="nvme0n1"):
    return {"spill": {"disk": {"id": spill_disk, "problem": None if spill_disk else "none"}},
            "weights": {"disk": {"id": weight_disk, "problem": None if weight_disk else "none"}}}


def _arm(comparison, block, position, arm, step_s, *, outcome="ok", planned=0,
         placement=None):
    tier = "ram" if comparison == "ram_store_vs_recompute" else "disk"
    store = "ram" if tier == "ram" else "spill"
    identity = {"comparison": comparison, "arm": arm, "block": block, "position": position,
                "tier": tier, "store": store, "k": 1}
    checks = {name: {"outcome": "ok"} for name in ("suspend", "power", "foreign_reads")}
    checks["power"]["outcome"] = outcome
    return {"meta": {"device": "cuda", "synthetic": False, "mode": "arm", "direct_io": True,
                     "steps": 1, **identity},
            "records": [{"kind": "arm", **identity, "variant_role": arm,
                         "variant": "recompute" if arm == "A" else "store", "ids_sha256": "abc",
                         "steps": [{"step_s": step_s, "losses": [1.0],
                                    "spill_written": planned, "spill_read": planned}],
                         "losses": [1.0], "planned_spill_bytes_per_step": planned,
                         "spill_direct": True if planned else None,
                         "placement": placement or _placement(),
                         "F1": {"outcome": outcome, "required": list(checks), "checks": checks,
                                "arm": arm, "round": block, "run": position},
                         "outcome": outcome}]}


def _session(tmp_path, *, device="cuda", correctness=None, arms=None):
    import g2_rule

    (tmp_path / "protocol.json").write_text(json.dumps({"rule": g2_rule.RULE}))
    for k in (1, 2, 4):
        payload = (correctness or {}).get(k, _correctness(k, device))
        (tmp_path / f"correctness_k{k}.json").write_text(json.dumps(payload))
    for (name, block, position, arm), payload in (arms or {}).items():
        path = tmp_path / "arms" / name / f"b{block:02d}_p{position}_{arm}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return tmp_path


def _blocks(name, n, a, b, *, planned=0, placement=None):
    arms = {}
    for block in range(n):
        jitter = 0.001 * block
        for position, arm in enumerate("ABBA"):
            value = (a if arm == "A" else b) + jitter
            arms[(name, block, position, arm)] = _arm(name, block, position, arm, value,
                                                      planned=planned if arm == "B" else 0,
                                                      placement=placement)
    return arms


# --------------------------------------------------------------------------
# G1 / G2
# --------------------------------------------------------------------------
def test_exact_stores_with_working_controls_pass_g1_and_g2(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path))
    assert out["gates"]["G1"]["status"] == "pass"
    assert out["gates"]["G2"]["status"] == "pass"


def test_one_unequal_gradient_fails_g1(rule, tmp_path):
    bad = _correctness(2, g2_spill_vs_ga=_pair(False))
    out = rule.evaluate(_session(tmp_path, correctness={2: bad}))
    assert out["gates"]["G1"]["status"] == "fail"
    assert "g2_spill vs ga" in out["gates"]["G1"]["detail"]


@pytest.mark.parametrize("field,value", [("tensors", 0), ("tensors", None),
                                       ("all_zero", None), ("equal", None)])
def test_vacuous_or_incomplete_gradient_comparisons_cannot_pass_g1(
        rule, tmp_path, field, value):
    bad = _correctness(1)
    comparison = bad["records"][0]["g2_ram_vs_ga"]["backward"]
    if value is None:
        comparison.pop(field)
    else:
        comparison[field] = value
    out = rule.evaluate(_session(tmp_path, correctness={1: bad}))
    assert out["gates"]["G1"]["status"] != "pass"
    assert not out["verdict"].startswith("BUILD")


def test_a_control_that_cannot_fail_removes_g1(rule, tmp_path):
    """If identity hooks give the right answer, this fixture does not show the
    weight-slot hazard, and passing it would prove nothing."""
    blind = _correctness(1, controls={"naive_identity": {"raised": None, **_pair(True)},
                                      "corrupt": {"record": [1, 0], **_pair(False, True)}})
    out = rule.evaluate(_session(tmp_path, correctness={1: blind}))
    assert out["gates"]["G1"]["status"] == "no verdict"


def test_an_identity_control_that_raises_counts_as_working(rule, tmp_path):
    raised = _correctness(1, controls={"naive_identity": {"raised": "modified by an inplace"},
                                       "corrupt": {"record": [1, 0], **_pair(False, True)}})
    out = rule.evaluate(_session(tmp_path, correctness={1: raised}))
    assert out["gates"]["G1"]["status"] == "pass"


def test_a_corruption_that_changes_the_forward_fails_g1(rule, tmp_path):
    leaky = _correctness(4, controls={"naive_identity": {"raised": "x"},
                                      "corrupt": {"record": [1, 0], **_pair(False, False)}})
    out = rule.evaluate(_session(tmp_path, correctness={4: leaky}))
    assert out["gates"]["G1"]["status"] == "fail"


def test_non_reproducible_ga_without_a_deterministic_rerun_has_no_verdict(rule, tmp_path):
    noisy = _correctness(2, ga_vs_ga=_pair(False))
    out = rule.evaluate(_session(tmp_path, correctness={2: noisy}))
    assert out["gates"]["G1"]["status"] == "no verdict"
    det = _correctness(2)
    det["records"][0]["deterministic"] = True
    (tmp_path / "correctness_k2_det.json").write_text(json.dumps(det))
    assert rule.evaluate(tmp_path)["gates"]["G1"]["status"] == "pass"


def test_g1_and_g2_read_the_same_deterministic_record(rule, tmp_path):
    """The audit's reproduction: the deterministic rerun has spill_read = 0, the default
    run's bytes are right. Before, G1 passed on the rerun while G2 passed on the default
    run's bytes; both must judge the one record G1 selected."""
    noisy = _correctness(2, ga_vs_ga=_pair(False))
    session = _session(tmp_path, correctness={2: noisy})
    det = _correctness(2, g2_spill_bytes={**_bytes(True), "spill_read": 0})
    det["records"][0]["deterministic"] = True
    (session / "correctness_k2_det.json").write_text(json.dumps(det))
    out = rule.evaluate(session)
    assert out["gates"]["G1"]["parts"][2]["source"] == "correctness_k2_det.json"
    assert out["gates"]["G2"]["status"] == "fail"
    assert "correctness_k2_det.json" in out["gates"]["G2"]["detail"]


def test_a_rerun_that_is_not_deterministic_cannot_replace_the_default(rule, tmp_path):
    noisy = _correctness(2, ga_vs_ga=_pair(False))
    session = _session(tmp_path, correctness={2: noisy})
    (session / "correctness_k2_det.json").write_text(json.dumps(_correctness(2)))
    out = rule.evaluate(session)
    assert out["gates"]["G1"]["status"] == "no verdict"
    assert out["gates"]["G2"]["status"] == "no verdict"


@pytest.mark.parametrize("placement,why", [
    (_placement("nvme0n1", "nvme1n1"), "nvme0n1 and nvme1n1"),
    (_placement(None, "nvme0n1"), "spill disk unknown"),
])
def test_disk_arms_need_the_os_to_show_one_physical_disk(rule, tmp_path, placement, why):
    """Two direct-I/O flags are not a same-drive proof."""
    arms = _blocks(rule.DISK_COMPARISON, 5, a=10.0, b=8.0, planned=8192, placement=placement)
    g4 = rule.evaluate(_session(tmp_path, arms=arms))["gates"]["G4"]
    assert g4["status"] == "no verdict" and "V10" in g4["detail"] and why in g4["detail"]


def test_weight_bytes_in_a_record_fail_g1_and_g2(rule, tmp_path):
    heavy = _correctness(1, g2_ram_bytes=_bytes(False, weight=4096))
    out = rule.evaluate(_session(tmp_path, correctness={1: heavy}))
    assert out["gates"]["G1"]["status"] == "fail"
    assert out["gates"]["G2"]["status"] == "fail"


def test_spill_bytes_off_plan_fail_g2(rule, tmp_path):
    short = _correctness(2, g2_spill_bytes=_bytes(True, written=4096))
    out = rule.evaluate(_session(tmp_path, correctness={2: short}))
    assert out["gates"]["G2"]["status"] == "fail"


# --------------------------------------------------------------------------
# G3 / G4 and the verdict
# --------------------------------------------------------------------------
def test_a_clear_gain_on_the_ram_tier_passes_g3(rule, tmp_path):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    out = rule.evaluate(_session(tmp_path, arms=arms))
    g3 = out["gates"]["G3"]
    assert g3["status"] == "pass", g3
    assert g3["interval"]["high"] < 1 - rule.MIN_GAIN
    assert out["verdict"].startswith("BUILD")


def test_no_gain_fails_g3(rule, tmp_path):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=10.0)
    assert rule.evaluate(_session(tmp_path, arms=arms))["gates"]["G3"]["status"] == "fail"


def test_a_gain_just_under_the_bar_is_not_a_pass(rule, tmp_path):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=9.6)  # 4% < MIN_GAIN
    assert rule.evaluate(_session(tmp_path, arms=arms))["gates"]["G3"]["status"] != "pass"


def test_speed_is_not_read_when_g1_fails(rule, tmp_path):
    bad = _correctness(1, g2_ram_vs_ga=_pair(False))
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    out = rule.evaluate(_session(tmp_path, correctness={1: bad}, arms=arms))
    assert out["gates"]["G3"]["status"] == "no verdict"
    assert out["verdict"].startswith("DO NOT BUILD")


def test_a_void_block_is_replaced_not_fatal(rule, tmp_path):
    arms = _blocks(rule.RAM_COMPARISON, 6, a=10.0, b=8.0)
    arms[(rule.RAM_COMPARISON, 0, 1, "B")] = _arm(rule.RAM_COMPARISON, 0, 1, "B", 8.0,
                                                  outcome="void")
    g3 = rule.evaluate(_session(tmp_path, arms=arms))["gates"]["G3"]
    assert g3["status"] == "pass" and len(g3["ratios"]) == 5


def test_other_token_rows_in_a_block_remove_the_gate(rule, tmp_path):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    arms[(rule.RAM_COMPARISON, 2, 1, "B")]["records"][0]["ids_sha256"] = "other"
    assert rule.evaluate(_session(tmp_path, arms=arms))["gates"]["G3"]["status"] == "no verdict"


def test_disk_arms_need_the_planned_spill_and_a_direct_weight_source(rule, tmp_path):
    arms = _blocks(rule.DISK_COMPARISON, 5, a=10.0, b=8.0, planned=8192)
    assert rule.evaluate(_session(tmp_path, arms=arms))["gates"]["G4"]["status"] == "pass"
    arms[(rule.DISK_COMPARISON, 1, 2, "B")]["records"][0]["steps"][0]["spill_read"] = 0
    assert rule.evaluate(_session(tmp_path, arms=arms))["gates"]["G4"]["status"] == "no verdict"


def test_a_virtual_disk_labels_g4_without_changing_its_status(rule, tmp_path):
    virtio = {"id": "vda", "virtual": True, "virtual_reason": "vendor 0x1af4 (virtio)"}
    placement = {"spill": {"disk": virtio}, "weights": {"disk": virtio}}
    arms = _blocks(rule.DISK_COMPARISON, 5, a=10.0, b=8.0, planned=8192, placement=placement)
    g4 = rule.evaluate(_session(tmp_path, arms=arms))["gates"]["G4"]
    assert g4["status"] == "pass"
    assert g4["disk_scope"]["virtual"] is True and "VIRTUAL disk" in g4["detail"]
    physical = _blocks(rule.DISK_COMPARISON, 5, a=10.0, b=8.0, planned=8192)
    (tmp_path / "p").mkdir()
    plain = rule.evaluate(_session(tmp_path / "p", arms=physical))["gates"]["G4"]
    assert plain["disk_scope"]["virtual"] is None and "VIRTUAL" not in plain["detail"]


def test_unknown_f1_cannot_hide_in_a_void_block(rule, tmp_path):
    arms = _blocks(rule.RAM_COMPARISON, 6, a=10.0, b=8.0)
    for position, role, outcome in ((0, "A", "unknown"), (1, "B", "void")):
        arms[(rule.RAM_COMPARISON, 0, position, role)] = _arm(
            rule.RAM_COMPARISON, 0, position, role, 10.0 if role == "A" else 8.0,
            outcome=outcome)
    out = rule.evaluate(_session(tmp_path, arms=arms))
    assert out["gates"]["G3"]["status"] == "no verdict"
    assert "F1" in out["gates"]["G3"]["detail"]
    assert not out["verdict"].startswith("BUILD")


@pytest.mark.parametrize("bad_time", [None, 0.0, -1.0, float("inf"), float("nan"), True])
def test_every_timed_step_needs_positive_finite_seconds(rule, tmp_path, bad_time):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    payload = arms[(rule.RAM_COMPARISON, 0, 1, "B")]
    row = payload["records"][0]
    # Two good steps keep the old median at 8 even when it ignores or sorts a bad step.
    row["steps"].extend([{**row["steps"][0]}, {**row["steps"][0], "step_s": bad_time}])
    row["losses"] = [1.0, 1.0, 1.0]
    payload["meta"]["steps"] = 3
    out = rule.evaluate(_session(tmp_path, arms=arms))
    assert out["gates"]["G3"]["status"] == "no verdict"
    assert "timed step" in out["gates"]["G3"]["detail"]
    assert not out["verdict"].startswith("BUILD")


@pytest.mark.parametrize("losses", [None, [], [None], [float("nan")], [float("inf")]])
def test_aggregate_losses_cannot_hide_an_invalid_timed_step(rule, tmp_path, losses):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    arms[(rule.RAM_COMPARISON, 0, 1, "B")]["records"][0]["steps"][0]["losses"] = losses
    out = rule.evaluate(_session(tmp_path, arms=arms))
    assert out["gates"]["G3"]["status"] == "no verdict"
    assert "V5" in out["gates"]["G3"]["detail"]
    assert not out["verdict"].startswith("BUILD")


@pytest.mark.parametrize("outcome", [None, "", "pass", "candidate"])
def test_unrecognized_f1_outcome_cannot_pass_a_speed_gate(rule, tmp_path, outcome):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    arms[(rule.RAM_COMPARISON, 0, 1, "B")]["records"][0]["F1"]["outcome"] = outcome
    out = rule.evaluate(_session(tmp_path, arms=arms))
    assert out["gates"]["G3"]["status"] == "no verdict"
    assert "F1" in out["gates"]["G3"]["detail"]
    assert not out["verdict"].startswith("BUILD")


@pytest.mark.parametrize("problem", ["missing F1", "missing check", "unknown", "invalid", "void",
                                    "missing required", "wrong summary"])
def test_f1_summary_cannot_override_required_checks(rule, tmp_path, problem):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    row = arms[(rule.RAM_COMPARISON, 0, 1, "B")]["records"][0]
    if problem == "missing F1":
        row.pop("F1")
    elif problem == "missing check":
        row["F1"]["checks"].pop("power")
    elif problem == "missing required":
        row["F1"]["required"] = []
    elif problem == "wrong summary":
        row["outcome"] = "void"
    else:
        row["F1"]["checks"]["power"]["outcome"] = problem
    out = rule.evaluate(_session(tmp_path, arms=arms))
    assert out["gates"]["G3"]["status"] == "no verdict"
    assert "F1" in out["gates"]["G3"]["detail"]
    assert not out["verdict"].startswith("BUILD")


@pytest.mark.parametrize("part,field,value", [
    ("row", "variant", "recompute"), ("row", "variant", None),
    ("row", "arm", "A"), ("row", "comparison", "disk_spill_vs_recompute"),
    ("row", "tier", "disk"), ("row", "store", "spill"),
    ("row", "block", 7), ("row", "position", 0),
    ("meta", "arm", "A"), ("meta", "comparison", "disk_spill_vs_recompute"),
    ("meta", "tier", "disk"), ("meta", "store", "spill"),
    ("meta", "block", 7), ("meta", "position", 0), ("meta", "k", 2),
])
def test_speed_gate_binds_records_and_metadata_to_the_named_arm(
        rule, tmp_path, part, field, value):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    payload = arms[(rule.RAM_COMPARISON, 0, 1, "B")]
    target = payload["records"][0] if part == "row" else payload["meta"]
    target[field] = value
    out = rule.evaluate(_session(tmp_path, arms=arms))
    assert out["gates"]["G3"]["status"] == "no verdict"
    assert not out["verdict"].startswith("BUILD")


@pytest.mark.parametrize("comparison,role,planned", [
    ("ram_store_vs_recompute", "A", 8192), ("ram_store_vs_recompute", "B", 8192),
    ("disk_spill_vs_recompute", "A", 8192), ("disk_spill_vs_recompute", "B", 0),
    ("disk_spill_vs_recompute", "B", -8192),
])
def test_spill_plan_must_match_the_arm_and_tier(rule, tmp_path, comparison, role, planned):
    arms = _blocks(comparison, 5, a=10.0, b=8.0,
                   planned=8192 if comparison == rule.DISK_COMPARISON else 0)
    position = 0 if role == "A" else 1
    row = arms[(comparison, 0, position, role)]["records"][0]
    row.update(planned_spill_bytes_per_step=planned, spill_direct=True)
    row["steps"][0].update(spill_written=planned, spill_read=planned)
    out = rule.evaluate(_session(tmp_path, arms=arms))
    gate = out["gates"]["G4" if comparison == rule.DISK_COMPARISON else "G3"]
    assert gate["status"] == "no verdict"
    assert "V6" in gate["detail"]
    assert not out["verdict"].startswith("BUILD")


def test_duplicate_position_cannot_overwrite_a_read_arm(rule, tmp_path):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    session = _session(tmp_path, arms=arms)
    payload = arms[(rule.RAM_COMPARISON, 0, 1, "B")]
    (session / "arms" / rule.RAM_COMPARISON / "b00_p1_A.json").write_text(json.dumps(payload))
    gate = rule.gate_speed(session, rule.RAM_COMPARISON, tier="ram")
    assert gate["status"] == "no verdict"
    assert "filename" in gate["detail"] or "multiple" in gate["detail"]


@pytest.mark.parametrize("part", ["correctness", "arm"])
def test_every_read_payload_needs_its_own_device_metadata(rule, tmp_path, part):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    correctness = {1: _correctness(1)}
    payload = correctness[1] if part == "correctness" else arms[
        (rule.RAM_COMPARISON, 0, 1, "B")]
    payload.pop("meta")
    out = rule.evaluate(_session(tmp_path, arms=arms, correctness=correctness))
    assert out["validity"]["V1"]["ok"] is False
    assert all(gate["status"] == "no verdict" for gate in out["gates"].values())
    assert not out["verdict"].startswith("BUILD")


def test_protocol_metadata_alone_cannot_certify_a_cuda_session(rule, tmp_path):
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    correctness = {k: _correctness(k) for k in (1, 2, 4)}
    for payload in [*correctness.values(), *arms.values()]:
        payload.pop("meta")
    session = _session(tmp_path, arms=arms, correctness=correctness)
    (session / "protocol.json").write_text(json.dumps(
        {"rule": rule.RULE, "meta": {"device": "cuda", "synthetic": False}}))
    out = rule.evaluate(session)
    assert out["validity"]["V1"]["ok"] is False
    assert not out["verdict"].startswith("BUILD")


@pytest.mark.parametrize("problem", ["missing", "v1", "changed threshold"])
def test_protocol_mismatch_is_not_silently_evaluated_as_rule_v2(rule, tmp_path, problem):
    session = _session(tmp_path, arms=_blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0))
    if problem == "missing":
        (session / "protocol.json").unlink()
    else:
        recorded = dict(rule.RULE)
        recorded["profile" if problem == "v1" else "G3"] = (
            "g2-recompute-vs-spill-v1" if problem == "v1" else "pass below 1.00")
        (session / "protocol.json").write_text(json.dumps({"rule": recorded}))
    out = rule.evaluate(session)
    assert out["verdict"] == "NO VERDICT: PROTOCOL MISMATCH"
    assert not out["validity"]["protocol"]["ok"]
    assert all(gate["status"] == "no verdict" for gate in out["gates"].values())
    assert out["measured_gates"] == {}
    if problem == "v1":
        assert "g2-recompute-vs-spill-v1" in out["validity"]["protocol"]["detail"]
    elif problem == "changed threshold":
        assert "G3" in out["validity"]["protocol"]["detail"]


def test_published_session2_retains_the_recorded_rule_v2_result(rule):
    session = ROOT / "benchmarks" / "results" / "g2-gpu-20261031" / "session2-rtx3090"
    out = rule.evaluate(session)
    assert out["validity"]["protocol"]["ok"] and out["validity"]["V1"]["ok"]
    assert [out["gates"][gate]["status"] for gate in ("G1", "G2", "G3", "G4")] == [
        "pass", "pass", "fail", "fail"]
    assert out["verdict"] == "DO NOT BUILD: recompute is as fast or faster on both tiers"


def test_published_session1_requires_the_historical_evaluator(rule):
    session = ROOT / "benchmarks" / "results" / "g2-gpu-20261031" / "session1-rtx4090"
    recorded = json.loads((session / "verdict.json").read_text(encoding="utf-8"))
    out = rule.evaluate(session)
    assert recorded["verdict"] == "NO VERDICT"
    assert out["verdict"] == "NO VERDICT: PROTOCOL MISMATCH"
    assert out["measured_gates"] == {}
    assert "g2-recompute-vs-spill-v1" in out["validity"]["protocol"]["detail"]


def _runs(rule, a_values, b_values):
    import arm_validity

    return [[arm_validity.Run(a, "ok"), arm_validity.Run(b, "ok"), arm_validity.Run(b, "ok"),
             arm_validity.Run(a, "ok")] for a, b in zip(a_values, b_values)]


def test_v7_v2_judges_each_arms_noise_against_its_own_mean(rule):
    """Session 1 (2026-10-31): B five times A's length with 25% relative noise asked F1's
    seconds-pooled rule for 2,041 blocks; each arm against its own mean asks for ~50."""
    import arm_validity

    a = [1.17, 1.16, 1.18, 1.17, 1.16]
    b = [5.0, 5.0, 5.0, 10.0, 6.5]
    blocks = _runs(rule, a, b)
    v2 = rule.blocks_needed(blocks)
    assert arm_validity.blocks_needed(blocks) > 20 * v2
    assert 30 < v2 < 120
    # Equal lengths: the v2 count is F1's to within the A/B pooling of the means.
    same = _runs(rule, [5.0, 5.2, 4.9, 5.1, 5.0], [5.1, 4.9, 5.0, 5.2, 5.0])
    assert abs(rule.blocks_needed(same) - arm_validity.blocks_needed(same)) <= 1


def test_v7_v2_keeps_f1s_pilot_and_single_top_up(rule):
    quiet = _runs(rule, [1.0] * 4, [2.0] * 4)
    assert rule.blocks_to_add(quiet) == 1  # the pilot is 5 blocks
    noisy = _runs(rule, [1.0, 1.1, 0.9, 1.0, 1.05], [2.0, 4.0, 2.5, 3.0, 2.2])
    first_n = rule.blocks_needed(noisy)
    assert rule.blocks_to_add(noisy) == first_n - 5
    calm = noisy + _runs(rule, [1.0] * (first_n - 5), [2.5] * (first_n - 5))
    assert rule.blocks_to_add(calm) == max(0, rule.blocks_needed(calm[:first_n]) - first_n)


def test_a_cpu_session_never_gets_a_verdict_but_keeps_the_measured_gates(rule, tmp_path):
    correctness = {k: _correctness(k, "cpu") for k in (1, 2, 4)}
    arms = _blocks(rule.RAM_COMPARISON, 5, a=10.0, b=8.0)
    out = rule.evaluate(_session(tmp_path, correctness=correctness, arms=arms))
    assert out["verdict"] == "NO VERDICT: CPU/SYNTHETIC PILOT"
    assert all(gate["status"] == "no verdict" for gate in out["gates"].values())
    assert out["measured_gates"]["G1"]["status"] == "pass"


def test_rule_text_carries_the_bar(rule):
    assert f"{1 - rule.MIN_GAIN:.2f}" in rule.RULE["G3"]


# --------------------------------------------------------------------------
# break-even
# --------------------------------------------------------------------------
def test_break_even_arithmetic(rule):
    out = rule.break_even(saved_s=2.0, spilled_bytes=1_000_000_000, write_Bps=2e9,
                          read_Bps=4e9, overhead_s=0.25)
    assert out["write_s"] == pytest.approx(0.5)
    assert out["read_s"] == pytest.approx(0.25)
    assert out["serial_margin_s"] == pytest.approx(2.0 - 0.5 - 0.25 - 0.25)
    # Symmetric rate at which the serial margin is zero: 2 GB / (2.0 - 0.25) s.
    assert out["breakeven_Bps"] == pytest.approx(2e9 / 1.75)
    at = rule.break_even(saved_s=2.0, spilled_bytes=1_000_000_000,
                         write_Bps=out["breakeven_Bps"], read_Bps=out["breakeven_Bps"],
                         overhead_s=0.25)
    assert at["serial_margin_s"] == pytest.approx(0.0, abs=1e-9)


def test_contention_is_charged_as_slower_weight_reads(rule):
    out = rule.break_even(saved_s=2.0, spilled_bytes=1_000_000_000, write_Bps=4e9,
                          read_Bps=4e9, weight_bytes=8_000_000_000, weight_ref_Bps=8e9,
                          weight_with_spill_Bps=4e9)
    assert out["contention_s"] == pytest.approx(1.0)
    assert out["serial_margin_s"] == pytest.approx(2.0 - 0.25 - 0.25 - 1.0)


def test_overlap_only_charges_io_that_compute_cannot_hide(rule):
    out = rule.break_even(saved_s=1.0, spilled_bytes=1_000_000_000, write_Bps=1e9,
                          read_Bps=1e9, forward_compute_s=0.75, backward_compute_s=2.0)
    assert out["overlapped_cost_s"] == pytest.approx(0.25)
    assert out["overlapped_margin_s"] == pytest.approx(0.75)
    assert out["serial_margin_s"] == pytest.approx(-1.0)


def test_a_missing_rate_gives_no_margin_rather_than_a_guess(rule):
    out = rule.break_even(saved_s=1.0, spilled_bytes=10, write_Bps=None, read_Bps=1e9)
    assert out["serial_margin_s"] is None and out["breakeven_Bps"] is None
    shared = rule.break_even(saved_s=1.0, spilled_bytes=10, write_Bps=1e9, read_Bps=1e9,
                             weight_bytes=100)
    assert shared["contention_s"] is None and shared["serial_margin_s"] is None


def _profile(spilled, weight_bytes, *, waits=True, machine="m-box", spill="nvme0n1",
             weights="nvme0n1"):
    row = {
        "kind": "profile", "recompute_s_per_step": 2.0, "store_overhead_s_per_step": 0.1,
        "record_capacity_bytes": 4096, "records_per_step": 10,
        "weight_bytes_per_step": weight_bytes, "bytes": {"planned_spill_bytes": spilled},
        "forward_compute_s_per_step": 1.0, "backward_compute_s_per_step": 2.0,
        "host": {"machine_id": machine},
        "placement": {"spill": {"disk": {"id": spill}}, "weights": {"disk": {"id": weights}}}}
    if waits:
        row["store_io_wait_s_per_step"] = {"put": 0.4, "get": 0.3}
    return {"meta": {"device": "cuda", "synthetic": False}, "records": [row]}


def _sustained_arm(mode, *, machine="m-box", spill="nvme0n1", weights="nvme0n1"):
    return {"mode": mode, "status": "candidate", "direct": True, "measured_s": 3600,
            "fixture_exceeds_ram": True, "source": f"{mode}/arm.json",
            "host": {"machine_id": machine}, "spill_disk": spill, "weight_disk": weights}


def test_each_tier_is_costed_from_its_own_profile(rule, tmp_path):
    session = _session(tmp_path)
    (session / "profile_ram.json").write_text(json.dumps(_profile(0, 0)))
    (session / "profile_disk.json").write_text(json.dumps(_profile(40960, 8_000_000_000)))
    (session / "disk").mkdir()
    rates = {"write_Bps": 1e9, "read_Bps": 2e9, "weight_ref_Bps": 4e9,
             "weight_with_spill_Bps": 2e9,
             "arms": [{"mode": "cycle", "status": "smoke_passed", "direct": True,
                       "measured_s": 20, "fixture_exceeds_ram": False}]}
    (session / "disk" / "rates.json").write_text(json.dumps(rates))
    out = rule.analyse_breakeven(session)
    assert out["status"] == "estimate" and not out["V9"]["ok"]
    ram = out["cases"]["ram_tier_records_in_ram"]
    disk = out["cases"]["disk_tier_records_spilled"]
    assert ram["spilled_bytes"] == 0 and ram["contention_s"] == 0.0
    assert ram["serial_margin_s"] == pytest.approx(2.0 - 0.1)
    assert disk["contention_s"] == pytest.approx(2.0)  # 8 GB at 2 GB/s instead of 4 GB/s
    assert disk["serial_margin_s"] < 0
    rates["arms"] = [_sustained_arm("cycle+weights"), _sustained_arm("weights")]
    (session / "disk" / "rates.json").write_text(json.dumps(rates))
    out = rule.analyse_breakeven(session)
    assert out["V11"]["ok"], out["V11"]
    assert out["status"] == "measured"


@pytest.mark.parametrize("arm,profile,why", [
    (dict(machine="m-other"), {}, "another machine"),
    (dict(machine=None), {}, "no machine id"),
    (dict(spill="nvme1n1"), {}, "spill disk nvme1n1"),
    (dict(weights=None), {}, "weight disk unknown"),
    ({}, dict(machine=None), "the profile names no machine id"),
    ({}, dict(spill=None), "no spill/weight disk"),
])
def test_rates_from_another_or_unknown_host_or_disk_are_an_estimate(rule, tmp_path, arm,
                                                                    profile, why):
    """Audit finding 5: rates.json from any disk session counted as measured, even one
    made on another box or another drive than the model profile."""
    session = _session(tmp_path)
    (session / "profile_disk.json").write_text(json.dumps(_profile(40960, 8_000_000_000,
                                                                   **profile)))
    (session / "disk").mkdir()
    rates = {"write_Bps": 1e9, "read_Bps": 2e9, "weight_ref_Bps": 4e9,
             "weight_with_spill_Bps": 2e9,
             "arms": [_sustained_arm("cycle+weights", **arm), _sustained_arm("weights")]}
    (session / "disk" / "rates.json").write_text(json.dumps(rates))
    out = rule.analyse_breakeven(session)
    assert out["V9"]["ok"] and not out["V11"]["ok"]
    assert why in out["V11"]["reason"]
    assert out["status"] == "estimate"


def test_the_cpu_shape_estimate_does_not_void_a_gpu_session(rule, tmp_path):
    session = _session(tmp_path)
    (session / "shapes.json").write_text(json.dumps({"meta": {"device": "cpu",
                                                              "synthetic": True}}))
    assert rule.evaluate(session)["validity"]["V1"]["ok"] is True


def test_a_profile_without_separated_waits_gets_no_disk_margin(rule, tmp_path):
    """The old profile's overhead held the spill waits, and break_even charged the drive
    again from its rates. Such a profile must not produce a margin."""
    session = _session(tmp_path)
    (session / "profile_disk.json").write_text(json.dumps(_profile(40960, 0, waits=False)))
    case = rule.analyse_breakeven(session)["cases"]["disk_tier_records_spilled"]
    assert case["serial_margin_s"] is None and "waits" in case["note"]


# --------------------------------------------------------------------------
# the session driver: placement preflight, and no timed arm after a failed G1
# --------------------------------------------------------------------------
def _session_module():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("g2_session", HARNESS / "g2_session.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path, *extra):
    return ["--weights", "w", "--shards", str(tmp_path / "shards"), "--spill-dir",
            str(tmp_path / "spill"), "--out", str(tmp_path / "session"), "--device", "cpu",
            "--allow-buffered", *extra]


@pytest.mark.parametrize("bad_k", [None, 4])
def test_the_session_runs_no_timed_arm_after_a_failed_correctness_check(
        rule, tmp_path, monkeypatch, bad_k):
    session = _session_module()
    ran = []

    def fake_child(entry, out, env, deadline):
        ran.append(entry["name"])
        if entry["name"].startswith("correctness_k") and not entry["name"].endswith("_det"):
            k = int(entry["name"][len("correctness_k"):])
            payload = _correctness(k, g2_ram_vs_ga=_pair(k != bad_k))
            Path(entry["output"]).write_text(json.dumps(payload))
        return 0

    monkeypatch.setattr(session, "run_child", fake_child)
    monkeypatch.setattr(session.g2_rule, "blocks_to_add", lambda runs: 0)
    monkeypatch.setattr(session, "placement_check", lambda args, names: {"refusals": []})
    code = session.main(_argv(tmp_path, "--comparisons", rule.RAM_COMPARISON, "--execute"))
    status = json.loads((tmp_path / "session" / "session.json").read_text())
    if bad_k is None:
        assert code == 0 and status["timing_allowed"]["ok"] is True
    else:
        assert code == 3 and status["timing_allowed"]["ok"] is False
        assert "timing arms not run" in status["reason"]
        assert not any(name.startswith(rule.RAM_COMPARISON) for name in ran)


@pytest.mark.parametrize("disks,refused", [(("d0", "d0"), False), (("d0", "d1"), True),
                                           (("d0", None), True)])
def test_the_session_refuses_a_disk_comparison_the_os_does_not_place_on_one_disk(
        tmp_path, monkeypatch, disks, refused):
    session = _session_module()
    import g2_disk

    where = {str(tmp_path / "shards"): disks[0], str(tmp_path / "spill"): disks[1]}

    def volume(path):
        disk = where.get(path)
        return {"path": path, "disk": {"id": disk, "problem": None if disk else "no device"}}

    (tmp_path / "shards").mkdir()
    (tmp_path / "spill").mkdir()
    monkeypatch.setattr(g2_disk, "volume_info", volume)
    args = session.args_parser().parse_args(_argv(tmp_path))
    record = session.placement_check(args, list(session.COMPARISONS))
    assert bool(record["refusals"]) is refused, record
    assert session.placement_check(args, ["ram_store_vs_recompute"])["refusals"] == []
