"""CPU tests for the A4 verdict (benchmarks/harness/a4_rule.py) over synthetic raw JSON.

Each test writes the files one session leaves and checks one branch of the
rule fixed before the GPU run: pass, fail and no verdict per gate, the validity rows that
take a gate away, and that a CPU pilot never gets a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_l2l_probe_harness import _load

CHUNK = 4 << 20
N_LAYERS = 32


@pytest.fixture()
def rule():
    return _load("a4_rule")


def _cmp(equal=True, zero=()):
    return {"tensors": 4, "unequal": [] if equal else ["x"], "max_abs": 0.0 if equal else 1e-3,
            "worst": [], "all_zero": list(zero), "equal": equal}


def _pair(forward=True, backward=True, zero=()):
    return {"backward": _cmp(backward, zero), "forward_equal": forward}


def _plan_dict(budget, k=4):
    spill = {"1": 0, "0.5": 16, "0.25": 24, "min": 32}[budget]
    # All-pinned holds every chunk (32 layers x k x 4 MiB = 512 MiB at k = 4); a spill arm
    # holds its resident chunks plus two rings of 4 x (chunk + 4096) rounded to 8 MiB.
    held = (N_LAYERS - spill) * k * CHUNK + (2 * 4 * (8 << 20) if spill else 0)
    cap = max(held, 256 << 20) if budget != "1" else 512 << 20
    requested = (N_LAYERS - spill) * k * CHUNK + (8 * (CHUNK + 4096) if spill else 0)
    allocations = []
    if spill < N_LAYERS:
        allocations.append({"kind": "resident_chunk", "request_bytes": CHUNK,
                            "count": (N_LAYERS - spill) * k, "host_bytes_each": CHUNK})
    if spill:
        allocations.append({"kind": "ring_buffer", "request_bytes": CHUNK + 4096,
                            "count": 8, "host_bytes_each": 8 << 20})
    return {"n_layers": N_LAYERS, "k": k, "chunk_bytes": CHUNK, "ring": 4, "pin": True,
            "budget_bytes": cap, "spill_layers": spill, "resident_layers": N_LAYERS - spill,
            "spill_file_bytes": spill * k * CHUNK, "host_bytes": held,
            "pinned_floor_bytes": 0, "requested_bytes": requested,
            "all_pinned_bytes": N_LAYERS * k * CHUNK, "minimum_bytes": 64 << 20,
            "unused_bytes": cap - held, "granularity_waste_bytes": 0,
            "allocations": allocations}


def _correctness(k, *, budget_ok=True, control_differs=True, reproducible=True,
                 device="cuda", deterministic=False):
    budgets = {spec: {"plan": _plan_dict(spec, k), "vs_pinned": _pair(backward=budget_ok),
                      "vs_ga": _pair(backward=budget_ok)} for spec in ("0.5", "0.25", "min")}
    return {"meta": {"device": device}, "records": [{
        "kind": "correctness", "k": k, "deterministic": deterministic,
        "ga_vs_ga": _pair(backward=reproducible), "pinned_vs_pinned": _pair(),
        "pinned_vs_ga": _pair(), "budgets": budgets,
        "control": {"spill_layers": 32,
                    "vs_pinned": {"backward": _cmp(not control_differs), "forward_equal": True}},
    }]}


def _cleanup(ok=True, message="SPILL_ENOSPC: ... the plan spills 32 layers x k=4 x 4,194,304 B",
             allocator=1 << 30):
    names = [("ctrl_c_mid_backward", "KeyboardInterrupt"),
             ("exception_mid_forward", "RuntimeError"),
             ("enospc_injected_mid_step", "SpillDiskFullError")]
    return {"meta": {}, "records": [{"kind": "cleanup", "scenarios": [
        {"scenario": name, "expected": exp, "raised": exp, "ok": ok,
         "file_exists_after": not ok, "leftover_spill_files": [], "spill_threads_after": [],
         "allocator_baseline_bytes": allocator, "allocator_after_bytes": allocator,
         "message": message if name.startswith("enospc") else "x"} for name, exp in names],
        "ok": ok}]}


def _arm(comparison, block, position, budget, value, *, tier="ram", weight=None,
         outcome="ok", device="cuda", cap_ok=True, extra_a=4, extra_b=4, rows="r",
         disks=("nvme0n1", "nvme0n1")):
    """``extra_a`` / ``extra_b``: host-allocator bytes the arm allocates outside its store
    during the timed steps (the real run's +4 B loss readback by default)."""
    plan = _plan_dict(budget)
    spill = plan["spill_file_bytes"]
    return {"meta": {"device": device, "direct_io": True if tier == "disk" else None,
                     "seq": 512},
            "records": [{
                "kind": "arm", "skipped": None, "comparison": comparison,
                "arm": "A" if position in (0, 3) else "B", "block": block, "position": position,
                "k": 4, "budget": budget, "tier": tier, "plan": plan, "split": None,
                "volume": {"scope": "local", "direct_io": True, "free_bytes": 100 << 30},
                "ids": {"row_sha256": [rows]}, "outcome": outcome, "value": value,
                "finite": True, "hidden_share": 0.9 if spill else None,
                "spill_write_gb_per_s": 2.0 if spill else None,
                "spill_read_gb_per_s": 3.0 if spill else None,
                "weight_read_gb_per_s": weight,
                "records": [{"spill_written": spill, "spill_read": spill}] * 3,
                "store": {"direct": True if spill else None},
                "cap": {"ok": cap_ok, "reason": "within the plan" if cap_ok else "over",
                        "held_bytes": plan["host_bytes"], "plan_host_bytes": plan["host_bytes"],
                        "budget_bytes": plan["budget_bytes"]},
                "pinned": {"allocator_baseline_bytes": 1 << 30,
                           "allocator_peak_bytes": (1 << 30) + plan["host_bytes"]
                           + (extra_a if position in (0, 3) else extra_b)},
                "process_memory": {"peak_rss": 1 << 33},
                "spill_file_exists_after": False, "leftover_spill_files": [],
                "placement": {"spill": {"drive": {"device": disks[0]}},
                              "weights": {"drive": {"device": disks[1]}}},
            }]}


def _blocks(tmp_path, comparison, a_values, b_values, *, budget_b="0.5", **kw):
    """One A-B-B-A block per (a, b) pair; a/b are the step medians."""
    directory = tmp_path / "arms" / comparison
    directory.mkdir(parents=True, exist_ok=True)
    weights = kw.pop("weights", None)
    for block, (a, b) in enumerate(zip(a_values, b_values)):
        for position, arm in enumerate("ABBA"):
            value = a if arm == "A" else b
            weight = None if weights is None else (weights[0] if arm == "A" else weights[1][block])
            payload = _arm(comparison, block, position, "1" if arm == "A" else budget_b, value,
                           weight=weight, **kw)
            (directory / f"b{block:02d}_p{position}_{arm}.json").write_text(json.dumps(payload))


def _session(tmp_path, *, g1=None, cleanup=None, k2=(1.04,), k4=(1.04,), blocks=5,
             interference=None, disks=("nvme0n1", "nvme0n1"), **arm_kw):
    for k in (2, 3, 4):
        (tmp_path / f"correctness_k{k}.json").write_text(json.dumps((g1 or {}).get(k)
                                                                    or _correctness(k)))
    (tmp_path / "cleanup.json").write_text(json.dumps(cleanup or _cleanup()))
    jitter = [0.0, 0.002, -0.001, 0.001, -0.002, 0.0015, -0.0005]
    for name, ratio in (("ram_k2_half", k2), ("ram_k4_half", k4)):
        a = [10.0 + j for j in jitter[:blocks]]
        b = [(10.0 + j) * ratio[0] + j for j in jitter[:blocks]]
        _blocks(tmp_path, name, a, b, **arm_kw)
    if interference is not None:
        a = [20.0 + j for j in jitter[:blocks]]
        _blocks(tmp_path, "disk_k4_same_drive", a, a, budget_b="min", tier="disk", disks=disks,
                weights=(3.0, [3.0 * interference + j for j in jitter[:blocks]]))
    return str(tmp_path)


def test_everything_passes(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, interference=0.99))
    assert {g: out["gates"][g]["status"] for g in out["gates"]} == {
        "G1": "pass", "G2": "pass", "G3": "pass", "G4": "pass", "G5": "pass"}
    assert out["verdict"].startswith("SPILL READY FOR A3")
    assert out["tokens_per_step"]


def test_a_spilled_budget_that_differs_fails_g1_and_the_verdict(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, g1={4: _correctness(4, budget_ok=False)}))
    assert out["gates"]["G1"]["status"] == "fail"
    assert "k=4: budget 0.25 vs_pinned: backward" in out["gates"]["G1"]["detail"]
    assert out["verdict"].startswith("SPILL NOT SAFE")


def test_a_control_that_does_not_differ_takes_g1_away(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, g1={2: _correctness(2, control_differs=False)}))
    assert out["gates"]["G1"]["status"] == "no verdict"
    assert "cannot fail" in out["gates"]["G1"]["detail"]
    assert out["verdict"] == "NO VERDICT"


def test_unreproducible_ga_uses_the_deterministic_rerun(rule, tmp_path):
    path = _session(tmp_path, g1={3: _correctness(3, reproducible=False)})
    out = rule.evaluate(path)
    assert out["gates"]["G1"]["status"] == "no verdict"
    (tmp_path / "correctness_k3_det.json").write_text(
        json.dumps(_correctness(3, deterministic=True)))
    out = rule.evaluate(path)
    assert out["gates"]["G1"]["status"] == "pass"
    assert out["gates"]["G1"]["parts"][1]["deterministic"] is True


@pytest.mark.parametrize("k", (2, 3, 4))
@pytest.mark.parametrize("spec", ("0.5", "0.25", "min"))
@pytest.mark.parametrize("omission", ("budget", "refused", "plan", "plan_field", "comparison",
                                     "forward", "backward", "gradient_field", "skipped"))
def test_g1_requires_every_registered_budget_and_complete_evidence(rule, tmp_path, k, spec,
                                                                 omission):
    payload = _correctness(k)
    budgets = payload["records"][0]["budgets"]
    entry = budgets[spec]
    if omission == "budget":
        del budgets[spec]
    elif omission == "refused":
        budgets[spec] = {"refused": "BUDGET_BELOW_MINIMUM: reviewer corruption",
                         "code": "BUDGET_BELOW_MINIMUM"}
    elif omission == "plan":
        del entry["plan"]
    elif omission == "plan_field":
        del entry["plan"]["host_bytes"]
    elif omission == "comparison":
        del entry["vs_ga"]
    elif omission == "forward":
        del entry["vs_pinned"]["forward_equal"]
    elif omission == "backward":
        del entry["vs_ga"]["backward"]
    elif omission == "gradient_field":
        del entry["vs_ga"]["backward"]["all_zero"]
    else:
        entry["skipped"] = "not run"
    out = rule.evaluate(_session(tmp_path, g1={k: payload}))
    assert out["gates"]["G1"]["status"] == "no verdict"
    assert out["verdict"] == "NO VERDICT"


def test_g1_does_not_accept_only_the_minimum_budget(rule, tmp_path):
    payload = _correctness(4)
    budgets = payload["records"][0]["budgets"]
    del budgets["0.5"], budgets["0.25"]
    assert rule.evaluate(_session(tmp_path, g1={4: payload}))["gates"]["G1"]["status"] == (
        "no verdict")


@pytest.mark.parametrize("rerun", (False, True))
def test_g1_checks_all_budgets_in_the_selected_run_only(rule, tmp_path, rerun):
    default = _correctness(3, reproducible=not rerun)
    deterministic = _correctness(3, deterministic=True)
    selected = deterministic if rerun else default
    del selected["records"][0]["budgets"]["0.25"]
    path = _session(tmp_path, g1={3: default})
    (tmp_path / "correctness_k3_det.json").write_text(json.dumps(deterministic))
    assert rule.evaluate(path)["gates"]["G1"]["status"] == "no verdict"


def test_g1_uses_complete_deterministic_evidence_without_combining_the_default(rule, tmp_path):
    default = _correctness(3, reproducible=False)
    default["records"][0]["budgets"] = {"min": {"refused": "not run"}}
    path = _session(tmp_path, g1={3: default})
    (tmp_path / "correctness_k3_det.json").write_text(
        json.dumps(_correctness(3, deterministic=True)))
    assert rule.evaluate(path)["gates"]["G1"]["status"] == "pass"


@pytest.mark.parametrize("field", ("tensors", "unequal", "max_abs", "worst", "all_zero", "equal"))
def test_g1_requires_the_complete_gradient_comparison(rule, tmp_path, field):
    payload = _correctness(4)
    del payload["records"][0]["budgets"]["min"]["vs_pinned"]["backward"][field]
    assert rule.evaluate(_session(tmp_path, g1={4: payload}))["gates"]["G1"]["status"] == (
        "no verdict")


def test_g1_accepts_a_nan_control_gap_in_the_form_the_probe_emits(rule, tmp_path):
    """``l2l_schedule.compare`` reports max_abs None and names the non-finite tensors when a
    corrupted read turns the gradients NaN: that control differs, it is not incomplete."""
    payload = _correctness(4)
    control = payload["records"][0]["control"]["vs_pinned"]["backward"]
    control.update(equal=False, unequal=["x"], max_abs=None, nonfinite=["x"])
    assert rule.evaluate(_session(tmp_path, g1={4: payload}))["gates"]["G1"]["status"] == "pass"


@pytest.mark.parametrize("nonfinite", (None, [], "x"))
def test_g1_does_not_accept_a_null_gap_without_named_nonfinite_tensors(
        rule, tmp_path, nonfinite):
    payload = _correctness(4)
    control = payload["records"][0]["control"]["vs_pinned"]["backward"]
    control.update(equal=False, unequal=["x"], max_abs=None)
    if nonfinite is not None:
        control["nonfinite"] = nonfinite
    assert rule.evaluate(_session(tmp_path, g1={4: payload}))["gates"]["G1"]["status"] == (
        "no verdict")


def test_g1_still_fails_a_budget_whose_gradients_went_nonfinite(rule, tmp_path):
    payload = _correctness(4)
    budget = payload["records"][0]["budgets"]["min"]["vs_ga"]["backward"]
    budget.update(equal=False, unequal=["x"], max_abs=None, nonfinite=["x"])
    out = rule.evaluate(_session(tmp_path, g1={4: payload}))["gates"]["G1"]
    assert out["status"] == "fail" and "max_abs None" in out["detail"]


def test_g1_without_any_spilling_budget_is_no_verdict(rule, tmp_path):
    payload = _correctness(4)
    for entry in payload["records"][0]["budgets"].values():
        entry["plan"] = _plan_dict("1")
    assert rule.evaluate(_session(tmp_path, g1={4: payload}))["gates"]["G1"]["status"] == (
        "no verdict")


@pytest.mark.parametrize("ref", ("vs_pinned", "vs_ga"))
@pytest.mark.parametrize("direction", ("forward", "backward", "all_zero"))
def test_g1_fails_observed_mismatches_even_when_another_budget_is_missing(
        rule, tmp_path, ref, direction):
    payload = _correctness(2)
    budgets = payload["records"][0]["budgets"]
    del budgets["0.5"]
    budgets["min"][ref] = _pair(forward=direction != "forward",
                                backward=direction != "backward",
                                zero=("x",) if direction == "all_zero" else ())
    assert rule.evaluate(_session(tmp_path, g1={2: payload}))["gates"]["G1"]["status"] == "fail"


@pytest.mark.parametrize("field", ("ga_vs_ga", "pinned_vs_pinned", "pinned_vs_ga", "control"))
def test_g1_missing_reference_or_control_evidence_is_no_verdict(rule, tmp_path, field):
    payload = _correctness(4)
    del payload["records"][0][field]
    assert rule.evaluate(_session(tmp_path, g1={4: payload}))["gates"]["G1"]["status"] == (
        "no verdict")


def test_g1_cannot_use_a_different_k_as_evidence(rule, tmp_path):
    assert rule.evaluate(_session(tmp_path, g1={2: _correctness(4)}))["gates"]["G1"]["status"] == (
        "no verdict")


def test_cost_over_ten_percent_fails_g4_but_spill_stays_safe(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, k4=(1.25,)))
    assert out["gates"]["G4"]["status"] == "fail"
    assert out["verdict"].startswith("SPILL SAFE BUT COSTLY")


def test_cost_straddling_ten_percent_is_no_verdict(rule, tmp_path):
    path = _session(tmp_path)
    directory = Path(path) / "arms" / "ram_k2_half"
    for i, b in enumerate([10.6, 11.6, 10.5, 11.7, 10.4]):
        for position in (1, 2):
            file = directory / f"b{i:02d}_p{position}_B.json"
            payload = json.loads(file.read_text())
            payload["records"][0]["value"] = b
            file.write_text(json.dumps(payload))
    out = rule.evaluate(path)
    assert out["gates"]["G4"]["parts"][0]["status"] == "no verdict"
    assert "straddles" in out["gates"]["G4"]["parts"][0]["detail"]


def test_too_few_blocks_takes_the_cost_gate_away(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, blocks=3))
    assert out["gates"]["G4"]["status"] == "no verdict"
    assert "V7" in out["gates"]["G4"]["detail"]


def test_an_unknown_f1_check_takes_the_cost_gate_away(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, outcome="unknown"))
    assert out["gates"]["G4"]["status"] == "no verdict"
    assert "V1" in out["gates"]["G4"]["detail"]


def test_a_void_block_is_replaced_not_counted(rule, tmp_path):
    path = _session(tmp_path, blocks=6)
    file = Path(path) / "arms" / "ram_k2_half" / "b02_p1_B.json"
    payload = json.loads(file.read_text())
    payload["records"][0]["outcome"] = "void"
    payload["records"][0]["value"] = 99.0
    file.write_text(json.dumps(payload))
    part = rule.evaluate(path)["gates"]["G4"]["parts"][0]
    assert part["void_blocks"] == 1 and part["valid_blocks"] == 5
    assert part["status"] == "pass"


def test_spill_bytes_that_do_not_match_the_plan_invalidate_the_arm(rule, tmp_path):
    path = _session(tmp_path)
    file = Path(path) / "arms" / "ram_k4_half" / "b00_p1_B.json"
    payload = json.loads(file.read_text())
    payload["records"][0]["records"][0]["spill_read"] = 0
    file.write_text(json.dumps(payload))
    out = rule.evaluate(path)
    assert out["gates"]["G4"]["parts"][1]["status"] == "no verdict"
    assert "V6" in out["gates"]["G4"]["parts"][1]["detail"]


def test_a_network_spill_volume_invalidates_the_arm(rule, tmp_path):
    path = _session(tmp_path)
    file = Path(path) / "arms" / "ram_k2_half" / "b01_p2_B.json"
    payload = json.loads(file.read_text())
    payload["records"][0]["volume"]["scope"] = "network"
    file.write_text(json.dumps(payload))
    out = rule.evaluate(path)
    assert "V2: spill volume scope network" in out["gates"]["G4"]["parts"][0]["detail"]


def test_a_store_over_its_budget_fails_g2(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, cap_ok=False))
    assert out["gates"]["G2"]["status"] == "fail"
    assert out["verdict"].startswith("SPILL NOT SAFE")


def _edit_arm(path, comparison, name, change):
    file = Path(path) / "arms" / comparison / name
    payload = json.loads(file.read_text())
    change(payload["records"][0])
    file.write_text(json.dumps(payload))


def test_a_common_excess_outside_the_store_passes_g2(rule, tmp_path):
    """The 2026-10-25 shape: every arm, with and without spill, allocates the same 4 B
    outside its store (the float32 loss readback). v1 failed this; v2 passes it."""
    out = rule.evaluate(_session(tmp_path, extra_a=4, extra_b=4))
    assert out["gates"]["G2"]["status"] == "pass"
    assert out["rule_version"] == "v2" and out["summary"]["rule_version"] == "v2"
    assert set(out["gates"]["G2"]["excess_bytes"].values()) == {4}


def test_one_byte_over_the_a_reference_fails_g2(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, extra_a=4, extra_b=5))
    assert out["gates"]["G2"]["status"] == "fail"
    assert "over the 4 B the block's A arms" in out["gates"]["G2"]["detail"]
    assert out["verdict"].startswith("SPILL NOT SAFE")


def test_the_a_arms_raw_growth_gives_b_no_room(rule, tmp_path):
    """A's allocator growth includes its whole store (512 MiB at k = 4). Were that the
    reference, B (holding ~320 MiB) could allocate ~192 MiB outside its store unseen."""
    path = _session(tmp_path)
    _edit_arm(path, "ram_k4_half", "b02_p2_B.json", lambda row: row["pinned"].update(
        allocator_peak_bytes=row["pinned"]["allocator_baseline_bytes"] + (512 << 20)))
    out = rule.evaluate(path)
    assert out["gates"]["G2"]["status"] == "fail"
    assert "ram_k4_half/b02_p2_B.json" in out["gates"]["G2"]["detail"]


@pytest.mark.parametrize("break_a,why", [
    (lambda row: row["pinned"].update(allocator_peak_bytes=row["pinned"]["allocator_peak_bytes"]
                                      + 1), "excesses differ"),
    (lambda row: row.update(outcome="void"), "void"),
    (lambda row: row["plan"].update(spill_layers=1), "spilled"),
    (lambda row: row["pinned"].update(allocator_peak_bytes=None), "no allocator counter"),
])
def test_a_block_without_a_clean_equal_a_reference_is_no_verdict(rule, tmp_path, break_a, why):
    path = _session(tmp_path)
    _edit_arm(path, "ram_k2_half", "b01_p3_A.json", break_a)
    out = rule.evaluate(path)
    g2 = out["gates"]["G2"]
    assert g2["status"] == "no verdict", g2["detail"][:300]
    assert why in g2["references"]["ram_k2_half/b01"]["why"]
    assert "ram_k2_half/b01_p1_B.json" in g2["detail"]
    assert out["verdict"] == "NO VERDICT"


def test_a_missing_a_arm_is_no_verdict_for_its_block(rule, tmp_path):
    path = _session(tmp_path)
    (Path(path) / "arms" / "ram_k4_half" / "b00_p0_A.json").unlink()
    g2 = rule.evaluate(path)["gates"]["G2"]
    assert g2["status"] == "no verdict"
    assert "A at position 0 missing" in g2["references"]["ram_k4_half/b00"]["why"]


def test_a_b_arm_without_its_counter_is_no_verdict(rule, tmp_path):
    path = _session(tmp_path)
    _edit_arm(path, "ram_k2_half", "b00_p1_B.json",
              lambda row: row["pinned"].update(allocator_baseline_bytes=None))
    g2 = rule.evaluate(path)["gates"]["G2"]
    assert g2["status"] == "no verdict"
    assert "ram_k2_half/b00_p1_B.json" in g2["detail"]


def test_a_failure_beats_a_missing_reference_elsewhere(rule, tmp_path):
    path = _session(tmp_path)
    _edit_arm(path, "ram_k2_half", "b01_p3_A.json", lambda row: row.update(outcome="void"))
    _edit_arm(path, "ram_k4_half", "b03_p1_B.json", lambda row: row["pinned"].update(
        allocator_peak_bytes=row["pinned"]["allocator_peak_bytes"] + 1))
    assert rule.evaluate(path)["gates"]["G2"]["status"] == "fail"


def test_a_budget_violation_in_one_comparison_survives_a_clean_namesake(rule, tmp_path):
    """Arm basenames repeat across comparisons. The audit's reproduction: one byte over
    in ram_k2_half's b00_p1_B, while ram_k4_half's b00_p1_B (read later, same basename) is
    clean, used to come out as a pass."""
    path = _session(tmp_path)
    _edit_arm(path, "ram_k2_half", "b00_p1_B.json", lambda row: row["pinned"].update(
        allocator_peak_bytes=row["pinned"]["allocator_peak_bytes"] + 1))
    out = rule.evaluate(path)
    assert out["gates"]["G2"]["status"] == "fail"
    assert "ram_k2_half/b00_p1_B.json" in out["gates"]["G2"]["detail"]
    assert "ram_k4_half/b00_p1_B.json" not in out["gates"]["G2"]["detail"]


def test_cleanup_that_leaves_a_file_fails_g3(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, cleanup=_cleanup(ok=False)))
    assert out["gates"]["G3"]["status"] == "fail"


def test_enospc_without_the_arithmetic_fails_g3(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, cleanup=_cleanup(message="OSError: no space")))
    assert out["gates"]["G3"]["status"] == "fail"
    assert "enospc_injected_mid_step" in out["gates"]["G3"]["detail"]


def test_cleanup_without_allocator_counters_is_no_verdict(rule, tmp_path):
    """The CPU pilot: file and threads checked, the return of pinned memory not measured."""
    out = rule.evaluate(_session(tmp_path, cleanup=_cleanup(allocator=None)))
    assert out["gates"]["G3"]["status"] == "no verdict"
    assert "not measured" in out["gates"]["G3"]["detail"]


@pytest.mark.parametrize("scenario", (0, 1, 2))
@pytest.mark.parametrize("change", (
    {"file_exists_after": True},
    {"leftover_spill_files": ["a4-spill-left.bin"]},
    {"spill_threads_after": ["l2l-spill-writer"]},
    {"raised": "ValueError"},
    {"raised": None},
    {"expected": "ValueError", "raised": "ValueError"},
    {"allocator_after_bytes": (1 << 30) + 1},
    {"allocator_after_bytes": (1 << 30) - 1},
))
def test_g3_derives_failures_from_raw_fields_despite_stale_ok(rule, tmp_path, scenario, change):
    payload = _cleanup()
    payload["records"][0]["scenarios"][scenario].update(change)
    out = rule.evaluate(_session(tmp_path, cleanup=payload))
    assert out["gates"]["G3"]["status"] == "fail"
    assert out["verdict"].startswith("SPILL NOT SAFE")


@pytest.mark.parametrize("field", ("raised", "expected", "file_exists_after",
                                   "leftover_spill_files", "spill_threads_after",
                                   "allocator_baseline_bytes", "allocator_after_bytes"))
def test_g3_missing_raw_evidence_is_no_verdict_despite_ok(rule, tmp_path, field):
    payload = _cleanup()
    del payload["records"][0]["scenarios"][0][field]
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == (
        "no verdict")


@pytest.mark.parametrize("field", ("allocator_baseline_bytes", "allocator_after_bytes"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1, True, "1073741824"))
def test_g3_invalid_allocator_counters_cannot_pass(rule, tmp_path, field, value):
    payload = _cleanup()
    payload["records"][0]["scenarios"][0][field] = value
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == (
        "no verdict")


def test_g3_equal_but_negative_allocator_counters_cannot_pass(rule, tmp_path):
    assert rule.evaluate(_session(tmp_path, cleanup=_cleanup(allocator=-1)))["gates"]["G3"][
        "status"] == "no verdict"


def test_g3_accepts_zero_allocator_baseline(rule, tmp_path):
    assert rule.evaluate(_session(tmp_path, cleanup=_cleanup(allocator=0)))["gates"]["G3"][
        "status"] == "pass"


@pytest.mark.parametrize("level", ("scenario", "record"))
def test_g3_rejects_false_ok_that_contradicts_complete_safe_raw_fields(rule, tmp_path, level):
    payload = _cleanup()
    record = payload["records"][0]
    target = record["scenarios"][0] if level == "scenario" else record
    target["ok"] = False
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == "fail"


def test_g3_does_not_require_summary_ok_when_raw_evidence_is_complete(rule, tmp_path):
    payload = _cleanup()
    record = payload["records"][0]
    del record["ok"]
    for scenario in record["scenarios"]:
        del scenario["ok"]
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == "pass"


def test_g3_missing_counter_is_not_a_failure_even_with_false_ok(rule, tmp_path):
    payload = _cleanup()
    scenario = payload["records"][0]["scenarios"][0]
    scenario.update(allocator_after_bytes=None, ok=False)
    payload["records"][0]["ok"] = False
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == (
        "no verdict")


@pytest.mark.parametrize("missing", ("scenario", "counter", "cleanup_record"))
def test_g3_observed_failure_beats_missing_evidence(rule, tmp_path, missing):
    payload = _cleanup()
    record = payload["records"][0]
    if missing == "scenario":
        record["scenarios"].pop()
    elif missing == "counter":
        del record["scenarios"][0]["allocator_after_bytes"]
    else:
        payload["records"] = []
    path = _session(tmp_path, cleanup=payload)
    _edit_arm(path, "ram_k2_half", "b00_p1_B.json",
              lambda row: row.update(spill_file_exists_after=True))
    assert rule.evaluate(path)["gates"]["G3"]["status"] == "fail"


@pytest.mark.parametrize("field", ("spill_file_exists_after", "leftover_spill_files"))
def test_g3_requires_timed_arm_file_cleanup_evidence(rule, tmp_path, field):
    path = _session(tmp_path)
    _edit_arm(path, "ram_k2_half", "b00_p1_B.json", lambda row: row.pop(field))
    assert rule.evaluate(path)["gates"]["G3"]["status"] == "no verdict"


def test_g3_checks_optional_real_enospc_with_the_same_raw_rules(rule, tmp_path):
    payload = _cleanup()
    scenarios = payload["records"][0]["scenarios"]
    scenarios.append({**scenarios[2], "scenario": "enospc_real",
                      "spill_threads_after": ["l2l-spill-writer"]})
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == "fail"


def test_g3_missing_enospc_message_is_no_verdict(rule, tmp_path):
    payload = _cleanup()
    del payload["records"][0]["scenarios"][2]["message"]
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == (
        "no verdict")


def test_g3_duplicate_scenarios_do_not_supply_complete_evidence(rule, tmp_path):
    payload = _cleanup()
    scenarios = payload["records"][0]["scenarios"]
    scenarios.append(dict(scenarios[0]))
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == (
        "no verdict")


@pytest.mark.parametrize("field", ("file_exists_after", "leftover_spill_files",
                                   "spill_threads_after"))
def test_g3_null_cleanup_observations_are_not_evidence(rule, tmp_path, field):
    payload = _cleanup()
    payload["records"][0]["scenarios"][0][field] = None
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == (
        "no verdict")


def test_g3_scenario_failure_beats_another_missing_scenario(rule, tmp_path):
    payload = _cleanup()
    scenarios = payload["records"][0]["scenarios"]
    scenarios.pop()
    scenarios[0]["allocator_after_bytes"] += 1
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == "fail"


def test_g3_a_skipped_cleanup_scenario_is_no_verdict(rule, tmp_path):
    payload = _cleanup()
    payload["records"][0]["scenarios"][0]["skipped"] = "not run"
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == (
        "no verdict")


def test_g3_no_scenario_is_not_a_raw_failure_despite_false_summary(rule, tmp_path):
    payload = _cleanup()
    payload["records"][0].update(scenarios=[], ok=False)
    assert rule.evaluate(_session(tmp_path, cleanup=payload))["gates"]["G3"]["status"] == (
        "no verdict")


def test_g3_checks_timed_arm_leftovers_even_without_cleanup_record(rule, tmp_path):
    path = _session(tmp_path, cleanup={"records": []})
    _edit_arm(path, "ram_k2_half", "b00_p1_B.json",
              lambda row: row.update(leftover_spill_files=["a4-spill-left.bin"]))
    assert rule.evaluate(path)["gates"]["G3"]["status"] == "fail"


@pytest.mark.parametrize("disks,why", [
    (("nvme0n1", "nvme1n1"), "nvme0n1 and nvme1n1"),
    ((None, "nvme0n1"), "physical disk unknown"),
])
def test_g5_needs_the_os_to_show_spill_and_weights_on_one_disk(rule, tmp_path, disks, why):
    """A same-drive label is a claim: two spill/shard directories on different disks, or
    a disk the OS could not name, must not produce a G5 verdict however clean the data."""
    out = rule.evaluate(_session(tmp_path, interference=0.99, disks=disks))
    assert out["gates"]["G5"]["status"] == "no verdict"
    assert "V9" in out["gates"]["G5"]["detail"] and why in out["gates"]["G5"]["detail"]


def test_interference_past_ten_percent_fails_g5_and_adds_the_advice(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, interference=0.7))
    assert out["gates"]["G5"]["status"] == "fail"
    assert out["verdict"].endswith("spill to a drive the weights are not read from")


def test_a_cpu_pilot_never_gets_a_verdict(rule, tmp_path):
    out = rule.evaluate(_session(tmp_path, device="cpu",
                                 g1={k: _correctness(k, device="cpu") for k in (2, 3, 4)}))
    assert out["verdict"] == "NO VERDICT: SYNTHETIC/CPU PILOT"
    assert "V3" in out["gates"]["G4"]["detail"]


def test_missing_files_are_no_verdict_not_a_crash(rule, tmp_path):
    out = rule.evaluate(str(tmp_path))
    assert set(g["status"] for g in out["gates"].values()) == {"no verdict"}
    assert out["verdict"] == "NO VERDICT"


def test_t_interval(rule):
    ci = rule.interval([0.01, 0.03, 0.02, 0.02, 0.02])
    assert ci["mean"] == pytest.approx(0.02)
    assert ci["low"] == pytest.approx(0.02 - 2.776 * 0.00707107 / 5 ** 0.5, rel=1e-4)
    assert rule.interval([0.1])["low"] is None


def test_the_cli_writes_the_verdict(rule, tmp_path, capsys):
    path = _session(tmp_path, interference=0.99)
    assert rule.main([path, "--out", str(tmp_path / "verdict.json")]) == 0
    saved = json.loads((tmp_path / "verdict.json").read_text())
    assert saved["summary"]["G4"] == "pass"
    assert "G1 pass" in capsys.readouterr().out


def test_published_v3_replays_with_unchanged_gates_and_verdict(rule):
    root = (Path(__file__).resolve().parents[1] / "benchmarks" / "results"
            / "a4-v3-rtx3090-20261026")
    published = json.loads((root / "verdict.json").read_text(encoding="utf-8"))
    replay = rule.evaluate(str(root))
    assert replay["gates"] == published["gates"]
    assert replay["verdict"] == published["verdict"]
    assert replay["summary"] == published["summary"]


def test_published_v1_current_rule_illustration_keeps_its_safety_gates(rule):
    root = (Path(__file__).resolve().parents[1] / "benchmarks" / "results"
            / "a4-rtx3090-20261025")
    replay = rule.evaluate(str(root))
    assert {name: gate["status"] for name, gate in replay["gates"].items()} == {
        "G1": "pass", "G2": "pass", "G3": "pass", "G4": "no verdict", "G5": "no verdict"}
    assert replay["verdict"] == "SPILL SAFE, COST NO VERDICT"
