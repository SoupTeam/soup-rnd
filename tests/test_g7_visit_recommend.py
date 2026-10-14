"""Tests for the G7 recommendation evaluator (benchmarks/harness/visit_verdicts.py, ticket 07).

Seam 2: the validity rows, the experiment verdict and the prototype result in, one
recommendation and a verdict per branch out. All numbers are fixed and synthetic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"

TIE = "Continue: `tie_weights=False` upstream"
GRAPHS = "Continue: CUDA graphs in the runtime"


@pytest.fixture(scope="module")
def verdicts():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_visit_verdicts_recommend_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "visit_verdicts.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def row(verdict="pass", reason="fine"):
    return {"verdict": verdict, "reason": reason, "evidence": {}}


def rows(**verdicts):
    """The four profile rows, all passing unless named."""
    names = ("timer_is_free", "profile_is_readable", "control", "cpu_bound")
    return {name: row(verdicts.get(name, "pass"), f"{name} is {verdicts.get(name, 'pass')}")
            for name in names}


def prototype(gate=True, saving=0.40, payback=50, pool_bytes=30e6, blocked=None):
    return {
        "gate_passed": gate,
        "blocked_reason": blocked,
        "saving_fraction": saving,
        "payback_steps": payback,
        "pool_bytes_per_layer": pool_bytes,
    }


def recommend(verdicts, profile=None, experiment=None, proto=None):
    return verdicts.recommend(
        profile if profile is not None else rows(),
        experiment if experiment is not None else row("fail", "no gain"),
        proto,
    )


# --- each recommendation alone ------------------------------------------------------------


def test_tie_weights_alone_continues_when_the_experiment_passes(verdicts):
    result = recommend(
        verdicts,
        profile=rows(cpu_bound="fail"),
        experiment=row("pass", "B is faster"),
    )
    assert result["recommendation"] == TIE
    assert result["branches"]["tie_weights"]["verdict"] == "pass"
    assert result["branches"]["cuda_graphs"]["verdict"] == "fail"


def test_cuda_graphs_alone_continue_when_every_condition_holds(verdicts):
    result = recommend(verdicts, proto=prototype())
    assert result["recommendation"] == GRAPHS
    assert result["branches"]["cuda_graphs"]["verdict"] == "pass"
    assert result["branches"]["tie_weights"]["verdict"] == "fail"


def test_stop_when_neither_branch_holds(verdicts):
    result = recommend(verdicts, profile=rows(cpu_bound="fail"))
    assert result["recommendation"] == "Stop"
    assert result["branches"]["tie_weights"]["verdict"] == "fail"
    assert result["branches"]["cuda_graphs"]["verdict"] == "fail"


def test_both_continue_branches_at_once(verdicts):
    result = recommend(verdicts, experiment=row("pass", "B is faster"), proto=prototype())
    assert result["recommendation"] == f"{TIE}; {GRAPHS}"
    assert result["continue"] == [TIE, GRAPHS]


# --- the prototype conditions -------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, named",
    [
        ({"saving": 0.29}, "30%"),
        ({"payback": 101}, "100 steps"),
        ({"pool_bytes": 1.1e9 / 32 + 1e6}, "1.1 GB"),
    ],
)
def test_a_missed_prototype_threshold_stops_the_graph_branch(verdicts, kwargs, named):
    result = recommend(verdicts, proto=prototype(**kwargs))
    assert result["recommendation"] == "Stop"
    assert result["branches"]["cuda_graphs"]["verdict"] == "fail"
    assert named in result["branches"]["cuda_graphs"]["reason"]


@pytest.mark.parametrize(
    "kwargs", [{"saving": 0.30}, {"payback": 100}, {"pool_bytes": 1.1e9 / 32}]
)
def test_prototype_thresholds_hold_at_their_edge(verdicts, kwargs):
    assert recommend(verdicts, proto=prototype(**kwargs))["recommendation"] == GRAPHS


def test_a_prototype_that_failed_its_gate_is_blocked_with_no_verdict(verdicts):
    branch = recommend(verdicts, proto=prototype(gate=False))["branches"]["cuda_graphs"]
    assert branch["verdict"] == "no_verdict"
    assert branch["row"] == "prototype"
    assert "gate" in branch["reason"]


# --- no verdict ---------------------------------------------------------------------------


def test_no_verdict_on_the_graph_branch_while_the_experiment_stands(verdicts):
    result = recommend(
        verdicts,
        profile=rows(control="no_verdict"),
        experiment=row("pass", "B is faster"),
    )
    assert result["recommendation"] == TIE
    branch = result["branches"]["cuda_graphs"]
    assert branch["verdict"] == "no_verdict"
    assert branch["row"] == "control"
    assert "control arm" in branch["rerun"]


def test_no_verdict_on_the_experiment_while_the_graph_branch_stands(verdicts):
    result = recommend(verdicts, experiment=row("no_verdict", "the gate failed"), proto=prototype())
    assert result["recommendation"] == GRAPHS
    branch = result["branches"]["tie_weights"]
    assert branch["verdict"] == "no_verdict"
    assert branch["row"] == "experiment"
    assert "tie_experiment.py" in branch["rerun"]


def test_a_stopped_branch_beside_a_no_verdict_branch_is_no_recommendation(verdicts):
    result = recommend(verdicts, experiment=row("no_verdict", "a run is void"),
                       profile=rows(cpu_bound="fail"))
    assert result["recommendation"] == "No verdict"


def test_both_branches_without_a_verdict(verdicts):
    result = recommend(verdicts, experiment=row("no_verdict", "x"), profile=rows(control="fail"))
    assert result["recommendation"] == "No verdict"
    assert result["continue"] == []


@pytest.mark.parametrize("name", ["timer_is_free", "profile_is_readable", "control"])
def test_a_validity_row_with_no_pass_names_itself_in_the_graph_branch(verdicts, name):
    result = recommend(verdicts, profile=rows(**{name: "fail"}), proto=prototype())
    branch = result["branches"]["cuda_graphs"]
    assert branch["verdict"] == "no_verdict"
    assert branch["row"] == name
    assert branch["rerun"]


def test_a_threshold_row_without_a_verdict_names_itself(verdicts):
    branch = recommend(verdicts, profile=rows(cpu_bound="no_verdict"))["branches"]["cuda_graphs"]
    assert branch["verdict"] == "no_verdict"
    assert branch["row"] == "cpu_bound"
    assert branch["rerun"]


def test_a_fired_threshold_without_a_prototype_is_no_verdict_naming_the_prototype(verdicts):
    result = recommend(verdicts, proto=None)
    branch = result["branches"]["cuda_graphs"]
    assert branch["verdict"] == "no_verdict"
    assert branch["row"] == "prototype"


def test_a_blocked_prototype_is_no_verdict_with_the_reason(verdicts):
    result = recommend(verdicts, proto=prototype(gate=False, blocked="graph capture OOM"))
    branch = result["branches"]["cuda_graphs"]
    assert branch["verdict"] == "no_verdict"
    assert "prototype blocked: graph capture OOM" in branch["reason"]


def test_a_missing_prototype_number_is_no_verdict_not_an_error(verdicts):
    proto = prototype()
    del proto["payback_steps"]
    branch = recommend(verdicts, proto=proto)["branches"]["cuda_graphs"]
    assert branch["verdict"] == "no_verdict"
    assert "payback_steps" in branch["reason"]


# --- the rounds ---------------------------------------------------------------------------


def test_rounds_that_agree_keep_their_verdict(verdicts):
    combined = verdicts.combine_rounds([rows(), rows()])
    assert all(r["verdict"] == "pass" for r in combined.values())
    combined = verdicts.combine_rounds([rows(cpu_bound="fail"), rows(cpu_bound="fail")])
    assert combined["cpu_bound"]["verdict"] == "fail"


def test_agreeing_rounds_keep_both_rounds_evidence(verdicts):
    first, second = rows(), rows()
    first["cpu_bound"]["evidence"] = {"ratio": 3.0}
    second["cpu_bound"]["evidence"] = {"ratio": 3.4}
    combined = verdicts.combine_rounds([first, second])
    assert [e["ratio"] for e in combined["cpu_bound"]["evidence"]["rounds"]] == [3.0, 3.4]


def test_rounds_that_disagree_give_no_verdict(verdicts):
    combined = verdicts.combine_rounds([rows(), rows(cpu_bound="fail")])
    assert combined["cpu_bound"]["verdict"] == "no_verdict"
    assert "disagree" in combined["cpu_bound"]["reason"]
    assert combined["control"]["verdict"] == "pass"


def test_a_round_without_a_verdict_gives_no_verdict(verdicts):
    combined = verdicts.combine_rounds([rows(), rows(control="no_verdict")])
    assert combined["control"]["verdict"] == "no_verdict"
