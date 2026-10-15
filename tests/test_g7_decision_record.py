"""The G7 record holds the rule's committed numbers (benchmarks/probe-rtx3050-short-step-visit.md).

The rule is committed before the first arm and no threshold may change afterwards. These tests
tie the record's text to the constants the harness executes, so the two cannot drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks" / "probe-rtx3050-short-step-visit.md"
HARNESS = ROOT / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def text():
    return RECORD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def constants():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    import visit_profile
    import visit_verdicts

    return visit_profile, visit_verdicts


def test_the_record_has_the_sections_the_ticket_asks_for(text):
    for heading in (
        "## 1. Question",
        "## 2. Fixture and step shape",
        "## 3. The rule",
        "### 3.2 Validity rows",
        "### 3.3 The control",
        "### 3.4 CPU-bound threshold",
        "### 3.5 Experiment",
        "### 3.6 Recommendations",
        "### 3.7 Prototype gate, cap and thresholds",
        "## 5. Words",
        "## 6. Why",
        "## 8. Numbers from other records",
    ):
        assert heading in text


def test_the_thresholds_in_the_record_match_the_code(text, constants):
    _, verdicts = constants
    assert f"at most {verdicts.TIMER_EXCESS_LIMIT:.0%} longer" in text
    assert f"at most {verdicts.INFLATION_LIMIT} times" in text
    assert f"{verdicts.CONTROL_SPIN_MS * 1000:.0f} µs ± {verdicts.CONTROL_TOLERANCE:.0%}" in text
    assert f"exactly {verdicts.CONTROL_LAUNCHES}" in text
    assert f"at least {verdicts.CPU_BOUND_RATIO:g} times the GPU kernel time" in text
    assert f"alpha {verdicts.ALPHA}" in text
    assert f"at least {verdicts.GRAPH_SAVING_MIN:.0%} of capturable host time" in text
    assert f"at most {verdicts.GRAPH_PAYBACK_MAX_STEPS} steps" in text
    assert f"{verdicts.GRAPH_LAYERS} private pools" in text
    assert f"within {verdicts.GRAPH_POOL_BUDGET_BYTES / 1e9:g} GB" in text


def test_the_step_counts_in_the_record_match_the_code(text, constants):
    profile, _ = constants
    assert (
        f"The bare and timer arms take {profile.WARMUP_STEPS} warm-up steps "
        f"and {profile.TIMED_STEPS} timed steps"
    ) in text
    assert (
        f"The profile and control arms take {profile.WARMUP_STEPS} warm-up steps "
        f"and {profile.PROFILE_STEPS} profiled steps"
    ) in text


def test_the_three_recommendations_are_worded_as_the_evaluator_returns_them(text, constants):
    _, verdicts = constants
    assert f'"{verdicts.CONTINUE_TIE}"' in text
    assert f'"{verdicts.CONTINUE_GRAPHS}"' in text
    assert f'"{verdicts.STOP}"' in text
    assert verdicts.DETECTION_LIMIT in text


def test_the_old_figures_sit_only_in_the_other_records_table(text):
    before_table, table = text.split("## 8. Numbers from other records")
    for figure in ("0.65 ms", "~10 ms", "2.9x"):
        assert figure not in before_table
        assert figure in table
    assert "3d2be4d0" in table and "57c18e5" in table and "8b54b472" in table
    assert "No ratio or difference" in table


def test_every_arm_label_is_named(text, constants):
    profile, _ = constants
    assert profile.LABEL in text
