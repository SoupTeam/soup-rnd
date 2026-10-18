"""F1 count, validity and per-pair gates cannot be bypassed by a good aggregate."""
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def f1():
    folder = Path(__file__).parents[1] / "benchmarks/harness"
    spec = importlib.util.spec_from_file_location("c2_f1_campaign", folder / "c2_f1_campaign.py")
    module = importlib.util.module_from_spec(spec)
    old = sys.path[:]
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


def blocks(count, kind="primitive"):
    return [[{"depth": depth, "value": (1 if depth == 1 else 1.2) if kind == "primitive"
              else (12 if depth == 1 else 10), "watch": {"outcome": "ok"},
              "staging": 100, "vram": 4.0} for depth in (1, 2, 2, 1)]
            for _ in range(count)]


def test_three_blocks_do_not_complete_f1_despite_six_good_pairs(f1):
    result = f1.evaluate(blocks(3), "primitive")
    assert result["additional_blocks"] == 2
    assert result["verdict"] == "NO VERDICT"


@pytest.mark.parametrize("kind, expected", [("primitive", "PASS primitive"), ("step", "PASS step")])
def test_five_stable_blocks_complete_both_repeat_and_effect_rules(f1, kind, expected):
    result = f1.evaluate(blocks(5, kind), kind)
    assert result["additional_blocks"] == 0
    assert result["verdict"] == expected


def test_large_spread_requires_more_blocks_even_with_five_pilot_blocks(f1):
    data = blocks(5)
    for index, block in enumerate(data):
        for arm in block:
            arm["value"] *= (1 if index % 2 else 3)
    result = f1.evaluate(data, "primitive")
    assert result["additional_blocks"] > 0
    assert result["verdict"] == "NO VERDICT"


def test_unknown_sensor_withholds_otherwise_passing_result(f1):
    data = blocks(5)
    data[0][0]["watch"]["outcome"] = "unknown"
    assert f1.evaluate(data, "primitive")["verdict"] == "NO VERDICT"


def test_one_bad_pair_withholds_step_benefit_despite_good_median(f1):
    data = blocks(5, "step")
    data[-1][2]["value"] = 12
    result = f1.evaluate(data, "step")
    assert result["median_ratio"] == 1.2
    assert result["verdict"] == "FAIL step benefit"


def test_extra_staging_withholds_step_pass(f1):
    data = blocks(5, "step")
    data[2][1]["staging"] += 1
    result = f1.evaluate(data, "step")
    assert result["memory_ok"] is False
    assert result["verdict"] == "FAIL step benefit"
