"""F1 count, validity and per-pair gates cannot be bypassed by a good aggregate."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
              "mean_step_s": 12 if depth == 1 else 10,
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


def test_median_cannot_rescue_a_failed_original_mean_rule(f1):
    data = blocks(5, "step")
    data[-1][2]["mean_step_s"] = 12
    result = f1.evaluate(data, "step")
    assert result["median_ratio"] == 1.2
    assert result["original_mean_rule_ok"] is False
    assert result["verdict"] == "FAIL step benefit"


@pytest.mark.parametrize("bad_arm", [None, 2])
def test_driver_uses_fresh_abba_children_and_stops_the_whole_invalid_sequence(
    f1, tmp_path, monkeypatch, bad_arm
):
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"physical_device": "nvme0n1", "correctness_ok": True,
                                     "direct_io_verified": True}))
    args = SimpleNamespace(out=str(tmp_path / "sequence"), admission=str(admission),
                           device="nvme0n1", kind="primitive", fixture="fixed-fixture",
                           max_blocks=20, shards="")
    calls = []

    class Watch:
        def __init__(self, **kwargs):
            assert kwargs["required"] == ["suspend", "power", "foreign_reads"]
            assert kwargs["disk_device"]("overlay") == "nvme0n1"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.record = {"outcome": "unknown" if len(calls) == bad_arm else "ok"}

    def child(command, **kwargs):
        assert kwargs["env"]["OMP_NUM_THREADS"] == "8"
        depth = int(command[command.index("--single-depth") + 1])
        calls.append(depth)
        Path(command[command.index("--out") + 1]).write_text(json.dumps({
            "complete": True, "invalid_reasons": [],
            "rounds": [{"arms": {str(depth): {"gb_s": 1 if depth == 1 else 1.2}}}],
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(f1, "ArmWatch", Watch)
    monkeypatch.setattr(f1.subprocess, "run", child)
    monkeypatch.setattr(f1.subprocess, "check_output", lambda *a, **k: "fixed-sha")
    result = f1.campaign(args)
    record = json.loads((Path(args.out) / "campaign.json").read_text())
    if bad_arm:
        assert result == 1
        assert calls == [1, 2]
        assert not record["complete"]
        assert record["decision"]["verdict"] == "NO VERDICT"
    else:
        assert result == 0
        assert calls == [1, 2, 2, 1] * 5
        assert record["complete"]
        assert record["decision"]["verdict"] == "PASS primitive"
