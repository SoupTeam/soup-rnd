"""A whole fixed sequence is required; a good median cannot replace a bad round."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def campaign():
    folder = Path(__file__).parents[1] / "benchmarks/harness"
    spec = importlib.util.spec_from_file_location(
        "c2_linux_campaign", folder / "c2_linux_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    old = sys.path[:]
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


def run_fake(tmp_path, monkeypatch, campaign, *, ratios=None, primitive="PASS primitive",
             staging_changed=False, vram_changed=False, invalid_arm=None):
    primitive_path = tmp_path / "primitive.json"
    primitive_path.write_text(json.dumps({"decision": {"verdict": primitive}}))
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "layer.safetensors").write_bytes(b"immutable fixture")
    args = SimpleNamespace(out=str(tmp_path / "sequence"), primitive=str(primitive_path),
                           shards=str(shards), weights="fixed-weights", device="nvme0n1")
    calls = []
    ratios = ratios or [1.2] * 6

    def child(command, **kwargs):
        depth = int(command[command.index("--c2-depth") + 1])
        index = len(calls) // 2
        calls.append(depth)
        path = Path(command[command.index("--out") + 1])
        data = {
            "c2_validity": {"reasons": ["foreign writes"] if len(calls) == invalid_arm else []},
            "meta": {"c2_source_admission": {
                "staging_bytes": 101 if staging_changed and depth == 2 else 100}},
            "records": [{"label": "step_plain", "step_s_mean": ratios[index] if depth == 1 else 1,
                         "peak_alloc_gb": 4.2 if vram_changed and depth == 2 else 4.0,
                         "records": [{"loss": 1.0}] * 3}],
        }
        path.write_text(json.dumps(data))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", child)
    result = campaign.campaign(args)
    return result, calls, json.loads((Path(args.out) / "campaign.json").read_text())


def test_all_twelve_arms_alternate_and_pass_only_both_gates(tmp_path, monkeypatch, campaign):
    result, calls, record = run_fake(tmp_path, monkeypatch, campaign)
    assert result == 0
    assert calls == [1, 2, 2, 1] * 3
    assert record["decision"]["verdict"].startswith("PASS candidate")


@pytest.mark.parametrize("settings", [
    {"ratios": [1.2] * 5 + [0.99]},
    {"ratios": [1.06] * 6},
    {"primitive": "NO VERDICT"},
    {"staging_changed": True},
    {"vram_changed": True},
])
def test_bad_round_or_unequal_budget_cannot_ship(tmp_path, monkeypatch, campaign, settings):
    result, calls, record = run_fake(tmp_path, monkeypatch, campaign, **settings)
    assert result == 0  # complete measurements; the decision itself does not permit shipping
    assert len(calls) == 12
    assert record["decision"]["verdict"].startswith("DO NOT IMPLEMENT")


def test_invalid_arm_stops_without_cherry_picking_a_replacement(tmp_path, monkeypatch, campaign):
    result, calls, record = run_fake(tmp_path, monkeypatch, campaign, invalid_arm=2)
    assert result == 1
    assert calls == [1, 2]
    assert record["complete"] is False
    assert "foreign writes" in record["invalid_reasons"][0]
    assert record["decision"]["verdict"].startswith("NO VERDICT")
