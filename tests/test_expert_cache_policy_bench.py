"""Fail-closed scheduling, reference accuracy and paired static-hotset comparisons."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.harness import expert_cache_policy_bench as bench
from benchmarks.harness.two_read_cache_common import atomic_json, sha256_file
from tests.test_two_read_cache_evidence import _capture
from tests.test_two_read_cache_runner import _spec


def _base() -> dict:
    return {**_spec(), "host_cache_bytes": 2 * 2**30, "device_cache_bytes": 2**28,
            "host_cache_policy": "lru"}


def _snapshot(label: str, *, before: bool, large: bool = False) -> dict:
    cached = label != "two_read"
    host, device = (20, 10) if cached else (0, 0)
    logical = 80
    scale = 0 if before else 1
    return {"expert_cache": {
        "host_cache_policy": "static" if label.startswith("two_read_static") else "lru",
        "logical_fetches": logical * scale,
        "storage_reads": (logical - host - device) * scale,
        "storage_bytes_read": (logical - host - device) * scale * 128,
        "materializations": (logical - device) * scale,
        "host": {"hits": host * scale, "budget_bytes": 2 * 2**30 if cached else 0,
                 "payload_bytes": 64 if cached and not before else 0, "evictions": 0},
        "device": {"hits": device * scale,
                   "budget_bytes": int(1.5 * 2**30) if large else 2**28 if cached else 0,
                   "payload_bytes": 32 if cached and not before else 0},
    }}


def _result(label: str, seconds: float, *, large: bool = False) -> dict:
    return {"status": "passed", "timing": {
        "update_seconds": [seconds] * 6, "tokens_per_update": 6,
        "cuda": {"peak_allocated_bytes": 150, "peak_reserved_bytes": 200},
        "resources": {"sampled_peak_rss_bytes": 1000},
        "runtime_after": _snapshot(label, before=False, large=large),
    }, "setup": {"seconds": 1.0, "cache_plan": _plan(label, large=large)},
        "fresh_process_wall_seconds": 2.0}


def _plan(label: str, *, large: bool = False) -> dict | None:
    if label == "two_read":
        return None
    return {"host_cache_policy": "static" if label.startswith("two_read_static") else "lru",
            "host_cache_bytes": 2 * 2**30,
            "device_cache_bytes": int(1.5 * 2**30) if large else 2**28}


def _blocks(*, large: bool = False, ratios: dict | None = None) -> list[dict]:
    arms, orders = bench.protocol(1.5 if large else None)
    ratios = ratios or {"two_read_lru": .985, "two_read_static": .9,
                        bench.LARGE_ARM: .85}
    blocks = []
    for number, order in enumerate(orders):
        baseline = (1., 2., 4., 8., 3., 6.)[number]
        for position, arm in enumerate(order):
            blocks.append({"round": number, "position": position, "arm": arm,
                           "result": _result(arm, baseline * ratios.get(arm, 1),
                                             large=arm == bench.LARGE_ARM)})
    assert set(arms) == {row["arm"] for row in blocks}
    return blocks


def test_three_arm_specs_change_only_host_policy_and_keep_native_protocol() -> None:
    base = _base()
    original = copy.deepcopy(base)
    for label in bench.ARMS:
        child, worker = bench.arm_spec(base, label)
        assert {key: value for key, value in child.items() if key != "host_cache_policy"} == {
            key: value for key, value in base.items() if key != "host_cache_policy"
        }
        assert child["rounds"] == 4
        assert worker == ("two_read" if label == "two_read" else "two_read_cache")
    assert base == original
    assert len(bench.ORDERS) == len(set(bench.ORDERS)) == 6
    for label in bench.ARMS:
        assert [sum(order[position] == label for order in bench.ORDERS)
                for position in range(3)] == [2, 2, 2]


def test_optional_large_device_arm_requires_explicit_opt_in_and_preserves_base_caps() -> None:
    with pytest.raises(ValueError):
        bench.arm_spec(_base(), bench.LARGE_ARM)
    child, worker = bench.arm_spec(_base(), bench.LARGE_ARM, 1.5)
    assert child["device_cache_bytes"] == 1.5 * 2**30
    assert child["host_cache_bytes"] == 2 * 2**30
    assert child["host_cache_policy"] == "static" and worker == "two_read_cache"
    for bad in (0, .25, 1, float("nan"), True):
        with pytest.raises(ValueError):
            bench.protocol(bad)
    with pytest.raises(ValueError):
        bench.arm_spec({**_base(), "device_cache_bytes": 2**30}, "two_read", 1.5)
    arms, orders = bench.protocol(1.5)
    assert len(arms) == len(orders) == 4
    for label in arms:
        assert [sum(order[position] == label for order in orders)
                for position in range(4)] == [1, 1, 1, 1]


@pytest.mark.parametrize("large", (False, True))
def test_exact_paired_round_ratios_and_memory_scopes(large: bool) -> None:
    summary = bench.timing_summary(
        _blocks(large=large), large_device_cache_gib=1.5 if large else None,
    )
    lru = summary["comparisons"]["two_read_lru_vs_two_read"]
    static = summary["comparisons"]["two_read_static_vs_two_read"]
    assert lru["step_time_reduction_percent"] == pytest.approx(1.5)
    assert not lru["passes_practical_2_percent_gate"]
    assert static["step_time_reduction_percent"] == pytest.approx(10)
    assert static["paired_bootstrap_95_percent"] == pytest.approx([10, 10])
    assert static["passes_practical_2_percent_gate"]
    assert summary["static_vs_lru"]["step_time_reduction_percent"] == pytest.approx(
        100 * (1 - .9 / .985),
    )
    memory = summary["arms"]["two_read_static"]
    assert memory["max_observed_cache_payload_bytes"] == {"host": 64, "device": 32}
    assert memory["max_sampled_rss_bytes"] == 1000
    if large:
        assert summary["large_device_vs_static"]["step_time_reduction_percent"] == pytest.approx(
            100 * (1 - .85 / .9),
        )


def test_positive_point_estimate_cannot_hide_wide_paired_interval() -> None:
    blocks = _blocks()
    baselines = (1., 2., 4., 8., 3., 6.)
    for block in blocks:
        if block["arm"] == "two_read_static":
            ratio = .8 if block["round"] < 5 else 1.4
            block["result"]["timing"]["update_seconds"] = [
                baselines[block["round"]] * ratio,
            ] * 6
    result = bench.timing_summary(blocks)["comparisons"]["two_read_static_vs_two_read"]
    assert result["step_time_reduction_percent"] > 2
    assert result["paired_bootstrap_95_percent"][0] < 2
    assert not result["passes_practical_2_percent_gate"]


@pytest.mark.parametrize("fault", (
    "order", "missing", "duplicate", "nan", "zero", "boolean", "short", "failed", "tokens",
))
def test_timing_matrix_cannot_hide_incomplete_or_invalid_blocks(fault: str) -> None:
    blocks = _blocks()
    if fault == "order":
        blocks[0]["position"] = 2
    elif fault == "missing":
        blocks.pop()
    elif fault == "duplicate":
        blocks.append(copy.deepcopy(blocks[0]))
    elif fault == "failed":
        blocks[0]["result"]["status"] = "failed"
    elif fault == "tokens":
        blocks[0]["result"]["timing"]["tokens_per_update"] = 7
    elif fault == "short":
        blocks[0]["result"]["timing"]["update_seconds"].pop()
    else:
        blocks[0]["result"]["timing"]["update_seconds"][0] = {
            "nan": float("nan"), "zero": 0, "boolean": True,
        }[fault]
    with pytest.raises(ValueError):
        bench.timing_summary(blocks)


def _mock_run(tmp_path: Path, monkeypatch, fault: str | None = None, *, large: bool = False):
    import torch

    source = tmp_path / "spec.json"
    atomic_json(source, _base())
    events = []
    monkeypatch.setattr(bench.native, "verify_spec", lambda *_args, **_kwargs: None)

    def worker(path, spec, arm, phase, out):
        out.mkdir(parents=True)
        label = out.name
        events.append((phase, label))
        if fault == "source" and label == "two_read_static":
            atomic_json(source, {**_base(), "seed": 999})
        result = {"status": "passed", "phase": phase, "arm": arm,
                  "spec_sha256": sha256_file(path),
                  "setup": {"cache_plan": _plan(label, large=label == bench.LARGE_ARM)}}
        if label == "two_read_static":
            if fault == "worker_arm":
                result["arm"] = "two_read"
            elif fault == "plan":
                result["setup"]["cache_plan"]["host_cache_policy"] = "lru"
        if phase == "accuracy":
            capture = _capture()
            if label == "two_read_lru":
                capture["microbatches"][0]["logits"][0] += .0008
            elif label.startswith("two_read_static"):
                capture["microbatches"][0]["logits"][0] -= .0008
            if label == "two_read_static":
                if fault == "gradient":
                    capture["microbatches"][0]["gradients"]["adapter_a"][0] += .1
                elif fault == "inputs":
                    capture["microbatches"][0]["input_ids"][0, 0] += 1
                elif fault == "init":
                    capture["initial_adapters"]["adapter_a"][0] += .0001
                elif fault == "short":
                    capture["microbatches"] = capture["microbatches"][:2]
                    capture["updates"] = capture["updates"][:1]
            capture_path = out / "capture.pt"
            torch.save(capture, capture_path)
            before = _snapshot(label, before=True, large=label == bench.LARGE_ARM)
            after = _snapshot(label, before=False, large=label == bench.LARGE_ARM)
            if label == "two_read_static":
                if fault == "policy":
                    after["expert_cache"]["host_cache_policy"] = "lru"
                elif fault == "gpu_budget":
                    after["expert_cache"]["device"]["budget_bytes"] *= 2
                elif fault == "no_hits":
                    after["expert_cache"]["host"]["hits"] = 0
            result["accuracy"] = {
                "capture_sha256": "bad" if fault == "sha" else sha256_file(capture_path),
                "runtime_before": before, "runtime_after": after,
            }
        else:
            result.update(_result(label, {"two_read": 1., "two_read_lru": .99,
                                          "two_read_static": .9,
                                          bench.LARGE_ARM: .85}[label],
                                  large=label == bench.LARGE_ARM))
        assert spec["host_cache_policy"] == ("static" if label.startswith("two_read_static")
                                              else "lru")
        return result

    monkeypatch.setattr(bench.native, "launch_worker", worker)
    options = {"large_device_cache_gib": 1.5} if large else {}
    return source, tmp_path / "out", events, options


@pytest.mark.parametrize("large", (False, True))
def test_all_fresh_accuracy_arms_precede_all_balanced_timing(
    tmp_path: Path, monkeypatch, large: bool,
) -> None:
    source, out, events, options = _mock_run(tmp_path, monkeypatch, large=large)
    result = bench.run(source, out, **options)
    arms, orders = bench.protocol(1.5 if large else None)
    assert events[:len(arms)] == [("accuracy", arm) for arm in arms]
    assert events[len(arms):] == [("timing", arm) for order in orders for arm in order]
    assert result["status"] == "completed" and result["accuracy_passed"]
    assert all(result["accuracy"][arm]["gate"]["passed"] for arm in arms)
    # Each candidate is acceptable against baseline; they are deliberately not
    # acceptable against each other under the unchanged 1% pointwise gate.
    assert not result["accuracy"]["two_read_lru"]["gate"]["bitwise"]
    assert len(result["timing_blocks"]) == len(arms) * len(orders)


@pytest.mark.parametrize("fault", (
    "gradient", "inputs", "init", "short", "sha", "policy", "gpu_budget", "no_hits", "source",
    "worker_arm", "plan",
))
def test_failed_accuracy_identity_or_policy_never_starts_timing(
    tmp_path: Path, monkeypatch, fault: str,
) -> None:
    source, out, events, options = _mock_run(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError):
        bench.run(source, out, **options)
    assert not any(phase == "timing" for phase, _arm in events)
    saved = json.loads((out / "run.json").read_text())
    assert saved["status"] == "failed"


def test_cli_defaults_and_invalid_cohort_rounds(tmp_path: Path) -> None:
    args = bench.parse_args(["--spec", "spec.json", "--out", "out"])
    assert args.rounds is None and args.large_device_cache_gib is None
    for rounds, large in ((4, None), (6, 1.5), (0, None), (True, None)):
        with pytest.raises(ValueError, match="balanced"):
            bench.run(tmp_path / "absent", tmp_path / "out", rounds=rounds,
                      large_device_cache_gib=large)
