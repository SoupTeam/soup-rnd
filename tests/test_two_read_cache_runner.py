"""The GPU controller must qualify every trajectory before starting any timing."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harness import two_read_cache as runner
from benchmarks.harness.two_read_cache_common import ARMS, atomic_json, sha256_file
from tests.test_two_read_cache_evidence import _capture


def _spec() -> dict:
    return {
        "seq_len": 3, "batch_size": 1, "accumulation": 2, "accuracy_updates": 2,
        "warmup_updates": 2, "timed_updates": 6, "rounds": 4, "timeout_seconds": 60,
        "lora_r": 4, "lora_alpha": 8, "lora_targets": ["q_proj", "o_proj"],
        "host_cache_bytes": 1024, "device_cache_bytes": 1024, "dtype": "bfloat16",
        "quant": "nf4", "max_host_bytes": 32 * 2**30, "device_cap_bytes": 8 * 2**30,
    }


def _cache_counters(arm: str, *, before: bool) -> dict:
    logical = 120 if arm in {"ordinary", "cache"} else 80
    cached = arm in {"cache", "two_read_cache"}
    reads = logical - 20 if cached else logical
    return {"expert_cache": {
        "logical_fetches": 0 if before else logical,
        "storage_reads": 0 if before else reads,
        "storage_bytes_read": 0 if before else reads * 128,
        "materializations": 0 if before else logical - (10 if cached else 0),
        "host": {"hits": 0 if before or not cached else 10},
        "device": {"hits": 0 if before or not cached else 10},
    }}


def _accuracy_counters() -> dict:
    return {arm: {"worker": {"accuracy": {
        "runtime_before": _cache_counters(arm, before=True),
        "runtime_after": _cache_counters(arm, before=False),
    }}} for arm in ARMS}


def test_cache_behavior_verifies_three_two_reads_and_both_real_cache_tiers() -> None:
    result = runner.validate_cache_behavior(_accuracy_counters())
    deltas = result["expert_counter_deltas"]
    assert deltas["ordinary"]["logical_fetches"] == 120
    assert deltas["two_read"]["logical_fetches"] == 80
    assert deltas["cache"]["host_hits"] == deltas["cache"]["device_hits"] == 10
    assert "not physical SSD" in result["storage_bytes_read_meaning"]


@pytest.mark.parametrize("corruption", [
    "bad_ratio", "no_host_hit", "no_device_hit", "cache_visits",
])
def test_cache_behavior_cannot_claim_an_unexercised_optimization(corruption: str) -> None:
    accuracy = _accuracy_counters()
    if corruption == "bad_ratio":
        after = accuracy["two_read"]["worker"]["accuracy"]["runtime_after"]["expert_cache"]
        after["logical_fetches"] += 1
        after["storage_reads"] += 1
    else:
        after = accuracy["cache"]["worker"]["accuracy"]["runtime_after"]["expert_cache"]
        if corruption == "no_host_hit":
            after["host"]["hits"] = 0
        elif corruption == "no_device_hit":
            after["device"]["hits"] = 0
        else:
            after["logical_fetches"] += 1
    with pytest.raises(ValueError):
        runner.validate_cache_behavior(accuracy)


def _blocks(ratios: dict[str, float] | None = None, rounds: int = 4) -> list[dict]:
    ratios = ratios or {"two_read": 0.9, "cache": 0.8, "two_read_cache": 0.75}
    blocks = []
    for number in range(rounds):
        # Pair within round: large host drift must not become an arm's benefit.
        baseline = (1, 4, 2, 8)[number % 4]
        for position, arm in enumerate(runner.ORDERS[number % 4]):
            seconds = baseline * ratios.get(arm, 1)
            blocks.append({
                "round": number, "position": position, "arm": arm,
                "result": {"timing": {"update_seconds": [seconds] * 6}},
            })
    return blocks


def test_timing_is_paired_by_round_and_combined_savings_are_not_added() -> None:
    summary = runner.timing_summary(_blocks())
    comparisons = summary["comparisons"]
    assert comparisons["two_read"]["step_time_reduction_percent"] == pytest.approx(10)
    assert comparisons["cache"]["step_time_reduction_percent"] == pytest.approx(20)
    assert comparisons["two_read_cache"]["step_time_reduction_percent"] == pytest.approx(25)
    incremental = summary["incremental_comparisons"]
    assert incremental["two_read_cache_vs_two_read"]["reference"] == "two_read"
    assert incremental["two_read_cache_vs_two_read"]["step_time_reduction_percent"] == (
        pytest.approx(100 * (1 - .75 / .9))
    )
    assert incremental["two_read_cache_vs_cache"]["reference"] == "cache"
    for comparison in comparisons.values():
        assert comparison["reference"] == "ordinary"
        assert comparison["paired_rounds"] == 4
        assert comparison["passes_practical_2_percent_gate"]


def test_positive_speed_difference_below_fixed_practical_gate_does_not_pass() -> None:
    summary = runner.timing_summary(_blocks({
        "two_read": 0.985, "cache": 1.01, "two_read_cache": 0.97,
    }))
    comparisons = summary["comparisons"]
    assert not comparisons["two_read"]["passes_practical_2_percent_gate"]
    assert not comparisons["cache"]["passes_practical_2_percent_gate"]
    assert comparisons["two_read_cache"]["passes_practical_2_percent_gate"]


def test_wide_paired_interval_cannot_be_replaced_by_positive_point_estimate() -> None:
    blocks = _blocks()
    for block in blocks:
        if block["arm"] == "cache":
            baseline = (1, 4, 2, 8)[block["round"]]
            ratio = (0.8, 0.8, 0.8, 1.4)[block["round"]]
            block["result"]["timing"]["update_seconds"] = [baseline * ratio] * 6
    cache = runner.timing_summary(blocks)["comparisons"]["cache"]
    assert cache["step_time_reduction_percent"] > 2
    assert cache["paired_bootstrap_95_percent"][0] < 2
    assert not cache["passes_practical_2_percent_gate"]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_nonfinite_or_nonpositive_timing_cannot_be_used(bad: float) -> None:
    blocks = _blocks()
    blocks[0]["result"]["timing"]["update_seconds"][0] = bad
    with pytest.raises(ValueError):
        runner.timing_summary(blocks)


@pytest.mark.parametrize("mutation", ["missing_arm", "short_block", "duplicate", "unknown"])
def test_incomplete_or_duplicate_rounds_do_not_produce_a_speed_claim(mutation: str) -> None:
    blocks = _blocks()
    if mutation == "missing_arm":
        blocks.pop()
    elif mutation == "short_block":
        blocks[0]["result"]["timing"]["update_seconds"].pop()
    elif mutation == "duplicate":
        blocks.append(copy.deepcopy(blocks[0]))
    else:
        blocks[0]["arm"] = "bogus"
    with pytest.raises(ValueError):
        runner.timing_summary(blocks)


@pytest.mark.parametrize("mutation", ["five_rounds", "missing_round", "wrong_position"])
def test_rounds_must_retain_balanced_process_order(mutation: str) -> None:
    blocks = _blocks(rounds=5 if mutation == "five_rounds" else 4)
    if mutation == "missing_round":
        for block in blocks:
            if block["round"] == 3:
                block["round"] = 9
    elif mutation == "wrong_position":
        blocks[0]["position"] = 3
    with pytest.raises(ValueError):
        runner.timing_summary(blocks)


def test_valid_spec_and_cli_keep_fixed_protocol_defaults() -> None:
    runner.validate_spec(_spec())
    args = runner.parse_args([
        "prepare", "--weights", "weights", "--shards", "shards", "--out", "results",
        "--host-cache-gib", "1", "--device-cache-gib", "0.25",
    ])
    assert (args.seq_len, args.accumulation, args.accuracy_updates) == (512, 2, 2)
    assert (args.warmup_updates, args.timed_updates, args.rounds) == (2, 6, 4)
    assert args.host_cache_policy == "lru"
    execute = runner.parse_args(["run", "--spec", "spec.json", "--out", "run", "--accuracy-only"])
    assert execute.accuracy_only and execute.spec == Path("spec.json")


@pytest.mark.parametrize("field,value", [
    ("seq_len", 1), ("accumulation", 1), ("accuracy_updates", 1), ("timed_updates", 5),
    ("rounds", 5), ("warmup_updates", 1), ("host_cache_bytes", 0),
    ("device_cache_bytes", True), ("dtype", "float32"), ("quant", "none"),
    ("lora_targets", []), ("lora_targets", ["expert_proj"]),
    ("lora_targets", "q_proj"), ("lora_targets", [""]),
    ("lora_targets", ["q_proj", "q_proj"]),
    ("host_cache_policy", "fifo"), ("host_cache_policy", None),
])
def test_invalid_spec_is_refused(field: str, value) -> None:
    spec = _spec()
    spec[field] = value
    with pytest.raises(ValueError):
        runner.validate_spec(spec)


def _mock_run(tmp_path: Path, monkeypatch, *, corruption: str | None = None):
    import torch

    spec_path = tmp_path / "spec.json"
    atomic_json(spec_path, _spec())
    events = []
    gates = []
    verify_calls = []
    real_compare = runner.compare_captures

    def verify(_spec, *, content):
        verify_calls.append(content)

    def compare(reference, candidate):
        result = real_compare(reference, candidate)
        gates.append(result["passed"])
        return result

    def fake_worker(_path, _run_spec, arm, phase, out):
        events.append((phase, arm))
        out.mkdir(parents=True)
        result = {"status": "passed", "phase": phase, "arm": arm,
                  "spec_sha256": sha256_file(spec_path)}
        if phase == "accuracy":
            if corruption == "failure" and arm == "cache":
                raise RuntimeError("simulated worker failure")
            capture = _capture()
            if corruption == "short_all":
                capture["microbatches"] = capture["microbatches"][:2]
                capture["updates"] = capture["updates"][:1]
            if arm == "cache":
                if corruption == "input":
                    capture["microbatches"][0]["input_ids"][0, 0] += 1
                elif corruption == "init":
                    capture["initial_adapters"]["adapter_a"][0] += 1e-4
                elif corruption == "gradient":
                    capture["microbatches"][0]["gradients"]["adapter_a"][0] += 0.1
            capture_path = out / "capture.pt"
            torch.save(capture, capture_path)
            result["accuracy"] = {
                "capture_sha256": sha256_file(capture_path),
                "runtime_before": _cache_counters(arm, before=True),
                "runtime_after": _cache_counters(arm, before=False),
            }
            if corruption == "capture_hash" and arm == "cache":
                result["accuracy"]["capture_sha256"] = "0" * 64
            if corruption in {"bad_ratio", "no_host_hit", "no_device_hit"}:
                after = result["accuracy"]["runtime_after"]["expert_cache"]
                if corruption == "bad_ratio" and arm == "two_read":
                    after["logical_fetches"] += 1
                    after["storage_reads"] += 1
                elif corruption == "no_host_hit" and arm == "cache":
                    after["host"]["hits"] = 0
                elif corruption == "no_device_hit" and arm == "cache":
                    after["device"]["hits"] = 0
        else:
            # Every one of the three candidate gates must pass before ANY timing.
            assert gates == [True, True, True]
            assert events[:4] == [("accuracy", name) for name in ARMS]
            result["timing"] = {"update_seconds": [1.0 if arm == "ordinary" else 0.9] * 6}
        atomic_json(out / "result.json", result)
        return result

    monkeypatch.setattr(runner, "verify_spec", verify)
    monkeypatch.setattr(runner, "compare_captures", compare)
    monkeypatch.setattr(runner, "launch_worker", fake_worker)
    return spec_path, events, gates, verify_calls


def test_orchestration_checks_all_four_captures_before_balanced_fresh_process_timing(
    tmp_path: Path, monkeypatch,
) -> None:
    spec_path, events, gates, verifies = _mock_run(tmp_path, monkeypatch)
    out = tmp_path / "run"
    result = runner.run(spec_path, out)
    assert result["status"] == "completed" and result["accuracy_passed"]
    assert gates == [True, True, True]
    assert events[:4] == [("accuracy", arm) for arm in ARMS]
    assert events[4:] == [("timing", arm) for order in runner.ORDERS for arm in order]
    assert len(result["timing_blocks"]) == 16
    assert verifies == [True, True, True]
    retained = json.loads((out / "run.json").read_text())
    assert retained["status"] == "completed" and not retained["gpu_rented_by_runner"]


@pytest.mark.parametrize("corruption", [
    "input", "init", "gradient", "failure", "capture_hash", "short_all",
    "bad_ratio", "no_host_hit", "no_device_hit",
])
def test_accuracy_failure_preserves_failure_report_and_never_launches_timing(
    tmp_path: Path, monkeypatch, corruption: str,
) -> None:
    spec_path, events, _gates, _verifies = _mock_run(tmp_path, monkeypatch, corruption=corruption)
    out = tmp_path / "run"
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(spec_path, out)
    assert all(phase == "accuracy" for phase, _arm in events)
    retained = json.loads((out / "run.json").read_text())
    assert retained["status"] == "failed" and retained["timing_blocks"] == []
    assert "error" in retained


def test_accuracy_only_mode_does_not_implicitly_start_timing(tmp_path: Path, monkeypatch) -> None:
    spec_path, events, _gates, _verifies = _mock_run(tmp_path, monkeypatch)
    result = runner.run(spec_path, tmp_path / "run", accuracy_only=True)
    assert result["status"] == "accuracy_passed"
    assert events == [("accuracy", arm) for arm in ARMS]
    assert result["timing_blocks"] == []


@pytest.mark.parametrize("field,bad", [
    ("status", "running"), ("phase", "timing"), ("arm", "two_read"),
    ("spec_sha256", "another-spec"),
])
def test_launcher_rejects_success_from_wrong_worker_identity(
    tmp_path: Path, monkeypatch, field: str, bad: str,
) -> None:
    spec_path = tmp_path / "spec.json"
    atomic_json(spec_path, _spec())
    out = tmp_path / "worker"
    monkeypatch.setattr(runner, "verify_spec", lambda *_args, **_kwargs: None)

    def fake_process(command, **kwargs):
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert command[1:3] == ["-m", "benchmarks.harness.two_read_cache_worker"]
        payload = {"status": "passed", "phase": "accuracy", "arm": "ordinary",
                   "spec_sha256": sha256_file(spec_path)}
        payload[field] = bad
        atomic_json(out / "result.json", payload)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_process)
    with pytest.raises(ValueError, match="belongs to another run"):
        runner.launch_worker(spec_path, _spec(), "ordinary", "accuracy", out)
    assert (out / "result.json").exists()
    assert not (out / "controller-result.json").exists()


def test_launcher_preserves_a_nonzero_worker_failure(tmp_path: Path, monkeypatch) -> None:
    spec_path = tmp_path / "spec.json"
    atomic_json(spec_path, _spec())
    out = tmp_path / "worker"
    monkeypatch.setattr(runner, "verify_spec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=7,
    ))
    with pytest.raises(RuntimeError, match="failed \\(7\\)"):
        runner.launch_worker(spec_path, _spec(), "ordinary", "accuracy", out)
    assert (out / "worker.log").exists()
