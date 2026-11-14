"""The Kanana launcher ``oc_arm_run.py`` on the tiny DeepSeek-V3, on CPU.

Runs the real ``prepare``, ``run`` and ``judge`` commands on the tiny native checkpoint of
``b13_moe_fixture`` (random weights, fp32). A card, NF4 and the packed levers are not covered:
they are measured on the rented host. What is checked here is the launcher itself: the arms switch
the levers they claim, the reads per hit of every arm match the tiny table of ticket 03, the
record has the fields the protocol lists, and ``judge`` passes pairs that are equal and fails a
pair that is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness import oc_arm_run
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

K = 2
UPDATES = 3
SEQ = 6


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """One tiny checkpoint and one shared ``per_expert`` cache for every arm."""
    root = tmp_path_factory.mktemp("oc")
    weights = fixture.moe_weights_dir(root, fixture.moe_config())
    cache = root / "cache"
    args = oc_arm_run.build_parser().parse_args(
        ["prepare", "--source", str(weights), "--cache", str(cache), "--device", "cpu"])
    oc_arm_run.prepare(args)
    return {"root": root, "weights": weights, "cache": cache}


def run_args(stand: Dict[str, Path], arm: str, *, capture: bool, out: Path) -> Any:
    argv = [
        "run", "--arm", arm, "--k", str(K), "--source", str(stand["weights"]),
        "--cache", str(stand["cache"]), "--out", str(out), "--device", "cpu",
        "--updates", str(UPDATES), "--warmup", "1", "--seq", str(SEQ), "--lr", "1e-2",
        "--lora-r", "2", "--lora-alpha", "4", "--targets", *fixture.TARGETS, "--tier", "ram",
        "--hits-probe", "--revision", "tiny",
    ]
    if capture:
        argv.append("--capture")
    return oc_arm_run.build_parser().parse_args(argv)


_RECORDS: Dict[str, Dict[str, Any]] = {}


def record_of(stand: Dict[str, Path], arm: str) -> Dict[str, Any]:
    """Each arm runs once for the module, with a capture, in the folder ``<arm>-k<K>``."""
    if arm not in _RECORDS:
        out = stand["root"] / "runs" / f"{arm}-k{K}"
        _RECORDS[arm] = oc_arm_run.run_arm(run_args(stand, arm, capture=True, out=out))
    return _RECORDS[arm]


# ---------------------------------------------------------------- levers
def test_ordinary_arms_switch_no_lever() -> None:
    for arm in ("ga_ordinary", "l2l_ordinary", "ga_floor"):
        assert oc_arm_run.levers_for(arm, device="cuda", host_cache_bytes=8 << 30) == ({}, [])


def test_complete_on_a_card_switches_all_four_levers() -> None:
    levers, off = oc_arm_run.levers_for("ga_complete", device="cuda", host_cache_bytes=8 << 30)
    assert levers == {
        "two_read_experts": True, "grouped_expert_routes": True,
        "expert_transfer": "packed", "expert_host_cache_bytes": 8 << 30,
    }
    assert off == []


def test_complete_on_cpu_says_which_levers_it_leaves_off() -> None:
    levers, off = oc_arm_run.levers_for("l2l_complete", device="cpu", host_cache_bytes=8 << 30)
    assert levers == {"two_read_experts": True, "grouped_expert_routes": True}
    assert off == ["expert_transfer", "expert_host_cache_bytes"]


def test_a_zero_host_cache_leaves_the_cache_lever_out() -> None:
    levers, _ = oc_arm_run.levers_for("ga_complete", device="cuda", host_cache_bytes=0)
    assert "expert_host_cache_bytes" not in levers


def test_an_unknown_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        oc_arm_run.levers_for("ga_cache", device="cpu", host_cache_bytes=0)


def test_windows_repeat_across_calls_and_differ_across_updates() -> None:
    import torch

    first = oc_arm_run.window_ids(64, SEQ, 0, K)
    again = oc_arm_run.window_ids(64, SEQ, 0, K)
    later = oc_arm_run.window_ids(64, SEQ, 1, K)
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert not torch.equal(first[0], later[0])
    assert first[0].shape == (1, SEQ)


# ---------------------------------------------------------------- run
@pytest.mark.parametrize(
    ("arm", "reads_per_hit"),
    [("ga_ordinary", 3.0), ("ga_complete", 2.0), ("l2l_ordinary", 3.0), ("l2l_complete", 3.0)],
)
@_REQUIRES_TRAINING_FLOOR
def test_reads_per_hit_match_the_tiny_table(
    stand: Dict[str, Path], arm: str, reads_per_hit: float
) -> None:
    record = record_of(stand, arm)
    assert record["reads_per_hit"] == reads_per_hit
    assert record["hits_window0"] > 0


@_REQUIRES_TRAINING_FLOOR
def test_the_record_has_the_protocol_fields(stand: Dict[str, Path]) -> None:
    record = record_of(stand, "ga_complete")
    for field in (
        "arm", "mode", "k", "kind", "median_step_seconds", "measured_step_seconds",
        "setup_seconds", "run_seconds", "expert_fetches_per_window", "layer_loads_per_window",
        "rss_bytes", "host_cache_bytes", "versions", "repo_commit", "model_revision",
        "levers_applied", "levers_off_on_this_device", "tokens_per_second",
        "vram_peak_allocated_bytes",
    ):
        assert field in record, field
    assert record["kind"] == "synthetic"
    assert record["model_revision"] == "tiny"
    assert len(record["measured_step_seconds"]) == UPDATES - 1  # one warm-up update
    assert record["median_step_seconds"] > 0
    assert record["vram_peak_allocated_bytes"] is None  # CPU
    assert record["levers_off_on_this_device"] == ["expert_transfer", "expert_host_cache_bytes"]
    written = json.loads(
        (stand["root"] / "runs" / f"ga_complete-k{K}" / "record.json").read_text("utf-8"))
    assert written["median_step_seconds"] == record["median_step_seconds"]


@_REQUIRES_TRAINING_FLOOR
def test_layer_loads_follow_the_mode(stand: Dict[str, Path]) -> None:
    ga, l2l = record_of(stand, "ga_ordinary"), record_of(stand, "l2l_ordinary")
    assert set(ga["layer_loads_per_window"]) == {3 * K}  # three layers, once per micro-batch
    assert set(l2l["layer_loads_per_window"]) == {3}  # once per window


@_REQUIRES_TRAINING_FLOOR
def test_a_run_folder_is_never_reused(stand: Dict[str, Path]) -> None:
    record_of(stand, "ga_ordinary")
    again = run_args(stand, "ga_ordinary", capture=False,
                     out=stand["root"] / "runs" / f"ga_ordinary-k{K}")
    with pytest.raises(FileExistsError):
        oc_arm_run.run_arm(again)


# ---------------------------------------------------------------- judge
@_REQUIRES_TRAINING_FLOOR
def test_judge_passes_the_pairs_and_reports_bit_equality(stand: Dict[str, Path]) -> None:
    for arm in ("ga_ordinary", "ga_complete", "l2l_ordinary", "l2l_complete"):
        record_of(stand, arm)
    args = oc_arm_run.build_parser().parse_args(
        ["judge", "--runs", str(stand["root"] / "runs"), "--k", str(K)])
    verdict = oc_arm_run.judge(args)
    assert verdict["mode"] == "fp32"
    assert verdict["all_pass"], verdict["pairs"]
    pairs = verdict["pairs"]
    assert pairs["ga_complete_vs_ga_ordinary"]["bit_equal"] is True
    assert pairs["l2l_complete_vs_l2l_ordinary"]["bit_equal"] is True
    assert (stand["root"] / "runs" / f"accuracy-k{K}.json").is_file()


def _capture(scale: float) -> Dict[str, Any]:
    import torch

    grads = [{"w": torch.tensor([1.0, 2.0, 3.0]) * scale} for _ in range(2)]
    return {"lr": 1e-2, "losses": [1.0 * scale, 2.0 * scale], "grads": grads,
            "params": {"w": torch.ones(3)}, "adam": {}}


def test_judge_pair_fails_a_gradient_outside_the_fp32_limit() -> None:
    ref, cand = oc_arm_run._as_run(_capture(1.0)), oc_arm_run._as_run(_capture(1.01))
    result = oc_arm_run.judge_pair(ref, cand, mode="fp32", floor_grad=None, floor_loss=None)
    assert not result["pass"]
    assert {row["what"] for row in result["failing"]} == {"gradient", "loss"}


def test_judge_pair_passes_equal_runs() -> None:
    ref = oc_arm_run._as_run(_capture(1.0))
    result = oc_arm_run.judge_pair(ref, oc_arm_run._as_run(_capture(1.0)), mode="fp32",
                                   floor_grad=None, floor_loss=None)
    assert result["pass"] and result["failing"] == []


def test_judge_pair_bf16_needs_a_floor() -> None:
    ref, cand = oc_arm_run._as_run(_capture(1.0)), oc_arm_run._as_run(_capture(1.0))
    result = oc_arm_run.judge_pair(ref, cand, mode="bf16", floor_grad=None, floor_loss=None)
    assert result["judged"] is False and result["pass"] is False


def test_judge_pair_bf16_limit_is_one_and_a_half_times_the_floor() -> None:
    ref = oc_arm_run._as_run(_capture(1.0))
    floor = [0.01, 0.01]
    inside = oc_arm_run._as_run(_capture(1.0 + 0.014 / 1.0))  # 1.4 x the floor
    outside = oc_arm_run._as_run(_capture(1.0 + 0.016))  # 1.6 x the floor
    kwargs: List[Any] = dict(mode="bf16", floor_grad=floor, floor_loss=floor)
    assert oc_arm_run.judge_pair(ref, inside, **kwargs)["pass"]
    assert not oc_arm_run.judge_pair(ref, outside, **kwargs)["pass"]
