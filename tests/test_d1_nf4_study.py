"""D1 evidence must reject incomplete, contaminated and vacuous comparisons."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness"
sys.path.insert(0, str(HARNESS))


def rules():
    import d1_rules

    return d1_rules


def test_empty_evidence_cannot_close_d1():
    report = rules().summarize([])
    assert report["status"] == "NOT_RUN"
    assert report["d1_complete"] is False


def arm(name, run, value=10.0, outcome="ok", **changes):
    row = {
        "arm": name,
        "block": 0,
        "run": run,
        "median_step_ms": value,
        "source_sha": "source",
        "rule_sha256": "rule",
        "case": "q-128-bfloat16",
        "gpu_uuid": "GPU-one",
        "settings_sha256": "settings",
        "path": "fused" if name == "B" else "dense",
        "validity": {"outcome": outcome},
        "clock_sm_mhz": [1740, 1740],
        "peak_allocated_bytes": 1234,
        "peak_reserved_bytes": 2345,
        "peers_clean": True,
        "power_limit_mw": [350000, 350000],
        "memory_util_pct": [40, 50],
    }
    row.update(changes)
    return row


def clean_blocks(count=5):
    rows = []
    for block in range(count):
        for run, name in enumerate("ABBA"):
            rows.append(arm(name, run, block=block, median_step_ms=10 if name == "A" else 9))
    return rows


def test_projection_results_alone_are_never_full_d1_acceptance():
    report = rules().summarize(clean_blocks())
    assert report["status"] == "MEASURED_SUBSET"
    assert report["d1_complete"] is False
    assert report["comparisons"][0]["ratio_a_over_b"] == pytest.approx(10 / 9)


@pytest.mark.parametrize("outcome", ["void", "unknown"])
def test_one_contaminated_arm_invalidates_its_whole_abba_block(outcome):
    rows = clean_blocks()
    rows[1]["validity"]["outcome"] = outcome
    comparison = rules().summarize(rows)["comparisons"][0]
    assert comparison["accepted_blocks"] == 4
    assert comparison["status"] == "NO_VERDICT"
    assert comparison["ratio_a_over_b"] is None
    assert comparison["rejected_blocks"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_uuid", "GPU-other"),
        ("source_sha", "other-source"),
        ("rule_sha256", "other-rule"),
        ("settings_sha256", "other-settings"),
        ("clock_sm_mhz", []),
        ("clock_sm_mhz", [1000, 1740]),
        ("median_step_ms", float("nan")),
        ("median_step_ms", 0),
        ("peers_clean", None),
        ("power_limit_mw", []),
        ("power_limit_mw", [300000, 350000]),
        ("memory_util_pct", []),
    ],
)
def test_mixed_identity_missing_sensors_and_invalid_timings_are_rejected(field, value):
    rows = clean_blocks()
    rows[0][field] = value
    comparison = rules().summarize(rows)["comparisons"][0]
    assert comparison["status"] == "NO_VERDICT"
    assert comparison["ratio_a_over_b"] is None


def test_missing_or_reordered_run_cannot_be_relabelled_as_abba():
    rows = clean_blocks()
    rows[0]["arm"] = "B"
    assert rules().summarize(rows)["comparisons"][0]["status"] == "NO_VERDICT"
    assert rules().summarize(clean_blocks()[:-1])["comparisons"][0]["status"] == "NO_VERDICT"


def test_same_kernel_comparison_is_labelled_as_such():
    rows = clean_blocks()
    for row in rows:
        row["path"] = "dense"
    comparison = rules().summarize(rows)["comparisons"][0]
    assert comparison["status"] == "SAME_PATH"
    assert comparison["speedup_claim"] is False


def test_duplicate_records_are_not_extra_repeats():
    with pytest.raises(ValueError, match="duplicate"):
        rules().summarize(clean_blocks() + [clean_blocks()[0]])


def test_evidence_writer_preserves_previous_attempts(tmp_path):
    path = tmp_path / "arm.json"
    rules().write_new_json(path, {"status": "failed"})
    with pytest.raises(FileExistsError):
        rules().write_new_json(path, {"status": "passed"})
    assert json.loads(path.read_text())["status"] == "failed"


@pytest.mark.parametrize("name", ["../outside.pt", "C:/outside.pt", "/outside.pt"])
def test_replay_refuses_paths_outside_evidence_root(tmp_path, name):
    with pytest.raises(ValueError, match="containment"):
        rules().evidence_path(tmp_path, name)


def test_tensor_replay_checks_every_expected_quantity_and_detects_nonfinite():
    torch = pytest.importorskip("torch")
    reference = {"logits": torch.ones(2), "grad_A": torch.ones(3)}
    candidate = {key: value.clone() for key, value in reference.items()}
    assert rules().compare_tensors(candidate, reference)["passed"] is True
    candidate.pop("grad_A")
    assert rules().compare_tensors(candidate, reference)["passed"] is False
    candidate["grad_A"] = torch.tensor([1.0, float("nan"), 1.0])
    assert rules().compare_tensors(candidate, reference)["passed"] is False


def test_negative_control_must_change_a_finite_gradient():
    torch = pytest.importorskip("torch")
    expected = {"grad": torch.ones(3)}
    assert rules().negative_control(expected, expected) is False
    assert rules().negative_control({"grad": torch.tensor([1.0, 2.0, 1.0])}, expected)
    assert not rules().negative_control({"grad": torch.full((3,), float("nan"))}, expected)


def test_peer_identity_comes_from_empty_before_and_after_not_a_random_single_pid():
    def sample(pids):
        return {"available": True, "uuid": "GPU-one", "compute": pids, "graphics": []}

    assert rules().peer_check(sample([]), [sample([101]), sample([101])], sample([]), "GPU-one")
    assert not rules().peer_check(sample([101]), [sample([101])], sample([]), "GPU-one")
    assert not rules().peer_check(sample([]), [sample([101]), sample([102])], sample([]), "GPU-one")
    assert not rules().peer_check(sample([]), [], sample([]), "GPU-one")
    assert not rules().peer_check(sample([]), [sample([101])], sample([]), "GPU-other")


def test_harness_import_does_not_load_heavy_dependencies():
    import subprocess

    script = HARNESS / "d1_nf4_study.py"
    code = (
        "import runpy,sys; runpy.run_path(sys.argv[1],run_name='import_check'); "
        "assert not ({'torch','transformers','peft','bitsandbytes'} & set(sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code, str(script)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_rules_import_spec_exists():
    assert importlib.util.find_spec("d1_rules") is not None


def test_invalid_training_or_ablation_plan_is_rejected_before_a_gpu_child():
    from d1_nf4_study import parser, validate_args

    args = parser().parse_args(["run", "--kind", "train", "--output", "unused"])
    with pytest.raises(ValueError, match="weights.*data"):
        validate_args(args)
    args = parser().parse_args(
        ["run", "--kind", "ablation", "--weights", "weights", "--output", "unused"]
    )
    with pytest.raises(ValueError, match="baseline-nodequant"):
        validate_args(args)


def test_gpu_uuid_prefix_normalization_does_not_accept_another_gpu():
    from d1_rules import canonical_gpu_uuid

    value = "0acc0e0b-b6d2-6310-47c7-922123d0dc8c"
    assert canonical_gpu_uuid(value) == canonical_gpu_uuid("GPU-" + value)
    assert canonical_gpu_uuid(value) != canonical_gpu_uuid("1acc0e0b-b6d2-6310-47c7-922123d0dc8c")
    with pytest.raises(ValueError):
        canonical_gpu_uuid("not-a-GPU")


def test_all_block_reads_bound_includes_own_io_and_rejects_sensor_gaps():
    from d1_rules import all_block_reads_check

    start = {"nvme0n1": 100, "nvme0n1p1": 100}
    assert all_block_reads_check(start, {"nvme0n1": 200, "nvme0n1p1": 200})["outcome"] == "ok"
    # This counts physical disks and partitions; even own I/O is not subtracted.
    assert (
        all_block_reads_check(start, {"nvme0n1": 600000100, "nvme0n1p1": 600000100})["outcome"]
        == "void"
    )
    assert all_block_reads_check(start, {"nvme0n1": 200})["outcome"] == "unknown"
    assert all_block_reads_check({}, {})["outcome"] == "unknown"
    assert all_block_reads_check(start, {"nvme0n1": 99, "nvme0n1p1": 200})["outcome"] == "unknown"
