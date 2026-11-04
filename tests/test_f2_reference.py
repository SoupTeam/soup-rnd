"""F2 campaign safety and evidence gates; no model download or GPU work."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness"
sys.path.insert(0, str(HARNESS))

import f2_reference as f2  # noqa: E402


def valid_evidence():
    return {
        "returncode": 0,
        "validity": {"outcome": "ok"},
        "audit": {
            "complete": True,
            "physical_vram_bytes": 8 * 1024**3,
            "fingerprint_first": "before",
            "fingerprint_counted_first": "after_warmup",
            "fingerprint_last": "after",
            "steps": [
                {"grad_norm": 1.0, "useful_tokens": 100, "total_tokens": 128}
                for _ in range(f2.STEPS)
            ],
            "peak_allocated_bytes": 3 * 1024**3,
            "peak_reserved_bytes": 4 * 1024**3,
        },
        "telemetry_complete": True,
        "tier_confirmed": True,
        "adapter_saved": True,
        "bench": {"valid": True, "checks": {"parameters_changed": True}},
    }


def test_order_is_abba_and_keeps_five_refusals():
    runs = f2.schedule()
    assert len(runs) == 75
    for length in f2.LENGTHS:
        arms = [arm for arm in runs if arm[1] == length]
        assert arms == [
            (mode, length) for _ in range(5) for mode in ("ram", "nvme", "nvme", "ram", "refusal")
        ]


@pytest.mark.parametrize("kind", ["train", "bench"])
def test_accepts_actual_updates(kind):
    assert f2.judge(valid_evidence(), kind, "ram")[0] == "PASS"


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("fingerprint_last", "before"),
        ("physical_vram_bytes", 24 * 1024**3),
        ("peak_reserved_bytes", 9 * 1024**3),
    ],
)
def test_rejects_incomplete_updates_large_gpu_and_spill(field, value):
    evidence = valid_evidence()
    evidence["audit"][field] = value
    assert f2.judge(evidence, "bench", "ram")[0] != "PASS"


@pytest.mark.parametrize("norm", [None, 0, float("nan"), float("inf")])
def test_all_counted_gradients_must_be_finite_and_nonzero(norm):
    evidence = valid_evidence()
    evidence["audit"]["steps"][-1]["grad_norm"] = norm
    assert f2.judge(evidence, "bench", "ram")[0] != "PASS"


@pytest.mark.parametrize("key", ["telemetry_complete", "tier_confirmed", "adapter_saved"])
def test_missing_required_evidence_is_no_verdict(key):
    evidence = valid_evidence()
    evidence[key] = False
    assert f2.judge(evidence, "train", "ram")[0] == "NO_VERDICT"


@pytest.mark.parametrize("outcome", ["void", "unknown"])
def test_contaminated_arm_never_counts(outcome):
    evidence = valid_evidence()
    evidence["validity"]["outcome"] = outcome
    assert f2.judge(evidence, "bench", "ram")[0] == "NO_VERDICT"


def test_generic_error_is_not_expected_refusal():
    evidence = valid_evidence()
    evidence.update(returncode=1, log="CUDA out of memory")
    assert f2.judge(evidence, "bench", "refusal")[0] == "FAIL"
    evidence["log"] = "layer streaming needs NVMe or more RAM"
    assert f2.judge(evidence, "bench", "refusal")[0] == "EXPECTED_REFUSAL"
    evidence["returncode"] = 0
    assert f2.judge(evidence, "bench", "refusal")[0] == "FAIL"


def test_early_stop_and_missing_tokens_are_rejected():
    evidence = valid_evidence()
    evidence["audit"]["steps"].pop()
    assert f2.judge(evidence, "bench", "ram")[0] == "FAIL"
    evidence = valid_evidence()
    evidence["audit"]["steps"][-1]["useful_tokens"] = 0
    assert f2.judge(evidence, "bench", "ram")[0] == "FAIL"


def test_preparation_has_nine_unmeasured_cells(tmp_path):
    f2.prepare(tmp_path / "campaign", tmp_path / "snapshot", tmp_path / "cache")
    plan = json.loads((tmp_path / "campaign" / "plan.json").read_text())
    assert len(plan["cells"]) == 9
    assert plan["matrix_approved"] is False
    assert plan["checkpoint_revision"] == f2.REVISION
    assert not (tmp_path / "snapshot").exists()
    assert not (tmp_path / "cache").exists()
    assert all(row["status"] == "NOT_RUN" for row in plan["cells"])
    assert len(list((tmp_path / "campaign" / "configs").glob("*.json"))) == 9
    assert "NOT_RUN" in (tmp_path / "campaign" / "matrix.md").read_text()


def test_existing_campaign_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        f2.prepare(tmp_path, tmp_path / "snapshot", tmp_path / "cache")


def test_unapproved_campaign_cannot_freeze(tmp_path):
    root = tmp_path / "campaign"
    f2.prepare(root, tmp_path / "snapshot", tmp_path / "cache")
    with pytest.raises(ValueError, match="approval"):
        f2.freeze(root)
    assert not (root / "frozen.json").exists()


def test_report_does_not_turn_unexecuted_cells_into_failures(tmp_path):
    root = tmp_path / "campaign"
    f2.prepare(root, tmp_path / "snapshot", tmp_path / "cache")
    f2.report(root)
    assert (root / "matrix.md").read_text().count("| NOT_RUN |") == 9


def test_trace_samples_must_overlap_counted_training_window():
    rows = [
        {
            "perf_counter": 9,
            "gpu": "20, 2000, 3000, 40, 115, P0, 60",
            "rss_bytes": 100,
            "read_bytes": 10,
            "write_bytes": 20,
        }
    ]
    audit = {"counted_window_started": 10, "counted_window_ended": 20}
    assert not f2.trace_summary(rows, audit)["complete"]
    rows = [dict(rows[0], perf_counter=stamp) for stamp in (11, 14, 17, 19)]
    assert f2.trace_summary(rows, audit)["complete"]
    rows[0]["gpu"] = "[N/A], 2000, 3000, 40, 115, P0, 60"
    assert not f2.trace_summary(rows, audit)["complete"]


def test_sparse_trace_is_incomplete():
    rows = [
        {
            "perf_counter": stamp,
            "gpu": "20, 2000, 3000, 40, 115, P0, 60",
            "rss_bytes": 100,
            "read_bytes": 10,
            "write_bytes": 20,
        }
        for stamp in (11, 19)
    ]
    assert not f2.trace_summary(rows, {"counted_window_started": 10, "counted_window_ended": 20})[
        "complete"
    ]


def test_tier_evidence_must_be_the_actual_ready_line():
    ram = "Layer streaming ready: 48 layers, 9.93 GB pinned RAM store"
    disk = (
        "Layer streaming ready: 48 layers, streamed from DISK (9.93 GB on an NVMe volume) "
        "by an async reader, read_ahead=2, 300 MB pinned host staging"
    )
    assert f2.tier_confirmed("ram", ram)
    assert f2.tier_confirmed("nvme", disk)
    assert not f2.tier_confirmed("nvme", ram + " detected NVMe")
    assert not f2.tier_confirmed("ram", ram.replace("pinned", "pageable"))
    assert not f2.tier_confirmed("nvme", disk.replace("pinned", "pageable"))


def test_native_bench_report_cannot_be_missing():
    evidence = valid_evidence()
    evidence["bench"] = None
    assert f2.judge(evidence, "bench", "ram")[0] == "FAIL"


def test_warmup_only_updates_do_not_validate_counted_window():
    evidence = valid_evidence()
    evidence["audit"]["fingerprint_counted_first"] = "after"
    assert f2.judge(evidence, "bench", "ram")[0] == "FAIL"


def test_bad_numbers_are_preserved_as_strings_not_invalid_json(tmp_path):
    path = tmp_path / "failure.json"
    f2.write_json(path, {"grad_norm": float("nan")})
    assert json.loads(path.read_text())["grad_norm"] == "nan"


@pytest.mark.parametrize("kind", ["train", "bench"])
def test_observer_runs_real_cli_and_sees_actual_updates(tmp_path, kind):
    """A tiny local model exercises both CLI paths; no Hub or GPU is used."""
    for name in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(name)
    from tests.test_issue341_seed_and_fullft import _ROWS, _tiny_llama_dir

    base = _tiny_llama_dir(tmp_path)
    data = tmp_path / "data.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}
        for user, answer in list(_ROWS) * 3
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = tmp_path / "config.json"
    f2.write_json(
        config,
        {
            "base": base,
            "task": "sft",
            "output": str(tmp_path / "adapter"),
            "data": {
                "train": str(data),
                "format": "chatml",
                "chat_template": "chatml",
                "max_length": 64,
                "val_split": 0,
            },
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "quantization": "none",
                "lr": 0.01,
                "logging_steps": 1,
                "lora": {"r": 4, "alpha": 8, "dropout": 0, "target_modules": ["q_proj"]},
            },
        },
    )
    audit = tmp_path / "audit.json"
    cli = ["train", "--config", str(config), "--yes"]
    if kind == "bench":
        cli = [
            "bench",
            "train",
            "--config",
            str(config),
            "--steps",
            "6",
            "--warmup",
            "2",
            "--output",
            str(tmp_path / "bench.json"),
        ]
    env = dict(
        os.environ,
        PYTHONPATH=str(HARNESS.parents[1] / "src"),
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, str(HARNESS / "f2_cli_observe.py"), "--audit", str(audit), "--", *cli],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(audit.read_text())
    assert observed["complete"]
    assert observed["fingerprint_first"]
    assert observed["fingerprint_last"] != observed["fingerprint_first"]
    assert observed["fingerprint_last"] != observed["fingerprint_counted_first"]
    assert len(observed["steps"]) >= 6
    assert all(step["useful_tokens"] > 0 for step in observed["steps"])
    if kind == "bench":
        assert json.loads((tmp_path / "bench.json").read_text())["valid"]
    else:
        assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
