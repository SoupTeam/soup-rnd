"""Evidence review must retain failed tensors and exclude invalid ABBA rounds."""

import json

import pytest


def test_saved_tensor_replay_rejects_a_real_mismatch(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from benchmarks.harness.d2_replay_tensors import replay

    path = tmp_path / "mismatch.safetensors"
    safetensors.save_file({
        "fast/Y/Y": torch.tensor([2.0], dtype=torch.bfloat16),
        "PEFT/Y/Y": torch.tensor([1.0], dtype=torch.bfloat16),
        "float64/Y/Y": torch.tensor([1.0], dtype=torch.float64),
    }, str(path))
    result = replay(path, "bf16", expected_keys={"Y/Y"})
    assert result["passed"] is False
    assert result["rows"][0]["fast_float64_max_abs_error"] == 1.0
    assert result["rows"][0]["proposed_max_error_bound"] == 1e-8


def test_replay_rejects_gradient_omitted_from_all_arms(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from benchmarks.harness.d2_replay_tensors import replay

    path = tmp_path / "single-dx0.safetensors"
    safetensors.save_file({
        f"{arm}/Y/Y": torch.ones(2, dtype=torch.float64 if arm == "float64" else torch.bfloat16)
        for arm in ("fast", "PEFT", "float64")
    }, str(path))
    with pytest.raises(AssertionError, match="missing expected"):
        replay(path, "bf16")


@pytest.mark.parametrize("fault", [None, "missing_arm", "peer", "clock_change", "no_power"])
def test_timing_review_excludes_faults_without_promoting_verdict(tmp_path, fault):
    from benchmarks.harness.d2_audit_evidence import audit_timing

    rows = []
    for index, implementation in enumerate(("PEFT", "fast", "fast", "PEFT"), 1):
        rows.append({
            "path": "qkv", "request_dX": True, "round": 1, "arm_index": index,
            "implementation": implementation, "correctness_passed": True,
            "forward_ms": 2.0, "backward_ms": 3.0,
            "validity": {
                "status": "UNVERIFIED", "available": True, "own_pid": 123,
                "target_uuid": "abc", "peers": [],
                "gpu_samples": [["0", "GPU-abc", "595.71.05", "1785", "P2", "100.0"]],
                "process_samples": [["GPU-abc", "123", "python", "700"]],
            },
        })
    if fault == "missing_arm":
        rows.pop()
    elif fault == "peer":
        rows[0]["validity"]["process_samples"].append(["GPU-abc", "456", "python", "400"])
    elif fault == "clock_change":
        rows[0]["validity"]["gpu_samples"][0][3] = "1980"
    elif fault == "no_power":
        rows[0]["validity"]["gpu_samples"][0][5] = "N/A"
    path = tmp_path / "timing.json"
    path.write_text(json.dumps({"dtype": "bf16", "rows": rows, "memory_scope": "absolute"}))
    result = audit_timing(path)
    assert result["verdict"].startswith("NO VERDICT")
    assert result["rounds"][0]["sampled_telemetry_passed"] is (fault is None)
    assert len(result["rounds"]) == 1
    assert result["summaries"][2]["included_rounds"] == int(fault is None)
