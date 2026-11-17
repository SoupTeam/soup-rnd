"""Reject forged PASS summaries when retained trainer payloads disagree."""

import json

import pytest

from benchmarks.harness.d3_fused_nf4_verify import verify_matrix


def matrix_fixture():
    files, cells = {}, []
    for codec in ("nf4", "nf4-dq"):
        for l2l in (0, 1):
            for mode in ("sequential", "fused_nf4"):
                case = f"matrix-01/{codec}-{l2l}-{mode}"
                cells.append({"case": case, "status": "passed"})
                row = {"status": "passed", "final_parameters": {"factor": "same"},
                       "runtime": {"expert_gemm": {"fused_matrices": 1}},
                       "reload_comparison": {"actual_dtype": "bf16", "wanted_dtype": "bf16",
                                             "actual_sha256": "same", "wanted_sha256": "same"}}
                files[case + "/status.json"] = json.dumps(row).encode()
                for name in ("gradients-0.safetensors", "gradients-1.safetensors",
                             "trained-logits.safetensors"):
                    files[case + "/" + name] = b"same actual payload"
    files["matrix-01/status.json"] = json.dumps({"status": "passed", "cells": cells}).encode()
    return files


@pytest.mark.parametrize("payload", ["gradients-0.safetensors", "gradients-1.safetensors",
                                     "trained-logits.safetensors"])
def test_false_pass_cannot_hide_changed_actual_trainer_payload(payload):
    files = matrix_fixture()
    assert verify_matrix(files, "") == 8
    files[f"matrix-01/nf4-dq-1-fused_nf4/{payload}"] = b"different payload"
    with pytest.raises(ValueError, match="actual gradient/logit"):
        verify_matrix(files, "")


def test_false_pass_cannot_hide_inactive_fusion_or_reload_dtype_change():
    files = matrix_fixture()
    name = "matrix-01/nf4-0-fused_nf4/status.json"
    row = json.loads(files[name])
    row["runtime"]["expert_gemm"]["fused_matrices"] = 0
    files[name] = json.dumps(row).encode()
    with pytest.raises(ValueError, match="inactive"):
        verify_matrix(files, "")
    row["runtime"]["expert_gemm"]["fused_matrices"] = 1
    row["reload_comparison"]["actual_dtype"] = "fp32"
    files[name] = json.dumps(row).encode()
    with pytest.raises(ValueError, match="reload mismatch"):
        verify_matrix(files, "")
