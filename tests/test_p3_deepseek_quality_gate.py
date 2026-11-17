"""Task 1.3 numerical integration runs on approved native-BF16 Ampere CUDA only."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import p3_deepseek_quality_gate as gate


def _window() -> dict[str, Any]:
    ids = [position % 64 for position in range(4096)]
    labels = list(ids)
    labels[0] = -100
    labels[17] = -100
    labels[70:75] = [-100] * 5
    return {
        "window_id": "fixture/window-000", "common_boundary_mask_interval": 1024,
        "micro_batches": [
            {"input_ids": [ids[start:start + 1024]], "labels": [labels[start:start + 1024]]}
            for start in range(0, 4096, 1024)
        ],
    }


def test_prefix_masks_local_boundaries_and_recounts_shifted_targets() -> None:
    # Reusing the 1024-window denominator or retaining a cross-chunk target fails this.
    inputs, labels, identity = gate.gate_prefix(_window())
    assert len(inputs) == len(labels) == 8
    assert all(len(row) == 1 and len(row[0]) == 64 for row in inputs + labels)
    assert [row[0][0] for row in labels] == [-100] * 8
    assert labels[0][0][17] == -100
    assert labels[1][0][6:11] == [-100] * 5
    assert identity["input_tokens"] == 512
    assert identity["valid_target_count"] == 498
    assert identity["per_microbatch_targets"] == [62, 58, 63, 63, 63, 63, 63, 63]
    assert _window()["micro_batches"][0]["labels"][0][64] == 0


@pytest.mark.parametrize("bad", ["short", "no_targets", "wrong_boundary", "label_shape"])
def test_bad_prefix_cannot_be_silently_repacked(bad: str) -> None:
    window = _window()
    if bad == "short":
        window["micro_batches"][0]["input_ids"][0] = [1] * 511
    elif bad == "no_targets":
        window["micro_batches"][0]["labels"][0] = [-100] * 1024
    elif bad == "wrong_boundary":
        window["common_boundary_mask_interval"] = 64
    else:
        window["micro_batches"][0]["labels"] = [[1]]
    with pytest.raises(ValueError):
        gate.gate_prefix(window)


def _zero_metric() -> dict[str, Any]:
    return {
        "rows": 512, "rms_row_rel": 0.0, "worst_row_rel": 0.0,
        "rel_l2": 0.0, "max_abs": 0.0, "max_abs_over_absmax": 0.0,
        "err_sq": 0.0, "ref_sq": 1.0, "row_rel_sq_sum": 0.0,
    }


def _metric_input() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layers = [{
        "layer": index, "restore": _zero_metric(), "branch": _zero_metric(),
        "output": _zero_metric(),
        "pairs": [{"microbatch": mb, "rows": 64, "finite": True, "forward_equal": True}
                  for mb in range(8)],
    } for index in range(3)]
    gradients = {
        "tensors": 2, "global_rel_l2": 0.0, "min_cosine": 1.0, "max_rel_l2": 0.0,
        "per_tensor": {name: {
            "cosine": 1.0, "rel_l2": 0.0, "reference_finite": True,
            "candidate_finite": True, "shape_equal": True,
        } for name in ("attention.lora_A", "attention.lora_B")},
    }
    return layers, gradients


def _judge(layers: list[dict[str, Any]], gradients: dict[str, Any]) -> dict[str, Any]:
    return gate.assess_metrics(
        layers, gradients, expected_layers=[0, 1, 2],
        expected_tensors=["attention.lora_A", "attention.lora_B"],
    )


@pytest.mark.parametrize("bad", [
    "missing_layer", "duplicate_layer", "nan_layer", "nan_tensor", "missing_tensor",
    "false_finite", "missing_pair", "wrong_minimum", "nan_global", "short_rows",
])
def test_incomplete_or_nonfinite_metrics_never_read_pass(bad: str) -> None:
    # A max/min reduction must not hide NaN or silently shrink its coverage denominator.
    layers, gradients = _metric_input()
    assert {row["status"] for row in _judge(layers, gradients).values()} == {"PASS"}
    if bad == "missing_layer":
        layers.pop()
    elif bad == "duplicate_layer":
        layers[-1]["layer"] = 1
    elif bad == "nan_layer":
        layers[-1]["branch"]["rms_row_rel"] = float("nan")
    elif bad == "nan_tensor":
        gradients["per_tensor"]["attention.lora_A"]["rel_l2"] = float("nan")
    elif bad == "missing_tensor":
        del gradients["per_tensor"]["attention.lora_B"]
    elif bad == "false_finite":
        layers[1]["pairs"][4]["finite"] = False
    elif bad == "missing_pair":
        layers[2]["pairs"].pop()
    elif bad == "wrong_minimum":
        gradients["per_tensor"]["attention.lora_B"]["cosine"] = 0.5
    elif bad == "nan_global":
        gradients["global_rel_l2"] = float("nan")
    else:
        layers[0]["restore"]["rows"] = 511
    assert {row["status"] for row in _judge(layers, gradients).values()} == {"NO_VERDICT"}


def test_unchanged_g4_bounds_are_inclusive_and_control_is_not_forced_to_fail() -> None:
    layers, gradients = _metric_input()
    layers[2]["restore"].update(rms_row_rel=0.05, worst_row_rel=0.25)
    layers[1]["branch"]["rms_row_rel"] = 0.10
    gradients.update(global_rel_l2=0.05, min_cosine=0.99)
    gradients["per_tensor"]["attention.lora_B"]["cosine"] = 0.99
    gates = _judge(layers, gradients)
    assert all(value["status"] == "PASS" for value in gates.values())
    discrimination = gate.assess_discrimination(gates)
    assert discrimination["status"] == "NO_VERDICT"
    assert discrimination["not_failed"] == ["E1", "E2", "E3"]
    layers[2]["restore"]["rms_row_rel"] = 0.050001
    layers[1]["branch"]["rms_row_rel"] = 0.100001
    gradients["global_rel_l2"] = 0.050001
    gates = _judge(layers, gradients)
    assert all(value["status"] == "FAIL" for value in gates.values())
    assert gate.assess_discrimination(gates)["status"] == "PASS"


def test_missing_independent_floor_is_no_verdict_even_for_bitwise_gradients() -> None:
    import torch

    gradients = {"attention.lora_A": torch.tensor([1.0, 2.0])}
    result = gate.assess_a3(
        gradients, gradients, gradients, losses=[0.5] * 8,
        repeat_losses=[0.5] * 8, l2l_losses=[0.5] * 8, dtype="bfloat16",
    )
    assert result["status"] == "NO_VERDICT"
    assert result["rule"]["status"] == "no_floor"
    assert result["rule"]["l2l_vs_ga"]["equal"] is True


def test_a3_uses_independent_backend_floor_factor_and_exact_repeat() -> None:
    import torch

    reference = {"attention.lora_A": torch.tensor([1.0])}
    candidate = {"attention.lora_A": torch.tensor([1.125])}
    floor = {"attention.lora_A": torch.tensor([1.125])}
    result = gate.assess_a3(
        reference, reference, candidate, losses=[1.0] * 8,
        repeat_losses=[1.0] * 8, l2l_losses=[1.0] * 8, dtype="bfloat16",
        floor_gradients=floor, floor_losses=[1.0] * 8, floor_observed=True,
    )
    assert result["status"] == "PASS"
    assert result["rule"]["tolerance"]["grad_limit"] == [0.1875]
    bad = gate.assess_a3(
        reference, candidate, reference, losses=[1.0] * 8,
        repeat_losses=[1.0] * 8, l2l_losses=[1.0] * 8, dtype="bfloat16",
        floor_gradients=floor, floor_losses=[1.0] * 8, floor_observed=True,
    )
    assert bad["status"] == "NO_VERDICT"
    assert bad["rule"]["status"] == "a_vs_a_failed"


def test_a3_unobserved_or_nonfinite_floor_cannot_relax_exact_ga() -> None:
    import torch

    values = {"attention.lora_A": torch.tensor([1.0])}
    floor = {"attention.lora_A": torch.tensor([float("nan")])}
    unobserved = gate.assess_a3(
        values, values, values, losses=[1.0] * 8, repeat_losses=[1.0] * 8,
        l2l_losses=[1.0] * 8, dtype="bfloat16", floor_gradients=values,
        floor_losses=[1.0] * 8, floor_observed=False,
    )
    assert unobserved["status"] == "NO_VERDICT"
    broken = gate.assess_a3(
        values, values, values, losses=[1.0] * 8, repeat_losses=[1.0] * 8,
        l2l_losses=[1.0] * 8, dtype="bfloat16", floor_gradients=floor,
        floor_losses=[1.0] * 8, floor_observed=True,
    )
    assert broken["status"] == "NO_VERDICT"


@pytest.mark.parametrize("changed", ["gradient", "loss", "loss_cancellation"])
def test_fp32_qualification_refuses_nonbitwise_error_below_a3_tolerance(changed: str) -> None:
    import torch

    reference = {"attention.lora_A": torch.tensor([1.0])}
    candidate = {name: value.clone() for name, value in reference.items()}
    losses = [1.0] * 8
    if changed == "gradient":
        candidate["attention.lora_A"][0] += 2.0**-16
    else:
        losses[0] += 2.0**-16
        if changed == "loss_cancellation":
            losses[1] -= 2.0**-16
    result = gate.assess_a3(
        reference, reference, candidate, losses=[1.0] * 8, repeat_losses=[1.0] * 8,
        l2l_losses=losses, dtype="float32",
    )
    assert result["status"] == "FAIL"
    assert result["bitwise_qualification"]["equal"] is False
    assert result["rule"]["pass"] is True
    assert result["rule"]["tolerance"]["grad_rel"][0] < 1e-4
    assert result["rule"]["tolerance"]["loss_rel"][0] < 1e-4


def test_fp32_qualification_requires_exact_microbatch_losses_and_every_named_gradient() -> None:
    import torch

    reference = {"attention.lora_A": torch.tensor([1.0]),
                 "attention.lora_B": torch.tensor([2.0])}
    result = gate.assess_a3(
        reference, reference, reference, losses=[0.5] * 8, repeat_losses=[0.5] * 8,
        l2l_losses=[0.5] * 8, dtype="float32",
    )
    assert result["status"] == "PASS"
    assert result["bitwise_qualification"] == {
        "equal": True, "per_microbatch_losses_equal": True, "gradients_equal": True,
    }


def test_fp32_qualification_refuses_microbatch_repeat_cancellation() -> None:
    import torch

    gradients = {"attention.lora_A": torch.tensor([1.0])}
    repeated = [1.0 + 2.0**-16, 1.0 - 2.0**-16, *([1.0] * 6)]
    result = gate.assess_a3(
        gradients, gradients, gradients, losses=[1.0] * 8, repeat_losses=repeated,
        l2l_losses=[1.0] * 8, dtype="float32",
    )
    assert result["status"] == "NO_VERDICT"
    assert result["rule"]["a_vs_a"]["equal"] is True


@pytest.fixture(scope="module")
def ampere_cuda() -> str:
    from benchmarks.harness.p3_deepseek_quality_attention import require_ampere_cuda

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require_ampere_cuda("cuda")
    return "cuda"


@pytest.fixture(scope="module")
def tiny_source(tmp_path_factory: Any, ampere_cuda: str) -> Path:
    import torch

    from benchmarks.harness.b13_moe_fixture import moe_config, moe_weights_dir

    root = tmp_path_factory.mktemp("quality-gate-native")
    config = moe_config(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=3, first_k_dense_replace=1, q_lora_rank=16, kv_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=1024, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        return moe_weights_dir(root, config, seed=61)


def _run_native(
    tmp_path: Path, source: Path, dtype: str, device: str = "cuda", *, publish: bool = True,
) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from benchmarks.harness.p3_deepseek_quality_model import open_session, write_initial_adapter

    with open_session(
        source, tmp_path / "cache", device=device, dtype=dtype, prepare=True,
    ) as session:
        adapter = tmp_path / "initial.safetensors"
        write_initial_adapter(session.model, adapter, seed=23)
    del session
    gc.collect()
    report = gate.run_gate(
        source=source, cache=tmp_path / "cache", initial_adapter=adapter,
        window=_window(), output=tmp_path / "evidence", dtype=dtype, device=device,
        fixture=True,
    )
    if publish and os.environ.get("P3_QUALITY_EVIDENCE") and device == "cuda":
        destination = Path(os.environ["P3_QUALITY_EVIDENCE"]) / f"gate-{dtype}-{device}"
        shutil.copytree(tmp_path / "evidence", destination)
    return report


@pytest.mark.gpu
def test_native_fp32_l2l_matches_ordinary_streamed_ga_bitwise(
    tmp_path: Path, tiny_source: Path,
) -> None:
    # A changed accumulation denominator, input order, or copied rather than real GA fails here.
    report = _run_native(tmp_path, tiny_source, "float32")
    assert report["status"] == "COMPLETE"
    assert report["a3"]["status"] == "PASS"
    assert report["a3"]["rule"]["a_vs_a"]["equal"]
    assert report["a3"]["rule"]["l2l_vs_ga"]["equal"]
    assert report["verdict"] == "NO_VERDICT"
    assert report["qualification_only"] is True
    for name in ("ga", "ga_repeat", "bf16_l2l"):
        row = report["passes"][name]
        assert row["optimizer_applied"] is False
        assert row["adapter_unchanged"] is True
        assert row["valid_target_count"] == 498
        assert row["initial_adapter_sha256"] == hashlib.sha256(
            (tmp_path / "initial.safetensors").read_bytes()
        ).hexdigest()
        assert row["backend_requested"] == "EFFICIENT_ATTENTION"
        for sample in row["attention"]["samples"]:
            assert "aten::_scaled_dot_product_efficient_attention" in (
                sample["forward"]["operators"]
            )
            assert "aten::_scaled_dot_product_efficient_attention_backward" in (
                sample["backward"]["operators"]
            )
            assert any("fmha_cutlassF" in kernel
                       for kernel in sample["forward"]["cuda_kernel_names"])
            assert any("fmha_cutlassB" in kernel
                       for kernel in sample["backward"]["cuda_kernel_names"])
    assert report["passes"]["fp8_oracle"]["status"] == "NOT_APPLICABLE"
    assert report["determinism"]["restored"] is True
    assert report["determinism"]["execution"]["warn_only"] is False
    assert report["cublas_environment_restored"] is True
    assert report["attention_policy"]["sha256"] == (
        "fc887c5a996bd58536ef0325b61405cd73a2a83cf363fcc4156204ab180e1ed6"
    )


@pytest.mark.gpu
def test_native_bf16_measures_every_layer_tensor_and_real_oracle_vjp(
    tmp_path: Path, tiny_source: Path,
) -> None:
    # Missing replay VJPs, skipped decoder layers, or aggregate-only finite checks fail here.
    report = _run_native(tmp_path, tiny_source, "bfloat16")
    assert report["status"] == "COMPLETE", report.get("error")
    assert report["qualification_only"] is True
    for name in ("fp8_oracle", "int4_oracle"):
        row = report["passes"][name]
        if name == "int4_oracle" and row["status"] == "NOT_APPLICABLE":
            assert any(value["status"] != "PASS"
                       for value in report["candidate_gates"].values())
            assert report["stopped_after"] == "fp8_oracle"
            continue
        assert row["oracle"]["expected_pairs"] == 16
        assert row["oracle"]["counts"] == {
            "native_forward": 16, "oracle_recompute": 16, "oracle_vjp": 16,
        }
        assert row["oracle"]["complete"] is True
        assert row["oracle"]["natural_routing_measured"] is False
        assert row["routing_guard"] == "production_L2LStep_enabled"
        assert row["adapter_unchanged"] is True
        assert gate._oracle_complete(row, 16) is True
        missing_vjp = copy.deepcopy(row)
        missing_vjp["oracle"]["counts"]["oracle_vjp"] -= 1
        assert gate._oracle_complete(missing_vjp, 16) is False
        assert [layer["layer"] for layer in row["layers"]] == [0, 1, 2]
        for layer in row["layers"]:
            assert len(layer["pairs"]) == 8
            for kind in ("restore", "branch", "output"):
                assert layer[kind]["rows"] == 512
                assert layer[kind]["rms_row_rel"] >= 0
                assert all(math.isfinite(value) for value in layer[kind].values())
        assert set(row["gradient_agreement"]["per_tensor"]) == set(report["trainable_names"])
        assert all(value["reference_finite"] and value["candidate_finite"]
                   for value in row["gradient_agreement"]["per_tensor"].values())
        assert all(math.isfinite(value[metric])
                   for value in row["gradient_agreement"]["per_tensor"].values()
                   for metric in ("rel_l2", "cosine"))
        assert math.isfinite(row["gradient_agreement"]["global_rel_l2"])
        assert math.isfinite(row["gradient_agreement"]["min_cosine"])
    control = report["control"]["gates"]
    not_failed = [name for name in ("E1", "E2", "E3") if control[name]["status"] != "FAIL"]
    assert report["control"]["discrimination"]["not_failed"] == not_failed
    assert report["control"]["discrimination"]["status"] == (
        "NO_VERDICT" if not_failed else "PASS"
    )
    assert report["passes"]["ga"]["losses"] == report["passes"]["ga_repeat"]["losses"]
    assert report["determinism"]["restored"] is True
    assert report["determinism"]["execution"]["warn_only"] is False
    assert report["environment"]["gpu"]["native_bf16"] is True
    from benchmarks.harness import p3_deepseek_quality_train as training

    with training._deterministic_run():
        execution = training._execution_identity("cuda")
    training.require_gate_environment(report["environment"], execution["gate_environment"])
    assert report["cublas_environment_restored"] is True
    assert report["floor"]["observed"] is True
    assert report["floor"]["backend"] == "MATH"
    assert report["floor"]["main_backend"] == "EFFICIENT_ATTENTION"
    assert report["a3"]["status"] == "PASS"
    assert report["a3"]["independent_floor_observed"] is True
    for name, row in report["passes"].items():
        if row["status"] == "NOT_APPLICABLE":
            assert name == "int4_oracle"
            continue
        assert row["valid_target_count"] == 498
        assert row["optimizer_applied"] is False
        assert row["backend_requested"] == (
            "MATH" if name == "ga_floor" else "EFFICIENT_ATTENTION"
        )
        assert row["attention"]["executed_sdpa_calls"] >= 24
        assert all(call["dtype"] == "torch.bfloat16" and call["device"] == "cuda"
                   and call["query_shape"][-2] == call["key_shape"][-2] == 64
                   and call["value_shape"][-2] == 64
                   for call in row["attention"]["unique_calls"])
        expected = row["backend_requested"]
        for sample in row["attention"]["samples"]:
            if expected == "EFFICIENT_ATTENTION":
                assert "aten::_scaled_dot_product_efficient_attention" in (
                    sample["forward"]["operators"]
                )
                assert "aten::_scaled_dot_product_efficient_attention_backward" in (
                    sample["backward"]["operators"]
                )
                assert any("fmha_cutlassF" in kernel
                           for kernel in sample["forward"]["cuda_kernel_names"])
                assert any("fmha_cutlassB" in kernel
                           for kernel in sample["backward"]["cuda_kernel_names"])
            else:
                assert "aten::_scaled_dot_product_attention_math" in (
                    sample["forward"]["operators"]
                )
                assert {"aten::bmm", "aten::_softmax_backward_data"} <= set(
                    sample["backward"]["operators"]
                )
            assert sample["forward"]["cuda_kernel_names"]
            assert sample["backward"]["cuda_kernel_names"]
    manifest = json.loads((tmp_path / "evidence" / "manifest.json").read_text("utf-8"))
    assert manifest["verdict"] in {"FAIL", "NO_VERDICT"}
    for relative, descriptor in manifest["files"].items():
        assert hashlib.sha256((tmp_path / "evidence" / relative).read_bytes()).hexdigest() == (
            descriptor["sha256"]
        )


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_cpu_cannot_execute_native_gate_even_with_fixture_permission(
    tmp_path: Path, dtype: str,
) -> None:
    # Fixture access must not resurrect CPU/MATH numerical qualification.
    report = gate.run_gate(
        source=tmp_path / "missing", cache=tmp_path / "cache",
        initial_adapter=tmp_path / "missing.safetensors", window=_window(),
        output=tmp_path / "evidence", device="cpu", dtype=dtype, fixture=True,
    )
    assert report["status"] == "FAILED"
    assert report["passes"] == {}
    assert report["optimizer_updates"] == 0
    assert "CUDA" in report["error"]["message"]
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("workspace", [None, ":4096:8", "incompatible"])
def test_gate_failure_restores_cublas_environment_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: str | None,
) -> None:
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    report = gate.run_gate(
        source=tmp_path / "missing", cache=tmp_path / "cache",
        initial_adapter=tmp_path / "missing.safetensors", window=_window(),
        output=tmp_path / "evidence", device="cpu", fixture=True,
    )
    assert report["status"] == "FAILED"
    assert report["passes"] == {}
    if workspace == "incompatible":
        assert "CUBLAS_WORKSPACE_CONFIG" in report["error"]["message"]
    else:
        assert "CUDA" in report["error"]["message"]
    assert report["cublas_environment_restored"] is True
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == workspace


def test_hardware_rejection_precedes_source_and_cache_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def refuse_hardware(_device: str) -> dict[str, Any]:
        raise ValueError("native-BF16 Ampere CUDA required")

    monkeypatch.setattr(gate, "require_ampere_cuda", refuse_hardware)
    report = gate.run_gate(
        source=tmp_path / "missing", cache=tmp_path / "cache",
        initial_adapter=tmp_path / "missing.safetensors", window=_window(),
        output=tmp_path / "evidence", device="cuda", fixture=True,
    )
    assert report["status"] == "FAILED" and report["passes"] == {}
    assert report["error"] == {"type": "ValueError", "message": "native-BF16 Ampere CUDA required"}
    assert report["optimizer_updates"] == 0
    assert not (tmp_path / "cache").exists()


def test_existing_evidence_is_never_reused_or_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        gate.run_gate(
            source=tmp_path / "missing", cache=tmp_path / "cache",
            initial_adapter=tmp_path / "missing.safetensors", window=_window(),
            output=output, device="cpu", fixture=True,
        )
    assert sentinel.read_text("utf-8") == "unchanged"
    assert not (output / "manifest.json").exists()


@pytest.mark.gpu
def test_failed_source_keeps_failure_manifest_and_never_passes(
    tmp_path: Path, ampere_cuda: str,
) -> None:
    report = gate.run_gate(
        source=tmp_path / "missing", cache=tmp_path / "cache",
        initial_adapter=tmp_path / "missing.safetensors", window=_window(),
        output=tmp_path / "evidence", device=ampere_cuda, fixture=True,
    )
    assert report["status"] == "FAILED"
    assert report["verdict"] == "NO_VERDICT"
    assert report["error"]["type"] == "FileNotFoundError"
    assert (tmp_path / "evidence" / "failure.json").is_file()
    assert json.loads((tmp_path / "evidence" / "manifest.json").read_text("utf-8"))["status"] == (
        "FAILED"
    )


def test_overall_cannot_admit_missing_floor_or_uninformative_control() -> None:
    candidate = {name: {"status": "PASS"} for name in ("E1", "E2", "E3")}
    assert gate.overall_verdict(
        {"status": "NO_VERDICT"}, candidate, {"status": "PASS"}, eligible=True,
    ) == "NO_VERDICT"
    assert gate.overall_verdict(
        {"status": "PASS"}, candidate, {"status": "NO_VERDICT"}, eligible=True,
    ) == "NO_VERDICT"
    assert gate.overall_verdict(
        {"status": "PASS"}, candidate, {"status": "PASS"}, eligible=True,
    ) == "PASS"
    broken = copy.deepcopy(candidate)
    broken["E2"]["status"] = "FAIL"
    assert gate.overall_verdict(
        {"status": "PASS"}, broken, {"status": "PASS"}, eligible=True,
    ) == "FAIL"


@pytest.mark.gpu
def test_loader_refuses_fixture_receipt_even_when_numerics_are_complete(
    tmp_path: Path, tiny_source: Path,
) -> None:
    _run_native(tmp_path, tiny_source, "float32", publish=False)
    path = tmp_path / "evidence" / "manifest.json"
    with pytest.raises(ValueError, match="production|PASS|qualification"):
        gate.load_gate(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_loader_rejects_hash_mismatch_before_reading_claimed_pass(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"status":"COMPLETE","verdict":"PASS"}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        gate.load_gate(path, "0" * 64)


def test_gradient_metrics_keep_frozen_values_and_named_finite_coverage() -> None:
    import torch

    reference = {"attention.lora_A": torch.tensor([3.0, 4.0])}
    candidate = {"attention.lora_A": torch.tensor([3.0, 4.5])}
    result = gate.gradient_agreement(reference, candidate)
    assert result["global_rel_l2"] == pytest.approx(0.1)
    assert result["per_tensor"]["attention.lora_A"]["reference_finite"] is True
    candidate["attention.lora_A"][1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        gate.gradient_agreement(reference, candidate)


def test_layer_archive_reads_one_original_pair_and_refuses_mutation(tmp_path: Path) -> None:
    import torch

    original = gate._LayerArchive(tmp_path / "bf16", 1)
    inputs = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
    original.observe(0, 0, "forward", inputs, inputs * 2)
    candidate = gate._LayerArchive(tmp_path / "fp8", 1, reference=original)
    candidate.observe(0, 0, "forward", inputs, inputs * 2)
    restored = inputs + 0.125
    candidate.observe(0, 0, "recompute", restored, restored * 2)
    assert candidate.table()[0]["restore"]["rows"] == 2
    assert candidate.table()[0]["branch"]["rms_row_rel"] > 0
    artifact = Path(original.forward[(0, 0)]["artifact"]["path"])
    artifact.write_bytes(b"corrupted immutable boundary")
    with pytest.raises(ValueError, match="hash"):
        original.load((0, 0))


def test_cli_has_no_fixture_or_precision_bypass(capsys: Any) -> None:
    with pytest.raises(SystemExit) as error:
        gate.main(["--fixture"])
    assert error.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_nonfinite_evidence_is_named_and_never_serialized_as_nan(tmp_path: Path) -> None:
    report = gate._publish(tmp_path, {
        "schema": gate.SCHEMA, "status": "COMPLETE", "verdict": "PASS",
        "qualification_only": False, "passes": {}, "measurement": float("nan"),
    })
    assert report["verdict"] == "NO_VERDICT"
    assert report["measurement"] is None
    assert report["nonfinite_evidence_fields"] == ["$.measurement"]
    parsed = json.loads((tmp_path / "result.json").read_text("utf-8"))
    assert parsed["measurement"] is None


def test_layer_archive_refuses_nonfinite_tensor_without_claiming_coverage(tmp_path: Path) -> None:
    import torch

    archive = gate._LayerArchive(tmp_path, 1)
    inputs = torch.ones((1, 64, 2))
    inputs[0, 9, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        archive.observe(0, 0, "forward", inputs, inputs)
    assert archive.forward == {}
    assert archive.table()[0]["pairs"] == []


def test_receipt_hashes_every_listed_artifact_before_claimed_numeric_admission(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.json"
    result.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": gate.SCHEMA, "status": "COMPLETE", "verdict": "PASS",
        "qualification_only": False,
        "files": {"result.json": {"bytes": 2, "sha256": "0" * 64},
                  "gate-input.json": {"bytes": 2, "sha256": "0" * 64}},
    }), encoding="utf-8")
    (tmp_path / "gate-input.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        gate.load_gate(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())


@pytest.mark.gpu
def test_nonfinite_executed_ga_stops_with_failure_and_releases_session(
    tmp_path: Path, tiny_source: Path,
) -> None:
    import torch

    from benchmarks.harness.p3_deepseek_quality_model import (
        load_initial_adapter,
        open_session,
        sha256,
        trainable_parameters,
        write_initial_adapter,
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with open_session(
        tiny_source, tmp_path / "cache", device="cuda", dtype="float32", prepare=True,
    ) as session:
        adapter = tmp_path / "initial.safetensors"
        write_initial_adapter(session.model, adapter)
        load_initial_adapter(session.model, adapter)
        parameter = next(iter(trainable_parameters(session.model).values()))
        handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        raw_ids, raw_labels, identity = gate.gate_prefix(_window())
        inputs = [torch.tensor(row, device="cuda") for row in raw_ids]
        labels = [torch.tensor(row, device="cuda") for row in raw_labels]
        try:
            row, gradients, _ = gate._run_pass(
                session, name="ga", initial_adapter=adapter, initial_sha256=sha256(adapter),
                ids=inputs, labels=labels, targets=identity["valid_target_count"],
                directory=tmp_path / "ga", device="cuda", dtype="float32",
                host_budget_bytes=32 * 1024**2,
            )
        finally:
            handle.remove()
        assert row["status"] == "FAILED"
        assert gradients == {}
        assert any(not check["finite"] for check in row["gradient_checks"].values())
        assert row["optimizer_applied"] is False and row["adapter_unchanged"] is True
        assert row["host_budget_after_close_bytes"] == 0
        assert row["cleanup_errors"] == []
    assert session.closed is True


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "a/../b", "a\\b", "a:b"])
def test_artifact_paths_cannot_escape_evidence_root(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="path"):
        gate._child(tmp_path, relative)


@pytest.mark.parametrize("layers,gradient", [(None, {}), ([], None), ([{}], {"per_tensor": {}})])
def test_malformed_metric_records_cannot_raise_or_read_pass(layers: Any, gradient: Any) -> None:
    result = gate.assess_metrics(
        layers, gradient, expected_layers=[0], expected_tensors=["attention.lora_A"],
    )
    assert {row["status"] for row in result.values()} == {"NO_VERDICT"}


def _attention_receipt(backend: str) -> dict[str, Any]:
    """Serialized validator fixture only; public admission always refuses this evidence."""
    call = {
        "dtype": "torch.bfloat16", "device": "cuda",
        "query_shape": [1, 2, 64, 16], "key_shape": [1, 2, 64, 16],
        "value_shape": [1, 2, 64, 8],
        "query_dtype": "torch.bfloat16", "key_dtype": "torch.bfloat16",
        "value_dtype": "torch.bfloat16", "query_device": "cuda",
        "key_device": "cuda", "value_device": "cuda",
        "device_index": 0, "key_device_index": 0, "value_device_index": 0,
    }
    efficient = backend == "EFFICIENT_ATTENTION"
    return {
        "backend_requested": backend, "executed_sdpa_calls": 8, "unique_calls": [call],
        "deterministic_flags": {
            "algorithms_enabled": True, "warn_only": False, "allow_tf32": False,
            "cudnn_allow_tf32": False, "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "backend_flags": {"efficient": efficient, "math": not efficient,
                          "flash": False, "cudnn": False},
        "hardware": {
            "device": "cuda:0", "name": "serialized Ampere fixture", "capability": [8, 0],
            "native_bf16": True, "bf16_supported": True, "cuda_version": "12.8",
            "native_bf16_check": "torch.cuda.is_bf16_supported(including_emulation=False)",
        },
        "instrumentation": {
            "scope": "bounded_attention_operators", "max_signatures": 16,
            "retains_tensors": False, "full_window_trace": False,
            "max_profiled_operations": 80, "profiled_operations": 2,
            "signature_limit_exceeded": False, "profiling_errors": [],
        },
        "samples": [{
            **copy.deepcopy(call),
            "forward": {
                "operators": ["aten::_scaled_dot_product_efficient_attention" if efficient
                              else "aten::_scaled_dot_product_attention_math"],
                "cuda_kernel_names": ["fmha_cutlassF_bf16_aligned_32x128_gmem_sm80"
                                      if efficient else "ampere_bf16_s16816gemm_bf16_128x64_nn"],
            },
            "backward": {
                "operators": ["aten::_scaled_dot_product_efficient_attention_backward"]
                if efficient else ["aten::bmm", "aten::_softmax_backward_data"],
                "cuda_kernel_names": ["fmha_cutlassB_bf16_aligned_128x64_k65536_sm80"
                                      if efficient else "softmax_backward_cuda_kernel"],
            },
        }],
    }


def _numeric_receipt(root: Path) -> Path:
    """Hand-derived CPU tensor receipt, not native codec or production GPU evidence."""
    import torch

    targets = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
    names = sorted(f"model.layers.0.self_attn.{target}.lora_{factor}.default.weight"
                   for target in targets for factor in ("A", "B"))
    passes = {}
    for arm in ("ga", "ga_repeat", "bf16_l2l", "ga_floor", "fp8_oracle", "int4_oracle"):
        factor = 1.125 if arm == "ga_floor" else 2.0 if arm == "int4_oracle" else 1.0
        values = {name: torch.tensor([factor, factor * 2]) for name in names}
        folder = root / arm
        folder.mkdir(parents=True)
        with (folder / "gradients.pt").open("wb") as stream:
            torch.save(values, stream)
        passes[arm] = {
            "status": "COMPLETE", "optimizer_applied": False, "adapter_unchanged": True,
            "adapter_before_sha256": "a" * 64, "adapter_after_sha256": "a" * 64,
            "initial_adapter_sha256": "b" * 64, "valid_target_count": 504,
            "host_budget_after_close_bytes": 0, "cleanup_errors": [], "losses": [1.0] * 8,
            "diagnostic_errors": [],
            "runtime_load_deltas": {key: 1 for key in (
                "layer_loads", "large_loads", "expert_fetches", "expert_bytes_read",
            )},
            "gradient_checks": {name: {"present": True, "finite": True, "nonzero": True,
                                       "shape": [2]} for name in names},
            "routing_guard": "production_L2LStep_enabled",
            "backend_requested": "MATH" if arm == "ga_floor" else "EFFICIENT_ATTENTION",
            "attention": _attention_receipt(
                "MATH" if arm == "ga_floor" else "EFFICIENT_ATTENTION"
            ),
        }

    def errors(relative: float, maximum: float, reference: float) -> dict[str, Any]:
        # All 512 rows are equal, so the pooled and rowwise relative errors coincide.
        return {
            "rows": 512, "rms_row_rel": relative, "worst_row_rel": relative,
            "rel_l2": relative, "max_abs": maximum, "max_abs_over_absmax": relative,
            "err_sq": reference * relative**2, "ref_sq": reference,
            "row_rel_sq_sum": 512 * relative**2,
        }

    for mb in range(8):
        filename = f"layer-000-mb-{mb:03d}.pt"
        for arm, phase, input_value, output_value in (
            ("bf16_l2l", "forward", 1.0, 2.0),
            ("fp8_oracle", "recompute", 1.0, 2.0),
            ("int4_oracle", "recompute", 1.5, 4.0),
        ):
            folder = root / arm / phase
            folder.mkdir(exist_ok=True)
            values = {"input": torch.full((1, 64, 2), input_value, dtype=torch.bfloat16),
                      "output": torch.full((1, 64, 2), output_value, dtype=torch.bfloat16)}
            with (folder / filename).open("wb") as stream:
                torch.save(values, stream)
    for arm in ("fp8_oracle", "int4_oracle"):
        coarse = arm == "int4_oracle"
        passes[arm]["layers"] = [{
            "layer": 0,
            "pairs": [{"microbatch": mb, "rows": 64, "finite": True, "forward_equal": True}
                      for mb in range(8)],
            "restore": errors(0.5 if coarse else 0.0, 0.5 if coarse else 0.0, 1024.0),
            "branch": errors(1.5 if coarse else 0.0, 1.5 if coarse else 0.0, 1024.0),
            "output": errors(1.0 if coarse else 0.0, 2.0 if coarse else 0.0, 4096.0),
        }]
        error = 1.0 if coarse else 0.0
        passes[arm]["gradient_agreement"] = {
            "tensors": 10, "global_rel_l2": error, "min_cosine": 1.0, "max_rel_l2": error,
            "worst": [{"name": name, "cosine": 1.0, "rel_l2": error} for name in names[:5]],
            "per_tensor": {name: {"cosine": 1.0, "rel_l2": error, "reference_finite": True,
                                   "candidate_finite": True, "shape_equal": True}
                           for name in names},
        }
        passes[arm]["oracle"] = {
            "complete": True, "expected_pairs": 8,
            "counts": {key: 8 for key in ("native_forward", "oracle_recompute", "oracle_vjp")},
            "forced_routing": True, "natural_routing_measured": False,
            "records_remaining": 0, "pending_replays": 0, "failure": None, "cleanup_errors": [],
        }
        passes[arm]["oracle_after_close"] = {
            "closed": True, "host_reserved_bytes": 0, "records_remaining": 0,
            "pending_replays": 0, "owned_host_storage_bytes": 0, "owned_device_storage_bytes": 0,
            "cleanup": {"original_forwards_restored": True, "hooks_removed": True},
            "cleanup_errors": [],
        }

    def gates(coarse: bool) -> dict[str, Any]:
        return {
            "E1": {"status": "FAIL" if coarse else "PASS", "layers_over": [0] if coarse else [],
                   "max_rms_row_rel": 0.5 if coarse else 0.0,
                   "max_worst_row_rel": 0.5 if coarse else 0.0,
                   "rms_bound": 0.05, "worst_row_bound": 0.25},
            "E2": {"status": "FAIL" if coarse else "PASS", "layers_over": [0] if coarse else [],
                   "max_branch_rms_row_rel": 1.5 if coarse else 0.0, "bound": 0.1,
                   "semantics": "one executed Oracle recompute residual branch "
                   "from restored input"},
            "E3": {"status": "FAIL" if coarse else "PASS",
                   "global_rel_l2": 1.0 if coarse else 0.0, "min_cosine": 1.0, "tensors": 10,
                   "global_rel_l2_bound": 0.05, "min_cosine_bound": 0.99},
        }

    comparison = {
        "grads_equal": True, "losses_equal": True, "rates_equal": True, "steps_equal": True,
        "steps": [1, 1], "max_abs": 0.0, "worst": [], "equal": True,
    }
    result = {
        "schema": gate.SCHEMA, "status": "COMPLETE", "verdict": "PASS",
        "qualification_only": True, "fixture": "hand-derived CPU numerical-verifier receipt",
        "attention_policy": gate.policy_identity(),
        "cublas_environment_restored": True,
        "determinism": {"restored": True, "execution": {
            "algorithms_enabled": True, "warn_only": False, "allow_tf32": False,
            "cudnn_allow_tf32": False, "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        }},
        "environment": {"cublas_workspace_config": ":4096:8", "gpu": {
            "name": "serialized Ampere fixture", "capability": [8, 0],
            "bf16_supported": True, "native_bf16": True,
        }},
        "initial_adapter_sha256": "b" * 64, "input": {"per_microbatch_targets": [63] * 8},
        "trainable_names": names, "layers": [0], "passes": passes,
        "model": {"cache": {"n_layers": 1},
                  "source": {"shape": {"first_k_dense_replace": 0, "hidden_size": 2}}},
        "floor": {"observed": True, "backend": "MATH",
                  "main_backend": "EFFICIENT_ATTENTION", "factor": 1.5},
        "a3": {
            "status": "PASS", "independent_floor_observed": True,
            "optimizer_updates": 0, "comparison_observations": 1,
            "rule": {"a_vs_a": comparison, "l2l_vs_ga": comparison,
                     "pass": True, "note": "", "status": "within_tolerance",
                     "tolerance": {
                         "mode": "bf16", "factor": 1.5, "limit_fp32": 1e-4,
                         "grad_rel": [0.0], "loss_rel": [0.0], "floor_grad": [0.125],
                         "floor_loss": [0.0], "judged": True, "failing": [], "pass": True,
                         "floor_problem": None, "grad_limit": [0.1875], "loss_limit": [0.0],
                     }},
        },
        "candidate_gates": gates(False),
        "control": {"gates": gates(True), "discrimination": {
            "status": "PASS", "not_failed": [],
            "detail": "Frozen G4 V5 requires the int4 control to fail E1, E2 and E3; "
            "a nondiscriminating fixture never gets a fabricated control failure",
        }},
    }
    return _rehash_numeric_receipt(root, result)


def _rehash_numeric_receipt(root: Path, result: dict[str, Any]) -> Path:
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": gate.SCHEMA, "qualification_only": True,
        "fixture": "hand-derived CPU numerical-verifier receipt", "files": files,
    }), encoding="utf-8")
    return manifest


def _verify_numeric_receipt(manifest: Path) -> dict[str, Any]:
    wrapper = json.loads(manifest.read_text("utf-8"))
    # Every mutation below gets fresh wrappers, so refusal must come from raw evidence, not hashes.
    for relative, descriptor in wrapper["files"].items():
        path = manifest.parent / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]
        assert path.stat().st_size == descriptor["bytes"]
    result = json.loads((manifest.parent / "result.json").read_text("utf-8"))
    return gate._verify_numerical_evidence(manifest.parent, wrapper["files"], result)


def test_numeric_verifier_recomputes_coherent_raw_tensor_receipt(tmp_path: Path) -> None:
    result = _verify_numeric_receipt(_numeric_receipt(tmp_path))
    assert result["a3"]["rule"]["tolerance"]["floor_grad"] == [0.125]
    assert result["candidate_gates"]["E3"]["global_rel_l2"] == 0.0
    assert result["control"]["gates"]["E2"]["max_branch_rms_row_rel"] == 1.5


def test_public_loader_still_refuses_numerically_valid_private_fixture(tmp_path: Path) -> None:
    manifest = _numeric_receipt(tmp_path)
    _verify_numeric_receipt(manifest)
    with pytest.raises(ValueError, match="production|qualification"):
        gate.load_gate(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())


@pytest.mark.parametrize("changed,reason", [
    ("layer_metric", "layer metrics differ"),
    ("raw_layer", "layer metrics differ"),
    ("missing_pair", "missing raw boundary pair"),
    ("gradient", "E3 raw tensor metrics differ"),
    ("oracle", "Oracle VJP coverage"),
    ("a3_gradient", "A3 raw tensor recomputation"),
    ("repeat_loss_cancellation", "per-microbatch repeat losses"),
])
def test_numeric_verifier_rejects_rehashed_false_evidence(
    tmp_path: Path, changed: str, reason: str,
) -> None:
    import torch

    manifest = _numeric_receipt(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text("utf-8"))
    if changed == "layer_metric":
        result["passes"]["fp8_oracle"]["layers"][0]["restore"]["rms_row_rel"] = 0.01
    elif changed == "oracle":
        result["passes"]["fp8_oracle"]["oracle"]["counts"]["oracle_vjp"] = 7
    elif changed == "repeat_loss_cancellation":
        result["passes"]["ga_repeat"]["losses"] = [1.25, 0.75] + [1.0] * 6
    elif changed == "a3_gradient":
        path = tmp_path / "bf16_l2l" / "gradients.pt"
        values = torch.load(path, map_location="cpu", weights_only=True)
        values[result["trainable_names"][0]][0] = 3.0
        torch.save(values, path)
    elif changed == "missing_pair":
        (tmp_path / "fp8_oracle" / "recompute" / "layer-000-mb-003.pt").unlink()
    elif changed == "gradient":
        path = tmp_path / "fp8_oracle" / "gradients.pt"
        values = torch.load(path, map_location="cpu", weights_only=True)
        values[result["trainable_names"][0]][0] = 3.0
        torch.save(values, path)
    else:
        path = tmp_path / "fp8_oracle" / "recompute" / "layer-000-mb-003.pt"
        values = torch.load(path, map_location="cpu", weights_only=True)
        values["input"] += 0.125
        torch.save(values, path)
    _rehash_numeric_receipt(tmp_path, result)
    with pytest.raises(ValueError, match=reason):
        _verify_numeric_receipt(manifest)


@pytest.mark.parametrize("arm", [
    "ga", "ga_repeat", "bf16_l2l", "ga_floor", "fp8_oracle", "int4_oracle",
])
@pytest.mark.parametrize("changed", [
    "old_math_main", "missing_observation", "requested_backend", "missing_kernel",
    "missing_backward", "wrong_operator", "warn_only", "tf32", "cpu", "t4",
    "missing_shape", "wrong_length", "empty_calls", "wrong_kernel", "missing_sample",
    "full_window_trace", "profiling_error", "fallback_enabled",
])
def test_numeric_verifier_rejects_old_or_incomplete_attention_for_every_pass(
    tmp_path: Path, arm: str, changed: str,
) -> None:
    manifest = _numeric_receipt(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text("utf-8"))
    row = result["passes"][arm]
    observation = row["attention"]
    if changed == "old_math_main":
        row["backend_requested"] = "EFFICIENT_ATTENTION" if arm == "ga_floor" else "MATH"
    elif changed == "missing_observation":
        del row["attention"]
    elif changed == "requested_backend":
        observation["backend_requested"] = "FLASH_ATTENTION"
    elif changed == "missing_kernel":
        observation["samples"][0]["forward"]["cuda_kernel_names"] = []
    elif changed == "missing_backward":
        observation["samples"][0]["backward"] = {"operators": [], "cuda_kernel_names": []}
    elif changed == "wrong_kernel":
        observation["samples"][0]["forward"]["cuda_kernel_names"] = [
            "fmha_cutlassF_bf16_aligned_32x128_gmem_sm80" if arm == "ga_floor"
            else "ampere_bf16_s16816gemm_bf16_128x64_nn"
        ]
    elif changed == "wrong_operator":
        observation["samples"][0]["forward"]["operators"] = [
            "aten::_scaled_dot_product_flash_attention"
        ]
    elif changed == "warn_only":
        observation["deterministic_flags"]["warn_only"] = True
    elif changed == "tf32":
        observation["deterministic_flags"]["allow_tf32"] = True
    elif changed == "missing_sample":
        observation["samples"] = []
    elif changed == "full_window_trace":
        observation["instrumentation"]["full_window_trace"] = True
    elif changed == "profiling_error":
        observation["instrumentation"]["profiling_errors"] = ["incomplete profiler range"]
    elif changed == "fallback_enabled":
        observation["backend_flags"]["flash"] = True
    elif changed == "cpu":
        observation["unique_calls"][0]["device"] = "cpu"
    elif changed == "t4":
        observation["hardware"]["capability"] = [7, 5]
    elif changed == "missing_shape":
        del observation["unique_calls"][0]["value_shape"]
    elif changed == "wrong_length":
        for call in observation["unique_calls"] + observation["samples"]:
            for field in ("query_shape", "key_shape", "value_shape"):
                call[field][-2] = 32
    else:
        observation["unique_calls"] = []
    _rehash_numeric_receipt(tmp_path, result)
    with pytest.raises(ValueError, match="attention|backend|SDPA|kernel|CUDA|BF16"):
        _verify_numeric_receipt(manifest)


@pytest.mark.parametrize("arm", ["ga", "ga_repeat", "bf16_l2l", "fp8_oracle", "int4_oracle"])
def test_efficient_pass_cannot_swap_forward_and_backward_kernel_receipts(arm: str) -> None:
    observation = _attention_receipt("EFFICIENT_ATTENTION")
    sample = observation["samples"][0]
    sample["forward"]["cuda_kernel_names"], sample["backward"]["cuda_kernel_names"] = (
        sample["backward"]["cuda_kernel_names"], sample["forward"]["cuda_kernel_names"]
    )
    with pytest.raises(ValueError, match="kernel"):
        gate._require_pass_attention(
            arm, {"backend_requested": "EFFICIENT_ATTENTION", "attention": observation},
            min_calls=8,
        )


def test_attention_sample_shape_cannot_differ_from_executed_signature() -> None:
    observation = _attention_receipt("EFFICIENT_ATTENTION")
    observation["samples"][0]["value_shape"][-1] = 16
    with pytest.raises(ValueError, match="signatures"):
        gate._require_pass_attention(
            "ga", {"backend_requested": "EFFICIENT_ATTENTION", "attention": observation},
            min_calls=8,
        )


@pytest.mark.parametrize("changed", ["backend", "main_backend", "factor", "observed"])
def test_numeric_verifier_rejects_changed_alternative_floor_policy(
    tmp_path: Path, changed: str,
) -> None:
    manifest = _numeric_receipt(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text("utf-8"))
    result["floor"][changed] = False if changed == "observed" else "changed"
    _rehash_numeric_receipt(tmp_path, result)
    with pytest.raises(ValueError, match="floor"):
        _verify_numeric_receipt(manifest)


@pytest.mark.parametrize("changed", ["missing", "path", "sha256", "contract"])
def test_numeric_verifier_refuses_rehashed_changed_immutable_backend_policy(
    tmp_path: Path, changed: str,
) -> None:
    manifest = _numeric_receipt(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text("utf-8"))
    if changed == "missing":
        del result["attention_policy"]
    elif changed == "contract":
        result["attention_policy"]["contract"]["main_backend"] = "MATH"
    else:
        result["attention_policy"][changed] = "changed"
    _rehash_numeric_receipt(tmp_path, result)
    with pytest.raises(ValueError, match="policy"):
        _verify_numeric_receipt(manifest)


@pytest.mark.parametrize("changed", [
    "warn_only", "tf32", "not_restored", "workspace", "environment_not_restored",
    "t4", "emulated_bf16", "different_gpu",
])
def test_numeric_verifier_refuses_changed_hardware_or_determinism(
    tmp_path: Path, changed: str,
) -> None:
    manifest = _numeric_receipt(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text("utf-8"))
    if changed == "warn_only":
        result["determinism"]["execution"]["warn_only"] = True
    elif changed == "tf32":
        result["determinism"]["execution"]["allow_tf32"] = True
    elif changed == "not_restored":
        result["determinism"]["restored"] = False
    elif changed == "workspace":
        result["environment"]["cublas_workspace_config"] = "incompatible"
    elif changed == "environment_not_restored":
        result["cublas_environment_restored"] = False
    elif changed == "t4":
        result["environment"]["gpu"]["capability"] = [7, 5]
    elif changed == "emulated_bf16":
        result["environment"]["gpu"]["native_bf16"] = False
    else:
        result["passes"]["bf16_l2l"]["attention"]["hardware"]["name"] = "another GPU"
    _rehash_numeric_receipt(tmp_path, result)
    with pytest.raises(ValueError, match="determin|Ampere|hardware"):
        _verify_numeric_receipt(manifest)


def test_production_attention_requires_exact_mla_heads_and_dimensions() -> None:
    row = {"backend_requested": "EFFICIENT_ATTENTION",
           "attention": _attention_receipt("EFFICIENT_ATTENTION")}
    with pytest.raises(ValueError, match="MLA"):
        gate._require_pass_attention("ga", row, min_calls=8, production=True)
    for call in row["attention"]["unique_calls"] + row["attention"]["samples"]:
        call["query_shape"] = [1, 128, 64, 192]
        call["key_shape"] = [1, 128, 64, 192]
        call["value_shape"] = [1, 128, 64, 128]
    gate._require_pass_attention("ga", row, min_calls=8, production=True)


def test_gate_deterministic_scope_disallows_warnings_and_restores_process_state() -> None:
    import random

    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    backend_flags = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with gate._deterministic("cpu") as state:
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            assert state["execution"]["warn_only"] is False
            random.random()
            torch.rand(4)
        assert state["restored"] is True
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert random.getstate() == python_rng
        assert torch.equal(torch.get_rng_state(), torch_rng)
        assert torch.backends.cuda.matmul.allow_tf32 == backend_flags["matmul_tf32"]
        assert torch.backends.cudnn.allow_tf32 == backend_flags["cudnn_tf32"]
        assert torch.backends.cudnn.benchmark == backend_flags["cudnn_benchmark"]
        assert torch.backends.cudnn.deterministic == backend_flags["cudnn_deterministic"]
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def test_source_identity_binds_shared_executor_and_immutable_attention_protocol() -> None:
    identity = gate.policy_identity()
    hashes = gate._source_hashes()
    assert hashes[identity["path"]] == (
        "fc887c5a996bd58536ef0325b61405cd73a2a83cf363fcc4156204ab180e1ed6"
    )
    shared = "benchmarks/harness/p3_deepseek_quality_attention.py"
    assert hashes[shared] == hashlib.sha256(
        (Path(__file__).parents[1] / shared).read_bytes()
    ).hexdigest()


@pytest.mark.gpu
def test_missing_executed_kernel_refuses_completed_pass_and_raw_gradient_artifact(
    tmp_path: Path, tiny_source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    import torch

    from benchmarks.harness.p3_deepseek_quality_model import (
        open_session,
        sha256,
        write_initial_adapter,
    )

    native_execution = gate.attention_execution

    @contextmanager
    def missing_kernel(backend: str, *, device: str) -> Any:
        with native_execution(backend, device=device) as observed:
            class MissingKernel:
                def report(self) -> dict[str, Any]:
                    report = copy.deepcopy(observed.report())
                    for sample in report["samples"]:
                        sample["forward"]["cuda_kernel_names"] = []
                    return report

                def require_complete(self, **kwargs: Any) -> None:
                    gate.require_execution_evidence(self.report(), backend=backend, **kwargs)

            yield MissingKernel()

    with open_session(
        tiny_source, tmp_path / "cache", device="cuda", dtype="bfloat16", prepare=True,
    ) as session:
        adapter = tmp_path / "initial.safetensors"
        write_initial_adapter(session.model, adapter)
        raw_ids, raw_labels, identity = gate.gate_prefix(_window())
        monkeypatch.setattr(gate, "attention_execution", missing_kernel)
        row, gradients, _ = gate._run_pass(
            session, name="ga", initial_adapter=adapter, initial_sha256=sha256(adapter),
            ids=[torch.tensor(value, device="cuda") for value in raw_ids],
            labels=[torch.tensor(value, device="cuda") for value in raw_labels],
            targets=identity["valid_target_count"], directory=tmp_path / "ga",
            device="cuda", dtype="bfloat16", host_budget_bytes=32 * 1024**2,
        )
        assert row["status"] == "FAILED" and gradients == {}
        assert "kernel" in row["error"]["message"]
        assert row["optimizer_applied"] is False and row["adapter_unchanged"] is True
        assert row["host_budget_after_close_bytes"] == 0
        assert row["cleanup_errors"] == []
        assert not (tmp_path / "ga" / "gradients.pt").exists()
    assert session.closed is True


@pytest.mark.gpu
@pytest.mark.parametrize("operation", ["gradient_snapshot", "archive_table", "oracle_report"])
def test_diagnostic_failure_cannot_skip_native_oracle_store_and_lease_cleanup(
    tmp_path: Path, tiny_source: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    import torch

    from benchmarks.harness.p3_deepseek_oracle import DeepseekOracleProbe
    from benchmarks.harness.p3_deepseek_quality_model import (
        open_session,
        sha256,
        write_initial_adapter,
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    closed: list[dict[str, Any]] = []
    native_close = DeepseekOracleProbe.close
    native_report = DeepseekOracleProbe.report

    def close_probe(probe: Any) -> None:
        native_close(probe)
        closed.append(native_report(probe))

    def fail_diagnostic(*_args: Any, **_kwargs: Any) -> Any:
        raise MemoryError("injected diagnostic allocation failure")

    monkeypatch.setattr(DeepseekOracleProbe, "close", close_probe)
    with open_session(
        tiny_source, tmp_path / "cache", device="cuda", dtype="bfloat16", prepare=True,
    ) as session:
        adapter = tmp_path / "initial.safetensors"
        write_initial_adapter(session.model, adapter)
        raw_ids, raw_labels, identity = gate.gate_prefix(_window())
        inputs = [torch.tensor(row, device="cuda") for row in raw_ids]
        labels = [torch.tensor(row, device="cuda") for row in raw_labels]
        if operation == "gradient_snapshot":
            monkeypatch.setattr(gate, "_gradient_snapshot", fail_diagnostic)
        elif operation == "archive_table":
            monkeypatch.setattr(gate._LayerArchive, "table", fail_diagnostic)
        else:
            monkeypatch.setattr(DeepseekOracleProbe, "report", fail_diagnostic)
        row, gradients, _ = gate._run_pass(
            session, name="int4_oracle", initial_adapter=adapter, initial_sha256=sha256(adapter),
            ids=inputs, labels=labels, targets=identity["valid_target_count"],
            directory=tmp_path / "int4", device="cuda", dtype="bfloat16",
            host_budget_bytes=32 * 1024**2,
        )
        assert row["status"] == "FAILED"
        assert gradients == {}
        assert row["diagnostic_errors"]
        assert row["host_budget_after_close_bytes"] == 0
        assert row["cleanup_errors"] == []
        assert session.retained_resources == []
        assert row["store"]["codec_stats"]["compress_calls"] == 24
        assert closed and all(report["closed"] and report["host_reserved_bytes"] == 0
                              and report["records_remaining"] == report["pending_replays"] == 0
                              for report in closed)
    assert session.closed is True


@pytest.mark.gpu
def test_observer_close_failure_is_retained_before_runtime_can_close(
    tmp_path: Path, tiny_source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from benchmarks.harness.p3_deepseek_quality_model import (
        open_session,
        sha256,
        write_initial_adapter,
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    session = open_session(
        tiny_source, tmp_path / "cache", device="cuda", dtype="float32", prepare=True,
    )
    native_close = gate._LayerFiniteObservation.close

    def fail_close(_observer: Any) -> None:
        raise MemoryError("injected observer close failure")

    try:
        adapter = tmp_path / "initial.safetensors"
        write_initial_adapter(session.model, adapter)
        monkeypatch.setattr(gate._LayerFiniteObservation, "close", fail_close)
        raw_ids, raw_labels, identity = gate.gate_prefix(_window())
        row, gradients, _ = gate._run_pass(
            session, name="ga", initial_adapter=adapter, initial_sha256=sha256(adapter),
            ids=[torch.tensor(value, device="cuda") for value in raw_ids],
            labels=[torch.tensor(value, device="cuda") for value in raw_labels],
            targets=identity["valid_target_count"], directory=tmp_path / "ga",
            device="cuda", dtype="float32", host_budget_bytes=32 * 1024**2,
        )
        assert row["status"] == "FAILED" and gradients == {}
        assert row["cleanup_errors"][0]["resource"] == "observer"
        assert session.retained_resources
        with pytest.raises(RuntimeError, match="retained"):
            session.close()
        assert session.closed is False
    finally:
        for resource in session.retained_resources:
            if isinstance(resource, gate._LayerFiniteObservation):
                native_close(resource)
        session.retained_resources.clear()
        session.close()


def test_failure_gradient_diagnostics_observe_without_copying_tensors() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([0.5, 0.25])
    values, checks = gate._gradient_snapshot({"attention.lora_A": parameter}, capture=False)
    assert values == {}
    assert checks["attention.lora_A"]["finite"] is True
    assert checks["attention.lora_A"]["nonzero"] is True
