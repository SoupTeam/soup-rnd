"""D2 evidence harness tests: real PEFT, SYNTHETIC weights and inputs."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "fast_lora_probe.py"


def _probe():
    assert HARNESS.is_file(), "D2 Fast-LoRA evidence harness is missing"
    spec = importlib.util.spec_from_file_location("d2_fast_lora_probe", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fp32_parity_reports_every_forward_and_adapter_gradient():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    report = _probe().run_parity(seed=792, dtype="fp32", device="cpu", gradcheck=False)
    assert report["passed"] is True
    assert report["fixture"] == "SYNTHETIC"
    assert report["reference"] == "unpatched PEFT"
    assert report["base_frozen"] is True
    assert report["dropout"] == 0.0
    assert report["criterion"] == "torch.testing.assert_close defaults (fp32)"
    rows = report["rows"]
    for path, projections in {
        "single": ["o_proj"],
        "qkv": ["q_proj", "k_proj", "v_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
    }.items():
        group = [row for row in rows if row["path"] == path]
        backward = {row["quantity"] for row in group if row["phase"] == "backward"}
        assert backward == {"dX"} | {
            f"{grad}/{projection}" for grad in ("dA", "dB") for projection in projections
        }
        assert len([row for row in group if row["phase"] == "forward"]) == (
            3 if path == "qkv" else 1
        )
        assert report["patch_counts"][path] > 0
        assert all(row["passed"] and isinstance(row["bit_exact"], bool) for row in group)
    assert len(rows) == 22


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_low_precision_uses_named_float64_error_bound(dtype):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    report = _probe().run_parity(seed=792, dtype=dtype, device="cpu", gradcheck=False)
    assert report["passed"]
    assert report["criterion"] == "PROPOSED: float64 max error <= 2*PEFT max error + 1e-8"
    for row in report["rows"]:
        assert row["fast_float64_max_abs_error"] <= row["proposed_max_error_bound"]
        assert row["proposed_max_error_bound"] == (2 * row["peft_float64_max_abs_error"] + 1e-8)
        assert row["peft_float64_rmse"] >= 0
        assert "fast_peft_float64_max_error_ratio" in row
        assert row["finite"] is True


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_evidence_rejects_missing_or_nonfinite_tensor(value):
    torch = pytest.importorskip("torch")
    tensor = None if value is None else torch.tensor([value])
    with pytest.raises(AssertionError, match="missing|nonfinite"):
        _probe().require_finite(tensor, "dA/o_proj")


def test_gradcheck_covers_input_and_all_adapter_matrices_for_each_path():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    report = _probe().run_parity(seed=792, gradcheck=True)
    for path, projections in {
        "single": ["o_proj"],
        "qkv": ["q_proj", "k_proj", "v_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
    }.items():
        check = report["gradcheck"][path]
        assert check["passed"] is True
        assert check["dtype"] == "float64"
        assert set(check["variables"]) == {"X"} | {
            f"{matrix}/{projection}" for matrix in ("A", "B") for projection in projections
        }


def test_fifty_step_synthetic_llama_loss_gate_checks_real_fast_paths():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    probe = _probe()
    report = probe.run_loss(seed=792, steps=50, device="cpu", dtype="fp32")
    assert report["fixture"] == "SYNTHETIC"
    assert report["steps"] == 50
    assert len(report["rows"]) == 50
    assert report["passed"] and report["base_unchanged"]
    assert set(report["target_modules"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert all(count > 0 for count in report["patch_counts"].values())
    assert all(row["rounded_3_equal"] for row in report["rows"])
    assert report["rows"][-1]["baseline_loss"] < report["rows"][0]["baseline_loss"]
    assert report["negative_control"]["unpatched_rejected"] is True
    assert all(report["execution"][path] for path in ("single", "qkv", "mlp"))
    assert torch.get_num_threads() == previous_threads
    assert torch.are_deterministic_algorithms_enabled() == previous_deterministic
    repeat = probe.run_loss(seed=792, steps=2)
    assert repeat["rows"] == report["rows"][:2]


def test_noop_patcher_cannot_pass_a_loss_curve(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    probe = _probe()
    monkeypatch.setattr(probe, "_patch", lambda model, path: 1)
    with pytest.raises(AssertionError, match="custom autograd"):
        probe.run_loss(steps=1)


def test_cpu_timing_is_only_cpu_evidence_with_all_paths_and_dx_branches():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    report = _probe().run_timing(warmup=1, repeats=2, tokens=3)
    assert report["fixture"] == "SYNTHETIC"
    assert report["measurement_scope"] == "CPU DEBUG-ONLY; no CUDA or performance multiplier claim"
    assert report["timer"] == "perf_counter"
    assert report["shapes"] == "tiny"
    assert report["warmup"] == 1
    assert report["repeats"] == 2
    rows = report["rows"]
    assert len(rows) == 48
    assert report["round_order"] == "ABBA"
    assert report["correctness_gate"]["passed"]
    assert report["timing_verdict"] == "NO VERDICT (CPU DEBUG-ONLY)"
    for path in ("single", "qkv", "mlp"):
        for dx in (True, False):
            for round_index in (1, 2):
                arms = [
                    row
                    for row in rows
                    if row["path"] == path
                    and row["request_dX"] == dx
                    and row["round"] == round_index
                ]
                assert [row["arm"] for row in arms] == ["A", "B", "B", "A"]
                assert [row["arm_index"] for row in arms] == [1, 2, 3, 4]
    assert {(row["path"], row["request_dX"], row["implementation"]) for row in rows} == {
        (path, dx, implementation)
        for path in ("single", "qkv", "mlp")
        for dx in (True, False)
        for implementation in ("PEFT", "fast")
    }
    assert all(row["forward_ms"] > 0 and row["backward_ms"] > 0 for row in rows)
    assert all(row["dtype"] == "fp32" for row in rows)
    assert report["passed"] is True


def test_large_cpu_shapes_require_explicit_opt_in():
    with pytest.raises(ValueError, match="allow_large_cpu"):
        _probe().run_timing(shapes="llama3.1-8b", device="cpu")


@pytest.mark.gpu
def test_cuda_fp16_timing_uses_events_without_bf16_or_triton():
    report = _probe().run_timing(
        device="cuda",
        dtype="fp16",
        warmup=1,
        repeats=2,
        tokens=3,
    )
    assert report["timer"] == "CUDA events"
    assert report["measurement_scope"] == "CUDA single-layer ONLY; not a full-run multiplier"
    assert report["passed"]


def test_cli_emits_separate_parity_tables_and_raw_json_csv(tmp_path):
    prefix = tmp_path / "parity"
    env = {**os.environ, "PYTHONPATH": str(HARNESS.parents[2] / "src")}
    process = subprocess.run(
        [
            sys.executable,
            str(HARNESS),
            "parity",
            "--skip-gradcheck",
            "--output-prefix",
            str(prefix),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "Forward parity" in process.stdout
    assert "Backward parity" in process.stdout
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["passed"]
    assert Path(report["environment"]["soup_cli_file"]).samefile(
        HARNESS.parents[2] / "src" / "soup_cli" / "__init__.py"
    )
    assert set(report["environment"]["kernel_sha256"]) == {
        "fast_lora.py",
        "fast_lora_qkv.py",
        "fast_lora_mlp.py",
    }
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["rows"]) == 22
    assert {row["phase"] for row in rows} == {"forward", "backward"}
    assert report["nf4_status"] == "not implemented/unverified by this harness"
    assert len(report["environment"]["decision_rule_sha256"]) == 64


def test_cli_help_never_imports_heavy_training_dependencies():
    code = (
        "import builtins, runpy, sys; "
        "original = builtins.__import__; "
        "blocked = {'torch', 'peft', 'transformers', 'bitsandbytes'}; "
        "builtins.__import__ = lambda name, *a, **k: "
        "(_ for _ in ()).throw(AssertionError('heavy import: ' + name)) "
        "if name.split('.')[0] in blocked else original(name, *a, **k); "
        "sys.argv = [sys.argv[1], '--help']; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    process = subprocess.run(
        [sys.executable, "-c", code, str(HARNESS)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "parity" in process.stdout and "loss" in process.stdout and "timing" in process.stdout


def test_nf4_request_fails_honestly_and_writes_failure_evidence(tmp_path):
    prefix = tmp_path / "nf4"
    assert (
        _probe().main(
            [
                "parity",
                "--quantization",
                "nf4",
                "--output-prefix",
                str(prefix),
            ]
        )
        == 2
    )
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert "not implemented/unverified" in report["error"]
    assert report["rows"] == []


@pytest.mark.parametrize("steps", [0, -1])
def test_loss_rejects_empty_runs_instead_of_vacuous_pass(steps):
    with pytest.raises(ValueError, match="steps"):
        _probe().run_loss(steps=steps)


@pytest.mark.parametrize("options", [{"repeats": 0}, {"warmup": -1}, {"tokens": 0}])
def test_timing_rejects_empty_or_invalid_measurements(options):
    with pytest.raises(ValueError, match="repeats|warmup|tokens"):
        _probe().run_timing(**options)


def test_parity_preserves_rng_threads_and_deterministic_settings():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    enabled = torch.are_deterministic_algorithms_enabled()
    report = _probe().run_parity()
    assert report["deterministic_algorithms"] is True
    assert report["cpu_threads"] == 1
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == enabled


def test_fp32_parity_fails_real_numerical_mismatch():
    torch = pytest.importorskip("torch")
    with pytest.raises(AssertionError):
        _probe()._comparison(
            torch.tensor([2.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0], dtype=torch.float64),
            "fp32",
        )


def test_low_precision_bound_fails_excessive_error():
    torch = pytest.importorskip("torch")
    with pytest.raises(AssertionError, match="named PROPOSED bound"):
        _probe()._comparison(
            torch.tensor([2.0], dtype=torch.float16),
            torch.tensor([1.0], dtype=torch.float16),
            torch.tensor([1.0], dtype=torch.float64),
            "fp16",
        )


def test_fp16_loss_uses_float32_adapter_master_weights():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    report = _probe().run_loss(steps=2, dtype="fp16")
    assert report["passed"]
    assert report["adapter_dtype"] == "fp32"
    assert all(row["rounded_3_equal"] for row in report["rows"])


def test_fp16_loss_reports_reference_arithmetic_for_each_declared_module_and_step():
    probe = _probe()
    report = probe.run_loss(seed=792, steps=2, device="cpu", dtype="fp16")
    metadata = report["reference_arithmetic"]
    assert metadata["required_true"] is True
    assert metadata["ctx_attribute"] == "grad_fn.reference_order"
    expected = {
        "qkv": [
            f"base_model.model.model.layers.{layer}.self_attn.{projection}"
            for layer in range(report["model"]["num_hidden_layers"])
            for projection in probe.PROJECTIONS["qkv"]
        ],
        "mlp": [
            f"base_model.model.model.layers.{layer}.mlp"
            for layer in range(report["model"]["num_hidden_layers"])
        ],
    }
    assert metadata["expected_modules"] == expected
    for row in report["rows"]:
        assert row["reference_arithmetic_status"] == "observed"
        for path, names in expected.items():
            assert set(row["reference_arithmetic_modes"][path]) == set(names)
            assert all(row["reference_arithmetic_modes"][path][name] is True for name in names)
    assert report["negative_control"]["unpatched_rejected"] is True


def test_loss_cli_exports_mode_scope_in_json_csv_help_and_render(tmp_path):
    prefix = tmp_path / "loss-modes"
    process = subprocess.run(
        [sys.executable, str(HARNESS), "loss", "--dtype", "fp32", "--steps", "2",
         "--output-prefix", str(prefix)],
        env={**os.environ, "PYTHONPATH": str(HARNESS.parents[2] / "src")},
        capture_output=True, text=True, check=False, timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["reference_arithmetic"]["required_true"] is False
    assert all(mode is False for row in report["rows"]
               for values in row["reference_arithmetic_modes"].values() for mode in values.values())
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(report["rows"]) == 2
    for raw, row in zip(csv_rows, report["rows"]):
        assert json.loads(raw["reference_arithmetic_modes"]) == row["reference_arithmetic_modes"]
        assert raw["reference_arithmetic_status"] == "observed"
    assert "reference_order" in process.stdout and "fp32" in process.stdout
    assert "CPU" in process.stdout and "not global bit-exactness" in process.stdout
    assert "finite SYNTHETIC fixture" in process.stdout
    help_result = subprocess.run(
        [sys.executable, str(HARNESS), "--help"], capture_output=True,
        text=True, check=False, timeout=30,
    )
    assert help_result.returncode == 0
    assert "reference_order" in help_result.stdout
    assert "CPU" in help_result.stdout and "not global bit-exactness" in help_result.stdout


def test_loss_gate_retains_real_measured_rows_when_rounding_fails(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    probe = _probe()
    original = probe._loss_model

    def biased_model(device, dtype):
        model = original(device, dtype)

        def bias_fast_loss(module, inputs, output):
            if any(getattr(child, "_soup_fast_lora_mlp", False) for child in module.modules()):
                output.loss = output.loss + 0.01
            return output

        model.register_forward_hook(bias_fast_loss)
        return model

    monkeypatch.setattr(probe, "_loss_model", biased_model)
    report = probe.run_loss(steps=2)
    assert report["passed"] is False
    assert len(report["rows"]) == 2
    assert all(row["abs_error"] > 0.009 for row in report["rows"])
    assert "3 decimals" in report["error"]


def test_parity_records_actual_custom_grad_functions():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    probe = _probe()
    report = probe.run_parity()
    assert report["execution"] == {
        "single": [probe.EXPECTED_GRAD_FN["single"]],
        "qkv": [probe.EXPECTED_GRAD_FN["qkv"]] * 3,
        "mlp": [probe.EXPECTED_GRAD_FN["mlp"]],
    }


def test_noop_patcher_cannot_pass_parity(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    probe = _probe()
    monkeypatch.setattr(probe, "_patch", lambda model, path: 1)
    with pytest.raises(AssertionError, match="custom autograd"):
        probe.run_parity()


def test_t4_bf16_request_is_rejected_not_skipped_or_marked_pass(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(ValueError, match="native bf16.*fp16"):
        _probe().validate_runtime("cuda", "bf16")


def test_missing_nvidia_smi_is_disclosed_as_no_validity_verdict(monkeypatch):
    probe = _probe()

    def absent(*args, **kwargs):
        raise FileNotFoundError("SYNTHETIC test: nvidia-smi unavailable")

    monkeypatch.setattr(probe.subprocess, "run", absent)
    validity = probe.collect_cuda_validity()
    assert validity["status"] == "UNVERIFIED"
    assert validity["available"] is False
    assert "nvidia-smi" in validity["reason"]


def test_parity_without_dx_still_checks_every_adapter_gradient():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    report = _probe().run_parity(request_dx=False)
    assert report["passed"]
    assert report["request_dX"] is False
    assert len(report["rows"]) == 19
    assert not any(row["quantity"] == "dX" for row in report["rows"])


def test_parity_detects_real_deliberately_changed_adapter_negative_control():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    report = _probe().run_parity(dtype="fp16")
    assert set(report["negative_control"]) == {"single", "qkv", "mlp"}
    assert all(
        control["changed_adapter_detected"] for control in report["negative_control"].values()
    )
    assert all(
        control["max_abs_output_delta"] > 0 for control in report["negative_control"].values()
    )


def test_void_timing_arms_are_retained_raw_but_excluded_from_rendered_medians():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    probe = _probe()
    report = probe.run_timing(warmup=0, repeats=1, tokens=3)
    for row in report["rows"]:
        row["validity"]["status"] = "VOID"  # Explicit SYNTHETIC interference fault injection.
    stream = StringIO()
    probe._render(report, probe.Console(file=stream, width=160))
    assert "VOID" in stream.getvalue()
    assert "n/a" in stream.getvalue()
    assert len(report["rows"]) == 24
