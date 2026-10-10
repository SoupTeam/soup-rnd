"""Fail-closed validity regressions using real CPU PEFT/autograd paths."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from io import StringIO
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "fast_lora_probe.py"


def _probe():
    spec = importlib.util.spec_from_file_location("d2_validity_probe", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("optimization", ["flag", "environment"])
@pytest.mark.parametrize(
    "check", ["bound", "finite", "noop_parity", "noop_loss", "control", "reference_mode"]
)
def test_optimized_python_cannot_disable_validity_checks(optimization, check):
    actions = {
        "bound": "p._comparison(torch.tensor([2.], dtype=torch.float16), "
        "torch.tensor([1.], dtype=torch.float16), torch.tensor([1.], dtype=torch.float64), 'fp16')",
        "finite": "p.require_finite(torch.tensor([float('nan')]), 'dA/o_proj')",
        "noop_parity": "p._patch = lambda model, path: 1; p.run_parity()",
        "noop_loss": "p._patch = lambda model, path: 1; p.run_loss(steps=1)",
        "control": "p._comparison = lambda *args: {}; p.run_parity(dtype='fp16')",
        "reference_mode": "from soup_cli.utils import fast_lora_mlp as k; "
        "original = k._mlp_function; "
        "k._mlp_function = lambda **kw: original(reference_order=False); "
        "p.run_loss(steps=1, dtype='fp16')",
    }
    code = (
        "import importlib.util, sys, torch\n"
        "spec = importlib.util.spec_from_file_location('probe', sys.argv[1])\n"
        "p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)\n"
        "try:\n"
        f"    {actions[check]}\n"
        "except AssertionError:\n"
        "    sys.exit(0)\n"
        "sys.exit(99)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(HARNESS.parents[2] / "src")}
    env.pop("PYTHONOPTIMIZE", None)
    flags = ["-O"] if optimization == "flag" else []
    if optimization == "environment":
        env["PYTHONOPTIMIZE"] = "1"
    result = subprocess.run(
        [sys.executable, *flags, "-c", code, str(HARNESS)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr + f"exit={result.returncode}"


def test_unregistered_adapter_cannot_disappear_from_parity(monkeypatch):
    probe = _probe()
    original = probe._patch

    def unregister(model, path):
        count = original(model, path)
        if path == "single":
            adapter = model.o_proj.lora_A.default
            weight = adapter._parameters.pop("weight")
            object.__setattr__(adapter, "weight", weight)  # Forward still works.
        return count

    monkeypatch.setattr(probe, "_patch", unregister)
    with pytest.raises(AssertionError, match="missing|unregistered|keys"):
        probe.run_parity()


@pytest.mark.parametrize("snapshot_index", [1, 2, 3])
def test_missing_snapshot_key_fails_against_independent_expectations(monkeypatch, snapshot_index):
    probe = _probe()
    original = probe._snapshot
    calls = 0

    def omit(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == snapshot_index:
            result.pop("dA/o_proj")
        return result

    monkeypatch.setattr(probe, "_snapshot", omit)
    with pytest.raises(AssertionError, match="keys|missing"):
        probe.run_parity()


@pytest.mark.parametrize("mode", ["parity", "loss", "timing"])
@pytest.mark.parametrize("exception", [RuntimeError, KeyError])
def test_cli_retains_completed_measurements_and_failure_case(
    monkeypatch, tmp_path, mode, exception
):
    probe = _probe()
    name = {"parity": "_comparison", "loss": "_check_training_gradients", "timing": "_measure"}[
        mode
    ]
    original = getattr(probe, name)
    calls = 0

    def fail_after_measurement(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise exception("SYNTHETIC injected failure after completed measurement")
        return original(*args, **kwargs)

    monkeypatch.setattr(probe, name, fail_after_measurement)
    prefix = tmp_path / mode
    assert (
        probe.main(
            [
                mode,
                "--skip-gradcheck",
                "--steps",
                "2",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--tokens",
                "3",
                "--output-prefix",
                str(prefix),
            ]
        )
        == 2
    )
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    completed = [row for row in report["rows"] if row.get("row_type") != "failure"]
    assert len(completed) == 1
    failed = [row for row in report["rows"] if row.get("row_type") == "failure"]
    assert len(failed) == 1
    assert failed[0]["failure_case"] == report["failure_case"]
    assert "SYNTHETIC injected failure" in failed[0]["error"]
    assert "Traceback" in failed[0]["traceback"]
    assert exception.__name__ in report["traceback"]
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["rows"])
    assert rows[-1]["row_type"] == "failure"
    assert "SYNTHETIC injected failure" in rows[-1]["error"]
    assert "Traceback" in rows[-1]["traceback"]


def test_timing_gates_exact_timed_shape_before_any_measurement(monkeypatch):
    probe = _probe()
    original_patch = probe._patch
    original_measure = probe._measure
    measured = 0

    def corrupt_exact_shape(model, path):
        count = original_patch(model, path)
        if path == "single":

            def corrupt(module, inputs, output):
                if tuple(inputs[0].shape) == (1, 3, 8):
                    assert type(output.grad_fn).__name__ == probe.EXPECTED_GRAD_FN[path]
                    output.detach().add_(10.0)  # Retain the genuine custom grad_fn.

            model.register_forward_hook(corrupt)
        return count

    def measure(*args, **kwargs):
        nonlocal measured
        measured += 1
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(probe, "_patch", corrupt_exact_shape)
    monkeypatch.setattr(probe, "_measure", measure)
    with pytest.raises(AssertionError):
        probe.run_timing(warmup=0, repeats=1, tokens=3)
    assert measured == 0


@pytest.mark.parametrize("fault", ["nonfinite_gradient", "finite_gradient", "finite_output"])
def test_first_bad_timing_arm_fails_and_retains_raw_measurement(monkeypatch, tmp_path, fault):
    probe = _probe()
    original_patch, original_measure = probe._patch, probe._measure
    models = {}
    calls = 0

    def patch(model, path):
        count = original_patch(model, path)
        models[path] = model
        return count

    def measure(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_measure(*args, **kwargs)
        if calls == 3 and fault == "finite_output":
            assert type(result[0][0].grad_fn).__name__ == probe.EXPECTED_GRAD_FN["single"]
            result[0][0].detach().add_(10.0)
        if calls == 4 and fault != "finite_output":
            gradient = models["single"].o_proj.lora_A.default.weight.grad
            if fault == "nonfinite_gradient":
                gradient.fill_(float("nan"))
            else:
                gradient.add_(10.0)
        return result

    monkeypatch.setattr(probe, "_patch", patch)
    monkeypatch.setattr(probe, "_measure", measure)
    prefix = tmp_path / fault
    assert (
        probe.main(
            [
                "timing",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--tokens",
                "3",
                "--output-prefix",
                str(prefix),
            ]
        )
        == 2
    )
    assert calls == 4  # Do not hide the fault behind a later healthy B arm.
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    bad = [row for row in report["rows"] if row.get("arm_index") == 2][0]
    assert bad["implementation"] == "fast"
    assert bad["correctness_passed"] is False
    assert bad["validity"]["status"] == "VOID"
    assert bad["forward_ms"] > 0 and bad["backward_ms"] > 0
    assert "Traceback" in bad["traceback"]
    assert "NO VERDICT" in report["timing_verdict"]
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row.get("arm_index") == "2" and row["correctness_passed"] == "False" for row in rows)


@pytest.mark.parametrize("refusal", ["cuda", "nf4", "missing_torch"])
@pytest.mark.parametrize("git_context", ["reported", "no_repository", "clean"])
def test_early_refusal_still_records_available_environment(
    monkeypatch, tmp_path, refusal, git_context,
):
    probe = _probe()
    if git_context != "reported":
        from types import SimpleNamespace

        original_run = probe.subprocess.run

        def git_probe(command, *args, **kwargs):
            if command and command[0] == "git":
                if git_context == "no_repository":
                    return SimpleNamespace(
                        returncode=128, stdout="", stderr="SYNTHETIC no .git snapshot",
                    )
                return SimpleNamespace(
                    returncode=0,
                    stdout="a" * 40 if command[-1] == "HEAD" else "",
                    stderr="",
                )
            return original_run(command, *args, **kwargs)

        monkeypatch.setattr(probe.subprocess, "run", git_probe)
    options = []
    if refusal == "cuda":
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        options = ["--device", "cuda"]
    elif refusal == "nf4":
        options = ["--quantization", "nf4"]
    else:
        import builtins

        original = builtins.__import__

        def unavailable(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("SYNTHETIC unavailable torch")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", unavailable)
    prefix = tmp_path / refusal
    assert probe.main(["parity", *options, "--output-prefix", str(prefix)]) == 2
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    environment = report["environment"]
    assert environment["harness_sha256"] == hashlib.sha256(HARNESS.read_bytes()).hexdigest()
    assert (
        environment["decision_rule_sha256"]
        == hashlib.sha256(
            (HARNESS.parents[2] / "benchmarks" / "gate-d2-fast-lora-rule.md").read_bytes()
        ).hexdigest()
    )
    for field, query in (
        ("git_head", "git rev-parse HEAD"), ("git_status", "git status --short")
    ):
        value = environment[field]
        if value is None:
            assert environment["collection_errors"][query]
        else:
            assert isinstance(value, str)
            assert query not in environment["collection_errors"]
            if field == "git_head":
                assert value
    if git_context == "no_repository":
        assert environment["git_head"] is None and environment["git_status"] is None
    elif git_context == "clean":
        assert environment["git_head"] == "a" * 40 and environment["git_status"] == ""
    assert set(environment["kernel_sha256"]) == {
        "fast_lora.py",
        "fast_lora_qkv.py",
        "fast_lora_mlp.py",
    }
    for name, digest in environment["kernel_sha256"].items():
        source = HARNESS.parents[2] / "src" / "soup_cli" / "utils" / name
        assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert environment["python_executable"] == sys.executable
    assert "torch" in environment["packages"]
    assert report["passed"] is False
    assert "Traceback" in report["traceback"]
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["row_type"] == "failure"
    assert rows[0]["error"] == report["error"]


def test_harness_fingerprint_changes_with_content_without_git_status_change(monkeypatch, tmp_path):
    probe = _probe()
    copy = tmp_path / "benchmarks" / "harness" / "fast_lora_probe.py"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(HARNESS.read_bytes())
    monkeypatch.setattr(probe, "__file__", str(copy))
    before = probe._environment("cpu")
    copy.write_bytes(copy.read_bytes() + b"\n# SYNTHETIC fingerprint fault injection\n")
    after = probe._environment("cpu")
    assert before["git_status"] == after["git_status"]
    assert before["harness_sha256"] != after["harness_sha256"]
    assert after["harness_sha256"] == hashlib.sha256(copy.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ["parity", "loss", "timing"])
def test_direct_runtime_failure_keeps_pre_refusal_provenance(monkeypatch, mode):
    import torch

    probe = _probe()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA was requested") as failure:
        getattr(probe, f"run_{mode}")(device="cuda")
    environment = failure.value.evidence_report["environment"]
    assert environment["cuda_available"] is False
    assert environment["harness_sha256"] == hashlib.sha256(HARNESS.read_bytes()).hexdigest()


def test_loss_rejects_unregistered_adapter_even_for_one_step(monkeypatch):
    probe = _probe()
    original = probe._patch

    def unregister(model, path):
        count = original(model, path)
        if path == "single":
            module = next(
                module for name, module in model.named_modules() if name.endswith("o_proj")
            )
            adapter = module.lora_A.default
            weight = adapter._parameters.pop("weight")
            object.__setattr__(adapter, "weight", weight)
        return count

    monkeypatch.setattr(probe, "_patch", unregister)
    with pytest.raises(AssertionError, match="missing|unregistered"):
        probe.run_loss(steps=1)


def test_direct_gradcheck_cannot_accept_noop_patcher(monkeypatch):
    probe = _probe()
    monkeypatch.setattr(probe, "_patch", lambda model, path: 1)
    with pytest.raises(AssertionError, match="custom autograd"):
        probe._gradcheck("single", "cpu")


@pytest.mark.parametrize("optimization", ["flag", "environment"])
def test_optimized_cli_writes_truthful_noop_failure_artifacts(tmp_path, optimization):
    prefix = tmp_path / optimization
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('probe', sys.argv[1])\n"
        "p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)\n"
        "p._patch = lambda model, path: 1\n"
        "sys.exit(p.main(['parity', '--skip-gradcheck', '--output-prefix', sys.argv[2]]))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(HARNESS.parents[2] / "src")}
    env.pop("PYTHONOPTIMIZE", None)
    flags = ["-O"] if optimization == "flag" else []
    if optimization == "environment":
        env["PYTHONOPTIMIZE"] = "1"
    result = subprocess.run(
        [sys.executable, *flags, "-c", code, str(HARNESS), str(prefix)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert "custom autograd missing" in report["error"]
    assert "Traceback" in report["traceback"]
    assert report["environment"]["python_optimization"] == 1
    assert (
        report["environment"]["harness_sha256"] == hashlib.sha256(HARNESS.read_bytes()).hexdigest()
    )
    assert report["rows"][-1]["row_type"] == "failure"
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["rows"])
    assert rows[-1]["passed"] == "False"


def test_timing_checks_every_arm_outside_measured_regions(monkeypatch):
    probe = _probe()
    original_measure, original_collect = probe._measure, probe._collect_snapshot
    inside = False
    checks = 0
    measures = 0

    def measure(*args, **kwargs):
        nonlocal inside, measures
        inside = True
        measures += 1
        try:
            return original_measure(*args, **kwargs)
        finally:
            inside = False

    def collect(*args, **kwargs):
        nonlocal checks
        assert not inside
        checks += 1
        return original_collect(*args, **kwargs)

    monkeypatch.setattr(probe, "_measure", measure)
    monkeypatch.setattr(probe, "_collect_snapshot", collect)
    report = probe.run_timing(warmup=1, repeats=1, tokens=3)
    assert report["passed"]
    assert measures == 96  # Six cases, eight arms, two measured phases.
    assert checks == 66  # Three gate snapshots + eight arms, for each of six cases.
    assert len(report["correctness_gate"]["cases"]) == 6
    assert all(case["input_shape"] == [1, 3, 8] for case in report["correctness_gate"]["cases"])
    assert all(row["correctness_passed"] for row in report["rows"])
    for row in report["rows"]:
        assert {check["quantity"] for check in row["checks"]} == probe._expected_quantities(
            row["path"],
            row["request_dX"],
        )


def test_timing_backward_exception_preserves_completed_forward_measurement(monkeypatch, tmp_path):
    probe = _probe()
    original = probe._measure
    calls = 0

    def fail_backward(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("SYNTHETIC first B backward failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(probe, "_measure", fail_backward)
    prefix = tmp_path / "partial-arm"
    assert (
        probe.main(
            [
                "timing",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--tokens",
                "3",
                "--output-prefix",
                str(prefix),
            ]
        )
        == 2
    )
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert len(report["rows"]) == 2
    partial = report["rows"][-1]
    assert partial["row_type"] == "failure"
    assert partial["arm_index"] == 2
    assert partial["forward_ms"] > 0
    assert partial["backward_ms"] is None  # Never invent an unfinished timing.
    assert partial["validity"]["status"] == "VOID"
    assert "Traceback" in partial["traceback"]


def test_timing_early_refusal_has_no_validity_verdict(monkeypatch, tmp_path):
    import torch

    probe = _probe()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError) as failure:
        probe.run_timing(device="cuda")
    assert "NO VERDICT" in failure.value.evidence_report["timing_verdict"]
    prefix = tmp_path / "early-timing"
    assert probe.main(["timing", "--device", "cuda", "--output-prefix", str(prefix)]) == 2
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert "NO VERDICT" in report["timing_verdict"]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "fault", ["fast_dtype", "fast_shape", "oracle_shape", "fast_device", "oracle_device"]
)
def test_low_precision_comparison_rejects_tensor_contract_mismatch(dtype, fault):
    import torch

    probe = _probe()
    precision = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    expected = torch.ones(1, 3, 8, dtype=precision)
    actual, oracle = expected.clone(), expected.double()
    if fault == "fast_dtype":
        actual = actual.float()  # Numerically identical, but not the measured precision.
    elif fault == "fast_shape":
        actual = actual.expand(2, 3, 8)  # Broadcast subtraction would silently pass.
    elif fault == "oracle_shape":
        oracle = oracle.expand(2, 3, 8)
    elif fault == "fast_device":
        actual = actual.to("meta")
    else:
        oracle = oracle.to("meta")
    with pytest.raises(AssertionError, match="dtype|shape|device"):
        probe._comparison(actual, expected, oracle, dtype)


@pytest.mark.parametrize("start_at", ["gate", "first_fast_arm"])
def test_timing_rejects_promoted_output_inside_genuine_custom_function(
    monkeypatch, tmp_path, start_at
):
    import torch

    from soup_cli.utils.fast_lora import _single_projection_function

    probe = _probe()
    function = _single_projection_function()
    original_forward, original_measure = function.forward, probe._measure
    measure_calls = 0
    promoted = 0

    def forward(ctx, *args):
        nonlocal promoted
        output = original_forward(ctx, *args)
        if output.dtype == torch.float16 and (start_at == "gate" or measure_calls >= 3):
            promoted += 1
            return output.float()  # Cast *inside* Function: retain genuine backward node.
        return output

    def measure(*args, **kwargs):
        nonlocal measure_calls
        measure_calls += 1
        result = original_measure(*args, **kwargs)
        if measure_calls == 3:
            assert type(result[0][0].grad_fn).__name__ == probe.EXPECTED_GRAD_FN["single"]
        return result

    monkeypatch.setattr(function, "forward", staticmethod(forward))
    monkeypatch.setattr(probe, "_measure", measure)
    with pytest.raises(AssertionError, match="dtype") as failure:
        probe.run_timing(dtype="fp16", warmup=0, repeats=1, tokens=3)
    report = failure.value.evidence_report
    probe._write_evidence(report, str(tmp_path / f"promoted-{start_at}"))
    assert promoted > 0
    assert report["passed"] is False
    assert measure_calls == (0 if start_at == "gate" else 4)
    assert "NO VERDICT" in report["timing_verdict"]
    if start_at == "first_fast_arm":
        bad = next(row for row in report["rows"] if row.get("arm_index") == 2)
        assert bad["validity"]["status"] == "VOID"
        assert bad["forward_ms"] > 0 and bad["backward_ms"] > 0


def test_low_precision_comparison_respects_reference_fp32_adapter_gradients():
    import torch

    expected = torch.ones(2, 8, dtype=torch.float32)
    result = _probe()._comparison(expected.clone(), expected, expected.double(), "fp16")
    assert result["passed"] and result["bit_exact"]


@pytest.mark.parametrize("fail_arm", ["PEFT", "fast"])
def test_loss_failure_retains_each_available_forward_loss(monkeypatch, tmp_path, fail_arm):
    probe = _probe()
    original_model, original_check = probe._loss_model, probe._check_training_gradients
    measured_losses = []
    check_calls = 0

    def model(*args, **kwargs):
        fixture = original_model(*args, **kwargs)

        def record(_model, _inputs, output):
            measured_losses.append(output.loss.detach().item())

        fixture.register_forward_hook(record)
        return fixture

    def check(fixture):
        nonlocal check_calls
        check_calls += 1
        if check_calls == (1 if fail_arm == "PEFT" else 2):
            raise RuntimeError(f"SYNTHETIC {fail_arm} gradient check interrupted")
        return original_check(fixture)

    monkeypatch.setattr(probe, "_loss_model", model)
    monkeypatch.setattr(probe, "_check_training_gradients", check)
    prefix = tmp_path / f"partial-loss-{fail_arm}"
    assert probe.main(["loss", "--steps", "1", "--output-prefix", str(prefix)]) == 2
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert len(report["rows"]) == 1
    partial = report["rows"][0]
    assert partial["row_type"] == "failure"
    assert partial["step"] == 1
    assert partial["incomplete"] is True
    assert partial["phase"] == "gradient_check"
    assert partial["implementation"] == fail_arm
    assert partial["baseline_loss"] == measured_losses[1]  # Skip negative-control forward.
    if fail_arm == "fast":
        assert partial["fast_loss"] == measured_losses[2]
    else:
        assert "fast_loss" not in partial  # That forward never ran.
    assert "rounded_3_equal" not in partial
    assert not any("grad" in key or key.endswith("_ms") for key in partial)
    assert "SYNTHETIC" in partial["error"] and "Traceback" in partial["traceback"]
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 1
    assert float(csv_rows[0]["baseline_loss"]) == partial["baseline_loss"]
    if fail_arm == "fast":
        assert float(csv_rows[0]["fast_loss"]) == partial["fast_loss"]
    else:
        assert "fast_loss" not in csv_rows[0]


@pytest.mark.parametrize("fail_step", [1, 2], ids=["first", "late"])
@pytest.mark.parametrize("fail_arm", ["fast", "PEFT"])
def test_zero_grad_interruption_keeps_truthful_unexecuted_mode_metadata(
    tmp_path, fail_arm, fail_step
):
    prefix = tmp_path / f"zero-grad-{fail_arm}-{fail_step}"
    code = (
        "import importlib.util, json, sys, torch\n"
        "from pathlib import Path\n"
        "spec = importlib.util.spec_from_file_location('probe', sys.argv[1])\n"
        "p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)\n"
        "original_model = p._loss_model\n"
        "original_hooks = p._execution_hooks\n"
        "original_modes = p._reference_arithmetic_hooks\n"
        "original_zero_grad = torch.optim.AdamW.zero_grad\n"
        "measured = {'losses': [], 'execution': [], 'modes': {}, 'zero_grad_calls': 0}\n"
        "def model(*args, **kwargs):\n"
        "    fixture = original_model(*args, **kwargs)\n"
        "    def record(_model, _inputs, output):\n"
        "        measured['losses'].append(output.loss.detach().item())\n"
        "    fixture.register_forward_hook(record)\n"
        "    return fixture\n"
        "def hooks(model):\n"
        "    observed, handles = original_hooks(model)\n"
        "    measured['execution'].append(observed)\n"
        "    return observed, handles\n"
        "def modes(model):\n"
        "    observed, handles = original_modes(model)\n"
        "    measured['modes'] = observed\n"
        "    return observed, handles\n"
        "def zero_grad(optimizer, *args, **kwargs):\n"
        "    measured['zero_grad_calls'] += 1\n"
        "    fail_call = 2 * int(sys.argv[3]) - (sys.argv[4] == 'PEFT')\n"
        "    if measured['zero_grad_calls'] == fail_call:\n"
        "        raise RuntimeError('SYNTHETIC zero_grad interruption')\n"
        "    return original_zero_grad(optimizer, *args, **kwargs)\n"
        "p._loss_model = model\n"
        "p._execution_hooks = hooks\n"
        "p._reference_arithmetic_hooks = modes\n"
        "torch.optim.AdamW.zero_grad = zero_grad\n"
        "status = p.main(['loss', '--dtype', 'fp16', '--steps', '2',\n"
        "                 '--output-prefix', sys.argv[2]])\n"
        "Path(sys.argv[2] + '.measured.json').write_text(json.dumps(measured), encoding='utf-8')\n"
        "sys.exit(status)\n"
    )
    command = [sys.executable, "-c", code, str(HARNESS), str(prefix), str(fail_step), fail_arm]
    (tmp_path / "command.json").write_text(json.dumps(command), encoding="utf-8")
    with (tmp_path / "stdout.log").open("wb") as out, (tmp_path / "stderr.log").open("wb") as err:
        result = subprocess.run(
            command,
            env={**os.environ, "PYTHONPATH": str(HARNESS.parents[2] / "src")},
            stdout=out, stderr=err, check=False, timeout=120,
        )
    (tmp_path / "exit_code.txt").write_text(str(result.returncode), encoding="utf-8")
    assert result.returncode == 2
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    measured = json.loads(Path(str(prefix) + ".measured.json").read_text(encoding="utf-8"))
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert report["passed"] is False
    assert len(report["rows"]) == len(csv_rows) == fail_step
    assert measured["zero_grad_calls"] == 2 * fail_step - (fail_arm == "PEFT")
    assert len(measured["losses"]) == 1 + 2 * (fail_step - 1) + (fail_arm == "fast")
    expected = {
        "qkv": [
            f"base_model.model.model.layers.{layer}.self_attn.{projection}"
            for layer in range(2) for projection in ("q_proj", "k_proj", "v_proj")
        ],
        "mlp": [f"base_model.model.model.layers.{layer}.mlp" for layer in range(2)],
    }
    assert report["reference_arithmetic"]["expected_modules"] == expected
    assert report["reference_arithmetic"]["required_true"] is True
    for index, (completed, raw) in enumerate(zip(report["rows"][:-1], csv_rows[:-1])):
        assert completed["step"] == index + 1
        assert completed["baseline_loss"] == measured["losses"][1 + 2 * index]
        assert completed["fast_loss"] == measured["losses"][2 + 2 * index]
        assert completed["rounded_3_equal"] is True
        assert completed["reference_arithmetic_status"] == "observed"
        assert completed["reference_arithmetic_modes"] == {
            path: dict.fromkeys(names, True) for path, names in expected.items()
        }
        assert float(raw["baseline_loss"]) == completed["baseline_loss"]
        assert float(raw["fast_loss"]) == completed["fast_loss"]
        assert raw["rounded_3_equal"] == "True"
        assert raw["reference_arithmetic_status"] == completed["reference_arithmetic_status"]
        assert json.loads(raw["reference_arithmetic_modes"]) == (
            completed["reference_arithmetic_modes"]
        )
    partial, raw = report["rows"][-1], csv_rows[-1]
    assert partial["failure_case"] == report["failure_case"] == {
        "stage": "training", "step": fail_step, "implementation": fail_arm, "phase": "zero_grad"
    }
    assert partial["step"] == fail_step and partial["phase"] == "zero_grad"
    assert partial["row_type"] == "failure" and partial["incomplete"] is True
    assert "fast_loss" not in partial and not raw.get("fast_loss")
    assert "rounded_3_equal" not in partial and not raw.get("rounded_3_equal")
    assert raw["row_type"] == "failure" and raw["incomplete"] == "True"
    assert raw["step"] == str(fail_step) and raw["phase"] == "zero_grad"
    assert raw["implementation"] == fail_arm
    assert json.loads(raw["failure_case"]) == partial["failure_case"]
    assert "SYNTHETIC zero_grad interruption" in partial["error"]
    assert "Traceback" in partial["traceback"]
    assert raw["error"] == partial["error"] == report["error"]
    assert raw["traceback"] == partial["traceback"]
    assert "Traceback" in report["traceback"]
    assert "SYNTHETIC zero_grad interruption" in report["traceback"]
    if fail_arm == "fast":
        assert partial["baseline_loss"] == measured["losses"][-1]
        assert float(raw["baseline_loss"]) == partial["baseline_loss"]
        assert partial["reference_arithmetic_status"] == raw["reference_arithmetic_status"] == (
            "INSUFFICIENT"
        )
        sentinels = {path: dict.fromkeys(names, "not executed") for path, names in expected.items()}
        assert partial["reference_arithmetic_modes"] == measured["modes"] == sentinels
        assert json.loads(raw["reference_arithmetic_modes"]) == sentinels
        execution_expected = {
            **expected,
            "single": [
                f"base_model.model.model.layers.{layer}.self_attn.o_proj" for layer in range(2)
            ],
        }
        assert measured["execution"][0] == {
            path: dict.fromkeys(names, "not executed") for path, names in execution_expected.items()
        }
    else:
        assert "baseline_loss" not in partial and not raw.get("baseline_loss")
        for key in ("reference_arithmetic_status", "reference_arithmetic_modes"):
            assert key not in partial and not raw.get(key)  # This fast arm was never entered.


@pytest.mark.parametrize("fallback_path", ["all", "single", "qkv", "mlp"])
def test_loss_rejects_replaced_peft_modules_after_a_valid_fast_step(monkeypatch, fallback_path):
    probe = _probe()
    original_model, original_patch = probe._loss_model, probe._patch
    original_verify = probe._verify_execution
    measured_losses, execution_checks, fallback_nodes, replaced = [], [], [], []
    handles = []
    fast_forwards = 0

    def model(*args, **kwargs):
        fixture = original_model(*args, **kwargs)

        def record(_model, _inputs, output):
            measured_losses.append(output.loss.detach().item())

        fixture.register_forward_hook(record)
        return fixture

    def patch(fixture, path):
        count = original_patch(fixture, path)
        if path == "single":

            def switch_after_first_step(fixture, _inputs):
                nonlocal fast_forwards
                fast_forwards += 1
                if fast_forwards != 2:
                    return
                parameters = dict(fixture.named_parameters())
                named = list(fixture.named_modules())
                if fallback_path == "all":
                    targets = {name for names in probe.PROJECTIONS.values() for name in names}
                    selected = [
                        (name, module)
                        for name, module in named
                        if name.split(".")[-1] in targets
                        or all(hasattr(module, name) for name in probe.PROJECTIONS["mlp"])
                    ]
                else:
                    suffix = {"single": ".o_proj", "qkv": ".v_proj", "mlp": ".mlp"}[
                        fallback_path
                    ]
                    selected = [(name, module) for name, module in named if name.endswith(suffix)][
                        -1:
                    ]
                    if fallback_path == "mlp":
                        # The single patcher also wraps the MLP's projection forwards.
                        prefix = selected[0][0] + "."
                        selected.extend(
                            (name, module)
                            for name, module in named
                            if name.startswith(prefix)
                            and name.split(".")[-1] in probe.PROJECTIONS["mlp"]
                        )
                assert selected
                for name, module in reversed(selected):
                    replacement = copy.copy(module)
                    replacement._forward_hooks = OrderedDict()
                    replacement._forward_pre_hooks = OrderedDict()
                    replacement.forward = type(module).forward.__get__(replacement, type(module))
                    parent_name, _, child = name.rpartition(".")
                    setattr(fixture.get_submodule(parent_name), child, replacement)
                    replaced.append(name)

                    def observe(_module, _inputs, output):
                        fallback_nodes.append(type(output.grad_fn).__name__)

                    handles.append(replacement.register_forward_hook(observe))
                # Keep optimizer ownership and numerical parity: replace no Parameters.
                assert dict(fixture.named_parameters()).keys() == parameters.keys()
                assert all(
                    parameter is parameters[name] for name, parameter in fixture.named_parameters()
                )

            handles.append(fixture.register_forward_pre_hook(switch_after_first_step))
        return count

    def verify(counts, observed):
        execution_checks.append(copy.deepcopy(observed))
        return original_verify(counts, observed)

    monkeypatch.setattr(probe, "_loss_model", model)
    monkeypatch.setattr(probe, "_patch", patch)
    monkeypatch.setattr(probe, "_verify_execution", verify)
    try:
        with pytest.raises(
            AssertionError, match="custom autograd missing:.*not executed"
        ) as failure:
            probe.run_loss(seed=792, steps=2, device="cpu", dtype="fp32")
    finally:
        for handle in handles:
            handle.remove()

    assert fast_forwards == 2
    assert fallback_nodes and all(node == "AddBackward0" for node in fallback_nodes)
    assert len(execution_checks) == 3  # Unpatched control, valid step 1, rejected step 2.
    control, first, second = execution_checks
    for path in probe.PROJECTIONS:
        assert control[path] and all(node == "AddBackward0" for node in control[path].values())
        assert first[path] and all(
            node == probe.EXPECTED_GRAD_FN[path] for node in first[path].values()
        )
        assert second[path].keys() == first[path].keys()  # Never drop missing expected modules.
        for name, node in second[path].items():
            assert node == ("not executed" if name in replaced else probe.EXPECTED_GRAD_FN[path])
    assert any(name in replaced for group in first.values() for name in group)
    report = failure.value.evidence_report
    assert report["passed"] is False
    assert len(report["rows"]) == 2 and len(measured_losses) == 5
    completed, partial = report["rows"]
    assert completed["step"] == 1 and completed["rounded_3_equal"] is True
    assert completed["baseline_loss"] == measured_losses[1]
    assert completed["fast_loss"] == measured_losses[2]
    assert partial["row_type"] == "failure" and partial["incomplete"] is True
    assert partial["failure_case"] == {
        "stage": "training", "step": 2, "implementation": "fast", "phase": "execution_check"
    }
    assert partial["baseline_loss"] == measured_losses[3]
    assert partial["fast_loss"] == measured_losses[4]
    assert "rounded_3_equal" not in partial and "Traceback" in partial["traceback"]
    for path, names in report["reference_arithmetic"]["expected_modules"].items():
        assert set(completed["reference_arithmetic_modes"][path]) == set(names)
        assert set(partial["reference_arithmetic_modes"][path]) == set(names)
        for name in names:
            assert partial["reference_arithmetic_modes"][path][name] == (
                "not executed" if name in replaced else False
            )
    assert partial["reference_arithmetic_status"] == "INSUFFICIENT"


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("path", ["qkv", "mlp"])
def test_loss_rejects_late_genuine_legacy_arithmetic_with_unchanged_function_names(
    monkeypatch, dtype, path
):
    from soup_cli.utils import fast_lora_mlp, fast_lora_qkv

    probe = _probe()
    kernel = {"qkv": fast_lora_qkv, "mlp": fast_lora_mlp}[path]
    factory_name = f"_{path}_function"
    original_factory = getattr(kernel, factory_name)
    original_patch, original_verify = probe._patch, probe._verify_execution
    fast_forwards = 0
    execution = []

    def patch(model, group):
        count = original_patch(model, group)
        if group == "single":
            def begin(_model, _inputs):
                nonlocal fast_forwards
                fast_forwards += 1

            model.register_forward_pre_hook(begin)
        return count

    def select(*, reference_order=False):
        # Invoke the actual legacy Function after a genuine scoped first step.
        return original_factory(reference_order=reference_order and fast_forwards < 2)

    def verify(counts, observed):
        execution.append(copy.deepcopy(observed))
        return original_verify(counts, observed)

    monkeypatch.setattr(probe, "_patch", patch)
    monkeypatch.setattr(probe, "_verify_execution", verify)
    monkeypatch.setattr(kernel, factory_name, select)
    with pytest.raises(AssertionError, match="reference arithmetic.*True") as failure:
        probe.run_loss(seed=792, steps=2, device="cpu", dtype=dtype)
    report = failure.value.evidence_report
    assert fast_forwards == 2
    assert len(execution) == 3
    for group in probe.PROJECTIONS:
        assert all(node == probe.EXPECTED_GRAD_FN[group] for node in execution[-1][group].values())
    completed, partial = report["rows"]
    assert completed["reference_arithmetic_status"] == "observed"
    assert all(mode is True for values in completed["reference_arithmetic_modes"].values()
               for mode in values.values())
    assert partial["reference_arithmetic_status"] == "INSUFFICIENT"
    assert all(mode is False for mode in partial["reference_arithmetic_modes"][path].values())
    assert partial["phase"] == "reference_arithmetic_check" and partial["step"] == 2
    assert isinstance(partial["baseline_loss"], float) and isinstance(partial["fast_loss"], float)
    assert "rounded_3_equal" not in partial


@pytest.mark.parametrize("path", ["qkv", "mlp"])
@pytest.mark.parametrize("fault", ["missing", "none", "truthy_int"])
def test_loss_rejects_late_missing_or_nonboolean_mode_on_genuine_node(monkeypatch, path, fault):
    from soup_cli.utils import fast_lora_mlp, fast_lora_qkv

    probe = _probe()
    kernel = {"qkv": fast_lora_qkv, "mlp": fast_lora_mlp}[path]
    function = getattr(kernel, f"_{path}_function")(reference_order=True)
    original_forward, original_patch = function.forward, probe._patch
    forwards = 0

    def patch(model, group):
        count = original_patch(model, group)
        if group == "single":
            def begin(_model, _inputs):
                nonlocal forwards
                forwards += 1

            model.register_forward_pre_hook(begin)
        return count

    def forward(ctx, *args):
        result = original_forward(ctx, *args)
        if forwards == 2:
            if fault == "missing":
                del ctx.reference_order
            else:
                ctx.reference_order = None if fault == "none" else 1
        return result

    monkeypatch.setattr(probe, "_patch", patch)
    monkeypatch.setattr(function, "forward", staticmethod(forward))
    with pytest.raises(AssertionError, match="insufficient reference arithmetic") as failure:
        probe.run_loss(steps=2, dtype="fp16")
    completed, partial = failure.value.evidence_report["rows"]
    assert forwards == 2 and completed["reference_arithmetic_status"] == "observed"
    assert all(mode is True for mode in completed["reference_arithmetic_modes"][path].values())
    expected = {"missing": "missing reference_order", "none": "invalid reference_order (NoneType)",
                "truthy_int": "invalid reference_order (int)"}[fault]
    assert set(partial["reference_arithmetic_modes"][path].values()) == {expected}
    assert partial["reference_arithmetic_status"] == "INSUFFICIENT"
    assert partial["phase"] == "reference_arithmetic_check"


@pytest.mark.parametrize("path", ["qkv", "mlp"])
def test_loss_resets_each_mode_key_when_its_observer_is_removed_late(monkeypatch, path, tmp_path):
    probe = _probe()
    original = probe._reference_arithmetic_hooks
    forwards = 0
    target = None

    def hooks(model):
        nonlocal target
        observed, handles = original(model)
        target = list(observed[path])[-1]
        module = model.get_submodule(target)
        selected = next(handle for handle in handles if handle.id in module._forward_hooks)

        def begin(_model, _inputs):
            nonlocal forwards
            forwards += 1
            if forwards == 2:
                selected.remove()  # Function-name observers remain installed and genuine.

        model.register_forward_pre_hook(begin)
        return observed, handles

    monkeypatch.setattr(probe, "_reference_arithmetic_hooks", hooks)
    prefix = tmp_path / f"late-unhooked-{path}"
    assert probe.main(["loss", "--dtype", "fp16", "--steps", "2",
                       "--output-prefix", str(prefix)]) == 2
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    completed, partial = report["rows"]
    assert forwards == 2 and report["passed"] is False
    assert completed["reference_arithmetic_modes"][path][target] is True
    assert partial["reference_arithmetic_modes"][path][target] == "not executed"
    assert partial["reference_arithmetic_status"] == "INSUFFICIENT"
    assert partial["phase"] == "reference_arithmetic_check"
    for group, names in report["reference_arithmetic"]["expected_modules"].items():
        assert set(completed["reference_arithmetic_modes"][group]) == set(names)
        assert set(partial["reference_arithmetic_modes"][group]) == set(names)
    assert "rounded_3_equal" not in partial
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert json.loads(csv_rows[-1]["reference_arithmetic_modes"]) == (
        partial["reference_arithmetic_modes"]
    )


def test_partial_fast_forward_preserves_only_modes_that_really_executed(monkeypatch):
    probe = _probe()
    original = probe._patch
    forwards = 0
    handles = []

    def patch(model, path):
        count = original(model, path)
        if path == "single":
            def begin(_model, _inputs):
                nonlocal forwards
                forwards += 1

            def interrupt(_module, _inputs, _output):
                if forwards == 2:
                    raise RuntimeError("SYNTHETIC interrupted second fast forward")

            handles.append(model.register_forward_pre_hook(begin))
            handles.append(model.get_base_model().model.layers[1].self_attn.q_proj.register_forward_hook(
                interrupt
            ))
        return count

    monkeypatch.setattr(probe, "_patch", patch)
    try:
        with pytest.raises(RuntimeError, match="interrupted second fast forward") as failure:
            probe.run_loss(steps=2, dtype="fp16")
    finally:
        for handle in handles:
            handle.remove()
    completed, partial = failure.value.evidence_report["rows"]
    assert completed["reference_arithmetic_status"] == "observed"
    assert partial["phase"] == "forward"
    assert partial["reference_arithmetic_status"] == "INSUFFICIENT"
    assert "fast_loss" not in partial and isinstance(partial["baseline_loss"], float)
    for group, values in partial["reference_arithmetic_modes"].items():
        for name, mode in values.items():
            assert mode == (True if ".layers.0." in name else "not executed"), (group, name, mode)


@pytest.fixture(scope="module")
def real_cpu_timing_report():
    return _probe().run_timing(warmup=0, repeats=1, tokens=3)


@pytest.mark.parametrize(
    "metadata",
    [
        "missing", None, {}, [], "bad", {"available": False}, {"status": None},
        {"status": []}, {"status": "UNKNOWN"}, {"status": "CPU DEBUG-ONLY"},
        {"status": "CPU DEBUG-ONLY", "available": "False"},
    ],
)
def test_timing_renderer_excludes_missing_or_malformed_validity(real_cpu_timing_report, metadata):
    probe = _probe()
    report = copy.deepcopy(real_cpu_timing_report)
    for row in report["rows"]:
        if metadata == "missing":
            row.pop("validity")
        else:
            row["validity"] = copy.deepcopy(metadata)
    raw_rows = copy.deepcopy(report["rows"])
    stream = StringIO()
    probe._render(report, probe.Console(file=stream, width=160))
    output = stream.getvalue()
    assert "NO VERDICT" in output and "insufficient" in output.lower()
    assert "n/a" in output
    assert report["rows"] == raw_rows  # Never replace the malformed raw evidence.
    assert len(report["rows"]) == 24
    assert "NO VERDICT" in report["timing_verdict"]


def test_timing_renderer_cannot_promote_cpu_debug_to_a_verdict(real_cpu_timing_report):
    probe = _probe()
    report = copy.deepcopy(real_cpu_timing_report)
    report["timing_verdict"] = "SYNTHETIC erroneous performance approval"
    stream = StringIO()
    probe._render(report, probe.Console(file=stream, width=160))
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stream.getvalue())
    assert "NO VERDICT (CPU DEBUG-ONLY)" in plain
    assert "erroneous performance approval" not in plain


def test_cli_persists_insufficient_validity_without_losing_raw_arms(
    monkeypatch, tmp_path, real_cpu_timing_report
):
    probe = _probe()
    result = copy.deepcopy(real_cpu_timing_report)
    for row in result["rows"]:
        row.pop("validity")
    raw_rows = copy.deepcopy(result["rows"])
    monkeypatch.setattr(probe, "run_timing", lambda **kwargs: result)
    prefix = tmp_path / "missing-validity"
    assert probe.main(["timing", "--output-prefix", str(prefix)]) == 0
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert "NO VERDICT" in report["timing_verdict"]
    assert "insufficient" in report["timing_verdict"].lower()
    assert report["rows"] == raw_rows
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(raw_rows)


@pytest.mark.parametrize("runtime_failure", [False, True])
@pytest.mark.parametrize("broken_console", [False, True])
def test_cli_render_failure_is_recorded_without_losing_probe_evidence(
    monkeypatch, tmp_path, real_cpu_timing_report, runtime_failure, broken_console
):
    probe = _probe()
    result = copy.deepcopy(real_cpu_timing_report)
    if runtime_failure:
        result["passed"] = False
        result["error"] = "SYNTHETIC earlier runtime failure"
        result["traceback"] = "SYNTHETIC earlier runtime traceback"

    def render(*args, **kwargs):
        raise KeyError("SYNTHETIC renderer failure")

    monkeypatch.setattr(probe, "run_timing", lambda **kwargs: result)
    monkeypatch.setattr(probe, "_render", render)
    if broken_console:
        class BrokenConsole:
            def print(self, *args, **kwargs):
                raise OSError("SYNTHETIC unavailable console after renderer failure")

        monkeypatch.setattr(probe, "Console", BrokenConsole)
    prefix = tmp_path / "render-failure"
    assert probe.main(["timing", "--output-prefix", str(prefix)]) == 2
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert "NO VERDICT" in report["timing_verdict"]
    assert report["rows"][:24] == real_cpu_timing_report["rows"]
    failed = report["rows"][-1]
    assert failed["row_type"] == "failure"
    assert failed["failure_case"]["stage"] == "render"
    assert "SYNTHETIC renderer failure" in failed["error"]
    assert "Traceback" in failed["traceback"]
    if runtime_failure:
        assert report["error"] == "SYNTHETIC earlier runtime failure"
        assert report["traceback"] == "SYNTHETIC earlier runtime traceback"
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(report["rows"]) == 25
    assert csv_rows[-1]["row_type"] == "failure"
