"""Reproducible D2 Fast-LoRA evidence, not a full-run performance benchmark.

All weights/inputs are SYNTHETIC. The baseline is real, unpatched PEFT.
Heavy training dependencies are imported only when a probe is run.
Loss mode records per-step grad_fn.reference_order for declared QKV/MLP modules;
fp16/bf16 require literal True, fp32 records mode without requiring True.
CPU evidence covers a finite SYNTHETIC fixture, not global bit-exactness,
CUDA/NF4 validation or a performance claim.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from rich import box
from rich.console import Console
from rich.table import Table

PROJECTIONS = {
    "single": ("o_proj",),
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}
EXPECTED_GRAD_FN = {
    "single": "_FastLoraSingleProjectionBackward",
    "qkv": "_FastLoraQKVBackward",
    "mlp": "_FastLoraSwiGLUBackward",
}
SHAPES = {
    "tiny": {"hidden": 8, "kv": 4, "intermediate": 12, "rank": 2},
    "llama3.1-8b": {"hidden": 4096, "kv": 1024, "intermediate": 14336, "rank": 16},
}


def _make_fixture(path: str, dtype: Any, device: str, shapes: str = "tiny") -> Any:
    """Construct structural single/GQA/SwiGLU fixtures using actual PEFT layers."""
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from torch import nn

    shape = SHAPES[shapes]
    hidden, kv, intermediate, rank = (
        shape[key] for key in ("hidden", "kv", "intermediate", "rank")
    )

    class Group(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if path == "qkv":
                self.q_proj = nn.Linear(hidden, hidden, bias=False)
                self.k_proj = nn.Linear(hidden, kv, bias=False)
                self.v_proj = nn.Linear(hidden, kv, bias=False)
            else:
                self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
                self.up_proj = nn.Linear(hidden, intermediate, bias=False)
                self.down_proj = nn.Linear(intermediate, hidden, bias=False)
                self.act_fn = nn.SiLU()

        def forward(self, x: Any) -> Any:
            if path == "qkv":
                return self.q_proj(x), self.k_proj(x), self.v_proj(x)
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class Fixture(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_act="silu")
            if path == "single":
                self.o_proj = nn.Linear(hidden, hidden, bias=False)
            else:
                self.group = Group()

        def forward(self, x: Any) -> Any:
            return self.o_proj(x) if path == "single" else self.group(x)

    model = Fixture()
    inject_adapter_in_model(
        LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(PROJECTIONS[path]),
        ),
        model,
    )
    # PEFT normally initialises B to zero; nonzero B exercises dA and LoRA dX.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    return model.to(device=device, dtype=dtype)


def _patch(model: Any, path: str) -> int:
    from soup_cli.utils.fast_lora import patch_fast_lora_single_projection
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    patchers = {
        "single": patch_fast_lora_single_projection,
        "qkv": patch_fast_lora_qkv,
        "mlp": patch_fast_lora_mlp,
    }
    return patchers[path](model)


def _outputs(model: Any, x: Any) -> tuple[Any, ...]:
    result = model(x)
    return result if isinstance(result, tuple) else (result,)


def _require(condition: Any, message: str) -> None:
    """Evidence checks must survive python -O and PYTHONOPTIMIZE."""
    if not condition:
        raise AssertionError(message)


def require_finite(tensor: Any, label: str) -> None:
    """Reject missing gradients and NaN/Inf rather than omitting evidence rows."""
    import torch

    _require(tensor is not None, f"missing tensor/gradient: {label}")
    _require(torch.isfinite(tensor).all().item(), f"nonfinite tensor/gradient: {label}")


def _expected_quantities(path: str, request_dx: bool) -> set[str]:
    outputs = PROJECTIONS[path] if path == "qkv" else ("Y",)
    return (
        {f"Y/{label}" for label in outputs}
        | ({"dX"} if request_dx else set())
        | {
            f"{gradient}/{projection}"
            for gradient in ("dA", "dB")
            for projection in PROJECTIONS[path]
        }
    )


def _validate_snapshot_keys(path: str, request_dx: bool, **snapshots: Any) -> None:
    required = _expected_quantities(path, request_dx)
    for label, snapshot in snapshots.items():
        found = set(snapshot)
        _require(
            found == required,
            f"{label} {path} snapshot keys: missing={sorted(required - found)}, "
            f"unexpected={sorted(found - required)}",
        )


def _validate_adapter_registration(model: Any, path: str) -> None:
    """Derive mandatory A/B from fixture topology, not discovered parameters."""
    registered = dict(model.named_parameters())
    required = set()
    for projection in PROJECTIONS[path]:
        prefix = projection if path == "single" else f"group.{projection}"
        module = model.get_submodule(prefix)
        for matrix in ("A", "B"):
            name = f"{prefix}.lora_{matrix}.default.weight"
            required.add(name)
            parameter = getattr(module, f"lora_{matrix}")["default"].weight
            _require(registered.get(name) is parameter, f"missing/unregistered adapter: {name}")
            _require(parameter.requires_grad, f"adapter not trainable: {name}")
    found = {name for name in registered if "lora_" in name}
    _require(found == required, f"adapter keys mismatch: {path}: {sorted(found ^ required)}")


def _snapshot(model: Any, x: Any, upstream: tuple[Any, ...], path: str) -> dict[str, Any]:
    import torch

    _validate_adapter_registration(model, path)
    model.zero_grad(set_to_none=True)
    x.grad = None
    outputs = _outputs(model, x)
    _require(len(outputs) == len(upstream) == (3 if path == "qkv" else 1), "output count mismatch")
    for out in outputs:
        require_finite(out, "output")
    torch.autograd.backward(outputs, upstream)
    return _collect_snapshot(model, x, outputs, path)


def _collect_snapshot(model: Any, x: Any, outputs: tuple[Any, ...], path: str) -> dict[str, Any]:
    """Read the just-executed arm without another forward or backward."""
    _validate_adapter_registration(model, path)
    _require(len(outputs) == (3 if path == "qkv" else 1), "output count mismatch")
    for out in outputs:
        require_finite(out, "output")
    result = {}
    if x.requires_grad:
        require_finite(x.grad, "dX")
        result["dX"] = x.grad.detach().clone()
    else:
        _require(x.grad is None, "unexpected dX")
    labels = PROJECTIONS["qkv"] if len(outputs) == 3 else ("Y",)
    result.update({f"Y/{label}": out.detach().clone() for label, out in zip(labels, outputs)})
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            require_finite(parameter.grad, name)
            gradient = "dA" if "lora_A" in name else "dB"
            projection = name.split(".lora_")[0].split(".")[-1]
            result[f"{gradient}/{projection}"] = parameter.grad.detach().clone()
        else:
            _require(
                not parameter.requires_grad and parameter.grad is None, f"base not frozen: {name}"
            )
    _validate_snapshot_keys(path, x.requires_grad, snapshot=result)
    return result


def _comparison(actual: Any, expected: Any, oracle: Any, dtype: str) -> dict[str, Any]:
    """Float32 uses torch defaults; low precision uses the PROPOSED max-error bound.

    Per tensor: fast float64 max error <= 2 * PEFT float64 max error + 1e-8.
    This permits dtype rounding, not an assertion of bit-exact low-precision math.
    The float64 oracle uses the SAME rounded weights, input and upstream gradient.
    """
    import torch

    # Compare the original tensor contracts before casts or broadcast arithmetic.
    # Adapter gradients may be fp32 even when the requested base precision is fp16;
    # the PEFT tensor, not the CLI label, defines each quantity's dtype contract.
    for label, tensor in (("fast", actual), ("PEFT", expected), ("float64", oracle)):
        _require(tensor is not None, f"missing tensor/gradient: {label}")
    _require(
        actual.dtype == expected.dtype,
        f"dtype mismatch: fast {actual.dtype}, PEFT {expected.dtype}",
    )
    _require(
        actual.shape == expected.shape == oracle.shape,
        f"shape mismatch: fast {actual.shape}, PEFT {expected.shape}, float64 {oracle.shape}",
    )
    _require(
        actual.device == expected.device == oracle.device,
        f"device mismatch: fast {actual.device}, PEFT {expected.device}, float64 {oracle.device}",
    )
    for label, tensor in (("fast", actual), ("PEFT", expected), ("float64", oracle)):
        require_finite(tensor, label)
    actual64 = actual.double()
    expected64 = expected.double()
    fast_rmse = (actual64 - oracle).square().mean().sqrt().item()
    peft_rmse = (expected64 - oracle).square().mean().sqrt().item()
    fast_max_error = (actual64 - oracle).abs().max().item()
    peft_max_error = (expected64 - oracle).abs().max().item()
    bound = 2.0 * peft_max_error + 1e-8
    if dtype == "fp32":
        torch.testing.assert_close(actual, expected)
    else:
        _require(
            fast_max_error <= bound,
            f"float64 max error {fast_max_error} exceeds named PROPOSED bound {bound}",
        )
    return {
        "bit_exact": torch.equal(actual, expected),
        "max_abs_error": (actual64 - expected64).abs().max().item(),
        "fast_float64_rmse": fast_rmse,
        "peft_float64_rmse": peft_rmse,
        "proposed_max_error_bound": bound,
        "fast_float64_max_abs_error": fast_max_error,
        "peft_float64_max_abs_error": peft_max_error,
        "fast_peft_float64_max_error_ratio": (
            fast_max_error / peft_max_error if peft_max_error else None
        ),
        "finite": True,
        "passed": True,
    }


def _changed_adapter_control(
    baseline: Any,
    x: Any,
    expected: dict[str, Any],
    oracle: dict[str, Any],
    dtype: str,
) -> dict[str, Any]:
    """Prove the numerical gate rejects a real, deliberately perturbed PEFT adapter."""
    import torch

    changed = copy.deepcopy(baseline)
    name, parameter = next(
        (name, parameter) for name, parameter in changed.named_parameters() if "lora_B" in name
    )
    with torch.no_grad():
        parameter[0, 0].add_(0.5)
        outputs = _outputs(changed, x.detach())
    labels = PROJECTIONS["qkv"] if len(outputs) == 3 else ("Y",)
    detected = False
    max_delta = 0.0
    for label, output in zip(labels, outputs):
        quantity = f"Y/{label}"
        max_delta = max(
            max_delta, (output.double() - expected[quantity].double()).abs().max().item()
        )
        try:
            _comparison(output, expected[quantity], oracle[quantity], dtype)
        except AssertionError:
            detected = True
    _require(detected, "changed-adapter negative control was accepted by the numerical criterion")
    return {
        "changed_adapter_detected": True,
        "parameter": name,
        "perturbation": "B[0,0] += 0.5",
        "max_abs_output_delta": max_delta,
    }


def _gradcheck(path: str, device: str) -> dict[str, Any]:
    """Numerically differentiate real patched PEFT X and every adapter A/B."""
    import torch
    from torch.func import functional_call

    model = _make_fixture(path, torch.float64, device)
    _require(_patch(model, path) > 0, "evidence gate failed")
    _validate_adapter_registration(model, path)
    parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    names = list(parameters)
    values = tuple(
        parameter.detach().clone().requires_grad_(True) for parameter in parameters.values()
    )
    x = torch.randn(1, 1, 8, dtype=torch.float64, device=device, requires_grad=True)
    _require(
        all(type(out.grad_fn).__name__ == EXPECTED_GRAD_FN[path] for out in _outputs(model, x)),
        "custom autograd missing in gradcheck",
    )

    def call(input_x: Any, *adapters: Any) -> Any:
        return functional_call(model, dict(zip(names, adapters)), (input_x,))

    passed = torch.autograd.gradcheck(
        call,
        (x, *values),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    variables = ["X"]
    for name in names:
        matrix = "A" if "lora_A" in name else "B"
        variables.append(f"{matrix}/{name.split('.lora_')[0].split('.')[-1]}")
    return {
        "passed": passed,
        "dtype": "float64",
        "variables": variables,
        "eps": 1e-6,
        "atol": 1e-5,
        "rtol": 1e-4,
        "higher_order": "not tested; kernels are once_differentiable",
    }


def _retain_evidence(mode: str) -> Any:
    """Keep completed rows on original exceptions; direct callers still fail closed."""

    def decorate(function: Any) -> Any:
        @wraps(function)
        def call(**kwargs: Any) -> dict[str, Any]:
            report: dict[str, Any] = {
                "mode": mode,
                "rows": [],
                "passed": False,
                "failure_case": {"stage": "initialization"},
            }
            if mode == "timing":
                report["timing_verdict"] = "NO VERDICT (run incomplete)"
            try:
                report["environment"] = _environment(kwargs.get("device", "cpu"))
                _validate_environment(report["environment"])
                result = function(**kwargs, _evidence=report)
                result["environment"] = report["environment"]
                return result
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
                report["traceback"] = traceback.format_exc()
                partial = report.pop("pending_arm", {})
                if partial:
                    partial["validity"] = {
                        **partial["validity"],
                        "status": "VOID",
                        "reason": "arm interrupted",
                    }
                step = report.pop("pending_step", {})
                if step:
                    partial.update({**step, **report["failure_case"], "incomplete": True})
                report["rows"].append(
                    {
                        **partial,
                        "row_type": "failure",
                        "passed": False,
                        "failure_case": report["failure_case"],
                        "error": report["error"],
                        "traceback": report["traceback"],
                    }
                )
                exc.evidence_report = report
                raise

        return call

    return decorate


@_retain_evidence("parity")
def run_parity(
    *,
    seed: int = 792,
    dtype: str = "fp32",
    device: str = "cpu",
    gradcheck: bool = False,
    request_dx: bool = True,
    shapes: str = "tiny",
    allow_large_cpu: bool = False,
    _evidence: dict[str, Any],
) -> dict[str, Any]:
    """Compare every output and dX/dA/dB independently against unpatched PEFT."""
    import torch

    validate_runtime(device, dtype)
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    rows = _evidence["rows"]
    counts = {}
    if shapes != "tiny" and torch.device(device).type == "cpu" and not allow_large_cpu:
        raise ValueError("large CPU shapes require allow_large_cpu=True")
    checks = {}
    execution = {}
    controls = {}
    with _deterministic(seed, device):
        for path in PROJECTIONS:
            _evidence["failure_case"] = {"path": path, "stage": "fixture/patch"}
            baseline = _make_fixture(path, torch_dtype, device, shapes)
            oracle_model = copy.deepcopy(baseline).double()
            fast = copy.deepcopy(baseline)
            counts[path] = _patch(fast, path)
            _require(counts[path] > 0, f"no patch applied: {path}")
            x = torch.randn(
                2,
                3,
                SHAPES[shapes]["hidden"],
                dtype=torch_dtype,
                device=device,
                requires_grad=request_dx,
            )
            fast_x = x.detach().clone().requires_grad_(request_dx)
            fast_outputs = _outputs(fast, fast_x)
            execution[path] = [type(out.grad_fn).__name__ for out in fast_outputs]
            _require(
                all((grad_fn == EXPECTED_GRAD_FN[path] for grad_fn in execution[path])),
                f"custom autograd missing: {path}: {execution[path]}",
            )
            upstream = tuple(torch.randn_like(out) for out in fast_outputs)
            _evidence["failure_case"] = {"path": path, "stage": "snapshot"}
            actual = _snapshot(fast, fast_x, upstream, path)
            expected = _snapshot(baseline, x, upstream, path)
            oracle = _snapshot(
                oracle_model,
                x.detach().double().requires_grad_(request_dx),
                tuple(gradient.double() for gradient in upstream),
                path,
            )
            _validate_snapshot_keys(path, request_dx, fast=actual, PEFT=expected, oracle=oracle)
            _evidence["failure_case"] = {"path": path, "stage": "negative_control"}
            controls[path] = _changed_adapter_control(baseline, x, expected, oracle, dtype)
            for quantity, tensor in actual.items():
                _evidence["failure_case"] = {
                    "path": path,
                    "stage": "comparison",
                    "quantity": quantity,
                }
                reference = expected[quantity]
                rows.append(
                    {
                        "path": path,
                        "phase": "forward" if quantity.startswith("Y/") else "backward",
                        "quantity": quantity,
                        **_comparison(tensor, reference, oracle[quantity], dtype),
                    }
                )
            if gradcheck:
                _evidence["failure_case"] = {"path": path, "stage": "gradcheck"}
                checks[path] = _gradcheck(path, "cpu")
    return {
        "mode": "parity",
        "fixture": "SYNTHETIC",
        "reference": "unpatched PEFT",
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "base_frozen": True,
        "dropout": 0.0,
        "criterion": (
            "torch.testing.assert_close defaults (fp32)"
            if dtype == "fp32"
            else "PROPOSED: float64 max error <= 2*PEFT max error + 1e-8"
        ),
        "patch_counts": counts,
        "rows": rows,
        "passed": True,
        "gradcheck": checks,
        "execution": execution,
        "negative_control": controls,
        "deterministic_algorithms": True,
        "cpu_threads": 1,
        "request_dX": request_dx,
        "shapes": shapes,
        "shape_dimensions": SHAPES[shapes],
    }


def validate_runtime(device: str, dtype: str) -> None:
    """An unsupported CUDA/bf16 request is a failure, never a skipped passing row."""
    import torch

    if dtype not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"unsupported dtype: {dtype}")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU debug or CUDA evidence is supported")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; CPU is debug-only")
        if dtype == "bf16" and torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("native bf16 is unavailable on this CUDA card; request fp16 (T4)")


@contextmanager
def _deterministic(seed: int, device: str) -> Iterator[None]:
    """Restore RNG, CPU threads and deterministic flags even when a gate fails."""
    import torch

    threads = torch.get_num_threads()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cudnn_benchmark = torch.backends.cudnn.benchmark
    devices = []
    if torch.device(device).type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        index = torch.device(device).index
        devices = [torch.cuda.current_device() if index is None else index]
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.set_num_threads(threads)


def _loss_model(device: str, dtype: str) -> Any:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        hidden_act="silu",
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    targets = [name for names in PROJECTIONS.values() for name in names]
    model = get_peft_model(
        LlamaForCausalLM(config),
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    ).to(device=device)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    model.to(dtype=torch_dtype)
    # Keep AdamW states/updates in fp32 even on a T4; both paths use the same
    # mixed base/adapter dtypes. No native bf16 or Triton is needed for fp16.
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    return model


def _execution_hooks(model: Any) -> tuple[dict[str, dict[str, str]], list[Any]]:
    observed: dict[str, dict[str, str]] = {path: {} for path in PROJECTIONS}
    handles = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        path = None
        if leaf == "o_proj":
            path = "single"
        elif leaf in PROJECTIONS["qkv"]:
            path = "qkv"
        elif all(hasattr(module, projection) for projection in PROJECTIONS["mlp"]):
            path = "mlp"
        if path is not None:
            observed[path][name] = "not executed"

            def hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                key: str = name,
                group: str = path,
            ) -> None:
                observed[group][key] = type(output.grad_fn).__name__

            handles.append(module.register_forward_hook(hook))
    return observed, handles


def _reference_arithmetic_hooks(model: Any) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    """Pure output observers on every declared canonical loss-fixture QKV/MLP module.

    Keep this telemetry separate from the existing Function-name execution gates.
    Neither a Function name nor a previous forward establishes selected arithmetic.
    """
    layers = model.get_base_model().model.layers
    _require(len(layers) == model.config.num_hidden_layers, "reference arithmetic layer mismatch")
    registered = {id(module): name for name, module in model.named_modules()}
    observed: dict[str, dict[str, Any]] = {"qkv": {}, "mlp": {}}
    handles = []
    try:
        for layer in layers:
            modules = [("qkv", getattr(layer.self_attn, name)) for name in PROJECTIONS["qkv"]]
            modules.append(("mlp", layer.mlp))
            for path, module in modules:
                name = registered.get(id(module))
                if name is None:
                    raise AssertionError(f"unregistered reference arithmetic module: {path}")
                _require(name not in observed[path], f"shared reference arithmetic module: {name}")
                observed[path][name] = "not executed"

                def hook(
                    _module: Any,
                    _inputs: Any,
                    output: Any,
                    *,
                    key: str = name,
                    group: str = path,
                ) -> None:
                    missing = object()
                    mode = getattr(output.grad_fn, "reference_order", missing)
                    if type(mode) is bool:
                        observed[group][key] = mode
                    elif mode is missing:
                        observed[group][key] = "missing reference_order"
                    else:
                        observed[group][key] = f"invalid reference_order ({type(mode).__name__})"

                handles.append(module.register_forward_hook(hook))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return observed, handles


def _verify_reference_arithmetic(
    observed: dict[str, dict[str, Any]], expected: dict[str, list[str]], *, required_true: bool
) -> None:
    _require(set(observed) == set(expected), "reference arithmetic group keys mismatch")
    for path, names in expected.items():
        _require(
            names and set(observed[path]) == set(names),
            f"reference arithmetic module keys mismatch: {path}",
        )
        for name in names:
            mode = observed[path][name]
            _require(
                type(mode) is bool and (not required_true or mode is True),
                f"insufficient reference arithmetic evidence: {name}: {mode!r}; "
                f"expected {'literal True' if required_true else 'a literal bool'}",
            )


def _verify_execution(counts: dict[str, int], observed: dict[str, dict[str, str]]) -> None:
    for path in PROJECTIONS:
        _require(counts[path] > 0, f"no patch applied: {path}")
        _require(observed[path], f"no modules observed: {path}")
        for name, grad_fn in observed[path].items():
            _require(
                grad_fn == EXPECTED_GRAD_FN[path],
                f"custom autograd missing: {name}: {grad_fn}; expected {EXPECTED_GRAD_FN[path]}",
            )


def _check_training_gradients(model: Any) -> None:
    registered = dict(model.named_parameters())
    projections = {projection for names in PROJECTIONS.values() for projection in names}
    counts = dict.fromkeys(projections, 0)
    required = set()
    for prefix, module in model.named_modules():
        projection = prefix.split(".")[-1]
        if projection not in projections:
            continue
        counts[projection] += 1
        for matrix in ("A", "B"):
            name = f"{prefix}.lora_{matrix}.default.weight"
            required.add(name)
            parameter = getattr(module, f"lora_{matrix}")["default"].weight
            _require(registered.get(name) is parameter, f"missing/unregistered adapter: {name}")
    layers = model.config.num_hidden_layers
    _require(all(count == layers for count in counts.values()), f"missing projections: {counts}")
    found = {name for name in registered if "lora_" in name}
    _require(found == required, f"training adapter keys mismatch: {sorted(found ^ required)}")
    for name, parameter in registered.items():
        if "lora_" in name:
            _require(parameter.requires_grad, f"adapter not trainable: {name}")
            require_finite(parameter.grad, name)
        else:
            _require(
                not parameter.requires_grad and parameter.grad is None, f"base not frozen: {name}"
            )


@_retain_evidence("loss")
def run_loss(
    *,
    seed: int = 792,
    steps: int = 50,
    device: str = "cpu",
    dtype: str = "fp32",
    _evidence: dict[str, Any],
) -> dict[str, Any]:
    """Train the same random tiny Llama on fixed synthetic tokens, twice."""
    import torch

    if steps < 1:
        raise ValueError("steps must be >= 1")
    validate_runtime(device, dtype)
    rows = _evidence["rows"]
    with _deterministic(seed, device):
        baseline = _loss_model(device, dtype)
        fast = copy.deepcopy(baseline)
        # Group patchers MUST precede singles; otherwise QKV silently declines.
        counts = {path: _patch(fast, path) for path in ("mlp", "qkv", "single")}
        frozen = {
            name: parameter.detach().clone()
            for name, parameter in baseline.named_parameters()
            if not parameter.requires_grad
        }
        tokens = torch.randint(0, 64, (2, 12), device=device)
        observed, handles = _execution_hooks(fast)
        control_observed, control_handles = _execution_hooks(baseline)
        arithmetic_modes, arithmetic_handles = _reference_arithmetic_hooks(fast)
        arithmetic_metadata = {
            "ctx_attribute": "grad_fn.reference_order",
            "required_true": dtype in {"fp16", "bf16"},
            "expected_modules": {path: list(values) for path, values in arithmetic_modes.items()},
            "scope": "declared canonical HF tiny-Llama loss fixture, dense bases, fp32 adapters",
            "limits": [
                "per-forward selected arithmetic only; Function names alone do not establish mode",
                "after Function-name gates: True is scoped arithmetic, False is legacy fusion",
                "fp32 disables the scoped path by dtype; True is not required",
                "finite SYNTHETIC fixture, not global bit-exactness or a performance claim",
                "CPU evidence does not validate CUDA, NF4, other graphs, shapes or schedulers",
            ],
        }
        _evidence["reference_arithmetic"] = arithmetic_metadata
        optimizers = [
            torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=0.003,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
                foreach=False,
            )
            for model in (baseline, fast)
        ]
        try:
            baseline(input_ids=tokens, labels=tokens)
            try:
                _verify_execution(counts, control_observed)
            except AssertionError as exc:
                control = {"unpatched_rejected": True, "reason": str(exc)}
            else:
                raise AssertionError("negative control failed: unpatched PEFT was accepted")
            for step in range(steps):
                losses = []
                pending_step: dict[str, Any] = {"step": step + 1}
                _evidence["pending_step"] = pending_step
                for model, optimizer in zip((baseline, fast), optimizers):
                    _evidence["failure_case"] = {
                        "stage": "training",
                        "step": step + 1,
                        "implementation": "fast" if model is fast else "PEFT",
                        "phase": "zero_grad",
                    }
                    if model is fast:
                        # Reset on arm entry, even if zero_grad interrupts before forward.
                        for path_observed in observed.values():
                            for name in path_observed:
                                path_observed[name] = "not executed"
                        for path_modes in arithmetic_modes.values():
                            for name in path_modes:
                                path_modes[name] = "not executed"
                        pending_step["reference_arithmetic_status"] = "INSUFFICIENT"
                        pending_step["reference_arithmetic_modes"] = copy.deepcopy(arithmetic_modes)
                    optimizer.zero_grad(set_to_none=True)
                    _evidence["failure_case"]["phase"] = "forward"
                    try:
                        loss = model(input_ids=tokens, labels=tokens).loss
                    finally:
                        if model is fast:
                            # Preserve actually observed partial forwards even when they raise.
                            pending_step["reference_arithmetic_modes"] = copy.deepcopy(
                                arithmetic_modes
                            )
                    raw_loss = loss.detach().item()
                    loss_key = "fast_loss" if model is fast else "baseline_loss"
                    # Record the measured loss before any validity/backward/update failure.
                    # Nonfinite values use explicit strings for strict JSON, not invented numbers.
                    pending_step[loss_key] = (
                        raw_loss if math.isfinite(raw_loss) else str(raw_loss)
                    )
                    _evidence["failure_case"]["phase"] = "loss_check"
                    require_finite(loss, "loss")
                    if model is fast:
                        _evidence["failure_case"]["phase"] = "execution_check"
                        _verify_execution(counts, observed)
                        _evidence["failure_case"]["phase"] = "reference_arithmetic_check"
                        _verify_reference_arithmetic(
                            arithmetic_modes, arithmetic_metadata["expected_modules"],
                            required_true=arithmetic_metadata["required_true"],
                        )
                        pending_step["reference_arithmetic_status"] = "observed"
                    _evidence["failure_case"]["phase"] = "backward"
                    loss.backward()
                    _evidence["failure_case"]["phase"] = "gradient_check"
                    _check_training_gradients(model)
                    _evidence["failure_case"]["phase"] = "optimizer_step"
                    optimizer.step()
                    losses.append(raw_loss)
                rounded_equal = f"{losses[0]:.3f}" == f"{losses[1]:.3f}"
                rows.append(
                    {
                        **pending_step,
                        "step": step + 1,
                        "baseline_loss": losses[0],
                        "fast_loss": losses[1],
                        "abs_error": abs(losses[0] - losses[1]),
                        "baseline_rounded_3": f"{losses[0]:.3f}",
                        "fast_rounded_3": f"{losses[1]:.3f}",
                        "rounded_3_equal": rounded_equal,
                    }
                )
                _evidence.pop("pending_step")
            for model in (baseline, fast):
                _evidence["failure_case"] = {"stage": "final_parameters"}
                for name, parameter in model.named_parameters():
                    require_finite(parameter, name)
                    if name in frozen:
                        _require(torch.equal(parameter, frozen[name]), f"base changed: {name}")
        finally:
            for handle in handles + control_handles + arithmetic_handles:
                handle.remove()
        mismatches = [row["step"] for row in rows if not row["rounded_3_equal"]]
        return {
            "mode": "loss",
            "fixture": "SYNTHETIC",
            "reference": "unpatched PEFT",
            "seed": seed,
            "steps": steps,
            "device": device,
            "dtype": dtype,
            "model": baseline.config.to_dict(),
            "adapter_dtype": "fp32",
            "tokens": tokens.cpu().tolist(),
            "target_modules": [name for names in PROJECTIONS.values() for name in names],
            "optimizer": {
                "name": "AdamW",
                "lr": 0.003,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "foreach": False,
            },
            "dropout": 0.0,
            "base_frozen": True,
            "base_unchanged": True,
            "criterion": "every per-step loss rounds identically to 3 decimals",
            "patch_counts": counts,
            "execution": observed,
            "reference_arithmetic": arithmetic_metadata,
            "negative_control": control,
            "rows": rows,
            "passed": not mismatches,
            "failing_steps": mismatches,
            **(
                {"error": f"loss mismatch at 3 decimals; failing steps: {mismatches}"}
                if mismatches
                else {}
            ),
        }


def _measure(action: Any, device: str) -> tuple[Any, float]:
    """Measure one phase; CUDA events exclude unrelated host bookkeeping."""
    import torch

    if torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            result = action()
            end.record()
            end.synchronize()
            return result, start.elapsed_time(end)
    start_time = time.perf_counter()
    result = action()
    return result, (time.perf_counter() - start_time) * 1000.0


def collect_cuda_validity(target_uuid: str | None = None) -> dict[str, Any]:
    """Best-effort raw driver/SM-clock/power/peer samples, never automatic approval."""
    queries = {
        "gpu_samples": "--query-gpu=index,uuid,driver_version,clocks.current.sm,pstate,power.draw",
        "process_samples": "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    }
    result: dict[str, Any] = {
        "status": "UNVERIFIED",
        "available": False,
        "utc": datetime.now(timezone.utc).isoformat(),
        "own_pid": os.getpid(),
        "reason": "clock/power/peer validity requires review; NO VERDICT",
    }
    try:
        for name, query in queries.items():
            process = subprocess.run(
                ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if process.returncode:
                result["reason"] = f"nvidia-smi failed: {process.stderr.strip()}"
                return result
            result[name] = list(csv.reader(process.stdout.splitlines(), skipinitialspace=True))
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["reason"] = f"nvidia-smi unavailable: {exc}"
        return result
    result["available"] = True
    normalized_uuid = (target_uuid or "").lower().removeprefix("gpu-")
    peers = [
        row
        for row in result["process_samples"]
        if len(row) >= 2
        and normalized_uuid
        and row[0].lower().removeprefix("gpu-") == normalized_uuid
        and row[1] != str(os.getpid())
    ]
    result["target_uuid"] = target_uuid
    result["peers"] = peers
    if peers:
        result["status"] = "VOID"
        result["reason"] = (
            "other compute processes on this GPU; retain raw arm but exclude headline"
        )
    return result


@_retain_evidence("timing")
def run_timing(
    *,
    seed: int = 792,
    device: str = "cpu",
    dtype: str = "fp32",
    shapes: str = "tiny",
    tokens: int = 16,
    warmup: int = 5,
    repeats: int = 20,
    allow_large_cpu: bool = False,
    _evidence: dict[str, Any],
) -> dict[str, Any]:
    """Single/GQA/SwiGLU forward and backward timings, with/without dX requested."""
    import torch

    if repeats < 1 or warmup < 0 or tokens < 1:
        raise ValueError("repeats and tokens must be >= 1; warmup must be >= 0")
    is_cuda = torch.device(device).type == "cuda"
    if shapes != "tiny" and not is_cuda and not allow_large_cpu:
        raise ValueError("large CPU shapes require allow_large_cpu=True")
    validate_runtime(device, dtype)
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    rows = _evidence["rows"]
    _evidence["timing_verdict"] = "NO VERDICT (incomplete/failed correctness or arm validity)"
    with _deterministic(seed, device):
        correctness: dict[str, Any] = {"passed": False, "cases": []}
        _evidence["correctness_gate"] = correctness
        for path in PROJECTIONS:
            baseline = _make_fixture(path, torch_dtype, device, shapes)
            fast = copy.deepcopy(baseline)
            _require(_patch(fast, path) > 0, f"no patch applied: {path}")
            for request_dx in (True, False):
                source = torch.randn(
                    1, tokens, SHAPES[shapes]["hidden"], dtype=torch_dtype, device=device
                )
                models = {"PEFT": baseline, "fast": fast}
                inputs = {
                    name: source.detach().clone().requires_grad_(request_dx) for name in models
                }
                probe_outputs = _outputs(fast, inputs["fast"])
                _require(
                    all(
                        (
                            type(out.grad_fn).__name__ == EXPECTED_GRAD_FN[path]
                            for out in probe_outputs
                        )
                    ),
                    "custom autograd missing",
                )
                upstream = tuple(torch.randn_like(out) for out in probe_outputs)
                del probe_outputs
                _evidence["failure_case"] = {
                    "stage": "correctness_gate",
                    "path": path,
                    "request_dX": request_dx,
                    "input_shape": list(source.shape),
                    "dtype": dtype,
                }
                # These are the actual timed models, rounded weights, X and upstream.
                # No separately seeded or differently shaped correctness fixture is used.
                oracle_model = copy.deepcopy(baseline).double()
                expected = _snapshot(baseline, inputs["PEFT"], upstream, path)
                actual = _snapshot(fast, inputs["fast"], upstream, path)
                oracle = _snapshot(
                    oracle_model,
                    source.double().requires_grad_(request_dx),
                    tuple(gradient.double() for gradient in upstream),
                    path,
                )
                _validate_snapshot_keys(path, request_dx, fast=actual, PEFT=expected, oracle=oracle)
                case: dict[str, Any] = {
                    "path": path,
                    "request_dX": request_dx,
                    "input_shape": list(source.shape),
                    "dtype": dtype,
                    "passed": False,
                    "rows": [],
                }
                correctness["cases"].append(case)
                for quantity in expected:
                    _evidence["failure_case"]["quantity"] = quantity
                    case["rows"].append(
                        {
                            "quantity": quantity,
                            **_comparison(
                                actual[quantity], expected[quantity], oracle[quantity], dtype
                            ),
                        }
                    )
                case["negative_control"] = _changed_adapter_control(
                    baseline,
                    inputs["PEFT"],
                    expected,
                    oracle,
                    dtype,
                )
                case["passed"] = True
                del oracle_model, actual
                arms = (("A", "PEFT"), ("B", "fast"), ("B", "fast"), ("A", "PEFT"))
                for iteration in range(warmup + repeats):
                    for arm_index, (arm, implementation) in enumerate(arms, start=1):
                        _evidence["failure_case"] = {
                            "stage": "timing",
                            "path": path,
                            "request_dX": request_dx,
                            "iteration": iteration,
                            "warmup": iteration < warmup,
                            "arm": arm,
                            "arm_index": arm_index,
                            "implementation": implementation,
                        }
                        model, x = models[implementation], inputs[implementation]
                        model.zero_grad(set_to_none=True)
                        x.grad = None
                        validity: dict[str, Any] = {"status": "CPU DEBUG-ONLY", "available": False}
                        if is_cuda:
                            torch.cuda.synchronize(device)
                            torch.cuda.reset_peak_memory_stats(device)
                            if iteration >= warmup:
                                target_uuid = getattr(
                                    torch.cuda.get_device_properties(device), "uuid", None
                                )
                                validity = collect_cuda_validity(
                                    str(target_uuid) if target_uuid else None
                                )
                        _evidence["pending_arm"] = {
                            **_evidence["failure_case"],
                            "dtype": dtype,
                            "forward_ms": None,
                            "backward_ms": None,
                            "validity": validity,
                            "correctness_passed": False,
                        }
                        outputs, forward_ms = _measure(lambda: _outputs(model, x), device)
                        _evidence["pending_arm"]["forward_ms"] = forward_ms
                        _, backward_ms = _measure(
                            lambda tensors=outputs, gradients=upstream: torch.autograd.backward(
                                tensors, gradients
                            ),
                            device,
                        )
                        _evidence.pop("pending_arm")
                        row: dict[str, Any] = {
                            "path": path,
                            "implementation": implementation,
                            "request_dX": request_dx,
                            "round": iteration - warmup + 1,
                            "repeat": iteration - warmup + 1,
                            "warmup": iteration < warmup,
                            "arm": arm,
                            "arm_index": arm_index,
                            "dtype": dtype,
                            "forward_ms": forward_ms,
                            "backward_ms": backward_ms,
                            "peak_allocated_bytes": (
                                torch.cuda.max_memory_allocated(device) if is_cuda else None
                            ),
                            "peak_reserved_bytes": (
                                torch.cuda.max_memory_reserved(device) if is_cuda else None
                            ),
                            "validity": validity,
                            "correctness_passed": False,
                            "checks": [],
                        }
                        if iteration >= warmup:
                            rows.append(row)
                        # All checks and CPU transfers are AFTER both measured event regions.
                        try:
                            if implementation == "fast":
                                _require(
                                    all(
                                        type(out.grad_fn).__name__ == EXPECTED_GRAD_FN[path]
                                        for out in outputs
                                    ),
                                    "custom autograd missing in timed arm",
                                )
                            measured = _collect_snapshot(model, x, outputs, path)
                            _validate_snapshot_keys(
                                path, request_dx, arm=measured, PEFT=expected, oracle=oracle
                            )
                            for quantity in expected:
                                _evidence["failure_case"]["quantity"] = quantity
                                row["checks"].append(
                                    {
                                        "quantity": quantity,
                                        **_comparison(
                                            measured[quantity],
                                            expected[quantity],
                                            oracle[quantity],
                                            dtype,
                                        ),
                                    }
                                )
                            row["correctness_passed"] = True
                            del measured
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            row["traceback"] = traceback.format_exc()
                            row["validity"] = {
                                **validity,
                                "status": "VOID",
                                "reason": "arm correctness failed",
                            }
                            if iteration < warmup:
                                rows.append(row)  # Bad warmups are evidence too.
                            raise
                        del outputs
            del baseline, fast
        correctness["passed"] = True
    return {
        "mode": "timing",
        "fixture": "SYNTHETIC",
        "reference": "unpatched PEFT",
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "shapes": shapes,
        "shape_dimensions": SHAPES[shapes],
        "tokens": tokens,
        "batch": 1,
        "warmup": warmup,
        "repeats": repeats,
        "round_order": "ABBA",
        "arm_mapping": {"A": "unpatched PEFT", "B": "fast"},
        "repeat_unit": "ABBA round; two samples per implementation per round",
        "correctness_gate": correctness,
        "timing_verdict": (
            "NO VERDICT (CUDA arm validity requires review)"
            if is_cuda
            else "NO VERDICT (CPU DEBUG-ONLY)"
        ),
        "memory_scope": (
            "absolute peak allocation/reservation, includes both models, inputs "
            "and cached correctness snapshots"
        ),
        "process_isolation": "same-process ABBA; not fresh-process arms",
        "base_frozen": True,
        "dropout": 0.0,
        "timer": "CUDA events" if is_cuda else "perf_counter",
        "measurement_scope": (
            "CUDA single-layer ONLY; not a full-run multiplier"
            if is_cuda
            else "CPU DEBUG-ONLY; no CUDA or performance multiplier claim"
        ),
        "timed_region": "forward/backward separately; excludes zero_grad, optimizer, validation",
        "rows": rows,
        "passed": True,
    }


def _environment(device: str) -> dict[str, Any]:
    """Collect what is available before runtime refusals, even without torch/CUDA."""
    root = Path(__file__).parents[2]
    errors: dict[str, str] = {}

    def available(label: str, action: Any) -> Any:
        try:
            return action()
        except Exception as exc:
            errors[label] = f"{type(exc).__name__}: {exc}"
            return None

    def digest(path: Path) -> Any:
        return available(str(path), lambda: hashlib.sha256(path.read_bytes()).hexdigest())

    def git(*args: str) -> Any:
        def query() -> str:
            process = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if process.returncode:
                raise RuntimeError(process.stderr.strip())
            return process.stdout.strip()

        return available(f"git {' '.join(args)}", query)

    packages = {}
    for package in ("torch", "peft", "transformers", "bitsandbytes"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
        except Exception as exc:
            packages[package] = f"unavailable: {exc}"

    result: dict[str, Any] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "python_optimization": sys.flags.optimize,
        "platform": platform.platform(),
        "packages": packages,
        "harness_file": os.path.realpath(__file__),
        "harness_sha256": digest(Path(__file__)),
        "decision_rule_sha256": digest(root / "benchmarks" / "gate-d2-fast-lora-rule.md"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "kernel_sha256": {
            filename: digest(root / "src" / "soup_cli" / "utils" / filename)
            for filename in ("fast_lora.py", "fast_lora_qkv.py", "fast_lora_mlp.py")
        },
        "cpu_threads_during_probe": 1,
        "collection_errors": errors,
        "soup_cli_file": None,
        "checkout_matches": False,
        "torch_cuda_build": None,
        "cuda_available": None,
    }
    try:
        import soup_cli

        result["soup_cli_file"] = os.path.realpath(soup_cli.__file__)
        expected = os.path.realpath(root / "src" / "soup_cli" / "__init__.py")
        result["checkout_matches"] = os.path.normcase(result["soup_cli_file"]) == os.path.normcase(
            expected
        )
    except Exception as exc:
        errors["soup_cli"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch

        result["torch_cuda_build"] = torch.version.cuda
        result["cuda_available"] = available("cuda_available", torch.cuda.is_available)
        if device.startswith("cuda") and result["cuda_available"]:
            result["cuda_device"] = available(
                "cuda_device", lambda: torch.cuda.get_device_name(device)
            )
            result["cuda_capability"] = available(
                "cuda_capability",
                lambda: list(torch.cuda.get_device_capability(device)),
            )
            result["cuda_total_memory_bytes"] = available(
                "cuda_memory",
                lambda: torch.cuda.get_device_properties(device).total_memory,
            )
    except Exception as exc:
        errors["torch"] = f"{type(exc).__name__}: {exc}"
    return result


def _validate_environment(environment: dict[str, Any]) -> None:
    if not environment["checkout_matches"]:
        root = Path(__file__).parents[2]
        raise ValueError(
            f"wrong/unavailable soup_cli checkout: {environment['soup_cli_file']}; "
            f"set PYTHONPATH={root / 'src'}"
        )
    hashes = [
        environment["harness_sha256"],
        environment["decision_rule_sha256"],
        *environment["kernel_sha256"].values(),
    ]
    _require(all(hashes), f"source provenance unavailable: {environment['collection_errors']}")


def _write_evidence(report: dict[str, Any], prefix: str) -> tuple[Path, Path]:
    """JSON preserves provenance; CSV preserves every raw per-tensor/step/timing row."""
    json_path, csv_path = Path(f"{prefix}.json"), Path(f"{prefix}.csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metadata = {key: report.get(key) for key in ("mode", "fixture", "device", "dtype", "seed")}
    raw_rows = report["rows"]
    if not raw_rows and not report["passed"]:
        raw_rows = [
            {
                "row_type": "failure",
                "passed": False,
                "error": report["error"],
                "failure_case": report.get("failure_case"),
                "traceback": report.get("traceback"),
            }
        ]
    rows = [
        {
            **metadata,
            **{
                key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            },
        }
        for row in raw_rows
    ]
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(metadata)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _timing_summary_status(row: dict[str, Any], device: str) -> str:
    """Fail closed for absent/malformed validity without rewriting raw samples."""
    validity = row.get("validity")
    if not isinstance(validity, dict):
        return "INSUFFICIENT"
    status = validity.get("status")
    if status == "VOID":
        return "VOID"
    if device == "cpu" and status == "CPU DEBUG-ONLY" and validity.get("available") is False:
        return "CPU DEBUG-ONLY"
    if (
        device.startswith("cuda")
        and status == "UNVERIFIED"
        and validity.get("available") is True
        and isinstance(validity.get("gpu_samples"), list)
        and bool(validity["gpu_samples"])
        and isinstance(validity.get("process_samples"), list)
    ):
        return "UNVERIFIED"
    return "INSUFFICIENT"


def _render(report: dict[str, Any], console: Console) -> None:
    console.print(f"D2 {report['mode']} | SYNTHETIC | {report['device']} | {report['dtype']}")
    if report["mode"] == "loss":
        console.print("Loss mode evidence: per-step grad_fn.reference_order (QKV/MLP)")
        console.print("Required: literal True for fp16/bf16; fp32 does not require True")
        console.print("Scope: finite SYNTHETIC fixture; not global bit-exactness")
        if report["device"] == "cpu":
            console.print("CPU DEBUG-ONLY; CUDA/NF4 and other graphs remain unverified")
    if not report["passed"]:
        console.print(report["error"], markup=False)
        return
    if report["mode"] == "parity":
        console.print(report["criterion"], markup=False)
        for phase in ("forward", "backward"):
            table = Table(title=f"{phase.title()} parity", box=box.ASCII)
            for column in ("Path", "Quantity", "Bit exact", "Max abs PEFT error", "PASS"):
                table.add_column(column)
            for row in report["rows"]:
                if row["phase"] == phase:
                    table.add_row(
                        row["path"],
                        row["quantity"],
                        str(row["bit_exact"]),
                        f"{row['max_abs_error']:.3e}",
                        str(row["passed"]),
                    )
            console.print(table)
        console.print(f"Gradcheck: {report['gradcheck'] or 'not requested'}", markup=False)
    elif report["mode"] == "loss":
        table = Table(title="Per-step SYNTHETIC tiny-Llama loss", box=box.ASCII)
        for column in ("Step", "PEFT", "Fast", "Abs error", "Equal at 3 decimals"):
            table.add_column(column)
        for row in report["rows"]:
            table.add_row(
                str(row["step"]),
                f"{row['baseline_loss']:.6f}",
                f"{row['fast_loss']:.6f}",
                f"{row['abs_error']:.3e}",
                str(row["rounded_3_equal"]),
            )
        console.print(table)
        console.print(f"Patches: {report['patch_counts']}; unpatched negative control rejected")
    else:
        statuses = [_timing_summary_status(row, report["device"]) for row in report["rows"]]
        cpu = report["device"] == "cpu"
        insufficient_count = statuses.count("INSUFFICIENT")
        report["timing_verdict"] = (
            "NO VERDICT (CPU DEBUG-ONLY)"
            if cpu
            else "NO VERDICT (CUDA arm validity requires review)"
        )
        if insufficient_count:
            scope = "CPU DEBUG-ONLY; " if cpu else ""
            report["timing_verdict"] = f"NO VERDICT ({scope}insufficient arm validity evidence)"
        console.print(report["measurement_scope"], markup=False)
        console.print(report["timing_verdict"], markup=False)
        console.print(
            f"VOID arms excluded from medians: {statuses.count('VOID')}; "
            f"insufficient validity arms excluded: {insufficient_count}; raw samples retained"
        )
        table = Table(
            title="Single-layer debug/unverified medians (raw repeats in CSV)", box=box.ASCII
        )
        for column in ("Path", "dX requested", "Implementation", "Forward ms", "Backward ms"):
            table.add_column(column)
        for path in PROJECTIONS:
            for dx in (True, False):
                for implementation in ("PEFT", "fast"):
                    rows = [
                        row
                        for row, status in zip(report["rows"], statuses)
                        if row.get("path") == path
                        and row.get("request_dX") == dx
                        and row.get("implementation") == implementation
                        and status not in ("VOID", "INSUFFICIENT")
                    ]
                    table.add_row(
                        path,
                        str(dx),
                        implementation,
                        f"{statistics.median(row['forward_ms'] for row in rows):.4f}"
                        if rows
                        else "n/a",
                        f"{statistics.median(row['backward_ms'] for row in rows):.4f}"
                        if rows
                        else "n/a",
                    )
        console.print(table)
    console.print("PASS; no full-run, Liger, Unsloth or upstream multiplier claim")


def main(argv: list[str] | None = None) -> int:
    """CLI: parity (default gradcheck), 50-step loss, or bounded single-layer timing."""
    console = Console()

    class RichParser(argparse.ArgumentParser):
        def _print_message(self, message: str, file: Any = None) -> None:
            Console(file=file).print(message, end="", markup=False, highlight=False)

    parser = RichParser(description=__doc__)
    parser.add_argument("mode", choices=("parity", "loss", "timing"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument(
        "--dtype", choices=("fp32", "fp16", "bf16"), help="default: fp32 CPU, fp16 CUDA (T4 first)"
    )
    parser.add_argument("--seed", type=int, default=792)
    parser.add_argument("--output-prefix", required=True, help="writes PREFIX.json and PREFIX.csv")
    parser.add_argument("--quantization", choices=("dense", "nf4"), default="dense")
    parser.add_argument("--skip-gradcheck", action="store_true")
    parser.add_argument("--no-dx", action="store_true", help="parity without requesting dX")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shapes", choices=tuple(SHAPES), default="tiny")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--allow-large-cpu", action="store_true")
    args = parser.parse_args(argv)
    dtype = args.dtype or ("fp16" if args.device.startswith("cuda") else "fp32")
    report: dict[str, Any] = {
        "schema_version": 2,
        "mode": args.mode,
        "fixture": "SYNTHETIC",
        "device": args.device,
        "dtype": dtype,
        "seed": args.seed,
        "arguments": vars(args),
        "rows": [],
        "passed": False,
        "nf4_status": "not implemented/unverified by this harness",
        "limitations": [
            "random weights/tokens, not a pretrained checkpoint or real training dataset",
            "no full-run, Liger, Unsloth or upstream multiplier evidence",
            "CUDA routes require an actual compatible GPU; CPU timings are CPU ONLY",
        ],
    }
    if args.mode == "timing":
        report["timing_verdict"] = "NO VERDICT (run incomplete)"
    try:
        report["environment"] = _environment(args.device)
        _validate_environment(report["environment"])
        if args.quantization == "nf4":
            raise NotImplementedError(report["nf4_status"])
        validate_runtime(args.device, dtype)
        common = {"seed": args.seed, "device": args.device, "dtype": dtype}
        if args.mode == "parity":
            result = run_parity(
                **common,
                gradcheck=not args.skip_gradcheck,
                request_dx=not args.no_dx,
                shapes=args.shapes,
                allow_large_cpu=args.allow_large_cpu,
            )
        elif args.mode == "loss":
            result = run_loss(**common, steps=args.steps)
        else:
            result = run_timing(
                **common,
                shapes=args.shapes,
                tokens=args.tokens,
                warmup=args.warmup,
                repeats=args.repeats,
                allow_large_cpu=args.allow_large_cpu,
            )
        report.update(result)
    except Exception as exc:
        report.update(getattr(exc, "evidence_report", {}))
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report.setdefault("failure_case", {"stage": "runtime"})
        report["passed"] = False
    render_error = None
    try:
        _render(report, console)
    except Exception as exc:
        render_error = f"{type(exc).__name__}: {exc}"
        render_traceback = traceback.format_exc()
        failure_case = {"stage": "render"}
        report["rows"].append(
            {
                "row_type": "failure",
                "passed": False,
                "failure_case": failure_case,
                "error": render_error,
                "traceback": render_traceback,
            }
        )
        # A renderer fault must not replace an earlier probe failure or its evidence.
        report.setdefault("error", render_error)
        report.setdefault("traceback", render_traceback)
        report.setdefault("failure_case", failure_case)
        report["passed"] = False
        if args.mode == "timing":
            report["timing_verdict"] = "NO VERDICT (render failed; raw evidence retained)"
    paths = _write_evidence(report, args.output_prefix)
    if render_error is not None:
        # Persist first: the console itself may be the reason rendering failed.
        try:
            console.print(render_error, markup=False)
            console.print(f"Evidence: {paths[0]} | {paths[1]}", markup=False)
        except Exception:
            pass
        return 2
    console.print(f"Evidence: {paths[0]} | {paths[1]}", markup=False)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
