"""Synthetic NF4 parity with identical packed bases and replayable tensor evidence.

This is a correctness probe, not a speed or real-checkpoint quality claim.
Run from the repository root with PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

from rich.console import Console

try:
    from . import fast_lora_probe as probe
except ImportError:
    import fast_lora_probe as probe


def _digest(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _bases(model: Any, path: str) -> dict[str, Any]:
    return {
        name: model.get_submodule(name if path == "single" else f"group.{name}").base_layer
        for name in probe.PROJECTIONS[path]
    }


def make_fixture(path: str, dtype: Any, device: str, shapes: str = "tiny") -> Any:
    """Quantize once before copying the reference, retaining fp32 adapters."""
    import bitsandbytes as bnb
    import torch

    model = probe._make_fixture(path, torch.float32, "cpu", shapes)
    for name in probe.PROJECTIONS[path]:
        projection = model.get_submodule(name if path == "single" else f"group.{name}")
        dense = projection.base_layer
        base = bnb.nn.Linear4bit(
            dense.in_features,
            dense.out_features,
            bias=False,
            compute_dtype=dtype,
            quant_type="nf4",
            compress_statistics=True,
        )
        base.weight = bnb.nn.Params4bit(
            dense.weight.detach().clone(),
            requires_grad=False,
            quant_type="nf4",
            compress_statistics=True,
        )
        projection.base_layer = base.to(device)
        projection.lora_A.to(device=device, dtype=torch.float32)
        projection.lora_B.to(device=device, dtype=torch.float32)
    return model


def make_oracle(model: Any, path: str) -> Any:
    """Float64 graph over the same dequantized NF4 weights, no re-quantization."""
    import bitsandbytes as bnb
    import torch

    oracle = copy.deepcopy(model)
    for name, base in _bases(oracle, path).items():
        weight = bnb.functional.dequantize_4bit(base.weight, base.weight.quant_state)
        dense = torch.nn.Linear(
            base.in_features,
            base.out_features,
            bias=False,
            dtype=torch.float64,
            device=weight.device,
        )
        dense.weight = torch.nn.Parameter(weight.double(), requires_grad=False)
        projection = oracle.get_submodule(name if path == "single" else f"group.{name}")
        projection.base_layer = dense
    return oracle.double()


def _quant_evidence(model: Any, path: str) -> dict[str, Any]:
    import torch

    def state_digest(state: Any) -> Any:
        # QuantState.as_dict(packed=True) covers nested state, dtype, shape and codes.
        return {
            key: _digest(value) if isinstance(value, torch.Tensor) else str(value)
            for key, value in state.as_dict(packed=True).items()
        }

    return {
        name: {
            "packed_sha256": _digest(base.weight),
            "quant_state": state_digest(base.weight.quant_state),
        }
        for name, base in _bases(model, path).items()
    }


def _snapshot(model: Any, inputs: Any, upstream: Any, path: str, checkpoint: bool) -> Any:
    import torch
    from torch.utils.checkpoint import checkpoint as recompute

    model.zero_grad(set_to_none=True)
    inputs.grad = None
    result = recompute(model, inputs, use_reentrant=False) if checkpoint else model(inputs)
    outputs = result if isinstance(result, tuple) else (result,)
    torch.autograd.backward(outputs, upstream)
    return probe._collect_snapshot(model, inputs, outputs, path)


def run_case(
    path: str,
    dtype: str,
    device: str,
    request_dx: bool,
    checkpoint: bool,
    destination: Path,
    shapes: str = "tiny",
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    baseline = make_fixture(path, torch_dtype, device, shapes)
    fast = copy.deepcopy(baseline)
    original_quant = _quant_evidence(baseline, path)
    probe._require(original_quant == _quant_evidence(fast, path), "different packed bases")
    oracle_model = make_oracle(baseline, path)
    patch_count = probe._patch(fast, path)
    probe._require(patch_count > 0, "no patch applied")
    source = torch.randn(2, 3, probe.SHAPES[shapes]["hidden"], dtype=torch_dtype, device=device)
    inputs = source.clone().requires_grad_(request_dx)
    outputs = probe._outputs(fast, inputs)
    execution = [type(value.grad_fn).__name__ for value in outputs]
    probe._require(
        all(name == probe.EXPECTED_GRAD_FN[path] for name in execution), "silent fallback"
    )
    upstream = tuple(torch.randn_like(value) for value in outputs)
    del outputs
    snapshots = {
        "fast": _snapshot(fast, inputs, upstream, path, checkpoint),
        "PEFT": _snapshot(
            baseline, source.clone().requires_grad_(request_dx), upstream, path, checkpoint
        ),
        "float64": _snapshot(
            oracle_model,
            source.double().requires_grad_(request_dx),
            tuple(value.double() for value in upstream),
            path,
            False,
        ),
    }
    probe._validate_snapshot_keys(path, request_dx, **snapshots)
    probe._require(original_quant == _quant_evidence(fast, path), "fast base mutated")
    probe._require(original_quant == _quant_evidence(baseline, path), "reference base mutated")
    artifact = destination / f"{path}-dx{int(request_dx)}-checkpoint{int(checkpoint)}.safetensors"
    save_file(
        {
            **{
                f"{arm}/{key}": value.detach().cpu().contiguous()
                for arm, snapshot in snapshots.items()
                for key, value in snapshot.items()
            },
            "inputs": source.cpu().contiguous(),
            **{
                f"upstream/{index}": value.cpu().contiguous()
                for index, value in enumerate(upstream)
            },
        },
        str(artifact),
    )
    row = {
        "path": path,
        "request_dX": request_dx,
        "checkpoint": checkpoint,
        "patch_count": patch_count,
        "execution": execution,
        "quantization": original_quant,
        "tensor_file": artifact.name,
        "tensor_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "comparisons": [],
        "passed": False,
    }
    try:
        for key, actual in snapshots["fast"].items():
            row["comparisons"].append(
                {
                    "quantity": key,
                    "phase": "forward" if key.startswith("Y/") else "backward",
                    **probe._comparison(
                        actual, snapshots["PEFT"][key], snapshots["float64"][key], dtype
                    ),
                }
            )
        row["negative_control"] = probe._changed_adapter_control(
            baseline, source, snapshots["PEFT"], snapshots["float64"], dtype
        )
        row["passed"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shapes", choices=("tiny", "nf4-aligned"), default="tiny")
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "fixture": "SYNTHETIC",
        "dtype": args.dtype,
        "device": args.device,
        "seed": 792,
        "shapes": args.shapes,
        "shape_dimensions": probe.SHAPES[args.shapes],
        "quantization": "NF4 double-quantized",
        "adapter_dtype": "fp32",
        "rows": [],
        "passed": False,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "streaming": "UNVERIFIED by this projection probe",
    }
    try:
        report["environment"] = probe._environment(args.device)
        probe._validate_environment(report["environment"])
        probe.validate_runtime(args.device, args.dtype)
        with probe._deterministic(792, args.device):
            for path in probe.PROJECTIONS:
                for request_dx in (True, False):
                    for checkpoint in (False, True):
                        try:
                            row = run_case(
                                path,
                                args.dtype,
                                args.device,
                                request_dx,
                                checkpoint,
                                destination,
                                args.shapes,
                            )
                        except Exception as exc:
                            row = {
                                "path": path,
                                "request_dX": request_dx,
                                "checkpoint": checkpoint,
                                "passed": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                            }
                        report["rows"].append(row)
        report["passed"] = len(report["rows"]) == 12 and all(
            row["passed"] for row in report["rows"]
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    Console().print(f"NF4 {'PASS' if report['passed'] else 'FAIL'}: {destination}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
