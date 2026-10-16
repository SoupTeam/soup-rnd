"""Independently replay the committed D2 criterion from saved safetensors."""

from __future__ import annotations

import argparse
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


def _expected_keys(path: Path) -> set[str]:
    if path.name == "streaming.safetensors":
        return {"Y/logits", "Y/loss"} | {
            f"gradient/base_model.model.model.layers.{layer}.{group}.{projection}"
            f".lora_{matrix}.default.weight"
            for layer in range(4)
            for group, projections in (
                ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                ("mlp", ("gate_proj", "up_proj", "down_proj")),
            )
            for projection in projections for matrix in ("A", "B")
        }
    parts = path.stem.split("-")
    if len(parts) < 2 or parts[0] not in probe.PROJECTIONS or parts[1] not in ("dx0", "dx1"):
        raise ValueError("unrecognized fixture; expected quantities must be supplied explicitly")
    return probe._expected_quantities(parts[0], parts[1] == "dx1")


def replay(path: Path, dtype: str, *, expected_keys: set[str] | None = None) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    arms = {
        arm: {
            key.removeprefix(arm + "/"): value
            for key, value in tensors.items()
            if key.startswith(arm + "/")
        }
        for arm in ("fast", "PEFT", "float64")
    }
    report: dict[str, Any] = {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dtype": dtype,
        "rows": [],
        "passed": False,
    }
    required = _expected_keys(path) if expected_keys is None else expected_keys
    probe._require(required, "no expected quantities")
    probe._require(
        all(set(arm) == required for arm in arms.values()),
        "incomplete comparison arms or missing expected output/gradient",
    )
    for key, actual in arms["fast"].items():
        expected, oracle = arms["PEFT"][key], arms["float64"][key]
        row: dict[str, Any] = {
            "quantity": key, "phase": "forward" if key.startswith("Y/") else "backward",
            "passed": False,
        }
        try:
            row.update(probe._comparison(actual, expected, oracle, dtype))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if actual.shape == expected.shape == oracle.shape and all(
                torch.isfinite(value).all().item() for value in (actual, expected, oracle)
            ):
                fast_error = (actual.double() - oracle.double()).abs().max().item()
                peft_error = (expected.double() - oracle.double()).abs().max().item()
                row.update(
                    {
                        "bit_exact": torch.equal(actual, expected),
                        "max_abs_error": (actual.double() - expected.double()).abs().max().item(),
                        "fast_float64_max_abs_error": fast_error,
                        "peft_float64_max_abs_error": peft_error,
                        "proposed_max_error_bound": 2 * peft_error + 1e-8,
                    }
                )
        report["rows"].append(row)
    report["passed"] = all(row["passed"] for row in report["rows"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_file")
    parser.add_argument("--dtype", required=True, choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = replay(Path(args.tensor_file), args.dtype)
    except Exception as exc:
        report = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    with Path(args.output).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    Console().print(f"Replay {'PASS' if report['passed'] else 'FAIL'}: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
