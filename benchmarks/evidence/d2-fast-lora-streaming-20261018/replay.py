"""Replay stored Soup streamed-training pairs on CPU; no speed or fresh-GPU verdict."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from rich.console import Console


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def replay(root: Path, device: str) -> dict:
    import torch
    from safetensors.torch import load_file

    adapters = {
        f"base_model.model.model.layers.{layer}.{group}.{projection}.lora_{matrix}.default.weight"
        for layer in range(4)
        for group, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
        for matrix in ("A", "B")
    }
    suffixes = {f"{group}/{name}" for group in ("initial", "final") for name in adapters}
    suffixes |= {f"gradient/{step}/{name}" for step in range(2) for name in adapters}
    suffixes |= {f"logits/{step}" for step in range(4)} | {"loss"}
    expected_keys = {f"{arm}/{suffix}" for arm in ("lora", "fast") for suffix in suffixes}
    rtol, atol = (1e-2, 1e-5) if device == "cuda" else (1e-5, 1e-7)
    rows = []
    cases = []
    for precision, source, quant in itertools.product(
        ("bf16", "fp16") if device == "cuda" else ("fp32",),
        ("ram", "disk"),
        ("none", "4bit"),
    ):
        stem = f"{device}-{precision}-{source}-{quant}"
        metadata_path = root / f"{stem}.json"
        tensor_path = root / f"{stem}.safetensors"
        require(metadata_path.is_file() and tensor_path.is_file(), f"missing case: {stem}")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        require(meta.get("passed") is True, f"case did not pass: {stem}")
        require(
            (meta["device"], meta["precision"], meta["source"], meta["quantization"])
            == (device, precision, source, quant),
            f"case identity differs: {stem}",
        )
        require((meta["rtol"], meta["atol"]) == (rtol, atol), "changed numerical tolerance")
        for runtime in meta["runtime_stats"]:
            require(
                runtime["n_layers"] == 4 and runtime["buffers"] == 2, "wrong layer/buffer matrix"
            )
            require(
                runtime["tier"] == source and runtime["layer_loads"] > 16,
                "no streamed recomputation",
            )
        tensors = load_file(str(tensor_path), device="cpu")
        require(set(tensors) == expected_keys, f"missing/extra tensor quantities: {stem}")
        for suffix in sorted(suffixes):
            first, second = tensors[f"lora/{suffix}"], tensors[f"fast/{suffix}"]
            require(
                bool(torch.isfinite(first).all() and torch.isfinite(second).all()),
                f"nonfinite quantity: {stem}/{suffix}",
            )
            exact = suffix.startswith("initial/")
            torch.testing.assert_close(
                first,
                second,
                rtol=0 if exact else rtol,
                atol=0 if exact else max(atol, 1e-6) if suffix.startswith("logits/") else atol,
            )
            if suffix == "loss":
                require(first.numel() == second.numel() == 2, "missing optimizer loss")
                require(
                    all(
                        f"{a:.3f}" == f"{b:.3f}"
                        for a, b in zip(first.tolist(), second.tolist(), strict=True)
                    ),
                    "three-decimal loss mismatch",
                )
            rows.append(
                {
                    "case": stem,
                    "quantity": suffix,
                    "max_abs": float((first.double() - second.double()).abs().max()),
                    "exact": bool(torch.equal(first, second)),
                }
            )
        cases.append(
            {"name": stem, "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest()}
        )
    return {
        "status": "PASS",
        "scope": "CPU replay of stored training pairs only",
        "cases": cases,
        "comparisons": len(rows),
        "rows": rows,
    }


def replay_oracles(root: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    rows = []
    for dtype in ("bf16", "fp16"):
        folder = root / f"stream-oracle-{dtype}"
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        require(report["passed"] is True and report["loss_3f_equal"], "oracle gate failed")
        tensors = load_file(str(folder / "streaming.safetensors"), device="cpu")
        quantities = {key.removeprefix("PEFT/") for key in tensors if key.startswith("PEFT/")}
        require(
            len(quantities) == 58 and {"Y/loss", "Y/logits"} <= quantities,
            "oracle quantity count differs",
        )
        require(
            set(tensors)
            == {f"{arm}/{key}" for arm in ("PEFT", "fast", "float64") for key in quantities},
            "oracle arm is incomplete",
        )
        for key in sorted(quantities):
            values = [tensors[f"{arm}/{key}"] for arm in ("PEFT", "fast", "float64")]
            require(all(bool(torch.isfinite(value).all()) for value in values), "nonfinite oracle")
            peft, fast, reference = (value.double() for value in values)
            error = float((fast - reference).abs().max())
            baseline_error = float((peft - reference).abs().max())
            require(error <= 2 * baseline_error + 1e-8, f"oracle bound failed: {dtype}/{key}")
            rows.append(
                {
                    "dtype": dtype,
                    "quantity": key,
                    "exact": bool(torch.equal(peft, fast)),
                    "fast_max_error": error,
                    "peft_max_error": baseline_error,
                }
            )
    return {"status": "PASS", "comparisons": len(rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracles", type=Path)
    args = parser.parse_args()
    result = replay(args.path, args.device)
    if args.oracles:
        result["oracles"] = replay_oracles(args.oracles)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Console().print(
        {
            "status": result["status"],
            "cases": len(result["cases"]),
            "comparisons": result["comparisons"],
            "scope": result["scope"],
        }
    )


if __name__ == "__main__":
    main()
