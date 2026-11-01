"""Finite development CUDA screen; no timing and no pretrained/trainer verdict.

Compares a grouped fused launch with singleton fused launches AND an independent
bitsandbytes dequantize/BF16 mm reference. Native mismatches fail, even if the
grouped/singleton comparison passes. Run only on a physical Linux CUDA host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path


def evaluate_cells(cells: list[dict]) -> list[str]:
    """Never turn an incomplete grid or singleton-only parity into a PASS."""
    expected = {
        (shape, nested, transpose, rows)
        for shape in ((2048, 2048), (2048, 1024))
        for nested in (False, True)
        for transpose in (False, True)
        for rows in ((0, 1, 17, 129), (31, 128, 481, 512))
    }
    controls = {(shape, nested) for shape in ((2048, 2048), (2048, 1024))
                for nested in (False, True)}
    failures = []
    for cell in cells:
        shape = tuple(cell["shape"])
        if cell.get("kind") == "control":
            key = (shape, cell["nested"])
            if key not in controls:
                failures.append("unexpected/duplicate control")
            else:
                controls.remove(key)
            if not cell["negative_effective"] or not cell["restored_exact"]:
                failures.append("control")
        else:
            key = (shape, cell["nested"], cell["transpose"], tuple(cell["rows"]))
            if key not in expected:
                failures.append("unexpected/duplicate arithmetic cell")
            else:
                expected.remove(key)
            fields = ("group_singleton_exact", "group_native_exact", "finite")
            if len(cell["experts"]) != 4 or any(
                not all(e[k] for k in fields) for e in cell["experts"]
            ):
                failures.append("native/grouped exactness")
    if expected or controls:
        failures.append("incomplete grid")
    return failures


def run(out: Path, *, backend: str = "triton") -> dict:
    import bitsandbytes as bnb
    import torch

    from soup_cli.utils.expert_group_fused import PackedNF4
    from soup_cli.utils.expert_group_fused import fused_group_linear as launch

    fused_group_linear = partial(launch, backend=backend)

    if sys.platform != "linux" or torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("Linux CUDA sm_80+ required")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("native BF16 required")
    torch.manual_seed(314159)
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    root = Path(__file__).parents[2]
    sources = [Path(__file__), root / "src/soup_cli/utils/expert_group_fused.py",
               root / "src/soup_cli/utils/expert_group_fused_wmma.py"]
    report = {
        "kind": "unqualified D3 fused packed NF4 development screen",
        "torch": torch.__version__, "bitsandbytes": bnb.__version__,
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "seed": 314159, "backend": backend, "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        }, "cells": [], "status": "incomplete", "production_enabled": False,
    }
    out.mkdir(parents=True, exist_ok=False)

    def exact(a, b):
        return (a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a.view(torch.int16), b.view(torch.int16)))

    try:
        for shape in ((2048, 2048), (2048, 1024)):
            for nested in (False, True):
                weights, references = [], []
                for _ in range(4):
                    base = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                    packed, state = bnb.functional.quantize_4bit(
                        base, blocksize=64, compress_statistics=nested, quant_type="nf4")
                    references.append(bnb.functional.dequantize_4bit(packed, state))
                    weights.append(PackedNF4(
                        packed.flatten(), state.absmax, state.code, shape,
                        nested_scales=state.state2.absmax if nested else None,
                        nested_code=state.state2.code if nested else None,
                        offset=state.offset.reshape(1) if nested else None,
                    ))
                for transpose in (False, True):
                    for rows in ([0, 1, 17, 129], [31, 128, 481, 512]):
                        inputs = [torch.randn(m, shape[0 if transpose else 1],
                                              device="cuda", dtype=torch.bfloat16) for m in rows]
                        grouped = fused_group_linear(inputs, weights, transpose=transpose,
                                                     call_budget_bytes=64 * 2**20)
                        singleton = [fused_group_linear([x], [w], transpose=transpose,
                                                       call_budget_bytes=64 * 2**20)[0]
                                     for x, w in zip(inputs, weights)]
                        native = [x @ (w if transpose else w.t())
                                  for x, w in zip(inputs, references)]
                        per_expert = []
                        for a, b, c in zip(grouped, singleton, native):
                            per_expert.append({
                                "group_singleton_exact": exact(a, b),
                                "group_native_exact": exact(a, c),
                                "finite": bool(torch.isfinite(a).all().item()),
                                "native_differing_bits": int((a.view(torch.int16)
                                                              != c.view(torch.int16)).sum().item()),
                                "native_max_abs": float((a.float() - c.float()).abs().max().item())
                                if a.numel() else 0.0,
                            })
                        report["cells"].append({"shape": shape, "nested": nested,
                                                "transpose": transpose, "rows": rows,
                                                "planned_paths": [
                                                    "empty" if m == 0 else
                                                    "native-small" if backend == "hybrid_wmma"
                                                    and m <= 128 else "fused" for m in rows],
                                                "experts": per_expert})
                # Effective packed-weight corruption control followed by restored positive.
                x = torch.ones((129, shape[1]), dtype=torch.bfloat16, device="cuda")
                w = weights[0]
                before = fused_group_linear([x], [w], call_budget_bytes=64 * 2**20)[0]
                original = w.packed.clone()
                w.packed.bitwise_xor_(255)
                changed = fused_group_linear([x], [w], call_budget_bytes=64 * 2**20)[0]
                w.packed.copy_(original)
                restored = fused_group_linear([x], [w], call_budget_bytes=64 * 2**20)[0]
                report["cells"].append({"shape": shape, "nested": nested, "kind": "control",
                                        "negative_effective": not exact(before, changed),
                                        "restored_exact": exact(before, restored)})
        failures = evaluate_cells(report["cells"])
        report["status"] = "fail" if failures else "development_numerics_pass"
        report["failures"] = failures
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = repr(exc)
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("triton", "wmma", "hybrid_wmma"), default="triton")
    args = parser.parse_args()
    result = run(args.out, backend=args.backend)
    Console().print(f"Status: {result['status']}; cells: {len(result['cells'])}")
    raise SystemExit(0 if result["status"] == "development_numerics_pass" else 1)
