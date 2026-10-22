#!/usr/bin/env python3
"""G4 ticket 02: one linear layer, native FP4 GEMM against NF4 plus a bf16 matmul.

Two paths run the same layer on the same synthetic data:

- ``fp4``: ``te.Linear`` under TransformerEngine's ``NVFP4BlockScaling``, which
  multiplies on the packed weights (the native FP4 GEMM of GLOSSARY.md).
- ``nf4``: bitsandbytes NF4 weights, restored to bf16, then a plain bf16 matmul.

Each path records its error against the bf16 product and its time. A shape the FP4
GEMM cannot run (see ``FP4_MULTIPLES``) is refused with ``UnsupportedShapeError``. The
layer never falls back to NF4, because that would mix two formats in one arm
(ADR 0003, ADR 0001). The cost of padding to a multiple of 128 is measured
separately, on synthetic shapes.

All numbers are synthetic: random data, one layer, not training. Benchmark
harness code, not shipped. Heavy dependencies are imported inside functions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

#: Multiple each dimension must have for the native FP4 GEMM. TransformerEngine's own check is 16
#: for all three (``NVFP4_BLOCK_SIZE`` in ``csrc/quantizer.cpp``). On an RTX 5090 (sm_120) cuBLAS
#: finds no FP4 algorithm for in_features (K) = 4112, so K is required to be a multiple of 32.
#: One value was tried; this is a conservative guard (see benchmarks/gate-g4-single-layer.md). tokens and out_features only need 16.
FP4_MULTIPLES = {"tokens": 16, "in_features": 32, "out_features": 16}


class UnsupportedShapeError(ValueError):
    """The shape is not one TransformerEngine can run as native FP4."""


def check_fp4_shape(*, tokens: int, in_features: int, out_features: int) -> None:
    """Refuse a shape with a dimension that is not a multiple of what the FP4 GEMM needs."""
    bad = [
        f"{name}={value} (needs a multiple of {FP4_MULTIPLES[name]})"
        for name, value in (
            ("tokens", tokens),
            ("in_features", in_features),
            ("out_features", out_features),
        )
        if value % FP4_MULTIPLES[name]
    ]
    if bad:
        raise UnsupportedShapeError(
            f"native FP4 GEMM needs every dimension divisible by its multiple; got "
            f"{', '.join(bad)}. There is no fallback to NF4: it would mix two formats in "
            "one arm. Change the shape or pad it."
        )


#: The multiple the torchao NVFP4 training path requires; the padding cost is measured against it.
PAD_MULTIPLE = 128


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def padding_cost(
    *, tokens: int, in_features: int, out_features: int, multiple: int = PAD_MULTIPLE
) -> dict:
    """What padding every dimension up to ``multiple`` adds, as fractions of the unpadded size.

    ``flop_overhead`` is the extra multiply-adds, ``weight_overhead`` the extra weight elements.
    """
    padded = {
        "tokens": _round_up(tokens, multiple),
        "in_features": _round_up(in_features, multiple),
        "out_features": _round_up(out_features, multiple),
    }
    flops = tokens * in_features * out_features
    padded_flops = padded["tokens"] * padded["in_features"] * padded["out_features"]
    weights = in_features * out_features
    padded_weights = padded["in_features"] * padded["out_features"]
    return {
        "padded": padded,
        "flop_overhead": padded_flops / flops - 1,
        "weight_overhead": padded_weights / weights - 1,
    }

PATHS = {"A": "native_fp4", "B": "nf4_dequant_bf16"}

REPORT_LABEL = "RTX 5090, synthetic (random data, one layer)"


def _mean(values: list) -> float:
    return sum(values) / len(values)


def build_report(blocks: list, *, shape: dict, padding: list, label: str = REPORT_LABEL) -> dict:
    """Merge A-B-B-A blocks into one report. A block with a void run is left out, not averaged.

    Each run is ``{"arm", "value", "outcome", "error": {"rel_error", "cosine"}}``. Arm A is
    native FP4, arm B is NF4 restored to bf16.
    """
    valid = [block for block in blocks if all(run["outcome"] != "void" for run in block)]
    if not valid:
        raise ValueError("no block without a void run; nothing to report")
    arms = {}
    for arm, path in PATHS.items():
        runs = [run for block in valid for run in block if run["arm"] == arm]
        arms[arm] = {
            "path": path,
            "runs": len(runs),
            "mean_seconds": _mean([run["value"] for run in runs]),
            "rel_error": _mean([run["error"]["rel_error"] for run in runs]),
            "cosine": _mean([run["error"]["cosine"] for run in runs]),
        }
    return {
        "label": label,
        "data_kind": "synthetic",
        "shape": shape,
        "valid_blocks": len(valid),
        "void_blocks": len(blocks) - len(valid),
        "arms": arms,
        "padding_to_128": padding,
    }


#: The arm watch's required checks, as committed in benchmarks/gate-g4-single-layer.md.
REQUIRED = ("suspend", "power")


def _error_against(reference, result) -> Dict[str, float]:
    import torch

    ref, out = reference.float(), result.float()
    return {
        "rel_error": ((out - ref).norm() / ref.norm()).item(),
        "cosine": torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item(),
        "max_abs_diff": (out - ref).abs().max().item(),
    }


def _build_fp4_step(weight, x, pad: bool) -> Callable[[], object]:
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    recipe = NVFP4BlockScaling()
    out_features, in_features = weight.shape
    tokens = x.shape[0]
    if pad:
        out_p = _round_up(out_features, PAD_MULTIPLE)
        in_p = _round_up(in_features, PAD_MULTIPLE)
        tokens_p = _round_up(tokens, PAD_MULTIPLE)
        padded = torch.zeros(out_p, in_p, dtype=weight.dtype, device=weight.device)
        padded[:out_features, :in_features] = weight
        weight = padded
    with te.quantized_model_init(enabled=True, recipe=recipe):
        layer = te.Linear(weight.shape[1], weight.shape[0], bias=False, params_dtype=weight.dtype)
    layer = layer.cuda()
    with torch.no_grad():
        layer.weight.copy_(weight)

    def step():
        with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
            if pad:
                xp = torch.nn.functional.pad(x, (0, in_p - in_features, 0, tokens_p - tokens))
                return layer(xp)[:tokens, :out_features]
            return layer(x)

    step.weight_storage = type(layer.weight).__name__  # type: ignore[attr-defined]
    return step


def _build_nf4_step(weight, x) -> Callable[[], object]:
    import bitsandbytes.functional as bnb_f
    import torch

    packed, state = bnb_f.quantize_4bit(weight, quant_type="nf4")

    def step():
        with torch.no_grad():
            restored = bnb_f.dequantize_4bit(packed, state).to(weight.dtype)
            return torch.nn.functional.linear(x, restored)

    return step


def run_layer(
    *,
    path: str,
    tokens: int,
    in_features: int,
    out_features: int,
    steps: int,
    warmup: int,
    pad: bool = False,
    seed: int = 0,
) -> Dict[str, object]:
    """Run one arm once: build the layer, check its output against bf16, time ``steps`` calls.

    ``path`` is ``fp4`` or ``nf4``. The number is the median of the timed steps, in seconds.
    """
    if path == "fp4":
        check_fp4_shape(
            tokens=_round_up(tokens, PAD_MULTIPLE) if pad else tokens,
            in_features=_round_up(in_features, PAD_MULTIPLE) if pad else in_features,
            out_features=_round_up(out_features, PAD_MULTIPLE) if pad else out_features,
        )
    elif path != "nf4":
        raise ValueError(f"path must be 'fp4' or 'nf4', got {path!r}")
    import torch

    torch.manual_seed(seed)
    weight = (torch.randn(out_features, in_features, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(tokens, in_features, device="cuda").to(torch.bfloat16)
    reference = torch.nn.functional.linear(x, weight)
    step = _build_fp4_step(weight, x, pad) if path == "fp4" else _build_nf4_step(weight, x)
    error = _error_against(reference, step())
    millis: List[float] = []
    for index in range(warmup + steps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        if index >= warmup:
            millis.append(start.elapsed_time(end))
    return {
        "median_seconds": statistics.median(millis) / 1000,
        "error": error,
        "weight_storage": getattr(step, "weight_storage", "bf16 restored per step"),
        "settings": {
            "path": path, "pad": pad, "tokens": tokens, "in_features": in_features,
            "out_features": out_features, "steps": steps, "warmup": warmup, "seed": seed,
        },
    }  # fmt: skip


def _worker(args: argparse.Namespace) -> int:
    result = run_layer(
        path=args.path, tokens=args.tokens, in_features=args.k, out_features=args.n,
        steps=args.steps, warmup=args.warmup, pad=args.pad,
    )  # fmt: skip
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def refusal(args: argparse.Namespace) -> int:
    """Show the refusal for a shape that is not a multiple of 16: ours, then TransformerEngine's."""
    shape = dict(tokens=args.tokens, in_features=args.k, out_features=args.n)
    report: Dict[str, object] = {"shape": shape}
    try:
        check_fp4_shape(**shape)
    except UnsupportedShapeError as exc:
        report["harness_error"] = f"{type(exc).__name__}: {exc}"
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    layer = te.Linear(args.k, args.n, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(args.tokens, args.k, device="cuda", dtype=torch.bfloat16)
    try:
        with torch.no_grad(), te.autocast(enabled=True, recipe=NVFP4BlockScaling()):
            layer(x)
        report["transformer_engine"] = "accepted the shape"
    except Exception as exc:  # noqa: BLE001
        report["transformer_engine"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    return 0


def _driver(args: argparse.Namespace) -> int:
    """A-B-B-A blocks of fresh worker processes, each under an ``ArmWatch`` (METHOD.md)."""
    from arm_validity import (
        ArmWatch,
        Run,
        block_order,
        blocks_to_add,
        repeats,
        valid_blocks,
    )

    arms = {"A": args.arm_a, "B": args.arm_b}  # each "fp4", "nf4" or "fp4+pad"
    out_dir = Path(args.out).parent / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks: List[List[Dict[str, object]]] = []
    runs = lambda: [[Run(r["value"], r["outcome"]) for r in b] for b in blocks]  # noqa: E731
    while (todo := blocks_to_add(runs())) > 0:
        if len(blocks) >= args.max_blocks:
            raise RuntimeError(f"{len(blocks)} blocks run and {todo} still due; stopping")
        block = []
        for position, arm in enumerate(block_order(1)[0]):
            run_no = len(blocks) * 4 + position + 1
            result_file = out_dir / f"run{run_no:03d}-{arm}.json"
            path, _, pad = arms[arm].partition("+")
            command = [
                sys.executable, __file__, "worker", "--path", path, "--tokens", str(args.tokens),
                "--k", str(args.k), "--n", str(args.n), "--steps", str(args.steps),
                "--warmup", str(args.warmup), "--out", str(result_file),
            ] + (["--pad"] if pad else [])  # fmt: skip
            with ArmWatch(
                arm=arm, round=2 * len(blocks) + 1 + position // 2, run=run_no,
                label=REPORT_LABEL, required=REQUIRED, model_path=str(out_dir),
            ) as watch:  # fmt: skip
                subprocess.run(command, check=True)
            result = json.loads(result_file.read_text(encoding="utf-8"))
            block.append({
                "arm": arm, "value": result["median_seconds"], "error": result["error"],
                "outcome": watch.record["outcome"], "record": watch.record,
                "weight_storage": result["weight_storage"],
            })  # fmt: skip
            print(f"run {run_no} arm {arm} ({arms[arm]}): {result['median_seconds']:.6f} s, "
                  f"{watch.record['outcome']}", flush=True)  # fmt: skip
        blocks.append(block)
    valid = valid_blocks(runs())
    rep = repeats(runs())
    report = build_report(
        blocks, shape={"tokens": args.tokens, "in_features": args.k, "out_features": args.n},
        padding=[padding_cost(tokens=args.tokens, in_features=args.k, out_features=args.n)],
    )  # fmt: skip
    report["arms"]["A"]["path"], report["arms"]["B"]["path"] = arms["A"], arms["B"]
    report["sigma"], report["blocks_needed"] = rep.sigma, rep.blocks_needed
    report["blocks"] = blocks
    report["valid_block_count"] = len(valid)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "blocks"}, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("worker", "driver", "refusal"):
        p = sub.add_parser(name)
        p.add_argument("--tokens", type=int, default=4096)
        p.add_argument("--k", type=int, default=4096, help="in_features")
        p.add_argument("--n", type=int, default=4096, help="out_features")
        if name != "refusal":
            p.add_argument("--steps", type=int, default=200)
            p.add_argument("--warmup", type=int, default=50)
            p.add_argument("--out", required=True)
        if name == "worker":
            p.add_argument("--path", choices=("fp4", "nf4"), required=True)
            p.add_argument("--pad", action="store_true")
        if name == "driver":
            p.add_argument("--arm-a", default="fp4")
            p.add_argument("--arm-b", default="nf4")
            p.add_argument("--max-blocks", type=int, default=12)
    args = parser.parse_args(argv)
    return {"worker": _worker, "driver": _driver, "refusal": refusal}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
