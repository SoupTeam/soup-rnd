"""Where does one MBR (or L2L control) step spend its time? PR #72 diagnostics.

One command, one verdict. It builds a streamed DeepSeek-V3 model, runs the step with and
without the phase clock of ``p72_phase_profiler`` in A-B-B-A blocks, prints the phases and
exits 1 when any gate fails:

* gap gate: ``|step wall - sum of phases| / step wall`` must stay under ``--max-gap`` (5%).
* coverage gate: ``unattributed_run_glue`` (time no phase claims) must stay under
  ``--max-unattributed`` (5%).
* parity gate: loss and every LoRA gradient of the instrumented step must be bit-equal
  (``torch.equal``) to the uninstrumented step. An uninstrumented A/A pair is the control.

CPU, seconds (a tiny synthetic checkpoint)::

    PYTHONPATH=$PWD/src:$PWD python benchmarks/harness/p72_mbr_phase_profile.py

``--coarse`` installs no phases. It must exit 1 (everything is unattributed); that is the
red state of this loop.

The Moonlight GPU step is measured by ``p3_route_preserving_mbr_soup_bridge.py
--phase-profile``, which feeds the same ``PhaseClock`` (see ``docs`` of that flag).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "src"), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from p72_phase_profiler import ROOT, PhaseClock, install  # noqa: E402

SMALL_M_BUCKETS = (8, 32, 128, 512)


def build_rig(root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    from benchmarks.harness import b13_moe_fixture as fixture
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )

    overrides = dict(
        hidden_size=args.hidden,
        moe_intermediate_size=args.moe_inter,
        n_routed_experts=args.experts,
        num_experts_per_tok=args.topk,
        num_hidden_layers=args.layers,
        vocab_size=args.vocab,
    )
    config = fixture.moe_config(**overrides)
    weights = fixture.moe_weights_dir(root, config)
    shards = root / "shards"
    nf4 = args.storage != "dense"
    arch = stream_arch_of(load_stream_config(str(weights), trust_remote_code=False))
    quant_args: Dict[str, Any] = {}
    if nf4:
        quant_args = dict(
            quant="nf4",
            quant_suffixes=quantised_layer_suffixes(
                build_meta_skeleton(str(weights), dtype="float32", quant="nf4")
            ),
            quant_device="cpu",
            double_quant=args.storage == "nf4-nested",
        )
    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch=arch,
        expert_layout="per_expert", **quant_args,
    )
    lora = fixture.moe_lora_config(dropout=0.0)
    model, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
        device=args.device, dtype="float32", buffers=2, pin=False, seed=71,
        quant="nf4" if nf4 else "none", double_quant=args.storage == "nf4-nested",
        tier="ram", read_ahead=2, two_read_experts=True, grouped_expert_routes=True,
    )
    fixture.randomize_adapters(model)
    model.train()
    batches = [
        fixture.moe_batch(config, seed=i, rows=args.rows, seq=args.seq)["input_ids"].to(args.device)
        for i in range(args.k)
    ]
    stepper = L2LStep(model, runtime, capture_layer_call(model, batches[0]))
    store = ActivationStore(
        n_layers=config.num_hidden_layers, k=args.k,
        chunk_shape=(args.rows, args.seq, config.hidden_size),
        dtype=torch.float32, device=args.device, pin=False,
    )
    params = fixture.trainables(model)
    initial = {name: p.detach().clone() for name, p in params.items()}
    return dict(
        config=config, model=model, runtime=runtime, stepper=stepper, store=store,
        batches=batches, params=params, initial=initial, fixture=fixture,
    )


def one_step(rig: Dict[str, Any], kind: str, args: argparse.Namespace, clock: Optional[PhaseClock]):
    """Reset adapters, run the step and a clipped AdamW step; return (seconds, losses, grads)."""
    import torch

    model, params = rig["model"], rig["params"]
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(rig["initial"][name])
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(list(params.values()), lr=1e-4, weight_decay=0.01)
    if clock is not None:
        clock.reset()

    def sync() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    sync()
    started = time.perf_counter()
    context = clock.run() if clock is not None else None
    if context is not None:
        context.__enter__()
    try:
        losses = rig["stepper"].run(
            rig["batches"], rig["store"],
            route_preserving_mbr=(kind == "mbr"),
            mbr_group_size=(args.group_size if kind == "mbr" else None),
        )
        grads = rig["fixture"].snapshot(model, gradients=True)
        if clock is not None:
            with clock.frame("optimizer_clip_and_step"):
                torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
                optimizer.step()
        else:
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
            optimizer.step()
    finally:
        if context is not None:
            context.__exit__(None, None, None)
    sync()
    return time.perf_counter() - started, [float(x.detach()) for x in losses], grads


def same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    import torch

    return a.keys() == b.keys() and all(
        (a[k] is None and b[k] is None)
        or (a[k] is not None and b[k] is not None and torch.equal(a[k], b[k]))
        for k in a
    )


def bucket(rows: int) -> str:
    for edge in SMALL_M_BUCKETS:
        if rows <= edge:
            return f"M<={edge}"
    return f"M>{SMALL_M_BUCKETS[-1]}"


def profile_kind(rig: Dict[str, Any], kind: str, args: argparse.Namespace) -> Dict[str, Any]:
    clock = PhaseClock(args.device)
    for _ in range(args.warmup):
        one_step(rig, kind, args, None)
    base: List[float] = []
    runs: List[Dict[str, Any]] = []
    base_ref: Dict[str, Any] = {}
    parity = {"instrumented_equals_baseline": True, "baseline_AA_equal": True}

    def baseline() -> None:
        seconds, losses, grads = one_step(rig, kind, args, None)
        base.append(seconds)
        if not base_ref:
            base_ref.update(losses=losses, grads=grads)
        else:
            parity["baseline_AA_equal"] &= losses == base_ref["losses"] and same(
                grads, base_ref["grads"]
            )

    def instrumented() -> None:
        if args.coarse:
            seconds, losses, grads = one_step(rig, kind, args, clock)
        else:
            with install(clock):
                seconds, losses, grads = one_step(rig, kind, args, clock)
        parity["instrumented_equals_baseline"] &= losses == base_ref["losses"] and same(
            grads, base_ref["grads"]
        )
        summed = sum(clock.totals.values())
        runs.append(
            dict(
                wall=seconds, summed=summed, totals=dict(clock.totals),
                counts=dict(clock.counts), background=dict(clock.background),
                gemm_rows=list(clock.gemm_rows),
            )
        )

    for _ in range(args.blocks):  # A-B-B-A: baseline, instrumented, instrumented, baseline
        baseline()
        instrumented()
        instrumented()
        baseline()

    return summarize_runs(
        kind, runs, uninstrumented_seconds=base, parity_ok=all(parity.values()),
        max_gap=args.max_gap, max_unattributed=args.max_unattributed, parity=parity,
    )


def summarize_runs(
    kind: str,
    runs: List[Dict[str, Any]],
    *,
    uninstrumented_seconds: List[float],
    parity_ok: bool,
    max_gap: float = 0.05,
    max_unattributed: float = 0.05,
    parity: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Medians over instrumented runs plus the three gates (shared with the GPU bridge)."""
    for run in runs:
        run.setdefault("summed", sum(run["totals"].values()))
    labels = sorted({label for run in runs for label in run["totals"]})
    wall = statistics.median(run["wall"] for run in runs)
    phases = {
        label: {
            "seconds": statistics.median(run["totals"].get(label, 0.0) for run in runs),
            "calls": statistics.median(run["counts"].get(label, 0) for run in runs),
        }
        for label in labels
    }
    for row in phases.values():
        row["share"] = row["seconds"] / wall
    gap = statistics.median(abs(run["wall"] - run["summed"]) / run["wall"] for run in runs)
    unattributed = phases.get(ROOT, {"seconds": 0.0})["seconds"] / wall
    base_median = statistics.median(uninstrumented_seconds)

    buckets: Dict[str, Dict[str, float]] = {}
    for run in runs:
        for label, rows, seconds in run.get("gemm_rows", []):
            row = buckets.setdefault(
                f"{label}|{bucket(rows)}", {"leases": 0, "seconds": 0.0, "rows": 0}
            )
            row["leases"] += 1
            row["seconds"] += seconds
            row["rows"] += rows
    for row in buckets.values():
        row["leases_per_step"] = row["leases"] / len(runs)
        row["seconds_per_step"] = row["seconds"] / len(runs)
        row["share_of_step"] = row["seconds_per_step"] / wall
        row["mean_rows"] = row["rows"] / row["leases"]
    background = {
        label: statistics.median(run.get("background", {}).get(label, 0.0) for run in runs)
        for label in sorted({label for run in runs for label in run.get("background", {})})
    }
    return dict(
        kind=kind, baseline_seconds=uninstrumented_seconds, baseline_median=base_median,
        background_seconds=background,
        instrumented_wall_median=wall, instrumentation_ratio=wall / base_median,
        phases=phases, gap=gap, unattributed=unattributed,
        parity=parity if parity is not None else {"instrumented_equals_baseline": parity_ok},
        gemm_by_bucket=buckets,
        gates=dict(
            gap_ok=gap <= max_gap, coverage_ok=unattributed <= max_unattributed,
            parity_ok=parity_ok,
        ),
    )


def render(result: Dict[str, Any]) -> str:
    lines = [
        f"== {result['kind']}: baseline {result['baseline_median']:.4f}s "
        f"(A-B-B-A medians), instrumented {result['instrumented_wall_median']:.4f}s "
        f"(x{result['instrumentation_ratio']:.2f})",
        f"{'phase':44s} {'sec':>9s} {'share':>7s} {'calls':>7s}",
    ]
    for label, row in sorted(result["phases"].items(), key=lambda kv: -kv[1]["seconds"]):
        lines.append(
            f"{label:44s} {row['seconds']:9.4f} {row['share']*100:6.1f}% {row['calls']:7.0f}"
        )
    lines.append(
        f"gap |wall-sum|/wall = {result['gap']*100:.2f}%   unattributed = "
        f"{result['unattributed']*100:.2f}%   parity = {result['parity']}"
    )
    for label, seconds in sorted(result.get("background_seconds", {}).items()):
        lines.append(f"  background (overlapped, not in the sum): {label} {seconds:.4f}s")
    lines.append("expert GEMM by rows per call (M), per step:")
    for key, row in sorted(result["gemm_by_bucket"].items()):
        lines.append(
            f"  {key:28s} leases {row['leases_per_step']:7.1f}  mean M {row['mean_rows']:7.1f}  "
            f"{row['seconds_per_step']:8.4f}s  {row['share_of_step']*100:5.1f}% of step"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--storage", default="dense", choices=("dense", "nf4", "nf4-nested"))
    parser.add_argument("--kinds", default="mbr,l2l")
    parser.add_argument("--k", type=int, default=4, help="micro-batches in the window")
    parser.add_argument("--group-size", type=int, default=4, help="MBR group size")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--seq", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--moe-inter", type=int, default=16)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=3, help="A-B-B-A blocks")
    parser.add_argument("--max-gap", type=float, default=0.05)
    parser.add_argument("--max-unattributed", type=float, default=0.05)
    parser.add_argument("--coarse", action="store_true", help="no phases: must exit 1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        rig = build_rig(Path(tmp), args)
        try:
            for kind in args.kinds.split(","):
                results.append(profile_kind(rig, kind, args))
        finally:
            rig["store"].close()
            rig["runtime"].close()
    failed = False
    for result in results:
        print(render(result))
        for gate, ok in result["gates"].items():
            print(f"  {gate}: {'PASS' if ok else 'FAIL'}")
            failed |= not ok
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("VERDICT:", "RED" if failed else "GREEN")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
