"""CPU profile of the routing Oracle on the tiny DeepSeek-V3 MoE (ticket 07, part 1).

The same route-preserving MBR step with and without
:class:`soup_cli.utils.routing_oracle.DeepseekRoutingOracle`, on the shipped tiny
fixture: BF16 compute, dense storage, k micro-batches of one row each. Three runs:

* a count run per arm, counting native router executions and ``aten.topk`` calls;
* a warmup step, then ``--steps`` timed steps per arm.

Reported as JSON and a short human summary: router and top-k calls per step, the
skipped fraction, and the step-time win. The Oracle skips exactly the recompute's
router and top-k work; the forward router stays native. A negative step-time win is a
result, not an error: the Oracle adds host copies and its own dispatch.

This is a CPU profile of call counts and wall time, not a GPU speed measurement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

DEFAULT_STEPS = 7
DEFAULT_WARMUP = 2


def _arguments(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=4, help="micro-batches per step")
    parser.add_argument("--seq", type=int, default=16, help="tokens per micro-batch")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="timed steps per arm")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="untimed steps per arm")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16",
        help="compute dtype (Moonlight is BF16)",
    )
    parser.add_argument("--json", type=str, default=None, help="write the report here")
    return parser.parse_args(argv)


@contextmanager
def _count_topk() -> Iterator[Dict[str, int]]:
    """Count ``aten.topk`` calls below the dispatcher, including the routers' own."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    topk = torch.ops.aten.topk.default
    tally = {"calls": 0}

    class Counter(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is topk:
                tally["calls"] += 1
            return func(*args, **(kwargs or {}))

    with Counter():
        yield tally


@contextmanager
def _count_native_router() -> Iterator[Dict[str, int]]:
    """Count executions of the native ``DeepseekV3TopkRouter.forward``.

    Installed before the Oracle is created, so the Oracle's captured native method is
    this counter too: its ``forward`` phase counts, its replay does not.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    tally = {"calls": 0}
    original = DeepseekV3TopkRouter.forward

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        tally["calls"] += 1
        return original(self, *args, **kwargs)

    DeepseekV3TopkRouter.forward = forward
    try:
        yield tally
    finally:
        DeepseekV3TopkRouter.forward = original


def _fixture(work: Path, *, k: int, seq: int, dtype: str) -> Dict[str, Any]:
    import torch

    from benchmarks.harness import b13_moe_fixture as fixture
    from soup_cli.utils import l2l_step
    from soup_cli.utils.l2l_activations import ActivationStore
    from tests.test_b13_moe_runtime import build_moe_streamed

    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = fixture.moe_config()
    weights = fixture.moe_weights_dir(work, config)
    moe = build_moe_streamed(
        weights,
        work / "shards",
        dtype=dtype,
        two_read_experts=True,
        grouped_expert_routes=True,
    )
    model = moe.model
    model.train()
    ids = [
        fixture.moe_batch(config, seed=seed, rows=1, seq=seq)["input_ids"]
        for seed in range(k)
    ]
    calls = [l2l_step.capture_layer_call(model, item) for item in ids]
    stepper = l2l_step.L2LStep(model, moe.runtime, calls[0])
    store_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    store = ActivationStore(
        n_layers=config.num_hidden_layers,
        k=k,
        chunk_shape=(1, seq, config.hidden_size),
        dtype=store_dtype,
        device="cpu",
        pin=False,
    )
    return {
        "torch": torch.__version__,
        "config": config,
        "moe": moe,
        "model": model,
        "ids": ids,
        "calls": calls,
        "stepper": stepper,
        "store": store,
        "dtype": dtype,
        "k": k,
        "seq": seq,
    }


def _zero_grads(model: Any) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _step(rig: Dict[str, Any], oracle: Any = None) -> float:
    _zero_grads(rig["model"])
    start = time.perf_counter()
    rig["stepper"].run(
        rig["ids"],
        rig["store"],
        calls=rig["calls"],
        route_preserving_mbr=True,
        mbr_group_size=rig["k"],
        collect_phase_stats=True,
        routing_oracle=oracle,
    )
    return time.perf_counter() - start


def _counts(rig: Dict[str, Any], oracle: Any) -> Dict[str, int]:
    if oracle is not None:
        _step(rig, oracle)
        stats = rig["stepper"].routing_oracle_stats
        assert stats is not None
        return {
            "router_calls": stats["step_counts"]["native_forward"],
            "router_forward": stats["step_counts"]["native_forward"],
            "router_recomputed": stats["step_counts"]["oracle_recompute"],
            "oracle_vjp": stats["step_counts"]["oracle_vjp"],
        }
    with _count_native_router() as router:
        _step(rig)
    return {
        "router_calls": router["calls"],
        "router_forward": router["calls"] // 2,
        "router_recomputed": router["calls"] // 2,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from soup_cli.utils.routing_oracle import DeepseekRoutingOracle

    report: Dict[str, Any] = {
        "kind": "oracle-cpu-profile",
        "device": "cpu",
        "k": args.k,
        "seq": args.seq,
        "dtype": args.dtype,
        "steps": args.steps,
        "warmup": args.warmup,
    }
    with tempfile.TemporaryDirectory(prefix="oracle-cpu-profile-") as tmp:
        rig = _fixture(Path(tmp), k=args.k, seq=args.seq, dtype=args.dtype)
        try:
            report["torch"] = rig["torch"]
            report["model"] = {
                "layers": int(rig["config"].num_hidden_layers),
                "moe_layers": int(
                    rig["config"].num_hidden_layers - rig["config"].first_k_dense_replace
                ),
                "experts": int(rig["config"].n_routed_experts),
                "top_k": int(rig["config"].num_experts_per_tok),
            }
            with _count_topk() as topk:
                baseline_counts = _counts(rig, None)
            with DeepseekRoutingOracle(rig["model"], pin=False) as oracle:
                with _count_topk() as oracle_topk:
                    oracle_counts = _counts(rig, oracle)

            for _ in range(args.warmup):
                _step(rig)
            baseline_times = [_step(rig) for _ in range(args.steps)]

            with DeepseekRoutingOracle(rig["model"], pin=False) as oracle:
                for _ in range(args.warmup):
                    _step(rig, oracle)
                oracle_times = [_step(rig, oracle) for _ in range(args.steps)]
                report["oracle"] = oracle.report()
                report["oracle_host"] = oracle.host_bytes()
        finally:
            rig["store"].close()
            rig["moe"].close()

    baseline_counts["topk_calls"] = topk["calls"]
    oracle_counts["topk_calls"] = oracle_topk["calls"]
    # The baseline recompute runs the router but two-read's TopkDecisions mode replays
    # every topk below it, so those decisions are dispatch entries, not aten executions.
    # With the Oracle the recompute runs no router at all. The per-router count is
    # measured in the baseline forward (one router call per forward pair).
    forward_pairs = baseline_counts["router_forward"]
    per_router = (
        int(round(baseline_counts["topk_calls"] / forward_pairs)) if forward_pairs else 0
    )
    baseline_counts["topk_dispatches"] = baseline_counts["topk_calls"] + (
        per_router * baseline_counts["router_recomputed"]
    )
    oracle_counts["topk_dispatches"] = oracle_counts["topk_calls"]
    report["topk_per_router_forward"] = per_router
    report["counts"] = {"without_oracle": baseline_counts, "with_oracle": oracle_counts}
    report["router_skipped_fraction"] = 1.0 - (
        oracle_counts["router_calls"] / baseline_counts["router_calls"]
    )
    report["topk_skipped_fraction"] = 1.0 - (
        oracle_counts["topk_dispatches"] / baseline_counts["topk_dispatches"]
    )
    report["time_seconds"] = {
        "without_oracle": baseline_times,
        "with_oracle": oracle_times,
        "without_oracle_median": statistics.median(baseline_times),
        "with_oracle_median": statistics.median(oracle_times),
    }
    baseline_median = report["time_seconds"]["without_oracle_median"]
    oracle_median = report["time_seconds"]["with_oracle_median"]
    report["time_win_percent"] = 100.0 * (baseline_median - oracle_median) / baseline_median
    return report


def _summary(report: Dict[str, Any]) -> str:
    baseline = report["counts"]["without_oracle"]
    oracle = report["counts"]["with_oracle"]
    lines = [
        f"CPU Oracle profile: k={report['k']} S={report['seq']} {report['dtype']} "
        f"{report['model']['moe_layers']} MoE layers, top-k {report['model']['top_k']}",
        f"router calls/step: {baseline['router_calls']} -> {oracle['router_calls']} "
        f"({100 * report['router_skipped_fraction']:.1f}% skipped)",
        f"top-k dispatches/step: {baseline['topk_dispatches']} -> "
        f"{oracle['topk_dispatches']} "
        f"({100 * report['topk_skipped_fraction']:.1f}% skipped); "
        f"aten::topk executions {baseline['topk_calls']} -> {oracle['topk_calls']} "
        f"(two-read already replays the recompute's top-k)",
        f"step median: {report['time_seconds']['without_oracle_median']:.4f} s "
        f"-> {report['time_seconds']['with_oracle_median']:.4f} s "
        f"({report['time_win_percent']:+.1f}%)",
    ]
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    from rich.console import Console

    args = _arguments(argv)
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")
    Console().print(_summary(report), markup=False, highlight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
