#!/usr/bin/env python3
"""Ticket 04: L2L equals gradient accumulation on a small real model, k=1/2/4.

For each k the driver runs gradient accumulation twice (A-vs-A, to show the run is
repeatable) and one shipped L2L step, compares losses and LoRA gradients bit for bit,
and records the layer loads and NF4 restores of the accumulation and L2L steps.
LoRA dropout is 0. Raw records are rewritten after every k. Benchmark code, not
shipped. torch is imported inside functions.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_schedule  # noqa: E402
import stream_probe  # noqa: E402

DEFAULT_WEIGHTS = "HuggingFaceTB/SmolLM2-135M"


def _loads(counters: Any) -> Dict[str, int]:
    return {
        "layer_loads": int(counters.layer_loads),
        "large_loads": int(counters.large_loads),
        "restores": int(counters.restores),
    }


def measure_k(
    model: Any,
    runtime: Any,
    micro_batches: Sequence[Any],
    store_factory: Callable[[], Any],
    cold: Optional[Callable[[Any], None]] = None,
) -> Dict[str, Any]:
    """One k: GA twice, one L2L step, equality flags and load counters."""
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, count_loads

    mbs = list(micro_batches)
    k = len(mbs)

    def prepare() -> None:
        # Loads depend on what the previous walk left in the slots: every counted
        # span starts after the same forward-only walk.
        l2l_schedule.zero_grads(model)
        capture_layer_call(model, mbs[0])
        if cold is not None:
            cold(model)

    l2l_schedule.zero_grads(model)
    ga1 = l2l_schedule.ga_step(model, mbs)
    g1 = l2l_schedule.collect_grads(model)

    prepare()
    with count_loads(runtime) as ga_counters:
        ga2 = l2l_schedule.ga_step(model, mbs)
    g2 = l2l_schedule.collect_grads(model)

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]), seed=0)
    prepare()
    store = store_factory()
    try:
        losses = step.run(mbs, store, step=0)
    finally:
        store.close()
    g3 = l2l_schedule.collect_grads(model)

    a_vs_a = l2l_schedule.compare(g1, g2)
    l2l = l2l_schedule.compare(g1, g3)
    return {
        "k": k,
        "tensors": l2l["tensors"],
        "grads_equal": bool(l2l["equal"]),
        "max_abs": l2l["max_abs"],
        "worst": l2l["worst"],
        "all_zero": l2l["all_zero"],
        "losses_equal": bool(l2l_schedule.losses_equal(ga1, losses)),
        "a_vs_a_equal": bool(a_vs_a["equal"] and l2l_schedule.losses_equal(ga1, ga2)),
        "losses_ga": [float(x) for x in ga1],
        "losses_l2l": [float(x) for x in losses],
        "ga": _loads(ga_counters),
        "l2l": _loads(step.counters),
    }


def verdict(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass when every k is bit-equal, no gradient is vacuous, and L2L loads do not
    grow k-fold against k=1."""
    by_k = {rec["k"]: rec for rec in records}
    base = by_k.get(1)
    equal = bool(records) and all(
        rec["grads_equal"] and rec["losses_equal"] and not rec["all_zero"] for rec in records
    )
    loads_flat = base is not None and all(
        rec["l2l"]["layer_loads"] < rec["k"] * base["l2l"]["layer_loads"]
        for rec in records
        if rec["k"] > 1
    )
    return {
        "equal": equal,
        "loads_flat": loads_flat,
        "has_k1_baseline": base is not None,
        "l2l_layer_loads": {k: rec["l2l"]["layer_loads"] for k, rec in sorted(by_k.items())},
        "ga_layer_loads": {k: rec["ga"]["layer_loads"] for k, rec in sorted(by_k.items())},
        "pass": equal and loads_flat,
    }


def _cold_views(model: Any) -> None:
    """Drop the cached rebuilt NF4 views so a step's restores are counted (#841)."""
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    for layer in decoder_owner(model).layers:
        layer._cached_substitution_buffers = None
        layer._cached_substitution_weights = None


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--quant", choices=("none", "nf4"), required=True)
    parser.add_argument("--ks", default="1,2,4")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", required=True)
    parsed = parser.parse_args(argv)

    if parsed.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not stream_probe.cuda_available():
        print("SKIP: CUDA is required for l2l_smollm2_check.py")
        return 0
    import l2l_activations
    import l2l_probe
    import torch

    from soup_cli.utils.layer_stream import resolve_stream_dtype

    if parsed.deterministic:
        torch.use_deterministic_algorithms(True)
    out = Path(parsed.out)
    if out.exists():
        print(f"REFUSED: {out} exists; attempts are never overwritten")
        return 2
    args = l2l_probe.parse_args([
        "--mode", "correctness", "--weights", parsed.weights, "--quant", parsed.quant,
        "--seq", str(parsed.seq), "--tier", "ram", "--out", str(out / "unused.json"),
    ])
    device = "cuda"
    dtype = resolve_stream_dtype(device)
    torch_dtype = getattr(torch, dtype)
    facts = stream_probe.gpu_facts(device)
    print("RUN: ticket 04 L2L vs gradient accumulation,", parsed.weights, parsed.quant)
    for key, value in facts.items():
        print(f"{key:<16}{value}")
    model, runtime, config, *_ = stream_probe.build(args, device, dtype)
    l2l_schedule.make_non_vacuous(model)
    vocab = int(config.vocab_size)
    meta = {
        **facts, "weights": parsed.weights, "quant": parsed.quant, "dtype": dtype,
        "seq": parsed.seq, "deterministic": bool(parsed.deterministic),
        "arch": str(config.model_type), "n_layers": int(config.num_hidden_layers),
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    sink = stream_probe.Sink(str(out / "records.json"), meta)
    records: List[Dict[str, Any]] = []
    cold = _cold_views if parsed.quant == "nf4" else None
    try:
        for k in [int(part) for part in parsed.ks.split(",") if part.strip()]:
            gen = torch.Generator(device="cuda").manual_seed(args.input_seed)
            mbs = [torch.randint(0, vocab, (1, parsed.seq), generator=gen, device="cuda")
                   for _ in range(k)]

            def make_store(k=k) -> Any:
                return l2l_activations.ActivationStore(
                    n_layers=int(config.num_hidden_layers), k=k,
                    chunk_shape=(1, parsed.seq, int(config.hidden_size)),
                    dtype=torch_dtype, device="cuda", pin=True,
                )

            try:
                record = measure_k(model, runtime, mbs, make_store, cold=cold)
            except RuntimeError as exc:
                if not parsed.deterministic:
                    raise
                sink.add("error", {"k": k, "deterministic_error": str(exc)})
                continue
            records.append(record)
            sink.add("k", record)
            print(f"k={k}: grads_equal {record['grads_equal']} losses_equal "
                  f"{record['losses_equal']} (max |d| {record['max_abs']:.3e}) "
                  f"loads l2l {record['l2l']} ga {record['ga']}")
    finally:
        sink.payload["verdict"] = verdict(records)
        sink.flush()
        runtime.close()
    print(f"verdict {sink.payload['verdict']}\nwrote {out / 'records.json'}")
    return 0 if sink.payload["verdict"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
