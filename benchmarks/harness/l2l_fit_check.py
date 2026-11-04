#!/usr/bin/env python3
"""Memory fit check before the A2-03 SmolLM2 runs: refuses to start if the card is too small.

With ``--records`` it instead checks the allocator peak a probe run measured. Otherwise reads
the model config (no weights), adds up the bytes the check needs on the GPU at the
largest k, and compares the total with ``cudaMemGetInfo``. The estimate counts every
resident weight, the LoRA adapters with gradients and Adam-sized slack, a dozen
layer-sized tensors per micro-batch and the fp32 logits of every micro-batch, then
doubles the sum. Benchmark code, not shipped. torch is imported inside functions.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict


def estimate_bytes(config: Any, *, seq: int, k: int, lora_r: int = 16, element: int = 2,
                   n_targets: int = 4) -> Dict[str, int]:
    """GPU bytes for the weights, adapters, activations and logits, before the safety factor."""
    hidden = int(config.hidden_size)
    inter = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    vocab = int(config.vocab_size)
    heads = int(config.num_attention_heads)
    kv = int(getattr(config, "num_key_value_heads", None) or heads)
    head_dim = hidden // heads
    per_layer = (
        hidden * hidden * 2 + hidden * kv * head_dim * 2  # q, o, k, v
        + 3 * hidden * inter + 2 * hidden  # mlp and norms
    )
    embed = vocab * hidden * (1 if getattr(config, "tie_word_embeddings", False) else 2)
    weights = (layers * per_layer + embed + hidden) * element
    lora = layers * n_targets * lora_r * (hidden + hidden) * (element + 4 + 4 + 4)
    activations = 12 * seq * max(hidden, inter) * element * k
    logits = k * seq * vocab * 4 * 3
    return {"weights": weights, "lora": lora, "activations": activations, "logits": logits}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def measured_fit(*, peak_bytes: int, free_bytes: int, margin: float = 1.25) -> Dict[str, Any]:
    """Fit from a measured allocator peak: the peak times ``margin`` must stay under free."""
    return {
        "checked_at": _now(), "peak_bytes": int(peak_bytes), "free_bytes": int(free_bytes),
        "margin": margin, "fits": peak_bytes * margin < free_bytes,
    }


def check(weights: str, *, seq: int, k: int, factor: float = 2.0) -> Dict[str, Any]:
    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(weights)
    parts = estimate_bytes(config, seq=seq, k=k)
    need = int(sum(parts.values()) * factor)
    free, total = torch.cuda.mem_get_info()
    return {
        "weights": weights, "seq": seq, "k": k, "factor": factor, "parts": parts,
        "need_bytes": need, "free_bytes": int(free), "total_bytes": int(total),
        "fits": need < free, "checked_at": _now(),
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument(
        "--records", help="records.json of a probe run: check its measured peak instead"
    )
    parsed = parser.parse_args(argv)
    if parsed.records:
        import torch

        records = json.loads(open(parsed.records).read())["records"]
        peaks = [rec["peak_gpu_bytes"] for rec in records if rec.get("peak_gpu_bytes")]
        if not peaks:
            print("no peak_gpu_bytes in the records", file=sys.stderr)
            return 2
        report = measured_fit(peak_bytes=max(peaks), free_bytes=torch.cuda.mem_get_info()[0])
        print(json.dumps(report, indent=2))
        return 0 if report["fits"] else 1
    report = check(parsed.weights, seq=parsed.seq, k=parsed.max_k)
    print(json.dumps(report, indent=2))
    return 0 if report["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
