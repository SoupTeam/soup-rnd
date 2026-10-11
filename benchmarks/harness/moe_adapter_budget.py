"""Rank-16 LoRA parameter counts per E1 arm for DeepSeek-V3 and Kimi K2, from their configs.

Usage: moe_adapter_budget.py
Prints, for each pinned config, the trainable parameters and the persistent AdamW fp32
bytes (16 per parameter: weight, gradient, two moments) of the four arms of
probe-moe-adapter-placement.md, with the same target rules as the harness: every
linear of the token mixer, the shared expert's three projections, and a rank-16 pair
per projection of each adapted routed expert. ``AT`` adapts a quarter of the routed
experts per MoE layer. Dense MLP layers (``first_k_dense_replace``) and the
multi-token-prediction layer get no adapter in any arm, as in the harness, which
adapts only decoder layers.

Two expert layouts are printed. ``fused``: ``gate`` and ``up`` are one ``gate_up``
matrix with one rank-16 pair, as transformers 5 stores experts and as the harness
adapts them. ``separate``: one pair each on ``gate``, ``up`` and ``down``, as in the
original checkpoints' per-expert modules; the research brief's "~10.2B on K2" is this
layout. Standard library only; the configs are fetched once into the Hugging Face
cache (``huggingface_hub`` is a Soup dependency).
"""

from __future__ import annotations

import json
import sys
from typing import Dict

RANK = 16
ADAMW_FP32_BYTES = 16
HOT_FRACTION = 0.25
CONFIGS = {
    "DeepSeek-V3": ("deepseek-ai/DeepSeek-V3", "e815299b0bcbac849fa540c768ef21845365c9eb"),
    "Kimi-K2": ("moonshotai/Kimi-K2-Instruct", "fd1984e2b7a3350dbf7305fe73a4ede25c14de50"),
}


def lora(fan_in: int, fan_out: int) -> int:
    return RANK * (fan_in + fan_out)


def mla_attention(cfg: Dict) -> int:
    """q_a, q_b, kv_a (with the rope key), kv_b and o projections of one MLA layer."""
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    nope, rope, value = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    q_rank, kv_rank = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    return (lora(hidden, q_rank) + lora(q_rank, heads * (nope + rope))
            + lora(hidden, kv_rank + rope) + lora(kv_rank, heads * (nope + value))
            + lora(heads * value, hidden))


def arms(cfg: Dict, *, fused: bool) -> Dict[str, int]:
    hidden, inter = cfg["hidden_size"], cfg["moe_intermediate_size"]
    layers = cfg["num_hidden_layers"]
    moe_layers = layers - cfg["first_k_dense_replace"]
    experts = cfg["n_routed_experts"]
    shared_inter = inter * cfg["n_shared_experts"]
    gate_up = lora(hidden, 2 * inter) if fused else 2 * lora(hidden, inter)
    expert = gate_up + lora(inter, hidden)
    shared = 2 * lora(hidden, shared_inter) + lora(shared_inter, hidden)
    attention = layers * mla_attention(cfg)
    attention_shared = attention + moe_layers * shared
    hot = max(1, round(experts * HOT_FRACTION))
    return {"A": attention, "AS": attention_shared,
            "AT": attention_shared + moe_layers * hot * expert,
            "AE": attention_shared + moe_layers * experts * expert}


def load_config(repo: str, revision: str) -> Dict:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "config.json", revision=revision), encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    from rich.console import Console

    console = Console(width=200)
    for name, (repo, revision) in CONFIGS.items():
        cfg = load_config(repo, revision)
        for layout in ("fused", "separate"):
            counts = arms(cfg, fused=layout == "fused")
            cells = ", ".join(f"{arm} {n / 1e6:,.0f}M = {n * ADAMW_FP32_BYTES / 2**30:,.1f} GiB"
                              for arm, n in counts.items())
            console.print(f"{name} ({repo}@{revision[:8]}) {layout}: {cells}", markup=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
