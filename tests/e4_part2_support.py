"""Shared test helpers for the E4 part 2 harness: harness import path and a tiny random OLMoE.

The tiny model has the real model's structure (fused 3-D experts, q/k norms, router) at a
size a CPU test can build in a second.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO_ROOT / "benchmarks" / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

TINY = dict(
    vocab_size=128, hidden_size=64, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4, num_experts=4, num_experts_per_tok=2,
    max_position_embeddings=128, pad_token_id=1, eos_token_id=2, tie_word_embeddings=False,
)


def tiny_config(**overrides):
    from transformers import OlmoeConfig

    return OlmoeConfig(**{**TINY, **overrides})


def tiny_model(seed: int = 0, dtype=None, experts_implementation: str = "eager", **overrides):
    """A tiny random OLMoE (fused experts, weights drawn from the seed)."""
    import torch
    from transformers import OlmoeForCausalLM

    torch.manual_seed(seed)
    config = tiny_config(**overrides)
    config._experts_implementation = experts_implementation
    model = OlmoeForCausalLM(config)
    for param in model.parameters():  # std 0.02 init leaves the experts near-degenerate
        with torch.no_grad():
            param.copy_(torch.randn(param.shape, generator=torch.Generator().manual_seed(
                seed + hash(tuple(param.shape)) % 1000)) * 0.05)
    return model.to(dtype) if dtype is not None else model


def write_legacy_checkpoint(model, directory: Path, shards: int = 2) -> Path:
    """Save ``model`` in the original OLMoE layout: one 2-D tensor per expert projection.

    This is the layout of the real checkpoint (``experts.{e}.{gate,up,down}_proj.weight``);
    transformers 5 fuses it on load.
    """
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    inter = model.config.intermediate_size
    flat = {}
    for key, tensor in model.state_dict().items():
        if key.endswith("experts.gate_up_proj"):
            base = key[: -len("gate_up_proj")]
            for e in range(tensor.shape[0]):
                flat[f"{base}{e}.gate_proj.weight"] = tensor[e, :inter].clone()
                flat[f"{base}{e}.up_proj.weight"] = tensor[e, inter:].clone()
        elif key.endswith("experts.down_proj"):
            base = key[: -len("down_proj")]
            for e in range(tensor.shape[0]):
                flat[f"{base}{e}.down_proj.weight"] = tensor[e].clone()
        else:
            flat[key] = tensor.clone()
    names = sorted(flat)
    weight_map, total = {}, 0
    for index in range(shards):
        chunk = {k: flat[k] for k in names[index::shards]}
        fname = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        save_file(chunk, str(directory / fname), metadata={"format": "pt"})
        for k, v in chunk.items():
            weight_map[k] = fname
            total += v.numel() * v.element_size()
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}))
    model.config.save_pretrained(directory)
    return directory
