"""Exploratory Mixtral frozen-prefix cache proof of concept.

Not a production E2 implementation or performance benchmark.
"""

import json
import tempfile
from pathlib import Path

import torch
from transformers import MixtralConfig, MixtralForCausalLM

from soup_cli.utils.frozen_prefix_cache import FrozenPrefixCache

torch.manual_seed(42)

config = MixtralConfig(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_local_experts=4,
    num_experts_per_tok=2,
    max_position_embeddings=128,
    attention_dropout=0.0,
    router_aux_loss_coef=0.0,
)

model = MixtralForCausalLM(config)
model.train()

decoder = model.model
cutoff = 2

assert len(decoder.layers) == 4

# Freeze embeddings and the first two decoder layers.
for parameter in decoder.embed_tokens.parameters():
    parameter.requires_grad_(False)

for layer in decoder.layers[:cutoff]:
    for parameter in layer.parameters():
        parameter.requires_grad_(False)

# This PoC assumes no active dropout in the frozen prefix.
for module in decoder.layers[:cutoff].modules():
    if isinstance(module, torch.nn.Dropout):
        assert module.p == 0.0

input_ids = torch.randint(0, 256, (1, 16))
labels = input_ids.clone()


def gradients():
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def run():
    model.zero_grad(set_to_none=True)
    result = model(
        input_ids=input_ids,
        labels=labels,
        use_cache=False,
    )
    result.loss.backward()
    return (
        result.logits.detach().clone(),
        result.loss.item(),
        gradients(),
    )


# Ordinary forward/backward.
baseline_logits, baseline_loss, baseline_grads = run()

with tempfile.TemporaryDirectory() as directory:
    cache = FrozenPrefixCache(directory)

    # Cache only the output of the final frozen layer.
    # Earlier frozen layers still execute in this prototype.
    metadata = {
        "experiment": "small-moe-hook-poc",
        "seed": 42,
        "cutoff": cutoff,
        "input_ids": input_ids.tolist(),
    }

    state = {"hits": 0, "misses": 0}

    def hook(module, args, output):
        saved = cache.load(metadata)

        if saved is None:
            state["misses"] += 1
            cache.save(output, metadata)
            return output

        state["hits"] += 1
        return saved.to(
            device=output.device,
            dtype=output.dtype,
        )

    handle = decoder.layers[cutoff - 1].register_forward_hook(hook)

    try:
        # First run populates cache.
        first_logits, first_loss, first_grads = run()

        # Second run reuses cached activation.
        cached_logits, cached_loss, cached_grads = run()
    finally:
        handle.remove()

    assert state == {"hits": 1, "misses": 1}

    assert torch.equal(baseline_logits, first_logits)
    assert torch.equal(baseline_logits, cached_logits)

    assert baseline_loss == first_loss == cached_loss

    assert baseline_grads.keys() == cached_grads.keys()

    exact_gradients = all(
        torch.equal(baseline_grads[name], cached_grads[name]) for name in baseline_grads
    )

    assert exact_gradients

    summary = {
        "kind": "small_moe_hook_cache_poc",
        "architecture": "mixtral",
        "random_initialization": True,
        "layers": 4,
        "experts": 4,
        "frozen_layers": 2,
        "cache_hits": state["hits"],
        "cache_misses": state["misses"],
        "exact_logits": True,
        "exact_loss": True,
        "exact_gradients": exact_gradients,
        "frozen_prefix_execution_skipped": False,
        "production_e2_supported": False,
        "performance_benchmark": False,
    }

    destination = Path("benchmarks/e2/moe/small_moe_cache_poc_result.json")

    destination.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== SMALL MOE CACHE POC ===")
    print(json.dumps(summary, indent=2))
    print("Saved:", destination)
