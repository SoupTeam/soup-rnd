import json
import tempfile
from pathlib import Path

import torch
from transformers import MixtralConfig, MixtralForCausalLM

from soup_cli.utils.frozen_prefix_cache import FrozenPrefixCache
from soup_cli.utils.frozen_prefix_forward import install_frozen_prefix_cache
from soup_cli.utils.moe import detect_moe_model

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

print("=== SMALL MOE ARCHITECTURE ===")
print("Model type:", config.model_type)
print("Layers:", config.num_hidden_layers)
print("Experts:", config.num_local_experts)
print("Experts per token:", config.num_experts_per_tok)
print("Detected MoE:", detect_moe_model(model))

assert detect_moe_model(model)

input_ids = torch.randint(
    0, config.vocab_size, (1, 16)
)
labels = input_ids.clone()

print("\n=== FORWARD / BACKWARD ===")

output = model(
    input_ids=input_ids,
    labels=labels,
    use_cache=False,
)

assert torch.isfinite(output.loss)

output.loss.backward()

router_grads = [
    p.grad for n, p in model.named_parameters()
    if "gate" in n and p.grad is not None
]

expert_grads = [
    p.grad for n, p in model.named_parameters()
    if "experts" in n and p.grad is not None
]

print("Loss:", output.loss.item())
print("Router gradients:", len(router_grads))
print("Expert gradients:", len(expert_grads))

assert expert_grads, "No expert gradients detected"

print("\n=== E2 COMPATIBILITY ===")

# Current E2 only supports MistralModel.
with tempfile.TemporaryDirectory() as directory:
    cache = FrozenPrefixCache(directory)

    try:
        with install_frozen_prefix_cache(
            model=model,
            cache=cache,
            cutoff=2,
            metadata_factory=lambda **kwargs: {},
        ):
            raise RuntimeError(
                "Unexpectedly accepted Mixtral"
            )
    except ValueError as exc:
        message = str(exc)
        print("E2 rejected Mixtral:", message)

        assert "MistralModel only" in message

summary = {
    "kind": "small_moe_exploratory_smoke",
    "architecture": "mixtral",
    "random_initialization": True,
    "layers": config.num_hidden_layers,
    "experts": config.num_local_experts,
    "experts_per_token": config.num_experts_per_tok,
    "forward_backward_passed": True,
    "expert_gradients_present": True,
    "router_gradients_present": bool(router_grads),
    "e2_currently_unsupported": True,
    "e2_cache_equivalence_tested": False,
    "quality_benchmark": False,
    "speed_benchmark": False,
}

destination = Path(
    "benchmarks/e2/moe/small_moe_smoke_result.json"
)

destination.write_text(
    json.dumps(summary, indent=2) + "\n"
)

print("\n=== RESULT ===")
print(json.dumps(summary, indent=2))
print("Saved:", destination)
