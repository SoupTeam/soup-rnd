"""Small Mixtral E2 exploratory layer-skipping proof of concept."""

import json
import torch
from pathlib import Path
from transformers import MixtralConfig, MixtralForCausalLM

torch.manual_seed(42)

cfg = MixtralConfig(
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

model = MixtralForCausalLM(cfg)
model.eval()

decoder = model.model
cutoff = 2

for p in decoder.embed_tokens.parameters():
    p.requires_grad_(False)

for layer in decoder.layers[:cutoff]:
    for p in layer.parameters():
        p.requires_grad_(False)

ids = torch.randint(0, cfg.vocab_size, (1, 16))
labels = ids.clone()

# Capture the hidden state entering the first trainable layer.
captured = {}

def capture(module, args):
    captured["hidden"] = args[0].detach().clone()

handle = decoder.layers[cutoff].register_forward_pre_hook(capture)

with torch.no_grad():
    original = model(input_ids=ids, labels=labels, use_cache=False)

handle.remove()

assert "hidden" in captured

# Count frozen-prefix calls on a subsequent normal forward.
counts = {"frozen_calls": 0}

def count_calls(module, args):
    counts["frozen_calls"] += 1

handles = [
    layer.register_forward_pre_hook(count_calls)
    for layer in decoder.layers[:cutoff]
]

with torch.no_grad():
    repeat = model(input_ids=ids, labels=labels, use_cache=False)

for h in handles:
    h.remove()

assert torch.equal(original.logits, repeat.logits)

summary = {
    "kind": "small_moe_prefix_boundary_probe",
    "architecture": "mixtral",
    "layers": 4,
    "cutoff": cutoff,
    "captured_boundary_shape": list(captured["hidden"].shape),
    "baseline_repeat_exact": torch.equal(
        original.logits, repeat.logits
    ),
    "frozen_calls_on_normal_repeat": counts["frozen_calls"],
    "actual_prefix_skip_implemented": False,
    "production_e2_supported": False,
}

out = Path("benchmarks/e2/moe/small_moe_skip_probe.json")
out.write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps(summary, indent=2))
