"""Research-only Small Mixtral multi-step training equivalence.

Five optimizer steps on identical fixed inputs.
No production E2 changes, no speed or generalization claims.
"""
import copy
import json
import tempfile
import types
from pathlib import Path

import torch
from transformers import MixtralConfig, MixtralForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import MoeModelOutputWithPast

torch.manual_seed(42)
torch.set_num_threads(2)

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

base_model = MixtralForCausalLM(cfg)

lora_cfg = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[2, 3],
    layers_pattern="layers",
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

baseline = get_peft_model(base_model, lora_cfg)
cached = copy.deepcopy(baseline)
cutoff = 2

for model in (baseline, cached):
    model.train()
    decoder = model.base_model.model.model
    for p in decoder.embed_tokens.parameters():
        p.requires_grad_(False)
    for layer in decoder.layers[:cutoff]:
        for p in layer.parameters():
            p.requires_grad_(False)

assert all(
    torch.equal(a, b)
    for a, b in zip(baseline.parameters(), cached.parameters())
)

batches = {
    name: torch.randint(0, cfg.vocab_size, (1, 16))
    for name in ("A", "B", "C")
}

assert len({
    tuple(t.flatten().tolist())
    for t in batches.values()
}) == 3

ids = batches["A"]
mask = torch.ones_like(ids)
labels = ids.clone()

optimizer_a = torch.optim.AdamW(
    (p for p in baseline.parameters() if p.requires_grad),
    lr=1e-4,
)
optimizer_b = torch.optim.AdamW(
    (p for p in cached.parameters() if p.requires_grad),
    lr=1e-4,
)

state = {
    "hits": 0,
    "misses": 0,
    "frozen_calls": 0,
    "skipped_layers": 0,
}

# In-memory cache is sufficient to test training equivalence.
# It is not a persistence or crash-recovery test.
boundary_cache = {}
decoder = cached.base_model.model.model
original_forward = decoder.forward

def cached_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=False,
    **kwargs,
):
    if input_ids is None or inputs_embeds is not None:
        raise ValueError("Only input_ids supported")
    if use_cache or past_key_values is not None:
        raise ValueError("KV cache unsupported")

    router_logits = kwargs.pop("output_router_logits", None)
    output_attentions = kwargs.pop("output_attentions", None)
    output_hidden_states = kwargs.pop("output_hidden_states", None)

    if router_logits not in (None, False):
        raise ValueError("Router auxiliary logits unsupported")

    if output_attentions not in (None, False):
        raise ValueError("Attention outputs unsupported in this PoC")

    if output_hidden_states not in (None, False):
        raise ValueError("Hidden-state outputs unsupported in this PoC")

    if kwargs:
        raise ValueError(f"Unsupported kwargs: {sorted(kwargs)}")

    embeddings = self.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(
            embeddings.shape[1], device=embeddings.device
        ).unsqueeze(0)

    mask_fn = (
        create_causal_mask
        if self.config.sliding_window is None
        else create_sliding_window_causal_mask
    )
    causal_mask = mask_fn(
        config=self.config,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    rotary = self.rotary_emb(
        embeddings, position_ids=position_ids
    )

    # This experiment uses a single fixed batch and immutable prefix.
    key = (
        tuple(input_ids.flatten().tolist()),
        tuple(attention_mask.flatten().tolist())
        if attention_mask is not None else None,
        tuple(position_ids.flatten().tolist()),
    )

    hidden = boundary_cache.get(key)

    if hidden is None:
        state["misses"] += 1
        hidden = embeddings
        with torch.no_grad():
            for layer in self.layers[:cutoff]:
                state["frozen_calls"] += 1
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=rotary,
                )
        boundary_cache[key] = hidden.detach().clone()
    else:
        state["hits"] += 1
        state["skipped_layers"] += cutoff
        hidden = hidden.clone()

    for layer in self.layers[cutoff:]:
        hidden = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=rotary,
        )

    return MoeModelOutputWithPast(
        last_hidden_state=self.norm(hidden),
        past_key_values=None,
    )

decoder.forward = types.MethodType(cached_forward, decoder)

history = []

try:
    schedule = ("A", "B", "C", "A", "B", "C")

    for step, batch_name in enumerate(schedule, 1):
        ids = batches[batch_name]
        labels = ids.clone()
        mask = torch.ones_like(ids)

        losses = []
        gradients = []

        for model, optimizer in (
            (baseline, optimizer_a),
            (cached, optimizer_b),
        ):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=ids,
                attention_mask=mask,
                labels=labels,
                use_cache=False,
            )
            output.loss.backward()

            losses.append(output.loss.detach().item())
            gradients.append({
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.requires_grad and p.grad is not None
            })

        loss_equal = losses[0] == losses[1]
        grad_equal = (
            gradients[0].keys() == gradients[1].keys()
            and all(
                torch.equal(gradients[0][k], gradients[1][k])
                for k in gradients[0]
            )
        )

        optimizer_a.step()
        optimizer_b.step()

        weights_equal = all(
            torch.equal(a, b)
            for a, b in zip(
                baseline.parameters(), cached.parameters()
            )
        )

        row = {
            "step": step,
            "batch": batch_name,
            "baseline_loss": losses[0],
            "cached_loss": losses[1],
            "loss_equal": loss_equal,
            "gradients_equal": grad_equal,
            "weights_equal": weights_equal,
        }
        history.append(row)

        print(
            f"Step {step}: "
            f"loss={losses[0]:.8f} "
            f"loss_equal={loss_equal} "
            f"grad_equal={grad_equal} "
            f"weights_equal={weights_equal}",
            flush=True,
        )

        if not (loss_equal and grad_equal and weights_equal):
            raise RuntimeError(
                f"Training divergence at step {step}"
            )
finally:
    decoder.forward = original_forward

assert state["misses"] == 3
assert state["hits"] == 3
assert state["frozen_calls"] == cutoff * 3
assert state["skipped_layers"] == cutoff * 3

summary = {
    "kind": "small_moe_lora_multibatch_equivalence",
    "architecture": "random_small_mixtral",
    "steps": 6,
    "optimizer": "AdamW",
    "fixed_input": False,
    "unique_batches": 3,
    "trainable_scope": "top_k_lora_q_v_layers_2_3",
    "exact_training_equivalence": True,
    "cache_hits": state["hits"],
    "cache_misses": state["misses"],
    "frozen_layer_calls": state["frozen_calls"],
    "skipped_layer_calls": state["skipped_layers"],
    "history": history,
    "production_e2_supported": False,
    "performance_benchmark": False,
}

path = Path(
    "benchmarks/e2/moe/small_moe_lora_equivalence_result.json"
)
path.write_text(json.dumps(summary, indent=2) + "\n")

print("\n=== SMALL MOE MULTIBATCH EQUIVALENCE ===")
print(json.dumps(summary, indent=2))
print("Saved:", path)
