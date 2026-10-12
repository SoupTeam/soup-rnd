"""Research-only Mixtral frozen-prefix skipping proof of concept.

No production E2 changes. No pretrained checkpoint or speed claims.
"""

import json
import tempfile
import types
from pathlib import Path

import torch
from transformers import MixtralConfig, MixtralForCausalLM
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import MoeModelOutputWithPast

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
    fingerprint_frozen_prefix,
)

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
model.train()
decoder = model.model
cutoff = 2

for p in decoder.embed_tokens.parameters():
    p.requires_grad_(False)
for layer in decoder.layers[:cutoff]:
    for p in layer.parameters():
        p.requires_grad_(False)

ids = torch.randint(0, cfg.vocab_size, (1, 16))
mask = torch.ones_like(ids)
labels = ids.clone()


def run():
    model.zero_grad(set_to_none=True)
    result = model(
        input_ids=ids,
        attention_mask=mask,
        labels=labels,
        use_cache=False,
    )
    result.loss.backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    return result.logits.detach().clone(), result.loss.item(), grads


baseline_logits, baseline_loss, baseline_grads = run()

with tempfile.TemporaryDirectory() as directory:
    cache = FrozenPrefixCache(directory)

    state = {
        "hits": 0,
        "misses": 0,
        "frozen_layer_calls": 0,
        "skipped_layers": 0,
    }

    original_forward = decoder.forward

    prefix_fingerprint = fingerprint_frozen_prefix(decoder, cutoff)

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
            raise ValueError("PoC requires input_ids only")
        if use_cache or past_key_values is not None:
            raise ValueError("KV cache unsupported in PoC")
        output_router_logits = kwargs.pop("output_router_logits", None)

        if output_router_logits not in (None, False):
            raise ValueError("Router auxiliary logits are not supported by this PoC")

        if kwargs:
            raise ValueError(f"Unsupported kwargs: {sorted(kwargs)}")

        embeddings = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                embeddings.shape[1],
                device=embeddings.device,
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

        position_embeddings = self.rotary_emb(embeddings, position_ids=position_ids)

        metadata = build_cache_metadata(
            model_revision="random-mixtral-seed42",
            frozen_prefix_fingerprint=prefix_fingerprint,
            cutoff=cutoff,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            config_fingerprint="mixtral-4layers-cutoff2-v1",
        )

        hidden = cache.load(metadata)

        if hidden is None:
            state["misses"] += 1
            hidden = embeddings

            with torch.no_grad():
                for layer in self.layers[:cutoff]:
                    state["frozen_layer_calls"] += 1
                    hidden = layer(
                        hidden,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                    )

            cache.save(hidden, metadata)
        else:
            state["hits"] += 1
            state["skipped_layers"] += cutoff
            hidden = hidden.to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )

        for layer in self.layers[cutoff:]:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )

        hidden = self.norm(hidden)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=None,
        )

    decoder.forward = types.MethodType(cached_forward, decoder)

    try:
        first_logits, first_loss, first_grads = run()
        after_first = dict(state)

        second_logits, second_loss, second_grads = run()
        after_second = dict(state)
    finally:
        decoder.forward = original_forward

    exact_logits = torch.equal(baseline_logits, first_logits) and torch.equal(
        baseline_logits, second_logits
    )
    exact_loss = baseline_loss == first_loss == second_loss
    exact_grads = baseline_grads.keys() == first_grads.keys() == second_grads.keys() and all(
        torch.equal(baseline_grads[k], first_grads[k])
        and torch.equal(baseline_grads[k], second_grads[k])
        for k in baseline_grads
    )

    assert exact_logits, "Logits mismatch"
    assert exact_loss, "Loss mismatch"
    assert exact_grads, "Gradients mismatch"

    assert after_first["misses"] == 1
    assert after_first["frozen_layer_calls"] == cutoff

    assert after_second["hits"] == 1
    assert after_second["frozen_layer_calls"] == cutoff
    assert after_second["skipped_layers"] == cutoff

    summary = {
        "kind": "small_moe_real_prefix_skip_poc",
        "architecture": "mixtral",
        "random_initialization": True,
        "layers": 4,
        "experts": 4,
        "cutoff": cutoff,
        "cache_hits": state["hits"],
        "cache_misses": state["misses"],
        "frozen_layer_calls_total": state["frozen_layer_calls"],
        "frozen_layers_skipped_on_hit": cutoff,
        "exact_logits": exact_logits,
        "exact_loss": exact_loss,
        "exact_trainable_gradients": exact_grads,
        "production_e2_supported": False,
        "performance_benchmark": False,
    }

    output = Path("benchmarks/e2/moe/small_moe_real_skip_result.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== SMALL MOE REAL PREFIX SKIP ===")
    print(json.dumps(summary, indent=2))
    print("Saved:", output)
