
"""E2: two-epoch training with a persistent frozen-prefix cache."""

import copy
import re

import torch
from peft import get_peft_model
from torch.nn import functional
from transformers import MistralConfig, MistralForCausalLM
from transformers.masking_utils import create_causal_mask

from soup_cli.config.schema import LoraConfig
from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
)
from soup_cli.utils.peft_wiring import (
    build_lora_config,
    resolve_top_k_layers,
)


def make_model():
    config = MistralConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )

    base = MistralForCausalLM(config)

    lora = LoraConfig(
        r=4,
        alpha=8,
        dropout=0.0,
        top_k_layers=2,
        target_modules=["q_proj", "v_proj"],
    )

    selected = resolve_top_k_layers(base, 2)

    peft_config = build_lora_config(
        lora,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        layers_to_transform=selected,
    )

    return get_peft_model(base, peft_config)


def cached_forward_loss(model, input_ids, cache, metadata):
    decoder = model.base_model.model.model
    sequence_length = input_ids.shape[1]

    position_ids = torch.arange(
        sequence_length,
        device=input_ids.device,
    ).unsqueeze(0)

    with torch.no_grad():
        embeddings = decoder.embed_tokens(input_ids)

        causal_mask = create_causal_mask(
            config=decoder.config,
            inputs_embeds=embeddings,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = decoder.rotary_emb(
            embeddings,
            position_ids=position_ids,
        )

    kwargs = {
        "attention_mask": causal_mask,
        "position_ids": position_ids,
        "past_key_values": None,
        "use_cache": False,
        "position_embeddings": position_embeddings,
    }

    hidden = cache.load(metadata)
    cache_hit = hidden is not None

    if not cache_hit:
        hidden = embeddings
        with torch.no_grad():
            for layer in decoder.layers[:2]:
                hidden = layer(hidden, **kwargs)

        cache.save(hidden, metadata)
    else:
        hidden = hidden.to(
            device=input_ids.device,
            dtype=embeddings.dtype,
        )

    for layer in decoder.layers[2:]:
        hidden = layer(hidden, **kwargs)

    hidden = decoder.norm(hidden)
    logits = model.base_model.model.lm_head(hidden)

    loss = functional.cross_entropy(
        logits[:, :-1, :].contiguous().reshape(
            -1, logits.shape[-1]
        ),
        input_ids[:, 1:].contiguous().reshape(-1),
    )

    return loss, cache_hit


def test_two_epoch_cached_training(tmp_path):
    torch.manual_seed(42)

    normal_model = make_model()
    cached_model = copy.deepcopy(normal_model)

    normal_model.train()
    cached_model.train()

    normal_optimizer = torch.optim.SGD(
        (p for p in normal_model.parameters() if p.requires_grad),
        lr=0.01,
    )
    cached_optimizer = torch.optim.SGD(
        (p for p in cached_model.parameters() if p.requires_grad),
        lr=0.01,
    )

    cache = FrozenPrefixCache(tmp_path / "cache")
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    metadata = build_cache_metadata(
        model_revision="tiny-mistral-test-v1",
        frozen_prefix_fingerprint="fixed-frozen-weights-v1",
        config_fingerprint="dropout0-no-packing-v1",
        cutoff=2,
        input_ids=input_ids,
        position_ids=torch.arange(5).unsqueeze(0),
    )

    cache_hits = []
    normal_losses = []
    cached_losses = []

    frozen_calls = [0]

    def count_frozen_forward(module, inputs):
        frozen_calls[0] += 1

    handles = [
        layer.register_forward_pre_hook(count_frozen_forward)
        for layer in cached_model.base_model.model.model.layers[:2]
    ]

    try:
        for epoch in range(2):
            normal_optimizer.zero_grad(set_to_none=True)
            cached_optimizer.zero_grad(set_to_none=True)

            normal_loss = normal_model(
                input_ids=input_ids,
                labels=input_ids,
                use_cache=False,
            ).loss

            cached_loss, hit = cached_forward_loss(
                cached_model,
                input_ids,
                cache,
                metadata,
            )

            torch.testing.assert_close(
                normal_loss,
                cached_loss,
                rtol=1e-5,
                atol=1e-6,
            )

            normal_loss.backward()
            cached_loss.backward()

            normal_grads = {
                name: p.grad
                for name, p in normal_model.named_parameters()
                if p.requires_grad
            }
            cached_grads = {
                name: p.grad
                for name, p in cached_model.named_parameters()
                if p.requires_grad
            }

            assert normal_grads.keys() == cached_grads.keys()
            assert normal_grads

            for name in normal_grads:
                assert re.search(r"\.layers\.(2|3)\.", name)
                assert normal_grads[name] is not None
                assert cached_grads[name] is not None

                torch.testing.assert_close(
                    normal_grads[name],
                    cached_grads[name],
                    rtol=1e-4,
                    atol=1e-6,
                )

            normal_optimizer.step()
            cached_optimizer.step()

            normal_losses.append(normal_loss.item())
            cached_losses.append(cached_loss.item())
            cache_hits.append(hit)

            print(
                f"Epoch {epoch + 1}: "
                f"normal={normal_loss.item():.6f}, "
                f"cached={cached_loss.item():.6f}, "
                f"cache_hit={hit}"
            )
    finally:
        for handle in handles:
            handle.remove()

    assert cache_hits == [False, True]

    # Two frozen layers execute only on the first epoch.
    assert frozen_calls[0] == 2

    normal_params = dict(normal_model.named_parameters())
    cached_params = dict(cached_model.named_parameters())

    for name, parameter in normal_params.items():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter,
                cached_params[name],
                rtol=1e-4,
                atol=1e-6,
            )

    assert len(normal_losses) == 2
    assert len(cached_losses) == 2

