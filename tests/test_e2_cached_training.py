
import copy
import re

import torch
from peft import get_peft_model
from transformers import MistralConfig, MistralForCausalLM
from transformers.masking_utils import create_causal_mask

from soup_cli.config.schema import LoraConfig
from soup_cli.utils.peft_wiring import (
    build_lora_config,
    resolve_top_k_layers,
)


def test_cached_training_matches_normal():
    torch.manual_seed(42)

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
    cfg = LoraConfig(
        r=4,
        alpha=8,
        dropout=0.0,
        top_k_layers=2,
        target_modules=["q_proj", "v_proj"],
    )

    selected = resolve_top_k_layers(base, 2)
    peft_cfg = build_lora_config(
        cfg,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        layers_to_transform=selected,
    )

    normal_model = get_peft_model(base, peft_cfg)
    cached_model = copy.deepcopy(normal_model)

    normal_model.train()
    cached_model.train()

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = input_ids.clone()

    # Normal training
    normal_model.zero_grad(set_to_none=True)
    normal_loss = normal_model(
        input_ids=input_ids,
        labels=labels,
        use_cache=False,
    ).loss
    normal_loss.backward()

    # Cached training: skip frozen prefix
    cached_model.zero_grad(set_to_none=True)
    decoder = cached_model.base_model.model.model

    position_ids = torch.arange(
        input_ids.shape[1]
    ).unsqueeze(0)

    with torch.no_grad():
        hidden = decoder.embed_tokens(input_ids)

        causal_mask = create_causal_mask(
            config=decoder.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = decoder.rotary_emb(
            hidden, position_ids=position_ids
        )

        kwargs = dict(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )

        for layer in decoder.layers[:2]:
            hidden = layer(hidden, **kwargs)

        cached_activation = hidden.detach().clone()

    # Make sure frozen layers are not executed again
    def forbidden_forward(module, inputs):
        raise AssertionError("Frozen prefix executed!")

    handles = [
        layer.register_forward_pre_hook(forbidden_forward)
        for layer in decoder.layers[:2]
    ]

    try:
        hidden = cached_activation

        for layer in decoder.layers[2:]:
            hidden = layer(hidden, **kwargs)

        hidden = decoder.norm(hidden)
        logits = cached_model.base_model.model.lm_head(hidden)

        # Same next-token loss convention as causal LM
        from torch.nn import functional

        cached_loss = functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, config.vocab_size),
            labels[:, 1:].contiguous().view(-1),
        )
        cached_loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(
        normal_loss, cached_loss, rtol=1e-5, atol=1e-6
    )

    normal_grads = {
        n: p.grad
        for n, p in normal_model.named_parameters()
        if p.requires_grad
    }
    cached_grads = {
        n: p.grad
        for n, p in cached_model.named_parameters()
        if p.requires_grad
    }

    assert normal_grads.keys() == cached_grads.keys()
    assert normal_grads

    for name in normal_grads:
        assert normal_grads[name] is not None, name
        assert cached_grads[name] is not None, name
        assert re.search(r"\.layers\.(2|3)\.", name), name

        torch.testing.assert_close(
            normal_grads[name],
            cached_grads[name],
            rtol=1e-4,
            atol=1e-6,
        )

