
import torch
from transformers import MistralConfig, MistralModel
from transformers.masking_utils import create_causal_mask


def test_cached_forward_skips_frozen_prefix():
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

    model = MistralModel(config)
    model.eval()
    model.requires_grad_(False)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    position_ids = torch.arange(4).unsqueeze(0)

    with torch.no_grad():
        normal = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

    # Prepare the same embeddings, mask and rotary positions.
    hidden = model.embed_tokens(input_ids)

    causal_mask = create_causal_mask(
        config=model.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    position_embeddings = model.rotary_emb(
        hidden,
        position_ids=position_ids,
    )

    layer_kwargs = dict(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )

    # Epoch 1: compute the frozen prefix once.
    with torch.no_grad():
        for layer in model.layers[:2]:
            hidden = layer(hidden, **layer_kwargs)

        cached_activation = hidden.detach().clone()

    # Epoch 2: use cached activation and skip layers 0-1.
    calls = []

    def forbidden_forward(module, inputs):
        calls.append(True)
        raise AssertionError("Frozen prefix was executed!")

    handles = [
        layer.register_forward_pre_hook(forbidden_forward)
        for layer in model.layers[:2]
    ]

    try:
        with torch.no_grad():
            hidden = cached_activation.clone()

            for layer in model.layers[2:]:
                hidden = layer(hidden, **layer_kwargs)

            cached_output = model.norm(hidden)
    finally:
        for handle in handles:
            handle.remove()

    assert not calls
    torch.testing.assert_close(
        normal,
        cached_output,
        rtol=0,
        atol=0,
    )

