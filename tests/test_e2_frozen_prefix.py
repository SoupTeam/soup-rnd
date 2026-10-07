
import torch
from transformers import MistralConfig, MistralModel


def test_frozen_prefix_determinism():
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

    for param in model.parameters():
        param.requires_grad_(False)

    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        output1 = model(input_ids=input_ids).last_hidden_state
        output2 = model(input_ids=input_ids).last_hidden_state

    assert torch.equal(output1, output2)

def test_frozen_prefix_boundary():
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
    captured = []

    def capture_boundary(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach().clone())

    handle = model.layers[1].register_forward_hook(capture_boundary)

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
            model(input_ids=input_ids)
    finally:
        handle.remove()

    assert len(captured) == 2
    assert torch.equal(captured[0], captured[1])
    assert captured[0].shape == (1, 4, 64)


def test_cached_boundary_output_matches_normal():
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

    boundary = []

    def capture(module, inputs, output):
        boundary.append(output.detach().clone())

    handle = model.layers[1].register_forward_hook(capture)

    try:
        with torch.no_grad():
            normal = model(input_ids=input_ids).last_hidden_state
    finally:
        handle.remove()

    cached_activation = boundary[0]

    def replace_boundary(module, inputs, output):
        return cached_activation.clone()

    handle = model.layers[1].register_forward_hook(replace_boundary)

    try:
        with torch.no_grad():
            cached = model(input_ids=input_ids).last_hidden_state
    finally:
        handle.remove()

    assert torch.equal(normal, cached)
