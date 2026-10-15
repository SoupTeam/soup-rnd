"""Zero-copy frozen-prefix fingerprint compatibility."""

import hashlib

import pytest
import torch
from transformers import MistralConfig, MistralModel

from soup_cli.utils.frozen_prefix_cache import (
    fingerprint_frozen_prefix,
    fingerprint_prefix_configuration,
)


def legacy_fingerprint(decoder, cutoff):
    """Reference implementation using the original tobytes path."""
    digest = hashlib.sha256()

    digest.update(b"e2-frozen-prefix-v2")
    digest.update(
        fingerprint_prefix_configuration(
            decoder, cutoff
        ).encode("ascii")
    )

    modules = [
        ("embed_tokens", decoder.embed_tokens),
        *[
            (f"layers.{i}", decoder.layers[i])
            for i in range(cutoff)
        ],
    ]

    with torch.no_grad():
        for prefix, module in modules:
            for name, tensor in sorted(
                module.state_dict().items()
            ):
                value = tensor.detach().cpu().contiguous()

                digest.update(
                    f"{prefix}.{name}".encode()
                )
                digest.update(
                    str(value.dtype).encode()
                )
                digest.update(
                    str(tuple(value.shape)).encode()
                )
                digest.update(
                    value.view(torch.uint8)
                    .numpy().tobytes()
                )

    return digest.hexdigest()


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
)
def test_zero_copy_matches_legacy(dtype):
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

    model = MistralModel(config).eval()
    model = model.to(dtype=dtype)

    cutoff = 2

    original = legacy_fingerprint(model, cutoff)
    optimized = fingerprint_frozen_prefix(
        model, cutoff
    )

    assert optimized == original

    # Changing an upper layer must not affect the key.
    with torch.no_grad():
        next(model.layers[3].parameters()).add_(0.01)

    assert fingerprint_frozen_prefix(
        model, cutoff
    ) == original

    # Changing a frozen layer must invalidate it.
    with torch.no_grad():
        next(model.layers[0].parameters()).add_(0.01)

    changed = fingerprint_frozen_prefix(
        model, cutoff
    )

    assert changed != original
    assert changed == legacy_fingerprint(
        model, cutoff
    )
