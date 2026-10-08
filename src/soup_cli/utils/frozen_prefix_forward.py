
"""Frozen-prefix forward execution for E2 research.

Initial scope:
- Mistral decoder
- Resident model (no layer streaming)
- No KV cache
- Deterministic frozen prefix
"""
from __future__ import annotations

import contextlib
import types
from typing import Any

from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    fingerprint_frozen_prefix,
)


class FrozenPrefixRunner:
    """Execute Mistral with reusable frozen-prefix activations."""

    def __init__(
        self,
        decoder: Any,
        cache: FrozenPrefixCache,
        cutoff: int,
        metadata_factory: Any,
    ):
        if not 0 < cutoff < len(decoder.layers):
            raise ValueError("Invalid frozen-prefix cutoff")

        self.decoder = decoder
        self.cache = cache
        self.cutoff = cutoff
        self.metadata_factory = metadata_factory
        self.frozen_prefix_fingerprint = fingerprint_frozen_prefix(
            decoder,
            cutoff,
        )
        self._frozen_parameters = tuple(
            [
                *decoder.embed_tokens.parameters(),
                *(
                    parameter
                    for layer in decoder.layers[:cutoff]
                    for parameter in layer.parameters()
                ),
            ]
        )

        if any(p.requires_grad for p in self._frozen_parameters):
            raise ValueError("E2 requires a fully frozen prefix")

        self._frozen_versions = tuple(
            p._version for p in self._frozen_parameters
        )
        self.hits = 0
        self.misses = 0

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if past_key_values is not None or use_cache:
            raise ValueError("E2 does not support KV caching")

        if inputs_embeds is not None or input_ids is None:
            raise ValueError("E2 requires input_ids")

        kwargs.pop("num_items_in_batch", None)
        output_attentions = kwargs.pop("output_attentions", None)
        output_hidden_states = kwargs.pop("output_hidden_states", None)

        if output_attentions not in (None, False):
            raise ValueError("E2 does not support output_attentions=True")

        if output_hidden_states not in (None, False):
            raise ValueError("E2 does not support output_hidden_states=True")

        if kwargs:
            raise ValueError(
                f"E2 unsupported forward arguments: {sorted(kwargs)}"
            )

        decoder = self.decoder

        embeddings = decoder.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = (
                embeddings.new_zeros(
                    (1, embeddings.shape[1]),
                    dtype=input_ids.dtype,
                )
                + __import__("torch").arange(
                    embeddings.shape[1],
                    device=embeddings.device,
                ).unsqueeze(0)
            )

        mask_function = (
            create_causal_mask
            if decoder.config.sliding_window is None
            else create_sliding_window_causal_mask
        )

        causal_mask = mask_function(
            config=decoder.config,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = decoder.rotary_emb(
            embeddings,
            position_ids=position_ids,
        )

        metadata = self.metadata_factory(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        metadata = dict(metadata)
        metadata["frozen_prefix_fingerprint"] = (
            self.frozen_prefix_fingerprint
        )
        current_versions = tuple(
            p._version for p in self._frozen_parameters
        )

        if current_versions != self._frozen_versions:
            raise RuntimeError(
                "E2 frozen-prefix weights changed during training. "
                "Restart the runner to rebuild its cache fingerprint."
            )
        hidden = self.cache.load(metadata)

        if hidden is None:
            self.misses += 1
            hidden = embeddings

            import torch

            with torch.no_grad():
                for layer in decoder.layers[: self.cutoff]:
                    hidden = layer(
                        hidden,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                    )

            self.cache.save(hidden, metadata)

        else:
            self.hits += 1
            hidden = hidden.to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )

        for layer in decoder.layers[self.cutoff :]:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )

        hidden = decoder.norm(hidden)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=None,
        )


@contextlib.contextmanager
def install_frozen_prefix_cache(
    model: Any,
    cache: FrozenPrefixCache,
    cutoff: int,
    metadata_factory: Any,
):
    """Temporarily install E2 cached forward on a resident Mistral."""

    from transformers import MistralModel

    from soup_cli.utils.layer_stream_runtime import decoder_owner

    decoder = decoder_owner(model)

    if not isinstance(decoder, MistralModel):
        raise ValueError(
            "E2 frozen-prefix cache currently supports MistralModel only"
        )

    if not 0 < cutoff < len(decoder.layers):
        raise ValueError("E2 requires a nonempty frozen prefix and upper stack")

    if any(p.requires_grad for p in decoder.embed_tokens.parameters()):
        raise ValueError("E2 requires frozen input embeddings")

    for layer in decoder.layers[:cutoff]:
        if any(p.requires_grad for p in layer.parameters()):
            raise ValueError("E2 requires all prefix parameters frozen")

    runner = FrozenPrefixRunner(
        decoder=decoder,
        cache=cache,
        cutoff=cutoff,
        metadata_factory=metadata_factory,
    )

    original_forward = decoder.forward

    def cached_forward(self, *args, **kwargs):
        return runner.forward(*args, **kwargs)

    decoder.forward = types.MethodType(cached_forward, decoder)

    try:
        yield runner
    finally:
        decoder.forward = original_forward
