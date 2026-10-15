#!/usr/bin/env python3
"""Research-only fast fingerprint guard for the SHA-verified OLMoE v1 cache.

This is deliberately NOT a production-safe, general-purpose invalidation scheme.

The v1 cache hashes EVERY frozen tensor on EVERY batch, which costs ~10 seconds
per lookup on pinned pretrained OLMoE K8. This wrapper computes the original
cryptographic fingerprint at first access, then monitors tensor identity and
PyTorch's in-place mutation counter (Tensor._version) on later accesses.
A change forces a fresh *full* fingerprint and therefore a cache miss.

IMPORTANT CONTRACT:
 - Frozen prefix is immutable during ordinary LoRA training.
 - All supported mutations use normal PyTorch in-place methods, replacing
   registered tensors, or call mark_prefix_changed() explicitly.
 - DO NOT modify frozen tensors through `.data.add_()`, raw storage writes,
   custom CUDA kernels, or other mutation paths bypassing Tensor._version
   WITHOUT calling mark_prefix_changed() BEFORE the next forward.
 - No concurrent prefix mutation/forward; no cross-process cache writers.
 - K8 research-only; no production integration, no performance acceptance.

Usage:
  FastCache = guarded_cache_type(probe.ResearchOlmoeCache)
  with FastCache(base_model, cache_dir, cutoff=8) as cache: ...
"""
from __future__ import annotations

import time
from typing import Any

import torch

from soup_cli.utils.frozen_prefix_cache import (
    build_cache_metadata,
    fingerprint_frozen_prefix,
    fingerprint_prefix_configuration,
)


def guarded_cache_type(base_cache_class: type) -> type:
    """Return subclass preserving v1 disk format, router logic and safety checks."""

    class GuardedResearchOlmoeCache(base_cache_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._prefix_signature = None
            self._config_fingerprint = None
            self._full_fingerprint = None
            self.fingerprint_recomputes = 0
            self.fingerprint_seconds = 0.0
            self.metadata_calls = 0
            self.guard_seconds = 0.0
            self.metadata_seconds = 0.0
            self.explicit_invalidations = 0

            # Bind the model revision explicitly to avoid importing a second
            # unverified copy of v1 and silently using its default revision.
            self._research_model_revision = None

            # Fail closed if frozen state_dict has entries not monitored by the
            # registered parameters/buffers. This is a one-time scan.
            for module in self._frozen_modules():
                watched = set(dict(module.named_parameters(recurse=True)))
                watched |= set(dict(module.named_buffers(recurse=True)))
                untracked = set(module.state_dict()) - watched
                if untracked:
                    raise RuntimeError(
                        f"Untracked frozen prefix state_dict entries: {sorted(untracked)[:5]}"
                    )

        def _frozen_modules(self):
            return [self.decoder.embed_tokens, *self.decoder.layers[: self.cutoff]]

        def set_model_revision(self, revision: str) -> None:
            """Set pinned checkpoint identity before the first cache forward."""
            if self.metadata_calls or self._full_fingerprint is not None:
                raise RuntimeError("Cannot change model revision after cache use")
            if not isinstance(revision, str) or "@" not in revision:
                raise ValueError("A pinned model_id@revision is required")
            self._research_model_revision = revision

        def mark_prefix_changed(self) -> None:
            """MANDATORY before next lookup for any mutation bypassing _version.

            It is okay to call this for ordinary in-place mutations too.
            The old disk file remains, but the cryptographic key changes if
            the underlying frozen weights/configuration really changed.
            """
            self._prefix_signature = None
            self._full_fingerprint = None
            self._config_fingerprint = None
            self.explicit_invalidations += 1

        def _signature(self) -> tuple:
            items = []
            for module_index, module in enumerate(self._frozen_modules()):
                for kind, pairs in (
                    ("p", module.named_parameters(recurse=True)),
                    ("b", module.named_buffers(recurse=True)),
                ):
                    for name, tensor in pairs:
                        if tensor.is_meta:
                            raise RuntimeError("Meta frozen tensor is unsupported")
                        if kind == "p" and tensor.requires_grad:
                            raise RuntimeError("Frozen-prefix parameter became trainable")
                        # Version counters catch ordinary PyTorch in-place edits.
                        # identity/address/shape/dtype catch parameter/buffer replacement.
                        items.append((
                            module_index,
                            kind,
                            name,
                            id(tensor),
                            tensor._version,
                            tensor.data_ptr(),
                            tuple(tensor.shape),
                            tuple(tensor.stride()),
                            tensor.storage_offset(),
                            str(tensor.dtype),
                            str(tensor.device),
                        ))
            return tuple(items)

        def _metadata(self, input_ids, attention_mask, position_ids) -> dict:
            started = time.perf_counter()
            if not self._research_model_revision:
                raise RuntimeError("Pinned model revision not set")
            with torch.no_grad():
                guard_start = time.perf_counter()
                signature = self._signature()
                # Includes actual effective RoPE state and decoder config.
                # Profiling showed this costs ~0.0004 s on pretrained OLMoE.
                config_hash = fingerprint_prefix_configuration(self.decoder, self.cutoff)
                self.guard_seconds += time.perf_counter() - guard_start

                if (
                    signature != self._prefix_signature
                    or config_hash != self._config_fingerprint
                    or self._full_fingerprint is None
                ):
                    hash_start = time.perf_counter()
                    full_hash = fingerprint_frozen_prefix(self.decoder, self.cutoff)
                    self.fingerprint_seconds += time.perf_counter() - hash_start
                    # Do not accept a concurrent mutation during fingerprinting.
                    if self._signature() != signature:
                        raise RuntimeError("Frozen prefix mutated during fingerprinting")
                    if fingerprint_prefix_configuration(self.decoder, self.cutoff) != config_hash:
                        raise RuntimeError("OLMoE config/RoPE mutated during fingerprinting")
                    self._full_fingerprint = full_hash
                    self._prefix_signature = signature
                    self._config_fingerprint = config_hash
                    self.fingerprint_recomputes += 1

                metadata = build_cache_metadata(
                    model_revision=self._research_model_revision,
                    frozen_prefix_fingerprint=self._full_fingerprint,
                    config_fingerprint=config_hash,
                    cutoff=self.cutoff,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            self.metadata_calls += 1
            self.metadata_seconds += time.perf_counter() - started
            return metadata

    GuardedResearchOlmoeCache.__name__ = "GuardedResearchOlmoeCache"
    return GuardedResearchOlmoeCache
