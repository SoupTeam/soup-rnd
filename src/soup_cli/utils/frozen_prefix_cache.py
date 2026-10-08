
"""Frozen-prefix activation cache for E2 experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def cache_key(metadata: dict[str, Any]) -> str:
    """Create a deterministic key from cache metadata."""
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_frozen_prefix(
    decoder: Any,
    cutoff: int,
) -> str:
    """SHA-256 fingerprint of embeddings and frozen decoder layers."""
    import torch

    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        raise ValueError("cutoff must be an integer")

    if not 0 < cutoff <= len(decoder.layers):
        raise ValueError("Invalid frozen-prefix cutoff")

    digest = hashlib.sha256()
    digest.update(b"e2-frozen-prefix-v1")

    modules = [
        ("embed_tokens", decoder.embed_tokens),
        *[
            (f"layers.{i}", decoder.layers[i])
            for i in range(cutoff)
        ],
    ]

    with torch.no_grad():
        for prefix, module in modules:
            for name, tensor in sorted(module.state_dict().items()):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"Unexpected state entry: {prefix}.{name}"
                    )

                if tensor.is_meta:
                    raise ValueError(
                        "Cannot fingerprint meta tensors; "
                        "use a checkpoint/shard fingerprint instead"
                    )

                value = tensor.detach().cpu().contiguous()

                digest.update(f"{prefix}.{name}".encode())
                digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.view(torch.uint8).numpy().tobytes())

    return digest.hexdigest()


def build_cache_metadata(
    *,
    model_revision: str,
    frozen_prefix_fingerprint: str,
    cutoff: int,
    input_ids: Any,
    attention_mask: Any = None,
    position_ids: Any = None,
    config_fingerprint: str,
) -> dict[str, Any]:
    """Build cache metadata from actual model inputs."""

    import torch

    if not model_revision or not frozen_prefix_fingerprint:
        raise ValueError("Model identity and fingerprint are required")

    if not config_fingerprint:
        raise ValueError("Configuration fingerprint is required")

    if cutoff < 1:
        raise ValueError("cutoff must be positive")

    def tensor_hash(value: Any) -> str | None:
        if value is None:
            return None

        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")

        tensor = value.detach().cpu().contiguous()

        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())

        return digest.hexdigest()

    return {
        "schema_version": 1,
        "model_revision": model_revision,
        "frozen_prefix_fingerprint": frozen_prefix_fingerprint,
        "config_fingerprint": config_fingerprint,
        "cutoff": cutoff,
        "input_ids_hash": tensor_hash(input_ids),
        "attention_mask_hash": tensor_hash(attention_mask),
        "position_ids_hash": tensor_hash(position_ids),
    }


class FrozenPrefixCache:
    """Store and retrieve frozen-prefix activations on disk."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, metadata: dict[str, Any]) -> Path:
        return self.directory / f"{cache_key(metadata)}.safetensors"

    def save(self, activation: Any, metadata: dict[str, Any]) -> Path:
        """Atomically save a frozen-prefix activation."""
        from safetensors.torch import save_file

        path = self._path(metadata)
        tensor = activation.detach().cpu().contiguous()

        fd, temporary_path = tempfile.mkstemp(
            prefix=".e2-cache-",
            suffix=".tmp",
            dir=self.directory,
        )

        try:
            os.close(fd)

            save_file(
                {"hidden_states": tensor},
                temporary_path,
            )

            os.replace(temporary_path, path)

        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

        return path

    def load(self, metadata: dict[str, Any]) -> Any | None:
        from safetensors.torch import load_file

        path = self._path(metadata)

        if not path.is_file():
            return None

        tensors = load_file(str(path), device="cpu")
        return tensors["hidden_states"]

