
"""Frozen-prefix activation cache for E2 experiments."""

from __future__ import annotations

import hashlib
import json
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


class FrozenPrefixCache:
    """Store and retrieve frozen-prefix activations on disk."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, metadata: dict[str, Any]) -> Path:
        return self.directory / f"{cache_key(metadata)}.safetensors"

    def save(self, activation: Any, metadata: dict[str, Any]) -> Path:
        from safetensors.torch import save_file

        path = self._path(metadata)
        tensor = activation.detach().cpu().contiguous()

        save_file(
            {"hidden_states": tensor},
            str(path),
        )
        return path

    def load(self, metadata: dict[str, Any]) -> Any | None:
        from safetensors.torch import load_file

        path = self._path(metadata)

        if not path.is_file():
            return None

        tensors = load_file(str(path), device="cpu")
        return tensors["hidden_states"]

