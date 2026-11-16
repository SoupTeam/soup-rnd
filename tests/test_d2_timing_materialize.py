"""Deduplicated evidence must reconstruct exactly and remain contained."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def materializer():
    source = Path(__file__).parents[1] / "benchmarks/harness/d2_materialize_timing.py"
    spec = importlib.util.spec_from_file_location("timing_materialize", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(root, alias="copy.safetensors"):
    data = b"synthetic evidence payload"
    (root / "original.safetensors").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (root / "dedup-map.json").write_text(json.dumps({alias: "original.safetensors"}))
    (root / "evidence-manifest.json").write_text(
        json.dumps(
            {name: {"sha256": digest, "included": True} for name in ("original.safetensors", alias)}
        )
    )
    return data


def test_restore_and_verify_identical_alias(tmp_path):
    data = bundle(tmp_path)
    assert materializer().materialize(tmp_path) == 2
    assert (tmp_path / "copy.safetensors").read_bytes() == data


def test_reject_corrupt_alias_source(tmp_path):
    bundle(tmp_path)
    (tmp_path / "original.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        materializer().materialize(tmp_path)


def test_reject_alias_path_escape(tmp_path):
    bundle(tmp_path, "../escape.safetensors")
    with pytest.raises(ValueError, match="escapes"):
        materializer().materialize(tmp_path)
