"""Byte-preserving source prefix selection, without loading model dependencies."""

import hashlib
import json
import struct

import pytest

from benchmarks.harness.p3_deepseek_quality_kaggle import (
    _child,
    _header,
    copy_prefix,
    select_prefix,
)


def test_prefix_keeps_scale_and_all_selected_experts_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tensors = {
        "model.embed_tokens.weight": b"embd",
        "model.layers.0.mlp.experts.0.gate_proj.weight": b"\x01\x02\x03\x04",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": b"scal",
        "model.layers.0.mlp.experts.1.gate_proj.weight": b"\xff\xfe\xfd\xfc",
        "model.layers.1.self_attn.q_a_proj.weight": b"skip",
        "lm_head.weight": b"head",
        "model.norm.weight": b"norm",
    }
    header = {}
    payload = bytearray()
    for name, raw in tensors.items():
        header[name] = {"dtype": "U8", "shape": [4],
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (source / "part.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "part.safetensors" for key in tensors},
    }))
    mapping, headers, size, elements = select_prefix(source, 1)
    kept = {key: raw for key, raw in tensors.items() if ".layers.1." not in key}
    assert set(mapping) == set(kept)
    assert size == elements == 4 * len(kept)
    destination = tmp_path / "prefix"
    config = {"num_hidden_layers": 2, "hidden_size": 7168, "n_routed_experts": 256}
    record = copy_prefix(source, destination, config=config, layers=1,
                         mapping=mapping, headers=headers)
    copied = json.loads((destination / "config.json").read_text())
    assert copied == dict(config, num_hidden_layers=1)
    actual_header, start = _header(destination / "part.safetensors")
    data = (destination / "part.safetensors").read_bytes()
    for name, expected in kept.items():
        begin, end = actual_header[name]["data_offsets"]
        assert data[start + begin:start + end] == expected
        assert record["tensors"][name]["source_tensor_bytes_sha256"] == (
            hashlib.sha256(expected).hexdigest()
        )
    with pytest.raises(FileExistsError):
        copy_prefix(source, destination, config=config, layers=1, mapping=mapping, headers=headers)


def test_source_index_cannot_escape_mount(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        _child(tmp_path, "../outside.safetensors")


@pytest.mark.parametrize("config_exists", [False, True])
def test_admission_failure_preserves_unowned_scratch_and_records_failure(tmp_path, config_exists):
    from benchmarks.harness.p3_deepseek_quality_kaggle import qualify

    source = tmp_path / "source"
    source.mkdir()
    if config_exists:
        (source / "config.json").write_text('{"model_type":"wrong"}')
    scratch = tmp_path / "scratch"
    preserved = scratch / "real-weight-bounded"
    preserved.mkdir(parents=True)
    sentinel = preserved / "user-data"
    sentinel.write_bytes(b"preserve this")
    output = tmp_path / "output"
    with pytest.raises((FileNotFoundError, ValueError)):
        qualify(source, output, scratch)
    assert sentinel.read_bytes() == b"preserve this"
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "FAILED"
    assert report["temporary_weights_removed"] is True
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["files"]["result.json"] == hashlib.sha256(
        (output / "result.json").read_bytes()).hexdigest()
