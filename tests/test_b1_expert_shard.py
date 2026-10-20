"""B1.1 storage acceptance: independent bytes, B2/B3 compatibility and refusals.

These tests do not claim B1.2/B1.3 training support or a throughput improvement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _checkpoint(root: Path, count: int = 12):
    import torch
    from safetensors.torch import save_file

    root.mkdir()
    tensors = {"model.norm.weight": torch.ones(4)}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "self_attn.q_proj.weight"] = torch.eye(4)
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = torch.arange(count).float()
        tensors[prefix + "mlp.shared_experts.down_proj.weight"] = torch.ones(4, 3)
        for expert in range(count):
            for projection, shape in (
                ("gate_proj", (3, 4)),
                ("up_proj", (3, 4)),
                ("down_proj", (4, 3)),
            ):
                tensors[prefix + f"mlp.experts.{expert}.{projection}.weight"] = (
                    torch.arange(12).reshape(shape).float()
                    + 100 * expert
                    + 1000 * layer
                    + (10 if projection == "up_proj" else 0)
                )
    save_file(tensors, str(root / "model.safetensors"))
    return tensors


def _shard(source: Path, out: Path, **kwargs):
    from soup_cli.utils.layer_shard import ShardLayout, shard_checkpoint

    return shard_checkpoint(
        str(source),
        str(out),
        dtype=kwargs.pop("dtype", "float32"),
        arch="deepseek_v3",
        layout=ShardLayout(
            n_layers=2,
            stack_experts=True,
            fp32_keys=frozenset({"mlp.gate.e_score_correction_bias"}),
        ),
        **kwargs,
    )


def _reader(out: Path, index, layer: int = 0, direct: bool = False):
    from soup_cli.utils.expert_shard import ExpertShardReader
    from soup_cli.utils.layer_shard import expert_paths

    return ExpertShardReader(
        expert_paths(str(out), index)[layer],
        count=index.expert_count,
        shapes=index.expert_shapes,
        quant_specs=index.expert_quant_specs,
        dtype=index.dtype,
        direct=direct,
    )


def test_checkpoint_roundtrip_preserves_experts_and_dense_parts(tmp_path):
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.expert_shard import aligned_staging
    from soup_cli.utils.layer_shard import layer_paths, read_shard_index

    source, out = tmp_path / "source", tmp_path / "cache"
    expected = _checkpoint(source)
    index = _shard(source, out, expert_layout="per_expert")
    assert read_shard_index(str(out)) == index
    assert index.expert_layers == (0, 1) and index.expert_count == 12
    assert index.expert_layout_version == 1 and index.format_version == 5
    assert index.total_params == sum(t.numel() for t in expected.values())
    assert index.fp32_keys == ("mlp.gate.e_score_correction_bias",)
    for layer, path in enumerate(layer_paths(str(out), index)):
        dense = load_file(path)
        assert not any(key.startswith("mlp.experts.") for key in dense)
        assert "mlp.shared_experts.down_proj.weight" in dense
        assert torch.equal(dense["mlp.gate.e_score_correction_bias"], torch.arange(12).float())
        reader = _reader(out, index, layer)
        owner, staging = aligned_staging(reader.max_span)
        try:
            for expert in (10, 2, 11, 0):
                host = reader.read_host(expert, staging)
                prefix = f"model.layers.{layer}.mlp.experts.{expert}."
                want = torch.cat(
                    (expected[prefix + "gate_proj.weight"], expected[prefix + "up_proj.weight"]), 0
                )
                assert torch.equal(host[f"{expert}.gate_up"], want)
                assert torch.equal(host[f"{expert}.down"], expected[prefix + "down_proj.weight"])
            assert reader.fetches == 4
        finally:
            reader.close()
        assert owner.numel() >= reader.max_span


def test_selected_expert_reads_only_its_spans(tmp_path, monkeypatch):
    from soup_cli.utils import safetensors_reader
    from soup_cli.utils.expert_shard import aligned_staging

    source, out = tmp_path / "source", tmp_path / "cache"
    _checkpoint(source, count=12)
    index = _shard(source, out, expert_layout="per_expert")
    reader = _reader(out, index)
    owner, staging = aligned_staging(reader.max_span)
    calls = []
    actual = safetensors_reader.read_range_into

    def traced(handle, offset, target, length):
        calls.append((offset, length))
        return actual(handle, offset, target, length)

    monkeypatch.setattr(safetensors_reader, "read_range_into", traced)
    try:
        result = reader.read_host(7, staging)
        assert set(result) == {"7.gate_up", "7.down"}
        assert len(calls) == 2
        assert reader.bytes_read == (6 * 4 + 4 * 3) * 4
        # Sector alignment can overlap a neighbour's bytes; no other tensor is fetched.
        assert sum(size for _, size in calls) <= reader.max_span
    finally:
        reader.close()
    assert owner.numel() > 0


def test_default_fused_cache_still_has_identical_expert_weights(tmp_path):
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import layer_paths

    source, out = tmp_path / "source", tmp_path / "cache"
    weights = _checkpoint(source)
    index = _shard(source, out)
    assert not index.expert_layout and not index.expert_layers
    blob = load_file(layer_paths(str(out), index)[0])
    expected = torch.stack(
        [
            torch.cat(
                (
                    weights[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"],
                    weights[f"model.layers.0.mlp.experts.{e}.up_proj.weight"],
                ),
                0,
            )
            for e in range(12)
        ]
    )
    assert torch.equal(blob["mlp.experts.gate_up_proj"], expected)


def test_legacy_index_without_expert_metadata_is_readable(tmp_path):
    from soup_cli.utils.layer_shard import read_shard_index

    source, out = tmp_path / "source", tmp_path / "cache"
    _checkpoint(source)
    index = _shard(source, out)
    path = out / "index.json"
    data = json.loads(path.read_text())
    for key in list(data):
        if key.startswith("expert_"):
            del data[key]
    path.write_text(json.dumps(data))
    assert read_shard_index(str(out)) == index
    assert _shard(source, out) == index


def test_layout_switch_and_missing_file_invalidate_cache(tmp_path):
    from soup_cli.utils.layer_shard import expert_paths

    source, out = tmp_path / "source", tmp_path / "cache"
    _checkpoint(source)
    _shard(source, out)
    notices = []
    index = _shard(source, out, expert_layout="per_expert", notify=notices.append)
    assert any("expert layout changed" in message for message in notices)
    Path(expert_paths(str(out), index)[0]).unlink()
    notices.clear()
    _shard(source, out, expert_layout="per_expert", notify=notices.append)
    assert any("cached expert shard" in message and "missing" in message for message in notices)
    notices.clear()
    fused = _shard(source, out, notify=notices.append)
    assert not fused.expert_layout and any("expert layout changed" in n for n in notices)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("expert_layout", "unknown", "unknown expert_layout"),
        ("expert_layout_version", 2, "unsupported expert layout version"),
        ("expert_layout_version", True, "unsupported expert layout version"),
        ("expert_count", 0, "expert_count"),
        ("expert_count", True, "expert_count"),
        ("expert_layers", [1, 0], "expert_layers"),
        ("expert_layers", [0, 0], "expert_layers"),
        ("expert_layers", [2], "expert_layers"),
        ("expert_shapes", {}, "expert_shapes"),
        ("expert_quant_specs", {"down": {}}, "NF4 specs"),
    ],
)
def test_malformed_expert_index_is_refused(tmp_path, field, value, match):
    from soup_cli.utils.layer_shard import read_shard_index

    source, out = tmp_path / "source", tmp_path / "cache"
    _checkpoint(source)
    _shard(source, out, expert_layout="per_expert")
    path = out / "index.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        read_shard_index(str(out))


@pytest.mark.parametrize("missing", ["gate_proj", "up_proj", "down_proj"])
def test_incomplete_expert_source_is_refused(tmp_path, missing):
    from safetensors.torch import save_file

    source = tmp_path / "source"
    state = _checkpoint(source)
    del state[f"model.layers.0.mlp.experts.2.{missing}.weight"]
    save_file(state, str(source / "model.safetensors"))
    with pytest.raises(ValueError, match="expert ids must be contiguous"):
        _shard(source, tmp_path / "out", expert_layout="per_expert")


def test_fp8_sources_use_b3_decoder_before_expert_storage(tmp_path):
    import torch

    from soup_cli.utils.expert_shard import aligned_staging
    from tests.test_b2_b3_integration import _fp8_expert_checkpoint

    source, reference = _fp8_expert_checkpoint(tmp_path / "source")
    out = tmp_path / "out"
    index = _shard(Path(source), out, dtype="bfloat16", expert_layout="per_expert")
    assert index.double_quant is False
    reader = _reader(out, index)
    owner, staging = aligned_staging(reader.max_span)
    try:
        host = reader.read_host(2, staging)
        prefix = "model.layers.0.mlp.experts.2."
        expected = torch.cat(
            (reference[prefix + "gate_proj.weight"], reference[prefix + "up_proj.weight"]), 0
        )
        assert torch.equal(host["2.gate_up"], expected)
        assert torch.equal(host["2.down"], reference[prefix + "down_proj.weight"])
    finally:
        reader.close()
    assert owner.numel() > 0


def test_fp8_to_nf4_expert_storage_matches_b3_reference(tmp_path):
    import torch
    from bitsandbytes.functional import quantize_4bit
    from safetensors.torch import load_file, save_file

    from soup_cli.utils.expert_shard import aligned_staging
    from tests.test_b2_b3_integration import _fp8_expert_checkpoint

    source, reference = _fp8_expert_checkpoint(tmp_path / "source")
    path = Path(source) / "model.safetensors"
    tensors = load_file(str(path))
    tensors["model.layers.0.self_attn.q_proj.weight"] = torch.eye(128).bfloat16()
    save_file(tensors, str(path))
    out = tmp_path / "cache"
    index = _shard(
        Path(source),
        out,
        dtype="bfloat16",
        expert_layout="per_expert",
        quant="nf4",
        quant_suffixes=("self_attn.q_proj.weight",),
        quant_device="cpu",
    )
    assert index.double_quant is False
    reader = _reader(out, index)
    owner, staging = aligned_staging(reader.max_span)
    try:
        host = reader.read_host(1, staging)
        prefix = "model.layers.0.mlp.experts.1."
        values = (
            torch.cat(
                (reference[prefix + "gate_proj.weight"], reference[prefix + "up_proj.weight"]), 0
            ),
            reference[prefix + "down_proj.weight"],
        )
        for part, value in zip(("gate_up", "down"), values):
            packed, stats = quantize_4bit(
                value, blocksize=64, quant_type="nf4", compress_statistics=False
            )
            assert torch.equal(host[f"1.{part}"], packed)
            assert torch.equal(host[f"1.{part}::absmax"], stats.absmax)
    finally:
        reader.close()
    assert owner.numel() > 0


def test_training_explicitly_refuses_storage_only_cache():
    from types import SimpleNamespace

    from soup_cli.utils.layer_stream_runtime import install_streaming

    with pytest.raises(ValueError, match="B1.2/B1.3"):
        install_streaming(
            None,
            shard_dir="unused",
            index=SimpleNamespace(expert_layout="per_expert"),
            device="cpu",
        )


def test_model_builder_refuses_before_materializing_weights(monkeypatch):
    from types import SimpleNamespace

    from soup_cli.utils import layer_stream_runtime

    def unexpected(*args, **kwargs):
        pytest.fail("a B1.1 storage-only cache must be refused before weight materialization")

    built = []

    def meta_skeleton(*args, **kwargs):
        built.append(True)
        return object()

    monkeypatch.setattr(layer_stream_runtime, "build_meta_skeleton", meta_skeleton)
    monkeypatch.setattr(layer_stream_runtime, "materialize_extras", unexpected)
    monkeypatch.setattr(layer_stream_runtime, "materialize_layer_buffers", unexpected)
    with pytest.raises(ValueError, match="B1.2/B1.3"):
        layer_stream_runtime.build_streamed_model(
            model_id="unused",
            shard_dir="unused",
            lora_config=None,
            index=SimpleNamespace(expert_layout="per_expert"),
            device="cpu",
        )
    assert built == [True]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu(reason="B1.1 NF4 CUDA codec roundtrip")),
    ],
)
@pytest.mark.parametrize("double_quant", [False, True])
def test_nf4_expert_bytes_match_independent_quantization(tmp_path, double_quant, device):
    import torch
    from bitsandbytes.functional import quantize_4bit

    from soup_cli.utils.expert_shard import ExpertShardReader, aligned_staging, write_expert_shard

    generator = torch.Generator().manual_seed(7)
    pairs = [
        (torch.randn(128, 64, generator=generator), torch.randn(64, 64, generator=generator))
        for _ in range(3)
    ]
    path = str(tmp_path / "experts.safetensors")
    info = write_expert_shard(path, pairs, quantise=True, double_quant=double_quant, device=device)
    reader = ExpertShardReader(
        path,
        count=3,
        shapes=info.shapes,
        quant_specs=info.quant_specs,
        dtype="float32",
        direct=False,
    )
    owner, staging = aligned_staging(reader.max_span)
    try:
        for expert in (2, 0, 1):
            host = reader.read_host(expert, staging)
            for part, value in zip(("gate_up", "down"), pairs[expert]):
                packed, state = quantize_4bit(
                    value.to(device),
                    blocksize=64,
                    quant_type="nf4",
                    compress_statistics=double_quant,
                )
                assert torch.equal(host[f"{expert}.{part}"], packed.cpu())
                assert torch.equal(host[f"{expert}.{part}::absmax"], state.absmax.cpu())
                if double_quant:
                    assert torch.equal(
                        host[f"{expert}.{part}::nested_absmax"], state.state2.absmax.cpu()
                    )
                    assert torch.equal(host[f"{expert}.{part}::nested_offset"], state.offset.cpu())
    finally:
        reader.close()
    assert owner.numel() > 0


@pytest.mark.parametrize("double_quant", [False, True])
def test_nf4_checkpoint_and_index_roundtrip_on_cpu(tmp_path, double_quant):
    import torch
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit
    from safetensors.torch import load_file, save_file

    from soup_cli.utils.expert_shard import aligned_staging
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    source, out = tmp_path / "source", tmp_path / "cache"
    state = _checkpoint(source, count=3)
    generator = torch.Generator().manual_seed(13)
    for key in list(state):
        if "mlp.experts." in key:
            rows = 64
            state[key] = torch.randn(rows, 64, generator=generator)
        elif key.endswith("q_proj.weight"):
            state[key] = torch.randn(64, 64, generator=generator)
    save_file(state, str(source / "model.safetensors"))
    index = _shard(
        source,
        out,
        expert_layout="per_expert",
        quant="nf4",
        quant_suffixes=("self_attn.q_proj.weight",),
        quant_device="cpu",
        double_quant=double_quant,
    )
    assert index == read_shard_index(str(out))
    index_path = out / "index.json"
    original_index = index_path.read_text()
    malformed = json.loads(original_index)
    malformed["double_quant"] = not double_quant
    index_path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="double_quant"):
        read_shard_index(str(out))
    index_path.write_text(original_index)
    codes = load_file(str(out / "extras.safetensors"))
    reader = _reader(out, index)
    owner, staging = aligned_staging(reader.max_span)
    try:
        host = reader.read_host(1, staging)
        prefix = "model.layers.0.mlp.experts.1."
        values = (
            torch.cat((state[prefix + "gate_proj.weight"], state[prefix + "up_proj.weight"]), 0),
            state[prefix + "down_proj.weight"],
        )
        for part, value in zip(("gate_up", "down"), values):
            packed, expected = quantize_4bit(
                value, blocksize=64, quant_type="nf4", compress_statistics=double_quant
            )
            key = f"1.{part}"
            rebuilt = rebuild_quant_state(key, host, index.expert_quant_specs[part], codes)
            assert torch.equal(host[key], packed)
            assert torch.equal(
                dequantize_4bit(host[key], rebuilt), dequantize_4bit(packed, expected)
            )
    finally:
        reader.close()
    assert owner.numel() > 0


def test_reader_rejects_a_replaced_file_before_fetch(tmp_path):
    import os

    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.expert_shard import aligned_staging

    source, out = tmp_path / "source", tmp_path / "cache"
    _checkpoint(source)
    index = _shard(source, out, expert_layout="per_expert")
    reader = _reader(out, index)
    replacement = tmp_path / "replacement.safetensors"
    save_file({"0.gate_up": torch.zeros(6, 4)}, str(replacement))
    os.replace(replacement, reader.path)
    owner, staging = aligned_staging(reader.max_span)
    with pytest.raises(ValueError, match="changed on disk"):
        reader.read_host(0, staging)
    reader.close()
    assert owner.numel() > 0


@pytest.mark.parametrize("expert", [-1, 12, True, 1.5])
def test_invalid_expert_id_is_refused(tmp_path, expert):
    from soup_cli.utils.expert_shard import aligned_staging

    source, out = tmp_path / "source", tmp_path / "cache"
    _checkpoint(source)
    index = _shard(source, out, expert_layout="per_expert")
    reader = _reader(out, index)
    owner, staging = aligned_staging(reader.max_span)
    try:
        with pytest.raises(IndexError):
            reader.read_host(expert, staging)
    finally:
        reader.close()
    assert owner.numel() > 0


def test_direct_and_buffered_reads_agree(tmp_path):
    import torch

    from soup_cli.utils.expert_shard import aligned_staging

    source, out = tmp_path / "source", tmp_path / "cache"
    _checkpoint(source)
    index = _shard(source, out, expert_layout="per_expert")
    actual = []
    for direct in (False, True):
        reader = _reader(out, index, direct=direct)
        owner, staging = aligned_staging(reader.max_span)
        try:
            host = reader.read_host(9, staging)
            actual.append({key: value.clone() for key, value in host.items()})
        finally:
            reader.close()
        assert owner.numel() > 0
    assert all(torch.equal(actual[0][key], actual[1][key]) for key in actual[0])


def test_expert_files_follow_layer_stripe_placement(tmp_path):
    import os

    from soup_cli.utils.layer_shard import expert_paths, layer_paths

    source = tmp_path / "source"
    out, stripe = tmp_path / "cache" / "model", tmp_path / "stripe"
    _checkpoint(source)
    stripe.mkdir()
    index = _shard(source, out, expert_layout="per_expert", stripe_roots=(str(stripe),))
    assert index.layer_roots == (0, 1)
    for layer, path in expert_paths(str(out), index).items():
        assert Path(path).is_file()
        assert os.path.dirname(path) == os.path.dirname(layer_paths(str(out), index)[layer])
    before = {
        layer: Path(path).stat().st_mtime_ns
        for layer, path in expert_paths(str(out), index).items()
    }
    assert _shard(source, out, expert_layout="per_expert", stripe_roots=(str(stripe),)) == index
    assert before == {
        layer: Path(path).stat().st_mtime_ns
        for layer, path in expert_paths(str(out), index).items()
    }
