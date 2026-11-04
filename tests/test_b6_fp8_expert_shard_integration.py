"""Synthetic integration checks for FP8 input and per-expert NF4 storage.

These checks do not demonstrate a real giant-checkpoint training run.
"""

from pathlib import Path

import pytest


def _split_expert_checkpoint(tmp_path: Path):
    import torch
    from safetensors.torch import load_file

    from tests.test_b2_b3_integration import _fp8_expert_checkpoint
    from tests.test_fp8_source_checkpoints import FP8_CONFIG, _write_checkpoint

    source, reference = _fp8_expert_checkpoint(tmp_path / "initial")
    tensors = load_file(str(Path(source) / "model.safetensors"))
    tensors["model.layers.0.self_attn.q_proj.weight"] = torch.eye(128).bfloat16()
    scale_key = "model.layers.0.mlp.experts.1.up_proj.weight_scale_inv"
    files = {
        "model-00001-of-00002.safetensors": [key for key in tensors if key != scale_key],
        "model-00002-of-00002.safetensors": [scale_key],
    }
    source = _write_checkpoint(tmp_path / "split", tensors, FP8_CONFIG, files)
    return Path(source), reference


def test_cross_file_expert_scale_to_nf4_uses_fp8_default(tmp_path, monkeypatch):
    import torch
    from bitsandbytes.functional import quantize_4bit

    from soup_cli.utils import layer_shard
    from soup_cli.utils.expert_shard import aligned_staging
    from tests.test_b1_expert_shard import _reader, _shard

    class OneHandle(layer_shard._SourceHandles):
        def __init__(self, shards, opener, capacity=1):
            super().__init__(shards, opener, capacity=1)

    monkeypatch.setattr(layer_shard, "_SourceHandles", OneHandle)
    source, reference = _split_expert_checkpoint(tmp_path)
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget

    budget = new_expert_cache_budget([str(path) for path in source.glob("*.safetensors")])
    out = tmp_path / "cache"
    index = _shard(
        source,
        out,
        dtype="bfloat16",
        expert_layout="per_expert",
        quant="nf4",
        quant_suffixes=("self_attn.q_proj.weight",),
        quant_device="cpu",
    )
    assert sum(path.stat().st_size for path in out.iterdir() if path.is_file()) < budget
    assert index.double_quant is False
    assert index.expert_layout == "per_expert"
    reader = _reader(out, index)
    owner, staging = aligned_staging(reader.max_span)
    try:
        host = reader.read_host(1, staging)
        prefix = "model.layers.0.mlp.experts.1."
        expected = {
            "gate_up": torch.cat(
                (reference[prefix + "gate_proj.weight"], reference[prefix + "up_proj.weight"]),
                dim=0,
            ),
            "down": reference[prefix + "down_proj.weight"],
        }
        for part, value in expected.items():
            packed, state = quantize_4bit(
                value, blocksize=64, quant_type="nf4", compress_statistics=False
            )
            assert torch.equal(host[f"1.{part}"], packed)
            assert torch.equal(host[f"1.{part}::absmax"], state.absmax)
            assert f"1.{part}::nested_absmax" not in host
    finally:
        reader.close()
    assert owner.numel() >= reader.max_span


def test_fp8_per_expert_refuses_explicit_double_quant_before_writing(tmp_path):
    from tests.test_b1_expert_shard import _shard

    source, _ = _split_expert_checkpoint(tmp_path)
    out = tmp_path / "cache"
    with pytest.raises(ValueError, match="gate-b3-fp8-source"):
        _shard(
            source,
            out,
            dtype="bfloat16",
            expert_layout="per_expert",
            quant="nf4",
            quant_suffixes=("self_attn.q_proj.weight",),
            quant_device="cpu",
            double_quant=True,
        )
    assert not (out / "index.json").exists()
