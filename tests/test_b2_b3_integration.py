"""B2 and B3 together: block-scaled FP8 per-expert keys are decoded, then stacked.

A DeepSeek-V3 download stores every routed expert as its own float8 matrix with a
``weight_scale_inv`` grid. The sharder must decode each one (B3) and stack the decoded
tensors into the fused 3-D parameters the skeleton holds (B2), and the cache check must
still compare the whole layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from tests.test_fp8_source_checkpoints import (
    FP8_CONFIG,
    _Exponents,
    _fp8_weight,
    _reference_decode,
    _write_checkpoint,
)

_N_EXPERTS = 3
_SIZE = 128


def _fp8_expert_checkpoint(root: Path, n_layers: int = 2):
    generator = torch.Generator().manual_seed(7)
    exponents = _Exponents()
    tensors = {"model.embed_tokens.weight": torch.randn(16, _SIZE).to(torch.bfloat16)}
    reference = {}
    for layer in range(n_layers):
        for expert in range(_N_EXPERTS):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                q, scale = _fp8_weight((_SIZE, _SIZE), exponents, generator=generator)
                tensors[key] = q
                tensors[key + "_scale_inv"] = scale
                reference[key] = _reference_decode(q, scale)
    return _write_checkpoint(root, tensors, FP8_CONFIG), reference


def test_fp8_per_expert_keys_are_decoded_then_stacked(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import ShardLayout, layer_shard_path, shard_checkpoint

    source, reference = _fp8_expert_checkpoint(tmp_path / "src")
    shards = str(tmp_path / "shards")

    index = shard_checkpoint(
        source,
        shards,
        dtype="bfloat16",
        arch="llama",
        layout=ShardLayout(n_layers=2, stack_experts=True),
    )

    assert index.stacked_experts is True
    blob = load_file(layer_shard_path(shards, 1))
    assert blob["mlp.experts.gate_up_proj"].dtype == torch.bfloat16
    assert not any("scale_inv" in key for key in blob)
    for expert in range(_N_EXPERTS):
        prefix = f"model.layers.1.mlp.experts.{expert}."
        want_gate_up = torch.cat(
            (reference[prefix + "gate_proj.weight"], reference[prefix + "up_proj.weight"]), 0
        )
        assert torch.equal(blob["mlp.experts.gate_up_proj"][expert], want_gate_up), expert
        assert torch.equal(
            blob["mlp.experts.down_proj"][expert], reference[prefix + "down_proj.weight"]
        ), expert


def test_a_decoded_fp8_cache_is_resharded_when_the_layer_count_changes(
    tmp_path: Path,
) -> None:
    from soup_cli.utils.layer_shard import ShardLayout, shard_checkpoint

    source, _reference = _fp8_expert_checkpoint(tmp_path / "src")
    shards = str(tmp_path / "shards")
    shard_checkpoint(
        source, shards, dtype="bfloat16", arch="llama",
        layout=ShardLayout(n_layers=2, stack_experts=True),
    )

    notices: List[str] = []
    index = shard_checkpoint(
        source, shards, dtype="bfloat16", arch="llama",
        layout=ShardLayout(n_layers=1, stack_experts=True),
        notify=notices.append,
    )

    assert [n for n in notices if "Re-sharding" in n] == [
        "[yellow]Re-sharding layer cache:[/] decoder layer count changed (2 -> 1)."
    ]
    assert index.n_layers == 1
