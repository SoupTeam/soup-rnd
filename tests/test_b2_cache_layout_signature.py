"""B2 review fix: cache acceptance compares the whole skeleton-derived layout (ADR 0004).

The layout decides the layer count (MTP filtering), expert stacking and the fp32 keys. A cache
that agrees on the stacking alone but not on the rest must be rebuilt, not returned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest
import torch

from tests.test_b2_deepseek_v3_stream import _save_fixture, _tiny_deepseek_v3_config
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

_ROUTER_BIAS = "mlp.gate.e_score_correction_bias"


def _write_checkpoint(weights: Path, n_layers: int = 2) -> None:
    from safetensors.torch import save_file

    from tests._tiny_hf_models import write_tiny_llama_config

    weights.mkdir()
    state = {"model.embed_tokens.weight": torch.randn(32, 8)}
    for layer in range(n_layers):
        state[f"model.layers.{layer}.mlp.up_proj.weight"] = torch.randn(4, 8)
        state[f"model.layers.{layer}.{_ROUTER_BIAS}"] = torch.randn(4)
    save_file(state, str(weights / "model.safetensors"))
    write_tiny_llama_config(str(weights), n_layers)


def _drop_index_fields(shards: str, *fields: str) -> None:
    """Make the index look like one written before those fields existed."""
    path = os.path.join(shards, "index.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for name in fields:
        payload.pop(name, None)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _resharding_notices(notices: List[str]) -> List[str]:
    return [notice for notice in notices if "Re-sharding" in notice]


def _shard(weights: Path, shards: str, layout, notices: List[str]):
    from soup_cli.utils.layer_shard import shard_checkpoint

    return shard_checkpoint(
        str(weights), shards, dtype="float32", layout=layout, notify=notices.append
    )


def test_a_cache_with_another_layer_count_is_resharded(tmp_path: Path) -> None:
    from soup_cli.utils.layer_shard import ShardLayout

    weights = tmp_path / "weights"
    _write_checkpoint(weights, n_layers=2)
    shards = str(tmp_path / "shards")
    _shard(weights, shards, ShardLayout(n_layers=2, stack_experts=False), [])

    notices: List[str] = []
    index = _shard(weights, shards, ShardLayout(n_layers=1, stack_experts=False), notices)

    assert _resharding_notices(notices) == [
        "[yellow]Re-sharding layer cache:[/] decoder layer count changed (2 -> 1)."
    ]
    assert index.n_layers == 1


def test_a_cache_with_other_fp32_keys_is_resharded(tmp_path: Path) -> None:
    from soup_cli.utils.layer_shard import ShardLayout

    weights = tmp_path / "weights"
    _write_checkpoint(weights)
    shards = str(tmp_path / "shards")
    _shard(weights, shards, ShardLayout(n_layers=2, stack_experts=False), [])

    notices: List[str] = []
    index = _shard(
        weights,
        shards,
        ShardLayout(n_layers=2, stack_experts=False, fp32_keys=frozenset({_ROUTER_BIAS})),
        notices,
    )

    assert _resharding_notices(notices) == [
        "[yellow]Re-sharding layer cache:[/] fp32 keys changed "
        f"(none -> {_ROUTER_BIAS})."
    ]
    assert index.fp32_keys == (_ROUTER_BIAS,)


def test_a_cache_with_the_same_layout_is_reused(tmp_path: Path) -> None:
    from soup_cli.utils.layer_shard import ShardLayout

    weights = tmp_path / "weights"
    _write_checkpoint(weights)
    shards = str(tmp_path / "shards")
    layout = ShardLayout(n_layers=2, stack_experts=False, fp32_keys=frozenset({_ROUTER_BIAS}))
    _shard(weights, shards, layout, [])

    notices: List[str] = []
    _shard(weights, shards, layout, notices)

    assert _resharding_notices(notices) == []


def test_a_cache_written_before_fp32_keys_were_recorded_is_reused_without_any(
    tmp_path: Path,
) -> None:
    """A layout with no fp32 keys cannot have changed them, so older caches stay valid."""
    from soup_cli.utils.layer_shard import ShardLayout

    weights = tmp_path / "weights"
    _write_checkpoint(weights)
    shards = str(tmp_path / "shards")
    layout = ShardLayout(n_layers=2, stack_experts=False)
    _shard(weights, shards, layout, [])
    _drop_index_fields(shards, "fp32_keys")

    notices: List[str] = []
    _shard(weights, shards, layout, notices)

    assert _resharding_notices(notices) == []


def test_a_cache_written_before_fp32_keys_were_recorded_is_resharded_with_some(
    tmp_path: Path,
) -> None:
    """The old index cannot prove the router bias was kept in fp32, so it is rebuilt."""
    from soup_cli.utils.layer_shard import ShardLayout

    weights = tmp_path / "weights"
    _write_checkpoint(weights)
    shards = str(tmp_path / "shards")
    _shard(weights, shards, ShardLayout(n_layers=2, stack_experts=False), [])
    _drop_index_fields(shards, "fp32_keys")

    notices: List[str] = []
    index = _shard(
        weights,
        shards,
        ShardLayout(n_layers=2, stack_experts=False, fp32_keys=frozenset({_ROUTER_BIAS})),
        notices,
    )

    assert len(_resharding_notices(notices)) == 1
    assert index.fp32_keys == (_ROUTER_BIAS,)


@pytest.mark.parametrize("cache_layers, charged", [(2, True), (1, False)])
def test_the_disk_preflight_charges_a_cache_whose_layer_count_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_layers: int, charged: bool
) -> None:
    """The setup's skeleton has one layer, so only a one-layer cache avoids a write."""
    from soup_cli.utils.layer_shard import ShardLayout, inspect_shard_cache

    weights = tmp_path / "weights"
    _write_checkpoint(weights, n_layers=2)
    shards = str(tmp_path / "shards")
    layout = ShardLayout(n_layers=cache_layers, stack_experts=False)
    index = _shard(weights, shards, layout, [])

    cached, reason = inspect_shard_cache(
        shards,
        "float32",
        index.source_fingerprint,
        index.source_files,
        "none",
        False,
        "",
        layout=ShardLayout(n_layers=1, stack_experts=False),
    )

    assert (cached is None) is charged
    assert ("layer count" in reason) is charged


@_REQUIRES_TRAINING_FLOOR
def test_a_config_edit_that_drops_a_layer_reshards_the_cache(tmp_path: Path) -> None:
    """The source files are unchanged, so only the skeleton-derived layout can tell."""
    from transformers import AutoConfig

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of

    weights = tmp_path / "weights"
    _save_fixture(weights, _tiny_deepseek_v3_config())
    arch = stream_arch_of(AutoConfig.from_pretrained(str(weights)))
    shards = str(tmp_path / "shards")
    first = shard_checkpoint(str(weights), shards, dtype="float32", arch=arch)
    assert first.n_layers == 3
    assert first.fp32_keys == (_ROUTER_BIAS,)

    config_path = weights / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_hidden_layers"] = 2
    config_path.write_text(json.dumps(config), encoding="utf-8")

    notices: List[str] = []
    second = shard_checkpoint(
        str(weights), shards, dtype="float32", arch=arch, notify=notices.append
    )

    assert _resharding_notices(notices) == [
        "[yellow]Re-sharding layer cache:[/] decoder layer count changed (3 -> 2)."
    ]
    assert second.n_layers == 2
    assert second.source_fingerprint == first.source_fingerprint
