"""B2 ticket 04 — the shard index records whether experts were stacked (ADR 0004)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest
import torch

_N_EXPERTS = 3


def _write_per_expert_checkpoint(weights: Path) -> None:
    from safetensors.torch import save_file

    from tests._tiny_hf_models import write_tiny_llama_config

    weights.mkdir()
    state = {"model.embed_tokens.weight": torch.randn(32, 8)}
    for expert in range(_N_EXPERTS):
        for projection in ("gate_proj", "up_proj"):
            key = f"model.layers.0.mlp.experts.{expert}.{projection}.weight"
            state[key] = torch.randn(4, 8)
        state[f"model.layers.0.mlp.experts.{expert}.down_proj.weight"] = torch.randn(8, 4)
    save_file(state, str(weights / "model.safetensors"))
    write_tiny_llama_config(str(weights), 1)


def _write_dense_checkpoint(weights: Path) -> None:
    from safetensors.torch import save_file

    from tests._tiny_hf_models import write_tiny_llama_config

    weights.mkdir()
    state = {
        "model.embed_tokens.weight": torch.randn(32, 8),
        "model.layers.0.mlp.up_proj.weight": torch.randn(4, 8),
    }
    save_file(state, str(weights / "model.safetensors"))
    write_tiny_llama_config(str(weights), 1)


def _drop_stacking_field(shards: str) -> None:
    """Make the index look like one written before the field existed."""
    path = os.path.join(shards, "index.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    del payload["stacked_experts"]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _resharding_notices(notices: List[str]) -> List[str]:
    return [notice for notice in notices if "Re-sharding" in notice]


@pytest.mark.parametrize("written_before_the_field", [False, True])
def test_a_per_expert_cache_is_resharded_when_the_layout_says_stack(
    tmp_path: Path, written_before_the_field: bool
) -> None:
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import ShardLayout, layer_shard_path, shard_checkpoint

    weights = tmp_path / "weights"
    _write_per_expert_checkpoint(weights)
    shards = str(tmp_path / "shards")
    shard_checkpoint(
        str(weights),
        shards,
        dtype="float32",
        layout=ShardLayout(n_layers=1, stack_experts=False),
    )
    if written_before_the_field:
        _drop_stacking_field(shards)

    notices: List[str] = []
    index = shard_checkpoint(
        str(weights),
        shards,
        dtype="float32",
        layout=ShardLayout(n_layers=1, stack_experts=True),
        notify=notices.append,
    )

    assert _resharding_notices(notices) == [
        "[yellow]Re-sharding layer cache:[/] expert stacking changed (False -> True)."
    ]
    assert index.stacked_experts is True
    assert "mlp.experts.gate_up_proj" in load_file(layer_shard_path(shards, 0))


def test_a_dense_cache_written_before_the_field_is_reused(tmp_path: Path) -> None:
    from soup_cli.utils.layer_shard import ShardLayout, shard_checkpoint

    weights = tmp_path / "weights"
    _write_dense_checkpoint(weights)
    shards = str(tmp_path / "shards")
    layout = ShardLayout(n_layers=1, stack_experts=False)
    shard_checkpoint(str(weights), shards, dtype="float32", layout=layout)
    _drop_stacking_field(shards)

    notices: List[str] = []
    index = shard_checkpoint(
        str(weights), shards, dtype="float32", layout=layout, notify=notices.append
    )

    assert _resharding_notices(notices) == []
    assert index.stacked_experts is False


def test_a_stacked_qwen4_cache_written_before_the_field_is_reused(tmp_path: Path) -> None:
    from soup_cli.utils.layer_shard import ShardLayout, shard_checkpoint

    weights = tmp_path / "weights"
    _write_per_expert_checkpoint(weights)
    shards = str(tmp_path / "shards")
    layout = ShardLayout(n_layers=1, stack_experts=True)
    shard_checkpoint(str(weights), shards, dtype="float32", arch="qwen4_exp", layout=layout)
    _drop_stacking_field(shards)

    notices: List[str] = []
    index = shard_checkpoint(
        str(weights),
        shards,
        dtype="float32",
        arch="qwen4_exp",
        layout=layout,
        notify=notices.append,
    )

    assert _resharding_notices(notices) == []
    assert index.stacked_experts is True


@pytest.mark.parametrize("cache_stacked, charged", [(False, True), (True, False)])
def test_the_disk_preflight_charges_a_cache_whose_stacking_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_stacked: bool, charged: bool
) -> None:
    """The setup's layout says stack, so only a per-expert cache needs a write."""
    import types

    import soup_cli.trainer.stream_setup as stream_setup
    from soup_cli.utils.layer_shard import (
        ShardLayout,
        _discover_safetensors,
        _source_file_components,
        shard_checkpoint,
    )
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    weights = tmp_path / "weights"
    _write_per_expert_checkpoint(weights)
    shard_checkpoint(
        str(weights),
        str(tmp_path / "shards"),
        dtype="float32",
        arch="qwen4_exp",
        layout=ShardLayout(n_layers=1, stack_experts=cache_stacked),
    )
    plan = types.SimpleNamespace(
        weights_dir=str(weights),
        source_bytes=1_024,
        materialized_copy_bytes=0,
        materialize_bytes=0,
        needs_materialization=False,
        source_files=_source_file_components(_discover_safetensors(str(weights))),
    )

    def _resolve(_base, *, before_materialize=None):
        before_materialize(plan)
        return str(weights)

    rendered = {}
    monkeypatch.setattr(
        stream_setup,
        "_render_stream_disk_preflight",
        lambda **kwargs: rendered.update(kwargs),
    )

    _drive_qwen4_streaming_setup(tmp_path, monkeypatch, resolve_weights=_resolve)

    assert (rendered["shard_write_bytes"] > 0) is charged
