"""B2 ticket 02 — the sharder stacks per-expert Qwen4 keys into fused 3-D tensors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytest

from tests.test_issue602_qwen4_ple_streaming import _tiny_qwen4_config

# More than ten, so ordering expert ids as strings (0, 1, 10, 11, 2, ...) would fail.
_N_EXPERTS = 12


def _save_tiny_qwen4_with_experts(path: Path) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    config = _tiny_qwen4_config()
    config.text_config.num_experts = _N_EXPERTS
    torch.manual_seed(2)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.save_pretrained(path, safe_serialization=True)
    return model


def _checkpoint_file(weights_dir: Path) -> str:
    names = [name for name in os.listdir(weights_dir) if name.endswith(".safetensors")]
    assert len(names) == 1, names
    return os.path.join(weights_dir, names[0])


def _rewrite_checkpoint(
    weights_dir: Path, edit: Callable[[Dict[str, Any]], None]
) -> None:
    from safetensors.torch import load_file, save_file

    path = _checkpoint_file(weights_dir)
    state = load_file(path)
    edit(state)
    save_file(state, path, metadata={"format": "pt"})


@pytest.fixture
def tiny_qwen4(tmp_path: Path) -> Tuple[Path, Any]:
    weights = tmp_path / "weights"
    weights.mkdir()
    model = _save_tiny_qwen4_with_experts(weights)
    return weights, model


def _shard(weights: Path, out: Path) -> Any:
    from soup_cli.utils.layer_shard import shard_checkpoint

    return shard_checkpoint(str(weights), str(out), dtype="float32", arch="qwen4_exp")


def test_sharder_stacks_experts_in_integer_order(
    tiny_qwen4: Tuple[Path, Any], tmp_path: Path
) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import layer_shard_path

    weights, model = tiny_qwen4
    shards = tmp_path / "shards"
    index = _shard(weights, shards)
    source = load_file(_checkpoint_file(weights))

    assert "mlp.experts.gate_up_proj" in index.layer_keys
    assert "mlp.experts.down_proj" in index.layer_keys
    for idx in range(index.n_layers):
        prefix = f"model.layers.{idx}.mlp.experts"

        def expert(projection: str, expert_id: int, prefix: str = prefix) -> Any:
            return source[f"{prefix}.{expert_id}.{projection}.weight"]

        expected_gate_up = torch.stack(
            [
                torch.cat((expert("gate_proj", i), expert("up_proj", i)), dim=0)
                for i in range(_N_EXPERTS)
            ],
            dim=0,
        )
        expected_down = torch.stack(
            [expert("down_proj", i) for i in range(_N_EXPERTS)], dim=0
        )
        with safe_open(layer_shard_path(str(shards), idx), framework="pt") as handle:
            keys = set(handle.keys())
            gate_up = handle.get_tensor("mlp.experts.gate_up_proj")
            down = handle.get_tensor("mlp.experts.down_proj")

        assert not any(key.startswith("mlp.experts.0.") for key in keys)
        assert torch.equal(gate_up, expected_gate_up)
        assert torch.equal(down, expected_down)
        # The resident model's fused parameters are an independent reference.
        experts = model.model.layers[idx].mlp.experts
        assert torch.equal(gate_up, experts.gate_up_proj.detach())
        assert torch.equal(down, experts.down_proj.detach())
        assert not torch.equal(gate_up[2], gate_up[10])


def test_sharder_refuses_a_layer_missing_one_projection(
    tiny_qwen4: Tuple[Path, Any], tmp_path: Path
) -> None:
    weights, _model = tiny_qwen4

    def drop_up(state: Dict[str, Any]) -> None:
        for expert_id in range(_N_EXPERTS):
            del state[f"model.layers.1.mlp.experts.{expert_id}.up_proj.weight"]

    _rewrite_checkpoint(weights, drop_up)

    with pytest.raises(ValueError, match="Qwen4 layer 1 has incomplete expert projections"):
        _shard(weights, tmp_path / "shards")


@pytest.mark.parametrize(
    "dropped",
    [
        pytest.param(("gate_proj", "up_proj", "down_proj"), id="gap-in-ids"),
        pytest.param(("down_proj",), id="ids-differ-across-projections"),
    ],
)
def test_sharder_refuses_non_contiguous_or_mismatched_expert_ids(
    tiny_qwen4: Tuple[Path, Any], tmp_path: Path, dropped: Tuple[str, ...]
) -> None:
    weights, _model = tiny_qwen4

    def drop_expert_five(state: Dict[str, Any]) -> None:
        for projection in dropped:
            del state[f"model.layers.0.mlp.experts.5.{projection}.weight"]

    _rewrite_checkpoint(weights, drop_expert_five)

    with pytest.raises(ValueError, match="Qwen4 layer 0 expert ids must be contiguous"):
        _shard(weights, tmp_path / "shards")


def test_sharder_refuses_a_layer_with_fused_and_per_expert_keys(
    tiny_qwen4: Tuple[Path, Any], tmp_path: Path
) -> None:
    import torch

    weights, _model = tiny_qwen4

    def add_fused(state: Dict[str, Any]) -> None:
        state["model.layers.0.mlp.experts.gate_up_proj"] = torch.zeros(_N_EXPERTS, 16, 16)

    _rewrite_checkpoint(weights, add_fused)

    with pytest.raises(ValueError, match="Qwen4 layer 0 contains both packed and per-expert"):
        _shard(weights, tmp_path / "shards")
