"""B2 ticket 03 — the meta skeleton decides the shard layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_qwen35_streaming import _copy_toy_lora, _tiny_qwen35_lora

# More than ten, so ordering expert ids as strings (0, 1, 10, 11, 2, ...) would fail.
_N_EXPERTS = 12


def _tiny_qwen35_moe_config() -> Any:
    from transformers import Qwen3_5MoeTextConfig

    return Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        num_experts=_N_EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        max_position_embeddings=128,
    )


def _save_resident(weights: Path, seed: int) -> Any:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    resident = AutoModelForCausalLM.from_config(
        _tiny_qwen35_moe_config(), dtype=torch.float32
    ).eval()
    resident.save_pretrained(weights, safe_serialization=True)
    return resident


def _logits_match_resident(resident: Any, weights: Path, shards: str, index: Any) -> None:
    from peft import get_peft_model

    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    streamed, runtime = build_streamed_model(
        model_id=str(weights),
        shard_dir=shards,
        index=index,
        lora_config=_tiny_qwen35_lora(),
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=41,
    )
    try:
        reference = get_peft_model(resident, _tiny_qwen35_lora())
        _copy_toy_lora(streamed, reference)
        streamed.eval()
        reference.eval()
        input_ids = torch.tensor([[1, 7, 3, 11, 5, 2, 9, 4, 13, 8, 6, 10]])
        with torch.no_grad():
            got = streamed(input_ids=input_ids).logits
            want = reference(input_ids=input_ids).logits
        assert torch.equal(got, want), (got - want).abs().max().item()
    finally:
        runtime.close()


@_REQUIRES_TRAINING_FLOOR
def test_fused_qwen35_moe_from_per_expert_keys_matches_resident(tmp_path: Path) -> None:
    """Parity arm 9: fused experts in the skeleton, per-expert keys on disk."""
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of

    weights = tmp_path / "weights"
    resident = _save_resident(weights, seed=43)
    with safe_open(str(weights / "model.safetensors"), framework="pt") as handle:
        assert "model.layers.0.mlp.experts.11.gate_proj.weight" in handle.keys()

    shards = str(tmp_path / "shards")
    index = shard_checkpoint(
        str(weights), shards, dtype="float32", arch=stream_arch_of(resident.config)
    )
    with safe_open(layer_shard_path(shards, 0), framework="pt") as handle:
        assert handle.get_slice("mlp.experts.gate_up_proj").get_shape() == [
            _N_EXPERTS,
            16,
            16,
        ]

    _logits_match_resident(resident, weights, shards, index)



@_REQUIRES_TRAINING_FLOOR
def test_keys_beyond_the_skeletons_layers_are_dropped_and_counted(tmp_path: Path) -> None:
    """A layer the skeleton lacks, written into the checkpoint by hand like an MTP layer."""
    import os

    from safetensors.torch import load_file, save_file

    from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

    weights = tmp_path / "weights"
    resident = _save_resident(weights, seed=47)
    path = str(weights / "model.safetensors")
    state = load_file(path)
    extra = {
        key.replace("model.layers.1.", "model.layers.2.", 1): value.clone()
        for key, value in state.items()
        if key.startswith("model.layers.1.")
    }
    assert len(extra) > 3 * _N_EXPERTS
    save_file({**state, **extra}, path, metadata={"format": "pt"})

    shards = str(tmp_path / "shards")
    notices: list = []
    index = shard_checkpoint(
        str(weights), shards, dtype="float32", arch="qwen3", notify=notices.append
    )

    assert index.n_layers == 2
    assert not os.path.exists(layer_shard_path(shards, 2))
    assert [notice for notice in notices if "Dropped" in notice] == [
        f"[yellow]Dropped {len(extra)} checkpoint keys of decoder layers at or beyond "
        "the model's 2 layers (an MTP layer, for example).[/]"
    ]
    _logits_match_resident(resident, weights, shards, index)


def test_per_expert_keys_stay_when_the_skeleton_has_no_fused_experts(tmp_path: Path) -> None:
    """A skeleton without ``mlp.experts.gate_up_proj`` keeps per-expert keys as they are."""
    from safetensors.torch import load_file, save_file

    from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint
    from tests._tiny_hf_models import write_tiny_llama_config

    weights = tmp_path / "weights"
    weights.mkdir()
    state = {"model.embed_tokens.weight": torch.randn(32, 8)}
    for expert in range(_N_EXPERTS):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.0.mlp.experts.{expert}.{projection}.weight"
            state[key] = torch.randn(4, 8)
    save_file(state, str(weights / "model.safetensors"))
    write_tiny_llama_config(str(weights), 1)

    shards = str(tmp_path / "shards")
    index = shard_checkpoint(str(weights), shards, dtype="float32")

    layer = load_file(layer_shard_path(shards, 0))
    assert "mlp.experts.gate_up_proj" not in layer
    assert len(layer) == 3 * _N_EXPERTS
    assert torch.equal(
        layer["mlp.experts.11.up_proj.weight"],
        state["model.layers.0.mlp.experts.11.up_proj.weight"],
    )
    assert index.n_layers == 1
