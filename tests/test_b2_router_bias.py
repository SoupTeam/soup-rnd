"""B2 ticket 05 — a streamed DeepSeek-V3 model routes with the checkpoint's bias in fp32."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

_BIAS = "mlp.gate.e_score_correction_bias"


def _tiny_deepseek_v3_config() -> Any:
    from transformers import DeepseekV3Config

    return DeepseekV3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=4,
        topk_group=2,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


def _save_checkpoint(weights: Path, seed: int = 53) -> dict:
    """Save a fp32 checkpoint with a non-zero router bias; return the bias per MoE layer."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    resident = AutoModelForCausalLM.from_config(_tiny_deepseek_v3_config(), dtype=torch.float32)
    biases = {}
    with torch.no_grad():
        for idx, layer in enumerate(resident.model.layers):
            if idx >= 1:
                layer.mlp.gate.e_score_correction_bias.normal_(0.0, 0.01)
                biases[idx] = layer.mlp.gate.e_score_correction_bias.clone()
    resident.save_pretrained(weights, safe_serialization=True)
    return biases


@_REQUIRES_TRAINING_FLOOR
def test_shard_layout_lists_the_router_bias_as_an_fp32_key(tmp_path: Path) -> None:
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, shard_layout_of

    weights = tmp_path / "weights"
    _save_checkpoint(weights)

    layout = shard_layout_of(build_meta_skeleton(str(weights), dtype="bfloat16"))

    assert layout.fp32_keys == frozenset({_BIAS})


@_REQUIRES_TRAINING_FLOOR
def test_sharding_at_bf16_stores_the_router_bias_in_fp32(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

    weights = tmp_path / "weights"
    biases = _save_checkpoint(weights)
    shards = str(tmp_path / "shards")

    shard_checkpoint(str(weights), shards, dtype="bfloat16")

    assert _BIAS not in load_file(layer_shard_path(shards, 0))
    for idx, bias in biases.items():
        layer = load_file(layer_shard_path(shards, idx))
        assert layer[_BIAS].dtype == torch.float32
        assert torch.equal(layer[_BIAS], bias)
        assert layer["mlp.gate.weight"].dtype == torch.bfloat16


@_REQUIRES_TRAINING_FLOOR
def test_nf4_sharding_leaves_the_router_bias_unquantized(tmp_path: Path) -> None:
    pytest.importorskip("bitsandbytes")
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import QUANT_NF4, layer_shard_path, shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    weights = tmp_path / "weights"
    biases = _save_checkpoint(weights)
    shards = str(tmp_path / "shards")
    skeleton = build_meta_skeleton(str(weights), dtype="bfloat16", quant=QUANT_NF4)

    index = shard_checkpoint(
        str(weights),
        shards,
        dtype="bfloat16",
        quant=QUANT_NF4,
        quant_suffixes=quantised_layer_suffixes(skeleton),
        quant_device="cpu",
    )

    assert "self_attn.o_proj.weight" in index.quant_specs
    assert _BIAS not in index.quant_specs
    for idx, bias in biases.items():
        layer = load_file(layer_shard_path(shards, idx))
        assert sorted(key for key in layer if key.startswith(_BIAS)) == [_BIAS]
        assert layer[_BIAS].dtype == torch.float32
        assert torch.equal(layer[_BIAS], bias)


def _lora() -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )


def _build_streamed(weights: Path, shards: str, index: Any, device: str = "cpu") -> Any:
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    return build_streamed_model(
        model_id=str(weights),
        shard_dir=shards,
        index=index,
        lora_config=_lora(),
        device=device,
        dtype="bfloat16",
        buffers=2,
        pin=False,
    )


def _router(model: Any, idx: int) -> Any:
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    return decoder_owner(model).layers[idx].inner.mlp.gate


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu(reason="checks the bias lands on CUDA")),
    ],
)
def test_streamed_router_bias_is_the_checkpoints_in_fp32_on_the_device(
    tmp_path: Path, device: str
) -> None:
    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = tmp_path / "weights"
    biases = _save_checkpoint(weights)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(str(weights), shards, dtype="bfloat16")

    model, runtime = _build_streamed(weights, shards, index, device)
    try:
        for idx, bias in biases.items():
            got = _router(model, idx).e_score_correction_bias
            assert got.dtype == torch.float32
            assert got.device.type == device
            assert bool(got.ne(0).any())
            assert torch.equal(got.cpu(), bias)
    finally:
        runtime.close()


@_REQUIRES_TRAINING_FLOOR
def test_a_shard_without_the_router_bias_refuses_the_build(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

    weights = tmp_path / "weights"
    _save_checkpoint(weights)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(str(weights), shards, dtype="bfloat16")
    path = layer_shard_path(shards, 2)
    layer = load_file(path)
    del layer[_BIAS]
    save_file(layer, path)

    with pytest.raises(ValueError, match=rf"decoder layer 2 .*'{_BIAS}'"):
        _build_streamed(weights, shards, index)


@_REQUIRES_TRAINING_FLOOR
def test_non_persistent_layer_buffers_are_left_alone(tmp_path: Path, monkeypatch: Any) -> None:
    """A non-persistent buffer is not in the shard: the build neither refuses nor fills it."""
    from transformers.models.deepseek_v3 import modeling_deepseek_v3

    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = tmp_path / "weights"
    _save_checkpoint(weights)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(str(weights), shards, dtype="bfloat16")

    router_init = modeling_deepseek_v3.DeepseekV3TopkRouter.__init__

    def init_with_scratch(self: Any, config: Any) -> None:
        router_init(self, config)
        self.register_buffer("scratch", torch.full((4,), 7.0), persistent=False)

    monkeypatch.setattr(modeling_deepseek_v3.DeepseekV3TopkRouter, "__init__", init_with_scratch)
    model, runtime = _build_streamed(weights, shards, index)
    try:
        for idx in (1, 2):
            assert torch.equal(_router(model, idx).scratch, torch.full((4,), 7.0))
    finally:
        runtime.close()


@_REQUIRES_TRAINING_FLOOR
def test_nf4_sharding_refuses_to_quantize_an_fp32_key(tmp_path: Path) -> None:
    pytest.importorskip("bitsandbytes")
    from soup_cli.utils.layer_shard import QUANT_NF4, shard_checkpoint

    weights = tmp_path / "weights"
    _save_checkpoint(weights)

    with pytest.raises(ValueError, match=rf"keeps in fp32: {_BIAS}"):
        shard_checkpoint(
            str(weights),
            str(tmp_path / "shards"),
            dtype="bfloat16",
            quant=QUANT_NF4,
            quant_suffixes=("self_attn.o_proj.weight", _BIAS),
            quant_device="cpu",
        )
