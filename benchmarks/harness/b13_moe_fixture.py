"""Tiny native DeepSeek-V3 fixture for the standalone B1.3 smoke (no pytest imports)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")


def trainables(model: Any) -> dict[str, Any]:
    """Normalize only the streaming decoder wrapper in parameter names."""
    return {
        name.replace(".inner.", "."): param
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def snapshot(model: Any, *, gradients: bool = False) -> dict[str, Any]:
    result = {}
    for name, param in trainables(model).items():
        value = param.grad if gradients else param
        result[name] = None if value is None else value.detach().cpu().clone()
    return result


@dataclass
class Fixture:
    config: Any
    streamed: Any
    resident: Any
    runtime: Any

    def close(self) -> None:
        self.runtime.close()


def build_fixture(root: Path, *, device: str = "cpu") -> Fixture:
    """Checkpoint -> native per-expert sharder -> real runtime, plus independent eager twin.

    The sizes/seeds follow test_b13_moe_runtime's fp32 fixture. The resident model loads
    the checkpoint independently; only nonzero adapter weights are copied from the stream.
    """
    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, DeepseekV3Config

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.peft_wiring import attach_adapter

    config = DeepseekV3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_routed_experts=8,
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
        use_cache=False,
    )
    root.mkdir(parents=True, exist_ok=True)
    weights, shards = root / "weights", root / "shards"
    torch.manual_seed(61)
    source = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    generator = torch.Generator().manual_seed(62)
    with torch.no_grad():
        for layer in source.model.layers[1:]:
            layer.mlp.gate.weight.normal_(0.0, 0.6, generator=generator)
            layer.mlp.gate.e_score_correction_bias.normal_(0.0, 0.3, generator=generator)
            layer.mlp.experts.gate_up_proj.normal_(0.0, 0.2, generator=generator)
            layer.mlp.experts.down_proj.normal_(0.0, 0.2, generator=generator)
    source.save_pretrained(weights, safe_serialization=True)
    del source
    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch=stream_arch_of(config),
        expert_layout="per_expert",
    )
    lora = LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, bias="none", target_modules=list(TARGETS),
        task_type=TaskType.CAUSAL_LM,
    )
    streamed, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
        device=device, dtype="float32", buffers=2, pin=False, seed=71, quant="none",
    )
    try:
        generator = torch.Generator().manual_seed(67)
        with torch.no_grad():
            for name, param in sorted(trainables(streamed).items()):
                if "lora_B" in name:
                    param.copy_(torch.randn(param.shape, generator=generator).to(param) * 0.05)
        base = AutoModelForCausalLM.from_pretrained(
            str(weights), dtype=torch.float32,
            experts_implementation="eager",
        )
        resident = attach_adapter(base, lora).to(device)
        source_params, target_params = trainables(streamed), trainables(resident)
        if not source_params or source_params.keys() != target_params.keys():
            raise AssertionError("independent models have different trainable parameters")
        with torch.no_grad():
            for name, param in target_params.items():
                param.copy_(source_params[name])
        streamed.train()
        resident.train()
        return Fixture(config, streamed, resident, runtime)
    except BaseException:
        runtime.close()
        raise


def micro_batches(config: Any, k: int, *, first: int, device: str) -> tuple[list, list]:
    """Different input/routing per micro-batch, with masked SFT labels."""
    import torch

    batches, labels = [], []
    for index in range(first, first + k):
        generator = torch.Generator().manual_seed(1000 + index)
        ids = torch.randint(0, config.vocab_size, (2, 6), generator=generator).to(device)
        lab = ids.clone()
        lab[0, : 1 + index % 3] = -100
        batches.append(ids)
        labels.append(lab)
    return batches, labels
