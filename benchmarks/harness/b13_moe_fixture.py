"""Shared tiny native DeepSeek-V3 construction; no pytest or eager training imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

N_EXPERTS = 8
TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
EMBEDDING_TARGETS = (*TARGETS, "embed_tokens")
HEAD_SHARED_TARGETS = (
    *TARGETS, "lm_head", "shared_experts.gate_proj", "shared_experts.up_proj",
    "shared_experts.down_proj",
)
ALL_TARGETS = (*HEAD_SHARED_TARGETS, "embed_tokens")


def trainables(model: Any) -> dict[str, Any]:
    """Normalize only the streaming wrapper in supported trainable parameter names."""
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


def moe_config(**overrides: Any) -> Any:
    """One dense layer, two MoE layers and native grouped top-k routing."""
    from transformers import DeepseekV3Config

    fields: dict[str, Any] = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=N_EXPERTS,
        n_shared_experts=1, num_experts_per_tok=2, n_group=4, topk_group=2,
        q_lora_rank=24, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8,
        v_head_dim=8, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, use_cache=False,
    )
    fields.update(overrides)
    return DeepseekV3Config(**fields)


def moe_weights_dir(root: Path, config: Any = None, *, seed: int = 61) -> Path:
    """Save the common non-degenerate native checkpoint, independently loaded by each arm."""
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    config = config if config is not None else moe_config()
    weights = root / "weights"
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for layer in model.model.layers[config.first_k_dense_replace:]:
            layer.mlp.gate.weight.normal_(0.0, 0.6, generator=generator)
            layer.mlp.gate.e_score_correction_bias.normal_(0.0, 0.3, generator=generator)
            layer.mlp.experts.gate_up_proj.normal_(0.0, 0.2, generator=generator)
            layer.mlp.experts.down_proj.normal_(0.0, 0.2, generator=generator)
    model.save_pretrained(weights, safe_serialization=True)
    last = config.num_hidden_layers - 1
    with safe_open(str(weights / "model.safetensors"), framework="pt") as handle:
        keys = set(handle.keys())
    last_key = f"model.layers.{last}.mlp.experts.{config.n_routed_experts - 1}.gate_proj.weight"
    if last_key not in keys:
        raise AssertionError("native checkpoint did not save separate routed expert keys")
    if f"model.layers.{last}.mlp.experts.gate_up_proj" in keys:
        raise AssertionError("native checkpoint unexpectedly saved fused routed expert keys")
    return weights


def moe_lora_config(*, dropout: float = 0.0, targets: Sequence[str] = TARGETS) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=2, lora_alpha=4, lora_dropout=dropout, bias="none", target_modules=list(targets),
        task_type=TaskType.CAUSAL_LM,
    )


def randomize_adapters(model: Any, *, seed: int = 67) -> None:
    """Make both factors differentiable, including PEFT's initially zero embedding A."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(trainables(model).items()):
            if "lora_B" in name or "lora_embedding_" in name:
                param.copy_(torch.randn(param.shape, generator=generator).to(param) * 0.05)


def copy_adapter_weights(source: Any, target: Any) -> None:
    import torch

    source_params, target_params = trainables(source), trainables(target)
    if not source_params or source_params.keys() != target_params.keys():
        raise AssertionError("independent models have different trainable parameters")
    with torch.no_grad():
        for name, param in target_params.items():
            param.copy_(source_params[name])


def resident_base(weights: Path, *, dtype: str = "float32") -> Any:
    """Load native eager weights without any streaming construction or execution helpers."""
    import torch
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.kimi_k2 import load_stream_config

    config = load_stream_config(str(weights), trust_remote_code=False)
    config.use_cache = False
    base = AutoModelForCausalLM.from_pretrained(
        str(weights), config=config, dtype=getattr(torch, dtype), experts_implementation="eager",
    )
    if base.config._experts_implementation != "eager":
        raise AssertionError("resident reference must use native eager experts")
    return base


def resident_adapter(base: Any, lora: Any, *, source: Any = None) -> Any:
    from soup_cli.utils.peft_wiring import attach_adapter

    resident = attach_adapter(base, lora)
    if source is not None:
        copy_adapter_weights(source, resident)
    return resident


def moe_batch(config: Any, *, seed: int = 0, rows: int = 2, seq: int = 6) -> dict[str, Any]:
    import torch

    generator = torch.Generator().manual_seed(1000 + seed)
    ids = torch.randint(0, config.vocab_size, (rows, seq), generator=generator)
    return {"input_ids": ids, "labels": ids.clone()}


@dataclass
class Fixture:
    config: Any
    streamed: Any
    resident: Any
    runtime: Any

    def close(self) -> None:
        self.runtime.close()


def build_fixture(
    root: Path, *, device: str = "cpu", targets: Sequence[str] = TARGETS, dropout: float = 0.0,
) -> Fixture:
    """Checkpoint -> real per-expert runtime, plus an independently loaded eager twin."""
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    config = moe_config()
    root.mkdir(parents=True, exist_ok=True)
    weights = moe_weights_dir(root, config)
    shards = root / "shards"
    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch=stream_arch_of(config),
        expert_layout="per_expert",
    )
    lora = moe_lora_config(dropout=dropout, targets=targets)
    streamed, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
        device=device, dtype="float32", buffers=2, pin=False, seed=71, quant="none",
    )
    try:
        randomize_adapters(streamed)
        resident = resident_adapter(resident_base(weights), lora, source=streamed).to(device)
        streamed.train()
        resident.train()
        return Fixture(config, streamed, resident, runtime)
    except BaseException:
        runtime.close()
        raise


def micro_batches(
    config: Any, k: int, *, first: int, device: str,
) -> tuple[list[Any], list[Any]]:
    """Different input/routing per micro-batch, with masked SFT labels."""
    batches, labels = [], []
    for index in range(first, first + k):
        batch = moe_batch(config, seed=index)
        ids = batch["input_ids"].to(device)
        lab = batch["labels"].to(device)
        lab[0, : 1 + index % 3] = -100
        batches.append(ids)
        labels.append(lab)
    return batches, labels
