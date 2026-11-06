"""Independently bind saved giant-adapter tensors to the training/reload digest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(adapter, config_path, events_path, output):
    import torch
    from peft import LoraConfig, get_peft_model
    from rich.console import Console
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, DeepseekV3Config

    torch.set_num_threads(1)
    real = json.loads(config_path.read_text())
    settings = json.loads((adapter / "adapter_config.json").read_text())
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    steps = [row for row in events if row["event"] == "step"]
    reloads = [row for row in events if row["event"] == "reload"]
    assert len(steps) == 2 and len(reloads) == 1 and reloads[0]["equal"]
    assert settings["r"] == 2 and settings["lora_alpha"] == 4
    assert real["num_hidden_layers"] == 61 and real["n_routed_experts"] == 256
    tiny = DeepseekV3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
        num_hidden_layers=real["num_hidden_layers"], num_attention_heads=2,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        first_k_dense_replace=3, n_group=2, topk_group=1, q_lora_rank=8,
        kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
        max_position_embeddings=64, use_cache=False,
    )
    model = get_peft_model(AutoModelForCausalLM.from_config(tiny, dtype=torch.float32),
                           LoraConfig(r=2, lora_alpha=4, lora_dropout=0, bias="none",
                                      target_modules=settings["target_modules"],
                                      task_type="CAUSAL_LM"))
    names = [name.replace(".default.", ".") for name, value in model.named_parameters()
             if value.requires_grad]
    assert len(names) == len(set(names)) == 610
    widths = {
        "q_a_proj": (real["hidden_size"], real["q_lora_rank"]),
        "q_b_proj": (real["q_lora_rank"], real["num_attention_heads"]
                     * (real["qk_nope_head_dim"] + real["qk_rope_head_dim"])),
        "kv_a_proj_with_mqa": (real["hidden_size"], real["kv_lora_rank"]
                               + real["qk_rope_head_dim"]),
        "kv_b_proj": (real["kv_lora_rank"], real["num_attention_heads"]
                      * (real["qk_nope_head_dim"] + real["v_head_dim"])),
        "o_proj": (real["num_attention_heads"] * real["v_head_dim"], real["hidden_size"]),
    }
    file = adapter / "adapter_model.safetensors"
    digest = hashlib.sha256()
    parameters = 0
    zero_b = 0
    with safe_open(file, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(names)
        for name in names:
            tensor = handle.get_tensor(name)
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all(), name
            module = name.split(".self_attn.")[1].split(".")[0]
            inp, out = widths[module]
            is_a = ".lora_A." in name
            expected_shape = (2, inp) if is_a else (out, 2)
            assert tuple(tensor.shape) == expected_shape, (name, tensor.shape, expected_shape)
            parameters += tensor.numel()
            if not is_a and torch.count_nonzero(tensor) == 0:
                zero_b += 1
            digest.update(tensor.contiguous().numpy().tobytes())
    assert parameters == 12125824
    expected = steps[-1]["adapter_sha256_after"]
    assert digest.hexdigest() == expected == reloads[0]["parameter_sha256"]
    result = {
        "method": "native parameter order from a resident tiny 61-layer DeepSeek-V3/PEFT "
                  "topology; actual shapes and values read from the saved giant adapter; "
                  "no giant forward rerun and no reuse of the harness hash helper",
        "tensor_count": len(names), "parameter_count": parameters,
        "all_fp32": True, "all_finite": True, "shapes_match_pinned_giant_config": True,
        "zero_lora_b_tensors_after_training": zero_b,
        "native_parameter_sha256": digest.hexdigest(),
        "matches_final_training_and_reload": True,
        "adapter_file_bytes": file.stat().st_size,
        "adapter_file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    Console().print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.adapter, args.config, args.events, args.output)
