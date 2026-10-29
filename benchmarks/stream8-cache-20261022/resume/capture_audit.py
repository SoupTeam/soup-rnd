"""Count required original capture operands without allocating model weights."""

import argparse
import hashlib
import json
from pathlib import Path

from rich.console import Console


def audit(protocol):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model_info = protocol["models"][0]
    config = Qwen3Config(**model_info["config"])
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config).to(torch.bfloat16)
        model = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=4, lora_dropout=0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]), low_cpu_mem_usage=True)
    adapters = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                       "bytes": value.numel() * value.element_size()}
                for name, value in model.named_parameters() if value.requires_grad}
    assert len(adapters) == config.num_hidden_layers * 14
    assert all(value.device.type == "meta" for value in model.parameters())
    adapter_bytes = sum(row["bytes"] for row in adapters.values())
    fixed_training = adapter_bytes * 4 + 4 * 3 * 1024 * 8 + 4 * 8
    # Original semantic fixture has 2x16 tokens, seven left padding tokens;
    # all25 valid logits plus9 unpadded diagnostic logits must remain present.
    fixed_semantics = adapter_bytes * 2 + 4 * 2 * 16 * 8 + 8
    cases = {}
    for name, itemsize in (("bf16_logits", 2), ("fp32_logits", 4)):
        training_logits = 4 * 1024 * config.vocab_size * itemsize
        semantic_logits = (25 + 9) * config.vocab_size * itemsize
        training = fixed_training + training_logits
        semantics = fixed_semantics + semantic_logits
        cases[name] = {
            "training_logits_bytes": training_logits,
            "training_capture_payload_bytes": training,
            "semantic_capture_payload_bytes": semantics,
            "four_arm_capture_payload_bytes": 4 * (training + semantics),
            "24_timing_final_adapters_payload_bytes": 24 * adapter_bytes,
            "all_scientific_tensor_payload_bytes": 4 * (training + semantics) + 24 * adapter_bytes,
        }
    shapes = {name: row["shape"] for name, row in adapters.items()}
    return {
        "model": model_info["model_id"], "revision": model_info["revision"],
        "layers": config.num_hidden_layers, "vocab_size": config.vocab_size,
        "adapter_tensor_count": len(adapters), "adapter_state_bytes": adapter_bytes,
        "adapters_all_meta_no_base_allocated": True,
        "adapter_dtype_counts": {dtype: sum(row["dtype"] == dtype for row in adapters.values())
                                  for dtype in sorted({row["dtype"] for row in adapters.values()})},
        "adapter_shapes_sha256": hashlib.sha256(
            json.dumps(shapes, sort_keys=True).encode()).hexdigest(),
        "required_training_groups": {"initial": len(adapters), "gradient/0": len(adapters),
                                     "gradient/1": len(adapters), "final": len(adapters),
                                     "logits": 4, "loss": 1,
                                     "batch": "all Tensor forward kwargs for all4 microbatches"},
        "required_semantic_groups": ["initial", "gradient", "batch", "logits/valid", "loss",
                                     "diagnostic/unpadded_logits"],
        "cases": cases,
        "limits": "Payload sizes exclude safetensors headers/JSON/logs and any additional Tensor "
                  "forward kwargs. Runtime dtype/shape/key audit determines actual exact bytes; "
                  "both logits dtype cases retained prospectively, no tensor may be omitted.",
        "retention_requirement": "lossless reconstruction of every original raw byte and filename; "
                                 "do not replace tensors with hashes or reduced-precision values",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.protocol.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    Console().print({key: result[key] for key in ("adapter_tensor_count", "adapter_state_bytes",
                                                 "adapter_dtype_counts", "cases")})
