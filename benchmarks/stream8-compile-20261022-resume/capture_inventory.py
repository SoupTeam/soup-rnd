"""Exact tensor-payload accounting from pinned model shapes, without base weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def adapter_inventory(configuration: dict) -> list[dict]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    values = dict(configuration)
    config = AutoConfig.for_model(values.pop("model_type"), **values)
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16)
        model = get_peft_model(base, LoraConfig(
            r=4, lora_alpha=4, lora_dropout=0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    result = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
               "bytes": value.numel() * value.element_size()}
              for name, value in model.named_parameters() if value.requires_grad]
    if len(result) != configuration["num_hidden_layers"] * 14:
        raise ValueError("unexpected seven-projection adapter topology")
    if {row["dtype"] for row in result} != {"torch.float32"}:
        raise ValueError("adapter payload contract changed")
    return result


def payloads(model: dict) -> dict:
    parameters = adapter_inventory(model["config"])
    adapter = sum(row["bytes"] for row in parameters)
    vocab = model["config"]["vocab_size"]
    # Actual tiny SFTTrainer+Accelerator hooks observe F32 logits, three I64
    # [1, context] inputs and one I64 num_items_in_batch scalar per microbatch.
    # Standalone semantics never prepares the model with Accelerator, hence BF16.
    capture_groups = {
        "initial": adapter, "gradient_2_updates": 2 * adapter, "final": adapter,
        "logits_4_microbatches_fp32": 4 * 1024 * vocab * 4,
        "batch_4_microbatches_int64": 4 * (3 * 1024 * 8 + 8), "loss_float64": 4 * 8,
    }
    semantics_groups = {
        "initial": adapter, "gradient": adapter, "logits_25_valid_bf16": 25 * vocab * 2,
        "diagnostic_9_unpadded_bf16": 9 * vocab * 2, "batch": 4 * 2 * 16 * 8,
        "loss_float64": 8,
    }
    return {
        "model_id": model["model_id"], "revision": model["revision"],
        "base_weight_bytes": model["weight_bytes"],
        "adapter_tensors": len(parameters), "adapter_state_payload_bytes": adapter,
        "training_trace_tensor_count": 4 * len(parameters) + 21,
        "training_trace_groups": capture_groups,
        "training_trace_payload_bytes": sum(capture_groups.values()),
        "semantics_trace_tensor_count": 2 * len(parameters) + 7,
        "semantics_trace_groups": semantics_groups,
        "semantics_trace_payload_bytes": sum(semantics_groups.values()),
        "one_capture_adapter_payload_bytes": adapter,
        "one_timing_adapter_payload_bytes": adapter,
        "maximum_attempted_accuracy_arms": 4,
        "maximum_qualified_timing_comparisons": 2,
        "timing_fresh_processes_per_comparison": 8,
        "maximum_matrix_tensor_payload_bytes": (
            4 * (sum(capture_groups.values()) + sum(semantics_groups.values()) + adapter)
            + 16 * adapter
        ),
        "accounting_scope": "exact expected tensor payload, excludes safetensors JSON headers, "
                            "JSON/text/logs, setup scripts and model-independent source bundle; "
                            "not an observed archive or compression prediction",
        "admission": "use actual packed archive index bytes plus measured codec and transfer "
                     "cost, preserve hashes and reserve before next paid stage",
    }


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--extra-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = json.loads(args.protocol.read_text())["models"]
    if args.extra_model:
        models.append(json.loads(args.extra_model.read_text()))
    result = {str(model["size_b"]): payloads(model) for model in models}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    Console().print(result)


if __name__ == "__main__":
    main()
