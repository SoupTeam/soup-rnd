"""Verify real-data consumption plans and synthetic tiny-Mistral CPU updates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

from consumer import (
    audit_plan,
    audit_trace,
    cpu_train,
    inspect_source,
    source_rows,
    windows,
)
from prepare import make_order_study_arms
from rich.console import Console


def state_hash(model: object, *, frozen_only: bool = False) -> str:
    import torch

    result = hashlib.sha256()
    for name, value in model.named_parameters():
        if frozen_only and value.requires_grad:
            continue
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def synthetic_cpu_check(out: Path) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import MistralConfig, MistralForCausalLM

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(901)
    config = MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=4,
                           num_key_value_heads=2, max_position_embeddings=64,
                           attention_dropout=0.0, sliding_window=None,
                           bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = "eager"
    base = MistralForCausalLM(config).cpu()
    rows = [{"id": f"synthetic-{index}",
             "input_ids": [1, *[(index + offset) % 59 + 3 for offset in range(index + 2)], 2]}
            for index in range(16)]
    scores = {row["id"]: 0.6 if index % 4 == 0 else 0.8 for index, row in enumerate(rows)}
    arms = make_order_study_arms(rows, scores, budget=83, seed=901)
    (out / "synthetic_inputs.json").write_text(json.dumps(arms, indent=2) + "\n")
    initial_hashes, results = set(), {}
    for name, data in arms.items():
        directory = out / name
        directory.mkdir(exist_ok=False)
        torch.manual_seed(901)
        model = get_peft_model(copy.deepcopy(base), LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0,
            task_type="CAUSAL_LM", bias="none"))
        initial = state_hash(model)
        initial_hashes.add(initial)
        frozen_before = state_hash(model, frozen_only=True)
        planned = list(windows(data, budget=83, targets_per_update=16))
        plan = audit_plan(planned, 83, 16)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3, weight_decay=0)
        trace = directory / "actual_trace.jsonl"
        execution = cpu_train(model, optimizer, planned, budget=83,
                               base_lr=1e-3, trace=trace)
        actual = audit_trace(trace, planned, 83, 1e-3)
        final = state_hash(model)
        if initial == final or state_hash(model, frozen_only=True) != frozen_before:
            raise ValueError("Adapter did not update or frozen base weights changed")
        adapter = {key: value.detach().cpu().contiguous()
                   for key, value in model.named_parameters() if value.requires_grad}
        save_file(adapter, str(directory / "adapter.safetensors"))
        results[name] = {"plan": plan, "execution": execution, "trace_audit": actual,
                         "initial_state_sha256": initial, "final_state_sha256": final,
                         "frozen_base_unchanged": True,
                         "adapter_sha256": hashlib.sha256(
                             (directory / "adapter.safetensors").read_bytes()).hexdigest()}
    if len(initial_hashes) != 1:
        raise ValueError("CPU proof arms did not start from identical base and adapters")
    return {"status": "PASS_SYNTHETIC_TINY_MISTRAL_CPU", "arms": results,
            "config": config.to_dict(), "all_initial_model_states_identical": True,
            "synthetic_tokens_per_arm": 83, "optimizer_updates_per_arm": 6,
            "Mistral7B_training_or_quality_claim": False, "CUDA_AMP_exercised": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Existing proof directory is retained; choose a new output")
    protocol_path = Path(__file__).with_name("protocol_draft.json")
    protocol = json.loads(protocol_path.read_text())
    manifest_sha = protocol["prepared_data"]["sha256"]
    args.out.mkdir(parents=True, exist_ok=False)
    rule = {"status": "PRE_CPU_ENGINEERING_CHECK_RULE", "real_budget": 5_000_000,
            "real_targets_per_optimizer_update": 8192, "expected_updates": 611,
            "expected_last_update_tokens": 2880,
            "normalization": "sum of shifted nonignored token CE / window target count",
            "lr": "base_lr*(1-consumed_targets_before_update/total_target_budget)",
            "cross_update_context": "repeat full causal prefix; disjoint supervision masks",
            "dropout": 0, "padding": "right, multiple8, pad_token_id0, labels-100",
            "synthetic_scope": "random tiny Mistral; LoRA q/v r2 alpha4;83targets perarm",
            "numerical_gradient_reference": "float64,rtol1e-12,atol1e-12;unit negative control",
            "manifest_sha256": manifest_sha,
            "source_sha256": {name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("consumer.py", "check_consumer.py", "prepare.py")},
            "not_GPU_readiness_or_model_quality_acceptance": True}
    (args.out / "rule.json").write_text(json.dumps(rule, indent=2) + "\n")
    try:
        real_plans = {}
        for arm in protocol["policies"]:
            source = inspect_source(args.data, arm, manifest_sha)
            planned = windows(source_rows(Path(source["path"])), 5_000_000, 8192)
            receipt = audit_plan(planned, 5_000_000, 8192)
            if (receipt["optimizer_updates"] != 611
                    or receipt["last_update_useful_tokens"] != 2880):
                raise ValueError("Different optimizer exposure")
            if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Training source changed during planning")
            real_plans[arm] = {"source": source, "consumption_plan_audit": receipt}
        (args.out / "real_data_plans.json").write_text(json.dumps(real_plans, indent=2) + "\n")
        synthetic = synthetic_cpu_check(args.out)
        (args.out / "synthetic_cpu_result.json").write_text(json.dumps(synthetic, indent=2) + "\n")
        receipt = {"status": "PASS_CPU_CONSUMER_CONFORMANCE_ONLY", "real_data_plans": real_plans,
                   "synthetic_proof": "synthetic_cpu_result.json", "GPU_used": False,
                   "Mistral7B_training_started": False, "CUDA_AMP_validated": False,
                   "rule_sha256": hashlib.sha256((args.out / "rule.json").read_bytes()).hexdigest(),
                   "runtime": {name: importlib.metadata.version(name)
                               for name in ("torch", "transformers", "peft", "safetensors")},
                   "next_gate": "CUDA FP16/AMP equivalence, checkpoint/reload/eval and F1 "
                                "integration, hardware fit and actual budget before pilot"}
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        Console().print({"status": receipt["status"], "real_planned_updates": 611,
                         "synthetic_updates_each": 6, "no_GPU_or_Mistral7B_training": True})
    except Exception as error:
        (args.out / "failure.json").write_text(json.dumps({
            "status": "CPU_ENGINEERING_CHECK_FAILED", "error": repr(error),
            "partial_traces_retained": True, "no_automatic_retry": True}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
