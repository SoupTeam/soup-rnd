"""Prospective paid tiny CUDA gate. Must not be run during local CPU preparation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

from check_consumer import state_hash
from consumer import windows
from engine import audit_runtime, train
from rich.console import Console
from run import build_model, write
from run_support import Guard, admit, contract, gpu_inventory, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = contract(args.contract, args.contract_sha)
    spec["launch_contract_sha256"] = args.contract_sha
    billing = json.loads(args.admission.read_text())
    admit(billing, spec, "cuda-gate")
    if os.environ.get("SOUP_G6_INSTANCE_ID") != str(billing["instance_id"]):
        parser.error("Instance identity mismatch")
    if runtime() != spec["runtime_target"]:
        parser.error("Runtime differs from prospective package pins")
    args.out.mkdir(parents=True, exist_ok=False)
    write(
        args.out / "prospective_rule.json",
        {
            "contract_sha256": args.contract_sha,
            "purpose": "TINY_CUDA_NUMERICAL_GATE",
            "max_relative_adapter_delta_l2_error": 0.2,
            "max_absolute_adapter_delta_error": 2e-5,
            "max_absolute_initial_logits_error": 0.005,
            "base": "identical FP16-rounded tiny frozen base; CPU FP32 math vs CUDA AMP",
            "scope": "not full7B fit, quality or F1 timing acceptance",
        },
    )
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        inventory = gpu_inventory(billing["gpu_uuid"])
        guard = Guard(args.out, 600, gpu=True, uuid=billing["gpu_uuid"])
        guard()
        cpu = build_model(spec, None, synthetic=True)
        for parameter in cpu.parameters():
            if not parameter.requires_grad:
                parameter.data = parameter.data.half().float()
        gpu = copy.deepcopy(cpu).to("cuda:0")
        for parameter in gpu.parameters():
            if not parameter.requires_grad:
                parameter.data = parameter.data.half()
        guard()
        initial = {
            name: value.detach().clone()
            for name, value in cpu.named_parameters()
            if value.requires_grad
        }
        cpu.eval()
        gpu.eval()
        ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 2]])
        with torch.inference_mode():
            first = cpu(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            with torch.autocast("cuda", dtype=torch.float16):
                second = (
                    gpu(
                        input_ids=ids.cuda(),
                        attention_mask=torch.ones_like(ids).cuda(),
                        use_cache=False,
                    )
                    .logits.float()
                    .cpu()
                )
        error = (first - second).abs().max().item()
        if error > 0.005:
            raise ValueError("Initial CUDA/CPU logits error exceeds frozen .005 bound")
        rows = [
            {"id": "short", "input_ids": [1, 3, 2]},
            {"id": "long", "input_ids": [1, 4, 5, 6, 7, 8, 9, 2]},
        ]
        before = {
            "cpu": state_hash(cpu, frozen_only=True),
            "gpu": state_hash(gpu, frozen_only=True),
        }
        executions = {}
        for name, model, device in (("cpu", cpu, "cpu"), ("gpu", gpu, "cuda:0")):
            optimizer = torch.optim.AdamW(
                [value for value in model.parameters() if value.requires_grad],
                lr=spec["recipe"]["lr"],
                weight_decay=0,
                foreach=False,
                fused=False,
            )
            executions[name] = train(
                model,
                optimizer,
                windows(rows, 9, 4),
                budget=9,
                base_lr=spec["recipe"]["lr"],
                trace=args.out / f"{name}_trace.jsonl",
                telemetry=args.out / f"{name}_telemetry.jsonl",
                device=device,
                guard=guard,
            )
            audit_runtime(
                args.out / f"{name}_trace.jsonl", windows(rows, 9, 4), 9, spec["recipe"]["lr"], 9
            )
            if state_hash(model, frozen_only=True) != before[name]:
                raise ValueError("Frozen tiny base changed")
        deltas = []
        references = []
        for name, value in gpu.named_parameters():
            if value.requires_grad:
                reference = dict(cpu.named_parameters())[name].detach() - initial[name]
                actual = value.detach().cpu() - initial[name]
                deltas.append((actual - reference).flatten())
                references.append(reference.flatten())
        difference, reference = torch.cat(deltas), torch.cat(references)
        if reference.norm().item() == 0:
            raise ValueError("No actual adapter update")
        relative = (difference.norm() / reference.norm()).item()
        maximum = difference.abs().max().item()
        if relative > 0.2 or maximum > 2e-5:
            raise ValueError("CUDA adapter update differs beyond frozen tolerance")
        write(
            args.out / "receipt.json",
            {
                "status": "PASS_TINY_CUDA_NUMERICAL_GATE",
                "contract_sha256": args.contract_sha,
                "runtime": runtime(),
                "inventory": inventory,
                "completed_unix_s": time.time(),
                "relative_delta_l2_error": relative,
                "absolute_delta_max_error": maximum,
                "initial_logits_max_error": error,
                "executions": executions,
                "full7B_fit_validated": False,
                "quality_or_performance_verdict": "NO_VERDICT",
            },
        )
        Console().print("PASS_TINY_CUDA_NUMERICAL_GATE")
    except Exception as error:
        write(args.out / "failure.json", {"error": repr(error), "no_retry": True})
        raise


if __name__ == "__main__":
    main()
