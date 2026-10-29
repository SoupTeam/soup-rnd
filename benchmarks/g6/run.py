"""G6 offline entrypoint: audit plans, CPU smoke, prospective gated CUDA pilot/full arm.

No download, rental, server control, automatic retry, or optimizer resume.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from check_consumer import state_hash
from checkpoint_eval import load_adapter, sha256
from consumer import audit_plan, inspect_source, source_rows, windows
from engine import audit_runtime, evaluate, train
from eval_suite import generate_suite
from prepare import make_order_study_arms, take_budget
from rich.console import Console
from run_support import (
    Guard,
    admit,
    contract,
    gpu_inventory,
    harness,
    preceding_initial,
    runtime,
    verify_model,
)


def write(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def base_model(spec: dict, model_dir: Path | None, synthetic: bool) -> object:
    import torch
    from transformers import AutoModelForCausalLM, MistralConfig, MistralForCausalLM

    torch.manual_seed(spec["recipe"]["seed"])
    if synthetic:
        config = MistralConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            attention_dropout=0.0,
            sliding_window=None,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = spec["recipe"]["attention"]
        model = MistralForCausalLM(config).cpu()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16,
            attn_implementation=spec["recipe"]["attention"],
            device_map=None,
        ).to("cuda:0")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def build_model(spec: dict, model_dir: Path | None, synthetic: bool) -> object:
    import torch
    from peft import LoraConfig, get_peft_model

    model = base_model(spec, model_dir, synthetic)
    torch.manual_seed(spec["recipe"]["seed"])
    settings = spec["recipe"]["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=settings["r"],
            lora_alpha=settings["alpha"],
            target_modules=settings["modules"],
            lora_dropout=0,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model


def tokenizer_for(spec: dict, model_dir: Path) -> object:
    from prepare import digest
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    if (
        digest(tokenizer.get_vocab()) != spec["tokenizer"]["vocab_sha256"]
        or digest(tokenizer.special_tokens_map) != spec["tokenizer"]["special_sha256"]
    ):
        raise ValueError("Tokenizer differs from data preparation")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    return tokenizer


def phase(
    out: Path,
    name: str,
    guard: Guard,
    seconds: int,
    function: object,
    *,
    gpu: bool,
    model_dir: Path | None,
    arm: str,
) -> dict:
    guard.stage(seconds)
    started = time.monotonic()
    watch = (
        harness().ArmWatch(arm=arm, round=1, run=1, label=name, model_path=str(model_dir))
        if gpu
        else None
    )
    try:
        if watch:
            watch.start()
        result = function()
        guard()
        return result
    finally:
        record = watch.stop() if watch else {"outcome": "unknown", "reason": "CPU synthetic"}
        write(out / (name + "_phase.json"), {"elapsed_s": time.monotonic() - started, "F1": record})


def execute_arm(
    spec: dict,
    data: list[dict],
    evaluation: list[dict],
    out: Path,
    guard: Guard,
    *,
    model_dir: Path | None,
    synthetic: bool,
    budget: int,
    lr_budget: int,
    arm: str,
    suite: dict,
    expected_initial: str | None = None,
) -> dict:
    import torch
    from peft import PeftModel
    from safetensors.torch import save_file

    device = "cpu" if synthetic else "cuda:0"
    limits = spec["limits"]
    guard.stage(limits["setup_s"])
    setup_started = time.monotonic()
    model = build_model(spec, model_dir, synthetic)
    guard()
    tokenizer = None if synthetic else tokenizer_for(spec, model_dir)
    initial, frozen = state_hash(model), state_hash(model, frozen_only=True)
    if expected_initial is not None and initial != expected_initial:
        raise ValueError("Initial model differs from preceding/prospective pilot identity")
    write(
        out / "initial_identity.json",
        {
            "initial_state_sha256": initial,
            "frozen_base_sha256": frozen,
            "setup_s": time.monotonic() - setup_started,
        },
    )
    targets = sum(len(row["input_ids"]) - 1 for row in evaluation)

    def ce(name: str) -> dict:
        return phase(
            out,
            name,
            guard,
            limits["eval_s"],
            lambda: evaluate(
                model,
                evaluation,
                expected_targets=targets,
                trace=out / (name + ".jsonl"),
                device=device,
                guard=guard,
            ),
            gpu=not synthetic,
            model_dir=model_dir,
            arm=arm,
        )

    initial_eval = ce("initial_eval")
    if tokenizer:
        phase(
            out,
            "initial_diagnostics",
            guard,
            limits["diagnostic_s"],
            lambda: generate_suite(
                model, tokenizer, suite, out / "initial_diagnostics.jsonl", device, guard
            ),
            gpu=True,
            model_dir=model_dir,
            arm=arm,
        )

    def plan():
        size = spec["recipe"]["targets_per_update"] if not synthetic else 16
        return windows(data, budget, size)

    write(
        out / "consumption_plan.json",
        audit_plan(plan(), budget, spec["recipe"]["targets_per_update"] if not synthetic else 16),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=spec["recipe"]["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        foreach=False,
        fused=False,
    )
    training = phase(
        out,
        "train",
        guard,
        limits["train_s"],
        lambda optimizer=optimizer: train(
            model,
            optimizer,
            plan(),
            budget=budget,
            base_lr=spec["recipe"]["lr"],
            trace=out / "actual_trace.jsonl",
            telemetry=out / "training_telemetry.jsonl",
            device=device,
            guard=guard,
            clip=1,
            lr_budget=lr_budget,
        ),
        gpu=not synthetic,
        model_dir=model_dir,
        arm=arm,
    )
    audit = audit_runtime(
        out / "actual_trace.jsonl", plan(), budget, spec["recipe"]["lr"], lr_budget
    )
    if state_hash(model, frozen_only=True) != frozen or state_hash(model) == initial:
        raise ValueError("Base changed or adapter did not change")
    final_eval = ce("final_eval")
    if tokenizer:
        diagnostics = phase(
            out,
            "final_diagnostics",
            guard,
            limits["diagnostic_s"],
            lambda: generate_suite(
                model, tokenizer, suite, out / "final_diagnostics.jsonl", device, guard
            ),
            gpu=True,
            model_dir=model_dir,
            arm=arm,
        )
    else:
        diagnostics = {"status": "NOT_SCORED_SYNTHETIC_CPU", "safety": "NOT_EVALUATED"}
    guard.stage(limits["setup_s"])
    saved = time.monotonic()
    final = state_hash(model)
    checkpoint = out / "adapter.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.named_parameters()
            if value.requires_grad
        },
        str(checkpoint),
    )
    model.save_pretrained(out / "peft_adapter", safe_serialization=True)
    write(
        out / "checkpoint.json",
        {
            "raw_sha256": sha256(checkpoint),
            "final_state": final,
            "frozen_base": frozen,
            "optimizer_state_saved": False,
            "resume_supported": False,
            "peft_files": {
                p.name: sha256(p) for p in (out / "peft_adapter").iterdir() if p.is_file()
            },
        },
    )
    del optimizer, model
    gc.collect()
    if not synthetic:
        torch.cuda.empty_cache()
    model = build_model(spec, model_dir, synthetic)
    load_adapter(model, checkpoint, sha256(checkpoint), frozen)
    if state_hash(model) != final:
        raise ValueError("Strict raw reload state differs")
    raw_reload = ce("raw_reload_eval")
    if raw_reload["mean_ce"] != final_eval["mean_ce"]:
        raise ValueError("Raw reload eval differs")
    del model
    gc.collect()
    if not synthetic:
        torch.cuda.empty_cache()
    guard.stage(limits["setup_s"])
    base = base_model(spec, model_dir, synthetic)
    model = PeftModel.from_pretrained(
        base,
        out / "peft_adapter",
        is_trainable=True,
        local_files_only=True,
        autocast_adapter_dtype=True,
    )
    if state_hash(model) != final or state_hash(model, frozen_only=True) != frozen:
        raise ValueError("Standard PEFT reload differs")
    peft_reload = ce("peft_reload_eval")
    if peft_reload["mean_ce"] != final_eval["mean_ce"]:
        raise ValueError("PEFT reload eval differs")
    if tokenizer:
        phase(
            out,
            "reload_diagnostics",
            guard,
            limits["diagnostic_s"],
            lambda: generate_suite(
                model, tokenizer, suite, out / "reload_diagnostics.jsonl", device, guard
            ),
            gpu=True,
            model_dir=model_dir,
            arm=arm,
        )
        if (out / "final_diagnostics.jsonl").read_bytes() != (
            out / "reload_diagnostics.jsonl"
        ).read_bytes():
            raise ValueError("Reload generated tokens/text/scores differ")
    phase_records = [json.loads(path.read_text()) for path in out.glob("*_phase.json")]
    valid_timing = not synthetic and all(
        row["F1"]["outcome"] == "ok" and row["F1"].get("gpu_clock", {}).get("outcome") == "ok"
        for row in phase_records
    )
    result = {
        "status": "CPU_SYNTHETIC_PIPELINE_PASS" if synthetic else "ARM_COMPLETED_REVIEW_PENDING",
        "initial_state_sha256": initial,
        "final_state_sha256": final,
        "frozen_base_unchanged": True,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "raw_reload_eval": raw_reload,
        "peft_reload_eval": peft_reload,
        "reload_exact": True,
        "training": training,
        "runtime_audit": audit,
        "diagnostics": diagnostics,
        "checkpoint_and_reload_s": time.monotonic() - saved,
        "F1_timing_accepted": valid_timing,
        "quality_verdict": "NO_VERDICT",
        "CPU_evidence_does_not_validate_CUDA_AMP": True,
    }
    write(out / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "cpu-smoke", "pilot", "full"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--cuda-gate", type=Path)
    parser.add_argument("--pilot-summary", type=Path)
    args = parser.parse_args()
    spec = contract(args.contract, args.contract_sha)
    spec["launch_contract_sha256"] = args.contract_sha
    if args.mode in {"pilot", "full"}:
        if (
            not args.arm
            or args.arm not in spec["arm_order"]
            or not args.admission
            or not args.model_dir
        ):
            parser.error("CUDA requires arm, offline model-dir and fresh account admission")
        receipt = json.loads(args.admission.read_text())
        if not args.cuda_gate:
            parser.error("A successful tiny CUDA numerical gate is required")
        gate = json.loads(args.cuda_gate.read_text())
        if (
            gate.get("status") != "PASS_TINY_CUDA_NUMERICAL_GATE"
            or gate.get("contract_sha256") != args.contract_sha
            or gate.get("runtime") != spec["runtime_target"]
            or gate.get("inventory", {}).get("uuid") != receipt["gpu_uuid"]
        ):
            parser.error("CUDA numerical gate is missing or belongs to another setup")
        if args.mode == "full":
            if not args.pilot_summary:
                parser.error("Four-arm pilot review is required before a full arm")
            pilot = json.loads(args.pilot_summary.read_text())
            if (
                pilot.get("mode") != "pilot"
                or pilot.get("contract_sha256") != args.contract_sha
                or pilot.get("F1_all_accepted") is not True
                or set(pilot.get("rows", {})) != set(spec["arm_order"])
            ):
                parser.error("Four valid matched pilot arms are required")
        admit(receipt, spec, args.mode)
        if os.environ.get("SOUP_G6_INSTANCE_ID") != str(receipt["instance_id"]):
            parser.error("Instance identity differs from explicit admission")
    expected_initial = None
    if args.mode in {"pilot", "full"}:
        claim = Path(__file__).parent / "claims" / args.contract_sha
        expected_initial = preceding_initial(claim, spec, args.mode, args.arm)
        if args.mode == "full":
            if expected_initial and expected_initial != pilot["initial_state_sha256"]:
                parser.error("Full arm initial state differs from pilot")
            expected_initial = pilot["initial_state_sha256"]
    args.out.mkdir(parents=True, exist_ok=False)
    write(
        args.out / "prospective_run.json",
        {"mode": args.mode, "contract_sha": args.contract_sha, "arm": args.arm, "resume": False},
    )
    try:
        if sha256(args.data / "manifest.json") != spec["data"]["manifest_sha256"]:
            raise ValueError("Data manifest mismatch")
        suite = json.loads((Path(__file__).parent / "spec/eval_suite_v1.json").read_text())
        if args.mode == "plan":
            plans = {}
            for arm in spec["arm_order"]:
                source = inspect_source(args.data, arm, spec["data"]["manifest_sha256"])
                plans[arm] = audit_plan(
                    windows(source_rows(Path(source["path"])), 5_000_000, 8192), 5_000_000, 8192
                )
            write(args.out / "plans.json", plans)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch

            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            synthetic = args.mode == "cpu-smoke"
            guard = Guard(
                args.out,
                spec["limits"][args.mode + "_total_s"],
                gpu=not synthetic,
                uuid=None if synthetic else receipt["gpu_uuid"],
            )
            write(args.out / "runtime.json", runtime())
            if synthetic:
                rows = [
                    {
                        "id": f"cpu-{index}",
                        "input_ids": [
                            1,
                            *[(index + offset) % 59 + 3 for offset in range(index + 2)],
                            2,
                        ],
                    }
                    for index in range(16)
                ]
                scores = {
                    row["id"]: 0.6 if index % 4 == 0 else 0.8 for index, row in enumerate(rows)
                }
                data = make_order_study_arms(rows, scores, budget=83, seed=901)
                evaluation = [
                    {"id": "probe-short", "input_ids": [1, 61, 60, 2]},
                    {"id": "probe-long", "input_ids": [1, 62, 60, 59, 58, 57, 2]},
                ]
                budget = lr_budget = 83
                arms = spec["arm_order"]
                write(args.out / "synthetic_inputs.json", {"train": data, "eval": evaluation})
            else:
                expected_runtime = spec["runtime_target"]
                if runtime() != expected_runtime:
                    raise ValueError("Runtime differs from prospective package pins")
                inventory = gpu_inventory(receipt["gpu_uuid"])
                if inventory != gate["inventory"]:
                    raise ValueError("GPU/driver differs from numerical gate")
                write(args.out / "inventory.json", inventory)
                guard()
                write(
                    args.out / "model_verification.json",
                    verify_model(args.model_dir, spec["model_files"], guard),
                )
                guard()
                # Global per-contract stage/arm claim: a failed launch cannot be redeployed.
                claim = Path(__file__).parent / "claims" / args.contract_sha
                claim.mkdir(parents=True, exist_ok=True)
                write(claim / f"{args.mode}_{args.arm}.json", {"out": str(args.out)})
                source = inspect_source(args.data, args.arm, spec["data"]["manifest_sha256"])
                selected = list(source_rows(Path(source["path"])))
                budget = 65536 if args.mode == "pilot" else 5_000_000
                data = {args.arm: take_budget(selected, budget)}
                evaluation_path = args.data / "eval.jsonl"
                if sha256(evaluation_path) != spec["data"]["eval_sha256"]:
                    raise ValueError("Eval bytes mismatch")
                evaluation = list(source_rows(evaluation_path))
                lr_budget = 5_000_000
                arms = [args.arm]
                write(args.out / "admission.json", receipt)
            results = {}
            for arm in arms:
                directory = args.out / arm
                directory.mkdir(exist_ok=False)
                results[arm] = execute_arm(
                    spec,
                    data[arm],
                    evaluation,
                    directory,
                    guard,
                    model_dir=args.model_dir,
                    synthetic=synthetic,
                    budget=budget,
                    lr_budget=lr_budget,
                    arm=arm,
                    suite=suite,
                    expected_initial=expected_initial,
                )
            if synthetic and len({row["initial_state_sha256"] for row in results.values()}) != 1:
                raise ValueError("CPU arms did not start identically")
            write(
                args.out / "receipt.json",
                {
                    "status": "CPU_PIPELINE_PASS" if synthetic else "COMPLETED_REVIEW_PENDING",
                    "arms": results,
                    "Mistral7B_training": not synthetic,
                    "quality_verdict": "NO_VERDICT",
                },
            )
        write(
            args.out / "files_sha256.json",
            {
                str(path.relative_to(args.out)): sha256(path)
                for path in sorted(args.out.rglob("*"))
                if path.is_file()
            },
        )
        Console().print({"mode": args.mode, "out": str(args.out), "complete": True})
    except Exception as error:
        write(
            args.out / "failure.json",
            {
                "error": repr(error),
                "partial_outputs_retained": True,
                "no_retry": True,
                "quality_verdict": "NO_VERDICT",
            },
        )
        raise


if __name__ == "__main__":
    main()
