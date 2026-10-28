"""One isolated C3 calibration arm; historical compare.py remains frozen.

The workload mirrors its synthetic pilot. Each process builds its own model,
source and optimizer; the parent must verify identity and A/B numerical parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from benchmarks.c3.compare import ReadCounter, save, tensor_digest


def run(destination: Path, mode: str, arm: str) -> dict[str, Any]:
    if arm not in ("A", "B"):
        raise ValueError("arm must be A or B")
    if mode not in ("cpu-smoke", "cuda-pilot"):
        raise ValueError("unknown comparison mode")
    if destination.exists():
        raise FileExistsError("output must be a new directory; no retry or overwrite")
    destination.mkdir(parents=True)
    try:
        return _run(destination, mode, arm)
    except BaseException as error:
        receipt = destination / "result.json"
        result = json.loads(receipt.read_text()) if receipt.exists() else {"mode": mode}
        result.update(status="FAILED", error=repr(error), performance_verdict="NO_VERDICT")
        save(receipt, result)
        raise


def _run(destination: Path, mode: str, arm: str) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType
    from transformers import LlamaConfig, LlamaForCausalLM

    from benchmarks.c3.io_calibration import difference, snapshot
    from benchmarks.harness.arm_validity import ArmWatch
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.partial_ram_source import PartialRamBudget

    cuda = mode == "cuda-pilot"
    device = "cuda:0" if cuda else "cpu"
    if cuda and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("the pilot requires exactly one CUDA device")
    ownership = None
    if cuda:
        from benchmarks.harness.l2l_box import confirm_gpu_owner, gpu_compute_pids

        before = gpu_compute_pids()
        if before != []:
            raise RuntimeError(f"GPU must be observably idle before initialization: {before!r}")
        context_probe = torch.empty(1, device=device)
        torch.cuda.synchronize()
        ownership = confirm_gpu_owner(before, context_active=True, single_device=True)
        del context_probe
        if ownership["method"] != "exclusive-local-CUDA-initialization":
            raise RuntimeError(f"cannot establish GPU ownership: {ownership!r}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=256 if cuda else 64,
        hidden_size=1024 if cuda else 32,
        intermediate_size=4096 if cuda else 64,
        num_hidden_layers=8 if cuda else 4,
        num_attention_heads=8 if cuda else 4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        max_position_embeddings=256,
    )
    base = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = destination / "weights"
    base.save_pretrained(weights, safe_serialization=True)
    del base
    directory = destination / "shards"
    index = shard_checkpoint(str(weights), str(directory), dtype="float32", arch="llama")
    identity = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in weights.iterdir()
        if p.is_file()
    }
    records = []
    order = [arm]
    steps, warmup = (6, 2) if cuda else (3, 0)
    cache = 128 << 20 if cuda else 45_000
    envelope = 512 << 20 if cuda else 10_000_000
    reserved = 64 << 20 if cuda else 0
    result: dict[str, Any] = {
        "mode": mode,
        "source_model": "synthetic random dense Llama",
        "config": config.to_dict(),
        "weight_sha256": identity,
        "steps_per_arm": steps,
        "warmup_steps": warmup,
        "order": order,
        "torch": torch.__version__,
        "records": records,
        "gpu_ownership": ownership,
        "performance_verdict": "NO_VERDICT",
        "status": "RUNNING",
        "pid": os.getpid(),
        "scope": "single fresh process including source generation and cache fill",
    }
    save(destination / "result.json", result)
    for position, arm in enumerate(order):
        runtime = None
        row: dict[str, Any] = {"position": position, "arm": arm, "updates": []}
        records.append(row)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        setup_io = snapshot(directory) if cuda else None
        try:
            with ReadCounter() as reader:
                model, runtime = build_streamed_model(
                    model_id=str(weights),
                    shard_dir=str(directory),
                    index=index,
                    lora_config=LoraConfig(
                        r=4,
                        lora_alpha=8,
                        lora_dropout=0,
                        bias="none",
                        target_modules=["q_proj", "v_proj"],
                        task_type=TaskType.CAUSAL_LM,
                    ),
                    device=device,
                    dtype="float32",
                    pin=cuda,
                    seed=3,
                    tier="disk",
                    read_ahead=1,
                    partial_ram_budget=PartialRamBudget(envelope, reserved, cache)
                    if arm == "B"
                    else None,
                )
                source = runtime.source
                disk = getattr(source, "_disk", source)
                row["direct_io"] = disk.direct_io if disk is not None else True
                row["resident_indices"] = list(getattr(source, "resident_indices", ()))
                row["setup_seconds"] = time.perf_counter() - started
                row["initial_stats"] = runtime.stats()
                if cuda:
                    row["setup_physical_io"] = difference(setup_io, snapshot(directory))
                if arm == "B" and not 0 < len(row["resident_indices"]) < config.num_hidden_layers:
                    raise RuntimeError("the comparison requires strictly partial residency")
                if row["initial_stats"]["host_allocated_bytes"] > envelope - reserved:
                    raise RuntimeError("owned source allocations exceed the common envelope")
                if cuda and not row["direct_io"]:
                    raise RuntimeError("pilot refuses buffered disk fallback")
                params = {
                    name: param for name, param in model.named_parameters() if param.requires_grad
                }
                row["initial_adapter_sha256"] = tensor_digest(params)
                optimizer = torch.optim.AdamW(params.values(), lr=0.001)
                model.train()
                watch = ArmWatch(
                    arm=arm, round=position // 4, run=position, model_path=str(directory)
                )
                with watch:
                    for step in range(steps):
                        ids = (
                            (torch.arange(128 if cuda else 8).reshape(1, -1) + step)
                            % config.vocab_size
                        ).to(device)
                        optimizer.zero_grad()
                        if cuda:
                            torch.cuda.synchronize()
                        before = reader.snapshot()
                        io_before = snapshot(directory) if cuda else None
                        tick = time.perf_counter()
                        loss = model(input_ids=ids, labels=ids).loss
                        loss.backward()
                        gradients = {name: param.grad for name, param in params.items()}
                        if not torch.isfinite(loss) or not all(
                            value is not None and torch.isfinite(value).all()
                            for value in gradients.values()
                        ):
                            raise RuntimeError("non-finite loss or missing/non-finite gradient")
                        # Numerical hashing is outside the reported training time.
                        if cuda:
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - tick
                        grad_hash = tensor_digest(gradients)
                        tick = time.perf_counter()
                        optimizer.step()
                        if cuda:
                            torch.cuda.synchronize()
                        elapsed += time.perf_counter() - tick
                        after = reader.snapshot()
                        row["updates"].append(
                            {
                                "step": step,
                                "warmup": step < warmup,
                                "loss": float(loss.detach().cpu()),
                                "gradient_sha256": grad_hash,
                                "training_seconds": elapsed,
                                "range_read_calls": after["calls"] - before["calls"],
                                "range_read_bytes": after["bytes"] - before["bytes"],
                            }
                        )
                        if cuda:
                            row["updates"][-1]["physical_io"] = difference(
                                io_before, snapshot(directory)
                            )
                row["f1"] = watch.record
                row["final_adapter_sha256"] = tensor_digest(params)
                row["changed_adapter"] = (
                    row["final_adapter_sha256"] != row["initial_adapter_sha256"]
                )
                row["final_stats"] = runtime.stats()
                if cuda:
                    row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    row["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
                if cuda:
                    import resource

                    row["process_lifetime_maxrss_kib"] = resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                runtime.close()
                runtime = None
                row["range_reads_total"] = reader.snapshot()
                row["status"] = "COMPLETED"
                if position and (
                    row["final_adapter_sha256"] != records[0]["final_adapter_sha256"]
                    or [(x["loss"], x["gradient_sha256"]) for x in row["updates"]]
                    != [(x["loss"], x["gradient_sha256"]) for x in records[0]["updates"]]
                ):
                    raise RuntimeError("exact numerical comparison failed; no further arms")
                del optimizer, params, gradients, model
        except BaseException as error:
            if runtime is not None:
                runtime.close()
            row["status"] = "FAILED"
            row["error"] = repr(error)
            result["status"] = "FAILED"
            save(destination / "result.json", result)
            raise
        save(destination / "result.json", result)
    # A single arm cannot establish A/B parity; the parent compares all records.
    reference = records[0]
    matches = all(
        row["changed_adapter"]
        and row["final_adapter_sha256"] == reference["final_adapter_sha256"]
        and [(x["loss"], x["gradient_sha256"]) for x in row["updates"]]
        == [(x["loss"], x["gradient_sha256"]) for x in reference["updates"]]
        for row in records
    )
    result["adapter_changed"] = matches
    result["status"] = "ARM_COMPLETE" if matches else "NUMERICAL_FAIL"
    # F1 outcomes, repetitions, physical device attribution and memory/clock
    # review remain independent gates; a numerical pass cannot approve speed.
    save(destination / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu-smoke", "cuda-pilot"], default="cpu-smoke")
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    args = parser.parse_args()
    result = run(args.output, args.mode, args.arm)
    from rich.console import Console

    Console().print(f"{result['status']}; performance={result['performance_verdict']}")
    return 0 if result["adapter_changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
