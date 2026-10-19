"""Short, fail-closed pretrained 7B screen through Soup's RAM streaming runtime."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from d1_final_f1 import configure, route
from d1_rules import all_block_reads_check, compare_tensors, digest, peer_check, write_new_json
from rich.console import Console

ROOT = Path(__file__).parents[2]
RULE = ROOT / "benchmarks/d1-real7b-rule-20261019.md"
MODEL = "mistralai/Mistral-7B-v0.3"
REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"
TEXT = """A small engineering team is building a tool for training language models.
The team stores frozen model weights in host memory and moves one layer at a time
to a graphics processor. This makes large models accessible on smaller computers.
However, reducing memory usage can increase training time. A useful experiment
measures both costs on the same workload, checks gradients against a reference,
and saves the original observations. A short test can establish numerical
correctness without proving the quality of a model after weeks of training.
The engineers compare complete optimizer steps rather than isolated operations.
They keep the input text, initial parameters, precision and hardware identical.
Results are useful even when an optimization fails to improve the product.
"""


def prepare(data: Path) -> dict:
    torch = configure()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    data.mkdir()
    weights = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            allow_patterns=["*.json", "model-*.safetensors", "tokenizer.model*"],
            max_workers=4,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(str(weights), trust_remote_code=False)
    tokens = tokenizer(TEXT, return_tensors="pt")["input_ids"]
    if tokens.shape[1] < 128:
        raise RuntimeError("natural text contains fewer than128 tokens")
    inputs = tokens[:, :128].contiguous()
    torch.save(inputs, data / "inputs.pt")
    probe = build_meta_skeleton(str(weights), dtype="bfloat16", quant="nf4")
    suffixes = quantised_layer_suffixes(probe)
    del probe
    index = shard_checkpoint(
        str(weights),
        str(data / "shards"),
        dtype="bfloat16",
        arch="mistral",
        quant="nf4",
        quant_suffixes=suffixes,
        quant_device="cuda",
        notify=Console().print,
    )
    result = {
        "model_id": MODEL,
        "revision": REVISION,
        "weights_dir": str(weights),
        "layers": index.n_layers,
        "inputs_sha256": digest(data / "inputs.pt"),
        "model_file_hashes": {
            path.name: digest(path) for path in sorted(weights.glob("*.safetensors"))
        },
        "config_sha256": digest(weights / "config.json"),
        "sequence": 128,
        "tier": "ram",
        "pin": False,
        "buffers": 2,
        "lora_initialization": "standard zero B, seeded A; no randomized B",
    }
    if result["layers"] != 32:
        raise RuntimeError("unexpected model layer count")
    write_new_json(data / "prepared.json", result)
    return result


def model_and_runtime(data: Path) -> tuple[Any, Any]:
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    prepared = json.loads((data / "prepared.json").read_text())
    return build_streamed_model(
        model_id=prepared["weights_dir"],
        shard_dir=str(data / "shards"),
        index=read_shard_index(str(data / "shards")),
        lora_config=LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
        device="cuda",
        dtype="bfloat16",
        buffers=2,
        pin=False,
        seed=842,
        quant="nf4",
        tier="ram",
    )


def step(model: Any, optimizer: Any, inputs: Any, capture: bool = False) -> dict:
    import torch

    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids=inputs, use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), inputs[:, 1:].reshape(-1)
    )
    loss.backward()
    optimizer.step()
    if not capture:
        return {}
    return {
        "logits": logits.detach().cpu().clone(),
        "loss": loss.detach().cpu().clone(),
        **{
            f"parameter:{key.replace('.inner.', '.')}": value.detach().cpu().clone()
            for key, value in model.named_parameters()
            if value.requires_grad
        },
        **{
            f"gradient:{key.replace('.inner.', '.')}": value.grad.detach().cpu().clone()
            for key, value in model.named_parameters()
            if value.requires_grad
        },
    }


def child(args: argparse.Namespace) -> dict:
    torch = configure()
    from d1_nf4_study import timed_steps

    counts = route(args.route)
    model, runtime = model_and_runtime(args.data)
    model.train()
    inputs = torch.load(args.data / "inputs.pt", weights_only=True).cuda()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)
    try:
        if args.command == "correctness":
            snapshots = [step(model, optimizer, inputs, True) for _ in range(3)]
            torch.save(snapshots, args.output / "witness.pt")
            result = {
                "dispatch_counts": counts,
                "witness_sha256": digest(args.output / "witness.pt"),
                "losses": [float(row["loss"]) for row in snapshots],
                "nonzero_gradients": [
                    sum(
                        bool(value.count_nonzero())
                        for key, value in row.items()
                        if key.startswith("gradient:")
                    )
                    for row in snapshots
                ],
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        else:
            measured = timed_steps(lambda: step(model, optimizer, inputs), 20, 5)
            value = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
            dense = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
            reference = timed_steps(
                lambda: torch.nn.functional.linear(value, dense),
                200,
                20,
                warmup_seconds=1,
                timed_seconds=1,
            )
            result = {
                **measured,
                "dense_gemm_reference": reference,
                "reference_reduced_precision": True,
                "dispatch_counts": counts,
                "tokens_per_second": 128000 / measured["median_step_ms"],
            }
        result.update(
            route=args.route,
            source_sha=os.environ["D1_SOURCE_SHA"],
            rule_sha256=digest(RULE),
            model_revision=REVISION,
        )
        write_new_json(args.output / "result.json", result)
        if args.route == "candidate" and not counts["candidate"]:
            raise RuntimeError("candidate path did not execute")
        return result
    finally:
        runtime.close()


def invoke(command: str, data: Path, output: Path, mode: str) -> None:
    output.mkdir(parents=True)
    with (output / "stdout.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                __file__,
                command,
                "--data",
                str(data),
                "--output",
                str(output),
                "--route",
                mode,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run(args: argparse.Namespace) -> dict:
    torch = configure()
    from arm_validity import ArmWatch, arm_outcome
    from d1_nf4_study import GPUMonitor, block_readings
    from d1_reference import validate_reference

    # configure() does not create a CUDA context in this controller.
    for mode in ("native", "candidate"):
        invoke("correctness", args.data, args.output / mode, mode)
    bundles = [
        torch.load(args.output / mode / "witness.pt", weights_only=True)
        for mode in ("native", "candidate")
    ]
    comparisons = [compare_tensors(b, a) for a, b in zip(*bundles, strict=True)]
    gate = {"passed": all(row["passed"] for row in comparisons), "updates": comparisons}
    write_new_json(args.output / "correctness.json", gate)
    if not gate["passed"]:
        raise RuntimeError("real7B parity failed; timing prohibited")
    monitor, rows = GPUMonitor(), []
    for index, arm in enumerate("ABBA"):
        output = args.output / f"arm-{index}-{arm}"
        before = monitor.sample()
        if not before["available"] or before.get("compute") or before.get("graphics"):
            raise RuntimeError("occupied/unknown pre-context GPU")
        peers, stop = [], threading.Event()

        def sample() -> None:
            while not stop.is_set():
                peers.append(monitor.sample())
                stop.wait(0.1)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        watch = ArmWatch(
            arm=arm,
            round=index // 2,
            run=index,
            model_path=str(args.data),
            required=("power", "suspend", "foreign_reads"),
        )
        block_before = block_readings()
        try:
            with watch:
                invoke("arm", args.data, output, "native" if arm == "A" else "candidate")
        finally:
            stop.set()
            thread.join(timeout=10)
        after, block_after = monitor.sample(), block_readings()
        validity = watch.record
        foreign = validity["checks"]["foreign_reads"]
        if foreign["outcome"] == "unknown" and "maps to no block device" in foreign["reason"]:
            checks = {
                **validity["checks"],
                "all_block_reads": all_block_reads_check(block_before, block_after),
            }
            required = ["power", "suspend", "all_block_reads"]
            validity = {
                **validity,
                "checks": checks,
                "required": required,
                "outcome": arm_outcome(checks, required),
            }
        row = json.loads((output / "result.json").read_text())
        row.update(
            arm=arm,
            run=index,
            validity=validity,
            peer_lifecycle={"before": before, "during": peers, "after": after},
            peers_clean=peer_check(before, peers, after, row["gpu_uuid"]),
        )
        row["reference_validation"] = validate_reference(row, row["dense_gemm_reference"])
        write_new_json(output / "watched.json", row)
        rows.append(row)
        Console().print(
            {
                "arm": arm,
                "validity": validity["outcome"],
                "median_step_ms": row["median_step_ms"],
                "allocated": row["peak_allocated_bytes"],
            }
        )
    windows = [window for row in rows for window in (row, row["dense_gemm_reference"])]
    clocks = [clock for row in windows for clock in row["clock_sm_mhz"]]
    limits = [limit for row in windows for limit in row["power_limit_mw"]]
    clean = (
        bool(clocks)
        and min(clocks) > 0
        and max(clocks) <= min(clocks) * 1.05
        and len(set(limits)) == 1
        and all(
            row["validity"]["outcome"] == "ok"
            and row["peers_clean"]
            and row["reference_validation"]["passed"]
            for row in rows
        )
    )
    result = {
        "status": "CLEAN_DESCRIPTIVE_SCREEN" if clean else "NO_VERDICT_TIMING",
        "formal_f1": False,
        "quality_evaluation": False,
        "clock_range_mhz": [min(clocks), max(clocks)] if clocks else [],
        "median_ms": {
            arm: statistics.median(row["median_step_ms"] for row in rows if row["arm"] == arm)
            for arm in "AB"
        },
        "peak_allocated_bytes": {
            arm: max(row["peak_allocated_bytes"] for row in rows if row["arm"] == arm)
            for arm in "AB"
        },
        "peak_reserved_bytes": {
            arm: max(row["peak_reserved_bytes"] for row in rows if row["arm"] == arm)
            for arm in "AB"
        },
        "rows": rows,
    }
    write_new_json(args.output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "correctness", "arm", "run"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("native", "candidate"), default="native")
    args = parser.parse_args()
    if args.command == "prepare":
        Console().print(prepare(args.data))
    elif args.command == "run":
        Console().print(run(args))
    else:
        Console().print(child(args))
