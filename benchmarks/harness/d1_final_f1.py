"""Bounded fresh-process ABBA test of the scoped D1 backend through Soup."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from d1_rules import (
    all_block_reads_check,
    compare_tensors,
    digest,
    peer_check,
    settings_digest,
    summarize,
    write_new_json,
)
from rich.console import Console

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
RULE = ROOT / "benchmarks/d1-final-f1-rule-20261016.md"


def configure() -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(842)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    return torch


def route(mode: str) -> dict[str, int]:
    import d1_fused_nf4 as candidate

    from soup_cli.utils import layer_stream_runtime as runtime

    original_linear = runtime.checkpoint_visible_nf4_linear
    original_gate = runtime._can_use_checkpoint_visible_nf4_gemm
    counts = {"candidate": 0, "native_custom": 0}

    def covered(x: Any, state: Any) -> bool:
        return x.dtype == state.dtype and (x.numel() // x.shape[-1], *map(int, state.shape)) in {
            (128, 4096, 4096),
            (128, 64, 64),
            (2048, 64, 64),
            (128, 32768, 12288),
        }

    def selected(x: Any, packed: Any, state: Any, bias: Any = None) -> Any:
        if mode != "native" and covered(x, state):
            counts["candidate"] += 1
            return candidate.linear(x, packed, state, bias, backend="wmma_native64")
        counts["native_custom"] += 1
        return original_linear(x, packed, state, bias)

    runtime.checkpoint_visible_nf4_linear = selected
    runtime._can_use_checkpoint_visible_nf4_gemm = lambda x, state: (
        (mode != "native" and covered(x, state)) or original_gate(x, state)
    )
    return counts


def build_model(directory: Path) -> tuple[Any, Any]:
    from test_v07202 import _randomise_lora_b, _tiny_lora

    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    model, runtime = build_streamed_model(
        model_id=str(directory / "model"),
        shard_dir=str(directory / "shards"),
        index=read_shard_index(str(directory / "shards")),
        lora_config=_tiny_lora(),
        device="cuda",
        dtype="bfloat16",
        buffers=2,
        pin=False,
        seed=3,
        quant="nf4",
    )
    _randomise_lora_b(model)
    model.train()
    return model, runtime


def training_step(model: Any, optimizer: Any, inputs: Any, capture: bool = False) -> dict:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids=inputs, use_cache=False).logits
    loss = logits.float().square().mean()
    loss.backward()
    optimizer.step()
    if not capture:
        return {}
    return {
        "logits": logits.detach().cpu().clone(),
        "loss": loss.detach().cpu().clone(),
        **{
            f"parameter:{key.replace('.inner.', '.')}": p.detach().cpu().clone()
            for key, p in model.named_parameters()
            if p.requires_grad
        },
        **{
            f"gradient:{key.replace('.inner.', '.')}": p.grad.detach().cpu().clone()
            for key, p in model.named_parameters()
            if p.requires_grad
        },
    }


def prepare(directory: Path) -> dict:
    torch = configure()
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=4096,
        intermediate_size=4096,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=8,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = directory / "model"
    weights.mkdir()
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    del state, model
    probe = build_meta_skeleton(str(weights), dtype="bfloat16", quant="nf4")
    suffixes = quantised_layer_suffixes(probe)
    del probe
    shard_checkpoint(
        str(weights),
        str(directory / "shards"),
        dtype="bfloat16",
        arch="llama",
        quant="nf4",
        quant_suffixes=suffixes,
        quant_device="cuda",
    )
    # Separate processes preserve native dispatch and identical initialization.
    for mode in ("native", "candidate"):
        subprocess.run(
            [
                sys.executable,
                __file__,
                "correctness",
                "--data",
                str(directory),
                "--route",
                mode,
                "--output",
                str(directory / f"{mode}.pt"),
            ],
            check=True,
        )
    bundles = [
        torch.load(directory / f"{mode}.pt", map_location="cpu", weights_only=True)
        for mode in ("native", "candidate")
    ]
    comparisons = [
        compare_tensors(actual, native)
        for actual, native in zip(bundles[1], bundles[0], strict=True)
    ]
    result = {
        "passed": all(row["passed"] for row in comparisons),
        "comparisons": comparisons,
        "seed": 842,
        "synthetic_weights": True,
        "hidden": 4096,
        "layers": 4,
        "model_sha256": digest(weights / "model.safetensors"),
    }
    write_new_json(directory / "correctness.json", result)
    if not result["passed"]:
        raise RuntimeError("target-shape full-step correctness failed; no timing")
    subprocess.run(
        [
            sys.executable,
            __file__,
            "resident",
            "--data",
            str(directory),
            "--route",
            "candidate",
            "--output",
            str(directory / "resident-gate"),
        ],
        check=True,
    )
    return result


def child(args: argparse.Namespace) -> dict:
    torch = configure()
    from d1_nf4_study import timed_steps

    counts = route(args.route)
    model, runtime = build_model(args.data)
    inputs = torch.randint(0, 64, (1, 128), device="cuda")
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)
    try:
        if args.command == "dispatch":
            actual = training_step(model, optimizer, inputs, True)
            expected = torch.load(args.data / "native.pt", map_location="cpu", weights_only=True)[0]
            comparison = compare_tensors(actual, expected)
            result = {
                "passed": comparison["passed"] and counts["candidate"] > 0,
                "comparison": comparison,
                "dispatch_counts": counts,
                "timing_run": False,
                "source_sha": os.environ["D1_SOURCE_SHA"],
            }
            write_new_json(args.output / "result.json", result)
            if not result["passed"]:
                raise RuntimeError("scoped candidate dispatch proof failed")
            return result
        if args.command == "resident":
            from test_v07202 import _randomise_lora_b, _resident_nf4, _sync_adapters

            resident = _resident_nf4(str(args.data / "model"), dtype="bfloat16", device=0)
            resident.train()
            _randomise_lora_b(resident)
            _sync_adapters(model, resident)
            resident_optimizer = torch.optim.SGD(
                [p for p in resident.parameters() if p.requires_grad], lr=0.001
            )
            comparisons, witnesses = [], []
            for _ in range(3):
                expected = training_step(resident, resident_optimizer, inputs, True)
                actual = training_step(model, optimizer, inputs, True)
                comparisons.append(compare_tensors(actual, expected))
                witnesses.append({"resident": expected, "candidate": actual})
            torch.save(witnesses, args.output / "witness.pt")
            result = {
                "passed": all(row["passed"] for row in comparisons),
                "comparisons": comparisons,
                "witness_sha256": digest(args.output / "witness.pt"),
                "dispatch_counts": counts,
            }
            write_new_json(args.output / "result.json", result)
            if not result["passed"]:
                raise RuntimeError("unpatched resident parity failed; no timing")
            return result
        if args.command == "correctness":
            tensors = [training_step(model, optimizer, inputs, True) for _ in range(3)]
            torch.save(tensors, args.output)
            return {"saved": True}
        measured = timed_steps(lambda: training_step(model, optimizer, inputs), 200, 20)
        # Post-primary allocation keeps the dense reference out of peak accounting.
        primary_peaks = {
            key: measured[key] for key in ("peak_allocated_bytes", "peak_reserved_bytes")
        }
        dense = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
        value = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
        reference = timed_steps(
            lambda: torch.nn.functional.linear(value, dense),
            200,
            20,
            warmup_seconds=1,
            timed_seconds=1,
        )
        result = {
            **measured,
            **primary_peaks,
            "status": "MEASURED",
            "path": args.route,
            "source_sha": os.environ["D1_SOURCE_SHA"],
            "rule_sha256": digest(RULE),
            "settings_sha256": settings_digest(
                {
                    "hidden": 4096,
                    "layers": 4,
                    "seq": 128,
                    "steps": 200,
                    "warmup": 20,
                    "dtype": "bfloat16",
                    "nested": True,
                    "sgd": 0.001,
                }
            ),
            "dense_gemm_reference": reference,
            "reference_reduced_precision": True,
            "tokens_per_second": 128000 / measured["median_step_ms"],
            "dispatch_counts": counts,
        }
        if args.route == "candidate" and not counts["candidate"]:
            raise RuntimeError("no experimental calls; cannot label this a candidate arm")
        write_new_json(args.output / "result.json", result)
        return result
    finally:
        runtime.close()


def run(args: argparse.Namespace) -> dict:
    from arm_validity import ArmWatch, Run, arm_outcome, blocks_to_add
    from d1_nf4_study import GPUMonitor, block_readings

    monitor = GPUMonitor()
    rows = []
    for block in range(5):
        for index, arm in enumerate("ABBA"):
            output = args.output / f"block-{block}-run-{index}-{arm}"
            output.mkdir()
            command = [
                sys.executable,
                __file__,
                "arm",
                "--data",
                str(args.data),
                "--route",
                "native" if arm == "A" else "candidate",
                "--output",
                str(output),
            ]
            before = monitor.sample()
            if not before["available"] or before.get("compute") or before.get("graphics"):
                raise RuntimeError("GPU pre-context occupied/unknown")
            peers, stop = [], threading.Event()

            def sample() -> None:
                while not stop.is_set():
                    peers.append(monitor.sample())
                    stop.wait(0.1)

            thread = threading.Thread(target=sample, daemon=True)
            thread.start()
            watch = ArmWatch(
                arm=arm,
                round=block * 2 + index // 2,
                run=index,
                model_path=str(args.data),
                required=("power", "suspend", "foreign_reads"),
            )
            block_before = block_readings()
            with watch:
                with (output / "stdout.log").open("x") as log:
                    try:
                        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
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
            path = output / "result.json"
            record = json.loads(path.read_text()) if path.exists() else {}
            record.update(
                arm=arm,
                block=block,
                run=index,
                case="soup-h4096-bf16-native-wmma64",
                validity=validity,
                returncode=process.returncode,
                peer_lifecycle={"before": before, "during": peers, "after": after},
                peers_clean=peer_check(before, peers, after, record.get("gpu_uuid")),
            )
            write_new_json(output / "watched.json", record)
            rows.append(record)
            Console().print(
                {
                    "block": block,
                    "arm": arm,
                    "status": record.get("status"),
                    "validity": validity["outcome"],
                }
            )
            if process.returncode:
                result = {"status": "NO_VERDICT", "reason": "child failed", "rows": rows}
                write_new_json(args.output / "summary.json", result)
                return result
        provisional = summarize(rows)
        # A later invalid block also ends this cohort; never spend on known-bad timing.
        if any(item["rejected_blocks"] for item in provisional["comparisons"]):
            result = {**provisional, "rows": rows, "stopped_early": True}
            write_new_json(args.output / "summary.json", result)
            return result
    result = {**summarize(rows), "rows": rows}
    result["sizing_extra"] = blocks_to_add(
        [
            [
                Run(row["median_step_ms"], row["validity"]["outcome"])
                for row in rows
                if row["block"] == b
            ]
            for b in range(5)
        ]
    )
    write_new_json(args.output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "correctness", "resident", "dispatch", "arm", "run")
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("native", "candidate"), default="native")
    args = parser.parse_args()
    if args.command in ("prepare", "run", "resident", "dispatch"):
        args.output.mkdir(parents=True, exist_ok=False)
    if args.command == "prepare":
        args.data.mkdir(parents=True, exist_ok=False)
        prepare(args.data)
    elif args.command == "run":
        run(args)
    else:
        child(args)
