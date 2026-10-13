"""Prepared full-model LoRA comparison; GPU results are not supplied by this script.

Each backend runs in a separate interpreter. CUDA timings include zero_grad,
forward, loss, backward and AdamW.step, and exclude loading/data transfer.
The tiny CPU fixture is a smoke test, not a representative performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rich.console import Console

try:
    from . import fast_lora_probe as probe
    from .d2_audit_timing import clock_verdict
    from .d2_isolated_timing import Monitor, lifecycle_verdict, wait_file, write_json
except ImportError:
    import fast_lora_probe as probe
    from d2_audit_timing import clock_verdict
    from d2_isolated_timing import Monitor, lifecycle_verdict, wait_file, write_json

BACKENDS = ("peft", "soup", "liger", "unsloth")
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
TARGETS = [leaf for leaves in probe.PROJECTIONS.values() for leaf in leaves]
COMMON_PACKAGES = ("torch", "transformers", "peft", "triton", "safetensors")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: Any) -> str:
    import torch

    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def environment() -> dict:
    packages = {}
    for package in (*COMMON_PACKAGES, "liger-kernel", "unsloth", "unsloth_zoo", "xformers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version, "packages": packages}


def verify_fixture(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    probe._require(manifest["profile"] in ("smoke", "tinyllama-1.1b"), "unknown profile")
    probe._require(manifest["dtype"] in ("float32", "float16", "bfloat16"), "unknown dtype")
    for name, expected in manifest["files"].items():
        candidate = os.path.realpath(root / name)
        parent = os.path.realpath(root)
        probe._require(os.path.commonpath([parent, candidate]) == parent, "fixture escape")
        probe._require(digest_file(Path(candidate)) == expected, f"fixture changed: {name}")
    return manifest


def attach_peft(model: Any, rank: int) -> Any:
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGETS,
            task_type="CAUSAL_LM",
        ),
    )


def prepare(root: Path, profile: str, dtype: str) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(792)
    model_dir = root / "model"
    if profile == "smoke":
        config = LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            use_cache=False,
            tie_word_embeddings=False,
        )
        model = LlamaForCausalLM(config).to(dtype=getattr(torch, dtype))
        model.save_pretrained(model_dir)
        tokens, rank = 16, 4
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL,
            revision=REVISION,
            local_dir=model_dir,
            allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
            local_files_only=True,
        )
        tokens, rank = 512, 16
    model = attach_peft(model, rank)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
                if "lora_B" in name:
                    parameter.normal_(std=0.02)
    adapters = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    # Fixed synthetic tokens exclude tokenizer/packing/data-loader variation.
    batch = torch.randint(3, model.config.vocab_size, (1, tokens))
    save_file({**adapters, "input_ids": batch}, str(root / "fixture.safetensors"))
    bases = {
        name: tensor_digest(parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    write_json(root / "base-digests.json", bases)
    files = {
        path.relative_to(root).as_posix(): digest_file(path)
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    write_json(
        root / "manifest.json",
        {
            "profile": profile,
            "dtype": dtype,
            "seed": 792,
            "rank": rank,
            "tokens": tokens,
            "batch": 1,
            "model": MODEL if profile != "smoke" else "random",
            "revision": REVISION if profile != "smoke" else None,
            "quantization": "none",
            "adapters": "float32",
            "checkpointing": False,
            "attention_requested": "sdpa",
            "files": files,
            "environment": environment(),
            "criterion": {"loss": "all steps equal at .3f", "per_tensor_relative_l2": 0.01},
        },
    )


def callable_name(value: Any) -> str:
    value = getattr(value, "__func__", value)
    value = getattr(value, "func", value)
    return f"{getattr(value, '__module__', '')}.{getattr(value, '__name__', type(value).__name__)}"


def restore_frozen_bases(model: Any, root: Path, expected: dict, dtype: Any) -> list[dict]:
    """Undo loader input mutations using the exact frozen checkpoint, never a tolerance.

    Unsloth can rewrite padding embeddings on load. The benchmark supplies
    canonical weights to every backend and records every restoration.
    """
    from safetensors import safe_open

    frozen = {name: value for name, value in model.named_parameters() if not value.requires_grad}
    probe._require(set(frozen) == set(expected), "backend frozen parameter set differs")
    adjustments = []
    with safe_open(root / "model/model.safetensors", framework="pt", device="cpu") as checkpoint:
        for name, parameter in frozen.items():
            actual = tensor_digest(parameter)
            if actual == expected[name]:
                continue
            key = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
            probe._require(key in checkpoint.keys(), f"canonical weight missing: {key}")
            canonical = checkpoint.get_tensor(key).to(dtype=dtype)
            probe._require(
                tensor_digest(canonical) == expected[name], f"canonical source differs: {name}"
            )
            parameter.data = canonical.to(device=parameter.device)
            adjustments.append(
                {"name": name, "loaded_sha256": actual, "restored_sha256": tensor_digest(parameter)}
            )
    return adjustments


def load_backend(
    root: Path, manifest: dict, backend: str, device: str, *, deterministic: bool = False
) -> tuple[Any, dict]:
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Unsloth explicitly requires import before Transformers/PEFT; each arm is fresh.
    if backend == "unsloth":
        from unsloth import FastLanguageModel
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    torch.manual_seed(manifest["seed"])
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dtype = getattr(torch, manifest["dtype"])
    if device.startswith("cuda"):
        probe._require(torch.cuda.device_count() == 1, "require one CUDA-visible GPU")
        probe._require(torch.cuda.get_device_capability(0)[0] >= 8, "Ampere or newer required")
        probe._require(torch.cuda.is_bf16_supported(), "native CUDA BF16 required")
        check = torch.ones((64, 64), device=device, dtype=torch.bfloat16)
        probe.require_finite(check @ check, "native BF16 preflight")
        del check
    if backend == "unsloth":
        probe._require(device.startswith("cuda"), "Unsloth requires CUDA in this protocol")
        model, _ = FastLanguageModel.from_pretrained(
            model_name=str((root / "model").absolute()),
            max_seq_length=manifest["tokens"],
            dtype=dtype,
            load_in_4bit=False,
            load_in_8bit=False,
            fast_inference=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=manifest["rank"],
            lora_alpha=2 * manifest["rank"],
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGETS,
            use_gradient_checkpointing=False,
            random_state=manifest["seed"],
        )
        FastLanguageModel.for_training(model, use_gradient_checkpointing=False)
    else:
        if backend == "liger":
            from liger_kernel.transformers import apply_liger_kernel_to_llama

            apply_liger_kernel_to_llama(
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=False,
                fused_linear_cross_entropy=True,
            )
        from transformers import AutoModelForCausalLM

        model = attach_peft(
            AutoModelForCausalLM.from_pretrained(
                root / "model",
                dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=True,
            ),
            manifest["rank"],
        ).to(device)
    model.config.use_cache = False
    model.train()
    fixture = load_file(str(root / "fixture.safetensors"))
    found = {name for name, value in model.named_parameters() if value.requires_grad}
    expected = set(fixture) - {"input_ids"}
    probe._require(found == expected, "backend trainable parameter set differs")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in expected:
                parameter.data = parameter.data.float()
                parameter.copy_(fixture[name].to(parameter.device))
    bases = json.loads((root / "base-digests.json").read_text())
    restored_bases = restore_frozen_bases(model, root, bases, dtype) if backend == "unsloth" else []
    actual_bases = {
        name: tensor_digest(parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    probe._require(actual_bases == bases, "base weights/dtypes differ; comparison refused")
    counts = {}
    if backend == "soup":
        counts = {path: probe._patch(model, path) for path in ("mlp", "qkv", "single")}
    layers = model.get_base_model().model.layers
    signatures = [
        {
            "mlp": callable_name(layer.mlp.forward),
            "norm": callable_name(layer.input_layernorm.forward),
            "qkv": callable_name(getattr(layer.self_attn, "apply_qkv", None)),
            "o": callable_name(getattr(layer.self_attn, "apply_o", None)),
        }
        for layer in layers
    ]
    if backend == "liger":
        probe._require(
            all(
                "liger_kernel" in row["mlp"] and "liger_kernel" in row["norm"] for row in signatures
            ),
            "Liger patch absent",
        )
    if backend == "unsloth":
        probe._require(
            all(all("unsloth" in row[key] for key in ("mlp", "qkv", "o")) for row in signatures),
            "Unsloth LoRA patches absent",
        )
    probe._require(
        not deterministic or torch.are_deterministic_algorithms_enabled(),
        "backend disabled requested deterministic algorithms",
    )
    return model, {
        "patch_counts": counts,
        "signatures": signatures,
        "canonical_base_restoration": restored_bases,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def optimizer_for(model: Any) -> Any:
    import torch

    return torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=0.003,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )


def snapshot(model: Any, prefix: str) -> dict:
    probe._check_training_gradients(model)
    return {
        f"{prefix}/{kind}/{name}": value.detach().cpu().clone().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        for kind, value in (("weight", parameter), ("gradient", parameter.grad))
    }


def arm(args: argparse.Namespace) -> None:
    root, output = Path(args.fixture), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = verify_fixture(root)
    model, execution = load_backend(
        root,
        manifest,
        args.backend,
        args.device,
        deterministic=getattr(args, "deterministic", False),
    )
    import torch
    from safetensors.torch import load_file, save_file

    cuda = args.device.startswith("cuda")
    probe._require(cuda or args.phase == "gate", "CPU cannot supply performance evidence")
    fixture = load_file(str(root / "fixture.safetensors"))
    tokens = fixture.pop("input_ids").to(args.device)
    observed, handles = probe._execution_hooks(model) if args.backend == "soup" else ({}, [])
    # Check backend execution outside measured regions; hooks are then removed.
    with torch.autograd.profiler.profile(use_cuda=False) as trace:
        initial_loss = model(input_ids=tokens, labels=tokens).loss
        initial_loss.backward()
    probe.require_finite(initial_loss, "preflight loss")
    if args.backend == "soup":
        probe._verify_execution(execution["patch_counts"], observed)
    for handle in handles:
        handle.remove()
    execution["observed_custom_paths"] = observed
    execution["loss_grad_fn"] = type(initial_loss.grad_fn).__name__
    execution["autograd_events"] = sorted(
        {
            event.name
            for event in trace.function_events
            if any(marker in event.name for marker in ("Liger", "LoRA_", "_FastLora"))
        }
    )
    if args.backend == "liger":
        probe._require(
            any("LigerFusedLinearCrossEntropy" in name for name in execution["autograd_events"]),
            "Liger loss not executed",
        )
    if args.backend == "unsloth":
        for marker in ("LoRA_MLP", "LoRA_QKV", "LoRA_W"):
            probe._require(
                any(marker in name for name in execution["autograd_events"]),
                f"Unsloth {marker} not executed",
            )
    del initial_loss
    optimizer = optimizer_for(model)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cuda and manifest["dtype"] == "float16",
        init_scale=128.0,
        growth_interval=2000,
    )

    def step() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=tokens).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss

    warmup_iterations = 0
    warmup_started = time.monotonic()
    if cuda:
        while warmup_iterations < args.warmup or time.monotonic() - warmup_started < 2:
            step()
            torch.cuda.synchronize()
            warmup_iterations += 1
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(fixture[name].to(parameter.device))
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        optimizer = optimizer_for(model)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=manifest["dtype"] == "float16",
            init_scale=128.0,
            growth_interval=2000,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    write_json(
        output / "ready.json",
        {
            "pid": os.getpid(),
            "execution": execution,
            "environment": environment(),
            "fixture_sha256": digest_file(root / "manifest.json"),
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid) if cuda else None,
            "device": args.device,
            "backend": args.backend,
            "harness_sha256": digest_file(Path(__file__)),
            "adapter_keys": sorted(fixture),
            "warmup_iterations": warmup_iterations,
        },
    )
    if args.managed:
        wait_file(output / "go", seconds=300)
    measurements, tensors = [], {}
    for index in range(args.steps):
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
        started = time.perf_counter()
        loss = step()
        if cuda:
            end.record()
            end.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000
        event_ms = start.elapsed_time(end) if cuda else None
        value = loss.detach().float().item()
        probe._require(math.isfinite(value), "nonfinite loss")
        # An overflow-skipped update cannot count as a completed optimizer step.
        if index in (0, args.steps - 1):
            probe._require(
                all(int(state["step"].item()) == index + 1 for state in optimizer.state.values())
                and len(optimizer.state) == len(fixture),
                "optimizer update skipped",
            )
        measurements.append(
            {"step": index + 1, "loss": value, "wall_ms": wall_ms, "cuda_ms": event_ms}
        )
        if index in (0, args.steps - 1):
            # Capture peaks before validation copies. Snapshot work is outside both timers.
            peaks = (
                {
                    "allocated": torch.cuda.max_memory_allocated(),
                    "reserved": torch.cuda.max_memory_reserved(),
                }
                if cuda
                else None
            )
            tensors.update(snapshot(model, f"step-{index + 1}"))
    result = {
        "backend": args.backend,
        "phase": args.phase,
        "steps": args.steps,
        "profile": manifest["profile"],
        "dtype": manifest["dtype"],
        "rows": measurements,
        "peaks_bytes": peaks,
        "input_tokens_per_step": tokens.numel(),
        "predicted_tokens_per_step": tokens.shape[0] * (tokens.shape[1] - 1),
        "predicted_tokens_per_second_of_steps": args.steps
        * tokens.shape[0]
        * (tokens.shape[1] - 1)
        / (sum(row["wall_ms"] for row in measurements) / 1000),
        "scope": "zero_grad + forward/loss + backward + AdamW.step; preloaded fixed tokens",
        "fp16_scaler": {
            "enabled": scaler.is_enabled(),
            "initial_scale": 128.0,
            "growth_interval": 2000,
            "final_scale": scaler.get_scale(),
        },
        "performance_status": "PENDING_SUPERVISOR_AND_REPLAY" if cuda else "CPU_SMOKE_ONLY",
    }
    write_json(output / "result.json", result)
    (output / "timed.json").touch()
    # Do not include large post-run CPU hash work in the timed NVML window.
    bases = json.loads((root / "base-digests.json").read_text())
    probe._require(
        {
            name: tensor_digest(value)
            for name, value in model.named_parameters()
            if not value.requires_grad
        }
        == bases,
        "frozen base changed during training",
    )
    save_file(tensors, str(output / "snapshots.safetensors"))
    write_json(output / "validation.json", {"base_unchanged": True})
    if args.managed:
        wait_file(output / "exit", seconds=300)


def compare(reference: Path, candidate: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    lhs, rhs = [
        json.loads((folder / "result.json").read_text()) for folder in (reference, candidate)
    ]
    probe._require(lhs["steps"] >= 2, "insufficient steps")
    ready_a, ready_b = [
        json.loads((folder / "ready.json").read_text()) for folder in (reference, candidate)
    ]
    probe._require(
        lhs["steps"] == rhs["steps"]
        and len(lhs["rows"]) == lhs["steps"]
        and len(rhs["rows"]) == rhs["steps"],
        "missing training steps",
    )
    probe._require(ready_a["fixture_sha256"] == ready_b["fixture_sha256"], "different fixtures")
    probe._require(
        ready_a["harness_sha256"] == ready_b["harness_sha256"], "different harness source"
    )
    probe._require(
        ready_a.get("execution", {}).get("deterministic_algorithms")
        == ready_b.get("execution", {}).get("deterministic_algorithms"),
        "different deterministic execution modes",
    )
    probe._require(
        all(
            ready_a["environment"]["packages"][name] == ready_b["environment"]["packages"][name]
            for name in COMMON_PACKAGES
        ),
        "different common package versions",
    )
    expected, actual = [
        load_file(str(folder / "snapshots.safetensors")) for folder in (reference, candidate)
    ]
    probe._require(set(expected) == set(actual) and expected, "missing snapshot quantities")
    adapter_keys = ready_a["adapter_keys"]
    probe._require(adapter_keys and adapter_keys == ready_b["adapter_keys"], "adapter set differs")
    required_keys = {
        f"step-{step}/{kind}/{name}"
        for step in (1, lhs["steps"])
        for kind in ("weight", "gradient")
        for name in adapter_keys
    }
    probe._require(set(expected) == required_keys, "required adapter snapshot missing")
    for folder in (reference, candidate):
        probe._require(
            json.loads((folder / "validation.json").read_text())["base_unchanged"] is True,
            "base immutability check missing",
        )
    required_prefixes = {
        f"step-{step}/{kind}" for step in (1, lhs["steps"]) for kind in ("weight", "gradient")
    }
    found_prefixes = {"/".join(key.split("/")[:2]) for key in expected}
    probe._require(required_prefixes == found_prefixes, "missing first/final weights or gradients")
    rows = []
    for key, value in expected.items():
        other = actual[key]
        passed = value.shape == other.shape and value.dtype == other.dtype
        finite = torch.isfinite(value).all().item() and torch.isfinite(other).all().item()
        relative = None
        if passed and finite:
            denominator = value.double().norm().item()
            error = (value.double() - other.double()).norm().item()
            relative = error / max(denominator, 1e-12)
            passed = relative <= 0.01
        else:
            passed = False
        rows.append({"quantity": key, "relative_l2": relative, "passed": passed})
    loss_checks = [
        math.isfinite(a["loss"])
        and math.isfinite(b["loss"])
        and a["step"] == b["step"] == index + 1
        and f"{a['loss']:.3f}" == f"{b['loss']:.3f}"
        for index, (a, b) in enumerate(zip(lhs["rows"], rhs["rows"]))
    ]
    return {
        "passed": all(loss_checks) and all(row["passed"] for row in rows),
        "loss_checks": loss_checks,
        "tensors": rows,
        "criterion": "all losses .3f; each saved weight/gradient relative L2 <= 1%",
    }


def supervise(args: argparse.Namespace) -> None:
    root = Path(args.output)
    manifests = [verify_fixture(Path(name)) for name in args.fixtures]
    probe._require(
        {item["dtype"] for item in manifests} == {"float16", "bfloat16"},
        "require distinct FP16 and BF16 fixtures",
    )
    root.mkdir(parents=True, exist_ok=False)
    monitor = Monitor()
    deadline = time.monotonic() + args.seconds
    summary = {"status": "PREPARING", "arms": [], "scope": "dense complete TinyLlama step"}
    write_json(root / "arguments.json", vars(args))

    def run_one(fixture: Path, backend: str, phase: str, label: str) -> Path:
        folder = root / label
        remaining = deadline - time.monotonic()
        probe._require(remaining > 30, "run deadline reached")
        before = [monitor.sample() for _ in range(3)]
        probe._require(
            all(
                row["available"] and not row["processes"] and not row["graphics"] for row in before
            ),
            "GPU occupied before arm",
        )
        command = [
            sys.executable,
            str(Path(__file__).absolute()),
            "arm",
            "--fixture",
            str(fixture.absolute()),
            "--output",
            str(folder.absolute()),
            "--backend",
            backend,
            "--phase",
            phase,
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.warmup),
            "--device",
            "cuda:0",
            "--managed",
        ]
        if getattr(args, "deterministic", False):
            command.append("--deterministic")
        write_json(root / f"{label}-command.json", command)
        with (root / f"{label}.log").open("w", encoding="utf-8") as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            live, startup = [], []
            try:
                arm_deadline = min(deadline, time.monotonic() + 300)
                while not (folder / "ready.json").exists():
                    startup.append(monitor.sample())
                    probe._require(worker.poll() is None, "arm failed before ready; see log")
                    probe._require(time.monotonic() < arm_deadline, "arm startup timeout")
                    time.sleep(0.02)
                (folder / "go").touch()
                while not (folder / "timed.json").exists():
                    live.append(monitor.sample())
                    probe._require(worker.poll() is None, "arm failed during steps; see log")
                    probe._require(time.monotonic() < arm_deadline, "arm timeout")
                    time.sleep(0.02)
                (folder / "exit").touch()
                code = worker.wait(timeout=min(30, max(1, deadline - time.monotonic())))
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()
                after = []
                for _ in range(5):
                    after.append(monitor.sample())
                    time.sleep(0.02)
                write_json(
                    root / f"{label}-telemetry.json",
                    {
                        "before": before,
                        "context_birth": startup,
                        "live": live,
                        "after": after,
                    },
                )
                write_json(root / f"{label}-process.json", {"exit_code": worker.returncode})
        validity = lifecycle_verdict(before, live, after, code)
        ready = json.loads((folder / "ready.json").read_text())
        # NVML can report host PIDs while os.getpid() is container-local.
        # Link the sole persistent context through lifecycle and GPU UUID.
        owner = validity["nvml_pid"]
        uuid_matches = str(ready["gpu_uuid"]).lower().removeprefix("gpu-") == str(
            monitor.uuid
        ).lower().removeprefix("gpu-")
        birth = all(
            row["available"] and not row["graphics"] and row["processes"] in ([], [owner])
            for row in startup
        )
        clocks = clock_verdict(live)
        probe._require(
            validity["status"] == "LIFECYCLE_VALID"
            and owner is not None
            and uuid_matches
            and birth,
            "VOID GPU lifecycle/ownership",
        )
        probe._require(clocks["passed"], "VOID unstable/missing arm clocks")
        summary["arms"].append(
            {
                "label": label,
                "backend": backend,
                "phase": phase,
                "validity": validity,
                "clocks": clocks,
            }
        )
        write_json(root / "summary.json", summary)
        Console().print(label, "collected; replay pending")
        return folder

    try:
        gates_by_dtype, checks_by_dtype = {}, {}
        for fixture_name in args.fixtures:
            fixture = Path(fixture_name)
            manifest = verify_fixture(fixture)
            probe._require(manifest["profile"] == "tinyllama-1.1b", "smoke is not performance")
            probe._require(manifest["dtype"] in ("float16", "bfloat16"), "half dtype required")
            dtype = manifest["dtype"]
            gates = {
                backend: run_one(fixture, backend, "gate", f"{dtype}-gate-{backend}")
                for backend in BACKENDS
            }
            checks = {backend: compare(gates["peft"], gates[backend]) for backend in BACKENDS}
            write_json(root / f"{dtype}-gate-checks.json", checks)
            gates_by_dtype[dtype], checks_by_dtype[dtype] = gates, checks
        # Collect both precision controls before deciding on any performance run.
        # Failed numerical criteria remain failures; no timing is admitted.
        probe._require(
            all(
                check["passed"] for checks in checks_by_dtype.values() for check in checks.values()
            ),
            "correctness gate failed; no performance verdict",
        )
        if getattr(args, "gates_only", False):
            summary["status"] = "CORRECTNESS_GATES_PASS; timing not requested"
            return
        for fixture_name in args.fixtures:
            fixture = Path(fixture_name)
            dtype = verify_fixture(fixture)["dtype"]
            gates = gates_by_dtype[dtype]
            for candidate in BACKENDS[1:]:
                cycle_ratios = []
                for cycle in range(args.cycles):
                    measured, clocks = {"peft": [], candidate: []}, []
                    for position, backend in enumerate(("peft", candidate, candidate, "peft")):
                        label = f"{dtype}-{candidate}-{cycle}-{position}-{backend}"
                        folder = run_one(fixture, backend, "timing", label)
                        check = compare(gates["peft"], folder)
                        write_json(folder / "replay.json", check)
                        probe._require(check["passed"], "measured trajectory failed replay")
                        result = json.loads((folder / "result.json").read_text())
                        times = [row["wall_ms"] for row in result["rows"]]
                        probe._require(
                            all(math.isfinite(t) and t > 0 for t in times), "invalid wall timings"
                        )
                        measured[backend].append(sum(times))
                        clocks.extend(
                            json.loads((root / f"{label}-telemetry.json").read_text())["live"]
                        )
                    probe._require(clock_verdict(clocks)["passed"], "ABBA clock mismatch")
                    cycle_ratios.append(
                        statistics.mean(measured["peft"]) / statistics.mean(measured[candidate])
                    )
                summary[f"{dtype}/{candidate}"] = {
                    "peft_over_candidate_full_step_wall": cycle_ratios,
                    "median_ratio": statistics.median(cycle_ratios),
                }
        summary["status"] = "COLLECTED; independent replay and team acceptance required"
    except Exception:
        summary["status"] = "VOID/INCOMPLETE; preserve all partial evidence"
        summary["error"] = traceback.format_exc()
        raise
    finally:
        write_json(root / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    prep.add_argument("--profile", choices=("smoke", "tinyllama-1.1b"), default="tinyllama-1.1b")
    prep.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), required=True)
    worker = sub.add_parser("arm")
    worker.add_argument("--fixture", required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument("--backend", choices=BACKENDS, required=True)
    worker.add_argument("--phase", choices=("gate", "timing"), default="gate")
    worker.add_argument("--device", default="cuda:0")
    worker.add_argument("--steps", type=int, default=50)
    worker.add_argument("--warmup", type=int, default=20)
    worker.add_argument("--managed", action="store_true")
    worker.add_argument("--deterministic", action="store_true")
    check = sub.add_parser("compare")
    check.add_argument("--reference", required=True)
    check.add_argument("--candidate", required=True)
    check.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--fixtures", nargs=2, required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--steps", type=int, default=50)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("--cycles", type=int, default=3)
    run.add_argument("--seconds", type=int, default=4500)
    run.add_argument("--deterministic", action="store_true")
    run.add_argument("--gates-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(Path(args.output), args.profile, args.dtype)
        elif args.command == "arm":
            probe._require(args.steps >= 2 and args.warmup >= 1, "invalid step count")
            arm(args)
        elif args.command == "compare":
            result = compare(Path(args.reference), Path(args.candidate))
            write_json(Path(args.output), result)
            probe._require(result["passed"], "comparison failed")
        else:
            probe._require(
                args.steps == 50
                and args.warmup >= 20
                and args.cycles >= 3
                and 60 <= args.seconds <= 7200,
                "production protocol bounds",
            )
            supervise(args)
    except Exception:
        Console().print(traceback.format_exc(), markup=False)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
