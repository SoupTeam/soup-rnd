#!/usr/bin/env python3
"""Pretrained OLMoE K=4/8/12 FP16+GradScaler short performance pilot (Turing SM7.5).

Execute in SEPARATE processes with --mode baseline and --mode cached.
Both runs train the SAME LoRA configuration with the SAME fixed synthetic
examples, optimizer, deterministic seed, and two epochs. In cached mode,
the verified research-only frozen-prefix disk cache actually skips layers
on hits; first epoch builds the cache, second reuses it.

Measurements include actual FP16 autocast and enabled GradScaler, step wall clock
(with CUDA sync), epoch step sums,
training wall clock, cache hits/misses, adapter/dataset hashes and peak VRAM.
They deliberately include real cache fingerprinting and IO overhead.
They deliberately do NOT include model loading in training wall clock.

No production E2 changes. Not an ABBA run, full-data run, quality test,
or E2 acceptance. Synthetic fixed random token IDs are for timing ONLY.
Requires original SHA256-verified research cache, guarded cache v2 and transformers 5.19.0.
FP16/SM7.5 and full GPU residency must match both arms.
Does NOT establish accepted speed or model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, __version__ as peft_version
from transformers import OlmoeForCausalLM, __version__ as transformers_version

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
RESEARCH_CACHE_SHA256 = "20c3dbdd1e7515c89d7c22967f4d2e87a1b3d10bcb2134b3d81a0c73edc05b6b"
GUARD_CACHE_SHA256 = "e0852684edbc6fc740506b7212f896ae48d835ab8f5dab5b247afa6201804a26"
LAYER_COUNT = 16


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.digest()


def named_tensors_sha256(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(named_tensors, key=lambda x: x[0]):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_digest(value))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "cached"), required=True)
    parser.add_argument("--k", type=int, choices=(4, 8, 12), required=True, help="Upper layers with trainable LoRA; frozen cutoff is 16-k")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("/workspace/olmoe_disk_cache_research_v1.py"))
    parser.add_argument("--guard-module", type=Path, default=Path("/workspace/olmoe_guarded_cache_v2.py"))
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    K = args.k
    CUTOFF = LAYER_COUNT - K
    assert 0 < CUTOFF < LAYER_COUNT

    if args.output_dir.exists():
        parser.error("Output directory already exists: refusing to reuse an old cache/report")
    if args.examples < 2 or args.epochs != 2 or args.seq_len < 4 or args.seq_len > 128:
        parser.error("Use at least 2 examples, exactly 2 epochs, sequence length 4..128")
    if args.batch_size < 1 or args.batch_size > 2:
        parser.error("Pilot batch-size must be 1 or 2")
    if transformers_version != "5.19.0":
        raise RuntimeError(f"Unsupported Transformers version: {transformers_version}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU: export CUDA_VISIBLE_DEVICES=0")
    if torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError("This pilot requires Turing SM7.5 GPU")
    if not args.probe.is_file() or file_sha256(args.probe) != RESEARCH_CACHE_SHA256:
        raise RuntimeError("Research cache v1 implementation SHA256 mismatch")
    if not args.guard_module.is_file() or file_sha256(args.guard_module) != GUARD_CACHE_SHA256:
        raise RuntimeError("Guarded cache v2 missing or SHA256 mismatch")

    snapshot = (
        Path.home() / ".cache/huggingface/hub"
        / "models--allenai--OLMoE-1B-7B-0924"
        / "snapshots" / REVISION
    )
    if not (snapshot / "config.json").is_file() or len(list(snapshot.glob("model-*.safetensors"))) != 3:
        raise RuntimeError("Pinned pretrained OLMoE checkpoint is not present")

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"Loading ONE pretrained OLMoE FP16; K={K}; mode={args.mode}", flush=True)
    load_started = time.perf_counter()
    base = OlmoeForCausalLM.from_pretrained(
        str(snapshot), dtype=torch.float16, device_map={"": 0},
        local_files_only=True, trust_remote_code=False,
    )
    load_seconds = time.perf_counter() - load_started
    if len(base.model.layers) != LAYER_COUNT or base.config.num_experts != 64:
        raise RuntimeError("Unexpected OLMoE pretrained architecture")
    if base.config.attention_dropout != 0:
        raise RuntimeError("Nonzero attention dropout unsupported for reproducible timing")
    base.requires_grad_(False)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(CUTOFF, LAYER_COUNT)),
        layers_pattern="layers", bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    model.train()
    if model.get_base_model() is not base:
        raise RuntimeError("Unexpected PEFT model object")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if len(trainable) != 4 * K:
        raise RuntimeError(f"Unexpected trainable tensor count: {len(trainable)}")
    for name, p in trainable:
        if p.dtype != torch.float32:
            raise RuntimeError(f"Expected PEFT trainable LoRA in FP32: {name} {p.dtype}")
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if not match or int(match.group(1)) < CUTOFF or "lora_" not in name:
            raise RuntimeError(f"Invalid trainable tensor: {name}")
    for p in base.model.embed_tokens.parameters():
        if p.requires_grad:
            raise RuntimeError("Embedding is not frozen")
    for layer in base.model.layers[:CUTOFF]:
        if any(p.requires_grad for p in layer.parameters()):
            raise RuntimeError("Frozen prefix contains trainable parameters")

    # Random valid token IDs. This is NOT natural-language data or quality evidence.
    generator = torch.Generator(device="cpu").manual_seed(20261014)
    examples = torch.randint(
        low=3, high=base.config.vocab_size,
        size=(args.examples, args.seq_len), generator=generator,
        dtype=torch.long,
    )
    if torch.unique(examples, dim=0).shape[0] != args.examples:
        raise RuntimeError("Synthetic input batches contain duplicate rows; cache counts invalid")
    data_sha256 = named_tensors_sha256([("synthetic_ids", examples)])
    ids_gpu = examples.to("cuda:0")
    mask_gpu = torch.ones((args.batch_size, args.seq_len), device="cuda:0", dtype=torch.long)
    positions_gpu = torch.arange(args.seq_len, device="cuda:0", dtype=torch.long).unsqueeze(0)

    adapter_sha_start = named_tensors_sha256(trainable)
    optimizer = torch.optim.AdamW(
        (p for _, p in trainable), lr=2e-4, weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    if not scaler.is_enabled() or scaler.get_scale() != 1024.0:
        raise RuntimeError("FP16 GradScaler not enabled or initial scale mismatch")
    completed_optimizer_steps = 0

    args.output_dir.mkdir(parents=True)
    cache = None
    if args.mode == "cached":
        import importlib.util
        spec = importlib.util.spec_from_file_location("olmoe_cache_verified_pilot", args.probe)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load verified cache implementation")
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        if probe.MODEL_REVISION != "random-olmoe-4layers-seed42-research-only":
            raise RuntimeError("Verified cache identity has unexpectedly changed")
        probe.MODEL_REVISION = f"{MODEL_ID}@{REVISION}"
        fast_spec = importlib.util.spec_from_file_location("olmoe_fast_cache_guard_v2", args.guard_module)
        if fast_spec is None or fast_spec.loader is None:
            raise RuntimeError("Cannot load guarded cache v2")
        fast_guard = importlib.util.module_from_spec(fast_spec)
        fast_spec.loader.exec_module(fast_guard)
        CacheV2 = fast_guard.guarded_cache_type(probe.ResearchOlmoeCache)
        cache = CacheV2(base, args.output_dir / "cache", CUTOFF)
        cache.set_model_revision(f"{MODEL_ID}@{REVISION}")

    steps = []
    epochs = []
    epoch_count = args.examples // args.batch_size
    if args.examples % args.batch_size:
        epoch_count += 1
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()

    # Baseline does not install any cache hooks/monkey patches.
    with (cache if cache is not None else nullcontext()):
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            durations = []
            for batch_index, offset in enumerate(range(0, args.examples, args.batch_size), start=1):
                ids = ids_gpu[offset:offset + args.batch_size]
                batch = ids.shape[0]
                mask = mask_gpu[:batch]
                positions = positions_gpu  # broadcast over batch in OLMoE
                labels = ids.clone()
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                step_started = time.perf_counter()
                # Explicit FP16 execution contract on Turing: autocast + enabled scaler.
                scale_before = float(scaler.get_scale())
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = model(
                        input_ids=ids,
                        attention_mask=mask,
                        position_ids=positions,
                        labels=labels,
                        output_router_logits=True,
                        use_cache=False,
                    )
                if (
                    result.aux_loss is None or result.router_logits is None
                    or len(result.router_logits) != LAYER_COUNT
                    or not torch.isfinite(result.loss).item()
                    or not torch.isfinite(result.aux_loss).item()
                    or not torch.isfinite(result.logits).all().item()
                ):
                    raise RuntimeError("Invalid/nonfinite OLMoE loss/logits/router output")
                scaler.scale(result.loss).backward()
                scaler.unscale_(optimizer)
                # A skipped update is not a training-equivalent step, even if both
                # arms report the same losses. Fail closed on nonfinite gradients.
                for name, p in trainable:
                    if p.grad is None or not torch.isfinite(p.grad).all().item():
                        raise RuntimeError(f"Nonfinite/missing unscaled gradient: {name}")
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                if scale_after < scale_before or not optimizer.state:
                    raise RuntimeError(
                        f"GradScaler overflow/optimizer step skipped: {scale_before} -> {scale_after}"
                    )
                completed_optimizer_steps += 1
                torch.cuda.synchronize()
                seconds = time.perf_counter() - step_started
                durations.append(seconds)
                steps.append({
                    "epoch": epoch,
                    "batch": batch_index,
                    "batch_size": batch,
                    "step_seconds": round(seconds, 6),
                    "loss": float(result.loss.detach().float().item()),
                    "aux_loss": float(result.aux_loss.detach().float().item()),
                    "scaler_before": scale_before,
                    "scaler_after": scale_after,
                    "optimizer_step_verified": True,
                })
                if batch_index == 1 or batch_index % 16 == 0 or batch_index == epoch_count:
                    print(
                        f"MODE={args.mode} epoch={epoch} batch={batch_index}/{epoch_count} "
                        f"step={seconds:.3f}s loss={result.loss.item():.5f} "
                        f"scale={scale_after:.0f} "
                        f"cache_hits={cache.hits if cache else 0} "
                        f"cache_misses={cache.misses if cache else 0}",
                        flush=True,
                    )
            torch.cuda.synchronize()
            epoch_wall = time.perf_counter() - epoch_started
            epochs.append({
                "epoch": epoch,
                "step_sum_seconds": round(sum(durations), 6),
                "wall_seconds": round(epoch_wall, 6),
                "optimizer_steps": len(durations),
            })
            print(f"EPOCH {epoch}: step_sum={sum(durations):.3f}s wall={epoch_wall:.3f}s", flush=True)
    torch.cuda.synchronize()
    training_wall = time.perf_counter() - training_started

    if completed_optimizer_steps != args.epochs * epoch_count:
        raise RuntimeError("Not every batch completed an optimizer update")
    if named_tensors_sha256(trainable) == adapter_sha_start:
        raise RuntimeError("LoRA weights did not change during training")
    if cache is not None:
        expected_batches = epoch_count
        if cache.misses != expected_batches or cache.hits != expected_batches:
            raise RuntimeError(f"Wrong cache hit/miss counts: {cache.hits}/{cache.misses}, expected {expected_batches}")
        if cache.skipped_frozen_layer_calls != expected_batches * CUTOFF:
            raise RuntimeError("Frozen prefix did not actually skip expected layers")
        if cache.fingerprint_recomputes != 1 or cache.metadata_calls != 2 * expected_batches:
            raise RuntimeError(
                "Unexpected guard rehashes/calls; frozen weights may have changed: "
                f"rehashes={cache.fingerprint_recomputes}, calls={cache.metadata_calls}"
            )

    report = {
        "kind": "olmoe_pretrained_kmatrix_fp16_gradscaler_research_performance_pilot_v1",
        "k": K,
        "frozen_layers": CUTOFF,
        "model_layers": LAYER_COUNT,
        "mode": args.mode,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "precision": "pretrained FP16 weights, FP16 CUDA autocast, FP32 PEFT LoRA, GradScaler enabled",
        "precision_flags": {"fp16": True, "bf16": False},
        "grad_scaler": {"enabled": scaler.is_enabled(), "initial_scale": 1024.0, "final_scale": float(scaler.get_scale()), "completed_optimizer_steps": completed_optimizer_steps, "skipped_optimizer_steps": 0},
        "lora": {"r": 4, "alpha": 8, "dropout": 0, "modules": ["q_proj", "v_proj"], "upper_layers": K},
        "optimizer": "AdamW lr=2e-4 wd=0",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "examples": args.examples,
        "data_kind": "fixed synthetic token IDs: NOT quality evidence",
        "synthetic_data_sha256": data_sha256,
        "adapter_sha256_before": adapter_sha_start,
        "adapter_sha256_after": named_tensors_sha256(trainable),
        "epochs": epochs,
        "steps": steps,
        "training_wall_seconds": round(training_wall, 6),
        "model_load_seconds_excluded": round(load_seconds, 6),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 5),
        "cache": {
            "enabled": cache is not None,
            "hits": cache.hits if cache is not None else 0,
            "misses": cache.misses if cache is not None else 0,
            "skipped_frozen_layer_calls": cache.skipped_frozen_layer_calls if cache is not None else 0,
        },
        "guard_profile": {
            "guard_kind": "torch_tensor_version_identity_plus_config_sha",
            "full_sha256_recomputes": cache.fingerprint_recomputes if cache else 0,
            "full_sha256_seconds": round(cache.fingerprint_seconds, 6) if cache else 0.0,
            "guard_seconds": round(cache.guard_seconds, 6) if cache else 0.0,
            "metadata_seconds_total": round(cache.metadata_seconds, 6) if cache else 0.0,
            "metadata_calls": cache.metadata_calls if cache else 0,
            "explicit_invalidations": cache.explicit_invalidations if cache else 0,
        },
        "code": {
            "pilot_sha256": file_sha256(Path(__file__)),
            "verified_cache_sha256": file_sha256(args.probe),
            "guard_cache_sha256": file_sha256(args.guard_module),
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers_version,
            "peft": peft_version,
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "caveats": [
            "Exploratory pilot only; no ABBA randomization or acceptance decision",
            "Uses fixed synthetic token IDs; not a quality evaluation",
            "Single new process per arm; cold CUDA kernels and IO may vary",
            "RTX 2080 Ti FP16 results cannot be directly combined with RTX 3090 BF16 timing",
            "No continuous GPU-peer/process, suspension, or foreign-IO monitoring: not valid for final ABBA acceptance",
            "One full frozen-prefix SHA256 per run, with version/identity guard per lookup",
            "Version guard cannot detect .data/raw-storage writes; such writes require explicit mark_prefix_changed()",
            "No concurrent prefix mutation or multi-process writers supported",
            "Step measurements include cache reads/writes and hashing, without profiler",
            "Model loading excluded from reported training wall; separate setup overhead",
        ],
        "performance_acceptance": "NO VERDICT",
        "quality_acceptance": "NO VERDICT",
        "production_e2_modified": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nOLMOE K{K} FP16 {args.mode.upper()} PERFORMANCE PILOT COMPLETE", flush=True)
    print("REPORT:", args.output_dir / "report.json", flush=True)


if __name__ == "__main__":
    main()
