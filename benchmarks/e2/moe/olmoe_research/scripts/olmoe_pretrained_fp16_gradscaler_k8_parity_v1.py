#!/usr/bin/env python3
"""Research-only OLMoE pretrained FP16 + GradScaler K8 cache optimizer parity.

Runs two sequential, independent AdamW/GradScaler branches against ONE pinned
pretrained FP16 OLMoE instance. The baseline runs all 16 decoder layers;
the cached branch skips 8 frozen layers on cache hits and restores the frozen
router logits for auxiliary loss. Each step compares outputs, unscaled LoRA
gradients, updated adapters, AdamW state and scaler evolution.

Does not modify production E2. No timing, model-quality, or acceptance claim.
Must run with CUDA_VISIBLE_DEVICES=0 on NVIDIA Turing GPU (SM 7.5).
Original OLMoE cache v1 and fast-guard v2 source hashes are pinned.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, __version__ as peft_version
from transformers import OlmoeForCausalLM, AutoTokenizer, __version__ as transformers_version

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
PROBE_SHA = "20c3dbdd1e7515c89d7c22967f4d2e87a1b3d10bcb2134b3d81a0c73edc05b6b"
GUARD_SHA = "e0852684edbc6fc740506b7212f896ae48d835ab8f5dab5b247afa6201804a26"
K, CUTOFF, LAYERS = 8, 8, 16


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imported(path: Path, module_name: str, expected_sha: str):
    if not path.is_file() or sha(path) != expected_sha:
        raise RuntimeError(f"Missing or unverified {module_name}: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(name: str, first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape or first.dtype != second.dtype:
        raise RuntimeError(f"{name}: shape/dtype mismatch")
    torch.testing.assert_close(first, second, atol=2e-3, rtol=2e-3, msg=name)
    return float((first.float() - second.float()).abs().max().item())


def weights_snapshot(trainable):
    return {n: p.detach().cpu().clone() for n, p in trainable}


def restore_weights(trainable, original):
    with torch.no_grad():
        for n, p in trainable:
            p.copy_(original[n].to(device=p.device, dtype=p.dtype))


def run_branch(model, trainable, optimizer, scaler, ids, mask, positions, labels):
    model.zero_grad(set_to_none=True)
    scale_before = float(scaler.get_scale())
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            labels=labels,
            output_router_logits=True,
            use_cache=False,
        )
    if output.aux_loss is None or output.router_logits is None or len(output.router_logits) != LAYERS:
        raise RuntimeError("Incomplete router logits or aux loss")
    for label, tensor in (("loss", output.loss), ("aux_loss", output.aux_loss), ("logits", output.logits)):
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"Nonfinite {label} in FP16")
    scaler.scale(output.loss).backward()
    scaler.unscale_(optimizer)
    gradients = {}
    for n, p in trainable:
        if p.grad is None or not torch.isfinite(p.grad).all().item():
            raise RuntimeError(f"Missing/nonfinite unscaled gradient: {n}")
        gradients[n] = p.grad.detach().cpu().clone()
    if not any(t.abs().max().item() > 0 for t in gradients.values()):
        raise RuntimeError("All LoRA gradients are zero")
    outputs = {
        "logits": output.logits.detach().cpu().clone(),
        "loss": output.loss.detach().cpu().clone(),
        "aux_loss": output.aux_loss.detach().cpu().clone(),
        "routers": [t.detach().cpu().clone() for t in output.router_logits],
        "gradients": gradients,
    }
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if scale_after < scale_before:
        raise RuntimeError(f"GradScaler overflow or skipped optimizer step: {scale_before} -> {scale_after}")
    if not optimizer.state:
        raise RuntimeError("AdamW optimizer state empty: optimizer step may have been skipped")
    outputs["weights"] = weights_snapshot(trainable)
    outputs["scale_before"] = scale_before
    outputs["scale_after"] = scale_after
    return outputs


def compare_optimizer_states(left_opt, right_opt, trainable):
    max_delta = 0.0
    for n, param in trainable:
        left, right = left_opt.state.get(param, {}), right_opt.state.get(param, {})
        if not left or left.keys() != right.keys():
            raise RuntimeError(f"Missing/different AdamW states: {n}")
        for key in left:
            a, b = left[key], right[key]
            if isinstance(a, torch.Tensor):
                if not isinstance(b, torch.Tensor):
                    raise RuntimeError(f"AdamW state type mismatch {n}.{key}")
                max_delta = max(max_delta, compare(f"adamw.{n}.{key}", a, b))
            elif a != b:
                raise RuntimeError(f"AdamW state scalar mismatch {n}.{key}")
    return max_delta


def compare_branches(a, b, trainable, opt_a, opt_b):
    diffs = {k: compare(k, a[k], b[k]) for k in ("logits", "loss", "aux_loss")}
    if len(a["routers"]) != LAYERS or len(b["routers"]) != LAYERS:
        raise RuntimeError("Router logits count mismatch")
    diffs["router_logits"] = max(compare(f"router.{i}", x, y) for i, (x, y) in enumerate(zip(a["routers"], b["routers"])))
    diffs["gradients"] = max(compare(f"grad.{n}", a["gradients"][n], b["gradients"][n]) for n, _ in trainable)
    diffs["adapter_weights"] = max(compare(f"adapter.{n}", a["weights"][n], b["weights"][n]) for n, _ in trainable)
    diffs["optimizer_states"] = compare_optimizer_states(opt_a, opt_b, trainable)
    if a["scale_before"] != b["scale_before"] or a["scale_after"] != b["scale_after"]:
        raise RuntimeError("GradScaler state diverged")
    return diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=Path("/workspace/olmoe_disk_cache_research_v1.py"))
    parser.add_argument("--guard", type=Path, default=Path("/workspace/olmoe_guarded_cache_v2.py"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output already exists; refusing stale cache/results")
    if args.steps < 2:
        parser.error("Must run at least two optimizer steps to test cache hit")
    if transformers_version != "5.19.0":
        raise RuntimeError(f"Unexpected Transformers version: {transformers_version}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one GPU via CUDA_VISIBLE_DEVICES=0")
    if torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError("This smoke is specifically for SM7.5 FP16 GPU")
    probe = imported(args.probe, "olmoe_verified_cache_probe", PROBE_SHA)
    if probe.MODEL_REVISION != "random-olmoe-4layers-seed42-research-only":
        raise RuntimeError("Unexpected verified cache model revision")
    probe.MODEL_REVISION = f"{MODEL_ID}@{REVISION}"
    guard = imported(args.guard, "olmoe_verified_fast_guard", GUARD_SHA)
    snapshot = (Path.home() / ".cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924"
                / "snapshots" / REVISION)
    if not (snapshot / "config.json").is_file() or len(list(snapshot.glob("model-*.safetensors"))) != 3:
        raise RuntimeError("Pinned OLMoE snapshot missing")

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("Loading ONE pinned pretrained OLMoE in FP16...", flush=True)
    base = OlmoeForCausalLM.from_pretrained(
        str(snapshot), dtype=torch.float16, device_map={"": 0},
        local_files_only=True, trust_remote_code=False,
    )
    if len(base.model.layers) != LAYERS or base.config.num_experts != 64:
        raise RuntimeError("Unexpected pretrained OLMoE architecture")
    if base.config.attention_dropout != 0:
        raise RuntimeError("Nonzero attention dropout unsupported")
    base.requires_grad_(False)
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(CUTOFF, LAYERS)),
        layers_pattern="layers", bias="none",
    )
    model = get_peft_model(base, cfg)
    model.train()
    if model.get_base_model() is not base:
        raise RuntimeError("Unexpected PEFT base identity")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable) != K * 4:
        raise RuntimeError(f"Wrong number of LoRA tensors {len(trainable)}")
    for n, p in trainable:
        matched = re.search(r"(?:^|\.)layers\.(\d+)\.", n)
        if not matched or int(matched.group(1)) < CUTOFF or "lora_" not in n:
            raise RuntimeError(f"Unexpected trainable tensor {n}")
        if p.dtype != torch.float32:
            raise RuntimeError(f"Expected PEFT FP32 trainable LoRA tensors, got {p.dtype}: {n}")
    for layer in base.model.layers[:CUTOFF]:
        if any(p.requires_grad for p in layer.parameters()):
            raise RuntimeError("Frozen prefix requires_grad unexpectedly True")
    if any(p.requires_grad for p in base.model.embed_tokens.parameters()):
        raise RuntimeError("Embeddings must be frozen")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    enc = tokenizer("The capital of Kazakhstan is Astana.", return_tensors="pt")
    ids = enc["input_ids"].to("cuda:0")
    mask = enc["attention_mask"].to("cuda:0")
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    labels = ids.clone()
    labels[mask == 0] = -100
    if not 2 <= ids.shape[1] <= 32:
        raise RuntimeError("Unexpected sequence length")

    args.output_dir.mkdir(parents=True)
    params = [p for _, p in trainable]
    opt_a = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    opt_b = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    scale_a = torch.amp.GradScaler("cuda", init_scale=1024.0)
    scale_b = torch.amp.GradScaler("cuda", init_scale=1024.0)
    if not scale_a.is_enabled() or not scale_b.is_enabled():
        raise RuntimeError("FP16 GradScaler unexpectedly disabled")
    GuardedCache = guard.guarded_cache_type(probe.ResearchOlmoeCache)
    cache = GuardedCache(base, args.output_dir / "cache", CUTOFF)
    cache.set_model_revision(f"{MODEL_ID}@{REVISION}")
    torch.cuda.reset_peak_memory_stats()
    results = []
    with cache:
        for step in range(1, args.steps + 1):
            start_weights = weights_snapshot(trainable)
            patched = base.model.forward
            base.model.forward = cache._original_forward
            try:
                baseline = run_branch(model, trainable, opt_a, scale_a, ids, mask, positions, labels)
            finally:
                base.model.forward = patched
            restore_weights(trainable, start_weights)
            cached = run_branch(model, trainable, opt_b, scale_b, ids, mask, positions, labels)
            differences = compare_branches(baseline, cached, trainable, opt_a, opt_b)
            if (cache.hits, cache.misses) != (step - 1, 1):
                raise RuntimeError(f"Unexpected cache counters: hits={cache.hits}, misses={cache.misses}")
            if cache.skipped_frozen_layer_calls != (step - 1) * CUTOFF:
                raise RuntimeError("Frozen skip calls count mismatch")
            if cache.fingerprint_recomputes != 1:
                raise RuntimeError("Frozen prefix fingerprint recalculated unexpectedly")
            values = {
                "step": step,
                "max_abs_differences": differences,
                "grad_scaler_scale_before": baseline["scale_before"],
                "grad_scaler_scale_after": baseline["scale_after"],
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
            }
            results.append(values)
            print(f"STEP {step}: FP16 GradScaler LoRA parity PASS; hits={cache.hits}, misses={cache.misses}; max differences={differences}", flush=True)
    report = {
        "kind": "pretrained_olmoe_fp16_gradscaler_k8_cache_lora_optimizer_parity_v1",
        "model_id": MODEL_ID, "revision": REVISION,
        "pretrained": True, "hardware": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "precision": "FP16 pretrained weights + FP16 CUDA autocast + FP32 PEFT LoRA + enabled GradScaler",
        "grad_scaler_init_scale": 1024.0,
        "k_trainable_layers": K, "frozen_cutoff": CUTOFF,
        "steps": results,
        "cache": {
            "hits": cache.hits, "misses": cache.misses,
            "skipped_frozen_layer_calls": cache.skipped_frozen_layer_calls,
            "full_sha256_recomputes": cache.fingerprint_recomputes,
            "fingerprint_seconds": cache.fingerprint_seconds,
            "guard_seconds": cache.guard_seconds,
        },
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (2**30), 4),
        "script_sha256": sha(Path(__file__)),
        "probe_sha256": sha(args.probe), "guard_sha256": sha(args.guard),
        "torch": torch.__version__, "transformers": transformers_version, "peft": peft_version,
        "quality_evaluation": False, "performance_benchmark": False,
        "production_e2_modified": False,
        "research_acceptance": "NO VERDICT",
        "verdict": "EXPLORATORY_PRETRAINED_FP16_LORA_OPTIMIZER_PARITY_ONLY",
    }
    output_path = args.output_dir / "report.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\nPRETRAINED OLMOE K8 FP16 GRADSCALER MULTISTEP PARITY PASSED", flush=True)
    print("Report:", output_path, flush=True)


if __name__ == "__main__":
    main()
