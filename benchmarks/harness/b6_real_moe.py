"""Measured real-checkpoint per-expert LoRA pilot; no synthetic substitution."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

from rich.console import Console


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    from peft import LoraConfig, TaskType, load_peft_weights, set_peft_model_state_dict
    from transformers import AutoTokenizer

    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget
    from soup_cli.utils.layer_shard import (
        checkpoint_source_components,
        fingerprint_source_files,
        inspect_shard_cache,
        shard_checkpoint,
        source_identity_includes_config,
    )
    from soup_cli.utils.layer_stream_runtime import (
        assert_streamed_adapter_saved,
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
        shard_layout_of,
    )
    from soup_cli.utils.peft_wiring import MOE_TEXT_LORA_TARGETS

    console = Console()

    @contextlib.contextmanager
    def meta_construction():
        # Accelerate redirects parameters after construction. Poison-filling a
        # temporary CPU torch.empty before that redirection touches giant arrays.
        # No base values from construction are used: the cache supplies them all.
        previous = torch.utils.deterministic.fill_uninitialized_memory
        try:
            torch.utils.deterministic.fill_uninitialized_memory = False
            yield
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = previous
    source, cache, output = Path(args.source), Path(args.cache), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    events = output / "events.jsonl"

    def emit(kind: str, **values) -> None:
        import psutil

        memory = psutil.Process().memory_info()
        row = {"event": kind, "unix": time.time(), "rss_bytes": memory.rss,
               "host_ram_bytes": psutil.virtual_memory().total,
               "disk_free_bytes_now": shutil.disk_usage(output).free, **values}
        status = Path("/proc/self/status")
        if status.is_file():
            for line in status.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    row["peak_rss_bytes"] = int(line.split()[1]) * 1024
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        console.print(row)

    if not torch.cuda.is_available():
        raise RuntimeError("B6 requires a real CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261028)
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or props.total_memory > 9 * 2**30:
        raise RuntimeError("B6 pilot requires one physical GPU of approximately 8 GB")
    dtype = "bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float16"
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "deepseek_v3":
        raise ValueError("this pilot accepts DeepSeek-V3 FP8 only")
    budget = new_expert_cache_budget([str(path) for path in source.glob("*.safetensors")])
    cache.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache.parent).free
    existing_partial = (
        args.restart_incomplete_cache and cache.is_dir() and not (cache / "index.json").exists()
    )
    reuse = args.reuse_cache and (cache / "index.json").is_file()
    if args.reuse_cache and not reuse:
        raise ValueError("--reuse-cache requires a committed cache index")
    if (cache.exists() and not existing_partial and not reuse) or (free < budget and not reuse):
        raise ValueError(f"new cache requires {budget} free bytes, available {free}")
    emit("environment", gpu=props.name, vram_bytes=props.total_memory, dtype=dtype,
         torch=torch.__version__, cuda=torch.version.cuda, cache_budget_bytes=budget,
         disk_free_bytes=free, source_revision=args.revision)
    with meta_construction():
        skeleton = build_meta_skeleton(str(source), dtype=dtype, quant="nf4", double_quant=False)
    suffixes, layout = quantised_layer_suffixes(skeleton), shard_layout_of(skeleton)
    del skeleton
    gc.collect()
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.shard_threads)
        if reuse:
            files = tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns)
                          for path in sorted(source.glob("*.safetensors")))
            components = checkpoint_source_components(
                str(source), files,
                include_config=source_identity_includes_config(str(source), "deepseek_v3"),
            )
            index, reason = inspect_shard_cache(
                str(cache), dtype, fingerprint_source_files(components), components,
                "nf4", False, "cuda", layout=layout, expert_layout="per_expert",
            )
            if index is None:
                raise ValueError(f"existing cache failed validation: {reason}")
        else:
            index = shard_checkpoint(
                str(source), str(cache), dtype=dtype, arch="deepseek_v3", layout=layout,
                quant="nf4", quant_suffixes=suffixes, quant_device="cuda",
                double_quant=False, expert_layout="per_expert",
            )
    finally:
        torch.set_num_threads(previous_threads)
    if index.total_params < 500_000_000_000:
        raise ValueError("B6 real giant pilot requires at least 500B source parameters")
    emit("sharded", seconds=time.perf_counter() - started,
         cache_reused=reuse,
         shard_threads=args.shard_threads, training_threads=previous_threads,
         source_parameters=index.total_params,
         cache_bytes=sum(path.stat().st_size for path in cache.iterdir() if path.is_file()))
    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0, bias="none",
                      target_modules=list(MOE_TEXT_LORA_TARGETS["deepseek_v3"]),
                      task_type=TaskType.CAUSAL_LM)

    def build():
        with meta_construction():
            model, runtime = build_streamed_model(
                model_id=str(source), shard_dir=str(cache), index=index, lora_config=lora,
                device="cuda", dtype=dtype, buffers=2, pin=False, quant="nf4",
                double_quant=False, tier="disk", read_ahead=1,
            )
        if dtype == "float16":
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
        return model, runtime

    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=False)
    texts = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if not texts.get("train") or not texts.get("heldout"):
        raise ValueError("data JSON requires separate train and heldout strings")
    emit("data", sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
         max_length=args.max_length)

    def batch(text):
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=args.max_length)["input_ids"].to("cuda")
        return {"input_ids": ids, "labels": ids}

    model, runtime = build()
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    def adapter_hash(params):
        digest = hashlib.sha256()
        for parameter in params:
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()

    validation = batch(texts["heldout"][0])
    torch.cuda.reset_peak_memory_stats()
    try:
        model.eval()
        with torch.no_grad():
            before = float(model(**validation).loss)
        if not math.isfinite(before):
            raise RuntimeError("non-finite heldout loss before training")
        emit("heldout_before", loss=before)
        for step in range(args.steps):
            model.train()
            inputs = batch(texts["train"][step % len(texts["train"])])
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = model(**inputs)
            if not torch.isfinite(result.loss):
                raise RuntimeError("non-finite training loss")
            result.loss.backward()
            gradients = [param.grad for param in parameters if param.grad is not None]
            if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
                raise RuntimeError("missing or non-finite adapter gradients")
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError("adapter gradient norm is zero or non-finite")
            old_hash = adapter_hash(parameters)
            optimizer.step()
            new_hash = adapter_hash(parameters)
            if new_hash == old_hash:
                raise RuntimeError("optimizer step did not change the adapter")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            emit("step", step=step, loss=float(result.loss), gradient_norm=norm,
                 adapter_sha256_before=old_hash, adapter_sha256_after=new_hash,
                 gradient_tensors=len(gradients), seconds=seconds,
                 tokens=int(inputs["input_ids"].numel()),
                 tokens_per_second=inputs["input_ids"].numel() / seconds,
                 peak_allocated=torch.cuda.max_memory_allocated(),
                 peak_reserved=torch.cuda.max_memory_reserved())
        model.eval()
        with torch.no_grad():
            result = model(**validation)
            logits = result.logits.cpu()
            if not torch.isfinite(result.loss):
                raise RuntimeError("non-finite heldout loss after training")
            emit("heldout_after", loss=float(result.loss))
        from safetensors.torch import save_file

        save_file({"logits": logits.contiguous()}, str(output / "reference-logits.safetensors"))
        adapter = output / "adapter"
        model.save_pretrained(adapter)
        assert_streamed_adapter_saved(model, str(adapter))
        emit("runtime", stats=runtime.stats())
        emit("expert_io", stats=runtime.expert_pool.stats())
    finally:
        runtime.close()
    del model, runtime, optimizer, parameters, gradients, result
    gc.collect()
    torch.cuda.empty_cache()
    model, runtime = build()
    try:
        state = load_peft_weights(str(adapter), device="cpu")
        set_peft_model_state_dict(model, state)
        reloaded_hash = adapter_hash([param for param in model.parameters() if param.requires_grad])
        if reloaded_hash != new_hash:
            raise RuntimeError("reloaded adapter parameters differ from the trained parameters")
        model.eval()
        with torch.no_grad():
            reloaded = model(**validation).logits.cpu()
        equal = torch.equal(logits, reloaded)
        error = float((logits.float() - reloaded.float()).abs().max())
        emit("reload", equal=equal, max_abs_error=error,
             parameter_digest_matches=True, parameter_sha256=reloaded_hash)
        if not equal:
            raise RuntimeError("reloaded logits differ")
    finally:
        runtime.close()
    emit("complete")


if __name__ == "__main__":
    import faulthandler
    import signal

    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)
    parser = argparse.ArgumentParser()
    for name in ("source", "cache", "output", "data", "revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--shard-threads", type=int, default=8)
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument("--restart-incomplete-cache", action="store_true")
    cache_mode.add_argument("--reuse-cache", action="store_true")
    run(parser.parse_args())
