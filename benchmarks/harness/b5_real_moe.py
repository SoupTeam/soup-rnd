"""B5 ticket 03: trains a DeepSeek-V3-family MoE on the ``per_expert`` path and writes the journal.

Used for Moonlight-16B-A3B on one 8 GB card (rule:
``benchmarks/b5-moonlight-acceptance-20261109.md``) and, on CPU, for the tiny fixture of the
tests. It makes the same calls as ``b6_real_moe.py`` (``shard_checkpoint`` with the
``per_expert`` layout, ``build_streamed_model``), takes its settings from the ``soup.yaml`` that
``SoupConfig`` validates, and writes every row through ``b5_recorder`` so ``b5_judge`` and
``b5_acceptance`` can read it. ``soup train`` is not run: it writes no journal for the judge and
keeps no gradients.

With ``--cold`` (a store run only) the page cache of the store is dropped right before step 0
(after the model is built), and the ``cold_cache`` event says how and what was left
(``b5_cold.py``). Every step row has ``disk_read_bytes``, the ``read_bytes`` counter of the
process before and after the step, or null where the host has no counter. A ``shape`` event
gives the store's shape and bytes.

With ``--store`` the cache is a synthetic store written by ``b5_synth_store.py`` (ticket 05):
the sharding step is skipped, ``build_streamed_model`` reads the store as it reads a real cache,
the batches are random token ids, and the journal is signed ``synthetic`` with the store's own
generator string. A folder without the store's ``SYNTHETIC.json`` marker is refused.

One journal, ``<output>/journal.jsonl``. Besides the recorder's rows it has ``environment``,
``phase_end`` (peak RSS, disk, seconds), ``training_clock`` (GPU clock samples during the steps),
``runtime`` and ``complete``. An existing output folder is refused.

Benchmark harness code, not shipped. Heavy imports sit inside the functions.

    python benchmarks/harness/b5_real_moe.py \\
        --config benchmarks/protocols/b5-moonlight-20261109/soup.yaml \\
        --source SRC --cache CACHE --output OUT \\
        --data benchmarks/protocols/b5-moonlight-20261109/data.json \\
        --model moonshotai/Moonlight-16B-A3B --revision 476b36a4... --kind real
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

try:
    from benchmarks.harness import b5_cold, b5_recorder, b5_synth_store
except ImportError:  # run as a script from benchmarks/harness
    import b5_cold  # type: ignore[no-redef]
    import b5_recorder  # type: ignore[no-redef]
    import b5_synth_store  # type: ignore[no-redef]

console = Console()

SEED = 20261109
CLIP_NORM = 1.0
#: Clock event reasons that make a speed reading untrustworthy (arm_validity.CLOCK_EVENT_REASONS).
THROTTLE_REASONS = ("sw_power_cap", "sw_thermal_slowdown", "hw_slowdown",
                    "hw_thermal_slowdown", "hw_power_brake_slowdown")


def _peak_rss() -> Optional[int]:
    """Peak resident bytes of this process: VmHWM on Linux, ru_maxrss elsewhere."""
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, ValueError):
        return None
    return peak if sys.platform == "darwin" else peak * 1024


def _reset_peak_rss() -> bool:
    try:
        from benchmarks.harness import host_box
    except ImportError:
        import host_box  # type: ignore[no-redef]
    return host_box.reset_peak_rss()


def _folder_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _version(name: str) -> Optional[str]:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@contextlib.contextmanager
def _meta_construction():
    # Accelerate redirects parameters after construction. Poison-filling a temporary CPU
    # torch.empty before that redirection touches giant arrays (the same guard as B6).
    import torch

    previous = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


def _clock_summary(samples: List[Any]) -> Dict[str, Any]:
    readings = [item.reading for item in samples if item.reading is not None]
    throttled = [item for item in readings if set(item["reasons"]) & set(THROTTLE_REASONS)]
    counts: Dict[str, int] = {}
    for item in readings:
        for reason in item["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    clocks = [item["sm_clock_mhz"] for item in readings]
    return {
        "samples": len(readings), "throttled_samples": len(throttled), "reason_counts": counts,
        "unavailable_samples": len(samples) - len(readings),
        "sm_clock_mhz": ({"min": min(clocks), "median": statistics.median(clocks),
                          "max": max(clocks)} if clocks else None),
    }


def _text_batches(args: argparse.Namespace, max_length: int, device: str) -> List[Dict[str, Any]]:
    """One batch per step: a text cut to exactly ``max_length`` tokens, so nothing is padded."""
    from transformers import AutoTokenizer

    texts = json.loads(Path(args.data).read_text(encoding="utf-8"))["train"]
    if len(texts) < args.steps:
        raise ValueError(f"{len(texts)} training texts for {args.steps} steps")
    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=False)
    batches = []
    for number, text in enumerate(texts[:args.steps]):
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length)["input_ids"]
        if ids.shape[1] != max_length:
            raise ValueError(f"text {number} has {ids.shape[1]} tokens, need {max_length}")
        batches.append({"input_ids": ids, "labels": ids.clone()})
    return batches


def _random_batches(vocab_size: int, steps: int, max_length: int) -> List[Dict[str, Any]]:
    """One batch per step of random token ids: a synthetic store has no tokenizer or text."""
    import torch

    batches = []
    for step in range(steps):
        generator = torch.Generator().manual_seed(SEED + step)
        ids = torch.randint(0, vocab_size, (1, max_length), generator=generator)
        batches.append({"input_ids": ids, "labels": ids.clone()})
    return batches


def _open_store(store: Path, *, dtype: str, quant: str) -> Any:
    """The marker and the index of a synthetic store, or a ValueError naming what is wrong."""
    from soup_cli.utils.layer_shard import read_shard_index

    marker = b5_synth_store.read_marker(store)
    index = read_shard_index(str(store))
    if index.dtype != dtype:
        raise ValueError(f"the store holds dtype {index.dtype}, this run needs {dtype}")
    if index.quant != quant:
        raise ValueError(f"the store is quant {index.quant!r}, the config asks for {quant!r}")
    if index.expert_layout != "per_expert":
        raise ValueError("the store is not a per_expert cache")
    return marker, index


def run(args: argparse.Namespace, *, batches: Optional[List[Dict[str, Any]]] = None) -> None:
    """Prepare the cache, train ``args.steps`` steps, save, reload; write the journal."""
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    import yaml
    from peft import LoraConfig, TaskType, load_peft_weights, set_peft_model_state_dict

    from soup_cli.config.schema import SoupConfig
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        assert_streamed_adapter_saved,
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
        shard_layout_of,
    )

    config = SoupConfig.model_validate(yaml.safe_load(Path(args.config).read_text("utf-8")))
    tcfg = config.training
    if tcfg.stream_expert_layout != "per_expert":
        raise ValueError("the config must set stream_expert_layout: per_expert")
    quant = "nf4" if tcfg.quantization == "4bit" else "none"
    store = Path(args.store) if getattr(args, "store", None) else None
    if store is not None:
        source = cache = store
        if args.kind != "synthetic":
            raise ValueError("a synthetic store can only be run with --kind synthetic")
    else:
        source, cache = Path(args.source), Path(args.cache)
    output = Path(args.output)
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if source_config.get("model_type") != "deepseek_v3":
        raise ValueError("this harness accepts a deepseek_v3 checkpoint only")
    max_length = config.data.max_length
    device = args.device
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("the run needs exactly one CUDA device")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("the rule needs bf16 without emulation (compute capability 8.0+)")
        dtype = "bfloat16"
        props = torch.cuda.get_device_properties(0)
    else:
        dtype, props = "float32", None
    tier = args.tier or tcfg.stream_source
    marker, prebuilt = _open_store(store, dtype=dtype, quant=quant) if store else (None, None)
    free_before = shutil.disk_usage(cache.parent if cache.parent.exists() else Path(".")).free
    output.mkdir(parents=True, exist_ok=False)
    recorder = b5_recorder.AdapterRecorder(output / "journal.jsonl")
    torch.manual_seed(SEED)

    def event(name: str, **values: Any) -> None:
        recorder.event(name, **values)
        console.print({"event": name, **{k: v for k, v in values.items() if k != "argv"}})

    if store is not None:
        recorder.signature("synthetic", args.model or f"synthetic MoE store {store.name}",
                           generator=marker["generator"])
    elif args.kind == "real":
        recorder.signature("real", args.model, revision=args.revision)
    else:
        recorder.signature("synthetic", args.model, generator=args.generator)
    lora = LoraConfig(r=tcfg.lora.r, lora_alpha=tcfg.lora.alpha, lora_dropout=tcfg.lora.dropout,
                      bias="none", target_modules=list(tcfg.lora.target_modules),
                      task_type=TaskType.CAUSAL_LM)
    event(
        "environment", argv=sys.argv, python=platform.python_version(),
        platform=platform.platform(), gpu=props.name if props else "cpu",
        device_count=torch.cuda.device_count() if props else 0,
        vram_bytes=props.total_memory if props else None,
        compute_capability=[props.major, props.minor] if props else None, dtype=dtype,
        quant=quant, double_quant=False, expert_layout="per_expert", tier=tier,
        trust_remote_code=False, lora_targets=list(tcfg.lora.target_modules),
        lora_r=tcfg.lora.r, lora_alpha=tcfg.lora.alpha, lr=tcfg.lr, max_length=max_length,
        steps=args.steps, seed=SEED, config_sha256=_sha256(Path(args.config)),
        versions={name: _version(name) for name in
                  ("torch", "transformers", "peft", "bitsandbytes", "tiktoken", "safetensors",
                   "accelerate", "soup-cli")},
        cuda=torch.version.cuda, host_ram_bytes=_host_ram(), ram_limit=_ram_limit(),
        disk_free_bytes=free_before,
        source_bytes=_folder_bytes(source), store=store.name if store else None,
    )

    # ---- preparation: the new per-expert cache ------------------------------------------
    started = time.perf_counter()
    _reset_peak_rss()
    if store is not None:
        index = prebuilt  # nothing to shard: the store is the cache
    elif quant == "nf4":
        with _meta_construction():
            skeleton = build_meta_skeleton(str(source), dtype=dtype, quant="nf4",
                                           double_quant=False)
        suffixes, layout = quantised_layer_suffixes(skeleton), shard_layout_of(skeleton)
        del skeleton
        gc.collect()
        budget = new_expert_cache_budget([str(path) for path in source.glob("*.safetensors")])
        if free_before < budget:
            raise ValueError(f"the new cache needs {budget} free bytes, {free_before} available")
        event("cache_budget", bytes=budget)
        index = shard_checkpoint(
            str(source), str(cache), dtype=dtype, arch="deepseek_v3", layout=layout, quant="nf4",
            quant_suffixes=suffixes, quant_device=device, double_quant=False,
            expert_layout="per_expert")
    else:
        index = shard_checkpoint(str(source), str(cache), dtype=dtype, arch="deepseek_v3",
                                 expert_layout="per_expert")
    event("phase_end", phase="preparation", seconds=time.perf_counter() - started,
          peak_rss_bytes=_peak_rss(), cache_bytes=_folder_bytes(cache),
          disk_free_bytes=shutil.disk_usage(cache).free, source_parameters=index.total_params,
          prebuilt_store=store is not None)

    if store is not None:
        event("shape", **{key: marker["shape"][key] for key in
                          ("layers", "moe_layers", "experts", "hidden", "expert_size")},
              parameters=index.total_params, store_bytes=b5_synth_store.store_bytes(store))

    def build() -> Any:
        with _meta_construction():
            return build_streamed_model(
                model_id=str(source), shard_dir=str(cache), index=index, lora_config=lora,
                device=device, dtype=dtype, buffers=2, pin=False, seed=SEED, quant=quant,
                double_quant=False, tier=tier, read_ahead=1)

    if batches is None:
        batches = (_random_batches(source_config["vocab_size"], args.steps, max_length)
                   if store is not None else _text_batches(args, max_length, device))
    batches = [{key: value.to(device) for key, value in batch.items()} for batch in batches]
    if len(batches) < args.steps:
        raise ValueError(f"{len(batches)} batches for {args.steps} steps")

    if getattr(args, "cold", False) and store is None:
        raise ValueError("--cold needs --store: only a store of known files can be dropped")

    # ---- training ---------------------------------------------------------------------
    model, runtime = build()
    adapter = output / "adapter"
    try:
        params = b5_recorder.adapter_params(model)
        optimizer = torch.optim.AdamW(list(params.values()), lr=tcfg.lr)
        recorder.start(params)
        _reset_peak_rss()
        if getattr(args, "cold", False):
            # Last thing before step 0: the model build above may have read some files. The
            # drop runs before the clock sampler starts, so its idle seconds are not samples.
            event("cold_cache", **b5_cold.drop_page_cache(store))
        sampler = None
        if device == "cuda":
            try:
                from benchmarks.harness import arm_validity
            except ImportError:
                import arm_validity  # type: ignore[no-redef]
            sampler = arm_validity.Sampler("gpu_clock", arm_validity.gpu_clock, 1.0)
            torch.cuda.reset_peak_memory_stats()
            sampler.start()
        started = time.perf_counter()
        event("training_start")
        model.train()
        try:
            for step in range(args.steps):
                inputs = batches[step]
                tokens = int(inputs["input_ids"].numel())  # batch of one, no padding
                before = runtime.stats()
                read_before = b5_cold.process_read_bytes()
                optimizer.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.synchronize()
                step_started = time.perf_counter()
                loss = model(**inputs).loss
                loss.backward()
                recorder.gradients(params)  # before clipping, as the rule says
                torch.nn.utils.clip_grad_norm_(list(params.values()), CLIP_NORM)
                optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - step_started
                after = runtime.stats()
                read_after = b5_cold.process_read_bytes()
                recorder.step(
                    step, float(loss.detach()), params, seconds=seconds, tokens=tokens,
                    tokens_per_second=tokens / seconds,
                    peak_allocated=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                    peak_reserved=torch.cuda.max_memory_reserved() if device == "cuda" else None,
                    expert_fetches=after["expert_fetches"] - before["expert_fetches"],
                    expert_bytes_read=after["expert_bytes_read"] - before["expert_bytes_read"],
                    disk_read_bytes=(None if read_before is None or read_after is None
                                     else read_after - read_before),
                    peak_rss_bytes=_peak_rss())
                console.print({"step": step, "loss": float(loss.detach()), "seconds": seconds})
        finally:
            if sampler is not None:
                sampler.stop()
        event("training_clock", **(_clock_summary(sampler.samples) if sampler else
                                   {"samples": 0, "throttled_samples": 0}))
        event("phase_end", phase="training", seconds=time.perf_counter() - started,
              peak_rss_bytes=_peak_rss(), disk_free_bytes=shutil.disk_usage(cache).free)
        model.save_pretrained(adapter)
        assert_streamed_adapter_saved(model, str(adapter))
        event("runtime", stats=_json_safe(runtime.stats()))
    finally:
        runtime.close()
    del model, runtime, optimizer, params
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- reload: the saved adapter, read back into a fresh model -----------------------------
    model, runtime = build()
    try:
        set_peft_model_state_dict(model, load_peft_weights(str(adapter), device="cpu"))
        recorder.reload(b5_recorder.adapter_params(model))
    finally:
        runtime.close()
    event("complete")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _host_ram() -> Optional[int]:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        return None


def _ram_limit() -> Dict[str, Any]:
    """The cgroup memory limit reading (``host_box.ram_limit``); None limit means unlimited."""
    try:
        from benchmarks.harness import host_box
    except ImportError:
        import host_box  # type: ignore[no-redef]
    return host_box.ram_limit()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output"):
        parser.add_argument("--" + name, required=True)
    for name in ("source", "cache", "model"):
        parser.add_argument("--" + name)
    parser.add_argument("--store", help="a synthetic store from b5_synth_store.py")
    parser.add_argument("--data")
    parser.add_argument("--kind", choices=("real", "synthetic"), default="real")
    parser.add_argument("--revision")
    parser.add_argument("--generator")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--tier", choices=("ram", "disk"))
    parser.add_argument("--cold", action="store_true",
                        help="drop the page cache of the --store right before step 0")
    parser.add_argument("--shard-threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.cold and not args.store:
        parser.error("--cold needs --store: only a store of known files can be dropped")
    if args.store:
        args.kind = "synthetic"
    else:
        missing = [name for name in ("source", "cache", "model") if not getattr(args, name)]
        if missing:
            parser.error("need " + ", ".join("--" + name for name in missing) + " (or --store)")
        if args.kind == "real" and not args.revision:
            parser.error("--kind real needs --revision")
        if args.kind == "synthetic" and not args.generator:
            parser.error("--kind synthetic needs --generator")
        if not args.data:
            parser.error("--data is required for a text run")
    run(args)
    return 0


if __name__ == "__main__":
    import faulthandler
    import signal

    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)
    sys.exit(main())
