"""H3: exercise Soup's SFT wrapper, then reload its adapter in another process.

See benchmarks/h3-fsdp-lora-20261026.md. CUDA results are pending until rented
hardware is approved. The offline CPU fixture is explicitly not a scaling arm.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from h3_fsdp_gate import (
    accumulation,
    adapter_delta,
    assert_adapter_equal,
    assert_data_window,
    assert_fsdp2_shards,
    assert_shards,
    parameter_local_numel,
    state_digest,
    tensor_digest,
    throughput,
)
from rich.console import Console

console = Console()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(argv: list[str]) -> str:
    cwd = Path(__file__).parents[2] if argv[0] == "git" else None
    result = subprocess.run(argv, capture_output=True, text=True, check=False, cwd=cwd)
    return result.stdout if result.returncode == 0 else result.stderr


def prepare(args: argparse.Namespace) -> None:
    """Resolve the base to an immutable snapshot and materialize fixed-length data."""
    from datasets import Dataset
    from transformers import AutoConfig

    root = Path(os.path.realpath(args.prepared))
    root.mkdir(parents=True, exist_ok=False)
    if args.cpu_fixture:
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(42)
        base = root / "tiny-base"
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=1024,
                             bos_token_id=1, eos_token_id=2, pad_token_id=0)
        Qwen2ForCausalLM(config).save_pretrained(base)
        tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t3"))
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="t0",
                               bos_token="t1", eos_token="t2").save_pretrained(base)
        revision = "offline-random-fixture"
    else:
        from huggingface_hub import model_info, snapshot_download

        revision = model_info(args.base, revision=args.revision).sha
        base = Path(snapshot_download(args.base, revision=revision,
                                     allow_patterns=["*.json", "*.safetensors", "*.txt",
                                                     "*.model", "*.jinja"]))
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    generator = random.Random(314159)
    rows = []
    for _ in range(args.rows):
        ids = [generator.randrange(4, config.vocab_size) for _ in range(args.sequence_length)]
        rows.append({"input_ids": ids, "labels": ids.copy(),
                     "attention_mask": [1] * len(ids)})
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    Dataset.from_list(rows).save_to_disk(str(root / "data"))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in base.iterdir() if path.is_file()}
    write_json(root / "prepared.json", {
        "base": str(base), "model_id": args.base, "revision": revision,
        "base_files_sha256": hashes, "data_sha256": digest, "rows": args.rows,
        "sequence_length": args.sequence_length, "cpu_fixture": args.cpu_fixture,
        "description": "SYNTHETIC seeded random tokens; plumbing, not quality evidence",
    })
    console.print(f"Prepared {args.rows} fixed-length rows at {root}")


class RamSampler:
    """Sample this rank's RSS and host used RAM every 50 ms, outside CUDA work."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.rank_rss_peak = 0
        self.host_used_peak = 0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        import psutil

        process = psutil.Process()
        while not self.stop_event.is_set():
            self.rank_rss_peak = max(self.rank_rss_peak, process.memory_info().rss)
            self.host_used_peak = max(self.host_used_peak, psutil.virtual_memory().used)
            self.stop_event.wait(0.05)

    def __enter__(self) -> RamSampler:
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop_event.set()
        self.thread.join()


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from fsdp_sharding_probe import module_class_histogram
    from peft import get_peft_model_state_dict
    from torch.distributed.fsdp import FSDPModule, FullyShardedDataParallel, MixedPrecisionPolicy
    from transformers import TrainerCallback

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.fsdp import get_fsdp_training_args

    prepared = read_json(Path(args.prepared) / "prepared.json")
    for name, expected in prepared["base_files_sha256"].items():
        path = Path(prepared["base"]) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"base snapshot changed: {name}")
    cpu = args.device == "cpu"
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if cpu and (not prepared["cpu_fixture"] or world != 1):
        raise ValueError("CPU mode is only for the offline one-process fixture")
    if not cpu:
        if platform.system() != "Linux" or not torch.cuda.is_available():
            raise ValueError("the measured pilot requires Linux + CUDA")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is required; do not silently change precision")
    if world not in (1, 2):
        raise ValueError("this pilot only covers one or two GPUs")
    accum = accumulation(args.global_batch, args.micro_batch, world)
    steps = args.correctness_steps + args.warmup_steps + args.measured_steps + 1
    if prepared["rows"] < steps * args.global_batch:
        raise ValueError("dataset too small: would repeat rows within the pilot")
    output = Path(os.path.realpath(args.output))
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"rank-{rank}.json").exists() or (output / "adapter_model.safetensors").exists():
        raise ValueError("refusing to overwrite an earlier arm")
    config = SoupConfig.model_validate({
        "base": prepared["base"], "output": str(output),
        "data": {"train": str(Path(args.prepared) / "data"), "format": "pre_tokenized",
                 "tokenized_path": str(Path(args.prepared) / "data"), "val_split": 0,
                 "max_length": prepared["sequence_length"]},
        "training": {"batch_size": args.micro_batch, "gradient_accumulation_steps": accum,
                     "epochs": 1, "lr": 0.0002, "weight_decay": 0, "warmup_ratio": 0,
                     "scheduler": "constant", "optimizer": "adamw_torch",
                     "seed": 42, "data_seed": 42, "gradient_checkpointing": False,
                     "quantization": "none", "packing": False, "rewind_log": False,
                     "logging_steps": steps + 1, "save_steps": steps + 1,
                     "lora": {"r": 8, "alpha": 16, "dropout": 0,
                              "target_modules": ["q_proj", "v_proj"]}},
    })
    fsdp = get_fsdp_training_args("full_shard") if world == 2 else None
    wrapper = SFTTrainerWrapper(config, device=args.device, fsdp_config=fsdp)
    setup_start = time.perf_counter()
    wrapper.setup({"train": []})
    trainer = wrapper.trainer
    plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
    if world == 2 and getattr(plugin, "fsdp_version", None) == 2:
        # Fixed pilot control, declared before the first distributed optimizer
        # step: match the baseline's FP32 LoRA gradient accumulation/reduction.
        plugin.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16)
    actual_rows = [{key: row[key] for key in ("input_ids", "labels", "attention_mask")}
                   for row in trainer.train_dataset]
    actual_hash = hashlib.sha256(json.dumps(actual_rows, separators=(",", ":")).encode())
    if actual_hash.hexdigest() != prepared["data_sha256"]:
        raise ValueError("actual Trainer data differs from the prepared token/label dataset")
    if bool(trainer.args.bf16) != (not cpu) or trainer.args.fp16 or trainer.args.torch_compile:
        raise ValueError("actual precision/compile differs from pilot specification")
    trainer.args.max_steps = steps
    trainer.args.disable_tqdm = True
    trainer.args.save_strategy = "steps"
    trainer.args.save_steps = steps
    trainer.args.logging_strategy = "no"
    trainer.model.config.use_cache = False
    initial_adapter = {key: value.detach().cpu().clone()
                       for key, value in get_peft_model_state_dict(wrapper.model).items()}
    total = sum(p.numel() for p in wrapper.model.parameters())
    prewrap_shapes = {}
    prewrap_aliases = {}
    for name, parameter in wrapper.model.named_parameters(remove_duplicate=False):
        prewrap_shapes[name] = list(parameter.shape)
        prewrap_aliases.setdefault(id(parameter), []).append(name)
    alias_groups = list(prewrap_aliases.values())
    fixed_ids = torch.tensor([trainer.train_dataset[0]["input_ids"][:64]],
                             device=trainer.args.device)
    setup_seconds = time.perf_counter() - setup_start
    # Benchmark-only controls are recorded separately; product config stays SoupConfig.
    write_json(output / f"config-rank-{rank}.json", config.model_dump(mode="json"))

    def gathered(value: Any) -> list:
        if world == 1:
            return [value]
        results = [None] * world
        dist.all_gather_object(results, value)
        return results

    def sync() -> None:
        if not cpu:
            torch.cuda.synchronize()

    def full_context(with_grads: bool = False) -> Any:
        if world == 1 or record["fsdp_version"] == 2:
            return contextlib.nullcontext()
        return FullyShardedDataParallel.summon_full_params(
            trainer.model_wrapped, writeback=False, with_grads=with_grads)

    def adapter_snapshot(gradients: bool = False) -> dict:
        result = {}
        with full_context(with_grads=gradients):
            for name, parameter in trainer.model_wrapped.named_parameters():
                if not parameter.requires_grad:
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace(".default.", ".")
                value = parameter.grad if gradients else parameter
                if value is None:
                    raise ValueError(f"missing adapter gradient: {name}")
                if callable(getattr(value, "full_tensor", None)):
                    value = value.full_tensor()
                result[name] = value.detach().cpu().clone()
        if result.keys() != initial_adapter.keys():
            raise ValueError("canonical adapter keys changed under wrapping")
        return result

    def base_digest() -> str:
        return state_digest({name: p.to_local() if callable(getattr(p, "to_local", None)) else p
                             for name, p in trainer.model_wrapped.named_parameters()
                             if not p.requires_grad})

    record: dict[str, Any] = {"status": "incomplete", "world_size": world, "rank": rank,
                              "cpu_fixture": cpu, "setup_seconds": setup_seconds,
                              "gradient_accumulation_steps": accum,
                              "micro_batch": args.micro_batch, "global_batch": args.global_batch,
                              "steps": steps, "correctness_steps": args.correctness_steps,
                              "warmup_steps": args.warmup_steps,
                              "measured_steps": args.measured_steps}
    times: list[float] = []
    losses = []
    windows: dict[int, list[str]] = {}
    counts: dict[int, int] = {}
    input_tokens: dict[int, int] = {}
    sampler = RamSampler()
    original_step = trainer.training_step

    def observed_step(model: Any, inputs: dict, *rest: Any, **kwargs: Any) -> Any:
        step = trainer.state.global_step
        ids = inputs["input_ids"]
        if ids.shape != (args.micro_batch, prepared["sequence_length"]):
            raise ValueError(f"actual micro batch/sequence shape differs: {ids.shape}")
        counts[step] = counts.get(step, 0) + 1
        input_tokens[step] = input_tokens.get(step, 0) + ids.numel()
        if step < args.correctness_steps:
            if not torch.all(inputs["attention_mask"] == 1).item():
                raise ValueError("unexpected padding invalidates token accounting")
            if not torch.equal(inputs["labels"], ids):
                raise ValueError("unexpected label masking invalidates pilot loss accounting")
            windows.setdefault(step, []).extend(tensor_digest(row) for row in ids)
        loss = original_step(model, inputs, *rest, **kwargs)
        losses.append(loss.detach())  # inspect outside the measured window
        return loss

    trainer.training_step = observed_step
    start_measure = args.correctness_steps + args.warmup_steps
    end_measure = start_measure + args.measured_steps

    class Observer(TrainerCallback):
        def on_train_begin(self, callback_args: Any, state: Any, control: Any,
                           **kwargs: Any) -> None:
            plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
            record["fsdp_version"] = getattr(plugin, "fsdp_version", None)
            policy = getattr(plugin, "mixed_precision_policy", None)
            record["fsdp_policy"] = ({name: str(getattr(policy, name, None)) for name in
                                      ("param_dtype", "reduce_dtype", "output_dtype")}
                                     if policy is not None else None)
            if world == 2 and record["fsdp_version"] not in (1, 2):
                raise ValueError("unknown FSDP version; refuse to infer sharding")
            histogram = dict(module_class_histogram(trainer.model_wrapped))
            units = (sum(isinstance(m, FSDPModule) for m in trainer.model_wrapped.modules())
                     if record["fsdp_version"] == 2
                     else histogram.get("FullyShardedDataParallel", 0))
            local = sum(parameter_local_numel(p) for p in trainer.model_wrapped.parameters())
            shards = gathered({"local": local, "units": units})
            record["shards"] = shards
            record["logical_total_parameters"] = total
            record["wrapper_classes"] = histogram
            record["parameter_storage"] = {
                name: {"global_shape": list(p.shape),
                       "local_numel": parameter_local_numel(p),
                       "dtype": str(p.dtype),
                       "sharded": any(placement.is_shard()
                                      for placement in getattr(p, "placements", ())),
                       "placements": str(getattr(p, "placements", None)),
                       "trainable": p.requires_grad}
                for name, p in trainer.model_wrapped.named_parameters()
            }
            record["prewrap_tied_aliases"] = [group for group in alias_groups if len(group) > 1]
            if world == 2 and record["fsdp_version"] == 2:
                if not all(row["units"] for row in shards):
                    raise ValueError("FSDP2 modules missing on a rank")
                storage_rows = gathered(record["parameter_storage"])
                record["sharding_accounting"] = assert_fsdp2_shards(
                    storage_rows, prewrap_shapes, alias_groups)
            elif world == 2:
                assert_shards([r["local"] for r in shards], total,
                              [r["units"] for r in shards])
            elif local != total or units:
                raise ValueError("single-GPU control must hold full parameters without FSDP")
            record["base_before_sha256"] = base_digest()
            assert_adapter_equal(initial_adapter, adapter_snapshot())

        def on_step_begin(self, callback_args: Any, state: Any, control: Any,
                          **kwargs: Any) -> None:
            if state.global_step == start_measure:
                if world > 1:
                    dist.barrier()
                sync()
                if not cpu:
                    torch.cuda.reset_peak_memory_stats()
                sampler.__enter__()
            if start_measure <= state.global_step < end_measure:
                sync()
                self.started = time.perf_counter()

        def on_pre_optimizer_step(self, callback_args: Any, state: Any, control: Any,
                                  **kwargs: Any) -> None:
            # Extra inspection steps sit outside throughput. Gradients here have
            # been reduced and clipped by Trainer; zero A at LoRA init is valid.
            if state.global_step < args.correctness_steps or state.global_step == end_measure:
                if any(p.grad is not None for p in trainer.model_wrapped.parameters()
                       if not p.requires_grad):
                    raise ValueError("frozen base unexpectedly has gradients")
                grads = adapter_snapshot(gradients=True)
                if any(value.dtype != torch.float32 for value in grads.values()):
                    raise ValueError("pilot requires FP32 LoRA gradients in both arms")
                if not all(torch.isfinite(v).all().item() for v in grads.values()):
                    raise ValueError("nonfinite adapter gradient")
                if not any(torch.count_nonzero(v).item() for v in grads.values()):
                    raise ValueError("all adapter gradients are zero")
                hashes = gathered(state_digest(grads))
                if len(set(hashes)) != 1:
                    raise ValueError("reconstructed reduced gradients disagree across ranks")
                record.setdefault("gradient_checks", []).append({
                    "optimizer_step": state.global_step + 1, "finite": True,
                    "some_nonzero": True, "frozen_gradients_absent": True,
                    "reconstructed_sha256_per_rank": hashes})
                if state.global_step == 0:
                    if rank == 0:
                        torch.save(grads, output / "first-gradients.pt")
                    record["first_gradient_sha256"] = hashes[0]

        def on_step_end(self, callback_args: Any, state: Any, control: Any,
                        **kwargs: Any) -> None:
            completed = state.global_step - 1
            if start_measure <= completed < end_measure:
                sync()
                times.append(time.perf_counter() - self.started)
            if counts.get(completed) != accum:
                raise ValueError(f"actual accumulation {counts.get(completed)} != {accum}")
            if completed < args.correctness_steps:
                rows = gathered(windows[completed])
                assert_data_window(rows, args.global_batch)
                record.setdefault("data_windows", []).append(sorted(row for r in rows for row in r))
                adapted = adapter_snapshot()
                record.setdefault("updates", []).append(adapter_delta(initial_adapter, adapted))
                hashes = gathered(state_digest(adapted))
                if len(set(hashes)) != 1:
                    raise ValueError("reconstructed adapter disagrees across ranks")
                if completed == 0 and rank == 0:
                    torch.save(adapted, output / "first-adapter.pt")
            if state.global_step == end_measure:
                sampler.__exit__()
                record["training_memory"] = {
                    "allocated_peak_bytes": None if cpu else torch.cuda.max_memory_allocated(),
                    "reserved_peak_bytes": None if cpu else torch.cuda.max_memory_reserved(),
                    "rank_rss_peak_sampled_bytes": sampler.rank_rss_peak,
                    "host_used_peak_sampled_bytes": sampler.host_used_peak,
                    "host_sample_interval_seconds": 0.05,
                }

        def on_train_end(self, callback_args: Any, state: Any, control: Any,
                         **kwargs: Any) -> None:
            record["base_after_sha256"] = base_digest()
            if record["base_before_sha256"] != record["base_after_sha256"]:
                raise ValueError("frozen base shards changed")
            if not all(torch.isfinite(loss).all().item() for loss in losses):
                raise ValueError("nonfinite training loss")
            record["losses"] = [float(loss.cpu()) for loss in losses]
            adapted = adapter_snapshot()
            record["final_update"] = adapter_delta(initial_adapter, adapted)
            trainer.model_wrapped.eval()
            with torch.no_grad(), trainer.compute_loss_context_manager():
                logits = trainer.model_wrapped(input_ids=fixed_ids,
                                               attention_mask=torch.ones_like(fixed_ids)).logits
            if not torch.isfinite(logits).all().item():
                raise ValueError("nonfinite final eval logits")
            if rank == 0:
                torch.save({"input_ids": fixed_ids.cpu(), "logits": logits.cpu(),
                            "adapter": adapted}, output / "reload-reference.pt")
            record["rank_step_seconds"] = times
            record["rank_input_tokens"] = sum(input_tokens[s]
                                               for s in range(start_measure, end_measure))
            results = gathered({"seconds": times, "tokens": record["rank_input_tokens"]})
            record.update(throughput([r["seconds"] for r in results],
                                     [r["tokens"] for r in results]))

    trainer.add_callback(Observer())
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "accelerate", "peft", "trl", "datasets", "psutil")}
    write_json(output / f"environment-rank-{rank}.json", {
        "versions": versions, "python": sys.version, "platform": platform.platform(),
        "commit": command_output(["git", "rev-parse", "HEAD"]).strip(),
        "git_status": command_output(["git", "status", "--porcelain"]),
        "cuda": torch.version.cuda, "argv": sys.argv,
        "device": None if cpu else torch.cuda.get_device_name(), "pid": os.getpid(),
        "host": platform.node(), "nccl": None if cpu else torch.cuda.nccl.version(),
        "harness_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("h3_fsdp_gate.py", "h3_fsdp_run.py", "fsdp_sharding_probe.py")},
    })
    if rank == 0:
        (output / "pip-freeze.txt").write_text(command_output([sys.executable, "-m", "pip",
                                                              "freeze"]), encoding="utf-8")
        if not cpu:
            (output / "nvidia-smi.txt").write_text(command_output(["nvidia-smi", "-q"]),
                                                  encoding="utf-8")
            (output / "topology.txt").write_text(command_output(["nvidia-smi", "topo", "-m"]),
                                                encoding="utf-8")
    try:
        wrapper.train()
        # All ranks participate in save; reload happens only after torchrun exits.
        if world > 1:
            dist.barrier()
        record["checkpoint_files"] = {
            str(p.relative_to(output)): p.stat().st_size
            for p in output.rglob("*") if p.is_file()
        }
        record["status"] = "training_passed_reload_pending"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if sampler.thread.is_alive():
            sampler.__exit__()
        write_json(output / f"rank-{rank}.json", record)
        if dist.is_initialized():
            dist.destroy_process_group()


def reload_adapter(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    root = Path(args.output)
    prepared = read_json(Path(args.prepared) / "prepared.json")
    cpu = args.device == "cpu"
    reference = torch.load(root / "reload-reference.pt", map_location="cpu", weights_only=True)
    adapter_dir = root
    if args.checkpoint:
        adapter_dir = root / f"checkpoint-{read_json(root / 'rank-0.json')['steps']}"
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert_adapter_equal(reference["adapter"], saved)
    dtype = torch.float32 if cpu else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(prepared["base"], torch_dtype=dtype,
                                              local_files_only=True).to(args.device)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()
    assert_adapter_equal(saved, get_peft_model_state_dict(model))
    ids = reference["input_ids"].to(args.device)
    context = contextlib.nullcontext() if cpu else torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.no_grad(), context:
        actual = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.cpu()
    # BF16: one relative ULP plus two absolute ULPs near one. Same dtype,
    # backend and input; exact adapter key/value equality is checked separately.
    atol, rtol = (1e-5, 1e-5) if cpu else (1 / 64, 1 / 128)
    torch.testing.assert_close(actual.float(), reference["logits"].float(), atol=atol, rtol=rtol)
    error = float((actual.float() - reference["logits"].float()).abs().max())
    target = "reload-checkpoint.json" if args.checkpoint else "reload.json"
    write_json(root / target, {"status": "passed", "pid": os.getpid(),
                                      "adapter_path": str(adapter_dir),
                                      "atol": atol, "rtol": rtol,
                                      "max_abs_logit_error": error,
                                      "adapter_tensors": len(saved),
                                      "resume": "unmeasured"})
    console.print(f"Adapter reload passed: {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--prepared", required=True)
    prep.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    prep.add_argument("--revision", default="main")
    prep.add_argument("--rows", type=int, default=512)
    prep.add_argument("--sequence-length", type=int, default=512)
    prep.add_argument("--cpu-fixture", action="store_true")
    for command in ("train", "reload"):
        sub = commands.add_parser(command)
        sub.add_argument("--prepared", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if command == "train":
            sub.add_argument("--global-batch", type=int, default=8)
            sub.add_argument("--micro-batch", type=int, default=1)
            sub.add_argument("--correctness-steps", type=int, default=2)
            sub.add_argument("--warmup-steps", type=int, default=5)
            sub.add_argument("--measured-steps", type=int, default=20)
        else:
            sub.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.rows < 1 or args.sequence_length < 64:
            parser.error("rows must be positive and sequence length >= 64")
        prepare(args)
    elif args.command == "train":
        if min(args.correctness_steps, args.warmup_steps, args.measured_steps) < 1:
            parser.error("correctness, warmup and measured steps must be positive")
        run(args)
    else:
        reload_adapter(args)


if __name__ == "__main__":
    main()
