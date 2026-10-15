"""D1: checkpoint-safe NF4 correctness, projection timing and real-data training.

Run from a repository checkout. CUDA runs require the rule and harness committed
before the first arm. CPU correctness is preparation only. No cloud API is used.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from d1_rules import (  # noqa: E402
    compare_tensors,
    digest,
    evidence_path,
    negative_control,
    peer_check,
    settings_digest,
    summarize,
    write_new_json,
)

RULE = ROOT / "benchmarks" / "gate-d1-nf4-rule.md"
CONSOLE = Console()
SEED = 842


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def provenance(require_clean: bool) -> dict:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if require_clean and dirty:
        raise RuntimeError("commit the rule and source before CUDA arms; tracked tree is dirty")
    if require_clean:
        git("ls-files", "--error-unmatch", "benchmarks/gate-d1-nf4-rule.md")
        for name in ("d1_nf4_study.py", "d1_rules.py", "d1_qa.py"):
            git("ls-files", "--error-unmatch", "benchmarks/harness/" + name)
    packages = {}
    for name in ("torch", "bitsandbytes", "transformers", "peft", "psutil"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "source_sha": git("rev-parse", "HEAD"),
        "rule_sha256": digest(RULE),
        "packages": packages,
        "python": sys.version,
        "dirty": dirty,
        "harness_sha256": digest(Path(__file__)),
        "runtime_sha256": digest(ROOT / "src/soup_cli/utils/layer_stream_runtime.py"),
        "qa_harness_sha256": digest(Path(__file__).with_name("d1_qa.py")),
    }


def configure(device: str, dtype: str):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no GPU result can be produced on this machine")
    if device == "cpu" and dtype != "float32":
        raise ValueError("CPU preparation uses float32; half-precision acceptance requires CUDA")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and dtype == "bfloat16":
        probe = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        if not bool(torch.isfinite(probe @ probe).all()):
            raise RuntimeError("native CUDA bf16 probe failed")
        del probe
    return torch, getattr(torch, dtype)


def tensor_bundle(path: Path, tensors: dict) -> dict:
    import torch

    if path.exists():
        raise FileExistsError(path)
    torch.save({name: value.detach().cpu().clone() for name, value in tensors.items()}, path)
    return {"file": path.name, "sha256": digest(path)}


def _gemm(x, packed, state):
    import torch

    kwargs = {}
    scales = state.absmax
    if state.nested:
        scales = state.state2.absmax
        kwargs = {
            "absmax_8bit": state.absmax,
            "absmax_code": state.state2.code,
            "absmax_offset": state.offset,
        }
    return torch.ops.bitsandbytes.gemm_4bit.default(
        x, packed, state.shape, scales, state.blocksize, state.quant_type, **kwargs
    )


def alias_control(directory: Path, device: str, dtype: str) -> dict:
    """Positive and intentionally broken lifetimes, with a >165 MiB CUDA weight."""
    torch, precision = configure(device, dtype)
    import bitsandbytes.functional as functional
    from torch.utils.checkpoint import checkpoint

    from soup_cli.utils.layer_stream_runtime import (
        _can_use_checkpoint_visible_nf4_gemm,
        checkpoint_visible_nf4_linear,
    )

    # Exceed the historical threshold even when interpreted as packed bytes.
    n, k = (32768, 12288) if device == "cuda" else (64, 64)
    sources = []
    for _index in range(2):
        weight = torch.randn(n, k, device=device, dtype=precision) / math.sqrt(k)
        sources.append(functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True))
        del weight
    (packed0, state0), (packed1, state1) = sources
    packed = packed0.clone()
    state = copy.deepcopy(state0)
    x0 = torch.randn(3, k, device=device, dtype=precision)
    dispatch = bool(_can_use_checkpoint_visible_nf4_gemm(x0, state))
    if device == "cuda" and not dispatch:
        raise RuntimeError("large-alias fixture has no fused dispatch; retain failure, re-scope")

    def refill(source, quant_state):
        with torch.no_grad():
            packed.copy_(source)
            for name in ("absmax", "code", "offset"):
                target = getattr(state, name, None)
                if target is not None:
                    target.copy_(getattr(quant_state, name))
            if state.nested:
                state.state2.absmax.copy_(quant_state.state2.absmax)
                state.state2.code.copy_(quant_state.state2.code)

    class PlainContext(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.packed = packed  # Intentionally broken negative control.
            ctx.state = state
            return _gemm(value, packed, state)

        @staticmethod
        def backward(ctx, grad):
            dense = functional.dequantize_4bit(ctx.packed, ctx.state).to(grad.dtype)
            return grad @ dense

    counters = {"safe": 0, "broken": 0}
    values = {}
    for mode in ("private", "safe", "broken"):
        value = x0.clone().requires_grad_(True)

        def body(current):
            counters[mode] += 1
            refill(packed0, state0)
            if mode == "broken":
                return PlainContext.apply(current)
            return checkpoint_visible_nf4_linear(current, packed, state)

        if mode == "private":
            output = checkpoint_visible_nf4_linear(value, packed0, state0)
        else:
            output = checkpoint(body, value, use_reentrant=False)
            refill(packed1, state1)
        output.float().square().mean().backward()
        values[mode] = {"output": output.detach(), "grad": value.grad.detach()}
    safe = compare_tensors(values["safe"], values["private"])
    detected = negative_control(
        {"grad": values["broken"]["grad"]}, {"grad": values["private"]["grad"]}
    )
    bundles = {
        mode: tensor_bundle(directory / f"alias-{mode}.pt", value) for mode, value in values.items()
    }
    dense_bytes = n * k * torch.empty((), dtype=precision).element_size()
    return {
        "passed": safe["passed"] and detected and counters == {"safe": 2, "broken": 1},
        "safe": safe,
        "negative_control_detected": detected,
        "calls": counters,
        "shape": [n, k],
        "dense_weight_bytes": dense_bytes,
        "packed_weight_bytes": packed0.numel() * packed0.element_size(),
        "above_165_mib": packed0.numel() * packed0.element_size() > 165 * 1024**2,
        "native_fused_dispatch": dispatch,
        "tensors": bundles,
    }


def streaming_control(directory: Path, device: str, dtype: str, seq: int) -> dict:
    """Reuse the production fixture; the resident model remains unpatched."""
    torch, _precision = configure(device, dtype)
    sys.path.insert(0, str(ROOT / "tests"))
    import bitsandbytes as bnb
    from test_v07202 import _nf4_stream, _randomise_lora_b, _resident_nf4, _sync_adapters

    from soup_cli.utils import layer_stream_runtime as runtime_module

    with tempfile.TemporaryDirectory(prefix="d1-stream-") as temporary:
        streamed, runtime, weights, _index, _shards = _nf4_stream(
            Path(temporary), n_layers=4, buffers=2, device=device, dtype=dtype
        )
        resident = _resident_nf4(weights, dtype=dtype, device=0 if device == "cuda" else "cpu")
        _randomise_lora_b(resident)
        if _sync_adapters(streamed, resident) <= 0:
            raise RuntimeError("no adapters synchronized")
        calls = {"function": 0, "streamed_matmul": 0, "resident_matmul": 0}
        decisions = []
        phase = "streamed"
        real_function = runtime_module.checkpoint_visible_nf4_linear
        real_matmul = bnb.matmul_4bit
        real_gate = runtime_module._can_use_checkpoint_visible_nf4_gemm

        def counted_function(*args, **kwargs):
            calls["function"] += 1
            return real_function(*args, **kwargs)

        def counted_matmul(*args, **kwargs):
            calls[phase + "_matmul"] += 1
            return real_matmul(*args, **kwargs)

        def counted_gate(*args, **kwargs):
            result = real_gate(*args, **kwargs)
            decisions.append(bool(result))
            return result

        runtime_module.checkpoint_visible_nf4_linear = counted_function
        runtime_module._can_use_checkpoint_visible_nf4_gemm = counted_gate
        bnb.matmul_4bit = counted_matmul
        bundles, tensors = {}, {}
        try:
            inputs = torch.randint(0, 64, (1, seq), device=device)
            for name, model in (("streamed", streamed), ("resident", resident)):
                phase = name
                model.train()
                logits = model(input_ids=inputs, use_cache=False).logits
                logits.float().square().mean().backward()
                parameters = {
                    key.replace(".inner.", "."): value
                    for key, value in model.named_parameters()
                    if value.requires_grad
                }
                if not parameters or any(value.grad is None for value in parameters.values()):
                    raise RuntimeError("missing trainable gradients")
                if not any(bool(value.grad.count_nonzero()) for value in parameters.values()):
                    raise RuntimeError("vacuous all-zero adapter gradients")
                tensors[name] = {
                    "logits": logits.detach().cpu(),
                    **{
                        "grad:" + key: value.grad.detach().cpu()
                        for key, value in parameters.items()
                    },
                }
                bundles[name] = tensor_bundle(directory / f"stream-{seq}-{name}.pt", tensors[name])
            comparison = compare_tensors(tensors["streamed"], tensors["resident"])
            route_ok = (
                calls["streamed_matmul"] == 0
                and calls["resident_matmul"] > 0
                and bool(decisions)
                and (calls["function"] > 0) == any(decisions)
            )
            if device == "cuda":
                route_ok = route_ok and any(decisions) == (seq <= 8)
            return {
                "passed": comparison["passed"] and route_ok,
                "comparison": comparison,
                "route_ok": route_ok,
                "calls": calls,
                "gate_decisions": decisions,
                "pool_loads": runtime.pool.loads,
                "tensors": bundles,
            }
        finally:
            runtime_module.checkpoint_visible_nf4_linear = real_function
            runtime_module._can_use_checkpoint_visible_nf4_gemm = real_gate
            bnb.matmul_4bit = real_matmul
            runtime.close()


def correctness(args) -> dict:
    source = provenance(args.device == "cuda")
    configure(args.device, args.dtype)
    alias = alias_control(args.output, args.device, args.dtype)
    sequences = (8, 128) if args.device == "cpu" else (3, 2048)
    streams = [streaming_control(args.output, args.device, args.dtype, seq) for seq in sequences]
    passed = alias["passed"] and all(row["passed"] for row in streams)
    return {
        **source,
        "status": "PASS"
        if passed and args.device == "cuda"
        else "CPU_PREPARATION_PASS"
        if passed
        else "FAIL",
        "device": args.device,
        "dtype": args.dtype,
        "alias": alias,
        "streaming": streams,
        "d1_complete": False,
    }


@contextlib.contextmanager
def route(mode: str):
    """Harness-only route choice; production configuration is unchanged."""
    from soup_cli.utils import layer_stream_runtime as runtime_module

    original = runtime_module._can_use_checkpoint_visible_nf4_gemm
    if mode != "native":
        runtime_module._can_use_checkpoint_visible_nf4_gemm = lambda *_args: mode == "forced"
    try:
        yield
    finally:
        runtime_module._can_use_checkpoint_visible_nf4_gemm = original


class GPUMonitor:
    """Read-only NVML lifecycle observations; unavailable is never clean."""

    def __init__(self):
        try:
            import pynvml
            from arm_validity import workload_gpu

            pynvml.nvmlInit()
            gpu = workload_gpu()["id"]
            if gpu is None:
                raise RuntimeError("no workload GPU")
            self.nvml = pynvml
            self.handle = (
                pynvml.nvmlDeviceGetHandleByUUID(gpu)
                if gpu.startswith("GPU-")
                else pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
            )
            self.error = None
        except Exception as exc:
            self.error = str(exc)

    def sample(self):
        result = {"time_ns": time.monotonic_ns(), "available": False}
        if self.error:
            return {**result, "error": self.error}
        try:
            nvml, handle = self.nvml, self.handle
            result.update(
                uuid=nvml.nvmlDeviceGetUUID(handle),
                compute=sorted(
                    {row.pid for row in nvml.nvmlDeviceGetComputeRunningProcesses(handle)}
                ),
                graphics=sorted(
                    {row.pid for row in nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)}
                ),
                sm_mhz=nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
                memory_util_pct=nvml.nvmlDeviceGetUtilizationRates(handle).memory,
                power_limit_mw=nvml.nvmlDeviceGetEnforcedPowerLimit(handle),
                power_mw=nvml.nvmlDeviceGetPowerUsage(handle),
                temperature_c=nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                throttle=nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
                driver=nvml.nvmlSystemGetDriverVersion(),
                available=True,
            )
        except Exception as exc:
            result["error"] = str(exc)
        return result


class Telemetry:
    """Sample SM clock and memory-controller utilization during timed work."""

    def __init__(self):
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.monitor = GPUMonitor()
        self.thread = threading.Thread(target=self.sample_loop, daemon=True)

    def sample_loop(self):
        while not self.stop_event.is_set():
            row = self.monitor.sample()
            self.samples.append(row)
            if not row["available"]:
                self.errors.append(row.get("error", "unavailable NVML"))
            self.stop_event.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop_event.set()
        self.thread.join(timeout=10)


def timed_steps(step, count: int, warmup: int) -> dict:
    import torch

    for _index in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pairs, wall_ms = [], []
    with Telemetry() as telemetry:
        started_ns = time.monotonic_ns()
        for _index in range(count):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            began = time.perf_counter()
            start.record()
            step()
            end.record()
            end.synchronize()
            wall_ms.append(1000 * (time.perf_counter() - began))
            pairs.append(start.elapsed_time(end))
        ended_ns = time.monotonic_ns()
    samples = [
        row
        for row in telemetry.samples
        if started_ns <= row["time_ns"] <= ended_ns and row["available"]
    ]
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    return {
        "step_cuda_ms": pairs,
        "step_wall_ms": wall_ms,
        "median_step_ms": statistics.median(wall_ms),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "clock_sm_mhz": [row["sm_mhz"] for row in samples],
        "power_limit_mw": [row["power_limit_mw"] for row in samples],
        "memory_util_pct": [row["memory_util_pct"] for row in samples],
        "telemetry": telemetry.samples,
        "telemetry_errors": telemetry.errors,
        "gpu_uuid": actual_uuid,
        "telemetry_gpu_matches_cuda": bool(samples)
        and all(row["uuid"] == actual_uuid for row in samples),
    }


def projection(args) -> dict:
    source = provenance(True)
    torch, precision = configure("cuda", args.dtype)
    import bitsandbytes.functional as functional
    from torch.utils.checkpoint import checkpoint

    from soup_cli.utils.layer_stream_runtime import (
        _can_use_checkpoint_visible_nf4_gemm,
        checkpoint_visible_nf4_linear,
    )

    weight = torch.randn(args.n, args.k, device="cuda", dtype=precision) / math.sqrt(args.k)
    packed, state = functional.quantize_4bit(
        weight, quant_type="nf4", compress_statistics=not args.no_nested
    )
    del weight
    value = torch.randn(args.m, args.k, device="cuda", dtype=precision, requires_grad=True)
    native_fused = bool(_can_use_checkpoint_visible_nf4_gemm(value, state))
    fused = args.route == "forced" or args.route == "native" and native_fused
    counts = {"function": 0, "dense": 0}

    def forward(current):
        counts["function" if fused else "dense"] += 1
        if fused:
            return checkpoint_visible_nf4_linear(current, packed, state)
        dense = functional.dequantize_4bit(packed, state).to(current.dtype)
        return torch.nn.functional.linear(current, dense)

    def step():
        value.grad = None
        result = checkpoint(forward, value, use_reentrant=False)
        result.float().square().mean().backward()

    measured = timed_steps(step, args.steps, args.warmup)
    # Oracle tensors and the dense GEMM ceiling are created AFTER recording peaks.
    from bitsandbytes.backends.cuda import ops as cuda_ops

    backend_counts = {"custom": 0, "fallback": 0}
    real_custom = cuda_ops._gemm_4bit_kernel_impl
    real_fallback = cuda_ops._dequant_linear_fallback

    def counted_custom(*values, **keywords):
        backend_counts["custom"] += 1
        return real_custom(*values, **keywords)

    def counted_fallback(*values, **keywords):
        backend_counts["fallback"] += 1
        return real_fallback(*values, **keywords)

    cuda_ops._gemm_4bit_kernel_impl = counted_custom
    cuda_ops._dequant_linear_fallback = counted_fallback
    try:
        result = forward(value)
    finally:
        cuda_ops._gemm_4bit_kernel_impl = real_custom
        cuda_ops._dequant_linear_fallback = real_fallback
    if fused and (backend_counts["custom"] > 0) != native_fused:
        raise RuntimeError("actual backend kernel differs from recorded dispatch")
    value.grad = None
    result.float().square().mean().backward()
    tensors = {"output": result.detach(), "grad_x": value.grad.detach()}
    witness = tensor_bundle(args.output / "projection.pt", tensors)
    import bitsandbytes as bnb

    reference_x = value.detach().clone().requires_grad_(True)
    reference = bnb.matmul_4bit(reference_x, packed.t(), quant_state=state)
    reference.float().square().mean().backward()
    reference_tensors = {"output": reference.detach(), "grad_x": reference_x.grad.detach()}
    native_comparison = compare_tensors(tensors, reference_tensors)
    reference_witness = tensor_bundle(args.output / "native-reference.pt", reference_tensors)
    if args.route != "dense" and not native_comparison["passed"]:
        raise RuntimeError("candidate differs from unpatched native reference; reject timing")
    dense = functional.dequantize_4bit(packed, state).to(value.dtype)
    with torch.no_grad():
        for _index in range(5):
            torch.nn.functional.linear(value, dense)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _index in range(20):
            torch.nn.functional.linear(value, dense)
        end.record()
        end.synchronize()
    ceiling_ms = start.elapsed_time(end) / 20
    return {
        **source,
        **measured,
        "status": "MEASURED",
        "path": ("function:fused" if native_fused else "function:dense_fallback")
        if fused
        else "dense",
        "native_fused": native_fused,
        "route": args.route,
        "counters": counts,
        "backend_counters_untimed": backend_counts,
        "native_comparison": native_comparison,
        "native_reference_tensors": reference_witness,
        "tensors": witness,
        "shape": [args.m, args.n, args.k],
        "dense_gemm_ms": ceiling_ms,
        "dense_gemm_tflops": 2 * args.m * args.n * args.k / (ceiling_ms * 1e9),
        "settings_sha256": settings_digest(
            {
                key: getattr(args, key)
                for key in ("dtype", "m", "n", "k", "no_nested", "steps", "warmup")
            }
        ),
        "timing_scope": "checkpointed frozen projection forward/backward; no optimizer",
    }


def train(args) -> dict:
    """Full SFT updates and held-out greedy QA on the fixed real-data pilot."""
    source = provenance(True)
    torch, _precision = configure("cuda", args.dtype)
    from types import SimpleNamespace

    from d1_qa import load_split, prompt, score, tokenize
    from stream_probe import build
    from transformers import AutoTokenizer

    from soup_cli.utils import layer_stream_runtime as runtime_module

    dataset = json.loads((args.data / "dataset.json").read_text(encoding="utf-8"))
    if args.weights.name != dataset["model_revision"]:
        raise ValueError("weights must be the local Hub snapshot at the pinned model revision")
    for name, expected in dataset["tokenizer_sha256"].items():
        if digest(args.weights / name) != expected:
            raise ValueError(f"tokenizer SHA-256 mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.weights), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    training = [tokenize(tokenizer, row, args.seq) for row in load_split(args.data, "train")]
    held_out = load_split(args.data, "test")
    # Hash all frozen model files outside the timed region; each arm reports the same bytes.
    model_hashes = {path.name: digest(path) for path in sorted(args.weights.glob("*.safetensors"))}
    if not model_hashes:
        raise ValueError("no local safetensors model files")
    build_args = SimpleNamespace(
        weights=str(args.weights),
        shards=str(args.output / "shards"),
        quant="nf4",
        lazy_shard_handles=False,
        control_sync_source=False,
        lora_targets="q_proj,k_proj,v_proj,o_proj",
        lora_r=16,
        buffers=2,
        read_ahead=2,
        no_pin=False,
        seed=SEED,
        tier="ram",
        stripe_dir=[],
    )
    model, runtime, _config, _index, *_rest = build(build_args, "cuda", args.dtype)
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("expected frozen base with trainable LoRA only")
    initial = tensor_bundle(args.output / "initial-adapters.pt", trainable)
    optimizer = torch.optim.AdamW(
        trainable.values(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False
    )
    losses, token_counts, supervised_counts = [], [], []
    decisions = []
    real_gate = runtime_module._can_use_checkpoint_visible_nf4_gemm

    def step():
        row = training[len(losses) % len(training)]
        batch = {name: torch.tensor([value], device="cuda") for name, value in row.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch, use_cache=False)
        output.loss.backward()
        optimizer.step()
        losses.append(output.loss.detach())
        token_counts.append(sum(row["attention_mask"]))
        supervised_counts.append(sum(value != -100 for value in row["labels"]))

    predictions = []
    try:
        with route(args.route):
            active_gate = runtime_module._can_use_checkpoint_visible_nf4_gemm

            def record_active_gate(*values):
                decision = active_gate(*values)
                decisions.append(bool(decision))
                return decision

            runtime_module._can_use_checkpoint_visible_nf4_gemm = record_active_gate
            model.train()
            measured = timed_steps(step, args.steps, args.warmup)
            training_decisions = decisions.copy()
            if not bool(torch.isfinite(torch.stack(losses)).all()) or any(
                value.grad is None or not bool(torch.isfinite(value.grad).all())
                for value in trainable.values()
            ):
                raise RuntimeError("nonfinite loss or missing/nonfinite final LoRA gradient")
            losses = [float(value) for value in losses]
            final = tensor_bundle(args.output / "final-adapters.pt", trainable)
            model.eval()
            with torch.no_grad():
                for row in held_out:
                    ids = prompt(tokenizer, row)
                    if len(ids) + 64 > args.seq:
                        raise ValueError("held-out prompt exceeds predeclared sequence budget")
                    inputs = torch.tensor([ids], device="cuda")
                    generated = model.generate(
                        input_ids=inputs,
                        attention_mask=torch.ones_like(inputs),
                        max_new_tokens=64,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    predictions.append(
                        {
                            "id": row["id"],
                            "unanswerable": row["unanswerable"],
                            "answers": [value["text"] for value in row["answers"]],
                            "prediction": tokenizer.decode(
                                generated[0, len(ids) :], skip_special_tokens=True
                            ).strip(),
                        }
                    )
            runtime_module._can_use_checkpoint_visible_nf4_gemm = active_gate
        total_seconds = sum(measured["step_wall_ms"]) / 1000
        settings = {
            "dtype": args.dtype,
            "steps": args.steps,
            "warmup": args.warmup,
            "seq": args.seq,
            "seed": SEED,
            "lr": 2e-4,
            "rank": 16,
            "dataset_sha256": digest(args.data / "dataset.json"),
            "model_hashes": model_hashes,
        }
        return {
            **source,
            **measured,
            "status": "MEASURED",
            "d1_complete": False,
            "settings_sha256": settings_digest(settings),
            "settings": settings,
            "path": "mixed"
            if any(training_decisions) and not all(training_decisions)
            else "function"
            if any(training_decisions)
            else "dense",
            "gate_decisions": training_decisions,
            "route": args.route,
            "updates": len(losses),
            "warmup_updates": args.warmup,
            "losses": losses,
            "nonpadding_tokens_s": sum(token_counts[args.warmup :]) / total_seconds,
            "supervised_tokens_s": sum(supervised_counts[args.warmup :]) / total_seconds,
            "predictions": predictions,
            "quality": score(predictions),
            "initial_adapters": initial,
            "final_adapters": final,
            "timing_scope": "full forward/loss/backward/AdamW step, fixed real QA pilot",
        }
    finally:
        runtime_module._can_use_checkpoint_visible_nf4_gemm = real_gate
        runtime.close()


def ablation(args) -> dict:
    """A/C mechanism ablation; cached-zero C is explicitly garbage mathematics."""
    source = provenance(True)
    torch, _precision = configure("cuda", args.dtype)
    from types import SimpleNamespace

    from stream_probe import Instruments, build, measure_ceiling, model_shapes

    build_args = SimpleNamespace(
        weights=str(args.weights),
        shards=str(args.output / "shards"),
        quant="nf4",
        lazy_shard_handles=False,
        control_sync_source=False,
        lora_targets="q_proj,k_proj,v_proj,o_proj",
        lora_r=16,
        buffers=2,
        read_ahead=2,
        no_pin=False,
        seed=SEED,
        tier="ram",
        stripe_dir=[],
    )
    model, runtime, config, _index, *_rest = build(build_args, "cuda", args.dtype)
    instruments = Instruments(runtime, model, "nf4")
    instruments.events_on = False
    instruments.nocopy = False
    instruments.nodequant = args.route == "nodequant"
    inputs = torch.randint(0, config.vocab_size, (1, args.seq), device="cuda")
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=2e-4, foreach=False
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
        loss.backward()
        optimizer.step()

    try:
        model.train()
        measured = timed_steps(step, args.steps, args.warmup)
        ceiling = measure_ceiling("cuda", args.dtype, model_shapes(config), args.seq)
        return {
            **source,
            **measured,
            "status": "MEASURED",
            "d1_complete": False,
            "path": args.route,
            "timing_only_garbage": instruments.nodequant,
            "patched_linears": instruments.patched_linears,
            "ceiling": ceiling,
            "quality_claim": False,
            "settings_sha256": settings_digest(
                {
                    "dtype": args.dtype,
                    "seq": args.seq,
                    "steps": args.steps,
                    "warmup": args.warmup,
                    "seed": SEED,
                    "model": str(args.weights),
                    "config_sha256": digest(args.weights / "config.json"),
                }
            ),
            "timing_scope": "RAM-tier synthetic-token A/C ablation; nodequant uses zero "
            "weights, so its ratio is an upper-bound probe, not real kernel speedup",
        }
    finally:
        runtime.close()


def run_blocks(args) -> dict:
    """One new child/CUDA context per arm; F1 watches every child, including failures."""
    provenance(True)
    from arm_validity import ArmWatch

    routes = {"A": args.pair.split("-")[0], "B": args.pair.split("-")[1]}
    records = []
    monitor = GPUMonitor()
    for block in range(args.blocks):
        for run, arm in enumerate("ABBA"):
            output = args.output / f"block-{block:03d}-run-{run}-{arm}"
            output.mkdir(parents=True, exist_ok=False)
            command = [
                sys.executable,
                str(Path(__file__)),
                args.kind,
                "--output",
                str(output),
                "--route",
                routes[arm],
                "--dtype",
                args.dtype,
            ]
            keys = (
                ("m", "n", "k", "steps", "warmup")
                if args.kind == "projection"
                else ("steps", "warmup", "seq", "weights", "data")
                if args.kind == "train"
                else ("steps", "warmup", "seq", "weights")
            )
            for key in keys:
                command.extend(["--" + key, str(getattr(args, key))])
            if args.kind == "projection" and args.no_nested:
                command.append("--no-nested")
            watch = ArmWatch(
                arm=arm,
                round=block * 2 + run // 2,
                run=run,
                model_path=str(args.weights) if args.kind != "projection" else str(ROOT),
                required=("power", "suspend", "foreign_reads"),
            )
            before = monitor.sample()
            write_new_json(output / "gpu-before.json", before)
            if not before["available"] or before.get("compute") or before.get("graphics"):
                raise RuntimeError("GPU pre-context is occupied or unknown; no child launched")
            peer_samples, stop = [], threading.Event()

            def sample_peers():
                while not stop.is_set():
                    peer_samples.append(monitor.sample())
                    stop.wait(0.1)

            thread = threading.Thread(target=sample_peers, daemon=True)
            thread.start()
            with watch:
                with (output / "stdout.log").open("x", encoding="utf-8") as log:
                    try:
                        process = subprocess.run(
                            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                        )
                    finally:
                        stop.set()
                        thread.join(timeout=10)
            after = monitor.sample()
            record_path = output / "result.json"
            result = json.loads(record_path.read_text()) if record_path.exists() else {}
            result.update(
                arm=arm,
                block=block,
                run=run,
                case=f"{args.kind}-{args.pair}-{args.m}x{args.n}x{args.k}-{args.dtype}"
                if args.kind == "projection"
                else f"{args.kind}-{args.pair}-{args.seq}-{args.dtype}",
                validity=watch.record,
                returncode=process.returncode,
                artifact=str(record_path.relative_to(args.output)),
                peer_lifecycle={"before": before, "during": peer_samples, "after": after},
                peers_clean=peer_check(before, peer_samples, after, result.get("gpu_uuid")),
            )
            if (
                process.returncode
                or result.get("status") != "MEASURED"
                or result.get("telemetry_errors")
                or not result.get("telemetry_gpu_matches_cuda")
            ):
                result["validity"] = {**(watch.record or {}), "outcome": "void"}
            write_new_json(output / "watched.json", result)
            records.append(result)
            CONSOLE.print(f"block {block + 1}/{args.blocks}, arm {arm}: {result.get('status')}")
            if process.returncode:
                return {
                    "status": "NO_VERDICT",
                    "d1_complete": False,
                    "rows": records,
                    "reason": "child failed; every attempt retained; no automatic retry",
                }
    return {**summarize(records), "rows": records}


def replay(args) -> dict:
    import torch

    record = json.loads((args.input / "result.json").read_text())
    comparisons = []

    def load(bundle):
        path = evidence_path(args.input, bundle["file"])
        if digest(path) != bundle["sha256"]:
            raise ValueError("tensor SHA-256 mismatch")
        return torch.load(path, map_location="cpu", weights_only=True)

    alias = record["alias"]
    tensors = {name: load(bundle) for name, bundle in alias["tensors"].items()}
    comparisons.append(compare_tensors(tensors["safe"], tensors["private"]))
    detected = negative_control(
        {"grad": tensors["broken"]["grad"]}, {"grad": tensors["private"]["grad"]}
    )
    for row in record["streaming"]:
        comparisons.append(
            compare_tensors(load(row["tensors"]["streamed"]), load(row["tensors"]["resident"]))
        )
    passed = all(row["passed"] for row in comparisons) and detected
    return {
        "status": "PASS" if passed else "FAIL",
        "comparisons": comparisons,
        "negative_control_detected": detected,
        "original_status": record["status"],
        "scope": "stored finite tensors only; CPU replay cannot establish CUDA timing",
    }


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    commands = argument_parser.add_subparsers(dest="command", required=True)
    for name in (
        "doctor",
        "correctness",
        "projection",
        "train",
        "ablation",
        "run",
        "replay",
        "report",
    ):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name in ("correctness", "doctor"):
            command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if name in ("correctness", "projection", "train", "ablation", "run"):
            command.add_argument(
                "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
            )
        if name in ("projection", "train", "ablation", "run"):
            command.add_argument("--m", type=int, default=8)
            command.add_argument("--n", type=int, default=4096)
            command.add_argument("--k", type=int, default=4096)
            command.add_argument("--steps", type=int, default=100)
            command.add_argument("--warmup", type=int, default=20)
            command.add_argument("--no-nested", action="store_true")
        if name in ("projection", "train"):
            command.add_argument("--route", choices=("dense", "native", "forced"), required=True)
        if name == "ablation":
            command.add_argument("--route", choices=("baseline", "nodequant"), required=True)
        if name == "run":
            command.add_argument(
                "--kind", choices=("projection", "train", "ablation"), default="projection"
            )
            command.add_argument(
                "--pair",
                choices=("dense-native", "native-forced", "baseline-nodequant"),
                default="dense-native",
            )
            command.add_argument("--blocks", type=int, default=5)
        if name in ("train", "ablation", "run"):
            command.add_argument("--weights", type=Path, required=name != "run")
            command.add_argument("--data", type=Path)
            command.add_argument("--seq", type=int, default=1024)
        if name in ("replay", "report"):
            command.add_argument("--input", type=Path, required=True)
    return argument_parser


def validate_args(args) -> None:
    kind = args.kind if args.command == "run" else args.command
    if kind == "train" and (args.weights is None or args.data is None):
        raise ValueError("train requires local --weights and --data")
    if args.command == "run" and kind == "ablation" and args.pair != "baseline-nodequant":
        raise ValueError("ablation requires --pair baseline-nodequant")
    if args.command == "run" and kind != "ablation" and args.pair == "baseline-nodequant":
        raise ValueError("baseline-nodequant is only an ablation pair")
    if kind == "ablation" and args.weights is None:
        raise ValueError("ablation requires local --weights")
    if kind in ("train", "ablation") and not args.weights.is_dir():
        raise ValueError("--weights must name an existing local model snapshot")
    if kind == "train" and not (args.data / "dataset.json").is_file():
        raise ValueError("--data must name a prepared dataset directory")
    for name in ("m", "n", "k", "steps", "blocks", "seq"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if hasattr(args, "warmup") and args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "result.json"
    if result_path.exists():
        raise FileExistsError("choose a new output directory; previous evidence is immutable")
    try:
        validate_args(args)
        if args.command == "doctor":
            torch, _precision = configure(args.device, "float32")
            result = {
                **provenance(False),
                "status": "READY" if torch.cuda.is_available() else "CPU_ONLY",
                "cuda_available": torch.cuda.is_available(),
                "imported_runtime": str(
                    __import__(
                        "soup_cli.utils.layer_stream_runtime", fromlist=["__file__"]
                    ).__file__
                ),
            }
        elif args.command == "correctness":
            result = correctness(args)
        elif args.command == "projection":
            result = projection(args)
        elif args.command == "train":
            result = train(args)
        elif args.command == "ablation":
            result = ablation(args)
        elif args.command == "run":
            result = run_blocks(args)
        elif args.command == "replay":
            result = replay(args)
        else:
            rows = [
                json.loads(path.read_text()) for path in sorted(args.input.glob("*/watched.json"))
            ]
            result = summarize(rows)
    except Exception as exc:
        result = {
            "status": "NOT_RUN" if "CUDA unavailable" in str(exc) else "FAIL",
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "d1_complete": False,
        }
        write_new_json(result_path, result)
        CONSOLE.print(result)
        return 2
    write_new_json(result_path, result)
    CONSOLE.print({"status": result["status"], "result": str(result_path)})
    return (
        0
        if result["status"]
        in ("PASS", "CPU_PREPARATION_PASS", "CPU_ONLY", "READY", "MEASURED", "MEASURED_SUBSET")
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
