"""B1.4 diagnostic of grouped expert compute against sequential, by routed-row bucket.

Answers the D3 owner's question on PR #28: how many CUDA kernels does grouping remove, and
what does it cost for small row counts. A bucket is the number of routed rows each visited
expert receives (1, 2, 4 ... 128). Capacity and kernel-count diagnostic, no speed claim.
"""

from __future__ import annotations

import contextlib
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Routed rows per visited expert, as agreed with the D3 owner in the #28 thread.
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


def route_for_bucket(rows: int, *, experts: int) -> Tuple[Any, Any]:
    """Routing that sends exactly ``rows`` tokens to each of ``experts`` experts (top-k = 1)."""
    import torch

    if rows < 1 or experts < 1:
        raise ValueError("rows and experts must be positive")
    tokens = rows * experts
    index = (torch.arange(tokens, dtype=torch.int64) % experts).reshape(tokens, 1)
    weights = torch.full((tokens, 1), 0.5, dtype=torch.float32)
    return index, weights


PHASE_PREFIX = "phase:"
_LAUNCH_CATEGORIES = ("cuda_runtime", "cuda_driver")
UNATTRIBUTED = "unattributed"


def _attribute(events: List[Dict[str, Any]], category: str,
               weight: Callable[[Dict[str, Any]], int]) -> Dict[str, int]:
    phases = [
        event for event in events
        if event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith(PHASE_PREFIX)
    ]
    launches: Dict[Any, Dict[str, Any]] = {}
    for event in events:
        if event.get("cat") in _LAUNCH_CATEGORIES:
            correlation = (event.get("args") or {}).get("correlation")
            if correlation is not None:
                launches.setdefault(correlation, event)
    totals: Dict[str, int] = {}
    for event in events:
        if event.get("cat") != category:
            continue
        launch = launches.get((event.get("args") or {}).get("correlation"))
        name = UNATTRIBUTED
        if launch is not None:
            when = launch["ts"]
            covering = [
                phase for phase in phases
                if phase.get("pid") == launch.get("pid")
                and phase["ts"] <= when <= phase["ts"] + phase.get("dur", 0)
            ]
            same_thread = [phase for phase in covering if phase.get("tid") == launch.get("tid")]
            chosen = same_thread or covering
            if chosen:
                innermost = min(chosen, key=lambda phase: (phase.get("dur", 0), -phase["ts"]))
                name = innermost["name"][len(PHASE_PREFIX):]
        totals[name] = totals.get(name, 0) + weight(event)
    return totals


def attribute_kernels(events: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count CUDA kernels per phase from the events of an exported ``torch.profiler`` trace.

    A phase is a ``record_function("phase:<name>")`` range. A kernel belongs to the innermost
    phase that was open when the kernel was launched; the launch is the runtime or driver event
    with the kernel's correlation id. Phases of the launching thread come first. Autograd runs
    the backward on its own thread inside the main thread's range, so when no phase of that
    thread covers the launch, a phase of another thread of the process may claim it. This is
    valid only while the harness runs one stream of work at a time. A kernel with no such phase
    or no launch event counts as ``unattributed``, so the counts always add up to the number of
    kernel events.
    """
    return _attribute(events, "kernel", lambda event: 1)


def attribute_memcpy(events: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count device copies per phase, by the same rule as ``attribute_kernels``."""
    return _attribute(events, "gpu_memcpy", lambda event: 1)


def attribute_kernel_time_us(events: List[Dict[str, Any]]) -> Dict[str, int]:
    """Sum the device time of the kernels per phase, in microseconds, by the same rule."""
    return _attribute(events, "kernel", lambda event: int(event.get("dur", 0)))


ARMS = ("sequential", "grouped_restore", "grouped_gemm")
GROUP_WIDTH = 4
#: D3 restricts grouped GEMM to the OLMoE shapes it validated; the diagnostic adds the
#: geometry of the layer under test to that set and checks the result against a resident layer.
_PASS = {"name": "fwd"}


class ArmRefusedError(Exception):
    """The arm cannot run here; the reason goes into the record."""


class Layer:
    """One MoE expert layer: a resident reference, its shard file and the arms built on it."""

    def __init__(self, *, module: Any, reader: Any, codes: Dict[str, Any], hidden: int,
                 intermediate: int, experts: int, device: str, dtype: str) -> None:
        self.module, self.reader, self.codes = module, reader, codes
        self.hidden, self.intermediate, self.experts = hidden, intermediate, experts
        self.device, self.dtype = device, dtype
        self._arms: Dict[str, Any] = {}
        self._closers: List[Callable[[], None]] = []

    def close(self) -> None:
        for closer in reversed(self._closers):
            closer()
        self._closers = []
        self.reader.close()

    def _activation_budget(self) -> int:
        from soup_cli.utils.expert_group_compute import plan_grouped_forward

        plan = plan_grouped_forward(
            tokens=max(BUCKETS) * self.experts, top_k=1, experts=self.experts,
            hidden=self.hidden, intermediate=self.intermediate, width=GROUP_WIDTH,
        )
        return max(2**24, int(plan["activation_bytes"] * 1.1))

    def arm(self, name: str) -> Any:
        if name not in self._arms:
            self._arms[name] = self._build_arm(name)
        return self._arms[name]

    def _build_arm(self, name: str) -> Any:
        act_fn = self.module.act_fn
        if name == "sequential":
            from soup_cli.utils.expert_stream import (
                ExpertBufferPool,
                plan_expert_buffer,
                streamed_experts_class,
            )

            plan = plan_expert_buffer(self.reader, dtype=self.dtype)
            pool = ExpertBufferPool(
                self.reader, dtype=self.dtype, device=self.device,
                weight_budget_bytes=plan.weight_bytes, codes=self.codes,
            )
            self._closers.append(pool.close)
            return streamed_experts_class()(self.reader, pool, act_fn, 0)
        if name not in ("grouped_restore", "grouped_gemm"):
            raise ValueError(f"unknown arm: {name}")
        from soup_cli.utils.expert_group import GroupedExpertPool, plan_grouped_expert_buffer
        from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class

        gemm = None
        if name == "grouped_gemm":
            gemm = self._build_gemm()
        plan = plan_grouped_expert_buffer(self.reader, dtype=self.dtype, group_width=GROUP_WIDTH)
        pool = GroupedExpertPool(
            self.reader, device=self.device, dtype=self.dtype, group_width=GROUP_WIDTH,
            weight_budget_bytes=plan.weight_bytes, codes=self.codes,
        )
        self._closers.append(pool.close)
        return grouped_autograd_experts_class()(
            self.reader, pool, act_fn, activation_budget_bytes=self._activation_budget(),
            gemm=gemm,
        )

    def _build_gemm(self) -> Any:
        if not self.device.startswith("cuda"):
            raise ArmRefusedError(f"refused: grouped GEMM needs CUDA bf16 on sm_80+ (device is "
                             f"{self.device})")
        if self.dtype != "bfloat16":
            raise ArmRefusedError("refused: grouped GEMM needs CUDA bf16, the layer is not bf16")
        library = find_cublas()
        if library is None:
            raise ArmRefusedError("refused: no absolute libcublas.so.12/13 found next to torch")
        from soup_cli.utils import expert_group_gemm

        expert_group_gemm.SUPPORTED_SHAPES.update({
            (self.hidden, 2 * self.intermediate), (self.intermediate, self.hidden),
        })
        rows = max(BUCKETS) * self.experts
        budget = rows * 2 * self.intermediate * 2 * 2 + 2**20
        try:
            gemm = expert_group_gemm.GroupedGemm(
                library, call_budget_bytes=budget, native_olmoe_rows_max=16,
            )
        except (ValueError, RuntimeError, AttributeError) as error:
            raise ArmRefusedError(f"refused: {type(error).__name__}: {error}") from error
        self._closers.append(gemm.close)
        return gemm


def find_cublas() -> Optional[Path]:
    """Absolute path of the cuBLAS 12/13 library that ships with the torch wheel, if any."""
    import glob
    import os

    import torch

    site = os.path.dirname(os.path.dirname(torch.__file__))
    for pattern in ("nvidia/cublas/lib/libcublas.so.13", "nvidia/cublas/lib/libcublas.so.12"):
        for found in glob.glob(os.path.join(site, pattern)):
            return Path(os.path.realpath(found)).parent / os.path.basename(found)
    return None


def set_blas_policy() -> Dict[str, Any]:
    """The D3 reference policy, applied to every arm: cuBLASLt and no reduced-precision bf16."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.preferred_blas_library("cublaslt")
        matmul = torch.backends.cuda.matmul
        try:  # torch 2.11: one (allow, split-K) tuple, as D3's own harness assigns it
            matmul.allow_bf16_reduced_precision_reduction = (False, False)
        except RuntimeError:  # torch 2.7: a bool and a separate split-K flag
            matmul.allow_bf16_reduced_precision_reduction = False
            matmul.allow_bf16_reduced_precision_reduction_split_k = False
    cuda = torch.backends.cuda
    try:
        split_k: Any = bool(cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
    except AttributeError:
        split_k = "attribute missing in this torch"
    return {
        "preferred_blas_library": str(cuda.preferred_blas_library()),
        "allow_bf16_reduced_precision_reduction": bool(
            cuda.matmul.allow_bf16_reduced_precision_reduction),
        "allow_bf16_reduced_precision_reduction_split_k": split_k,
    }


def build_layer(workdir: Any, *, hidden: int, intermediate: int, experts: int, device: str,
                dtype: str, seed: int = 5) -> Layer:
    """A DeepSeek-V3-style expert layer with random weights, resident and as a shard file."""
    import torch
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts

    from soup_cli.utils.expert_shard import ExpertShardReader, write_expert_shard
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    config = DeepseekV3Config(
        hidden_size=hidden, intermediate_size=2 * intermediate, moe_intermediate_size=intermediate,
        n_routed_experts=experts, n_group=1, topk_group=1, num_experts_per_tok=1,
        n_shared_experts=1, norm_topk_prob=True,
    )
    config._experts_implementation = "eager"
    torch.manual_seed(seed)
    module = DeepseekV3Experts(config)
    with torch.no_grad():
        module.gate_up_proj.normal_(0, hidden ** -0.5)
        module.down_proj.normal_(0, intermediate ** -0.5)
    module = module.to(device=device, dtype=getattr(torch, dtype)).requires_grad_(False).eval()
    path = str(Path(workdir) / "experts.safetensors")
    info = write_expert_shard(
        path,
        ((gate.detach(), down.detach()) for gate, down in zip(
            module.gate_up_proj, module.down_proj)),
        quantise=False, double_quant=False, device=device,
    )
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs=info.quant_specs, dtype=dtype,
        direct=False,
    )
    codes = {NF4_CODE_KEY: info.codes[0], NF4_NESTED_CODE_KEY: info.codes[1]}
    return Layer(module=module, reader=reader, codes=codes, hidden=hidden,
                 intermediate=intermediate, experts=experts, device=device, dtype=dtype)


# --------------------------------------------------------------------------
# One bucket: correctness, kernel counts per phase and paired timing, arm by arm.
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _range(name: str, annotate: bool) -> Any:
    if not annotate:
        yield
        return
    from torch.profiler import record_function

    with record_function(f"{PHASE_PREFIX}{name}"):
        yield


def _instrument(module: Any) -> Callable[[], None]:
    """Label restore, GEMM and activation ranges inside an arm; returns the undo function."""
    from torch.profiler import record_function

    pool = module.pool
    original_lease = pool.lease
    undo: List[Callable[[], None]] = []

    @contextlib.contextmanager
    def lease(*args: Any, **kwargs: Any) -> Any:
        manager = original_lease(*args, **kwargs)
        with record_function(f"{PHASE_PREFIX}restore_{_PASS['name']}"):
            weights = manager.__enter__()
        try:
            yield weights
        except BaseException as error:
            if not manager.__exit__(type(error), error, error.__traceback__):
                raise
        else:
            manager.__exit__(None, None, None)

    pool.lease = lease
    undo.append(lambda: pool.__dict__.pop("lease", None))
    for method, label in (("_linear", "gemm"), ("_activate", "act")):
        if hasattr(module, method):
            original = getattr(module, method)

            def wrapper(*args: Any, _original: Any = original, _label: str = label,
                        **kwargs: Any) -> Any:
                with record_function(f"{PHASE_PREFIX}{_label}"):
                    return _original(*args, **kwargs)

            setattr(module, method, wrapper)
            undo.append(lambda name=method: module.__dict__.pop(name, None))

    def restore() -> None:
        for step in undo:
            step()

    return restore


def _inputs(layer: Layer, rows: int) -> Tuple[Any, Any, Any]:
    import torch

    index, weights = route_for_bucket(rows, experts=layer.experts)
    generator = torch.Generator().manual_seed(9 + rows)
    hidden = torch.randn(index.shape[0], layer.hidden, generator=generator)
    dtype = getattr(torch, layer.dtype)
    return hidden.to(layer.device, dtype), index.to(layer.device), weights.to(layer.device, dtype)


def _sync(layer: Layer) -> None:
    if layer.device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _execute(layer: Layer, module: Any, inputs: Tuple[Any, Any, Any], *,
             annotate: bool = False, timed: bool = False) -> Dict[str, Any]:
    """One forward and one backward; returns the tensors and, if ``timed``, the two durations."""
    import torch

    hidden, index, weights = inputs
    x = hidden.clone().requires_grad_(True)
    w = weights.clone().requires_grad_(True)
    _sync(layer)
    started = time.perf_counter()
    _PASS["name"] = "fwd"
    with _range("fwd", annotate):
        out = module(x, index, w)
    _sync(layer)
    middle = time.perf_counter()
    probe = (torch.arange(out.numel(), device=out.device, dtype=torch.float32)
             .reshape(out.shape).remainder(13).div(13).to(out.dtype))
    _PASS["name"] = "bwd"
    with _range("bwd", annotate):
        out.backward(probe)
    _sync(layer)
    ended = time.perf_counter()
    result: Dict[str, Any] = {"output": out.detach(), "hidden_grad": x.grad,
                              "route_weight_grad": w.grad}
    if timed:
        result["forward_ms"] = (middle - started) * 1e3
        result["backward_ms"] = (ended - middle) * 1e3
    return result


def _compare(result: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    names = ("output", "hidden_grad", "route_weight_grad")
    return {
        "bitwise_equal_to_resident": {
            name: bool(torch.equal(result[name], reference[name])) for name in names},
        "max_abs_diff_vs_resident": {
            name: float((result[name].float() - reference[name].float()).abs().max())
            for name in names},
    }


def _profile_kernels(layer: Layer, module: Any, inputs: Tuple[Any, Any, Any]) -> Optional[Dict]:
    if not layer.device.startswith("cuda"):
        return None
    from torch.profiler import ProfilerActivity, profile

    restore = _instrument(module)
    try:
        _execute(layer, module, inputs, annotate=True)  # warm-up outside the profile
        _sync(layer)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            _execute(layer, module, inputs, annotate=True)
            _sync(layer)
        with tempfile.TemporaryDirectory(prefix="b14-group-trace-") as folder:
            path = Path(folder) / "trace.json"
            profiler.export_chrome_trace(str(path))
            events = json.loads(path.read_text()).get("traceEvents", [])
    finally:
        restore()
    by_phase = attribute_kernels(events)
    copies = attribute_memcpy(events)
    return {
        "total": sum(by_phase.values()), "by_phase": dict(sorted(by_phase.items())),
        "memcpy_events": sum(copies.values()), "memcpy_by_phase": dict(sorted(copies.items())),
        "gpu_time_us_by_phase": dict(sorted(attribute_kernel_time_us(events).items())),
    }


def run_bucket(layer: Layer, *, rows: int, arms: Sequence[str] = ARMS,
               repeats: int = 20) -> Dict[str, Any]:
    """Run every arm on the same routed inputs: correctness, kernels per phase, paired timing.

    Timing runs the arms in rotating order, one repeat at a time, after one untimed warm-up,
    with no profiler. It is a diagnostic of this layer on this host, not a speed claim.
    """
    inputs = _inputs(layer, rows)
    reference = _execute(layer, layer.module, inputs)
    record: Dict[str, Any] = {
        "rows": rows, "experts": layer.experts, "tokens": int(inputs[1].shape[0]), "arms": {},
    }
    live: Dict[str, Any] = {}
    for name in arms:
        try:
            module = layer.arm(name)
        except ArmRefusedError as refusal:
            record["arms"][name] = {"status": str(refusal)}
            continue
        before = layer.reader.fetches
        result = _execute(layer, module, inputs)
        entry = {"status": "ok", "expert_fetches": layer.reader.fetches - before}
        entry.update(_compare(result, reference))
        entry["kernels"] = _profile_kernels(layer, module, inputs)
        record["arms"][name] = entry
        live[name] = module
    samples: Dict[str, List[Dict[str, float]]] = {name: [] for name in live}
    names = list(live)
    for repeat in range(repeats):
        turn = repeat % max(len(names), 1)
        for name in names[turn:] + names[:turn]:
            timed = _execute(layer, live[name], inputs, timed=True)
            samples[name].append({"forward": timed["forward_ms"],
                                  "backward": timed["backward_ms"]})
    for name, rows_ms in samples.items():
        totals = [row["forward"] + row["backward"] for row in rows_ms]
        record["arms"][name]["timing_ms"] = {
            "samples": len(rows_ms),
            "forward_median": statistics.median(row["forward"] for row in rows_ms),
            "backward_median": statistics.median(row["backward"] for row in rows_ms),
            "total_median": statistics.median(totals),
            "total_min": min(totals), "total_max": max(totals),
        }
    return record


def _environment(device: str) -> Dict[str, Any]:
    import platform
    import subprocess

    import torch

    env: Dict[str, Any] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device": device,
    }
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        env["transformers"] = None
    if device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        env.update(gpu=properties.name, capability=list(
            torch.cuda.get_device_capability()), vram_bytes=int(properties.total_memory))
    try:
        env["source_sha"] = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["source_sha"] = None
    return env


def main() -> int:
    """Remote entry. Environment: B14G_OUT, B14G_WORK, B14G_DEVICE, B14G_DTYPE, B14G_HIDDEN,
    B14G_INTERMEDIATE, B14G_EXPERTS, B14G_REPEATS, B14G_BUCKETS (comma list), B14G_TAG."""
    import os

    import torch

    cuda = torch.cuda.is_available()
    device = os.environ.get("B14G_DEVICE", "cuda" if cuda else "cpu")
    dtype = os.environ.get("B14G_DTYPE", "bfloat16" if cuda else "float32")
    hidden = int(os.environ.get("B14G_HIDDEN", "2048"))
    intermediate = int(os.environ.get("B14G_INTERMEDIATE", "1408"))
    experts = int(os.environ.get("B14G_EXPERTS", "8"))
    repeats = int(os.environ.get("B14G_REPEATS", "20"))
    buckets = tuple(int(item) for item in os.environ.get(
        "B14G_BUCKETS", ",".join(str(rows) for rows in BUCKETS)).split(","))
    tag = os.environ.get("B14G_TAG", "run")
    out = Path(os.environ.get("B14G_OUT", "out"))
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    policy = set_blas_policy()
    with tempfile.TemporaryDirectory(dir=os.environ.get("B14G_WORK"), prefix="b14g-") as work:
        layer = build_layer(work, hidden=hidden, intermediate=intermediate, experts=experts,
                            device=device, dtype=dtype)
        try:
            records = []
            for rows in buckets:
                records.append(run_bucket(layer, rows=rows, repeats=repeats))
                print(f"[{time.time() - started:7.1f}s] bucket {rows} done", flush=True)
        finally:
            layer.close()
    result = {
        "environment": _environment(device), "blas_policy": policy,
        "geometry": {"hidden": hidden, "moe_intermediate": intermediate, "experts": experts,
                     "dtype": dtype, "top_k": 1, "group_width": GROUP_WIDTH,
                     "grouped_gemm_native_rows_max": 16,
                     "supported_shapes_added": [[hidden, 2 * intermediate],
                                                [intermediate, hidden]]},
        "repeats": repeats, "timing_is_diagnostic_not_a_speed_claim": True,
        "buckets": records,
    }
    (out / f"result-{tag}.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {out / f'result-{tag}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
