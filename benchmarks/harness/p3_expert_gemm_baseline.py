#!/usr/bin/env python3
"""A2 stage 0: the routed-expert GEMMs on cuBLAS and on a fixed-config Triton GEMM.

A2 wants one GEMM per expert over a whole MBR window (``merged``) to give the same bits, row by
row, as one GEMM per micro-batch (``separate``). That holds when a row's output does not depend
on how many rows M share the call. This harness measures, for the four matmuls of a frozen
routed expert (``gate_up`` and ``down`` forward, and their input gradients; no weight gradient):

* **M-invariance.** Against one reference call over more rows than any probe: ``op(X[:m])``
  against its first m rows, the same at a row offset (rows move inside a tile), and separate
  patterns (several calls over consecutive chunks, concatenated: equal chunks, and one ragged
  window whose chunk sizes cross the tile boundaries) against the reference rows. Compared as
  raw bit patterns, so -0.0 and 0.0 or two NaN payloads differ.
* **Validity.** The reference call must be finite and close to a float64 product of the same
  operands (``TOLERANCE``); a kernel that returns the same wrong bits for every M is
  ``valid=False`` and never ``invariant``. For Triton, the number of compiled binaries must not
  grow after the first call: every M runs the same binary.
* **Speed.** An M sweep (median of CUDA-event timings, L2 flushed before each call by
  ``triton.testing.do_bench``), and per pattern the separate calls against the merged call two
  ways: ``eager`` (one L2 flush, then the calls as Python issues them: later separate calls may
  hit the weight in L2, as in the one-lease MBR path, and launch gaps count) and ``graph``
  (``do_bench_cudagraph``: GPU time only, L2 warm). During timing PyTorch's deterministic NaN
  fill of new tensors is off, so it does not cost a kernel per output.

Kernels:

* ``cublas``: ``F.linear`` for the forward and ``torch.mm`` for the input gradient, as autograd
  runs them for 2-D row-major operands, with cuBLAS reduced-precision reductions allowed (the
  PyTorch default). ``cublas-norpr``: the same with them off (fp16/bf16 only). Split-K stays
  allowed in both, as by default. The verdict holds for the ``CUBLAS_WORKSPACE_CONFIG`` recorded
  in ``environment`` (``:4096:8`` unless set): cuBLAS picks kernels by workspace too.
* ``triton:<config>``: ``fixed_gemm`` below. One tile config for every M, no autotune, no split-K,
  one fp32 accumulator per output tile over a sequential K loop (``acc = tl.dot(a, b, acc)``),
  M not specialized (``do_not_specialize``). Each output element is reduced over K in the same
  order whatever M is; masked rows are zeros and never mix with valid rows. The design follows
  Thinking Machines' batch-invariant matmul
  (https://github.com/thinking-machines-lab/batch_invariant_ops, MIT) with small-M tiles.
  ``input_precision="ieee"`` matters for fp32 only: no TF32, so fp32 runs on FMA everywhere.

Bit-identity holds per GPU architecture, Triton version and config, not across them. Triton
lowers ``tl.dot`` below compute capability 8.0 (Tesla T4) to FMA, not tensor cores
(``AccelerateMatmul.cpp``: "falls back to the FMA path"), so a T4 checks the harness and cuBLAS;
the mma path and the speed that matters need an sm_80+ card (RTX 30xx/40xx/50xx).

``--device cpu`` runs the invariance and validity checks only (cuBLAS becomes the CPU BLAS; the
Triton kernel needs ``TRITON_INTERPRET=1``, which has no bfloat16). It exists for smoke tests.
The exit status is 1 when an entry failed for a reason other than an unsupported combination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.markup import escape
from rich.table import Table

CONSOLE = Console()


@dataclass(frozen=True)
class Model:
    name: str
    hidden: int
    intermediate: int  # moe_intermediate_size of one routed expert


MODELS = {
    model.name: model
    for model in (
        Model("deepseek-v3", hidden=7168, intermediate=2048),
        Model("kanana-30b-a3b", hidden=2048, intermediate=768),
        Model("tiny", hidden=64, intermediate=32),  # smoke shape only
    )
}


@dataclass(frozen=True)
class Op:
    """``C[M, n] = A[M, k] @ B[k, n]`` with ``B`` a view of the stored expert weight.

    ``transposed``: the weight is stored ``[n, k]`` and ``B`` is its transpose (the forward,
    ``F.linear``); otherwise the weight is stored ``[k, n]`` and ``B`` is the weight itself (the
    input gradient of that linear, ``grad @ W``).
    """

    name: str
    k: int
    n: int
    transposed: bool


def expert_ops(model: Model) -> List[Op]:
    hidden, inter = model.hidden, model.intermediate
    return [
        Op("gate_up.fwd", k=hidden, n=2 * inter, transposed=True),  # x @ W_gu.T, W_gu [2I, H]
        Op("down.fwd", k=inter, n=hidden, transposed=True),  # a @ W_d.T, W_d [H, I]
        Op("down.dx", k=hidden, n=inter, transposed=False),  # g @ W_d
        Op("gate_up.dx", k=2 * inter, n=hidden, transposed=False),  # g @ W_gu
    ]


@dataclass(frozen=True)
class TileConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int

    @property
    def name(self) -> str:
        return (
            f"m{self.block_m}n{self.block_n}k{self.block_k}"
            f"w{self.num_warps}s{self.num_stages}"
        )


CONFIGS = {
    config.name: config
    for config in (
        TileConfig(16, 32, 32, 4, 2),
        TileConfig(16, 64, 64, 4, 3),
        TileConfig(16, 128, 64, 4, 3),
        TileConfig(32, 64, 64, 4, 3),
        TileConfig(64, 64, 32, 4, 3),
    )
}
DEFAULT_CONFIG = "m16n32k32w4s2"
DTYPES = ("float16", "bfloat16", "float32")
#: Relative L2 error against float64 above which a kernel's output is not a valid product. Far
#: above correct rounding (T4: fp16 <= 3.3e-4, bf16 1.7e-3, fp32 <= 1.5e-6), far below a wrong
#: tile; fp32 catches an accidental TF32 (~1e-3).
TOLERANCE = {"float16": 2e-3, "bfloat16": 1e-2, "float32": 1e-4}
DEFAULT_MS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
#: (micro-batches k, rows per micro-batch r): DeepSeek-V3 top-8 of 256 is ~2 rows per expert per
#: 64-token micro-batch, Kanana top-6 of 128 is ~3; the last two are denser windows.
DEFAULT_PATTERNS = ((64, 2), (64, 3), (16, 8), (4, 32))
#: One expert's rows over a window whose micro-batches routed it unevenly, crossing 16/32/64.
RAGGED = (1, 2, 3, 5, 8, 13, 17, 31, 33, 16)
ROW_OFFSET = 3
#: Rows of the reference call beyond the largest probe, so no probe compares a call with itself.
REFERENCE_EXTRA = 29


@lru_cache(maxsize=None)
def _fixed_gemm_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["m"])
    def fixed_gemm_kernel(
        a_ptr, b_ptr, c_ptr, m, n, k,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    ):
        # Offsets in int64: row * stride overflows int32 well before the tensors do.
        rows = (tl.program_id(0) * block_m + tl.arange(0, block_m)).to(tl.int64)
        cols = (tl.program_id(1) * block_n + tl.arange(0, block_n)).to(tl.int64)
        depth = tl.arange(0, block_k).to(tl.int64)
        a_ptrs = a_ptr + rows[:, None] * stride_am + depth[None, :] * stride_ak
        b_ptrs = b_ptr + depth[:, None] * stride_bk + cols[None, :] * stride_bn
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        for start in range(0, k, block_k):
            left = k - start
            a = tl.load(a_ptrs, mask=(rows[:, None] < m) & (depth[None, :] < left), other=0.0)
            b = tl.load(b_ptrs, mask=(depth[:, None] < left) & (cols[None, :] < n), other=0.0)
            acc = tl.dot(a, b, acc, input_precision="ieee")
            a_ptrs += block_k * stride_ak
            b_ptrs += block_k * stride_bk
        c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
        mask = (rows[:, None] < m) & (cols[None, :] < n)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)

    return fixed_gemm_kernel


def triton_binaries() -> Optional[int]:
    """Compiled binaries of the fixed GEMM so far (None under the interpreter)."""
    caches = getattr(_fixed_gemm_kernel(), "device_caches", None)
    if caches is None:
        return None
    return sum(len(cache[0]) for cache in caches.values())


def fixed_gemm(a: Any, b: Any, config: TileConfig) -> Any:
    """``a @ b`` with one tile config for every M; ``b`` may be any strided view."""
    import torch

    m, k = a.shape
    k_b, n = b.shape
    if k != k_b:
        raise ValueError(f"inner dimensions differ: {k} and {k_b}")
    if a.dtype != b.dtype:
        raise ValueError(f"operand dtypes differ: {a.dtype} and {b.dtype}")
    if a.dtype == torch.bfloat16 and os.environ.get("TRITON_INTERPRET") == "1":
        # The interpreter runs on numpy, which has no bfloat16: its products come out garbage.
        raise NotImplementedError("bfloat16 under TRITON_INTERPRET gives wrong products")
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    if m == 0:
        return out
    grid = (-(-m // config.block_m), -(-n // config.block_n))
    _fixed_gemm_kernel()[grid](
        a, b, out, m, n, k, *a.stride(), *b.stride(), *out.stride(),
        block_m=config.block_m, block_n=config.block_n, block_k=config.block_k,
        num_warps=config.num_warps, num_stages=config.num_stages,
    )
    return out


@contextmanager
def cublas_reductions(reduced_precision: bool) -> Iterator[None]:
    """fp16/bf16 cuBLAS reduced-precision reductions on or off, split-K allowed; restored after."""
    import torch

    getters = (
        torch._C._get_cublas_allow_fp16_reduced_precision_reduction,
        torch._C._get_cublas_allow_bf16_reduced_precision_reduction,
    )
    setters = (
        torch._C._set_cublas_allow_fp16_reduced_precision_reduction,
        torch._C._set_cublas_allow_bf16_reduced_precision_reduction,
    )
    saved = [tuple(get()) for get in getters]
    for setter in setters:
        setter(reduced_precision, True)
    try:
        yield
    finally:
        for setter, (allow, split_k) in zip(setters, saved):
            setter(allow, split_k)


@contextmanager
def no_nan_fill() -> Iterator[None]:
    """Deterministic mode fills new tensors with NaN (a kernel per output); off while timing."""
    import torch

    saved = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = saved


def kernel_fn(kernel: str, op: Op, weight: Any) -> Callable[[Any], Any]:
    """The callable ``a -> op(a)`` for one kernel; the weight is bound in its stored layout.

    cuBLAS flags are not set here (that would cost CPU time inside every timed call): run the
    callable under ``kernel_context(kernel)``.
    """
    import torch
    import torch.nn.functional as functional

    if kernel.startswith("triton:"):
        config = CONFIGS[kernel.split(":", 1)[1]]
        b = weight.t() if op.transposed else weight
        return lambda a: fixed_gemm(a, b, config)
    if kernel not in ("cublas", "cublas-norpr"):
        raise ValueError(f"unknown kernel {kernel!r}")
    if op.transposed:
        return lambda a: functional.linear(a, weight)
    return lambda a: torch.mm(a, weight)


@contextmanager
def kernel_context(kernel: str) -> Iterator[None]:
    if kernel == "cublas-norpr":
        with cublas_reductions(False):
            yield
    elif kernel == "cublas":
        with cublas_reductions(True):
            yield
    else:
        yield


def make_operands(op: Op, rows: int, dtype: str, device: str, seed: int) -> Tuple[Any, Any]:
    """Activations ``[rows, k]`` and the weight in its stored layout, unit-scale outputs."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (op.n, op.k) if op.transposed else (op.k, op.n)
    weight = torch.randn(shape, generator=generator) / op.k**0.5
    a = torch.randn((rows, op.k), generator=generator)
    return a.to(device=device, dtype=getattr(torch, dtype)), weight.to(
        device=device, dtype=getattr(torch, dtype)
    )


def _bits(tensor: Any) -> Any:
    import torch

    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()]
    return tensor.contiguous().view(view)


def bits_sha256(tensor: Any) -> str:
    return hashlib.sha256(_bits(tensor).cpu().numpy().tobytes()).hexdigest()


def bit_diff(got: Any, want: Any) -> Dict[str, Any]:
    """Raw bit-pattern comparison: equal bits, differing elements and rows, max |difference|."""
    if got.shape != want.shape or got.dtype != want.dtype:
        raise ValueError(f"cannot compare {got.shape}/{got.dtype} with {want.shape}/{want.dtype}")
    differ = _bits(got) != _bits(want)
    count = int(differ.sum())
    return {
        "bit_equal": count == 0,
        "diff_elements": count,
        "diff_rows": int(differ.any(dim=1).sum()),
        "max_abs": _finite_or_none((got.double() - want.double()).abs().max()) if count else 0.0,
    }


def _finite_or_none(value: Any) -> Optional[float]:
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class Pattern:
    """One expert's rows over a window: consecutive chunks, one call per micro-batch."""

    name: str
    chunks: Tuple[int, ...]

    @property
    def rows(self) -> int:
        return sum(self.chunks)


def equal_pattern(micro: int, per: int) -> Pattern:
    return Pattern(f"{micro}x{per}", (per,) * micro)


RAGGED_PATTERN = Pattern("ragged", RAGGED)


def reference_rows(ms: Sequence[int], patterns: Sequence[Pattern]) -> int:
    return max([*ms, *(pattern.rows for pattern in patterns)]) + ROW_OFFSET + REFERENCE_EXTRA


def _chunks(a: Any, pattern: Pattern) -> List[Any]:
    out, start = [], 0
    for size in pattern.chunks:
        out.append(a[start:start + size])
        start += size
    return out


def invariance(
    fn: Callable[[Any], Any],
    a: Any,
    b_reference: Any,
    *,
    ms: Sequence[int],
    patterns: Sequence[Pattern],
    tolerance: float,
    binaries: Callable[[], Optional[int]] = lambda: None,
) -> Dict[str, Any]:
    """M-invariance and validity of ``fn`` against one reference call over all of ``a``."""
    import torch

    rows = a.shape[0]
    needed = reference_rows(ms, patterns) - REFERENCE_EXTRA
    if needed > rows:
        raise ValueError(f"{rows} reference rows cannot hold probes needing {needed}")
    full = fn(a)
    compiled_after_first = binaries()
    reference64 = a.double() @ b_reference.double()
    error = (full.double() - reference64).norm() / reference64.norm()
    relative = _finite_or_none(error)
    finite = bool(torch.isfinite(full).all())
    result: Dict[str, Any] = {
        "rows": rows,
        "reference_sha256": bits_sha256(full),
        "repeat": bit_diff(fn(a), full),
        "finite": finite,
        "fp64_relative_l2": relative,
        "fp64_max_abs": _finite_or_none((full.double() - reference64).abs().max()),
        "tolerance": tolerance,
        "prefix": [{"m": m, **bit_diff(fn(a[:m]), full[:m])} for m in ms],
        "shifted": [],
        "patterns": [],
    }
    for m in ms:
        window = slice(ROW_OFFSET, ROW_OFFSET + m)
        result["shifted"].append({"m": m, **bit_diff(fn(a[window]), full[window])})
    for pattern in patterns:
        separate = torch.cat([fn(chunk) for chunk in _chunks(a, pattern)])
        result["patterns"].append({
            "name": pattern.name, "rows": pattern.rows,
            **bit_diff(separate, full[:pattern.rows]),
        })
    new_binaries = None
    if compiled_after_first is not None:
        new_binaries = binaries() - compiled_after_first
    result["new_binaries_after_first_call"] = new_binaries
    result["valid"] = finite and relative is not None and relative < tolerance
    result["bit_equal_all"] = result["repeat"]["bit_equal"] and all(
        item["bit_equal"] for key in ("prefix", "shifted", "patterns") for item in result[key]
    )
    result["invariant"] = (
        result["valid"] and result["bit_equal_all"] and new_binaries in (None, 0)
    )
    return result


def time_us(fn: Callable[[], Any], *, graph: bool = False) -> Tuple[float, float, float]:
    """Median, p20, p80 in microseconds: eager with L2 flushed per call, or CUDA-graph replay."""
    from triton.testing import do_bench, do_bench_cudagraph

    quantiles = [0.5, 0.2, 0.8]
    if graph:
        median, low, high = do_bench_cudagraph(fn, rep=100, quantiles=quantiles)
    else:
        median, low, high = do_bench(fn, warmup=25, rep=100, quantiles=quantiles)
    return median * 1e3, low * 1e3, high * 1e3


def speed(
    fn: Callable[[Any], Any],
    op: Op,
    a: Any,
    *,
    ms: Sequence[int],
    patterns: Sequence[Pattern],
    timer: Callable[..., Tuple[float, float, float]] = time_us,
) -> Dict[str, Any]:
    weight_bytes = op.k * op.n * a.element_size()
    sweep = []
    for m in ms:
        rows = a[:m]
        median, low, high = timer(lambda rows=rows: fn(rows))
        sweep.append({
            "m": m, "median_us": median, "p20_us": low, "p80_us": high,
            "tflops": 2 * m * op.k * op.n / median / 1e6,
            "weight_gbps": weight_bytes / median / 1e3,
        })
    pattern_rows = []
    for pattern in patterns:
        chunks = _chunks(a, pattern)
        merged_rows = a[:pattern.rows]
        row: Dict[str, Any] = {"name": pattern.name, "rows": pattern.rows}
        for mode, graph in (("eager", False), ("graph", True)):
            separate = timer(lambda chunks=chunks: [fn(chunk) for chunk in chunks], graph=graph)
            merged = timer(lambda rows=merged_rows: fn(rows), graph=graph)
            row[mode] = {
                "separate_us": separate[0], "separate_p20_us": separate[1],
                "separate_p80_us": separate[2],
                "merged_us": merged[0], "merged_p20_us": merged[1], "merged_p80_us": merged[2],
                "separate_over_merged": separate[0] / merged[0],
            }
        pattern_rows.append(row)
    return {"sweep": sweep, "patterns": pattern_rows}


def environment(device: str) -> Dict[str, Any]:
    import torch

    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton_interpret": os.environ.get("TRITON_INTERPRET"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sms": props.multi_processor_count,
            "memory_gib": round(props.total_memory / 2**30, 2),
            "l2_mib": round(getattr(props, "L2_cache_size", 0) / 2**20, 2),
        })
    return info


def configure(device: str) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda needs a CUDA device")


def kernels_for(dtype: str, configs: Sequence[str]) -> List[str]:
    names = ["cublas"]
    if dtype != "float32":
        names.append("cublas-norpr")
    return names + [f"triton:{name}" for name in configs]


def expected_entries(
    models: Sequence[str], dtypes: Sequence[str], configs: Sequence[str]
) -> int:
    return len(models) * 4 * sum(len(kernels_for(dtype, configs)) for dtype in dtypes)


def run(
    *,
    models: Sequence[str],
    dtypes: Sequence[str],
    configs: Sequence[str],
    ms: Sequence[int],
    patterns: Sequence[Pattern],
    device: str,
    timing: bool,
    on_entry: Callable[[List[Dict[str, Any]]], None] = lambda results: None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    import torch

    rows = reference_rows(ms, patterns)
    results: List[Dict[str, Any]] = []
    for model_name in models:
        for op_index, op in enumerate(expert_ops(MODELS[model_name])):
            for dtype in dtypes:
                a, weight = make_operands(op, rows, dtype, device, seed + op_index)
                b_reference = weight.t() if op.transposed else weight
                for kernel in kernels_for(dtype, configs):
                    entry: Dict[str, Any] = {
                        "model": model_name, "op": asdict(op), "dtype": dtype, "kernel": kernel,
                    }
                    fn = kernel_fn(kernel, op, weight)
                    binaries = triton_binaries if kernel.startswith("triton:") else lambda: None
                    try:
                        with kernel_context(kernel):
                            entry["invariance"] = invariance(
                                fn, a, b_reference, ms=ms, patterns=patterns,
                                tolerance=TOLERANCE[dtype], binaries=binaries,
                            )
                            if timing and kernel != "cublas-norpr":
                                with no_nan_fill():
                                    entry["speed"] = speed(fn, op, a, ms=ms, patterns=patterns)
                    except NotImplementedError as exc:
                        entry["unsupported"] = str(exc)
                    except Exception as exc:  # noqa: BLE001 - recorded; exit status reports it
                        entry["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                        if device == "cuda":
                            torch.cuda.synchronize()  # surface a sticky CUDA error here
                    results.append(entry)
                    _report_line(entry)
                    on_entry(results)
    return results


def _report_line(entry: Dict[str, Any]) -> None:
    label = escape(f"{entry['model']} {entry['op']['name']} {entry['dtype']} {entry['kernel']}")
    for key, style in (("unsupported", "yellow"), ("error", "red")):
        if key in entry:
            message = escape(entry[key].splitlines()[0][:160])
            CONSOLE.print(f"[{style}]{label}: {key}: {message}[/{style}]")
            return
    inv = entry["invariance"]
    broken = [item["m"] for item in inv["prefix"] if not item["bit_equal"]]
    patterns = [p["name"] for p in inv["patterns"] if not p["bit_equal"]]
    relative = inv["fp64_relative_l2"]
    CONSOLE.print(
        f"{label}: invariant={inv['invariant']} valid={inv['valid']} "
        f"prefix-broken-m={broken} pattern-broken={escape(str(patterns))} "
        f"fp64-rel={'nan' if relative is None else f'{relative:.2e}'}"
    )


def summary_table(results: Sequence[Dict[str, Any]], m_columns: Sequence[int]) -> Table:
    table = Table(title="expert GEMM baseline (median us, L2 cold; sep/merged eager | graph)")
    for column in ("model", "op", "dtype", "kernel", "invariant", "valid"):
        table.add_column(column)
    for m in m_columns:
        table.add_column(f"M={m}", justify="right")
    table.add_column("sep/merged", justify="right")
    for entry in results:
        if "invariance" not in entry:
            continue
        times = {row["m"]: row["median_us"] for row in entry.get("speed", {}).get("sweep", [])}
        ratios = [
            f"{row['name']}:{row['eager']['separate_over_merged']:.1f}"
            f"|{row['graph']['separate_over_merged']:.1f}"
            for row in entry.get("speed", {}).get("patterns", [])
        ]
        table.add_row(
            entry["model"], entry["op"]["name"], entry["dtype"], entry["kernel"],
            str(entry["invariance"]["invariant"]), str(entry["invariance"]["valid"]),
            *(f"{times[m]:.0f}" if m in times else "-" for m in m_columns),
            " ".join(ratios) or "-",
        )
    return table


def _csv(kind: Callable[[str], Any], text: str) -> List[Any]:
    return [kind(item) for item in text.split(",") if item]


def _patterns(text: str) -> List[Pattern]:
    patterns = []
    for item in _csv(str, text):
        if item == "ragged":
            patterns.append(RAGGED_PATTERN)
            continue
        micro, per = item.split("x")
        patterns.append(equal_pattern(int(micro), int(per)))
    return patterns


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--models", default="deepseek-v3,kanana-30b-a3b")
    parser.add_argument("--dtypes", default=",".join(DTYPES))
    parser.add_argument("--configs", default=",".join(CONFIGS),
                        help=f"Triton tile configs (default all; stage 0 uses {DEFAULT_CONFIG})")
    parser.add_argument("--ms", default=",".join(map(str, DEFAULT_MS)))
    parser.add_argument(
        "--patterns",
        default=",".join([*(f"{k}x{r}" for k, r in DEFAULT_PATTERNS), "ragged"]),
        help="separate-vs-merged patterns: KxR (k micro-batches of r rows) or 'ragged'",
    )
    parser.add_argument("--no-timing", action="store_true", help="bit checks and validity only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    models, dtypes = _csv(str, args.models), _csv(str, args.dtypes)
    configs = _csv(str, args.configs)
    for name, values, known in (
        ("model", models, MODELS), ("dtype", dtypes, DTYPES), ("config", configs, CONFIGS),
    ):
        unknown = [value for value in values if value not in known]
        if unknown or not values and name != "config":
            parser.error(f"unknown or no {name}(s) {unknown}; choose from {sorted(known)}")
    ms = sorted(set(_csv(int, args.ms)))
    try:
        patterns = _patterns(args.patterns)
    except ValueError:
        parser.error("--patterns takes KxR items or 'ragged'")
    if not ms or ms[0] < 1:
        parser.error("--ms needs positive row counts")
    if any(len(p.chunks) < 2 or min(p.chunks) < 1 for p in patterns):
        parser.error("--patterns needs k >= 2 micro-batches of r >= 1 rows")
    timing = not args.no_timing
    interpreted = os.environ.get("TRITON_INTERPRET") == "1"
    if args.device == "cpu":
        if timing:
            parser.error("--device cpu has no timing; pass --no-timing")
        if configs and not interpreted:
            parser.error("Triton kernels on --device cpu need TRITON_INTERPRET=1")
    elif interpreted:
        parser.error("TRITON_INTERPRET=1 is not GPU evidence; unset it for --device cuda")
    configure(args.device)
    header = {
        "environment": environment(args.device),
        "settings": {
            "models": models, "dtypes": dtypes, "configs": configs, "ms": ms,
            "patterns": [asdict(pattern) for pattern in patterns], "row_offset": ROW_OFFSET,
            "reference_rows": reference_rows(ms, patterns), "tolerance": TOLERANCE,
            "timing": timing,
            "configs_detail": {name: asdict(CONFIGS[name]) for name in configs},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def dump(results: List[Dict[str, Any]], complete: bool = False) -> None:
        payload = {**header, "complete": complete, "results": results}
        if complete:
            payload["triton_binaries"] = triton_binaries() if configs else None
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    results = run(models=models, dtypes=dtypes, configs=configs, ms=ms, patterns=patterns,
                  device=args.device, timing=timing, on_entry=dump)
    dump(results, complete=True)
    CONSOLE.print(summary_table(results, [m for m in (1, 2, 16, 128, 1024) if m in ms]))
    errors = [entry for entry in results if "error" in entry]
    if errors or len(results) != expected_entries(models, dtypes, configs):
        CONSOLE.print(f"[red]{len(errors)} entries failed; see {escape(str(args.output))}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
