"""A0-exact merged expert GEMM through cuBLASLt (optional JIT extension, ADR 0002).

A0 (L2L) calls the native cuBLAS GEMM once per micro-batch segment. The kernel cuBLAS picks
depends on the segment's row count M, and a row's bits depend only on that kernel, not on its
neighbours (micro results in ``benchmarks/results/a0-exact-levers-20261119/``: 1359 of 1359 pairs).
So segments that A0 serves with one kernel can share one call, provided that call is forced onto
that kernel. Native ``cublasGemmEx`` cannot be forced, cuBLASLt can: for every (weight shape,
layout, M) the table asks the Lt heuristic for candidates once and keeps the one whose output
equals the native result bit for bit on that segment. Segments with no such algo (1-row GEMV
calls, for instance) and non-contiguous weights keep the native per-segment call.

Layouts, as torch hands them to cuBLAS: forward ``x @ W.T`` (``transposed=True``, W is [N, K]) and
the input gradient ``g @ W`` (``transposed=False``).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

HEURISTIC_LIMIT = 32
_EXTENSION_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a0_exact_lt_ext.cpp")


def native_matmul(x: Any, w: Any, transposed: bool) -> Any:
    """The A0 call: what ``F.linear`` (forward) and its autograd (input gradient) run."""
    import torch

    return torch.mm(x, w.t()) if transposed else torch.mm(x, w)


Bucket = Tuple[Optional[Hashable], List[int]]


def bucket_segments(keys: Sequence[Optional[Hashable]]) -> List[Bucket]:
    """Group segment indices by key in first-seen order; a ``None`` key stays alone."""
    buckets: List[Bucket] = []
    where: Dict[Hashable, int] = {}
    for index, key in enumerate(keys):
        if key is None:
            buckets.append((None, [index]))
        elif key in where:
            buckets[where[key]][1].append(index)
        else:
            where[key] = len(buckets)
            buckets.append((key, [index]))
    return buckets


def algo_key(algo: Any) -> Tuple[int, ...]:
    return tuple(int(v) for v in algo.tolist())


class AlgoTable:
    """Per (weight shape, dtype, layout, M): the Lt algo that reproduces the native bits, or None.

    ``backend`` has ``heuristics(x, w, transposed, limit)`` (candidate algos) and
    ``matmul(x, w, transposed, algo)``. ``native`` is the A0 call used as the reference."""

    def __init__(self, backend: Any, native: Callable[[Any, Any, bool], Any] = native_matmul,
                 limit: int = HEURISTIC_LIMIT) -> None:
        self.backend = backend
        self.native = native
        self.limit = limit
        self.entries: Dict[Tuple[Any, ...], Optional[Any]] = {}
        # (weight shape, dtype, layout, algo) -> smallest merged row count Lt refused
        self.refused: Dict[Tuple[Any, ...], int] = {}
        self.stats = {"discovered": 0, "no_algo": 0, "merged_calls": 0, "merge_refused": 0}
        self._keys: Dict[int, Tuple[int, ...]] = {}  # id(algo) -> algo_key, algos live here
        # (weight shape, dtype, layout) -> [[uses, algo], ...] of the algos chosen so far
        self._chosen: Dict[Tuple[Any, ...], List[List[Any]]] = {}

    def key_of(self, algo: Any) -> Tuple[int, ...]:
        """``algo_key`` of a table algo, cached (``tolist`` per segment is measurable)."""
        key = self._keys.get(id(algo))
        if key is None:
            key = self._keys[id(algo)] = algo_key(algo)
        return key

    def _reproduces(self, x: Any, w: Any, transposed: bool, algo: Any, want: Any) -> bool:
        """True when ``algo`` gives the native bits on ``x`` and on three more inputs."""
        import torch

        try:
            if not torch.equal(self.backend.matmul(x, w, transposed, algo), want):
                return False
            for scale in (0.5, 2.0, 4.0):
                probe = (torch.randn_like(x.float()) * scale).to(x.dtype)
                if not torch.equal(self.backend.matmul(probe, w, transposed, algo),
                                   self.native(probe, w, transposed)):
                    return False
        except RuntimeError:
            return False
        return True

    def lookup(self, x: Any, w: Any, transposed: bool) -> Optional[Any]:
        import torch

        key = (tuple(w.shape), str(w.dtype), bool(transposed), int(x.shape[0]))
        if key in self.entries:
            return self.entries[key]
        want = self.native(x, w, transposed)
        found = None
        form = key[:3]
        chosen = self._chosen.setdefault(form, [])
        # An algo the form already uses comes first: neighbouring row counts then share one algo and
        # one call merges more segments. It has to reproduce the native bits on the segment itself
        # and on other data of the same shape (the first-match policy trusts the one sample).
        for entry in sorted(chosen, key=lambda e: -e[0]):
            if self._reproduces(x, w, transposed, entry[1], want):
                found = entry
                break
        if found is not None:
            found[0] += 1
            found = found[1]
        else:
            for algo in self.backend.heuristics(x, w, transposed, self.limit):
                try:
                    got = self.backend.matmul(x, w, transposed, algo)
                except RuntimeError:
                    continue
                if torch.equal(got, want):
                    found = algo
                    chosen.append([1, algo])
                    break
        self.entries[key] = found
        self.stats["discovered" if found is not None else "no_algo"] += 1
        return found


def merged_matmul(x: Any, w: Any, transposed: bool, counts: Sequence[int], table: AlgoTable) -> Any:
    """``native_matmul`` per segment of ``counts``, with same-algo segments merged into one call."""
    import torch

    parts = list(x.split(list(counts)))
    if not w.is_contiguous() or not table.backend.supports(x, w):
        return torch.cat([table.native(p, w, transposed) for p in parts])
    algos = [table.lookup(p, w, transposed) for p in parts]
    keys = [None if a is None else table.key_of(a) for a in algos]
    if hasattr(table.backend, "merged"):
        return _fused_matmul(x, w, transposed, [int(c) for c in counts], keys, table)
    outs: List[Any] = [None] * len(parts)
    for key, idx in bucket_segments(keys):
        if key is None:
            outs[idx[0]] = table.native(parts[idx[0]], w, transposed)
            continue
        algo = algos[idx[0]]
        rows = sum(int(counts[i]) for i in idx)
        refusal = (tuple(w.shape), str(w.dtype), bool(transposed), key)
        if len(idx) > 1 and rows < table.refused.get(refusal, rows + 1):
            joined = torch.cat([parts[i] for i in idx]).contiguous()
            try:
                result = table.backend.matmul(joined, w, transposed, algo)
            except RuntimeError:  # an algo found at small M may not accept a larger M
                table.refused[refusal] = rows
                table.stats["merge_refused"] += 1
            else:
                table.stats["merged_calls"] += 1
                for out, i in zip(result.split([int(counts[i]) for i in idx]), idx):
                    outs[i] = out
                continue
        for i in idx:  # one call per segment, each with its own (A0-exact) algo
            outs[i] = table.backend.matmul(parts[i].contiguous(), w, transposed, algo)
    return torch.cat(outs)


def _fused_matmul(x: Any, w: Any, transposed: bool, counts: List[int],
                  keys: List[Optional[Tuple[int, ...]]], table: AlgoTable) -> Any:
    """The same plan, run by the backend's ``merged`` in one call (C++: gather, forced-algo Lt
    call and scatter per group, native ``mm`` for groups without an algo). Groups Lt refuses
    come back as indices; their segments run again one per group and the refusal is kept."""
    shape = (tuple(w.shape), str(w.dtype), bool(transposed))
    groups: List[List[int]] = []
    group_algos: List[List[int]] = []
    for key, idx in bucket_segments(keys):
        if key is None:
            groups.append(idx)
            group_algos.append([])
            continue
        rows = sum(counts[i] for i in idx)
        if len(idx) > 1 and rows >= table.refused.get(shape + (key,), rows + 1):
            groups.extend([i] for i in idx)
            group_algos.extend(list(key) for _ in idx)
        else:
            groups.append(idx)
            group_algos.append(list(key))
    out, failed = table.backend.merged(x.contiguous(), w, transposed, counts, groups, group_algos)
    failed = set(failed)
    table.stats["merged_calls"] += sum(
        1 for i, g in enumerate(groups) if len(g) > 1 and i not in failed
    )
    if not failed:
        return out
    redo: List[List[int]] = []
    redo_algos: List[List[int]] = []
    for index in sorted(failed):
        rows = sum(counts[i] for i in groups[index])
        refusal = shape + (tuple(group_algos[index]),)
        table.refused[refusal] = min(rows, table.refused.get(refusal, rows))
        table.stats["merge_refused"] += 1
        redo.extend([i] for i in groups[index])
        redo_algos.extend(group_algos[index] for _ in groups[index])
    again, still = table.backend.merged(x.contiguous(), w, transposed, counts, redo, redo_algos)
    if still:
        raise RuntimeError(f"cuBLASLt refused single-segment calls {still} (algo found at that M)")
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    for (i,) in redo:
        out[offsets[i]:offsets[i + 1]] = again[offsets[i]:offsets[i + 1]]
    return out


def _merged_linear_class() -> Any:
    import torch

    class _MergedLinear(torch.autograd.Function):
        """``x @ W.T`` per segment (merged calls); backward gives the input gradient only."""

        @staticmethod
        def forward(ctx: Any, x: Any, w: Any, counts: Sequence[int], table: AlgoTable) -> Any:
            ctx.counts, ctx.table = list(counts), table
            ctx.save_for_backward(w)
            return merged_matmul(x, w, True, counts, table)

        @staticmethod
        def backward(ctx: Any, grad: Any) -> Tuple[Any, None, None, None]:
            (w,) = ctx.saved_tensors
            if ctx.needs_input_grad[1]:
                raise RuntimeError("lt-merged expert GEMM serves frozen weights only")
            gx = merged_matmul(grad.contiguous(), w, False, ctx.counts, ctx.table)
            return gx, None, None, None

    return _MergedLinear


class _LazyFunction:
    _cls: Any = None

    def apply(self, *args: Any) -> Any:
        if _LazyFunction._cls is None:
            _LazyFunction._cls = _merged_linear_class()
        return _LazyFunction._cls.apply(*args)


MergedLinear = _LazyFunction()


class LtExtensionBackend:
    """The compiled ``a0_exact_lt_ext`` (cuBLASLt on torch's Lt handle and stream)."""

    def __init__(self, module: Any) -> None:
        self.module = module

    @staticmethod
    def supports(x: Any, w: Any) -> bool:
        import torch

        return bool(x.is_cuda and w.is_cuda and x.dtype == torch.bfloat16 and w.dtype == x.dtype)

    def heuristics(self, x: Any, w: Any, transposed: bool, limit: int) -> List[Any]:
        algos = self.module.heuristics(x, w, transposed, limit)
        return list(algos.unbind(0)) if algos.numel() else []

    def enumerate_algos(self, x: Any, w: Any, transposed: bool, max_custom: int = 2) -> List[Any]:
        """Every split-K-free algo config cuBLASLt accepts for this problem (candidate pool)."""
        algos = self.module.enumerate_algos(x, w, transposed, max_custom)
        return list(algos.unbind(0)) if algos.numel() else []

    def matmul(self, x: Any, w: Any, transposed: bool, algo: Any) -> Any:
        return self.module.matmul(x, w, transposed, algo)

    def register_algo(self, out_cols: int, k: int, transposed: bool, rows: int,
                      algo: List[int]) -> None:
        self.module.register_algo(out_cols, k, transposed, rows, algo)

    def set_streams(self, count: int) -> None:
        """Spread the Lt calls of one segment matmul over ``count`` CUDA streams (1: off)."""
        self.module.set_streams(count)

    def expert_forward(self, hidden: Any, weights: Any, gate_up: Any, down: Any,
                       counts: List[int]) -> Any:
        return self.module.expert_forward(hidden, weights, gate_up, down, counts)

    def expert_backward(self, hidden: Any, weights: Any, gate_up: Any, down: Any, grad_out: Any,
                        counts: List[int], need_hidden: bool, need_weights: bool) -> Any:
        return self.module.expert_backward(hidden, weights, gate_up, down, grad_out, counts,
                                           need_hidden, need_weights)

    def merged(self, x: Any, w: Any, transposed: bool, counts: List[int],
               groups: List[List[int]], algos: List[List[int]]) -> Tuple[Any, List[int]]:
        out, failed = self.module.merged(x, w, transposed, counts, groups, algos)
        return out, list(failed)



class FusedExpert:
    """The fused expert (``--expert-gemm lt-fused``): finds the A0-exact Lt algo of every
    (form, row count) once, registers it in the extension, then forward and direct backward of
    one expert visit are one extension call each (``expert_forward``, ``expert_backward``)."""

    def __init__(self, backend: Any, require_cuda: bool = True) -> None:
        self.backend = backend
        self.table = AlgoTable(backend)
        self.require_cuda = require_cuda
        self.registered: set = set()
        self._known: Dict[Tuple[Any, ...], set] = {}  # weight shapes -> rows done in all forms

    def usable(self, hidden: Any, gate_up: Any, down: Any) -> bool:
        if not self.require_cuda:
            return True
        import torch

        return bool(
            hidden.is_cuda and hidden.dtype == torch.bfloat16 and hidden.dim() == 2
            and gate_up.dtype == torch.bfloat16 and down.dtype == torch.bfloat16
            and gate_up.is_contiguous() and down.is_contiguous()
        )

    def ensure(self, counts: Sequence[int], gate_up: Any, down: Any) -> bool:
        """Register the algo of every new row count in the four GEMM forms; False when one
        has no A0-exact algo (the caller then takes the Python path)."""
        shapes = (tuple(gate_up.shape), tuple(down.shape))
        known = self._known.setdefault(shapes, set())
        if all(c in known for c in counts):
            return True
        for rows in set(int(c) for c in counts):
            samples: Dict[int, Any] = {}  # one input per K for this row count, dropped after
            for w, transposed in ((gate_up, True), (down, True), (down, False), (gate_up, False)):
                k = int(w.shape[1] if transposed else w.shape[0])
                out_cols = int(w.shape[0] if transposed else w.shape[1])
                key = (out_cols, k, transposed, rows)
                if key in self.registered:
                    continue
                if k not in samples:
                    samples[k] = self._sample(rows, k, w)
                algo = self.table.lookup(samples[k], w, transposed)
                if algo is None:
                    return False
                self.backend.register_algo(out_cols, k, transposed, rows,
                                           [int(v) for v in algo.tolist()])
                self.registered.add(key)
            known.add(rows)
        return True

    @staticmethod
    def _sample(rows: int, k: int, like: Any) -> Any:
        import torch

        return torch.randn(rows, k, device=like.device).to(like.dtype)


_KERNELS_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "a0_exact_lt_kernels.cu")


def load_extension(
    verbose: bool = False, kernels: bool = True, strict: bool = False
) -> LtExtensionBackend:
    """JIT-build the extension (CUDA only; needs cublasLt headers). With ``kernels`` the
    elementwise CUDA kernels of the fused expert are built too (needs nvcc, from CUDA_HOME or
    the ``nvidia-cuda-nvcc`` wheel). When that build fails the ATen-only extension is used,
    unless ``strict``: then the build error is raised, because ``lt-fused`` needs the kernels."""
    from torch.utils.cpp_extension import load

    include_paths, ldflags = _cuda_paths()
    if kernels:
        try:
            module = _build(load, include_paths, ldflags, verbose, with_kernels=True)
            return LtExtensionBackend(module)
        except Exception as exc:  # noqa: BLE001 - nvcc missing or a build error: say so, fall back
            if strict:
                raise
            from rich.console import Console

            Console(stderr=True).out(
                f"[a0_exact_lt] kernels build failed ({exc!r:.300}); using the ATen path")
    return LtExtensionBackend(_build(load, include_paths, ldflags, verbose, with_kernels=False))


def _build(load: Any, include_paths: List[str], ldflags: List[str], verbose: bool,
           with_kernels: bool) -> Any:
    import torch
    import torch.utils.cpp_extension as cpp_ext

    sources = [_EXTENSION_SOURCE]
    name = "a0_exact_lt_ext"
    extra: Dict[str, Any] = {}
    if with_kernels:
        home = _nvcc_home()
        cpp_ext.CUDA_HOME = home
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        sources.append(_KERNELS_SOURCE)
        name = "a0_exact_lt_ext_k"
        extra = {"extra_cflags": ["-DA0_HAVE_KERNELS"], "extra_cuda_cflags": ["-DA0_HAVE_KERNELS"]}
    return load(
        name=name, sources=sources, extra_include_paths=include_paths, extra_ldflags=ldflags,
        verbose=verbose, **extra,
    )


def _nvcc_home() -> str:
    """A directory with ``bin/nvcc``: CUDA_HOME, /usr/local/cuda, or the nvidia-cuda-nvcc wheel
    (``nvidia/cu13``). A wheel layout gets ``lib64`` and ``libcudart.so`` links when missing."""
    import glob

    for home in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        if home and os.path.isfile(os.path.join(home, "bin", "nvcc")):
            _link_wheel_libs(home)
            return home
    import nvidia

    for root in getattr(nvidia, "__path__", []):
        for bin_dir in sorted(glob.glob(os.path.join(root, "*", "bin"))):
            home = os.path.dirname(bin_dir)
            if os.path.isfile(os.path.join(bin_dir, "nvcc")):
                _link_wheel_libs(home)
                return home
    raise RuntimeError("no nvcc found (set CUDA_HOME or install nvidia-cuda-nvcc)")


def _link_wheel_libs(home: str) -> None:
    """Pip wheels ship ``lib/libcudart.so.13`` only; the linker wants ``lib64`` and ``.so``."""
    import glob

    lib = os.path.join(home, "lib")
    if not os.path.isdir(lib):
        return
    try:
        if not os.path.exists(os.path.join(home, "lib64")):
            os.symlink("lib", os.path.join(home, "lib64"))
        runtime = os.path.join(lib, "libcudart.so")
        versions = sorted(glob.glob(runtime + ".*"))
        if not os.path.exists(runtime) and versions:
            os.symlink(os.path.basename(versions[0]), runtime)
    except OSError:
        pass


def toolchain_error() -> Optional[str]:
    """Why the extension cannot be built here (None when it can): no C++ compiler, no ninja, no
    nvcc, or no cublasLt headers. Looks only; builds nothing."""
    import shutil

    if not any(shutil.which(name) for name in ("c++", "g++", "clang++")):
        return "no C++ compiler found"
    from torch.utils.cpp_extension import is_ninja_available

    if not is_ninja_available():
        return "ninja is not installed (pip install ninja)"
    try:
        _nvcc_home()
        include_paths, _ = _cuda_paths()
    except (RuntimeError, ImportError) as exc:
        return str(exc) if isinstance(exc, RuntimeError) else f"no CUDA toolkit wheel ({exc})"
    return _cublas_mismatch(include_paths[0])


def _cublas_mismatch(include: str) -> Optional[str]:
    """The headers' cuBLAS major against the CUDA major torch runs on. A build against the other
    major links a cuBLASLt that torch does not load and crashes in its first call."""
    import re

    import torch

    try:
        with open(os.path.join(include, "cublas_api.h"), encoding="utf-8", errors="ignore") as fh:
            found = re.search(r"#define\s+CUBLAS_VER_MAJOR\s+(\d+)", fh.read())
    except OSError:
        return None
    torch_cuda = str(torch.version.cuda or "")
    if found is None or not torch_cuda or found.group(1) == torch_cuda.split(".")[0]:
        return None
    return (f"the CUDA headers in {include} are cuBLAS {found.group(1)}, torch runs on CUDA "
            f"{torch_cuda}; point CUDA_HOME at a matching toolkit (the nvidia-cuda-nvcc wheel)")


def _cuda_paths() -> Tuple[List[str], List[str]]:
    """CUDA headers and cuBLASLt for the build: the toolkit under CUDA_HOME when there is one,
    else the ``nvidia`` pip wheels torch itself loads (``nvidia/cu13/{include,lib}``)."""
    import glob

    home = os.environ.get("CUDA_HOME") or ("/usr/local/cuda" if os.path.isdir("/usr/local/cuda")
                                           else None)
    if home and os.path.isfile(os.path.join(home, "include", "cublasLt.h")):
        include = os.path.join(home, "include")
        cccl = os.path.join(include, "cccl")  # pip wheels keep nv/target here; toolkits find it
        paths = [include] + ([cccl] if os.path.isdir(cccl) else [])
        lib = next((d for d in (os.path.join(home, "lib64"), os.path.join(home, "lib"))
                    if glob.glob(os.path.join(d, "libcublasLt.so*"))), os.path.join(home, "lib64"))
        if os.path.exists(os.path.join(lib, "libcublasLt.so")):
            return paths, [f"-L{lib}", "-lcublasLt", f"-Wl,-rpath,{lib}"]
        libs = sorted(glob.glob(os.path.join(lib, "libcublasLt.so*")))  # wheels: only the .so.13
        link = f"-l:{os.path.basename(libs[0])}" if libs else "-lcublasLt"
        return paths, [f"-L{lib}", link, f"-Wl,-rpath,{lib}"]
    import nvidia  # the CUDA wheels torch depends on

    for root in getattr(nvidia, "__path__", []):
        for include in sorted(glob.glob(os.path.join(root, "*", "include"))):
            lib = os.path.join(os.path.dirname(include), "lib")
            if os.path.isfile(os.path.join(include, "cublasLt.h")) and os.path.isdir(lib):
                libs = sorted(glob.glob(os.path.join(lib, "libcublasLt.so*")))
                link = f"-l:{os.path.basename(libs[0])}" if libs else "-lcublasLt"
                # CUDA 13 keeps the CCCL headers (nv/target, cuda/std) in include/cccl
                cccl = os.path.join(include, "cccl")
                paths = [include] + ([cccl] if os.path.isdir(cccl) else [])
                return paths, [f"-L{lib}", link, f"-Wl,-rpath,{lib}"]
    raise RuntimeError("no CUDA headers with cublasLt.h found (set CUDA_HOME)")
