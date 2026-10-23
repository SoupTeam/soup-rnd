"""Development-only direct cuBLAS grouped BF16 probe; no D3 acceptance or timing.

Original ctypes binding of NVIDIA's documented C API. No third-party kernel
source is copied. Explicit absolute library path, caller-owned CUDA workspace,
pointer arrays retain their tensors. Nothing is installed in the trainer.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def column_major_problem(rows: int, inner: int, columns: int) -> dict:
    """Row-major H @ W.T becomes column-major W.T @ H.T without copies."""
    if any(type(value) is not int or not 1 <= value < 2**31
           for value in (rows, inner, columns)):
        raise ValueError("nonempty 32-bit matrix dimensions required")
    return {"transa": 1, "transb": 0, "m": columns, "n": rows, "k": inner,
            "lda": inner, "ldb": inner, "ldc": columns, "group_size": 1}


class CublasGroupedProbe:
    """Synchronous development ownership; not yet a bounded runtime component."""

    def __init__(
        self, library: Path, *, workspace_bytes: int = 4 * 2**20,
        native_single_rows: bool = False, native_rows_max: int | None = None,
    ) -> None:
        import torch

        if (sys.platform != "linux" or not library.is_absolute() or not library.is_file()
                or library.name not in ("libcublas.so.12", "libcublas.so.13")):
            raise ValueError("explicit absolute Linux cuBLAS 12/13 shared library required")
        if type(workspace_bytes) is not int or workspace_bytes < 16 * 1024:
            raise ValueError("explicit workspace of at least 16 KiB required")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.native_rows_max = (1 if native_single_rows else 0) if native_rows_max is None else (
            native_rows_max
        )
        if type(self.native_rows_max) is not int or not 0 <= self.native_rows_max <= 16:
            raise ValueError("native row cutoff must be an integer from 0 through 16")
        self.lib = ctypes.CDLL(str(library))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        signatures = {
            "cublasCreate_v2": [ctypes.POINTER(pointer)],
            "cublasDestroy_v2": [pointer],
            "cublasSetStream_v2": [pointer, pointer],
            "cublasSetWorkspace_v2": [pointer, pointer, ctypes.c_size_t],
            "cublasGetVersion_v2": [pointer, ctypes.POINTER(integer)],
            "cublasGemmGroupedBatchedEx": [
                pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
                integer, pointer, pointer, integer, pointer, pointer, pointer, integer,
                pointer, integer, pointer, integer,
            ],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, integer
        self.handle = pointer()
        self.workspace = None
        self._lock = threading.Lock()
        self._check(self.lib.cublasCreate_v2(ctypes.byref(self.handle)), "create")
        try:
            version = integer()
            self._check(self.lib.cublasGetVersion_v2(self.handle, ctypes.byref(version)), "version")
            self.version = version.value
            self.workspace = torch.empty(workspace_bytes, device=self.device, dtype=torch.uint8)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status:
            raise RuntimeError(f"cuBLAS {operation} returned status {status}")

    def prepare(self, inputs: list, weights: list) -> tuple[list, Any]:
        import torch

        if not self.handle or not 1 <= len(inputs) <= 4 or len(inputs) != len(weights):
            raise ValueError("live handle and 1 through 4 input/weight pairs required")
        selected = []
        singles = []
        outputs = []
        problems = []
        for hidden, weight in zip(inputs, weights):
            if (hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]
                    or hidden.dtype != torch.bfloat16 or weight.dtype != hidden.dtype
                    or hidden.device != self.device or weight.device != self.device
                    or not hidden.is_contiguous() or not weight.is_contiguous()
                    or hidden.requires_grad or weight.requires_grad
                    or hidden.data_ptr() % 16 or weight.data_ptr() % 16):
                raise ValueError("contiguous aligned non-autograd CUDA BF16 pairs required")
            output = torch.empty((hidden.shape[0], weight.shape[0]), device=self.device,
                                 dtype=torch.bfloat16)
            outputs.append(output)
            if 0 < hidden.shape[0] <= self.native_rows_max:
                singles.append((hidden, weight, output))
            elif hidden.shape[0]:
                problems.append(column_major_problem(hidden.shape[0], hidden.shape[1],
                                                     weight.shape[0]))
                selected.append((hidden, weight, output))
        if not selected and not singles:
            return outputs, lambda: outputs
        host = {}
        for name in problems[0] if problems else ():
            host[name] = (ctypes.c_int * len(problems))(*(problem[name] for problem in problems))
        host["alpha"] = (ctypes.c_float * len(problems))(*([1.0] * len(problems)))
        host["beta"] = (ctypes.c_float * len(problems))(*([0.0] * len(problems)))
        pointers = [torch.tensor([entry[index].data_ptr() for entry in selected],
                                 device=self.device, dtype=torch.int64)
                    for index in (1, 0, 2)] if selected else []

        def launch() -> list:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError("grouped cuBLAS probe is already active")
            try:
                if not self.handle:
                    raise RuntimeError("grouped cuBLAS probe is closed")
                with torch.cuda.device(self.device):
                    stream = torch.cuda.current_stream(self.device)
                    if selected:
                        self._check(self.lib.cublasSetStream_v2(
                            self.handle, stream.cuda_stream,
                        ), "stream")
                        # Setting the stream resets cuBLAS workspace, so set ours afterwards.
                        self._check(self.lib.cublasSetWorkspace_v2(
                            self.handle, self.workspace.data_ptr(), self.workspace.numel(),
                        ), "workspace")
                        self._check(self.lib.cublasGemmGroupedBatchedEx(
                            self.handle, host["transa"], host["transb"], host["m"], host["n"],
                            host["k"], host["alpha"], pointers[0].data_ptr(), 14, host["lda"],
                            pointers[1].data_ptr(), 14, host["ldb"], host["beta"],
                            pointers[2].data_ptr(), 14, host["ldc"], len(problems),
                            host["group_size"], 68,
                        ), "grouped GEMM")
                    for hidden, weight, output in singles:
                        torch.mm(hidden, weight.T, out=output)
                return outputs
            finally:
                # Keep host descriptors, pointer arrays and selected weights/outputs
                # alive until cuBLAS has finished, including failed API calls.
                _keep_alive = selected, singles, host, pointers
                try:
                    torch.cuda.synchronize(self.device)
                finally:
                    self._lock.release()

        return outputs, launch

    def close(self) -> None:
        import torch

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close an active grouped cuBLAS probe")
        try:
            torch.cuda.synchronize(self.device)
            if self.handle:
                self._check(self.lib.cublasDestroy_v2(self.handle), "destroy")
                self.handle = ctypes.c_void_p()
            self.workspace = None
        finally:
            self._lock.release()


@contextmanager
def reference_policy(mode: str) -> Any:
    """Explicit process-local policy applied to BOTH arms, then fully restored."""
    import torch

    if mode not in ("current", "cublaslt-no-splitk"):
        raise ValueError("unknown reference policy")
    cuda = torch.backends.cuda
    old_blas = cuda.preferred_blas_library()
    old_flags = (cuda.matmul.allow_bf16_reduced_precision_reduction,
                 cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
    try:
        if mode == "cublaslt-no-splitk":
            cuda.preferred_blas_library("cublaslt")
            cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
        yield {
            "mode": mode, "blas": str(cuda.preferred_blas_library()),
            "reduced_precision": cuda.matmul.allow_bf16_reduced_precision_reduction,
            "split_k": cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
            "scope": "same explicit settings in native and grouped development checks",
        }
    finally:
        cuda.matmul.allow_bf16_reduced_precision_reduction = old_flags
        cuda.preferred_blas_library(old_blas)


def run_probe(
    folder: Path, library: Path, *, native_single_rows: bool = False,
    native_rows_max: int | None = None,
) -> dict:
    import torch
    import torch.nn.functional as functional

    backend = CublasGroupedProbe(library, native_single_rows=native_single_rows,
                                 native_rows_max=native_rows_max)
    report = {"status": "running", "records": [], "profiles": [],
              "formal_measurement": False, "d3_accepted": False,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "library": str(library), "cublas_version": backend.version,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "workspace_bytes": backend.workspace.numel(),
              "native_rows_max": backend.native_rows_max,
              "row_policy": "native mm at/below cutoff; grouped GEMM above; empty is no-op"}
    try:
        for seed in (11, 29):
            for inner, columns, rows in (
                (32, 32, (0, 1, 7, 17)), (33, 47, (1, 17, 31, 33)),
                (2048, 2048, (1, 17, 63, 129)), (1024, 2048, (1, 17, 64, 128)),
            ):
                gen = torch.Generator().manual_seed(seed)
                inputs = [(torch.randn(count, inner, generator=gen) * 0.2).to(
                    device="cuda", dtype=torch.bfloat16,
                ) for count in rows]
                weights = [(torch.randn(columns, inner, generator=gen) * 0.2).to(
                    device="cuda", dtype=torch.bfloat16,
                ) for _ in rows]
                with torch.no_grad():
                    expected = [functional.linear(hidden, weight)
                                for hidden, weight in zip(inputs, weights)]
                    outputs, launch = backend.prepare(inputs, weights)
                    launch()
                    alone = []
                    for hidden, weight in zip(inputs, weights):
                        single, call = backend.prepare([hidden], [weight])
                        call()
                        alone.append(single[0])
                    for index, (want, got, single) in enumerate(zip(expected, outputs, alone)):
                        report["records"].append({
                            "seed": seed, "inner": inner, "columns": columns, "rows": rows[index],
                            "equal": torch.equal(want, got),
                            "width_equal": torch.equal(got, single),
                            "finite": bool(torch.isfinite(got).all()),
                            "unequal_elements": int(torch.count_nonzero(want != got)),
                            "max_abs_diff": float((want.float() - got.float()).abs().max())
                            if want.numel() else 0.0,
                        })
                (folder / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
                if inner == 2048 and seed == 11:
                    for name, call in (
                        ("native", lambda: [functional.linear(a, b)
                                            for a, b in zip(inputs, weights)]),
                        ("prepared-grouped", launch),
                        ("prepare-and-grouped", lambda: backend.prepare(inputs, weights)[1]()),
                    ):
                        with torch.no_grad(), torch.profiler.profile(activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]) as profile:
                            call()
                            torch.cuda.synchronize()
                        path = folder / f"{name}.json"
                        profile.export_chrome_trace(str(path))
                        events = json.loads(path.read_text())["traceEvents"]
                        kernels = [event for event in events
                                   if event.get("cat") == "kernel" and event.get("ph") == "X"]
                        report["profiles"].append({
                            "name": name, "kernel_count": len(kernels),
                            "kernel_names": [row["name"] for row in kernels],
                        })
        exact = all(row["equal"] and row["width_equal"] and row["finite"]
                    for row in report["records"])
        report["status"] = "arithmetic_exact" if exact else "arithmetic_mismatch"
    except (Exception, KeyboardInterrupt):
        report["status"] = "no_verdict"
        report["traceback"] = traceback.format_exc()
    finally:
        backend.close()
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", choices=("current", "cublaslt-no-splitk"), default="current")
    parser.add_argument("--native-single-rows", action="store_true")
    parser.add_argument("--native-rows-max", type=int)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        with reference_policy(args.reference) as policy:
            (args.out / "reference-policy.json").write_text(json.dumps(policy, indent=2) + "\n")
            report = run_probe(args.out, args.library, native_single_rows=args.native_single_rows,
                               native_rows_max=args.native_rows_max)
    except (Exception, KeyboardInterrupt):
        report = {"status": "no_verdict", "traceback": traceback.format_exc(), "d3_accepted": False}
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    Console().print(f"{report['status']}: {args.out}")
    return 0 if report["status"] == "arithmetic_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
