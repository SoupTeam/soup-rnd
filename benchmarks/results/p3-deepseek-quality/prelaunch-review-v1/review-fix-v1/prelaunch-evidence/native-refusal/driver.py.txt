"""Portable stage c: synthetic native-CUDA/BF16 smoke, NOT scientific admission.

Run only on an approved rental, inside the process-only stage deadline::

    python -m benchmarks.harness.p3_deepseek_quality_smoke --output NEW_DIRECTORY

The only input is a fresh evidence directory. Workload sizes cannot be changed.
No production source, download, provider operation or gradient gate is invoked.
This reproduces the archived attention probe and standalone fixture smoke using
current production attention/training/resource APIs. COMPLETE means fixture
execution only: never a full-model gate PASS, E3 result or budget authorization.
A failure is terminal; retain/upload evidence and follow the approved teardown
procedure. There is no retry, resume, fallback or automatic downsizing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from rich.console import Console

_SCOPE = (
    "Synthetic tiny native DeepSeek-V3 and attention fixtures only; NOT scientific "
    "admission, a full-model gate, E3, quality evidence or phase-3 budget approval."
)
_LENGTHS = (1024, 4096)
_ARCHIVES = (
    "benchmarks/results/p3-deepseek-quality/efficient-vast-v1/sdpa-probe-v1/probe.py.txt",
    "benchmarks/results/p3-deepseek-quality/efficient-vast-v1/standalone-smoke-v1/driver.py.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("invalid fixture directory or evidence name")
    target = Path(os.path.realpath(root / name))
    if os.path.commonpath((str(root), str(target))) != str(root) or target == root:
        raise ValueError("fixture directory or evidence name escapes root")
    return target


class _Directories:
    """Minimal local tmp_path_factory interface for the existing study-pack fixture."""

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.realpath(root))

    def mktemp(self, name: str) -> Path:
        target = _child(self.root, name)
        target.mkdir(exist_ok=False)
        return target


def _new_output(output: Path | str) -> Path:
    if not isinstance(output, (str, os.PathLike)):
        raise TypeError("output must be a nonempty filesystem path")
    raw = os.fspath(output)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("output must be a nonempty filesystem path")
    if os.path.lexists(raw):
        raise FileExistsError(f"output must be fresh: {raw}")
    root = Path(os.path.realpath(raw))
    root.mkdir(parents=True, exist_ok=False)
    return root


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


@contextmanager
def _attempt(root: Path, report: dict, name: str) -> Iterator[dict]:
    """Write an immutable start and terminal receipt, including failed partial work."""
    record: dict[str, Any] = {
        "name": name, "status": "RUNNING", "test_fixture": True,
        "scientific_admission": False, "scope": _SCOPE,
    }
    report["attempts"].append(record)
    receipts = _child(root, "receipts")
    _write_json(_child(receipts, name + "-start.json"), record)
    started = time.monotonic()
    measurement = None
    try:
        from .p3_fp8_smoke import MemoryObservation

        measurement = MemoryObservation("cuda", scope=(
            "Fixture-only whole-process/allocator observation, including residency, "
            "profiling and earlier allocator caches; not full-model resource capacity. "
            "Training windows reset CUDA peak counters internally: use their individual "
            "memory receipts for window peaks, not this enclosing allocator peak."
        ))
        with measurement:
            yield record
        record["status"] = "COMPLETE"
    except BaseException as error:
        record["status"] = "FAILED"
        record["error"] = _error(error)
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        record["memory"] = {} if measurement is None else measurement.result
        _write_json(_child(receipts, name + ".json"), record)
        Console().print_json(json.dumps(record, allow_nan=False))


def _tensor_hash(tensor: Any) -> str:
    import torch

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _attention_cases(root: Path, report: dict) -> None:
    import torch
    import torch.nn.functional as functional

    from . import p3_deepseek_quality_attention as attention

    for backend, length in ((attention.MAIN_BACKEND, 1024),
                            (attention.MAIN_BACKEND, 4096), (attention.FLOOR_BACKEND, 64)):
        hashes = []
        for repeat in range(2):
            name = f"attention-{backend}-L{length}-repeat-{repeat}"
            with _attempt(root, report, name) as record:
                record.update(backend=backend, length=length, batch=1, heads=128,
                              qk_dim=192, v_dim=128, dtype="bfloat16", repeat=repeat)
                observation = None
                try:
                    torch.manual_seed(23)
                    with attention.attention_execution(backend, device="cuda") as observation:
                        tensors = [torch.randn(1, 128, length, width, device="cuda",
                                               dtype=torch.bfloat16, requires_grad=True)
                                   for width in (192, 192, 128)]
                        value = functional.scaled_dot_product_attention(
                            *tensors, is_causal=True, dropout_p=0.0,
                        )
                        value.sum().backward()
                        torch.cuda.synchronize()
                        observation.require_complete(dtype="bfloat16", backward=True)
                        values = [value] + [tensor.grad for tensor in tensors]
                        if not all(tensor is not None and bool(torch.isfinite(tensor).all())
                                   for tensor in values):
                            raise ValueError("nonfinite attention output or Q/K/V gradient")
                        digests = [_tensor_hash(tensor) for tensor in values]
                        record["output_qkv_gradient_sha256"] = digests
                        record["finite_output_and_gradients"] = True
                        hashes.append(digests)
                        if repeat == 1:
                            if hashes[0] != hashes[1]:
                                raise ValueError(
                                    "attention output/gradients differ on deterministic repeat")
                            record["bitwise_repeat_equal"] = True
                        del values, value, tensors
                finally:
                    if observation is not None:
                        record["attention"] = observation.report()
            gc.collect()
            torch.cuda.empty_cache()


def _fixture_runs(root: Path, report: dict) -> None:
    from tests.test_p3_deepseek_quality_model import _checkpoint
    from tests.test_p3_deepseek_quality_train import study_pack

    from . import p3_deepseek_quality_model as model_api
    from . import p3_deepseek_quality_timing as timing
    from . import p3_deepseek_quality_train as train

    fixture_root = _child(root, "fixtures")
    fixture_root.mkdir(exist_ok=False)
    with _attempt(root, report, "fixture-preparation") as record:
        pack, pack_hash = study_pack.__wrapped__(_Directories(fixture_root))
        source = _checkpoint(fixture_root, exact_mla=True)
        cache = _child(fixture_root, "cache")
        initial = _child(fixture_root, "initial.safetensors")
        with model_api.open_session(source, cache, device="cuda", prepare=True) as session:
            model_api.write_initial_adapter(session.model, initial)
        initial_hash = model_api.sha256(initial)
        record.update(pack_sha256=pack_hash, initial_adapter_sha256=initial_hash,
                      exact_mla=True, shared_initial=True)
        report["fixture"] = {
            "pack_sha256": pack_hash, "initial_adapter_sha256": initial_hash,
            "exact_mla": True, "test_fixture": True,
        }
    training = _child(root, "training")
    training.mkdir(exist_ok=False)
    for length in _LENGTHS:
        for arm in train.ARMS:
            name = f"{arm}-L{length}"
            output = _child(training, name)
            with _attempt(root, report, name) as record:
                record.update(arm=arm, length=length, steps=3, micro_batch_size=1,
                              k=4096 // length, initial_adapter_sha256=initial_hash)
                try:
                    manifest = train.run_training(
                        source=source, cache=cache, pack=pack, pack_sha256=pack_hash,
                        initial_adapter=initial, arm=arm, length=length, steps=3,
                        output=output, device="cuda", test_fixture=True,
                    )
                    if manifest.get("status") != "COMPLETE" or manifest.get("completed_steps") != 3:
                        raise RuntimeError("fixture training did not complete all three steps")
                    if model_api.sha256(initial) != initial_hash:
                        raise ValueError("shared initial adapter changed during fixture training")
                    record["completed_steps"] = manifest["completed_steps"]
                finally:
                    if (output / "manifest.json").is_file():
                        manifest_path = output / "manifest.json"
                        record["training_manifest"] = manifest_path.relative_to(root).as_posix()
                        record["training_manifest_sha256"] = model_api.sha256(manifest_path)

    from transformers import AutoTokenizer

    with _attempt(root, report, "resource-only-forward") as record:
        tokenizer = AutoTokenizer.from_pretrained(
            pack.parent / "train-fixture", local_files_only=True)
        observations: list[dict] = []
        record["observations"] = observations
        record["quality_evaluated"] = False
        record["resource_admission_sufficient"] = False
        with model_api.open_session(source, cache, device="cuda") as session:
            model_api.load_initial_adapter(session.model, initial)
            timing.time_inputs(
                session, tokenizer, [{"input_ids": [3, 4, 5, 3], "positions": [1, 2, 3]}],
                max_tokens_per_batch=4, deadline=time.monotonic() + 120,
                test_fixture=True, observations=observations,
                evidence_prefix=_child(root / "receipts", "resource-forward"),
            )
        if model_api.sha256(initial) != initial_hash:
            raise ValueError("shared initial adapter changed during resource-only forward")


def run_smoke(*, output: Path | str) -> dict[str, Any]:
    """Execute one fixed fixture attempt; failure commits evidence then raises."""
    root = _new_output(output)
    report: dict[str, Any] = {
        "schema": "p3-deepseek-quality-smoke-v1", "status": "FAILED", "scope": _SCOPE,
        "scientific_admission": False, "quality_evaluated": False, "test_fixture": True,
        "device": "cuda", "dtype": "bfloat16", "seed": 23, "attempts": [],
        "training_lengths": list(_LENGTHS), "steps_per_arm_and_length": 3,
        "first_scheduler_learning_rate": 0, "retry_allowed": False,
        "deadline": "Caller must use the process-only stage helper; no in-process hard deadline.",
    }
    started = time.monotonic()
    stage = "native_admission"
    failure = None
    try:
        (root / "driver.py.txt").write_bytes(Path(__file__).read_bytes())
        (root / "receipts").mkdir(exist_ok=False)
        _write_json(root / "RUNNING.json", report)
        from . import p3_deepseek_quality_attention as attention
        from . import p3_deepseek_quality_train as train

        # This context establishes CUBLAS configuration before any CUDA initialization.
        with train._deterministic_run():
            report["hardware"] = attention.require_ampere_cuda("cuda")
            report["attention_policy"] = attention.policy_identity()
            stage = "dependency_identity"
            repository = Path(__file__).resolve().parents[2]
            dependencies = (*_ARCHIVES,
                            "benchmarks/harness/p3_deepseek_quality_attention.py",
                            "benchmarks/harness/p3_deepseek_quality_model.py",
                            "benchmarks/harness/p3_deepseek_quality_train.py",
                            "benchmarks/harness/p3_deepseek_quality_timing.py",
                            "benchmarks/harness/p3_deepseek_quality_data.py",
                            "benchmarks/harness/b13_moe_fixture.py",
                            "benchmarks/harness/p3_fp8_smoke.py",
                            "tests/test_p3_deepseek_quality_model.py",
                            "tests/test_p3_deepseek_quality_train.py",
                            "tests/test_p3_deepseek_quality_data.py")
            report["source_dependencies_sha256"] = {
                name: _sha256(repository / name) for name in dependencies
            }
            stage = "attention_probe"
            _attention_cases(root, report)
            stage = "native_fixture"
            _fixture_runs(root, report)
        report["status"] = "COMPLETE"
    except BaseException as error:
        failure = error
        report.update(status="FAILED", failed_stage=stage, error=_error(error))
        _write_json(root / "FAILED.json", {
            "scope": _SCOPE, "scientific_admission": False, "retry_allowed": False,
            "failed_stage": stage, "error": _error(error),
        })
        # Do not retain native tensors through exception tracebacks after session cleanup.
        traceback.clear_frames(error.__traceback__)
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["files"] = {
            path.relative_to(root).as_posix(): _sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file() and path != root / "manifest.json"
        }
        _write_json(root / "manifest.json", report)
        Console().print_json(json.dumps(report, allow_nan=False))
    if failure is not None:
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise failure
        raise RuntimeError(f"smoke FAILED; evidence preserved at {root}") from failure
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Fresh evidence directory; never reused")
    arguments = parser.parse_args(argv)
    Console().print(_SCOPE)
    try:
        run_smoke(output=arguments.output)
    except KeyboardInterrupt:
        Console().print("Smoke interrupted; retain partial receipts. No retry or admission.")
        return 130
    except Exception as error:
        Console().print(f"Smoke refused/failed: {type(error).__name__}: {error}")
        return 1
    Console().print("Fixture smoke COMPLETE only; no scientific admission or quality claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
