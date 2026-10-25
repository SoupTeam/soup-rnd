"""Evidence controls shared by the four-arm expert-streaming GPU benchmark.

No GPU run is implied by a passing comparator.  In particular, input/adapters
must be identical before candidate trajectories are compared with the ordinary
streaming reference, and every candidate is compared with that reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

ARMS = ("ordinary", "two_read", "cache", "two_read_cache")
RTOL = 0.01
ATOL = 1e-5


def atomic_json(path: str | os.PathLike[str], data: Any) -> None:
    """Replace a JSON artifact only after its complete, finite payload is durable."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash an artifact without retaining its contents in host memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(
    paths: Iterable[str | os.PathLike[str]], root: str | os.PathLike[str],
) -> dict[str, str]:
    """Map contained files to hashes, refusing escape and duplicate aliases."""
    base = os.path.realpath(root)
    result: dict[str, str] = {}
    seen: set[str] = set()
    for item in paths:
        filename = os.fspath(item)
        if not os.path.isabs(filename):
            filename = os.path.join(base, filename)
        canonical = os.path.realpath(filename)
        try:
            contained = os.path.normcase(os.path.commonpath((base, canonical))) == (
                os.path.normcase(base)
            )
        except ValueError:
            contained = False
        if not contained:
            raise ValueError(f"Fingerprint path escapes root: {item}")
        identity = os.path.normcase(canonical)
        if identity in seen:
            raise ValueError(f"Duplicate fingerprint path: {item}")
        if not os.path.isfile(canonical):
            raise ValueError(f"Fingerprint path is not a file: {item}")
        seen.add(identity)
        key = Path(os.path.relpath(canonical, base)).as_posix()
        result[key] = sha256_file(canonical)
    return dict(sorted(result.items()))


def _report() -> dict[str, Any]:
    return {"passed": True, "bitwise": True, "checks": 0, "failures": []}


def _fail(report: dict[str, Any], path: tuple[Any, ...], reason: str) -> None:
    report["passed"] = False
    report["bitwise"] = False
    report["failures"].append({"path": list(path), "reason": reason})


def _tensor(
    value: Any, report: dict[str, Any], path: tuple[Any, ...],
) -> bool:
    import torch

    if not isinstance(value, torch.Tensor):
        _fail(report, path, "expected tensor")
        return False
    if value.layout != torch.strided or value.numel() == 0:
        _fail(report, path, "expected nonempty dense tensor")
        return False
    if not torch.isfinite(value).all().item():
        _fail(report, path, "nonfinite tensor")
        return False
    return True


def validate_capture(capture: Any) -> dict[str, Any]:
    """Reject incomplete trajectories before their numbers can grant a timing gate.

    Expected keys are ``initial_adapters``, ``microbatches`` and ``updates``.
    Each microbatch contains inputs, loss/logits, all accumulated adapter
    gradients, decoder-input gradients, router logits and routing weights.
    Each update contains all adapters and AdamW state per adapter.  Diagnostic
    tensors can have arbitrary string keys but cannot silently disappear.
    """
    import torch

    report = _report()
    if not isinstance(capture, dict) or set(capture) != {
        "initial_adapters", "microbatches", "updates",
    }:
        _fail(report, (), "capture keys must be initial_adapters, microbatches, updates")
        return report
    initial = capture["initial_adapters"]
    if not isinstance(initial, dict) or not initial:
        _fail(report, ("initial_adapters",), "missing initial adapters")
        return report
    if not all(isinstance(name, str) for name in initial):
        _fail(report, ("initial_adapters",), "adapter names must be strings")
        return report
    initial_valid = {
        name: _tensor(value, report, ("initial_adapters", name))
        for name, value in initial.items()
    }
    batches = capture["microbatches"]
    updates = capture["updates"]
    if not isinstance(batches, list) or not batches:
        _fail(report, ("microbatches",), "missing microbatches")
        return report
    if not isinstance(updates, list) or not updates:
        _fail(report, ("updates",), "missing optimizer updates")
        return report
    if len(batches) % len(updates):
        _fail(report, ("microbatches",), "microbatches do not divide equally into updates")
    required_batch = {
        "input_ids", "loss", "logits", "gradients", "layer_inputs",
        "router_logits", "routing_weights",
    }
    for index, batch in enumerate(batches):
        path = ("microbatches", index)
        if not isinstance(batch, dict) or set(batch) != required_batch:
            _fail(report, path, "incomplete microbatch fields")
            continue
        for field in ("input_ids", "loss", "logits"):
            value = batch[field]
            if _tensor(value, report, (*path, field)):
                report["checks"] += 1
                if field == "loss" and value.ndim != 0:
                    _fail(report, (*path, field), "loss must be scalar")
                if field == "input_ids" and value.dtype not in (torch.int32, torch.int64):
                    _fail(report, (*path, field), "input IDs must be integers")
        for field in ("gradients", "layer_inputs", "router_logits", "routing_weights"):
            values = batch[field]
            if not isinstance(values, dict) or not values:
                _fail(report, (*path, field), "missing gradient or diagnostic tensors")
                continue
            if not all(isinstance(name, str) for name in values):
                _fail(report, (*path, field), "diagnostic names must be strings")
            if field == "gradients" and values.keys() != initial.keys():
                _fail(report, (*path, field), "adapter gradient keys differ from initial state")
            for name, value in values.items():
                if field in {"router_logits", "routing_weights"}:
                    required = {"values", "gradients"}
                    if field == "routing_weights":
                        required.add("indices")
                    if not isinstance(value, dict) or set(value) != required:
                        _fail(report, (*path, field, name), "router values and gradients required")
                        continue
                    valid = {}
                    for key, item in value.items():
                        valid[key] = _tensor(item, report, (*path, field, name, key))
                        report["checks"] += int(valid[key])
                    if all(valid.values()) and (
                        value["values"].shape != value["gradients"].shape
                        or value["values"].dtype != value["gradients"].dtype
                    ):
                        _fail(report, (*path, field, name), "router value/gradient shape or dtype")
                    if field == "routing_weights" and valid["indices"] and valid["values"] and (
                        value["indices"].dtype != torch.int64
                        or value["indices"].shape != value["values"].shape
                    ):
                        _fail(report, (*path, field, name), "router indices shape or dtype")
                elif _tensor(value, report, (*path, field, name)):
                    report["checks"] += 1
        if isinstance(batch["router_logits"], dict) and isinstance(batch["routing_weights"], dict):
            if batch["router_logits"].keys() != batch["routing_weights"].keys():
                _fail(report, path, "router logits and weights have different layer keys")
    final: dict[str, Any] | None = None
    for index, update in enumerate(updates):
        path = ("updates", index)
        if not isinstance(update, dict) or set(update) != {"adapters", "optimizer"}:
            _fail(report, path, "incomplete update fields")
            continue
        adapters = update["adapters"]
        optimizer = update["optimizer"]
        if not isinstance(adapters, dict) or adapters.keys() != initial.keys():
            _fail(report, (*path, "adapters"), "adapter keys differ from initial state")
            continue
        if not isinstance(optimizer, dict) or optimizer.keys() != initial.keys():
            _fail(report, (*path, "optimizer"), "optimizer keys differ from adapter state")
            continue
        final = adapters
        for name, value in adapters.items():
            if _tensor(value, report, (*path, "adapters", name)):
                report["checks"] += 1
                if initial_valid[name] and (
                    value.shape != initial[name].shape or value.dtype != initial[name].dtype
                ):
                    _fail(report, (*path, "adapters", name), "adapter shape or dtype changed")
            state = optimizer[name]
            if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
                _fail(report, (*path, "optimizer", name), "missing AdamW optimizer state")
                continue
            for key, item in state.items():
                if not isinstance(key, str):
                    _fail(report, (*path, "optimizer", name), "optimizer names must be strings")
                if not _tensor(item, report, (*path, "optimizer", name, key)):
                    continue
                report["checks"] += 1
                if key == "step" and (item.numel() != 1 or item.item() != index + 1):
                    _fail(report, (*path, "optimizer", name, key), "unexpected optimizer step")
                if key in {"exp_avg", "exp_avg_sq"} and item.shape != initial[name].shape:
                    _fail(
                        report, (*path, "optimizer", name, key), "optimizer moment shape mismatch",
                    )
    if final is not None:
        for name, reference in initial.items():
            value = final[name]
            if initial_valid[name] and isinstance(value, torch.Tensor):
                if value.shape == reference.shape and torch.equal(value, reference):
                    _fail(report, ("updates", "final", "adapters", name), "adapter did not change")
    return report


def compare_captures(reference: Any, candidate: Any) -> dict[str, Any]:
    """Apply fixed pointwise gates to a candidate against the ordinary reference.

    ``rtol`` scales the reference, i.e. torch's SECOND argument. Initial adapters
    and input IDs must match exactly; numerical CUDA results need not be bitwise.
    A passing return value is conditional on complete ``validate_capture`` reports.
    """
    import torch

    report = _report()
    for label, value in (("reference", reference), ("candidate", candidate)):
        validation = validate_capture(value)
        if not validation["passed"]:
            report["passed"] = report["bitwise"] = False
            report["failures"].extend(
                {**failure, "path": [label, *failure["path"]]}
                for failure in validation["failures"]
            )
    if not report["passed"]:
        return report

    def visit(expected: Any, actual: Any, path: tuple[Any, ...]) -> None:
        if isinstance(expected, torch.Tensor):
            report["checks"] += 1
            if not isinstance(actual, torch.Tensor):
                _fail(report, path, "expected tensor")
                return
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                _fail(report, path, "tensor shape or dtype mismatch")
                return
            if not torch.isfinite(expected).all().item() or not torch.isfinite(actual).all().item():
                _fail(report, path, "nonfinite tensor")
                return
            exact = torch.equal(actual, expected)
            report["bitwise"] = report["bitwise"] and exact
            if (path[0] == "initial_adapters" or "input_ids" in path
                    or not (expected.is_floating_point() or expected.is_complex())):
                if not exact:
                    _fail(report, path, "initial adapters, inputs and integers must match bitwise")
                return
            if not torch.allclose(actual, expected, rtol=RTOL, atol=ATOL):
                _fail(report, path, "reference-relative pointwise tolerance exceeded")
            if path[-1] == "loss" and f"{actual.item():.3f}" != f"{expected.item():.3f}":
                _fail(report, path, "loss differs at three decimal places")
        elif isinstance(expected, dict):
            if not isinstance(actual, dict) or actual.keys() != expected.keys():
                _fail(report, path, "mapping keys mismatch")
                return
            for key, value in expected.items():
                visit(value, actual[key], (*path, key))
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                _fail(report, path, "sequence length or type mismatch")
                return
            for index, value in enumerate(expected):
                visit(value, actual[index], (*path, index))
        else:
            report["checks"] += 1
            if isinstance(expected, float) and not math.isfinite(expected):
                _fail(report, path, "nonfinite scalar")
            elif type(actual) is not type(expected) or actual != expected:
                _fail(report, path, "scalar value or type mismatch")

    visit(reference, candidate, ())
    return report
