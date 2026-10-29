#!/usr/bin/env python3
"""Read-only, CPU analysis of ``a3_1b_diagnose.py`` artifacts.

Usage::

    python benchmarks/harness/a3_1b_analyze.py RUN_DIR --out NEW_REPORT.json
    python benchmarks/harness/a3_1b_analyze.py RUN_DIR --out NEW_REPORT.json \
        --baseline PINNED_CHECKOUT

The report is created exclusively; existing files are never replaced. Input arm
records, gradients and traces are never written. Each trace row identifies one
bare CPU tensor with ``file``. Float64 arithmetic is chunked, with at most one
trace tensor from each arm being compared. No CUDA operation, model import, tolerance adjustment or
statistical inference of a cause is performed.

``--baseline`` optionally invokes the *unchanged* pinned stand's ``decide``.
That statistical comparison is separate from causal localization and is not a
rerun of the A3 spill gate. CPU fixtures are explicitly rehearsal evidence, not
a historical CUDA/BF16 criterion-1b verdict. Missing or incomplete trace evidence
can never produce a causal PASS. Adaptive ``--pair NAME:LEFT:RIGHT`` comparisons
are observations only: a human-reviewed single-factor experiment is still
required for a causal conclusion.

The JSON includes all raw metrics and a concise ``text_summary``. Exit 0 means
a report was written, not that parity or causality passed. Exit 2 means the CLI
or report creation was refused. Missing evidence never counts as equality.
The per-key trace-file schema is intentionally the runner's only supported layout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from rich.console import Console

BASELINE_COMMIT = "ec795ef175c222b60ff91b9ba44f7f2bfad55dd5"
SCHEMA = "a3-1b-analysis-v1"
INITIAL_PHASES = {"gradforward", "nogradforward"}
STAGES = {"embed", "decoder", "norm", "head", "ce_sum"}
PAIRS = (
    ("native_observer_control", "ga_native_1step", "ga_native_trace_1step", "control"),
    ("padded_observer_control", "ga_padded_1step", "ga_padded_trace_1step", "control"),
    ("l2l_observer_control", "l2l_3step", "l2l_trace_1step", "first_window_control"),
    ("shape_mask_intervention", "ga_native_trace_1step", "ga_padded_trace_1step", "trace"),
    ("residual_execution_path", "ga_padded_trace_1step", "l2l_trace_1step", "trace"),
    ("ga_repeat", "ga_3step", "ga_repeat_3step", "control"),
    ("original_first_window_l2l_vs_ga", "ga_3step", "l2l_3step", "first_window_gradients"),
    ("original_l2l_vs_ga", "ga_3step", "l2l_3step", "gradients"),
    ("ga_floor", "ga_3step", "ga_floor_3step", "gradients"),
)
TRACE_FIELDS = (
    "key", "stage", "layer", "phase", "micro_batch", "occurrence", "shape", "dtype",
    "grad_enabled", "autocast_enabled", "valid_length", "window", "row_id",
    "valid_positions", "slicing", "sha256", "file",
)
WINDOW_FIELDS = (
    "window", "row_ids", "row_sha256", "valid_lengths", "supervised_tokens", "label_tokens",
    "n_items", "n_items_definition",
)
INPUT_FIELDS = (
    "row_id", "input_ids_sha256", "labels_sha256", "attention_mask_sha256",
    "position_ids_sha256", "mask_policy", "position_policy", "input_shape", "label_shape",
    "input_dtype", "label_dtype",
)
CONTROLS = ("native_observer_control", "padded_observer_control", "l2l_observer_control")
CONTRASTS = ("shape_mask_intervention", "residual_execution_path")


class ArtifactError(ValueError):
    """An artifact is missing, unsafe to address, or structurally invalid."""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def contained_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ArtifactError(f"not an arm-relative path: {relative!r}")
    base, target = os.path.realpath(root), os.path.realpath(root / relative)
    try:
        inside = os.path.commonpath((base, target)) == base
    except ValueError:
        inside = False
    if not inside:
        raise ArtifactError(f"artifact path escapes its arm: {relative!r}")
    return Path(target)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_json(left: Any, right: Any) -> bool:
    """Preserve float signs and JSON number types, unlike Python's == on numbers."""
    return json.dumps(safe_numbers(left), sort_keys=True, allow_nan=False) == json.dumps(
        safe_numbers(right), sort_keys=True, allow_nan=False
    )


def safe_numbers(value: Any) -> Any:
    """Keep nonfinite measurements explicit without emitting nonstandard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: safe_numbers(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_numbers(item) for item in value]
    return value


def dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def load_cpu(path: Path) -> Any:
    import torch

    # No unsafe-pickle or non-mmap fallback: these artifacts are new-format torch.save files.
    try:
        return torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    except (
        OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError
    ) as error:
        raise ArtifactError(f"cannot safely load CPU artifact {path}: {error}") from error


def check_tensor(tensor: Any) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ArtifactError("saved value is not a tensor")
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise ArtifactError("saved tensor is not a dense CPU tensor")
    if not tensor.is_contiguous() or tensor.is_complex() or tensor.is_quantized:
        raise ArtifactError("saved tensor must be contiguous, real and unquantized")


def descriptor_problem(descriptor: Any) -> str | None:
    if not isinstance(descriptor, dict):
        return "descriptor is not an object"
    if not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("sha256", ""))):
        return "missing or malformed raw SHA-256"
    if not isinstance(descriptor.get("dtype"), str):
        return "missing dtype"
    shape = descriptor.get("shape")
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        return "missing or invalid shape"
    return None


def tensor_description(tensor: Any, chunk_elements: int) -> dict[str, Any]:
    import torch

    check_tensor(tensor)
    flat = tensor.detach().view(-1)
    digest = hashlib.sha256()
    nonfinite = {"nan": 0, "positive_infinity": 0, "negative_infinity": 0}
    nonzero = 0
    squared_norm = 0.0
    max_abs = 0.0
    for start in range(0, flat.numel(), chunk_elements):
        part = flat[start:start + chunk_elements]
        digest.update(part.view(torch.uint8).numpy().tobytes())
        nonfinite["nan"] += int(torch.isnan(part).sum())
        nonfinite["positive_infinity"] += int(torch.isposinf(part).sum())
        nonfinite["negative_infinity"] += int(torch.isneginf(part).sum())
        nonzero += int(torch.count_nonzero(part))
        numeric = part.to(torch.float64)
        squared_norm += float(numeric.square().sum())
        part_max = float(numeric.abs().max()) if numeric.numel() else 0.0
        max_abs = (
            math.nan if math.isnan(part_max) or math.isnan(max_abs) else max(max_abs, part_max)
        )
    return {
        "sha256": digest.hexdigest(), "dtype": dtype_name(tensor),
        "shape": list(tensor.shape), "finite": not any(nonfinite.values()),
        "nonzero": bool(nonzero), "nonzero_elements": nonzero, "nonfinite": nonfinite,
        "numel": flat.numel(), "norm": math.sqrt(squared_norm),
        "squared_norm": squared_norm, "max_abs": max_abs,
    }


def compare_descriptor(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    problems = []
    for name in ("sha256", "dtype", "shape", "finite", "nonzero"):
        if name in expected and expected[name] != actual[name]:
            problems.append(f"{name} differs from recorded metadata")
    return problems


def relative_l2(diff_squared: float, reference_squared: float) -> float:
    if math.isnan(diff_squared) or math.isnan(reference_squared):
        return math.nan
    if diff_squared == 0:
        return 0.0
    if reference_squared == 0:
        return math.inf
    return math.sqrt(diff_squared / reference_squared)


def tensor_metrics(left: Any, right: Any, chunk_elements: int) -> dict[str, Any]:
    """Raw bytes plus chunked float64 metrics; no inferred common gradient scale."""
    import torch

    check_tensor(left)
    check_tensor(right)
    out: dict[str, Any] = {
        "left_shape": list(left.shape), "right_shape": list(right.shape),
        "left_dtype": dtype_name(left), "right_dtype": dtype_name(right),
        "comparable": left.shape == right.shape and left.dtype == right.dtype,
        "arithmetic": "float64, chunked", "chunk_elements": chunk_elements,
    }
    if not out["comparable"]:
        out.update(bitmatch=False, reason="shape or dtype mismatch")
        return out
    ref, cand = left.detach().view(-1), right.detach().view(-1)
    ref_sq = cand_sq = gap_sq = dot = 0.0
    ref_nonfinite = cand_nonfinite = 0
    max_abs = 0.0
    first = None
    bitmatch = True
    for start in range(0, ref.numel(), chunk_elements):
        ref_raw, cand_raw = ref[start:start + chunk_elements], cand[start:start + chunk_elements]
        different = ref_raw.view(torch.uint8) != cand_raw.view(torch.uint8)
        if bool(different.any()):
            bitmatch = False
            if first is None:
                first_byte = int(different.to(torch.uint8).argmax())
                first = start + first_byte // ref.element_size()
        a, b = ref_raw.to(torch.float64), cand_raw.to(torch.float64)
        gap = b - a
        ref_nonfinite += int((~torch.isfinite(a)).sum())
        cand_nonfinite += int((~torch.isfinite(b)).sum())
        ref_sq += float(a.square().sum())
        cand_sq += float(b.square().sum())
        gap_sq += float(gap.square().sum())
        dot += float((a * b).sum())
        part_max = float(gap.abs().max()) if gap.numel() else 0.0
        max_abs = (
            math.nan if math.isnan(part_max) or math.isnan(max_abs) else max(max_abs, part_max)
        )
    out.update(
        bitmatch=bitmatch, numel=ref.numel(), max_abs=max_abs,
        relative_l2=relative_l2(gap_sq, ref_sq),
        reference_norm=math.sqrt(ref_sq), candidate_norm=math.sqrt(cand_sq),
        difference_norm=math.sqrt(gap_sq), reference_squared_norm=ref_sq,
        candidate_squared_norm=cand_sq, difference_squared_norm=gap_sq, dot_product=dot,
        reference_nonfinite=ref_nonfinite, candidate_nonfinite=cand_nonfinite,
        finite=not (ref_nonfinite or cand_nonfinite), first_different_flat_index=first,
    )
    if first is not None:
        coordinate, remainder = [], first
        for dimension in reversed(left.shape):
            coordinate.append(remainder % dimension)
            remainder //= dimension
        out["first_different_coordinate"] = list(reversed(coordinate))
    return out


@dataclass
class Arm:
    name: str
    directory: Path
    record: dict[str, Any] = field(default_factory=dict)
    source_record: Any = None
    problems: list[str] = field(default_factory=list)
    trace_problems: list[str] = field(default_factory=list)
    trace_rows: list[dict[str, Any]] = field(default_factory=list)
    gradient_descriptions: list[dict[str, Any]] = field(default_factory=list)
    trace_descriptions: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def gradient_path(self, step: int) -> Path:
        return self.directory / "tensors" / f"step_{step:04d}.pt"

    def gradients(self, step: int) -> dict[str, Any]:
        value = load_cpu(self.gradient_path(step))
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ArtifactError("gradient file is not a parameter-name mapping")
        return value

    def trace(self, row: dict[str, Any]) -> Any:
        return load_cpu(contained_path(self.directory, row["file"]))

    def summary(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory), "valid": not self.problems,
            "problems": self.problems, "trace_problems": self.trace_problems,
            "record": self.record, "artifacts": self.artifacts,
            "source_record": self.source_record,
            "gradients": self.gradient_descriptions, "traces": self.trace_rows,
            "trace_tensors": self.trace_descriptions,
        }


def validate_effective_inputs(arm: Arm, chunk_elements: int) -> None:
    """Verify saved post-padding tensors and original, valid-token row identities."""
    import torch

    fingerprint = arm.record.get("rows") or {}
    for window in arm.record.get("windows", []):
        if not isinstance(window, dict):
            continue
        inputs = window.get("effective_inputs")
        row_ids = window.get("row_ids", [])
        if not isinstance(inputs, list) or len(inputs) != len(row_ids):
            arm.problems.append(f"window {window.get('window')}: missing effective_inputs")
            continue
        for micro_batch, item in enumerate(inputs):
            prefix = f"window {window.get('window')} mb {micro_batch}"
            try:
                if not isinstance(item, dict) or any(key not in item for key in INPUT_FIELDS):
                    raise ArtifactError("missing post-padding input/mask/position metadata")
                if item["row_id"] != row_ids[micro_batch]:
                    raise ArtifactError("effective row_id differs from fetched row_id")
                path = contained_path(arm.directory, item["file"])
                tensors = load_cpu(path)
                if not isinstance(tensors, dict):
                    raise ArtifactError("effective-input file is not a tensor mapping")
                descriptions = {}
                for name, hash_field in (
                    ("input_ids", "input_ids_sha256"), ("labels", "labels_sha256"),
                    ("attention_mask", "attention_mask_sha256"),
                    ("position_ids", "position_ids_sha256"),
                ):
                    tensor = tensors.get(name)
                    if tensor is None:
                        if name in {"input_ids", "labels"} or item[hash_field] is not None:
                            raise ArtifactError(f"missing {name} tensor despite declared hash")
                        descriptions[name] = None
                        continue
                    actual = tensor_description(tensor, chunk_elements)
                    descriptions[name] = actual
                    if actual["sha256"] != item[hash_field]:
                        raise ArtifactError(f"{name} raw hash differs from metadata")
                ids, labels = tensors["input_ids"], tensors["labels"]
                if ids.ndim != 2 or ids.shape[0] != 1 or labels.shape != ids.shape:
                    raise ArtifactError("expected one-row, same-shaped IDs and labels")
                for name, tensor, shape_field, dtype_field in (
                    ("input_ids", ids, "input_shape", "input_dtype"),
                    ("labels", labels, "label_shape", "label_dtype"),
                ):
                    if (
                        list(tensor.shape) != item[shape_field]
                        or dtype_name(tensor) != item[dtype_field]
                    ):
                        raise ArtifactError(f"{name} saved shape/dtype differs from metadata")
                    if tensor.dtype not in (torch.int32, torch.int64):
                        raise ArtifactError(f"{name} must have integer token dtype")
                valid_length = window["valid_lengths"][micro_batch]
                if type(valid_length) is not int or not 1 <= valid_length <= ids.shape[1]:
                    raise ArtifactError("invalid original valid length")
                if bool((labels[:, valid_length:] != -100).any()):
                    raise ArtifactError("right-padding labels contain supervised targets")
                row_hash = hashlib.sha256(
                    ids[0, :valid_length].numpy().tobytes() + b"|"
                    + labels[0, :valid_length].numpy().tobytes()
                ).hexdigest()
                if row_hash != window["row_sha256"][micro_batch]:
                    raise ArtifactError(
                        "valid-token IDs/labels do not match the consumed row fingerprint"
                    )
                row_id = row_ids[micro_batch]
                if type(row_id) is not int or not (
                    0 <= row_id < len(fingerprint.get("row_sha256", []))
                ):
                    raise ArtifactError("row_id cannot index pinned dataset fingerprint")
                if row_hash != fingerprint["row_sha256"][row_id]:
                    raise ArtifactError("consumed row fingerprint differs from pinned dataset row")
                label_count = int((labels != -100).sum())
                causal_positions = (labels[0, 1:] != -100).nonzero().view(-1).tolist()
                if window["supervised_tokens"][micro_batch] != len(causal_positions):
                    raise ArtifactError(
                        "recorded supervised_tokens differs from shifted actual labels"
                    )
                if window["label_tokens"][micro_batch] != label_count:
                    raise ArtifactError(
                        "recorded label_tokens differs from unshifted actual labels"
                    )
                item["verified"] = {
                    "tensors": descriptions, "row_sha256": row_hash,
                    "label_tokens": label_count, "causal_supervised_tokens": len(causal_positions),
                    "causal_positions": causal_positions,
                }
                del tensors, ids, labels
            except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
                arm.problems.append(f"{prefix}: {error}")
        definition = window.get("n_items_definition")
        count_field = {
            "causal_supervised_tokens": "supervised_tokens", "label_tokens": "label_tokens"
        }
        if definition not in count_field:
            arm.problems.append(f"window {window.get('window')}: unknown original N definition")
        else:
            counts = window.get(count_field[definition])
            if not isinstance(counts, list) or any(
                type(count) is not int or count < 0 for count in counts
            ):
                arm.problems.append(f"window {window.get('window')}: invalid denominator counts")
            elif sum(counts) != window.get("n_items"):
                arm.problems.append(
                    f"window {window.get('window')}: original N differs from saved label counts"
                )


def validate_trace_context(arm: Arm, row: dict[str, Any]) -> None:
    """Reject a self-consistent trace crop if it describes the wrong actual row."""
    windows = [window for window in arm.record.get("windows", [])
               if isinstance(window, dict) and window.get("window") == row["window"]]
    if len(windows) != 1:
        raise ArtifactError("trace window does not identify one actual fetched window")
    window, micro_batch = windows[0], row["micro_batch"]
    if micro_batch >= len(window["row_ids"]) or window["row_ids"][micro_batch] != row["row_id"]:
        raise ArtifactError("trace microbatch/row_id differs from actual fetched window")
    if window["valid_lengths"][micro_batch] != row["valid_length"]:
        raise ArtifactError("trace valid_length differs from original row")
    effective = window.get("effective_inputs", [])[micro_batch]
    verified = effective.get("verified")
    if not verified:
        raise ArtifactError("trace has no hash-validated effective inputs")
    if row["slicing"] not in {"valid_tokens", "causal_targets", "ce_sum"}:
        raise ArtifactError("unknown valid-token slicing convention")
    if row["phase"] != "adjoint":
        expected_slicing = "ce_sum" if row["stage"] == "ce_sum" else (
            "causal_targets" if row["stage"] == "head" else "valid_tokens"
        )
        if row["slicing"] != expected_slicing:
            raise ArtifactError("stage has an incompatible valid-token slicing convention")
    if row["slicing"] in {"causal_targets", "ce_sum"}:
        expected_positions = verified["causal_positions"]
        if row["valid_positions"] != expected_positions:
            raise ArtifactError("valid target positions differ from actual saved causal labels")
        if row["stage"] == "head" and (
            len(row["shape"]) != 2 or row["shape"][0] != len(expected_positions)
        ):
            raise ArtifactError("head trace is not all valid targets by vocabulary")
        if row["stage"] == "ce_sum" and row["shape"] != []:
            raise ArtifactError("CE numerator is not a scalar")
    else:
        if row["valid_positions"] != list(range(row["valid_length"])):
            raise ArtifactError("valid-token positions are not the complete original row")
        if len(row["shape"]) != 3 or row["shape"][:2] != [1, row["valid_length"]]:
            raise ArtifactError("hidden trace is not cropped to original valid tokens")
    if row["phase"] == "adjoint" and row.get("boundary") not in {"input", "output"}:
        raise ArtifactError("adjoint trace needs explicit input/output boundary")
    if row["phase"] == "adjoint" and row.get("source_phase") not in INITIAL_PHASES | {"recompute"}:
        raise ArtifactError("adjoint needs explicit initial/recompute source_phase")


def validate_windows(arm: Arm) -> None:
    record = arm.record
    if not isinstance(record.get("row_order"), list) or not record["row_order"]:
        arm.problems.append("missing actual consumed row_order")
    windows = record.get("windows")
    if not isinstance(windows, list) or not windows:
        arm.problems.append("missing fetched windows and supervised-token counts")
        return
    consumed = []
    for index, window in enumerate(windows):
        if not isinstance(window, dict) or any(key not in window for key in WINDOW_FIELDS):
            arm.problems.append(f"window {index}: missing required row/N metadata")
            continue
        row_ids = window["row_ids"]
        if not isinstance(row_ids, list) or not row_ids:
            arm.problems.append(f"window {index}: missing row_ids")
            continue
        consumed.extend(row_ids)
        for name in ("row_sha256", "valid_lengths", "supervised_tokens", "label_tokens"):
            value = window[name]
            if not isinstance(value, list) or len(value) != len(row_ids):
                arm.problems.append(f"window {index}: {name} length differs from row_ids")
        if not isinstance(window["n_items"], (int, float)) or window["n_items"] <= 0:
            arm.problems.append(f"window {index}: invalid original n_items")
    if consumed != record.get("row_order"):
        arm.problems.append("row_order is not the concatenation of actual fetched window row_ids")


def validate_trace_rows(arm: Arm, chunk_elements: int) -> None:
    path = arm.directory / "traces.json"
    if not path.is_file():
        arm.trace_problems.append("missing traces.json; initial-forward localization unavailable")
        return
    try:
        rows = read_json(path)
        if not isinstance(rows, list) or not rows:
            raise ArtifactError("traces.json is not a nonempty array")
        seen = set()
        for index, row in enumerate(rows):
            try:
                if not isinstance(row, dict) or any(key not in row for key in TRACE_FIELDS):
                    raise ArtifactError("missing required trace alignment metadata")
                if not isinstance(row["key"], str) or row["key"] in seen:
                    raise ArtifactError("duplicate or invalid trace key")
                seen.add(row["key"])
                if row["stage"] not in STAGES:
                    raise ArtifactError("unknown stage")
                if row["phase"] not in INITIAL_PHASES | {"capture", "recompute", "adjoint"}:
                    raise ArtifactError("unknown phase")
                if row["stage"] == "decoder" and type(row["layer"]) is not int:
                    raise ArtifactError("decoder layer is not an integer")
                if row["stage"] != "decoder" and row["layer"] is not None:
                    raise ArtifactError("nondecoder layer must be null")
                if type(row["row_id"]) is not int or row["row_id"] < 0:
                    raise ArtifactError("invalid actual dataset row_id")
                for name in ("window", "micro_batch", "occurrence", "valid_length"):
                    if type(row[name]) is not int or row[name] < 0:
                        raise ArtifactError(f"invalid {name}")
                for name in ("grad_enabled", "autocast_enabled"):
                    if type(row[name]) is not bool:
                        raise ArtifactError(f"{name} is not an observed boolean")
                if not isinstance(row["valid_positions"], list):
                    raise ArtifactError("valid_positions is not an array")
                problem = descriptor_problem(row)
                if problem:
                    raise ArtifactError(problem)
                validate_trace_context(arm, row)
                value = arm.trace(row)
                actual = tensor_description(value, chunk_elements)
                if row["stage"] == "ce_sum":
                    actual["scalar_value"] = float(value.item())
                del value
                arm.trace_descriptions[row["key"]] = actual
                problems = compare_descriptor(row, actual)
                if not actual["finite"]:
                    problems.append("nonfinite trace tensor")
                if problems:
                    raise ArtifactError("; ".join(problems))
                arm.trace_rows.append(row)
            except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
                arm.trace_problems.append(f"trace row {index}: {error}")
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        arm.trace_problems.append(str(error))


def validate_layer_calls(arm: Arm, chunk_elements: int) -> None:
    def descriptors(value: Any) -> Iterator[dict[str, Any]]:
        if isinstance(value, dict):
            if "tensor" in value:
                yield value
            else:
                for child in value.values():
                    yield from descriptors(child)
        elif isinstance(value, list):
            for child in value:
                yield from descriptors(child)

    for index, call in enumerate(arm.record.get("layer_calls", [])):
        try:
            for key in ("window", "micro_batch", "row_id", "phase", "args", "kwargs", "file"):
                if key not in call:
                    raise ArtifactError(f"captured call missing {key}")
            window = arm.record["windows"][call["window"]]
            if window["row_ids"][call["micro_batch"]] != call["row_id"]:
                raise ArtifactError("captured call does not identify the actual consumed row")
            expected = list(descriptors({"args": call["args"], "kwargs": call["kwargs"]}))
            tensors = load_cpu(contained_path(arm.directory, call["file"]))
            if not isinstance(tensors, dict) or (
                tensors.keys() != {item["tensor"] for item in expected}
            ):
                raise ArtifactError("captured kwargs tensor names do not match metadata")
            for item in expected:
                problem = descriptor_problem(item)
                if problem:
                    raise ArtifactError(problem)
                actual = tensor_description(tensors[item["tensor"]], chunk_elements)
                issues = compare_descriptor(item, actual)
                if issues:
                    raise ArtifactError("; ".join(issues))
            if call["file"] not in arm.artifacts:
                raise ArtifactError("captured-call file is not covered by artifact manifest")
            del tensors
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
            arm.trace_problems.append(f"layer call {index}: {error}")


def compare_layer_calls(left: Arm, right: Arm) -> dict[str, Any]:
    indices = []
    for arm in (left, right):
        indices.append({
            (call["window"], call["micro_batch"], call["row_id"]): call
            for call in arm.record.get("layer_calls", [])
            if isinstance(call, dict) and call.get("phase") in INITIAL_PHASES
            and all(key in call for key in ("window", "micro_batch", "row_id", "args", "kwargs"))
        })
    ref, cand = indices
    expected = {
        (window["window"], mb, row_id) for window in left.record.get("windows", [])
        for mb, row_id in enumerate(window.get("row_ids", []))
    }
    complete = bool(ref) and ref.keys() == cand.keys() == expected
    rows = [
        {"left": ref[key], "right": cand[key],
         "args_match": exact_json(ref[key]["args"], cand[key]["args"]),
         "kwargs_match": exact_json(ref[key]["kwargs"], cand[key]["kwargs"])}
        for key in sorted(ref.keys() & cand.keys())
    ]
    return {"complete": complete, "rows": rows,
            "observed_kwargs_match": complete and all(
                row["kwargs_match"] and row["args_match"] for row in rows
            ),
            "scope": "actual decoder0 call mask/position kwargs; "
            "not every decoder's internal operations"}


def compare_dispatch(left: Arm, right: Arm) -> dict[str, Any]:
    indices = []
    for arm in (left, right):
        counts: dict[tuple[Any, ...], int] = {}
        index = {}
        for row in arm.record.get("dispatch", []):
            if not isinstance(row, dict) or row.get("phase") not in INITIAL_PHASES:
                continue
            identity = tuple(row.get(key) for key in ("window", "micro_batch", "row_id", "module"))
            occurrence = counts.get(identity, 0)
            counts[identity] = occurrence + 1
            index[(*identity, occurrence)] = row
        indices.append(index)
    ref, cand = indices
    needed = any(arm.record.get("device") == "cuda" and arm.record.get("quant") == "nf4"
                 for arm in (left, right))
    complete = not needed or (bool(ref) and ref.keys() == cand.keys())
    rows = [{"left": ref[key], "right": cand[key],
             "dispatch_result_match": ref[key].get("result") == cand[key].get("result")}
            for key in sorted(ref.keys() & cand.keys(), key=str)]
    return {"complete": complete, "required_for_cuda_nf4": needed, "rows": rows,
            "missing_left": [list(key) for key in sorted(cand.keys() - ref.keys(), key=str)],
            "missing_right": [list(key) for key in sorted(ref.keys() - cand.keys(), key=str)],
            "interpretation": "actual initial-forward NF4 decisions only; "
            "aggregate error cannot identify dispatch as cause"}


def validate_training_returns(arm: Arm) -> None:
    expected = {
        (window["window"], mb, row_id): window
        for window in arm.record.get("windows", [])
        for mb, row_id in enumerate(window.get("row_ids", []))
    }
    seen = set()
    for index, row in enumerate(arm.record.get("training_step_losses", [])):
        try:
            context = row["window"], row["micro_batch"], row["row_id"]
            if context not in expected or context in seen:
                raise ArtifactError("unknown or duplicate consumed-row context")
            seen.add(context)
            if type(row.get("loss")) not in (int, float) or not math.isfinite(row["loss"]):
                raise ArtifactError("unrounded training-step return is not a finite scalar")
            window = expected[context]
            if row.get("n_items") != window["n_items"]:
                raise ArtifactError("training-step denominator differs from actual fetched N")
            if row.get("valid_length") != window["valid_lengths"][row["micro_batch"]]:
                raise ArtifactError("training-step valid length differs from consumed row")
        except (ValueError, TypeError, KeyError, IndexError) as error:
            arm.problems.append(f"training_step_losses {index}: {error}")
    if seen != expected.keys():
        arm.problems.append(
            "unrounded training_step_losses does not cover every consumed row exactly once"
        )


def load_arm(root: Path, name: str, chunk_elements: int) -> Arm:
    arm = Arm(name, root / "arms" / name)
    try:
        record = read_json(arm.directory / "arm.json")
        if not isinstance(record, dict):
            raise ArtifactError("arm.json is not an object")
        arm.source_record = read_json(arm.directory / "arm.json")
        lists = ("row_order", "windows", "logs", "rates", "applied_rates", "grads",
                 "training_step_losses")
        invalid_lists = [key for key in lists if not isinstance(record.get(key), list)]
        invalid_rows = [key for key in ("windows", "logs", "grads", "training_step_losses")
                        if isinstance(record.get(key), list)
                        and any(not isinstance(item, dict) for item in record[key])]
        if invalid_lists or invalid_rows:
            arm.problems.append(f"invalid list/object schema: {invalid_lists + invalid_rows}")
            return arm
        arm.record = record
        if record.get("exit") != "normal":
            arm.problems.append(f"arm did not exit normally: {record.get('exit')!r}")
        initial = record.get("initial_adapter_hashes")
        if not isinstance(initial, dict) or not initial:
            arm.problems.append("missing initial_adapter_hashes")
        else:
            for parameter, descriptor in initial.items():
                problem = descriptor_problem(descriptor)
                if problem or descriptor.get("finite") is not True:
                    arm.problems.append(f"initial {parameter}: {problem or 'not finite'}")
        if not isinstance(record.get("rows"), dict) or not record["rows"]:
            arm.problems.append("missing pinned row fingerprints")
        validate_windows(arm)
        validate_training_returns(arm)
        for name_field in ("logs", "rates", "grads"):
            if not isinstance(record.get(name_field), list) or not record[name_field]:
                arm.problems.append(f"missing nonempty {name_field}")
        steps = record.get("steps")
        if type(steps) is not int or steps < 1:
            arm.problems.append("missing positive optimizer-step count")
            steps = 0
        if len(record.get("grads") or []) != steps or len(record.get("rates") or []) != steps:
            arm.problems.append("gradient/rate counts do not equal optimizer steps")
        applied = record.get("applied_rates")
        if not isinstance(applied, list) or len(applied) != steps or any(
            not isinstance(rate, (int, float)) or not math.isfinite(rate) for rate in applied
        ):
            arm.problems.append("missing/invalid actually applied learning rates")
        expected_files = {f"step_{step:04d}.pt" for step in range(steps)}
        if {path.name for path in (arm.directory / "tensors").glob("step_*.pt")} != expected_files:
            arm.problems.append("saved gradient files do not exactly match optimizer-step indices")
        for index, log in enumerate(record.get("logs") or []):
            if not isinstance(log, dict) or not isinstance(log.get("loss"), (int, float)):
                arm.problems.append(f"log {index}: missing numeric loss")
            elif not math.isfinite(log["loss"]):
                arm.problems.append(f"log {index}: nonfinite loss")
        if any(not isinstance(rate, (int, float)) or not math.isfinite(rate)
               for rate in record.get("rates") or []):
            arm.problems.append("nonfinite or invalid learning rate")
        manifest = record.get("artifacts")
        if not isinstance(manifest, dict) or not manifest:
            arm.problems.append("missing artifact SHA-256 manifest")
        else:
            for relative, expected in manifest.items():
                try:
                    path = contained_path(arm.directory, relative)
                    actual = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                    arm.artifacts[relative] = actual
                    if not isinstance(expected, dict) or any(
                        expected.get(key) != actual[key] for key in ("sha256", "bytes")
                    ):
                        arm.problems.append(f"artifact {relative}: file digest/size mismatch")
                except (OSError, ValueError, TypeError) as error:
                    arm.problems.append(f"artifact {relative}: {error}")
        for step in range(steps):
            try:
                gradients = arm.gradients(step)
                expected = record["grads"][step]
                if (
                    not isinstance(expected, dict) or not gradients
                    or gradients.keys() != expected.keys()
                ):
                    raise ArtifactError("gradient parameter names differ from recorded metadata")
                if not isinstance(initial, dict) or gradients.keys() != initial.keys():
                    raise ArtifactError(
                        "gradient names do not cover the exact initial trainable adapter universe"
                    )
                descriptions = {}
                for parameter, tensor in gradients.items():
                    problem = descriptor_problem(expected[parameter])
                    if problem:
                        raise ArtifactError(f"{parameter}: {problem}")
                    actual = tensor_description(tensor, chunk_elements)
                    descriptions[parameter] = actual
                    issues = compare_descriptor(expected[parameter], actual)
                    if not actual["finite"]:
                        issues.append("nonfinite gradient")
                    arm.problems.extend(f"step {step} {parameter}: {issue}" for issue in issues)
                if not any(item["nonzero"] for item in descriptions.values()):
                    arm.problems.append(f"step {step}: all-zero gradients (vacuous)")
                arm.gradient_descriptions.append(descriptions)
                del gradients
            except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
                arm.problems.append(f"step {step}: {error}")
        validate_effective_inputs(arm, chunk_elements)
        validate_trace_rows(arm, chunk_elements)
        validate_layer_calls(arm, chunk_elements)
        required_artifacts = {f"tensors/step_{step:04d}.pt" for step in range(steps)}
        required_artifacts.update(
            item["file"] for window in record.get("windows", []) if isinstance(window, dict)
            for item in window.get("effective_inputs", [])
            if isinstance(item, dict) and "file" in item
        )
        if (arm.directory / "traces.json").is_file():
            required_artifacts.add("traces.json")
        required_artifacts.update(row["file"] for row in arm.trace_rows)
        for relative in sorted(required_artifacts - arm.artifacts.keys()):
            arm.problems.append(f"required artifact {relative} is not covered by the manifest")
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        arm.problems.append(str(error))
    return arm


def prerequisites(left: Arm, right: Arm, first_window: bool = False) -> dict[str, Any]:
    fields = ("initial_adapter_hashes", "rows") if first_window else (
        "initial_adapter_hashes", "rows", "row_order"
    )
    matches = {
        field_name: bool(left.record.get(field_name)) and exact_json(
            left.record.get(field_name), right.record.get(field_name)
        ) for field_name in fields
    }
    windows = []
    for arm in (left, right):
        actual_windows = arm.record.get("windows", [])
        if first_window:
            actual_windows = actual_windows[:1]
        windows.append([
            {key: window.get(key) for key in WINDOW_FIELDS}
            for window in actual_windows if isinstance(window, dict)
        ])
    matches["actual_window_rows_lengths_and_N"] = bool(windows[0]) and exact_json(*windows)
    problems = [f"{name}: {problem}" for name, arm in ((left.name, left), (right.name, right))
                for problem in arm.problems]
    matches["baseline_commit"] = all(
        arm.record.get("baseline_commit") == BASELINE_COMMIT for arm in (left, right)
    )
    matches["device_dtype_quant_layers"] = all(
        key in left.record and key in right.record
        and exact_json(left.record[key], right.record[key])
        for key in ("device", "dtype", "quant", "n_layers")
    )
    problems.extend(f"{name} differs or is missing" for name, match in matches.items() if not match)
    return {"valid": not problems, "matches": matches, "problems": problems}


def compare_gradients(left: Arm, right: Arm, chunk_elements: int,
                      first_window: bool = False) -> dict[str, Any]:
    rows = []
    count_left, count_right = left.record.get("steps", 0), right.record.get("steps", 0)
    if type(count_left) is not int or type(count_right) is not int:
        return {"bitmatch": False, "left_steps": 0, "right_steps": 0,
                "steps": [], "problem": "invalid optimizer-step count"}
    if first_window:
        count_left, count_right = min(count_left, 1), min(count_right, 1)
    all_equal = count_left == count_right and count_left > 0
    for step in range(min(count_left, count_right)):
        result: dict[str, Any] = {"step": step, "parameters": {}}
        try:
            ref, cand = left.gradients(step), right.gradients(step)
            missing_left = sorted(cand.keys() - ref.keys())
            missing_right = sorted(ref.keys() - cand.keys())
            result.update(missing_left=missing_left, missing_right=missing_right)
            equal = not (missing_left or missing_right)
            ref_sq = cand_sq = gap_sq = 0.0
            comparable = bool(ref) and bool(cand)
            for parameter in sorted(ref.keys() & cand.keys()):
                metrics = tensor_metrics(ref[parameter], cand[parameter], chunk_elements)
                result["parameters"][parameter] = metrics
                equal = equal and metrics["bitmatch"]
                comparable = comparable and metrics["comparable"] and metrics.get("finite", False)
                if metrics["comparable"]:
                    ref_sq += metrics["reference_squared_norm"]
                    cand_sq += metrics["candidate_squared_norm"]
                    gap_sq += metrics["difference_squared_norm"]
            result.update(
                bitmatch=equal, comparable=comparable and not missing_left and not missing_right
            )
            if result["comparable"]:
                result.update(
                    relative_l2=relative_l2(gap_sq, ref_sq), reference_norm=math.sqrt(ref_sq),
                    candidate_norm=math.sqrt(cand_sq), difference_norm=math.sqrt(gap_sq),
                    reference_squared_norm=ref_sq, candidate_squared_norm=cand_sq,
                    difference_squared_norm=gap_sq,
                )
            all_equal = all_equal and equal and result["comparable"]
            del ref, cand
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            result.update(bitmatch=False, comparable=False, problem=str(error))
            all_equal = False
        rows.append(result)
    return {"bitmatch": all_equal, "left_steps": count_left,
            "right_steps": count_right, "steps": rows}


def trace_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    # Phase is deliberately normalized ONLY for initial forward. Capture and backward
    # recompute never enter this index. Actual modes remain in every comparison row.
    return (
        row["window"], row["row_id"], row["micro_batch"], row["stage"], row["layer"],
        "initial_forward", row["occurrence"], row["valid_length"],
        json.dumps(row["valid_positions"], sort_keys=True),
        json.dumps(row["slicing"], sort_keys=True),
    )


def trace_order(row: dict[str, Any]) -> tuple[int, ...]:
    stage = {"embed": 0, "decoder": 1, "norm": 2, "head": 3, "ce_sum": 4}[row["stage"]]
    layer = row["layer"] if row["stage"] == "decoder" else -1
    return stage, layer, row["window"], row["micro_batch"], row["occurrence"]


def initial_trace_index(arm: Arm) -> tuple[dict[tuple[Any, ...], dict[str, Any]], list[str]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    problems = list(arm.trace_problems)
    for row in arm.trace_rows:
        if row["phase"] not in INITIAL_PHASES:
            continue
        identity = trace_identity(row)
        if identity in index:
            problems.append(f"duplicate initial-forward identity: {row['key']}")
        index[identity] = row
    n_layers = arm.record.get("n_layers")
    if type(n_layers) is not int or n_layers < 1:
        problems.append("missing positive n_layers; full decoder coverage cannot be certified")
    else:
        expected_stages = {("embed", None), ("norm", None), ("head", None), ("ce_sum", None)}
        expected_stages.update(("decoder", layer) for layer in range(n_layers))
        for window in arm.record.get("windows", []):
            if not isinstance(window, dict) or not isinstance(window.get("row_ids"), list):
                continue
            for micro_batch, row_id in enumerate(window["row_ids"]):
                rows = [row for row in index.values() if row["window"] == window.get("window")
                        and row["micro_batch"] == micro_batch and row["row_id"] == row_id]
                observed = {(row["stage"], row["layer"]) for row in rows if row["occurrence"] == 0}
                if observed != expected_stages or len(rows) != len(expected_stages):
                    problems.append(
                        f"window {window.get('window')} mb {micro_batch}: incomplete or repeated "
                        f"initial stages; missing={sorted(map(str, expected_stages - observed))}, "
                        f"unexpected={sorted(map(str, observed - expected_stages))}"
                    )
    if not index:
        problems.append("no initial-forward traces (capture/recompute are not substitutes)")
    return index, problems


def compare_traces(left: Arm, right: Arm, chunk_elements: int,
                   allow_localization: bool) -> dict[str, Any]:
    ref, left_problems = initial_trace_index(left)
    cand, right_problems = initial_trace_index(right)
    problems = [f"{left.name}: {problem}" for problem in left_problems]
    problems += [f"{right.name}: {problem}" for problem in right_problems]
    missing_left, missing_right = cand.keys() - ref.keys(), ref.keys() - cand.keys()
    if missing_left or missing_right:
        problems.append(
            "initial-forward metadata identities do not align; keys alone are not evidence"
        )
    rows = []
    for identity in sorted(ref.keys() & cand.keys(), key=lambda key: trace_order(ref[key])):
        a, b = ref[identity], cand[identity]
        row: dict[str, Any] = {
            "left": a, "right": b,
            "phase_mapping": {"left": a["phase"], "right": b["phase"], "role": "initial_forward"},
            "execution_modes_match": all(a[name] == b[name] for name in
                                         ("grad_enabled", "autocast_enabled")),
            "phase_mapping_deliberate": a["phase"] == b["phase"] or (
                b["phase"] == "nogradforward" and bool(right.record.get("l2l"))
                and not bool(left.record.get("l2l")) and a["phase"] == "gradforward"
            ) or (
                a["phase"] == "nogradforward" and bool(left.record.get("l2l"))
                and not bool(right.record.get("l2l")) and b["phase"] == "gradforward"
            ),
        }
        if not row["phase_mapping_deliberate"]:
            problems.append(f"{a['key']} vs {b['key']}: unapproved initial-forward phase mapping")
        try:
            ref_tensor, cand_tensor = left.trace(a), right.trace(b)
            row["metrics"] = metrics = tensor_metrics(ref_tensor, cand_tensor, chunk_elements)
            del ref_tensor, cand_tensor
            if not metrics["comparable"] or not metrics.get("finite", False):
                problems.append(f"{a['key']} vs {b['key']}: incompatible or nonfinite tensors")
            coordinate = metrics.get("first_different_coordinate")
            if coordinate is not None and a["stage"] == "head" and coordinate:
                row["first_different_causal_token_position"] = a["valid_positions"][coordinate[0]]
            elif coordinate is not None and a["stage"] in {"embed", "decoder", "norm"}:
                row["first_different_valid_token_position"] = coordinate[-2]
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
            row["problem"] = str(error)
            problems.append(f"{a['key']} vs {b['key']}: {error}")
        rows.append(row)
    complete = not problems and bool(rows)
    differences = [row for row in rows if not row.get("metrics", {}).get("bitmatch", True)]
    token_differences = [row for row in differences if row["left"]["stage"] != "ce_sum"]
    earliest = (
        token_differences[0] if token_differences and complete and allow_localization else None
    )
    per_micro_batch = []
    if complete and allow_localization:
        seen = set()
        for row in token_differences:
            context = row["left"]["window"], row["left"]["micro_batch"]
            if context not in seen:
                seen.add(context)
                per_micro_batch.append(row)
    return {
        "complete": complete, "problems": problems,
        "excluded_phases": ["capture", "recompute"],
        "phase_policy": "gradforward/nogradforward map only to initial_forward; "
        "actual modes retained",
        "ordering": "stage, decoder layer, window, micro_batch; not wall-clock schedule order",
        "missing_left_metadata": [list(key) for key in sorted(missing_left, key=str)],
        "missing_right_metadata": [list(key) for key in sorted(missing_right, key=str)],
        "matched_tensor_count": len(rows), "different_tensor_count": len(differences),
        "bitmatch": bool(complete and not differences),
        "localization_allowed": complete and allow_localization,
        "earliest_valid_token_difference": earliest,
        "earliest_difference_by_micro_batch": per_micro_batch, "tensors": rows,
    }


def compare_adjoints(left: Arm, right: Arm, chunk_elements: int,
                     allow_localization: bool) -> dict[str, Any]:
    indices = []
    problems = []
    excluded = {}
    cross_schedule = bool(left.record.get("l2l")) != bool(right.record.get("l2l"))
    for arm in (left, right):
        groups = {}
        excluded[arm.name] = []
        for row in arm.trace_rows:
            if row["phase"] != "adjoint":
                continue
            metadata_identity = trace_identity(row)
            if cross_schedule and row["stage"] == "decoder":
                # Occurrence counts include redundant checkpoint hooks; select a
                # unique source-phase role before alignment, retaining raw counts.
                metadata_identity = (*metadata_identity[:6], *metadata_identity[7:])
            identity = (*metadata_identity, row["boundary"])
            groups.setdefault(identity, []).append(row)
        index = {}
        for identity, observations in groups.items():
            if not cross_schedule or observations[0]["stage"] != "decoder":
                for row in observations:
                    exact_identity = (*identity, row["source_phase"])
                    if exact_identity in index:
                        problems.append(f"{arm.name}: duplicate adjoint boundary/source identity")
                    index[exact_identity] = row
                continue
            # Decoder adjoints describe the backward computational boundary. GA's
            # original graph and L2L's explicit reverse recompute play that same role.
            # Prefer original GA graph gradients; never merge redundant observations.
            phases = ("recompute",) if arm.record.get("l2l") else ("gradforward", "recompute")
            candidates = []
            for phase in phases:
                candidates = [row for row in observations if row["source_phase"] == phase]
                if candidates:
                    break
            if len(candidates) != 1:
                problems.append(f"{arm.name}: missing or ambiguous computational decoder adjoint")
                excluded[arm.name].extend(observations)
                continue
            selected = candidates[0]
            index[(*identity, "decoder_backward_boundary")] = selected
            excluded[arm.name].extend(row for row in observations if row is not selected)
        indices.append(index)
    ref, cand = indices
    if not ref and not cand:
        return {"status": "NOT PROVIDED", "complete": False, "boundaries": [],
                "reason": "matching forwards with differing gradients require boundary adjoints"}
    missing_left, missing_right = cand.keys() - ref.keys(), ref.keys() - cand.keys()
    if missing_left or missing_right:
        problems.append("adjoint metadata/boundaries do not align")
    rows = []
    def ordering(key: tuple[Any, ...]) -> tuple[Any, ...]:
        # Reverse computational boundary order, retaining every actual source phase.
        return trace_order(ref[key])[:2], ref[key]["boundary"] == "output"
    for identity in sorted(ref.keys() & cand.keys(), key=ordering, reverse=True):
        a, b = ref[identity], cand[identity]
        try:
            ref_tensor, cand_tensor = left.trace(a), right.trace(b)
            metrics = tensor_metrics(ref_tensor, cand_tensor, chunk_elements)
            del ref_tensor, cand_tensor
            rows.append({"left": a, "right": b, "metrics": metrics,
                         "source_phase_mapping": {"left": a["source_phase"],
                                                  "right": b["source_phase"]}})
            if not metrics["comparable"] or not metrics.get("finite", False):
                problems.append(f"{a['key']}: incompatible/nonfinite adjoint")
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            problems.append(str(error))
    complete = not problems and bool(rows)
    differences = [row for row in rows if not row["metrics"]["bitmatch"]]
    return {
        "status": "ALIGNED OBSERVED BOUNDARIES" if complete else "INCOMPLETE",
        "complete": complete, "problems": problems, "boundaries": rows,
        "excluded_observations": excluded,
        "phase_policy": "cross-GA/L2L decoder backward boundaries: GA gradforward preferred, "
        "recompute fallback; L2L recompute. Other stages retain exact source phases.",
        "ordering": "reverse forward boundary order (output before input); "
        "observed boundaries only",
        "missing_left_metadata": [list(key) for key in sorted(missing_left, key=str)],
        "missing_right_metadata": [list(key) for key in sorted(missing_right, key=str)],
        "first_observed_backward_difference": differences[0]
        if complete and allow_localization and differences else None,
        "coverage_limit": "unobserved boundaries cannot be ruled out; "
        "adjoints alone do not prove cause",
    }


def compare_row_losses(left: Arm, right: Arm) -> dict[str, Any]:
    indices = []
    problems = []
    for arm in (left, right):
        index = {}
        for row in arm.record.get("per_microbatch_losses", []):
            try:
                identity = row["window"], row["micro_batch"], row["row_id"]
                if identity in index:
                    raise ArtifactError("duplicate per-microbatch loss")
                if not math.isfinite(row["loss"]):
                    raise ArtifactError("nonfinite actual model loss")
                window = next(window for window in arm.record["windows"]
                              if window["window"] == row["window"])
                micro_batch = row["micro_batch"]
                if window["row_ids"][micro_batch] != row["row_id"]:
                    raise ArtifactError("loss does not identify actual row")
                if row["n_items"] != window["n_items"]:
                    raise ArtifactError("loss denominator differs from original fetched-window N")
                if row["supervised_tokens"] != window["supervised_tokens"][micro_batch]:
                    raise ArtifactError("loss supervised-token metadata differs from actual row")
                passed_n = row.get("passed_n_items", row["n_items"])
                if passed_n != window["n_items"]:
                    raise ArtifactError(
                        "actual loss_function denominator argument differs from window N"
                    )
                index[identity] = row
            except (ValueError, TypeError, KeyError, IndexError, StopIteration) as error:
                problems.append(f"{arm.name}: invalid per-microbatch loss: {error}")
        indices.append(index)
        expected_contexts = {
            (window["window"], micro_batch, row_id)
            for window in arm.record.get("windows", []) if isinstance(window, dict)
            for micro_batch, row_id in enumerate(window.get("row_ids", []))
        }
        if index.keys() != expected_contexts:
            problems.append(f"{arm.name}: actual per-row losses do not cover every consumed row")
    ref, cand = indices
    if not ref or not cand or ref.keys() != cand.keys():
        problems.append("missing or unaligned actual per-microbatch losses")
    rows = []
    for identity in sorted(ref.keys() & cand.keys()):
        a, b = ref[identity], cand[identity]
        rows.append({
            "left": a, "right": b, "loss_bitmatch": exact_json(a["loss"], b["loss"]),
            "n_items_match": a["n_items"] == b["n_items"],
            "supervised_tokens_match": a["supervised_tokens"] == b["supervised_tokens"],
            "left_ce_numerator": next((left.trace_descriptions[row["key"]].get("scalar_value")
                for row in left.trace_rows if row["stage"] == "ce_sum"
                and row["phase"] in INITIAL_PHASES
                and (row["window"], row["micro_batch"], row["row_id"]) == identity), None),
            "right_ce_numerator": next((right.trace_descriptions[row["key"]].get("scalar_value")
                for row in right.trace_rows if row["stage"] == "ce_sum"
                and row["phase"] in INITIAL_PHASES
                and (row["window"], row["micro_batch"], row["row_id"]) == identity), None),
            "loss_max_abs": abs(b["loss"] - a["loss"]),
            "loss_relative": 0.0 if a["loss"] == b["loss"] else
            abs(b["loss"] - a["loss"]) / abs(a["loss"]) if a["loss"] else math.inf,
        })
    if any(not row["n_items_match"] or not row["supervised_tokens_match"] for row in rows):
        problems.append("paired row denominators/supervised-token counts differ")
    if any(row["left_ce_numerator"] is None or row["right_ce_numerator"] is None for row in rows):
        problems.append("missing aligned per-row CE numerator")
    return {"complete": not problems, "problems": problems, "rows": rows,
            "note": "actual model losses and N, separate from observer CPU FP32 CE numerators"}


def effective_input_contrast(left: Arm, right: Arm, first_window: bool,
                             policy: str) -> dict[str, Any]:
    collected = []
    for arm in (left, right):
        windows = arm.record.get("windows", [])
        if first_window:
            windows = windows[:1]
        collected.append([
            {key: item.get(key) for key in INPUT_FIELDS}
            for window in windows if isinstance(window, dict)
            for item in window.get("effective_inputs", []) if isinstance(item, dict)
        ])
    ref, cand = collected
    comparisons = []
    for a, b in zip(ref, cand):
        comparisons.append({
            "row_id": a["row_id"], "left": a, "right": b,
            "fields_match": {key: exact_json(a[key], b[key]) for key in INPUT_FIELDS},
        })
    exact = bool(ref) and exact_json(ref, cand)
    mask_matches = bool(comparisons) and all(
        row["fields_match"][key] for row in comparisons
        for key in ("attention_mask_sha256", "mask_policy")
    )
    position_matches = bool(comparisons) and all(
        row["fields_match"][key] for row in comparisons
        for key in ("position_ids_sha256", "position_policy")
    )
    dtype_matches = bool(comparisons) and all(
        row["fields_match"][key] for row in comparisons
        for key in ("input_dtype", "label_dtype")
    )
    valid = len(ref) == len(cand) and bool(ref) and dtype_matches
    if policy == "matched":
        valid = valid and exact
    classification = "matched post-padding inputs/mask/positions"
    if policy == "padding":
        classification = "padding/shape intervention with matched mask and position policy"
        if not mask_matches:
            classification = "padding plus mask-policy intervention (not shape alone)"
        if not position_matches:
            classification += "; position policy also differs"
    elif policy == "adaptive":
        classification = "adaptive contrast; intervention factors require explicit human review"
    return {
        "valid": valid, "exact_match": exact, "mask_match": mask_matches,
        "position_match": position_matches, "dtype_match": dtype_matches,
        "classification": classification, "policy": policy, "rows": comparisons,
    }



def compare_pair(name: str, left: Arm, right: Arm, kind: str, chunk_elements: int,
                 observer_controls_valid: bool = True) -> dict[str, Any]:
    first_window = kind in {"first_window_control", "first_window_gradients"}
    prerequisite = prerequisites(left, right, first_window)
    gradients = compare_gradients(left, right, chunk_elements, first_window)
    policy = "padding" if name in {
        "shape_mask_intervention", "original_l2l_vs_ga", "original_first_window_l2l_vs_ga"
    } else (
        "adaptive" if kind == "trace" and name != "residual_execution_path" else "matched"
    )
    effective_inputs = effective_input_contrast(left, right, first_window, policy)
    if not effective_inputs["valid"]:
        prerequisite["problems"].append(
            "post-padding IDs/labels/mask/position invariants not satisfied"
        )
    settings = {
        key: key in left.record and key in right.record and exact_json(
            left.record[key], right.record[key]
        ) for key in ("device", "dtype", "quant", "n_layers")
    }
    prerequisite["matches"]["execution_settings"] = all(settings.values())
    if not all(settings.values()):
        prerequisite["problems"].append("device/dtype/quant/n_layers differs or is missing")
    prerequisite["valid"] = not prerequisite["problems"]
    a_logs, b_logs = left.record.get("logs", []), right.record.get("logs", [])
    if first_window:
        a_logs, b_logs = a_logs[:1], b_logs[:1]
    losses_equal = bool(a_logs) and exact_json(
        [{key: row.get(key) for key in ("loss", "step")}
         for row in a_logs if isinstance(row, dict)],
        [{key: row.get(key) for key in ("loss", "step")}
         for row in b_logs if isinstance(row, dict)],
    )
    applied = []
    for arm in (left, right):
        rates = arm.record.get("applied_rates", [])
        applied.append(rates[:1] if first_window else rates)
    applied_equal = bool(applied[0]) and exact_json(*applied)
    returned_losses = []
    for arm in (left, right):
        rows = arm.record.get("training_step_losses", [])
        if first_window:
            rows = [row for row in rows if isinstance(row, dict) and row.get("window") == 0]
        returned_losses.append(rows)
    returned_losses_equal = bool(returned_losses[0]) and exact_json(*returned_losses)
    losses_equal = losses_equal and returned_losses_equal
    if first_window and any(not rates or rates[0] != 0.0 for rates in applied):
        prerequisite["problems"].append("first-window control requires actually applied LR zero")
        prerequisite["valid"] = False
    poststep_equal = bool(left.record.get("rates")) and exact_json(
        left.record.get("rates"), right.record.get("rates")
    ) and exact_json(
        [row.get("learning_rate") for row in a_logs if isinstance(row, dict)],
        [row.get("learning_rate") for row in b_logs if isinstance(row, dict)],
    )
    rates_equal = applied_equal and (first_window or poststep_equal)
    steps_equal = (
        bool(gradients["left_steps"]) and gradients["left_steps"] == gradients["right_steps"]
    )
    equal = (
        prerequisite["valid"] and gradients["bitmatch"] and losses_equal
        and rates_equal and steps_equal
    )
    result: dict[str, Any] = {
        "name": name, "left": left.name, "right": right.name, "kind": kind,
        "prerequisites": prerequisite, "gradients": gradients,
        "losses_bitmatch": losses_equal, "rates_bitmatch": rates_equal,
        "training_step_losses_bitmatch": returned_losses_equal,
        "applied_rates_bitmatch": applied_equal, "poststep_rates_bitmatch": poststep_equal,
        "comparison_scope": "first optimizer window only" if first_window else "all saved steps",
        "effective_inputs": effective_inputs, "execution_settings_match": settings,
        "steps_equal": steps_equal,
        "exact_control_valid": equal if kind in {"control", "first_window_control"} else None,
        "status": "INVALID" if not prerequisite["valid"] else ("BITMATCH" if equal else "DIFFERS"),
    }
    if kind == "trace":
        result["observer_controls_valid"] = observer_controls_valid
        result["traces"] = compare_traces(
            left, right, chunk_elements, prerequisite["valid"] and observer_controls_valid
        )
        result["captured_layer_calls"] = compare_layer_calls(left, right)
        result["dispatch"] = compare_dispatch(left, right)
        result["effective_decoder_kwargs_match"] = (
            result["captured_layer_calls"]["observed_kwargs_match"]
        )
        result["mask_position_classification"] = (
            "observed decoder0 kwargs match" if result["effective_decoder_kwargs_match"] else
            "observed decoder0 mask/position/other kwargs differ; "
            "not an isolated shape or schedule effect"
        )
        if not result["captured_layer_calls"]["complete"] or not result["dispatch"]["complete"]:
            result["traces"]["localization_allowed"] = False
            result["traces"]["earliest_valid_token_difference"] = None
            result["traces"]["earliest_difference_by_micro_batch"] = []
            result["traces"]["complete"] = False
            result["traces"]["problems"].append(
                "missing captured decoder kwargs or actual NF4 dispatch evidence"
            )
        result["per_row_losses"] = compare_row_losses(left, right)
        if not result["per_row_losses"]["complete"]:
            result["traces"]["localization_allowed"] = False
            result["traces"]["earliest_valid_token_difference"] = None
            result["traces"]["earliest_difference_by_micro_batch"] = []
            result["traces"]["problems"].append(
                "missing/invalid actual per-row loss and denominator evidence"
            )
            result["traces"]["complete"] = False
        result["adjoints"] = compare_adjoints(
            left, right, chunk_elements, result["traces"]["localization_allowed"]
        )
        forward_tokens = [row for row in result["traces"]["tensors"]
                          if row["left"]["stage"] != "ce_sum"]
        forwards_match = bool(forward_tokens) and all(
            row.get("metrics", {}).get("bitmatch", False) for row in forward_tokens
        )
        result["backward_localization_needed"] = forwards_match and not gradients["bitmatch"]
        if (
            not prerequisite["valid"] or not result["traces"]["complete"]
            or not observer_controls_valid
        ):
            result["status"] = "NO CAUSAL VERDICT"
        elif not result["per_row_losses"]["complete"]:
            result["status"] = "NO CAUSAL VERDICT"
        if name == "residual_execution_path":
            result["interpretation"] = (
                "Residual execution-path difference with inputs/shapes held fixed; not a pure "
                "schedule cause. Grad-mode, checkpoint, wrapper and backward scope "
                "remain confounded."
            )
    return result


class LazyGradientSteps(Sequence[dict[str, Any]]):
    """Let unchanged baseline code iterate steps without eagerly loading all steps."""

    def __init__(self, arm: Arm) -> None:
        self.arm = arm

    def __len__(self) -> int:
        return int(self.arm.record["steps"])

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.arm.gradients(index)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


def pinned_decide(baseline: Path, arms: dict[str, Arm]) -> dict[str, Any]:
    names = ("ga_3step", "ga_repeat_3step", "l2l_3step", "ga_floor_3step")
    problems = []
    for name in names[1:]:
        check = prerequisites(arms[names[0]], arms[name])
        problems.extend(check["problems"])
        if not all(arms[names[0]].record.get(key) == arms[name].record.get(key)
                   for key in ("device", "dtype", "quant", "n_layers")):
            problems.append(
                f"{name}: execution device/dtype/quant/layers differs from GA reference"
            )
        if arms[name].record.get("baseline_commit") != BASELINE_COMMIT:
            problems.append(f"{name}: recorded product baseline is not pinned")
    if arms[names[0]].record.get("baseline_commit") != BASELINE_COMMIT:
        problems.append(f"{names[0]}: recorded product baseline is not pinned")
    if problems:
        return {"status": "NO STATISTICAL VERDICT", "problems": problems}
    relative = "benchmarks/harness/l2l_mistral_stand.py"
    try:
        revision = subprocess.run(
            ["git", "-C", str(baseline), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if revision != BASELINE_COMMIT:
            raise ArtifactError(f"baseline HEAD is {revision}, expected {BASELINE_COMMIT}")
        tracked = subprocess.run(
            ["git", "-C", str(baseline), "show", f"{BASELINE_COMMIT}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        source = baseline / relative
        working_source = source.read_bytes()
        normalized_source = working_source.replace(b"\r\n", b"\n")
        if normalized_source != tracked.replace(b"\r\n", b"\n"):
            raise ArtifactError("pinned stand working file differs from the pinned Git blob")
        module_name = "_a3_1b_unchanged_pinned_stand"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ArtifactError("cannot load pinned stand")
        module = importlib.util.module_from_spec(spec)
        old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
        previous_module = sys.modules.get(module_name)
        try:
            sys.dont_write_bytecode = True
            sys.modules[module_name] = module
            exec(compile(normalized_source, str(source), "exec"), module.__dict__)
            runs = {}
            for name in names:
                arm = arms[name]
                runs[name] = module.RunRecord(
                    logs=arm.record["logs"], rates=arm.record["rates"],
                    grads=LazyGradientSteps(arm), steps=arm.record["steps"],
                    l2l=bool(arm.record.get("l2l")),
                )
            mode = "bf16" if arms[names[0]].record.get("dtype") in ("bf16", "bfloat16") else "fp32"
            decision = module.decide(
                runs[names[0]], runs[names[1]], runs[names[2]], mode=mode,
                floor=runs[names[3]] if mode == "bf16" else None,
            )
        finally:
            sys.path[:] = old_path
            sys.dont_write_bytecode = old_bytecode
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
        historical_device = all(arms[name].record.get("device") == "cuda" for name in names)
        statistical_status = "PASS" if decision.get("pass") else (
            "FAIL" if decision.get("status") in {"l2l_out_of_tolerance", "l2l_differs"}
            else "NO STATISTICAL VERDICT"
        )
        return {
            "status": statistical_status if historical_device and mode == "bf16"
            else "CPU/FP32 REHEARSAL ONLY",
            "comparison": "unchanged pinned stand.decide; not the spill-gate validity wrapper",
            "baseline_commit": revision, "source_sha256": hashlib.sha256(tracked).hexdigest(),
            "working_raw_source_sha256": hashlib.sha256(working_source).hexdigest(),
            "normalized_source_sha256": hashlib.sha256(normalized_source).hexdigest(),
            "source_comparison": "Python source CRLF normalized to LF only; tensors remain raw",
            "mode": mode, "decision": decision,
            "historical_records_modified": False, "causal_verdict": False,
        }
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        return {"status": "NO STATISTICAL VERDICT", "problems": [str(error)]}


def parse_pair(value: str) -> tuple[str, str, str, str]:
    parts = value.split(":")
    if len(parts) != 3 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in parts):
        raise argparse.ArgumentTypeError("pair must be NAME:LEFT_ARM:RIGHT_ARM (safe simple names)")
    return parts[0], parts[1], parts[2], "trace"


def analyze(root: Path, chunk_elements: int, baseline: Path | None,
            extra_pairs: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    specs = [*PAIRS, *extra_pairs]
    names = sorted({name for _, left, right, _ in specs for name in (left, right)})
    arms = {name: load_arm(root, name, chunk_elements) for name in names}
    pairs = {}
    for name, left_name, right_name, kind in specs:
        controls_valid = all(pairs.get(control, {}).get("exact_control_valid", False)
                             for control in CONTROLS)
        pairs[name] = compare_pair(
            name, arms[left_name], arms[right_name], kind, chunk_elements, controls_valid
        )
    statistical = pinned_decide(baseline, arms) if baseline else {
        "status": "NOT EVALUATED", "reason": "--baseline required to call unchanged pinned decide",
        "required_baseline_commit": BASELINE_COMMIT,
    }
    missing = []
    for control in CONTROLS:
        if not pairs[control]["exact_control_valid"]:
            missing.append(f"{control}: gradients, losses, rates and consumed inputs must bitmatch")
    for name in CONTRASTS:
        pair = pairs[name]
        if not pair["prerequisites"]["valid"]:
            missing.append(f"{name}: invalid or unequal initial adapters/actual consumed rows/N")
        if not pair["traces"]["complete"]:
            missing.append(
                f"{name}: complete, hash-validated, metadata-aligned initial-forward traces"
            )
        if pair.get("backward_localization_needed") and not pair["adjoints"]["complete"]:
            missing.append(
                f"{name}: forward tokens match but gradients differ; aligned boundary adjoints"
            )
    if statistical.get("status") != "FAIL":
        missing.append(
            "reproduction of the CUDA/BF16 statistical discrepancy using unchanged pinned decide"
        )
    if not any(pairs[name]["traces"]["earliest_valid_token_difference"]
               for name in CONTRASTS):
        missing.append("an earliest certified valid-token divergence after valid observer controls")
    missing.append(
        "a reviewed single-factor counterfactual removing/reducing the localized divergence "
        "as predicted, "
        "and the original comparator rerun with only that factor aligned"
    )
    return {
        "schema": SCHEMA, "input": str(root), "device": "CPU analysis only",
        "metric_policy": {
            "relative_l2": "||candidate-reference||_2 / ||reference||_2",
            "zero_reference": "0 if both equal, Infinity if nonzero difference and zero reference",
            "bitmatch": "raw tensor bytes, shape and dtype; includes signed zero/NaN payloads",
            "nonfinite_json": "NaN/Infinity/-Infinity strings, never silently dropped",
            "accumulation": "float64 chunk reductions; deterministic file/parameter order; "
            "no tolerances",
            "chunk_elements": chunk_elements,
            "scope_limit": "baseline decide is invoked unchanged; it may use whole-parameter "
            "float64 temporaries. Independent raw metrics are chunked; mmap paging is OS-managed.",
        },
        "arms": {name: arm.summary() for name, arm in arms.items()}, "pairs": pairs,
        "statistical_1b": statistical,
        "causal_mechanism": {
            "status": "NO CAUSAL VERDICT", "missing_evidence": missing,
            "interpretation": "earliest divergence is localization, not a cause; "
            "aggregate gradient error does not establish a scalar normalization error "
            "or any other mechanism",
            "counterfactual_policy": "additional --pair evidence is measured but never "
            "automatically promoted to cause found; single-factor design and "
            "original-comparator alignment need review",
        },
    }


def summary_text(report: dict[str, Any]) -> str:
    lines = ["A3 criterion 1b diagnostic analysis (CPU; no tolerance changes)"]
    for name, pair in report["pairs"].items():
        lines.append(
            f"{name}: {pair['status']}; grads bitmatch={pair['gradients']['bitmatch']}, "
            f"losses={pair['losses_bitmatch']}, rates={pair['rates_bitmatch']}"
        )
        if pair.get("effective_inputs"):
            lines.append(f"  contrast: {pair['effective_inputs']['classification']}")
        if pair.get("mask_position_classification"):
            lines.append(f"  effective kwargs: {pair['mask_position_classification']}")
        traces = pair.get("traces")
        if traces is not None:
            earliest = traces["earliest_valid_token_difference"]
            if earliest:
                row, metrics = earliest["left"], earliest["metrics"]
                lines.append(
                    f"  earliest valid-token difference: {row['stage']} layer={row['layer']} "
                    f"window={row['window']} mb={row['micro_batch']}; "
                    f"max_abs={metrics['max_abs']:.9g}, rel_l2={metrics['relative_l2']:.9g}"
                    f"; bitmatch={metrics['bitmatch']}, "
                    f"norms={metrics['reference_norm']:.9g}/{metrics['candidate_norm']:.9g}, "
                    f"nonfinite={metrics['reference_nonfinite']}/{metrics['candidate_nonfinite']}"
                )
            elif traces["localization_allowed"]:
                lines.append("  all aligned initial-forward valid-token tensors bitmatch")
            else:
                lines.append(
                    "  no certified earliest divergence (missing/invalid traces or controls)"
                )
            backward = pair.get("adjoints", {}).get("first_observed_backward_difference")
            if backward:
                row = backward["left"]
                lines.append(
                    f"  first observed backward difference: {row['stage']} layer={row['layer']} "
                    f"boundary={row['boundary']}; not an unobserved-boundary exclusion"
                )
            elif pair.get("backward_localization_needed"):
                lines.append(
                    "  forward tokens match but gradients differ; inspect boundary adjoints"
                )
    lines.append(f"Statistical 1b comparison: {report['statistical_1b']['status']}")
    decision = report["statistical_1b"].get("decision")
    if decision:
        lines.append(f"  unchanged stand.decide: {decision['status']}")
    lines.append("Causal mechanism: NO CAUSAL VERDICT")
    lines.extend(f"  missing: {item}" for item in report["causal_mechanism"]["missing_evidence"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="runner output directory (read-only)")
    parser.add_argument(
        "--out", required=True, type=Path, help="new JSON report; never overwritten"
    )
    parser.add_argument(
        "--baseline", type=Path, help="optional checkout of the exact pinned baseline"
    )
    parser.add_argument("--chunk-elements", type=int, default=1 << 20)
    parser.add_argument("--pair", action="append", type=parse_pair, default=[],
                        help="additional metadata-aligned adaptive comparison NAME:LEFT:RIGHT")
    args = parser.parse_args(argv)
    console = Console(highlight=False)
    if args.chunk_elements < 1:
        parser.error("--chunk-elements must be positive")
    if not args.run_dir.is_dir():
        parser.error("run_dir must be an existing directory")
    arms_root = os.path.realpath(args.run_dir / "arms")
    output_real = os.path.realpath(args.out)
    try:
        output_in_arms = os.path.commonpath((arms_root, output_real)) == arms_root
    except ValueError:
        output_in_arms = False
    if output_in_arms:
        parser.error("--out must not be inside input arms; artifact trees are read-only")
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out already exists; reports are never overwritten")
    if not args.out.parent.is_dir():
        parser.error("--out parent directory must already exist")
    pair_names = [spec[0] for spec in (*PAIRS, *args.pair)]
    if len(set(pair_names)) != len(pair_names):
        parser.error("comparison names must be unique, including built-in names")
    try:
        report = analyze(args.run_dir, args.chunk_elements, args.baseline, args.pair)
        report["text_summary"] = summary_text(report)
        serialized = json.dumps(safe_numbers(report), indent=2, allow_nan=False) + "\n"
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        console.print(report["text_summary"], markup=False)
        console.print(f"Raw metrics: {args.out}", markup=False)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        console.print(f"Analysis refused: {error}", markup=False)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
