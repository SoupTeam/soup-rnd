"""One unchanged A3/G4 gradient gate on the first training window, without updates.

Production CLI::

    python -m benchmarks.harness.p3_deepseek_quality_gate \
        --source SOURCE --cache CACHE --pack PACK --pack-sha256 HEX \
        --initial-adapter initial.safetensors --output NEW_DIRECTORY

The three measured arms are ordinary streamed GA, BF16 L2L, and block64 E4M3 plus
original-forward Oracle L2L. Main GA, its repeat, and both L2L candidates strictly
use EFFICIENT_ATTENTION. An independent MATH GA supplies the unchanged A3 floor.
The int4 plus Oracle arm tests discrimination, not a second candidate.

Every arm restores the same adapter. No optimizer is constructed. Layer inputs
and outputs are archived one pair at a time, on disk, so E1/E2 use the candidate's
executed recompute and the frozen G4 metrics without retaining all boundaries in
an instrumentation cache. This extra synchronous I/O is included in gate time;
these timings are not training throughput measurements. A fresh directory and
exclusive writes preserve partial failures. The manifest is the last file.

Tiny native checkpoints and FP32 on native-BF16 Ampere-or-newer CUDA are admitted
only by ``fixture=True``. Neither can yield a full-model PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import struct
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console

from benchmarks.harness.p3_deepseek_quality_attention import (
    FLOOR_BACKEND,
    MAIN_BACKEND,
    attention_execution,
    policy_identity,
    require_ampere_cuda,
    require_execution_evidence,
)
from benchmarks.harness.p3_deepseek_quality_profiles import get_profile

SCHEMA = "p3-deepseek-quality-gate-v1"
SEED = 23
K = 8
MICROBATCH_LENGTH = 64
PREFIX_TOKENS = 512
HOST_BUDGET_BYTES = 2 * 1024**3
GATES = ("E1", "E2", "E3")


def _child(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or Path(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("evidence path must be a contained relative artifact name")
    parent = os.path.realpath(root)
    child = os.path.realpath(root / relative)
    if os.path.commonpath([parent, child]) != parent or child == parent:
        raise ValueError("evidence path escapes its output directory")
    return Path(child)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any, *, path: str = "$", nonfinite: list[str] | None = None) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if nonfinite is not None:
            nonfinite.append(path)
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, path=f"{path}.{key}", nonfinite=nonfinite)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, path=f"{path}[{index}]", nonfinite=nonfinite)
                for index, item in enumerate(value)]
    raise TypeError(f"non-JSON gate evidence: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _save_tensors(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(values, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _ids_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}q", *values)).hexdigest()


def gate_prefix(window: dict[str, Any]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Split window zero's first 512 tokens, preserving response-only labels."""
    batches = window.get("micro_batches")
    if window.get("common_boundary_mask_interval") != 1024 or not isinstance(batches, list):
        raise ValueError("gate requires the unchanged L1024 packed window")
    if len(batches) != 4:
        raise ValueError("gate requires four original L1024 micro-batches")
    for batch in batches:
        for key in ("input_ids", "labels"):
            rows = batch.get(key)
            if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list)
                    or len(rows[0]) != 1024 or any(type(token) is not int for token in rows[0])):
                raise ValueError("gate requires B1/L1024 integer input_ids and labels")
        if any(token < 0 for token in batch["input_ids"][0]):
            raise ValueError("negative input token")
        if any(token < 0 and token != -100 for token in batch["labels"][0]):
            raise ValueError("negative label other than the ignored label")
    ids = list(batches[0]["input_ids"][0][:PREFIX_TOKENS])
    labels = list(batches[0]["labels"][0][:PREFIX_TOKENS])
    inputs, targets, counts = [], [], []
    for start in range(0, PREFIX_TOKENS, MICROBATCH_LENGTH):
        input_row = ids[start:start + MICROBATCH_LENGTH]
        target_row = labels[start:start + MICROBATCH_LENGTH]
        target_row[0] = -100
        inputs.append([input_row])
        targets.append([target_row])
        counts.append(sum(token != -100 for token in target_row[1:]))
    total = sum(counts)
    if total <= 0:
        raise ValueError("gate prefix has no supervised causal targets")
    return inputs, targets, {
        "window_id": window.get("window_id"), "window_index": 0, "pack_length": 1024,
        "prefix_start": 0, "prefix_stop": PREFIX_TOKENS, "input_tokens": PREFIX_TOKENS,
        "k": K, "microbatch_length": MICROBATCH_LENGTH, "batch_size": 1,
        "masked_local_first_positions": list(range(0, PREFIX_TOKENS, MICROBATCH_LENGTH)),
        "valid_target_count": total, "per_microbatch_targets": counts,
        "input_sha256": _ids_sha256(ids),
        "labels_sha256": _ids_sha256([token for row in targets for token in row[0]]),
        "loss_normalization": "sum causal unmasked losses / recomputed prefix target count",
    }


def _no_metrics(detail: str) -> dict[str, Any]:
    return {name: {"status": "NO_VERDICT", "detail": detail} for name in GATES}


def assess_metrics(
    layers: Sequence[dict[str, Any]], gradient: dict[str, Any], *,
    expected_layers: Sequence[int], expected_tensors: Sequence[str],
) -> dict[str, Any]:
    """Apply frozen G4 bounds only after checking every layer, pair and tensor."""
    from benchmarks.harness import g4_rule

    problems = []
    try:
        if not expected_layers or [row["layer"] for row in layers] != list(expected_layers):
            problems.append("missing, duplicated, or out-of-order decoder layers")
        for row in layers:
            pairs = row["pairs"]
            if [part["microbatch"] for part in pairs] != list(range(K)):
                problems.append(f"layer {row['layer']}: incomplete recompute pairs")
            if any(part.get("finite") is not True or part.get("forward_equal") is not True
                   or part.get("rows") != MICROBATCH_LENGTH for part in pairs):
                problems.append(f"layer {row['layer']}: nonfinite or unmatched boundary pair")
            for kind in ("restore", "branch", "output"):
                metric = row[kind]
                if metric.get("rows") != PREFIX_TOKENS:
                    problems.append(f"layer {row['layer']}: incomplete {kind} row coverage")
                for key in (
                    "rms_row_rel", "worst_row_rel", "rel_l2", "max_abs",
                    "max_abs_over_absmax", "err_sq", "ref_sq", "row_rel_sq_sum",
                ):
                    if not _finite(metric.get(key)) or metric[key] < 0:
                        problems.append(f"layer {row['layer']}: invalid {kind}.{key}")
        per = gradient["per_tensor"]
        if (not expected_tensors or set(per) != set(expected_tensors)
                or gradient.get("tensors") != len(expected_tensors)):
            problems.append("incomplete named gradient tensor coverage")
        for name, metric in per.items():
            for key in ("cosine", "rel_l2"):
                if not _finite(metric.get(key)):
                    problems.append(f"gradient {name}: nonfinite {key}")
            if (_finite(metric.get("rel_l2")) and metric["rel_l2"] < 0
                    or _finite(metric.get("cosine"))
                    and not -1.000001 <= metric["cosine"] <= 1.000001):
                problems.append(f"gradient {name}: invalid metric range")
            if (metric.get("reference_finite") is not True
                    or metric.get("candidate_finite") is not True
                    or metric.get("shape_equal") is not True):
                problems.append(f"gradient {name}: unverified finite/shape coverage")
        for key in ("global_rel_l2", "min_cosine", "max_rel_l2"):
            if not _finite(gradient.get(key)):
                problems.append(f"nonfinite gradient.{key}")
        if any(_finite(gradient.get(key)) and gradient[key] < 0
               for key in ("global_rel_l2", "max_rel_l2")):
            problems.append("negative global gradient error")
        if (per and all(_finite(part.get("cosine")) for part in per.values())
                and gradient.get("min_cosine") != min(part["cosine"] for part in per.values())):
            problems.append("minimum cosine does not cover every tensor")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        problems.append(f"malformed metric evidence: {error}")
    if problems:
        return _no_metrics("; ".join(problems))
    rms = max(row["restore"]["rms_row_rel"] for row in layers)
    worst = max(row["restore"]["worst_row_rel"] for row in layers)
    branch = max(row["branch"]["rms_row_rel"] for row in layers)
    over1 = [row["layer"] for row in layers if (
        row["restore"]["rms_row_rel"] > g4_rule.RESTORE_RMS_BOUND
        or row["restore"]["worst_row_rel"] > g4_rule.RESTORE_WORST_ROW_BOUND
    )]
    over2 = [row["layer"] for row in layers
             if row["branch"]["rms_row_rel"] > g4_rule.BRANCH_RMS_BOUND]
    return {
        "E1": {"status": "FAIL" if over1 else "PASS", "layers_over": over1,
               "max_rms_row_rel": rms, "max_worst_row_rel": worst,
               "rms_bound": g4_rule.RESTORE_RMS_BOUND,
               "worst_row_bound": g4_rule.RESTORE_WORST_ROW_BOUND},
        "E2": {"status": "FAIL" if over2 else "PASS", "layers_over": over2,
               "max_branch_rms_row_rel": branch, "bound": g4_rule.BRANCH_RMS_BOUND,
               "semantics": "one executed Oracle recompute residual branch from restored input"},
        "E3": {"status": "PASS" if (
            gradient["global_rel_l2"] <= g4_rule.GRAD_GLOBAL_BOUND
            and gradient["min_cosine"] >= g4_rule.GRAD_MIN_COSINE
        ) else "FAIL", "global_rel_l2": gradient["global_rel_l2"],
            "min_cosine": gradient["min_cosine"], "tensors": gradient["tensors"],
            "global_rel_l2_bound": g4_rule.GRAD_GLOBAL_BOUND,
            "min_cosine_bound": g4_rule.GRAD_MIN_COSINE},
    }


def assess_discrimination(gates: dict[str, Any]) -> dict[str, Any]:
    not_failed = [name for name in GATES if gates.get(name, {}).get("status") != "FAIL"]
    return {
        "status": "NO_VERDICT" if not_failed else "PASS", "not_failed": not_failed,
        "detail": "Frozen G4 V5 requires the int4 control to fail E1, E2 and E3; "
        "a nondiscriminating fixture never gets a fabricated control failure",
    }


def assess_a3(
    gradients: dict[str, Any], repeat_gradients: dict[str, Any],
    l2l_gradients: dict[str, Any], *, losses: Sequence[float],
    repeat_losses: Sequence[float], l2l_losses: Sequence[float], dtype: str,
    floor_gradients: dict[str, Any] | None = None,
    floor_losses: Sequence[float] | None = None, floor_observed: bool = False,
) -> dict[str, Any]:
    """Keep BF16 PR #34 unchanged; require bitwise losses/gradients for FP32 qualification."""
    import torch

    from benchmarks.harness import l2l_mistral_stand as stand

    inputs = [(gradients, losses), (repeat_gradients, repeat_losses),
              (l2l_gradients, l2l_losses)]
    if floor_observed:
        inputs.append((floor_gradients, floor_losses))
    if dtype not in ("bfloat16", "float32"):
        return {"status": "NO_VERDICT", "detail": "unsupported A3 precision"}
    for values, logs in inputs:
        if (not isinstance(values, dict) or not values or not logs or len(logs) != K
                or not all(_finite(row) for row in logs)
                or any(not isinstance(value, torch.Tensor)
                       or not bool(torch.isfinite(value).all()) for value in values.values())):
            return {"status": "NO_VERDICT", "detail": "missing/nonfinite A3 tensors or losses"}
        if set(values) != set(gradients):
            return {"status": "NO_VERDICT", "detail": "A3 named gradient coverage differs"}
        if any(values[name].shape != gradients[name].shape for name in gradients):
            return {"status": "NO_VERDICT", "detail": "A3 gradient shapes differ"}

    def record(values: dict[str, Any], logs: Sequence[float], l2l: bool = False) -> Any:
        # One gradient observation, not an optimizer update. LR is zero in every record.
        return stand.RunRecord(
            logs=[{"loss": sum(logs), "learning_rate": 0.0}], rates=[0.0],
            grads=[values], steps=1, l2l=l2l,
        )

    floor = record(floor_gradients, floor_losses) if floor_observed else None
    rule = stand.decide(
        record(gradients, losses), record(repeat_gradients, repeat_losses),
        record(l2l_gradients, l2l_losses, True),
        mode="fp32" if dtype == "float32" else "bf16", floor=floor,
    )
    status = "PASS" if rule["pass"] else (
        "FAIL" if rule["status"] in ("l2l_out_of_tolerance", "l2l_differs") else "NO_VERDICT"
    )
    bitwise = None
    if dtype == "float32":
        gradients_equal = all(torch.equal(gradients[name], l2l_gradients[name])
                              for name in gradients)
        losses_equal = list(losses) == list(l2l_losses)
        bitwise = {"equal": gradients_equal and losses_equal,
                   "per_microbatch_losses_equal": losses_equal,
                   "gradients_equal": gradients_equal}
        if list(losses) != list(repeat_losses):
            status = "NO_VERDICT"
        elif status == "PASS" and not bitwise["equal"]:
            status = "FAIL"
    result = {"status": status, "rule": rule, "independent_floor_observed": floor_observed,
              "optimizer_updates": 0, "comparison_observations": 1}
    if bitwise is not None:
        result["bitwise_qualification"] = bitwise
    return result


def overall_verdict(
    a3: dict[str, Any], candidate: dict[str, Any], discrimination: dict[str, Any], *,
    eligible: bool,
) -> str:
    statuses = [a3.get("status"), *(candidate.get(name, {}).get("status") for name in GATES)]
    if "FAIL" in statuses:
        return "FAIL"
    if not eligible or discrimination.get("status") != "PASS" or any(
        status != "PASS" for status in statuses
    ):
        return "NO_VERDICT"
    return "PASS"


def _adapter_digest(parameters: dict[str, Any]) -> str:
    from benchmarks.harness.p3_deepseek_quality_model import adapter_digest

    return adapter_digest(parameters)[0]


def _tensor_check(tensor: Any) -> dict[str, Any]:
    import torch

    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "elements": tensor.numel(), "finite": bool(torch.isfinite(tensor).all())}


def gradient_agreement(
    reference: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, Any]:
    """Frozen G4 metric plus observed per-tensor finite and shape coverage."""
    from benchmarks.harness import g4_activation_codec as metrics

    if set(reference) != set(candidate) or not reference:
        raise ValueError("gradient tensor names differ or are empty")
    checks = {}
    for name in reference:
        left, right = _tensor_check(reference[name]), _tensor_check(candidate[name])
        checks[name] = {"reference_finite": left["finite"], "candidate_finite": right["finite"],
                        "shape_equal": left["shape"] == right["shape"]}
    if any(not all(check.values()) for check in checks.values()):
        raise ValueError("missing, nonfinite, or mismatched gradient tensor")
    result = metrics.grad_agreement(reference, candidate)
    for name, check in checks.items():
        result["per_tensor"][name].update(check)
    return result


class _LayerFiniteObservation:
    """Check ordinary GA's executed decoder inputs and outputs without replacing forward."""

    def __init__(self, model: Any) -> None:
        from soup_cli.utils.l2l_step import model_parts

        self.counts = [0] * len(model_parts(model)[1].layers)
        self.handles = []
        for index, layer in enumerate(model_parts(model)[1].layers):
            def before(_module: Any, args: Any, index: int = index) -> None:
                if not _tensor_check(args[0])["finite"]:
                    raise ValueError(f"nonfinite GA input at decoder layer {index}")

            def after(_module: Any, _args: Any, output: Any, index: int = index) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                if not _tensor_check(hidden)["finite"]:
                    raise ValueError(f"nonfinite GA output at decoder layer {index}")
                self.counts[index] += 1

            self.handles.extend((layer.register_forward_pre_hook(before),
                                 layer.register_forward_hook(after)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class _LayerArchive:
    """One disk-backed original BF16 boundary per pair, and measured actual recomputes."""

    def __init__(
        self, directory: Path, n_layers: int, *, reference: _LayerArchive | None = None,
    ) -> None:
        self.directory = Path(os.path.realpath(directory))
        self.reference = reference
        self.forward: dict[tuple[int, int], dict[str, Any]] = {}
        self.recomputed: dict[tuple[int, int], dict[str, Any]] = {}
        self.parts: list[dict[str, list[Any]]] = [
            {kind: [] for kind in ("restore", "branch", "output")} for _ in range(n_layers)
        ]
        self.io_seconds = 0.0
        self.io_bytes = 0

    @staticmethod
    def _name(layer: int, mb: int) -> str:
        return f"layer-{layer:03d}-mb-{mb:03d}.pt"

    def observe(self, layer: int, mb: int, phase: str, inputs: Any, outputs: Any) -> None:
        import torch

        from benchmarks.harness import g4_activation_codec as metrics

        checks = {"input": _tensor_check(inputs), "output": _tensor_check(outputs)}
        if any(not check["finite"] for check in checks.values()):
            raise ValueError(f"nonfinite decoder {phase} at layer {layer}, microbatch {mb}")
        pair = (layer, mb)
        started = time.perf_counter()
        values = {"input": inputs.detach().to("cpu", copy=True),
                  "output": outputs.detach().to("cpu", copy=True)}
        if phase == "forward":
            if pair in self.forward:
                raise RuntimeError("duplicated original layer forward")
            if self.reference is None:
                path = _child(self.directory, "forward/" + self._name(layer, mb))
                descriptor = _save_tensors(path, values)
                descriptor["path"] = path.as_posix()
                self.io_bytes += descriptor["bytes"]
                equal = True
            else:
                descriptor, original = self.reference.load(pair)
                self.io_bytes += descriptor["bytes"]
                equal = all(torch.equal(original[key], values[key]) for key in values)
                if not equal:
                    raise ValueError("candidate original forward differs from BF16 L2L")
            self.forward[pair] = {"artifact": descriptor, "finite": True,
                                  "forward_equal": equal, "checks": checks}
            self.io_seconds += time.perf_counter() - started
            return
        if pair not in self.forward or pair in self.recomputed:
            raise RuntimeError("missing original forward or duplicate layer recompute")
        source = self.reference if self.reference is not None else self
        descriptor, original = source.load(pair)
        self.io_bytes += descriptor["bytes"]
        path = _child(self.directory, "recompute/" + self._name(layer, mb))
        recompute_artifact = _save_tensors(path, values)
        recompute_artifact["path"] = path.as_posix()
        self.io_bytes += recompute_artifact["bytes"]
        self.io_seconds += time.perf_counter() - started
        self.parts[layer]["restore"].append(
            metrics.restore_error(original["input"], values["input"]),
        )
        self.parts[layer]["output"].append(
            metrics.restore_error(original["output"], values["output"]),
        )
        self.parts[layer]["branch"].append(metrics.restore_error(
            original["output"].double() - original["input"].double(),
            values["output"].double() - values["input"].double(),
        ))
        self.recomputed[pair] = {
            "microbatch": mb, "rows": int(inputs.shape[0]) * int(inputs.shape[1]),
            "finite": True, "forward_equal": self.forward[pair]["forward_equal"],
            "checks": checks, "original": descriptor, "recompute": recompute_artifact,
        }

    def load(self, pair: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        descriptor = self.forward[pair]["artifact"]
        path = Path(descriptor["path"])
        if _sha256(path) != descriptor["sha256"]:
            raise ValueError("immutable original boundary archive hash changed")
        with path.open("rb") as stream:
            return descriptor, torch.load(stream, map_location="cpu", weights_only=True)

    def table(self) -> list[dict[str, Any]]:
        from benchmarks.harness import g4_activation_codec as metrics

        rows = []
        for layer, kinds in enumerate(self.parts):
            row: dict[str, Any] = {
                "layer": layer, "pairs": [
                    self.recomputed[(layer, mb)] for mb in range(K)
                    if (layer, mb) in self.recomputed
                ],
            }
            for kind, parts in kinds.items():
                row[kind] = metrics.merge_errors(parts) if parts else None
            rows.append(row)
        return rows


def _stepper(model: Any, runtime: Any, calls: Sequence[Any], archive: Any, oracle: Any) -> Any:
    import torch

    from soup_cli.utils.l2l_step import L2LStep

    if len(calls) != K or len({id(call) for call in calls}) != K:
        raise ValueError("every gate microbatch requires a distinct captured layer call")

    class ObservedStep(L2LStep):
        def _layer(self, module: Any, hidden: Any, state: Any, call: Any) -> Any:
            layer, mb = layer_ids[id(module)], call_ids[id(call)]
            phase = "recompute" if torch.is_grad_enabled() else "forward"
            visit = oracle.visit(layer, mb, phase) if oracle is not None else nullcontext()
            with visit:
                output = super()._layer(module, hidden, state, call)
            archive.observe(layer, mb, phase, hidden, output)
            return output

    step = ObservedStep(model, runtime, calls[0])
    layer_ids = {id(layer): index for index, layer in enumerate(step.layers)}
    call_ids = {id(call): index for index, call in enumerate(calls)}
    return step


def _store(
    name: str, *, budget: Any, shapes: Sequence[Any], layers: int, dtype: Any, device: str,
) -> tuple[Any, Any]:
    import torch

    from soup_cli.utils.budgeted_activations import (
        BudgetedActivationStore,
        activation_allocation_bytes,
    )

    shape = tuple(shapes[0])
    common = dict(n_layers=layers, k=K, chunk_shape=shape, dtype=dtype,
                  device=device, pin=torch.device(device).type == "cuda", spill_layers=0, ring=1)
    if name == "bf16_l2l":
        return BudgetedActivationStore(host_budget=budget, mb_shapes=shapes, **common), None
    if name == "fp8_oracle":
        from benchmarks.harness.p3_realdata_codec import CodecPolicy, make_store

        store = make_store(
            CodecPolicy("block64_e4m3", "e4m3", group_size=64), host_budget=budget,
            n_layers=layers, mb_shapes=shapes, dtype=dtype, device=device, pin=common["pin"],
        )
        store.k = K
        return store, None
    if name != "int4_oracle":
        raise ValueError("unknown gate storage policy")
    from benchmarks.harness.g4_activation_codec import CodecActivationStore
    from soup_cli.utils import activation_codec

    payload = activation_codec.payload_chunk_bytes(shape, activation_codec.COARSE)
    scales = layers * K * activation_codec.scale_chunk_bytes(shape, activation_codec.COARSE)
    charge = activation_allocation_bytes(
        n_layers=layers, k=K, chunk_bytes=payload, spill_layers=0, ring=1, pin=common["pin"],
    ) + ((1 << (scales - 1).bit_length()) if common["pin"] else scales)
    lease = budget.reserve("gate.int4.g4_diagnostic", charge)
    try:
        return CodecActivationStore(codec=activation_codec.COARSE, **common), lease
    except BaseException:
        lease.close()
        raise


def _gradient_snapshot(
    parameters: dict[str, Any], *, capture: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Observe gradient coverage, with optional CPU copies only for successful pass artifacts."""
    checks, values = {}, {}
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None:
            checks[name] = {"present": False, "finite": False, "nonzero": False}
            continue
        checks[name] = {"present": True, **_tensor_check(gradient),
                        "nonzero": bool(gradient.detach().ne(0).any())}
        if capture and checks[name]["finite"]:
            values[name] = gradient.detach().to("cpu", copy=True)
    return values, checks


def _run_pass(
    session: Any, *, name: str, initial_adapter: Path, initial_sha256: str,
    ids: Sequence[Any], labels: Sequence[Any], targets: int, directory: Path,
    device: str, dtype: str, host_budget_bytes: int, reference: _LayerArchive | None = None,
    production: bool = False, profile_id: str = "deepseek-v3",
) -> tuple[dict[str, Any], dict[str, Any], _LayerArchive | None]:
    import torch

    from benchmarks.harness.p3_deepseek_oracle import DeepseekOracleProbe
    from benchmarks.harness.p3_deepseek_quality_model import (
        load_initial_adapter,
        trainable_parameters,
    )
    from benchmarks.harness.p3_deepseek_quality_prepare import (
        cgroup_observation,
        require_memory_evidence,
    )
    from benchmarks.harness.p3_fp8_smoke import MemoryObservation
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.l2l_step import capture_layer_call, count_loads, model_parts

    directory.mkdir(parents=True, exist_ok=False)
    row: dict[str, Any] = {
        "status": "FAILED", "name": name, "profile_id": profile_id, "optimizer_applied": False,
        "valid_target_count": targets, "initial_adapter_sha256": initial_sha256,
        "backend_requested": FLOOR_BACKEND if name == "ga_floor" else MAIN_BACKEND,
        "routing_guard": "production_L2LStep_enabled" if "l2l" in name or "oracle" in name
        else "ordinary_streamed_GA",
        "host_budget_bytes": host_budget_bytes,
        "instrumentation": "synchronous per-pair CPU transfer, SHA256 and disk archive; "
        "included in wall time, not a training speed measurement",
    }
    model, runtime = session.model, session.runtime
    budget = HostMemoryBudget(host_budget_bytes)
    store = lease = oracle = observer = archive = None
    attention = None
    gradients: dict[str, Any] = {}
    started = time.perf_counter()
    adapter_before = None
    parameters: dict[str, Any] = {}
    losses: list[float] = []
    counters = None
    memory = MemoryObservation(torch.device(device).type, scope=(
        "Whole gate pass including gradient snapshots, archive I/O, diagnostics and cleanup. "
        "RSS is sampled; CUDA peaks are allocator observations, not activation attribution."))
    memory_started = False
    try:
        require_ampere_cuda(device)
        memory.__enter__()
        memory_started = True
        if load_initial_adapter(model, initial_adapter, profile_id=profile_id) != initial_sha256:
            raise ValueError("initial adapter file changed between gate passes")
        model.zero_grad(set_to_none=True)
        model.train()
        random.seed(SEED)
        torch.manual_seed(SEED)
        parameters = trainable_parameters(model)
        adapter_before = _adapter_digest(parameters)
        row["adapter_before_sha256"] = adapter_before
        layers = len(model_parts(model)[1].layers)
        if any(not _tensor_check(value)["finite"] for value in parameters.values()):
            raise ValueError("nonfinite initial adapter")
        with (attention_execution(row["backend_requested"], device=device,
                                  profile_id=profile_id) as attention,
              torch.autocast(device_type=torch.device(device).type, enabled=False),
              count_loads(runtime) as counters):
            if name in ("ga", "ga_repeat", "ga_floor"):
                observer = _LayerFiniteObservation(model)
                losses.clear()
                for batch, target in zip(ids, labels):
                    loss = model(input_ids=batch, labels=target, num_items_in_batch=targets).loss
                    if not bool(torch.isfinite(loss).all()):
                        raise ValueError("nonfinite ordinary GA loss")
                    losses.append(float(loss.detach()))
                    loss.backward()
                row["layer_forward_counts"] = list(observer.counts)
                if any(count < K for count in observer.counts):
                    raise ValueError("ordinary GA is missing an executed decoder layer")
            else:
                calls = [capture_layer_call(model, batch, draws=False) for batch in ids]
                shapes = [(1, MICROBATCH_LENGTH, int(session.config.hidden_size))] * K
                store, lease = _store(
                    name, budget=budget, shapes=shapes, layers=layers,
                    dtype=getattr(torch, dtype), device=device,
                )
                archive = _LayerArchive(directory, layers, reference=reference)
                if name in ("fp8_oracle", "int4_oracle"):
                    oracle = DeepseekOracleProbe(model, shapes, budget)
                step = _stepper(model, runtime, calls, archive, oracle)
                measured = step.run(ids, store, labels=labels, n_items=targets, calls=calls)
                losses.extend(float(loss.detach()) for loss in measured)
                if oracle is not None:
                    row["oracle"] = oracle.report()
                    oracle.require_complete()
                row["layers"] = archive.table()
                row["archive_io_seconds"] = archive.io_seconds
                row["archive_read_write_bytes"] = archive.io_bytes
                if any(len(layer["pairs"]) != K for layer in row["layers"]):
                    raise ValueError("incomplete executed decoder recompute coverage")
            row["losses"] = losses
            if len(losses) != K or not all(_finite(value) for value in losses):
                raise ValueError("missing or nonfinite per-microbatch loss")
            attention.require_complete(dtype=dtype, backward=True, min_calls=layers * K)
            gradients, checks = _gradient_snapshot(parameters)
            row["gradient_checks"] = checks
            if (not parameters or set(gradients) != set(parameters)
                    or (profile_id == "deepseek-v3"
                        and any(not check["nonzero"] for check in checks.values()))):
                raise ValueError("attention adapter gradient coverage violates the profile policy")
            if any(value.grad is not None for value in model.parameters()
                   if not value.requires_grad):
                raise ValueError("frozen base received a gradient")
            row["gradient_artifact"] = _save_tensors(directory / "gradients.pt", gradients)
        row["runtime_load_deltas"] = asdict(counters)
        _require_pass_attention(
            name, {**row, "attention": attention.report()}, dtype=dtype,
            min_calls=layers * K, production=production, profile_id=profile_id,
        )
        if any(row["runtime_load_deltas"][key] <= 0 for key in (
            "layer_loads", "large_loads", "expert_fetches", "expert_bytes_read",
        )):
            raise RuntimeError("gate did not execute the production per-expert disk runtime")
        row["status"] = "COMPLETE"
    except BaseException as error:
        row["error"] = {"type": type(error).__name__, "message": str(error)}
        row["interrupted"] = isinstance(error, (KeyboardInterrupt, SystemExit))
        traceback.clear_frames(error.__traceback__)
        gradients = {}
    finally:
        cleanup_errors: list[dict[str, str]] = []
        diagnostic_errors: list[dict[str, str]] = []

        def diagnostic(operation: str, body: Any) -> None:
            try:
                body()
            except BaseException as error:
                diagnostic_errors.append({"operation": operation, "type": type(error).__name__,
                                          "message": str(error)})

        def observe_tail() -> None:
            row["losses"] = losses
            if counters is not None:
                row["runtime_load_deltas"] = asdict(counters)
            if parameters and "gradient_checks" not in row:
                _, row["gradient_checks"] = _gradient_snapshot(parameters, capture=False)

        def observe_attention() -> None:
            if attention is not None:
                row["attention"] = attention.report()

        def observe_layer_counts() -> None:
            row["layer_forward_counts"] = list(observer.counts)

        def observe_archive() -> None:
            row["layers"] = archive.table()
            row["archive_io_seconds"] = archive.io_seconds
            row["archive_read_write_bytes"] = archive.io_bytes

        def observe_resource(resource_name: str, resource: Any, *, closed: bool) -> None:
            if hasattr(resource, "report"):
                row[resource_name + "_after_close" if closed else resource_name] = resource.report()
            elif resource_name == "store" and not closed:
                row["store"] = {"class": type(resource).__name__, "stats": asdict(resource.stats),
                                "dtype": str(resource.dtype), "k": resource.k,
                                "n_layers": resource.n_layers}
                if hasattr(resource, "host_bytes"):
                    row["store"]["storage"] = resource.host_bytes()
                if hasattr(resource, "codec_stats"):
                    resource.flush_timing()
                    row["store"]["codec_stats"] = asdict(resource.codec_stats)
            if closed and resource_name == "store" and hasattr(resource, "report"):
                value = row["store_after_close"]
                if (not value["cleanup"]["closed"] or value["owned_host_storage_bytes"]
                        or value["owned_device_storage_bytes"] or value["charged_host_bytes"]):
                    cleanup_errors.append({"resource": "store", "type": "CleanupFailure",
                                           "message": "owned storage remains after close"})
                    session.retained_resources.extend((resource, budget))
                    raise RuntimeError("compressed store retained owned storage or reservation")
            if closed and resource_name == "oracle":
                value = row["oracle_after_close"]
                if (not value["closed"] or value["records_remaining"]
                        or value["pending_replays"] or value["host_reserved_bytes"]):
                    cleanup_errors.append({"resource": "oracle", "type": "CleanupFailure",
                                           "message": "replay storage remains after close"})
                    session.retained_resources.extend((resource, budget))
                    raise RuntimeError("Oracle cleanup retained replay state or reservation")

        try:
            diagnostic("pass_tail", observe_tail)
            diagnostic("attention.report", observe_attention)
            if archive is not None:
                diagnostic("archive.table", observe_archive)
        finally:
            # Diagnostics may OOM. Their errors must never skip closing a live resource.
            for resource_name, resource in (("observer", observer), ("oracle", oracle),
                                            ("store", store), ("lease", lease)):
                if resource is None:
                    continue
                if resource_name == "lease" and cleanup_errors:
                    session.retained_resources.extend((resource, budget))
                    continue
                try:
                    try:
                        if resource_name == "observer":
                            diagnostic("observer.counts", observe_layer_counts)
                        elif resource_name in ("oracle", "store"):
                            diagnostic(resource_name + ".report", lambda: observe_resource(
                                resource_name, resource, closed=False,
                            ))
                    finally:
                        resource.close()
                except BaseException as error:
                    cleanup_errors.append({"resource": resource_name,
                                           "type": type(error).__name__, "message": str(error)})
                    session.retained_resources.extend((resource, budget))
                else:
                    if resource_name in ("oracle", "store"):
                        diagnostic(resource_name + ".report_after_close", lambda: observe_resource(
                            resource_name, resource, closed=True,
                        ))
        row["host_budget_after_close_bytes"] = budget.used_bytes
        row["adapter_unchanged"] = False

        def observe_adapter() -> None:
            if parameters:
                row["adapter_after_sha256"] = _adapter_digest(parameters)
                row["adapter_unchanged"] = row["adapter_after_sha256"] == adapter_before

        diagnostic("adapter_digest", observe_adapter)
        diagnostic("zero_grad", lambda: model.zero_grad(set_to_none=True))
        if memory_started:
            diagnostic("memory", lambda: memory.__exit__(None, None, None))
            diagnostic("memory_evidence", lambda: require_memory_evidence(
                memory.result, cuda=torch.device(device).type == "cuda"))
        row["memory"] = memory.result
        diagnostic("cgroup", lambda: row.update(cgroup=cgroup_observation()))
        row["resources_are_not_summed"] = True
        row["cleanup_errors"] = cleanup_errors
        row["diagnostic_errors"] = diagnostic_errors
        row["wall_seconds"] = time.perf_counter() - started
        if cleanup_errors or diagnostic_errors or budget.used_bytes or not row["adapter_unchanged"]:
            row["status"] = "FAILED"
            gradients = {}
    return row, gradients, archive


@contextmanager
def _deterministic(device: str) -> Iterator[dict[str, Any]]:
    import torch

    from benchmarks.harness.p3_deepseek_fp8_smoke import _preserve_process_state

    with _preserve_process_state(device) as evidence:
        torch.use_deterministic_algorithms(True, warn_only=False)
        evidence["execution"] = {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }
        yield evidence


def _source_hashes(*, profile_id: str = "deepseek-v3") -> dict[str, str]:
    root = Path(__file__).parents[2]
    names = (
        "benchmarks/harness/p3_deepseek_quality_gate.py",
        "benchmarks/harness/p3_deepseek_quality_attention.py",
        "benchmarks/harness/p3_deepseek_quality_model.py",
        "benchmarks/harness/p3_deepseek_quality_prepare.py",
        "benchmarks/harness/p3_deepseek_quality_profiles.py",
        "benchmarks/harness/p3_fp8_smoke.py", "benchmarks/harness/host_box.py",
        "benchmarks/harness/p3_deepseek_quality_data.py",
        "benchmarks/harness/p3_deepseek_oracle.py",
        "benchmarks/harness/p3_deepseek_fp8_smoke.py",
        "benchmarks/harness/p3_realdata_codec.py",
        "benchmarks/harness/g4_rule.py",
        "benchmarks/harness/g4_activation_codec.py",
        "benchmarks/harness/l2l_mistral_stand.py",
        "benchmarks/harness/l2l_schedule.py", "benchmarks/harness/p3_stage2_dataset.py",
        "benchmarks/harness/p3_stage2_training.py",
        "src/soup_cli/utils/l2l_step.py", "src/soup_cli/utils/l2l_rng.py",
        "src/soup_cli/utils/l2l_activations.py", "src/soup_cli/utils/budgeted_activations.py",
        "src/soup_cli/utils/activation_codec.py", "src/soup_cli/utils/layer_stream_runtime.py",
        "src/soup_cli/utils/expert_stream.py",
        "src/soup_cli/utils/layer_shard.py", "src/soup_cli/utils/host_memory_budget.py",
        policy_identity(profile_id=profile_id)["path"],
    )
    for name in names:
        if name.endswith(".py"):
            module_name = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
            imported = importlib.import_module(module_name)
            actual = getattr(imported, "__file__", None)
            if actual is None or os.path.realpath(actual) != os.path.realpath(root / name):
                raise ValueError(f"gate imported module is outside the admitted checkout: {name}")
    return {name: _sha256(root / name) for name in names}


def _derived_nonfinite_paths(report: dict[str, Any]) -> set[str]:
    """Classify publication diagnostics, never substitute for raw-tensor replay."""
    paths: list[str] = []
    _json_safe(report.get("a3", {}).get("rule"), path="$.a3.rule", nonfinite=paths)
    for name in ("bf16_l2l", "fp8_oracle", "int4_oracle"):
        row = report.get("passes", {}).get(name, {})
        for index, layer in enumerate(row.get("layers", [])):
            for kind in ("restore", "output", "branch"):
                _json_safe(layer.get(kind), path=f"$.passes.{name}.layers[{index}].{kind}",
                           nonfinite=paths)
        if "oracle" in name:
            _json_safe(row.get("gradient_agreement"),
                       path=f"$.passes.{name}.gradient_agreement", nonfinite=paths)
    _json_safe(report.get("candidate_gates"), path="$.candidate_gates", nonfinite=paths)
    _json_safe(report.get("control", {}).get("gates"), path="$.control.gates", nonfinite=paths)
    return set(paths)


def _publish(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    nonfinite: list[str] = []
    safe = _json_safe(report, nonfinite=nonfinite)
    safe.setdefault("profile_id", "deepseek-v3")
    diagnostic = safe["profile_id"] == "moonlight-16b-a3b"
    safe["nonfinite_evidence_fields"] = sorted(nonfinite) if diagnostic else nonfinite
    # Frozen ratios can be undefined for finite raw observations. Preserve their numerical
    # verdict; the loader must reconstruct every null and its exact path from raw evidence.
    technical = set(nonfinite) - _derived_nonfinite_paths(report) if diagnostic else nonfinite
    if technical:
        safe["verdict"] = "NO_VERDICT"
        if diagnostic:
            safe["status"] = "FAILED"
    for name, row in safe["passes"].items():
        row.setdefault("profile_id", safe["profile_id"])
        if row.get("status") != "NOT_APPLICABLE":
            _write_json(_child(output, name + "/pass.json"), row)
    failed_passes = {name: row for name, row in safe["passes"].items()
                     if row.get("status") == "FAILED"}
    if safe["status"] == "FAILED" or failed_passes:
        _write_json(_child(output, "failure.json"), {
            "profile_id": safe["profile_id"],
            "error": safe.get("error"), "failed_passes": failed_passes,
            "detail": "failed observations are preserved; no pass or floor was substituted",
        })
    _write_json(_child(output, "result.json"), safe)
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            relative = os.path.relpath(path, output).replace("\\", "/")
            admitted = _child(output, relative)
            files[relative] = {"bytes": admitted.stat().st_size, "sha256": _sha256(admitted)}
    _write_json(_child(output, "manifest.json"), {
        "schema": SCHEMA, "profile_id": safe["profile_id"],
        "status": safe["status"], "verdict": safe["verdict"],
        "qualification_only": safe["qualification_only"], "files": files,
        "manifest_policy": (
            "exclusive writes; all input, raw tensor and result files precede manifest"),
    })
    return safe


def _record_a3(
    report: dict[str, Any], results: dict[str, Any], dtype: str,
) -> None:
    floor_row = report["passes"].get("ga_floor", {})
    observed_floor = "ga_floor" in results and dtype == "bfloat16"
    report["floor"] = {
        "observed": observed_floor, "backend": FLOOR_BACKEND,
        "main_backend": MAIN_BACKEND, "factor": 1.5,
        "unavailable_reason": None if observed_floor else floor_row.get("error", {
            "message": "independent CUDA BF16 SDPA execution not observed",
        }),
    }
    report["a3"] = assess_a3(
        results["ga"], results["ga_repeat"], results["bf16_l2l"],
        losses=report["passes"]["ga"]["losses"],
        repeat_losses=report["passes"]["ga_repeat"]["losses"],
        l2l_losses=report["passes"]["bf16_l2l"]["losses"], dtype=dtype,
        floor_gradients=results.get("ga_floor"), floor_losses=floor_row.get("losses"),
        floor_observed=observed_floor,
    )


def _pass_metrics(report: dict[str, Any], name: str) -> dict[str, Any]:
    row = report["passes"][name]
    if row.get("status") == "COMPLETE" and row.get("forward_losses_equal"):
        return assess_metrics(
            row["layers"], row["gradient_agreement"],
            expected_layers=report["layers"], expected_tensors=report["trainable_names"],
        )
    return _no_metrics("pass unavailable or original forward loss differs")


def run_gate(
    *, source: Path, cache: Path, initial_adapter: Path, output: Path,
    window: dict[str, Any] | None = None, pack: Path | None = None,
    pack_sha256: str | None = None, device: str = "cuda", dtype: str = "bfloat16",
    prepare: bool = False, fixture: bool = False, host_budget_bytes: int = HOST_BUDGET_BYTES,
    profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    """Preserve numerical verdicts; only Moonlight completes all diagnostic passes after FAIL."""
    profile = get_profile(profile_id)
    diagnostic = profile.profile_id == "moonlight-16b-a3b"
    output = Path(os.path.realpath(output))
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": SCHEMA, "profile_id": profile_id, "status": "FAILED", "verdict": "NO_VERDICT",
        "qualification_only": fixture or dtype != "bfloat16" or device != "cuda",
        "source": str(source), "cache": str(cache), "dtype": dtype, "device": device,
        "seed": SEED, "optimizer_updates": 0, "passes": {},
        "a3": {"status": "NO_VERDICT", "detail": "required passes not completed"},
        "candidate_gates": _no_metrics("candidate not measured"),
        "control": {"gates": _no_metrics("int4 not measured"),
                    "discrimination": {"status": "NO_VERDICT", "not_failed": list(GATES)}},
        "routing_claim": "Oracle replays original routes; no natural routing-stability claim",
    }
    started = time.perf_counter()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        # Must precede CUDA initialization. Never quietly accept an incompatible existing value.
        if workspace is not None and workspace != ":4096:8":
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA starts")
        import torch
        if workspace is None:
            if device == "cuda" and torch.cuda.is_initialized():
                raise ValueError("fresh CUDA process required to set CUBLAS_WORKSPACE_CONFIG")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        from benchmarks.harness.p3_deepseek_quality_model import (
            open_session,
            require_production_source,
            trainable_parameters,
        )

        if dtype not in ("bfloat16", "float32") or (not fixture and (
            dtype != "bfloat16" or device != "cuda"
        )):
            raise ValueError("production gate requires CUDA/BF16; FP32 is fixture-only")
        report["attention_policy"] = policy_identity(profile_id=profile_id)
        hardware = require_ampere_cuda(device)
        if type(host_budget_bytes) is not int or host_budget_bytes < 1:
            raise ValueError("host budget must be positive integer bytes")
        if window is None:
            from benchmarks.harness.p3_deepseek_quality_data import load_pack

            if pack is None or pack_sha256 is None:
                raise ValueError("gate requires a verified pack and its SHA256")
            windows, manifest = load_pack(pack, pack_sha256, length=1024, steps=1,
                                          profile_id=profile_id)
            window = windows[0]
            report["pack"] = {"path": str(pack), "manifest_sha256": pack_sha256,
                              "manifest": manifest}
            if not fixture and manifest.get("test_only") is not False:
                raise ValueError("production gate refuses test-only training packs")
        elif pack is not None or pack_sha256 is not None or not fixture:
            raise ValueError("direct windows are explicit Python fixture inputs only")
        inputs, targets, identity = gate_prefix(window)
        report["input"] = identity
        _write_json(_child(output, "gate-input.json"), {
            "profile_id": profile_id, "identity": identity, "input_ids": inputs, "labels": targets,
        })
        report["initial_adapter_sha256"] = _sha256(initial_adapter)
        report["source_files_sha256"] = (
            _source_hashes() if profile_id == "deepseek-v3"
            else _source_hashes(profile_id=profile_id)
        )
        report["environment"] = {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "packages": {name: importlib.metadata.version(name) for name in (
                "transformers", "peft", "bitsandbytes", "accelerate", "safetensors",
            )},
        }
        report["environment"]["gpu"] = {
            key: hardware[key] for key in ("name", "capability", "bf16_supported", "native_bf16")
        }
        _write_json(_child(output, "execution-start.json"), {
            "profile_id": profile_id,
            "admission_sufficient": False, "partial_execution": _json_safe(report, nonfinite=[]),
        })
        with _deterministic(device) as deterministic:
            report["determinism"] = deterministic
            with open_session(
                source, cache, device=device, dtype=dtype, prepare=prepare, profile_id=profile_id,
            ) as session:
                report["model"] = session.metadata
                if not fixture:
                    require_production_source(session.metadata["source"], profile_id=profile_id)
                parameters = trainable_parameters(session.model)
                report["trainable_names"] = sorted(parameters)
                report["layers"] = list(range(int(session.config.num_hidden_layers)))
                if len(report["layers"]) != len(session.model.get_base_model().model.layers):
                    raise ValueError("native config and streamed decoder layer counts differ")
                if any(token >= int(session.config.vocab_size)
                       for row in inputs for token in row[0]):
                    raise ValueError("pack token outside model vocabulary")
                if any(token >= int(session.config.vocab_size)
                       for row in targets for token in row[0] if token != -100):
                    raise ValueError("supervised label outside model vocabulary")
                _write_json(_child(output, "session-open.json"), {
                    "profile_id": profile_id,
                    "admission_sufficient": False, "model": session.metadata,
                    "trainable_names": report["trainable_names"], "layers": report["layers"],
                })
                ids = [torch.tensor(row, dtype=torch.long, device=device) for row in inputs]
                labels = [torch.tensor(row, dtype=torch.long, device=device) for row in targets]
                results, archives = {}, {}
                candidate_assessed = False
                arms = ["ga", "ga_repeat", "bf16_l2l"]
                if dtype == "bfloat16":
                    arms += ["ga_floor", "fp8_oracle", "int4_oracle"]
                for name in arms:
                    row, gradients, archive = _run_pass(
                        session, name=name, initial_adapter=initial_adapter,
                        initial_sha256=report["initial_adapter_sha256"], ids=ids, labels=labels,
                        targets=identity["valid_target_count"], directory=_child(output, name),
                        device=device, dtype=dtype, host_budget_bytes=host_budget_bytes,
                        reference=archives.get("bf16_l2l") if "oracle" in name else None,
                        production=not fixture, profile_id=profile_id,
                    )
                    report["passes"][name] = row
                    nonfinite: list[str] = []
                    _write_json(_child(output, name + "/pass-observation.json"), {
                        "profile_id": profile_id,
                        "admission_sufficient": False,
                        "pass": _json_safe(row, nonfinite=nonfinite),
                        "nonfinite_evidence_fields": nonfinite,
                    })
                    if row["status"] != "COMPLETE":
                        if row.get("interrupted"):
                            raise RuntimeError(f"gate pass {name} was interrupted")
                        if row["cleanup_errors"] or row["host_budget_after_close_bytes"]:
                            raise RuntimeError(f"gate pass {name} cleanup failed")
                        if not row["adapter_unchanged"]:
                            raise RuntimeError(f"gate pass {name} changed the shared adapter")
                        if diagnostic:
                            raise RuntimeError(f"technical gate pass {name} failed: "
                                               f"{row.get('error')}")
                        if name not in ("ga_floor", "int4_oracle", "fp8_oracle"):
                            raise RuntimeError(
                                f"required gate pass {name} failed: {row.get('error')}")
                        report["stopped_after"] = name
                        break
                    results[name] = gradients
                    archives[name] = archive
                    if name == "ga_repeat" and (
                        row["losses"] != report["passes"]["ga"]["losses"]
                        or any(not torch.equal(results["ga"][key], gradients[key])
                               for key in report["trainable_names"])
                    ) and not diagnostic:
                        report["a3"] = {"status": "NO_VERDICT",
                                        "detail": "bitwise ordinary GA repeat failed"}
                        raise ValueError("ordinary GA repeat failed; L2L is not judged")
                    if name == "ga_floor":
                        _record_a3(report, results, dtype)
                        if report["a3"]["status"] != "PASS" and not diagnostic:
                            report["stopped_after"] = name
                            break
                    if name in ("fp8_oracle", "int4_oracle"):
                        row["forward_losses_equal"] = row["losses"] == report["passes"][
                            "bf16_l2l"
                        ]["losses"]
                        try:
                            row["gradient_agreement"] = gradient_agreement(
                                results["bf16_l2l"], gradients,
                            )
                        except ValueError as error:
                            row["status"] = "FAILED"
                            row["error"] = {"type": type(error).__name__, "message": str(error)}
                            if name == "fp8_oracle":
                                break
                        if name == "fp8_oracle":
                            report["candidate_gates"] = _pass_metrics(report, name)
                            candidate_assessed = True
                            if not diagnostic and any(
                                value["status"] != "PASS"
                                for value in report["candidate_gates"].values()
                            ):
                                report["stopped_after"] = name
                                break
                if dtype == "float32":
                    for name in ("fp8_oracle", "int4_oracle", "ga_floor"):
                        report["passes"][name] = {
                            "status": "NOT_APPLICABLE", "detail": "FP32 identity qualification; "
                            "frozen block64 E4M3 study is BF16 only, never silently cast",
                        }
                for name in ("fp8_oracle", "int4_oracle"):
                    report["passes"].setdefault(name, {
                        "status": "NOT_APPLICABLE", "detail": "not run after earlier gate refusal",
                    })
                if "floor" not in report:
                    _record_a3(report, results, dtype)
                for name, target in (("fp8_oracle", "candidate_gates"),
                                     ("int4_oracle", "control")):
                    if target == "candidate_gates" and candidate_assessed:
                        continue
                    gates = _pass_metrics(report, name)
                    if target == "control":
                        report[target] = {"gates": gates,
                                          "discrimination": assess_discrimination(gates)}
                    else:
                        report[target] = gates
                report["verdict"] = overall_verdict(
                    report["a3"], report["candidate_gates"], report["control"]["discrimination"],
                    eligible=not report["qualification_only"],
                )
                report["status"] = (
                    "FAILED" if any(row.get("status") == "FAILED"
                                    for row in report["passes"].values()) else "COMPLETE"
                )
        report["session_closed"] = True
    except BaseException as error:
        report["status"] = "FAILED"
        report["verdict"] = "NO_VERDICT"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.clear_frames(error.__traceback__)
    finally:
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
        report["cublas_environment_restored"] = (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") == workspace
        )
    report["wall_seconds"] = time.perf_counter() - started
    return _publish(output, report)


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError("gate admission: " + detail)


def _require_profile(value: dict[str, Any], profile_id: str, *, detail: str) -> None:
    _require(value.get("profile_id", "deepseek-v3") == profile_id,
             detail + " profile differs or is missing")


def _require_attention_contract(
    result: dict[str, Any], *, profile_id: str = "deepseek-v3",
) -> None:
    _require_profile(result, profile_id, detail="gate")
    _require(result.get("attention_policy") == policy_identity(profile_id=profile_id),
             "immutable attention backend policy differs or is missing")
    deterministic = result.get("determinism", {})
    _require(deterministic.get("restored") is True and deterministic.get("execution") == {
        "algorithms_enabled": True, "warn_only": False, "allow_tf32": False,
        "cudnn_allow_tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
    }, "strict deterministic execution/restoration evidence missing")
    environment = result.get("environment", {})
    _require(environment.get("cublas_workspace_config") == ":4096:8"
             and result.get("cublas_environment_restored") is True,
             "deterministic CUBLAS workspace/restoration evidence missing")
    hardware = environment.get("gpu", {})
    capability = hardware.get("capability")
    _require(isinstance(capability, list) and len(capability) == 2
             and all(type(value) is int and value >= 0 for value in capability)
             and tuple(capability) >= (8, 0) and hardware.get("native_bf16") is True
             and hardware.get("bf16_supported") is True
             and isinstance(hardware.get("name"), str) and bool(hardware["name"]),
             "native-BF16 Ampere hardware evidence missing")


def _require_pass_attention(
    name: str, row: dict[str, Any], *, dtype: str = "bfloat16", min_calls: int,
    production: bool = False, hardware: dict[str, Any] | None = None,
    profile_id: str = "deepseek-v3",
) -> None:
    profile = get_profile(profile_id)
    backend = FLOOR_BACKEND if name == "ga_floor" else MAIN_BACKEND
    _require(row.get("backend_requested") == backend, f"attention backend changed: {name}")
    observation = row.get("attention", {})
    require_execution_evidence(
        observation, backend=backend, dtype=dtype, backward=True, min_calls=min_calls,
        profile_id=profile_id,
    )
    if hardware is not None:
        _require(all(observation["hardware"].get(key) == hardware.get(key) for key in (
            "name", "capability", "bf16_supported", "native_bf16",
        )), f"attention hardware differs from gate environment: {name}")
    for call in observation["unique_calls"]:
        query, key, value = (call[field + "_shape"] for field in ("query", "key", "value"))
        _require(all(len(shape) == 4 and shape[0] == 1 and shape[-2] == MICROBATCH_LENGTH
                     for shape in (query, key, value)),
                 f"attention does not cover unchanged B1/L64 gate workload: {name}")
        if production:
            _require(query == key == [1, profile.attention_heads, MICROBATCH_LENGTH,
                                      profile.qk_head_dim]
                     and value == [1, profile.attention_heads, MICROBATCH_LENGTH,
                                   profile.v_head_dim],
                     f"attention MLA heads{profile.attention_heads} Q/K{profile.qk_head_dim} "
                     f"V{profile.v_head_dim} execution missing: {name}")


def _oracle_complete(row: dict[str, Any], expected_pairs: int) -> bool:
    oracle = row.get("oracle", {})
    closed = row.get("oracle_after_close", {})
    return bool(
        expected_pairs > 0 and oracle.get("complete") is True
        and oracle.get("expected_pairs") == expected_pairs
        and oracle.get("counts") == {name: expected_pairs for name in (
            "native_forward", "oracle_recompute", "oracle_vjp",
        )}
        and oracle.get("forced_routing") is True
        and oracle.get("natural_routing_measured") is False
        and oracle.get("records_remaining") == oracle.get("pending_replays") == 0
        and oracle.get("failure") is None and not oracle.get("cleanup_errors")
        and closed.get("closed") is True and closed.get("host_reserved_bytes") == 0
        and closed.get("records_remaining") == closed.get("pending_replays") == 0
        and closed.get("owned_host_storage_bytes") == 0
        and closed.get("owned_device_storage_bytes") == 0
        and closed.get("cleanup", {}).get("original_forwards_restored") is True
        and closed.get("cleanup", {}).get("hooks_removed") is True
        and not closed.get("cleanup_errors")
    )


def _require_layer_coverage(
    rows: Any, *, layers: Sequence[int], diagnostic: bool = False,
) -> None:
    _require(isinstance(rows, list) and len(rows) == len(layers)
             and all(isinstance(row, dict) for row in rows)
             and [row.get("layer") for row in rows] == list(layers),
             "exact raw boundary layer coverage missing")
    for row in rows:
        pairs = row.get("pairs")
        _require(isinstance(pairs, list) and len(pairs) == K
                 and all(isinstance(pair, dict) for pair in pairs)
                 and [pair.get("microbatch") for pair in pairs] == list(range(K))
                 and all(pair.get("rows") == MICROBATCH_LENGTH
                         and pair.get("finite") is True and pair.get("forward_equal") is True
                         for pair in pairs), "incomplete/nonfinite raw boundary pair coverage")
        for kind in ("restore", "output", "branch"):
            metric = row.get(kind)
            _require(isinstance(metric, dict) and metric.get("rows") == PREFIX_TOKENS
                     and all(key in metric and (
                         _finite(metric[key]) and metric[key] >= 0
                         or diagnostic and metric[key] is None
                     ) for key in (
                         "rms_row_rel", "worst_row_rel", "rel_l2", "max_abs",
                         "max_abs_over_absmax", "err_sq", "ref_sq", "row_rel_sq_sum",
                     )), "incomplete/nonfinite raw boundary metric coverage")


def _load_gate(
    path: Path, sha256: str, *, profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    """Admit an immutable production receipt, recomputing A3/E1/E2/E3 from its tensors.

    The training caller must then match ``model.source``, ``model.cache``,
    ``initial_adapter_sha256`` and ``pack.manifest_sha256`` to its live inputs.
    Moonlight numerical FAIL/NO_VERDICT is diagnostic, never a fabricated PASS.
    """
    from benchmarks.harness.p3_deepseek_quality_data import _tokenizer_contract
    from benchmarks.harness.p3_deepseek_quality_model import require_production_source

    profile = get_profile(profile_id)
    diagnostic = profile.profile_id == "moonlight-16b-a3b"
    verdicts = ("PASS", "FAIL", "NO_VERDICT") if diagnostic else ("PASS",)

    path = Path(path)
    _require(path.name == "manifest.json", "--gate must name manifest.json")
    _require(isinstance(sha256, str) and len(sha256) == 64 and _sha256(path) == sha256,
             "manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require_profile(manifest, profile_id, detail="manifest")
    _require(manifest.get("schema") == SCHEMA and manifest.get("status") == "COMPLETE"
             and manifest.get("verdict") in verdicts
             and manifest.get("qualification_only") is False,
             "production COMPLETE/PASS receipt required; qualification evidence is not admission")
    root = path.parent
    files = manifest.get("files")
    _require(isinstance(files, dict) and "result.json" in files and "gate-input.json" in files,
             "missing result/input artifact inventory")
    actual = {os.path.relpath(item, root).replace("\\", "/") for item in root.rglob("*")
              if item.is_file() and item != path}
    _require(actual == set(files), "artifact inventory has missing or unlisted files")
    for relative, descriptor in files.items():
        artifact = _child(root, relative)
        _require(isinstance(descriptor, dict) and artifact.is_file()
                 and artifact.stat().st_size == descriptor.get("bytes")
                 and _sha256(artifact) == descriptor.get("sha256"),
                 f"artifact size/hash mismatch: {relative}")
    if diagnostic:
        _require(not {"failure.json", "FAILED.json"}.intersection(files),
                 "technical failure artifact cannot admit a diagnostic gate")
    result = json.loads(_child(root, "result.json").read_text(encoding="utf-8"))
    _require_profile(result, profile_id, detail="result")
    _require(result.get("schema") == SCHEMA and result.get("status") == "COMPLETE"
             and result.get("verdict") == manifest.get("verdict")
             and result.get("qualification_only") is False
             and result.get("dtype") == "bfloat16" and result.get("device") == "cuda"
             and result.get("seed") == SEED and result.get("optimizer_updates") == 0
             and result.get("session_closed") is True
             and (diagnostic or result.get("nonfinite_evidence_fields") == []),
             "result is not complete finite production evidence")
    if diagnostic:
        _require(isinstance(result.get("initial_adapter_sha256"), str)
                 and re.fullmatch(r"[0-9a-f]{64}", result["initial_adapter_sha256"]) is not None,
                 "initial adapter identity missing or malformed")
    _require_attention_contract(result, profile_id=profile_id)
    require_production_source(result["model"]["source"], profile_id=profile_id)
    cache = result["model"]["cache"]
    precision = result["model"]["precision"]
    _require_profile(result["model"], profile_id, detail="model")
    for name in ("source", "cache"):
        _require_profile(result["model"][name], profile_id, detail=name)
    _require(cache.get("dtype") == "bfloat16" and cache.get("quant") == "nf4"
             and cache.get("double_quant") is False and cache.get("expert_layout") == "per_expert"
             and cache.get("tier") == "disk" and precision.get("compute_dtype") == "bfloat16"
             and precision.get("trainable_dtype") == "float32"
             and precision.get("autocast") is False
             and precision.get("attention_implementation") == "sdpa",
             "NF4 noDQ per-expert disk/BF16 precision contract differs")
    source_hashes = (_source_hashes() if profile_id == "deepseek-v3"
                     else _source_hashes(profile_id=profile_id))
    _require(result.get("source_files_sha256") == source_hashes,
             "gate executing source hashes differ from this checkout")
    pack = result.get("pack", {})
    _require(pack.get("manifest", {}).get("test_only") is False
             and isinstance(pack.get("manifest_sha256"), str), "production pack identity missing")
    _require(pack["manifest"].get("tokenizer", {}).get("test_only") is False
             and pack["manifest"].get("training_source_row_count") == 7801,
             "production tokenizer/source-row identity missing")
    _require_profile(pack["manifest"], profile_id, detail="pack")
    _require_profile(pack["manifest"]["tokenizer"], profile_id, detail="tokenizer")
    if diagnostic:
        _tokenizer_contract(pack["manifest"], profile_id=profile_id)
        contract = result["attention_policy"]["contract"]
        _require(pack["manifest_sha256"] == contract["training_pack_manifest_sha256"],
                 "pack differs from immutable attention policy")
        _require(result["model"]["source"]["config_sha256"]
                 == contract["model"]["config_sha256"],
                 "source config differs from immutable attention policy")
        _require(cache.get("source_fingerprint")
                 == result["model"]["source"].get("cache_source_fingerprint")
                 and cache.get("source_parameters") == profile.parameters
                 and cache.get("n_layers") == profile.shape["num_hidden_layers"]
                 and isinstance(cache.get("index_sha256"), str)
                 and re.fullmatch(r"[0-9a-f]{64}", cache["index_sha256"]) is not None,
                 "cache/source identity differs or is missing")
        _require(precision.get("lora") == {
            "rank": 2, "alpha": 4, "dropout": 0.0, "targets": list(profile.lora_targets),
        }, "profile rank-two attention adapter contract differs")
        environment = result.get("environment", {})
        _require(all(isinstance(environment.get(key), str) and bool(environment[key])
                     for key in ("torch", "cuda"))
                 and set(environment.get("packages", {})) == {
                     "transformers", "peft", "bitsandbytes", "accelerate", "safetensors",
                 } and all(isinstance(value, str) and bool(value)
                           for value in environment["packages"].values()),
                 "software/CUDA environment identity missing")
    input_record = json.loads(_child(root, "gate-input.json").read_text(encoding="utf-8"))
    _require_profile(input_record, profile_id, detail="input")
    identity = input_record.get("identity", {})
    _require(identity == result.get("input") and identity.get("k") == K
             and identity.get("microbatch_length") == MICROBATCH_LENGTH
             and identity.get("batch_size") == 1
             and identity.get("input_tokens") == PREFIX_TOKENS
             and identity.get("pack_length") == 1024 and identity.get("window_index") == 0
             and identity.get("prefix_start") == 0 and identity.get("prefix_stop") == PREFIX_TOKENS,
             "gate did not measure window zero, first512, k8/L64")
    ids, labels = input_record.get("input_ids"), input_record.get("labels")
    _require(isinstance(ids, list) and isinstance(labels, list) and len(ids) == len(labels) == K,
             "microbatch input/label coverage missing")
    for batch, target in zip(ids, labels):
        _require(isinstance(batch, list) and isinstance(target, list)
                 and len(batch) == len(target) == 1
                 and len(batch[0]) == len(target[0]) == MICROBATCH_LENGTH
                 and all(type(value) is int and value >= 0 for value in batch[0])
                 and all(type(value) is int and (value >= 0 or value == -100)
                         for value in target[0]) and target[0][0] == -100,
                 "local first-token masks or B1/L64 integer input policy differs")
    counts = [sum(token != -100 for token in target[0][1:]) for target in labels]
    _require(identity.get("valid_target_count") == sum(counts) > 0
             and identity.get("per_microbatch_targets") == counts
             and identity.get("input_sha256") == _ids_sha256([x for row in ids for x in row[0]])
             and identity.get("labels_sha256") == _ids_sha256(
                 [x for row in labels for x in row[0]]),
             "prefix target denominator or input/label hash mismatch")
    return _verify_numerical_evidence(root, files, result, profile_id=profile_id)


def _verify_numerical_evidence(
    root: Path, files: dict[str, Any], result: dict[str, Any],
    *, profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    """Verify raw tensors independently of production checkpoint admission.

    This private function cannot admit a training run. ``load_gate`` alone binds
    the production source, precision, immutable inventory and workload before it
    calls this verifier. Tiny CPU tensor receipts exercise the same numerical
    recomputation without forging a full-model or CUDA admission.
    """
    import torch

    from benchmarks.harness import g4_activation_codec as metrics

    profile = get_profile(profile_id)
    diagnostic = profile.profile_id == "moonlight-16b-a3b"
    _require_attention_contract(result, profile_id=profile_id)
    nonfinite: list[str] = []
    _json_safe(result, nonfinite=nonfinite)
    _require(not nonfinite, "nonfinite numerical evidence fields")
    declared = result.get("nonfinite_evidence_fields")
    if diagnostic:
        _require(isinstance(declared, list) and all(isinstance(path, str) for path in declared)
                 and declared == sorted(set(declared)),
                 "malformed nonfinite diagnostic paths")
    else:
        _require(not declared, "nonfinite numerical evidence fields")
    derived_nonfinite: set[str] = set()

    def canonical(value: Any, *, path: str) -> Any:
        # Only independently reconstructed values can declare a derived diagnostic.
        # Matching this representation also rejects nulls where raw replay was finite.
        paths: list[str] = []
        safe = _json_safe(value, path=path, nonfinite=paths)
        _require(diagnostic or not paths, "nonfinite numerical metric recomputation")
        derived_nonfinite.update(paths)
        return safe

    def same_json(expected: Any, recorded: Any) -> bool:
        if isinstance(expected, dict):
            return (isinstance(recorded, dict) and expected.keys() == recorded.keys()
                    and all(same_json(value, recorded[key])
                            for key, value in expected.items()))
        if isinstance(expected, list):
            return (isinstance(recorded, list) and len(expected) == len(recorded)
                    and all(same_json(left, right) for left, right in zip(expected, recorded)))
        if type(expected) is bool or type(recorded) is bool:
            return type(expected) is type(recorded) and expected == recorded
        return expected == recorded

    def matches(value: Any, recorded: Any, *, path: str) -> bool:
        reconstructed = canonical(value, path=path)
        return (same_json(reconstructed, recorded) if diagnostic
                else reconstructed == recorded)

    _require(not result.get("error"), "gate retains an execution error")

    counts = result["input"]["per_microbatch_targets"]
    _require(isinstance(counts, list) and len(counts) == K
             and all(type(value) is int and 0 <= value < MICROBATCH_LENGTH for value in counts)
             and sum(counts) > 0, "invalid numerical evidence target counts")
    names = result.get("trainable_names")
    n_layers = result["model"]["cache"]["n_layers"]
    _require(type(n_layers) is int and n_layers > 0, "invalid numerical layer count")
    layers = list(range(n_layers))
    _require(result.get("layers") == layers and isinstance(names, list)
             and all(isinstance(name, str) for name in names)
             and names == sorted(set(names))
             and len(names) == n_layers * len(profile.lora_targets) * 2,
             "exact layer/attention-adapter tensor inventory missing")
    observed = []
    for name in names:
        match = re.search(
            r"\.layers\.(\d+)\.self_attn\.([^.]+)\.lora_([AB])\.default\.weight$", name,
        )
        _require(match is not None, "unexpected attention-adapter tensor name")
        observed.append((int(match[1]), match[2], match[3]))
    _require(set(observed) == {(layer, target, factor) for layer in layers
                              for target in profile.lora_targets for factor in ("A", "B")},
             "exact layer/attention-adapter tensor inventory differs from profile")
    passes = result.get("passes", {})
    required = ("ga", "ga_repeat", "bf16_l2l", "ga_floor", "fp8_oracle", "int4_oracle")
    _require(set(passes) == set(required), "required measured passes missing or unexpected")
    if diagnostic:
        expected_raw = {name + "/gradients.pt" for name in required}
        expected_raw.update(
            f"{arm}/{phase}/layer-{layer:03d}-mb-{mb:03d}.pt"
            for arm, phase in (("bf16_l2l", "forward"), ("bf16_l2l", "recompute"),
                               ("fp8_oracle", "recompute"), ("int4_oracle", "recompute"))
            for layer in layers for mb in range(K)
        )
        _require({name for name in files if name.endswith(".pt")} == expected_raw,
                 "exact raw gradient/boundary artifact inventory missing or unexpected")
    adapter_identity = passes["ga"].get("adapter_before_sha256")
    _require(isinstance(adapter_identity, str) and len(adapter_identity) == 64,
             "shared adapter parameter digest missing")
    gradients = {}
    for name in required:
        row = passes[name]
        _require_profile(row, profile_id, detail="pass " + name)
        if diagnostic and result.get("qualification_only") is False:
            relative = name + "/pass.json"
            _require(relative in files and same_json(json.loads(
                _child(root, relative).read_text(encoding="utf-8")), row),
                f"pass receipt differs from result: {name}")
        _require(row.get("status") == "COMPLETE" and row.get("optimizer_applied") is False
                 and not row.get("error") and not row.get("interrupted")
                 and row.get("adapter_unchanged") is True
                 and row.get("adapter_before_sha256") == row.get("adapter_after_sha256")
                 and row.get("adapter_before_sha256") == adapter_identity
                 and row.get("initial_adapter_sha256") == result.get("initial_adapter_sha256")
                 and row.get("valid_target_count") == sum(counts)
                 and row.get("host_budget_after_close_bytes") == 0
                 and row.get("cleanup_errors") == [] and row.get("diagnostic_errors") == [],
                 f"incomplete/changed/unclean pass: {name}")
        if diagnostic and name == "fp8_oracle":
            store = row.get("store_after_close")
            _require(isinstance(store, dict) and isinstance(store.get("cleanup"), dict)
                     and store["cleanup"].get("closed") is True
                     and all(type(store.get(key)) is int and store[key] == 0 for key in (
                         "owned_host_storage_bytes", "owned_device_storage_bytes",
                         "charged_host_bytes",
                     )), "missing/unclean FP8 store cleanup")
        _require_pass_attention(
            name, row, min_calls=n_layers * K,
            production=result.get("qualification_only") is False,
            hardware=result["environment"]["gpu"],
            profile_id=profile_id,
        )
        _require(all(type(row.get("runtime_load_deltas", {}).get(key)) is int
                     and row["runtime_load_deltas"][key] > 0 for key in (
                         "layer_loads", "large_loads", "expert_fetches", "expert_bytes_read",
                     )), f"missing per-expert disk runtime execution: {name}")
        checks = row.get("gradient_checks", {})
        _require(set(checks) == set(names) and all(
            check.get("present") is True and check.get("finite") is True
            and type(check.get("nonzero")) is bool
            and (diagnostic or check["nonzero"]) for check in checks.values()
        ), f"missing per-tensor finite gradients or invalid nonzero diagnostics: {name}")
        artifact = _child(root, name + "/gradients.pt")
        _require(name + "/gradients.pt" in files, f"raw named gradient tensors missing: {name}")
        if diagnostic:
            descriptor = files[name + "/gradients.pt"]
            _require(artifact.is_file() and artifact.stat().st_size == descriptor.get("bytes")
                     and _sha256(artifact) == descriptor.get("sha256"),
                     f"raw gradient artifact size/hash differs: {name}")
        if diagnostic:
            _require(row.get("gradient_artifact") == {
                "path": "gradients.pt", "bytes": files[name + "/gradients.pt"]["bytes"],
                "sha256": files[name + "/gradients.pt"]["sha256"],
            }, f"gradient artifact descriptor differs: {name}")
        with artifact.open("rb") as stream:
            values = torch.load(stream, map_location="cpu", weights_only=True)
        _require(isinstance(values, dict) and set(values) == set(names)
                 and all(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                         and bool(torch.isfinite(value).all())
                         and bool(value.ne(0).any()) is checks[key]["nonzero"]
                         and list(value.shape) == checks[key].get("shape")
                         for key, value in values.items()),
                 f"raw gradients fail finite coverage or recorded nonzero diagnostics: {name}")
        gradients[name] = values
        if name != "ga":
            _require(all(value.shape == gradients["ga"][key].shape
                         for key, value in values.items()),
                     f"raw gradient shapes differ across passes: {name}")
        if diagnostic and name in ("ga", "ga_repeat", "ga_floor"):
            _require(row.get("layer_forward_counts") == [K] * n_layers,
                     f"ordinary GA layer coverage missing: {name}")
        losses = row.get("losses", [])
        _require(len(losses) == K and all(_finite(value) for value in losses),
                 f"nonfinite/incomplete losses: {name}")
        if "oracle" in name or "l2l" in name:
            _require(row.get("routing_guard") == "production_L2LStep_enabled",
                     "production routing guard was not retained")
    if not diagnostic:
        _require(passes["ga"]["losses"] == passes["ga_repeat"]["losses"],
                 "per-microbatch repeat losses differ")
    floor = result.get("floor", {})
    _require(floor.get("observed") is True and floor.get("backend") == FLOOR_BACKEND
             and floor.get("main_backend") == MAIN_BACKEND and floor.get("factor") == 1.5,
             "independent BF16 floor missing or backend policy differs")
    a3 = assess_a3(
        gradients["ga"], gradients["ga_repeat"], gradients["bf16_l2l"],
        losses=passes["ga"]["losses"], repeat_losses=passes["ga_repeat"]["losses"],
        l2l_losses=passes["bf16_l2l"]["losses"], dtype="bfloat16",
        floor_gradients=gradients["ga_floor"], floor_losses=passes["ga_floor"]["losses"],
        floor_observed=True,
    )
    _require(matches(a3, result.get("a3"), path="$.a3"),
             "A3 raw tensor recomputation differs from recorded verdict")
    if not diagnostic:
        _require(a3["status"] == "PASS", "A3 raw tensor recomputation did not reproduce PASS")
    derived_gates = {}
    expected_pairs = (n_layers - result["model"]["source"]["shape"]["first_k_dense_replace"]) * K
    archive_names = ("bf16_l2l", "fp8_oracle", "int4_oracle") if diagnostic else (
        "fp8_oracle", "int4_oracle",
    )
    for name in archive_names:
        row = passes[name]
        if "oracle" in name:
            _require(_oracle_complete(row, expected_pairs), "missing executed Oracle VJP coverage")
            _require(row.get("losses") == passes["bf16_l2l"]["losses"],
                     "candidate/control original-forward loss differs from BF16")
        _require_layer_coverage(row.get("layers"), layers=layers, diagnostic=diagnostic)
        table = []
        for layer in layers:
            parts = {kind: [] for kind in ("restore", "output", "branch")}
            for mb in range(K):
                basename = f"layer-{layer:03d}-mb-{mb:03d}.pt"
                pair = []
                for relative in ("bf16_l2l/forward/" + basename, name + "/recompute/" + basename):
                    _require(relative in files, "missing raw boundary pair")
                    if diagnostic:
                        artifact = _child(root, relative)
                        descriptor = files[relative]
                        _require(artifact.is_file()
                                 and artifact.stat().st_size == descriptor.get("bytes")
                                 and _sha256(artifact) == descriptor.get("sha256"),
                                 "raw boundary artifact size/hash differs")
                    with _child(root, relative).open("rb") as stream:
                        value = torch.load(stream, map_location="cpu", weights_only=True)
                    hidden = result["model"]["source"]["shape"]["hidden_size"]
                    _require(isinstance(value, dict) and set(value) == {"input", "output"}
                             and all(isinstance(tensor, torch.Tensor)
                                     and list(tensor.shape) == [1, MICROBATCH_LENGTH, hidden]
                                     and tensor.dtype == torch.bfloat16
                                     and bool(torch.isfinite(tensor).all())
                                     for tensor in value.values()), "invalid raw boundary tensor")
                    pair.append(value)
                original, restored = pair
                for kind in ("restore", "output"):
                    key = "input" if kind == "restore" else "output"
                    parts[kind].append(metrics.restore_error(original[key], restored[key]))
                parts["branch"].append(metrics.restore_error(
                    original["output"].double() - original["input"].double(),
                    restored["output"].double() - restored["input"].double(),
                ))
            recorded = row["layers"][layer]
            recomputed = {"layer": layer, "pairs": recorded["pairs"], **{
                kind: metrics.merge_errors(values) for kind, values in parts.items()
            }}
            _require(matches(recomputed, recorded, path=f"$.passes.{name}.layers[{layer}]"),
                     "layer metrics differ from raw one-recompute tensors")
            table.append(recomputed)
        if name == "bf16_l2l":
            continue
        agreement = gradient_agreement(gradients["bf16_l2l"], gradients[name])
        _require(matches(agreement, row.get("gradient_agreement"),
                         path=f"$.passes.{name}.gradient_agreement"),
                 "E3 raw tensor metrics differ")
        derived_gates[name] = assess_metrics(
            table, agreement, expected_layers=layers, expected_tensors=names,
        )
        statuses = ("PASS", "FAIL", "NO_VERDICT") if diagnostic else ("PASS", "FAIL")
        _require(all(value.get("status") in statuses for value in derived_gates[name].values()),
                 "incomplete/nonfinite recomputed G4 metrics")
    _require(matches(derived_gates["fp8_oracle"], result.get("candidate_gates"),
                     path="$.candidate_gates"), "candidate G4 verdict differs from raw metrics")
    discrimination = assess_discrimination(derived_gates["int4_oracle"])
    control = {"gates": derived_gates["int4_oracle"], "discrimination": discrimination}
    _require(matches(control, result.get("control"), path="$.control"),
             "int4 control evidence differs")
    if diagnostic:
        _require(declared == sorted(derived_nonfinite),
                 "nonfinite diagnostic paths differ from raw metric recomputation")
    verdict = overall_verdict(a3, derived_gates["fp8_oracle"], discrimination, eligible=True)
    _require(verdict == result.get("verdict"),
             "independently recomputed numerical verdict differs from receipt")
    if not diagnostic:
        _require(verdict == "PASS", "independently recomputed gate does not admit training")
    return result


def load_gate(
    path: Path, sha256: str, *, profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    """Return raw-verified production evidence with its unchanged numerical verdict.

    DeepSeek requires PASS. Moonlight also admits complete finite FAIL/NO_VERDICT
    diagnostics. Callers must still match live model, adapter, pack and environment.
    """
    try:
        return _load_gate(path, sha256, profile_id=profile_id)
    except (
        KeyError, IndexError, TypeError, AttributeError, OSError, RuntimeError, EOFError,
    ) as error:
        raise ValueError(f"gate admission: malformed or missing evidence: {error}") from error


class _Parser(argparse.ArgumentParser):
    def _print_message(self, message: str, file: TextIO | None = None) -> None:
        if message:
            Console(file=file or sys.stdout, markup=False, highlight=False).print(message, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    for name in ("source", "cache", "pack", "initial-adapter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--prepare", action="store_true", help="build only a missing NF4 cache")
    parser.add_argument("--host-budget-bytes", type=int, default=HOST_BUDGET_BYTES)
    parser.add_argument("--profile", choices=("deepseek-v3", "moonlight-16b-a3b"),
                        default="deepseek-v3")
    args = parser.parse_args(argv)
    report = run_gate(
        source=args.source, cache=args.cache, pack=args.pack, pack_sha256=args.pack_sha256,
        initial_adapter=args.initial_adapter, output=args.output, prepare=args.prepare,
        host_budget_bytes=args.host_budget_bytes, profile_id=args.profile,
    )
    Console().print(f"Gradient gate: {report['status']} / {report['verdict']}")
    if args.profile == "moonlight-16b-a3b":
        if report["status"] != "COMPLETE" or report["qualification_only"]:
            return 1
        manifest = args.output / "manifest.json"
        load_gate(manifest, _sha256(manifest), profile_id=args.profile)
        Console().print("Moonlight diagnostic execution verified; numerical verdict is unchanged.")
        return 0
    return 0 if report["verdict"] == "PASS" else 1 if report["verdict"] == "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
