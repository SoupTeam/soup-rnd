"""Hash-bound Stage2 v2 likelihood/NLL and initial-only development calibration.

v2 is an outcome-informed redesign chosen after v1 floors. It does not complete
v1, whose verdict remains BLOCKED_INCOMPLETE_COVERAGE. Only the parent executes
these entry points, in fresh Kaggle T4 processes after recording the protocol hash.
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import random
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from rich.console import Console

from . import p3_realdata_model as native
from . import p3_stage2_checkpoint_generation as checkpoint
from . import p3_stage2_dataset as data
from . import p3_stage2_generation as generation
from . import p3_stage2_preflight as preflight
from . import p3_stage2_training_audit as audit
from .p3_fp8_smoke import MemoryObservation

LOGITS_MEMORY_LIMIT_BYTES = 512 * 1024**2
REDUCTION_CHUNK_TOKENS = 128
DEFAULT_MAX_TOKENS_PER_BATCH = 8192
TRAINING_AUDIT_SHA256 = "b3e35ec28266529e37a2cd2b78bb4ed658d6d3d7861ac9649583d1fa3d17c3d2"
ENDPOINTS = ("hellaswag", "arc", "boolq", "lambada", "gsm8k", "mbpp")
METRICS = {"hellaswag": "acc_norm", "arc": "acc_norm", "boolq": "acc",
           "lambada": "acc", "gsm8k": "nll", "mbpp": "nll"}
DEV_COUNTS = {"hellaswag_dev": 2000, "arc_dev": 1000, "boolq_dev": 1000}
RUNTIME_PACKAGES = ("torch", "transformers", "peft", "accelerate", "safetensors",
                    "tokenizers", "numpy", "psutil", "pyyaml")
SOURCE_PATHS = (
    "benchmarks/harness/p3_stage2v2_loglik.py",
    "benchmarks/harness/p3_stage2v2_data.py",
    "benchmarks/harness/p3_realdata_model.py",
    "benchmarks/harness/p3_realdata_baseline.py",
    "benchmarks/harness/p3_realdata_corpus.py",
    "benchmarks/harness/p3_stage2_dataset.py",
    "benchmarks/harness/p3_stage2_generation.py",
    "benchmarks/harness/p3_stage2_preflight.py",
    "benchmarks/harness/p3_stage2_checkpoint_generation.py",
    "benchmarks/harness/p3_stage2_training_audit.py",
    "benchmarks/harness/p3_stage2_training.py",
    "benchmarks/harness/p3_fp8_smoke.py",
    "benchmarks/harness/l2l_schedule.py",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _positive(value: Any, name: str) -> int:
    _require(type(value) is int and value > 0, f"positive integer required: {name}")
    return value


def _source_hashes() -> dict:
    root = Path(__file__).parents[2]
    return {name: generation._file_hash(data._relative(root, name)) for name in SOURCE_PATHS}


def _libraries() -> dict:
    return {name: version(name) for name in RUNTIME_PACKAGES}


def _write_manifest(path: Path, value: dict) -> None:
    value["manifest_payload_sha256"] = generation._object_hash(value)
    generation._write_json(path, value)


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    state = generation._backend_state(tokenizer)
    try:
        if state[0] is not None:
            state[0].no_padding()
            state[0].no_truncation()
        ids = tokenizer(
            text, add_special_tokens=False, truncation=False, padding=False,
            return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]
    finally:
        generation._restore_backend(state)
    _require(isinstance(ids, list) and all(type(token) is int and 0 <= token < len(tokenizer)
                                         for token in ids), "invalid untruncated tokenization")
    return ids


def encode_pair(tokenizer: Any, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Apply lm-eval v0.4.9 joint-string slicing, without BOS or prefix repair."""
    _require(isinstance(context, str) and isinstance(continuation, str), "pair text required")
    _require(bool(continuation), "nonempty continuation text required")
    boundary = len(context.rstrip())
    continuation = context[boundary:] + continuation
    context = context[:boundary]
    context_ids = _tokenize(tokenizer, context)
    whole = _tokenize(tokenizer, context + continuation)
    continuation_ids = whole[len(context_ids):]
    _require(bool(context_ids), "nonempty context required; no implicit BOS")
    _require(bool(continuation_ids), "nonempty continuation tokens required")
    return context_ids, continuation_ids


def _device(model: Any, tokenizer: Any) -> Any:
    import torch

    device = model.get_input_embeddings().weight.device
    _require(device.type in ("cpu", "cuda"), "resident CPU/CUDA model required")
    _require(all(parameter.device == device for parameter in model.parameters()),
             "one-device resident model required")
    if device.type == "cuda":
        _require(device.index in (None, 0) and torch.cuda.current_device() == 0,
                 "CUDA-visible device 0 only")
    _require(model.config.vocab_size == len(tokenizer), "model/tokenizer vocabulary mismatch")
    _require(not getattr(model, "is_gradient_checkpointing", False),
             "active gradient checkpointing refused")
    generation._no_runtime_hooks(model)
    return device


def _rng_state() -> tuple:
    import numpy as np
    import torch

    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _restore_rng(state: tuple) -> None:
    import numpy as np
    import torch

    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def _restore_precision(flags: dict, warn_only: bool) -> None:
    import torch

    torch.set_float32_matmul_precision(flags["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = flags["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flags[
        "allow_fp16_reduced_precision_reduction"
    ]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flags[
        "allow_bf16_reduced_precision_reduction"
    ]
    torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
    torch.use_deterministic_algorithms(flags["deterministic_algorithms_enabled"],
                                       warn_only=warn_only)
    for device_type in ("cpu", "cuda"):
        dtype_name = flags[f"autocast_dtype_{device_type}"].removeprefix("torch.")
        torch.set_autocast_dtype(device_type, getattr(torch, dtype_name))
    for key, original in flags["arithmetic_environment"].items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


@contextmanager
def _scoring_session(model: Any, tokenizer: Any) -> Iterator[None]:
    import torch

    _device(model, tokenizer)
    flags = generation._precision_flags()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng = _rng_state()
    backend = generation._backend_state(tokenizer)
    try:
        with generation._inference_session(model, tokenizer):
            if backend[0] is not None:
                backend[0].no_padding()
                backend[0].no_truncation()
            yield
    finally:
        _restore_rng(rng)
        changed = (generation._precision_flags() != flags
                   or torch.is_deterministic_algorithms_warn_only_enabled() != warn_only)
        if changed:
            _restore_precision(flags, warn_only)
        _require(torch.is_autocast_enabled("cpu") == flags["autocast_enabled_cpu"]
                 and torch.is_autocast_enabled("cuda") == flags["autocast_enabled_cuda"],
                 "autocast context restoration failed")
        generation._no_runtime_hooks(model)
        _require(generation._precision_flags() == flags, "precision restoration failed")
        _require(not changed, "scoring changed caller precision permissions")


def _request_tokens(tokenizer: Any, requests: list[tuple[str, str]]) -> list[dict]:
    encoded = []
    for context, continuation in requests:
        context_ids, targets = encode_pair(tokenizer, context, continuation)
        ids = context_ids + targets
        encoded.append({"input_ids": ids[:-1], "targets": targets,
                        "positions": list(range(len(context_ids) - 1, len(ids) - 1))})
    return encoded


def _document_tokens(tokenizer: Any, rows: Sequence[tuple[str, str]]) -> list[dict]:
    encoded = []
    backend = generation._backend_state(tokenizer)
    try:
        if backend[0] is not None:
            backend[0].no_padding()
            backend[0].no_truncation()
        for prompt, response in rows:
            _require(isinstance(prompt, str) and bool(prompt), "NLL needs a nonempty prompt")
            document = data._encode_document(tokenizer, {"prompt": prompt, "response": response})
            _require(document["labels"][0] == -100, "first NLL token has no causal predictor")
            selected = [(index - 1, label) for index, label in
                        enumerate(document["labels"][1:], start=1) if label != -100]
            _require(bool(selected), "NLL has no supervised response/EOS targets")
            encoded.append({"input_ids": document["input_ids"][:-1],
                            "positions": [position for position, _ in selected],
                            "targets": [label for _, label in selected]})
    finally:
        generation._restore_backend(backend)
    return encoded


def _logits_bound(batch_size: int, positions: int, vocabulary: int,
                  reduction_chunk_tokens: int) -> int:
    # Conservative 16 B per native logit covers head output, scaling and FP32 conversion.
    # Two FP32 reduction buffers cover gather+log_softmax; allow another 8 B for transient
    # allocator overlap. This is a logits/scratch bound, not a whole-model peak claim.
    return vocabulary * (16 * batch_size * positions
                         + 16 * min(batch_size * positions, reduction_chunk_tokens))


def _batch_plan(encoded: list[dict], *, max_tokens_per_batch: int, vocabulary: int,
                logits_memory_limit_bytes: int, reduction_chunk_tokens: int) -> list[list[int]]:
    order = sorted(range(len(encoded)), key=lambda index: (-len(encoded[index]["input_ids"]),
                                                          index))
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    positions: set[int] = set()
    for index in order:
        length = len(encoded[index]["input_ids"])
        _require(length <= max_tokens_per_batch, "single request exceeds input token budget")
        proposed_size = len(current) + 1
        proposed_width = max(width, length)
        proposed_positions = positions.union(encoded[index]["positions"])
        fits = (proposed_size * proposed_width <= max_tokens_per_batch
                and _logits_bound(proposed_size, len(proposed_positions), vocabulary,
                                  reduction_chunk_tokens)
                <= logits_memory_limit_bytes)
        if current and not fits:
            batches.append(current)
            current, width = [], 0
            positions = set()
        _require(_logits_bound(1, 1, vocabulary, reduction_chunk_tokens)
                 <= logits_memory_limit_bytes, "logits memory budget cannot fit one position")
        current.append(index)
        width = max(width, length)
        positions.update(encoded[index]["positions"])
    if current:
        batches.append(current)
    return batches


def _position_chunks(
    positions: list[int], *, batch_size: int, vocabulary: int,
    logits_memory_limit_bytes: int, reduction_chunk_tokens: int,
) -> list[list[int]]:
    low, high = 1, len(positions)
    while low < high:
        middle = (low + high + 1) // 2
        if _logits_bound(batch_size, middle, vocabulary, reduction_chunk_tokens) <= (
            logits_memory_limit_bytes
        ):
            low = middle
        else:
            high = middle - 1
    return [positions[start:start + low] for start in range(0, len(positions), low)]


def _native_batch(model: Any, tokenizer: Any, rows: list[dict], *,
                  logits_memory_limit_bytes: int, reduction_chunk_tokens: int) -> tuple[list, dict]:
    import torch

    device = model.get_input_embeddings().weight.device
    pad = tokenizer.pad_token_id
    _require(type(pad) is int and 0 <= pad < len(tokenizer), "native padding ID required")
    width = max(len(row["input_ids"]) for row in rows)
    all_positions = sorted({position for row in rows for position in row["positions"]})
    chunks = _position_chunks(all_positions, batch_size=len(rows), vocabulary=len(tokenizer),
                              logits_memory_limit_bytes=logits_memory_limit_bytes,
                              reduction_chunk_tokens=reduction_chunk_tokens)
    greedy = [True] * len(rows)
    counts = [0] * len(rows)
    estimate = max(_logits_bound(len(rows), len(part), len(tokenizer), reduction_chunk_tokens)
                   for part in chunks)
    measurement = MemoryObservation(device.type, scope=(
        "One right-padded native likelihood batch, including input allocation, selected native "
        "logits and FP32 reduction scratch; resident base/adapters and allocator caches included. "
        "Not activation-attributed memory. No hooks, retries or precision fallback."
    ))
    try:
        with measurement:
            sums = torch.zeros(len(rows), dtype=torch.float32, device=device)
            inputs = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
            attention = torch.zeros_like(inputs)
            for index, row in enumerate(rows):
                length = len(row["input_ids"])
                inputs[index, :length] = torch.tensor(row["input_ids"], device=device)
                attention[index, :length] = 1
            for positions in chunks:
                slots = {position: slot for slot, position in enumerate(positions)}
                request_indices, time_indices, targets, owners = [], [], [], []
                for row_index, row in enumerate(rows):
                    for position, target in zip(row["positions"], row["targets"], strict=True):
                        if position in slots:
                            request_indices.append(row_index)
                            time_indices.append(slots[position])
                            targets.append(target)
                            owners.append(row_index)
                output = model(
                    input_ids=inputs, attention_mask=attention, use_cache=False,
                    output_attentions=False, output_hidden_states=False,
                    output_router_logits=False,
                    logits_to_keep=torch.tensor(positions, device=device),
                    return_dict=True,
                )
                logits = output.logits
                _require(tuple(logits.shape) == (len(rows), len(positions), len(tokenizer)),
                         "native logits_to_keep did not honor selected positions")
                _require(logits.dtype in (torch.float32, torch.bfloat16, torch.float16),
                         "unexpected native logits dtype")
                # Reduce many supervised rows together, never log_softmax the full padded cube.
                for start in range(0, len(targets), reduction_chunk_tokens):
                    stop = start + reduction_chunk_tokens
                    selected = logits[request_indices[start:stop], time_indices[start:stop]].float()
                    _require(bool(torch.isfinite(selected).all().item()), "nonfinite native logits")
                    labels = torch.tensor(targets[start:stop], device=device)
                    probabilities = selected.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
                    matches = selected.argmax(-1).eq(labels)
                    # Owners are grouped by request, so contiguous reductions need no atomic
                    # scatter_add and have the same FP32 arithmetic order on every invocation.
                    local_owners = owners[start:stop]
                    first = 0
                    while first < len(local_owners):
                        owner = local_owners[first]
                        last = first + 1
                        while last < len(local_owners) and local_owners[last] == owner:
                            last += 1
                        sums[owner] += probabilities[first:last].sum(dtype=torch.float32)
                        greedy[owner] = greedy[owner] and bool(matches[first:last].all().item())
                        counts[owner] += last - first
                        first = last
                    del selected, labels, probabilities, matches
                del logits, output
            del inputs, attention
    except BaseException as error:
        # The caller records this measurement before retaining the original failure.
        error.p3_loglik_measurement = measurement.result
        raise
    results = [{"logprob": float(total), "greedy": match, "n_tokens": count}
               for total, match, count in zip(sums.tolist(), greedy, counts, strict=True)]
    _require(all(result["n_tokens"] == len(row["targets"]) and math.isfinite(result["logprob"])
                 for result, row in zip(results, rows, strict=True)),
             "incomplete or nonfinite scored target coverage")
    return results, {"input_tokens": sum(len(row["input_ids"]) for row in rows),
                     "padded_input_tokens": len(rows) * width,
                     "scored_tokens": sum(counts), "forward_calls": len(chunks),
                     "logits_estimated_peak_bytes": estimate, "measurement": measurement.result}


def _score_encoded(model: Any, tokenizer: Any, encoded: list[dict], *, max_tokens_per_batch: int,
                   logits_memory_limit_bytes: int, reduction_chunk_tokens: int,
                   batch_observer: Callable[[dict], None] | None = None,
                   deadline: float | None = None) -> list[dict]:
    _positive(max_tokens_per_batch, "max_tokens_per_batch")
    _positive(logits_memory_limit_bytes, "logits_memory_limit_bytes")
    _positive(reduction_chunk_tokens, "reduction_chunk_tokens")
    limit = generation._context_limit(model, tokenizer)
    for row in encoded:
        _require(bool(row["input_ids"]) and len(row["input_ids"]) + 1 <= limit,
                 "untruncated request exceeds native context limit")
    plan = _batch_plan(encoded, max_tokens_per_batch=max_tokens_per_batch,
                       vocabulary=len(tokenizer),
                       logits_memory_limit_bytes=logits_memory_limit_bytes,
                       reduction_chunk_tokens=reduction_chunk_tokens)
    results: list[Any] = [None] * len(encoded)
    for batch_index, indices in enumerate(plan):
        _require(deadline is None or time.monotonic() < deadline, "controller deadline exceeded")
        event = {"schema": "p3-stage2-v2-loglik-batch-v1", "batch_index": batch_index,
                 "request_indices": indices}
        try:
            scored, observation = _native_batch(
                model, tokenizer, [encoded[index] for index in indices],
                logits_memory_limit_bytes=logits_memory_limit_bytes,
                reduction_chunk_tokens=reduction_chunk_tokens,
            )
        except BaseException as error:
            if batch_observer is not None:
                batch_observer({**event, "event": "batch_failed",
                                "measurement": getattr(error, "p3_loglik_measurement", {})})
            raise
        if batch_observer is not None:
            batch_observer({**event, "event": "batch_complete", **observation})
        for index, result in zip(indices, scored, strict=True):
            results[index] = result
        _require(deadline is None or time.monotonic() < deadline, "controller deadline exceeded")
    return results


def _score_requests(model: Any, tokenizer: Any, requests: list[tuple[str, str]], *,
                    max_tokens_per_batch: int,
                    logits_memory_limit_bytes: int = LOGITS_MEMORY_LIMIT_BYTES,
                    reduction_chunk_tokens: int = REDUCTION_CHUNK_TOKENS,
                    batch_observer: Callable[[dict], None] | None = None,
                    deadline: float | None = None) -> list[dict]:
    with _scoring_session(model, tokenizer):
        encoded = _request_tokens(tokenizer, requests)
        return _score_encoded(
            model, tokenizer, encoded, max_tokens_per_batch=max_tokens_per_batch,
            logits_memory_limit_bytes=logits_memory_limit_bytes,
            reduction_chunk_tokens=reduction_chunk_tokens, batch_observer=batch_observer,
            deadline=deadline,
        )


def score_requests(model: Any, tokenizer: Any, requests: list[tuple[str, str]], *,
                   max_tokens_per_batch: int) -> list[dict]:
    """Sum FP32 continuation log-probabilities, restoring input order and caller state."""
    return _score_requests(model, tokenizer, requests, max_tokens_per_batch=max_tokens_per_batch)


def score_task(row: dict, results: list[dict]) -> dict:
    for result in results:
        _require(isinstance(result, dict) and type(result.get("logprob")) in (int, float)
                 and math.isfinite(result["logprob"]) and type(result.get("greedy")) is bool
                 and type(result.get("n_tokens")) is int and result["n_tokens"] > 0,
                 "invalid likelihood result")
    if row.get("endpoint") == "lambada":
        _require(len(results) == 1 and row.get("metric") == "acc", "LAMBADA result required")
        return {"pred": 0, "pred_norm": 0, "pass": results[0]["greedy"]}
    choices = row.get("choices")
    _require(isinstance(choices, list) and len(choices) >= 2 and len(choices) == len(results)
             and all(isinstance(choice, str) and bool(choice) for choice in choices),
             "one result per nonempty choice required")
    gold = row.get("gold")
    _require(type(gold) is int and 0 <= gold < len(choices), "invalid gold choice")
    _require(row.get("metric") in ("acc", "acc_norm"), "unsupported likelihood metric")
    prediction = max(range(len(results)), key=lambda index: results[index]["logprob"])
    normalized = max(range(len(results)), key=lambda index:
                     results[index]["logprob"] / len(choices[index]))
    primary = normalized if row["metric"] == "acc_norm" else prediction
    return {"pred": prediction, "pred_norm": normalized, "pass": primary == gold}


def _nll(result: dict) -> dict:
    total = -result["logprob"]
    return {"nll_sum": total, "n_targets": result["n_tokens"],
            "nll_mean": total / result["n_tokens"]}


def teacher_forced_nll(model: Any, tokenizer: Any, prompt: str, response: str) -> dict:
    """Response-only FP32 NLL using frozen offset labels and exactly one appended EOS."""
    with _scoring_session(model, tokenizer):
        encoded = _document_tokens(tokenizer, [(prompt, response)])
        result = _score_encoded(
            # A single NLL document keeps its complete context. Logits use a separate byte cap.
            model, tokenizer, encoded,
            max_tokens_per_batch=max(DEFAULT_MAX_TOKENS_PER_BATCH, len(encoded[0]["input_ids"])),
            logits_memory_limit_bytes=LOGITS_MEMORY_LIMIT_BYTES,
            reduction_chunk_tokens=REDUCTION_CHUNK_TOKENS,
        )[0]
    return _nll(result)


def _read_manifest(root: Path, expected: str) -> dict:
    path = data._relative(root, "manifest.json")
    _require(isinstance(expected, str) and data._SHA256.fullmatch(expected) is not None
             and generation._file_hash(path) == expected, "data manifest file SHA256 mismatch")
    manifest = checkpoint._manifest(path)
    _require(manifest.get("schema") == "p3-stage2-v2-data-v1", "v2 dataset required")
    _require(isinstance(manifest.get("endpoints"), dict), "dataset endpoint mapping required")
    return manifest


def _task_rows(root: Path, manifest: dict, endpoint: str, *, development: bool) -> list[dict]:
    entry = manifest["endpoints"][endpoint]
    role = "development" if development else "evaluation"
    metric = METRICS[endpoint.removesuffix("_dev")]
    _require(entry.get("role") == role and entry.get("metric") == metric,
             "endpoint role/metric mismatch")
    if development:
        _require(entry.get("source_split") == "train", "pilot train split required")
    descriptor = entry["files"]["tasks"]
    _require(descriptor.get("path") == f"{endpoint}/tasks.jsonl"
             and manifest["files"].get(f"{endpoint}_tasks") == descriptor,
             "endpoint task descriptor binding mismatch")
    path = data._artifact(root, descriptor)
    rows = list(data._jsonl(path))
    ids = [row.get("task_id") for row in rows]
    _require(type(descriptor.get("rows")) is int and len(rows) == descriptor["rows"]
             == entry.get("retained_row_count") and bool(rows)
             and ids == entry.get("retained_task_ids")
             and all(isinstance(item, str) and bool(item) for item in ids)
             and len(set(ids)) == len(ids)
             and entry.get("ordered_task_ids_sha256") == generation._object_hash(ids),
             "complete ordered task coverage mismatch")
    refs = set()
    for index, row in enumerate(rows):
        _require(row.get("endpoint") == endpoint and row.get("role") == role
                 and row.get("metric") == metric and isinstance(row.get("context"), str)
                 and bool(row["context"]) and row.get("source_ref") == row["task_id"]
                 and isinstance(row.get("source_row_sha256"), str)
                 and data._SHA256.fullmatch(row["source_row_sha256"]) is not None,
                 "task source/role/metric identity mismatch")
        _require(row["source_ref"] not in refs, "duplicate source task")
        refs.add(row["source_ref"])
        if development:
            _require(row.get("source_split") == "train"
                     and type(row.get("source_row_ordinal")) is int
                     and row["source_row_ordinal"] == index
                     and row["source_ref"].endswith(f"#row={index}"),
                     "pilot must use first physical train rows")
        else:
            _require(row.get("source_split") in ("test", "validation"),
                     "evaluation test/validation split required")
        if metric == "nll" or endpoint == "lambada":
            _require(isinstance(row.get("target"), str) and bool(row["target"].strip())
                     and row.get("gold") == row["target"], "target/reference identity mismatch")
            if endpoint == "lambada":
                _require(row["target"].startswith(" "), "LAMBADA leading-space target required")
        else:
            # score_task validates choice cardinality, character denominators and gold.
            choices = row.get("choices", [])
            score_task(row, [{"logprob": 0.0, "greedy": False, "n_tokens": 1}
                             for _ in choices])
    return rows


def _admit_dataset(root: Path, protocol: dict, endpoint: str,
                   start: int, stop: int) -> tuple[list[dict], dict]:
    _require(endpoint in ENDPOINTS, "unsupported evaluation endpoint")
    expected = protocol["data"]["dataset_manifest_file_sha256"]
    manifest = _read_manifest(root, expected)
    counts = protocol["data"]["retained_evaluation_tasks"]
    ordered = protocol["data"]["ordered_task_ids_sha256"]
    full_counts = protocol["data"]["full_retained_evaluation_tasks"]
    source_ordered = protocol["data"]["source_ordered_task_ids_sha256"]
    _require(set(counts) == set(ENDPOINTS) == set(ordered) == set(full_counts)
             == set(source_ordered), "six full/selected endpoint counts/ID hashes required")
    for name in ENDPOINTS:
        entry = manifest["endpoints"][name]
        wanted = counts[name]
        _positive(wanted, f"{name} task count")
        available = entry["retained_row_count"]
        _require(type(full_counts[name]) is int and available == full_counts[name]
                 and source_ordered[name] == entry.get("ordered_task_ids_sha256"),
                 "full endpoint count/source ordering mismatch")
        _require(entry.get("role") == "evaluation" and entry.get("metric") == METRICS[name]
                 and wanted <= available and (name == "hellaswag" or wanted == available),
                 "complete non-Hella endpoint required")
    rows = _task_rows(root, manifest, endpoint, development=False)
    if endpoint == "hellaswag":
        wanted = protocol["loglik"]["hellaswag_n"]
        _require(wanted == counts[endpoint], "HellaSwag frozen N/count mismatch")
        if wanted < len(rows):
            from .p3_stage2v2_data import select_hellaswag_tasks

            rows = select_hellaswag_tasks(rows, wanted)
    ids = [row["task_id"] for row in rows]
    _require(len(rows) == counts[endpoint] and generation._object_hash(ids) == ordered[endpoint],
             "frozen selected task ordering mismatch")
    _require(type(start) is int and type(stop) is int and 0 <= start < stop <= len(rows),
             "invalid unmodified shard interval")
    selected = rows[start:stop]
    _require(generation._file_hash(data._relative(root, "manifest.json")) == expected,
             "dataset manifest changed during admission")
    return selected, {"data_manifest_sha256": expected, "complete_endpoint_row_count": len(rows),
                      "start": start, "stop": stop,
                      "ordered_task_ids": [row["task_id"] for row in selected],
                      "selected_rows_sha256": generation._object_hash(selected),
                      "tasks_file": copy.deepcopy(
                          manifest["endpoints"][endpoint]["files"]["tasks"],
                      )}


def _v1_protocol(path: Path) -> dict:
    _require(generation._file_hash(path) == audit.FROZEN_PROTOCOL_SHA256,
             "fixed v1 protocol hash mismatch")
    protocol = checkpoint._read_json(path)
    _require(protocol.get("schema") == "p3-stage2-frozen-paired-study-v1",
             "v1 protocol schema mismatch")
    root = Path(__file__).parents[2]
    for section in ("source_files_sha256", "contract_files_sha256"):
        base = path.parent if section == "contract_files_sha256" else root
        for relative, expected in protocol.get(section, {}).items():
            _require(generation._file_hash(data._relative(base, relative)) == expected,
                     "v1 frozen source/contract changed")
    return protocol


def _admit_protocol(path: Path, expected: str, v1_path: Path,
                    arm: str, seed: int, length: int) -> tuple[dict, dict]:
    _require(isinstance(expected, str) and data._SHA256.fullmatch(expected) is not None
             and generation._file_hash(path) == expected, "v2 protocol file SHA256 mismatch")
    protocol = checkpoint._read_json(path)
    _require(protocol.get("schema") == "p3-stage2-v2-protocol"
             and protocol.get("status") == "FROZEN",
             "recorded frozen v2 protocol required before evaluation")
    _require(protocol.get("v1_protocol_sha256") == audit.FROZEN_PROTOCOL_SHA256
             and protocol.get("training_audit_sha256") == TRAINING_AUDIT_SHA256,
             "v1 protocol/training audit binding mismatch")
    original = _v1_protocol(v1_path)
    _require(all(protocol.get(key) == original[key] for key in ("arms", "seeds", "lengths"))
             and arm in protocol["arms"] and type(seed) is int and seed in protocol["seeds"]
             and type(length) is int and length in protocol["lengths"],
             "unapproved trained configuration")
    _require(protocol.get("model", {}).get("id") == native.MODEL_ID
             and protocol["model"].get("revision") == native.REVISION,
             "pinned model identity mismatch")
    observed = _source_hashes()
    _require(all(protocol.get("source_files_sha256", {}).get(name) == digest
                 for name, digest in observed.items()), "frozen v2 implementation changed")
    policy = protocol["loglik"]
    for key in ("max_tokens_per_batch", "logits_memory_limit_bytes", "reduction_chunk_tokens",
                "hellaswag_n", "deadline_seconds"):
        _positive(policy.get(key), key)
    _require(policy["deadline_seconds"] <= 2700, "endpoint deadline exceeds 45 minutes")
    hellaswag_n = policy["hellaswag_n"]
    full_hellaswag = protocol["data"]["full_retained_evaluation_tasks"]["hellaswag"]
    _positive(full_hellaswag, "full retained HellaSwag count")
    _require(3000 <= hellaswag_n <= full_hellaswag <= 10042
             and (hellaswag_n == full_hellaswag or hellaswag_n % 100 == 0),
             "HellaSwag N violates frozen sizing rule")
    _require(policy["logits_memory_limit_bytes"] == LOGITS_MEMORY_LIMIT_BYTES
             and policy["reduction_chunk_tokens"] == REDUCTION_CHUNK_TOKENS
             and policy.get("add_special_tokens") is False and policy.get("prepend_bos") is False
             and policy.get("padding") == "right" and policy.get("logprob_dtype") == "float32"
             and policy.get("nll_label_policy") == "p3_stage2_dataset._encode_document",
             "frozen scoring semantics mismatch")
    _require(set(RUNTIME_PACKAGES).issubset(protocol.get("library_versions", {})),
             "explicit runtime package versions required")
    return protocol, original


def _admit_pilot_data(root: Path, expected: str) -> tuple[dict, dict]:
    manifest = _read_manifest(root, expected)
    # Check all development identities before opening any task artifact. No evaluation file is read.
    for endpoint in DEV_COUNTS:
        entry = manifest["endpoints"].get(endpoint, {})
        _require(entry.get("role") == "development" and entry.get("source_split") == "train"
                 and entry.get("retained_row_count") == DEV_COUNTS[endpoint],
                 "pilot requires exactly 2000/1000/1000 train-only development tasks")
    rows = {endpoint: _task_rows(root, manifest, endpoint, development=True)
            for endpoint in DEV_COUNTS}
    return rows, {"data_manifest_sha256": expected, "endpoints": {
        endpoint: {"role": "development", "source_split": "train", "task_count": len(values),
                   "ordered_task_ids": [row["task_id"] for row in values],
                   "tasks_file": manifest["endpoints"][endpoint]["files"]["tasks"]}
        for endpoint, values in rows.items()
    }}


def _admit_initial(matrix_root: Path, audit_path: Path, audit_sha256: str,
                   original: dict) -> dict:
    _require(audit_sha256 == TRAINING_AUDIT_SHA256
             and generation._file_hash(audit_path) == audit_sha256, "fixed training audit required")
    report = checkpoint._read_json(audit_path)
    _require(report.get("schema") == "p3-stage2-training-audit-v1"
             and report.get("status") == audit.VERIFIED and report.get("errors") == []
             and report.get("required_run_count") == report.get("verified_run_count") == 20
             and report.get("protocol_sha256") == audit.FROZEN_PROTOCOL_SHA256
             and report.get("pack_manifest_file_sha256")
             == original["data"]["training_pack_manifest_file_sha256"],
             "complete fixed training audit required")
    key = "L1024-seed23-bf16_l2l"
    entry = report["verified_final_checkpoints"][key]
    _require(entry.get("directory") == key and entry.get("seed") == 23
             and entry.get("length") == 1024 and entry.get("arm") == "bf16_l2l"
             and entry.get("protocol_sha256") == audit.FROZEN_PROTOCOL_SHA256,
             "pilot initial snapshot scope mismatch")
    relative = f"{key}/run.json"
    _require(entry.get("run_manifest_file") == relative, "initial wrapper path mismatch")
    wrapper_path = data._relative(matrix_root, relative)
    _require(wrapper_path.stat().st_size == entry["run_manifest_bytes"]
             and generation._file_hash(wrapper_path) == entry["run_manifest_file_sha256"],
             "initial wrapper bytes/hash mismatch")
    wrapper = checkpoint._read_json(wrapper_path)
    _require(wrapper.get("schema") == "p3-stage2-pretrained-arm-v1"
             and wrapper.get("seed") == 23 and wrapper.get("length") == 1024
             and wrapper.get("arm") == "bf16_l2l" and wrapper.get("preflight") is False
             and wrapper.get("protocol_sha256") == audit.FROZEN_PROTOCOL_SHA256,
             "initial training wrapper identity mismatch")
    descriptor = entry["initial_snapshot"]
    _require(descriptor.get("path") == f"{key}/initial.adapters.safetensors"
             and descriptor.get("file") == "initial.adapters.safetensors"
             and all(descriptor.get(field) == wrapper["initial_snapshot"].get(field)
                     for field in ("file", "bytes", "sha256", "tensor_manifest")),
             "audited initial snapshot binding mismatch")
    # Deliberately never inspect final_step64, arm tensors, or any evaluation task file.
    expected_metadata = {"kind": "paired-initial-adapters", "seed": "23",
                         "protocol_sha256": audit.FROZEN_PROTOCOL_SHA256}
    _require(descriptor.get("metadata") == expected_metadata, "audited initial header mismatch")
    return {"directory": key, "initializer": entry["initializer"],
            "checkpoint": copy.deepcopy(descriptor)}


def _requests(rows: Sequence[dict]) -> tuple[list[tuple[str, str]], list[int]]:
    requests, sizes = [], []
    for row in rows:
        continuations = [row["target"]] if "choices" not in row else [
            " " + choice for choice in row["choices"]
        ]
        sizes.append(len(continuations))
        requests.extend((row["context"], continuation) for continuation in continuations)
    return requests, sizes


def _score_rows(model: Any, tokenizer: Any, rows: list[dict], *, policy: dict,
                observer: Callable[[dict], None], deadline: float) -> list[dict]:
    kwargs = {"max_tokens_per_batch": policy["max_tokens_per_batch"],
              "logits_memory_limit_bytes": policy["logits_memory_limit_bytes"],
              "reduction_chunk_tokens": policy["reduction_chunk_tokens"],
              "batch_observer": observer, "deadline": deadline}
    if rows[0]["metric"] == "nll":
        with _scoring_session(model, tokenizer):
            encoded = _document_tokens(tokenizer, [(row["context"], row["target"]) for row in rows])
            scored = _score_encoded(model, tokenizer, encoded, **kwargs)
        return [_nll(result) for result in scored]
    requests, sizes = _requests(rows)
    scored = _score_requests(model, tokenizer, requests, **kwargs)
    output, start = [], 0
    for row, size in zip(rows, sizes, strict=True):
        results = scored[start:start + size]
        output.append({"results": results, **score_task(row, results)})
        start += size
    return output


def _failure(root: Path, result: dict, stage: str, error: BaseException) -> None:
    checkpoint._failure(root, result, stage, error)


def _runtime() -> tuple[Any, dict]:
    torch, environment = checkpoint._admit_runtime()
    _require(torch.cuda.current_device() == 0, "CUDA-visible device 0 only")
    environment["library_versions"] = _libraries()
    _require(environment["library_versions"]["psutil"] == "7.2.2"
             and environment["library_versions"]["pyyaml"] == "6.0.3",
             "explicit psutil/pyyaml pins required")
    return torch, environment


def _new_result(schema: str) -> dict:
    return {"schema": schema, "status": "STARTED", "selected_shard_complete": False,
            "v1_verdict": "BLOCKED_INCOMPLETE_COVERAGE", "v2_outcome_informed_redesign": True,
            "v2_completes_v1": False, "global_required_coverage_verified": False,
            "quality_admitted": False, "production_admitted": False, "retries": 0,
            "fallback_precision": False, "reservation_sum_is_peak": False}


def _admit_checkpoint_map(audit_path: Path, audit_sha256: str, protocol: dict) -> None:
    _require(audit_sha256 == TRAINING_AUDIT_SHA256
             and generation._file_hash(audit_path) == audit_sha256,
             "fixed training audit hash required for full checkpoint mapping")
    report = checkpoint._read_json(audit_path)
    expected = {
        f"L{length}-seed{seed}-{arm}"
        for length in protocol["lengths"]
        for seed in protocol["seeds"]
        for arm in protocol["arms"]
    }
    frozen = protocol.get("checkpoints")
    audited = report.get("verified_final_checkpoints")
    _require(len(expected) == 20 and isinstance(frozen, dict) and isinstance(audited, dict)
             and set(frozen) == set(audited) == expected and frozen == audited,
             "complete exact20 audited checkpoint mapping required")
    _require(generation._file_hash(audit_path) == audit_sha256,
             "training audit changed during full checkpoint mapping admission")


def _remember_failure(root: Path, result: dict, stage: str, error: BaseException,
                      interrupted: type[BaseException] | None) -> type[BaseException] | None:
    if not isinstance(error, Exception):
        result["interrupted"] = True
        if interrupted is None:
            interrupted = type(error)
    _failure(root, result, stage, error)
    return interrupted


def _cleanup(
    tensors: Any, rng: tuple | None, flags: dict | None, warn_only: bool | None,
    root: Path, result: dict, interrupted: type[BaseException] | None,
) -> type[BaseException] | None:
    try:
        if rng is not None:
            _restore_rng(rng)
            result["caller_rng_state_restored"] = True
    except BaseException as error:
        interrupted = _remember_failure(root, result, "caller_rng_restoration", error, interrupted)
    try:
        if flags is not None:
            import torch

            changed = (generation._precision_flags() != flags
                       or torch.is_deterministic_algorithms_warn_only_enabled() != warn_only)
            _restore_precision(flags, bool(warn_only))
            _require(generation._precision_flags() == flags, "caller precision restoration failed")
            result["caller_precision_flags_restored"] = True
            _require(not changed, "owned execution changed precision permissions")
    except BaseException as error:
        interrupted = _remember_failure(root, result, "caller_precision_restoration", error,
                                        interrupted)
    try:
        if tensors is not None:
            tensors.clear()
    except BaseException as error:
        interrupted = _remember_failure(root, result, "adapter_tensor_release", error, interrupted)
    return interrupted


def _cleanup_owned(torch: Any, allocated: int | None, root: Path, result: dict,
                   interrupted: type[BaseException] | None) -> type[BaseException] | None:
    # The owner has dropped model/tokenizer/tensor bindings before this function runs.
    try:
        gc.collect()
    except BaseException as error:
        interrupted = _remember_failure(root, result, "model_garbage_collection", error,
                                        interrupted)
    try:
        result["cleanup"] = (checkpoint._cleanup_cuda(torch, allocated)
                             if torch is not None and allocated is not None
                             else {"complete": True, "model_loading_not_started": True})
    except BaseException as error:
        result["cleanup"] = {"complete": False}
        interrupted = _remember_failure(root, result, "model_cuda_cleanup", error, interrupted)
    return interrupted


def run_endpoint(*, data_root: Path, endpoint: str, start: int, stop: int, arm: str,
                 seed: int, length: int, matrix_root: Path, audit_path: Path, audit_sha256: str,
                 v1_protocol_path: Path, protocol_path: Path, protocol_sha256: str,
                 cache_dir: Path, output: Path) -> dict:
    """Admit the frozen protocol/data/audited final adapter, execute one shard once."""
    root = checkpoint._fresh_output(
        output, (data_root, matrix_root, audit_path, v1_protocol_path, protocol_path), cache_dir,
    )
    started, clock = time.perf_counter(), time.monotonic()
    result = _new_result("p3-stage2-v2-loglik-run-v1")
    identity = {"endpoint": endpoint, "arm": arm, "seed": seed, "length": length,
                "protocol_sha256": protocol_sha256, "audit_sha256": audit_sha256}
    result.update(identity, start=start, stop=stop, requested_rows=stop - start, completed_rows=0)
    torch = model = tokenizer = metadata = tensors = rng = flags = warn_only = allocated = None
    interrupted = None
    stage = "frozen_protocol_admission"
    try:
        result["source_files_sha256"] = _source_hashes()
        protocol, original = _admit_protocol(protocol_path, protocol_sha256, v1_protocol_path,
                                             arm, seed, length)
        _require(audit_sha256 == protocol["training_audit_sha256"], "audit/protocol hash mismatch")
        deadline = clock + protocol["loglik"]["deadline_seconds"]
        rows, binding = _admit_dataset(data_root, protocol, endpoint, start, stop)
        result.update(binding)
        stage = "full_audited_checkpoint_mapping"
        _admit_checkpoint_map(audit_path, audit_sha256, protocol)
        stage = "fresh_kaggle_t4_admission"
        torch, environment = _runtime()
        result["environment"] = environment
        flags, rng = environment["precision_flags"], _rng_state()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        allocated = torch.cuda.memory_allocated(0)
        _require(environment["library_versions"] == {
            name: protocol["library_versions"][name] for name in RUNTIME_PACKAGES
        }, "frozen runtime library versions mismatch")
        stage = "fixed_audited_final_checkpoint_admission"
        entry = checkpoint._admit_checkpoint(matrix_root, audit_path, audit_sha256, original,
                                              audit.FROZEN_PROTOCOL_SHA256, arm, seed, length)
        key = f"L{length}-seed{seed}-{arm}"
        _require(protocol.get("checkpoints", {}).get(key) == entry,
                 "v2 protocol final checkpoint mapping mismatch")
        descriptor = entry["final_step64"]
        identity["checkpoint_sha256"] = descriptor["sha256"]
        result.update(identity)
        generation._write_json(root / "admission.json", {
            "schema": "p3-stage2-v2-loglik-admission-v1", **identity, **binding,
            "checkpoint": descriptor,
            "training_run_key": key, "training_audit": checkpoint._descriptor(audit_path),
            "v1_protocol_sha256": audit.FROZEN_PROTOCOL_SHA256,
            "protocol_file": checkpoint._descriptor(protocol_path),
            "source_files_sha256": result["source_files_sha256"],
        })
        stage = "final_adapter_and_base_loading"
        tensors = preflight._tensors(data._relative(matrix_root, f"{key}/arm"), descriptor,
                                     descriptor["metadata"])
        model, tokenizer, metadata = native.load_model(
            cache_dir=cache_dir, device="cuda:0", seed=seed,
        )
        _require(metadata["lora"] == entry["initializer"], "LoRA initializer mismatch")
        checkpoint._restore_final(model, tensors, descriptor, seed)
        tensors.clear()
        tensors = None
        result["native_model_admission"] = generation._native_guard(model, tokenizer)
        _require(generation._precision_flags() == flags, "loading changed precision permissions")
        generation._write_json(root / "model.json", metadata)
        result["setup_seconds"] = time.perf_counter() - started
        stage = "frozen_endpoint_scoring"
        observations: list[dict] = []
        with (root / "batches.jsonl").open("x", encoding="utf-8", newline="\n") as batches:
            def observe(event: dict) -> None:
                event = {**event, "endpoint": endpoint}
                observations.append(event)
                generation._append(batches, event)

            measured = _score_rows(model, tokenizer, rows, policy=protocol["loglik"],
                                   observer=observe, deadline=deadline)
        stage = "immutable_score_publication"
        with (root / "scores.jsonl").open("x", encoding="utf-8", newline="\n") as scores:
            for index, (row, values) in enumerate(zip(rows, measured, strict=True), start=start):
                score = values["nll_mean"] if row["metric"] == "nll" else float(values["pass"])
                record = {"schema": "p3-stage2-v2-loglik-row-v1", **identity, **values,
                          "task_index": index, "task_id": row["task_id"],
                          "source_ref": row["source_ref"],
                          "source_row_sha256": row["source_row_sha256"],
                          "metric": row["metric"], "gold": row["gold"], "score": score}
                record["row_sha256"] = generation._object_hash(record)
                generation._append(scores, record)
                result["completed_rows"] += 1
        stage = "loaded_final_and_immutable_input_revalidation"
        _require(checkpoint._loaded_adapter_manifest(model) == descriptor["tensor_manifest"],
                 "scoring changed final adapter tensors")
        _require(generation._file_hash(protocol_path) == protocol_sha256
                 and generation._file_hash(audit_path) == audit_sha256
                 and generation._file_hash(data._relative(data_root, "manifest.json"))
                 == binding["data_manifest_sha256"], "frozen admission inputs changed")
        _require(generation._file_hash(data._relative(data_root, binding["tasks_file"]["path"]))
                 == binding["tasks_file"]["sha256"], "admitted tasks changed during scoring")
        _require(_source_hashes() == result["source_files_sha256"], "scoring sources changed")
        generation._no_runtime_hooks(model)
        result.update(status="COMPLETE", selected_shard_complete=True,
                      batch_count=len(observations),
                      scored_tokens=sum(item.get("scored_tokens", 0) for item in observations),
                      request_count=sum(len(item["request_indices"]) for item in observations),
                      loaded_final_adapter_hashes_verified_after_scoring=True)
        _require(time.monotonic() < deadline, "controller deadline exceeded")
    except BaseException as error:
        interrupted = _remember_failure(root, result, stage, error, interrupted)
    finally:
        cleanup_started = time.perf_counter()
        interrupted = _cleanup(tensors, rng, flags, warn_only, root, result, interrupted)
        tensors = None
        model = tokenizer = metadata = rng = None
        interrupted = _cleanup_owned(torch, allocated, root, result, interrupted)
        result["cleanup_seconds"] = time.perf_counter() - cleanup_started
        result["wall_seconds"] = time.perf_counter() - started
        if result["status"] == "COMPLETE" and time.monotonic() >= deadline:
            _failure(root, result, "final_controller_deadline", ValueError("deadline exceeded"))
        result["files"] = generation._files(root)
        _write_manifest(root / "manifest.json", result)
    if interrupted is not None:
        raise interrupted("interruption preserved in failure evidence")
    return result


def run_pilot(*, data_root: Path, data_manifest_sha256: str, matrix_root: Path,
              audit_path: Path, audit_sha256: str, v1_protocol_path: Path,
              cache_dir: Path, output: Path,
              max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
              deadline_seconds: int = 1800) -> dict:
    """Time exactly the first 2000/1000/1000 train rows with the audited seed23 initial adapter."""
    root = checkpoint._fresh_output(
        output, (data_root, matrix_root, audit_path, v1_protocol_path), cache_dir,
    )
    started, clock = time.perf_counter(), time.monotonic()
    result = _new_result("p3-stage2-v2-loglik-pilot-v1")
    policy = {"max_tokens_per_batch": max_tokens_per_batch,
              "logits_memory_limit_bytes": LOGITS_MEMORY_LIMIT_BYTES,
              "reduction_chunk_tokens": REDUCTION_CHUNK_TOKENS}
    result.update(pilot_only=True, initial_adapter_only=True, evaluation_rows_opened=0,
                  final_checkpoint_files_opened=0, seed=23, length=1024, arm="bf16_l2l",
                  checkpoint_kind="paired-initial-adapters", audit_sha256=audit_sha256,
                  v1_protocol_sha256=audit.FROZEN_PROTOCOL_SHA256,
                  data_manifest_sha256=data_manifest_sha256, deadline_seconds=deadline_seconds,
                  endpoints={}, **policy)
    torch = model = tokenizer = metadata = tensors = rng = flags = warn_only = allocated = None
    interrupted = None
    stage = "development_only_data_and_initial_admission"
    try:
        result["source_files_sha256"] = _source_hashes()
        _positive(max_tokens_per_batch, "max_tokens_per_batch")
        _positive(deadline_seconds, "deadline_seconds")
        _require(deadline_seconds <= 1800, "pilot controller budget must not exceed 30 minutes")
        deadline = clock + deadline_seconds
        original = _v1_protocol(v1_protocol_path)
        rows_by_endpoint, binding = _admit_pilot_data(data_root, data_manifest_sha256)
        initial = _admit_initial(matrix_root, audit_path, audit_sha256, original)
        descriptor = initial["checkpoint"]
        result["checkpoint_sha256"] = descriptor["sha256"]
        generation._write_json(root / "admission.json", {
            "schema": "p3-stage2-v2-loglik-pilot-admission-v1", **binding,
            "initial_adapter_only": True, "checkpoint": descriptor,
            "training_run_key": initial["directory"], "audit_sha256": audit_sha256,
            "v1_protocol_sha256": audit.FROZEN_PROTOCOL_SHA256,
            "source_files_sha256": result["source_files_sha256"],
        })
        stage = "fresh_kaggle_t4_admission"
        torch, environment = _runtime()
        result["environment"] = environment
        result["library_versions"] = environment["library_versions"]
        flags, rng = environment["precision_flags"], _rng_state()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        allocated = torch.cuda.memory_allocated(0)
        stage = "initial_snapshot_and_native_base_loading"
        tensors = preflight._tensors(data._relative(matrix_root, initial["directory"]),
                                     descriptor, descriptor["metadata"])
        load_started = time.perf_counter()
        model, tokenizer, metadata = native.load_model(
            cache_dir=cache_dir, device="cuda:0", seed=23,
        )
        result["model_load_seconds"] = time.perf_counter() - load_started
        _require(metadata["lora"] == initial["initializer"], "pilot initializer mismatch")
        checkpoint._restore_final(model, tensors, descriptor, 23)
        tensors.clear()
        tensors = None
        result["native_model_admission"] = generation._native_guard(model, tokenizer)
        _require(generation._precision_flags() == flags, "loading changed precision permissions")
        generation._write_json(root / "model.json", metadata)
        result["setup_seconds"] = time.perf_counter() - started
        with (root / "batches.jsonl").open("x", encoding="utf-8", newline="\n") as batches:
            for endpoint, rows in rows_by_endpoint.items():
                stage = f"development_timing_{endpoint}"
                observations: list[dict] = []

                def observe(event: dict) -> None:
                    event = {**event, "endpoint": endpoint}
                    observations.append(event)
                    generation._append(batches, event)

                endpoint_started = time.perf_counter()
                requests, _ = _requests(rows)
                # Drop all development outcomes. Only request/token timing and memory are exported.
                scored = _score_requests(model, tokenizer, requests, **policy,
                                         batch_observer=observe, deadline=deadline)
                seconds = time.perf_counter() - endpoint_started
                scored_tokens = sum(item["n_tokens"] for item in scored)
                del scored
                input_tokens = sum(item["input_tokens"] for item in observations)
                result["endpoints"][endpoint] = {
                    "task_count": len(rows), "request_count": len(requests),
                    "scored_tokens": scored_tokens, "input_tokens": input_tokens,
                    "padded_input_tokens": sum(
                        item["padded_input_tokens"] for item in observations),
                    "batch_count": len(observations),
                    "forward_calls": sum(item["forward_calls"] for item in observations),
                    "seconds": seconds, "seconds_per_task": seconds / len(rows),
                    "seconds_per_request": seconds / len(requests),
                    "scored_tokens_per_second": scored_tokens / seconds,
                    "input_tokens_per_second": input_tokens / seconds,
                }
        stage = "initial_adapter_revalidation"
        generation._no_runtime_hooks(model)
        _require(checkpoint._loaded_adapter_manifest(model) == descriptor["tensor_manifest"],
                 "pilot changed initial adapters")
        _require(generation._file_hash(audit_path) == audit_sha256
                 and generation._file_hash(data._relative(data_root, "manifest.json"))
                 == data_manifest_sha256, "pilot inputs changed")
        _require(all(generation._file_hash(data._relative(data_root, item["tasks_file"]["path"]))
                     == item["tasks_file"]["sha256"] for item in binding["endpoints"].values()),
                 "pilot development task files changed")
        _require(_source_hashes() == result["source_files_sha256"], "pilot sources changed")
        result.update(status="COMPLETE", selected_shard_complete=True,
                      endpoint_scoring_seconds=sum(item["seconds"]
                                                   for item in result["endpoints"].values()))
        _require(time.monotonic() < deadline, "pilot controller deadline exceeded")
    except BaseException as error:
        interrupted = _remember_failure(root, result, stage, error, interrupted)
    finally:
        cleanup_started = time.perf_counter()
        interrupted = _cleanup(tensors, rng, flags, warn_only, root, result, interrupted)
        tensors = None
        model = tokenizer = metadata = rng = None
        interrupted = _cleanup_owned(torch, allocated, root, result, interrupted)
        result["cleanup_seconds"] = time.perf_counter() - cleanup_started
        result["wall_seconds"] = time.perf_counter() - started
        if result["status"] == "COMPLETE" and time.monotonic() >= deadline:
            _failure(root, result, "final_controller_deadline", ValueError("deadline exceeded"))
        result["files"] = generation._files(root)
        _write_manifest(root / "manifest.json", result)
    if interrupted is not None:
        raise interrupted("interruption preserved in failure evidence")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    endpoint = commands.add_parser("run-endpoint", help="frozen final-checkpoint evaluation shard")
    pilot = commands.add_parser("pilot", help="initial seed23 adapter, development train rows only")
    for command in (endpoint, pilot):
        for flag in ("data-root", "matrix-root", "audit-path", "v1-protocol-path", "cache-dir",
                     "output"):
            command.add_argument(f"--{flag}", type=Path, required=True)
        command.add_argument("--audit-sha256", required=True)
    endpoint.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    endpoint.add_argument("--start", type=int, required=True)
    endpoint.add_argument("--stop", type=int, required=True)
    endpoint.add_argument("--arm", choices=("bf16_l2l", "oracle_block64_e4m3"), required=True)
    endpoint.add_argument("--seed", type=int, required=True)
    endpoint.add_argument("--length", type=int, required=True)
    endpoint.add_argument("--protocol-path", type=Path, required=True)
    endpoint.add_argument("--protocol-sha256", required=True)
    pilot.add_argument("--data-manifest-sha256", required=True)
    pilot.add_argument("--max-tokens-per-batch", type=int, default=DEFAULT_MAX_TOKENS_PER_BATCH)
    pilot.add_argument("--deadline-seconds", type=int, default=1800)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    result = run_endpoint(**arguments) if command == "run-endpoint" else run_pilot(**arguments)
    Console().print({"schema": result["schema"], "status": result["status"],
                     "output": str(arguments["output"])})
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
