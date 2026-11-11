"""Native-HF generation for one immutable Stage2 shard and development-only profiling.

The parent loads the pinned model and final checkpoint, freezes the complete task
manifest, and schedules every shard. This module never loads weights, downloads
code, executes completions, selects checkpoints, or retries generation. Source
contracts are implemented here; GPU generation and arithmetic admission still
require the parent's recorded runtime evidence.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import platform
import re
import time
import traceback
from collections import Counter
from contextlib import contextmanager, nullcontext
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator, Sequence

if __package__:
    from . import p3_realdata_model as native
    from .p3_fp8_smoke import MemoryObservation
else:
    import p3_realdata_model as native
    from p3_fp8_smoke import MemoryObservation

BATCH_SIZE = 8
PROFILE_NEW_TOKENS = 16
PROFILE_MAX_PROMPT_TOKENS = 256
_CORE_FIELDS = frozenset((
    "benchmark", "task_id", "source_id", "source_ref", "source_row_sha256",
    "prompt", "prompt_sha256", "generation_policy",
))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPUTE_TYPE = re.compile(
    r"(?:computeType|compute_type|compute type|CUBLASLT_MATMUL_DESC_COMPUTE_TYPE)"
    r"[\"']?\s*[=:]\s*(?:type\s*=\s*[A-Za-z_][A-Za-z_0-9]*\s*;\s*val\s*=\s*)?"
    r"[\"']?(?!type\b)([A-Za-z_][A-Za-z_0-9]*|[0-9]+)", re.IGNORECASE,
)
_LOG_KEYS = (
    "CUBLAS_LOGINFO_DBG", "CUBLAS_LOGDEST_DBG", "CUBLASLT_LOG_LEVEL",
    "CUBLASLT_LOG_MASK", "CUBLASLT_LOG_FILE",
)
PRIMARY_SOURCES = {
    "cublas_logging": "https://docs.nvidia.com/cuda/cublas/index.html#cublasloggerconfigure",
    "cublaslt_logging": "https://docs.nvidia.com/cuda/cublas/index.html#cublaslt-logging",
    "cublas_compute_type": "https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t",
    "ampere_bf16": (
        "https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html"
        "#improved-tensor-core-operations"
    ),
    "torch_profiler": "https://docs.pytorch.org/docs/2.6/profiler.html",
    "torch_native_bf16_probe": (
        "https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/__init__.py"
        "#L133-L160"
    ),
    "hf_stopping": (
        "https://github.com/huggingface/transformers/blob/"
        f"{native.NATIVE_SOURCE_COMMIT}/src/transformers/generation/stopping_criteria.py"
    ),
    "hf_generation": (
        "https://github.com/huggingface/transformers/blob/"
        f"{native.NATIVE_SOURCE_COMMIT}/src/transformers/generation/utils.py"
    ),
}


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _object_hash(value: Any) -> str:
    return _text_hash(_compact(value))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_compact(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append(handle: Any, value: dict) -> None:
    handle.write(_compact(value) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _frozen_policy(benchmark: str) -> dict:
    package = f"{__package__}." if __package__ else ""
    if benchmark in ("gsm8k", "humaneval"):
        module = import_module(f"{package}p3_stage2_core_data")
        return copy.deepcopy(module.GENERATION_POLICIES[benchmark])
    if benchmark in ("apps", "math"):
        module = import_module(f"{package}p3_stage2_{benchmark}")
        return copy.deepcopy(module.GENERATION_POLICY)
    raise ValueError(f"unsupported frozen benchmark: {benchmark!r}")


def _validated_records(records: Sequence[dict], benchmark: str, *, profile: bool = False) -> list:
    policy = _frozen_policy(benchmark)
    if not records:
        raise ValueError("selected shard must not be empty")
    seen: dict[str, set[str]] = {
        key: set() for key in ("task_id", "source_id", "source_row_sha256", "prompt_sha256")
    }
    admitted = []
    for ordinal, original in enumerate(records):
        if not isinstance(original, dict):
            raise ValueError(f"record {ordinal} must be a dictionary")
        fields = _CORE_FIELDS | ({"profile_provenance"} if profile else set())
        missing, extra = fields - original.keys(), original.keys() - fields
        if missing or extra:
            raise ValueError(
                f"record {ordinal} fields: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        # A JSON roundtrip also isolates the shard from mutable caller dictionaries.
        row = json.loads(_compact(original))
        if row["benchmark"] != benchmark:
            raise ValueError(f"record {ordinal} benchmark mismatch")
        if not isinstance(row["task_id"], str) or not row["task_id"].strip():
            raise ValueError(f"record {ordinal} missing/invalid task_id")
        source_id = row["source_id"]
        if not ((type(source_id) is int and source_id >= 0)
                or isinstance(source_id, str) and source_id.strip()):
            raise ValueError(f"record {ordinal} missing/invalid source_id")
        if not isinstance(row["source_ref"], str) or not row["source_ref"].strip():
            raise ValueError(f"record {ordinal} missing/invalid source_ref")
        for key in ("source_row_sha256", "prompt_sha256"):
            if not isinstance(row[key], str) or _SHA256.fullmatch(row[key]) is None:
                raise ValueError(f"record {ordinal} missing/invalid {key}")
        if not isinstance(row["prompt"], str) or not row["prompt"]:
            raise ValueError(f"record {ordinal} prompt must be nonempty text")
        if row["prompt_sha256"] != _text_hash(row["prompt"]):
            raise ValueError(f"record {ordinal} prompt_sha256 mismatch")
        if _compact(row["generation_policy"]) != _compact(policy):
            raise ValueError(f"record {ordinal} differs from frozen generation_policy")
        for key, values in seen.items():
            identity = _compact(row[key])
            if identity in values:
                raise ValueError(f"record {ordinal} duplicate {key}")
            values.add(identity)
        if profile:
            provenance = row["profile_provenance"]
            if (not isinstance(provenance, dict)
                    or set(provenance) != {"scope", "evidence_ref"}
                    or provenance["scope"] not in (
                        "previously_consumed", "development", "synthetic",
                    )
                    or not isinstance(provenance["evidence_ref"], str)
                    or not provenance["evidence_ref"].strip()):
                raise ValueError(f"record {ordinal} needs explicit non-final profile_provenance")
        admitted.append(row)
    return admitted


def _stop_strings(policy: dict) -> tuple[str, ...]:
    return tuple(policy.get("stop_strings", policy.get("literal_text_stops", ())))


def _check_context(token_rows: Sequence[Sequence[int]], *, context_limit: int, budget: int) -> int:
    if type(context_limit) is not int or context_limit < 1 or type(budget) is not int or budget < 1:
        raise ValueError("context limit and budget must be positive integers")
    if not token_rows or any(not row for row in token_rows):
        raise ValueError("empty tokenized prompt is not admitted")
    width = max(map(len, token_rows))
    if width + budget > context_limit:
        raise ValueError(
            f"context budget exceeded: padded_input_width={width}, max_new_tokens={budget}, "
            f"context_limit={context_limit}; no truncation, dropping, or budget reduction"
        )
    return width


def _left_pad(token_rows: Sequence[Sequence[int]], *, device: Any) -> tuple[Any, Any]:
    import torch

    if not token_rows or any(not row for row in token_rows):
        raise ValueError("cannot pad empty tokenized prompts")
    width = max(map(len, token_rows))
    ids = torch.zeros((len(token_rows), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, tokens in enumerate(token_rows):
        ids[index, width - len(tokens):] = torch.tensor(tokens, dtype=torch.long, device=device)
        mask[index, width - len(tokens):] = 1
    return ids, mask


def _make_stop_tracker(
    tokenizer: Any, stop_strings: Sequence[str], *, input_width: int, batch_size: int,
    budget: int, device: Any,
) -> Any:
    import torch
    from transformers import StoppingCriteria, StopStringCriteria

    literal = StopStringCriteria(tokenizer, list(stop_strings)) if stop_strings else None

    class ContinuationStops(StoppingCriteria):
        def __init__(self) -> None:
            self.terminal_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
            self.literal_stops = torch.zeros(batch_size, dtype=torch.bool, device=device)
            self.eos_stops = torch.zeros_like(self.literal_stops)

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            continuation = input_ids[:, input_width:]
            steps = continuation.shape[1]
            if steps == 0:
                return self.terminal_steps.ne(0)
            eos = continuation[:, -1].eq(0)
            hit = literal(continuation, scores) if literal is not None else torch.zeros_like(eos)
            newly_finished = self.terminal_steps.eq(0) & (eos | hit | (steps >= budget))
            self.terminal_steps.masked_fill_(newly_finished, steps)
            self.literal_stops |= newly_finished & hit
            self.eos_stops |= newly_finished & eos
            return self.terminal_steps.ne(0)

    return ContinuationStops()


def _extract_continuation(
    tokenizer: Any, sequence: Sequence[int], *, input_width: int, terminal_step: int,
    budget: int, stop_strings: Sequence[str],
) -> dict:
    returned = list(sequence[input_width:])
    if not 1 <= terminal_step <= min(len(returned), budget):
        raise ValueError("missing or invalid per-row terminal step")
    raw, padding = returned[:terminal_step], returned[terminal_step:]
    if any(token != 0 for token in padding):
        raise ValueError("non-padding tokens after termination")
    eos_index = raw.index(0) if 0 in raw else None
    if eos_index is not None and eos_index != len(raw) - 1:
        raise ValueError("terminal step includes tokens after first generated EOS")
    prefix_ids = raw if eos_index is None else raw[:eos_index]
    decode = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    text = tokenizer.decode(prefix_ids, **decode)
    matches = [(text.find(stop), index, stop) for index, stop in enumerate(stop_strings)
               if text.find(stop) >= 0]
    match = min(matches) if matches else None
    scored = text[:match[0]] if match else text
    cap_hit = terminal_step == budget
    reason = "literal_stop" if match else "eos" if eos_index is not None else "max_new_tokens"
    if reason == "max_new_tokens" and not cap_hit:
        raise ValueError("unexplained early termination")
    raw_text = tokenizer.decode(raw, **decode)
    return {
        "returned_continuation_token_ids": returned,
        "returned_continuation_token_ids_sha256": _object_hash(returned),
        "raw_continuation_token_ids": raw,
        "raw_continuation_token_ids_sha256": _object_hash(raw),
        "raw_continuation": raw_text, "raw_continuation_sha256": _text_hash(raw_text),
        "batch_padding_token_ids": padding, "batch_padding_token_count": len(padding),
        "pre_eos_continuation_token_ids": prefix_ids,
        "pre_eos_token_count": len(prefix_ids),
        "scored_continuation": scored, "scored_continuation_sha256": _text_hash(scored),
        "scored_continuation_utf8_bytes": len(scored.encode("utf-8")),
        "scored_token_ids": None,
        "scored_token_ids_note": "Stops may split a token; scored text is never retokenized.",
        "new_token_count": terminal_step, "new_token_count_includes_terminal_eos": True,
        "eos_emitted": eos_index is not None, "eos_token_index": eos_index,
        "cap_hit": cap_hit, "stop_reason": reason,
        "stop_string": match[2] if match else None,
        "stop_character_offset": match[0] if match else None,
        "stop_offset_unit": "Unicode code points in decoded pre-EOS continuation only",
    }


def _no_runtime_hooks(model: Any) -> None:
    import torch

    for name, module in model.named_modules():
        if any(getattr(module, key, None) for key in (
            "_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks",
        )):
            raise RuntimeError(f"native inference refuses active module hooks: {name!r}")
        if "forward" in vars(module) or getattr(module, "_hf_hook", None) is not None:
            raise RuntimeError(f"native inference refuses replaced/offloaded forward: {name!r}")
    for key in ("_global_forward_pre_hooks", "_global_forward_hooks",
                "_global_backward_pre_hooks", "_global_backward_hooks"):
        if getattr(torch.nn.modules.module, key, None):
            raise RuntimeError(f"native inference refuses global module hooks: {key}")


def _native_guard(model: Any, tokenizer: Any) -> dict:
    import torch
    import transformers
    from transformers import GenerationMixin

    native._parts(model)
    dtypes = native._adapter_guard(model)
    if (transformers.__version__ != native.TRANSFORMERS_VERSION
            or version("peft") != native.PEFT_VERSION):
        raise RuntimeError("installed Transformers/PEFT versions differ from pinned native loader")
    if getattr(model.generate, "__func__", None) is not GenerationMixin.generate:
        raise RuntimeError("generation must use the installed native HF GenerationMixin.generate")
    if model.is_gradient_checkpointing:
        raise RuntimeError(
            "generation refuses active gradient checkpointing; caller must restore loader state"
        )
    _no_runtime_hooks(model)
    metadata = getattr(model, "_p3_realdata_metadata", {})
    if metadata.get("repo_id") != native.MODEL_ID or metadata.get("revision") != native.REVISION:
        raise RuntimeError("missing pinned caller-loaded model provenance")
    spec = metadata.get("tokenizer_spec", {})
    if (spec.get("repo_id") != native.MODEL_ID or spec.get("revision") != native.REVISION
            or not spec.get("files_sha256") or len(tokenizer) != model.config.vocab_size
            or tokenizer.eos_token_id != 0 or tokenizer.bos_token_id != 0
            or tokenizer.pad_token_id != 0):
        raise RuntimeError("tokenizer identity/vocabulary/BOS/EOS/pad differs from pinned source")
    device = model.get_input_embeddings().weight.device
    if (device.type not in ("cpu", "cuda")
            or any(param.device != device for param in model.parameters())):
        raise RuntimeError(
            "generation requires the caller's one-device resident model; no device fallback"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("caller model CUDA device is unavailable")
    return {"model": native.MODEL_ID, "model_revision": native.REVISION,
            "device": str(device), "parameter_dtypes": dtypes,
            "tokenizer_spec": copy.deepcopy(spec),
            "native_source": copy.deepcopy(metadata.get("source", {})),
            "library_versions": copy.deepcopy(metadata.get("library_versions", {}))}


def _backend_state(tokenizer: Any) -> tuple[Any, Any, Any]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return (backend, copy.deepcopy(backend.padding), copy.deepcopy(backend.truncation)) if (
        backend is not None
    ) else (None, None, None)


def _restore_backend(state: tuple) -> None:
    backend, padding, truncation = state
    if backend is None:
        return
    if backend.padding != padding:
        if padding is None:
            backend.no_padding()
        else:
            backend.enable_padding(**padding)
    if backend.truncation != truncation:
        if truncation is None:
            backend.no_truncation()
        else:
            backend.enable_truncation(**truncation)


@contextmanager
def _inference_session(model: Any, tokenizer: Any) -> Iterator[None]:
    import torch

    modules = list(model.modules())
    modes = [(module, module.training) for module in modules]
    bindings = [(module, module.config) for module in modules if hasattr(module, "config")]
    generation_config = model.generation_config
    configurations = {id(value): value for _, value in bindings}
    configurations[id(generation_config)] = generation_config
    snapshots = [(value, copy.deepcopy(vars(value))) for value in configurations.values()]
    token_attributes = {key: copy.deepcopy(getattr(tokenizer, key)) for key in (
        "padding_side", "truncation_side", "model_max_length",
    )}
    backend = _backend_state(tokenizer)
    parameters = [(name, param, param._version, param.dtype, param.device, param.requires_grad)
                  for name, param in model.named_parameters()]
    device_type = next(model.parameters()).device.type
    try:
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        for value, snapshot in snapshots:
            if vars(value) != snapshot:
                vars(value).clear()
                vars(value).update(snapshot)
        for module, original in bindings:
            if module.config is not original:
                module.config = original
        if model.generation_config is not generation_config:
            model.generation_config = generation_config
        for key, original in token_attributes.items():
            if getattr(tokenizer, key) != original:
                setattr(tokenizer, key, original)
        _restore_backend(backend)
        for module, training in modes:
            module.training = training
        for name, parameter, counter, dtype, device, requires_grad in parameters:
            if (parameter._version != counter or parameter.dtype != dtype
                    or parameter.device != device or parameter.requires_grad != requires_grad):
                raise RuntimeError(f"inference changed a model parameter: {name}")


def _tokens(tokenizer: Any, records: Sequence[dict], *, profile: bool = False) -> list[list[int]]:
    rows = []
    for row in records:
        encoded = tokenizer(
            row["prompt"], add_special_tokens=False, truncation=False, padding=False,
            return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]
        if not encoded or any(type(token) is not int or not 0 <= token < len(tokenizer)
                              for token in encoded):
            raise ValueError(f"invalid untruncated tokenization: {row['task_id']}")
        if profile and len(encoded) > PROFILE_MAX_PROMPT_TOKENS:
            raise ValueError(
                f"profile prompt exceeds declared {PROFILE_MAX_PROMPT_TOKENS}-token tiny probe: "
                f"{row['task_id']}; parent must supply another development shard, never truncate"
            )
        rows.append(encoded)
    return rows


def _context_limit(model: Any, tokenizer: Any) -> int:
    limit = model.config.max_position_embeddings
    token_limit = tokenizer.model_max_length
    if type(limit) is not int or limit < 1:
        raise ValueError("missing model context limit")
    if type(token_limit) is int and 0 < token_limit < 10**9:
        limit = min(limit, token_limit)
    return limit


def _generation_config(model: Any, *, width: int, budget: int) -> Any:
    from transformers import GenerationConfig

    overrides = {
        "max_length": width + budget, "max_new_tokens": budget,
        "min_length": 0, "min_new_tokens": 0, "max_time": None, "stop_strings": None,
        "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
        "use_cache": True, "cache_implementation": "dynamic", "disable_compile": True,
        "repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0,
        "remove_invalid_values": False, "renormalize_logits": False, "token_healing": False,
        "use_mtp": False, "is_assistant": False, "bos_token_id": 0, "eos_token_id": 0,
        "pad_token_id": 0, "output_attentions": False, "output_hidden_states": False,
        "output_scores": False, "output_logits": False, "return_dict_in_generate": False,
        "temperature": 1.0, "top_p": 1.0, "top_k": 50, "typical_p": 1.0,
    }
    config = GenerationConfig(**overrides)
    config, unused = model._prepare_generation_config(config)
    if unused:
        raise RuntimeError(f"unexpected native generation model kwargs: {sorted(unused)}")
    for key, value in overrides.items():
        if getattr(config, key) != value:
            raise RuntimeError(f"caller generation configuration overrides frozen {key}")
    for key in (
        "bad_words_ids", "forced_bos_token_id", "forced_eos_token_id", "force_words_ids",
        "constraints", "suppress_tokens", "begin_suppress_tokens", "sequence_bias",
        "watermarking_config", "exponential_decay_length_penalty", "guidance_scale",
        "cache_config", "max_cache_len", "prompt_lookup_num_tokens", "assistant_early_exit",
        "penalty_alpha", "dola_layers", "prefill_chunk_size", "compile_config",
        "continuous_batching_config", "speculation_type", "min_p", "top_h",
    ):
        if getattr(config, key, None) is not None:
            raise RuntimeError(f"undeclared caller generation control: {key}")
    known = set(GenerationConfig().to_dict())
    if set(config.to_dict()) - known:
        raise RuntimeError("custom generation configuration fields are not admitted")
    config.validate(strict=True)
    return config


def _device_context(device: Any) -> Any:
    if device.type == "cuda":
        import torch

        return torch.cuda.device(device)
    return nullcontext()


def _prepared_batch(
    model: Any, tokenizer: Any, token_rows: Sequence[Sequence[int]], stops: Sequence[str],
    budget: int,
) -> tuple[Any, Any, Any, Any]:
    device = model.get_input_embeddings().weight.device
    ids, mask = _left_pad(token_rows, device=device)
    config = _generation_config(model, width=ids.shape[1], budget=budget)
    tracker = _make_stop_tracker(
        tokenizer, stops, input_width=ids.shape[1], batch_size=len(token_rows), budget=budget,
        device=device,
    )
    return ids, mask, config, tracker


def _batch_inputs(ids: Any, mask: Any, token_rows: Sequence[Sequence[int]], config: Any) -> dict:
    padded, attention = ids.tolist(), mask.tolist()
    full_config = config.to_dict()
    return {
        "unpadded_input_ids": [list(row) for row in token_rows],
        "padded_input_ids": padded, "attention_mask": attention,
        "padded_input_ids_sha256": _object_hash(padded),
        "attention_mask_sha256": _object_hash(attention),
        "input_dtype": str(ids.dtype), "attention_mask_dtype": str(mask.dtype),
        "device": str(ids.device), "padded_input_width": ids.shape[1],
        "actual_batch_size": ids.shape[0], "nominal_batch_size": BATCH_SIZE,
        "full_generation_config": full_config,
        "generation_config_sha256": _object_hash(full_config),
        "stopping": (
            "HF EosTokenCriteria/MaxLengthCriteria plus continuation-only StopStringCriteria"
        ),
        "extra_logits_processors": [], "sampling_parameters_active": False,
    }


def _measured_generate(model: Any, batch: tuple, measurement: dict) -> Any:
    from transformers import StoppingCriteriaList

    ids, mask, config, tracker = batch
    scope = (
        "One native HF generate call, including cache allocation, prefill, decoding and internal "
        "cleanup. Includes resident model/adapter, input tensors, Python/runtime and existing "
        "allocator caches. Excludes model/checkpoint loading, tokenization, stop-table/config "
        "preparation, decoding output text and artifact writes. RAM and CUDA are separate facts; "
        "allocator peaks are process/device scoped, never sums across shards or resources."
    )
    observed = MemoryObservation(ids.device.type, scope=scope)
    started = time.perf_counter()
    try:
        with _device_context(ids.device), observed:
            return model.generate(
                input_ids=ids, attention_mask=mask, generation_config=config,
                stopping_criteria=StoppingCriteriaList([tracker]), synced_gpus=False,
            )
    finally:
        measurement.update(observed.result)
        measurement["call_including_observer_wall_seconds"] = time.perf_counter() - started
        measurement["reservation_sum_is_peak"] = False
        measurement["cuda_stats_reset_for_this_call"] = ids.device.type == "cuda"


def _batch_rows(
    tokenizer: Any, records: Sequence[dict], token_rows: Sequence[Sequence[int]], batch: tuple,
    output: Any, *, budget: int, stops: Sequence[str], batch_index: int, identity: dict,
) -> Iterator[dict]:
    import torch

    ids, mask, config, tracker = batch
    if (not isinstance(output, torch.Tensor) or output.dtype != torch.long or output.ndim != 2
            or output.shape[0] != len(records) or output.shape[1] <= ids.shape[1]
            or output.shape[1] > ids.shape[1] + budget):
        raise RuntimeError("native HF returned invalid shape/dtype/coverage")
    if not torch.equal(output[:, :ids.shape[1]], ids):
        raise RuntimeError("native HF output prefix differs from the complete padded input")
    sequences = output.tolist()
    terminal = tracker.terminal_steps.tolist()
    literal, eos = tracker.literal_stops.tolist(), tracker.eos_stops.tolist()
    for index, record in enumerate(records):
        extracted = _extract_continuation(
            tokenizer, sequences[index], input_width=ids.shape[1], terminal_step=terminal[index],
            budget=budget, stop_strings=stops,
        )
        if literal[index] != (extracted["stop_reason"] == "literal_stop"):
            raise RuntimeError(
                f"HF literal stop and decoded continuation disagree: {record['task_id']}"
            )
        if eos[index] != extracted["eos_emitted"]:
            raise RuntimeError(f"HF EOS tracker/continuation mismatch: {record['task_id']}")
        row = {
            **record, **identity, "batch_index": batch_index, "row_in_batch": index,
            "batch_task_ids": [item["task_id"] for item in records],
            "nominal_batch_size": BATCH_SIZE, "actual_batch_size": len(records),
            "final_partial_batch": len(records) < BATCH_SIZE,
            "unpadded_input_ids": list(token_rows[index]),
            "unpadded_input_ids_sha256": _object_hash(list(token_rows[index])),
            "padded_input_ids": ids[index].tolist(), "attention_mask": mask[index].tolist(),
            "input_width": ids.shape[1], "full_generation_config": config.to_dict(),
            "output_dtype": str(output.dtype), **extracted,
        }
        row["row_sha256"] = _object_hash(row)
        yield row


def _precision_flags() -> dict:
    import torch

    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "autocast_enabled_cuda": torch.is_autocast_enabled("cuda"),
        "autocast_enabled_cpu": torch.is_autocast_enabled("cpu"),
        "autocast_dtype_cuda": str(torch.get_autocast_dtype("cuda")),
        "autocast_dtype_cpu": str(torch.get_autocast_dtype("cpu")),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "arithmetic_environment": {key: os.environ.get(key) for key in (
            "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
            "CUBLAS_EMULATE_SINGLE_PRECISION", "CUBLAS_EMULATE_DOUBLE_PRECISION",
        )},
    }


def _source(target: Any) -> dict:
    target = inspect.unwrap(target)
    path = inspect.getsourcefile(target)
    if path is None or not Path(path).is_file():
        return {"path": path, "sha256": None, "available": False}
    return {"path": path, "sha256": _file_hash(Path(path)), "available": True}


def _runtime_provenance() -> dict:
    import torch
    from transformers import GenerationMixin, StopStringCriteria

    libraries = {}
    for package in ("torch", "transformers", "peft", "tokenizers", "nvidia-cublas-cu12",
                    "nvidia-cublas-cu13", "nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime-cu13"):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None
    loaded = []
    maps = Path("/proc/self/maps")
    if maps.is_file():
        loaded = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                         if any(name in line for name in ("libcublas", "libcuda", "libcudnn"))})
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "process_id": os.getpid(), "libraries": libraries,
        "torch_cuda_build": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "loaded_cuda_library_paths": loaded,
        "installed_sources": {"generation": _source(GenerationMixin.generate),
                              "stopping": _source(StopStringCriteria),
                              "runner": {"path": __file__, "sha256": _file_hash(Path(__file__))}},
        "reference_sources": PRIMARY_SOURCES,
    }


def _files(output_dir: Path) -> dict:
    return {path.relative_to(output_dir).as_posix(): {
        "bytes": path.stat().st_size, "sha256": _file_hash(path),
    } for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"}


def _failure(result: dict, output_dir: Path, stage: str, error: BaseException) -> None:
    failure = {"stage": stage, "error_type": type(error).__name__, "message": str(error),
               "traceback": traceback.format_exc(), "regeneration_or_retry_attempted": False,
               "completed_rows": result.get("completed_rows", 0)}
    result.update(status="FAILED", selected_shard_complete=False, failure=failure)
    _write_json(output_dir / "failure.json", failure)


def run_generation(
    model: Any, tokenizer: Any, records: Sequence[dict], *, output_dir: Path, benchmark: str,
    seed: int, length: int, arm: str, protocol_sha256: str, checkpoint_sha256: str,
) -> dict:
    """Generate once for one parent-selected immutable shard, retaining partial failure evidence.

    A pre-existing output directory raises FileExistsError without touching it.
    All subsequent admission/runtime errors return FAILED and write failure.json.
    COMPLETE describes only the supplied shard, never the parent's global task set.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    identity = {"benchmark": benchmark, "seed": seed, "length": length, "arm": arm,
                "protocol_sha256": protocol_sha256, "checkpoint_sha256": checkpoint_sha256}
    result = {
        "schema": "p3-stage2-generation-v1", "status": "STARTED", **identity,
        "requested_rows": None, "completed_rows": 0, "selected_shard_complete": False,
        "global_required_coverage_verified": False, "task_independence_verified": False,
        "quality_non_degradation_admitted": False, "nominal_batch_size": BATCH_SIZE,
        "checkpoint_authority": (
            "Caller-selected final checkpoint SHA256; runner does not load weights."
        ),
        "generated_code_executed": False, "retries": 0, "response_repair": False,
        "configuration_runtime_proof": "PENDING_PARENT_GPU_QUALIFICATION",
        "reservation_sum_is_peak": False,
    }
    stage = "request_admission"
    started = time.perf_counter()
    with (
        (output_dir / "generations.jsonl").open(
            "x", encoding="utf-8", newline="\n",
        ) as generations,
        (output_dir / "batches.jsonl").open(
            "x", encoding="utf-8", newline="\n",
        ) as batches,
    ):
        try:
            if type(seed) is not int or seed not in (23, 41, 67, 89, 113):
                raise ValueError("seed differs from frozen Stage2 seed set")
            if type(length) is not int or length not in (1024, 4096):
                raise ValueError("length must identify the L1024 or L4096 trained checkpoint")
            if arm not in ("bf16_l2l", "oracle_block64_e4m3"):
                raise ValueError("final generation accepts only the two frozen trained arms")
            for key, digest in (("protocol_sha256", protocol_sha256),
                                ("checkpoint_sha256", checkpoint_sha256)):
                if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                    raise ValueError(f"missing/invalid {key}")
            incoming = list(records)
            result["requested_rows"] = len(incoming)
            stage = "record_admission"
            admitted = _validated_records(incoming, benchmark)
            _write_json(output_dir / "request.json", {
                **identity, "records_sha256": _object_hash(admitted),
                "ordered_task_ids": [row["task_id"] for row in admitted],
                "records": admitted, "batch_size": BATCH_SIZE,
                "final_partial_batch_rule": (
                    "Keep the immutable remainder, never pad with duplicate tasks."
                ),
            })
            result["records_sha256"] = _object_hash(admitted)
            stage = "native_model_admission"
            result["model_provenance"] = _native_guard(model, tokenizer)
            result["runtime_provenance"] = _runtime_provenance()
            result["caller_precision_flags"] = _precision_flags()
            with _inference_session(model, tokenizer):
                result["effective_precision_flags"] = _precision_flags()
                stage = "tokenization_and_context_admission"
                token_rows = _tokens(tokenizer, admitted)
                policy = _frozen_policy(benchmark)
                budget, stops = policy["max_new_tokens"], _stop_strings(policy)
                limit = _context_limit(model, tokenizer)
                _check_context(token_rows, context_limit=limit, budget=budget)
                result.update(context_limit=limit, max_new_tokens=budget,
                              ordered_input_ids_sha256=_object_hash(token_rows))
                for start in range(0, len(admitted), BATCH_SIZE):
                    batch_index = start // BATCH_SIZE
                    rows = admitted[start:start + BATCH_SIZE]
                    tokens = token_rows[start:start + BATCH_SIZE]
                    stage = f"batch_{batch_index}_preparation"
                    batch = _prepared_batch(model, tokenizer, tokens, stops, budget)
                    audit = {"batch_index": batch_index,
                             "task_ids": [row["task_id"] for row in rows],
                             "final_partial_batch": len(rows) < BATCH_SIZE,
                             **_batch_inputs(batch[0], batch[1], tokens, batch[2])}
                    _append(batches, {"event": "batch_started", **audit})
                    stage = f"batch_{batch_index}_native_generate"
                    measurement: dict = {}
                    try:
                        output = _measured_generate(model, batch, measurement)
                    except BaseException:
                        _append(batches, {"event": "batch_failed", "batch_index": batch_index,
                                          "measurement": measurement})
                        raise
                    _append(batches, {
                        "event": "batch_returned", "batch_index": batch_index,
                        "returned_sequences": output.tolist(),
                        "returned_sequences_sha256": _object_hash(output.tolist()),
                        "terminal_steps": batch[3].terminal_steps.tolist(),
                        "literal_stops": batch[3].literal_stops.tolist(),
                        "eos_stops": batch[3].eos_stops.tolist(), "measurement": measurement,
                    })
                    stage = f"batch_{batch_index}_continuation_extraction"
                    for row in _batch_rows(
                        tokenizer, rows, tokens, batch, output, budget=budget, stops=stops,
                        batch_index=batch_index, identity=identity,
                    ):
                        row["measurement_ref"] = {
                            "path": "batches.jsonl", "batch_index": batch_index,
                            "event": "batch_returned",
                        }
                        row["row_sha256"] = _object_hash({key: value for key, value in row.items()
                                                         if key != "row_sha256"})
                        _append(generations, row)
                        result["completed_rows"] += 1
                    del output, batch
            stage = "caller_restoration"
            _no_runtime_hooks(model)
            if (_precision_flags() != result["caller_precision_flags"]
                    or model.config.use_cache is not False):
                raise RuntimeError("caller precision/config guards changed during generation")
            result.update(status="COMPLETE", selected_shard_complete=(
                result["completed_rows"] == result["requested_rows"]
            ), caller_configuration_restored=True, caller_module_modes_restored=True)
        except BaseException as error:
            _failure(result, output_dir, stage, error)
            if not isinstance(error, Exception):
                result["runner_wall_seconds"] = time.perf_counter() - started
                result["files"] = _files(output_dir)
                _write_json(output_dir / "manifest.json", result)
                raise
    result["runner_wall_seconds"] = time.perf_counter() - started
    result["files"] = _files(output_dir)
    _write_json(output_dir / "manifest.json", result)
    return result


def _logging_requirements() -> dict:
    return {
        "set_before_process_start": True,
        "values": {"CUBLAS_LOGINFO_DBG": "1",
                   "CUBLAS_LOGDEST_DBG": "/kaggle/working/<unique-process>/cublas.log",
                   "CUBLASLT_LOG_LEVEL": "5", "CUBLASLT_LOG_MASK": "31",
                   "CUBLASLT_LOG_FILE": "/kaggle/working/<unique-process>/cublaslt_%i.log"},
        "observed_environment": {key: os.environ.get(key) for key in _LOG_KEYS},
        "preprocess_timing_verifiable_here": False,
        "meaning": "Level 5 is API trace; mask 31 includes error/trace/hints/info/API trace.",
        "prerequisite": "Parent creates a new writable log directory before launching Python.",
        "sources": {key: PRIMARY_SOURCES[key] for key in ("cublas_logging", "cublaslt_logging")},
        "arithmetic_or_backend_overrides_required": False,
    }


def _log_positions() -> list[dict]:
    result = []
    for library, variable in (("cublas", "CUBLAS_LOGDEST_DBG"), ("cublaslt", "CUBLASLT_LOG_FILE")):
        configured = os.environ.get(variable)
        entry = {"library": library, "environment_key": variable, "configured_path": configured}
        if configured and configured not in ("stdout", "stderr"):
            log_path = configured.replace("%i", str(os.getpid())) if library == "cublaslt" \
                else configured
            path = Path(log_path)
            entry.update(path=str(path), byte_offset=path.stat().st_size if path.is_file() else 0)
        else:
            entry.update(path=None, byte_offset=None,
                         unavailable="No file destination; parent must retain stdout/stderr.")
        result.append(entry)
    return result


def _capture_logs(positions: list[dict], output_dir: Path, phase: str) -> dict:
    captures, counts = [], Counter()
    evidence_name = f"{phase}.library-compute-types.jsonl"
    with (output_dir / evidence_name).open("x", encoding="utf-8", newline="\n") as evidence:
        for position in positions:
            capture = dict(position)
            source = Path(position["path"]) if position["path"] else None
            if source is None or not source.is_file():
                capture["capture_status"] = "UNAVAILABLE_OR_NOT_FLUSHED"
                captures.append(capture)
                continue
            end = source.stat().st_size
            begin = position["byte_offset"]
            if end < begin:
                capture["capture_status"] = "SOURCE_TRUNCATED_DURING_PHASE"
                captures.append(capture)
                continue
            destination = output_dir / f"{phase}.{position['library']}.log"
            with source.open("rb") as handle, destination.open("xb") as copied:
                handle.seek(begin)
                remaining = end - begin
                while remaining:
                    chunk = handle.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise OSError("library log changed while preserving the phase byte range")
                    copied.write(chunk)
                    remaining -= len(chunk)
            capture.update(capture_status="COPIED_AVAILABLE_BYTE_RANGE", byte_end=end,
                           raw_path=destination.name, sha256=_file_hash(destination),
                           bytes=end - begin, asynchronous_library_buffering_may_omit_tail=True)
            with destination.open(encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    for match in _COMPUTE_TYPE.finditer(line):
                        field = match.group(1)
                        counts[(position["library"], field)] += 1
                        evidence.write(_compact({
                            "library": position["library"], "raw_path": destination.name,
                            "line": line_number, "compute_type_field_verbatim": field,
                            "raw_line": line.rstrip("\r\n"),
                            "scope": "Logged API field; not proof of every executed kernel.",
                        }) + "\n")
            captures.append(capture)
    return {
        "raw_logs": captures, "compute_type_evidence_path": evidence_name,
        "compute_type_fields": [{"library": library, "verbatim_value": value, "count": count}
                                for (library, value), count in sorted(counts.items())],
        "library_compute_type_status": "OBSERVED_LOG_FIELDS" if counts else "UNRESOLVED",
        "numeric_enum_interpretation": "Parent must match the installed CUDA API; not guessed.",
    }


def _tensor_metadata(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (list, tuple)):
        return [_tensor_metadata(item) for item in value]
    if isinstance(value, dict):
        return {key: _tensor_metadata(item) for key, item in value.items()}
    return None


@contextmanager
def _dtype_observers(model: Any, output_dir: Path) -> Iterator[dict]:
    from torch.utils._python_dispatch import TorchDispatchMode

    operations: Counter = Counter()
    modules: Counter = Counter()
    descriptions: dict[str, dict] = {}
    handles = []

    def observe(name: str):
        def hook(module: Any, args: Any, output: Any) -> None:
            metadata = {"module": name, "inputs": _tensor_metadata(args),
                        "output": _tensor_metadata(output)}
            key = _compact(metadata)
            descriptions[key] = metadata
            modules[key] += 1
        return hook

    class MatmulMetadata(TorchDispatchMode):
        def __torch_dispatch__(
            self, func: Any, types: Any, args: tuple = (), kwargs: Any = None,
        ) -> Any:
            output = func(*args, **(kwargs or {}))
            if str(func).split(".")[1] in ("mm", "bmm", "addmm", "matmul", "mv"):
                metadata = {
                    "operator": str(func), "inputs": _tensor_metadata(args),
                    "kwargs": _tensor_metadata(kwargs or {}), "output": _tensor_metadata(output),
                }
                key = _compact(metadata)
                descriptions[key] = metadata
                operations[key] += 1
            return output

    result: dict = {}
    try:
        selected = {"model.embed_tokens", "lm_head", "model.layers.0.self_attn.q_proj",
                    "model.layers.0.self_attn.q_proj.lora_A.default",
                    "model.layers.0.self_attn.q_proj.lora_B.default"}
        for name, module in model.named_modules():
            if name in selected:
                handles.append(module.register_forward_hook(observe(name)))
        with MatmulMetadata():
            yield result
    finally:
        for handle in handles:
            handle.remove()
        path = output_dir / "profile.tensor-dtypes.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for counter, kind in ((modules, "module_boundary"), (operations, "aten_matmul")):
                for key, count in sorted(counter.items()):
                    entry = {"kind": kind, "count": count, **descriptions[key]}
                    handle.write(_compact(entry) + "\n")
        result.update(path=path.name, module_observations=sum(modules.values()),
                      matmul_observations=sum(operations.values()),
                      accumulation_proven=False,
                      scope="Metadata observers only; no retained tensors or changed outputs.")


def _profiler_evidence(profiler: Any, output_dir: Path) -> dict:
    kernels: Counter = Counter()
    matmuls: Counter = Counter()
    path = output_dir / "profile.cuda-and-matmul-events.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for event in profiler.events():
            cuda = str(event.device_type).endswith("CUDA")
            matmul = event.name in (
                "aten::mm", "aten::bmm", "aten::addmm", "aten::matmul", "aten::linear",
            )
            if cuda or matmul:
                handle.write(_compact({
                    "name": event.name, "device_type": str(event.device_type),
                    "device_index": event.device_index, "input_shapes": event.input_shapes,
                    "cpu_time_total_us": event.cpu_time_total,
                    "device_time_total_us": event.device_time_total,
                }) + "\n")
                if cuda:
                    kernels[event.name] += 1
                if matmul:
                    matmuls[event.name] += 1
    return {
        "path": path.name, "cuda_event_count": sum(kernels.values()),
        "cuda_kernel_names_and_counts": dict(sorted(kernels.items())),
        "aten_matmul_names_and_counts": dict(sorted(matmuls.items())),
        "kernel_names_establish_arithmetic": False,
        "cuda_trace_status": "CAPTURED" if kernels else "UNRESOLVED_NO_CUDA_EVENTS",
    }


def profile_inference_backend(
    model: Any, tokenizer: Any, dev_records: Sequence[dict], *, output_dir: Path,
    protocol_sha256: str,
) -> dict:
    """Measure one unprofiled and one profiled batch8/16-token development generate call.

    Exactly eight non-final records must carry profile_provenance with scope and
    immutable evidence_ref. Neither pass is an endpoint result or throughput forecast.
    Native BF16 support and library compute-type evidence are reported separately.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "p3-stage2-inference-profile-v1", "status": "STARTED",
        "protocol_sha256": protocol_sha256, "completed_rows": 0,
        "selected_shard_complete": False, "final_test_outcomes_used": False,
        "development_scope_authority": (
            "Parent-provided profile_provenance and immutable evidence_ref; runner rejects "
            "final-only scope but does not independently establish source exposure."
        ),
        "max_new_tokens": PROFILE_NEW_TOKENS, "batch_size": BATCH_SIZE,
        "profile_max_prompt_tokens": PROFILE_MAX_PROMPT_TOKENS,
        "logging_requirements": _logging_requirements(),
        "backend_qualification": "PENDING_PARENT_KERNEL_AND_LIBRARY_REVIEW",
        "accumulation": {"status": "UNRESOLVED", "reason": (
            "Dtype and kernel labels do not establish accumulation. Logged compute-type fields "
            "apply only to their API/descriptor records; custom kernels and call correlation "
            "still require the parent's review of raw evidence."
        )},
        "full_model_admission": False, "r1_nf4_claim": False,
        "generated_code_executed": False, "full_evaluation_throughput_prediction": None,
        "quality_non_degradation_admitted": False, "reservation_sum_is_peak": False,
    }
    stage, profiler = "profile_record_admission", None
    trace = output_dir / "profile.trace.json"
    started = time.perf_counter()
    try:
        import torch

        if not isinstance(protocol_sha256, str) or _SHA256.fullmatch(protocol_sha256) is None:
            raise ValueError("missing/invalid protocol_sha256")
        incoming = list(dev_records)
        if len(incoming) != BATCH_SIZE:
            raise ValueError("profile requires exactly eight explicit development records")
        benchmark = incoming[0].get("benchmark") if isinstance(incoming[0], dict) else None
        admitted = _validated_records(incoming, benchmark, profile=True)
        _write_json(output_dir / "request.json", {
            "protocol_sha256": protocol_sha256, "records": admitted,
            "records_sha256": _object_hash(admitted),
            "scope": "development_only_not_final_outcomes",
            "bounded_new_tokens": PROFILE_NEW_TOKENS,
        })
        stage = "native_model_admission"
        result["model_provenance"] = _native_guard(model, tokenizer)
        result["runtime_provenance"] = _runtime_provenance()
        result["caller_precision_flags"] = _precision_flags()
        device = model.get_input_embeddings().weight.device
        if device.type != "cuda":
            raise RuntimeError(
                "CUDA kernel profiling requires the caller-resident CUDA model; no device fallback"
            )
        with _device_context(device):
            capability = torch.cuda.get_device_capability(device)
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            result["device"] = {
                "device": str(device), "name": torch.cuda.get_device_name(device),
                "compute_capability": list(capability),
                "native_bf16_supported_excluding_emulation": native_bf16,
                "bf16_tensor_support_including_emulation": torch.cuda.is_bf16_supported(),
                "native_support_probe": "torch.cuda.is_bf16_supported(including_emulation=False)",
                "t4_is_native_bf16": False, "precision_or_backend_switched": False,
            }
        with _inference_session(model, tokenizer):
            result["effective_precision_flags"] = _precision_flags()
            stage = "development_tokenization_and_context_admission"
            tokens = _tokens(tokenizer, admitted, profile=True)
            _check_context(
                tokens, context_limit=_context_limit(model, tokenizer), budget=PROFILE_NEW_TOKENS,
            )
            stops = _stop_strings(_frozen_policy(benchmark))
            for phase in ("smoke", "profile"):
                stage = f"{phase}_preparation"
                batch = _prepared_batch(model, tokenizer, tokens, stops, PROFILE_NEW_TOKENS)
                inputs = _batch_inputs(batch[0], batch[1], tokens, batch[2])
                _write_json(output_dir / f"{phase}.inputs.json", inputs)
                positions = _log_positions()
                measurement: dict = {}
                phase_result: dict = {"profiled": phase == "profile", "measurement": measurement,
                                      "input_path": f"{phase}.inputs.json"}
                result[phase] = phase_result
                phase_started = time.perf_counter()
                try:
                    stage = f"{phase}_native_generate"
                    if phase == "smoke":
                        output = _measured_generate(model, batch, measurement)
                    else:
                        activities = torch.profiler.supported_activities()
                        if torch.profiler.ProfilerActivity.CUDA not in activities:
                            raise RuntimeError(
                                "CUDA profiler activity is unavailable; CPU trace is insufficient"
                            )
                        profiler = torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True, profile_memory=True, with_stack=True,
                        )
                        with profiler, _dtype_observers(model, output_dir) as observations:
                            output = _measured_generate(model, batch, measurement)
                        phase_result["tensor_dtype_evidence"] = observations
                    sequences = output.tolist()
                    _write_json(output_dir / f"{phase}.returned.json", {
                        "returned_sequences": sequences,
                        "returned_sequences_sha256": _object_hash(sequences),
                        "terminal_steps": batch[3].terminal_steps.tolist(),
                        "literal_stops": batch[3].literal_stops.tolist(),
                        "eos_stops": batch[3].eos_stops.tolist(),
                    })
                    stage = f"{phase}_continuation_extraction"
                    generated, count = 0, 0
                    with (output_dir / f"{phase}.generations.jsonl").open(
                        "x", encoding="utf-8", newline="\n",
                    ) as handle:
                        for row in _batch_rows(
                            tokenizer, admitted, tokens, batch, output,
                            budget=PROFILE_NEW_TOKENS, stops=stops, batch_index=0,
                            identity={
                                "protocol_sha256": protocol_sha256, "profile_only": True,
                                "phase": phase,
                            },
                        ):
                            _append(handle, row)
                            generated += row["new_token_count"]
                            count += 1
                            phase_result["completed_rows"] = count
                            if phase == "smoke":
                                result["completed_rows"] = count
                    seconds = measurement["wall_seconds"]
                    phase_result.update(
                        generated_tokens=generated,
                        generated_token_count_includes_eos_excludes_batch_padding=True,
                        observed_new_tokens_per_generate_second=generated / seconds,
                        throughput_scope=(
                            "This batch8, at-most16-new-token development call only. No warmup "
                            "before smoke; library logging, if enabled, remains included. Profile "
                            "pass also includes profiler/observer overhead and follows smoke."
                        ),
                    )
                    del output, batch, sequences
                    if phase == "profile":
                        stage = "profile_trace_export"
                        profiler.export_chrome_trace(str(trace))
                        phase_result["trace_path"] = trace.name
                        phase_result["kernel_evidence"] = _profiler_evidence(profiler, output_dir)
                finally:
                    phase_result["phase_wall_seconds"] = time.perf_counter() - phase_started
                    phase_result["library_logs"] = _capture_logs(positions, output_dir, phase)
            result["completed_rows"] = BATCH_SIZE
        stage = "caller_restoration"
        _no_runtime_hooks(model)
        if (_precision_flags() != result["caller_precision_flags"]
                or model.config.use_cache is not False):
            raise RuntimeError("caller precision/configuration guards changed during profiling")
        cuda_events = result["profile"]["kernel_evidence"]["cuda_event_count"]
        result.update(status="PROFILE_CAPTURED" if cuda_events else "PROFILE_INCOMPLETE",
                      caller_configuration_restored=True, caller_module_modes_restored=True)
        result["timing_comparison"] = {
            "unprofiled_generate_seconds": result["smoke"]["measurement"]["wall_seconds"],
            "profiled_generate_seconds": result["profile"]["measurement"]["wall_seconds"],
            "profile_phase_including_setup_export_seconds": result["profile"]["phase_wall_seconds"],
            "causal_profiler_overhead_estimate": False,
            "reason": "Cold/warm sequential calls; not a full-evaluation forecast.",
        }
    except BaseException as error:
        _failure(result, output_dir, stage, error)
        if profiler is not None and not trace.exists():
            try:
                profiler.export_chrome_trace(str(trace))
                result["partial_trace_path"] = trace.name
            except Exception as trace_error:
                result["partial_trace_error"] = str(trace_error)
        if not isinstance(error, Exception):
            result["runner_wall_seconds"] = time.perf_counter() - started
            result["files"] = _files(output_dir)
            _write_json(output_dir / "manifest.json", result)
            raise
    result["runner_wall_seconds"] = time.perf_counter() - started
    result["files"] = _files(output_dir)
    _write_json(output_dir / "manifest.json", result)
    return result
