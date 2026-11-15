"""Pinned Granite BF16/attention-LoRA controls for the real-data P3 benchmark.

This is resident, benchmark-only layer-major replay, not production L2LStep or a
streaming-runtime admission. Native decoders retain Granite's residual scaling.
The native head and causal loss retain logits scaling and window-token reduction.
Routing observation and any Oracle interception belong exclusively to ``probe``.
Granite has no shared expert and does not establish DeepSeek numerical acceptance.

Runtime dependencies are lazy. The inspected native API is Transformers 5.19.0
with PEFT 0.21.2. Only the parent invokes this module through its Kaggle CLI run.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator, Sequence

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
TRANSFORMERS_VERSION = "5.19.0"
PEFT_VERSION = "0.21.2"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
NATIVE_SOURCE_COMMIT = "c4c4605f961877852b89b97478536bb3a26b6fd1"
_NATIVE_MODULE = "transformers.models.granitemoe.modeling_granitemoe"
_TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json",
}
_EXPECTED_CONFIG = {
    "model_type": "granitemoe", "hidden_size": 1024, "intermediate_size": 512,
    "num_hidden_layers": 24, "num_local_experts": 32, "num_experts_per_tok": 8,
    "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 49152,
    "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
    "logits_scaling": 6.0, "attention_multiplier": 0.015625,
    "attention_bias": False, "tie_word_embeddings": True,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"pinned Granite benchmark: {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(target: Any) -> dict[str, Any]:
    source = inspect.getsourcefile(target)
    if source is None or not Path(source).is_file():
        return {"source_file": source, "sha256": None, "available": False}
    path = Path(source)
    return {"source_file": str(path), "sha256": _sha256(path), "available": True}


def _config_guard(config: Any) -> None:
    for key, expected in _EXPECTED_CONFIG.items():
        _require(getattr(config, key, None) == expected, f"config.{key} != {expected!r}")
    _require(config.architectures == ["GraniteMoeForCausalLM"], "unexpected architecture")


def _parts(model: Any) -> tuple[Any, Sequence[Any]]:
    """Admit only the native, in-place PEFT-injected pinned Granite class."""
    import torch

    _require(type(model).__name__ == "GraniteMoeForCausalLM", "not native Granite causal LM")
    _require(type(model).__module__ == _NATIVE_MODULE, "non-native causal LM implementation")
    _config_guard(model.config)
    _require(model.config.attention_dropout == 0.0, "attention dropout override was lost")
    _require(model.config.use_cache is False, "use_cache must remain False")
    _require(model.config.output_router_logits is False, "router auxiliary loss must be off")
    _require(model.config._attn_implementation == "eager", "attention backend is not eager")
    _require(model.config._experts_implementation == "eager", "expert backend is not eager")
    owner = model.model
    _require(type(owner).__name__ == "GraniteMoeModel", "unexpected decoder owner")
    _require(owner.embedding_multiplier == 12.0, "native embedding multiplier changed")
    _require(model.logits_scaling == 6.0, "native logits multiplier changed")
    _require(len(owner.layers) == 24, "expected all 24 native decoders")
    _require(owner.embed_tokens.weight is model.lm_head.weight, "native tied weights lost")
    for module in (owner.embed_tokens, owner.norm, model.lm_head):
        _require(not any(param.requires_grad for param in module.parameters()),
                 "embedding, final norm and head must be frozen")
    for index, layer in enumerate(owner.layers):
        _require(type(layer).__name__ == "GraniteMoeDecoderLayer", "non-native decoder")
        _require(type(layer).__module__ == _NATIVE_MODULE, "non-native decoder implementation")
        _require(layer.residual_multiplier == 0.22, f"layer {index} residual scale changed")
        _require(layer.self_attn.attention_dropout == 0.0, "constructed attention dropout != 0")
        _require(layer.self_attn.scaling == 0.015625, "native attention multiplier changed")
        _require(layer.self_attn.config._attn_implementation == "eager",
                 "decoder attention is not eager")
        _require(layer.block_sparse_moe.experts.config._experts_implementation == "eager",
                 "decoder experts are not eager")
        router = layer.block_sparse_moe.router
        _require(router.num_experts == 32 and router.top_k == 8, "native router layout changed")
        _require(router.weight.dtype == torch.bfloat16 and not router.weight.requires_grad,
                 "router base must be frozen BF16")
    return owner, owner.layers


def _adapter_guard(model: Any) -> dict[str, Any]:
    import torch

    expected: set[str] = set()
    for index, layer in enumerate(model.model.layers):
        for target in TARGETS:
            module = getattr(layer.self_attn, target)
            prefix = f"model.layers.{index}.self_attn.{target}"
            expected.update(f"{prefix}.lora_{side}.default.weight" for side in ("A", "B"))
            _require(module.r == {"default": 2}, f"{prefix} rank is not 2")
            _require(module.lora_alpha == {"default": 4}, f"{prefix} alpha is not 4")
            _require(module.scaling == {"default": 2.0}, f"{prefix} scaling changed")
            dropout = module.lora_dropout["default"]
            _require(
                isinstance(dropout, torch.nn.Identity)
                or isinstance(dropout, torch.nn.Dropout) and dropout.p == 0.0,
                f"{prefix} has nonzero or unknown adapter dropout",
            )
            _require(module.active_adapters == ["default"] and not module.disable_adapters,
                     f"{prefix} default adapter is not active")
            _require(not module.merged_adapters, f"{prefix} adapter has been merged")
    observed = {name for name, param in model.named_parameters() if param.requires_grad}
    _require(observed == expected, "trainable tensors differ from the 192 attention LoRA weights")
    groups: dict[str, Counter[str]] = {"frozen_base": Counter(), "trainable_lora": Counter()}
    tensor_counts: Counter[str] = Counter()
    for name, param in model.named_parameters():
        group = "trainable_lora" if param.requires_grad else "frozen_base"
        required_dtype = torch.float32 if param.requires_grad else torch.bfloat16
        _require(param.dtype == required_dtype, f"{name} dtype is {param.dtype}")
        _require(param.device.type != "meta", f"{name} is still on meta")
        groups[group][str(param.dtype)] += param.numel()
        tensor_counts[group] += 1
    return {
        group: {"parameter_elements_by_dtype": dict(counts), "tensors": tensor_counts[group]}
        for group, counts in groups.items()
    }


def _observe(model: Any, name: str, tensor: Any) -> None:
    metadata = getattr(model, "_p3_realdata_metadata", None)
    if metadata is None:
        return
    observations = metadata["precision_observations"]
    entry = observations.setdefault(name, {"dtypes": [], "calls": 0})
    dtype = str(tensor.dtype)
    if dtype not in entry["dtypes"]:
        entry["dtypes"].append(dtype)
    entry["calls"] += 1
    entry["last_shape"] = list(tensor.shape)


def _disable_checkpointing(model: Any) -> None:
    model.gradient_checkpointing_disable()
    # In 5.19 native enable always installs this hook, but disable only removes it
    # for the Transformers PEFT mixin flag. In-place PEFT injection has no such flag.
    model.disable_input_require_grads()
    _require(not model.is_gradient_checkpointing, "native checkpointing did not disable")


@contextmanager
def _arithmetic(model: Any) -> Iterator[None]:
    import torch

    device_type = model.model.embed_tokens.weight.device.type
    _require(device_type in ("cpu", "cuda"), "only CPU/CUDA BF16 arithmetic is declared")
    with torch.autocast(device_type=device_type, enabled=False):
        yield


def load_model(
    *, cache_dir: Path, device: str = "cuda", seed: int = 23,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the pinned checkpoint without resizing its native tokenizer vocabulary."""
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, inject_adapter_in_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.modeling_layers import GradientCheckpointingLayer

    try:
        from .l2l_schedule import make_non_vacuous
    except ImportError:
        from l2l_schedule import make_non_vacuous

    _require(transformers.__version__ == TRANSFORMERS_VERSION,
             f"requires transformers=={TRANSFORMERS_VERSION}, got {transformers.__version__}")
    _require(version("peft") == PEFT_VERSION, f"requires peft=={PEFT_VERSION}")
    target_device = torch.device(device)
    _require(target_device.type in ("cuda", "cpu"), "only CPU/CUDA is supported")
    _require(target_device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    snapshot = Path(snapshot_download(
        MODEL_ID, revision=REVISION, cache_dir=str(cache_dir),
        allow_patterns=[
            "config.json", "generation_config.json", "model.safetensors.index.json",
            "*.safetensors", *sorted(_TOKENIZER_FILES),
        ],
    ))
    _require(snapshot.name == REVISION, "Hub snapshot did not resolve the pinned revision")
    config = AutoConfig.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
    )
    _config_guard(config)
    checkpoint_dropout = config.attention_dropout
    config.attention_dropout = 0.0
    config.use_cache = False
    config.output_router_logits = False
    config.output_hidden_states = False
    config.output_attentions = False
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
    )
    _require(len(tokenizer) == config.vocab_size, "tokenizer vocabulary differs from checkpoint")
    _require(tokenizer.bos_token_id == 0 and tokenizer.eos_token_id == 0,
             "native BOS/EOS IDs changed")
    model = None
    try:
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            str(snapshot), config=config, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={"": str(target_device)},
            attn_implementation="eager", experts_implementation="eager",
            use_safetensors=True, output_loading_info=True, use_kernels=False,
        )
        normalized_loading = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in loading_info.items()
        }
        _require(not any(normalized_loading.values()),
                 f"unexplained pretrained loading discrepancy: {normalized_loading!r}")
        model.requires_grad_(False)
        torch.manual_seed(seed)
        adapter = LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.0, target_modules=list(TARGETS),
            bias="none", init_lora_weights=True, use_rslora=False, use_dora=False,
        )
        model = inject_adapter_in_model(adapter, model, adapter_name="default")
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype=torch.float32)
        make_non_vacuous(model, seed=seed, scale=0.02)
        model.train()
        _disable_checkpointing(model)
        _parts(model)
        parameter_dtypes = _adapter_guard(model)
        files: dict[str, Any] = {}
        for path in sorted(snapshot.rglob("*")):
            if path.is_file():
                name = path.relative_to(snapshot).as_posix()
                files[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        weight_files = {
            name: item for name, item in files.items() if name.endswith(".safetensors")
        }
        _require(len(weight_files) == 2, "expected the pinned checkpoint's two weight shards")
        libraries = {
            name: version(name) for name in (
                "torch", "transformers", "peft", "huggingface_hub", "safetensors",
                "accelerate", "tokenizers",
            )
        }
        tokenizer_hashes = {
            name: item["sha256"] for name, item in files.items() if name in _TOKENIZER_FILES
        }
        _require("tokenizer.json" in tokenizer_hashes, "native tokenizer.json is missing")
        metadata: dict[str, Any] = {
            "repo_id": MODEL_ID, "revision": REVISION, "snapshot_path": str(snapshot),
            "execution": "benchmark-only resident native Granite layer-major replay",
            "production_model_admission": False, "streaming_runtime": False,
            "granite_proxy_not_deepseek_acceptance": True, "shared_experts": 0,
            "deepseek_difference": (
                "Granite top-logit softmax top8/32 GQA has no shared expert, grouped routing "
                "or correction bias; it is not DeepSeek-V3/R1 arithmetic or NF4 acceptance."
            ),
            "library_versions": libraries, "loading_info": normalized_loading,
            "config": json.loads(config.to_json_string()),
            "protocol": {
                "training": True, "base_dtype": "torch.bfloat16",
                "adapter_dtype": "torch.float32", "attention_dropout_original": checkpoint_dropout,
                "attention_dropout_override_before_construction": 0.0,
                "attn_implementation_observed": config._attn_implementation,
                "experts_implementation_observed": config._experts_implementation,
                "use_cache": False, "output_router_logits": False,
                "embedding_multiplier": model.model.embedding_multiplier,
                "residual_multiplier": model.model.layers[0].residual_multiplier,
                "logits_divisor": model.config.logits_scaling, "autocast": "disabled in both arms",
                "loss": "native CE only; shifted valid targets summed over the full window",
                "ga_checkpointing": "native all-layer nonreentrant, no activation offload",
                "l2l_checkpointing": "disabled; one native decoder graph at a time",
                "optimizer_step": False,
            },
            "lora": {
                "rank": 2, "alpha": 4, "dropout": 0.0, "targets": list(TARGETS),
                "seed": seed, "nonzero_B_scale": 0.02,
                "B_initializer": "l2l_schedule.make_non_vacuous",
                "injection": "PEFT inject_adapter_in_model; native causal LM class retained",
            },
            "parameter_dtype_observations": parameter_dtypes,
            "precision_observations": {}, "files": files, "weight_files": weight_files,
            "tokenizer_spec": {
                "repo_id": MODEL_ID, "revision": REVISION,
                "vocabulary_size": len(tokenizer), "eos_id": tokenizer.eos_token_id,
                "bos_id": tokenizer.bos_token_id,
                "prepend_bos_per_pack": bool(getattr(tokenizer, "add_bos_token", False)),
                "tokenizer_class": type(tokenizer).__name__, "files_sha256": tokenizer_hashes,
                "library_versions": libraries,
            },
            "source": {
                "inspected_transformers_commit": NATIVE_SOURCE_COMMIT,
                "installed_native_model": _source_record(type(model)),
                "installed_checkpoint_layer": _source_record(GradientCheckpointingLayer),
                "installed_loss": _source_record(ForCausalLMLoss),
                "installed_adapter_injection": _source_record(inject_adapter_in_model),
                "benchmark_module": _source_record(load_model),
            },
        }
        model._p3_realdata_metadata = metadata
        return model, tokenizer, metadata
    except BaseException:
        model = None
        raise


class _CaptureCompleteError(Exception):
    """Terminate the native forward before executing decoder zero."""


def _detach_call(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        _require(not value.requires_grad, "captured decoder kwarg unexpectedly has a graph")
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_call(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_call(item) for item in value)
    if isinstance(value, list):
        return [_detach_call(item) for item in value]
    _require(value is None or isinstance(value, (bool, int, float, str)),
             f"unsupported mutable native decoder kwarg: {type(value).__name__}")
    return value


def _batch_guard(model: Any, batches: Sequence[Any], labels: Sequence[Any] | None = None) -> int:
    import torch

    _require(len(batches) > 0, "empty accumulation window")
    _require(labels is None or len(labels) == len(batches), "batch/label window lengths differ")
    device = model.model.embed_tokens.weight.device
    total = 0
    for mb, batch in enumerate(batches):
        _require(isinstance(batch, torch.Tensor) and batch.dtype == torch.long,
                 "input IDs must be native torch.long tensors")
        _require(batch.ndim == 2 and batch.shape[0] > 0 and batch.shape[1] >= 2,
                 "input IDs must be nonempty [B,L] with L >= 2")
        _require(batch.device == device, "input IDs are not on the resident decoder device")
        _require(batch.shape[1] <= model.config.max_position_embeddings, "context limit exceeded")
        if labels is not None:
            label = labels[mb]
            _require(isinstance(label, torch.Tensor) and label.dtype == torch.long,
                     "labels must be native torch.long tensors")
            _require(label.shape == batch.shape and label.device == device,
                     "label shape/device differs from its input")
            total += int((label[:, 1:] != -100).sum().item())
    _require(labels is None or total > 0, "window has no valid shifted-label targets")
    return total


def _call_guard(call: dict[str, Any], batch: Any) -> None:
    import torch

    _require(isinstance(call, dict), "captured call must be a native kwargs dictionary")
    expected = {
        "attention_mask", "position_embeddings", "position_ids", "past_key_values", "use_cache",
    }
    _require(expected.issubset(call), "native decoder call is missing required captured arguments")
    _require("hidden_states" not in call, "captured call retains a decoder input")
    _require(call["past_key_values"] is None and call["use_cache"] is False,
             "captured call contains cache state")
    _require(not call.get("output_router_logits", False), "captured call enables auxiliary loss")
    rope = call["position_embeddings"]
    _require(isinstance(rope, tuple) and len(rope) == 2, "native rotary tuple was not captured")
    for tensor in rope:
        _require(isinstance(tensor, torch.Tensor) and tensor.device == batch.device
                 and tensor.dtype == torch.bfloat16 and not tensor.requires_grad,
                 "native rotary tensor has unexpected dtype/device/graph")
        _require(tensor.shape[-2:] == (batch.shape[1], 64), "captured rotary shape is stale")
    positions = call["position_ids"]
    _require(isinstance(positions, torch.Tensor) and positions.device == batch.device
             and positions.dtype == torch.long and positions.shape == (1, batch.shape[1]),
             "native full-sequence position IDs were not captured")
    mask = call["attention_mask"]
    _require(isinstance(mask, torch.Tensor) and mask.device == batch.device
             and mask.dtype == torch.bfloat16 and not mask.requires_grad,
             "native eager attention mask has unexpected dtype/device/graph")
    _require(mask.ndim == 4 and mask.shape[-2:] == (batch.shape[1], batch.shape[1]),
             "captured native full-sequence attention mask is stale")


def capture_calls(model: Any, batches: Sequence[Any]) -> list[dict[str, Any]]:
    """Capture native mask/RoPE kwargs, stopping before the first decoder executes."""
    import torch

    _, layers = _parts(model)
    _batch_guard(model, batches)
    model.train()
    _disable_checkpointing(model)
    calls: list[dict[str, Any]] = []
    completed = False
    handle = None

    def capture(_layer: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        _require(len(args) == 1, "native decoder positional signature changed")
        _require(args[0].dtype == torch.bfloat16 and not args[0].requires_grad,
                 "native scaled embedding is not graph-free BF16")
        _observe(model, "capture.native_scaled_embedding", args[0])
        calls.append(_detach_call(kwargs))
        raise _CaptureCompleteError

    try:
        handle = layers[0].register_forward_pre_hook(capture, with_kwargs=True)
        with torch.no_grad(), _arithmetic(model):
            for mb, batch in enumerate(batches):
                try:
                    model(
                        input_ids=batch, use_cache=False, output_router_logits=False,
                        logits_to_keep=0, return_dict=True,
                    )
                except _CaptureCompleteError:
                    pass
                else:
                    raise RuntimeError("native forward did not reach the decoder capture hook")
                _require(len(calls) == mb + 1, "decoder capture count differs from input count")
                _call_guard(calls[mb], batch)
        completed = True
        return calls
    finally:
        if handle is not None:
            handle.remove()
        if not completed:
            calls.clear()


def ga_step(
    model: Any, batches: Sequence[Any], labels: Sequence[Any], *, probe: Any,
) -> list[float]:
    """Accumulate native HF checkpointed gradients, leaving all adapter grads in place."""
    import torch

    _, layers = _parts(model)
    total = _batch_guard(model, batches, labels)
    model.train()
    _disable_checkpointing(model)
    losses: list[float] = []
    output = loss = None
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for layer in layers:
            _require(layer.gradient_checkpointing, "a native decoder was not checkpointed")
            checkpoint = layer._gradient_checkpointing_func
            _require(getattr(checkpoint, "keywords", {}).get("use_reentrant") is False,
                     "native checkpoint is not explicitly nonreentrant")
        with torch.enable_grad(), _arithmetic(model):
            for mb, (batch, label) in enumerate(zip(batches, labels)):
                with probe.visit(None, mb, "forward"):
                    output = model(
                        input_ids=batch, labels=label, num_items_in_batch=total,
                        use_cache=False, output_router_logits=False, logits_to_keep=0,
                        return_dict=True,
                    )
                    loss = output.loss
                    _require(loss is not None and loss.ndim == 0, "native loss is not scalar")
                    _observe(model, "ga.native_logits", output.logits)
                    _observe(model, "ga.native_loss", loss)
                # Do not retain the large logits result object while recomputing decoders.
                output = None
                with probe.visit(None, mb, "recompute"):
                    loss.backward()
                losses.append(float(loss.detach().cpu()))
                loss = None
        return losses
    finally:
        output = loss = None
        _disable_checkpointing(model)


def _hidden_guard(hidden: Any, batch: Any, *, leaf: bool = False) -> None:
    import torch

    _require(isinstance(hidden, torch.Tensor), "native decoder did not return a tensor")
    _require(hidden.shape == (*batch.shape, 1024), "native boundary shape changed")
    _require(hidden.dtype == torch.bfloat16 and hidden.device == batch.device,
             "native boundary is not decoder-device BF16")
    if leaf:
        _require(hidden.is_leaf and hidden.requires_grad, "restored boundary is not a VJP leaf")


def l2l_step(
    model: Any, batches: Sequence[Any], labels: Sequence[Any], calls: Sequence[dict[str, Any]],
    store: Any, *, probe: Any,
) -> list[float]:
    """Resident native decoder replay with codec-owned boundaries and no optimizer step.

    ``store.get`` returns a decoded BF16 tensor on the resident decoder device.
    The parent owns ``end_step``/``close`` in its finally block, including failures.
    This function retains only current forward boundaries and upstream gradients.
    """
    import torch

    owner, layers = _parts(model)
    total = _batch_guard(model, batches, labels)
    _require(len(calls) == len(batches), "captured call count differs from the window")
    for call, batch in zip(calls, batches):
        _call_guard(call, batch)
    model.train()
    _disable_checkpointing(model)
    losses: list[float] = []
    hidden: list[Any] = []
    upstream: list[Any] = []
    boundary = last = normed = logits = loss = restored = output = gradient = None
    try:
        with _arithmetic(model):
            with torch.no_grad():
                for batch in batches:
                    boundary = owner.embed_tokens(batch) * owner.embedding_multiplier
                    _hidden_guard(boundary, batch)
                    _observe(model, "l2l.scaled_embedding", boundary)
                    hidden.append(boundary)
                    boundary = None
                for index, layer in enumerate(layers):
                    _require(not layer.gradient_checkpointing, "nested checkpointing in replay")
                    for mb, batch in enumerate(batches):
                        store.put(index, mb, hidden[mb])
                        with probe.visit(index, mb, "forward"):
                            output = layer(hidden[mb], **calls[mb])
                        _hidden_guard(output, batch)
                        _observe(model, "l2l.native_decoder_forward", output)
                        hidden[mb] = output
                        output = None
            for mb, label in enumerate(labels):
                last = hidden[mb].detach().requires_grad_(True)
                hidden[mb] = None
                with torch.enable_grad():
                    normed = owner.norm(last)
                    logits = model.lm_head(normed) / model.config.logits_scaling
                    loss = model.loss_function(
                        logits, label, vocab_size=model.config.vocab_size,
                        num_items_in_batch=total,
                    )
                    _require(loss.ndim == 0, "native replay loss is not scalar")
                    _observe(model, "l2l.native_final_norm", normed)
                    _observe(model, "l2l.native_scaled_logits", logits)
                    _observe(model, "l2l.native_loss", loss)
                    logits = normed = None
                    loss.backward()
                losses.append(float(loss.detach().cpu()))
                _require(last.grad is not None, "head VJP did not reach the last boundary")
                upstream.append(last.grad.detach())
                last.grad = None
                last = loss = None
            hidden.clear()
            store.begin_backward()
            for index in range(len(layers) - 1, -1, -1):
                layer = layers[index]
                _require(not layer.gradient_checkpointing, "nested checkpointing in reverse replay")
                for mb, batch in enumerate(batches):
                    restored = store.get(index, mb).detach().requires_grad_(True)
                    _hidden_guard(restored, batch, leaf=True)
                    with torch.enable_grad(), probe.visit(index, mb, "recompute"):
                        output = layer(restored, **calls[mb])
                        _hidden_guard(output, batch)
                        _observe(model, "l2l.native_decoder_recompute", output)
                        gradient = upstream[mb]
                        _require(gradient is not None and gradient.shape == output.shape
                                 and gradient.dtype == output.dtype
                                 and gradient.device == output.device,
                                 "upstream VJP differs from the native decoder output")
                        output.backward(gradient)
                    _require(
                        restored.grad is not None, "native decoder VJP lost its input gradient",
                    )
                    upstream[mb] = restored.grad.detach() if index > 0 else None
                    restored.grad = None
                    # Release this decoded leaf and graph before retrieving the next boundary.
                    restored = output = gradient = None
        return losses
    finally:
        hidden.clear()
        upstream.clear()
        boundary = last = normed = logits = loss = restored = output = gradient = None
