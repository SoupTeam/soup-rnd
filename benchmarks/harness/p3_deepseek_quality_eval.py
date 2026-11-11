"""Hash-bound DeepSeek quality scoring and whole-endpoint pairing.

Run ``run-endpoint`` on a fresh directory per [start, stop) shard. Production
scoring requires a frozen protocol and a final trained adapter. This study
forbids untrained-base accuracy/NLL, including development endpoint scoring.
``--adapter-sha256`` names the final adapter manifest digest, not a weight digest.
``pair --data-root DIR --protocol PATH --protocol-sha256 HEX --baseline DIR...``
``--candidate DIR... --output NEW_DIR`` requires all six endpoints per kept length.
It writes primary ``pairs.jsonl`` for p3_stage2_statistics.summarize_pairs and
secondary ``nll-pairs.jsonl``. The manifest carries separate expected task maps.
Incomplete endpoints, overlapping shards and changed identities/files are refused.

The resident scorer's admission remains untouched. This wrapper admits one
production StreamRuntime and reuses the frozen encoder, batch planner, native
selected-logit reducer and metric functions, including response/EOS NLL labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Sequence

from rich.console import Console

from . import p3_deepseek_quality_attention as attention_policy
from . import p3_stage2_generation as generation
from . import p3_stage2v2_loglik as frozen

DATA_MANIFEST_SHA256 = "07f3c65cce76d0b209ce20aff7b096857077ac906156ac4ecf2c7955141b2b81"
SCORING_PROTOCOL_SHA256 = "685fa5bee528969d9b2eef423c90280d179432ba017fbebed12b687f608a8249"
ARMS = ("bf16_l2l", "oracle_block64_e4m3")
PRIMARY = ("hellaswag", "arc", "boolq", "lambada")
SECONDARY = ("gsm8k", "mbpp")
DEVELOPMENT = ("hellaswag_dev", "arc_dev", "boolq_dev")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _admit_native_execution(device: str) -> dict:
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
             "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    return attention_policy.require_ampere_cuda(device)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _child(root: Path, name: str) -> Path:
    _require(isinstance(name, str) and bool(name) and not Path(name).is_absolute(),
             "relative artifact filename required")
    parent, child = os.path.realpath(root), os.path.realpath(root / name)
    _require(os.path.commonpath((parent, child)) == parent and child != parent,
             "artifact escapes root")
    return Path(child)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON object required")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _write_rows(path: Path, rows: Sequence[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            text = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write(text + "\n")


def _manifest(root: Path, result: dict) -> None:
    result["files"] = {
        path.name: {"sha256": _hash(path), "bytes": path.stat().st_size}
        for path in sorted(root.iterdir()) if path.is_file() and path.name != "manifest.json"
    }
    result["manifest_payload_sha256"] = generation._object_hash(result)
    _write_json(root / "manifest.json", result)


def _scoring_protocol(path: Path | None = None) -> tuple[dict, dict]:
    if path is None:
        root = Path(__file__).parents[2]
        relative = Path("benchmarks/results/p3-stage2-v2/p3-stage2-v2-protocol.json")
        path = root / relative
    _require(path.is_file() and _hash(path) == SCORING_PROTOCOL_SHA256,
             "frozen v2 scoring protocol SHA256 mismatch")
    value = _json(path)
    _require(value.get("status") == "FROZEN" and value.get("loglik", {}).get(
        "logits_memory_limit_bytes") == 512 * 1024**2, "frozen scoring policy required")
    return value, {"path": str(path), "sha256": SCORING_PROTOCOL_SHA256}


def _source_files() -> dict:
    names = (
        __name__, frozen.__name__, generation.__name__, attention_policy.__name__,
        "benchmarks.harness.p3_stage2_dataset", "benchmarks.harness.p3_deepseek_quality_model",
        "benchmarks.harness.p3_deepseek_quality_data", "soup_cli.utils.layer_stream_runtime",
        "soup_cli.utils.expert_stream", "transformers.models.deepseek_v3.modeling_deepseek_v3",
    )
    sources = {name: _hash(Path(sys.modules[name].__file__)) for name in names
               if name in sys.modules and getattr(sys.modules[name], "__file__", None)}
    policy = attention_policy.policy_identity()
    sources[policy["path"]] = policy["sha256"]
    return sources


def admit_protocol(path: Path, expected: str, reference: dict) -> dict:
    _require(_digest(expected) and path.is_file() and _hash(path) == expected,
             "DeepSeek protocol SHA256 mismatch")
    value = _json(path)
    _require(value.get("schema") == "p3-deepseek-quality-protocol-v1",
             "DeepSeek protocol freeze required")
    _require(value.get("status") == "FROZEN", "protocol freeze required")
    _require(value.get("include_base_model_evaluation") is False,
             "study protocol must forbid untrained base evaluation")
    _require(value.get("attention_policy") == attention_policy.policy_identity(),
             "frozen attention policy missing or changed")
    policy = value.get("loglik", {})
    measured = {"hellaswag_n", "max_tokens_per_batch", "deadline_seconds"}
    _require(isinstance(policy, dict) and set(policy) == set(reference["loglik"])
             and all(policy[key] == original for key, original in reference["loglik"].items()
                     if key not in measured)
             and value.get("scoring") == reference["scoring"], "frozen scoring rules changed")
    for key in measured & policy.keys():
        _require(type(policy[key]) is int and policy[key] > 0, "positive frozen schedule required")
    if "hellaswag_n" in policy:
        _require(policy["hellaswag_n"] <= reference["data"]["full_retained_evaluation_tasks"][
            "hellaswag"], "HellaSwag frozen N exceeds retained manifest count")
    _require(value.get("data", {}).get("manifest_sha256") == DATA_MANIFEST_SHA256,
             "frozen task manifest identity changed")
    training = value.get("training", {})
    lengths = training.get("lengths")
    _require(training.get("seed") == 23 and isinstance(lengths, list) and bool(lengths)
             and all(type(item) is int and item in (1024, 4096) for item in lengths)
             and len(set(lengths)) == len(lengths) and type(training.get("steps")) is int
             and training["steps"] > 0, "frozen training seed/lengths/steps required")
    _require(all(_digest(value.get(name)) for name in ("initial_adapter_sha256", "pack_sha256")),
             "frozen initial adapter and pack identities required")
    _require(_hash(path) == expected, "protocol changed during admission")
    return value


def admit_tasks(
    root: Path, endpoint: str, start: int, stop: int, *,
    manifest_sha256: str = DATA_MANIFEST_SHA256, hellaswag_n: int | None = None,
) -> tuple[list[dict], dict]:
    """Validate the complete endpoint before selecting a nonempty exact interval."""
    _require(endpoint in (*PRIMARY, *SECONDARY, *DEVELOPMENT), "unsupported endpoint")
    manifest = frozen._read_manifest(root, manifest_sha256)
    rows = frozen._task_rows(root, manifest, endpoint, development=endpoint in DEVELOPMENT)
    if endpoint == "hellaswag" and hellaswag_n is not None and hellaswag_n < len(rows):
        from .p3_stage2v2_data import select_hellaswag_tasks

        rows = select_hellaswag_tasks(rows, hellaswag_n)
    _require(hellaswag_n is None or endpoint != "hellaswag" or len(rows) == hellaswag_n,
             "frozen HellaSwag count mismatch")
    _require(type(start) is int and type(stop) is int and 0 <= start < stop <= len(rows),
             "invalid unmodified shard interval")
    selected = rows[start:stop]
    _require(_hash(root / "manifest.json") == manifest_sha256,
             "data manifest changed during admission")
    return selected, {
        "data_manifest_sha256": manifest_sha256, "start": start, "stop": stop,
        "complete_endpoint_row_count": len(rows),
        "endpoint_ordered_task_ids_sha256": generation._object_hash(
            [row["task_id"] for row in rows]),
        "ordered_task_ids": [row["task_id"] for row in selected],
        "selected_rows_sha256": generation._object_hash(selected),
        "tasks_file": copy.deepcopy(manifest["endpoints"][endpoint]["files"]["tasks"]),
    }


def admit_adapter(root: Path, expected: str) -> dict:
    """Bind every saved file before opening tensors or modifying the live model."""
    path = root / "manifest.json"
    _require(_digest(expected) and path.is_file() and _hash(path) == expected,
             "adapter manifest file SHA256 mismatch")
    value = _json(path)
    _require(value.get("schema") == "p3-deepseek-quality-adapter-v1",
             "final adapter manifest required")
    files, identity = value.get("files"), value.get("identity")
    _require(isinstance(files, dict) and {"adapter_config.json", "adapter_model.safetensors"}
             <= set(files) and "manifest.json" not in files,
             "adapter config and weights file hashes required")
    _require(isinstance(identity, dict) and identity.get("arm") in ARMS
             and identity.get("seed") == 23 and type(identity.get("length")) is int
             and identity["length"] in (1024, 4096)
             and type(identity.get("completed_steps")) is int and identity["completed_steps"] > 0
             and all(_digest(identity.get(name)) for name in (
                 "initial_adapter_sha256", "pack_sha256", "protocol_sha256"))
             and isinstance(identity.get("source"), dict) and bool(identity["source"])
             and isinstance(identity.get("cache"), dict) and bool(identity["cache"]),
             "complete final adapter identity required")
    for name, digest in files.items():
        target = _child(root, name)
        _require(_digest(digest) and target.is_file() and _hash(target) == digest,
                 f"adapter file SHA256 mismatch: {name}")
    for target in root.rglob("*"):
        if target.is_file() and target.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth"):
            name = target.relative_to(root).as_posix()
            _require(name in files, f"unlisted adapter weight file: {name}")
            _require(name == "adapter_model.safetensors", "unexpected adapter weight file")
    config = _json(_child(root, "adapter_config.json"))
    targets = {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"}
    _require(config.get("peft_type") == "LORA" and config.get("r") == 2
             and config.get("lora_alpha") == 4 and config.get("lora_dropout") == 0
             and config.get("bias") == "none" and set(config.get("target_modules", [])) == targets
             and not config.get("modules_to_save") and not config.get("use_dora", False)
             and not config.get("use_rslora", False) and not config.get("rank_pattern")
             and not config.get("alpha_pattern"), "unexpected final adapter configuration")
    _require(_hash(path) == expected, "adapter manifest changed during admission")
    return value


def configure_adapter(session: Any, root: Path | None, expected: str | None) -> dict:
    """Base mode zeroes B in memory. Final mode copies an exact, validated PEFT state."""
    import torch

    from .p3_deepseek_quality_model import trainable_parameters

    parameters = trainable_parameters(session.model)
    factors = {name.rsplit(".lora_", 1)[0] for name in parameters}
    _require(bool(factors) and len(parameters) == 2 * len(factors) and all(
        ".self_attn." in name and (".lora_A.default." in name or ".lora_B.default." in name)
        and value.dtype == torch.float32 and not value.is_meta
        and bool(torch.isfinite(value).all().item()) for name, value in parameters.items()),
        "finite FP32 default attention LoRA parameters required")
    if root is None:
        _require(expected is None, "adapter hash without adapter directory")
        factors = [value for name, value in parameters.items() if ".lora_B." in name]
        _require(bool(factors), "base mode requires LoRA B factors")
        with torch.no_grad():
            for value in factors:
                value.zero_()
        return {"mode": "base", "adapter_manifest_sha256": None, "zero_lora_B": True}
    manifest = admit_adapter(root, expected)
    identity = manifest["identity"]
    _require(all(identity[key] == session.metadata[key] for key in ("source", "cache")),
             "adapter source/cache identity mismatch")
    from safetensors.torch import load_file

    tensors = load_file(str(_child(root, "adapter_model.safetensors")), device="cpu")
    names = {name.replace(".default.", "."): name for name in parameters}
    _require(len(names) == len(parameters) and set(tensors) == set(names),
             "exact adapter tensor names required")
    for saved, live in names.items():
        source, target = tensors[saved], parameters[live]
        _require(source.shape == target.shape and source.dtype == target.dtype == torch.float32
                 and not target.is_meta and bool(torch.isfinite(source).all().item()),
                 f"adapter tensor shape/dtype/finite mismatch: {saved}")
    # Recheck bytes after tensor loading, before the first copy. Never accept a partial load.
    admit_adapter(root, expected)
    with torch.no_grad():
        for saved, live in names.items():
            parameters[live].copy_(tensors[saved])
    return {"mode": "final", "adapter_manifest_sha256": expected,
            "files": manifest["files"], **identity}


def _module_nodes(model: Any) -> list[tuple[str, Any]]:
    # Canonical named_modules hides the streaming wrappers. Visit the structural
    # graph so wrapper hooks/forwards/modes cannot escape admission or restoration.
    nodes, pending, seen = [], [("", model)], set()
    while pending:
        name, module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        nodes.append((name, module))
        pending.extend((f"{name}.{key}".lstrip("."), child)
                       for key, child in module._modules.items() if child is not None)
    return nodes


def admit_runtime(session: Any, tokenizer: Any, *, test_fixture: bool = False) -> dict:
    """Allow only production streaming substitutions, never arbitrary runtime hooks."""
    import torch

    from soup_cli.utils import layer_stream_runtime as streaming
    from soup_cli.utils.expert_stream import streamed_experts_class

    model, runtime = session.model, session.runtime
    _require(type(runtime) is streaming.StreamRuntime, "expected StreamRuntime required")
    owner = streaming.decoder_owner(model)
    layers = list(owner.layers)
    _require(len(layers) == runtime.n_layers and bool(layers), "streamed layer coverage mismatch")
    for index, layer in enumerate(layers):
        _require(type(layer) is streaming._streamed_layer_class() and layer.idx == index
                 and layer.pool is runtime.pool and layer.prefetcher is runtime.prefetcher,
                 "foreign streamed decoder wrapper")
    _require(runtime.expert_pool is not None, "per-expert streamed runtime required")
    experts = [module for module in model.modules() if type(module) is streamed_experts_class()]
    _require(bool(experts) and len(experts) == len(runtime.expert_readers)
             and all(module.pool is runtime.expert_pool and any(
                 module.reader is reader for reader in runtime.expert_readers)
                 for module in experts),
             "streamed expert pool/readers mismatch")
    _require(runtime.hook is not None, "stream prefetch hook missing")
    hook = owner._forward_pre_hooks.get(runtime.hook.id)
    _require(callable(hook) and hook.__module__ == streaming.__name__
             and hook.__qualname__ == "install_streaming.<locals>._prime"
             and inspect.getclosurevars(hook).nonlocals.get("prefetcher") is runtime.prefetcher,
             "unexpected streamed prefetch hook")
    allowed_hook = (id(owner), runtime.hook.id)
    substituted = []
    for name, module in _module_nodes(model):
        _require(getattr(module, "_hf_hook", None) is None, f"foreign offload hook: {name}")
        for kind in ("_forward_pre_hooks", "_forward_hooks",
                     "_backward_pre_hooks", "_backward_hooks"):
            for key in getattr(module, kind, {}):
                _require(kind == "_forward_pre_hooks" and (id(module), key) == allowed_hook,
                         f"unexpected runtime hook: {name}")
        if "forward" in vars(module):
            forward = getattr(module.forward, "__func__", None)
            _require(type(module).__module__.startswith("bitsandbytes.") and forward is not None
                     and forward.__module__ == streaming.__name__
                     and forward.__qualname__ == "install_dequant_forward.<locals>._dequant_forward"
                     and getattr(module.forward, "__self__", None) is module,
                     f"unexpected replaced forward, Oracle must be OFF: {name}")
            substituted.append(name)
        _require(not getattr(module, "gradient_checkpointing", False),
                 "active gradient checkpointing refused")
    for key in ("_global_forward_pre_hooks", "_global_forward_hooks",
                "_global_backward_pre_hooks", "_global_backward_hooks"):
        _require(not getattr(torch.nn.modules.module, key, None), "global runtime hook refused")
    device = torch.device(runtime.device)
    _require(device.type in ("cpu", "cuda") and model.config.vocab_size >= len(tokenizer),
             "runtime device/tokenizer vocabulary mismatch")
    if device.type == "cuda":
        _require(device.index in (None, 0) and torch.cuda.current_device() == 0,
                 "CUDA-visible device 0 only")
        device = torch.device("cuda", 0)
    _require(test_fixture or runtime.expert_pool.dtype == "bfloat16",
             "production evaluation requires BF16 compute")
    _require(test_fixture or runtime.tier == "disk", "production disk streaming required")
    for parameter in model.parameters():
        _require(not parameter.dtype.is_floating_point or "float8" not in str(parameter.dtype),
                 "FP8 must be OFF")
        _require(parameter.is_meta or parameter.device == device, "foreign parameter device")
    for slot in runtime.pool.buffers:
        _require(all("float8" not in str(value.dtype) for value in slot.values()),
                 "FP8 streamed buffer refused")
    return {"class": f"{type(runtime).__module__}.{type(runtime).__qualname__}",
            "device": str(device), "tier": runtime.tier, "n_layers": len(layers),
            "expert_layers": len(experts), "nf4_forward_substitutions": substituted,
            "tokenizer_vocabulary_size": len(tokenizer),
            "native_head_vocabulary_size": model.config.vocab_size,
            "fp8": False, "oracle": False, "test_fixture": test_fixture,
            "source_sha256": _hash(Path(streaming.__file__))}


class _ReductionTokenizer:
    """Use the complete native head vocabulary in byte bounds and log-softmax.

    V3 has 129280 logits but 128815 tokenizer entries. Encoding still uses the
    original tokenizer; the padded logits retain their probability mass.
    """

    def __init__(self, tokenizer: Any, vocabulary: int) -> None:
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        return self.vocabulary

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


class _ScoringModel:
    """Supply the runtime device to the unchanged native batch reducer.

    A streamed embedding's weight is a meta placeholder. No tensors are allocated
    here and no model method or frozen admission function is replaced.
    """

    def __init__(self, session: Any, *, test_fixture: bool) -> None:
        import torch

        self.model = session.model
        self.config = session.model.config
        self.device = torch.device(session.runtime.device)
        self.test_fixture = test_fixture
        self.attention_reports: list[dict] = []

    def get_input_embeddings(self) -> Any:
        return SimpleNamespace(weight=SimpleNamespace(device=self.device))

    def __call__(self, **kwargs: Any) -> Any:
        if self.test_fixture:
            return self.model(**kwargs)
        # Reset the bounded observer per forward, not per variable-length shard.
        with attention_policy.attention_execution(
            attention_policy.MAIN_BACKEND, device=str(self.device),
        ) as observation:
            try:
                result = self.model(**kwargs)
                observation.require_complete(dtype="bfloat16", backward=False)
                return result
            finally:
                self.attention_reports.append(observation.report())


@contextmanager
def _scoring_session(
    session: Any, tokenizer: Any, *, test_fixture: bool = False,
) -> Iterator[_ScoringModel]:
    import torch

    identity = admit_runtime(session, tokenizer, test_fixture=test_fixture)
    model = session.model
    flags = generation._precision_flags()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng = frozen._rng_state()
    backend = generation._backend_state(tokenizer)
    nodes = _module_nodes(model)
    modes = [(module, module.training) for _, module in nodes]
    bindings = [(module, module.config) for _, module in nodes if hasattr(module, "config")]
    configurations = {id(config): config for _, config in bindings}
    configurations[id(model.generation_config)] = model.generation_config
    generation_config = model.generation_config
    snapshots = [(config, copy.deepcopy(vars(config))) for config in configurations.values()]
    attributes = {key: copy.deepcopy(getattr(tokenizer, key)) for key in (
        "padding_side", "truncation_side", "model_max_length")}
    parameters = [(name, value, value._version, value.dtype, value.device, value.requires_grad)
                  for name, value in model.named_parameters()]
    try:
        model.eval()
        tokenizer.padding_side = "right"
        if backend[0] is not None:
            backend[0].no_padding()
            backend[0].no_truncation()
        with torch.inference_mode(), torch.autocast("cpu", enabled=False), torch.autocast(
            "cuda", enabled=False,
        ):
            yield _ScoringModel(session, test_fixture=identity["test_fixture"])
    finally:
        for config, snapshot in snapshots:
            if vars(config) != snapshot:
                vars(config).clear()
                vars(config).update(snapshot)
        for module, config in bindings:
            if module.config is not config:
                module.config = config
        if model.generation_config is not generation_config:
            model.generation_config = generation_config
        for key, value in attributes.items():
            setattr(tokenizer, key, value)
        generation._restore_backend(backend)
        for module, mode in modes:
            module.training = mode
        frozen._restore_rng(rng)
        changed = (generation._precision_flags() != flags
                   or torch.is_deterministic_algorithms_warn_only_enabled() != warn_only)
        if changed:
            frozen._restore_precision(flags, warn_only)
        _require(admit_runtime(session, tokenizer, test_fixture=test_fixture) == identity,
                 "runtime identity changed during scoring")
        for name, value, counter, dtype, device, requires_grad in parameters:
            _require(value._version == counter and value.dtype == dtype and value.device == device
                     and value.requires_grad == requires_grad,
                     f"inference changed parameter: {name}")
        _require(not changed and generation._precision_flags() == flags,
                 "scoring changed caller precision permissions")


def _score(
    session: Any, tokenizer: Any, encode: Callable[[], list[dict]], *, max_tokens_per_batch: int,
    test_fixture: bool = False, observer: Callable[[dict], None] | None = None,
    deadline: float | None = None,
) -> list[dict]:
    with _scoring_session(session, tokenizer, test_fixture=test_fixture) as model:
        def observed_batch(event: dict) -> None:
            reports = model.attention_reports
            model.attention_reports = []
            if observer is not None:
                observer({**event, "attention": reports})

        return frozen._score_encoded(
            model, _ReductionTokenizer(tokenizer, model.config.vocab_size), encode(),
            max_tokens_per_batch=max_tokens_per_batch,
            logits_memory_limit_bytes=frozen.LOGITS_MEMORY_LIMIT_BYTES,
            reduction_chunk_tokens=frozen.REDUCTION_CHUNK_TOKENS,
            batch_observer=observed_batch, deadline=deadline,
        )


def score_requests(
    session: Any, tokenizer: Any, requests: list[tuple[str, str]], *, max_tokens_per_batch: int,
    test_fixture: bool = False,
) -> list[dict]:
    return _score(
        session, tokenizer, lambda: frozen._request_tokens(tokenizer, requests),
        max_tokens_per_batch=max_tokens_per_batch, test_fixture=test_fixture,
    )


def teacher_forced_nll(
    session: Any, tokenizer: Any, prompt: str, response: str, *, test_fixture: bool = False,
) -> dict:
    encoded = frozen._document_tokens(tokenizer, [(prompt, response)])
    results = _score(
        session, tokenizer, lambda: encoded,
        max_tokens_per_batch=max(frozen.DEFAULT_MAX_TOKENS_PER_BATCH, len(encoded[0]["input_ids"])),
        test_fixture=test_fixture,
    )
    return frozen._nll(results[0])


def _load_tokenizer(source: Path) -> tuple[Any, dict]:
    from .p3_deepseek_quality_data import load_tokenizer

    return load_tokenizer(source)


def run_endpoint(
    *, data_root: Path, endpoint: str, start: int, stop: int, source: Path, cache: Path,
    output: Path, adapter: Path | None = None, adapter_sha256: str | None = None,
    protocol: Path | None = None, protocol_sha256: str | None = None,
    scoring_protocol: Path | None = None, length: int = 1024, device: str = "cuda",
) -> dict:
    """Score once through BF16 streamed inference, with a failure manifest on refusal."""
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema": "p3-deepseek-quality-eval-v1", "status": "STARTED",
        "selected_shard_complete": False, "endpoint": endpoint, "length": length, "seed": 23,
        "protocol_sha256": protocol_sha256, "fp8": False, "oracle": False,
        "include_base_model_evaluation": False,
    }
    session = None
    started = time.monotonic()
    stage = "freeze"
    try:
        _require((protocol is None) == (protocol_sha256 is None),
                 "protocol path/hash pair required")
        _require((adapter is None) == (adapter_sha256 is None), "adapter path/hash pair required")
        _require(adapter is not None,
                 "untrained base quality scoring is forbidden in this study")
        _require(endpoint in (*PRIMARY, *SECONDARY),
                 "production scorer admits trained-arm evaluation endpoints only")
        _require(protocol is not None, "DeepSeek protocol freeze required before trained outcomes")
        reference, reference_identity = _scoring_protocol(scoring_protocol)
        study = admit_protocol(protocol, protocol_sha256, reference)
        result["attention_policy"] = study["attention_policy"]
        _require(length in study["training"]["lengths"], "length not kept by frozen protocol")
        policy = study["loglik"]
        stage = "dataset"
        rows, binding = admit_tasks(
            data_root, endpoint, start, stop,
            hellaswag_n=policy["hellaswag_n"] if endpoint == "hellaswag" else None,
        )
        result.update(binding=binding, scoring_protocol=reference_identity,
                      loglik=policy, scoring=reference["scoring"])
        stage = "adapter admission"
        adapter_manifest = admit_adapter(adapter, adapter_sha256) if adapter else None
        if adapter_manifest is not None:
            identity = adapter_manifest["identity"]
            _require(identity["protocol_sha256"] == protocol_sha256 and identity["length"] == length
                     and identity["completed_steps"] == study["training"]["steps"]
                     and all(identity[name] == study[name]
                             for name in ("initial_adapter_sha256", "pack_sha256")),
                     "final adapter protocol/length/steps/initial/pack identity mismatch")
            _require(identity.get("attention_policy") == study["attention_policy"],
                     "final adapter attention policy differs from frozen study")
        stage = "model"
        from .p3_deepseek_quality_model import open_session, require_production_source

        result["hardware"] = _admit_native_execution(device)
        session = open_session(source, cache, device=device, dtype="bfloat16", prepare=False)
        require_production_source(session.metadata["source"])
        result["model"] = copy.deepcopy(session.metadata)
        tokenizer, token_identity = _load_tokenizer(source)
        result["tokenizer"] = token_identity
        result["source_files_sha256"] = _source_files()
        result["runtime"] = admit_runtime(session, tokenizer)
        result["adapter_identity"] = configure_adapter(session, adapter, adapter_sha256)
        result["arm"] = result["adapter_identity"].get("arm", "base")
        stage = "scoring"
        events = []
        common = {"max_tokens_per_batch": policy["max_tokens_per_batch"],
                  "observer": lambda event: events.append({**event, "endpoint": endpoint}),
                  "deadline": started + policy["deadline_seconds"]}
        try:
            if endpoint in SECONDARY:
                scores = _score(
                    session, tokenizer,
                    lambda: frozen._document_tokens(
                        tokenizer, [(row["context"], row["target"]) for row in rows]), **common,
                )
                scored = [{"results": [score], "pass": None, **frozen._nll(score)}
                          for score in scores]
            else:
                requests, sizes = frozen._requests(rows)
                scores = _score(
                    session, tokenizer, lambda: frozen._request_tokens(tokenizer, requests),
                    **common,
                )
                scored, offset = [], 0
                for row, size in zip(rows, sizes, strict=True):
                    choices = scores[offset:offset + size]
                    metric_row = {**row, "endpoint": endpoint.removesuffix("_dev")}
                    scored.append({"results": choices, **frozen.score_task(metric_row, choices)})
                    offset += size
        finally:
            _write_rows(output / "batches.jsonl", events)
        scored_rows = [
            {"endpoint": endpoint, "task_id": row["task_id"], "task_index": start + index,
             "length": length, "seed": 23, "adapter_identity": result["adapter_identity"],
             "source_row_sha256": row["source_row_sha256"],
             "task_sha256": generation._object_hash(row),
             "task_scoring": {key: row[key] for key in (
                 "endpoint", "metric", "choices", "gold", "target") if key in row}, **score}
            for index, (row, score) in enumerate(zip(rows, scored, strict=True))
        ]
        stage = "input recheck"
        _, rebound = admit_tasks(
            data_root, endpoint, start, stop,
            hellaswag_n=policy["hellaswag_n"] if endpoint == "hellaswag" else None,
        )
        _require(binding == rebound, "task binding changed during scoring")
        if adapter:
            admit_adapter(adapter, adapter_sha256)
        if protocol:
            admit_protocol(protocol, protocol_sha256, reference)
        _require(all(descriptor["sha256"] == _hash(source / name)
                     for name, descriptor in token_identity["files"].items()),
                 "tokenizer changed during scoring")
        _require(_source_files() == result["source_files_sha256"],
                 "scoring source changed during execution")
        result["runtime_stats"] = session.runtime.stats()
        _write_rows(output / "rows.jsonl", scored_rows)
        result.update(status="COMPLETE", selected_shard_complete=True, row_count=len(scored_rows))
    except BaseException as error:
        result.update(status="FAILED", selected_shard_complete=False,
                      failure={"stage": stage, "type": type(error).__name__, "message": str(error)})
        if not isinstance(error, Exception):
            raise
    finally:
        if session is not None:
            try:
                session.close()
            except BaseException as error:
                result.update(status="FAILED", selected_shard_complete=False,
                              cleanup_failure={"type": type(error).__name__, "message": str(error)})
        result["wall_seconds"] = time.monotonic() - started
        _manifest(output, result)
    return result


def _read_shard(root: Path) -> tuple[dict, list[dict]]:
    value = _json(root / "manifest.json")
    payload = {key: entry for key, entry in value.items() if key != "manifest_payload_sha256"}
    _require(value.get("manifest_payload_sha256") == generation._object_hash(payload),
             "shard manifest payload SHA256 mismatch")
    _require(value.get("schema") == "p3-deepseek-quality-eval-v1"
             and value.get("status") == "COMPLETE" and value.get("selected_shard_complete") is True,
             "incomplete shard refused")
    for name, descriptor in value.get("files", {}).items():
        path = _child(root, name)
        _require(path.is_file() and _hash(path) == descriptor.get("sha256")
                 and path.stat().st_size == descriptor.get("bytes"), "shard file SHA256 mismatch")
    _require("rows.jsonl" in value.get("files", {}), "shard score file missing")
    lines = (root / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    _require(all(isinstance(row, dict) for row in rows), "score rows must be objects")
    _require(_json(root / "manifest.json") == value, "shard changed during admission")
    return value, rows


def _collect(paths: Sequence[Path], arm: str) -> dict[tuple, tuple[dict, list[dict]]]:
    _require(bool(paths), "empty shard coverage")
    cells: dict[tuple, list[tuple[dict, list[dict]]]] = {}
    for path in paths:
        manifest, rows = _read_shard(path)
        _require(manifest.get("arm") == arm, "arm identity mismatch")
        key = (manifest.get("endpoint"), manifest.get("length"), manifest.get("seed"))
        _require(key[0] in (*PRIMARY, *SECONDARY) and type(key[1]) is int
                 and key[1] in (1024, 4096) and key[2] == 23, "cell identity mismatch")
        identity = manifest.get("adapter_identity", {})
        _require(identity.get("mode") == "final" and identity.get("arm") == arm
                 and identity.get("length") == key[1] and identity.get("seed") == key[2]
                 and identity.get("protocol_sha256") == manifest.get("protocol_sha256")
                 and all(_digest(identity.get(field)) for field in (
                     "adapter_manifest_sha256", "initial_adapter_sha256", "pack_sha256",
                     "protocol_sha256")), "final adapter identity missing from shard")
        policy = attention_policy.policy_identity()
        _require(identity.get("attention_policy") == policy
                 and manifest.get("attention_policy") == policy,
                 "shard attention policy missing or changed")
        sources = manifest.get("source_files_sha256", {})
        _require(isinstance(sources, dict)
                 and sources.get(policy["path"]) == policy["sha256"]
                 and sources.get(attention_policy.__name__)
                 == _hash(Path(attention_policy.__file__)),
                 "shard attention executor/protocol source identity missing or changed")
        _require("batches.jsonl" in manifest.get("files", {}),
                 "shard attention batch evidence missing")
        batches = (path / "batches.jsonl").read_text(encoding="utf-8").splitlines()
        _require(bool(batches), "shard attention batch evidence empty")
        for line in batches:
            batch = json.loads(line)
            reports = batch.get("attention") if isinstance(batch, dict) else None
            _require(isinstance(reports, list) and bool(reports),
                     "batch attention execution evidence missing")
            _require(batch.get("event") == "batch_complete"
                     and type(batch.get("forward_calls")) is int
                     and batch["forward_calls"] == len(reports),
                     "batch attention forward coverage incomplete")
            for report in reports:
                attention_policy.require_execution_evidence(
                    report, backend=attention_policy.MAIN_BACKEND,
                    dtype="bfloat16", backward=False,
                )
        _require(isinstance(identity.get("source"), dict) and bool(identity["source"])
                 and isinstance(identity.get("cache"), dict) and bool(identity["cache"])
                 and identity["source"] == manifest.get("model", {}).get("source")
                 and identity["cache"] == manifest.get("model", {}).get("cache")
                 and type(identity.get("completed_steps")) is int
                 and identity["completed_steps"] > 0, "adapter/model identity mismatch")
        cells.setdefault(key, []).append((manifest, rows))
    complete = {}
    for key, shards in cells.items():
        shards.sort(key=lambda item: item[0]["binding"]["start"])
        original = shards[0][0]
        cursor, output = 0, []
        for manifest, rows in shards:
            binding = manifest["binding"]
            _require(all(manifest.get(field) == original.get(field) for field in (
                "adapter_identity", "model", "protocol_sha256", "loglik", "scoring", "tokenizer",
                "source_files_sha256")),
                "shard identity mismatch")
            _require(all(binding.get(field) == original["binding"].get(field) for field in (
                "complete_endpoint_row_count", "endpoint_ordered_task_ids_sha256",
                "data_manifest_sha256", "tasks_file")), "endpoint identity mismatch")
            stop = binding.get("stop")
            _require(type(binding.get("start")) is int and type(stop) is int
                     and binding["start"] == cursor and stop > cursor
                     and len(rows) == stop - cursor
                     and [row.get("task_id") for row in rows] == binding["ordered_task_ids"],
                     "incomplete or overlapping shard coverage")
            for index, row in enumerate(rows, start=cursor):
                _require(row.get("endpoint") == key[0] and row.get("length") == key[1]
                         and row.get("seed") == key[2] and row.get("task_index") == index
                         and row.get("adapter_identity") == manifest["adapter_identity"],
                         "score row identity/coverage mismatch")
                _require(row.get("task_scoring", {}).get("endpoint") == key[0],
                         "task metric endpoint identity mismatch")
                if key[0] in PRIMARY:
                    _require(type(row.get("pass")) is bool, "boolean primary result required")
                    expected = frozen.score_task(row["task_scoring"], row["results"])
                    _require(all(row.get(field) == expected[field] for field in expected),
                             "primary score/likelihood mismatch")
                else:
                    total = row.get("nll_sum")
                    targets, mean = row.get("n_targets"), row.get("nll_mean")
                    _require(type(targets) is int and targets > 0
                             and type(total) in (int, float) and math.isfinite(total) and total >= 0
                             and type(mean) in (int, float) and math.isfinite(mean)
                             and math.isclose(mean, total / targets, rel_tol=1e-12, abs_tol=1e-12),
                             "invalid secondary NLL result")
                    likelihoods = row.get("results", [])
                    _require(isinstance(likelihoods, list) and len(likelihoods) == 1
                             and isinstance(likelihoods[0], dict), "secondary likelihood required")
                    result = likelihoods[0]
                    _require(type(result.get("logprob")) in (int, float)
                             and math.isfinite(result["logprob"])
                             and type(result.get("greedy")) is bool,
                             "invalid secondary likelihood")
                    _require(result.get("n_tokens") == targets and row.get("pass") is None
                             and frozen._nll(row["results"][0]) == {
                                 "nll_sum": total, "n_targets": targets, "nll_mean": mean},
                             "secondary score/likelihood mismatch")
            output.extend(rows)
            cursor = stop
        ids = [row["task_id"] for row in output]
        _require(cursor == original["binding"]["complete_endpoint_row_count"]
                 and len(ids) == len(set(ids)) and generation._object_hash(ids)
                 == original["binding"]["endpoint_ordered_task_ids_sha256"],
                 "incomplete or reordered endpoint coverage")
        complete[key] = original, output
    return complete


def pair_shards(
    baseline: Sequence[Path], candidate: Sequence[Path],
) -> tuple[list[dict], list[dict]]:
    """Require full identical endpoint task coverage and one adapter per arm/cell."""
    left, right = _collect(baseline, ARMS[0]), _collect(candidate, ARMS[1])
    _require(set(left) == set(right), "paired cell coverage mismatch")
    primary, secondary = [], []
    for key in sorted(left):
        baseline_manifest, baseline_rows = left[key]
        candidate_manifest, candidate_rows = right[key]
        _require(all(baseline_manifest.get(field) == candidate_manifest.get(field) for field in (
            "model", "protocol_sha256", "loglik", "scoring", "tokenizer", "source_files_sha256")),
            "paired model/protocol/scoring identity mismatch")
        _require(all(baseline_manifest["adapter_identity"].get(field)
                     == candidate_manifest["adapter_identity"].get(field) for field in (
                         "source", "cache", "initial_adapter_sha256", "pack_sha256",
                         "protocol_sha256", "completed_steps", "attention_policy")),
                 "paired training/adapter identity mismatch")
        _require(all(baseline_manifest["binding"].get(field)
                     == candidate_manifest["binding"].get(field) for field in (
                         "data_manifest_sha256", "tasks_file", "endpoint_ordered_task_ids_sha256"))
                 and [row["task_id"] for row in baseline_rows]
                 == [row["task_id"] for row in candidate_rows], "paired task coverage mismatch")
        for baseline_row, candidate_row in zip(baseline_rows, candidate_rows, strict=True):
            _require(baseline_row.get("source_row_sha256")
                     == candidate_row.get("source_row_sha256"), "paired task source mismatch")
            row = {"benchmark": key[0], "length": key[1], "seed": key[2],
                   "task_id": baseline_row["task_id"]}
            if key[0] in PRIMARY:
                primary.append({**row, "baseline_pass": baseline_row["pass"],
                                "candidate_pass": candidate_row["pass"]})
            else:
                _require(baseline_row["n_targets"] == candidate_row["n_targets"],
                         "paired NLL target coverage mismatch")
                secondary.append({
                    **row, "n_targets": baseline_row["n_targets"],
                    "baseline_nll_sum": baseline_row["nll_sum"],
                    "candidate_nll_sum": candidate_row["nll_sum"],
                    "baseline_nll_mean": baseline_row["nll_mean"],
                    "candidate_nll_mean": candidate_row["nll_mean"],
                    "nll_mean_difference": candidate_row["nll_mean"] - baseline_row["nll_mean"],
                })
    return primary, secondary


def pair(
    *, baseline: Sequence[Path], candidate: Sequence[Path], output: Path,
    data_root: Path, protocol: Path, protocol_sha256: str,
    scoring_protocol: Path | None = None,
) -> dict:
    """Assemble all six frozen endpoints for every kept length, never a subset verdict."""
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {"schema": "p3-deepseek-quality-pairs-v1", "status": "STARTED"}
    try:
        reference, reference_identity = _scoring_protocol(scoring_protocol)
        study = admit_protocol(protocol, protocol_sha256, reference)
        policy = study["loglik"]
        inputs = [{"path": str(root / "manifest.json"), "sha256": _hash(root / "manifest.json")}
                  for root in (*baseline, *candidate)]
        required = {(endpoint, length, 23) for endpoint in (*PRIMARY, *SECONDARY)
                    for length in study["training"]["lengths"]}
        manifest = frozen._read_manifest(data_root, DATA_MANIFEST_SHA256)
        tasks = {}
        for endpoint in (*PRIMARY, *SECONDARY):
            count = (policy["hellaswag_n"] if endpoint == "hellaswag"
                     else manifest["endpoints"][endpoint]["retained_row_count"])
            rows, binding = admit_tasks(
                data_root, endpoint, 0, count,
                hellaswag_n=policy["hellaswag_n"] if endpoint == "hellaswag" else None,
            )
            tasks[endpoint] = rows, binding
        for paths, arm in ((baseline, ARMS[0]), (candidate, ARMS[1])):
            cells = _collect(paths, arm)
            _require(set(cells) == required, "complete frozen six-endpoint cell coverage required")
            adapters_by_length = {}
            for key, (shard, scored_rows) in cells.items():
                rows, binding = tasks[key[0]]
                _require(shard["protocol_sha256"] == protocol_sha256
                         and shard["loglik"] == study["loglik"]
                         and shard["scoring"] == study["scoring"], "frozen shard policy mismatch")
                _require(all(shard["binding"].get(field) == binding[field] for field in (
                    "data_manifest_sha256", "tasks_file", "complete_endpoint_row_count",
                    "endpoint_ordered_task_ids_sha256")), "frozen endpoint identity mismatch")
                identity = shard["adapter_identity"]
                _require(identity["completed_steps"] == study["training"]["steps"]
                         and all(identity[name] == study[name] for name in (
                             "initial_adapter_sha256", "pack_sha256")),
                         "frozen training adapter identity mismatch")
                previous = adapters_by_length.setdefault(key[1], identity)
                _require(previous == identity, "adapter changed between endpoints")
                for task, scored in zip(rows, scored_rows, strict=True):
                    task_scoring = {name: task[name] for name in (
                        "endpoint", "metric", "choices", "gold", "target") if name in task}
                    _require(scored["task_id"] == task["task_id"]
                             and scored.get("task_sha256") == generation._object_hash(task)
                             and scored.get("task_scoring") == task_scoring,
                             "frozen task coverage/content mismatch")
        primary, secondary = pair_shards(baseline, candidate)
        _require(all(_hash(Path(item["path"])) == item["sha256"] for item in inputs),
                 "input shard changed during pairing")
        admit_protocol(protocol, protocol_sha256, reference)
        for endpoint, (rows, binding) in tasks.items():
            _, rechecked = admit_tasks(
                data_root, endpoint, 0, len(rows),
                hellaswag_n=policy["hellaswag_n"] if endpoint == "hellaswag" else None,
            )
            _require(rechecked == binding, "frozen data changed during pairing")
        _write_rows(output / "pairs.jsonl", primary)
        _write_rows(output / "nll-pairs.jsonl", secondary)
        result.update(
            status="COMPLETE", primary_rows=len(primary), secondary_rows=len(secondary),
            protocol_sha256=protocol_sha256, data_manifest_sha256=DATA_MANIFEST_SHA256,
            include_base_model_evaluation=False,
            expected_tasks={name: [row["task_id"] for row in tasks[name][0]] for name in PRIMARY},
            expected_secondary_tasks={name: [row["task_id"] for row in tasks[name][0]]
                                      for name in SECONDARY},
            lengths=study["training"]["lengths"], seeds=[23], input_manifests=inputs,
            scoring_protocol=reference_identity, loglik=policy, scoring=study["scoring"],
        )
    except Exception as error:
        result.update(status="FAILED",
                      failure={"type": type(error).__name__, "message": str(error)})
    finally:
        _manifest(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    endpoint = commands.add_parser("run-endpoint")
    for name in ("data-root", "source", "cache", "output"):
        endpoint.add_argument("--" + name, type=Path, required=True)
    endpoint.add_argument("--endpoint", choices=(*PRIMARY, *SECONDARY), required=True)
    endpoint.add_argument("--start", type=int, required=True)
    endpoint.add_argument("--stop", type=int, required=True)
    endpoint.add_argument("--adapter", type=Path)
    endpoint.add_argument("--adapter-sha256")
    endpoint.add_argument("--protocol", type=Path)
    endpoint.add_argument("--protocol-sha256")
    endpoint.add_argument("--scoring-protocol", type=Path)
    endpoint.add_argument("--length", type=int, choices=(1024, 4096), default=1024)
    endpoint.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    pairing = commands.add_parser("pair")
    pairing.add_argument("--baseline", nargs="+", type=Path, required=True)
    pairing.add_argument("--candidate", nargs="+", type=Path, required=True)
    pairing.add_argument("--output", type=Path, required=True)
    pairing.add_argument("--data-root", type=Path, required=True)
    pairing.add_argument("--protocol", type=Path, required=True)
    pairing.add_argument("--protocol-sha256", required=True)
    pairing.add_argument("--scoring-protocol", type=Path)
    values = vars(parser.parse_args(argv))
    command = values.pop("command")
    try:
        result = run_endpoint(**values) if command == "run-endpoint" else pair(**values)
    except Exception as error:
        Console().print({"status": "FAILED", "type": type(error).__name__, "message": str(error)})
        return 1
    Console().print(result)
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
