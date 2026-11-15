"""Evaluator-only CUDA residency for the unchanged Moonlight NF4 cache.

Consumes a fresh admitted disk ModelSession after its adapter has been loaded.
Training, quantization, scoring and expert arithmetic remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

PROFILE = "moonlight-16b-a3b"


def _close_receipt(value: dict) -> dict:
    if not value["closed"]:
        error = RuntimeError("candidate cleanup failed")
        error.residency_cleanup = copy.deepcopy(value)
        raise error
    return copy.deepcopy(value)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _category(name: str) -> str:
    if ".lora_" in name:
        return "adapter"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "head"
    if ".self_attn." in name:
        return "attention"
    if ".shared_experts." in name:
        return "shared"
    if ".experts." in name:
        return "routed"
    if ".mlp." in name and ".gate." not in name:
        return "dense"
    return "other"


class _CudaExperts:
    """One layer's original packed expert payloads, copied once before timing."""

    def __init__(self, reader: Any, device: Any, staging: Any, layer_idx: int) -> None:
        from soup_cli.utils.expert_shard import EXPERT_PARTS, expert_key
        from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

        self.count = reader.count
        self.shapes = dict(reader.shapes)
        self.quant_specs = dict(reader.quant_specs)
        self.dtype = reader.dtype
        self.layer_idx = layer_idx
        self.bytes_read = 0  # Measured calls never read host/disk weight bytes.
        self.fetches = 0
        self.tensors: list[dict[str, Any]] = []
        self.preparation_bytes_read = 0
        before = reader.bytes_read
        try:
            _require(set(self.quant_specs) == set(EXPERT_PARTS), "all routed matrices need NF4")
            for expert in range(self.count):
                host = reader.read_host(expert, staging)
                tensors = {}
                self.tensors.append(tensors)
                for part in EXPERT_PARTS:
                    key = expert_key(expert, part)
                    spec = self.quant_specs[part]
                    _require(spec.quant_type == "nf4" and spec.blocksize == 64
                             and not spec.nested and spec.dtype == "bfloat16",
                             "routed payload must retain block64 nonnested BF16-compute NF4")
                    for name in quant_sidecar_keys(key, spec):
                        tensors[name] = host[name].to(device=device, non_blocking=False)
        except BaseException:
            self.close()
            raise
        finally:
            self.preparation_bytes_read = reader.bytes_read - before
            reader.close()

    def close(self) -> None:
        for tensors in self.tensors:
            tensors.clear()
        self.tensors.clear()


class _CudaExpertLease:
    """Use the existing dequantizer and one existing arena, with CUDA-only inputs.

    StreamedExperts.forward, its expert order, F.linear calls, routing multiply
    and index_add_ stay unchanged. Only the lease's weight source changes.
    """

    def __init__(self, pool: Any, readers: list[_CudaExperts]) -> None:
        self.pool, self.readers = pool, readers
        self.device, self.dtype = pool.device, pool.dtype
        self._shapes = pool._shapes
        self._buffers = dict(pool._buffers)
        self.fetches = 0

    def validate_reader(self, reader: Any) -> None:
        _require(not self.pool._closed and any(reader is item for item in self.readers),
                 "resident expert lease has a closed or foreign owner")

    @contextmanager
    def lease(self, reader: _CudaExperts, expert: int) -> Iterator[tuple[Any, Any]]:
        import torch

        from soup_cli.utils.expert_shard import DOWN, EXPERT_PARTS, GATE_UP, expert_key
        from soup_cli.utils.expert_stream import ExpertBufferPool
        from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

        _require(torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),
                 "resident NF4 supports inference only")
        if not self.pool._lock.acquire(blocking=False):
            raise RuntimeError("resident expert arena already has an active lease")
        try:
            self.validate_reader(reader)
            _require(type(expert) is int and 0 <= expert < reader.count, "expert out of range")
            payload = reader.tensors[expert]
            for part in EXPERT_PARTS:
                key = expert_key(expert, part)
                spec = reader.quant_specs[part]
                for index, name in enumerate(quant_sidecar_keys(key, spec)):
                    # Rebind views, not bytes. There is no host refill or packed CUDA copy.
                    self._buffers[f"{part}:{index}"] = payload[name]
                ExpertBufferPool._dequantize(self, part, spec)
            self.fetches += 1
            reader.fetches += 1
            yield self._buffers[GATE_UP], self._buffers[DOWN]
        finally:
            try:
                self.pool._synchronize()
            finally:
                self.pool._lock.release()

    def close(self) -> None:
        self._buffers.clear()
        self.readers.clear()


def _quiescent(runtime: Any) -> bool:
    source = runtime.source
    return (
        source._closed and not source._thread.is_alive()
        and all(group._closed and not any(thread.is_alive() for thread in group._threads)
                for group in source._pools.values())
        and all(reader._handle is None for reader in runtime.expert_readers)
        and runtime.hook is None
    )


class ResidentNF4:
    """One consumed source session. close() invalidates both model references."""

    mode = "A"

    def __init__(self, session: Any) -> None:
        self.model = session.model
        self._session = session
        self._readers: list[_CudaExperts] = []
        self._lease: _CudaExpertLease | None = None
        self._scoring = False
        self._receipt: dict | None = None
        self._adapter_sha256 = ""
        self._implementation_path = Path(__file__)
        self._implementation_sha256 = hashlib.sha256(
            self._implementation_path.read_bytes()).hexdigest()
        session.retained_resources.append(self)

    def _install(self) -> None:
        import torch

        from benchmarks.harness import p3_deepseek_quality_model as model_api
        from soup_cli.utils import layer_stream_runtime as stream
        from soup_cli.utils.async_disk_source import AsyncDiskSource
        from soup_cli.utils.expert_stream import streamed_experts_class

        session, model = self._session, self.model
        runtime = session.runtime
        _require(type(runtime) is stream.StreamRuntime and type(runtime.source) is AsyncDiskSource,
                 "a production asynchronous disk source session is required")
        _require(not session.closed and runtime.tier == "disk" and not runtime.external_sources,
                 "a fresh disk-only source session is required")
        _require(session.metadata.get("profile_id") == PROFILE
                 and session.metadata["precision"]["compute_dtype"] == "bfloat16"
                 and session.metadata["cache"]["quant"] == "nf4"
                 and session.metadata["cache"]["expert_layout"] == "per_expert",
                 "admitted Moonlight BF16-compute per_expert NF4 session required")
        model_api.require_production_source(session.metadata["source"], profile_id=PROFILE)
        _require(torch.device(runtime.device).type == "cuda" and torch.cuda.current_device() == 0,
                 "CUDA-visible device zero is required")
        _require(runtime.expert_pool is not None and runtime.expert_pool.dtype == "bfloat16"
                 and runtime.pool.loads == 0 and runtime.expert_pool.fetches == 0
                 and getattr(runtime.large_pool, "loads", 0) == 0,
                 "candidate must consume a fresh source session before any forward")
        self._adapter_sha256 = model_api.adapter_digest(model_api.trainable_parameters(model))[0]
        owner = stream.decoder_owner(model)
        _require(runtime.hook is not None, "source prefetch hook missing")
        runtime.hook.remove()
        runtime.hook = None
        torch.cuda.synchronize()
        # StreamRuntime.close does not free decoder/large slots. This owner does.
        runtime.pool.buffers.clear()
        runtime.pool.events.clear()
        runtime.pool.nbytes = 0
        if runtime.large_pool is not None:
            runtime.large_pool.buffer = None
            runtime.large_pool.event = None
            runtime.large_pool.nbytes = 0
        device = runtime.expert_pool.device
        experts = [module for module in model.modules() if type(module) is streamed_experts_class()]
        _require(len(experts) == len(runtime.expert_readers) > 0,
                 "routed reader coverage differs")
        for module in experts:
            _require(module.pool is runtime.expert_pool
                     and any(module.reader is reader for reader in runtime.expert_readers),
                     "foreign routed reader or pool")
            resident = _CudaExperts(module.reader, device, runtime.expert_pool._staging,
                                    module.layer_idx)
            self._readers.append(resident)
        self._lease = _CudaExpertLease(runtime.expert_pool, self._readers)
        for module, resident in zip(experts, self._readers, strict=True):
            module.reader, module.pool = resident, self._lease
        # No CPU staging remains reachable through the measured expert path.
        runtime.expert_pool._staging_owner = runtime.expert_pool._staging = None

        for index, wrapper in enumerate(list(owner.layers)):
            _require(type(wrapper) is stream._streamed_layer_class() and wrapper.idx == index,
                     "foreign decoder wrapper")
            keys = set()
            for name in wrapper.name_map.values():
                spec = wrapper.quant_specs.get(name)
                if spec is not None:
                    _require(spec.quant_type == "nf4" and spec.blocksize == 64
                             and not spec.nested and spec.dtype == "bfloat16",
                             "decoder NF4 spec differs from admitted cache")
                keys.update(stream.quant_sidecar_keys(name, spec) if spec is not None else (name,))
            tensors = {}
            try:
                for key in sorted(keys):
                    tensors[key] = runtime.source.get(index, key).to(
                        device=device, non_blocking=False)
            finally:
                runtime.source.release(index, None)
            for name, key in wrapper.name_map.items():
                prefix, _, leaf = name.rpartition(".")
                target = wrapper.inner.get_submodule(prefix) if prefix else wrapper.inner
                spec = wrapper.quant_specs.get(key)
                if spec is None:
                    target._parameters[leaf] = torch.nn.Parameter(tensors[key], requires_grad=False)
                else:
                    target._parameters[leaf] = stream.rebuild_params4bit(
                        key, tensors, spec, wrapper.codes)
                    target.quant_state = target._parameters[leaf].quant_state
            owner.layers[index] = wrapper.inner

        for wrapper in (model.get_input_embeddings(), model.get_output_embeddings()):
            if type(wrapper) is not stream._streamed_large_layer_class():
                continue
            key = wrapper.key
            source_index = wrapper.pool.source_indices[key]
            try:
                weight = runtime.source.get(source_index, key).to(device=device, non_blocking=False)
            finally:
                runtime.source.release(source_index, None)
            stream._set_module_param(wrapper.inner, wrapper._weight_param_name, weight)
            _require(stream._replace_module_references(model, wrapper, wrapper.inner) > 0,
                     "could not remove consumed large-layer wrapper")
        runtime.source.close()
        for reader in runtime.expert_readers:
            reader.close()
        torch.cuda.synchronize()
        _require(_quiescent(runtime), "source readers did not stop before resident inference")
        _require(model_api.adapter_digest(model_api.trainable_parameters(model))[0]
                 == self._adapter_sha256, "resident conversion changed adapter names or bytes")
        self._admit()

    def _admit(self) -> None:
        import torch

        from benchmarks.harness import p3_deepseek_quality_eval as scoring
        from soup_cli.utils import layer_stream_runtime as stream
        from soup_cli.utils.expert_stream import streamed_experts_class

        _require(self.model is not None and self._receipt is None, "candidate is closed")
        model, runtime = self.model, self._session.runtime
        _require(not self._session.closed and _quiescent(runtime), "source I/O is not quiescent")
        _require(model.config._attn_implementation == "sdpa" and not model.config.use_cache,
                 "native SDPA without a KV cache is required")
        _require(not any(name in vars(model.config) for name in ("aux_loss_alpha", "seq_aux")),
                 "original admitted aux normalization must remain in force")
        device = runtime.expert_pool.device
        for name, value in (*model.named_parameters(), *model.named_buffers()):
            _require(not value.is_meta and value.device == device
                     and "float8" not in str(value.dtype),
                     f"non-CUDA or FP8 model tensor: {name}")
            state = getattr(value, "quant_state", None)
            if state is not None:
                _require(state.quant_type == "nf4" and state.blocksize == 64
                         and not state.nested and state.dtype == torch.bfloat16
                         and value.dtype == torch.uint8
                         and state.absmax.device == state.code.device == device,
                         f"nonresident or changed decoder NF4 payload: {name}")
        for name, module in scoring._module_nodes(model):
            _require(type(module) not in (stream._streamed_layer_class(),
                                         stream._streamed_large_layer_class()),
                     f"disk wrapper remains: {name}")
            _require(getattr(module, "_hf_hook", None) is None
                     and not getattr(module, "gradient_checkpointing", False),
                     f"offload/checkpointing hook remains: {name}")
            for kind in ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks",
                         "_backward_hooks"):
                _require(not getattr(module, kind, {}), f"runtime hook remains: {name}")
            if "forward" in vars(module):
                forward = getattr(module.forward, "__func__", None)
                _require(type(module).__module__.startswith("bitsandbytes.")
                         and forward is not None and forward.__module__ == stream.__name__
                         and forward.__qualname__
                         == "install_dequant_forward.<locals>._dequant_forward",
                         f"foreign forward, Oracle must be off: {name}")
            if type(module) is streamed_experts_class():
                _require(module.pool is self._lease
                         and any(module.reader is item for item in self._readers),
                         "resident routed module ownership changed")
        for kind in ("_global_forward_pre_hooks", "_global_forward_hooks",
                     "_global_backward_pre_hooks", "_global_backward_hooks"):
            _require(not getattr(torch.nn.modules.module, kind, {}), "global runtime hook refused")
        for reader in self._readers:
            for tensors in reader.tensors:
                _require(all(value.device == device and value.dtype in (torch.uint8, torch.float32)
                             for value in tensors.values()),
                         "routed storage must remain CUDA packed NF4 and FP32 statistics")

    def inventory(self) -> dict:
        """Account unique allocations and tensor views separately, including quant states."""
        self._admit()
        pool = self._session.runtime.expert_pool
        tensors, storages, indices = [], [], {}
        logical_totals: dict[str, int] = {}
        unique_totals: dict[str, int] = {}
        roles: dict[str, int] = {}

        def add(name: str, value: Any, category: str, role: str, **extra: Any) -> None:
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            owner = key not in indices
            if owner:
                indices[key] = len(storages)
                storages.append({"storage_id": indices[key], "first_owner": name,
                                 "category": category, "dtype": str(value.dtype),
                                 "device": str(value.device), "storage_bytes": storage.nbytes()})
                group = f"{category}|{value.dtype}|{value.device}"
                unique_totals[group] = unique_totals.get(group, 0) + storage.nbytes()
                roles[role] = roles.get(role, 0) + storage.nbytes()
            logical = value.numel() * value.element_size()
            group = f"{category}|{value.dtype}|{value.device}"
            logical_totals[group] = logical_totals.get(group, 0) + logical
            tensors.append({"name": name, "category": category, "role": role,
                            "dtype": str(value.dtype), "device": str(value.device),
                            "shape": list(value.shape), "tensor_bytes": logical,
                            "storage_id": indices[key], "storage_bytes": storage.nbytes(),
                            "owns_unique_storage": owner, **extra})

        for name, value in self.model.named_parameters(remove_duplicate=False):
            state = getattr(value, "quant_state", None)
            add(name, value, _category(name), "nf4_packed" if state else "parameter",
                logical_weight_shape=list(state.shape) if state else list(value.shape),
                compute_dtype=str(state.dtype) if state else str(value.dtype))
            if state is not None:
                add(name + "::absmax", state.absmax, _category(name), "nf4_statistics")
                add(name + "::code", state.code, "other", "shared_nf4_code")
        for name, value in self.model.named_buffers(remove_duplicate=False):
            add(name, value, _category(name), "buffer")
        for reader in self._readers:
            for expert, payload in enumerate(reader.tensors):
                for name, value in payload.items():
                    add(f"routed_layer_{reader.layer_idx}.expert_{expert}.{name}", value, "routed",
                        "nf4_statistics" if "::" in name else "nf4_packed")
        add("expert_compute_arena", pool._arena, "compute_arena", "reusable_allocation")
        for name, value in pool._buffers.items():
            add("expert_compute_arena." + name, value, "compute_arena", "arena_view")
        return {
            "schema": "p3-moonlight-resident-nf4-storage-v1", "mode": self.mode,
            "evaluation_storage": "resident_nf4",
            "implementation_sha256": self._implementation_sha256,
            "implementation_path": str(self._implementation_path),
            "adapter_parameter_sha256": self._adapter_sha256,
            "storage_policy": "all frozen payloads, NF4 statistics and adapters CUDA-resident",
            "tensors": tensors, "unique_storages": storages,
            "unique_storage_bytes": sum(row["storage_bytes"] for row in storages),
            "unique_storage_bytes_by_category_dtype_device": unique_totals,
            "tensor_view_bytes_by_category_dtype_device_including_aliases": logical_totals,
            "unique_storage_bytes_by_role": roles,
            "compute_arena": {
                "plan": asdict(pool.plan), "storage_bytes": pool._arena.untyped_storage().nbytes(),
                "cuda_allocation_delta_at_original_construction": pool.cuda_allocation_delta,
                "reusable_bf16_dense_bytes": pool.plan.dense_bytes,
                "unused_packed_and_statistics_slot_bytes": pool.plan.packed_and_stats_bytes,
                "host_staging_bytes_retained": 0, "maximum_simultaneous_decoded_experts": 1,
                "semantics": "one overwritten expert pair, never a persistent decoded expert store",
            },
            "routed_preparation_bytes_read": sum(
                row.preparation_bytes_read for row in self._readers),
            "measured_routed_host_or_disk_weight_bytes": sum(
                row.bytes_read for row in self._readers),
            "resident_expert_leases": self._lease.fetches,
            "source_io_quiescent": _quiescent(self._session.runtime),
            "source_runtime_tier_unchanged": self._session.runtime.tier,
            "source_decoder_and_large_slots_released": True,
            "unattributed_memory": "activations, dense dequantization temporaries and CUDA "
                                   "allocator/context overhead require measured peaks",
        }

    def admit_runtime(
        self, session: Any, tokenizer: Any, *, test_fixture: bool, profile_id: str,
    ) -> dict:
        from . import p3_deepseek_quality_eval as scoring
        from .p3_deepseek_quality_profiles import get_profile

        _require(type(self) is ResidentNF4 and session is self._session
                 and profile_id == PROFILE and not test_fixture,
                 "resident inference requires its admitted native Moonlight session")
        self._admit()
        device = self._session.runtime.expert_pool.device
        scoring._admit_native_execution(str(device))
        _require(len(tokenizer) == self.model.config.vocab_size
                 == get_profile(PROFILE).tokenizer.vocabulary_size,
                 "resident native head/tokenizer vocabulary differs")
        return {
            "profile_id": PROFILE, "tier": "resident_nf4", "device": str(device),
            "source_runtime_tier": session.runtime.tier,
            "source_sha256": self._implementation_sha256,
            "tokenizer_vocabulary_size": len(tokenizer),
            "native_head_vocabulary_size": self.model.config.vocab_size,
            "fp8": False, "oracle": False, "test_fixture": False,
        }

    @contextmanager
    def scoring_context(self, tokenizer: Any) -> Iterator[Any]:
        from . import p3_deepseek_quality_eval as scoring

        _require(not self._scoring, "resident scoring context is already active")
        self._scoring = True
        try:
            with scoring._scoring_session(
                self._session, tokenizer, profile_id=PROFILE, resident=self,
            ) as wrapped:
                yield wrapped
        finally:
            self._scoring = False

    def close(self) -> dict:
        """Stop owned resources and invalidate the consumed model. Idempotent."""
        if self._receipt is not None:
            return _close_receipt(self._receipt)
        _require(not self._scoring, "close scoring context before candidate")
        session = self._session
        errors = []
        try:
            session.runtime.close()
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
        for reader in self._readers:
            reader.close()
        if self._lease is not None:
            self._lease.close()
        self._readers.clear()
        session.runtime.pool.buffers.clear()
        session.runtime.pool.events.clear()
        session.runtime.pool.nbytes = 0
        if session.runtime.large_pool is not None:
            session.runtime.large_pool.buffer = None
            session.runtime.large_pool.event = None
            session.runtime.large_pool.nbytes = 0
        session.retained_resources[:] = [value for value in session.retained_resources
                                        if value is not self]
        session.model = None
        self.model = None
        self._lease = None
        session.config = None
        arena = session.runtime.expert_pool
        released = (arena._closed and not arena._buffers and arena._arena is None
                    and arena._staging_owner is None and arena._staging is None
                    and _quiescent(session.runtime))
        self._session = None
        self._receipt = {
            "mode": self.mode, "closed": not errors and released, "cleanup_errors": errors,
            "candidate_and_session_model_references_dropped": True,
            "resident_expert_payloads_dropped": True,
            "compute_arena_closed": arena._closed, "owned_storage_released": released,
            "source_io_quiescent": _quiescent(session.runtime),
            "caller_must_close_source_session": True,
            "remaining_storage_policy": "drop external model/scoring-wrapper aliases, collect, "
                                        "then check CUDA allocation baseline",
        }
        return _close_receipt(self._receipt)


def prepare_resident(session: Any) -> ResidentNF4:
    """Consume a fresh disk session, preserving its already-admitted adapter."""
    candidate = ResidentNF4(session)
    try:
        candidate._install()
        return candidate
    except BaseException as error:
        try:
            error.residency_cleanup = candidate.close()
        except BaseException as cleanup_error:
            error.residency_cleanup = {"cleanup_error": f"{type(cleanup_error).__name__}: "
                                                       f"{cleanup_error}"}
        raise
