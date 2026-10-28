"""Opt-in D3 installation for the regular streamed SFT model builder.

Readers, a shared bounded weight pool and the optional GEMM handle are owned by
the returned resource. No benchmark modules are imported by the training path.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Any

_MATH_LOCK = threading.Lock()


def validate_expert_training(config: Any, options: Any, *, device: str, dtype: str) -> None:
    """Validate the model and requested backend before cache reads/allocations."""
    if getattr(config, "model_type", "") != "olmoe":
        raise ValueError("training.stream_experts currently supports native OLMoE only")
    if getattr(config, "output_router_logits", False):
        raise ValueError("stream_experts does not yet support the router auxiliary loss")
    if getattr(config, "hidden_act", "silu") != "silu":
        raise ValueError("stream_experts requires stateless SiLU experts")
    kind = str(device).split(":", 1)[0]
    if kind not in ("cpu", "cuda"):
        raise ValueError("stream_experts supports CPU or CUDA only")
    if options.backend == "triton" and (kind != "cuda" or dtype != "bfloat16"):
        raise ValueError("the Triton expert backend requires CUDA BF16")
    if kind == "cuda" and options.backend != "sequential":
        if sys.platform != "linux":
            raise ValueError("grouped CUDA expert compute currently requires Linux")
        if dtype != "bfloat16":
            raise ValueError("grouped CUDA expert compute requires native BF16")
        if (config.hidden_size, config.intermediate_size) != (2048, 1024):
            raise ValueError("grouped CUDA geometry is qualified for OLMoE H=2048, I=1024 only")
        path = options.cublas_library
        if not path or not os.path.isabs(path) or not os.path.isfile(path):
            raise ValueError("grouped CUDA requires an explicit existing cublas_library path")


def expert_training_reservations(shard_dir, index, options, *, dtype):
    """Read only a shard header, to charge the extra arena and staging in preflight."""
    from soup_cli.utils.expert_group import plan_grouped_expert_buffer
    from soup_cli.utils.expert_shard import ExpertShardReader
    from soup_cli.utils.layer_shard import expert_paths

    paths = expert_paths(shard_dir, index)
    if not paths:
        raise ValueError("stream_experts requires nonempty expert metadata")
    reader = ExpertShardReader(
        next(iter(paths.values())), count=index.expert_count, shapes=index.expert_shapes,
        quant_specs=index.expert_quant_specs, dtype=index.dtype, direct=False,
    )
    try:
        width = 1 if options.backend == "sequential" else options.group_width
        plan = plan_grouped_expert_buffer(reader, dtype=dtype, group_width=width)
        if plan.weight_bytes > options.weight_budget_bytes:
            raise ValueError("expert weight arena exceeds weight_budget_bytes before allocation")
        return {
            "host_bytes": plan.host_staging_bytes,
            "device_bytes": (plan.weight_bytes + options.activation_budget_bytes
                             + options.call_budget_bytes + 4 * 2**20),
        }
    finally:
        reader.close()


def _math_policy(stack: ExitStack) -> None:
    """Own and restore the explicit D3 process BLAS policy for this runtime."""
    import torch

    if not _MATH_LOCK.acquire(blocking=False):
        raise RuntimeError("another expert training runtime owns the CUDA math policy")
    stack.callback(_MATH_LOCK.release)
    cuda = torch.backends.cuda
    previous = (
        cuda.preferred_blas_library(), cuda.matmul.allow_tf32,
        cuda.matmul.allow_bf16_reduced_precision_reduction,
        cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
    )

    def restore():
        cuda.preferred_blas_library(previous[0])
        cuda.matmul.allow_tf32 = previous[1]
        cuda.matmul.allow_bf16_reduced_precision_reduction = previous[2:]

    stack.callback(restore)
    cuda.preferred_blas_library("cublaslt")
    cuda.matmul.allow_tf32 = False
    cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)


class ExpertTrainingResources:
    """Lifetime owner attached to StreamRuntime, including rollback on failure."""

    def __init__(self, stack, pool, readers, modules, gemm, plan):
        self._stack = stack
        self.pool, self.readers, self.modules, self.gemm = pool, readers, modules, gemm
        self.plan = plan
        self.closed = False

    def close(self):
        if not self.closed:
            self._stack.close()
            self.closed = True

    def stats(self):
        return {
            **self.plan,
            "expert_fetches": self.pool.fetches,
            "expert_logical_bytes_read": self.pool.logical_bytes_read,
            "expert_gemm": self.gemm.stats() if self.gemm is not None else None,
        }


def install_expert_training(model, shard_dir, index, options, *, device, dtype, codes):
    """Atomically replace frozen experts before the layer runtime enumerates meta weights."""
    import torch

    from soup_cli.utils.expert_group import GroupedExpertPool, plan_grouped_expert_buffer
    from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class
    from soup_cli.utils.expert_shard import ExpertShardReader
    from soup_cli.utils.expert_stream import _find_experts
    from soup_cli.utils.layer_shard import expert_paths
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    config = model.config
    validate_expert_training(config, options, device=device, dtype=dtype)
    if index.expert_layout != "per_expert":
        raise ValueError("stream_experts requires a per_expert cache; reshard the checkpoint")
    layers = decoder_owner(model).layers
    if tuple(index.expert_layers) != tuple(range(len(layers))):
        raise ValueError("OLMoE requires independently readable experts in every layer")
    stack = ExitStack()
    replacements = []
    try:
        readers = {}
        for idx, path in expert_paths(shard_dir, index).items():
            reader = ExpertShardReader(
                path, count=index.expert_count, shapes=index.expert_shapes,
                quant_specs=index.expert_quant_specs, dtype=index.dtype, direct=False,
            )
            stack.callback(reader.close)
            readers[idx] = reader
            found = _find_experts(layers[idx])
            if found is None:
                raise ValueError(f"layer {idx}: routed experts are absent")
            parent, attr, module = found
            expected = ((reader.count, *reader.shapes["gate_up"]),
                        (reader.count, *reader.shapes["down"]))
            params = dict(module.named_parameters())
            if (set(params) != {"gate_up_proj", "down_proj"}
                    or tuple(tuple(params[key].shape) for key in ("gate_up_proj", "down_proj"))
                    != expected or any(p.requires_grad for p in params.values())):
                raise ValueError(f"layer {idx}: expert shape/adapters/frozen-weight mismatch")
            replacements.append((parent, attr, module, reader))
        width = 1 if options.backend == "sequential" else options.group_width
        first = readers[0]
        plan = plan_grouped_expert_buffer(first, dtype=dtype, group_width=width)
        if plan.weight_bytes > options.weight_budget_bytes:
            raise ValueError("expert weight arena exceeds weight_budget_bytes before allocation")
        if str(device).startswith("cuda"):
            _math_policy(stack)
            required = (plan.weight_bytes + options.activation_budget_bytes
                        + options.call_budget_bytes + 4 * 2**20)
            if required > torch.cuda.mem_get_info(device)[0]:
                raise ValueError("expert arena/activation/workspace budgets exceed available VRAM")
        pool = GroupedExpertPool(
            first, device=device, dtype=dtype, group_width=width,
            weight_budget_bytes=options.weight_budget_bytes, codes=codes,
        )
        stack.callback(pool.close)
        gemm = None
        if str(device).startswith("cuda") and options.backend != "sequential":
            from soup_cli.utils.expert_group_gemm import GroupedGemm

            with torch.cuda.device(device):
                gemm = GroupedGemm(
                    Path(options.cublas_library), call_budget_bytes=options.call_budget_bytes,
                    native_olmoe_rows_max=options.native_rows_max,
                    native_olmoe_rows_from=options.native_rows_from,
                )
            stack.callback(gemm.close)
        cls = grouped_autograd_experts_class()
        modules = []
        for _parent, _attr, original, reader in replacements:
            modules.append(cls(
                reader, pool, original.act_fn, gemm=gemm,
                activation_budget_bytes=options.activation_budget_bytes,
                activation_backend="triton" if options.backend == "triton" else "native",
            ))
        installed = []
        try:
            for (parent, attr, original, _reader), module in zip(replacements, modules):
                installed.append((parent, attr, original))
                setattr(parent, attr, module)
        except BaseException:
            for parent, attr, original in reversed(installed):
                setattr(parent, attr, original)
            raise
        return ExpertTrainingResources(stack, pool, readers, modules, gemm, {
            "expert_backend": options.backend, "expert_group_width": width,
            "expert_source": "disk-buffered",
            "expert_compute_backend": "native-eager" if gemm is None else "cublas-hybrid",
            "expert_disk_bytes": sum(reader._size for reader in readers.values()),
            "expert_weight_bytes": plan.weight_bytes,
            "expert_host_staging_bytes": plan.host_staging_bytes,
            "expert_activation_budget_bytes": options.activation_budget_bytes,
        })
    except BaseException:
        stack.close()
        raise
