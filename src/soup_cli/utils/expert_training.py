"""Opt-in D3 installation for the regular streamed SFT model builder.

Readers, a shared bounded weight pool and the optional GEMM handle are owned by
the returned resource. No benchmark modules are imported by the training path.
"""

from __future__ import annotations

import inspect
import os
import platform
import sys
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Any

_MATH_LOCK = threading.Lock()


def _uses_grouped_gemm(architecture: str, backend: str, device: str) -> bool:
    """Whether this recipe has a validated CUDA grouped-GEMM training path."""
    return (str(device).split(":", 1)[0] == "cuda"
            and backend != "sequential"
            and architecture != "deepseek_v3")


def _cpu_training_platform() -> tuple[str, str]:
    return sys.platform, platform.machine().lower()


def validate_cpu_expert_recipe(*, dtype: str) -> None:
    """Admit the qualified CPU development recipe, not unverified Apple arithmetic."""
    operating_system, machine = _cpu_training_platform()
    if operating_system not in ("linux", "win32") or machine not in ("amd64", "x86_64"):
        raise ValueError(
            "CPU stream_experts exactness is qualified on Linux/Windows x86-64 only; "
            "Apple Silicon/other CPU platforms are unqualified. Use the qualified "
            "Linux CUDA recipe or disable training.stream_experts."
        )
    if str(dtype).removeprefix("torch.") != "float32":
        raise ValueError("CPU stream_experts exactness requires float32 model weights")
    import torch

    if torch.get_num_threads() != 1:
        raise ValueError(
            "CPU stream_experts exactness requires one CPU thread; set "
            "OMP_NUM_THREADS=1 before starting Python or torch.set_num_threads(1)."
        )


def validate_expert_training(config: Any, options: Any, *, device: str, dtype: str) -> None:
    """Validate the model and requested backend before cache reads/allocations."""
    arch = getattr(config, "model_type", "")
    kind = str(device).split(":", 1)[0]
    if arch not in ("olmoe", "deepseek_v3"):
        raise ValueError("training.stream_experts supports native OLMoE or DeepSeek-V3 only")
    if getattr(config, "output_router_logits", False):
        raise ValueError("stream_experts does not yet support the router auxiliary loss")
    if getattr(config, "hidden_act", "silu") != "silu":
        raise ValueError("stream_experts requires stateless SiLU experts")
    if kind not in ("cpu", "cuda"):
        raise ValueError("stream_experts supports CPU or CUDA only")
    if arch == "deepseek_v3" and kind == "cuda":
        if sys.platform != "linux":
            raise ValueError("DeepSeek-V3 CUDA expert training currently requires Linux")
        if dtype != "bfloat16":
            raise ValueError("DeepSeek-V3 CUDA expert training requires native BF16")
        if options.backend not in ("sequential", "grouped"):
            raise ValueError("DeepSeek-V3 CUDA expert training supports sequential/grouped only")
        geometry = (getattr(config, "hidden_size", None),
                    getattr(config, "moe_intermediate_size", None),
                    getattr(config, "n_routed_experts", None))
        if geometry != (2048, 1408, 64):
            raise ValueError(
                "DeepSeek-V3 CUDA expert training is qualified for academic-ds-9B "
                "H=2048, I=1408, E=64 only"
            )
    if options.backend == "triton" and (kind != "cuda" or dtype != "bfloat16"):
        raise ValueError("the Triton expert backend requires CUDA BF16")
    if options.backend == "fused_nf4" and (kind != "cuda" or dtype != "bfloat16"):
        raise ValueError("the fused NF4 expert backend requires Linux CUDA BF16")
    if kind == "cpu":
        validate_cpu_expert_recipe(dtype=dtype)
    if kind == "cuda" and options.backend != "sequential":
        if sys.platform != "linux":
            raise ValueError("grouped CUDA expert compute currently requires Linux")
        if dtype != "bfloat16":
            raise ValueError("grouped CUDA expert compute requires native BF16")
        if arch == "olmoe" and (config.hidden_size, config.intermediate_size) != (2048, 1024):
            raise ValueError("grouped CUDA geometry is qualified for OLMoE H=2048, I=1024 only")
        if _uses_grouped_gemm(arch, options.backend, device):
            path = options.cublas_library
            if not path or not os.path.isabs(path) or not os.path.isfile(path):
                raise ValueError("grouped CUDA requires an explicit existing cublas_library path")
        if options.backend == "fused_nf4":
            import torch

            if torch.cuda.get_device_capability(device) != (8, 6):
                raise ValueError("packed WMMA training candidate is currently limited to sm_86")


def initialize_deepseek_rotary(model: Any, *, device: str) -> None:
    """Initialize DeepSeek RoPE frequencies on the execution device.

    Transformers constructs these non-persistent buffers on CPU in the meta
    skeleton. Recomputing through the model's registered RoPE initializer on
    CUDA preserves the checkpoint's YaRN arithmetic order for exact replay.
    """
    if getattr(getattr(model, "config", None), "model_type", "") != "deepseek_v3":
        return
    if str(device).split(":", 1)[0] != "cuda":
        return

    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rotary = [module for module in model.modules()
              if type(module).__name__ == "DeepseekV3RotaryEmbedding"]
    if len(rotary) != 1:
        raise ValueError(f"DeepSeek-V3 requires one native rotary module, got {len(rotary)}")
    module = rotary[0]
    rope_type = getattr(module, "rope_type", None)
    initializer = ROPE_INIT_FUNCTIONS.get(rope_type)
    if not callable(initializer):
        raise ValueError(f"unsupported DeepSeek-V3 rotary type: {rope_type!r}")
    try:
        supports_device = "device" in inspect.signature(initializer).parameters
    except (TypeError, ValueError):
        supports_device = False
    if not supports_device:
        raise ValueError("installed Transformers RoPE initializer cannot target the CUDA device")
    target = torch.device(device)
    inv_freq, attention_scaling = initializer(module.config, device=target)
    device_matches = (inv_freq.device.type == target.type
                      and (target.index is None or inv_freq.device.index == target.index))
    if not device_matches or inv_freq.dtype != torch.float32:
        raise ValueError("DeepSeek-V3 RoPE initializer returned an unexpected tensor device/dtype")
    # Keep both registered, non-persistent buffers coherent. Dynamic RoPE
    # implementations restore original_inv_freq when returning to short inputs.
    module.register_buffer("inv_freq", inv_freq, persistent=False)
    module.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
    module.attention_scaling = attention_scaling


def expert_training_reservations(shard_dir, index, options, *, dtype, architecture, device):
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
        plan = plan_grouped_expert_buffer(reader, dtype=dtype, group_width=width,
                                          packed_only=options.backend == "fused_nf4")
        if plan.weight_bytes > options.weight_budget_bytes:
            raise ValueError("expert weight arena exceeds weight_budget_bytes before allocation")
        gemm_bytes = (options.call_budget_bytes + 4 * 2**20
                      if _uses_grouped_gemm(architecture, options.backend, device) else 0)
        return {
            "host_bytes": plan.host_staging_bytes,
            # Packed arenas can occupy an allocator block larger than their
            # requested size. Their constructor checks this configured ceiling.
            "device_bytes": ((options.weight_budget_bytes if options.backend == "fused_nf4"
                              else plan.weight_bytes) + options.activation_budget_bytes
                             + gemm_bytes),
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
    use_gemm = _uses_grouped_gemm(config.model_type, options.backend, device)
    if index.expert_layout != "per_expert":
        raise ValueError("stream_experts requires a per_expert cache; reshard the checkpoint")
    layers = decoder_owner(model).layers
    expected_layers = tuple(i for i, layer in enumerate(layers)
                            if _find_experts(layer) is not None)
    if not expected_layers or tuple(index.expert_layers) != expected_layers:
        raise ValueError("model and cache must identify the same independently readable experts")
    if config.model_type == "olmoe" and expected_layers != tuple(range(len(layers))):
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
        first = readers[expected_layers[0]]
        plan = plan_grouped_expert_buffer(first, dtype=dtype, group_width=width,
                                          packed_only=options.backend == "fused_nf4")
        if plan.weight_bytes > options.weight_budget_bytes:
            raise ValueError("expert weight arena exceeds weight_budget_bytes before allocation")
        if str(device).startswith("cuda"):
            if use_gemm:
                _math_policy(stack)
            required = ((options.weight_budget_bytes if options.backend == "fused_nf4"
                         else plan.weight_bytes) + options.activation_budget_bytes
                        + (options.call_budget_bytes + 4 * 2**20 if use_gemm else 0))
            if required > torch.cuda.mem_get_info(device)[0]:
                raise ValueError("expert arena/activation/workspace budgets exceed available VRAM")
        pool = GroupedExpertPool(
            first, device=device, dtype=dtype, group_width=width,
            weight_budget_bytes=options.weight_budget_bytes, codes=codes,
            packed_only=options.backend == "fused_nf4",
        )
        stack.callback(pool.close)
        gemm = None
        if use_gemm:
            from soup_cli.utils.expert_group_gemm import GroupedGemm

            with torch.cuda.device(device):
                if options.backend == "fused_nf4":
                    from soup_cli.utils.expert_group_fused_runtime import PackedGemm

                    gemm = PackedGemm(first, device=pool.device,
                                      call_budget_bytes=options.call_budget_bytes)
                else:
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
            "expert_compute_backend": ("packed-hybrid-wmma" if options.backend == "fused_nf4"
                                       else "grouped-restore-eager"
                                       if options.backend == "grouped" and gemm is None
                                       else "native-eager" if gemm is None else "cublas-hybrid"),
            "expert_disk_bytes": sum(reader._size for reader in readers.values()),
            "expert_weight_bytes": plan.weight_bytes,
            "expert_host_staging_bytes": plan.host_staging_bytes,
            "expert_activation_budget_bytes": options.activation_budget_bytes,
        })
    except BaseException:
        stack.close()
        raise
