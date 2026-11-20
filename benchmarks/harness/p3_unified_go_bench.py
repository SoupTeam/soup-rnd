"""Unified paired A-B-B-A benchmark & parity runner for the DeepSeek P3 GO stack.

Combines all four verified GO pillars into one matched full training step (L2LStep + AdamW):
1. Track 1 (Memory & High-k Unblock):
   - M1: Decoupled sub-window backward (``mbr_decoupled_subwindow=True``, ``subwindow=1``)
   - M2+: Zero-materialization streaming per-expert forward/backward (0 ``WFE`` / 0 ``dh_chunks``)
   - H2: ABA-safe window NF4 dequant reuse (``mbr_window_dequant=True``)
   - M3: ``LargeLayerBufferPool`` release during decoder forward & backward (−1.726 GiB on 671B)
2. Track 2 (NVMe I/O + Compute Engine):
   - ``AsyncExpertController`` (C++17 ``PackedStreamAsync``, ``io_workers=4``, ``ring_slots=8``,
     2-slot async DMA, cross-layer backward pre-staging, optional ``cuda_nf4_decode``)
   - ``--expert-gemm batch_invariant`` (PR #73 Triton fixed-tile ``merged_invariant`` on CUDA)
3. Track 3 (Long-Context Head):
   - ``l2l_sft_head(..., impl="chunked" | "cce" | "full", chunk=256)`` from ``stream_chunked_ce.py``

Supports ``--case tiny`` (CPU/CUDA parity + smoke), ``--case kanana`` (30B-A3B), and
``--case custom --model-dir <path>`` (DeepSeek-V3 671B) on RTX 4060 Ti 8 GB + NVMe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

KANANA_REPO = "kakaocorp/kanana-2-30b-a3b-base-2601"
KANANA_REV = "0d31365acf656e5f9a593cacc4913e9c0d36cb1c"
KANANA_TARGETS = ("q_b_proj", "kv_b_proj", "o_proj")

QWEN38_27B_REPO = "Qwen/Qwen3.8-27B"
QWEN38_27B_REV = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
QWEN38_27B_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj")

ARM_SPECS: Dict[str, Dict[str, Any]] = {
    "l2l_ref": dict(
        route_preserving_mbr=False,
        mbr_decoupled_subwindow=False,
        mbr_window_dequant=False,
        release_large_pool_during_layers=False,
        use_controller=False,
        head_impl="full",
    ),
    "std_mbr": dict(
        route_preserving_mbr=True,
        mbr_decoupled_subwindow=False,
        mbr_window_dequant=False,
        release_large_pool_during_layers=False,
        use_controller=False,
        head_impl="full",
    ),
    "mbr_h2_sync": dict(
        route_preserving_mbr=True,
        mbr_decoupled_subwindow=False,
        mbr_window_dequant=True,
        release_large_pool_during_layers=False,
        use_controller=False,
        head_impl="full",
    ),
    "unified_go_qd1": dict(
        route_preserving_mbr=True,
        mbr_decoupled_subwindow=True,
        mbr_window_dequant=True,
        release_large_pool_during_layers=True,
        use_controller=True,
        override_io_workers=1,
        override_ring_slots=4,
        override_cross_layer=False,
        head_impl=None,
    ),
    "unified_go_full_head": dict(
        route_preserving_mbr=True,
        mbr_decoupled_subwindow=True,
        mbr_window_dequant=True,
        release_large_pool_during_layers=True,
        use_controller=True,
        head_impl="full",
    ),
    "unified_go": dict(
        route_preserving_mbr=True,
        mbr_decoupled_subwindow=True,
        mbr_window_dequant=True,
        release_large_pool_during_layers=True,
        use_controller=True,
        head_impl=None,  # uses --head-loss (default: chunked)
        activation_codec="bf16",
        fp8_group_size=None,
        use_oracle=False,
    ),
    "l2l_ref_fp8_oracle": dict(
        route_preserving_mbr=False,
        mbr_decoupled_subwindow=False,
        mbr_window_dequant=False,
        release_large_pool_during_layers=False,
        use_controller=False,
        head_impl="full",
        activation_codec="fp8_e4m3",
        fp8_group_size=64,
        use_oracle=True,
    ),
    "unified_go_fp8_oracle": dict(
        route_preserving_mbr=True,
        mbr_decoupled_subwindow=True,
        mbr_window_dequant=True,
        release_large_pool_during_layers=True,
        use_controller=True,
        head_impl=None,
        activation_codec="fp8_e4m3",
        fp8_group_size=64,
        use_oracle=True,
    ),
    "unified_go_fp8_perrow_oracle": dict(
        route_preserving_mbr=True,
        mbr_decoupled_subwindow=True,
        mbr_window_dequant=True,
        release_large_pool_during_layers=True,
        use_controller=True,
        head_impl=None,
        activation_codec="fp8_e4m3",
        fp8_group_size=None,
        use_oracle=True,
    ),
    "unified_go_fp8_no_oracle": dict(
        route_preserving_mbr=True,
        mbr_decoupled_subwindow=True,
        mbr_window_dequant=True,
        release_large_pool_during_layers=True,
        use_controller=True,
        head_impl=None,
        activation_codec="fp8_e4m3",
        fp8_group_size=64,
        use_oracle=False,
    ),
}


def _git_sha() -> str:
    try:
        repo_root = str(Path(__file__).resolve().parents[2])
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _tensor_map_sha256(tensors: Dict[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        value = tensors[name]
        if value is None:
            digest.update(b"<None>")
            continue
        cpu = value.detach().contiguous().cpu()
        digest.update(str(cpu.dtype).encode("utf-8"))
        digest.update(str(tuple(cpu.shape)).encode("utf-8"))
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _compare_maps(ref: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    import math

    import torch

    max_abs = 0.0
    sq_diff = 0.0
    sq_ref = 0.0
    min_cosine = 1.0
    bit_equal = True
    for name in sorted(ref):
        a, b = ref[name], cand[name]
        if a is None and b is None:
            continue
        if a is None or b is None:
            return {
                "bit_equal": False,
                "max_abs": float("inf"),
                "relative_l2": float("inf"),
                "min_cosine": 0.0,
            }
        if not torch.equal(a, b):
            bit_equal = False
        af = a.detach().double().reshape(-1)
        bf = b.detach().double().reshape(-1)
        delta = bf - af
        cur_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        if cur_max > max_abs:
            max_abs = cur_max
        ref_sq = float(af.dot(af).item())
        cand_sq = float(bf.dot(bf).item())
        err_sq = float(delta.dot(delta).item())
        sq_diff += err_sq
        sq_ref += ref_sq
        if ref_sq > 0 and cand_sq > 0:
            cos = float(af.dot(bf).item()) / math.sqrt(ref_sq * cand_sq)
        else:
            cos = 1.0 if ref_sq == cand_sq == 0 else 0.0
        if cos < min_cosine:
            min_cosine = cos
    rel_l2 = (sq_diff**0.5) / max(sq_ref**0.5, 1e-30)
    return {
        "bit_equal": bit_equal,
        "max_abs": max_abs,
        "relative_l2": rel_l2,
        "min_cosine": min_cosine,
    }


def _rss_gib() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024**3)
    except Exception:
        return 0.0


def _nvml_used_gib(device_index: int = 0) -> Optional[float]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return float(info.used) / (1024**3)
    except Exception:
        return None


class _NvmlPeakPoller:
    """Background 20 ms NVML + RSS sampler to record true in-step GPU & host peaks."""

    def __init__(self, enabled: bool, device_index: int = 0, interval_s: float = 0.02) -> None:
        self.enabled = enabled
        self.device_index = device_index
        self.interval_s = interval_s
        self.peak_gib: Optional[float] = None
        self.rss_peak_gib: float = 0.0
        self.samples: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        proc = None
        try:
            import psutil

            proc = psutil.Process()
        except Exception:
            proc = None
        nvml_handle = None
        pynvml = None
        if self.enabled:
            try:
                import pynvml as _pynvml

                _pynvml.nvmlInit()
                nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                pynvml = _pynvml
            except Exception:
                nvml_handle = None
        try:
            while not self._stop.is_set():
                if nvml_handle is not None and pynvml is not None:
                    used = float(pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used) / (1024**3)
                    if self.peak_gib is None or used > self.peak_gib:
                        self.peak_gib = used
                if proc is not None:
                    rss = float(proc.memory_info().rss) / (1024**3)
                    if rss > self.rss_peak_gib:
                        self.rss_peak_gib = rss
                self.samples += 1
                self._stop.wait(self.interval_s)
        finally:
            if pynvml is not None:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def __enter__(self) -> "_NvmlPeakPoller":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)


def _prepare_shards(
    work: Path,
    *,
    case: str,
    model_dir: Optional[str],
    dtype: str,
    storage: str,
    device: str,
    prepared: bool,
) -> Tuple[Path, Path]:
    from benchmarks.harness.b5_real_moe import _meta_construction
    from benchmarks.harness.b13_moe_fixture import moe_config, moe_weights_dir
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        quantised_layer_suffixes,
        shard_layout_of,
    )

    work.mkdir(parents=True, exist_ok=True)
    shards = work / "shards"
    if model_dir is not None:
        weights = Path(model_dir)
    elif case == "tiny":
        weights = work / "weights"
        if not (weights / "config.json").exists():
            cfg = moe_config(
                vocab_size=256,
                tie_word_embeddings=False,
                hidden_size=128,
                intermediate_size=256,
                moe_intermediate_size=64,
                num_hidden_layers=3,
                first_k_dense_replace=1,
                n_routed_experts=8,
                n_shared_experts=1,
                num_experts_per_tok=2,
                q_lora_rank=64,
                kv_lora_rank=32,
                qk_rope_head_dim=16,
                qk_nope_head_dim=16,
                v_head_dim=16,
                num_attention_heads=4,
                num_key_value_heads=4,
            )
            moe_weights_dir(work, cfg)
    elif case == "tiny_qwen38":
        weights = work / "weights"
        if not (weights / "config.json").exists():
            import torch
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

            weights.mkdir(parents=True, exist_ok=True)
            qcfg = Qwen3_5TextConfig(
                vocab_size=512,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                layer_types=[
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
                tie_word_embeddings=False,
                pad_token_id=0,
                use_cache=False,
            )
            torch.manual_seed(42)
            Qwen3_5ForCausalLM(qcfg).to(torch.bfloat16).save_pretrained(weights)
    elif case in ("kanana", "qwen38_27b"):
        weights = work / "weights"
        if not (weights / "config.json").exists():
            if prepared:
                raise FileNotFoundError(f"--prepared set, but {weights}/config.json is missing")
            from huggingface_hub import snapshot_download

            repo_id = KANANA_REPO if case == "kanana" else QWEN38_27B_REPO
            rev_id = KANANA_REV if case == "kanana" else QWEN38_27B_REV
            snapshot_download(
                repo_id,
                revision=rev_id,
                local_dir=weights,
                token=False,
                allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
                max_workers=4,
            )
    else:
        raise ValueError(f"unsupported case {case!r}; pass --model-dir for custom checkpoints")

    if not (shards / "index.json").exists():
        if prepared:
            raise FileNotFoundError(f"--prepared set, but {shards}/index.json is missing")
        stream_cfg = load_stream_config(str(weights), False)
        stream_arch = stream_arch_of(stream_cfg)
        is_moe = stream_arch == "deepseek_v3"
        quant = "nf4" if storage == "nf4" else None
        with _meta_construction():
            skeleton = build_meta_skeleton(
                str(weights), dtype=dtype, quant=quant, double_quant=False
            )
        layout = shard_layout_of(skeleton)
        suffixes = quantised_layer_suffixes(skeleton) if quant == "nf4" else None
        del skeleton
        shard_checkpoint(
            str(weights),
            str(shards),
            dtype=dtype,
            arch=stream_arch,
            quant=quant,
            double_quant=False,
            quant_device=device,
            expert_layout="per_expert" if is_moe else "",
            layout=layout,
            quant_suffixes=suffixes,
        )
    return weights, shards


def _run_arm(
    arm: str,
    *,
    weights: Path,
    shards: Path,
    storage: str,
    k: int,
    seq_len: int,
    group_size: int,
    subwindow: int,
    device: str,
    dtype: str,
    expert_gemm: str,
    expert_loader: str,
    io_workers: int,
    ring_slots: int,
    cross_layer_prefetch: bool,
    cuda_nf4_decode: bool,
    head_loss: str,
    ce_chunk: int,
    host_cache_gib: float,
    lora_state: Optional[Dict[str, Any]],
    step_batches_cpu: Sequence[Sequence[Any]],
    track_saved_ledger: bool,
    lr: float = 1e-4,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    import torch
    from peft import LoraConfig, TaskType

    from benchmarks.harness import b13_moe_fixture as fixture
    from benchmarks.harness.b5_real_moe import _meta_construction
    from soup_cli.utils import l2l_step
    from soup_cli.utils.async_expert_controller import AsyncExpertController, cpu_packed_control
    from soup_cli.utils.fp8_activations import FP8ActivationStore, fp8_activation_allocation
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.routing_oracle import DeepseekRoutingOracle
    from soup_cli.utils.stream_chunked_ce import l2l_sft_head

    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_SPECS)}")
    spec = ARM_SPECS[arm]

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(71)

    index = read_shard_index(str(shards))
    is_moe = index.arch == "deepseek_v3"
    targets = list(KANANA_TARGETS) if is_moe else list(QWEN38_27B_TARGETS)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    use_packed = is_cuda and storage == "nf4" and is_moe
    cache_bytes = int(host_cache_gib * (1024**3)) if use_packed else 0
    with _meta_construction():
        model, runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=str(shards),
            index=index,
            lora_config=lora_cfg,
            device=device,
            dtype=dtype,
            quant="nf4" if storage == "nf4" else "none",
            double_quant=False,
            tier="disk" if is_cuda else "ram",
            pin=is_cuda,
            two_read_experts=is_moe,
            grouped_expert_routes=is_moe,
            expert_transfer="packed" if use_packed else "sync",
            expert_host_cache_bytes=cache_bytes,
            expert_gemm=(expert_gemm if (is_cuda and is_moe) else "native"),
        )
    controller: Optional[AsyncExpertController] = None
    try:
        if is_moe and spec["use_controller"] and expert_loader != "python_sync":
            if not is_cuda and storage == "nf4":
                cpu_packed_control(
                    runtime.expert_pool,
                    runtime.expert_readers[0],
                    int(host_cache_gib * (1024**3)) if host_cache_gib > 0 else 0,
                )
            eff_workers = int(spec.get("override_io_workers", io_workers))
            eff_ring = int(spec.get("override_ring_slots", ring_slots))
            eff_cl = bool(spec.get("override_cross_layer", cross_layer_prefetch))
            controller = AsyncExpertController(
                runtime,
                mode=expert_loader,
                cuda_nf4_decode=cuda_nf4_decode and is_cuda,
                io_workers=eff_workers,
                ring_slots=eff_ring,
                cross_layer_prefetch=eff_cl,
            )

        model.train()
        trainables = fixture.trainables(model)
        if lora_state is not None:
            with torch.no_grad():
                for name, param in trainables.items():
                    param.copy_(lora_state[name].to(param.device, dtype=param.dtype))
        else:
            fixture.randomize_adapters(model, seed=67)
        initial_lora = fixture.snapshot(model, gradients=False)

        step_batches = [[b.to(device) for b in sb] for sb in step_batches_cpu]
        cfg = getattr(model.config, "text_config", None) or model.config
        stepper = l2l_step.L2LStep(
            model, runtime, l2l_step.capture_layer_call(model, step_batches[0][0])
        )
        act_codec = spec.get("activation_codec", "bf16")
        fp8_group_size = spec.get("fp8_group_size", None)
        use_oracle = bool(spec.get("use_oracle", False)) and is_moe
        use_mbr = bool(spec["route_preserving_mbr"]) and is_moe
        mb_shapes = [(1, seq_len, int(cfg.hidden_size))] * k
        host_budget: Optional[HostMemoryBudget] = None
        if act_codec == "fp8_e4m3":
            alloc = fp8_activation_allocation(
                n_layers=int(cfg.num_hidden_layers),
                mb_shapes=mb_shapes,
                pin=is_cuda,
                compute_dtype=dtype,
                device_type="cuda" if is_cuda else "cpu",
                group_size=fp8_group_size,
            )
            # Headroom for Oracle host slabs (compact int16 ids + fp32 scores/norms).
            oracle_headroom = 512 * 1024**2
            host_budget = HostMemoryBudget(alloc.host_bytes + oracle_headroom)
            store: Any = FP8ActivationStore(
                host_budget=host_budget,
                n_layers=int(cfg.num_hidden_layers),
                k=k,
                chunk_shape=(1, seq_len, int(cfg.hidden_size)),
                dtype=getattr(torch, dtype),
                device=device,
                pin=is_cuda,
                group_size=fp8_group_size,
            )
            store_host_bytes = store.host_bytes()["owned_host_storage_bytes"]
        else:
            store = ActivationStore(
                n_layers=int(cfg.num_hidden_layers),
                k=k,
                chunk_shape=(1, seq_len, int(cfg.hidden_size)),
                dtype=getattr(torch, dtype),
                device=device,
                pin=is_cuda,
            )
            store_host_bytes = (
                int(cfg.num_hidden_layers) * k * seq_len * int(cfg.hidden_size)
                * (2 if dtype == "bfloat16" else 4)
            )
        oracle: Optional[DeepseekRoutingOracle] = None
        try:
            if use_oracle:
                oracle = DeepseekRoutingOracle(model, host_budget=host_budget, pin=is_cuda)
            opt = torch.optim.AdamW(trainables.values(), lr=lr)
            if is_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            ledger = None
            if track_saved_ledger:
                from tests.test_p3_decoupled_subwindow import SavedStorageLedger

                ledger = SavedStorageLedger()

            effective_head = spec["head_impl"] or head_loss
            loss_map: Dict[str, Any] = {}
            grads_snap: Dict[str, Any] = {}
            step_walls: List[float] = []
            recompute_fetches = 0
            forward_fetches = 0
            bwd_fetches = 0

            torch.manual_seed(101)
            poller = _NvmlPeakPoller(enabled=is_cuda)
            with poller:
                t_all = time.perf_counter()
                for step_i, batches in enumerate(step_batches):
                    opt.zero_grad(set_to_none=True)
                    n_items = l2l_step.label_tokens(batches)
                    head = l2l_sft_head(batches, n_items, impl=effective_head, chunk=ce_chunk)
                    run_kwargs: Dict[str, Any] = dict(
                        head=head,
                        route_preserving_mbr=use_mbr,
                        mbr_window_dequant=spec["mbr_window_dequant"],
                        release_large_pool_during_layers=spec["release_large_pool_during_layers"],
                        expert_controller=controller,
                        routing_oracle=oracle,
                        collect_phase_stats=is_moe,
                    )
                    if use_mbr:
                        run_kwargs["mbr_group_size"] = group_size
                        run_kwargs["mbr_decoupled_subwindow"] = spec["mbr_decoupled_subwindow"]
                        if spec["mbr_decoupled_subwindow"]:
                            run_kwargs["mbr_recompute_subwindow"] = subwindow
                    ts = time.perf_counter()
                    if ledger is not None:
                        with ledger.track():
                            losses = stepper.run(batches, store, **run_kwargs)
                    else:
                        losses = stepper.run(batches, store, **run_kwargs)
                    grads_snap = fixture.snapshot(model, gradients=True)
                    opt.step()
                    if is_cuda:
                        torch.cuda.synchronize(device)
                    step_walls.append(round(time.perf_counter() - ts, 4))
                    for mb_i, val in enumerate(losses):
                        loss_map[f"s{step_i}_mb{mb_i}"] = val.detach().cpu()
                    if stepper.expert_phase_stats:
                        for layer_row in stepper.expert_phase_stats["layers"].values():
                            recompute_fetches += int(
                                layer_row.get("recompute_forward", {}).get("expert_fetches", 0)
                            )
                            forward_fetches += int(
                                layer_row.get("init_forward", {}).get("expert_fetches", 0)
                            )
                            bwd_fetches += int(
                                layer_row.get("backward_dX", {}).get("expert_fetches", 0)
                            )
                wall_s = time.perf_counter() - t_all

            adapter_snap = fixture.snapshot(model, gradients=False)
            total_tokens = int(sum(b.numel() for sb in step_batches_cpu for b in sb))
            oracle_host_bytes = (
                oracle.host_bytes()["owned_host_storage_bytes"] if oracle is not None else 0
            )
            oracle_report = oracle.report() if oracle is not None else None
            max_gate_weight_rel_l2 = 0.0
            if stepper.route_preserving_mbr_stats:
                vals = [
                    float(row.get("gate_weight_rel_l2", 0.0))
                    for row in stepper.route_preserving_mbr_stats["layers"].values()
                ]
                if vals:
                    max_gate_weight_rel_l2 = max(vals)

            metrics = {
                "arm": arm,
                "activation_codec": act_codec,
                "fp8_group_size": fp8_group_size,
                "routing_oracle": use_oracle,
                "max_gate_weight_rel_l2": max_gate_weight_rel_l2,
                "store_host_bytes": int(store_host_bytes),
                "store_host_mib": round(store_host_bytes / (1024**2), 3),
                "oracle_host_bytes": int(oracle_host_bytes),
                "oracle_host_mib": round(oracle_host_bytes / (1024**2), 3),
                "steps": len(step_batches),
                "step_walls_s": step_walls,
                "head_loss": effective_head,
                "expert_loader": expert_loader if spec["use_controller"] else "python_sync",
                "k": k,
                "seq_len": seq_len,
                "tokens": total_tokens,
                "wall_s": round(wall_s, 4),
                "tokens_per_s": round(total_tokens / max(wall_s, 1e-9), 4),
                "cuda_allocated_peak_gib": (
                    round(torch.cuda.max_memory_allocated(device) / (1024**3), 4)
                    if is_cuda
                    else 0.0
                ),
                "cuda_reserved_peak_gib": (
                    round(torch.cuda.max_memory_reserved(device) / (1024**3), 4)
                    if is_cuda
                    else 0.0
                ),
                "nvml_peak_gib": (
                    round(poller.peak_gib, 4) if poller.peak_gib is not None else None
                ),
                "nvml_poll_samples": poller.samples,
                "nvml_used_gib": _nvml_used_gib() if is_cuda else None,
                "rss_peak_gib": round(poller.rss_peak_gib, 4),
                "rss_gib": round(_rss_gib(), 4),
                "saved_storage_peak_bytes": ledger.peak_bytes if ledger is not None else None,
                "saved_storage_peak_mib": (
                    round(ledger.peak_bytes / (1024**2), 3) if ledger is not None else None
                ),
                "expert_fetches_forward": forward_fetches,
                "expert_fetches_recompute": recompute_fetches,
                "expert_fetches_backward": bwd_fetches,
                "window_dequant_stats": stepper.window_dequant_stats,
                "large_pool_release_stats": stepper.large_pool_release_stats,
                "routing_oracle_stats": oracle_report,
                "native_telemetry": controller.native_telemetry() if controller else None,
                "cache_stats": controller.cache_stats() if controller else None,
                "losses_sha256": _tensor_map_sha256(loss_map),
                "gradients_sha256": _tensor_map_sha256(grads_snap),
                "adapter_sha256": _tensor_map_sha256(adapter_snap),
            }
            return metrics, initial_lora, loss_map, grads_snap, adapter_snap
        finally:
            if oracle is not None:
                oracle.close()
            store.close()
    finally:
        if controller is not None:
            controller.close()
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified A-B-B-A benchmark for DeepSeek P3 GO stack (Tracks 1 + 2 + 3)."
    )
    parser.add_argument(
        "--case",
        choices=["tiny", "tiny_qwen38", "kanana", "qwen38_27b", "custom"],
        default="tiny",
    )
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--work", type=str, required=True)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--storage", choices=["nf4", "dense"], default="nf4")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--subwindow", type=int, default=1)
    parser.add_argument("--git-sha", type=str, default="")
    parser.add_argument("--expert-gemm", choices=["native", "batch_invariant"], default="native")
    parser.add_argument(
        "--expert-loader",
        choices=["python_sync", "cpp_sync", "cpp_async_2slot"],
        default="cpp_async_2slot",
    )
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--ring-slots", type=int, default=8)
    parser.add_argument("--cross-layer-prefetch", action="store_true")
    parser.add_argument("--cuda-nf4-decode", action="store_true")
    parser.add_argument("--head-loss", choices=["full", "chunked", "cce"], default="chunked")
    parser.add_argument("--ce-chunk", type=int, default=256)
    parser.add_argument("--host-cache-gib", type=float, default=0.0)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["std_mbr", "unified_go", "unified_go", "std_mbr"],
    )
    parser.add_argument("--track-saved-ledger", action="store_true")
    parser.add_argument("--save-ref", type=str, default="")
    parser.add_argument("--compare-ref", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    import torch
    from rich.console import Console

    from soup_cli.utils.kimi_k2 import load_stream_config

    work = Path(args.work)
    weights, shards = _prepare_shards(
        work,
        case=args.case,
        model_dir=args.model_dir,
        dtype=args.dtype,
        storage=args.storage,
        device=args.device,
        prepared=args.prepared,
    )
    cfg = load_stream_config(str(weights), False)
    cfg = getattr(cfg, "text_config", None) or cfg
    gen = torch.Generator().manual_seed(2026)
    step_batches_cpu = [
        [
            torch.randint(0, int(cfg.vocab_size), (1, args.seq_len), generator=gen)
            for _ in range(args.k)
        ]
        for _ in range(max(1, args.steps))
    ]
    group_size = args.k if args.group_size <= 0 else args.group_size

    initial_lora: Optional[Dict[str, Any]] = None
    ref_losses: Optional[Dict[str, Any]] = None
    ref_grads: Optional[Dict[str, Any]] = None
    ref_adapter: Optional[Dict[str, Any]] = None
    if args.compare_ref:
        ref_blob = torch.load(args.compare_ref, map_location="cpu", weights_only=False)
        initial_lora = ref_blob["initial_lora"]
        ref_losses = ref_blob["losses"]
        ref_grads = ref_blob["gradients"]
        ref_adapter = ref_blob["adapter"]
    runs: List[Dict[str, Any]] = []

    console = Console()
    for run_idx, arm in enumerate(args.arms):
        try:
            metrics, lora_snap, loss_map, grads_snap, adapter_snap = _run_arm(
                arm,
                weights=weights,
                shards=shards,
                storage=args.storage,
                k=args.k,
                seq_len=args.seq_len,
                group_size=group_size,
                subwindow=args.subwindow,
                device=args.device,
                dtype=args.dtype,
                expert_gemm=args.expert_gemm,
                expert_loader=args.expert_loader,
                io_workers=args.io_workers,
                ring_slots=args.ring_slots,
                cross_layer_prefetch=args.cross_layer_prefetch,
                cuda_nf4_decode=args.cuda_nf4_decode,
                head_loss=args.head_loss,
                ce_chunk=args.ce_chunk,
                host_cache_gib=args.host_cache_gib,
                lora_state=initial_lora,
                step_batches_cpu=step_batches_cpu,
                track_saved_ledger=args.track_saved_ledger,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower():
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                err_row = {
                    "run_index": run_idx,
                    "arm": arm,
                    "status": "oom",
                    "error": msg[:300],
                    "cuda_allocated_peak_gib": (
                        round(torch.cuda.max_memory_allocated(args.device) / (1024**3), 4)
                        if args.device.startswith("cuda")
                        else 0.0
                    ),
                    "nvml_used_gib": _nvml_used_gib() if args.device.startswith("cuda") else None,
                }
                runs.append(err_row)
                console.print(f"[{run_idx}] arm={arm} [red]OOM[/red]: {err_row['error']}")
                continue
            if "route slots" in msg.lower() or "routing" in msg.lower():
                err_row = {
                    "run_index": run_idx,
                    "arm": arm,
                    "status": "refused",
                    "error": msg[:300],
                }
                runs.append(err_row)
                console.print(f"[{run_idx}] arm={arm} [yellow]REFUSED[/yellow]: {err_row['error']}")
                continue
            raise
        if ref_grads is None:
            initial_lora = lora_snap
            ref_losses, ref_grads, ref_adapter = loss_map, grads_snap, adapter_snap
            metrics["vs_first_arm"] = {"reference": True}
            if args.save_ref:
                ref_path = Path(args.save_ref)
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "initial_lora": initial_lora,
                        "losses": ref_losses,
                        "gradients": ref_grads,
                        "adapter": ref_adapter,
                    },
                    str(ref_path),
                )
        else:
            metrics["vs_first_arm"] = {
                "losses": _compare_maps(ref_losses, loss_map),
                "gradients": _compare_maps(ref_grads, grads_snap),
                "adapter": _compare_maps(ref_adapter, adapter_snap),
            }
        metrics["run_index"] = run_idx
        metrics["status"] = "ok"
        runs.append(metrics)
        grad_cmp = metrics["vs_first_arm"].get("gradients", {})
        grad_rel = grad_cmp.get("relative_l2", 0.0)
        grad_cos = grad_cmp.get("min_cosine", 1.0)
        console.print(
            f"[{run_idx}] arm={arm} codec={metrics['activation_codec']} "
            f"oracle={metrics['routing_oracle']} steps={metrics['steps']} "
            f"wall={metrics['wall_s']:.3f}s tok/s={metrics['tokens_per_s']:.2f} "
            f"store={metrics['store_host_mib']:.1f}MiB "
            f"oracle_mem={metrics['oracle_host_mib']:.1f}MiB "
            f"rss_pk={metrics['rss_peak_gib']:.2f}GiB "
            f"alloc={metrics['cuda_allocated_peak_gib']:.3f}GiB "
            f"nvml_pk={metrics['nvml_peak_gib']}GiB "
            f"grad_rel={grad_rel:.4f} min_cos={grad_cos:.5f}"
        )

    payload = {
        "git_sha": args.git_sha or _git_sha(),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "case": args.case,
        "dtype": args.dtype,
        "storage": args.storage,
        "expert_gemm": args.expert_gemm,
        "expert_loader": args.expert_loader,
        "io_workers": args.io_workers,
        "ring_slots": args.ring_slots,
        "cross_layer_prefetch": args.cross_layer_prefetch,
        "cuda_nf4_decode": args.cuda_nf4_decode,
        "head_loss": args.head_loss,
        "ce_chunk": args.ce_chunk,
        "k": args.k,
        "seq_len": args.seq_len,
        "group_size": group_size,
        "subwindow": args.subwindow,
        "runs": runs,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Saved unified A-B-B-A benchmark to {out_path}[/green]")


if __name__ == "__main__":
    main()
