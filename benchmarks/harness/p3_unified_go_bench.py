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
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

KANANA_REPO = "kakaocorp/kanana-2-30b-a3b-base-2601"
KANANA_REV = "0d31365acf656e5f9a593cacc4913e9c0d36cb1c"
KANANA_TARGETS = ("q_b_proj", "kv_b_proj", "o_proj")

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
    import torch

    max_abs = 0.0
    sq_diff = 0.0
    sq_ref = 0.0
    bit_equal = True
    for name in sorted(ref):
        a, b = ref[name], cand[name]
        if a is None and b is None:
            continue
        if a is None or b is None:
            return {"bit_equal": False, "max_abs": float("inf"), "relative_l2": float("inf")}
        if not torch.equal(a, b):
            bit_equal = False
        af = a.float()
        bf = b.float()
        delta = af - bf
        cur_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        if cur_max > max_abs:
            max_abs = cur_max
        sq_diff += float((delta * delta).sum().item())
        sq_ref += float((af * af).sum().item())
    rel_l2 = (sq_diff**0.5) / max(sq_ref**0.5, 1e-30)
    return {"bit_equal": bit_equal, "max_abs": max_abs, "relative_l2": rel_l2}


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
    from benchmarks.harness.b13_moe_fixture import moe_config, moe_weights_dir
    from soup_cli.utils.layer_shard import shard_checkpoint
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
    elif case == "kanana":
        weights = work / "weights"
        if not (weights / "config.json").exists():
            if prepared:
                raise FileNotFoundError(f"--prepared set, but {weights}/config.json is missing")
            from huggingface_hub import snapshot_download

            snapshot_download(
                KANANA_REPO,
                revision=KANANA_REV,
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
        quant = "nf4" if storage == "nf4" else None
        skeleton = build_meta_skeleton(str(weights), dtype=dtype, quant=quant, double_quant=False)
        layout = shard_layout_of(skeleton)
        suffixes = quantised_layer_suffixes(skeleton) if quant == "nf4" else None
        del skeleton
        shard_checkpoint(
            str(weights),
            str(shards),
            dtype=dtype,
            arch="deepseek_v3",
            quant=quant,
            double_quant=False,
            quant_device=device,
            expert_layout="per_expert",
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
    batches_cpu: Sequence[Any],
    track_saved_ledger: bool,
    lr: float = 1e-4,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    import torch
    from peft import LoraConfig, TaskType

    from benchmarks.harness import b13_moe_fixture as fixture
    from soup_cli.utils import l2l_step
    from soup_cli.utils.async_expert_controller import AsyncExpertController, cpu_packed_control
    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.stream_chunked_ce import l2l_sft_head

    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_SPECS)}")
    spec = ARM_SPECS[arm]

    torch.use_deterministic_algorithms(True)
    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(71)

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(KANANA_TARGETS),
    )
    use_packed = is_cuda and storage == "nf4"
    cache_bytes = int(host_cache_gib * (1024**3)) if use_packed else 0
    index = read_shard_index(str(shards))
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
        two_read_experts=True,
        grouped_expert_routes=True,
        expert_transfer="packed" if use_packed else "sync",
        expert_host_cache_bytes=cache_bytes,
        expert_gemm=expert_gemm if is_cuda else "native",
    )
    controller: Optional[AsyncExpertController] = None
    try:
        if spec["use_controller"] and expert_loader != "python_sync":
            if not is_cuda and storage == "nf4":
                cpu_packed_control(
                    runtime.expert_pool,
                    runtime.expert_readers[0],
                    int(host_cache_gib * (1024**3)) if host_cache_gib > 0 else 0,
                )
            controller = AsyncExpertController(
                runtime,
                mode=expert_loader,
                cuda_nf4_decode=cuda_nf4_decode and is_cuda,
                io_workers=io_workers,
                ring_slots=ring_slots,
                cross_layer_prefetch=cross_layer_prefetch,
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

        batches = [b.to(device) for b in batches_cpu]
        cfg = model.config
        stepper = l2l_step.L2LStep(
            model, runtime, l2l_step.capture_layer_call(model, batches[0])
        )
        store = ActivationStore(
            n_layers=cfg.num_hidden_layers,
            k=k,
            chunk_shape=(1, seq_len, cfg.hidden_size),
            dtype=getattr(torch, dtype),
            device=device,
            pin=is_cuda,
        )
        try:
            opt = torch.optim.AdamW(trainables.values(), lr=lr)
            opt.zero_grad(set_to_none=True)
            if is_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            ledger = None
            if track_saved_ledger:
                from tests.test_p3_decoupled_subwindow import SavedStorageLedger

                ledger = SavedStorageLedger()

            effective_head = spec["head_impl"] or head_loss
            n_items = l2l_step.label_tokens(batches)
            head = l2l_sft_head(batches, n_items, impl=effective_head, chunk=ce_chunk)

            run_kwargs: Dict[str, Any] = dict(
                head=head,
                route_preserving_mbr=spec["route_preserving_mbr"],
                mbr_window_dequant=spec["mbr_window_dequant"],
                release_large_pool_during_layers=spec["release_large_pool_during_layers"],
                expert_controller=controller,
                collect_phase_stats=True,
            )
            if spec["route_preserving_mbr"]:
                run_kwargs["mbr_group_size"] = group_size
                run_kwargs["mbr_decoupled_subwindow"] = spec["mbr_decoupled_subwindow"]
                if spec["mbr_decoupled_subwindow"]:
                    run_kwargs["mbr_recompute_subwindow"] = subwindow

            torch.manual_seed(101)
            t0 = time.perf_counter()
            if ledger is not None:
                with ledger.track():
                    losses = stepper.run(batches, store, **run_kwargs)
            else:
                losses = stepper.run(batches, store, **run_kwargs)
            grads_snap = fixture.snapshot(model, gradients=True)
            opt.step()
            if is_cuda:
                torch.cuda.synchronize(device)
            wall_s = time.perf_counter() - t0

            adapter_snap = fixture.snapshot(model, gradients=False)
            loss_map = {f"mb_{i}": v.detach().cpu() for i, v in enumerate(losses)}
            total_tokens = int(sum(b.numel() for b in batches_cpu))
            recompute_fetches = 0
            forward_fetches = 0
            bwd_fetches = 0
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

            metrics = {
                "arm": arm,
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
                "nvml_used_gib": _nvml_used_gib() if is_cuda else None,
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
                "native_telemetry": controller.native_telemetry() if controller else None,
                "cache_stats": controller.cache_stats() if controller else None,
                "losses_sha256": _tensor_map_sha256(loss_map),
                "gradients_sha256": _tensor_map_sha256(grads_snap),
                "adapter_sha256": _tensor_map_sha256(adapter_snap),
            }
            return metrics, initial_lora, loss_map, grads_snap, adapter_snap
        finally:
            store.close()
    finally:
        if controller is not None:
            controller.close()
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified A-B-B-A benchmark for DeepSeek P3 GO stack (Tracks 1 + 2 + 3)."
    )
    parser.add_argument("--case", choices=["tiny", "kanana", "custom"], default="tiny")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--work", type=str, required=True)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--storage", choices=["nf4", "dense"], default="nf4")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--subwindow", type=int, default=1)
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
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    import torch
    from rich.console import Console
    from transformers import AutoConfig

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
    cfg = AutoConfig.from_pretrained(str(weights))
    gen = torch.Generator().manual_seed(2026)
    batches_cpu = [
        torch.randint(0, int(cfg.vocab_size), (1, args.seq_len), generator=gen)
        for _ in range(args.k)
    ]
    group_size = args.k if args.group_size <= 0 else args.group_size

    initial_lora: Optional[Dict[str, Any]] = None
    ref_losses: Optional[Dict[str, Any]] = None
    ref_grads: Optional[Dict[str, Any]] = None
    ref_adapter: Optional[Dict[str, Any]] = None
    runs: List[Dict[str, Any]] = []

    console = Console()
    for run_idx, arm in enumerate(args.arms):
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
            batches_cpu=batches_cpu,
            track_saved_ledger=args.track_saved_ledger,
        )
        if initial_lora is None:
            initial_lora = lora_snap
            ref_losses, ref_grads, ref_adapter = loss_map, grads_snap, adapter_snap
            metrics["vs_first_arm"] = {"reference": True}
        else:
            metrics["vs_first_arm"] = {
                "losses": _compare_maps(ref_losses, loss_map),
                "gradients": _compare_maps(ref_grads, grads_snap),
                "adapter": _compare_maps(ref_adapter, adapter_snap),
            }
        metrics["run_index"] = run_idx
        runs.append(metrics)
        grad_rel = metrics["vs_first_arm"].get("gradients", {}).get("relative_l2", 0.0)
        console.print(
            f"[{run_idx}] arm={arm} head={metrics['head_loss']} loader={metrics['expert_loader']} "
            f"wall={metrics['wall_s']:.3f}s tok/s={metrics['tokens_per_s']:.2f} "
            f"cuda_alloc={metrics['cuda_allocated_peak_gib']:.3f}GiB "
            f"saved_peak={metrics['saved_storage_peak_mib']}MiB "
            f"recomp_reads={metrics['expert_fetches_recompute']} "
            f"grad_rel_l2={grad_rel:.2e}"
        )

    payload = {
        "git_sha": _git_sha(),
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
