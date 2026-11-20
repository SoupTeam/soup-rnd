"""Physical GPU VRAM & parity check of Full CE vs Chunked CE on DeepSeek-V3 head shape.

Tests real ``LargeLayerBufferPool`` + ``StreamedLargeLayer`` on ``H=7168, V=129280`` BF16
across ``S = 512, 1024, 2048, 4096`` on an 8 GB GPU, recording ``cuda_allocated_peak``,
``cuda_reserved_peak``, true 20 ms polled ``nvml_peak_gib``, wall time, and BF16 gradient parity.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List


def _run_case(seq_len: int, impl: str, chunk: int, device: str) -> Dict[str, Any]:
    import torch

    from benchmarks.harness.p3_unified_go_bench import _NvmlPeakPoller
    from soup_cli.utils.layer_stream_runtime import (
        LargeLayerBufferPool,
        _streamed_large_layer_class,
    )
    from soup_cli.utils.stream_chunked_ce import chunked_causal_lm_loss

    hidden_size, vocab_size = 7168, 129280
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(63)
    weight_cpu = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16) * 0.02
    pool = LargeLayerBufferPool(
        {"lm_head": ((vocab_size, hidden_size), "bfloat16")}, {"lm_head": 0}, device=device
    )
    pool.load_async("lm_head", SimpleNamespace(get=lambda _i, _k, w=weight_cpu: w))
    inner = torch.nn.Linear(hidden_size, vocab_size, bias=False, device="meta").requires_grad_(
        False
    )
    head = _streamed_large_layer_class()(inner, "lm_head", pool)
    hidden = torch.randn(1, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    hidden.requires_grad_(True)
    labels = torch.randint(0, vocab_size, (1, seq_len), device=device)
    n_items = seq_len - 1

    poller = _NvmlPeakPoller(enabled=True)
    try:
        with poller:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if impl == "full":
                logits = head(hidden).float()
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size),
                    labels[:, 1:].reshape(-1),
                    reduction="sum",
                ) / n_items
            else:
                loss = chunked_causal_lm_loss(hidden, head, labels, n_items, chunk=chunk)
            loss.backward()
            torch.cuda.synchronize(device)
            wall_s = time.perf_counter() - t0
        grad_cpu = hidden.grad.detach().cpu().clone()
        return {
            "seq_len": seq_len,
            "impl": impl,
            "chunk": chunk if impl == "chunked" else None,
            "status": "ok",
            "wall_s": round(wall_s, 4),
            "loss": float(loss.detach().cpu().item()),
            "cuda_allocated_peak_gib": round(
                torch.cuda.max_memory_allocated(device) / (1024**3), 4
            ),
            "cuda_reserved_peak_gib": round(
                torch.cuda.max_memory_reserved(device) / (1024**3), 4
            ),
            "nvml_peak_gib": round(poller.peak_gib, 4) if poller.peak_gib is not None else None,
            "_grad": grad_cpu,
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        return {
            "seq_len": seq_len,
            "impl": impl,
            "chunk": chunk if impl == "chunked" else None,
            "status": "oom",
            "error": str(exc)[:260],
            "cuda_allocated_peak_gib": round(
                torch.cuda.max_memory_allocated(device) / (1024**3), 4
            ),
            "cuda_reserved_peak_gib": round(
                torch.cuda.max_memory_reserved(device) / (1024**3), 4
            ),
            "nvml_peak_gib": round(poller.peak_gib, 4) if poller.peak_gib is not None else None,
        }
    finally:
        pool.release()
        del weight_cpu, hidden, labels, head, pool
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--seqs", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from rich.console import Console

    torch.cuda.init()
    console = Console()
    rows: List[Dict[str, Any]] = []
    for seq in args.seqs:
        full_row = _run_case(seq, "full", args.chunk, args.device)
        chunk_row = _run_case(seq, "chunked", args.chunk, args.device)
        if full_row["status"] == "ok" and chunk_row["status"] == "ok":
            g_full = full_row.pop("_grad")
            g_chunk = chunk_row.pop("_grad")
            chunk_row["grad_bit_equal_to_full"] = bool(torch.equal(g_full, g_chunk))
            chunk_row["grad_max_abs"] = float((g_full.float() - g_chunk.float()).abs().max().item())
            chunk_row["loss_abs_diff"] = abs(full_row["loss"] - chunk_row["loss"])
        else:
            full_row.pop("_grad", None)
            chunk_row.pop("_grad", None)
        rows.extend([full_row, chunk_row])
        f_alloc = full_row["cuda_allocated_peak_gib"]
        c_alloc = chunk_row["cuda_allocated_peak_gib"]
        c_resv = chunk_row["cuda_reserved_peak_gib"]
        console.print(
            f"S={seq:4d} | full={full_row['status']} alloc={f_alloc:.3f}GiB "
            f"nvml={full_row['nvml_peak_gib']}GiB | "
            f"chunked={chunk_row['status']} alloc={c_alloc:.3f}GiB "
            f"resv={c_resv:.3f}GiB nvml={chunk_row['nvml_peak_gib']}GiB "
            f"bit_eq={chunk_row.get('grad_bit_equal_to_full')}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "shape": {"hidden_size": 7168, "vocab_size": 129280, "dtype": "bfloat16"},
                "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
