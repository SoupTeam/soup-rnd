"""D3 development probe: grouped NF4 decode against the existing bnb decoder.

No full-expert layer, routing, GEMM or backward integration. No throughput or
capacity verdict. Original masked kernel; inputs retain bnb's packed bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _build_decoder() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def decode(
        packed_ptrs, stats_ptrs, nested_ptrs, offset_ptrs, output_ptrs,
        code_ptr, nested_code_ptr,
        elements: tl.constexpr, quant_block: tl.constexpr,
        nested_block: tl.constexpr, nested: tl.constexpr, block: tl.constexpr,
    ):
        expert = tl.program_id(1)
        index = tl.program_id(0) * block + tl.arange(0, block)
        valid = index < elements
        packed = tl.load(packed_ptrs + expert).to(tl.pointer_type(tl.uint8))
        output = tl.load(output_ptrs + expert).to(tl.pointer_type(tl.bfloat16))
        value = tl.load(packed + index // 2, valid, other=0).to(tl.int32)
        nibble = tl.where(index % 2 == 0, value >> 4, value & 15)
        code = tl.load(code_ptr + nibble)
        stat_index = index // quant_block
        if nested:
            stats = tl.load(stats_ptrs + expert).to(tl.pointer_type(tl.uint8))
            nested_stats = tl.load(nested_ptrs + expert).to(tl.pointer_type(tl.float32))
            offset = tl.load(offset_ptrs + expert).to(tl.pointer_type(tl.float32))
            stat_code = tl.load(stats + stat_index, valid, other=0).to(tl.int32)
            multiplier = tl.load(nested_stats + stat_index // nested_block, valid, other=0)
            scale = tl.load(nested_code_ptr + stat_code) * multiplier
            scale = scale + tl.load(offset)
        else:
            stats = tl.load(stats_ptrs + expert).to(tl.pointer_type(tl.float32))
            scale = tl.load(stats + stat_index, valid, other=0)
        tl.store(output + index, (code * scale).to(tl.bfloat16), valid)

    return decode


def decode_plan(packed: list, states: list, *, weight_budget_bytes: int) -> dict:
    """Validate metadata and a logical storage bound before output/metadata allocation."""
    import torch

    if not packed or len(packed) != len(states) or len(packed) > 4:
        raise ValueError("decode probe requires one to four matching packed/state pairs")
    if type(weight_budget_bytes) is not int or weight_budget_bytes < 1:
        raise ValueError("weight budget must be a positive integer")
    first = states[0]
    if (not first.shape or any(type(dim) is not int or dim < 1 for dim in first.shape)
            or first.blocksize not in (32, 64, 128, 256, 512, 1024, 2048, 4096)):
        raise ValueError("invalid NF4 shape/blocksize")
    device = packed[0].device
    elements = math.prod(first.shape)
    stat_count = -(-elements // first.blocksize)
    owned = []

    def check(tensor: Any, dtype: Any, count: int) -> None:
        if (tensor.device != device or tensor.dtype != dtype or tensor.numel() != count
                or not tensor.is_contiguous()):
            raise ValueError("incompatible NF4 tensor size, dtype, device or strides")
        owned.append(tensor)

    for value, state in zip(packed, states):
        if (state.shape != first.shape or state.dtype != torch.bfloat16
                or state.blocksize != first.blocksize or state.nested != first.nested
                or state.quant_type != "nf4"):
            raise ValueError("incompatible grouped NF4 input")
        check(value, torch.uint8, -(-elements // 2))
        check(state.code, torch.float32, 16)
        check(state.absmax, torch.uint8 if first.nested else torch.float32, stat_count)
        if not torch.equal(state.code, first.code):
            raise ValueError("incompatible NF4 code tables")
        if first.nested:
            block = state.state2.blocksize
            if (type(block) is not int or block < 1 or block > 4096 or block & (block - 1)
                    or block != first.state2.blocksize):
                raise ValueError("incompatible nested blocksize")
            check(state.state2.absmax, torch.float32, -(-stat_count // block))
            check(state.state2.code, torch.float32, 256)
            check(state.offset, torch.float32, 1)
            if not torch.equal(state.state2.code, first.state2.code):
                raise ValueError("incompatible nested code tables")
    unique = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
              for tensor in owned}
    output_bytes = len(packed) * elements * 2
    metadata_bytes = 5 * 512  # Conservative allocator-bin-sized pointer tables.
    total = sum(unique.values()) + output_bytes + metadata_bytes
    if total > weight_budget_bytes:
        raise ValueError(f"group decode needs {total} bytes, above budget {weight_budget_bytes}")
    return {
        "group_size": len(packed), "elements_per_expert": elements,
        "existing_input_storage_bytes": sum(unique.values()), "output_bytes": output_bytes,
        "metadata_bytes_upper_bound": metadata_bytes, "logical_weight_bytes": total,
        "weight_budget_bytes": weight_budget_bytes,
        "scope": "logical decode tensors; not whole-process allocator/context/VRAM bound",
    }


def prepare_decode(
    packed: list, states: list, *, weight_budget_bytes: int = 128 * 2**20,
) -> tuple[list, Any]:
    """Prepare only this bounded group; return outputs and a reusable launch closure."""
    import torch

    plan = decode_plan(packed, states, weight_budget_bytes=weight_budget_bytes)
    first = states[0]
    device = packed[0].device
    if device.type != "cuda":
        raise ValueError("CUDA BF16 NF4 states required")
    import triton

    elements = plan["elements_per_expert"]
    outputs = [torch.empty(first.shape, device=device, dtype=first.dtype) for _ in packed]
    lists = [packed, [state.absmax for state in states]]
    if first.nested:
        lists += [[state.state2.absmax for state in states], [state.offset for state in states]]
    else:
        lists += [packed, packed]  # Unused placeholders; never dereferenced by this specialization.
    lists += [outputs]
    pointers = [torch.tensor([tensor.data_ptr() for tensor in tensors], device=device,
                             dtype=torch.int64) for tensors in lists]
    kernel = _build_decoder()
    nested_code = first.state2.code if first.nested else first.code
    nested_block = first.state2.blocksize if first.nested else 1

    def launch() -> list:
        # Retain source states/tensors: pointer metadata does not own their storage.
        _keep_alive = packed, states
        kernel[(triton.cdiv(elements, 1024), len(packed))](
            *pointers, first.code, nested_code, elements, first.blocksize,
            nested_block, first.nested, 1024, num_warps=4, enable_fp_fusion=False,
        )
        return outputs

    return outputs, launch


def run_probe(folder: Path) -> dict:
    import torch
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Linux CUDA required")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("native BF16 required")
    records = []
    profiles = []
    for nested in (False, True):
        for shape in ((32, 32), (33, 47), (2048, 2048), (2048, 1024)):
            for seed in (11, 29):
                generator = torch.Generator().manual_seed(seed)
                weights = [(torch.randn(shape, generator=generator) * 0.2).to(
                    device="cuda", dtype=torch.bfloat16
                ) for _ in range(4)]
                quantized = [quantize_4bit(value, quant_type="nf4", compress_statistics=nested)
                             for value in weights]
                packed, states = map(list, zip(*quantized))
                expected = [dequantize_4bit(value, quant_state=state)
                            for value, state in quantized]
                outputs, launch = prepare_decode(packed, states)
                launch()
                torch.cuda.synchronize()
                batch = []
                for expert, (want, got) in enumerate(zip(expected, outputs)):
                    batch.append({
                        "nested": nested, "shape": shape, "seed": seed, "expert": expert,
                        "equal": torch.equal(want, got),
                        "finite": bool(torch.isfinite(got).all()),
                        "unequal_elements": int(torch.count_nonzero(want != got)),
                        "max_abs_diff": float((want.float() - got.float()).abs().max()),
                        "expected_sha256": hashlib.sha256(
                            want.cpu().view(torch.uint8).numpy()
                        ).hexdigest(),
                        "observed_sha256": hashlib.sha256(
                            got.cpu().view(torch.uint8).numpy()
                        ).hexdigest(),
                    })
                # Mutate only temporary packed GPU buffers; source checkpoints are untouched.
                for value in packed:
                    value.view(-1)[:1].bitwise_xor_(16)
                try:
                    launch()
                    torch.cuda.synchronize()
                    for row, want, got in zip(batch, expected, outputs):
                        row["negative_differs"] = not torch.equal(want, got)
                finally:
                    for value in packed:
                        value.view(-1)[:1].bitwise_xor_(16)
                if shape == (2048, 2048) and seed == 11:
                    from d3_expert_prepare import kernel_trace_summary

                    for arm in ("bnb-loop", "grouped-decode"):
                        # Separate development profiling, no throughput comparison.
                        with torch.profiler.profile(activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]) as profiler:
                            if arm == "bnb-loop":
                                _result = [dequantize_4bit(value, quant_state=state)
                                           for value, state in quantized]
                            else:
                                launch()
                            torch.cuda.synchronize()
                        path = folder / f"trace-{arm}-nested-{nested}.json"
                        profiler.export_chrome_trace(str(path))
                        profiles.append({
                            "arm": arm, "nested": nested,
                            "trace": kernel_trace_summary(json.loads(path.read_text())),
                            "scope": "decode invocation only; metadata prebuilt, no full MoE",
                        })
                records.extend(batch)
                (folder / "progress.json").write_text(
                    json.dumps(records, indent=2) + "\n", encoding="utf-8",
                )
    return {
        "status": "decode_exact" if all(
            row["equal"] and row["finite"] and row["negative_differs"] for row in records
        )
        else "decode_mismatch", "records": records, "profiles": profiles,
        "scope": "synthetic packed weights, decode only; no complete D3 candidate",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "no_verdict", "d3_accepted": False, "formal_measurement": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        report.update(run_probe(args.out))
    except (Exception, KeyboardInterrupt) as exc:
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path = args.out / "probe.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Console().print(f"{report['status']}: {path}")
    return 0 if report["status"] == "decode_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
