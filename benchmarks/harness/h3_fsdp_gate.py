"""Fail-closed accounting for the H3 pilot. No CUDA claims follow from CPU tests."""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any


def accumulation(global_batch: int, micro_batch: int, world_size: int) -> int:
    """Require exact global batch, rather than rounding accumulation down."""
    if any(type(value) is not int or value < 1
           for value in (global_batch, micro_batch, world_size)):
        raise ValueError("batch sizes and world size must be positive integers")
    divisor = micro_batch * world_size
    if global_batch % divisor:
        raise ValueError("global batch must be divisible by micro batch * world size")
    return global_batch // divisor


def parameter_local_numel(parameter: Any) -> int:
    """DTensor.numel is GLOBAL; to_local().numel reads the physical shard."""
    local = parameter.to_local() if callable(getattr(parameter, "to_local", None)) else parameter
    return int(local.numel())


def assert_shards(counts: list[int], total: int, fsdp_units: list[int]) -> None:
    """FSDP1 use_orig_params omits padding and permits uneven original views."""
    if len(counts) < 2 or len(fsdp_units) != len(counts) or not all(fsdp_units):
        raise ValueError("multi-rank FSDP wrappers required")
    if sum(counts) != total or any(count <= 0 or count >= total for count in counts):
        raise ValueError(f"resident shard accounting disagrees: {counts}, total={total}")


def assert_fsdp2_shards(rank_rows: list[dict], expected: dict, aliases: list[list[str]]) -> dict:
    """Validate actual local DTensor shards per parameter, including tied aliases.

    Wrapping can turn a shared embed/head into two separately sharded parameters.
    Account for that duplication explicitly; never divide the old unique total.
    """
    if len(rank_rows) < 2 or not rank_rows[0]:
        raise ValueError("missing FSDP2 rank parameter readings")
    names = set(rank_rows[0])
    if any(set(row) != names for row in rank_rows) or not names <= set(expected):
        raise ValueError("rank parameter names disagree with the pre-wrap model")
    if not aliases or any(not set(group) & names for group in aliases):
        raise ValueError("a pre-wrap parameter disappeared")
    if set(name for group in aliases for name in group) != set(expected):
        raise ValueError("incomplete pre-wrap alias inventory")
    for name in names:
        rows = [rank[name] for rank in rank_rows]
        if any(row["global_shape"] != expected[name] or not row["sharded"]
               or row["local_numel"] < 0 for row in rows):
            raise ValueError(f"not an actual shard of the expected parameter: {name}")
        if len({row["dtype"] for row in rows}) != 1:
            raise ValueError(f"rank dtypes differ: {name}")
        if sum(row["local_numel"] for row in rows) != math.prod(expected[name]):
            raise ValueError(f"local shard sizes do not reconstruct parameter: {name}")
    before = sum(math.prod(expected[group[0]]) for group in aliases)
    after = sum(math.prod(expected[name]) for name in names)
    return {"prewrap_unique_parameters": before, "postwrap_registered_parameters": after,
            "extra_alias_parameters": after - before,
            "local_parameters_per_rank": [sum(row["local_numel"] for row in rank.values())
                                          for rank in rank_rows]}


def assert_data_window(rank_rows: list[list[str]], global_batch: int) -> None:
    rows = [row for rank in rank_rows for row in rank]
    if len(rows) != global_batch:
        raise ValueError(f"actual global batch {len(rows)} != {global_batch}")
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate samples across ranks or within accumulation window")


def assert_forward_precision(dtypes: dict[str, str], autocast_modes: dict[str, bool],
                             expected_tensors: int, expected_autocast: bool) -> None:
    """Master dtype and requested BF16 do not establish the actual forward policy."""
    if len(dtypes) != expected_tensors or set(dtypes.values()) != {"torch.float32"}:
        raise ValueError("pilot requires all materialized LoRA parameters to stay FP32")
    if set(autocast_modes) != set(dtypes) or set(autocast_modes.values()) != {expected_autocast}:
        raise ValueError("actual forward autocast differs from the pilot's declared precision")


def throughput(rank_seconds: list[list[float]], rank_tokens: list[int]) -> dict:
    """Aggregate input tokens / slowest rank's sum of synchronized step durations."""
    if not rank_seconds or len(rank_tokens) != len(rank_seconds):
        raise ValueError("missing rank measurements")
    steps = len(rank_seconds[0])
    if (not steps or any(len(times) != steps for times in rank_seconds)
            or any(not math.isfinite(t) or t <= 0 for times in rank_seconds for t in times)
            or any(type(t) is not int or t <= 0 for t in rank_tokens)):
        raise ValueError("invalid or incomplete rank measurements")
    elapsed = max(sum(times) for times in rank_seconds)
    return {"tokens_per_second": sum(rank_tokens) / elapsed,
            "seconds_per_optimizer_step": elapsed / steps,
            "measured_seconds": elapsed, "measured_optimizer_steps": steps,
            "input_tokens": sum(rank_tokens)}


def tensor_digest(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    header = str((tuple(value.shape), str(value.dtype))).encode()
    data = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + data).hexdigest()


def state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(tensor_digest(value).encode())
    return digest.hexdigest()


def adapter_delta(before: dict, after: dict) -> dict:
    import torch

    if not before or before.keys() != after.keys():
        raise ValueError("adapter keys missing or changed")
    if not all(torch.isfinite(value).all().item() for value in after.values()):
        raise ValueError("adapter values must be finite")
    changed = [key for key in before if not torch.equal(before[key], after[key])]
    if not changed:
        raise ValueError("optimizer did not update any adapter tensor")
    return {"changed_tensors": sorted(changed), "tensor_count": len(after)}


def assert_adapter_equal(expected: dict, actual: dict) -> None:
    import torch

    if not expected or expected.keys() != actual.keys():
        raise ValueError("saved/reloaded adapter keys do not match the trained adapter")
    if any(not torch.equal(expected[key].cpu(), actual[key].cpu()) for key in expected):
        raise ValueError("saved/reloaded adapter values differ")


def scaling(runs: list[dict]) -> dict:
    """Invalid runs are errors; callers must retain them outside the result set."""
    if not runs or any(run.get("status") != "passed" for run in runs):
        raise ValueError("failed or incomplete arms cannot enter scaling results")
    if any(run["identity"] != runs[0]["identity"] for run in runs):
        raise ValueError("model/data/config/hardware identities differ")
    if any(not math.isfinite(run["tokens_per_second"]) or run["tokens_per_second"] <= 0
           or run["world_size"] not in (1, 2) for run in runs):
        raise ValueError("invalid throughput or world size")
    by_world = {size: [r["tokens_per_second"] for r in runs if r["world_size"] == size]
                for size in (1, 2)}
    if not all(by_world.values()):
        raise ValueError("both one-GPU and two-GPU measurements required")
    rates = {size: statistics.median(values) for size, values in by_world.items()}
    speedup = rates[2] / rates[1]
    return {"median_tokens_per_second": rates, "speedup_2": speedup,
            "efficiency_2": speedup / 2, "repeats": {k: len(v) for k, v in by_world.items()},
            "4_gpu": "unmeasured", "8_gpu": "unmeasured"}
