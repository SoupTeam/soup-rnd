"""Complete levers: packed NF4 expert transfer and prepacked host entries (CPU parts).

The H2D copy needs CUDA and is covered by ``test_complete_packed_transfer_cuda.py``. Here: the
byte layout that one coalesced copy relies on, the static host entry cache and its RAM budget,
the pool's refusals on CPU, and the plan the pool allocates from.
"""

from __future__ import annotations

import pytest
import torch


def _tensors() -> dict:
    gen = torch.Generator().manual_seed(11)
    return {
        "weights": torch.randint(0, 255, (101, 1), dtype=torch.uint8, generator=gen),
        "absmax": torch.randn(7, generator=gen),
        "nested_absmax": torch.randint(0, 255, (13,), dtype=torch.uint8, generator=gen),
        "offset": torch.randn((), generator=gen),
    }


def test_packing_layout_aligns_every_tensor_and_keeps_its_bytes_exact() -> None:
    from soup_cli.utils.expert_transfer import pack_into, packing_layout

    tensors = _tensors()
    rows, size = packing_layout(tensors)
    assert [row[0] for row in rows] == list(tensors)
    # every offset is on a 256-byte boundary, and regions do not overlap
    ends = [offset + count for _, offset, count, _, _ in rows]
    assert all(offset % 256 == 0 for _, offset, _, _, _ in rows)
    assert all(end <= row[1] for end, row in zip(ends, rows[1:]))
    assert size >= ends[-1]
    # a scalar sidecar is one four-byte region, not dropped by the alignment
    assert dict((row[0], row[2]) for row in rows)["offset"] == 4
    block = torch.zeros(size, dtype=torch.uint8)
    assert pack_into(block, rows, tensors) == sum(row[2] for row in rows)
    for name, offset, count, dtype, shape in rows:
        view = block[offset : offset + count].view(dtype).view(shape)
        assert torch.equal(view, tensors[name])
    covered = torch.zeros(size, dtype=torch.bool)
    for _, offset, count, _, _ in rows:
        covered[offset : offset + count] = True
    assert not block[~covered].any()


def test_a_stored_tensor_that_does_not_match_its_row_is_refused_not_converted() -> None:
    from soup_cli.utils.expert_transfer import pack_into, packing_layout

    tensors = _tensors()
    rows, size = packing_layout(tensors)
    wrong = dict(tensors, absmax=tensors["absmax"].to(torch.float64))
    with pytest.raises(ValueError, match="absmax"):
        pack_into(torch.zeros(size, dtype=torch.uint8), rows, wrong)
    with pytest.raises(ValueError, match="nonempty"):
        packing_layout({})
    with pytest.raises(ValueError, match="contiguous CPU"):
        packing_layout({"x": torch.zeros(4, 4).t()})


def test_the_host_hotset_charges_whole_blocks_and_never_evicts() -> None:
    from soup_cli.utils.expert_transfer import PackedHostHotset

    cache = PackedHostHotset(300)
    first = torch.arange(200, dtype=torch.uint8)
    assert cache.put("a", first) and not cache.put("a", first)  # a key is admitted once
    assert not cache.put("b", torch.zeros(101, dtype=torch.uint8))  # 200 + 101 > 300
    assert cache.put("c", torch.zeros(100, dtype=torch.uint8))  # exactly fills the budget
    assert not cache.put("d", torch.zeros(1, dtype=torch.uint8))
    first.fill_(0)  # the cache holds an owned copy
    got = cache.get("a")
    assert torch.equal(got, torch.arange(200, dtype=torch.uint8))
    assert cache.get("missing") is None
    assert cache.stats() == {
        "budget_bytes": 300, "payload_bytes": 300, "entries": 2, "hits": 1, "misses": 1,
        "rejected": 3,
    }
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            PackedHostHotset(bad)


def _reader(tmp_path, *, nf4: bool, nested: bool = True):
    from tests.test_b1_expert_forward import _experts, _store

    module, _ = _experts()
    reader, pool = _store(tmp_path, module, nf4=nf4, nested=nested)
    pool.close()
    return reader


def test_the_packed_plan_puts_every_stored_component_in_one_aligned_region_at_the_front(
    tmp_path,
) -> None:
    from soup_cli.utils.expert_stream import plan_expert_buffer

    reader = _reader(tmp_path, nf4=True)
    sync = plan_expert_buffer(reader, dtype="float32")
    packed = plan_expert_buffer(reader, dtype="float32", transfer="packed")
    assert sync.packed_region_bytes == 0 and sync.host_pinned_bytes == 0
    # the region holds the packed weights and statistics (padding between them included)
    assert packed.packed_region_bytes >= packed.packed_and_stats_bytes > 0
    assert packed.host_pinned_bytes == packed.packed_region_bytes
    assert (packed.dense_bytes, packed.packed_and_stats_bytes, packed.code_bytes) == (
        sync.dense_bytes, sync.packed_and_stats_bytes, sync.code_bytes,
    )
    assert packed.weight_bytes >= packed.packed_region_bytes + packed.dense_bytes
    assert packed.weight_bytes % 512 == 0


def test_a_dense_store_has_nothing_to_pack_and_a_bad_mode_is_refused(tmp_path) -> None:
    from soup_cli.utils.expert_stream import plan_expert_buffer

    dense = _reader(tmp_path, nf4=False)
    with pytest.raises(ValueError, match="packed transfer needs NF4"):
        plan_expert_buffer(dense, dtype="float32", transfer="packed")
    with pytest.raises(ValueError, match="transfer must be 'sync' or 'packed'"):
        plan_expert_buffer(dense, dtype="float32", transfer="prefetch")


def test_the_pool_refuses_packed_transfer_without_cuda_and_unsupported_combinations(
    tmp_path,
) -> None:
    from soup_cli.utils.expert_stream import ExpertBufferPool, plan_expert_buffer
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    reader = _reader(tmp_path, nf4=True)
    plan = plan_expert_buffer(reader, dtype="float32", transfer="packed")
    codes = {
        NF4_CODE_KEY: torch.linspace(-1, 1, 16),
        NF4_NESTED_CODE_KEY: torch.linspace(-1, 1, 256),
    }
    common = dict(device="cpu", dtype="float32", weight_budget_bytes=plan.weight_bytes, codes=codes)
    with pytest.raises(ValueError, match="packed expert transfer requires CUDA"):
        ExpertBufferPool(reader, transfer="packed", **common)
    with pytest.raises(ValueError, match="host_cache_bytes needs transfer='packed'"):
        ExpertBufferPool(reader, host_cache_bytes=1 << 20, **common)
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="host_cache_bytes must be a non-negative integer"):
            ExpertBufferPool(reader, host_cache_bytes=bad, **common)
    with pytest.raises(ValueError, match="transfer must be 'sync' or 'packed'"):
        ExpertBufferPool(reader, transfer="prefetch", **common)
    pool = ExpertBufferPool(reader, **common)
    try:
        stats = pool.stats()
        assert stats["transfer"] == "sync" and stats["host_cache"] is None
    finally:
        pool.close()
        reader.close()


def _tiny_cache(tmp_path, *, per_expert: bool):
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from tests.test_b2_deepseek_v3_stream import _save_fixture, _tiny_deepseek_v3_config

    config = _tiny_deepseek_v3_config()
    weights = tmp_path / "weights"
    _save_fixture(weights, config)
    arch = stream_arch_of(load_stream_config(str(weights), trust_remote_code=False))
    shards = str(tmp_path / "shards")
    extra = {"expert_layout": "per_expert"} if per_expert else {}
    index = shard_checkpoint(str(weights), shards, dtype="float32", arch=arch, **extra)
    return config, weights, shards, index


@pytest.mark.parametrize("lever", [
    {"expert_transfer": "packed"}, {"expert_host_cache_bytes": 1 << 20},
])
def test_the_transfer_levers_need_a_per_expert_cache(tmp_path, lever: dict) -> None:
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_b2_deepseek_v3_stream import _lora

    config, weights, shards, index = _tiny_cache(tmp_path, per_expert=False)
    name = next(iter(lever))
    with pytest.raises(ValueError, match=f"{name} needs a cache sharded"):
        build_streamed_model(
            model_id=str(weights), shard_dir=shards, index=index, lora_config=_lora(config),
            device="cpu", dtype="float32", pin=False, **lever,
        )


def test_build_streamed_model_hands_the_transfer_to_the_pool_which_refuses_it_on_cpu(
    tmp_path,
) -> None:
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_b2_deepseek_v3_stream import _lora

    config, weights, shards, index = _tiny_cache(tmp_path, per_expert=True)
    with pytest.raises(ValueError, match="packed expert transfer requires CUDA|needs NF4"):
        build_streamed_model(
            model_id=str(weights), shard_dir=shards, index=index, lora_config=_lora(config),
            device="cpu", dtype="float32", pin=False, expert_transfer="packed",
        )
