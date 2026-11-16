"""CUDA checks of the packed NF4 expert transfer and the prepacked host cache.

These need a card and were NOT run when they were written (the machine had none). They state
what a card run must show: packed transfer and the host cache change how the stored bytes reach
the arena, never the weights, so output and gradients equal the default (``sync``) pool bit for
bit, one H2D copy serves each load, and a host-cache hit reads nothing from the experts file.
Run with ``pytest -m gpu --no-cov tests/test_complete_packed_transfer_cuda.py``. The conftest
hook skips the tests without a card.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tests.test_b1_expert_forward import _experts, _inputs  # noqa: E402
from tests.test_b13_expert_backward import _assert_exact, _assert_signal, _hit, _run  # noqa: E402

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
_STORAGE = {"nf4": dict(nested=False), "nf4-nested": dict(nested=True)}


@pytest.fixture(autouse=True)
def deterministic_kernels() -> Iterator[None]:
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _pool(tmp_path: Path, module: Any, *, nested: bool, **pool_kwargs: Any) -> Any:
    """The B1 store and pool of ``test_b1_expert_forward._store`` with extra pool options."""
    from soup_cli.utils.expert_shard import ExpertShardReader, write_expert_shard
    from soup_cli.utils.expert_stream import ExpertBufferPool, plan_expert_buffer
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    path = str(tmp_path / "experts.safetensors")
    info = write_expert_shard(
        path,
        ((gate.detach(), down.detach()) for gate, down in zip(
            module.gate_up_proj, module.down_proj
        )),
        quantise=True, double_quant=nested, device=DEVICE,
    )
    dtype = str(module.gate_up_proj.dtype).removeprefix("torch.")
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs=info.quant_specs,
        dtype=dtype, direct=False,
    )
    codes = {NF4_CODE_KEY: info.codes[0], NF4_NESTED_CODE_KEY: info.codes[1]}
    transfer = pool_kwargs.get("transfer", "sync")
    plan = plan_expert_buffer(reader, dtype=dtype, transfer=transfer)
    pool = ExpertBufferPool(
        reader, dtype=dtype, device=DEVICE, weight_budget_bytes=plan.weight_bytes + 32 * 2**20,
        codes=codes, **pool_kwargs,
    )
    return reader, pool


def _streamed(module: Any, reader: Any, pool: Any) -> Any:
    from soup_cli.utils.expert_stream import streamed_experts_class

    return streamed_experts_class()(reader, pool, module.act_fn, 0)


@pytest.mark.parametrize("storage", _STORAGE)
def test_cuda_packed_transfer_equals_the_default_pool_and_copies_once_per_load(
    tmp_path: Path, storage: str
) -> None:
    pytest.importorskip("bitsandbytes")
    import torch

    module, _ = _experts()
    module = module.to(device=DEVICE, dtype=torch.bfloat16)
    args = _inputs(device=DEVICE, dtype=torch.bfloat16, count=24)
    results, pools = {}, []
    try:
        for transfer in ("sync", "packed"):
            directory = tmp_path / transfer
            directory.mkdir()
            reader, pool = _pool(directory, module, transfer=transfer, **_STORAGE[storage])
            pools.append((reader, pool))
            results[transfer] = _run(_streamed(module, reader, pool), args)
        _assert_signal(results["sync"])
        _assert_exact(results["sync"], results["packed"])
        reader, pool = pools[1]
        assert pool.fetches == reader.fetches == 2 * len(_hit(args))
        stats = pool.stats()
        assert stats["transfer"] == "packed" and stats["h2d_copies"] == pool.fetches
        assert pool.cuda_allocation_delta <= pool.weight_budget_bytes
    finally:
        for reader, pool in pools:
            reader.close()
            pool.close()


def test_cuda_a_host_cache_hit_reads_nothing_and_changes_nothing(tmp_path: Path) -> None:
    pytest.importorskip("bitsandbytes")
    import torch

    module, _ = _experts()
    module = module.to(device=DEVICE, dtype=torch.bfloat16)
    args = _inputs(device=DEVICE, dtype=torch.bfloat16, count=24)
    reader, pool = _pool(
        tmp_path, module, nested=True, transfer="packed", host_cache_bytes=64 * 2**20
    )
    try:
        experts = _streamed(module, reader, pool)
        first = _run(experts, args)
        reads_after_first = reader.fetches
        assert reads_after_first == len(_hit(args)) * 2 - pool.stats()["host_cache"]["hits"]
        second = _run(experts, args)
        # every expert of the second step comes from the cache: no new disk read
        assert reader.fetches == reads_after_first
        _assert_exact(first, second)
        assert pool.stats()["host_cache"]["rejected"] == 0
    finally:
        reader.close()
        pool.close()
