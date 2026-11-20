"""CUDA checks of ``ExpertBufferPool.async_lease``: same bits, and no device-wide sync per visit.

Needs a card (``pytest -m gpu --no-cov``); the conftest hook skips it otherwise. The packed
transfer with the host cache is the case ``async_lease`` changes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tests.test_b1_expert_forward import _experts, _inputs  # noqa: E402
from tests.test_b13_expert_backward import _assert_exact, _assert_signal, _hit, _run  # noqa: E402
from tests.test_complete_packed_transfer_cuda import _pool, _streamed  # noqa: E402

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def deterministic_kernels() -> Iterator[None]:
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("nested", [False, True])
def test_cuda_async_lease_equals_the_default_lease_bit_for_bit_and_skips_the_device_sync(
    tmp_path: Path, nested: bool
) -> None:
    pytest.importorskip("bitsandbytes")
    import torch

    module, _ = _experts()
    module = module.to(device=DEVICE, dtype=torch.bfloat16)
    args = _inputs(device=DEVICE, dtype=torch.bfloat16, count=24)
    results, syncs, pools = {}, {}, []
    real_sync = torch.cuda.synchronize
    try:
        for name, async_lease in (("default", False), ("async", True)):
            directory = tmp_path / name
            directory.mkdir()
            reader, pool = _pool(
                directory, module, nested=nested, transfer="packed",
                host_cache_bytes=64 * 2**20,
            )
            pools.append((reader, pool))
            pool.async_lease = async_lease
            counter = []
            torch.cuda.synchronize = lambda *a, **k: (counter.append(1), real_sync(*a, **k))[1]
            try:
                results[name] = _run(_streamed(module, reader, pool), args)
                results[name + "-again"] = _run(_streamed(module, reader, pool), args)
            finally:
                torch.cuda.synchronize = real_sync
            syncs[name] = len(counter)
        _assert_signal(results["default"])
        _assert_exact(results["default"], results["async"])
        _assert_exact(results["default"], results["async-again"])   # cache hits, same bits
        visits = 4 * len(_hit(args))                  # forward and backward lease, two steps
        assert syncs["default"] >= visits             # one device sync per lease at least
        assert syncs["async"] < visits // 2           # none per lease, the harness syncs a few
    finally:
        torch.cuda.synchronize = real_sync
        for reader, pool in pools:
            reader.close()
            pool.close()
