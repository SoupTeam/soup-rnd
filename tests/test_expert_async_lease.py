"""``ExpertBufferPool.async_lease``: a lease no longer synchronises the whole device when it ends.

By default every lease ends with ``torch.cuda.synchronize``, so the host cannot prepare the next
expert while the card still computes the last one. With ``async_lease`` and the packed transfer the
only host wait left is for the H2D copy that reads the pinned staging region (the region may not be
overwritten before that copy is done); weights, dequantisation and GEMMs stay ordered on their
stream. These tests pin the decision on CPU; the CUDA file pins the bits and the missing syncs.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import pytest

from tests.test_b1_expert_forward import _experts, _store

torch = pytest.importorskip("torch")

if TYPE_CHECKING:
    from pathlib import Path


def pool_with_counter(tmp_path: Path) -> tuple[object, object, list[int]]:
    module, _ = _experts()
    reader, pool = _store(tmp_path, module, nf4=False, nested=False, device="cpu")
    calls: list[int] = []
    pool._synchronize = lambda: calls.append(1)  # type: ignore[method-assign]
    return reader, pool, calls


def test_async_lease_is_off_by_default_and_a_lease_syncs_the_device(tmp_path: Path) -> None:
    from soup_cli.utils.expert_stream import ExpertBufferPool

    assert ExpertBufferPool.async_lease is False
    reader, pool, calls = pool_with_counter(tmp_path)
    try:
        with pool.lease(reader, 0):
            pass
        assert calls == [1]
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize(
    "transfer,device,async_lease,syncs",
    [
        ("packed", "cuda", True, 0),      # the one case that skips the device-wide wait
        ("packed", "cuda", False, 1),
        ("sync", "cuda", True, 1),        # the sync transfer copies blocking, keep the old rule
        ("packed", "cpu", True, 1),
    ],
)
def test_lease_end_skips_the_device_sync_only_for_async_packed_cuda_pools(
    tmp_path: Path, transfer: str, device: str, async_lease: bool, syncs: int
) -> None:
    reader, pool, calls = pool_with_counter(tmp_path)
    try:
        pool.transfer = transfer
        pool.device = types.SimpleNamespace(type=device)
        pool.async_lease = async_lease
        pool._lease_end()
        assert len(calls) == syncs
    finally:
        pool.device = torch.device("cpu")
        reader.close()
        pool.close()


def test_a_failing_lease_still_syncs_the_device_even_when_async(tmp_path: Path) -> None:
    reader, pool, calls = pool_with_counter(tmp_path)
    try:
        pool.async_lease = True
        with pytest.raises(RuntimeError, match="boom"):
            with pool.lease(reader, 0):
                raise RuntimeError("boom")
        assert calls            # the error path keeps the full wait
    finally:
        reader.close()
        pool.close()


def test_the_host_cache_key_resolves_the_path_every_time_by_default_and_once_when_async(
    tmp_path: Path, monkeypatch
) -> None:
    """``os.path.realpath`` is a chain of system calls. By default every load and prefetch hint
    resolves the reader path as before (no change of behaviour for anyone); with ``async_lease`` a
    pool resolves a path once."""
    import os

    reader, pool, _ = pool_with_counter(tmp_path)
    try:
        calls: list[str] = []
        real = os.path.realpath
        monkeypatch.setattr(os.path, "realpath", lambda p, **k: (calls.append(p), real(p, **k))[1])
        assert pool._host_key(reader, 3) == (real(reader.path), 3)
        pool._host_key(reader, 4)
        assert calls.count(reader.path) == 2             # default: resolved on every call
        pool.async_lease = True
        calls.clear()
        assert pool._host_key(reader, 3) == (real(reader.path), 3)
        pool._host_key(reader, 4)
        assert calls.count(reader.path) == 1             # async lease: resolved once
    finally:
        reader.close()
        pool.close()
