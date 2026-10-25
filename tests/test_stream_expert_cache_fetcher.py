"""Expert-tier ownership, read reduction and fail-closed source identities."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


def _reader(tmp_path, name="experts", *, offset=0, nf4=False, device="cpu"):
    import torch

    from soup_cli.utils.expert_stream import ExpertShardReader, write_expert_shard
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    path = str(tmp_path / (name + ".safetensors"))
    generator = torch.Generator().manual_seed(17 + offset)
    pairs = [(torch.randn(32, 32, generator=generator) + offset,
              torch.randn(32, 16, generator=generator) + offset) for _ in range(3)]
    info = write_expert_shard(path, pairs, quantise=nf4, device=device)
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs=info.quant_specs,
        dtype="float32", direct=False,
    )
    codes = ({NF4_CODE_KEY: info.codes[0].to(device),
              NF4_NESTED_CODE_KEY: info.codes[1].to(device)} if nf4 else {})
    return reader, pairs, codes


def _same(want, got):
    import torch

    assert len(want) == len(got) == 2
    for expected, actual in zip(want, got):
        assert torch.equal(expected, actual)
        assert torch.isfinite(actual).all()


@pytest.mark.parametrize("host,device", [(2**20, 0), (0, 2**20), (2**20, 2**20)])
def test_cache_hit_skips_real_read_and_preserves_staging_ownership(tmp_path, host, device):
    from soup_cli.utils.expert_stream import ExpertFetcher

    reader, pairs, _ = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32",
                            host_cache_bytes=host, device_cache_bytes=device)
    try:
        held = fetcher.fetch(reader, 0)
        _same(pairs[1], fetcher.fetch(reader, 1))
        _same(pairs[0], held)
        _same(pairs[0], fetcher.fetch(reader, 0))
        stats = fetcher.cache_stats()
        assert stats["logical_fetches"] == 3 and stats["storage_reads"] == reader.fetches == 2
        assert stats["materializations"] == (2 if device else 3)
        assert stats["host"]["hits"] == (0 if device else 1)
        assert stats["device"]["hits"] == (1 if device else 0)
        if device:
            assert stats["host"]["entries"] == 0
    finally:
        fetcher.close()
        reader.close()


def test_small_device_hotset_and_host_lru_cover_different_experts(tmp_path):
    from soup_cli.utils.expert_stream import ExpertFetcher

    reader, pairs, _ = _reader(tmp_path)
    one = sum(tensor.numel() * tensor.element_size() for tensor in pairs[0])
    fetcher = ExpertFetcher(device="cpu", dtype="float32",
                            host_cache_bytes=one, device_cache_bytes=one)
    try:
        for expert in (0, 1, 0, 1):
            _same(pairs[expert], fetcher.fetch(reader, expert))
        stats = fetcher.cache_stats()
        assert stats["host"]["entries"] == stats["device"]["entries"] == 1
        assert stats["host"]["hits"] == stats["device"]["hits"] == 1
        assert stats["host"]["payload_bytes"] == stats["device"]["payload_bytes"] == one
        assert reader.fetches == 2
    finally:
        fetcher.close()
        reader.close()


@pytest.mark.parametrize("device_cache", [False, True])
def test_equal_expert_ids_from_different_layers_never_share_weights(tmp_path, device_cache):
    from soup_cli.utils.expert_stream import ExpertFetcher

    first, a, _ = _reader(tmp_path, "layer_a")
    second, b, _ = _reader(tmp_path, "layer_b", offset=10)
    fetcher = ExpertFetcher(device="cpu", dtype="float32",
                            host_cache_bytes=0 if device_cache else 2**20,
                            device_cache_bytes=2**20 if device_cache else 0)
    try:
        for _ in range(2):
            _same(a[0], fetcher.fetch(first, 0))
            _same(b[0], fetcher.fetch(second, 0))
        assert first.fetches == second.fetches == 1
    finally:
        fetcher.close()
        first.close()
        second.close()


@pytest.mark.parametrize("expert", [-1, 3, True, 0.0])
def test_invalid_expert_is_refused_even_after_cache_warmup(tmp_path, expert):
    from soup_cli.utils.expert_stream import ExpertFetcher

    reader, _, _ = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", device_cache_bytes=2**20)
    try:
        fetcher.fetch(reader, 0)
        with pytest.raises(IndexError, match="outside"):
            fetcher.fetch(reader, expert)
    finally:
        fetcher.close()
        reader.close()


def test_reader_reopen_identity_check_is_not_bypassed_by_a_warm_hit(tmp_path):
    from soup_cli.utils.expert_stream import ExpertFetcher

    reader, _, _ = _reader(tmp_path)
    replacement, _, _ = _reader(tmp_path, "replacement", offset=10)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", device_cache_bytes=2**20)
    try:
        fetcher.fetch(reader, 0)
        reader.close()
        replacement.close()
        os.replace(replacement.path, reader.path)
        with pytest.raises(ValueError, match="changed on disk"):
            fetcher.fetch(reader, 0)
    finally:
        fetcher.close()
        reader.close()


@pytest.mark.parametrize("device_cache", [False, True])
def test_in_place_rewrite_is_refused_with_an_open_reader_and_warm_cache(tmp_path, device_cache):
    from soup_cli.utils.expert_stream import ExpertFetcher

    reader, _, _ = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32",
                            host_cache_bytes=0 if device_cache else 2**20,
                            device_cache_bytes=2**20 if device_cache else 0)
    try:
        fetcher.fetch(reader, 0)
        before = os.stat(reader.path)
        with open(reader.path, "r+b") as file:
            entry = reader._entries[0][0][1]
            file.seek(entry.start)
            file.write(b"\x00" * entry.nbytes)
        os.utime(reader.path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        with pytest.raises(ValueError, match="changed on disk"):
            fetcher.fetch(reader, 0)
    finally:
        fetcher.close()
        reader.close()


def test_close_releases_cache_and_staging_even_when_runtime_source_close_fails(tmp_path):
    from soup_cli.utils.expert_stream import ExpertFetcher
    from soup_cli.utils.layer_stream_runtime import StreamRuntime

    reader, _, _ = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", host_cache_bytes=2**20)
    fetcher.fetch(reader, 0)
    removed = []

    def broken():
        raise RuntimeError("source close failed")

    runtime = StreamRuntime(
        pool=None, source=SimpleNamespace(close=broken), prefetcher=None, n_layers=1,
        pinned=False, device="cpu", expert_readers=(reader,), expert_fetcher=fetcher,
        hook=SimpleNamespace(remove=lambda: removed.append(True)),
    )
    with pytest.raises(RuntimeError, match="source close failed"):
        runtime.close()
    stats = fetcher.cache_stats()
    assert stats["host"]["payload_bytes"] == stats["device"]["payload_bytes"] == 0
    assert stats["host_staging_bytes"] == 0 and reader._handle is None and removed == [True]
    fetcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        fetcher.fetch(reader, 0)


@pytest.mark.parametrize("host,device", [(1, 0), (0, 1), (1, 1)])
def test_undersized_cache_preserves_exact_uncached_path(tmp_path, host, device):
    from soup_cli.utils.expert_stream import ExpertFetcher

    reader, pairs, _ = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32",
                            host_cache_bytes=host, device_cache_bytes=device)
    try:
        for _ in range(2):
            _same(pairs[0], fetcher.fetch(reader, 0))
        assert reader.fetches == 2
        assert fetcher.cache_stats()["host"]["entries"] == 0
        assert fetcher.cache_stats()["device"]["entries"] == 0
    finally:
        fetcher.close()
        reader.close()


@pytest.mark.parametrize("host,device", [(2**20, 0), (0, 2**20), (2**20, 2**20)])
def test_real_cpu_nf4_cache_preserves_complete_nested_quantization_state(tmp_path, host, device):
    pytest.importorskip("bitsandbytes")
    from soup_cli.utils.expert_stream import ExpertFetcher

    reader, _, codes = _reader(tmp_path, nf4=True)
    baseline = ExpertFetcher(device="cpu", dtype="float32", codes=codes)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", codes=codes,
                            host_cache_bytes=host, device_cache_bytes=device)
    try:
        expected = baseline.fetch(reader, 0)
        for expert in (0, 1, 0):
            actual = fetcher.fetch(reader, expert)
            if expert == 0:
                _same(expected, actual)
        assert fetcher.cache_stats()["storage_reads"] == 2
    finally:
        baseline.close()
        fetcher.close()
        reader.close()


def test_nf4_dequant_dtype_changes_cannot_hit_an_old_dense_entry(tmp_path):
    pytest.importorskip("bitsandbytes")
    from dataclasses import replace

    from soup_cli.utils.expert_stream import GATE_UP, ExpertFetcher

    reader, _, codes = _reader(tmp_path, nf4=True)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", codes=codes,
                            device_cache_bytes=2**20)
    baseline = ExpertFetcher(device="cpu", dtype="float32", codes=codes)
    try:
        before = fetcher.fetch(reader, 0)
        reader.quant_specs[GATE_UP] = replace(reader.quant_specs[GATE_UP], dtype="bfloat16")
        expected = baseline.fetch(reader, 0)
        _same(expected, fetcher.fetch(reader, 0))
        import torch

        assert not torch.equal(before[0], expected[0])
        assert fetcher.cache_stats()["device"]["hits"] == 0
    finally:
        baseline.close()
        fetcher.close()
        reader.close()
