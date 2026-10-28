"""Actual CPU fetcher controls for opt-in usage observation and placement."""

from __future__ import annotations

import os
from collections import Counter
from types import SimpleNamespace

import pytest

from soup_cli.utils.expert_placement import ExpertUsageProfile
from soup_cli.utils.expert_stream import ExpertFetcher
from tests.test_stream_expert_cache_fetcher import _reader, _same


def _logical(reader, expert):
    return os.path.realpath(reader.path), expert


@pytest.mark.parametrize("nf4", [False, True])
@pytest.mark.parametrize("host,device", [(2**20, 0), (0, 2**20), (2**20, 6144)])
def test_profile_counts_real_cpu_fetches_once_including_both_cache_hits(
    tmp_path, nf4, host, device,
):
    reader, _pairs, codes = _reader(tmp_path, nf4=nf4)
    arguments = {"device": "cpu", "dtype": "float32", "codes": codes,
                 "host_cache_bytes": host, "device_cache_bytes": device,
                 "host_cache_policy": "static"}
    observed, reference = ExpertFetcher(**arguments), ExpertFetcher(**arguments)
    profile = ExpertUsageProfile()
    trace = (0, 1, 0, 2, 1, 0, 2)
    try:
        observed.configure_usage_profile(profile)
        held = []
        for expert in trace:
            expected = reference.fetch(reader, expert)
            actual = observed.fetch(reader, expert)
            _same(expected, actual)
            held.append((expected, actual))
        # Later source-staging reuse must not change any previously returned weights.
        for expected, actual in held:
            _same(expected, actual)
        assert observed.cache_stats() == reference.cache_stats()
        assert profile.stats()["total_requests"] == len(trace)
        assert observed.cache_stats()["logical_fetches"] == len(trace)
        records = {row.key: row for row in profile.snapshot()}
        for expert, count in Counter(trace).items():
            row = records[_logical(reader, expert)]
            assert row.requests == count
            assert row.transfer_bytes == sum(
                entry.nbytes for _name, entry in reader._entries[expert])
            assert row.dense_bytes == 6144
        if host:
            assert observed.cache_stats()["host"]["hits"] > 0
        if device:
            assert observed.cache_stats()["device"]["hits"] > 0
    finally:
        observed.close()
        reference.close()
        reader.close()


def test_profile_sizes_and_logical_ids_are_per_reader_not_global(tmp_path):
    dense, dense_pairs, _codes = _reader(tmp_path, "dense", offset=2)
    quantized, _pairs, codes = _reader(tmp_path, "quantized", offset=8, nf4=True)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", codes=codes,
                            host_cache_bytes=2**20, host_cache_policy="static")
    reference = ExpertFetcher(device="cpu", dtype="float32", codes=codes)
    profile = ExpertUsageProfile()
    try:
        fetcher.configure_usage_profile(profile)
        for _repeat in range(2):
            _same(dense_pairs[0], fetcher.fetch(dense, 0))
            _same(reference.fetch(quantized, 0), fetcher.fetch(quantized, 0))
        rows = {row.key: row for row in profile.snapshot()}
        assert set(rows) == {_logical(dense, 0), _logical(quantized, 0)}
        for reader in (dense, quantized):
            row = rows[_logical(reader, 0)]
            assert row.requests == 2 and row.dense_bytes == 6144
            assert row.transfer_bytes == sum(entry.nbytes for _name, entry in reader._entries[0])
        assert rows[_logical(dense, 0)].transfer_bytes > rows[_logical(quantized, 0)].transfer_bytes
        assert fetcher.cache_stats()["host"]["hits"] == 2
    finally:
        fetcher.close()
        reference.close()
        dense.close()
        quantized.close()


@pytest.mark.parametrize("nf4", [False, True])
def test_allowlist_selects_later_expert_and_keeps_host_gpu_entries_disjoint(tmp_path, nf4):
    reader, _pairs, codes = _reader(tmp_path, nf4=nf4)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", codes=codes,
                            host_cache_bytes=2**20, device_cache_bytes=6144,
                            host_cache_policy="static")
    reference = ExpertFetcher(device="cpu", dtype="float32", codes=codes)
    selected = [_logical(reader, 1)]
    try:
        fetcher.configure_placement(selected)
        selected.clear()  # The caller cannot later mutate the frozen runtime plan.
        for expert in (0, 1, 0, 1, 2, 1, 2, 0):
            _same(reference.fetch(reader, expert), fetcher.fetch(reader, expert))
        stats = fetcher.cache_stats()
        assert stats["device"]["entries"] == 1 and stats["device"]["payload_bytes"] == 6144
        assert stats["host"]["entries"] == 2
        assert stats["device"]["hits"] == 2 and stats["host"]["hits"] == 3
        assert stats["logical_fetches"] == 8 and stats["storage_reads"] == 3
        assert stats["device_placement"] == {"policy": "profile_allowlist", "keys": 1}
        for expert in range(3):
            key = fetcher._cache_key(reader, expert)
            device = fetcher._device_cache.peek(key)
            host = fetcher._host_cache.peek(key)
            assert (device is not None) == (expert == 1)
            assert (host is not None) == (expert != 1)
    finally:
        fetcher.close()
        reference.close()
        reader.close()


def test_same_expert_number_on_another_reader_is_not_selected(tmp_path):
    first, first_pairs, _codes = _reader(tmp_path, "first")
    second, second_pairs, _codes = _reader(tmp_path, "second", offset=10)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", host_cache_bytes=2**20,
                            device_cache_bytes=6144, host_cache_policy="static")
    try:
        fetcher.configure_placement((_logical(first, 0),))
        for _repeat in range(2):
            _same(second_pairs[0], fetcher.fetch(second, 0))
            _same(first_pairs[0], fetcher.fetch(first, 0))
        assert fetcher.cache_stats()["device"]["hits"] == 1
        assert fetcher.cache_stats()["host"]["hits"] == 1
        assert fetcher._device_cache.peek(fetcher._cache_key(second, 0)) is None
        assert fetcher._device_cache.peek(fetcher._cache_key(first, 0)) is not None
    finally:
        fetcher.close()
        first.close()
        second.close()


def test_empty_plan_disables_device_admission_and_none_restores_default_after_clear(tmp_path):
    reader, pairs, _codes = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", host_cache_bytes=2**20,
                            device_cache_bytes=6144, host_cache_policy="static")
    try:
        fetcher.configure_placement(())
        held = fetcher.fetch(reader, 2)
        _same(pairs[2], held)
        before = fetcher.cache_stats()
        assert before["device"]["entries"] == 0 and before["host"]["entries"] == 1
        fetcher.configure_placement(None)
        reset = fetcher.cache_stats()
        assert "device_placement" not in reset
        assert reset["device"]["entries"] == reset["host"]["entries"] == 0
        assert reset["device"]["budget_bytes"] == before["device"]["budget_bytes"]
        assert reset["host"]["budget_bytes"] == before["host"]["budget_bytes"]
        for counter in ("logical_fetches", "storage_reads", "materializations"):
            assert reset[counter] == before[counter]
        _same(pairs[0], fetcher.fetch(reader, 0))
        _same(pairs[1], fetcher.fetch(reader, 1))
        _same(pairs[2], held)
        assert fetcher._device_cache.peek(fetcher._cache_key(reader, 0)) is not None
        assert fetcher._host_cache.peek(fetcher._cache_key(reader, 0)) is None
        assert fetcher._host_cache.peek(fetcher._cache_key(reader, 1)) is not None
    finally:
        fetcher.close()
        reader.close()


def test_configure_observer_and_disable_it_preserve_existing_cache_entries(tmp_path):
    reader, pairs, _codes = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", device_cache_bytes=6144)
    profile = ExpertUsageProfile()
    try:
        held = fetcher.fetch(reader, 0)
        before = fetcher.cache_stats()
        fetcher.configure_usage_profile(profile)
        assert fetcher.cache_stats() == before
        observed = fetcher.fetch(reader, 0)
        assert all(left is right for left, right in zip(held, observed))
        assert profile.stats()["total_requests"] == 1
        fetcher.configure_usage_profile(None)
        frozen = profile.snapshot()
        _same(pairs[0], fetcher.fetch(reader, 0))
        assert profile.snapshot() == frozen
        assert fetcher.cache_stats()["device"]["hits"] == 2
    finally:
        fetcher.close()
        reader.close()


@pytest.mark.parametrize("bad", [True, "bad", {}, object()])
def test_invalid_observer_is_refused_before_existing_observer_or_cache_changes(tmp_path, bad):
    reader, _pairs, _codes = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", device_cache_bytes=6144)
    profile = ExpertUsageProfile()
    try:
        fetcher.configure_usage_profile(profile)
        fetcher.fetch(reader, 0)
        before = fetcher.cache_stats()
        with pytest.raises(TypeError, match="usage profile"):
            fetcher.configure_usage_profile(bad)
        assert fetcher.cache_stats() == before
        fetcher.fetch(reader, 0)
        assert profile.stats()["total_requests"] == 2
    finally:
        fetcher.close()
        reader.close()


@pytest.mark.parametrize("bad", [
    True, 5, "path", [("relative", 0)], [("", 0)], [("/absolute", True)],
    [("/absolute", -1)], [("/absolute", 1.0)], [("/absolute", 1, 2)],
])
def test_invalid_plan_refuses_before_clearing_warm_cache_or_previous_plan(tmp_path, bad):
    reader, pairs, _codes = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", host_cache_bytes=2**20,
                            device_cache_bytes=6144, host_cache_policy="static")
    try:
        fetcher.configure_placement((_logical(reader, 1),))
        fetcher.fetch(reader, 0)
        held = fetcher.fetch(reader, 1)
        before = fetcher.cache_stats()
        with pytest.raises((TypeError, ValueError)):
            fetcher.configure_placement(bad)
        assert fetcher.cache_stats() == before
        cached = fetcher.fetch(reader, 1)
        assert all(left is right for left, right in zip(held, cached))
        _same(pairs[0], fetcher.fetch(reader, 0))
    finally:
        fetcher.close()
        reader.close()


def test_partially_valid_plan_iterator_failure_does_not_clear_warm_entries(tmp_path):
    reader, pairs, _codes = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", device_cache_bytes=6144)
    try:
        held = fetcher.fetch(reader, 0)
        before = fetcher.cache_stats()

        def broken():
            yield _logical(reader, 1)
            raise LookupError("plan source failed")

        with pytest.raises(LookupError, match="plan source failed"):
            fetcher.configure_placement(broken())
        assert fetcher.cache_stats() == before
        _same(pairs[0], held)
        assert all(left is right for left, right in zip(held, fetcher.fetch(reader, 0)))
    finally:
        fetcher.close()
        reader.close()


def test_placement_normalizes_absolute_shard_alias_before_admission(tmp_path):
    reader, pairs, _codes = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", device_cache_bytes=6144)
    try:
        alias = os.path.join(str(tmp_path), "unused", "..", os.path.basename(reader.path))
        fetcher.configure_placement(((alias, 1),))
        _same(pairs[1], fetcher.fetch(reader, 1))
        _same(pairs[1], fetcher.fetch(reader, 1))
        assert fetcher.cache_stats()["device"]["hits"] == 1
    finally:
        fetcher.close()
        reader.close()


@pytest.mark.parametrize("invalid", [-1, 3, True, 1.0])
def test_invalid_expert_never_increments_usage_or_existing_cache_counters(tmp_path, invalid):
    reader, _pairs, _codes = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", device_cache_bytes=6144)
    profile = ExpertUsageProfile()
    try:
        fetcher.configure_usage_profile(profile)
        fetcher.fetch(reader, 0)
        before = fetcher.cache_stats(), profile.snapshot()
        with pytest.raises(IndexError, match="outside"):
            fetcher.fetch(reader, invalid)
        assert (fetcher.cache_stats(), profile.snapshot()) == before
    finally:
        fetcher.close()
        reader.close()


def test_speculative_cpu_prefetch_control_is_not_recorded_as_logical_demand(tmp_path):
    # Isolate observer placement with an inert transfer ticket, not simulated GPU math.
    reader, _pairs, codes = _reader(tmp_path, nf4=True)
    fetcher = ExpertFetcher(device="cpu", dtype="float32", codes=codes)
    profile = ExpertUsageProfile()
    released = []
    try:
        fetcher.configure_usage_profile(profile)
        fetcher._transfer = SimpleNamespace(
            stage=lambda _host: object(), release=lambda ticket, **_kwargs: released.append(ticket),
            close=lambda: None,
        )
        fetcher.transfer_mode = "prefetch"
        fetcher.prefetch(reader, 0)
        assert reader.fetches == 1
        assert profile.snapshot() == ()
        fetcher.cancel_prefetch()
        assert len(released) == 1
        fetcher._transfer = None
        fetcher.transfer_mode = "sync"
        fetcher.fetch(reader, 0)
        assert profile.snapshot()[0].requests == 1
        assert fetcher.cache_stats()["logical_fetches"] == 1
    finally:
        fetcher.close()
        reader.close()


def test_closed_fetcher_refuses_profile_and_placement_configuration(tmp_path):
    reader, _pairs, _codes = _reader(tmp_path)
    fetcher = ExpertFetcher(device="cpu", dtype="float32")
    profile = ExpertUsageProfile()
    fetcher.configure_usage_profile(profile)
    fetcher.fetch(reader, 0)
    snapshot = profile.snapshot()
    fetcher.close()
    try:
        for candidate in (None, profile, object()):
            with pytest.raises(RuntimeError, match="closed"):
                fetcher.configure_usage_profile(candidate)
        for candidate in (None, (), (_logical(reader, 0),)):
            with pytest.raises(RuntimeError, match="closed"):
                fetcher.configure_placement(candidate)
        assert profile.snapshot() == snapshot
    finally:
        reader.close()
