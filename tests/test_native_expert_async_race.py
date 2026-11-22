"""Race stress of the #77 C++ two-slot loader; skipped until the extension is built.

The vendored ``_native_expert_stream_async`` reads packed experts on worker threads into a
bounded ring, publishes by sequence position and frees a slot only after ``consume_next``
copied it out. These checks demand the bytes and the expert order for rings 2/4/8, a cancelled
prefetch, and a read failure that must raise, free its slots and leave the stream usable.

Run on a host with the extension built:

    python src/soup_cli/utils/_native_expert_async/build_ext.py --build-lib /tmp/native-ext
    PYTHONPATH=/tmp/native-ext pytest tests/test_native_expert_async_race.py --no-cov -v
"""

from __future__ import annotations

import os
import random
import types
from pathlib import Path
from typing import Any, List, Tuple

import pytest

torch = pytest.importorskip("torch")

RECORD_BYTES = 64
SECTOR = 4096


def _extension_or_skip() -> Any:
    try:
        import _native_expert_stream_async as extension
    except ImportError:
        pytest.skip(
            "_native_expert_stream_async is not built; run "
            "python src/soup_cli/utils/_native_expert_async/build_ext.py "
            "--build-lib /tmp/native-ext"
        )
    return extension


def _shard(tmp_path: Path, experts: int) -> Tuple[Path, bytes]:
    """``experts`` fixed-size records with distinct, position-dependent bytes."""
    data = bytearray()
    for expert in range(experts):
        data.extend(((expert * 29 + index * 7) % 251) for index in range(RECORD_BYTES))
    path = tmp_path / "experts.bin"
    path.write_bytes(bytes(data))
    return path, bytes(data)


def _records(experts: int) -> List[List[Tuple[int, int, int]]]:
    return [[(expert * RECORD_BYTES, (expert + 1) * RECORD_BYTES, 0)] for expert in range(experts)]


def _stream(
    extension: Any, path: Path, experts: int, *, ring_slots: int, io_workers: int = 4
) -> Tuple[Any, Any]:
    descriptor = os.open(path, os.O_RDONLY)
    reader = types.SimpleNamespace(fetches=0, bytes_read=0)
    try:
        stream = extension.PackedStreamAsync(
            [(reader, descriptor, _records(experts))],
            SECTOR,
            RECORD_BYTES,
            0,  # no host cache: every sequence reads, so a lost fetch would show
            io_workers,
            ring_slots,
        )
    finally:
        os.close(descriptor)
    return stream, reader


def _expected(data: bytes, expert: int) -> bytes:
    return data[expert * RECORD_BYTES : (expert + 1) * RECORD_BYTES]


def _consume(stream: Any, count: int) -> List[Tuple[int, bytes]]:
    out: List[Tuple[int, bytes]] = []
    buffer = torch.empty(RECORD_BYTES, dtype=torch.uint8)
    for _ in range(count):
        expert = int(stream.consume_next(buffer.data_ptr()))
        out.append((expert, bytes(buffer.tolist())))
    return out


@pytest.mark.parametrize("ring_slots", [2, 4, 8])
def test_every_sequence_keeps_its_order_and_bytes(tmp_path: Path, ring_slots: int) -> None:
    extension = _extension_or_skip()
    experts = 24
    path, data = _shard(tmp_path, experts)
    stream, reader = _stream(extension, path, experts, ring_slots=ring_slots)
    random.seed(ring_slots)
    try:
        for _ in range(40):
            wanted = random.sample(range(experts), random.randint(1, experts))
            stream.start_sequence(0, wanted)
            got = _consume(stream, len(wanted))
            stream.finish_sequence()
            assert [expert for expert, _ in got] == wanted
            assert all(payload == _expected(data, expert) for expert, payload in got)
        stats = stream.stats()
        assert stats["misses"] >= 40  # no host cache: each visited expert was read again
        assert reader.fetches == stream.native_stats()["range_reads"]
        assert stream.native_stats()["ring_slots"] == ring_slots
    finally:
        stream.close()


def test_a_matching_cross_layer_hint_is_promoted_and_cancel_is_clean(tmp_path: Path) -> None:
    extension = _extension_or_skip()
    experts = 8
    path, data = _shard(tmp_path, experts)
    stream, _ = _stream(extension, path, experts, ring_slots=2, io_workers=2)
    try:
        stream.enqueue_next_sequence(0, [5, 6])
        stream.start_sequence(0, [5, 6])
        got = _consume(stream, 2)
        stream.finish_sequence()
        assert [expert for expert, _ in got] == [5, 6]
        assert stream.native_stats()["cross_layer_promotions"] == 1

        stream.enqueue_next_sequence(0, [1, 2])
        stream.cancel_next_sequence()
        stream.start_sequence(0, [7])
        assert _consume(stream, 1)[0][1] == _expected(data, 7)
        stream.finish_sequence()
    finally:
        stream.close()


def test_a_read_failure_raises_frees_the_ring_and_the_stream_recovers(tmp_path: Path) -> None:
    extension = _extension_or_skip()
    experts = 4
    path, data = _shard(tmp_path, experts)
    stream, _ = _stream(extension, path, experts, ring_slots=2)
    stat = path.stat()
    try:
        stream.start_sequence(0, [0, 1])
        # Change the file identity the extension checked at registration. A truncate would
        # also shorten the file; moving the mtime is the smallest change that ``check`` sees
        # and the nanosecond value is restored exactly below.
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with pytest.raises(RuntimeError, match="expert shard"):
            _consume(stream, 1)
        stream.finish_sequence()

        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        if path.stat().st_mtime_ns != stat.st_mtime_ns:
            pytest.skip("this filesystem cannot restore mtime nanoseconds")
        stream.start_sequence(0, [2])
        got = _consume(stream, 1)
        stream.finish_sequence()
        assert got[0][0] == 2 and got[0][1] == _expected(data, 2)
    finally:
        stream.close()


def test_fill_sync_and_counters(tmp_path: Path) -> None:
    extension = _extension_or_skip()
    experts = 6
    path, data = _shard(tmp_path, experts)
    stream, reader = _stream(extension, path, experts, ring_slots=4)
    buffer = torch.empty(RECORD_BYTES, dtype=torch.uint8)
    try:
        stream.fill_sync(0, 4, buffer.data_ptr())
        assert bytes(buffer.tolist()) == _expected(data, 4)
        stats = stream.stats()
        assert stats["misses"] == 1 and stats["entries"] == 0  # budget 0: never cached
        assert reader.fetches == 1
    finally:
        stream.close()
