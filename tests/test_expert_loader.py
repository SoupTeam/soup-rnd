"""Loader selection and the CPU stand of the #77 two-slot packed stream.

``async_lease`` stays the default loader. ``cpp_async_2slot`` is an opt-in in the Pydantic
config and raises ``CppAsyncLoaderUnavailableError`` when the vendored extension is not built;
the Python lease is never substituted for it.

The sequence checks drive ``PackedSequenceLoader`` (the host-side controller) against
``PythonPackedStream``, a threaded stand of the extension's documented contract: worker threads
fill a bounded ring out of order, ``consume_next`` still hands experts back in sequence order,
a cancelled or failed sequence always frees its slots, and a wrong expert id is refused. The
last check feeds the loaded packed blocks to ``FusedCpuBackend`` (the CPU stand of the fused
node from ticket 03) and compares the output with the same backend fed directly.
"""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

torch = pytest.importorskip("torch")

from soup_cli.utils.expert_loader import (  # noqa: E402
    DEFAULT_EXPERT_LOADER,
    EXPERT_LOADER_CHOICES,
    CppAsyncLoaderUnavailableError,
    LoaderEnvironment,
    PackedSequenceLoader,
    detect_loader_environment,
    expert_loader_context,
    require_cpp_async_2slot,
)


def env(**changes: Any) -> LoaderEnvironment:
    values: Dict[str, Any] = {
        "extension_available": True,
        "cuda_available": True,
        "detail": None,
    }
    values.update(changes)
    return LoaderEnvironment(**values)


# ---------------------------------------------------------------- selection


def test_async_lease_is_the_default_and_the_cpp_arm_is_the_only_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert DEFAULT_EXPERT_LOADER == "async_lease"
    assert EXPERT_LOADER_CHOICES == ("async_lease", "cpp_async_2slot")

    def boom() -> LoaderEnvironment:
        raise AssertionError("the default loader must not probe the extension or CUDA")

    monkeypatch.setattr("soup_cli.utils.expert_loader.detect_loader_environment", boom)
    with expert_loader_context() as default, expert_loader_context("async_lease") as named:
        assert default is None and named is None


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        (
            {"extension_available": False, "detail": "no compiled module"},
            "build_ext.py",
        ),
        ({"cuda_available": False}, "CUDA"),
    ],
)
def test_cpp_async_2slot_names_the_missing_requirement(
    changes: Dict[str, Any], words: str
) -> None:
    with pytest.raises(CppAsyncLoaderUnavailableError, match=words):
        require_cpp_async_2slot(env(**changes))


def test_a_supported_environment_passes_the_requirement() -> None:
    require_cpp_async_2slot(env())


def test_the_refusal_is_a_runtime_error_that_names_the_option() -> None:
    with pytest.raises(RuntimeError, match="cpp_async_2slot"):
        require_cpp_async_2slot(env(extension_available=False, detail="missing"))


def test_selecting_the_cpp_loader_never_falls_back_to_the_python_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "soup_cli.utils.expert_loader.detect_loader_environment",
        lambda: env(extension_available=False, detail="missing"),
    )
    with pytest.raises(CppAsyncLoaderUnavailableError, match="build_ext.py"):
        with expert_loader_context("cpp_async_2slot"):
            pytest.fail("the body ran: the Python lease was substituted")


def test_a_built_extension_is_still_refused_until_the_pool_is_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU part selects the arm; the pool swap is the GPU part. No silent no-op."""
    monkeypatch.setattr(
        "soup_cli.utils.expert_loader.detect_loader_environment", lambda: env()
    )
    with pytest.raises(CppAsyncLoaderUnavailableError, match="not wired"):
        with expert_loader_context("cpp_async_2slot"):
            pytest.fail("the body ran without the two-slot loader in the pool")


def test_an_unknown_loader_is_refused() -> None:
    with pytest.raises(ValueError, match="async_lease"):
        with expert_loader_context("python_sync"):
            pass


def test_detect_loader_environment_reports_this_machine() -> None:
    found = detect_loader_environment()
    assert found.cuda_available == torch.cuda.is_available()
    if not found.extension_available:
        assert found.detail


def test_importing_the_loader_module_does_not_load_the_extension() -> None:
    """Selection is lazy: a clean checkout imports ``soup_cli`` without a compiled module."""
    import os
    import subprocess
    import sys

    import soup_cli

    source_root = Path(soup_cli.__file__).resolve().parent.parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root)] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    code = (
        "import sys, soup_cli.utils.expert_loader\n"
        "assert '_native_expert_stream_async' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(source_root),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------- config wiring


def test_the_config_default_is_async_lease() -> None:
    from soup_cli.config.schema import TrainingConfig

    assert TrainingConfig().stream_expert_loader == "async_lease"


def test_the_config_accepts_cpp_async_2slot_with_per_expert_streaming() -> None:
    from soup_cli.config.schema import SoupConfig

    config = SoupConfig(
        base="deepseek-ai/DeepSeek-V3",
        task="sft",
        data={"train": "data.jsonl"},
        training={
            "stream_layers": True,
            "stream_source": "disk",
            "stream_expert_layout": "per_expert",
            "stream_expert_loader": "cpp_async_2slot",
            "batch_size": 1,
        },
    )
    assert config.training.stream_expert_loader == "cpp_async_2slot"


def test_the_config_refuses_cpp_async_2slot_without_per_expert_layout() -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(Exception, match="per_expert"):
        SoupConfig(
            base="deepseek-ai/DeepSeek-V3",
            task="sft",
            data={"train": "data.jsonl"},
            training={
                "stream_layers": True,
                "stream_source": "disk",
                "stream_expert_loader": "cpp_async_2slot",
            },
        )


def test_the_config_refuses_a_non_default_loader_without_streaming() -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(Exception, match="stream_layers"):
        SoupConfig(
            base="some-model",
            task="sft",
            data={"train": "data.jsonl"},
            training={"stream_expert_loader": "cpp_async_2slot"},
        )


def test_the_config_refuses_an_unknown_loader() -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(Exception, match="async_lease"):
        SoupConfig(
            base="some-model",
            task="sft",
            data={"train": "data.jsonl"},
            training={"stream_expert_loader": "cpp_sync"},
        )


# ---------------------------------------------------------------- the Python stand


Source = Callable[[bytearray], None]


class PythonPackedStream:
    """Threaded CPU stand of ``PackedStreamAsync``'s sequence contract.

    Worker threads read into a bounded ring; publication is by sequence position, so a later
    position that finishes first waits in ``ready``. ``consume_next`` copies position order to
    the destination and only then frees the slot. Cancelling or failing a sequence frees every
    slot it held, which is what the slot-leak assertions read.
    """

    def __init__(
        self,
        sources: Mapping[Tuple[int, int], Source],
        *,
        packed_bytes: int,
        ring_slots: int = 2,
        io_workers: int = 1,
        delay: Optional[Callable[[int, int], float]] = None,
    ) -> None:
        if packed_bytes < 1:
            raise ValueError("packed_bytes must be positive")
        if not 2 <= ring_slots <= 64:
            raise ValueError("ring_slots must be between 2 and 64")
        if not 1 <= io_workers <= 16:
            raise ValueError("io_workers must be between 1 and 16")
        self.sources = dict(sources)
        self.packed_bytes = packed_bytes
        self.ring_slots = ring_slots
        self.io_workers = io_workers
        self.delay = delay or (lambda layer, expert: 0.0)
        self._ring = [bytearray(packed_bytes) for _ in range(ring_slots)]
        self._free = list(range(ring_slots))
        self._lock = threading.Condition()
        self._active: Optional[Dict[str, Any]] = None
        self._next: Optional[Dict[str, Any]] = None
        self._closed = False
        self.fetches = 0
        self.promotions = 0
        self.discards = 0
        self.finished = 0

    @property
    def ring_in_use(self) -> int:
        with self._lock:
            return self.ring_slots - len(self._free)

    @property
    def workers_alive(self) -> int:
        with self._lock:
            sequences = [sequence for sequence in (self._active, self._next) if sequence]
        return sum(1 for sequence in sequences for thread in sequence["threads"]
                   if thread.is_alive())

    def _new_sequence(self, layer: int, experts: Sequence[int]) -> Dict[str, Any]:
        return {
            "layer": int(layer),
            "experts": [int(expert) for expert in experts],
            "ready": {},
            "error": None,
            "cancelled": False,
            "next_consume": 0,
            "threads": [],
        }

    def _start_workers(self, sequence: Dict[str, Any]) -> None:
        gate = threading.Semaphore(self.io_workers)
        for position, expert in enumerate(sequence["experts"]):
            thread = threading.Thread(
                target=self._worker, args=(sequence, position, expert, gate), daemon=True
            )
            sequence["threads"].append(thread)
            thread.start()

    def _worker(
        self, sequence: Dict[str, Any], position: int, expert: int, gate: threading.Semaphore
    ) -> None:
        with gate:
            with self._lock:
                while (
                    not sequence["cancelled"]
                    and sequence["error"] is None
                    and self._active is sequence
                    and not self._free
                ):
                    self._lock.wait()
                if sequence["cancelled"] or sequence["error"] is not None:
                    return
                if self._active is not sequence:
                    return
                slot = self._free.pop(0)
            try:
                self.delay(sequence["layer"], expert)
                self.sources[(sequence["layer"], expert)](self._ring[slot])
            except BaseException as failure:  # noqa: BLE001 - forwarded to the consumer
                with self._lock:
                    self._free.append(slot)
                    if sequence["error"] is None:
                        sequence["error"] = failure
                    self._lock.notify_all()
                return
            with self._lock:
                if sequence["cancelled"] or self._active is not sequence:
                    self._free.append(slot)
                else:
                    sequence["ready"][position] = slot
                    self.fetches += 1
                self._lock.notify_all()

    def start_sequence(self, layer: int, experts: Sequence[int]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("stream is closed")
            if self._active is not None:
                raise RuntimeError("stream already has an active sequence")
            wanted = [int(expert) for expert in experts]
            if (
                self._next is not None
                and self._next["layer"] == int(layer)
                and self._next["experts"] == wanted
            ):
                self._active, self._next = self._next, None
                self.promotions += 1
                self._start_workers(self._active)
                self._lock.notify_all()
                return
            if self._next is not None:
                self._next = None
                self.discards += 1
            self._active = self._new_sequence(layer, wanted)
            self._start_workers(self._active)

    def enqueue_next_sequence(self, layer: int, experts: Sequence[int]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("stream is closed")
            if self._next is not None:
                self.discards += 1
            self._next = self._new_sequence(layer, experts)

    def cancel_next_sequence(self) -> None:
        with self._lock:
            if self._next is not None:
                self._next = None
                self.discards += 1

    def consume_next(self, destination: int) -> int:
        with self._lock:
            sequence = self._active
            if sequence is None or sequence["next_consume"] >= len(sequence["experts"]):
                raise RuntimeError("no active sequence item to consume")
            position = sequence["next_consume"]
            self._lock.wait_for(
                lambda: (
                    position in sequence["ready"]
                    or sequence["error"] is not None
                    or sequence["cancelled"]
                    or self._closed
                )
            )
            if sequence["error"] is not None:
                raise sequence["error"]
            if sequence["cancelled"] or self._closed:
                raise RuntimeError("sequence was cancelled")
            slot = sequence["ready"].pop(position)
            sequence["next_consume"] += 1
            ctypes.memmove(destination, bytes(self._ring[slot]), self.packed_bytes)
            self._free.append(slot)
            self._lock.notify_all()
            return sequence["experts"][position]

    def fill_sync(self, layer: int, expert: int, destination: int) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("stream is closed")
            if self._active is not None:
                raise RuntimeError("stream already has an active sequence")
        buffer = bytearray(self.packed_bytes)
        self.sources[(int(layer), int(expert))](buffer)
        ctypes.memmove(destination, bytes(buffer), self.packed_bytes)
        with self._lock:
            self.fetches += 1

    def finish_sequence(self) -> None:
        with self._lock:
            sequence, self._active = self._active, None
            if sequence is None:
                return
            sequence["cancelled"] = True
            self._lock.notify_all()
        for thread in sequence["threads"]:
            thread.join(timeout=5)
        with self._lock:
            for slot in sequence["ready"].values():
                self._free.append(slot)
            sequence["ready"].clear()
            self.finished += 1

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "ring_slots": self.ring_slots,
                "fetches": self.fetches,
                "promotions": self.promotions,
                "discards": self.discards,
                "finished": self.finished,
            }

    def close(self) -> None:
        self.finish_sequence()
        with self._lock:
            self._closed = True


def pattern_source(seed: int) -> Source:
    def fill(destination: bytearray) -> None:
        for index in range(len(destination)):
            destination[index] = (seed * 37 + index * 11) % 251

    return fill


def expected_bytes(seed: int, size: int) -> bytes:
    return bytes((seed * 37 + index * 11) % 251 for index in range(size))


def stream_for(
    experts: Sequence[int],
    *,
    ring_slots: int,
    io_workers: int = 2,
    delay: Optional[Callable[[int, int], float]] = None,
    layer: int = 0,
    packed_bytes: int = 64,
) -> PythonPackedStream:
    sources = {(layer, expert): pattern_source(expert) for expert in experts}
    return PythonPackedStream(
        sources,
        packed_bytes=packed_bytes,
        ring_slots=ring_slots,
        io_workers=io_workers,
        delay=delay,
    )


@pytest.mark.parametrize("ring_slots", [2, 4, 8])
def test_a_sequence_hands_experts_back_in_order_with_their_own_bytes(ring_slots: int) -> None:
    """Two workers publish out of order; consume still returns sequence order."""
    experts = list(range(8))

    def reverse_delay(layer: int, expert: int) -> float:
        return (7 - expert) * 0.002

    stream = stream_for(experts, ring_slots=ring_slots, delay=reverse_delay)
    loader = PackedSequenceLoader(stream, packed_bytes=stream.packed_bytes)
    seen: List[int] = []
    try:
        for expert, slot in loader.sequence(0, experts):
            seen.append(expert)
            assert bytes(slot.tolist()) == expected_bytes(expert, stream.packed_bytes)
        assert seen == experts
        assert stream.fetches == len(experts)
        assert stream.ring_in_use == 0
    finally:
        loader.close()


def test_a_partially_consumed_sequence_blocks_until_the_iterator_is_closed() -> None:
    """An abandoned iterator keeps the ring; closing it frees the slot for the next run."""
    stream = stream_for([0, 1, 2], ring_slots=2)
    loader = PackedSequenceLoader(stream, packed_bytes=stream.packed_bytes)
    try:
        partial = loader.sequence(0, [0, 1, 2])
        assert next(partial)[0] == 0
        with pytest.raises(RuntimeError, match="active sequence"):
            next(iter(loader.sequence(0, [2])))
        partial.close()
        assert stream.ring_in_use == 0
        assert stream.finished == 1
        assert [expert for expert, _ in loader.sequence(0, [2])] == [2]
    finally:
        loader.close()


def test_a_read_failure_is_raised_and_frees_every_ring_slot() -> None:
    experts = [0, 1, 2, 3]

    def source(seed: int) -> Source:
        if seed == 2:
            def fail(destination: bytearray) -> None:
                raise OSError("short read from expert shard")

            return fail
        return pattern_source(seed)

    stream = PythonPackedStream(
        {(0, expert): source(expert) for expert in experts},
        packed_bytes=32,
        ring_slots=2,
        io_workers=2,
    )
    loader = PackedSequenceLoader(stream, packed_bytes=32)
    try:
        with pytest.raises(OSError, match="short read"):
            list(loader.sequence(0, experts))
        assert stream.ring_in_use == 0
        assert stream.finished == 1
        # the next sequence still runs: the failure did not poison the ring
        assert [expert for expert, _ in loader.sequence(0, [3])] == [3]
    finally:
        loader.close()


def test_cancelling_a_prefetch_discards_the_hint_and_the_next_sequence_runs() -> None:
    stream = stream_for([0, 1], ring_slots=2)
    loader = PackedSequenceLoader(stream, packed_bytes=stream.packed_bytes)
    try:
        loader.prefetch(0, [0, 1])
        loader.cancel_prefetch()
        assert stream.stats()["discards"] == 1
        assert [expert for expert, _ in loader.sequence(0, [1])] == [1]
    finally:
        loader.close()


def test_a_matching_prefetch_is_promoted_without_a_second_read() -> None:
    stream = stream_for([0, 1], ring_slots=2)
    loader = PackedSequenceLoader(stream, packed_bytes=stream.packed_bytes)
    try:
        loader.prefetch(0, [0, 1])
        assert [expert for expert, _ in loader.sequence(0, [0, 1])] == [0, 1]
        assert stream.promotions == 1
        assert stream.fetches == 2
    finally:
        loader.close()


class WrongOrderStream(PythonPackedStream):
    """Hands back a valid expert id that is not the one the sequence asked for."""

    def consume_next(self, destination: int) -> int:
        expert = super().consume_next(destination)
        return expert + 1


def test_a_wrong_expert_id_is_refused_and_the_sequence_is_finished() -> None:
    stream = WrongOrderStream(
        {(0, 0): pattern_source(0), (0, 1): pattern_source(1)},
        packed_bytes=16,
        ring_slots=2,
    )
    loader = PackedSequenceLoader(stream, packed_bytes=16)
    try:
        with pytest.raises(RuntimeError, match="order"):
            list(loader.sequence(0, [0, 1]))
        assert stream.finished == 1
        assert stream.ring_in_use == 0
    finally:
        loader.close()


def test_fill_sync_returns_the_experts_packed_bytes() -> None:
    stream = stream_for([5], ring_slots=2)
    loader = PackedSequenceLoader(stream, packed_bytes=stream.packed_bytes)
    try:
        slot = loader.fill_sync(0, 5)
        assert bytes(slot.tolist()) == expected_bytes(5, stream.packed_bytes)
    finally:
        loader.close()


# ---------------------------------------------------------------- the fused CPU backend


def pack_expert(gate_up: Any, down: Any) -> Tuple[Any, List[tuple]]:
    from soup_cli.utils import expert_transfer

    rows, size = expert_transfer.packing_layout({"gate_up": gate_up, "down": down})
    block = torch.empty(size, dtype=torch.uint8)
    expert_transfer.pack_into(block, rows, {"gate_up": gate_up, "down": down})
    return block, rows


def unpack_expert(block: Any, rows: Sequence[tuple]) -> Dict[str, Any]:
    tensors: Dict[str, Any] = {}
    for name, offset, count, dtype, shape in rows:
        tensors[name] = block[offset : offset + count].view(dtype).view(shape).clone()
    return tensors


def test_loaded_experts_drive_the_fused_cpu_backend_like_direct_weights() -> None:
    """``FusedCpuBackend`` gets the bytes the sequence delivered, not the direct weights."""
    from tests.test_direct_expert_backward_mode import FusedCpuBackend

    torch.manual_seed(4)
    hidden_size, rows = 16, 3
    gate_up = {expert: torch.randn(2 * hidden_size, hidden_size) for expert in (2, 3)}
    down = {expert: torch.randn(hidden_size, hidden_size) for expert in (2, 3)}
    blocks, layouts = {}, {}
    for expert in (2, 3):
        blocks[expert], layouts[expert] = pack_expert(gate_up[expert], down[expert])
    size = max(block.numel() for block in blocks.values())
    padded = {
        key: torch.cat([block, torch.zeros(size - block.numel(), dtype=torch.uint8)])
        for key, block in blocks.items()
    }
    sources = {
        (0, expert): (lambda destination, expert=expert: destination.__setitem__(
            slice(None), bytes(padded[expert].tolist())
        ))
        for expert in (2, 3)
    }
    stream = PythonPackedStream(sources, packed_bytes=size, ring_slots=2, io_workers=2)
    loader = PackedSequenceLoader(stream, packed_bytes=size)
    hidden = torch.randn(rows, hidden_size)
    weights = torch.rand(rows, 1)
    reference_backend = FusedCpuBackend()
    reference: Dict[int, Any] = {}
    for expert in (2, 3):
        reference[expert] = reference_backend.expert_forward(
            hidden, weights, gate_up[expert], down[expert], [rows]
        )
    backend = FusedCpuBackend()
    try:
        for expert, slot in loader.sequence(0, [3, 2]):
            tensors = unpack_expert(slot, layouts[expert])
            got = backend.expert_forward(
                hidden, weights, tensors["gate_up"], tensors["down"], [rows]
            )
            assert torch.equal(got, reference[expert]), f"expert {expert} came back wrong"
    finally:
        loader.close()
    assert backend.forward_calls == 2


def test_the_stand_does_not_leak_threads_after_close() -> None:
    """A closed stand has no live workers and no held slots."""
    stream = stream_for(list(range(4)), ring_slots=4)
    loader = PackedSequenceLoader(stream, packed_bytes=stream.packed_bytes)
    list(loader.sequence(0, list(range(4))))
    loader.close()
    assert stream.ring_in_use == 0
    assert stream.workers_alive == 0
