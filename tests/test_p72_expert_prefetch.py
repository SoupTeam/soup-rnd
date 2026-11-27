"""Expert prefetch: the next hit expert is read on a worker thread while the current one computes.

Seam: a per-expert DeepSeek-V3 model from ``build_streamed_model(..., expert_prefetch=True)``.
One model runs with the pool's ``prefetch_enabled`` switched off and on, so every comparison
is ``torch.equal`` on the same weights. The read counters and the order of reads must not
change; only the thread that performs a read and the moment it starts may.

Interface under test:
* ``build_streamed_model(..., expert_prefetch=False)`` starts a worker thread named
  ``expert-prefetch*`` only when true; ``runtime.close()`` stops it.
* ``ExpertBufferPool.prefetch_enabled`` (attribute) and ``ExpertBufferPool.prefetch(reader,
  expert)``: a hint that reads ``expert`` into a second staging buffer in the background. A
  later ``lease`` for that same reader and expert uses it; any other lease discards it and
  reads synchronously. At most one read is in flight in the whole pool.
* ``StreamedExperts`` hints the next expert of its visit order in the forward and the previous
  one in the backward. There is no hint across layers.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest
import torch

from benchmarks.harness import b13_moe_fixture as fixture
from soup_cli.utils.expert_shard import GATE_UP
from tests.test_b13_moe_runtime import (
    build_moe_streamed,
    grads_differ,
    run_moe_pass,
    streamed_experts_modules,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR

WORKER_PREFIX = "expert-prefetch"


def _workers() -> List[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith(WORKER_PREFIX)]


@pytest.fixture()
def build(tmp_path: Any) -> Iterator[Callable[..., Any]]:
    made: List[Any] = []
    config = fixture.moe_config()
    weights = fixture.moe_weights_dir(tmp_path, config)

    def make(name: str, **options: Any) -> Any:
        moe = build_moe_streamed(
            weights, tmp_path / name, lora_config=fixture.moe_lora_config(), **options
        )
        made.append(moe)
        return moe

    make.config = config  # type: ignore[attr-defined]
    yield make
    for moe in made:
        with contextlib.suppress(Exception):  # a test may have closed it already
            moe.close()


class ReadLog:
    """Every ``read_host`` of every layer: (layer, expert, is_worker_thread), in order."""

    def __init__(self, model: Any, *, pause: float = 0.0) -> None:
        self.rows: List[Tuple[int, int, bool]] = []
        self.inflight = 0
        self.max_inflight = 0
        self._lock = threading.Lock()
        for layer, module in streamed_experts_modules(model).items():
            original = module.reader.read_host

            def wrapped(expert: int, staging: Any, *, _o: Any = original, _l: int = layer) -> Any:
                with self._lock:
                    self.inflight += 1
                    self.max_inflight = max(self.max_inflight, self.inflight)
                    self.rows.append(
                        (_l, expert, threading.current_thread() is not threading.main_thread())
                    )
                try:
                    if pause:
                        time.sleep(pause)
                    return _o(expert, staging)
                finally:
                    with self._lock:
                        self.inflight -= 1

            module.reader.read_host = wrapped

    def by_layer(self) -> Dict[int, List[Tuple[int, bool]]]:
        out: Dict[int, List[Tuple[int, bool]]] = {}
        for layer, expert, worker in self.rows:
            out.setdefault(layer, []).append((expert, worker))
        return out

    def clear(self) -> None:
        self.rows.clear()


def _batch(build: Any, seed: int = 3) -> Any:
    return fixture.moe_batch(build.config, seed=seed, rows=2, seq=6)


# ==========================================================================
# the numbers do not change
# ==========================================================================
@pytest.mark.parametrize("two_read", [False, True])
def test_prefetch_leaves_loss_and_every_gradient_bit_equal(build, two_read):
    moe = build("a", expert_prefetch=True, two_read_experts=two_read, grouped_expert_routes=True)
    pool = moe.runtime.expert_pool
    batch = _batch(build)
    pool.prefetch_enabled = False
    want = run_moe_pass(moe.model, batch)
    pool.prefetch_enabled = True
    got = run_moe_pass(moe.model, batch)
    assert want.grads and any(g is not None and bool(g.ne(0).any()) for g in want.grads.values())
    assert torch.equal(got.loss, want.loss)
    assert torch.equal(got.logits, want.logits)
    assert grads_differ(got.grads, want.grads) == []


@pytest.mark.parametrize("two_read", [False, True])
def test_prefetch_reads_the_same_experts_in_the_same_order_with_the_same_counters(build, two_read):
    moe = build("b", expert_prefetch=True, two_read_experts=two_read, grouped_expert_routes=True)
    pool = moe.runtime.expert_pool
    log = ReadLog(moe.model)
    batch = _batch(build)
    seen = {}
    for enabled in (False, True):
        pool.prefetch_enabled = enabled
        log.clear()
        fetches, logical = pool.fetches, pool.logical_bytes_read
        run_moe_pass(moe.model, batch)
        seen[enabled] = (
            {layer: [e for e, _ in rows] for layer, rows in log.by_layer().items()},
            pool.fetches - fetches,
            pool.logical_bytes_read - logical,
        )
    assert seen[True] == seen[False]
    assert seen[False][1] > 0 and seen[False][2] > 0


# ==========================================================================
# what moves to the worker
# ==========================================================================
def test_forward_reads_after_the_first_expert_of_a_layer_run_on_the_worker(build):
    moe = build("c", expert_prefetch=True, grouped_expert_routes=True)
    pool = moe.runtime.expert_pool
    log = ReadLog(moe.model)
    batch = _batch(build)
    for enabled in (False, True):
        pool.prefetch_enabled = enabled
        log.clear()
        run_moe_pass(moe.model, batch, backward=False)
        layers = log.by_layer()
        assert any(len(rows) >= 2 for rows in layers.values()), "fixture routes too little"
        for rows in layers.values():
            assert [worker for _, worker in rows] == (
                [False] + [enabled] * (len(rows) - 1)
            ), "no hint crosses a layer: the first expert of a layer is read on the caller"


@pytest.mark.parametrize("two_read", [False, True])
def test_backward_prefetches_in_reverse_visit_order(build, two_read):
    moe = build("d", expert_prefetch=True, two_read_experts=two_read, grouped_expert_routes=True)
    pool = moe.runtime.expert_pool
    log = ReadLog(moe.model)
    batch = _batch(build)
    pool.prefetch_enabled = True
    moe.model.train()
    moe.model.zero_grad(set_to_none=True)
    out = moe.model(**batch)
    forward = {layer: [e for e, _ in rows] for layer, rows in log.by_layer().items()}
    log.clear()
    out.loss.backward()
    backward = log.by_layer()
    assert any(len(rows) >= 2 for rows in backward.values())
    for layer, rows in backward.items():
        visit = forward[layer]
        # Without two-read the layer checkpoint recomputes the forward first (the same visit
        # order again), then each expert's own backward walks that order in reverse.
        order = list(reversed(visit)) if two_read else visit + list(reversed(visit))
        sweep = [False] + [True] * (len(visit) - 1)  # the first expert of a sweep has no hint
        assert [e for e, _ in rows] == order
        assert [worker for _, worker in rows] == (sweep if two_read else sweep + sweep)


def test_at_most_one_read_is_in_flight_in_the_whole_pool(build):
    moe = build("e", expert_prefetch=True, grouped_expert_routes=True)
    log = ReadLog(moe.model, pause=0.002)
    batch = _batch(build)
    moe.runtime.expert_pool.prefetch_enabled = True
    run_moe_pass(moe.model, batch)
    assert len(log.rows) > 4
    assert log.max_inflight == 1


def test_prefetch_hides_the_read_behind_the_compute(build, monkeypatch):
    moe = build("f", expert_prefetch=True, grouped_expert_routes=True)
    pool = moe.runtime.expert_pool
    ReadLog(moe.model, pause=0.03)
    gate_up_shape = tuple(pool._buffers[GATE_UP].shape)
    original = torch.nn.functional.linear

    def slow_linear(inp: Any, weight: Any, bias: Any = None) -> Any:
        if tuple(weight.shape) == gate_up_shape:
            time.sleep(0.03)
        return original(inp, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", slow_linear)
    batch = _batch(build)
    seconds = {}
    for enabled in (False, True):
        pool.prefetch_enabled = enabled
        started = time.perf_counter()
        run_moe_pass(moe.model, batch, backward=False)
        seconds[enabled] = time.perf_counter() - started
    assert seconds[True] < 0.8 * seconds[False], seconds


# ==========================================================================
# the hint is only a hint
# ==========================================================================
def _first_layer(moe: Any) -> Tuple[Any, Any]:
    module = streamed_experts_modules(moe.model)[min(streamed_experts_modules(moe.model))]
    return moe.runtime.expert_pool, module.reader


def _weights_of(pool: Any, reader: Any, expert: int) -> Tuple[Any, Any]:
    with pool.lease(reader, expert) as (gate_up, down):
        return gate_up.clone(), down.clone()


def test_a_hint_for_another_expert_is_discarded_and_the_asked_expert_is_read_directly(build):
    moe = build("g", expert_prefetch=True, grouped_expert_routes=True)
    pool, reader = _first_layer(moe)
    pool.prefetch_enabled = False
    want = _weights_of(pool, reader, 2)
    pool.prefetch_enabled = True
    fetches = pool.fetches
    pool.prefetch(reader, 5)
    got = _weights_of(pool, reader, 2)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert pool.fetches == fetches + 1, "a discarded hint is not a fetch"
    pool.prefetch(reader, 1)
    pool.prefetch_enabled = False
    sync_one = _weights_of(pool, reader, 1)
    pool.prefetch_enabled = True
    pool.prefetch(reader, 1)
    assert all(torch.equal(a, b) for a, b in zip(_weights_of(pool, reader, 1), sync_one))


def test_a_failed_background_read_raises_in_the_lease_that_asked_for_that_expert(build):
    moe = build("h", expert_prefetch=True, grouped_expert_routes=True)
    pool, reader = _first_layer(moe)
    pool.prefetch_enabled = True
    original = reader.read_host

    def failing(expert: int, staging: Any) -> Any:
        if threading.current_thread() is not threading.main_thread() and expert == 4:
            raise OSError("disk went away")
        return original(expert, staging)

    reader.read_host = failing
    pool.prefetch(reader, 4)
    with pytest.raises(OSError, match="disk went away"):
        with pool.lease(reader, 4):
            pass
    reader.read_host = original
    assert _weights_of(pool, reader, 4)[0].numel() > 0, "the pool stays usable after the error"


# ==========================================================================
# off when asked, and no thread outlives the runtime
# ==========================================================================
def test_prefetch_off_when_asked_and_starts_no_thread(build):
    before = _workers()
    moe = build("i", grouped_expert_routes=True, expert_prefetch=False)
    log = ReadLog(moe.model)
    run_moe_pass(moe.model, _batch(build))
    assert moe.runtime.expert_pool.prefetch_enabled is False
    assert _workers() == before
    assert log.rows and not any(worker for _, _, worker in log.rows)


def test_closing_the_runtime_stops_the_worker(build):
    before = _workers()
    moe = build("j", expert_prefetch=True, grouped_expert_routes=True)
    run_moe_pass(moe.model, _batch(build))
    assert len(_workers()) == len(before) + 1
    moe.close()
    deadline = time.time() + 5
    while _workers() != before and time.time() < deadline:
        time.sleep(0.01)
    assert _workers() == before
