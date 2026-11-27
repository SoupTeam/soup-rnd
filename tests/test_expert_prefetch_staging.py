"""Expert prefetch with two staging slots (``expert_prefetch_staging=2``).

Seam: ``ExpertBufferPool`` built by ``build_streamed_model(..., expert_prefetch=True,
expert_prefetch_staging=N)`` on the tiny DeepSeek-V3 fixture of ``test_p72_expert_prefetch``.
Everything here runs on CPU. The CUDA side of the same claim (the pinned packed region and
``async_lease``) is not covered here and stays for a GPU run.

Interface under test:
* ``build_streamed_model(..., expert_prefetch_staging=1)``: 1 is today's single staging slot,
  2 keeps up to two background reads pending. Any other value is refused.
* ``ExpertBufferPool.prefetch_staging`` and ``stats()["prefetch"]["staging"]``.
* A slot is reused only after the lease that consumed it has copied its bytes out.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any, Callable, Iterator, List

import pytest
import torch

from benchmarks.harness import b13_moe_fixture as fixture
from soup_cli.utils import layer_stream_runtime
from tests.test_b13_moe_runtime import (
    build_moe_streamed,
    grads_differ,
    run_moe_pass,
    streamed_experts_modules,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_p72_expert_prefetch import ReadLog

pytestmark = _REQUIRES_TRAINING_FLOOR


@pytest.fixture()
def build(tmp_path: Any) -> Iterator[Callable[..., Any]]:
    made: List[Any] = []
    config = fixture.moe_config()
    weights = fixture.moe_weights_dir(tmp_path, config)

    def make(name: str, **options: Any) -> Any:
        moe = build_moe_streamed(
            weights, tmp_path / name, lora_config=fixture.moe_lora_config(),
            grouped_expert_routes=True, **options,
        )
        made.append(moe)
        return moe

    make.config = config  # type: ignore[attr-defined]
    yield make
    for moe in made:
        with contextlib.suppress(Exception):
            moe.close()


def _first_layer(moe: Any) -> Any:
    modules = streamed_experts_modules(moe.model)
    module = modules[min(modules)]
    return moe.runtime.expert_pool, module.reader


def _weights_of(pool: Any, reader: Any, expert: int) -> Any:
    with pool.lease(reader, expert) as (gate_up, down):
        return gate_up.clone(), down.clone()


def test_default_is_prefetch_with_two_staging_slots():
    for name in ("build_streamed_model", "install_streaming"):
        params = inspect.signature(getattr(layer_stream_runtime, name)).parameters
        assert params["expert_prefetch"].default is None
        assert params["expert_prefetch_staging"].default is None


def test_default_pool_prefetches_with_two_slots(build):
    moe = build("def")
    pool = moe.runtime.expert_pool
    assert pool.prefetch_enabled is True
    assert pool.prefetch_staging == 2
    assert pool.stats()["prefetch"]["staging"] == 2


def test_explicit_prefetch_off_keeps_one_slot_and_no_prefetch(build):
    moe = build("off", expert_prefetch=False)
    pool = moe.runtime.expert_pool
    assert pool.prefetch_enabled is False
    assert pool.prefetch_staging == 1


@pytest.mark.parametrize("bad", [0, 3, True, "2", 2.0])
def test_refuses_any_staging_count_other_than_one_or_two(build, bad):
    with pytest.raises(ValueError, match="prefetch_staging"):
        build("bad", expert_prefetch=True, expert_prefetch_staging=bad)


def test_two_slots_need_prefetch_on(build):
    with pytest.raises(ValueError, match="prefetch_staging=2 needs prefetch"):
        build("off", expert_prefetch=False, expert_prefetch_staging=2)


def test_two_slots_keep_two_reads_pending_and_both_hit(build):
    moe = build("two", expert_prefetch=True, expert_prefetch_staging=2)
    pool, reader = _first_layer(moe)
    pool.prefetch_enabled = False
    want_1 = _weights_of(pool, reader, 1)
    want_2 = _weights_of(pool, reader, 2)
    pool.prefetch_enabled = True
    hits, discards = pool.prefetch_hits, pool.prefetch_discards
    pool.prefetch(reader, 1)
    pool.prefetch(reader, 2)
    got_1 = _weights_of(pool, reader, 1)
    got_2 = _weights_of(pool, reader, 2)
    assert pool.prefetch_hits == hits + 2
    assert pool.prefetch_discards == discards, "two slots hold both hints, none is dropped"
    assert all(torch.equal(a, b) for a, b in zip(got_1, want_1))
    assert all(torch.equal(a, b) for a, b in zip(got_2, want_2))


def test_a_third_hint_with_two_slots_drops_the_oldest_and_reuses_its_slot(build):
    moe = build("third", expert_prefetch=True, expert_prefetch_staging=2)
    pool, reader = _first_layer(moe)
    pool.prefetch_enabled = False
    want_3 = _weights_of(pool, reader, 3)
    pool.prefetch_enabled = True
    discards = pool.prefetch_discards
    pool.prefetch(reader, 1)
    pool.prefetch(reader, 2)
    pool.prefetch(reader, 3)
    assert pool.prefetch_discards == discards + 1
    got_3 = _weights_of(pool, reader, 3)
    assert all(torch.equal(a, b) for a, b in zip(got_3, want_3))


def test_a_slot_is_not_overwritten_while_its_lease_body_computes(build):
    moe = build("inuse", expert_prefetch=True, expert_prefetch_staging=2)
    pool, reader = _first_layer(moe)
    pool.prefetch_enabled = False
    want_1 = _weights_of(pool, reader, 1)
    pool.prefetch_enabled = True
    pool.prefetch(reader, 1)
    pool.prefetch(reader, 2)
    with pool.lease(reader, 1) as (gate_up, down):
        pool.prefetch(reader, 3)
        pool.prefetch(reader, 4)
        assert torch.equal(gate_up, want_1[0])
        assert torch.equal(down, want_1[1])


def test_staging_two_forward_and_gradients_are_bit_equal_to_staging_one(build):
    options = {"expert_prefetch": True, "two_read_experts": True}
    one = build("s1", expert_prefetch_staging=1, **options)
    two = build("s2", expert_prefetch_staging=2, **options)
    batch = fixture.moe_batch(build.config, seed=3, rows=2, seq=6)
    want = run_moe_pass(one.model, batch)
    got = run_moe_pass(two.model, batch)
    assert want.grads and any(g is not None and bool(g.ne(0).any()) for g in want.grads.values())
    assert torch.equal(got.loss, want.loss)
    assert torch.equal(got.logits, want.logits)
    assert grads_differ(got.grads, want.grads) == []


def _record_hints(pool: Any, modules: Any) -> List[Any]:
    """Every real ``pool.prefetch`` hint as (layer, expert), in call order.

    A call with ``expert=None`` (no next expert) is a no-op and is not recorded.
    """
    calls: List[Any] = []
    layer_of = {id(module.reader): layer for layer, module in modules.items()}
    original = pool.prefetch

    def recorder(reader: Any, expert: Any) -> None:
        if expert is not None:
            calls.append((layer_of[id(reader)], expert))
        original(reader, expert)

    pool.prefetch = recorder
    return calls


def _ahead_hints(order: List[int], depth: int) -> List[int]:
    """The hints a visit order gives: each expert names the next ``depth`` experts, no further."""
    return [
        order[i + step]
        for i in range(len(order))
        for step in range(1, depth + 1)
        if i + step < len(order)
    ]


def test_staging_one_hints_only_the_next_expert_inside_each_layer(build):
    moe = build("hint1", expert_prefetch=True, expert_prefetch_staging=1)
    modules = streamed_experts_modules(moe.model)
    pool = moe.runtime.expert_pool
    pool.prefetch_enabled = True
    calls = _record_hints(pool, modules)
    run_moe_pass(moe.model, fixture.moe_batch(build.config, seed=3, rows=2, seq=6), backward=False)
    assert any(len(m.last_trace) >= 3 for m in modules.values()), "fixture routes too little"
    for layer, module in modules.items():
        visit = [row["expert"] for row in module.last_trace]
        assert [e for lay, e in calls if lay == layer] == _ahead_hints(visit, 1)


def test_staging_two_hints_the_next_two_experts_inside_each_layer(build):
    moe = build("hint2", expert_prefetch=True, expert_prefetch_staging=2)
    modules = streamed_experts_modules(moe.model)
    pool = moe.runtime.expert_pool
    pool.prefetch_enabled = True
    calls = _record_hints(pool, modules)
    run_moe_pass(moe.model, fixture.moe_batch(build.config, seed=3, rows=2, seq=6), backward=False)
    assert any(len(m.last_trace) >= 3 for m in modules.values()), "fixture routes too little"
    for layer, module in modules.items():
        visit = [row["expert"] for row in module.last_trace]
        assert [e for lay, e in calls if lay == layer] == _ahead_hints(visit, 2)


def test_staging_two_backward_hints_the_two_previous_experts_inside_each_layer(build):
    moe = build("back2", expert_prefetch=True, expert_prefetch_staging=2, two_read_experts=True)
    modules = streamed_experts_modules(moe.model)
    pool = moe.runtime.expert_pool
    pool.prefetch_enabled = True
    batch = fixture.moe_batch(build.config, seed=3, rows=2, seq=6)
    moe.model.train()
    moe.model.zero_grad(set_to_none=True)
    out = moe.model(**batch)
    visits = {layer: [row["expert"] for row in m.last_trace] for layer, m in modules.items()}
    assert any(len(visit) >= 3 for visit in visits.values()), "fixture routes too little"
    calls = _record_hints(pool, modules)
    out.loss.backward()
    for layer, visit in visits.items():
        want = _ahead_hints(list(reversed(visit)), 2)
        assert [e for lay, e in calls if lay == layer] == want


def test_two_staging_slots_still_run_one_read_at_a_time(build):
    moe = build("serial2", expert_prefetch=True, expert_prefetch_staging=2)
    pool = moe.runtime.expert_pool
    pool.prefetch_enabled = True
    log = ReadLog(moe.model, pause=0.002)
    run_moe_pass(moe.model, fixture.moe_batch(build.config, seed=3, rows=2, seq=6), backward=False)
    assert len(log.rows) > 4
    assert log.max_inflight == 1


def test_staging_two_reads_the_same_experts_with_the_same_counters(build):
    one = build("c1", expert_prefetch=True, expert_prefetch_staging=1, two_read_experts=True)
    two = build("c2", expert_prefetch=True, expert_prefetch_staging=2, two_read_experts=True)
    batch = fixture.moe_batch(build.config, seed=3, rows=2, seq=6)
    counts = {}
    for name, moe in (("one", one), ("two", two)):
        pool = moe.runtime.expert_pool
        before = pool.fetches, pool.logical_bytes_read
        run_moe_pass(moe.model, batch)
        counts[name] = (pool.fetches - before[0], pool.logical_bytes_read - before[1])
    assert counts["one"] == counts["two"]
    assert counts["one"][0] > 0
