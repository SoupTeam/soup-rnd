"""CPU tests for the A4 activation-budget planner (benchmarks/harness/a4_activation_budget.py).

The planner decides which L2L boundary activations stay pinned and which spill,
in REAL host bytes. These tests pin its boundaries (exact fit, one byte short,
k=1, everything spilled), the refusals and the arithmetic they carry, the
page-lock rounding, and that its allocation model matches what
``l2l_activations.ActivationStore`` really allocates.
"""

from __future__ import annotations

import math

import pytest

from tests.test_l2l_probe_harness import _load

MIB = 1 << 20
SECTOR = 4096
# Mistral-7B at seq 512 in bf16: one chunk is 512 x 4096 x 2 B = 4 MiB.
MISTRAL = dict(n_layers=32, chunk_bytes=512 * 4096 * 2)


@pytest.fixture()
def bud():
    return _load("a4_activation_budget")


def _layout(bud, **over):
    kwargs = dict(MISTRAL, k=4, ring=8, pin=True)
    kwargs.update(over)
    return bud.StoreLayout(**kwargs)


def test_pinned_rounding_is_the_next_power_of_two(bud):
    assert bud.pinned_block_bytes(0) == 0
    assert bud.pinned_block_bytes(1) == 1
    assert bud.pinned_block_bytes(4096) == 4096
    assert bud.pinned_block_bytes(4097) == 8192
    assert bud.pinned_block_bytes(8 * MIB) == 8 * MIB
    # A staging buffer for an 8 MiB chunk asks for one sector more (#1531): 16 MiB page-locked.
    assert bud.pinned_block_bytes(8 * MIB + SECTOR) == 16 * MIB
    assert bud.allocation_cost(3 * MIB, pin=True) == 4 * MIB
    assert bud.allocation_cost(3 * MIB, pin=False) == 3 * MIB


def test_the_minimum_is_the_rings_at_their_rounded_size(bud):
    layout = _layout(bud)
    # 2 rings x 8 buffers x (4 MiB + 4096 B, page-locked as 8 MiB).
    assert layout.minimum_bytes == 2 * 8 * 8 * MIB
    assert layout.all_pinned_bytes == 32 * 4 * 4 * MIB
    assert layout.host_bytes(10) == 22 * 4 * 4 * MIB + 2 * 8 * 8 * MIB


def test_exact_fit_pins_everything_and_one_byte_short_spills(bud):
    layout = _layout(bud)
    exact = bud.plan_activations(layout.all_pinned_bytes, **MISTRAL, k=4)
    assert exact.spill_layers == 0
    assert exact.host_bytes == layout.all_pinned_bytes
    assert exact.spill_file_bytes == 0
    short = bud.plan_activations(layout.all_pinned_bytes - 1, **MISTRAL, k=4)
    assert short.spill_layers > 0
    assert short.host_bytes <= layout.all_pinned_bytes - 1


@pytest.mark.parametrize("resident", [0, 1, 5, 23])
def test_resident_layers_are_exactly_what_the_budget_buys(bud, resident):
    layout = _layout(bud)
    per_layer = 4 * 4 * MIB
    budget = layout.minimum_bytes + resident * per_layer
    plan = bud.plan_activations(budget, **MISTRAL, k=4)
    assert plan.resident_layers == resident
    assert plan.host_bytes == budget
    below = bud.plan_activations(budget - 1, **MISTRAL, k=4) if resident else None
    if below is not None:
        assert below.resident_layers == resident - 1


def test_a_budget_past_all_pinned_minus_the_rings_pins_everything(bud):
    """Rings (128 MiB) + 24 layers (384 MiB) = all-pinned (512 MiB): from there on the
    rings cost what the spilled layers would, so nothing spills."""
    layout = _layout(bud)
    plan = bud.plan_activations(layout.minimum_bytes + 31 * 16 * MIB, **MISTRAL, k=4)
    assert plan.spill_layers == 0
    assert plan.host_bytes == layout.all_pinned_bytes
    assert plan.unused_bytes == layout.minimum_bytes + 31 * 16 * MIB - layout.all_pinned_bytes


def test_the_lowest_layers_spill_first(bud):
    """Written first in the forward and read last in the backward: the most compute to
    hide their I/O behind."""
    layout = _layout(bud)
    plan = bud.plan_activations(layout.minimum_bytes + 3 * 16 * MIB, **MISTRAL, k=4)
    assert plan.spill_layers == 29
    assert plan.placement(0, 0) == "spill"
    assert plan.placement(28, 3) == "spill"
    assert plan.placement(29, 0) == "pinned"
    assert plan.spilled_chunks()[:2] == [(0, 0), (0, 1)]
    assert len(plan.spilled_chunks()) + len(plan.pinned_chunks()) == 32 * 4
    with pytest.raises(IndexError):
        plan.placement(32, 0)


def test_every_layer_spilled_at_the_minimum(bud):
    layout = _layout(bud)
    plan = bud.plan_activations(layout.minimum_bytes, **MISTRAL, k=4)
    assert plan.spill_layers == 32
    assert plan.pinned_chunks() == []
    assert plan.spill_file_bytes == 32 * 4 * 4 * MIB
    assert plan.host_bytes == layout.minimum_bytes


def test_below_the_minimum_is_refused_with_the_arithmetic(bud):
    layout = _layout(bud)
    with pytest.raises(bud.BudgetRefusalError) as info:
        bud.plan_activations(layout.minimum_bytes - 1, **MISTRAL, k=4)
    err = info.value
    assert err.code == bud.BUDGET_BELOW_MINIMUM
    text = str(err)
    assert f"{layout.minimum_bytes - 1:,} B" in text
    assert f"{layout.minimum_bytes:,} B" in text
    assert "2 staging rings x 8 buffers" in text
    assert f"{4 * MIB + SECTOR:,} B request" in text
    assert "1 B (0.0 MiB) short" in text


def test_k1_with_deep_rings_cannot_save_ram_by_spilling(bud):
    """At k=1 the two rings (16 x 8 MiB) cost what all 32 layers cost pinned (32 x 4 MiB):
    a budget below all-pinned is refused rather than planned into a bigger store."""
    layout = _layout(bud, k=1)
    assert layout.minimum_bytes == layout.all_pinned_bytes == 128 * MIB
    with pytest.raises(bud.BudgetRefusalError) as info:
        bud.plan_activations(layout.all_pinned_bytes - 1, **MISTRAL, k=1)
    assert info.value.code == bud.BUDGET_BELOW_MINIMUM
    shallow = bud.plan_activations(64 * MIB, **MISTRAL, k=1, ring=2)
    assert shallow.host_bytes <= 64 * MIB
    assert shallow.spill_layers == 32 - (64 - 32) // 4


def test_a_cheaper_all_pinned_store_wins_over_the_spill_minimum(bud):
    plan = bud.plan_activations(128 * MIB, **MISTRAL, k=1)
    assert plan.spill_layers == 0


def test_rounding_decides_how_many_layers_fit(bud):
    """A 3 MiB chunk pins as 4 MiB: the pinned plan keeps fewer layers than a pageable one."""
    common = dict(n_layers=8, k=2, chunk_bytes=3 * MIB, ring=1)
    budget = 2 * 4 * MIB + 3 * 2 * 3 * MIB  # pageable: rings + 3 layers
    pageable = bud.plan_activations(budget, pin=False, **common)
    pinned = bud.plan_activations(budget, pin=True, **common)
    assert pageable.host_bytes == 2 * (3 * MIB + SECTOR) + pageable.resident_layers * 6 * MIB
    assert pageable.resident_layers == 3
    assert pinned.resident_layers < pageable.resident_layers
    assert pinned.host_bytes <= budget


def test_spill_disk_short_is_refused_with_the_arithmetic(bud):
    layout = _layout(bud)
    file_bytes = 32 * 4 * 4 * MIB
    ok = bud.plan_activations(layout.minimum_bytes, **MISTRAL, k=4, spill_free_bytes=file_bytes)
    assert ok.spill_file_bytes == file_bytes
    with pytest.raises(bud.BudgetRefusalError) as info:
        bud.plan_activations(layout.minimum_bytes, **MISTRAL, k=4,
                             spill_free_bytes=file_bytes + MIB - 1, spill_reserve_bytes=MIB)
    assert info.value.code == bud.SPILL_DISK_SHORT
    text = str(info.value)
    assert "32 layers x k=4 x 4,194,304 B" in text
    assert f"{file_bytes + MIB:,} B" in text
    assert "1 B (0.0 MiB) short" in text


def test_an_all_pinned_plan_needs_no_spill_space(bud):
    layout = _layout(bud)
    plan = bud.plan_activations(layout.all_pinned_bytes, **MISTRAL, k=4, spill_free_bytes=0)
    assert plan.spill_layers == 0


def test_an_unaligned_chunk_cannot_spill_but_can_stay_pinned(bud):
    common = dict(n_layers=4, k=2, chunk_bytes=SECTOR * 10 + 2, ring=1)
    layout = bud.StoreLayout(**common)
    assert bud.plan_activations(layout.all_pinned_bytes, **common).spill_layers == 0
    with pytest.raises(bud.BudgetRefusalError) as info:
        bud.plan_activations(layout.minimum_bytes, **common)
    assert info.value.code == bud.UNALIGNED_SPILL
    assert "2 B past a multiple of the 4096 B sector" in str(info.value)


def test_the_floor_is_one_slab_paid_whether_the_payload_spills_or_not(bud):
    """G4's per-row scales stay pinned when the payload spills, in ONE slab of
    n_layers x k x floor (g4_activation_codec.py:461), rounded once."""
    common = dict(n_layers=4, k=2, chunk_bytes=64 * SECTOR, pinned_floor_bytes=1000, ring=1)
    layout = bud.StoreLayout(**common)
    # The slab asks for 8 x 1000 = 8,000 B and page-locks 8,192 B, once.
    assert layout.floor_slab_bytes == 8000
    assert layout.minimum_bytes == 2 * bud.pinned_block_bytes(65 * SECTOR) + 8192
    assert layout.all_pinned_bytes == 8 * 64 * SECTOR + 8192
    plan = bud.plan_activations(layout.minimum_bytes, **common)
    assert plan.spill_layers == 4
    assert plan.as_dict()["allocations"] == [
        {"kind": "floor_slab", "request_bytes": 8000, "count": 1, "host_bytes_each": 8192},
        {"kind": "ring_buffer", "request_bytes": 65 * SECTOR, "count": 2,
         "host_bytes_each": 128 * SECTOR},
    ]
    assert bud.StoreLayout(**{**common, "pin": False}).minimum_bytes == 2 * 65 * SECTOR + 8000


def test_a_power_of_two_scale_chunk_in_a_slab_that_is_not_rounds_up(bud):
    """G4Review's case: 30 layers, k=4, seq 256 -> 1,024 B of scales per chunk. Per chunk
    that is a power of two (120 x 1,024 = 122,880 B), but the real slab is one 122,880 B
    request, which page-locks 131,072 B."""
    layout = bud.StoreLayout(n_layers=30, k=4, chunk_bytes=256 * 576, pinned_floor_bytes=1024,
                             ring=2, pin=True)
    (slab,) = [a for a in layout.allocations(0) if a[0] == "floor_slab"]
    assert slab == ("floor_slab", 122_880, 1)
    assert layout.all_pinned_bytes == 120 * bud.pinned_block_bytes(256 * 576) + 131_072


def test_worked_mistral_7b_int8_example_by_hand(bud):
    """Mistral-7B, seq 512, hidden 4096, 32 layers, G4 int8 at k = 4, ring 4, pinned.

    Per chunk, unrounded (G4): payload 512 x 4096 x 1 B = 2,097,152 B (2 MiB), scales
    512 rows x 4 B = 2,048 B.

    Pinned sizes, by hand:
      resident payload chunk 2,097,152 B, already 2^21          -> 2,097,152 B each
      ring buffer 2,097,152 + 4,096 = 2,101,248 B, next 2^22     -> 4,194,304 B each
      scale slab 32 x 4 x 2,048 = 262,144 B = 2^18, rounded ONCE -> 262,144 B
    all-pinned  = 128 x 2 MiB + 256 KiB              = 268,435,456 + 262,144 = 268,697,600
    minimum     = 2 rings x 4 x 4 MiB + 256 KiB       =  33,554,432 + 262,144 =  33,816,576
    half budget = 268,697,600 // 2 = 134,348,800. Above the minimum: 100,532,224 B;
    one resident layer costs 4 x 2 MiB = 8,388,608 B, so 11 layers stay (92,274,688 B),
    21 spill. host = 33,816,576 + 92,274,688 = 126,091,264 B (<= the budget);
    spill file = 21 x 4 x 2,097,152 = 176,160,768 B, a multiple of 4096.
    """
    payload, scales = 512 * 4096, 512 * 4
    layout = bud.StoreLayout(n_layers=32, k=4, chunk_bytes=payload, pinned_floor_bytes=scales,
                             ring=4, pin=True)
    assert layout.all_pinned_bytes == 268_697_600
    assert layout.minimum_bytes == 33_816_576
    plan = bud.plan_activations(bud.arm_budget_bytes(layout, "0.5"), n_layers=32, k=4,
                                chunk_bytes=payload, pinned_floor_bytes=scales, ring=4)
    assert plan.budget_bytes == 134_348_800
    assert (plan.resident_layers, plan.spill_layers) == (11, 21)
    assert plan.host_bytes == 126_091_264
    assert plan.spill_file_bytes == 176_160_768 and plan.spill_file_bytes % SECTOR == 0
    assert plan.requested_bytes == 44 * payload + 8 * (payload + SECTOR) + 32 * 4 * scales
    # The same budget in bf16 (4 MiB chunks, no scales): 4 MiB + 4096 rounds to 8 MiB.
    bf16 = bud.StoreLayout(n_layers=32, k=4, chunk_bytes=2 * payload, ring=4)
    assert bf16.minimum_bytes == 8 * 8 * MIB
    assert bud.plan_activations(134_348_800, n_layers=32, k=4, chunk_bytes=2 * payload,
                                ring=4).resident_layers == (134_348_800 - 64 * MIB) // (16 * MIB)


def test_the_priority_order_charges_each_term_before_the_next(bud):
    """OS reserve -> process memory (pageable + page-locked weights) -> minimum rings ->
    resident activations -> C3 spare. Each test budget moves one boundary."""
    layout = bud.StoreLayout(n_layers=8, k=2, chunk_bytes=MIB, ring=1, pin=True)
    ring_min = layout.minimum_bytes          # 2 x 2 MiB (1 MiB + 4096 -> 2 MiB)
    per_layer = 2 * MIB
    assert ring_min == 4 * MIB and layout.all_pinned_bytes == 16 * MIB
    reserve, process, weights = 100 * MIB, 50 * MIB, 30 * MIB
    head = reserve + process + weights

    def split(total, **kw):
        return bud.split_host_ram(total, reserve_bytes=reserve, process_bytes=process,
                                  weight_pinned_bytes=weights, layout=layout, **kw)

    # 1-3: one byte short of reserve + process + weights + the rings: refused.
    with pytest.raises(bud.BudgetRefusalError) as info:
        split(head + ring_min - 1)
    assert info.value.code == bud.RAM_OVER_BUDGET
    # 3: exactly the rings: every layer spills, nothing spare.
    at_min = split(head + ring_min)
    assert at_min.plan.spill_layers == 8 and at_min.spare_bytes == 0
    # 4: rings + 3 layers: 3 resident, still nothing spare.
    three = split(head + ring_min + 3 * per_layer)
    assert three.plan.resident_layers == 3 and three.spare_bytes == 0
    # 4 -> 5: all-pinned fits, and everything past it is spare for C3.
    rich = split(head + layout.all_pinned_bytes + 7 * MIB)
    assert rich.plan.spill_layers == 0 and rich.spare_bytes == 7 * MIB
    # A rule arm's cap bounds the activations; what it frees goes to spare, not to them.
    capped = split(head + layout.all_pinned_bytes + 7 * MIB, activation_cap_bytes=ring_min)
    assert capped.plan.spill_layers == 8
    assert capped.spare_bytes == layout.all_pinned_bytes + 7 * MIB - ring_min
    assert capped.peak_pinned_bytes == weights + ring_min


def test_granularity_waste_is_under_one_layer(bud):
    layout = _layout(bud)
    budget = layout.minimum_bytes + 7 * 16 * MIB + 3 * 4 * MIB + 5
    plan = bud.plan_activations(budget, **MISTRAL, k=4)
    assert plan.resident_layers == 7
    assert plan.granularity_waste_bytes == 3 * 4 * MIB
    assert plan.granularity_waste_bytes <= (plan.k - 1) * 4 * MIB


def test_invalid_inputs_are_refused(bud):
    for bad in (dict(k=0), dict(n_layers=0), dict(chunk_bytes=0), dict(ring=0),
                dict(k=True), dict(chunk_bytes=1.5)):
        with pytest.raises(ValueError):
            bud.StoreLayout(**{**MISTRAL, "k": 4, **bad})
    with pytest.raises(ValueError):
        bud.plan_activations(-1, **MISTRAL, k=4)


def test_the_layout_matches_what_the_real_store_allocates(bud):
    """The planner's request sizes are ActivationStore's: resident chunks, and two rings
    of chunk+sector buffers only when something spills."""
    import torch

    acts = _load("l2l_activations")
    n_layers, k, shape = 4, 3, (1, 8, 128)
    chunk = math.prod(shape) * 4
    layout = bud.StoreLayout(n_layers=n_layers, k=k, chunk_bytes=chunk, ring=2, pin=False)
    for spill in (0, 1, 4):
        store = acts.ActivationStore(
            n_layers=n_layers, k=k, chunk_shape=shape, dtype=torch.float32, device="cpu",
            pin=False, spill_layers=spill, ring=2,
            spill_factory=(lambda size: _NullSpill()) if spill else None,
        )
        try:
            held = sum(t.untyped_storage().nbytes() for t in store._resident.values())
            if spill:
                for ring in (store._write_ring, store._read_ring):
                    held += sum(buf.untyped_storage().nbytes() for buf, _ in list(ring._free.queue))
            assert held == layout.host_bytes(spill), spill
        finally:
            store.close()


class _NullSpill:
    direct = False
    path = "<null>"
    left_behind = None

    def write(self, offset, buf):
        pass

    def read(self, offset, buf):
        pass

    def close(self, remove=True):
        pass


def test_split_gives_weights_first_then_activations_then_spare(bud):
    layout = _layout(bud)
    gib = 1 << 30
    split = bud.split_host_ram(16 * gib, reserve_bytes=4 * gib, process_bytes=3 * gib,
                               weight_pinned_bytes=4 * gib, layout=layout)
    assert split.plan.spill_layers == 0
    assert split.activation_budget_bytes == layout.all_pinned_bytes
    assert split.spare_bytes == 16 * gib - 11 * gib - layout.all_pinned_bytes
    assert split.peak_pinned_bytes == 4 * gib + layout.all_pinned_bytes
    capped = bud.split_host_ram(16 * gib, reserve_bytes=4 * gib, process_bytes=3 * gib,
                                weight_pinned_bytes=4 * gib, layout=layout,
                                activation_cap_bytes=layout.minimum_bytes)
    assert capped.plan.spill_layers == 32
    assert capped.spare_bytes == 16 * gib - 11 * gib - layout.minimum_bytes


def test_split_spills_when_the_weights_leave_too_little(bud):
    layout = _layout(bud)
    total = 1000 * MIB
    left = layout.minimum_bytes + 2 * 16 * MIB
    split = bud.split_host_ram(total, reserve_bytes=100 * MIB, process_bytes=0,
                               weight_pinned_bytes=total - 100 * MIB - left, layout=layout)
    assert split.plan.resident_layers == 2
    assert split.spare_bytes == 0


def test_split_refuses_with_the_whole_budget_in_the_message(bud):
    layout = _layout(bud)
    with pytest.raises(bud.BudgetRefusalError) as info:
        bud.split_host_ram(1000 * MIB, reserve_bytes=500 * MIB, process_bytes=200 * MIB,
                           weight_pinned_bytes=400 * MIB, layout=layout)
    assert info.value.code == bud.RAM_OVER_BUDGET
    text = str(info.value)
    assert "500.0 MiB) reserve" in text and "400.0 MiB) page-locked weights" in text
    assert "100.0 MiB) over before any activation" in text


def test_arm_budgets(bud):
    layout = _layout(bud)
    assert bud.arm_budget_bytes(layout, "1") == layout.all_pinned_bytes
    assert bud.arm_budget_bytes(layout, "0.5") == layout.all_pinned_bytes // 2
    assert bud.arm_budget_bytes(layout, "min") == layout.minimum_bytes
    with pytest.raises(ValueError):
        bud.arm_budget_bytes(layout, "0")


def test_max_k_without_spill_is_the_pinned_bound(bud):
    budget = 1 << 30
    k, bound = bud.max_k(budget, **MISTRAL, spill=False)
    assert k == budget // (32 * 4 * MIB) == 8
    assert bound == "all-pinned host bytes"


def test_max_k_with_spill_is_bounded_by_the_spill_volume(bud):
    budget = 1 << 30
    free = 64 << 30
    k, bound = bud.max_k(budget, **MISTRAL, spill=True, spill_free_bytes=free)
    assert bound == "spill volume free space"
    assert bud.plan_activations(budget, **MISTRAL, k=k, spill_free_bytes=free)
    with pytest.raises(bud.BudgetRefusalError):
        bud.plan_activations(budget, **MISTRAL, k=k + 1, spill_free_bytes=free)
    unbounded, why = bud.max_k(budget, **MISTRAL, spill=True, k_limit=4096)
    assert (unbounded, why) == (4096, "k_limit")


def test_tokens_per_step_reports_both_sides(bud):
    out = bud.tokens_per_step(1 << 30, **MISTRAL, seq=512, spill_free_bytes=64 << 30)
    assert out["without_spill"] == {"k": 8, "tokens": 4096, "bound": "all-pinned host bytes"}
    k = out["with_spill"]["k"]
    assert out["with_spill"]["tokens"] == 512 * k
    assert out["with_spill"]["spill_file_bytes"] <= 64 << 30
    assert k > 8


def test_budget_from_box_takes_the_smaller_of_ram_and_commit(bud):
    assert bud.budget_from_box(10_000, None, 1_000) == 9_000
    assert bud.budget_from_box(10_000, 6_000, 1_000) == 5_000
    assert bud.budget_from_box(500, None, 1_000) == 0
    with pytest.raises(bud.BudgetRefusalError):
        bud.budget_from_box(None, 6_000, 1_000)
