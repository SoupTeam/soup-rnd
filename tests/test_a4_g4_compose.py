"""CPU tests for the A4 x G4 composition (benchmarks/harness/a4_g4_compose.py).

G4 hands A4 one chunk's payload and scales, unrounded; A4 plans with the payload as
``chunk_bytes`` and the scales as ``pinned_floor_bytes``. These tests pin that contract
on the real G4 byte functions, and that a small real G4 store, built with the plan's
``spill_layers``, allocates exactly what the plan lists and round-trips bit for bit.
They skip when no G4 harness is found (before the tracks merge, the sibling worktree).
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load

SECTOR = 4096


@pytest.fixture()
def compose():
    return _load("a4_g4_compose")


@pytest.fixture()
def g4(compose):
    harness = compose.find_g4_harness()
    if harness is None:
        pytest.skip("no g4_activation_codec.py: neither merged here nor in ../soup-rnd-g4")
    return compose.load_g4(harness)


@pytest.mark.parametrize("shape,codec,element,payload,scale", [
    # Brief section 4: for (1, seq, H), int8/fp8 payload = seq x H, fp32 scales = 4 x seq;
    # bf16 identity payload = 2 x seq x H, no scales; CPU float32 identity at 4 B/element.
    ((1, 512, 4096), "int8", 2, 512 * 4096, 4 * 512),
    ((1, 512, 4096), "fp8_e4m3", 2, 512 * 4096, 4 * 512),
    ((1, 512, 4096), "bf16", 2, 2 * 512 * 4096, 0),
    ((1, 64, 576), "bf16", 4, 4 * 64 * 576, 0),
    # int4 packs two values a byte and pads an odd hidden size with one nibble.
    ((1, 256, 63), "int4", 4, 256 * 32, 4 * 256),
    ((1, 256, 4095), "int4", 4, 256 * 2048, 4 * 256),
    ((1, 64, 577), "int4", 4, 64 * 289, 4 * 64),
])
def test_g4_chunk_sizes_are_the_contract(compose, g4, shape, codec, element, payload, scale):
    sizes = compose.chunk_sizes(g4, shape, codec, element)
    assert sizes == {"payload": payload, "scale": scale, "compressed": payload + scale}


def test_the_planner_gets_the_sizes_unrounded_and_rounds_each_allocation_once(compose, g4):
    """Mistral-7B int8 at k=4 over 32 layers, ring 4, pinned: the 2 MiB payload is not
    rounded again, the ring buffer (2 MiB + 4096) page-locks 4 MiB, and the 32 x 4 scale
    chunks of 2,048 B are one 262,144 B slab."""
    sizes = compose.chunk_sizes(g4, (1, 512, 4096), "int8", 2)
    budget_bytes, plan, refusal = compose.plan_for(sizes, "min", n_layers=32, k=4, ring=4,
                                                   pin=True)
    assert refusal is None and plan.spill_layers == 32
    assert (plan.chunk_bytes, plan.layout.pinned_floor_bytes) == (2 << 20, 2048)
    assert compose.expected_allocations(plan) == {
        "floor_slab": [32 * 4 * 2048], "ring_buffer": [(2 << 20) + SECTOR] * 8}
    assert plan.host_bytes == budget_bytes == 8 * (4 << 20) + 262_144
    assert compose.pinned_equivalent([(2 << 20) + SECTOR] * 8 + [262_144]) == plan.host_bytes


def test_an_unaligned_int4_payload_is_refused_by_both_sides(compose, g4, tmp_path):
    """64 x 289 B = 18,496 B is no sector multiple: A4 refuses UNALIGNED_SPILL and G4's
    store refuses the same spill before it opens a file."""
    row = compose.Row("int4_577", (1, 64, 577), "int4", "float32", 3)
    got = compose.run_spec(g4, row, "min", n_layers=4, ring=1, steps=1,
                           spill_dir=str(tmp_path), backend="buffered")
    assert got["refusal"]["code"] == "UNALIGNED_SPILL"
    assert got["g4_refusal"]["refused"] and not got["g4_refusal"]["opened_file"]
    assert got["ok"], got["checks"]


@pytest.mark.parametrize("codec,dtype,spec", [
    ("int8", "float32", "0.5"), ("int8", "float32", "min"), ("bf16", "float32", "0.5"),
    ("int4", "float32", "min"),
])
def test_a_small_real_g4_store_holds_exactly_the_plan_and_round_trips(compose, g4, tmp_path,
                                                                     codec, dtype, spec):
    """Seq 64 x hidden 128 in float32 (int8 payload 8,192 B, int4 4,096 B, identity
    32,768 B: all sector multiples), 4 layers, k=3, ring 1, buffered spill."""
    row = compose.Row(f"small_{codec}", (1, 64, 128), codec, dtype, 3)
    got = compose.run_spec(g4, row, spec, n_layers=4, ring=1, steps=2,
                           spill_dir=str(tmp_path / "spill"), backend="buffered")
    failed = [name for name, ok in got["checks"].items() if not ok]
    assert got["ok"], failed
    assert got["plan"]["spill_layers"] > 0
    assert got["measured"]["pageable_bytes"] == got["plan"]["host_bytes"] <= got["budget_bytes"]
    assert all(step["written"] == step["read"] == got["plan"]["spill_file_bytes"]
               for step in got["steps"])
    assert got["after_close"]["leftover_spill_files"] == []
