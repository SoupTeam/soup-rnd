"""CPU tests for the pure parts of the ticket-11 harness (benchmarks/harness)."""

from __future__ import annotations

from tests.test_l2l_probe_harness import _load


# --------------------------------------------------------------------------
# arm without layer streaming
# --------------------------------------------------------------------------
def test_counters_of_a_resident_run_are_zero():
    arm = _load("l2l_ticket04_arm")
    assert arm.read_counters(None, None) == {
        "layer_loads": 0, "nf4_restores": 0, "large_loads": 0, "pinned_store_bytes": 0,
    }


def test_counters_of_a_streamed_run_come_from_the_runtime():
    arm = _load("l2l_ticket04_arm")

    class Runtime:
        @staticmethod
        def stats():
            return {"layer_loads": 5, "layer_restores": 3, "large_loads": 1}

    out = arm.read_counters(Runtime(), None)
    assert (out["layer_loads"], out["nf4_restores"], out["large_loads"]) == (5, 3, 1)


# --------------------------------------------------------------------------
# configs of the plain arms
# --------------------------------------------------------------------------
def _body(path):
    return [line for line in path.read_text().splitlines() if not line.startswith("#")]


def test_c1_differs_from_ga_in_the_streaming_lines_only():
    camp = _load("l2l_ticket11_campaign")
    ga = _body(camp.CONFIGS_04 / "ga.yaml")
    c1 = _body(camp.CONFIGS / "c1.yaml")
    removed = [line.strip() for line in ga if line not in c1]
    added = [line for line in c1 if line not in ga]
    assert removed == ["stream_layers: true", "stream_source: ram"]
    assert added == []


def test_c1_and_c2_load_as_plain_runs():
    from soup_cli.config.loader import load_config_from_string

    camp = _load("l2l_ticket11_campaign")
    c1 = load_config_from_string((camp.CONFIGS / "c1.yaml").read_text())
    assert (c1.training.batch_size, c1.training.gradient_accumulation_steps) == (1, 4)
    assert not c1.training.stream_layers and not c1.training.l2l
    c2 = load_config_from_string(camp.c2_config((camp.CONFIGS / "c1.yaml").read_text(), 8, 3))
    assert (c2.training.batch_size, c2.training.gradient_accumulation_steps) == (8, 1)
    assert c2.training.epochs == 3 and not c2.training.stream_layers


def test_c2_epochs_give_enough_warm_steps():
    camp = _load("l2l_ticket11_campaign")
    # 64 rows: batch 8 is 8 steps per epoch, batch 32 is 2.
    assert camp.epochs_for(8, rows=64, warm_steps=8) == 2
    assert camp.epochs_for(32, rows=64, warm_steps=8) == 5
    assert camp.epochs_for(1, rows=64, warm_steps=8) == 1


# --------------------------------------------------------------------------
# batch boundary
# --------------------------------------------------------------------------
def _fits_up_to(limit):
    calls = []

    def run(batch):
        calls.append(batch)
        if batch <= limit:
            return {"ok": True, "peak_allocated": batch * 100, "peak_reserved": batch * 110}
        return {"ok": False, "peak_allocated": 9999, "peak_reserved": 10000}

    return run, calls


def test_boundary_doubles_until_the_first_failure():
    camp = _load("l2l_ticket11_campaign")
    run, calls = _fits_up_to(9)
    out = camp.find_boundary(run)
    assert calls == [2, 4, 8, 16]
    assert out["largest_ok"] == 8 and out["failed_at"] == 16
    assert out["peak_at_largest_ok"] == {"peak_allocated": 800, "peak_reserved": 880}
    assert out["peak_at_failure"] == {"peak_allocated": 9999, "peak_reserved": 10000}


def test_boundary_is_none_when_batch_two_fails():
    camp = _load("l2l_ticket11_campaign")
    run, calls = _fits_up_to(1)
    out = camp.find_boundary(run)
    assert calls == [2]
    assert out["largest_ok"] is None and out["failed_at"] == 2


def test_boundary_stops_at_the_row_count_without_a_failure():
    camp = _load("l2l_ticket11_campaign")
    run, calls = _fits_up_to(10**6)
    out = camp.find_boundary(run, max_batch=16)
    assert calls == [2, 4, 8, 16]
    assert out["largest_ok"] == 16 and out["failed_at"] is None


# --------------------------------------------------------------------------
# blocks of the arms and the repeats rule
# --------------------------------------------------------------------------
def test_block_order_is_a_palindrome_of_the_arms_that_fit():
    camp = _load("l2l_ticket11_campaign")
    assert camp.block_order(["A", "B", "C1", "C2"]) == (
        "A", "B", "C1", "C2", "C2", "C1", "B", "A")
    assert camp.block_order(["A", "B", "C2"]) == ("A", "B", "C2", "C2", "B", "A")


def _runs(order, speeds, outcome="ok"):
    arm_validity = _load("arm_validity")
    return [arm_validity.Run(speeds[name], outcome) for name in order]


def test_pair_block_keeps_the_outer_a_runs_and_the_two_runs_of_the_other_arm():
    camp = _load("l2l_ticket11_campaign")
    arms = ["A", "B", "C1", "C2"]
    order = camp.block_order(arms)
    block = _runs(order, {"A": 10.0, "B": 20.0, "C1": 30.0, "C2": 40.0})
    block[0] = type(block[0])(11.0, "ok")  # first A differs from the last A
    pair = camp.pair_block(block, order, "C1")
    assert [r.value for r in pair] == [11.0, 30.0, 30.0, 10.0]


def test_a_void_run_anywhere_voids_the_block_for_every_pair():
    camp = _load("l2l_ticket11_campaign")
    arm_validity = _load("arm_validity")
    arms = ["A", "B", "C1", "C2"]
    order = camp.block_order(arms)
    block = _runs(order, {"A": 10.0, "B": 20.0, "C1": 30.0, "C2": 40.0})
    block[3] = arm_validity.Run(40.0, arm_validity.VOID)  # a C2 run
    for arm in ("B", "C1", "C2"):
        assert all(r.outcome == arm_validity.VOID for r in camp.pair_block(block, order, arm))


def test_blocks_to_add_is_the_largest_need_over_the_pairs():
    camp = _load("l2l_ticket11_campaign")
    arms = ["A", "B", "C1", "C2"]
    order = camp.block_order(arms)
    # Two blocks, A and B steady, C2 noisy: the pilot of 5 still applies, and C2 asks for more.
    steady = {"A": 100.0, "B": 90.0, "C1": 95.0}
    blocks = [
        _runs(order, {**steady, "C2": 100.0 + 40 * (-1) ** i}) for i in range(5)
    ]
    assert camp.blocks_to_add(blocks, order) > 0
    quiet = [_runs(order, {**steady, "C2": 80.0}) for _ in range(5)]
    quiet = [
        [type(r)(r.value + 0.01 * (i + j % 2), r.outcome) for j, r in enumerate(b)]
        for i, b in enumerate(quiet)
    ]
    assert camp.blocks_to_add(quiet, order) == 0


def test_summary_gives_the_mean_of_each_arm_and_the_interval_against_a():
    camp = _load("l2l_ticket11_campaign")
    order = camp.block_order(["A", "B", "C2"])
    # per block: A, B, C2, C2, B, A speeds
    blocks = [
        [{"A": 10.0 + i, "B": 20.0 + i, "C2": 5.0 + 2 * i}[name] for name in order]
        for i in range(4)
    ]
    out = camp.speed_summary(blocks, order)
    assert out["mean"]["A"] == 11.5 and out["mean"]["B"] == 21.5 and out["mean"]["C2"] == 8.0
    assert out["vs_a"]["B"]["mean_diff"] == 10.0
    assert out["vs_a"]["B"]["interval_95"] == [10.0, 10.0]
    assert out["vs_a"]["C2"]["mean_diff"] == -3.5
