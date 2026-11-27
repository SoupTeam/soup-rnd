from benchmarks.harness.abba_cache import (
    CACHE_BYTES,
    abba_cache_argv,
    abba_cache_plan,
)


def test_plan_has_four_items_per_block_in_abba_order():
    plan = abba_cache_plan(blocks=2)
    assert len(plan) == 8
    for block in (0, 1):
        items = plan[block * 4:block * 4 + 4]
        assert [item["block"] for item in items] == [block] * 4
        assert [item["host_cache_bytes"] for item in items] == [0, CACHE_BYTES, CACHE_BYTES, 0]


def test_items_carry_arm_window_and_kind():
    for item in abba_cache_plan(blocks=1):
        assert item["arm"] == "G"
        assert item["k"] == 8
        assert item["seq"] == 64
        assert item["kind"] == "speed"


def test_cache_bytes_are_five_gib_on_b_and_zero_on_a():
    assert CACHE_BYTES == 5368709120
    plan = abba_cache_plan(blocks=1)
    assert {item["host_cache_bytes"] for item in plan[1:3]} == {CACHE_BYTES}
    assert {item["host_cache_bytes"] for item in (plan[0], plan[3])} == {0}


def test_indexes_are_unique_and_increasing():
    indexes = [item["index"] for item in abba_cache_plan(blocks=5)]
    assert indexes == list(range(20))


def test_block_numbering_continues_from_start_block():
    plan = abba_cache_plan(blocks=2, start_block=5)
    assert [item["block"] for item in plan] == [5, 5, 5, 5, 6, 6, 6, 6]
    assert [item["index"] for item in plan] == list(range(20, 28))


def test_a_item_argv_has_no_host_cache_flag():
    item = abba_cache_plan(blocks=1)[0]
    argv = abba_cache_argv(item, source="/m", cache="/c", out="/o", lora_targets=("q_proj",))
    assert argv[0] == "run"
    assert "--host-cache-bytes" not in argv
    assert argv[argv.index("--arm") + 1] == "G"
    assert argv[argv.index("--k") + 1] == "8"
    assert argv[argv.index("--seq") + 1] == "64"
    assert argv[argv.index("--kind") + 1] == "speed"


def test_b_item_argv_passes_the_cache_budget():
    item = abba_cache_plan(blocks=1)[1]
    argv = abba_cache_argv(item, source="/m", cache="/c", out="/o", lora_targets=("q_proj",))
    assert argv[argv.index("--host-cache-bytes") + 1] == "5368709120"
