"""The L2L activation budget: psutil and the whole cgroup chain, on fake v2 and v1 trees.

Seam: ``soup_cli.utils.l2l_spill.ram_budget`` (and ``plan_spill`` on its result). The roots it
reads are module constants, pointed here at trees under ``tmp_path``; psutil is the real
``layer_stream.free_ram_bytes`` lookup, monkeypatched.
"""

from __future__ import annotations

import logging

import pytest

GIB = 2**30
#: psutil's answer when the test wants the cgroup to decide.
PLENTY = 10**15
#: Write a directory in place of a file: it exists and cannot be read.
DIRECTORY = object()


def _stat(inactive_file=0, shmem=0, *, v1=False):
    key = "total_inactive_file" if v1 else "inactive_file"
    return f"anon 0\nfile 0\n{'total_shmem' if v1 else 'shmem'} {shmem}\n{key} {inactive_file}\n"


def cgroup_tree(tmp_path, monkeypatch, levels, *, v1=False, own="/user.slice/run.scope",
                psutil=PLENTY, name="tree"):
    """A fake /proc/self/cgroup naming ``own`` and a cgroup mount whose directories hold the
    files in ``levels`` ({"" (root) | "user.slice" | ...: {file: text | DIRECTORY}}).
    Returns the mount root as the product spells level paths (root + "/part"...)."""
    from soup_cli.utils import l2l_spill

    base = tmp_path / name
    proc = base / "proc" / "self"
    proc.mkdir(parents=True)
    line = f"9:cpuset,memory:{own}" if v1 else f"0::{own}"
    (proc / "cgroup").write_text(f"12:pids:{own}\n{line}\n", encoding="utf-8")
    root = base / ("v1" if v1 else "v2")
    root.mkdir()
    for rel, files in levels.items():
        directory = root / rel if rel else root
        directory.mkdir(parents=True, exist_ok=True)
        for file_name, content in files.items():
            if content is DIRECTORY:
                (directory / file_name).mkdir()
            else:
                (directory / file_name).write_text(str(content), encoding="utf-8")
    monkeypatch.setattr(l2l_spill, "PROC_ROOT", str(base / "proc"))
    monkeypatch.setattr(l2l_spill, "CGROUP_V2_ROOT", str(base / "v2"))
    monkeypatch.setattr(l2l_spill, "CGROUP_V1_ROOT", str(base / "v1"))
    monkeypatch.setattr("soup_cli.utils.layer_stream.free_ram_bytes", lambda: psutil)
    return str(root)


def _v2(limit="max", current=0, *, high=None, stat=None):
    files = {"memory.max": limit, "memory.current": current,
             "memory.stat": _stat() if stat is None else stat}
    if high is not None:
        files["memory.high"] = high
    return files


def _budget():
    from soup_cli.utils.l2l_spill import ram_budget

    return ram_budget()


# --------------------------------------------------------------------------
# the share: 70%, or what the reserve leaves, whichever is smaller
# --------------------------------------------------------------------------
MIB = 2**20
#: Where 0.7 x available and available - reserve meet: reserve / 0.3, about 1.79 GB.
SWITCH = 512 * MIB * 10 // 3


@pytest.mark.parametrize("available, budget, rule", [
    (4 * GIB, int(4 * GIB * 0.7), "70% of"),
    (SWITCH + MIB, int((SWITCH + MIB) * 0.7), "70% of"),  # above it: the 70% side
    (SWITCH - MIB, SWITCH - MIB - 512 * MIB, "reserve"),  # below it: the reserve side
    (840 * MIB, 328 * MIB, "reserve"),  # A3's calibrated limit for 16 of Mistral's 32 layers
    (512 * MIB, 0, "reserve"),
    (100 * MIB, 0, "floored at 0"),
])
def test_the_budget_never_leaves_less_than_the_reserve(tmp_path, monkeypatch, available,
                                                       budget, rule):
    from soup_cli.utils.l2l_spill import RAM_RESERVE_BYTES

    assert RAM_RESERVE_BYTES == 512 * MIB
    cgroup_tree(tmp_path, monkeypatch, {}, psutil=available)
    seen = _budget()
    assert seen.available_bytes == available
    assert seen.budget_bytes == budget
    assert available - seen.budget_bytes >= min(available, RAM_RESERVE_BYTES)
    assert rule in seen.describe()


def test_the_refusal_threshold_is_the_reserve_plus_every_layer_spilled(tmp_path, monkeypatch):
    """Mistral-7B at k 4, seq 512, pinned: every layer spilled still needs the 64 MiB of
    staging rings, so below 512 + 64 MiB available the run stops and names the reserve."""
    from soup_cli.utils.l2l_spill import SpillRefusedError, plan_spill

    def plan(available):
        cgroup_tree(tmp_path, monkeypatch, {}, psutil=available, name=f"t{available}")
        budget = _budget()
        return plan_spill(budget.budget_bytes, n_layers=32, k=4, chunk_bytes=4 * MIB,
                          pin=True, budget_source=budget.describe())

    assert plan(576 * MIB).spill_layers == 32
    with pytest.raises(SpillRefusedError) as caught:
        plan(576 * MIB - 1)
    message = str(caught.value)
    assert "512 MiB reserve" in message and "67.1 MB" in message
    for fix in ("gradient_accumulation_steps", "data.max_length", "batch_size", "free RAM"):
        assert fix in message


def test_session_1s_check_child_is_refused_instead_of_spilling(tmp_path, monkeypatch):
    """A3 stage 2 session 1: 0.43 GB available at the plan point. The 0.7 share planned 18
    of 32 layers and the child was OOM-killed in its first step; now the run stops."""
    from soup_cli.utils.l2l_spill import SpillRefusedError, plan_spill

    cgroup_tree(tmp_path, monkeypatch, {}, psutil=430_000_000)
    budget = _budget()
    assert budget.budget_bytes == 0
    with pytest.raises(SpillRefusedError):
        plan_spill(budget.budget_bytes, n_layers=32, k=4, chunk_bytes=4 * MIB, pin=True)

# --------------------------------------------------------------------------
# v2
# --------------------------------------------------------------------------
def test_a_child_without_a_limit_is_limited_by_its_parent(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_spill import plan_spill

    root = cgroup_tree(tmp_path, monkeypatch, {
        "": {},
        "user.slice": _v2(limit=4 * GIB, current=1 * GIB),
        "user.slice/run.scope": _v2(limit="max", current=1 * GIB),
    })
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice"
    assert budget.available_bytes == 3 * GIB
    assert budget.budget_bytes == int(3 * GIB * 0.7)
    assert budget.limiting_numbers == {
        "limit": 4 * GIB, "limit_file": "memory.max", "used": 1 * GIB, "inactive_file": 0,
        "remainder": 3 * GIB,
    }
    # a store above that budget spills; psutil alone (10^15 bytes free) would keep it resident
    plan = plan_spill(budget.budget_bytes, n_layers=40, k=8, chunk_bytes=16 * 2**20, pin=False)
    assert plan.all_resident_bytes > budget.budget_bytes
    assert plan.spill_layers > 0
    assert plan.host_bytes <= budget.budget_bytes


def test_a_parent_with_a_larger_limit_but_a_smaller_remainder_decides(tmp_path, monkeypatch):
    root = cgroup_tree(tmp_path, monkeypatch, {
        "user.slice": _v2(limit=8 * GIB, current=7 * GIB),  # nearly full of siblings
        "user.slice/run.scope": _v2(limit=4 * GIB, current=1 * GIB),
    })
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice"
    assert budget.available_bytes == 1 * GIB


def test_the_own_group_decides_when_it_is_the_tightest(tmp_path, monkeypatch):
    root = cgroup_tree(tmp_path, monkeypatch, {
        "user.slice": _v2(limit=16 * GIB, current=2 * GIB),
        "user.slice/run.scope": _v2(limit=4 * GIB, current=2 * GIB),
    })
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice/run.scope"
    assert budget.available_bytes == 2 * GIB


def test_without_any_limit_psutil_decides(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, {
        "": {"memory.current": 5 * GIB},
        "user.slice": _v2(),
        "user.slice/run.scope": _v2(),
    }, psutil=6 * GIB)
    budget = _budget()
    assert budget.limiting_level == "psutil"
    assert budget.limiting_numbers == {"psutil_available": 6 * GIB}
    assert budget.budget_bytes == int(6 * GIB * 0.7)
    assert "psutil available" in budget.describe()


def test_psutil_decides_when_it_is_below_every_remainder(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, {"user.slice": _v2(limit=8 * GIB)}, psutil=1 * GIB)
    assert _budget().limiting_level == "psutil"


def test_without_psutil_and_without_a_limit_no_budget_is_known(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_spill import plan_spill

    cgroup_tree(tmp_path, monkeypatch, {"user.slice": _v2()}, psutil=None)
    budget = _budget()
    assert budget.budget_bytes is None and budget.limiting_level is None
    # nothing spills: everything stays resident, as before the spill existed
    assert plan_spill(None, n_layers=80, k=16, chunk_bytes=2**30, pin=True).spill_layers == 0


def test_without_psutil_a_cgroup_limit_still_gives_a_budget(tmp_path, monkeypatch):
    root = cgroup_tree(tmp_path, monkeypatch, {"user.slice": _v2(limit=2 * GIB)}, psutil=None)
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice"
    assert budget.budget_bytes == int(2 * GIB * 0.7)


@pytest.mark.parametrize("broken", ["garbage", DIRECTORY])
def test_an_unreadable_limit_is_not_no_limit(tmp_path, monkeypatch, caplog, broken):
    root = cgroup_tree(tmp_path, monkeypatch, {
        "user.slice": _v2(limit=3 * GIB, current=1 * GIB),
        "user.slice/run.scope": {**_v2(current=1 * GIB), "memory.max": broken},
    })
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.l2l_spill"):
        budget = _budget()
    own = root + "/user.slice/run.scope"
    assert budget.limiting_level == root + "/user.slice"
    assert budget.available_bytes == 2 * GIB
    assert [row["level"] for row in budget.failed_levels] == [own]
    assert any(own in record.getMessage() for record in caplog.records)
    assert own in budget.describe()


def test_inactive_file_cache_does_not_shrink_the_budget(tmp_path, monkeypatch):
    levels = {"user.slice/run.scope": _v2(limit=8 * GIB, current=2 * GIB, stat=_stat(0))}
    cgroup_tree(tmp_path, monkeypatch, levels, name="clean")
    clean = _budget()
    cached = {"user.slice/run.scope": _v2(limit=8 * GIB, current=6 * GIB, stat=_stat(4 * GIB))}
    cgroup_tree(tmp_path, monkeypatch, cached, name="cached")
    full_of_cache = _budget()
    assert clean.available_bytes == full_of_cache.available_bytes == 6 * GIB
    assert full_of_cache.limiting_numbers["inactive_file"] == 4 * GIB


def test_shmem_shrinks_the_budget(tmp_path, monkeypatch):
    """Pinned host memory is charged as shmem, on the anon LRU: it raises memory.current
    and never shows up as inactive_file."""
    base = {"user.slice/run.scope": _v2(limit=8 * GIB, current=3 * GIB, stat=_stat(1 * GIB))}
    cgroup_tree(tmp_path, monkeypatch, base, name="before")
    before = _budget()
    pinned = {"user.slice/run.scope": _v2(
        limit=8 * GIB, current=5 * GIB, stat=_stat(1 * GIB, shmem=2 * GIB))}
    cgroup_tree(tmp_path, monkeypatch, pinned, name="after")
    after = _budget()
    assert before.available_bytes == 6 * GIB
    assert after.available_bytes == 4 * GIB
    assert after.budget_bytes < before.budget_bytes


def test_memory_high_below_memory_max_is_the_limit(tmp_path, monkeypatch):
    root = cgroup_tree(tmp_path, monkeypatch, {
        "user.slice/run.scope": _v2(limit=8 * GIB, high=3 * GIB, current=1 * GIB),
    })
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice/run.scope"
    assert budget.limiting_numbers["limit"] == 3 * GIB
    assert budget.limiting_numbers["limit_file"] == "memory.high"
    assert budget.available_bytes == 2 * GIB


def test_memory_high_of_max_leaves_memory_max(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, {
        "user.slice/run.scope": _v2(limit=4 * GIB, high="max", current=1 * GIB),
    })
    assert _budget().limiting_numbers["limit_file"] == "memory.max"


@pytest.mark.parametrize("stat", [DIRECTORY, None])
def test_an_unreadable_memory_stat_counts_limit_minus_used(tmp_path, monkeypatch, caplog, stat):
    files = _v2(limit=8 * GIB, current=5 * GIB)
    if stat is DIRECTORY:
        files["memory.stat"] = DIRECTORY
    else:
        del files["memory.stat"]
    root = cgroup_tree(tmp_path, monkeypatch, {"user.slice/run.scope": files})
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.l2l_spill"):
        budget = _budget()
    own = root + "/user.slice/run.scope"
    assert budget.available_bytes == 3 * GIB
    assert budget.limiting_numbers["inactive_file"] is None
    assert any(own in r.getMessage() and "memory.stat" in r.getMessage() for r in caplog.records)
    assert "limit - used" in budget.describe()


def test_a_level_missing_from_this_mount_namespace_is_skipped(tmp_path, monkeypatch):
    # a container sees its host path in /proc/self/cgroup; only the mount root exists
    root = cgroup_tree(tmp_path, monkeypatch, {"": _v2(limit=2 * GIB)},
                       own="/docker/0123456789ab")
    assert _budget().limiting_level == root


# --------------------------------------------------------------------------
# v1
# --------------------------------------------------------------------------
def _v1(limit, usage=0, inactive=0, shmem=0):
    return {
        "memory.limit_in_bytes": limit,
        "memory.usage_in_bytes": usage,
        "memory.stat": _stat(inactive, shmem, v1=True),
    }


def test_v1_infinite_is_skipped_and_the_finite_parent_applies(tmp_path, monkeypatch):
    infinite = 9223372036854771712
    root = cgroup_tree(tmp_path, monkeypatch, {
        "": _v1(infinite, usage=50 * GIB),
        "user.slice": _v1(3 * GIB, usage=1 * GIB),
        "user.slice/run.scope": _v1(infinite, usage=1 * GIB),
    }, v1=True)
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice"
    assert budget.limiting_numbers["limit_file"] == "memory.limit_in_bytes"
    assert budget.available_bytes == 2 * GIB


def test_v1_total_inactive_file_is_reclaimable_and_shmem_is_not(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, {
        "user.slice/run.scope": _v1(8 * GIB, usage=6 * GIB, inactive=4 * GIB),
    }, v1=True, name="cache")
    assert _budget().available_bytes == 6 * GIB
    cgroup_tree(tmp_path, monkeypatch, {
        "user.slice/run.scope": _v1(8 * GIB, usage=6 * GIB, inactive=1 * GIB, shmem=3 * GIB),
    }, v1=True, name="shmem")
    assert _budget().available_bytes == 3 * GIB
