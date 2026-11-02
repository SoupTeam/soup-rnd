"""CPU tests for the host readings (host_box.py): cgroup limits and process memory.

The arithmetic is tested on fixture trees of /proc and the cgroup filesystems.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def box():
    return _load("host_box")


V1_UNLIMITED = "9223372036854771712"


def _limit_of(box, tmp_path, files):
    _tree(tmp_path, files)
    return box.ram_limit(proc_root=str(tmp_path / "proc"), v2_root=str(tmp_path / "cg2"),
                         v1_root=str(tmp_path / "cg1"))


def test_cgroup_dirs_walk_from_the_process_scope_to_the_root(box):
    text = "0::/user.slice/user-1000.slice/run-r1.scope\n"
    assert box.cgroup_dirs(text, root="/cg") == [
        "/cg/user.slice/user-1000.slice/run-r1.scope", "/cg/user.slice/user-1000.slice",
        "/cg/user.slice", "/cg"]
    assert box.cgroup_dirs("0::/\n", root="/cg") == ["/cg"]
    assert box.cgroup_dirs("4:memory:/docker/abc\n", root="/cg") == ["/cg"]


def test_status_parsing_splits_page_locked_from_pageable(box):
    text = "VmHWM:\t 2048 kB\nVmRSS:\t 1024 kB\nRssAnon:\t 512 kB\nRssShmem:\t 256 kB\nName: x\n"
    assert box.parse_status(text) == {"VmHWM": 2 << 20, "VmRSS": 1 << 20,
                                      "RssAnon": 512 << 10, "RssShmem": 256 << 10}


GIB = 1 << 30


def _tree(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


def test_v2_takes_the_tightest_limit_from_the_scope_up_and_reads_its_counters(box, tmp_path):
    """systemd-run puts MemoryMax on the scope; a slice above can be tighter still."""
    scope = "cg2/user.slice/user-1000.slice/run-r1.scope"
    got = _limit_of(box, tmp_path, {
        "proc/self/cgroup": "0::/user.slice/user-1000.slice/run-r1.scope",
        "proc/meminfo": "MemTotal: 16000000 kB\nSwapTotal:  8388604 kB",
        f"{scope}/memory.max": str(24 * GIB),
        f"{scope}/memory.swap.max": "0",
        f"{scope}/memory.current": "1000",
        "cg2/user.slice/user-1000.slice/memory.max": "max",
        "cg2/user.slice/user-1000.slice/memory.swap.max": "max",
        "cg2/user.slice/memory.max": str(20 * GIB),
        "cg2/user.slice/memory.current": "5000",
        "cg2/user.slice/memory.peak": "7000",
    })
    assert got["version"] == 2
    assert got["limit_bytes"] == 20 * GIB
    assert got["source"].endswith("cg2/user.slice/memory.max")
    assert (got["current_bytes"], got["peak_bytes"]) == (5000, 7000)
    assert got["swap_max_bytes"] == 0
    assert got["swap_source"].endswith("run-r1.scope/memory.swap.max")
    assert got["swap_total_bytes"] == 8388604 * 1024


def test_v2_max_everywhere_is_unlimited_and_swap_uncapped(box, tmp_path):
    got = _limit_of(box, tmp_path, {
        "proc/self/cgroup": "0::/system.slice/x.service",
        "proc/meminfo": "SwapTotal:  0 kB",
        "cg2/system.slice/x.service/memory.max": "max",
        "cg2/system.slice/x.service/memory.swap.max": "max",
        "cg2/system.slice/memory.max": "max",
    })
    assert got["limit_bytes"] is None and got["source"] is None
    assert got["swap_max_bytes"] is None and got["swap_total_bytes"] == 0


def test_v2_container_with_its_own_namespace_reads_the_root(box, tmp_path):
    """docker --memory 24g with a private cgroup namespace: the path is / and the
    container's own limit sits at the mount root."""
    got = _limit_of(box, tmp_path, {
        "proc/self/cgroup": "0::/",
        "cg2/memory.max": str(24 * GIB),
        "cg2/memory.swap.max": "0",
    })
    assert got["limit_bytes"] == 24 * GIB and got["swap_max_bytes"] == 0


def test_v1_limit_on_the_process_cgroup_is_found_not_only_the_root(box, tmp_path):
    """A v1 host with the limit on the process's own memory cgroup and the root unlimited.
    The previous ram_limit read only <v1 root>/memory.limit_in_bytes and said unlimited."""
    got = _limit_of(box, tmp_path, {
        "proc/self/cgroup": "11:cpuset:/\n4:blkio,memory:/user.slice/run-r1.scope\n1:name=x:/",
        "cg1/memory.limit_in_bytes": V1_UNLIMITED,
        "cg1/user.slice/memory.limit_in_bytes": V1_UNLIMITED,
        "cg1/user.slice/run-r1.scope/memory.limit_in_bytes": str(24 * GIB),
        "cg1/user.slice/run-r1.scope/memory.memsw.limit_in_bytes": str(24 * GIB),
        "cg1/user.slice/run-r1.scope/memory.usage_in_bytes": "123",
        "cg1/user.slice/run-r1.scope/memory.max_usage_in_bytes": "456",
    })
    assert got["version"] == 1
    assert got["limit_bytes"] == 24 * GIB
    assert got["source"].endswith("run-r1.scope/memory.limit_in_bytes")
    assert (got["current_bytes"], got["peak_bytes"]) == (123, 456)
    assert got["swap_max_bytes"] == 0  # memsw == memory: no swap on top


def test_v1_docker_reads_the_container_root_and_its_swap_allowance(box, tmp_path):
    """docker --memory 24g --memory-swap 32g on v1: the host path is listed, the
    container's cgroup is mounted at the root, and 8 GiB of swap come on top."""
    got = _limit_of(box, tmp_path, {
        "proc/self/cgroup": "4:memory:/docker/abc123",
        "cg1/memory.limit_in_bytes": str(24 * GIB),
        "cg1/memory.memsw.limit_in_bytes": str(32 * GIB),
    })
    assert got["limit_bytes"] == 24 * GIB
    assert got["swap_max_bytes"] == 8 * GIB


def test_v1_unlimited_everywhere_and_no_cgroup_file_at_all(box, tmp_path):
    got = _limit_of(box, tmp_path, {"proc/self/cgroup": "4:memory:/",
                                    "cg1/memory.limit_in_bytes": V1_UNLIMITED})
    assert got["limit_bytes"] is None and got["version"] is None
    empty = box.ram_limit(proc_root=str(tmp_path / "nothing"), v2_root=str(tmp_path / "x"),
                          v1_root=str(tmp_path / "y"))
    assert empty["limit_bytes"] is None and empty["swap_total_bytes"] is None


def test_a_garbage_limit_is_unknown_not_zero(box, tmp_path):
    got = _limit_of(box, tmp_path, {"proc/self/cgroup": "0::/a",
                                    "cg2/a/memory.max": "-1", "cg2/memory.max": "12abc"})
    assert got["limit_bytes"] is None


def test_process_memory_reads_proc_status_on_linux(box, tmp_path, monkeypatch):
    _tree(tmp_path, {"proc/self/status": "VmHWM:\t 4096 kB\nVmRSS:\t 3072 kB\nRssAnon:\t 1024 kB\n"
                                         "RssShmem:\t 2048 kB\nVmLck:\t 0 kB\nVmSwap:\t 512 kB"})
    monkeypatch.setattr(box.sys, "platform", "linux")
    got = box.process_memory(proc_root=str(tmp_path / "proc"))
    assert got == {"rss": 3 << 20, "rss_anon": 1 << 20, "rss_shmem": 2 << 20,
                   "peak_rss": 4 << 20, "locked": 0, "swap": 512 << 10}
