"""A container must not plan a host-sized resident store."""
from types import SimpleNamespace

import psutil
import pytest

from soup_cli.utils import layer_stream as streaming

GIB = 1024**3


@pytest.fixture
def container(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(
        total=128 * GIB, available=100 * GIB))
    files = {
        "/proc/self/cgroup": "0::/\n",
        "/sys/fs/cgroup/memory.max": str(32 * GIB),
        "/sys/fs/cgroup/memory.current": str(8 * GIB),
        "/sys/fs/cgroup/memory.stat": f"inactive_file {3 * GIB}\n",
    }

    def read(path, *args, **kwargs):
        normalized = path.as_posix()
        for suffix, content in files.items():
            if normalized.endswith(suffix):
                return content
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "read_text", read)
    return files


def test_effective_total_is_the_real_container_limit(container):
    assert streaming.total_ram_bytes() == 32 * GIB


def test_available_reclaims_only_inactive_file_cache(container):
    assert streaming.free_ram_bytes() == 27 * GIB


def test_real_bf16_store_automatically_uses_disk_without_memory_pressure(container):
    plan = streaming.build_stream_plan(
        arch="qwen2", n_layers=48, layer_bytes=550_000_000,
        embed_bytes=0, store_bytes=29_540_125_474,
        available_ram_bytes=streaming.free_ram_bytes(),
        total_ram_bytes=streaming.total_ram_bytes(), pinned_limit_bytes=None,
    )
    assert plan.tier == streaming.TIER_DISK


def test_host_available_remains_the_upper_bound(container, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(
        total=128 * GIB, available=5 * GIB))
    assert streaming.free_ram_bytes() == 5 * GIB


def test_unlimited_container_preserves_host_measurement(container):
    container["/sys/fs/cgroup/memory.max"] = "max"
    assert streaming.total_ram_bytes() == 128 * GIB
    assert streaming.free_ram_bytes() == 100 * GIB


def test_absent_controller_preserves_host_measurement(container):
    container.clear()
    assert streaming.total_ram_bytes() == 128 * GIB
    assert streaming.free_ram_bytes() == 100 * GIB


def test_visible_parent_limit_also_constrains_the_child(container):
    container["/proc/self/cgroup"] = "0::/worker/job\n"
    for group, limit, usage, inactive in (
        ("worker", 24, 10, 3), ("worker/job", 32, 8, 3),
    ):
        prefix = "/sys/fs/cgroup/" + group
        container[prefix + "/memory.max"] = str(limit * GIB)
        container[prefix + "/memory.current"] = str(usage * GIB)
        container[prefix + "/memory.stat"] = f"inactive_file {inactive * GIB}\n"
    assert streaming.total_ram_bytes() == 24 * GIB
    assert streaming.free_ram_bytes() == 17 * GIB


def test_v1_accounts_for_descendant_file_cache(container):
    container.clear()
    prefix = "/sys/fs/cgroup/memory"
    container.update({
        "/proc/self/cgroup": "7:cpu:/ignored\n9:memory:/\n",
        prefix + "/memory.limit_in_bytes": str(32 * GIB),
        prefix + "/memory.usage_in_bytes": str(8 * GIB),
        prefix + "/memory.stat": f"inactive_file 0\ntotal_inactive_file {3 * GIB}\n",
    })
    assert streaming.total_ram_bytes() == 32 * GIB
    assert streaming.free_ram_bytes() == 27 * GIB


def test_unlimited_v1_does_not_require_usage(container):
    container.clear()
    container["/sys/fs/cgroup/memory/memory.limit_in_bytes"] = str(2**63 - 4096)
    assert streaming.total_ram_bytes() == 128 * GIB
    assert streaming.free_ram_bytes() == 100 * GIB


def test_known_limit_without_usage_has_no_known_free_headroom(container):
    del container["/sys/fs/cgroup/memory.current"]
    assert streaming.total_ram_bytes() == 32 * GIB
    assert streaming.free_ram_bytes() == 0


@pytest.mark.parametrize("stat", ["", "broken", "inactive_file -100\n", "active_file 9000\n"])
def test_unknown_or_active_cache_is_not_reclaimed(container, stat):
    container["/sys/fs/cgroup/memory.stat"] = stat
    assert streaming.free_ram_bytes() == 24 * GIB


def test_bad_usage_cannot_produce_negative_or_excess_headroom(container):
    container["/sys/fs/cgroup/memory.current"] = str(40 * GIB)
    container["/sys/fs/cgroup/memory.stat"] = ""
    assert streaming.free_ram_bytes() == 0
    container["/sys/fs/cgroup/memory.stat"] = f"inactive_file {100 * GIB}\n"
    assert streaming.free_ram_bytes() == 32 * GIB


def test_membership_cannot_escape_the_controller_root(container):
    container["/proc/self/cgroup"] = "0::/../../outside\n"
    container["/outside/memory.max"] = str(GIB)
    assert streaming.total_ram_bytes() == 32 * GIB


def test_protected_allocation_does_not_require_bursting_to_the_hard_limit(container):
    container["/sys/fs/cgroup/memory.max"] = str(56 * GIB)
    container["/sys/fs/cgroup/memory.low"] = str(32 * GIB)
    assert streaming.total_ram_bytes() == 32 * GIB
    assert streaming.free_ram_bytes() == 27 * GIB


def test_reservation_is_still_bounded_by_a_smaller_hard_limit(container):
    container["/sys/fs/cgroup/memory.max"] = str(24 * GIB)
    container["/sys/fs/cgroup/memory.low"] = str(32 * GIB)
    assert streaming.total_ram_bytes() == 24 * GIB
    assert streaming.free_ram_bytes() == 19 * GIB


def test_zero_reservation_adds_no_artificial_constraint(container):
    container["/sys/fs/cgroup/memory.low"] = "0"
    assert streaming.total_ram_bytes() == 32 * GIB
    assert streaming.free_ram_bytes() == 27 * GIB


def test_v1_soft_allocation_is_a_conservative_budget(container):
    container.clear()
    prefix = "/sys/fs/cgroup/memory"
    container[prefix + "/memory.limit_in_bytes"] = str(56 * GIB)
    container[prefix + "/memory.soft_limit_in_bytes"] = str(32 * GIB)
    container[prefix + "/memory.usage_in_bytes"] = str(8 * GIB)
    assert streaming.total_ram_bytes() == 32 * GIB
    assert streaming.free_ram_bytes() == 24 * GIB
