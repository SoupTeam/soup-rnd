"""CPU tests for A4's spill-volume preflight (a4_box.py) and the probe's pure helpers.

The preflight is what keeps a "spill" off tmpfs (RAM again), off a container's
overlay root, and off a cloud network volume that is not a local SSD; these
tests pin how a mount table and a drive model are classified, and that a
volume whose unbuffered round trip fails is refused.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load

MOUNTINFO = """\
22 1 0:21 / / rw,relatime - overlay overlay rw,lowerdir=/x
30 22 259:1 / /workspace rw,relatime - ext4 /dev/nvme0n1p1 rw
31 22 0:40 / /dev/shm rw - tmpfs shm rw
32 22 0:41 / /mnt/net rw - nfs4 10.0.0.1:/vol rw
33 30 259:2 / /workspace/fast\\040disk rw - xfs /dev/nvme1n1 rw
"""


@pytest.fixture()
def box():
    return _load("a4_box")


@pytest.mark.parametrize("path,fstype,mount", [
    ("/root/x", "overlay", "/"),
    ("/workspace/spill", "ext4", "/workspace"),
    ("/workspace", "ext4", "/workspace"),
    ("/workspacex/spill", "overlay", "/"),
    ("/dev/shm/a4", "tmpfs", "/dev/shm"),
    ("/mnt/net/spill", "nfs4", "/mnt/net"),
    ("/workspace/fast disk/s", "xfs", "/workspace/fast disk"),
])
def test_the_longest_matching_mount_wins(box, path, fstype, mount):
    found = box.parse_mountinfo(MOUNTINFO, path)
    assert (found["fstype"], found["mount_point"]) == (fstype, mount)


@pytest.mark.parametrize("fstype,model,network,scope", [
    ("ext4", "Samsung SSD 990 PRO 2TB", False, "local"),
    ("xfs", "QEMU HARDDISK", False, "virtual"),
    ("ext4", "Virtual Disk", False, "virtual"),
    ("tmpfs", None, False, "refused"),
    ("overlay", "Samsung SSD", False, "refused"),
    ("nfs4", None, False, "network"),
    ("fuse.s3fs", None, False, "network"),
    ("virtiofs", None, False, "network"),
    ("NTFS", None, True, "network"),
    ("NTFS", None, False, "local"),
])
def test_filesystem_classification(box, fstype, model, network, scope):
    assert box.classify_fs(fstype, model, network)[0] == scope


def test_a_network_volume_is_refused_unless_labelled(box, tmp_path, monkeypatch):
    monkeypatch.setattr(box, "classify_fs", lambda *a: (box.NETWORK, "nfs4 is a network fs"))
    monkeypatch.setattr(box, "round_trip", lambda d: (True, "ok"))
    refused = box.spill_volume(str(tmp_path))
    assert not refused["ok"] and "--accept-network" in refused["refusals"][0]
    labelled = box.spill_volume(str(tmp_path), accept_network=True)
    assert labelled["ok"] and labelled["scope"] == "network"


def test_tmpfs_is_refused_without_writing_to_it(box, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(box, "classify_fs", lambda *a: (box.REFUSED, "tmpfs is RAM"))
    monkeypatch.setattr(box, "round_trip", lambda d: calls.append(d) or (True, "ok"))
    record = box.spill_volume(str(tmp_path))
    assert not record["ok"] and record["refusals"] == ["tmpfs is RAM"]
    assert calls == [] and record["direct_io"] is False


def test_a_failed_unbuffered_round_trip_is_refused(box, tmp_path, monkeypatch):
    monkeypatch.setattr(box, "classify_fs", lambda *a: (box.LOCAL, "ext4"))
    monkeypatch.setattr(box, "round_trip", lambda d: (False, "unbuffered open failed: EINVAL"))
    record = box.spill_volume(str(tmp_path))
    assert not record["ok"]
    assert "unbuffered I/O is not usable here: unbuffered open failed: EINVAL" in record["refusals"]


def test_the_real_round_trip_on_this_volume(box, tmp_path):
    ok, reason = box.round_trip(str(tmp_path))
    assert isinstance(ok, bool) and reason
    assert list(tmp_path.glob("a4-direct-probe-*")) == []


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


# ---------------------------------------------------------------------------
# cgroup limit, swap and process memory, on fixture trees (no real cgroup needed)
# ---------------------------------------------------------------------------
GIB = 1 << 30
#: What v1 memory.limit_in_bytes reads when unlimited on x86-64 (PAGE_COUNTER_MAX pages).
V1_UNLIMITED = "9223372036854771712"


def _tree(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


def _limit_of(box, tmp_path, files):
    _tree(tmp_path, files)
    return box.ram_limit(proc_root=str(tmp_path / "proc"), v2_root=str(tmp_path / "cg2"),
                         v1_root=str(tmp_path / "cg1"))


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


@pytest.mark.parametrize("mem,weights,expected", [
    # Linux, the runtime accounts for its pinned arenas: only the pageable rest is charged.
    ({"rss_anon": 3 * GIB, "rss_shmem": 4 * GIB, "rss": 7 * GIB}, 4 * GIB, 3 * GIB),
    # Linux, 1 GiB more page-locked than the runtime reports: charged, not lost.
    ({"rss_anon": 3 * GIB, "rss_shmem": 5 * GIB, "rss": 8 * GIB}, 4 * GIB, 4 * GIB),
    # A source that reports no pinned bytes (None -> 0): its shmem is charged here.
    ({"rss_anon": 3 * GIB, "rss_shmem": 2 * GIB, "rss": 5 * GIB}, 0, 5 * GIB),
    # Windows / macOS: the resident set minus the weights.
    ({"rss_anon": None, "rss_shmem": None, "rss": 9 * GIB}, 4 * GIB, 5 * GIB),
    ({"rss_anon": None, "rss_shmem": None, "rss": 1 * GIB}, 4 * GIB, 0),
    ({"rss_anon": None, "rss_shmem": None, "rss": None}, 0, None),
])
def test_process_bytes_charge_every_resident_byte_once(box, mem, weights, expected):
    assert box.non_activation_bytes(mem, weights) == expected


@pytest.mark.parametrize("declared,cgroup,expected", [
    (None, None, None), (None, 24 * GIB, 24 * GIB), (16 * GIB, None, 16 * GIB),
    (16 * GIB, 24 * GIB, 16 * GIB),
    # A declaration above the container's limit cannot raise it.
    (64 * GIB, 24 * GIB, 24 * GIB),
])
def test_the_host_total_is_the_tighter_of_declared_and_cgroup(probe, declared, cgroup, expected):
    assert probe.host_total_bytes(declared, cgroup) == expected


def test_a_host_total_with_unreadable_process_memory_is_refused(probe, monkeypatch):
    monkeypatch.setattr(probe.a4_box, "process_memory",
                        lambda: {"rss": None, "rss_anon": None, "rss_shmem": None})
    args = probe.parse_args(["--mode", "arm", "--device", "cpu", "--weights", "w", "--out", "o",
                             "--k", "2", "--ring", "1", "--host-total-bytes", str(8 * GIB)])
    meta = {"n_layers": 4, "chunk_bytes": 4096, "runtime": {"pinned_bytes": 0},
            "cgroup": {"limit_bytes": None}}
    with pytest.raises(probe.budget.BudgetRefusalError, match="cannot be read"):
        probe.plan_arm(args, meta, "1")


# ---------------------------------------------------------------------------
# probe helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def probe():
    return _load("a4_probe")


class _Source:
    def __init__(self):
        from types import SimpleNamespace

        self._plans = {0: SimpleNamespace(span=1000), 1: SimpleNamespace(span=3000)}

    def _range_works(self, idx, region):
        return [lambda: None, lambda: None]


def test_weight_read_rate_is_bytes_over_the_union_of_busy_time(probe, monkeypatch):
    source = _Source()
    meter = probe.WeightReadMeter(source)
    mark = meter.snapshot()
    times = iter([0.0, 2.0, 1.0, 3.0, 10.0, 11.0, 10.5, 10.6])
    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(times))
    for idx in (0, 1):
        for work in source._range_works(idx, None):
            work()
    got = meter.since(mark)
    # Intervals [0,2], [1,3], [10,11], [10.5,10.6]: the union is 3 + 1 = 4 s.
    assert got["bytes"] == 4000 and got["busy_s"] == pytest.approx(4.0)
    assert got["gb_per_s"] == pytest.approx(4000 / 4.0 / 1e9)


def test_summary_hidden_share_and_rates(probe):
    records = [
        {"step_s": s, "losses": [1.0], "put_wait_s": 0.1, "get_wait_s": 0.0,
         "spill_write_s": 0.5, "spill_read_s": 0.5, "spill_written": 1e9, "spill_read": 2e9}
        for s in (2.0, 3.0, 2.5)
    ]
    out = probe.summarise(records)
    assert out["value"] == 2.5
    assert out["hidden_share"] == pytest.approx(1 - 0.3 / 3.0)
    assert out["spill_write_gb_per_s"] == pytest.approx(2.0)
    assert out["spill_read_gb_per_s"] == pytest.approx(4.0)
    assert out["finite"] is True
    records[0]["losses"] = [None]
    assert probe.summarise(records)["finite"] is False


def test_a_no_spill_arm_has_no_hidden_share(probe):
    records = [{"step_s": 1.0, "losses": [1.0], "put_wait_s": 0.0, "get_wait_s": 0.0,
                "spill_write_s": 0.0, "spill_read_s": 0.0, "spill_written": 0, "spill_read": 0}]
    out = probe.summarise(records)
    assert out["hidden_share"] is None and out["spill_write_gb_per_s"] is None


def _usage(total, free):
    from collections import namedtuple

    return namedtuple("usage", "total used free")(total, total - free, free)


@pytest.mark.parametrize("mount,total,free,need,problem", [
    # A small dedicated filesystem with less room than the spill file: accepted.
    (True, 8 << 20, 3 << 20, 4 << 20, None),
    # Room for the whole file: the volume would not refuse, the case proves nothing.
    (True, 8 << 20, 5 << 20, 4 << 20, "enough for the"),
    # The 1 TB root a missing loop mount falls back to: refused before it is filled.
    (True, 1 << 40, 900 << 30, 4 << 20, "over the"),
    # Not the root of a filesystem (POSIX): a subdirectory of some other volume.
    (False, 8 << 20, 3 << 20, 4 << 20, "not a mount point"),
])
def test_the_real_enospc_dir_must_be_a_small_dedicated_volume(probe, tmp_path, monkeypatch,
                                                              mount, total, free, need, problem):
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: _usage(total, free))
    monkeypatch.setattr(probe.os.path, "ismount", lambda path: mount)
    monkeypatch.setattr(probe.sys, "platform", "linux")
    got = probe.enospc_dir_problem(str(tmp_path), need)
    assert (got is None) if problem is None else (problem in got), got


def test_a_wrong_enospc_dir_is_refused_before_any_model_is_built(probe, tmp_path):
    with pytest.raises(SystemExit):
        probe.parse_args(["--mode", "cleanup", "--device", "cpu", "--weights", "w",
                          "--spill-dir", str(tmp_path), "--enospc-dir", str(tmp_path / "no"),
                          "--out", "x.json"])


def test_the_cpu_pilot_is_labelled(probe):
    args = probe.parse_args(["--mode", "arm", "--device", "cpu", "--weights", "w",
                             "--out", "x.json", "--label", "smoke"])
    assert args.label == "SYNTHETIC/CPU PILOT smoke"
    assert args.quant == "none" and args.no_pin is True


def test_session_plan_alternates_and_puts_disk_spill_on_the_weights_drive(tmp_path):
    session = _load("a4_session")
    base = ["--weights", "w", "--shards", str(tmp_path / "shards"), "--spill-dir", "s",
            "--out", str(tmp_path / "o")]
    implicit = session.args_parser().parse_args(base)
    with pytest.raises(ValueError, match="weights-spill-dir"):
        session.arm_command(implicit, tmp_path, "disk_k4_same_drive", 0, 1, "B")
    args = session.args_parser().parse_args(base + ["--weights-spill-dir", "ws"])
    arms = [session.arm_command(args, tmp_path, "disk_k4_same_drive", 0, p, a)
            for p, a in enumerate("ABBA")]
    budgets = [entry["command"][entry["command"].index("--budget") + 1] for entry in arms]
    assert budgets == ["1", "min", "min", "1"]
    spill_dir = arms[1]["command"][arms[1]["command"].index("--spill-dir") + 1]
    assert spill_dir == "ws"
    with pytest.raises(ValueError, match="other-spill-dir"):
        session.arm_command(args, tmp_path, "disk_k4_other_drive", 0, 0, "A")


@pytest.mark.parametrize("disks,refused", [(("d0", "d0"), False), (("d0", "d1"), True),
                                           (("d0", None), True)])
def test_the_session_refuses_a_same_drive_comparison_the_os_does_not_confirm(
        tmp_path, monkeypatch, disks, refused):
    session = _load("a4_session")
    box = _load("a4_box")
    where = {str(tmp_path / "shards"): disks[0], str(tmp_path / "ws"): disks[1]}
    monkeypatch.setattr(box, "placement", lambda path: {"drive": {
        "device": where[path], "device_problem": None if where[path] else "no block device"}})
    args = session.args_parser().parse_args([
        "--weights", "w", "--shards", str(tmp_path / "shards"), "--spill-dir", "s",
        "--weights-spill-dir", str(tmp_path / "ws"), "--out", str(tmp_path / "o"),
        "--comparisons", "ram_k2_half,disk_k4_same_drive"])
    record = session.placement_check(args, ["ram_k2_half", "disk_k4_same_drive"])
    assert bool(record["refusals"]) is refused, record


@pytest.mark.parametrize("ref", ["ga_vs_ga", "pinned_vs_pinned"])
def test_the_deterministic_rerun_fires_when_either_reference_is_unreproducible(tmp_path, ref):
    import json

    session = _load("a4_session")
    exact = {"backward": {"equal": True}, "forward_equal": True}
    row = {"kind": "correctness", "ga_vs_ga": exact, "pinned_vs_pinned": exact}
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"records": [row]}))
    assert session.needs_fallback(path) is False
    row[ref] = {"backward": {"equal": False}, "forward_equal": True}
    path.write_text(json.dumps({"records": [row]}))
    assert session.needs_fallback(path) is True
    path.write_text(json.dumps({"records": []}))
    assert session.needs_fallback(path) is True
