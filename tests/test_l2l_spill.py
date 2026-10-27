"""The L2L activation spill without a trainer: the planner, the store's slots, the spill
file, the stale sweep, the volume checks, the disk-tier rule and the schema field.

Seams: ``soup_cli.utils.l2l_spill`` (``plan_spill``, ``open_spill_file``, ``sweep_stale``,
``spill_files``, ``prepare_spill_dir``), ``soup_cli.utils.l2l_activations.ActivationStore`` with
the real ``DirectSpillFile``, and ``load_config_from_string``.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import types

import pytest
import yaml

MIB, GIB = 2**20, 2**30


@pytest.fixture()
def direct_dir(tmp_path):
    """A directory on the temp volume, skipped where unbuffered I/O is refused (tmpfs)."""
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "direct"
    directory.mkdir()
    problem = l2l_spill.filesystem_refusal(os.path.realpath(directory))
    problem = problem or l2l_spill._round_trip(str(directory))
    if problem:
        pytest.skip(f"the temp directory cannot hold an unbuffered spill: {problem}")
    return str(directory)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------
def test_a_pinned_17_mib_chunk_costs_32_mib_and_the_rings_are_charged():
    from soup_cli.utils.l2l_spill import plan_spill

    plan = plan_spill(10 * GIB, n_layers=80, k=16, chunk_bytes=17 * MIB, pin=True)
    assert plan.ring == 4
    assert plan.chunk_cost == 32 * MIB
    assert plan.slot_bytes == 17 * MIB
    # one staging buffer asks a slot plus one sector, and that request is page-locked as
    # the next power of two
    assert plan.ring_buffer_cost == 32 * MIB
    assert plan.minimum_bytes == 2 * 4 * 32 * MIB
    assert plan.all_resident_bytes == 80 * 16 * 32 * MIB
    assert plan.spill_layers == 61
    assert plan.host_bytes == 19 * 16 * 32 * MIB + 2 * 4 * 32 * MIB == 10_468_982_784
    assert plan.host_bytes <= 10 * GIB
    assert plan.spill_file_bytes == 61 * 16 * 17 * MIB


def test_pageable_memory_is_charged_at_its_request():
    from soup_cli.utils.l2l_spill import plan_spill

    plan = plan_spill(10 * GIB, n_layers=80, k=16, chunk_bytes=17 * MIB, pin=False)
    assert plan.chunk_cost == 17 * MIB
    assert plan.ring_buffer_cost == 17 * MIB + 4096
    assert plan.minimum_bytes == 8 * (17 * MIB + 4096)
    assert plan.spill_layers == 43
    assert plan.host_bytes == 37 * 16 * 17 * MIB + 8 * (17 * MIB + 4096)


def test_pinned_rounding_decides_whether_the_store_fits():
    """bf16, 80 layers, hidden 8192, seq 1088, k=16: a 17 MiB chunk occupies a 32 MiB pinned
    block, so the store needs 40 GiB, not the 21.25 GiB of its logical bytes."""
    from soup_cli.utils.l2l_spill import plan_spill

    budget = int(32 * GIB * 0.7)
    kw = dict(n_layers=80, k=16, chunk_bytes=17 * MIB)
    assert plan_spill(budget, pin=False, **kw).spill_layers == 0
    pinned = plan_spill(budget, pin=True, **kw)
    assert pinned.spill_layers == 36
    assert pinned.host_bytes <= budget


def test_what_fits_stays_resident_and_no_budget_means_no_spill():
    from soup_cli.utils.l2l_spill import plan_spill

    kw = dict(n_layers=2, k=3, chunk_bytes=100, pin=False)
    fits = plan_spill(1000, **kw)
    assert fits.spill_layers == 0 and fits.host_bytes == 600
    assert plan_spill(None, **{**kw, "chunk_bytes": 10**15}).spill_layers == 0


def test_a_budget_below_every_layer_spilled_is_refused_with_what_to_change():
    from soup_cli.utils.l2l_spill import SpillRefusedError, plan_spill

    with pytest.raises(SpillRefusedError) as caught:
        plan_spill(1000, n_layers=2, k=4, chunk_bytes=200, pin=False)
    message = str(caught.value)
    assert "activations" in message
    for fix in ("gradient_accumulation_steps", "data.max_length", "batch_size", "free RAM"):
        assert fix in message
    # 10,000-byte chunks: slots of 12,288, ring buffers of 16,384, rings of 131,072 bytes,
    # the smallest budget that runs once 10 x 4 chunks no longer fit
    kw = dict(n_layers=10, k=4, chunk_bytes=10_000, pin=False)
    assert plan_spill(131_072, **kw).spill_layers == 10
    with pytest.raises(SpillRefusedError):
        plan_spill(131_071, **kw)


@pytest.mark.parametrize("pin", [False, True])
def test_every_budget_gets_the_fewest_spilled_layers_within_it(pin):
    from soup_cli.utils.l2l_spill import plan_spill

    kw = dict(n_layers=12, k=3, chunk_bytes=5 * MIB + 123, pin=pin)
    top = plan_spill(None, **kw)
    for budget in range(top.minimum_bytes, top.all_resident_bytes, 3 * MIB + 7):
        plan = plan_spill(budget, **kw)
        assert 1 <= plan.spill_layers <= 12
        assert plan.host_bytes <= budget, budget
        # one layer fewer spilled would not fit (or every other layer is already resident)
        assert plan.resident_layers == 11 or plan.host_bytes + 3 * plan.chunk_cost > budget


# --------------------------------------------------------------------------
# the store: slots, alignment, a smaller tensor in a larger slot
# --------------------------------------------------------------------------
def _direct_store(directory, *, n_layers, spill_layers, shape, dtype, k=2):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, DirectSpillFile

    path = os.path.join(directory, "spill.bin")
    return ActivationStore(
        n_layers=n_layers, k=k, chunk_shape=shape, dtype=dtype, device="cpu", pin=False,
        spill_layers=spill_layers, spill_factory=lambda size: DirectSpillFile(path, size),
        ring=4,
    ), path, torch


def test_a_bf16_chunk_off_the_sector_round_trips_through_the_direct_file(direct_dir):
    import torch

    shape = (1, 7, 5120)  # bf16, odd seq: 71,680 bytes, 17.5 sectors
    store, path, _ = _direct_store(
        direct_dir, n_layers=3, spill_layers=2, shape=shape, dtype=torch.bfloat16
    )
    try:
        assert store.chunk_bytes == 71_680 and store.slot_bytes == 73_728
        assert os.path.getsize(path) == 2 * 2 * 73_728
        gen = torch.Generator().manual_seed(0)
        sent = {
            (layer, mb): torch.randn(shape, generator=gen).to(torch.bfloat16)
            for layer in range(3) for mb in range(2)
        }
        for (layer, mb), tensor in sent.items():
            store.put(layer, mb, tensor)
        store.begin_backward()
        for layer in range(2, -1, -1):
            for mb in range(2):
                got = store.get(layer, mb)
                assert got.dtype == torch.bfloat16 and got.shape == shape
                assert torch.equal(got, sent[(layer, mb)])
        store.end_step()
        assert store.stats.bytes_written == store.stats.bytes_read == 2 * 2 * 73_728
    finally:
        store.close()
    assert not os.path.exists(path)


def test_a_smaller_tensor_lives_in_the_front_of_a_larger_slot(direct_dir):
    import torch

    store, path, _ = _direct_store(
        direct_dir, n_layers=3, spill_layers=2, shape=(1, 7, 5120), dtype=torch.bfloat16
    )
    try:
        for step, short in enumerate([(1, 3, 5120), (1, 7, 5120), (1, 1, 5120)]):
            sent = {
                (layer, mb): torch.full(short, float(step * 10 + layer * 2 + mb)).bfloat16()
                for layer in range(3) for mb in range(2)
            }
            for (layer, mb), tensor in sent.items():
                store.put(layer, mb, tensor)
            store.begin_backward()
            for layer in range(2, -1, -1):
                for mb in range(2):
                    got = store.get(layer, mb)
                    assert got.shape == short
                    assert torch.equal(got, sent[(layer, mb)])
            store.end_step()
        with pytest.raises(ValueError, match="does not fit"):
            store.put(2, 0, torch.zeros((1, 8, 5120), dtype=torch.bfloat16))
    finally:
        store.close()


def test_a_short_window_reads_only_what_it_put(tmp_path):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, BufferedSpillFile

    store = ActivationStore(
        n_layers=2, k=3, chunk_shape=(1, 4, 8), dtype=torch.float32, device="cpu", pin=False,
        spill_layers=1, spill_factory=lambda size: BufferedSpillFile(str(tmp_path / "s"), size),
    )
    try:
        for k in (3, 1, 2):
            for layer in range(2):
                for mb in range(k):
                    store.put(layer, mb, torch.full((1, 4, 8), float(k * 10 + mb)))
            store.begin_backward()
            for layer in (1, 0):
                for mb in range(k):
                    assert torch.equal(store.get(layer, mb), torch.full((1, 4, 8), k * 10.0 + mb))
            store.end_step()
    finally:
        store.close()


def test_a_ring_that_cannot_be_allocated_removes_the_file(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils import l2l_activations

    path = tmp_path / "spill.bin"

    def no_memory(nbytes, *, pin):
        raise MemoryError("no staging")

    monkeypatch.setattr(l2l_activations, "aligned_buffer", no_memory)
    with pytest.raises(MemoryError):
        l2l_activations.ActivationStore(
            n_layers=2, k=2, chunk_shape=(1, 8, 128), dtype=torch.float32, device="cpu",
            pin=False, spill_layers=1,
            spill_factory=lambda size: l2l_activations.BufferedSpillFile(str(path), size),
        )
    assert not path.exists()


# --------------------------------------------------------------------------
# a full drive
# --------------------------------------------------------------------------
def _enospc(*_a, **_k):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_a_full_drive_while_creating_the_file_names_size_and_directory(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_activations
    from soup_cli.utils.l2l_spill import SpillDiskFullError, open_spill_file, spill_file_name

    path = str(tmp_path / spill_file_name())
    monkeypatch.setattr(l2l_activations, "_write_all", _enospc)
    with pytest.raises(SpillDiskFullError) as caught:
        open_spill_file(path, 5 * 4096)
    message = str(caught.value)
    assert str(tmp_path) in message and "20,480 bytes" in message
    assert "training.l2l_spill_dir" in message
    assert not os.path.exists(path)


def test_a_full_drive_in_the_writer_thread_removes_the_file(tmp_path, monkeypatch):
    import time

    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, BufferedSpillFile
    from soup_cli.utils.l2l_spill import SpillDiskFullError, spill_file_name

    path = str(tmp_path / spill_file_name())
    monkeypatch.setattr(BufferedSpillFile, "write", lambda self, offset, buf: _enospc())
    store = ActivationStore(
        n_layers=3, k=2, chunk_shape=(1, 8, 128), dtype=torch.float32, device="cpu", pin=False,
        spill_layers=2, spill_factory=lambda size: BufferedSpillFile(path, size),
    )
    with pytest.raises(SpillDiskFullError) as caught:
        deadline = time.time() + 10
        while time.time() < deadline:
            store.put(0, 0, torch.zeros(1, 8, 128))
            time.sleep(0.01)
    message = str(caught.value)
    assert str(tmp_path) in message and f"{2 * 2 * 4096:,} bytes" in message
    assert "spill file was removed" in message
    assert not os.path.exists(path)
    store.close()


# --------------------------------------------------------------------------
# file names and the stale sweep
# --------------------------------------------------------------------------
def test_the_sweep_removes_only_files_of_dead_runs(tmp_path):
    from soup_cli.utils.l2l_spill import SPILL_FILE_GLOB, spill_files, sweep_stale

    assert SPILL_FILE_GLOB == "soup-l2l-spill-*.bin"
    dead, live = _dead_pid(), os.getpid()
    directory = tmp_path / "spill"
    directory.mkdir()
    names = {
        "dead": f"soup-l2l-spill-{dead}-0123abcd.bin",
        "dead_probe": f"soup-l2l-probe-{dead}-89abcdef.bin",
        "live": f"soup-l2l-spill-{live}-0123abcd.bin",
        "foreign": "notes.bin",
        "near_miss": f"soup-l2l-spill-{dead}.bin",
        "upper_hex": f"soup-l2l-spill-{dead}-FEDCBA98.bin",
    }
    for name in names.values():
        (directory / name).write_bytes(b"x")
    lines = []
    removed = sweep_stale(str(directory), notify=lines.append)
    gone = sorted(os.path.realpath(directory / names[key]) for key in ("dead", "dead_probe"))
    assert removed == gone
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        names[key] for key in ("live", "foreign", "near_miss", "upper_hex")
    )
    assert len(lines) == 1 and all(path in lines[0] for path in gone)
    assert spill_files(str(directory)) == [os.path.realpath(directory / names["live"])]
    assert sweep_stale(str(tmp_path / "missing")) == []


def _link_fixture(tmp_path):
    """A spill directory and, outside it, a file named like ours whose process is gone."""
    dead = _dead_pid()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / f"soup-l2l-spill-{dead}-00000000.bin"
    target.write_bytes(b"keep me")
    directory = tmp_path / "spill"
    directory.mkdir()
    return outside, target, directory, directory / f"soup-l2l-spill-{dead}-11111111.bin"


def _assert_link_untouched(directory, link, target):
    from soup_cli.utils.l2l_spill import spill_files, sweep_stale

    assert sweep_stale(str(directory)) == []
    assert spill_files(str(directory)) == []
    assert os.path.lexists(link)
    assert target.read_bytes() == b"keep me"


@pytest.mark.requires_symlink
def test_the_sweep_never_follows_or_removes_a_symlink(tmp_path):
    _outside, target, directory, link = _link_fixture(tmp_path)
    os.symlink(target, link)
    _assert_link_untouched(directory, link, target)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are a Windows construct")
def test_the_sweep_never_follows_or_removes_a_junction(tmp_path):
    import _winapi

    outside, target, directory, link = _link_fixture(tmp_path)
    _winapi.CreateJunction(str(outside), str(link))
    _assert_link_untouched(directory, link, target)


def test_a_process_is_alive_until_it_exits():
    from soup_cli.utils.l2l_spill import process_alive

    assert process_alive(os.getpid())
    assert not process_alive(_dead_pid())


# --------------------------------------------------------------------------
# the spill directory
# --------------------------------------------------------------------------
def _fake_mountinfo(tmp_path, monkeypatch, directory, fstype, source="none"):
    from soup_cli.utils import l2l_spill

    proc = tmp_path / "proc" / "self"
    proc.mkdir(parents=True)
    mount = os.path.realpath(directory).replace(os.sep, "/").replace(" ", "\\040")
    (proc / "mountinfo").write_text(
        "22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n"
        f"40 22 0:50 / {mount} rw,relatime - {fstype} {source} rw\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(l2l_spill, "PROC_ROOT", str(tmp_path / "proc"))


def _prepare(directory, **kw):
    from soup_cli.utils.l2l_spill import prepare_spill_dir

    options = dict(explicit=True, spill_layers=3, file_bytes=12 * MIB)
    options.update(kw)
    return prepare_spill_dir(str(directory), **options)


@pytest.mark.parametrize(
    "fstype,source,reason",
    [
        ("tmpfs", "tmpfs", "tmpfs"),
        ("overlay", "overlay", "overlay"),
        ("nfs4", "server:/export", "network filesystem"),
        ("cifs", "//server/share", "network filesystem"),
    ],
)
def test_ram_stacked_and_network_filesystems_are_refused(tmp_path, monkeypatch, fstype,
                                                         source, reason):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    directory = tmp_path / "spill"
    directory.mkdir()
    _fake_mountinfo(tmp_path, monkeypatch, directory, fstype, source)
    with pytest.raises(SpillRefusedError) as caught:
        _prepare(directory)
    message = str(caught.value)
    assert str(directory) in message and reason in message
    assert "training.l2l_spill_dir" in message


def test_a_volume_without_unbuffered_io_is_refused(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_activations, l2l_spill

    directory = tmp_path / "spill"

    def refuse(path):
        raise OSError(errno.EINVAL, "Invalid argument")

    monkeypatch.setattr(l2l_activations, "_open_unbuffered", refuse)
    monkeypatch.setattr(l2l_spill.shutil, "disk_usage", lambda p: types.SimpleNamespace(
        free=10**15))
    with pytest.raises(l2l_spill.SpillRefusedError, match=r"(?s)unbuffered.*l2l_spill_dir"):
        _prepare(directory)
    assert os.listdir(directory) == []  # the probe file is gone


def test_a_drive_short_of_the_file_plus_the_margin_is_refused(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill

    monkeypatch.setattr(l2l_spill.shutil, "disk_usage", lambda p: types.SimpleNamespace(
        free=GIB + 12 * MIB - 1))
    with pytest.raises(l2l_spill.SpillRefusedError) as caught:
        _prepare(tmp_path / "spill")
    message = str(caught.value)
    assert "1 GiB" in message and str(tmp_path / "spill") in message
    assert l2l_spill.space_margin_bytes(100 * GIB) == 10 * GIB


def test_a_usable_directory_is_created_and_returned(direct_dir):
    path = os.path.join(direct_dir, "nested", "spill")
    assert _prepare(path) == os.path.realpath(path)
    assert os.listdir(path) == []


# --------------------------------------------------------------------------
# the disk tier
# --------------------------------------------------------------------------
@pytest.fixture()
def disks(monkeypatch):
    """``physical_disks`` answered from a table keyed by directory name."""
    from soup_cli.utils import l2l_spill

    table = {}

    def physical(path):
        answer = table.get(os.path.basename(os.path.normpath(path)))
        if answer is None:
            return None, "not read on this platform"
        return frozenset(answer), ", ".join(answer)

    monkeypatch.setattr(l2l_spill, "physical_disks", physical)
    monkeypatch.setattr(l2l_spill, "_round_trip", lambda directory: None)
    monkeypatch.setattr(l2l_spill.shutil, "disk_usage", lambda p: types.SimpleNamespace(
        free=10**15))
    return table


@pytest.mark.parametrize("layout", ["same", "unknown"])
def test_the_default_dir_on_the_weights_disk_or_an_unknown_one_stops(tmp_path, disks, layout):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    disks["shards"] = ["nvme0n1"]
    if layout == "same":
        disks["_l2l_spill"] = ["nvme0n1"]
    with pytest.raises(SpillRefusedError) as caught:
        _prepare(tmp_path / "_l2l_spill", explicit=False, tier="disk",
                 weight_dirs=[str(tmp_path / "shards")])
    message = str(caught.value)
    assert "training.l2l_spill_dir" in message and "another disk" in message
    assert ("same physical disk" in message) == (layout == "same")


@pytest.mark.parametrize("layout", ["same", "unknown"])
def test_an_explicit_dir_on_the_weights_disk_or_an_unknown_one_warns(tmp_path, disks, layout):
    disks["shards"] = ["nvme0n1"]
    if layout == "same":
        disks["mine"] = ["nvme0n1"]
    lines = []
    _prepare(tmp_path / "mine", explicit=True, tier="disk",
             weight_dirs=[str(tmp_path / "shards")], notify=lines.append)
    assert len(lines) == 1
    if layout == "same":
        assert "13% slower" in lines[0]
    else:
        assert "could not verify" in lines[0]


def test_another_disk_or_the_ram_tier_is_silent(tmp_path, disks):
    disks["shards"] = ["nvme0n1"]
    disks["_l2l_spill"] = ["nvme1n1"]
    lines = []
    _prepare(tmp_path / "_l2l_spill", explicit=False, tier="disk",
             weight_dirs=[str(tmp_path / "shards")], notify=lines.append)
    _prepare(tmp_path / "elsewhere", explicit=False, tier="ram", notify=lines.append)
    assert lines == []


def _link_dir(target, link):
    """A directory link: a symlink, or a junction where symlinks need a privilege."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def test_a_partition_and_a_device_mapper_volume_resolve_to_their_whole_disks(tmp_path):
    """The /sys/devices layout: a partition sits inside its disk and has a ``partition``
    file; a dm device lists the block devices under it in ``slaves``."""
    from soup_cli.utils.l2l_spill import _whole_disks

    devices = tmp_path / "devices"
    nvme = devices / "nvme0n1"
    (nvme / "nvme0n1p2").mkdir(parents=True)
    (nvme / "nvme0n1p2" / "partition").write_text("2")
    sda = devices / "sda"
    (sda / "sda1").mkdir(parents=True)
    (sda / "sda1" / "partition").write_text("1")
    dm = devices / "dm-0"
    (dm / "slaves").mkdir(parents=True)
    _link_dir(nvme / "nvme0n1p2", dm / "slaves" / "nvme0n1p2")
    _link_dir(sda / "sda1", dm / "slaves" / "sda1")

    assert _whole_disks(str(nvme / "nvme0n1p2")) == ["nvme0n1"]
    assert _whole_disks(str(nvme)) == ["nvme0n1"]
    assert _whole_disks(str(dm)) == ["nvme0n1", "sda"]


def test_the_physical_disk_is_unknown_off_linux(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill

    for platform in ("win32", "darwin"):
        monkeypatch.setattr(l2l_spill.sys, "platform", platform)
        disks, why = l2l_spill.physical_disks(str(tmp_path))
        assert disks is None and platform in why


@pytest.mark.skipif(not hasattr(os, "major"), reason="st_dev major/minor is POSIX only")
def test_the_physical_disk_is_read_from_sys_dev_block(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill

    monkeypatch.setattr(l2l_spill.sys, "platform", "linux")
    st_dev = os.stat(tmp_path).st_dev
    sys_root = tmp_path / "sys"
    disk = sys_root / "devices" / "nvme1n1"
    (disk / "nvme1n1p1").mkdir(parents=True)
    (disk / "nvme1n1p1" / "partition").write_text("1")
    (sys_root / "dev" / "block").mkdir(parents=True)
    _link_dir(disk / "nvme1n1p1",
              sys_root / "dev" / "block" / f"{os.major(st_dev)}:{os.minor(st_dev)}")
    monkeypatch.setattr(l2l_spill, "SYS_ROOT", str(sys_root))
    disks, _why = l2l_spill.physical_disks(str(tmp_path / "not-yet" / "created"))
    assert disks == frozenset({"nvme1n1"})


# --------------------------------------------------------------------------
# the schema
# --------------------------------------------------------------------------
def _yaml(**training):
    fields = {
        "batch_size": 1, "quantization": "none", "stream_layers": True,
        "gradient_accumulation_steps": 2, "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
    }
    fields.update(training)
    return yaml.safe_dump({
        "base": "hf-internal-testing/tiny-random-LlamaForCausalLM", "task": "sft",
        "data": {"train": "data.jsonl"}, "training": fields,
    })


def test_the_spill_dir_loads_with_l2l_and_may_point_anywhere(tmp_path):
    from soup_cli.config.loader import load_config_from_string

    elsewhere = os.path.abspath(os.sep + "spill-volume")
    cfg = load_config_from_string(_yaml(l2l=True, l2l_spill_dir=elsewhere))
    assert cfg.training.l2l_spill_dir == elsewhere
    assert load_config_from_string(_yaml(l2l=True)).training.l2l_spill_dir is None


def test_the_spill_dir_without_l2l_is_refused():
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match=r"l2l_spill_dir.*l2l"):
        load_config_from_string(_yaml(l2l_spill_dir="spill"))


@pytest.mark.parametrize(
    "value,match", [("sp\x00ill", "null"), ("x" * 4097, "4096"), (" ", "non-empty")]
)
def test_a_bad_spill_dir_is_refused(value, match):
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match=match):
        load_config_from_string(_yaml(l2l=True, l2l_spill_dir=value))
