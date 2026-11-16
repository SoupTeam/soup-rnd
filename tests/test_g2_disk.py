"""Regression checks for the G2 sustained spill-drive instrument (benchmarks/harness/g2_disk.py).

CPU only, tmp_path only, kilobyte-sized files, sub-second arms.
"""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def g2():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("g2_disk", HARNESS / "g2_disk.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["g2_disk"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# summary math
# ---------------------------------------------------------------------------
def _rows(rates, *, busy_fraction=0.5, tail=None):
    """Cumulative rows sampled every second at the given per-second write rates, plus an
    optional (seconds, rate) tail sample that ends mid-window."""
    rows, total, t = [], 0.0, 0.0
    steps = [(1.0, r) for r in rates] + ([tail] if tail else [])
    for seconds, rate in steps:
        t += seconds
        total += rate * seconds
        row = {"elapsed_s": t}
        for direction in ("write", "read", "weight"):
            moved = total if direction == "write" else 0.0
            row[f"total_{direction}_bytes"] = moved
            row[f"{direction}_busy_s"] = t * busy_fraction if direction == "write" else 0.0
        rows.append(row)
    return rows


def test_summary_excludes_warmup_drops_partial_window_and_reports_p10(g2):
    # 3 s of warmup at 1000 B/s, six full windows at 10..60 B/s, then half a window at
    # 5000 B/s. Neither the warmup nor the partial tail may move any statistic.
    rows = _rows([1000, 1000, 1000, 10, 20, 30, 40, 50, 60], tail=(0.5, 5000))
    out = g2.summarize(rows, warmup_s=3, window_s=1, directions=("write",))
    wall = out["write"]["wall"]
    assert out["n_windows"] == 6 and out["measured_s"] == 6
    assert wall["windows_Bps"] == pytest.approx([10, 20, 30, 40, 50, 60])
    assert wall["mean_Bps"] == pytest.approx(35)
    assert wall["min_Bps"] == pytest.approx(10)
    assert wall["p10_Bps"] == pytest.approx(15)  # linear: 10 + 0.5 * (20 - 10)
    assert wall["first_5_Bps"] == pytest.approx([10, 20, 30, 40, 50])
    assert wall["last_5_Bps"] == pytest.approx([20, 30, 40, 50, 60])
    # Busy rate: all bytes over all I/O seconds of the full windows (0.5 s busy per s).
    assert out["write"]["busy"]["mean_Bps"] == pytest.approx(210 / 3)
    assert out["read"] is None and out["weight"] is None


def test_summary_keeps_every_full_window_including_outliers(g2):
    rows = _rows([10, 10, 10, 10_000, 10])
    out = g2.summarize(rows, warmup_s=0, window_s=1, directions=("write",))
    assert out["write"]["wall"]["windows_Bps"] == pytest.approx([10, 10, 10, 10_000, 10])
    assert out["write"]["wall"]["mean_Bps"] == pytest.approx(10_040 / 5)


def test_windows_interpolate_between_samples_that_straddle_a_boundary(g2):
    # Samples at 1.5 s and 3 s of a steady 100 B/s stream: the [0,1), [1,2), [2,3)
    # windows must each see 100 bytes, not the lumps the samples happened to carry.
    rows = []
    for t in (1.5, 3.0):
        rows.append({"elapsed_s": t, "total_write_bytes": 100 * t, "write_busy_s": t,
                     "total_read_bytes": 0, "read_busy_s": 0,
                     "total_weight_bytes": 0, "weight_busy_s": 0})
    out = g2.summarize(rows, warmup_s=0, window_s=1, directions=("write",))
    assert out["write"]["wall"]["windows_Bps"] == pytest.approx([100, 100, 100])


def test_no_full_window_after_warmup_gives_no_rate(g2):
    rows = _rows([10, 10, 10])
    out = g2.summarize(rows, warmup_s=2.5, window_s=1, directions=("write",))
    assert out["n_windows"] == 0 and out["write"]["wall"]["mean_Bps"] is None
    assert out["write"]["busy"]["mean_Bps"] is None


def test_a_counter_that_goes_backwards_is_refused(g2):
    rows = _rows([10, 10, 10])
    rows[2]["total_write_bytes"] = 5
    with pytest.raises(ValueError, match="decreased"):
        g2.summarize(rows, warmup_s=0, window_s=1)


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------
def test_read_back_mismatches_are_detected(g2):
    pattern = g2.ChunkPattern(16384, seed=7)
    buf = pattern.buffer()
    pattern.stamp(buf, 32768, 3)
    pattern.verify(buf, 32768, 3, body=True)  # the bytes as written pass

    stale = buf.clone()
    with pytest.raises(g2.IntegrityError, match="pass 3"):
        pattern.verify(stale, 32768, 4, body=False)  # an old pass, not the latest write
    with pytest.raises(g2.IntegrityError, match="offset 32768"):
        pattern.verify(stale, 0, 3, body=False)  # a chunk from another place

    torn = buf.clone()
    g2.ChunkPattern.words(torn)[2, 1] += g2.SECTOR  # one sector of another chunk
    with pytest.raises(g2.IntegrityError, match="sector 2"):
        pattern.verify(torn, 32768, 3, body=False)

    flipped = buf.clone()
    flipped[3 * g2.SECTOR + 100] ^= 0xFF
    pattern.verify(flipped, 32768, 3, body=False)  # sampled-out body: stamps still fine
    with pytest.raises(g2.IntegrityError, match="sector 3"):
        pattern.verify(flipped, 32768, 3, body=True)


# ---------------------------------------------------------------------------
# arms: backend, ENOSPC, cleanup, integrity, record
# ---------------------------------------------------------------------------
class _Watch:
    def __init__(self, outcome="ok"):
        self.outcome = outcome
        self.record = None

    def start(self):
        pass

    def stop(self):
        self.record = {"outcome": self.outcome, "checks": {}}
        return self.record


@pytest.fixture
def quiet(g2, monkeypatch):
    """Fast, deterministic box sensors: no nvidia-smi, tasklist or PowerShell."""
    monkeypatch.setattr(g2.l2l_box, "box_stamp", lambda: {"stub": True})
    monkeypatch.setattr(g2, "git_head", lambda: "0" * 40)
    monkeypatch.setattr(g2, "volume_info", lambda path: {
        "path": os.path.realpath(path), "fstype": "testfs", "scope": "local",
        "scope_reason": "test", "disk": {"key": None, "id": None, "model": None}})


def _arm(g2, tmp_path, mode, *extra, name="arm", outcome="ok"):
    args = g2.build_parser().parse_args([
        "run", "--mode", mode, "--out", str(tmp_path / name),
        "--spill-dir", str(tmp_path / "spill"), "--spill-bytes", "64KiB",
        "--chunk-bytes", "16KiB", "--layers", "2", "--duration", "0.4", "--warmup", "0",
        "--window", "0.1", "--reserve-bytes", "0", "--verify", "full", *extra])
    return g2.run_arm(args, str(tmp_path / name), watch_factory=lambda *_: _Watch(outcome),
                      ac_power=lambda: True)


def _spill_files(tmp_path):
    spill = tmp_path / "spill"
    return sorted(p.name for p in spill.iterdir()) if spill.exists() else []


@pytest.mark.parametrize("mode", ["write", "read", "cycle"])
def test_smoke_arm_measures_verifies_and_removes_its_spill(
    g2, quiet, tmp_path, mode, monkeypatch,
):
    sample = g2._sample_row

    def delayed_final_sample(*args):
        row = sample(*args)
        if row["elapsed_s"] >= 0.4:
            row["elapsed_s"] += 0.1
            row["interval_s"] += 0.1
        return row

    monkeypatch.setattr(g2, "_sample_row", delayed_final_sample)
    record = _arm(g2, tmp_path, mode, "--smoke", "--buffered")
    assert record["status"] == "smoke_passed", record["invalid_reasons"]
    assert record["kind"] == "SMOKE, excluded" and record["direct"] is False
    assert record["measured_s"] == pytest.approx(0.4)  # every window of --duration kept
    saved = json.loads((tmp_path / "arm" / "arm.json").read_text())
    assert saved["status"] == "smoke_passed"
    rows = (tmp_path / "arm" / "samples.jsonl").read_text().splitlines()
    assert rows and "ram_available_bytes" in json.loads(rows[0])
    assert json.loads(rows[-1])["elapsed_s"] >= 0.5  # the deadline caps the summary
    assert record["verify"]["stamp_checks"] > 0
    assert _spill_files(tmp_path) == []
    assert record["spill_file"]["removed"] is True


def test_direct_spill_round_trip_uses_shipped_io_helpers(g2, tmp_path, monkeypatch):
    import io

    # Keep the real read/write helpers, replacing only the OS direct-I/O opener.
    monkeypatch.setattr(g2.l2l_activations, "_open_unbuffered",
                        lambda path: io.FileIO(path, "r+b"))
    pattern = g2.ChunkPattern(16384, seed=7)
    written, restored = pattern.buffer(), pattern.buffer()
    pattern.stamp(written, 0, 3)
    path = tmp_path / "direct.bin"
    with g2.SpillFile(str(path), 16384, direct=True) as spill:
        spill.write(0, written)
        spill.read(0, restored)
        pattern.verify(restored, 0, 3, body=True)
    assert not path.exists()


def test_source_hashes_track_the_shipped_io_implementation(g2, tmp_path, monkeypatch):
    import hashlib

    name = "src/soup_cli/utils/l2l_activations.py"
    source = tmp_path / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"before")
    monkeypatch.setattr(g2, "ROOT", tmp_path)
    assert g2.source_hashes()[name] == hashlib.sha256(b"before").hexdigest()
    source.write_bytes(b"after")
    assert g2.source_hashes()[name] == hashlib.sha256(b"after").hexdigest()


def test_buffered_backend_is_refused_for_a_candidate(g2, quiet, tmp_path):
    with pytest.raises(g2.BackendRefusedError):
        g2.check_backend(smoke=False, buffered=True)
    record = _arm(g2, tmp_path, "write", "--buffered")
    assert record["status"] == "invalid"
    assert record["error_code"] == "G2_BUFFERED_REFUSED"
    assert _spill_files(tmp_path) == []  # refused before any file was created
    assert json.loads((tmp_path / "arm" / "arm.json").read_text())["status"] == "invalid"


def test_status_needs_every_condition_for_candidate(g2):
    ok = dict(smoke=False, direct=True, error=None, f1_outcome="ok", n_windows=3,
              ac_power=True, sources_unchanged=True, scope_override=False)
    assert g2.decide_status(**ok) == ("candidate", [])
    assert g2.decide_status(**{**ok, "smoke": True})[0] == "smoke_passed"
    for change in ({"direct": False}, {"error": "x"}, {"f1_outcome": "void"},
                   {"f1_outcome": "unknown"}, {"n_windows": 0}, {"ac_power": None},
                   {"sources_unchanged": False}, {"scope_override": True}):
        assert g2.decide_status(**{**ok, **change})[0] == "invalid", change


def test_spill_write_maps_enospc_to_a_named_error(g2, tmp_path, monkeypatch):
    spill = g2.SpillFile(str(tmp_path / "s.bin"), 16384, direct=False)
    try:
        buf = g2.ChunkPattern(16384, seed=1).buffer()

        def full(*_):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(g2, "_write_all", full)
        with pytest.raises(g2.SpillDiskFullError, match="G2_SPILL_ENOSPC"):
            spill.write(0, buf)
    finally:
        spill.close()
    assert not (tmp_path / "s.bin").exists()


# 64 KiB spill in 16 KiB chunks: writes 1-4 are the untimed prefill, 5+ the timed cycle.
@pytest.mark.parametrize("fail_after", [2, 6])
def test_enospc_mid_arm_is_recorded_and_the_spill_removed(g2, quiet, tmp_path, monkeypatch,
                                                          fail_after):
    real = g2._write_all
    calls = {"n": 0}

    def fills_up(handle, offset, buf):
        calls["n"] += 1
        if calls["n"] > fail_after:
            raise OSError(errno.ENOSPC, "No space left on device")
        real(handle, offset, buf)

    monkeypatch.setattr(g2, "_write_all", fills_up)
    record = _arm(g2, tmp_path, "cycle", "--smoke", "--buffered")
    assert calls["n"] == fail_after + 1  # the failure stopped the arm at once
    assert record["status"] == "invalid"
    assert record["error_code"] == "G2_SPILL_ENOSPC"
    assert "G2_SPILL_ENOSPC" in record["error"]
    assert _spill_files(tmp_path) == []
    assert json.loads((tmp_path / "arm" / "arm.json").read_text())["error_code"] == \
        "G2_SPILL_ENOSPC"


def test_windows_disk_full_codes_count_as_enospc(g2):
    class WinDiskFullError(OSError):
        winerror = 112  # ERROR_DISK_FULL, whatever errno Python maps it to

    assert g2.is_enospc(WinDiskFullError(errno.EINVAL, "There is not enough space on the disk"))
    assert g2.is_enospc(OSError(errno.ENOSPC, "full"))
    assert not g2.is_enospc(OSError(errno.EIO, "I/O error"))
    assert not g2.is_enospc(RuntimeError("full"))


@pytest.mark.parametrize("failure", [RuntimeError("boom"), KeyboardInterrupt()])
def test_failed_or_interrupted_arm_is_written_and_cleaned(g2, quiet, tmp_path, monkeypatch,
                                                          failure):
    real = g2._read_all
    calls = {"n": 0}

    def breaks(handle, offset, buf):
        calls["n"] += 1
        if calls["n"] > 3:
            raise failure
        real(handle, offset, buf)

    monkeypatch.setattr(g2, "_read_all", breaks)
    record = _arm(g2, tmp_path, "read", "--smoke", "--buffered")
    assert record["status"] == "invalid" and record["error"]
    assert record["interrupted"] is isinstance(failure, KeyboardInterrupt)
    assert _spill_files(tmp_path) == []
    assert (tmp_path / "arm" / "arm.json").exists()


def test_corrupted_read_back_fails_the_arm(g2, quiet, tmp_path, monkeypatch):
    real = g2._read_all

    def corrupts(handle, offset, buf):
        real(handle, offset, buf)
        buf[5000] ^= 0x01  # one body byte in sector 1

    monkeypatch.setattr(g2, "_read_all", corrupts)
    record = _arm(g2, tmp_path, "read", "--smoke", "--buffered")
    assert record["status"] == "invalid"
    assert record["error_code"] == "G2_SPILL_INTEGRITY"
    assert _spill_files(tmp_path) == []


def test_void_f1_invalidates_even_a_smoke_arm(g2, quiet, tmp_path):
    record = _arm(g2, tmp_path, "write", "--smoke", "--buffered", outcome="void")
    assert record["status"] == "invalid"
    assert any("F1" in reason for reason in record["invalid_reasons"])


def test_concurrent_arm_measures_spill_and_weights_separately(g2, quiet, tmp_path):
    weights = tmp_path / "w" / "weights.bin"
    g2.prepare_weights(str(weights), 65536, reserve_bytes=0, block=16384)
    record = _arm(g2, tmp_path, "cycle+weights", "--smoke", "--buffered",
                  "--weights", str(weights), "--weight-chunk-bytes", "16KiB")
    assert record["status"] == "smoke_passed", record["invalid_reasons"]
    summary = record["summary"]
    assert summary["write"]["bytes"] > 0 and summary["read"]["bytes"] > 0
    assert summary["weight"]["bytes"] > 0
    rates = g2.export_rates(record)
    assert rates["weight_with_spill_Bps"] > 0 and rates["weight_ref_Bps"] is None
    assert rates["fixture_exceeds_ram"] is False and rates["status"] == "smoke_passed"
    assert record["volumes"]["same_physical_disk"] is None  # mapping unknown here
    assert _spill_files(tmp_path) == []


def test_weight_mode_without_a_certified_fixture_is_invalid(g2, quiet, tmp_path):
    stray = tmp_path / "stray.bin"
    stray.write_bytes(b"\x00" * 16384)
    record = _arm(g2, tmp_path, "weights", "--smoke", "--buffered", "--weights", str(stray),
                  "--weight-chunk-bytes", "16KiB")
    assert record["status"] == "invalid" and record["error_code"] == "G2_FIXTURE"


# ---------------------------------------------------------------------------
# fixtures and volumes
# ---------------------------------------------------------------------------
def test_weight_fixture_certificate_and_tamper_refusal(g2, tmp_path):
    path = tmp_path / "w" / "weights.bin"
    cert = g2.prepare_weights(str(path), 65536, reserve_bytes=0, block=16384)
    assert cert["status"] == "complete" and cert["size_bytes"] == 65536
    assert cert["sha256"] == g2.sha256_file(str(path))
    assert g2.check_fixture(str(path), weight_chunk_bytes=16384, rehash=True)["rehash"] == \
        cert["sha256"]
    with pytest.raises(FileExistsError):  # exclusive creation: never overwritten
        g2.prepare_weights(str(path), 65536, reserve_bytes=0, block=16384)
    with pytest.raises(ValueError, match="whole"):
        g2.check_fixture(str(path), weight_chunk_bytes=65536 * 2)
    with open(path, "r+b") as handle:
        handle.write(b"\x00")
    os.utime(path, ns=(cert["identity"]["mtime_ns"] + 10**9,) * 2)
    with pytest.raises(g2.FixtureError, match="changed"):
        g2.check_fixture(str(path), weight_chunk_bytes=16384)


def test_fixture_preparation_refuses_without_free_space(g2, tmp_path):
    with pytest.raises(g2.SpillDiskShortError, match="G2_SPILL_DISK_SHORT"):
        g2.prepare_weights(str(tmp_path / "w.bin"), 16384, reserve_bytes=1 << 62, block=16384)
    assert not (tmp_path / "w.bin").exists()


@pytest.mark.requires_symlink
def test_links_are_refused(g2, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link, target_is_directory=True)
    with pytest.raises(g2.FixtureError, match="refused"):
        g2.check_no_links(str(link / "weights.bin"))


def test_ram_and_network_volumes_are_refused_unless_overridden(g2):
    for fstype in ("tmpfs", "overlay"):
        assert g2.classify_scope(fstype)[0] == "refused"
    assert g2.classify_scope("nfs4")[0] == "network"
    assert g2.classify_scope("ext4")[0] == "local"
    info = {"path": "/x", "scope": "refused", "scope_reason": "tmpfs is RAM"}
    with pytest.raises(g2.ScopeRefusedError):
        g2.check_scope(info, accept_nonlocal=False)
    assert g2.check_scope(info, accept_nonlocal=True) is True


def test_two_paths_are_not_two_disks_without_a_mapping(g2):
    known = {"disk": {"id": "nvme0n1"}}
    assert g2.same_disk(known, {"disk": {"id": "nvme0n1"}}) is True
    assert g2.same_disk(known, {"disk": {"id": "nvme1n1"}}) is False
    assert g2.same_disk(known, {"disk": {"id": None}}) is None


@pytest.mark.parametrize("device,model,vendor,virtual", [
    ("vda", None, "0x1af4", True),          # vast.ai KVM guest: no model, virtio vendor
    ("sda", "QEMU HARDDISK", "ATA", True),
    ("sdd", "Virtual Disk", "Msft", True),  # WSL2 / Hyper-V
    ("xvda", None, None, True),
    ("nvme0n1", "Samsung SSD 990 PRO 2TB", None, False),
    ("sda", None, None, None),             # nothing names it: unknown, not physical
])
def test_a_hypervisor_disk_is_labelled_virtual(g2, device, model, vendor, virtual):
    assert g2.virtual_disk(device, model, vendor)[0] is virtual


# ---------------------------------------------------------------------------
# contention and export
# ---------------------------------------------------------------------------
def _summary(**busy):
    out = {}
    for direction in ("write", "read", "weight"):
        rate = busy.get(direction)
        out[direction] = None if rate is None else {
            "busy": {"mean_Bps": rate}, "wall": {"mean_Bps": rate / 2}}
    return out


def test_contention_ratio_and_spill_rates_under_contention(g2):
    out = g2.contention(_summary(weight=4e9), _summary(write=1e9, read=2e9, weight=3e9))
    assert out["weight_ratio"] == pytest.approx(0.75)
    assert out["weight_ref_Bps"] == 4e9 and out["weight_with_spill_Bps"] == 3e9
    assert out["spill_write_Bps"] == 1e9 and out["spill_read_Bps"] == 2e9
    assert g2.contention(_summary(), _summary(weight=1.0))["weight_ratio"] is None


EXPORT_KEYS = {"write_Bps", "read_Bps", "weight_ref_Bps", "weight_with_spill_Bps", "source",
               "mode", "status", "direct", "measured_s", "fixture_bytes", "ram_total_bytes",
               "fixture_exceeds_ram", "host", "spill_disk", "weight_disk"}


def _record(mode, status="candidate", *, write=None, read=None, weight=None, ram=16 << 30,
            spill=32 << 30, weights=32 << 30, measured=1800.0):
    sizes = {"spill": spill if mode in ("write", "read", "cycle", "cycle+weights") else None,
             "weights": weights if mode in ("weights", "cycle+weights") else None}
    used = [s for s in sizes.values() if s is not None]
    return {"mode": mode, "status": status, "direct": True, "measured_s": measured,
            "summary": _summary(write=write, read=read, weight=weight),
            "fixture_bytes": sizes, "ram_total_bytes": ram,
            "fixture_exceeds_ram": all(s > ram for s in used), "arm_path": f"/x/{mode}"}


def test_export_rates_keys_and_mode_specific_weight_rates(g2):
    ref = g2.export_rates(_record("weights", weight=4e9))
    assert set(ref) == EXPORT_KEYS
    assert ref["weight_ref_Bps"] == 4e9 and ref["weight_with_spill_Bps"] is None
    assert ref["write_Bps"] is None and ref["source"] == "/x/weights"
    busy = g2.export_rates(_record("cycle+weights", write=1e9, read=2e9, weight=3e9))
    assert busy["weight_ref_Bps"] is None and busy["weight_with_spill_Bps"] == 3e9
    assert (busy["write_Bps"], busy["read_Bps"]) == (1e9, 2e9)
    small = g2.export_rates(_record("cycle+weights", weights=8 << 30))
    assert small["fixture_exceeds_ram"] is False  # one fixture fits in RAM


def _session(tmp_path, records):
    session = tmp_path / "session"
    session.mkdir()
    arms = []
    for i, record in enumerate(records):
        name = f"p{i}"
        (session / name).mkdir()
        (session / name / "arm.json").write_text(json.dumps(record))
        arms.append({"dir": name, "mode": record["mode"]})
    (session / "manifest.json").write_text(json.dumps({"arms": arms}))
    return session


def test_export_session_prefers_spill_rates_under_contention(g2, tmp_path):
    session = _session(tmp_path, [
        _record("write", write=9e9), _record("read", read=9e9),
        _record("weights", weight=4e9), _record("weights", weight=2e9),
        _record("cycle+weights", write=1e9, read=2e9, weight=1e9),
        _record("cycle+weights", write=3e9, read=4e9, weight=3e9),
        _record("cycle+weights", "invalid", write=100e9, read=100e9, weight=100e9)])
    out = g2.export_session(str(session))
    assert out["write_Bps"] == pytest.approx(2e9) and out["read_Bps"] == pytest.approx(3e9)
    assert out["weight_ref_Bps"] == pytest.approx(3e9)
    assert out["weight_with_spill_Bps"] == pytest.approx(2e9)
    assert len(out["arms"]) == 7 and [a["status"] for a in out["arms"]].count("invalid") == 1
    assert all(set(arm) == EXPORT_KEYS for arm in out["arms"])


def test_export_session_falls_back_to_write_and_read_arms(g2, tmp_path):
    session = _session(tmp_path, [_record("write", write=1e9), _record("read", read=3e9),
                                  _record("read", read=5e9)])
    out = g2.export_session(str(session))
    assert (out["write_Bps"], out["read_Bps"]) == (1e9, 4e9)
    assert out["weight_ref_Bps"] is None and out["weight_with_spill_Bps"] is None


def test_export_session_lists_an_arm_without_a_record_as_invalid(g2, tmp_path):
    session = _session(tmp_path, [_record("write", write=1e9)])
    manifest = json.loads((session / "manifest.json").read_text())
    manifest["arms"].append({"dir": "p9", "mode": "read"})
    (session / "manifest.json").write_text(json.dumps(manifest))
    out = g2.export_session(str(session))
    assert out["arms"][-1]["status"] == "invalid" and out["read_Bps"] is None


def test_export_session_refuses_a_manifest_that_escapes_the_session(g2, tmp_path):
    session = _session(tmp_path, [])
    (session / "manifest.json").write_text(json.dumps({"arms": [{"dir": "../..",
                                                                  "mode": "write"}]}))
    with pytest.raises(ValueError, match="escapes"):
        g2.export_session(str(session))


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
def test_abba_order(g2):
    assert g2.abba_order(["write", "weights"]) == ["write", "weights", "weights", "write"]
    with pytest.raises(ValueError):
        g2.abba_order(["write", "bogus"])


def test_session_passes_every_arm_option_to_its_children(g2):
    argv = ["session", "--modes", "write", "--out", "x", "--spill-dir", "/s",
            "--spill-bytes", "1GiB", "--chunk-bytes", "4MiB", "--layers", "8",
            "--weights", "/w.bin", "--weight-rate", "2GB", "--duration", "60",
            "--warmup", "10", "--verify", "full", "--smoke", "--buffered", "--keep"]
    session = g2.build_parser().parse_args(argv)
    child = g2.build_parser().parse_args(
        ["run", "--mode", "write", "--out", "y", *g2._arm_argv(session)])
    for name in ("spill_dir", "spill_bytes", "chunk_bytes", "layers", "weights", "weight_rate",
                 "duration", "warmup", "verify", "smoke", "buffered", "keep",
                 "reserve_bytes", "stop_temperature", "seed"):
        assert getattr(child, name) == getattr(session, name), name


def test_export_session_never_uses_a_killed_or_failed_child(g2, tmp_path):
    session = _session(tmp_path, [_record("write", write=1e9), _record("write", write=9e9)])
    manifest = json.loads((session / "manifest.json").read_text())
    manifest["arms"][0]["exit_code"] = 0
    manifest["arms"][1].update(exit_code=1, orchestrator_error="TimeoutExpired(...)")
    (session / "manifest.json").write_text(json.dumps(manifest))
    out = g2.export_session(str(session))
    assert out["write_Bps"] == 1e9
    assert out["arms"][1]["status"] == "invalid" and "Timeout" in out["arms"][1][
        "session_problem"]


# A stand-in child: writes the record a real arm would and, in the third arm, leaves its
# spill file behind and hangs, as a child the session must kill would.
_FAKE_CHILD = r'''
import json, os, sys, time
argv = sys.argv[1:]
opt = lambda name: argv[argv.index(name) + 1]
out, mode, tag = opt("--out"), opt("--mode"), opt("--spill-tag")
os.makedirs(out)
safe = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in tag)
if out.endswith("p2_" + mode):
    open(os.path.join(opt("--spill-dir"), "g2-spill-" + safe + ".bin"), "wb").close()
    time.sleep(60)
rate = {"busy": {"mean_Bps": 1e9 if mode == "write" else 3e9}, "wall": {"mean_Bps": 1.0}}
record = {"mode": mode, "status": "smoke_passed", "direct": True, "measured_s": 1.0,
          "summary": {"write": rate if mode == "write" else None,
                      "read": rate if mode == "read" else None, "weight": None},
          "fixture_bytes": {"spill": 1, "weights": None}, "ram_total_bytes": 2,
          "fixture_exceeds_ram": False, "pid": os.getpid()}
json.dump(record, open(os.path.join(out, "arm.json"), "w"))
'''


def test_session_runs_abba_fresh_processes_and_survives_a_killed_child(g2, tmp_path):
    script = tmp_path / "child.py"
    script.write_text(_FAKE_CHILD)
    (tmp_path / "spill").mkdir()
    out = tmp_path / "session"
    manifest = g2.run_session(["write", "read"], str(out),
                              ["--spill-dir", str(tmp_path / "spill"), "--smoke"],
                              timeout_s=3, script=str(script),
                              spill_dir=str(tmp_path / "spill"))
    assert manifest["order"] == ["write", "read", "read", "write"]
    assert [arm["mode"] for arm in manifest["arms"]] == manifest["order"]
    assert manifest["status"] == "complete"
    statuses = [arm["status"] for arm in manifest["arms"]]
    assert statuses == ["smoke_passed", "smoke_passed", "invalid", "smoke_passed"]
    killed = manifest["arms"][2]
    assert "TimeoutExpired" in killed["orchestrator_error"] and killed["missing"]
    assert killed["spill_removed_by_session"]  # the hung child's file went
    assert _spill_files(tmp_path) == []
    pids = {json.loads((out / arm["dir"] / "arm.json").read_text())["pid"]
            for arm in manifest["arms"] if arm["status"] != "invalid"}
    assert len(pids) == 3  # a fresh process per arm
    rates = json.loads((out / "rates.json").read_text())
    assert rates["write_Bps"] == 1e9 and rates["read_Bps"] == 3e9  # from the valid arms
    assert len(rates["arms"]) == 4


def test_mountinfo_picks_the_longest_mount_that_holds_the_path(g2):
    text = "\n".join([
        "22 1 259:2 / / rw,relatime - ext4 /dev/nvme0n1p2 rw",
        "30 22 259:5 / /mnt/nvme rw,relatime - xfs /dev/nvme1n1 rw",
        "31 22 0:40 / /mnt/nvme2 rw - tmpfs tmpfs rw",
        "32 22 0:41 / /mnt/my\\040disk rw - nfs4 server:/x rw",
    ])
    assert g2.parse_mountinfo(text, "/mnt/nvme/g2-spill")["fstype"] == "xfs"
    assert g2.parse_mountinfo(text, "/mnt/nvme2/x")["fstype"] == "tmpfs"
    assert g2.parse_mountinfo(text, "/mnt/nvmex/x")["fstype"] == "ext4"
    assert g2.parse_mountinfo(text, "/mnt/my disk/x")["fstype"] == "nfs4"
    assert g2.classify_scope("unknown")[0] == "unknown"  # unreadable is not local


def test_paced_weight_stream_never_runs_ahead_of_its_rate(g2, monkeypatch):
    import io
    from types import SimpleNamespace

    chunk, rate = 4096, 400_000  # ~100 requests per second
    now = 0.0
    arrivals = []

    class TimedReader(io.BytesIO):
        def readinto(self, buffer):
            arrivals.append(now)
            return super().readinto(buffer)

    class StopAfter50Reads:
        def is_set(self):
            return len(arrivals) >= 50

        def wait(self, delay):
            nonlocal now
            now += delay

    monkeypatch.setattr(g2, "time", SimpleNamespace(monotonic=lambda: now,
                                                   perf_counter=lambda: now))
    handle = TimedReader(b"\x01" * chunk * 8)
    buf = g2.l2l_activations.aligned_buffer(chunk, pin=False)
    meter = g2.Meter()
    g2.weights_loop(handle, chunk * 8, buf, meter, StopAfter50Reads(), rate_Bps=rate)
    assert meter.snapshot()["bytes"]["weight"] == 50 * chunk
    assert arrivals == pytest.approx([index * chunk / rate for index in range(50)])
