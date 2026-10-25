"""CPU tests for the A4 machine admission (benchmarks/harness/a4_admission.py).

A fake clock and an in-memory backend drive the real ``exercise`` / ``assess`` path:
a steady volume is admitted, a volume that stalls inside the run is refused, a stall
that crosses a window boundary is charged to both windows, and a volume that returns
other bytes is refused.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load

CHUNK = 4096


@pytest.fixture()
def adm():
    return _load("a4_admission")


class FakeVolume:
    """Positional writes and reads that advance a shared clock by a per-op cost."""

    def __init__(self, cost_s=0.01, stall_at=None, stall_s=0.0, corrupt=False):
        self.now = 0.0
        self.cost_s = cost_s
        self.stall_at, self.stall_s = stall_at, stall_s
        self.corrupt = corrupt
        self.data = {}
        self.ops = 0

    def clock(self):
        return self.now

    def _tick(self):
        self.ops += 1
        self.now += self.cost_s
        if self.stall_at is not None and self.stall_at[0] <= self.now < self.stall_at[1]:
            self.now += self.stall_s
            self.stall_at = None

    def backend(self, path, size):
        volume = self

        class Spill:
            def write(self, offset, buf):
                volume.data[offset] = buf.clone()
                volume._tick()

            def read(self, offset, buf):
                buf.copy_(volume.data[offset])
                if volume.corrupt:
                    buf[0] ^= 1
                volume._tick()

            def close(self, remove=True):
                pass

        return Spill()


def _chunks(n=4):
    import torch

    return [torch.full((CHUNK,), i + 1, dtype=torch.uint8) for i in range(n)]


def _run(adm, volume, tmp_path, duration=60.0, window=10.0):
    run = adm.exercise(str(tmp_path), duration_s=duration, chunk=CHUNK, n_layers=4, k=2,
                       clock=volume.clock, backend=volume.backend, chunks=_chunks())
    return adm.assess(run, duration_s=duration, window_s=window)


def test_a_steady_volume_is_admitted(adm, tmp_path):
    record = _run(adm, FakeVolume(), tmp_path)
    assert record["judgement"]["admitted"], record["judgement"]["problems"]
    assert len(record["windows"]) == 6
    assert record["mismatches"] == 0 and record["cycles"] > 0
    assert len(record["per_second"]) == 60


def test_a_stall_inside_the_run_is_refused(adm, tmp_path):
    """One write that takes 8 s in the fourth window: that window's write rate falls far
    below half the median, so the machine is refused."""
    volume = FakeVolume(stall_at=(31.0, 31.1), stall_s=8.0)
    record = _run(adm, volume, tmp_path)
    judgement = record["judgement"]
    assert not judgement["admitted"]
    low = judgement["write"]["low"] or judgement["read"]["low"]
    assert low and low[0]["index"] == 3
    assert any("below 50% of the median" in p for p in judgement["problems"])


def test_a_dip_just_above_half_the_median_is_admitted(adm, tmp_path):
    """The threshold is fixed at 50%: a window at ~60% of the median passes."""
    volume = FakeVolume(stall_at=(31.0, 31.1), stall_s=0.0)
    slow_until = 40.0

    def tick():
        volume.ops += 1
        volume.now += volume.cost_s * (1.6 if 30.0 <= volume.now < slow_until else 1.0)

    volume._tick = tick
    record = _run(adm, volume, tmp_path)
    assert record["judgement"]["admitted"], record["judgement"]["problems"]
    rates = [w["write_gb_per_s"] for w in record["windows"]]
    assert min(rates) < 0.7 * max(rates)


def test_an_operation_across_a_boundary_is_shared_by_both_windows(adm):
    ops = [("write", 9.0, 13.0, 4000)]
    windows = adm.bins(ops, 0.0, 10.0, 2)
    assert windows[0]["write_busy_s"] == pytest.approx(1.0)
    assert windows[1]["write_busy_s"] == pytest.approx(3.0)
    assert windows[0]["write_bytes"] == pytest.approx(1000)
    assert windows[1]["write_bytes"] == pytest.approx(3000)
    assert windows[0]["read_gb_per_s"] is None


def test_a_volume_that_returns_other_bytes_is_refused(adm, tmp_path):
    record = _run(adm, FakeVolume(corrupt=True), tmp_path)
    assert not record["judgement"]["admitted"]
    assert record["mismatches"] == record["cycles"]


def test_the_store_order_writes_forward_and_reads_backward(adm):
    write, read = adm.spill_order(3, 2, 10)
    assert write == [0, 10, 20, 30, 40, 50]
    assert read == [40, 50, 20, 30, 0, 10]


def test_the_cli_refuses_a_refused_volume_and_runs_a_smoke(adm, tmp_path, monkeypatch):
    import a4_box

    monkeypatch.setattr(a4_box, "spill_volume", lambda d, accept_network=False: {
        "ok": False, "refusals": ["tmpfs"]})
    monkeypatch.setattr(a4_box, "placement", lambda d: {})
    out = tmp_path / "admission.json"
    code = adm.main(["--spill-dir", str(tmp_path / "s"), "--out", str(out),
                     "--log", str(tmp_path / "s.csv")])
    assert code == 3 and "volume refused" in out.read_text()


def test_the_slow_write_listing_is_per_comparison_and_skips_no_spill_arms(adm, tmp_path):
    import json

    def arm(comparison, name, spill_layers, rate):
        path = tmp_path / "session" / "arms" / comparison / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"records": [{
            "kind": "arm", "skipped": None, "block": int(name[1:3]),
            "plan": {"spill_layers": spill_layers}, "spill_write_gb_per_s": rate}]}))

    arm("ram_k2_half", "b00_p0_A.json", 0, None)
    arm("ram_k2_half", "b00_p1_B.json", 16, 0.9)   # below 0.5 x 2.0
    arm("ram_k2_half", "b00_p2_B.json", 16, 1.0)   # exactly the floor: not below
    arm("disk_k4_same_drive", "b03_p1_B.json", 32, 0.2)
    admission = {"judgement": {"write": {"median_gb_per_s": 2.0}}}
    out = adm.slow_write_arms(str(tmp_path / "session"), admission)
    assert out["description_only"] and out["floor_gb_per_s"] == 1.0
    ram = out["comparisons"]["ram_k2_half"]
    assert ram["spill_arms"] == 2
    assert [b["arm"] for b in ram["below"]] == ["ram_k2_half/b00_p1_B.json"]
    disk = out["comparisons"]["disk_k4_same_drive"]
    assert disk["below"][0]["block"] == 3
