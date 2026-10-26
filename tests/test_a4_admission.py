"""CPU tests for the A4 machine admission, protocol v3 (benchmarks/harness/a4_admission.py).

A fake clock (whose sleep advances it) and an in-memory volume drive the real ``exercise``
/ ``assess`` path on the protocol's own schedule: 30 minutes, one cycle every 1.153 s, a
lateness limit of 5 s. Each fake operation stands for an eighth of the real 512 MiB cycle
file, so a volume's GB/s here means what it means on a box.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.test_l2l_probe_harness import _load

ROOT = Path(__file__).resolve().parents[1]
V1_RECORDS = ROOT / "benchmarks" / "results" / "a4-rtx3090-20261025"
CHUNK = 4096
N_LAYERS, K = 4, 2


@pytest.fixture()
def adm():
    return _load("a4_admission")


class FakeVolume:
    """Positional writes and reads that cost (an eighth of a cycle file) / rate seconds."""

    def __init__(self, adm, write_gbps=3.0, read_gbps=3.0, slow=None, stalls=(),
                 corrupt=False, drop_writes_after=None, leave_file=False):
        self.op_bytes = adm.CYCLE_BYTES / (N_LAYERS * K)
        self.now = 0.0
        self.rate = {"write": write_gbps, "read": read_gbps}
        self.slow = slow  # (from_s, to_s, write GB/s in between)
        self.stalls = sorted(stalls)  # (at_s, extra_s): once, after the first op past at_s
        self.corrupt = corrupt
        self.drop_writes_after = drop_writes_after
        self.leave_file = leave_file
        self.data = {}
        self.writes = 0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def _cost(self, kind):
        rate = self.rate[kind]
        if kind == "write" and self.slow and self.slow[0] <= self.now < self.slow[1]:
            rate = self.slow[2]
        self.now += self.op_bytes / (rate * 1e9)
        while self.stalls and self.stalls[0][0] <= self.now:
            self.now += self.stalls.pop(0)[1]

    def backend(self, path, size):
        volume = self

        class Spill:
            def write(self, offset, buf):
                volume.writes += 1
                if volume.drop_writes_after is None or volume.writes <= volume.drop_writes_after:
                    volume.data[offset] = buf.clone()
                volume._cost("write")

            def read(self, offset, buf):
                buf.copy_(volume.data[offset])
                if volume.corrupt:
                    buf[-1] ^= 1
                volume._cost("read")

            def close(self, remove=True):
                if volume.leave_file:
                    Path(path).write_bytes(b"")

        return Spill()


def _chunks(n=4):
    import torch

    return [torch.full((CHUNK,), i + 1, dtype=torch.uint8) for i in range(n)]


def _run(adm, volume, tmp_path):
    run = adm.exercise(str(tmp_path), chunk=CHUNK, n_layers=N_LAYERS, k=K, clock=volume.clock,
                       sleep=volume.sleep, backend=volume.backend, chunks=_chunks())
    return adm.assess(run)


def test_a_steady_volume_is_admitted_with_every_second_logged(adm, tmp_path):
    record = _run(adm, FakeVolume(adm), tmp_path)
    assert record["judgement"]["admitted"], record["judgement"]["problems"]
    assert record["mismatched_chunks"] == 0 and record["chunks_read"] > 0
    assert len(record["per_second"]) == 1800


def test_without_stalls_every_cycle_starts_on_its_schedule(adm, tmp_path):
    """The pause is the protocol: 1,561 cycles in 30 min, each started on time (±1%)."""
    record = _run(adm, FakeVolume(adm), tmp_path)
    rows = record["cycle_rows"]
    assert len(rows) == record["scheduled_cycles"] == math.ceil(1800 / adm.PERIOD_S)
    for index, (planned, start, _end, _lateness, _bad) in enumerate(rows):
        assert planned == pytest.approx(index * adm.PERIOD_S)
        assert abs(start - planned) <= 0.01 * adm.PERIOD_S


def test_a_volume_that_dips_but_still_covers_the_demand_is_admitted(adm, tmp_path):
    """10 -> 3 GB/s for ten minutes. v2 refused this (windows far below their own median);
    the schedule never falls behind, so v3 admits it."""
    volume = FakeVolume(adm, write_gbps=10.0, read_gbps=10.0, slow=(600.0, 1200.0, 3.0))
    record = _run(adm, volume, tmp_path)
    assert record["judgement"]["admitted"], record["judgement"]["problems"]
    assert record["judgement"]["max_lateness_s"] < 0
    assert record["description_only"]["stability"]["write"]["low"]


def test_a_steady_volume_slower_than_the_tested_rate_is_refused(adm, tmp_path):
    """A steady 0.3 GB/s write. v2 admitted this (no window below half its own median); it
    cannot write 512 MiB every 1.153 s, so the schedule falls behind and v3 refuses it."""
    record = _run(adm, FakeVolume(adm, write_gbps=0.3), tmp_path)
    assert not record["judgement"]["admitted"]
    assert record["judgement"]["max_lateness_s"] > 100
    assert not record["description_only"]["stability"]["write"]["low"]


def test_a_long_dip_to_the_rate_of_v1s_slow_arms_is_refused(adm, tmp_path):
    record = _run(adm, FakeVolume(adm, slow=(600.0, 900.0, 0.1)), tmp_path)
    assert not record["judgement"]["admitted"]


@pytest.mark.parametrize("stall_s, admitted", [(3.0, True), (8.0, False)])
def test_one_stall_is_absorbed_only_within_the_limit(adm, tmp_path, stall_s, admitted):
    record = _run(adm, FakeVolume(adm, stalls=[(900.0, stall_s)]), tmp_path)
    assert record["judgement"]["admitted"] is admitted


@pytest.mark.parametrize("lateness_s, admitted", [(4.9, True), (5.1, False)])
def test_the_lateness_limit_is_five_seconds(adm, tmp_path, lateness_s, admitted):
    cycle_s = 2 * adm.CYCLE_BYTES / 3e9  # the file written and read back at 3 GB/s
    volume = FakeVolume(adm, stalls=[(900.0, lateness_s + adm.PERIOD_S - cycle_s)])
    record = _run(adm, volume, tmp_path)
    assert record["judgement"]["max_lateness_s"] == pytest.approx(lateness_s)
    assert record["judgement"]["admitted"] is admitted


def test_a_volume_that_returns_other_bytes_is_refused(adm, tmp_path):
    record = _run(adm, FakeVolume(adm, corrupt=True), tmp_path)
    assert not record["judgement"]["admitted"]
    assert record["mismatched_chunks"] == record["chunks_read"] > 0


def test_a_volume_that_drops_writes_is_refused(adm, tmp_path):
    """After its first cycle the volume keeps the old bytes; only the cycle stamp tells."""
    record = _run(adm, FakeVolume(adm, drop_writes_after=N_LAYERS * K), tmp_path)
    assert not record["judgement"]["admitted"]
    assert record["mismatched_chunks"] == record["chunks_read"] - N_LAYERS * K


def test_a_test_file_left_behind_is_refused(adm, tmp_path):
    record = _run(adm, FakeVolume(adm, leave_file=True), tmp_path)
    assert record["left_behind"] and not record["judgement"]["admitted"]


def test_the_demand_is_v1s_largest_spill_arm(adm):
    demand = adm.v1_demand(str(V1_RECORDS))
    assert demand == {"bytes_per_s": adm.DEMAND_BYTES_PER_S, "arm": "ram_k3_min/b05_p1_B.json"}
    assert adm.PERIOD_S == pytest.approx(1.1532, abs=1e-4)


def test_the_store_order_writes_forward_and_reads_backward(adm):
    write, read = adm.spill_order(3, 2, 10)
    assert write == [0, 10, 20, 30, 40, 50]
    assert read == [40, 50, 20, 30, 0, 10]


def test_an_operation_across_a_boundary_is_shared_by_both_seconds(adm):
    seconds = adm.bins([("write", 0.9, 1.3, 4000)], 0.0, 1.0, 2)
    assert seconds[0]["write_busy_s"] == pytest.approx(0.1)
    assert seconds[1]["write_busy_s"] == pytest.approx(0.3)
    assert seconds[0]["write_bytes"] == pytest.approx(1000)
    assert seconds[1]["write_bytes"] == pytest.approx(3000)
    assert seconds[0]["read_gb_per_s"] is None


def test_the_cli_refuses_a_refused_volume(adm, tmp_path, monkeypatch):
    import a4_box

    monkeypatch.setattr(a4_box, "spill_volume", lambda d, accept_network=False: {
        "ok": False, "refusals": ["tmpfs"]})
    monkeypatch.setattr(a4_box, "placement", lambda d: {})
    out = tmp_path / "admission.json"
    code = adm.main(["--spill-dir", str(tmp_path / "s"), "--out", str(out),
                     "--log", str(tmp_path / "s.csv"), "--cycles", str(tmp_path / "c.csv")])
    assert code == 3 and "volume refused" in out.read_text()


def test_the_arm_description_is_per_comparison_against_the_demand(adm, tmp_path):
    demand = adm.DEMAND_BYTES_PER_S

    def arm(comparison, name, spill_layers, rate, asked=0.0):
        path = tmp_path / "arms" / comparison / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"records": [{
            "kind": "arm", "skipped": None, "block": int(name[1:3]),
            "plan": {"spill_layers": spill_layers}, "spill_write_gb_per_s": rate,
            "records": [{"step_s": 2.0, "spill_written": 2.0 * asked}] * 3}]}))

    arm("ram_k2_half", "b00_p0_A.json", 0, None)
    arm("ram_k2_half", "b00_p1_B.json", 16, 2.0, asked=0.5 * demand)
    arm("ram_k2_half", "b00_p2_B.json", 16, 0.3, asked=0.5 * demand)  # D < 0.3 < 2 D GB/s
    arm("disk_k4_same_drive", "b03_p1_B.json", 32, 0.1, asked=1.5 * demand)
    out = adm.describe_arms(str(tmp_path))
    assert out["description_only"]
    ram = out["comparisons"]["ram_k2_half"]
    assert ram["spill_arms"] == 2
    assert ram["below_tested_rate"] == ["ram_k2_half/b00_p2_B.json"]
    assert ram["below_demand"] == [] and ram["demand_above"] == []
    assert ram["arms"][0]["write_to_demand"] == pytest.approx(2.0e9 / demand)
    disk = out["comparisons"]["disk_k4_same_drive"]
    assert disk["below_demand"] == disk["demand_above"] == ["disk_k4_same_drive/b03_p1_B.json"]
