"""CPU tests for the A3 stage-2 machine admission (benchmarks/harness/a3_admission.py).

As A4's admission tests do: a fake clock (whose sleep advances it) and an in-memory volume
drive the real ``exercise`` / ``assess`` path on the schedule the two input files set. The
inputs are a calibration record and a no-spill A record shaped as the stand writes them,
with a 4 KiB slot so the cycles stay small. The fake volume's rates are multiples of the
demand D, so "below 2D" means what it means on a box.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import pytest

from tests.test_l2l_probe_harness import _load

SLOT = 4096
K = 2
N_LAYERS = 32
A_STEP_S = 2.0  # the A arm's median timed step: period A_STEP_S / 2


@pytest.fixture()
def adm():
    return _load("a3_admission")


def _plan(**over):
    plan = {"spill_layers": 0, "n_layers": N_LAYERS, "k": K, "chunk_bytes": SLOT,
            "slot_bytes": SLOT, "ring": 4, "pin": True}
    return {**plan, **over}


def _inputs(tmp_path, *, plan=None, a=None, a_value=A_STEP_S):
    calibration = {"arm": "calibration", "kind": "calibrate", "device": "cuda",
                   "exit": "normal", "spill_plan": _plan(**(plan or {}))}
    arm = {"arm": "cost/b00_p0_A", "kind": "cost", "role": "A", "device": "cuda",
           "exit": "normal", "spill_plan": _plan(), "value": a_value,
           "timed_step_s": [a_value] * 5, "env": {"gpu_facts": {"gpu": "RTX 3090"}}}
    arm.update(a or {})
    plan_path, a_path = tmp_path / "calibration.json", tmp_path / "A.json"
    plan_path.write_text(json.dumps(calibration))
    a_path.write_text(json.dumps(arm))
    return plan_path, a_path


class FakeVolume:
    """Positional writes and reads that cost one chunk / (rate x D) seconds."""

    def __init__(self, demand, write_d=10.0, read_d=10.0, slow=None, stalls=(),
                 corrupt=False, drop_writes_after=None, leave_file=False):
        self.demand = demand
        self.now = 0.0
        self.rate = {"write": write_d, "read": read_d}
        self.slow = slow  # (from_s, to_s, rate in D for both directions in between)
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
        if self.slow and self.slow[0] <= self.now < self.slow[1]:
            rate = self.slow[2]
        self.now += SLOT / (rate * self.demand)
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

    return [torch.full((SLOT,), i + 1, dtype=torch.uint8) for i in range(n)]


def _admit(adm, tmp_path, **volume_kw):
    inputs = adm.load_inputs(*_inputs(tmp_path))
    demand = adm.admission_plan(inputs)["demand_bytes_per_s"]
    volume = FakeVolume(demand, **volume_kw)
    record = adm.run_admission(str(tmp_path), inputs, clock=volume.clock, sleep=volume.sleep,
                               backend=volume.backend, chunks=_chunks())
    return record, volume


# --------------------------------------------------------------------------
# the schedule from the two input files
# --------------------------------------------------------------------------
def test_d_v_n_and_the_period_follow_the_formulas_from_the_two_jsons(adm, tmp_path):
    import hashlib

    plan_path, a_path = _inputs(tmp_path)
    inputs = adm.load_inputs(plan_path, a_path)
    out = adm.admission_plan(inputs)
    cycle = 16 * K * SLOT  # spill_layers (the stand's target) x k x slot_bytes
    demand = cycle / A_STEP_S
    volume = cycle * (7 * 2 * 12 + 3 * 4)
    assert out["spill_layers"] == 16 and out["cycle_bytes"] == cycle
    assert out["demand_bytes_per_s"] == pytest.approx(demand)
    assert out["period_s"] == pytest.approx(cycle / (2 * demand)) == pytest.approx(1.0)
    assert out["volume"]["bytes"] == volume
    assert out["n_cycles"] == max(20, math.ceil(volume / cycle)) == 180
    assert out["duration_s"] == pytest.approx(180 * out["period_s"])
    assert inputs["a"]["value_s"] == A_STEP_S
    assert inputs["a"]["sha256"] == hashlib.sha256(a_path.read_bytes()).hexdigest()
    assert inputs["plan"]["sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert (inputs["plan"]["n_layers"], inputs["plan"]["k"], inputs["plan"]["slot_bytes"]) == (
        N_LAYERS, K, SLOT)


@pytest.mark.parametrize("k, slot, a_value", [(4, 4 << 20, 3.1), (2, 8192, 0.5)])
def test_the_plan_sets_the_cycle_and_the_a_step_sets_d(adm, tmp_path, k, slot, a_value):
    plan_path, a_path = _inputs(tmp_path, plan={"k": k, "slot_bytes": slot},
                                a={"spill_plan": _plan(k=k, slot_bytes=slot),
                                   "value": a_value, "timed_step_s": [a_value] * 5})
    out = adm.admission_plan(adm.load_inputs(plan_path, a_path))
    assert out["cycle_bytes"] == 16 * k * slot
    assert out["demand_bytes_per_s"] == pytest.approx(16 * k * slot / a_value)
    assert out["period_s"] == pytest.approx(a_value / 2)


def test_v_counts_the_cost_b_arms_and_the_four_correctness_and_cleanup_spill_arms(adm):
    out = adm.session_volume(1000)
    assert out["cost_cycles"] == 7 * 2 * 12  # 2 warm-up + 5 timed steps, 2 B arms, 12 blocks
    assert out["other_cycles"] == 3 * 4  # l2l_spill, sigint, kill9, kill9_next at 3 steps
    assert out["bytes"] == 1000 * 180


@pytest.mark.parametrize("cycles, n", [(3, 20), (20, 20), (20.001, 21), (180, 180)])
def test_n_is_v_in_cycles_rounded_up_with_a_floor_of_20(adm, cycles, n):
    out = adm.schedule(cycle_bytes=1000, a_value_s=2.0, volume_bytes=cycles * 1000)
    assert out["n_cycles"] == n
    assert out["duration_s"] == pytest.approx(n * 1.0)


@pytest.mark.parametrize("plan, a, word", [
    ({}, {"device": "cpu"}, "not cuda"),
    ({}, {"exit": "error"}, "exited"),
    ({}, {"spill_plan": _plan(spill_layers=3)}, "spilled"),
    ({}, {"spill_plan": _plan(k=4)}, "k 4"),
    ({}, {"spill_plan": _plan(slot_bytes=8192)}, "slot_bytes 8192"),
    ({}, {"timed_step_s": [2.0] * 4}, "4 timed steps"),
    ({}, {"value": 0}, "value 0"),
    ({}, {"role": "B"}, "not a cost A arm"),
    ({"n_layers": 8}, {}, "fewer than the 16"),
    ({"slot_bytes": None}, {"spill_plan": _plan(slot_bytes=None)}, "slot_bytes"),
])
def test_inputs_that_cannot_set_the_schedule_are_refused(adm, tmp_path, plan, a, word):
    with pytest.raises(adm.InputError, match=word):
        adm.load_inputs(*_inputs(tmp_path, plan=plan, a=a))


def test_a_cpu_calibration_is_refused(adm, tmp_path):
    plan_path, a_path = _inputs(tmp_path)
    data = json.loads(plan_path.read_text())
    plan_path.write_text(json.dumps({**data, "device": "cpu"}))
    with pytest.raises(adm.InputError, match="--plan: device 'cpu'"):
        adm.load_inputs(plan_path, a_path)


def test_the_store_order_writes_forward_and_reads_backward(adm):
    write, read = adm.spill_order(3, 2, 10)
    assert write == [0, 10, 20, 30, 40, 50]
    assert read == [40, 50, 20, 30, 0, 10]


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def test_a_fast_steady_volume_is_admitted_with_exactly_n_cycles_on_schedule(adm, tmp_path):
    record, _volume = _admit(adm, tmp_path)
    assert record["admitted"], record["reason"]
    rows = record["cycle_rows"]
    period = record["schedule"]["period_s"]
    assert len(rows) == record["scheduled_cycles"] == record["schedule"]["n_cycles"] == 180
    for index, (planned, start, _end, _lateness, _bad) in enumerate(rows):
        assert planned == pytest.approx(index * period)
        assert start == pytest.approx(planned)
    assert record["mismatched_chunks"] == 0 and record["chunks_read"] == 180 * 16 * K


def test_the_admission_writes_no_more_than_v(adm, tmp_path):
    record, volume = _admit(adm, tmp_path)
    schedule = record["schedule"]
    cycle, n = schedule["cycle_bytes"], schedule["n_cycles"]
    assert schedule["volume"]["bytes"] >= 20 * cycle
    assert record["bytes"]["write"] == n * cycle == volume.writes * SLOT
    assert n * cycle <= schedule["volume"]["bytes"] + cycle
    assert record["bytes"]["read"] == n * cycle


def test_a_steady_volume_below_2d_is_refused_and_still_runs_every_cycle(adm, tmp_path):
    """Writes at 1.5 D: one cycle's writes alone take 2/1.5 = 1.33 periods (the schedule
    writes 2 D), so it falls 0.35 s behind on every 1 s cycle, about 63 s after 180."""
    record, _volume = _admit(adm, tmp_path, write_d=1.5, read_d=100.0)
    assert not record["admitted"]
    assert record["judgement"]["max_lateness_s"] > 50
    assert record["cycles"] == 180
    assert not record["description_only"]["stability"]["write"]["low"]


def test_a_volume_that_dips_but_still_covers_the_demand_is_admitted(adm, tmp_path):
    """10 D -> 4.5 D for a minute: a cycle then takes 0.89 of a period, so the schedule never
    falls behind; the description sees the dip and changes nothing."""
    record, _volume = _admit(adm, tmp_path, slow=(60.0, 120.0, 4.5))
    assert record["admitted"], record["reason"]
    assert record["judgement"]["max_lateness_s"] < 0
    assert record["description_only"]["stability"]["write"]["low"]


@pytest.mark.parametrize("lateness_s, admitted", [(4.9, True), (5.1, False)])
def test_the_lateness_limit_is_five_seconds(adm, tmp_path, lateness_s, admitted):
    # the period holds one cycle's writes and its reads: at 10 D each way a cycle takes
    # 2 x 2/10 = 0.4 of the 1 s period; a stall in cycle 30 ends it lateness_s late
    record, _volume = _admit(adm, tmp_path, stalls=[(30.05, lateness_s + 1.0 - 0.4)])
    assert record["judgement"]["max_lateness_s"] == pytest.approx(lateness_s)
    assert record["admitted"] is admitted


def test_after_a_stall_the_next_cycle_starts_at_once_not_on_its_slot(adm, tmp_path):
    record, _volume = _admit(adm, tmp_path, stalls=[(30.05, 3.0)])
    rows = record["cycle_rows"]
    assert rows[31][1] == pytest.approx(rows[30][2])  # cycle 31 starts where 30 ended
    assert rows[31][1] > rows[31][0]


def test_a_volume_that_returns_other_bytes_is_refused(adm, tmp_path):
    record, _volume = _admit(adm, tmp_path, corrupt=True)
    assert not record["admitted"]
    assert record["mismatched_chunks"] == record["chunks_read"] > 0


def test_a_volume_that_drops_writes_is_refused_by_the_stamp(adm, tmp_path):
    """After its first cycle the volume keeps the old bytes; only the cycle stamp tells."""
    record, _volume = _admit(adm, tmp_path, drop_writes_after=16 * K)
    assert not record["admitted"]
    assert record["mismatched_chunks"] == record["chunks_read"] - 16 * K


def test_a_test_file_left_behind_is_refused(adm, tmp_path):
    record, _volume = _admit(adm, tmp_path, leave_file=True)
    assert record["left_behind"] and not record["admitted"]
    assert "left" in record["reason"]


def test_fewer_cycles_than_scheduled_are_refused(adm):
    run = {"cycles": [(0.0, 0.0, 0.1, -0.9, 0)] * 19, "mismatched_chunks": 0,
           "left_behind": False, "path": "x"}
    judged = adm.judgement(run, n_cycles=20)
    assert not judged["admitted"] and "19 of 20" in judged["problems"][0]
    assert adm.judgement({**run, "cycles": run["cycles"] * 2}, n_cycles=38)["admitted"]


def test_a_sector_with_another_stamp_fails_even_with_the_right_bytes(adm):
    import torch

    source = torch.full((SLOT,), 7, dtype=torch.uint8)
    adm.stamp(source, 3, 8192)
    assert adm.matches(source.clone(), source, 3, 8192)
    assert not adm.matches(source.clone(), source, 2, 8192)
    assert not adm.matches(source.clone(), source, 3, 4096)
    other = source.clone()
    other[100] ^= 1
    assert not adm.matches(other, source, 3, 8192)


# --------------------------------------------------------------------------
# the volume and the command line
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fstype, model, scope", [
    ("tmpfs", None, "refused"), ("overlay", None, "refused"), ("nfs4", None, "network"),
    ("ext4", "QEMU HARDDISK", "virtual"), ("ext4", "Samsung SSD 990", "local"),
])
def test_the_filesystem_is_classified_as_a4_does(adm, fstype, model, scope):
    assert adm.classify_fs(fstype, model)[0] == scope


@pytest.mark.parametrize("scope, direct, ok", [
    ("network", True, False), ("refused", True, False), ("local", False, False),
    ("local", True, True), ("virtual", True, True),
])
def test_a_network_volume_or_no_direct_io_refuses_the_volume(adm, tmp_path, monkeypatch,
                                                             scope, direct, ok):
    calls = []
    monkeypatch.setattr(adm, "classify_fs", lambda *a, **kw: (scope, scope))
    monkeypatch.setattr(adm, "round_trip", lambda d: calls.append(d) or (direct, "probe"))
    out = adm.spill_volume(str(tmp_path / "spill"))
    assert out["ok"] is ok
    assert bool(calls) is (scope not in ("network", "refused"))
    if scope in ("network", "refused"):
        assert out["refusals"] == [scope]  # refused for what it is, not for its direct I/O


def _cli(adm, tmp_path, monkeypatch, *, volume_ok=True, **volume_kw):
    plan_path, a_path = _inputs(tmp_path)
    monkeypatch.setattr(adm, "spill_volume", lambda d: {"ok": volume_ok,
                                                        "refusals": [] if volume_ok else ["tmpfs"]})
    inputs = adm.load_inputs(plan_path, a_path)
    volume = FakeVolume(adm.admission_plan(inputs)["demand_bytes_per_s"], **volume_kw)
    monkeypatch.setattr(adm, "exercise", functools.partial(
        adm.exercise, clock=volume.clock, sleep=volume.sleep, backend=volume.backend,
        chunks=_chunks()))
    out = tmp_path / "admission.json"
    code = adm.main(["--spill-dir", str(tmp_path / "s"), "--plan", str(plan_path),
                     "--a", str(a_path), "--out", str(out), "--cycles",
                     str(tmp_path / "cycles.csv")])
    return code, out


def test_the_cli_admits_and_records_the_numbers_and_their_inputs(adm, tmp_path, monkeypatch):
    code, out = _cli(adm, tmp_path, monkeypatch)
    record = json.loads(out.read_text())
    assert code == 0 and record["admitted"]
    schedule = record["schedule"]
    for key in ("demand_bytes_per_s", "period_s", "n_cycles", "cycle_bytes", "duration_s"):
        assert schedule[key] is not None
    assert schedule["volume"]["bytes"] == schedule["cycle_bytes"] * 180
    assert record["inputs"]["a"]["value_s"] == A_STEP_S
    assert len(record["inputs"]["a"]["sha256"]) == len(record["inputs"]["plan"]["sha256"]) == 64
    lines = (tmp_path / "cycles.csv").read_text().splitlines()
    assert len(lines) == 1 + 180 and b"\r\n" not in out.read_bytes()


def test_the_cli_refuses_a_slow_volume_with_exit_4(adm, tmp_path, monkeypatch):
    code, out = _cli(adm, tmp_path, monkeypatch, write_d=1.5, read_d=100.0)
    assert code == 4 and not json.loads(out.read_text())["admitted"]


def test_the_cli_refuses_a_refused_volume_with_exit_3(adm, tmp_path, monkeypatch):
    code, out = _cli(adm, tmp_path, monkeypatch, volume_ok=False)
    record = json.loads(out.read_text())
    assert code == 3 and "volume refused" in record["reason"]
    assert record["schedule"]["n_cycles"] == 180  # the numbers are recorded all the same


def test_the_cli_refuses_inputs_and_existing_outputs_with_exit_2(adm, tmp_path, capsys):
    plan_path, a_path = _inputs(tmp_path, a={"device": "cpu"})
    args = ["--spill-dir", str(tmp_path / "s"), "--plan", str(plan_path), "--a", str(a_path),
            "--out", str(tmp_path / "o.json"), "--cycles", str(tmp_path / "c.csv")]
    assert adm.main(args) == 2 and "REFUSED INPUTS" in capsys.readouterr().out
    assert not (tmp_path / "o.json").exists()


@pytest.mark.parametrize("existing", ["out", "cycles"])
def test_the_cli_never_overwrites_a_result(adm, tmp_path, monkeypatch, capsys, existing):
    plan_path, a_path = _inputs(tmp_path)
    paths = {"out": tmp_path / "o.json", "cycles": tmp_path / "c.csv"}
    paths[existing].write_text("kept")
    monkeypatch.setattr(adm, "spill_volume", lambda d: pytest.fail("the volume was touched"))
    code = adm.main(["--spill-dir", str(tmp_path / "s"), "--plan", str(plan_path), "--a",
                     str(a_path), "--out", str(paths["out"]), "--cycles", str(paths["cycles"])])
    assert code == 2 and "refusing to overwrite" in capsys.readouterr().out
    assert paths[existing].read_text() == "kept"
