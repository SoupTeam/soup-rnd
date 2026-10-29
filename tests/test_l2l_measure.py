"""CPU tests for the A5 measurement record (l2l_measure.py).

One record per step and per run holds the useful speed (tokens without padding per
second), the GPU and RAM peaks and the I/O counters together. A source that is
missing gives an explicit null with its reason, never a zero.
"""

from __future__ import annotations

import json

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def measure():
    return _load("l2l_measure")


def _step(measure, **override):
    fields = dict(
        step_s=2.0,
        real_tokens=1000,
        layer_loads=64,
        nf4_restores=128,
        spill_written_bytes=0,
        spill_read_bytes=0,
        peak_vram_allocated_bytes=5_000_000_000,
        peak_vram_reserved_bytes=6_000_000_000,
        peak_rss_bytes=20_000_000_000,
        pinned_store_bytes=8_000_000_000,
        weight_read_bytes=14_000_000_000,
        os_read_bytes=14_000_000_000,
        os_write_bytes=0,
        device_read_bytes=14_500_000_000,
    )
    fields.update(override)
    return measure.step_record(**fields)


def test_real_tokens_leave_out_the_padding(measure):
    mask = [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
    assert measure.count_real_tokens(mask) == 5


def test_a_step_reports_tokens_without_padding_per_second(measure):
    record = _step(measure, step_s=2.0, real_tokens=1000)
    assert record["tokens_per_s"] == 500.0
    assert record["step_s"] == 2.0


def test_a_step_record_names_the_unit_of_every_byte_field(measure):
    record = _step(measure)
    assert record["peak_vram_allocated_bytes"] == 5_000_000_000
    assert record["peak_vram_reserved_bytes"] == 6_000_000_000
    assert record["peak_rss_bytes"] == 20_000_000_000
    assert record["pinned_store_bytes"] == 8_000_000_000
    assert record["weight_read_bytes"] == 14_000_000_000
    assert record["layer_loads"] == 64
    assert record["nf4_restores"] == 128


def test_a_missing_source_is_null_with_its_reason_not_zero(measure):
    record = _step(
        measure,
        peak_vram_allocated_bytes=None,
        peak_vram_reserved_bytes=None,
        unavailable={
            "peak_vram_allocated_bytes": "no CUDA device",
            "peak_vram_reserved_bytes": "no CUDA device",
        },
    )
    assert record["peak_vram_allocated_bytes"] is None
    assert record["unavailable"]["peak_vram_allocated_bytes"] == "no CUDA device"
    assert record["peak_rss_bytes"] == 20_000_000_000
    assert "peak_rss_bytes" not in record["unavailable"]


def test_a_null_without_a_reason_is_refused(measure):
    with pytest.raises(ValueError, match="peak_rss_bytes"):
        _step(measure, peak_rss_bytes=None)


def test_a_reason_for_a_field_that_has_a_value_is_refused(measure):
    with pytest.raises(ValueError, match="peak_rss_bytes"):
        _step(measure, unavailable={"peak_rss_bytes": "stale"})


@pytest.mark.parametrize("step_s,tokens", [(0.0, 10), (-1.0, 10), (1.0, 0), (1.0, -3)])
def test_a_step_with_no_time_or_no_tokens_is_refused(measure, step_s, tokens):
    with pytest.raises(ValueError):
        _step(measure, step_s=step_s, real_tokens=tokens)


def test_a_run_sums_tokens_and_time_and_takes_the_peaks(measure):
    steps = [
        _step(measure, step_s=2.0, real_tokens=1000, peak_rss_bytes=10, layer_loads=64,
              spill_written_bytes=100, spill_read_bytes=300, weight_read_bytes=7),
        _step(measure, step_s=4.0, real_tokens=1000, peak_rss_bytes=30, layer_loads=64,
              spill_written_bytes=100, spill_read_bytes=300, weight_read_bytes=5),
    ]
    run = measure.run_record(steps)
    assert run["steps"] == 2
    assert run["tokens_per_s"] == pytest.approx(2000 / 6.0)
    assert run["step_s_mean"] == 3.0
    assert run["peak_rss_bytes"] == 30
    assert run["layer_loads"] == 128
    assert run["spill_written_bytes"] == 200
    assert run["spill_read_bytes"] == 600
    assert run["weight_read_bytes"] == 12


def test_a_run_is_null_when_any_step_lacks_the_source(measure):
    steps = [
        _step(measure),
        _step(measure, peak_rss_bytes=None, unavailable={"peak_rss_bytes": "psutil missing"}),
    ]
    run = measure.run_record(steps)
    assert run["peak_rss_bytes"] is None
    assert "psutil missing" in run["unavailable"]["peak_rss_bytes"]


def test_a_run_of_no_steps_is_refused(measure):
    with pytest.raises(ValueError):
        measure.run_record([])


def test_the_records_are_plain_json_with_no_nan(measure):
    run = measure.run_record([_step(measure)])
    json.dumps(run, allow_nan=False)


def test_a_tiny_cpu_step_gives_every_field_a_value_or_a_reason(measure):
    import torch

    model = torch.nn.Linear(8, 8)
    ids = torch.ones(2, 8)
    mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3])

    def body():
        model(ids).sum().backward()

    record = measure.measure_step(
        body,
        attention_mask=mask,
        device="cpu",
        counters=lambda: {"layer_loads": 0, "nf4_restores": 0, "spill_written_bytes": 0,
                          "spill_read_bytes": 0, "pinned_store_bytes": 0},
    )
    assert record["real_tokens"] == 13
    assert record["tokens_per_s"] == pytest.approx(13 / record["step_s"])
    assert record["peak_vram_allocated_bytes"] is None
    assert record["unavailable"]["peak_vram_allocated_bytes"] == "no CUDA device"
    assert record["weight_read_bytes"] is None
    assert record["unavailable"]["weight_read_bytes"]
    for name in measure.MEASURED_FIELDS:
        assert (record[name] is None) == (name in record["unavailable"])
    json.dumps(record, allow_nan=False)


def test_counters_count_the_step_only(measure):
    state = {"loads": 10}

    def counters():
        return {"layer_loads": state["loads"], "nf4_restores": 0, "spill_written_bytes": 0,
                "spill_read_bytes": 0, "pinned_store_bytes": 7}

    def body():
        state["loads"] += 64

    record = measure.measure_step(
        body, attention_mask=[[1, 1]], device="cpu", counters=counters,
        read_bytes=lambda: 0,
    )
    assert record["layer_loads"] == 64
    assert record["pinned_store_bytes"] == 7
    assert record["weight_read_bytes"] == 0
    assert "weight_read_bytes" not in record["unavailable"]


def _counters():
    return {"layer_loads": 0, "nf4_restores": 0, "spill_written_bytes": 0,
            "spill_read_bytes": 0, "pinned_store_bytes": 0}


def test_the_rss_peak_is_the_steps_own_when_the_peak_can_be_reset(measure, monkeypatch):
    import host_box

    calls = []
    monkeypatch.setattr(host_box, "reset_peak_rss", lambda: calls.append("reset") or True)
    monkeypatch.setattr(host_box, "process_memory", lambda: {"peak_rss": 123})
    record = measure.measure_step(lambda: calls.append("body"), attention_mask=[[1]],
                                  device="cpu", counters=_counters)
    assert calls == ["reset", "body"]
    assert record["peak_rss_bytes"] == 123


def test_the_rss_peak_is_null_when_it_cannot_be_reset(measure, monkeypatch):
    import host_box

    monkeypatch.setattr(host_box, "reset_peak_rss", lambda: False)
    monkeypatch.setattr(host_box, "process_memory", lambda: {"peak_rss": 123})
    record = measure.measure_step(lambda: None, attention_mask=[[1]], device="cpu",
                                  counters=_counters)
    assert record["peak_rss_bytes"] is None
    assert "process-lifetime" in record["unavailable"]["peak_rss_bytes"]


def test_a_counter_the_trainer_does_not_have_is_null_with_a_reason(measure):
    def counters():
        found = _counters()
        del found["nf4_restores"]
        return found

    record = measure.measure_step(lambda: None, attention_mask=[[1]], device="cpu",
                                  counters=counters)
    assert record["nf4_restores"] is None
    assert "nf4_restores" in record["unavailable"]["nf4_restores"]


def test_a_read_counter_that_goes_backwards_is_null_with_a_reason(measure):
    readings = iter([100, 40])
    record = measure.measure_step(lambda: None, attention_mask=[[1]], device="cpu",
                                  counters=_counters, read_bytes=lambda: next(readings))
    assert record["weight_read_bytes"] is None
    assert "backwards" in record["unavailable"]["weight_read_bytes"]


# --------------------------------------------------------------------------
# seam 2: counters read from outside the training code
# --------------------------------------------------------------------------
def _proc_tree(tmp_path, text):
    (tmp_path / "self").mkdir()
    (tmp_path / "self" / "io").write_text(text, encoding="ascii")
    return str(tmp_path)


IO_TEXT = (
    "rchar: 999\nwchar: 888\nsyscr: 1\nsyscw: 1\n"
    "read_bytes: 4096\nwrite_bytes: 8192\ncancelled_write_bytes: 0\n"
)


def test_process_io_reads_storage_bytes_from_proc_not_syscall_chars(measure, tmp_path):
    io = measure.process_io(proc_root=_proc_tree(tmp_path, IO_TEXT), platform="linux")
    assert io == {"read_bytes": 4096, "write_bytes": 8192, "reason": None}


def test_process_io_is_null_with_a_reason_when_proc_is_missing(measure, tmp_path):
    io = measure.process_io(proc_root=str(tmp_path / "none"), platform="linux")
    assert io["read_bytes"] is None and io["write_bytes"] is None
    assert io["reason"]


def test_process_io_uses_psutil_counters_off_linux(measure):
    class Counters:
        read_bytes = 11
        write_bytes = 22

    io = measure.process_io(platform="win32", psutil_io=lambda: Counters())
    assert io == {"read_bytes": 11, "write_bytes": 22, "reason": None}


def test_process_io_is_null_when_psutil_has_no_counters(measure):
    def missing():
        raise AttributeError("io_counters")

    io = measure.process_io(platform="darwin", psutil_io=missing)
    assert io["read_bytes"] is None
    assert "io_counters" in io["reason"]


def test_a_step_takes_the_os_deltas_and_leaves_spill_reads_out_of_the_weights(measure):
    reads = iter([1_000, 11_000])
    writes = iter([0, 700])
    devices = iter([5_000, 25_000])
    step = {"loads": 0, "spill_read": 0}

    def counters():
        found = _counters()
        found["spill_read_bytes"] = step["spill_read"]
        return found

    def body():
        step["spill_read"] += 4_000

    record = measure.measure_step(
        body, attention_mask=[[1]], device="cpu", counters=counters,
        read_bytes=lambda: next(reads), write_bytes=lambda: next(writes),
        device_read_bytes=lambda: next(devices),
    )
    assert record["os_read_bytes"] == 10_000
    assert record["os_write_bytes"] == 700
    assert record["device_read_bytes"] == 20_000
    assert record["weight_read_bytes"] == 6_000


def test_weights_are_null_when_the_spill_read_more_than_the_os_counted(measure):
    reads = iter([0, 100])
    step = {"spill_read": 0}

    def counters():
        found = _counters()
        found["spill_read_bytes"] = step["spill_read"]
        return found

    def body():
        step["spill_read"] += 500

    record = measure.measure_step(
        body, attention_mask=[[1]], device="cpu", counters=counters,
        read_bytes=lambda: next(reads),
    )
    assert record["weight_read_bytes"] is None
    assert "spill" in record["unavailable"]["weight_read_bytes"]


def test_missing_os_sources_are_null_with_a_reason_for_every_io_field(measure):
    record = measure.measure_step(lambda: None, attention_mask=[[1]], device="cpu",
                                  counters=_counters)
    for name in ("os_read_bytes", "os_write_bytes", "device_read_bytes", "weight_read_bytes"):
        assert record[name] is None
        assert record["unavailable"][name]


def test_disk_read_counter_names_the_device_of_the_path(measure):
    seen = []
    counter = measure.device_read_counter(
        "/models", disk_device=lambda path: seen.append(path) or "nvme0n1",
        disk_reads=lambda device: {"nvme0n1": 77}[device],
    )
    assert counter() == 77
    assert seen == ["/models"]


def test_disk_read_counter_is_null_when_the_path_has_no_block_device(measure):
    counter = measure.device_read_counter(
        "/dev/shm", disk_device=lambda path: None, disk_reads=lambda device: 1)
    assert counter() is None


# --------------------------------------------------------------------------
# seam 3: validity gate and the summary
# --------------------------------------------------------------------------
def _arm_record(outcome, **checks):
    return {"outcome": outcome, "checks": checks}


def _entry(measure, arm, tokens_per_s, outcome="ok", reason="", round=1, run=1):
    run_rec = measure.run_record([_step(measure, step_s=1.0, real_tokens=int(tokens_per_s))])
    checks = {"foreign_reads": {"outcome": outcome, "reason": reason}} if outcome != "ok" else {}
    return measure.run_entry(run_rec, _arm_record(outcome, **checks), arm=arm, round=round,
                             run=run)


def test_a_void_run_is_marked_spoiled_with_the_failed_check_and_its_reason(measure):
    entry = _entry(measure, "A", 100, outcome="void",
                   reason="python (pid 9) read 2000000000 bytes")
    assert entry["spoiled"] is True
    assert entry["void_reasons"] == {"foreign_reads": "python (pid 9) read 2000000000 bytes"}
    assert entry["measurement"]["tokens_per_s"] == 100


def test_an_ok_run_is_not_spoiled_and_an_unknown_run_is_flagged_unverified(measure):
    ok = _entry(measure, "A", 100)
    unknown = _entry(measure, "A", 100, outcome="unknown", reason="no counter")
    assert ok["spoiled"] is False and ok["unverified"] is False
    assert unknown["spoiled"] is False and unknown["unverified"] is True


def _block(measure, a1, b1, b2, a2, **bad):
    values = [("A", a1), ("B", b1), ("B", b2), ("A", a2)]
    return [
        _entry(measure, arm, value, outcome=bad.get(f"p{index}", "ok"), reason="x",
               round=1 + index // 2, run=index + 1)
        for index, (arm, value) in enumerate(values)
    ]


def test_the_summary_gives_an_interval_for_the_block_differences(measure):
    blocks = [
        _block(measure, 110, 100, 100, 110),
        _block(measure, 108, 100, 100, 108),
        _block(measure, 112, 100, 100, 112),
    ]
    summary = measure.summarize(blocks)
    assert summary["valid_blocks"] == 3
    assert summary["mean_a_tokens_per_s"] == pytest.approx(110.0)
    assert summary["mean_b_tokens_per_s"] == pytest.approx(100.0)
    assert summary["difference_a_minus_b"]["mean"] == pytest.approx(10.0)
    low, high = summary["difference_a_minus_b"]["interval_95"]
    assert low < 10.0 < high
    # t(0.975, df=2) = 4.303, standard error = 2 / sqrt(3)
    assert high - 10.0 == pytest.approx(4.303 * 2 / 3 ** 0.5, rel=1e-3)


def test_a_block_with_a_spoiled_run_stays_out_of_the_summary_and_is_listed(measure):
    blocks = [
        _block(measure, 110, 100, 100, 110),
        _block(measure, 1, 1, 1, 1, p2="void"),
        _block(measure, 108, 100, 100, 108),
    ]
    summary = measure.summarize(blocks)
    assert summary["valid_blocks"] == 2
    assert summary["spoiled_blocks"] == [2]
    assert summary["mean_a_tokens_per_s"] == pytest.approx(109.0)


def test_the_summary_reports_the_f1_repeats_rule(measure):
    blocks = [_block(measure, 110, 100, 100, 110), _block(measure, 108, 100, 100, 108)]
    summary = measure.summarize(blocks)
    assert summary["blocks_to_add"] == 3
    assert summary["blocks_needed"] >= 2
    assert summary["sigma_tokens_per_s"] >= 0


def test_with_fewer_than_two_valid_blocks_the_interval_is_null_with_a_reason(measure):
    summary = measure.summarize([_block(measure, 110, 100, 100, 110)])
    assert summary["difference_a_minus_b"]["interval_95"] is None
    assert "at least 2" in summary["unavailable"]["difference_a_minus_b"]
    assert summary["blocks_to_add"] == 4


def test_unverified_runs_are_counted_in_the_summary(measure):
    blocks = [
        _block(measure, 110, 100, 100, 110, p1="unknown"),
        _block(measure, 108, 100, 100, 108),
    ]
    assert measure.summarize(blocks)["unverified_runs"] == 1


def test_a_block_that_is_not_a_b_b_a_is_refused(measure):
    with pytest.raises(ValueError):
        measure.summarize([[_entry(measure, "A", 1)] * 3])
    swapped = _block(measure, 1, 1, 1, 1)
    swapped[0]["arm"] = "B"
    with pytest.raises(ValueError, match="A, B, B, A"):
        measure.summarize([swapped])


# --------------------------------------------------------------------------
# seam 4: the launch passport
# --------------------------------------------------------------------------
def _passport(measure, **override):
    fields = dict(
        argv=["python", "bench.py", "--model", "my model", "--k", "16"],
        packages=lambda names: {name: "1.0" for name in names},
        gpu=lambda: {"name": "RTX 3090", "memory_total_mib": 24576, "driver": "580.1"},
        container_ram=lambda: {"limit_bytes": 64_000_000_000, "source": "memory.max"},
        commit=lambda: {"hash": "abc123", "dirty": False},
    )
    fields.update(override)
    return measure.launch_passport(**fields)


def test_the_passport_holds_the_whole_command_quoted_as_run(measure):
    passport = _passport(measure)
    assert passport["command"] == "python bench.py --model 'my model' --k 16"
    assert passport["argv"][3] == "my model"


def test_the_passport_holds_versions_gpu_ram_and_the_commit(measure):
    passport = _passport(measure)
    assert passport["packages"]["torch"] == "1.0"
    assert passport["gpu"]["name"] == "RTX 3090"
    assert passport["container_ram"]["limit_bytes"] == 64_000_000_000
    assert passport["commit"] == {"hash": "abc123", "dirty": False}
    assert passport["unavailable"] == {}


def test_a_source_that_fails_is_null_with_its_reason(measure):
    def no_gpu():
        raise RuntimeError("nvidia-smi not found")

    passport = _passport(measure, gpu=no_gpu, commit=lambda: None)
    assert passport["gpu"] is None
    assert "nvidia-smi not found" in passport["unavailable"]["gpu"]
    assert passport["commit"] is None
    assert passport["unavailable"]["commit"]


def test_an_uninstalled_package_is_named_with_null_not_a_version(measure):
    passport = _passport(measure, packages=lambda names: {n: None for n in names})
    assert passport["packages"]["torch"] is None
    assert "torch" in passport["unavailable"]["packages"]


def test_an_unlimited_container_ram_is_null_with_a_reason(measure):
    passport = _passport(measure, container_ram=lambda: {"limit_bytes": None, "source": None})
    assert passport["container_ram"]["limit_bytes"] is None
    assert "container_ram" in passport["unavailable"]


def test_the_default_commit_reader_finds_this_repository(measure):
    found = measure.git_commit()
    assert found is not None
    assert len(found["hash"]) == 40
    assert isinstance(found["dirty"], bool)


# --------------------------------------------------------------------------
# the record file and the summary beside it
# --------------------------------------------------------------------------
def test_the_record_and_its_summary_are_written_side_by_side(measure, tmp_path):
    blocks = [_block(measure, 110, 100, 100, 110), _block(measure, 108, 100, 100, 108)]
    paths = measure.write_report(tmp_path, passport=_passport(measure), blocks=blocks)
    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert paths["record"].parent == paths["summary"].parent == tmp_path
    assert record["passport"]["command"].startswith("python bench.py")
    assert len(record["blocks"]) == 2
    first = record["blocks"][0][0]["measurement"]
    for name in ("tokens_per_s", "peak_vram_allocated_bytes", "peak_rss_bytes",
                 "weight_read_bytes", "os_write_bytes", "layer_loads", "nf4_restores"):
        assert name in first
    assert summary["valid_blocks"] == 2


def test_with_one_valid_block_the_difference_is_null_not_a_lone_number(measure):
    summary = measure.summarize([_block(measure, 110, 100, 100, 110)])
    assert summary["difference_a_minus_b"] == {"mean": None, "interval_95": None}


def test_two_identical_valid_blocks_are_both_kept(measure):
    blocks = [_block(measure, 110, 100, 100, 110), _block(measure, 110, 100, 100, 110)]
    summary = measure.summarize(blocks)
    assert summary["valid_blocks"] == 2 and summary["spoiled_blocks"] == []


def test_a_report_never_leaves_one_file_when_the_other_exists(measure, tmp_path):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    blocks = [_block(measure, 110, 100, 100, 110), _block(measure, 108, 100, 100, 108)]
    with pytest.raises(FileExistsError):
        measure.write_report(tmp_path, passport=_passport(measure), blocks=blocks)
    assert not (tmp_path / "record.json").exists()
