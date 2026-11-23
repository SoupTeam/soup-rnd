"""CPU-only tests for the Moonlight memory-premium toolkit."""

import importlib
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks/mbr-moonlight-emem-20261123"
MIB = 2**20


@pytest.fixture
def analysis(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(BENCHMARK))
    return importlib.import_module("emem_analysis")


def test_window_label(analysis: ModuleType) -> None:
    assert analysis.window_label(8, 512) == "k=8,S=512"


def test_canonical_row_protects_measurement_identity(analysis: ModuleType) -> None:
    identity = {
        "model": "Moonlight-16B-A3B", "model_revision": "rev", "card": "3050",
        "window": "k=8,S=512", "arm": "E", "mode": "normal", "torch_version": "2.14.1",
    }
    assert analysis.canonical_row(**identity, extra={"status": "ok"}) == {
        **identity, "status": "ok",
    }
    for key in identity:
        with pytest.raises(ValueError, match=key):
            analysis.canonical_row(**identity, extra={key: "replacement"})


def _row(arm: str, k: int, seq: int, peak: int, **overrides: str) -> dict:
    return {
        "arm": arm, "k": k, "seq": seq, "status": "ok", "mode": "normal",
        "peak_allocated": peak, "peak_reserved": 2 * peak, "peak_nvml": 3 * peak,
        **overrides,
    }


def test_mem_premium_pairs_normal_rows_by_arm_and_window(analysis: ModuleType) -> None:
    rows = [
        _row("E", 8, 512, 15), _row("C", 64, 64, 20), _row("C", 8, 64, 6),
        _row("C", 8, 512, 10), _row("E", 64, 64, 18), _row("E", 8, 64, 8),
        _row("E", 8, 512, 500, mode="deterministic"),
    ]
    summary = analysis.mem_premium(rows)
    assert [(row["k"], row["seq"]) for row in summary] == [(8, 64), (8, 512), (64, 64)]
    assert [row["delta_alloc"] for row in summary] == [2, 5, -2]
    assert summary[1] == {
        "k": 8, "seq": 512, "window": "k=8,S=512", "status": "ok",
        "C": {"alloc_bytes": 10, "reserved_bytes": 20, "nvml_bytes": 30},
        "E": {"alloc_bytes": 15, "reserved_bytes": 30, "nvml_bytes": 45},
        "delta_alloc": 5, "delta_reserved": 10, "delta_nvml": 15,
    }


def test_mem_premium_names_the_gap_when_an_arm_is_missing(analysis: ModuleType) -> None:
    row, empty, _ = analysis.mem_premium([_row("C", 8, 64, 6)])
    assert row["C"]["alloc_bytes"] == 6
    assert row["E"] == {"alloc_bytes": None, "reserved_bytes": None, "nvml_bytes": None}
    assert (row["delta_alloc"], row["delta_reserved"], row["delta_nvml"]) == (None, None, None)
    assert row["status"] == "missing E"
    assert empty["status"] == "missing C and E"


def test_mem_premium_ignores_oom_peaks_and_marks_oom_only_gap(analysis: ModuleType) -> None:
    rows = [
        _row("C", 8, 64, 6), _row("E", 8, 64, 8), _row("E", 8, 64, 99, status="OOM"),
        _row("C", 8, 512, 10), _row("E", 8, 512, 99, status="OOM"),
    ]
    paired, unpaired, _ = analysis.mem_premium(rows)
    assert paired["delta_alloc"] == 2
    assert unpaired["E"]["alloc_bytes"] is None
    assert unpaired["delta_alloc"] is None
    assert unpaired["status"] == "missing E (OOM)"


@pytest.mark.parametrize(
    ("stack", "expected"),
    [
        ([], "other"),
        ([("/src/model.py", "matmul")], "other"),
        ([("/src/a0_exact_lt_kernels.cu", "dispatch")], "fused_scratch"),
        ([("/src/a0_exact_lt_ext.cpp", "dispatch"),
          (r"C:\src\a0_exact_lt_kernels.cu", "scratch")], "staging"),
        ([("/src/a0_exact_lt_ext.cpp", "merged"),
          ("/src/a0_exact_lt.py", "lookup")], "algo_discovery"),
        ([("/src/a0_exact_lt.py", "_sample"),
          (r"C:\src\a0_exact_lt_ext.cpp", "workspace_ptr")], "cublaslt_workspace"),
    ],
)
def test_classify_frames_uses_class_priority_across_the_stack(
    analysis: ModuleType, stack: list[tuple[str, str]], expected: str,
) -> None:
    frames = [{"filename": filename, "name": name} for filename, name in stack]
    assert analysis.classify_frames(frames) == expected


def test_holdings_flattens_segments_and_only_counts_allocated_blocks_as_live(
    analysis: ModuleType,
) -> None:
    frames = [{"filename": "/src/a0_exact_lt.py", "name": "ensure", "line": 32}]
    snapshot = {"segments": [
        {"blocks": [{"size": 512, "state": "active_allocated", "frames": frames}]},
        {"blocks": [
            {"size": 1024, "state": "active_awaiting_free", "frames": []},
            {"size": 2048, "state": "inactive"},
        ]},
    ]}
    assert analysis.holdings(snapshot) == [
        {"size": 512, "live": True, "klass": "algo_discovery", "frames": frames},
        {"size": 1024, "live": False, "klass": "other", "frames": []},
        {"size": 2048, "live": False, "klass": "other", "frames": []},
    ]


def test_holdings_diff_empty_snapshots_have_no_main_source(analysis: ModuleType) -> None:
    diff = analysis.holdings_diff({"segments": []}, {"segments": []})
    zero = {"live_bytes": 0, "reserved_bytes": 0}
    assert diff["totals"] == {"C": zero, "E": zero, "delta": zero}
    assert tuple(diff["by_class"]) == (
        "cublaslt_workspace", "algo_discovery", "staging", "fused_scratch", "other",
    )
    assert all(counts == diff["totals"] for counts in diff["by_class"].values())
    assert diff["extra"] == []
    assert diff["main_source"] is None


def _block(size: int, *, live: bool = True, native: str | None = None) -> dict:
    frames = [] if native is None else [{"filename": "a0_exact_lt_ext.cpp", "name": native}]
    return {"size": size, "state": "active_allocated" if live else "inactive", "frames": frames}


def _segment(*blocks: dict) -> dict:
    return {
        "total_size": sum(block["size"] for block in blocks),
        "active_size": sum(block["size"] for block in blocks if block["state"] != "inactive"),
        "blocks": list(blocks),
    }


def _snapshot(*blocks: dict) -> dict:
    return {"segments": [_segment(*blocks)]}


def test_segment_traps_empty_snapshot(analysis: ModuleType) -> None:
    assert analysis.segment_traps({"segments": []}) == {
        "releasable_bytes": 0, "trapped_bytes": 0, "pinned": [],
    }


def test_segment_traps_fully_cached_segment_is_releasable(analysis: ModuleType) -> None:
    segment = {"total_size": 2048, "active_size": 0, "blocks": [_block(2048, live=False)]}
    assert analysis.segment_traps({"segments": [segment]}) == {
        "releasable_bytes": 2048, "trapped_bytes": 0, "pinned": [],
    }


def test_segment_traps_attributes_free_space_to_live_holders_by_address(
    analysis: ModuleType,
) -> None:
    frames = _block(25 * MIB, native="dispatch")["frames"]
    snapshot = {
        "segments": [{
            "total_size": 125 * MIB, "active_size": 25 * MIB,
            "blocks": [
                {**_block(25 * MIB), "address": 4096},
                _block(100 * MIB, live=False, native="scratch"),
            ],
        }],
        "device_traces": [[
            {"action": "alloc", "addr": 4096, "frames": frames},
        ]],
    }
    assert analysis.segment_traps(snapshot) == {
        "releasable_bytes": 0, "trapped_bytes": 100 * MIB,
        "pinned": [{
            "total_size": 125 * MIB, "active_bytes": 25 * MIB, "trapped_bytes": 100 * MIB,
            "holders": ["fused_scratch"],
        }],
    }


def test_segment_traps_lists_pinned_segments_largest_first(analysis: ModuleType) -> None:
    awaiting = {**_block(8, native="matmul"), "state": "active_awaiting_free"}
    snapshot = {"segments": [
        _segment(_block(32), _block(16, live=False)),
        _segment(_block(2, native="scratch"), _block(3, native="dispatch"),
                 _block(3, native="dispatch"), awaiting, _block(64, live=False)),
        _segment(_block(128)),
    ]}
    assert analysis.segment_traps(snapshot) == {
        "releasable_bytes": 0, "trapped_bytes": 80,
        "pinned": [
            {"total_size": 80, "active_bytes": 16, "trapped_bytes": 64,
             "holders": ["fused_scratch", "staging"]},
            {"total_size": 48, "active_bytes": 32, "trapped_bytes": 16, "holders": ["other"]},
        ],
    }


def test_holdings_attributes_cached_blocks_from_alloc_events(analysis: ModuleType) -> None:
    frames = [{"filename": "a0_exact_lt_ext.cpp", "name": "scratch"}]
    snapshot = _snapshot({**_block(1024, live=False), "address": 4096})
    snapshot["device_traces"] = [[], [
        {"action": "alloc", "addr": 4096, "frames": frames},
        {"action": "free_completed", "addr": 4096, "frames": []},
    ]]
    assert analysis.holdings(snapshot) == [
        {"size": 1024, "live": False, "klass": "staging", "frames": frames},
    ]


def test_holdings_leaves_unmatched_cached_blocks_as_other(analysis: ModuleType) -> None:
    snapshot = _snapshot({**_block(1024, live=False), "address": 4096})
    snapshot["device_traces"] = [[
        {"action": "alloc", "addr": 8192, "frames": _block(1024, native="scratch")["frames"]},
    ]]
    assert analysis.holdings(snapshot) == [
        {"size": 1024, "live": False, "klass": "other", "frames": []},
    ]


def test_holdings_prefers_live_block_frames_over_alloc_events(analysis: ModuleType) -> None:
    block = {**_block(1024, native="matmul"), "address": 4096}
    snapshot = _snapshot(block)
    snapshot["device_traces"] = [[
        {"action": "alloc", "addr": 4096, "frames": _block(1024, native="scratch")["frames"]},
    ]]
    assert analysis.holdings(snapshot) == [
        {"size": 1024, "live": True, "klass": "cublaslt_workspace", "frames": block["frames"]},
    ]


def test_holdings_diff_counts_live_and_cached_bytes_and_excludes_matches(
    analysis: ModuleType,
) -> None:
    matched = _block(512)
    scratch = _block(1024, live=False, native="scratch")
    diff = analysis.holdings_diff(_snapshot(matched), _snapshot(matched, scratch))
    assert diff["totals"] == {
        "C": {"live_bytes": 512, "reserved_bytes": 512},
        "E": {"live_bytes": 512, "reserved_bytes": 1536},
        "delta": {"live_bytes": 0, "reserved_bytes": 1024},
    }
    assert diff["by_class"]["staging"] == {
        "C": {"live_bytes": 0, "reserved_bytes": 0},
        "E": {"live_bytes": 0, "reserved_bytes": 1024},
        "delta": {"live_bytes": 0, "reserved_bytes": 1024},
    }
    assert diff["extra"] == [
        {"size": 1024, "live": False, "klass": "staging", "count": 1, "frames": scratch["frames"]},
    ]
    assert diff["main_source"] == "staging"


def test_holdings_diff_lists_only_surplus_copies_largest_first(analysis: ModuleType) -> None:
    matched = _block(1024)
    cached = _block(1024, live=False)
    larger = _block(2048)
    diff = analysis.holdings_diff(
        _snapshot(matched), _snapshot(matched, cached, matched, larger, matched),
    )
    assert [(row["size"], row["live"], row["count"]) for row in diff["extra"]] == [
        (2048, True, 1), (1024, False, 1), (1024, True, 2),
    ]


def test_holdings_diff_breaks_main_source_ties_by_class_name(analysis: ModuleType) -> None:
    discovery = _block(512)
    discovery["frames"] = [{"filename": "a0_exact_lt.py", "name": "native_matmul"}]
    snapshot = _snapshot(_block(512, native="matmul"), discovery)
    assert analysis.holdings_diff({}, snapshot)["main_source"] == "algo_discovery"


def test_holdings_diff_reports_segment_and_holder_deltas(analysis: ModuleType) -> None:
    fused = _block(8, native="dispatch")
    staging = _block(4, native="scratch")
    c_snapshot = {"segments": [
        _segment(fused, staging, _block(24, live=False)),
        _segment(_block(2, native="matmul"), _block(10, live=False)),
        _segment(_block(16, live=False)),
    ]}
    e_snapshot = {"segments": [
        _segment(fused, fused, staging, _block(64, live=False)),
        _segment(fused, _block(8, live=False)),
        _segment(_block(4, live=False)),
    ]}
    diff = analysis.holdings_diff(c_snapshot, e_snapshot)
    assert diff["traps"] == {
        "C": analysis.segment_traps(c_snapshot),
        "E": analysis.segment_traps(e_snapshot),
        "delta": {"releasable_bytes": -12, "trapped_bytes": 38},
    }
    assert diff["trapped_by_holder"] == {
        "cublaslt_workspace": -10, "algo_discovery": 0, "staging": 40,
        "fused_scratch": 48, "other": 0,
    }


def test_describe_diff_reports_main_source_and_mebibytes(analysis: ModuleType) -> None:
    scratch = _block(3 * 2**19, native="scratch")
    cached = _block(2**20, live=False)
    summary = analysis.describe_diff(analysis.holdings_diff({}, _snapshot(scratch, cached)))
    assert "Main source: staging" in summary
    assert "total: live +1.5 MiB, reserved +2.5 MiB" in summary
    assert "staging: live +1.5 MiB, reserved +1.5 MiB" in summary
    assert "1 x 1.5 MiB staging" in summary


def test_describe_diff_names_the_holder_with_the_largest_trapped_delta(
    analysis: ModuleType,
) -> None:
    staging = _block(MIB, native="scratch")
    c_snapshot = _snapshot(staging)
    e_snapshot = {"segments": [
        _segment(_block(200 * MIB, native="dispatch"), _block(840 * MIB, live=False)),
        _segment(_block(65 * MIB, native="dispatch"), staging, _block(47 * MIB, live=False)),
    ]}
    diff = analysis.holdings_diff(c_snapshot, e_snapshot)
    assert diff["main_source"] == "other"
    assert analysis.describe_diff(diff).splitlines()[0] == (
        "Main source: other = 887.0 MiB of free space trapped in segments pinned by fused_scratch"
    )


def test_describe_diff_names_all_holders_tied_for_largest_trapped_delta(
    analysis: ModuleType,
) -> None:
    snapshot = _snapshot(
        _block(MIB, native="scratch"), _block(MIB, native="dispatch"), _block(5 * MIB, live=False),
    )
    assert analysis.describe_diff(analysis.holdings_diff({}, snapshot)).splitlines()[0] == (
        "Main source: other = 5.0 MiB of free space trapped in segments pinned by "
        "fused_scratch, staging"
    )


def test_describe_diff_keeps_other_when_free_space_is_releasable(analysis: ModuleType) -> None:
    diff = analysis.holdings_diff({}, _snapshot(_block(5 * MIB, live=False)))
    assert analysis.describe_diff(diff).splitlines()[0] == "Main source: other"


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARK.parents[1]))
    monkeypatch.syspath_prepend(str(BENCHMARK))


def _measure_args(path: Path) -> list[str]:
    return [
        "--source", str(path), "--cache", str(path / "cache"), "--out", str(path / "out"),
        "--arm", "E", "--k", "8", "--seq", "512", "--device", "cpu",
    ]


def test_run_mem_appends_normal_measurements(
    benchmark: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    from benchmarks.harness import mbr_measure

    run_mem = importlib.import_module("run_mem")
    (tmp_path / "config.json").write_text('{"_commit_hash": "rev"}', encoding="utf-8")
    raw = _row("E", 8, 512, 15)

    def run_one(args: SimpleNamespace) -> dict:
        assert (args.kind, args.deterministic, args.steps, args.lora_r) == ("speed", False, 3, 16)
        assert args.lora_targets == ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
        assert (args.arm, args.k, args.seq, args.device) == ("E", 8, 512, "cpu")
        return raw.copy()

    monkeypatch.setattr(mbr_measure, "run_one", run_one)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fake"))
    for _ in range(2):
        assert run_mem.main(_measure_args(tmp_path)) == 0
    root = tmp_path / "out"
    canonical = {
        **raw, "model": "Moonlight-16B-A3B", "model_revision": "rev", "card": "CPU",
        "window": "k=8,S=512", "torch_version": "fake",
    }
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 2
    assert lines[0] == lines[1]
    assert {key: lines[0][key] for key in canonical} == canonical
    assert lines[0]["power_ac_online"] in (0, 1, None)
    assert (root / "rows_canonical.jsonl").read_text(encoding="utf-8").splitlines() == [
        json.dumps(row) for row in lines
    ]
    raw_lines = (root / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in raw_lines] == [raw, raw]


@pytest.mark.parametrize(("flag", "value"), [
    ("--steps", "1"), ("--k", "0"), ("--seq", "-1"), ("--lora-r", "0"),
])
def test_run_mem_rejects_invalid_dimensions_before_the_harness(
    benchmark: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, value: str,
) -> None:
    from benchmarks.harness import mbr_measure

    run_mem = importlib.import_module("run_mem")

    def run_one(args: SimpleNamespace) -> dict:
        pytest.fail("an invalid run must not reach the harness")

    monkeypatch.setattr(mbr_measure, "run_one", run_one)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fake"))
    with pytest.raises(SystemExit) as error:
        run_mem.main([*_measure_args(tmp_path), flag, value])
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_run_mem_writes_a_canonical_crash_before_reraising(
    benchmark: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    from benchmarks.harness import mbr_measure

    run_mem = importlib.import_module("run_mem")

    def run_one(args: SimpleNamespace) -> dict:
        raise RuntimeError("cannot build the model")

    monkeypatch.setattr(mbr_measure, "run_one", run_one)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fake"))
    with pytest.raises(RuntimeError, match="cannot build"):
        run_mem.main(_measure_args(tmp_path))
    out = tmp_path / "out"
    crash = json.loads((out / "rows_canonical.jsonl").read_text(encoding="utf-8"))
    assert crash == json.loads(capsys.readouterr().out)
    assert crash["status"] == "CRASH"
    assert crash["error"] == "RuntimeError: cannot build the model"
    assert crash["model_revision"] == "unknown"
    assert not (out / "rows.jsonl").exists()


def test_capture_snapshot_cpu_skips_the_measurement(
    benchmark: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.harness import mbr_measure

    capture = importlib.import_module("capture_snapshot")
    monkeypatch.setattr(mbr_measure, "run_one", lambda args: pytest.fail("CPU capture ran"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fake"))
    assert capture.main(_measure_args(tmp_path)) == 0
    out = tmp_path / "out"
    row = json.loads((out / "snapshot-E-k8-s512.json").read_text(encoding="utf-8"))
    assert (row["status"], row["arm"], row["mode"], row["window"]) == (
        "cpu", "E", "normal", "k=8,S=512",
    )
    assert (row["card"], row["torch_version"], row["model"]) == (
        "CPU", "fake", "Moonlight-16B-A3B",
    )
    assert row["copy_peak_bytes"] is None
    assert not (out / "snapshot-E-k8-s512.pickle").exists()


@pytest.fixture
def snapshot_run(
    benchmark: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    from benchmarks.harness import mbr_measure
    from soup_cli.utils import l2l_step, layer_stream_runtime

    class FakeOOMError(Exception):
        pass

    state = SimpleNamespace(events=[], live=False, fail_copy=False, device="cuda")

    class Tensor:
        def fill_(self, value: int) -> None:
            assert value == 1
            state.events.append("fill")

        def __del__(self) -> None:
            state.live = False
            state.events.append("free")

    def empty(size: int, *, device: str, dtype: str) -> Tensor:
        assert (size, device, dtype) == (1024, "cuda", "uint8")
        state.events.append("probe")
        if state.fail_copy:
            state.events.append("oom")
            raise FakeOOMError("no space")
        state.live = True
        return Tensor()

    def history(**kwargs: Any) -> None:
        state.events.append(kwargs)

    def dump(path: str) -> None:
        assert not state.live
        state.events.append("dump")
        cached = _block(1024, live=False)
        snapshot = {"segments": []} if "oom" in state.events else _snapshot(cached)
        Path(path).write_bytes(pickle.dumps(snapshot))

    def run_one(args: SimpleNamespace) -> dict:
        assert (args.kind, args.deterministic, args.steps) == ("speed", False, 2)
        layer_stream_runtime.build_streamed_model()
        for _ in range(args.steps):
            l2l_step.L2LStep().run()
        state.events.append("close")
        return {**_row("E", 8, 512, 10), "peak_ram": 8000}

    class Step:
        def run(self) -> None:
            state.events.append("step")

    def stat(name: str, value: int) -> int:
        state.events.append(name)
        return value

    monkeypatch.setattr(layer_stream_runtime, "build_streamed_model",
                        lambda: state.events.append("build"))
    monkeypatch.setattr(l2l_step, "L2LStep", Step)
    monkeypatch.setattr(mbr_measure, "run_one", run_one)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="fake", uint8="uint8", empty=empty, cuda=SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda: "Fake 3050",
            current_device=lambda: state.device,
            synchronize=lambda: state.events.append("sync"),
            memory_allocated=lambda: stat("memory_allocated", 1000),
            memory_reserved=lambda: stat("memory_reserved", 2500),
            max_memory_allocated=lambda: stat("max_memory_allocated", 2000),
            max_memory_reserved=lambda: stat("max_memory_reserved", 3000),
            OutOfMemoryError=FakeOOMError,
            memory=SimpleNamespace(_record_memory_history=history, _dump_snapshot=dump),
        ),
    ))
    state.run = lambda *flags: importlib.import_module("capture_snapshot").main([
        *_measure_args(tmp_path), "--device", "auto", *flags,
    ])
    state.stem = tmp_path / "out" / "snapshot-E-k8-s512"
    return state


def test_capture_snapshot_dumps_and_stops_recording_before_probe(
    snapshot_run: SimpleNamespace,
) -> None:
    assert snapshot_run.run("--max-entries", "37") == 0
    stats = ["memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"]
    assert snapshot_run.events == [
        {"enabled": "all", "context": "all", "stacks": "all", "max_entries": 37},
        "build", "step", "step", "sync", *stats,
        "dump", {"enabled": None}, "probe", "fill", "sync", "free", "close", *stats,
    ]
    row = json.loads(snapshot_run.stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert (row["status"], row["card"], row["copy_peak_bytes"], row["copy_ok"]) == (
        "ok", "Fake 3050", 1024, True,
    )
    assert row["memory_allocated"] == 1000
    assert row["memory_reserved"] == 2500
    assert row["max_memory_allocated"] == 2000
    assert row["max_memory_reserved"] == 3000
    assert row["harness_peaks"] == {
        "peak_allocated": 10, "peak_reserved": 20, "peak_nvml": 30, "peak_ram": 8000,
    }
    assert snapshot_run.stem.with_suffix(".pickle").exists()


def test_capture_snapshot_preserves_cached_blocks_when_the_copy_probe_ooms(
    snapshot_run: SimpleNamespace,
) -> None:
    snapshot_run.fail_copy = True
    assert snapshot_run.run() == 0
    snapshot = pickle.loads(snapshot_run.stem.with_suffix(".pickle").read_bytes())
    assert snapshot == _snapshot(_block(1024, live=False))
    row = json.loads(snapshot_run.stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert row["status"] == "ok"
    assert row["copy_peak_bytes"] == 1024
    assert row["copy_ok"] is False
    assert row["copy_error"] == "no space"


def test_capture_snapshot_defaults_to_one_million_history_entries(
    snapshot_run: SimpleNamespace,
) -> None:
    assert snapshot_run.run() == 0
    assert snapshot_run.events[0]["max_entries"] == 1_000_000


def test_analyze_snapshots_writes_requested_reports(
    benchmark: None, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    analyze = importlib.import_module("analyze_snapshots")
    before, after = tmp_path / "c.pickle", tmp_path / "e.pickle"
    before.write_bytes(pickle.dumps(_snapshot(_block(3 * MIB, live=False))))
    after.write_bytes(pickle.dumps(_snapshot(
        _block(2 * MIB, native="scratch"), _block(MIB, live=False),
    )))
    output, markdown = tmp_path / "diff.json", tmp_path / "diff.md"
    assert analyze.main([
        "--c", str(before), "--e", str(after), "--json", str(output), "--markdown", str(markdown),
    ]) == 0
    diff = json.loads(output.read_text(encoding="utf-8"))
    assert diff["main_source"] == "staging"
    assert diff["totals"]["delta"]["live_bytes"] == 2**21
    assert diff["traps"]["delta"] == {"releasable_bytes": -3 * MIB, "trapped_bytes": MIB}
    summary = capsys.readouterr().out
    assert "Main source: staging" in summary
    assert "live +2.0 MiB" in summary
    assert "C: releasable 3.0 MiB, trapped 0.0 MiB" in summary
    assert "E: releasable 0.0 MiB, trapped 1.0 MiB" in summary
    assert summary == markdown.read_text(encoding="utf-8")


def test_analyze_snapshots_reads_plain_allocator_pickles_without_torch(
    benchmark: None, tmp_path: Path, capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    analyze = importlib.import_module("analyze_snapshots")
    before, after = tmp_path / "c.pickle", tmp_path / "e.pickle"
    before.write_bytes(pickle.dumps(_snapshot()))
    after.write_bytes(pickle.dumps(_snapshot(_block(2**21, native="matmul"))))
    assert analyze.main(["--c", str(before), "--e", str(after)]) == 0
    assert "Main source: cublaslt_workspace" in capsys.readouterr().out
