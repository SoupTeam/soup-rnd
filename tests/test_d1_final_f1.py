"""The scoped study must preserve the shipped path on uncovered projections."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


def test_uncovered_native_fused_route_is_preserved(monkeypatch):
    import d1_final_f1
    import d1_fused_nf4

    from soup_cli.utils import layer_stream_runtime as runtime

    calls = []
    old_gate = runtime._can_use_checkpoint_visible_nf4_gemm
    old_linear = runtime.checkpoint_visible_nf4_linear
    monkeypatch.setattr(runtime, "_can_use_checkpoint_visible_nf4_gemm", lambda *_: True)
    monkeypatch.setattr(runtime, "checkpoint_visible_nf4_linear", lambda *a: calls.append(a))
    monkeypatch.setattr(
        d1_fused_nf4,
        "linear",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("uncovered projection reached experimental arithmetic")
        ),
    )
    value = SimpleNamespace(dtype="bf16", shape=(128, 4096), numel=lambda: 128 * 4096)
    state = SimpleNamespace(dtype="bf16", shape=(1024, 4096))
    try:
        d1_final_f1.route("candidate")
        assert runtime._can_use_checkpoint_visible_nf4_gemm(value, state)
        runtime.checkpoint_visible_nf4_linear(value, "packed", state)
        assert calls == [(value, "packed", state, None)]
    finally:
        runtime._can_use_checkpoint_visible_nf4_gemm = old_gate
        runtime.checkpoint_visible_nf4_linear = old_linear


@pytest.mark.parametrize("invalid_block", [0, 1, 4, None])
def test_timing_stops_after_first_invalid_block(monkeypatch, tmp_path, invalid_block):
    import arm_validity
    import d1_final_f1
    import d1_nf4_study
    from test_d1_nf4_study import arm

    class Watch:
        record = {"outcome": "ok", "checks": {"foreign_reads": {"outcome": "ok"}}}

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monitor = SimpleNamespace(
        sample=lambda: {"available": True, "compute": [], "graphics": [], "uuid": "GPU-one"}
    )
    monkeypatch.setattr(d1_nf4_study, "GPUMonitor", lambda: monitor)
    monkeypatch.setattr(d1_nf4_study, "block_readings", lambda: {})
    monkeypatch.setattr(arm_validity, "ArmWatch", Watch)
    monkeypatch.setattr(d1_final_f1, "peer_check", lambda *_args: True)
    calls = []

    def execute(command, **_kwargs):
        block, index = divmod(len(calls), 4)
        name = "ABBA"[index]
        output = Path(command[command.index("--output") + 1])
        row = arm(name, index, value=10 if name == "A" else 9)
        if block == invalid_block:
            row["clock_sm_mhz"] = [1000, 1740]
        (output / "result.json").write_text(json.dumps(row), encoding="utf-8")
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(d1_final_f1.subprocess, "run", execute)
    result = d1_final_f1.run(SimpleNamespace(output=tmp_path, data=tmp_path / "fixture"))
    expected = 20 if invalid_block is None else (invalid_block + 1) * 4
    assert len(calls) == len(result["rows"]) == expected
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved == result
    if invalid_block is None:
        assert result["status"] == "MEASURED_SUBSET"
        assert not result.get("stopped_early")
    else:
        assert result["status"] == "NO_VERDICT"
        assert result["stopped_early"] is True
        assert result["comparisons"][0]["accepted_blocks"] == 0
