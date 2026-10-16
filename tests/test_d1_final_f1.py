"""The scoped study must preserve the shipped path on uncovered projections."""

import sys
from pathlib import Path
from types import SimpleNamespace

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
