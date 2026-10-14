"""Tests for the G7 profile arm's named ranges (benchmarks/harness/visit_ranges.py, ticket 04).

CPU only: fake layers from the runtime's own layer class, the CPU profiler, and the trace
analyser on the exported events. The GPU smoke test lives in test_g7_visit_ranges_gpu.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

import trace_analyser  # noqa: E402
import visit_ranges  # noqa: E402


class FakePool:
    def wait(self, idx):
        return {}


class FakePrefetcher:
    def advance(self, idx):
        return None


class Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Parameter(torch.eye(4))

    def forward(self, hidden_states):
        return hidden_states @ self.lora


def make_layers(count=3):
    from soup_cli.utils.layer_stream_runtime import _streamed_layer_class

    cls = _streamed_layer_class()
    pool, prefetcher = FakePool(), FakePrefetcher()
    return [cls(Inner(), idx, pool, prefetcher) for idx in range(count)]


def run_step(layers):
    hidden = torch.ones(1, 4, requires_grad=True)
    out = hidden
    for layer in layers:
        out = layer(out)
    out.sum().backward()


def profiled_events(layers, tmp_path, steps=2):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(steps):
            with visit_ranges.step_range():
                run_step(layers)
    path = tmp_path / "trace.json"
    prof.export_chrome_trace(str(path))
    return json.loads(path.read_text(encoding="utf-8"))["traceEvents"]


def test_every_range_appears_in_the_trace_and_the_analyser_finds_the_visits(tmp_path):
    layers = make_layers(3)
    with visit_ranges.install_ranges(layers):
        events = profiled_events(layers, tmp_path, steps=2)

    names = {e["name"] for e in events if e.get("cat") == "user_annotation"}
    r = trace_analyser.RANGES
    assert {r.step, r.forward_visit, r.recompute_visit, r.weight_prep, r.functional_call,
            r.layer_call, r.layer_backward} <= names  # fmt: skip
    result = trace_analyser.analyse_trace(events)
    assert result["summary"]["forward"]["count"] == 6
    assert result["summary"]["recompute"]["count"] == 6
    for visit in result["visits"]:
        parts = ("launch_ms", "weight_prep_ms", "functional_call_ms", "python_dispatcher_ms")
        assert sum(visit[p] for p in parts) == pytest.approx(visit["host_ms"])
        assert visit["weight_prep_ms"] > 0
        assert visit["functional_call_ms"] >= 0
    # Layer 0's input needs no gradient, so it has no layer backward in the real model. Here
    # the input requires grad, so every layer has one.
    assert result["summary"]["layer_backward"]["count"] == 6


def test_removing_the_ranges_restores_the_layers_and_drops_the_hooks(tmp_path):
    layers = make_layers(2)
    cls = type(layers[0])
    before = {name: cls.__dict__[name] for name in ("_body", "forward", "_substituted_weights")}

    with visit_ranges.install_ranges(layers):
        pass

    assert {name: cls.__dict__[name] for name in before} == before
    assert all("forward" not in vars(layer.inner) for layer in layers)
    events = profiled_events(layers, tmp_path, steps=1)
    names = {e["name"] for e in events if e.get("cat") == "user_annotation"}
    assert trace_analyser.RANGES.forward_visit not in names


def test_a_step_with_ranges_gives_the_same_loss_gradient_as_one_without():
    layers = make_layers(2)

    def grad():
        for layer in layers:
            layer.inner.lora.grad = None
        run_step(layers)
        return [layer.inner.lora.grad.clone() for layer in layers]

    plain = grad()
    with visit_ranges.install_ranges(layers):
        ranged = grad()
    assert all(torch.equal(a, b) for a, b in zip(plain, ranged))
