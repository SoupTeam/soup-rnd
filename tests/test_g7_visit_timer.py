"""Tests for the G7 timer arm (benchmarks/harness/visit_timer.py, G7 ticket 03).

The layers are real ``StreamedDecoderLayer`` objects on the CPU, built around a tiny
synthetic inner module with a fake pool and prefetcher. No test needs a GPU, a shard
folder or a model download. All numbers are synthetic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def timer():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_visit_timer_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "visit_timer.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class FakePool:
    def wait(self, idx):
        return {}


class FakePrefetcher:
    def advance(self, idx):
        return None


class Inner(torch.nn.Module):
    """One trainable matrix, standing in for a LoRA-adapted layer."""

    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Parameter(torch.eye(4))

    def forward(self, hidden_states):
        return hidden_states @ self.lora


def make_layers(count=3):
    from soup_cli.utils.layer_stream_runtime import _streamed_layer_class

    cls = _streamed_layer_class()
    pool, prefetcher = FakePool(), FakePrefetcher()
    return [cls(Inner(), idx, pool, prefetcher) for idx in range(count)], pool, prefetcher


def run_step(layers):
    hidden = torch.ones(1, 4, requires_grad=True)
    out = hidden
    for layer in layers:
        out = layer(out)
    out.sum().backward()


def counting_wrappers(log):
    """A wrapper for every patch point that logs the point's name when it runs."""

    def make(point):
        def wrap(orig):
            def wrapped(*args, **kwargs):
                log.append(point)
                return orig(*args, **kwargs)

            return wrapped

        return wrap

    return {point: make(point) for point in (
        "body", "substituted_weights", "forward", "functional_call", "checkpoint",
        "pool_wait", "prefetcher_advance",
    )}  # fmt: skip


# --- the patches ------------------------------------------------------------------------


def test_every_patch_point_runs_for_every_streamed_layer(timer):
    layers, _, _ = make_layers(3)
    log = []
    with timer.install_patches(layers, counting_wrappers(log)):
        run_step(layers)

    # Two visits per layer (forward, recompute), so six of each point inside the layers.
    for point in ("body", "substituted_weights", "functional_call", "pool_wait"):
        assert log.count(point) == 6, point
    assert log.count("forward") == 3
    assert log.count("checkpoint") == 3
    assert log.count("prefetcher_advance") == 6


def test_removing_the_patches_restores_the_original_behaviour(timer):
    layers, pool, prefetcher = make_layers(2)
    cls = type(layers[0])
    before = {name: cls.__dict__[name] for name in ("_body", "_substituted_weights", "forward")}
    body_cells = [cell.cell_contents for cell in before["_body"].__closure__]
    forward_cells = [cell.cell_contents for cell in before["forward"].__closure__]

    log = []
    patches = timer.install_patches(layers, counting_wrappers(log))
    patches.remove()
    run_step(layers)

    assert log == []
    for name, original in before.items():
        assert cls.__dict__[name] is original
    assert [cell.cell_contents for cell in cls.__dict__["_body"].__closure__] == body_cells
    assert [cell.cell_contents for cell in cls.__dict__["forward"].__closure__] == forward_cells
    assert "wait" not in vars(pool)
    assert "advance" not in vars(prefetcher)


def test_removing_the_patches_keeps_an_instance_attribute_that_was_there_before(timer):
    layers, pool, _ = make_layers(1)

    def own_wait(idx):
        return {}

    pool.wait = own_wait
    timer.install_patches(layers, counting_wrappers([])).remove()

    assert pool.wait is own_wait


def test_the_shared_pool_is_patched_once_not_once_per_layer(timer):
    layers, _, _ = make_layers(3)
    log = []
    with timer.install_patches(layers, counting_wrappers(log)):
        layers[0].pool.wait(0)

    assert log == ["pool_wait"]


def test_patching_twice_is_refused(timer):
    layers, _, _ = make_layers(1)
    with timer.install_patches(layers, counting_wrappers([])):
        with pytest.raises(RuntimeError, match="already"):
            timer.install_patches(layers, counting_wrappers([]))


def test_an_unknown_patch_point_is_refused(timer):
    layers, _, _ = make_layers(1)
    with pytest.raises(ValueError, match="nonsense"):
        timer.install_patches(layers, {"nonsense": lambda orig: orig})


def test_a_failed_install_leaves_the_class_unpatched(timer):
    layers, _, _ = make_layers(1)
    cls = type(layers[0])
    original = cls.__dict__["_body"]
    with pytest.raises(ValueError):
        timer.install_patches(layers, {"body": lambda orig: orig, "nonsense": lambda o: o})

    assert cls.__dict__["_body"] is original


# --- the timer wrappers -----------------------------------------------------------------


class Clock:
    """A fake monotonic clock that moves 1000 ns on every read."""

    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 1000
        return self.now


def timed_step(timer, layers, recorder):
    recorder.begin_step()
    run_step(layers)
    return recorder.end_step()


def test_each_layer_has_one_forward_and_one_recompute_visit_per_step(timer):
    layers, _, _ = make_layers(3)
    recorder = timer.Recorder(clock=Clock())
    with timer.install_patches(layers, timer.timer_wrappers(recorder)):
        timed_step(timer, layers, recorder)

    visits = recorder.steps[0].visits
    for idx in range(3):
        kinds = sorted(kind for kind, i, _, _ in visits if i == idx)
        assert kinds == ["forward", "recompute"]


def test_a_visit_is_forward_when_called_from_the_layer_forward_and_recompute_otherwise(timer):
    layers, _, _ = make_layers(2)
    recorder = timer.Recorder(clock=Clock())
    with timer.install_patches(layers, timer.timer_wrappers(recorder)):
        timed_step(timer, layers, recorder)

    order = [(kind, idx) for kind, idx, _, _ in recorder.steps[0].visits]
    # Forward walks 0, 1. Backward recomputes in reverse, 1 then 0.
    assert order == [("forward", 0), ("forward", 1), ("recompute", 1), ("recompute", 0)]


def test_visit_time_is_the_clock_difference_around_the_layer_body(timer):
    layers, _, _ = make_layers(1)
    clock = Clock()
    recorder = timer.Recorder(clock=clock)
    with timer.install_patches(layers, timer.timer_wrappers(recorder)):
        timed_step(timer, layers, recorder)

    for _, _, start, end in recorder.steps[0].visits:
        assert end > start
        assert (end - start) % 1000 == 0


def test_layer_backward_is_the_hook_window_without_the_recompute_inside_it(timer):
    layers, _, _ = make_layers(2)
    recorder = timer.Recorder(clock=Clock())
    with timer.install_patches(layers, timer.timer_wrappers(recorder)):
        timed_step(timer, layers, recorder)

    step = recorder.steps[0]
    assert sorted(idx for idx, _, _ in step.windows) == [0, 1]
    for idx, start, end in step.windows:
        recompute = [(s, e) for kind, i, s, e in step.visits if kind == "recompute" and i == idx]
        assert len(recompute) == 1
        # The non-reentrant checkpoint recomputes inside the layer's backward window.
        assert start <= recompute[0][0] and recompute[0][1] <= end


def test_the_input_hook_is_skipped_when_the_layer_input_needs_no_gradient(timer):
    layers, _, _ = make_layers(1)
    recorder = timer.Recorder(clock=Clock())
    with timer.install_patches(layers, timer.timer_wrappers(recorder)):
        recorder.begin_step()
        out = layers[0](torch.ones(1, 4))
        out.sum().backward()
        recorder.end_step()

    assert recorder.steps[0].windows == []


def test_no_hooks_are_made_when_gradients_are_off(timer):
    layers, _, _ = make_layers(1)
    recorder = timer.Recorder(clock=Clock())
    with timer.install_patches(layers, timer.timer_wrappers(recorder)):
        recorder.begin_step()
        with torch.no_grad():
            layers[0](torch.ones(1, 4, requires_grad=True))
        recorder.end_step()

    step = recorder.steps[0]
    assert [kind for kind, *_ in step.visits] == ["forward"]
    assert step.windows == []


# --- the summary ------------------------------------------------------------------------


def fake_step(recorder, forward_ns, recompute_ns, backward_ns, step_ns, layers=(0, 1)):
    """Add a step with the given durations in ns, one forward, recompute and backward each."""
    step = recorder.new_step_record()
    t = 0
    for idx in layers:
        step.visits.append(("forward", idx, t, t + forward_ns))
        t += forward_ns
    for idx in reversed(layers):
        window_start = t
        step.visits.append(("recompute", idx, t, t + recompute_ns))
        step.windows.append((idx, window_start, t + recompute_ns + backward_ns))
        t = t + recompute_ns + backward_ns
    step.ns = step_ns


def test_summary_gives_the_median_over_layers_and_timed_steps_and_the_raw_values(timer):
    recorder = timer.Recorder(clock=Clock())
    fake_step(recorder, 9_000_000, 9_000_000, 9_000_000, 1_000_000_000)  # warm-up, ignored
    fake_step(recorder, 1_000_000, 2_000_000, 3_000_000, 500_000_000)
    fake_step(recorder, 3_000_000, 4_000_000, 5_000_000, 700_000_000)

    summary = recorder.summary(warmup=1, timed=2)

    assert summary["forward_visit_ms"]["values"] == [1.0, 1.0, 3.0, 3.0]
    assert summary["forward_visit_ms"]["median"] == pytest.approx(2.0)
    assert summary["recompute_visit_ms"]["median"] == pytest.approx(3.0)
    assert summary["layer_backward_ms"]["median"] == pytest.approx(4.0)
    assert summary["layer_backward_ms"]["values"] == [3.0, 3.0, 5.0, 5.0]
    assert summary["step_ms"]["values"] == [500.0, 700.0]
    assert summary["step_ms"]["median"] == pytest.approx(600.0)


def test_summary_states_whether_every_layer_had_two_visits_per_step(timer):
    recorder = timer.Recorder(clock=Clock())
    fake_step(recorder, 1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert recorder.summary(warmup=0, timed=1)["two_visits_per_layer"] is True

    recorder = timer.Recorder(clock=Clock())
    fake_step(recorder, 1_000_000, 1_000_000, 1_000_000, 1_000_000)
    recorder.steps[0].visits.pop()  # the last recompute is missing
    assert recorder.summary(warmup=0, timed=1)["two_visits_per_layer"] is False


def test_summary_needs_all_the_steps_it_is_asked_for(timer):
    recorder = timer.Recorder(clock=Clock())
    fake_step(recorder, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="steps"):
        recorder.summary(warmup=1, timed=2)


# --- the LoRA window check --------------------------------------------------------------


def test_lora_accumulation_is_reported_against_the_backward_window(timer):
    layers, _, _ = make_layers(2)
    recorder = timer.Recorder(clock=Clock())
    with timer.install_patches(layers, timer.timer_wrappers(recorder)):
        with timer.watch_lora_gradients(layers, recorder):
            timed_step(timer, layers, recorder)

    report = recorder.lora_window_report()

    assert report["inside"] + report["outside"] == 2  # one LoRA matrix per layer
    assert report["state"] in ("inside", "outside", "mixed")
    assert (report["state"] == "inside") == (report["outside"] == 0)


def test_lora_report_calls_marks_inside_and_outside_by_time(timer):
    recorder = timer.Recorder(clock=Clock())
    step = recorder.new_step_record()
    step.windows.append((0, 100, 200))
    step.lora_marks.extend([(0, 150), (0, 200)])
    assert recorder.lora_window_report() == {"inside": 2, "outside": 0, "state": "inside"}

    step.lora_marks.append((0, 201))
    assert recorder.lora_window_report() == {"inside": 2, "outside": 1, "state": "mixed"}

    step.lora_marks[:] = [(0, 50), (1, 150)]  # layer 1 has no window at all
    assert recorder.lora_window_report() == {"inside": 0, "outside": 2, "state": "outside"}


def test_watching_lora_gradients_removes_its_hooks(timer):
    layers, _, _ = make_layers(1)
    recorder = timer.Recorder(clock=Clock())
    with timer.watch_lora_gradients(layers, recorder):
        pass
    timed_step(timer, layers, recorder)

    assert recorder.steps[0].lora_marks == []
