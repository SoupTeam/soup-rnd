"""GPU smoke test for the G7 profile arm (ticket 04): needs CUDA, runs with ``-m smoke``.

A small synthetic streamed model (3 layers of the ``llama-8b`` shape, a small vocabulary,
written by the synthetic checkpoint writer with the GPU hidden). The ranges install, appear in
a CUDA trace, and after removal a step is bit-equal to a step taken before they were installed.
Run: ``~/.venvs/soup-rnd-cuda/bin/python -m pytest <this file> -m smoke --no-cov``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))


@pytest.fixture(scope="module")
def streamed(tmp_path_factory):
    import stream_probe

    from soup_cli.utils.layer_stream import resolve_stream_dtype

    root = tmp_path_factory.mktemp("g7smoke")
    weights = root / "weights"
    subprocess.run(
        [sys.executable, str(HARNESS / "synth_checkpoint.py"), "--shape", "llama-8b",
         "--layers", "3", "--vocab", "2048", "--out", str(weights)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )  # fmt: skip
    args = argparse.Namespace(
        weights=str(weights), shards=str(root / "shards"), quant="nf4", tier="ram",
        buffers=2, read_ahead=2, no_pin=False, seed=3, lora_r=8,
        lora_targets="q_proj,k_proj,v_proj,o_proj", lazy_shard_handles=False,
        control_sync_source=False, stripe_dir=[],
    )  # fmt: skip
    device = "cuda"
    model, runtime, config, *_ = stream_probe.build(args, device, resolve_stream_dtype(device))
    ids = torch.randint(0, int(config.vocab_size), (1, 16), device=device,
                        generator=torch.Generator(device=device).manual_seed(17))  # fmt: skip
    yield model, ids
    runtime.close()


def one_step(model, ids):
    for param in model.parameters():
        param.grad = None
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    torch.cuda.synchronize()
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    return loss.detach().clone(), grads


def test_ranges_install_appear_in_a_cuda_trace_and_leave_the_step_bit_equal(streamed, tmp_path):
    import trace_analyser
    import visit_ranges
    from torch.profiler import ProfilerActivity, profile

    from soup_cli.utils.layer_stream_runtime import decoder_owner

    model, ids = streamed
    layers = list(decoder_owner(model).layers)
    one_step(model, ids)  # warm up
    base_loss, base_grads = one_step(model, ids)

    with visit_ranges.install_ranges(layers):
        one_step(model, ids)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with visit_ranges.step_range():
                one_step(model, ids)
    trace = tmp_path / "trace.json"
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text(encoding="utf-8"))["traceEvents"]

    r = trace_analyser.RANGES
    names = {e["name"] for e in events if e.get("cat") == "user_annotation"}
    assert {r.step, r.forward_visit, r.recompute_visit, r.weight_prep, r.functional_call,
            r.layer_call, r.layer_backward} <= names  # fmt: skip
    analysis = trace_analyser.analyse_trace(events)
    assert analysis["summary"]["forward"]["count"] == 3
    assert analysis["summary"]["recompute"]["count"] == 3
    assert analysis["summary"]["forward"]["launches_total"] > 0
    assert analysis["summary"]["forward"]["kernel_ms"] > 0

    loss, grads = one_step(model, ids)  # after removal
    assert torch.equal(loss, base_loss)
    assert len(grads) == len(base_grads) > 0
    assert all(torch.equal(a, b) for a, b in zip(grads, base_grads))


def test_control_injections_show_up_as_50_launches_and_about_200_us_per_visit(streamed, tmp_path):
    import trace_analyser
    import visit_control
    import visit_ranges
    from torch.profiler import ProfilerActivity, profile

    from soup_cli.utils.layer_stream_runtime import decoder_owner

    model, ids = streamed
    layers = list(decoder_owner(model).layers)

    def forward_summary(**wrappers):
        with visit_ranges.install_ranges(layers, **wrappers):
            one_step(model, ids)  # warm up
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with visit_ranges.step_range():
                    one_step(model, ids)
        trace = tmp_path / "trace.json"
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text(encoding="utf-8"))["traceEvents"]
        return trace_analyser.analyse_trace(events)["summary"]["forward"]

    base = forward_summary()
    injected = forward_summary(inner=visit_control.control_wrappers())

    assert injected["launches_total"] - base["launches_total"] == 50
    prep_delta_ms = injected["weight_prep_ms"] - base["weight_prep_ms"]
    assert 0.16 <= prep_delta_ms <= 0.24
