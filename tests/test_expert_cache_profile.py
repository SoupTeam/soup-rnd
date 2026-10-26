"""Behavioral controls for bounded, transparent profiling instrumentation."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harness.expert_cache_profile import (
    _training_hashes,
    accumulated_profile_update,
    operation_markers,
    profiler_summary,
    require_cuda_profile,
    retain_trace,
    run_profile,
)


def test_markers_preserve_return_identity_and_restore_after_exception() -> None:
    value = object()
    entries = []

    class Fetcher:
        def fetch(self, item):
            assert item is value
            return item

    original = Fetcher.fetch

    @contextlib.contextmanager
    def marker(name):
        entries.append(name)
        yield

    with pytest.raises(RuntimeError, match="interrupted"):
        with operation_markers(marker, ((Fetcher, "fetch", "test.fetch"),)):
            assert Fetcher().fetch(value) is value
            raise RuntimeError("interrupted")
    assert Fetcher.fetch is original
    assert entries == ["test.fetch"]


@pytest.mark.parametrize("limit", (1, 2**20))
def test_trace_gzip_is_lossless_and_overflow_is_retained(tmp_path: Path, limit: int) -> None:
    raw = b'{"traceEvents":[]}'
    profiler = SimpleNamespace(export_chrome_trace=lambda path: Path(path).write_bytes(raw))
    if limit == 1:
        with pytest.raises(RuntimeError, match="size bound"):
            retain_trace(profiler, tmp_path, limit)
        assert (tmp_path / "trace.json").read_bytes() == raw
    else:
        result = retain_trace(profiler, tmp_path, limit)
        assert result["decoded_verified"]
        assert not (tmp_path / "trace.json").exists()
    assert gzip.decompress((tmp_path / "trace.json.gz").read_bytes()) == raw
    proof = json.loads((tmp_path / "trace-retention.json").read_text())
    assert proof["decoded_sha256"] == hashlib.sha256(raw).hexdigest()
    assert proof["within_bound"] == (limit != 1)
    with pytest.raises(FileExistsError):
        retain_trace(profiler, tmp_path, 2**20)


def test_cuda_profile_refuses_cpu_only_and_copy_only_fallbacks() -> None:
    good = {"event_count": 6, "cuda_event_count": 3, "cuda_kernel_event_count": 2,
            "cuda_h2d_event_count": 1}
    require_cuda_profile(good)
    for key in ("cuda_kernel_event_count", "cuda_h2d_event_count"):
        with pytest.raises(RuntimeError, match="actual CUDA"):
            require_cuda_profile({**good, key: 0})
    with pytest.raises(RuntimeError, match="CPU events"):
        require_cuda_profile({**good, "event_count": 3})


def test_profile_retains_native_cuda_refusal(tmp_path: Path, monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    out = tmp_path / "profile"
    assert run_profile(spec, "two_read_cache", out) == 1
    result = json.loads((out / "profile-result.json").read_text())
    assert result["status"] == "failed"
    assert "native BF16 CUDA" in result["error"]
    assert result["unprofiled_speed_verdict"] == "NO VERDICT"
    with pytest.raises(FileExistsError):
        run_profile(spec, "two_read_cache", out)


def test_profile_records_only_first_microbatch_and_preserves_accumulated_adamw() -> None:
    pytest.importorskip("bitsandbytes")
    import torch

    from benchmarks.harness.two_read_cache_worker import _optimizer

    class AdapterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.tensor([.3]))
            self.lora_B = torch.nn.Parameter(torch.tensor([.7]))

        def forward(self, input_ids, labels, use_cache):
            assert not use_cache
            prediction = input_ids.float() * self.lora_A * self.lora_B
            return SimpleNamespace(loss=(prediction - labels).square().mean())

    reference, candidate = AdapterModel(), AdapterModel()
    spec = {"accumulation": 2}
    expected_optimizer, optimizer = _optimizer(reference, spec), _optimizer(candidate, spec)
    batches = torch.tensor([[[1, 2, 3]], [[3, 2, 1]]])
    reference.zero_grad(set_to_none=True)
    expected_losses = []
    for ids in batches:
        loss = reference(input_ids=ids, labels=ids, use_cache=False).loss
        expected_losses.append(float(loss.detach()))
        (loss / 2).backward()
    expected_optimizer.step()
    profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                     record_shapes=False, profile_memory=False,
                                     with_stack=False, with_flops=False)
    result = accumulated_profile_update(
        candidate, optimizer, batches, spec, device="cpu", profiler=profile,
        record_function=torch.profiler.record_function,
    )
    assert result["losses"] == expected_losses
    assert result["profiled_microbatches"] == [0]
    assert result["unprofiled_microbatches"] == [1]
    assert _training_hashes(reference, expected_optimizer) == _training_hashes(candidate, optimizer)
    forward = next(row for row in profiler_summary(profile)["key_averages"]
                   if row["name"] == "soup.profile.microbatch.forward")
    assert forward["count"] == 1


def test_real_cpu_nf4_profile_markers_preserve_complete_two_read_trajectory(tmp_path: Path) -> None:
    pytest.importorskip("bitsandbytes")
    import torch

    from benchmarks.harness.two_read_cache_common import compare_captures
    from benchmarks.harness.two_read_cache_worker import _adapter_values, _adapters, _run_accuracy
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes
    from tests.test_expert_stream import _shard, _tiny_deepseek_v3
    from tests.test_stream_expert_cache_nf4 import _build

    weights, resident = _tiny_deepseek_v3(tmp_path)
    del resident
    skeleton = build_meta_skeleton(weights, dtype="float32", quant="nf4")
    suffixes = quantised_layer_suffixes(skeleton)
    del skeleton
    index = _shard(tmp_path, weights, quant="nf4", expert_layout="per_expert",
                   quant_suffixes=suffixes, quant_device="cpu")
    model, runtime = _build(weights, index, str(tmp_path / "shards"), two_read=True, device="cpu")
    try:
        runtime.expert_fetcher.configure_cache(host_cache_bytes=2**20, device_cache_bytes=2**15,
                                               host_cache_policy="static")
        batches = torch.randint(0, 128, (4, 1, 12), generator=torch.Generator().manual_seed(29))
        spec = {"accuracy_updates": 2, "accumulation": 2}
        initial = _adapter_values(model)
        expected = _run_accuracy(model, runtime, batches, spec)
        with torch.no_grad():
            for name, value in _adapters(model).items():
                value.copy_(initial[name])
        runtime.expert_fetcher.clear_cache()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=False, profile_memory=False,
                                    with_stack=False, with_flops=False) as profile:
            with operation_markers(torch.profiler.record_function):
                actual = _run_accuracy(model, runtime, batches, spec)
        result = compare_captures(expected, actual)
        assert result["passed"] and result["bitwise"]
        names = {row["name"] for row in profiler_summary(profile)["key_averages"]}
        assert {"soup.expert.fetch", "soup.expert.storage_read", "soup.nf4.quant_state",
                "soup.nf4.dequantize_4bit"}.issubset(names)
    finally:
        runtime.close()


@pytest.mark.gpu(reason="Actual CUDA profiler and CUPTI availability require a GPU")
def test_real_cuda_profiler_collects_kernel_and_h2d_events(tmp_path: Path, record_property) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("actual CUDA profiler preflight requires CUDA")
    source = torch.arange(256, dtype=torch.float32).reshape(16, 16).pin_memory()
    torch.ones((16, 16), device="cuda").sin_()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False, with_stack=False, with_flops=False,
    ) as profile:
        with torch.profiler.record_function("soup.cuda_profiler_preflight"):
            value = source.to("cuda", copy=True).requires_grad_(True)
            (value @ value).square().mean().backward()
            torch.cuda.synchronize()
    summary = profiler_summary(profile)
    proof = tmp_path / "cuda-profiler-proof.json"
    proof.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record_property("cuda_profiler_proof", str(proof))
    require_cuda_profile(summary)
    assert any("backward" in row["name"].lower() for row in summary["key_averages"])
