"""Complete NF4 expert-cache updates against the same uncached NF4 base."""

from __future__ import annotations

import pytest


def _build(weights, index, shards, *, two_read, device):
    import torch

    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_expert_stream import _lora

    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=_lora(),
        device=device, dtype="float32", quant="nf4", buffers=2, pin=False, seed=3,
        two_read_experts=two_read,
    )
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return model.eval(), runtime


def _gate(expected, actual, *, bitwise):
    """Reference-first original tolerances; CPU additionally requires exact bytes."""
    import torch

    assert set(actual) == set(expected)
    for key, reference in expected.items():
        candidate = actual[key]
        if isinstance(reference, dict):
            _gate(reference, candidate, bitwise=bitwise)
        elif isinstance(reference, (list, tuple)):
            assert len(reference) == len(candidate)
            _gate(dict(enumerate(reference)), dict(enumerate(candidate)), bitwise=bitwise)
        elif isinstance(reference, torch.Tensor):
            assert reference.shape == candidate.shape and reference.dtype == candidate.dtype, key
            assert torch.isfinite(reference).all() and torch.isfinite(candidate).all(), key
            # PyTorch scales rtol by its SECOND argument: the frozen reference.
            assert torch.allclose(candidate, reference, rtol=.01, atol=1e-5), key
            if bitwise:
                assert torch.equal(reference, candidate), key
            if key == "loss":
                assert f"{reference.item():.3f}" == f"{candidate.item():.3f}"
        else:
            assert candidate == reference, key


def _nf4_trajectory_comparison(tmp_path, device):
    pytest.importorskip("bitsandbytes")
    import torch

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes
    from tests.test_expert_stream import _shard, _tiny_deepseek_v3
    from tests.test_stream_expert_cache_accuracy import _adapter_values, _configure, _trajectory

    weights, resident = _tiny_deepseek_v3(tmp_path)
    del resident  # The control is the same frozen NF4 base, not unquantised weights.
    skeleton = build_meta_skeleton(weights, dtype="float32", quant="nf4")
    suffixes = quantised_layer_suffixes(skeleton)
    del skeleton
    index = _shard(
        tmp_path, weights, quant="nf4", expert_layout="per_expert",
        quant_suffixes=suffixes, quant_device=device,
    )
    shards = str(tmp_path / "shards")
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(29)).to(device)
    batches = (ids[:1].clone(), ids[1:].clone())
    reference, baseline = _build(weights, index, shards, two_read=False, device=device)
    try:
        initial = _adapter_values(reference)
        expected = _trajectory(reference, batches)
    finally:
        baseline.close()
    del reference
    for two_read in (False, True):
        for arm in ("none", "host", "device", "both"):
            model, runtime = _build(weights, index, shards, two_read=two_read, device=device)
            try:
                _gate(initial, _adapter_values(model), bitwise=True)
                _configure(runtime, arm)
                _gate(expected, _trajectory(model, batches), bitwise=device == "cpu")
                stats = runtime.expert_fetcher.cache_stats()
                if arm != "none":
                    assert stats["storage_reads"] < stats["logical_fetches"]
            finally:
                runtime.close()


def test_real_cpu_nf4_expert_cache_preserves_complete_optimizer_trajectories(tmp_path):
    _nf4_trajectory_comparison(tmp_path, "cpu")


@pytest.mark.gpu(reason="Real CUDA NF4 kernels and device-cache parity require a GPU")
def test_real_cuda_nf4_expert_cache_preserves_complete_optimizer_trajectories(tmp_path):
    _nf4_trajectory_comparison(tmp_path, "cuda")
