"""Shared-cap refusals and exact training with retained activations and weights."""

from types import SimpleNamespace

import pytest

from soup_cli.utils.stream_cache_recompute import configure_cached_stream_recompute


class Memory:
    def total(self):
        return 10000

    def free(self):
        return 10000

    def reserved(self):
        return 0

    def synchronize(self):
        pass

    def release(self):
        pass


def instruments(*, peak=7000, error=None, count=2):
    import torch

    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.stream_cache_fixture import fixture

    model, runtime = fixture()
    seen = []

    def planner(current, **kwargs):
        seen.append(("plan", kwargs))
        set_stream_recompute(current, False, retain=count)
        with torch.no_grad():
            next(param for param in current.parameters() if param.requires_grad).add_(1)
        torch.rand(4)
        return count

    def measure(current, **kwargs):
        seen.append(("measure", kwargs))
        if error is not None:
            raise error
        runtime.prefetcher.prime()
        result = current(torch.rand(2, 4, requires_grad=True))
        result.square().sum().backward()
        current[0].training = not current[0].training
        return SimpleNamespace(peak_bytes=peak, reserved_bytes=peak, failed=False, oom=False)

    return model, runtime, seen, planner, measure


def configure(model, runtime, planner, measure, **kwargs):
    return configure_cached_stream_recompute(
        model, runtime, rows=2, seq_len=128, vocab_size=64,
        memory=Memory(), planner=planner, measure=measure,
        reserve_bytes=100, optimizer_reserve_bytes=300, **kwargs,
    )


def kept(model):
    from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

    return {layer.idx for layer in streamed_decoder_layers(model) if not layer.recompute}


def test_shared_cap_includes_resident_peak_optimizer_and_setup_and_preserves_state():
    import torch

    model, runtime, seen, planner, measure = instruments()
    model[0].eval()
    rng = torch.get_rng_state().clone()
    parameters = [param for param in model.parameters() if param.requires_grad]
    values = [param.detach().clone() for param in parameters]
    modes = [module.training for module in model.modules()]
    result = configure(model, runtime, planner, measure)
    assert result.retained_layers == 2
    assert result.cap_bytes == 8500
    assert seen[0][1]["override_bytes"] == 8100
    assert seen[1][1] == dict(rows=2, seq_len=128, vocab_size=64, device="cpu")
    assert kept(model) == {4, 5}
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(param, value) for param, value in zip(parameters, values))
    assert all(param.grad is None for param in model.parameters())
    assert [module.training for module in model.modules()] == modes


@pytest.mark.parametrize("peak, accepted", [(8100, True), (8101, False)])
def test_full_candidate_peak_not_single_layer_estimate_decides(peak, accepted):
    model, runtime, _seen, planner, measure = instruments(peak=peak)
    result = configure(model, runtime, planner, measure)
    assert result.retained_layers == (2 if accepted else 0)
    assert bool(kept(model)) is accepted


def test_override_is_shared_instead_of_spending_the_same_headroom_twice():
    model, runtime, seen, planner, measure = instruments(peak=7000)
    result = configure(model, runtime, planner, measure, override_bytes=7200)
    assert result.cap_bytes == 7200
    assert seen[0][1]["override_bytes"] == 6800
    assert result.retained_layers == 0
    assert kept(model) == set()


def test_zero_native_selection_does_not_run_a_second_probe():
    model, runtime, seen, planner, measure = instruments(count=0)
    result = configure(model, runtime, planner, measure)
    assert result.retained_layers == 0
    assert len(seen) == 1
    assert kept(model) == set()


def test_lazy_expert_cache_ceiling_is_held_back_from_activation_retention():
    model, runtime, seen, planner, _ = instruments()
    runtime.pool.is_cuda = True  # Synthetic memory reader; no CUDA execution.
    runtime.expert_fetcher = SimpleNamespace(
        cache_stats=lambda: {"device": {"budget_bytes": 1000, "payload_bytes": 0}},
    )
    measure = lambda *args, **kwargs: SimpleNamespace(  # noqa: E731
        peak_bytes=7000, reserved_bytes=7000, failed=False, oom=False,
    )
    result = configure(model, runtime, planner, measure, override_bytes=8000)
    assert seen[0][1]["override_bytes"] == 6600
    assert result.reserve_bytes == 1100 and result.retained_layers == 0
    assert kept(model) == set()


def test_oom_restores_recompute_and_other_errors_propagate():
    import torch

    for error in (torch.OutOfMemoryError("synthetic capacity"), ValueError("broken kernel")):
        model, runtime, _seen, planner, measure = instruments(error=error)
        if isinstance(error, torch.OutOfMemoryError):
            assert configure(model, runtime, planner, measure).retained_layers == 0
        else:
            with pytest.raises(ValueError, match="broken kernel"):
                configure(model, runtime, planner, measure)
        assert kept(model) == set()
        assert all(param.grad is None for param in model.parameters())


@pytest.mark.parametrize("resident", [(), (2,), (2, 3), tuple(range(6))])
@pytest.mark.parametrize("retained", [1, 3, 6])
def test_combined_four_update_trajectory_is_bit_exact(resident, retained):
    import torch

    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.stream_cache_fixture import fixture, trajectory

    reference, ordinary = fixture()
    candidate, runtime = fixture()
    runtime.pool.reserve_layers(resident, runtime.source)
    set_stream_recompute(candidate, False, retain=retained)
    expected = trajectory(reference, ordinary)
    actual = trajectory(candidate, runtime)
    assert len(actual) == len(expected)
    assert all(torch.equal(first, second) for first, second in zip(actual, expected))


@pytest.mark.parametrize("cached, cuda, calls", [(False, True, 0), (True, False, 0),
                                               (True, True, 1)])
def test_disk_setup_only_probes_retention_when_an_actual_cache_exists(
    monkeypatch, cached, cuda, calls,
):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    seen = []
    monkeypatch.setattr(
        "soup_cli.utils.stream_cache_recompute.configure_cached_stream_recompute",
        lambda *args, **kwargs: seen.append(kwargs),
    )
    wrapper = StreamingSetupMixin()
    wrapper._stream_runtime = SimpleNamespace(
        stats=lambda: {"ram_cache_indices": [1] if cached else [], "resident_layer_indices": []}
    )
    wrapper._configure_stream_recompute(
        None, SimpleNamespace(stream_recompute="auto", stream_vram_override=7000),
        tier="disk", on_cuda=cuda, task="sft", step_shape=(1, 512, 64),
    )
    assert len(seen) == calls
    if seen:
        assert seen[0]["override_bytes"] == 7000
        assert seen[0]["rows"] == 1 and seen[0]["seq_len"] == 512


@pytest.mark.parametrize("resident", [(), (1,)])
@pytest.mark.parametrize("retained", [1, 4])
def test_real_nf4_disk_ram_cache_and_retention_match_all_gradients(
    tmp_path, resident, retained, monkeypatch,
):
    pytest.importorskip("bitsandbytes")
    from soup_cli.utils.layer_stream_runtime import (
        build_streamed_model,
        clear_stream_substitution_cache,
        set_stream_recompute,
    )
    from tests.test_stream_recompute import _assert_same, _ids, _step
    from tests.test_v07202 import _nf4_stream, _randomise_lora_b, _tiny_lora

    unused, initial, weights, index, shards = _nf4_stream(tmp_path, n_layers=4)
    initial.close()
    del unused
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=_tiny_lora(),
        device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
        quant="nf4", tier="disk", read_ahead=2,
    )
    try:
        _randomise_lora_b(model)
        expected = _step(model, _ids())
        clear_stream_substitution_cache(model)
        # A tiny positive cache control must not depend on other desktop apps.
        # Actual host/cgroup refusal remains covered by the admission tests.
        monkeypatch.setattr(
            "soup_cli.utils.stream_ram_cache.available_memory",
            lambda: {"available_bytes": 4 * 2**30, "constraints": []},
        )
        runtime.source.configure_ram_cache(4 * 2**20)
        runtime.pool.reserve_layers(resident, runtime.source)
        runtime.source.set_resident_layers(resident)
        stats = runtime.source.ram_cache_stats()
        assert stats["indices"]
        assert not set(stats["indices"]).intersection(resident)
        set_stream_recompute(model, False, retain=retained)
        for _ in range(3):
            _assert_same(expected, _step(model, _ids()))
    finally:
        runtime.close()
