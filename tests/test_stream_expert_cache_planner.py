"""Whole-model expert-cache admission, lazy ceilings and setup-state preservation."""

from types import SimpleNamespace

import pytest

from soup_cli.utils.stream_expert_cache import (
    configure_stream_expert_cache,
    expert_device_reserve_bytes,
)

MIB = 2**20


class FakeMemory:
    def __init__(self, *, total=10000, free=3000, reserved=0):
        self.total_bytes = total
        self.free_bytes = free
        self.reserved_bytes = reserved
        self.synchronizations = 0
        self.releases = 0

    def total(self):
        return self.total_bytes

    def free(self):
        return self.free_bytes

    def reserved(self):
        return self.reserved_bytes

    def synchronize(self):
        self.synchronizations += 1

    def release(self):
        self.releases += 1


class FakeFetcher:
    def __init__(self):
        self.configurations = []
        self.configure_cache()

    def configure_cache(
        self, *, host_cache_bytes=0, device_cache_bytes=0, host_cache_policy="lru",
    ):
        self.host_budget = host_cache_bytes
        self.device_budget = device_cache_bytes
        self.host_policy = host_cache_policy
        self.host_payload = self.device_payload = 0
        self.configurations.append((host_cache_bytes, device_cache_bytes))

    def cache_stats(self):
        return {
            "host": {"budget_bytes": self.host_budget, "payload_bytes": self.host_payload},
            "device": {"budget_bytes": self.device_budget, "payload_bytes": self.device_payload},
            "host_staging_bytes": 99,
        }

    def clear_cache(self):
        self.host_payload = self.device_payload = 0


def fixture(*, is_cuda=True, warm_bytes=200):
    import torch

    from soup_cli.utils.layer_stream_runtime import StepPeak

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.2))
    fetcher = FakeFetcher()
    source = SimpleNamespace(hotset=object(), configure_ram_cache=lambda *args: None)
    runtime = SimpleNamespace(
        pool=SimpleNamespace(is_cuda=is_cuda, device="cpu"),
        expert_fetcher=fetcher, expert_readers=(object(),), source=source,
    )
    memory = FakeMemory()
    calls = []

    def measure(model, **shape):
        calls.append((fetcher.host_budget, fetcher.device_budget, dict(shape)))
        # Both probes do a real backward and exercise restoration of RNG,
        # adapter values, mixed module modes and freshly produced gradients.
        model(torch.rand(2, 2)).sum().backward()
        with torch.no_grad():
            next(model.parameters()).add_(7)
        model[0].training = not model[0].training
        fetcher.host_payload = min(fetcher.host_budget, warm_bytes)
        fetcher.device_payload = min(fetcher.device_budget, warm_bytes)
        peak = 1000 + fetcher.device_payload
        return StepPeak(
            peak_bytes=peak, reserved_bytes=peak, seconds=1,
            rows=shape["rows"], seq_len=shape["seq_len"],
        )

    return model, runtime, memory, calls, measure


def configure(model, runtime, memory, measure, **kwargs):
    controls = {
        "host_cache_bytes": 300, "device_cache_bytes": 500,
        "rows": 2, "seq_len": 128, "vocab_size": 10,
        "memory": memory, "measure": measure, "reserve_bytes": 100,
        "optimizer_reserve_bytes": 300, "host_available": 10 * 2**30,
    }
    return configure_stream_expert_cache(model, runtime, **(controls | kwargs))


def assert_disabled(runtime):
    stats = runtime.expert_fetcher.cache_stats()
    assert stats["host"] == {"budget_bytes": 0, "payload_bytes": 0}
    assert stats["device"] == {"budget_bytes": 0, "payload_bytes": 0}


def test_disabled_does_not_need_model_or_runtime():
    plan = configure_stream_expert_cache(None, None, rows=0, seq_len=0, vocab_size=0)
    assert plan.reason == "disabled"
    assert not plan.enabled


def test_complete_baseline_and_candidate_preserve_existing_source_and_staging():
    model, runtime, memory, calls, measure = fixture()
    source_hotset = runtime.source.hotset
    plan = configure(model, runtime, memory, measure)
    assert [(host, device) for host, device, _ in calls] == [(0, 0), (300, 500)]
    assert all(shape == {"rows": 2, "seq_len": 128, "vocab_size": 10, "device": "cpu"}
               for _, _, shape in calls)
    assert runtime.source.hotset is source_hotset
    assert runtime.expert_fetcher.cache_stats()["host_staging_bytes"] == 99
    assert plan.enabled
    assert (plan.host_cache_bytes, plan.device_cache_bytes) == (300, 500)
    assert plan.host_reserve_bytes == 800  # Both synthetic CPU tiers consume host RAM.
    assert runtime.source._expert_host_cache_reserve_bytes == 800
    assert (plan.baseline_peak_reserved_bytes, plan.candidate_peak_reserved_bytes) == (1000, 1200)
    assert plan.cap_bytes == 3000
    assert plan.device_available_bytes == 1400
    assert plan.reserve_bytes == 100
    assert plan.optimizer_reserve_bytes == 300
    assert plan.setup_seconds >= 0


def test_static_policy_is_explicitly_admitted_and_recorded():
    model, runtime, memory, _, measure = fixture()
    plan = configure(model, runtime, memory, measure, host_cache_policy="static")
    assert plan.host_cache_policy == runtime.expert_fetcher.host_policy == "static"
    assert plan.candidate_host_available_bytes == 10 * 2**30
    assert runtime.expert_fetcher.host_budget == 300
    assert runtime.expert_fetcher.device_budget == 500
    assert runtime.expert_fetcher.host_payload == runtime.expert_fetcher.device_payload == 0
    assert plan.candidate_peak_allocated_bytes == 1200  # Measured before the clear.


@pytest.mark.parametrize("policy", [None, "", "fifo", True, ["static"]])
def test_invalid_policy_is_refused_even_with_disabled_budgets(policy):
    with pytest.raises(ValueError, match="host_cache_policy"):
        configure_stream_expert_cache(
            None, None, rows=0, seq_len=0, vocab_size=0, host_cache_policy=policy,
        )


def test_host_capacity_is_rechecked_after_candidate_allocations(monkeypatch):
    import soup_cli.utils.stream_expert_cache as planner

    model, runtime, memory, _, measure = fixture()
    samples = iter((10 * 2**30, 100))
    monkeypatch.setattr(planner, "available_memory", lambda **_: {"available_bytes": next(samples)})
    with pytest.raises(ValueError, match="live RAM admission"):
        configure(model, runtime, memory, measure, host_available=None, host_cache_policy="static")
    assert_disabled(runtime)
    assert runtime.expert_fetcher.host_policy == "lru"


def test_probe_restores_rng_adapter_values_gradients_and_mixed_modes():
    import torch

    model, runtime, memory, _, measure = fixture()
    model[1].eval()
    values = [param.detach().clone() for param in model.parameters()]
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    configure(model, runtime, memory, measure)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, param) for value, param in zip(values, model.parameters()))
    assert all(param.grad is None for param in model.parameters())
    assert [module.training for module in model.modules()] == modes


def test_default_optimizer_reserve_counts_trainable_parameters_only():
    model, runtime, memory, _, measure = fixture()
    model[0].bias.requires_grad_(False)
    plan = configure(model, runtime, memory, measure, optimizer_reserve_bytes=None)
    assert plan.optimizer_reserve_bytes == 12 * model[0].weight.numel()


def test_preexisting_gradients_are_refused_without_destroying_them():
    import torch

    model, runtime, memory, calls, measure = fixture()
    param = next(model.parameters())
    param.grad = torch.ones_like(param)
    original = param.grad
    with pytest.raises(ValueError, match="pristine gradients"):
        configure(model, runtime, memory, measure)
    assert param.grad is original
    assert calls == []
    assert_disabled(runtime)


@pytest.mark.parametrize("kind", ["budget", "payload"])
def test_preconfigured_caches_are_refused_and_preserved(kind):
    model, runtime, memory, calls, measure = fixture()
    if kind == "budget":
        runtime.expert_fetcher.configure_cache(host_cache_bytes=5)
    else:
        runtime.expert_fetcher.device_payload = 5
    original = runtime.expert_fetcher.cache_stats()
    with pytest.raises(ValueError, match="disabled, empty"):
        configure(model, runtime, memory, measure)
    assert runtime.expert_fetcher.cache_stats() == original
    assert calls == []


@pytest.mark.parametrize("missing", ["memory", "measure", "both"])
def test_cpu_controls_require_both_explicit_instruments(missing):
    model, runtime, memory, calls, measure = fixture(is_cuda=False)
    with pytest.raises(ValueError, match="CUDA pool"):
        configure(
            model, runtime, None if missing in {"memory", "both"} else memory,
            None if missing in {"measure", "both"} else measure,
        )
    assert calls == []


def test_baseline_cap_refuses_before_enabling_caches():
    model, runtime, memory, calls, measure = fixture()
    with pytest.raises(ValueError, match="baseline exceeds"):
        configure(model, runtime, memory, measure, override_bytes=1300)
    assert len(calls) == 1
    assert_disabled(runtime)


def test_cold_cache_charges_full_requested_ceiling_before_enabling():
    model, runtime, memory, calls, measure = fixture(warm_bytes=0)
    with pytest.raises(ValueError, match="full expert device cache ceiling"):
        configure(model, runtime, memory, measure, device_cache_bytes=1601)
    assert len(calls) == 1
    assert_disabled(runtime)


@pytest.mark.parametrize("warm_bytes", [0, 200])
def test_candidate_gate_charges_the_full_ceiling_even_when_warm(warm_bytes):
    model, runtime, memory, calls, good = fixture(warm_bytes=warm_bytes)
    old_reserve = 71
    runtime.source._expert_host_cache_reserve_bytes = old_reserve

    def measure(model, **shape):
        peak = good(model, **shape)
        if len(calls) == 2:
            # The 2300-byte measured peak fits with 400 reserves, but the
            # still-lazy 500 ceiling makes 3200 and must be held back in full.
            return type(peak)(2300, 2300, 1, shape["rows"], shape["seq_len"])
        return peak

    with pytest.raises(ValueError, match="candidate and full device ceiling"):
        configure(model, runtime, memory, measure)
    assert len(calls) == 2
    assert_disabled(runtime)
    assert runtime.source._expert_host_cache_reserve_bytes == old_reserve
    assert memory.releases >= 1


def test_host_admission_uses_available_ram_after_the_baseline(monkeypatch):
    import soup_cli.utils.stream_expert_cache as planner

    model, runtime, memory, calls, measure = fixture()
    sampled = []

    def available_memory(**kwargs):
        sampled.append((len(calls), runtime.expert_fetcher.device_budget, kwargs))
        return {"available_bytes": 512 * MIB}

    monkeypatch.setattr(planner, "available_memory", available_memory)
    with pytest.raises(ValueError, match="available RAM admission"):
        configure(model, runtime, memory, measure)
    assert sampled == [(1, 0, {"host_available": 10 * 2**30})]
    assert_disabled(runtime)
    assert not hasattr(runtime.source, "_expert_host_cache_reserve_bytes")


def test_cpu_dense_cache_and_host_cache_share_the_host_admission():
    model, runtime, _, calls, measure = fixture(is_cuda=False)
    memory = FakeMemory(total=10 * 2**30, free=10 * 2**30)
    # Each 400 MiB ceiling individually fits the RAM budget (about 778 MiB),
    # but together both CPU-owned tiers exceed it.
    with pytest.raises(ValueError, match="available RAM admission"):
        configure(
            model, runtime, memory, measure, host_cache_bytes=400 * MIB,
            device_cache_bytes=400 * MIB, host_available=1844 * MIB,
        )
    assert len(calls) == 1
    assert_disabled(runtime)


def test_candidate_gate_resamples_live_capacity_after_external_allocations():
    model, runtime, memory, calls, good = fixture()

    def measure(model, **shape):
        peak = good(model, **shape)
        if len(calls) == 2:
            memory.free_bytes = 1900  # Required 2100 fits the old 3000, not this live cap.
        return peak

    with pytest.raises(ValueError, match="candidate and full device ceiling"):
        configure(model, runtime, memory, measure)
    assert len(calls) == 2
    assert_disabled(runtime)


def test_candidate_live_capacity_cannot_raise_the_original_cap():
    model, runtime, memory, calls, good = fixture()

    def measure(model, **shape):
        peak = good(model, **shape)
        if len(calls) == 2:
            memory.free_bytes = 9000
        return peak

    plan = configure(model, runtime, memory, measure)
    assert plan.cap_bytes == 3000


def test_original_probe_failure_survives_cleanup_synchronization_failure():
    import torch

    model, runtime, memory, _, good = fixture()

    def broken_sync():
        raise torch.OutOfMemoryError("cleanup out of memory")

    def measure(model, **shape):
        good(model, **shape)
        memory.synchronize = broken_sync
        raise RuntimeError("owner corruption")

    with pytest.raises(RuntimeError, match="owner corruption"):
        configure(model, runtime, memory, measure)
    assert_disabled(runtime)


@pytest.mark.parametrize("which", ["baseline", "candidate"])
@pytest.mark.parametrize("outcome", ["oom", "raised_oom", "none", "failed", "raised_error"])
def test_all_probe_failures_rollback_and_restore_state(which, outcome):
    import torch

    from soup_cli.utils.layer_stream_runtime import StepPeak

    model, runtime, memory, calls, good = fixture()
    model[1].eval()
    values = [param.detach().clone() for param in model.parameters()]
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()

    def measure(model, **shape):
        peak = good(model, **shape)
        if len(calls) != (1 if which == "baseline" else 2):
            return peak
        if outcome == "raised_oom":
            raise torch.OutOfMemoryError("out of memory")
        if outcome == "raised_error":
            raise RuntimeError("owner corruption")
        if outcome == "none":
            return None
        return StepPeak(
            0, 0, 1, shape["rows"], shape["seq_len"],
            oom=outcome == "oom", failed=outcome == "failed", error="AcceleratorError",
        )

    expected_type = RuntimeError if outcome in {"failed", "raised_error"} else ValueError
    with pytest.raises(expected_type):
        configure(model, runtime, memory, measure)
    assert_disabled(runtime)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, param) for value, param in zip(values, model.parameters()))
    assert all(param.grad is None for param in model.parameters())
    assert [module.training for module in model.modules()] == modes
    assert not hasattr(runtime.source, "_expert_host_cache_reserve_bytes")


@pytest.mark.parametrize("field", ["host_cache_bytes", "device_cache_bytes", "reserve_bytes",
                                   "optimizer_reserve_bytes"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_byte_counts_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        configure_stream_expert_cache(None, None, rows=0, seq_len=0, vocab_size=0,
                                     **{field: value})


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_host_available_is_refused(value):
    with pytest.raises(ValueError, match="host_available"):
        configure_stream_expert_cache(
            None, None, rows=0, seq_len=0, vocab_size=0, host_available=value,
        )


def test_shared_gpu_reserve_uses_ceiling_not_current_payload():
    _, runtime, _, _, _ = fixture()
    runtime.expert_fetcher.configure_cache(device_cache_bytes=900)
    assert runtime.expert_fetcher.device_payload == 0
    assert expert_device_reserve_bytes(runtime) == 900
    runtime.pool.is_cuda = False
    assert expert_device_reserve_bytes(runtime) == 0
    assert expert_device_reserve_bytes(None) == 0


def test_no_expert_layout_is_refused_before_probes():
    model, runtime, memory, calls, measure = fixture()
    runtime.expert_readers = ()
    with pytest.raises(ValueError, match="per-expert streaming"):
        configure(model, runtime, memory, measure)
    assert calls == []
