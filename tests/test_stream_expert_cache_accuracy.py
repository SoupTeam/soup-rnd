"""Expert caches preserve full training trajectories across the two autograd layouts.

CPU device-cache cells exercise storage and numerical semantics only. They make no
claim about CUDA memory accounting, transfer ordering or speed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.test_expert_stream import _paired, _three_way

_CACHE_BYTES = 1024 * 1024
_ARMS = ("none", "host", "device", "both")


def _configure(runtime: Any, arm: str, host_cache_policy: str = "lru") -> None:
    runtime.expert_fetcher.configure_cache(
        host_cache_bytes=_CACHE_BYTES if arm in {"host", "both"} else 0,
        device_cache_bytes=_CACHE_BYTES if arm in {"device", "both"} else 0,
        host_cache_policy=host_cache_policy,
    )


def _adapters(net: Any) -> dict[str, Any]:
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    return {
        name: parameter
        for name, parameter in canonical_named_parameters(net)
        if "lora_" in name
    }


def _adapter_values(net: Any) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in _adapters(net).items()}


def _capture(net: Any, ids: Any, *, autocast: bool) -> dict[Any, Any]:
    """Capture one accumulated microbatch, restoring all diagnostic instrumentation."""
    import torch

    captured: dict[Any, Any] = {}
    router_leaves: dict[int, list[Any]] = {}
    handles = []
    substitutions = []
    router_flags = []

    def keep(key: Any) -> Any:
        def hook(grad: Any) -> None:
            captured[key] = grad.detach().clone()

        return hook

    def on_input(idx: int) -> Any:
        def hook(_module: Any, args: Any) -> None:
            if args[0].requires_grad:
                args[0].register_hook(keep(("dX", idx)))

        return hook

    def on_router(idx: int) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            output[0].register_hook(keep(("router_logits", idx)))
            output[1].register_hook(keep(("routing_weights", idx)))

        return hook

    def private_router_weight(real: Any, leaves: list[Any]) -> Any:
        def substituted(buffers: Any) -> dict[str, Any]:
            weights = dict(real(buffers))
            leaf = weights["mlp.gate.weight"].detach().clone().requires_grad_(True)
            leaves.append(leaf)
            weights["mlp.gate.weight"] = leaf
            return weights

        return substituted

    for idx, layer in enumerate(net.get_submodule("base_model.model.model.layers")):
        handles.append(layer.register_forward_pre_hook(on_input(idx)))
        inner = getattr(layer, "inner", None)
        router = getattr((inner or layer).mlp, "gate", None)
        if router is None:
            continue
        handles.append(router.register_forward_hook(on_router(idx)))
        if inner is None:
            router_flags.append((router.weight, router.weight.requires_grad))
            router.weight.grad = None
            router.weight.requires_grad_(True)
            router_leaves[idx] = [router.weight]
        else:
            router_leaves[idx] = []
            original = layer._substituted_weights
            substitutions.append((layer, original))
            layer._substituted_weights = private_router_weight(original, router_leaves[idx])
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            output = net(input_ids=ids, labels=ids)
        # Two equal microbatches form each optimizer update.
        (output.loss / 2).backward()
    finally:
        for handle in handles:
            handle.remove()
        for layer, original in substitutions:
            layer._substituted_weights = original
        for weight, required_grad in router_flags:
            weight.requires_grad_(required_grad)
    captured["input_ids"] = ids.clone()
    captured["logits"] = output.logits.detach().clone()
    captured["loss"] = output.loss.detach().clone()
    for idx, leaves in router_leaves.items():
        grads = [leaf.grad for leaf in leaves if leaf.grad is not None]
        assert len(grads) == 1, (idx, len(grads))
        captured[("router_weight", idx)] = grads[0].detach().clone()
    for name, parameter in _adapters(net).items():
        assert parameter.requires_grad and parameter.grad is not None, name
        captured[("grad", name)] = parameter.grad.detach().clone()
    assert {key[0] for key in captured if isinstance(key, tuple)} == {
        "dX", "router_logits", "routing_weights", "router_weight", "grad",
    }
    for kind in ("routing_weights", "router_weight"):
        assert {key for key in captured if isinstance(key, tuple) and key[0] == kind} == {
            (kind, 1), (kind, 2),
        }
        assert captured[(kind, 1)].abs().sum().item() > 0
    return captured


def _trajectory(net: Any, batches: tuple[Any, Any], *, autocast: bool = False) -> dict[str, Any]:
    import torch

    adapters = _adapters(net)
    assert adapters and all(parameter.requires_grad for parameter in adapters.values())
    initial = _adapter_values(net)
    optimizer = torch.optim.AdamW(adapters.values(), lr=0.002, weight_decay=0.01)
    microbatches = []
    updates = []
    for update in range(2):
        net.zero_grad(set_to_none=True)
        # Reuse exactly the same token batches, reversing their order for update two.
        for ids in batches[::1 if update == 0 else -1]:
            microbatches.append(_capture(net, ids, autocast=autocast))
        optimizer.step()
        updates.append({
            "adapters": _adapter_values(net),
            "optimizer": {
                name: {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in optimizer.state[parameter].items()
                }
                for name, parameter in adapters.items()
            },
        })
    assert all(not torch.equal(parameter.detach(), initial[name])
               for name, parameter in adapters.items())
    return {"microbatches": microbatches, "updates": updates}


def _assert_equal(actual: Any, expected: Any, path: tuple[Any, ...] = ()) -> None:
    import torch

    if torch.is_tensor(expected):
        assert torch.is_tensor(actual), path
        if expected.is_floating_point():
            assert torch.isfinite(expected).all() and torch.isfinite(actual).all(), path
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-5, msg=str(path))
        assert torch.equal(actual, expected), path
        if path[-1:] == ("loss",):
            assert f"{actual.item():.3f}" == f"{expected.item():.3f}", path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            _assert_equal(actual[key], value, (*path, key))
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for idx, value in enumerate(expected):
            _assert_equal(actual[idx], value, (*path, idx))
    else:
        assert actual == expected, path


def _observe_visits(runtime: Any) -> list[int]:
    """Count logical fetch requests independently of the reader's physical counters."""
    count = [0]
    original = runtime.expert_fetcher.fetch

    def fetch(reader: Any, expert: int) -> Any:
        count[0] += 1
        return original(reader, expert)

    runtime.expert_fetcher.fetch = fetch
    return count


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("cell", ("fp32", "bf16", "bf16-autocast"))
@pytest.mark.parametrize("host_cache_policy", ("lru", "static"))
def test_expert_cache_preserves_accumulated_optimizer_trajectory(
    tmp_path: Path, arm: str, cell: str, host_cache_policy: str,
) -> None:
    """Every cache arm equals resident and three-read controls through complete updates."""
    ref, (default, default_runtime), (two, two_runtime), ids = _three_way(
        tmp_path, "bfloat16" if cell == "bf16" else "float32",
    )
    runtimes = (default_runtime, two_runtime)
    try:
        _configure(default_runtime, arm, host_cache_policy)
        _configure(two_runtime, arm, host_cache_policy)
        initial = _adapter_values(ref)
        _assert_equal(_adapter_values(default), initial)
        _assert_equal(_adapter_values(two), initial)
        batches = (ids[:1].clone(), ids[1:].clone())
        visits = [_observe_visits(runtime) for runtime in runtimes]
        expected = _trajectory(ref, batches, autocast=cell == "bf16-autocast")
        for net, runtime, count in zip((default, two), runtimes, visits):
            actual = _trajectory(net, batches, autocast=cell == "bf16-autocast")
            _assert_equal(actual, expected)
            stats = runtime.stats()
            cache = runtime.expert_fetcher.cache_stats()
            assert stats["expert_cache"] == cache
            assert cache["logical_fetches"] == count[0]
            reads = sum(reader.fetches for reader in runtime.expert_readers)
            assert reads == stats["expert_fetches"] == cache["storage_reads"] > 0
            assert sum(reader.bytes_read for reader in runtime.expert_readers) == (
                stats["expert_bytes_read"]
            ) == cache["storage_bytes_read"]
            for tier in ("host", "device"):
                assert 0 <= cache[tier]["payload_bytes"] <= cache[tier]["budget_bytes"]
            if arm == "none":
                assert reads == count[0]
                assert cache["host"]["hits"] == cache["device"]["hits"] == 0
            else:
                assert 0 < reads < count[0]
                hit_tier = "host" if arm == "host" else "device"
                assert cache[hit_tier]["hits"] > 0
                if arm == "both":
                    # Every tiny expert fits on the device; host entries must not duplicate it.
                    assert cache["host"]["entries"] == cache["host"]["payload_bytes"] == 0
        # Routing is equal; only the default checkpoint recompute requests experts again.
        assert visits[0][0] * 2 == visits[1][0] * 3
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.mark.parametrize("when", ("forward", "recompute"))
def test_warm_expert_caches_preserve_two_read_hook_refusal(tmp_path: Path, when: str) -> None:
    net, runtime, _ref, ids, _index = _paired(tmp_path, two_read_experts=True)
    try:
        _configure(runtime, "both")
        net(input_ids=ids, labels=ids).loss.backward()
        net.zero_grad(set_to_none=True)
        loss = net(input_ids=ids, labels=ids).loss if when == "recompute" else None
        calls = []
        mlp = net.get_submodule("base_model.model.model.layers")[2].inner.mlp
        handle = mlp.register_forward_hook(lambda _m, _a, _o: calls.append(True))
        try:
            with pytest.raises(RuntimeError, match="two-read expert streaming refuses decoder"):
                if when == "forward":
                    net(input_ids=ids, labels=ids)
                else:
                    loss.backward()
        finally:
            handle.remove()
        assert calls == []
    finally:
        runtime.close()


def test_warm_expert_caches_preserve_retained_and_recomputed_updates(tmp_path: Path) -> None:
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.test_expert_stream_two_read_contract import _build, _close

    nets, ids = _build(tmp_path, buffers=4, refill_before_backward=True)
    try:
        streamed = [nets[path] for path in ("default", "two-reads")]
        for _net, runtime in streamed:
            _configure(runtime, "both")
        visits = [_observe_visits(runtime) for _net, runtime in streamed]
        batches = (ids[:1].clone(), ids[1:2].clone())
        for recompute in (False, True):
            before = [count[0] for count in visits]
            expected = _trajectory(nets["resident"][0], batches)
            for net, _runtime in streamed:
                assert set_stream_recompute(net, recompute) == 3
                _assert_equal(_trajectory(net, batches), expected)
            delta = [count[0] - old for count, old in zip(visits, before)]
            if recompute:
                assert delta[0] * 2 == delta[1] * 3
            else:
                assert delta[0] == delta[1] > 0
    finally:
        _close(nets)
