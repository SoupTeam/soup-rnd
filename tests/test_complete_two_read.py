"""Complete lever: two reads of a streamed routed expert per step instead of three (CPU).

A tiny DeepSeek-V3 is streamed per expert through ``build_streamed_model``. Three models read the
same shards with the same adapters: the resident eager model, the default streamed path (reads
each hit expert in the forward, in the checkpoint recompute and in the backward) and the streamed
path with ``two_read_experts=True`` (no read in the recompute). Loss and every LoRA gradient are
compared with ``torch.equal``. Reads come from ``runtime.expert_pool.fetches``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from tests.test_b2_deepseek_v3_stream import (
    _copy_lora,
    _lora,
    _lora_params,
    _randomise_lora_b,
    _save_fixture,
    _tiny_deepseek_v3_config,
)

PATHS = ("resident", "default", "two-reads")


def _build(tmp_path: Path, *, buffers: int = 2, refill_before_backward: bool = False) -> Any:
    """``({path: (model, runtime or None)}, input ids)`` with equal adapters on every path."""
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.peft_wiring import attach_adapter

    config = _tiny_deepseek_v3_config()
    weights = tmp_path / "weights"
    _save_fixture(weights, config)
    stream_config = load_stream_config(str(weights), trust_remote_code=False)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(
        str(weights), shards, dtype="float32", arch=stream_arch_of(stream_config),
        expert_layout="per_expert",
    )
    lora = _lora(config)
    nets: Dict[str, Any] = {}
    for path, two_read_experts in (("default", False), ("two-reads", True)):
        model, runtime = build_streamed_model(
            model_id=str(weights), shard_dir=shards, index=index, lora_config=lora,
            device="cpu", dtype="float32", buffers=buffers, pin=False, seed=71,
            refill_before_backward=refill_before_backward, two_read_experts=two_read_experts,
        )
        nets[path] = (model, runtime)
    base = AutoModelForCausalLM.from_pretrained(
        str(weights), config=stream_config, dtype=torch.float32, experts_implementation="eager"
    )
    resident = attach_adapter(base, lora)
    _randomise_lora_b(nets["default"][0])
    _copy_lora(nets["default"][0], nets["two-reads"][0])
    _copy_lora(nets["default"][0], resident)
    nets["resident"] = (resident, None)
    for model, _ in nets.values():
        model.eval()
    ids = torch.randint(0, config.vocab_size, (3, 12), generator=torch.Generator().manual_seed(29))
    return nets, ids


def _close(nets: Any) -> None:
    for _, runtime in nets.values():
        if runtime is not None:
            runtime.close()


def _fetches(runtime: Any) -> Any:
    return runtime.expert_pool.fetches if runtime is not None else None


def _grads(net: Any) -> Dict[str, Any]:
    return {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in _lora_params(net).items()
    }


@pytest.fixture(scope="module")
def nets(tmp_path_factory: pytest.TempPathFactory) -> Any:
    built = _build(tmp_path_factory.mktemp("complete_two_read"))
    yield built
    _close(built[0])


def test_one_step_equals_resident_and_default_and_reads_each_hit_expert_twice(nets: Any) -> None:
    """Loss and every LoRA gradient are bit-equal on all three paths. Per step the default path
    reads each forward-visited expert three times and two reads read it twice."""
    models, ids = nets
    results, forward_reads, step_reads = {}, {}, {}
    for path in PATHS:
        net, runtime = models[path]
        net.zero_grad(set_to_none=True)
        before = _fetches(runtime)
        loss = net(input_ids=ids, labels=ids).loss
        forward_reads[path] = None if runtime is None else _fetches(runtime) - before
        loss.backward()
        step_reads[path] = None if runtime is None else _fetches(runtime) - before
        results[path] = (loss.detach(), _grads(net))
    assert forward_reads["default"] == forward_reads["two-reads"] > 0
    assert step_reads["default"] == 3 * forward_reads["default"]
    assert step_reads["two-reads"] == 2 * forward_reads["two-reads"]
    want_loss, want = results["resident"]
    assert want and all(grad is not None for grad in want.values())
    for path in ("default", "two-reads"):
        loss, got = results[path]
        assert torch.equal(loss, want_loss), path
        assert set(got) == set(want)
        assert all(torch.equal(got[name], want[name]) for name in want), path


def _stack() -> list:
    from soup_cli.utils import expert_stream

    return list(getattr(expert_stream._SCOPES, "stack", None) or [])


def _moe(net: Any, idx: int) -> Any:
    return net.get_submodule("base_model.model.model.layers")[idx].inner.mlp


def _step(net: Any, ids: Any) -> Any:
    net.zero_grad(set_to_none=True)
    loss = net(input_ids=ids, labels=ids).loss
    loss.backward()
    return loss.detach(), _grads(net)


def _assert_grads_equal(got: Dict[str, Any], want: Dict[str, Any]) -> None:
    assert got and set(got) == set(want)
    assert all(torch.equal(got[name], want[name]) for name in want)


@pytest.mark.parametrize("micro_batches", [2, 3])
def test_accumulated_micro_batches_equal_resident_and_default(
    nets: Any, micro_batches: int
) -> None:
    """Forward and backward per micro-batch with gradients accumulating in ``.grad``: every
    loss and the accumulated gradients are bit-equal on all paths, each micro-batch reads
    twice (default: three times) what its forward read, and no scope is left on the stack."""
    models, ids = nets
    results = {}
    for path in PATHS:
        net, runtime = models[path]
        net.zero_grad(set_to_none=True)
        losses = []
        for i in range(micro_batches):
            batch = ids[i : i + 1]
            before = _fetches(runtime)
            loss = net(input_ids=batch, labels=batch).loss
            forward = None if runtime is None else _fetches(runtime) - before
            loss.backward()
            losses.append(loss.detach())
            del loss
            assert _stack() == []
            if runtime is not None:
                per_read = 2 if path == "two-reads" else 3
                assert _fetches(runtime) - before == per_read * forward > 0
        results[path] = (losses, _grads(net))
    want_losses, want = results["resident"]
    for path in ("default", "two-reads"):
        losses, got = results[path]
        assert all(torch.equal(a, b) for a, b in zip(losses, want_losses)), path
        _assert_grads_equal(got, want)


def test_two_forwards_then_one_backward_equal_resident_and_default(
    tmp_path: Path,
) -> None:
    """Each forward's checkpoint call keeps its own record; one backward recomputes both and
    replays each from its own decisions. A layer pool that holds every layer and a refill before
    backward are what a graph that outlives the next forward needs."""
    models, ids = _build(tmp_path, buffers=4, refill_before_backward=True)
    try:
        results = {}
        for path in PATHS:
            net, runtime = models[path]
            net.zero_grad(set_to_none=True)
            before = _fetches(runtime)
            first = net(input_ids=ids[:1], labels=ids[:1]).loss
            second = net(input_ids=ids[1:], labels=ids[1:]).loss
            forward = None if runtime is None else _fetches(runtime) - before
            assert _stack() == []
            (first + second).backward()
            losses = (first.detach(), second.detach())
            del first, second
            assert _stack() == []
            if runtime is not None:
                per_read = 2 if path == "two-reads" else 3
                assert _fetches(runtime) - before == per_read * forward
            results[path] = (losses, _grads(net))
        for path in ("default", "two-reads"):
            assert all(
                torch.equal(a, b) for a, b in zip(results[path][0], results["resident"][0])
            )
            _assert_grads_equal(results[path][1], results["resident"][1])
    finally:
        _close(models)


def test_a_forward_hook_on_the_moe_block_is_refused_before_any_gradient(nets: Any) -> None:
    """A hook added after build on a module whose output the recompute changes: the forward
    raises before the layer runs, no gradient is written, the hook never ran, and after the
    hook is removed the next step equals the resident one."""
    models, ids = nets
    net, runtime = models["two-reads"]
    _, want = _step(models["resident"][0], ids)
    net.zero_grad(set_to_none=True)
    calls = []
    handle = _moe(net, 1).register_forward_hook(lambda _m, _a, out: calls.append(1))
    before = _fetches(runtime)
    try:
        with pytest.raises(RuntimeError, match="two-read expert streaming refuses decoder layer"):
            net(input_ids=ids, labels=ids)
    finally:
        handle.remove()
    assert calls == [] and _grads(net) == {name: None for name in _grads(net)}
    assert _stack() == [] and _fetches(runtime) == before
    _, got = _step(net, ids)
    _assert_grads_equal(got, want)


def test_a_hook_added_between_forward_and_backward_is_refused_in_the_recompute(
    nets: Any,
) -> None:
    models, ids = nets
    net, _ = models["two-reads"]
    _, want = _step(models["resident"][0], ids)
    net.zero_grad(set_to_none=True)
    scale = torch.ones((), requires_grad=True)
    loss = net(input_ids=ids, labels=ids).loss
    calls = []

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        calls.append(1)
        return output * scale

    handle = _moe(net, 2).register_forward_hook(hook)
    try:
        with pytest.raises(RuntimeError, match="the MoE block"):
            loss.backward()
    finally:
        handle.remove()
    assert calls == [] and scale.grad is None and _stack() == []
    del loss
    _, got = _step(net, ids)
    _assert_grads_equal(got, want)


def test_a_global_forward_hook_is_refused(nets: Any) -> None:
    models, ids = nets
    net, _ = models["two-reads"]
    handle = torch.nn.modules.module.register_module_forward_hook(lambda *_: None)
    try:
        with pytest.raises(RuntimeError, match="a global forward hook"):
            net(input_ids=ids, labels=ids)
    finally:
        handle.remove()
    assert _stack() == []


def test_layers_that_stream_recompute_switches_off_run_without_the_context_and_stay_exact(
    nets: Any,
) -> None:
    """Without the checkpoint there is no recompute to skip: a retained layer reads each expert
    in the forward and the backward, and the result still equals resident."""
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute

    models, ids = nets
    net, runtime = models["two-reads"]
    _, want = _step(models["resident"][0], ids)
    set_stream_recompute(net, False)
    try:
        net.zero_grad(set_to_none=True)
        before = _fetches(runtime)
        loss = net(input_ids=ids, labels=ids).loss
        forward = _fetches(runtime) - before
        loss.backward()
        assert _fetches(runtime) - before == 2 * forward
        _assert_grads_equal(_grads(net), want)
    finally:
        set_stream_recompute(net, True)
    assert _stack() == []


def test_two_reads_need_a_per_expert_cache(tmp_path: Path) -> None:
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    config = _tiny_deepseek_v3_config()
    weights = tmp_path / "weights"
    _save_fixture(weights, config)
    arch = stream_arch_of(load_stream_config(str(weights), trust_remote_code=False))
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(str(weights), shards, dtype="float32", arch=arch)
    assert index.expert_layout != "per_expert"
    with pytest.raises(ValueError, match="two_read_experts needs a cache sharded"):
        build_streamed_model(
            model_id=str(weights), shard_dir=shards, index=index, lora_config=_lora(config),
            device="cpu", dtype="float32", pin=False, two_read_experts=True,
        )


def test_grouped_routes_and_two_reads_together_equal_the_default_path(tmp_path: Path) -> None:
    """The comparison harness switches both levers through ``build_streamed_model``."""
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    models, ids = _build(tmp_path)
    try:
        weights = tmp_path / "weights"
        index_dir = str(tmp_path / "shards")
        from soup_cli.utils.layer_shard import read_shard_index

        grouped, runtime = build_streamed_model(
            model_id=str(weights), shard_dir=index_dir, index=read_shard_index(index_dir),
            lora_config=_lora(_tiny_deepseek_v3_config()), device="cpu", dtype="float32",
            pin=False, seed=71, two_read_experts=True, grouped_expert_routes=True,
        )
        try:
            experts = _moe(grouped, 1).experts
            assert experts.grouped_expert_routes and experts.two_reads
            assert not _moe(models["default"][0], 1).experts.grouped_expert_routes
            _copy_lora(models["default"][0], grouped)
            grouped.eval()
            want_loss, want = _step(models["default"][0], ids)
            before = runtime.expert_pool.fetches
            loss, got = _step(grouped, ids)
            assert torch.equal(loss, want_loss)
            _assert_grads_equal(got, want)
            assert runtime.expert_pool.fetches - before > 0
        finally:
            runtime.close()
    finally:
        _close(models)


@pytest.mark.parametrize("lever", ["two_read_experts", "grouped_expert_routes"])
def test_every_lever_needs_a_per_expert_cache_instead_of_being_ignored(
    tmp_path: Path, lever: str
) -> None:
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    config = _tiny_deepseek_v3_config()
    weights = tmp_path / "weights"
    _save_fixture(weights, config)
    arch = stream_arch_of(load_stream_config(str(weights), trust_remote_code=False))
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(str(weights), shards, dtype="float32", arch=arch)
    with pytest.raises(ValueError, match=f"{lever} needs a cache sharded"):
        build_streamed_model(
            model_id=str(weights), shard_dir=shards, index=index, lora_config=_lora(config),
            device="cpu", dtype="float32", pin=False, **{lever: True},
        )


def test_bf16_autocast_steps_are_bit_equal_between_default_and_two_reads(nets: Any) -> None:
    """fp32 weights under CPU bf16 autocast, the mixed-precision shape of a real recipe: the
    replayed zero never reaches a gradient, so two reads match the default path bit for bit."""
    models, ids = nets
    results = {}
    for path in ("default", "two-reads"):
        net, runtime = models[path]
        net.zero_grad(set_to_none=True)
        before = _fetches(runtime)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = net(input_ids=ids, labels=ids).loss
        forward = _fetches(runtime) - before
        loss.backward()
        per_read = 2 if path == "two-reads" else 3
        assert _fetches(runtime) - before == per_read * forward > 0
        results[path] = (loss.detach(), _grads(net))
    assert torch.equal(results["default"][0], results["two-reads"][0])
    _assert_grads_equal(results["two-reads"][1], results["default"][1])
