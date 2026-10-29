"""B1.3: backward through streamed routed experts, exact against a resident eager reference.

Every expected value comes from the resident ``DeepseekV3Experts`` / ``DeepseekV3MoE`` of
transformers on the same weights (for NF4: the independently dequantised copy), run with
autograd on the same input, routing, dtype and upstream gradient. Comparison is
``torch.equal``; nothing here is a tolerance. Reloads are observed through the public
``reader.read_host`` / ``reader.fetches`` / ``pool.fetches`` boundary, and a backward load is
told from a forward load by ``torch._C._current_autograd_node()``, which is set only while
autograd's engine runs a node.

The ``check_*`` helpers and ``exact_backward_case`` are shared with
``test_b13_expert_backward_cuda.py``, which must set ``CUBLAS_WORKSPACE_CONFIG`` before CUDA
starts, so this module imports no torch at the top level.
"""

from __future__ import annotations

import copy
from collections import Counter, namedtuple
from contextlib import contextmanager, nullcontext

import pytest

from tests.test_b1_expert_forward import (
    _dequantized_reference,
    _experts,
    _inputs,
    _store,
    _streamed,
)

_Run = namedtuple("_Run", "out hidden_grad weights_grad")
_EXPERTS = 8
#: name -> keyword arguments of ``_inputs``
_ROUTINGS = {
    "top2-all-hit": dict(count=24, top_k=2),
    "top1": dict(count=24, top_k=1),
    "top4": dict(count=24, top_k=4),
    "top8": dict(count=24, top_k=8),
    "partial": dict(count=24, partial=True),
    "sparse-top1": dict(count=6, top_k=1),
}
_ALL_HIT = ("top2-all-hit", "top4", "top8")


def _upstream(like):
    """A fixed dense upstream gradient: every output element carries signal."""
    import torch

    gen = torch.Generator().manual_seed(3)
    return torch.randn(like.shape, generator=gen).to(device=like.device, dtype=like.dtype)


def _region(device_type, active):
    import torch

    if not active:
        return nullcontext()
    return torch.autocast(device_type, dtype=torch.bfloat16)


def _run(module, args, *, amp=None, needs=("hidden", "weights"), checkpoint=None):
    """Forward and backward of ``module`` on clones of ``args``.

    ``amp``: None, ``"forward"`` (autocast around the forward only, the shape of a training
    loop), ``"both"`` (backward inside the region too) or ``"backward"`` (plain forward,
    backward inside an autocast region). ``checkpoint``: None, ``"reentrant"`` or
    ``"non-reentrant"``. ``needs`` names the inputs that require grad.
    """
    from torch.utils.checkpoint import checkpoint as run_checkpointed

    hidden, index, weights = args
    hidden = hidden.clone().requires_grad_("hidden" in needs)
    weights = weights.clone().requires_grad_("weights" in needs)
    kind = hidden.device.type
    with _region(kind, amp in ("forward", "both")):
        if checkpoint is None:
            out = module(hidden, index, weights)
        else:
            out = run_checkpointed(
                lambda h, w: module(h, index, w), hidden, weights,
                use_reentrant=checkpoint == "reentrant",
            )
    with _region(kind, amp in ("both", "backward")):
        out.backward(_upstream(out))
    return _Run(out.detach(), hidden.grad, weights.grad)


def _same(a, b):
    import torch

    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def _parts(want, got):
    """Which of output / hidden grad / routing-weight grad are bitwise equal."""
    return {
        "output": _same(want.out, got.out),
        "hidden grad": _same(want.hidden_grad, got.hidden_grad),
        "weights grad": _same(want.weights_grad, got.weights_grad),
    }


def _assert_exact(want, got):
    parts = _parts(want, got)
    assert all(parts.values()), f"not bitwise equal to the resident eager module: {parts}"


def _assert_signal(run):
    """The reference is dense signal, so equality is not a statement about zeros."""
    for name, tensor in zip(run._fields, run):
        if tensor is not None:
            assert (tensor.float() != 0).float().mean().item() > 0.9, f"{name} is mostly zero"


def _hit(args):
    return sorted(set(args[1].flatten().tolist()))


def _close(*resources):
    for resource in resources:
        resource.close()


class _Loads:
    """Wraps the PUBLIC ``reader.read_host`` of one reader.

    Notes every expert load and whether it ran inside autograd's engine (a backward load, or a
    checkpoint recompute that backward triggered), and can perturb or fail the loads that do.
    """

    def __init__(self, monkeypatch, reader):
        import torch

        self.records = []  # (expert, ran inside the engine)
        self.perturb = None  # callable(expert, host) applied to loads inside the engine
        self.perturbed = []
        self.fail = None  # exception raised by loads inside the engine
        original = reader.read_host

        def read_host(expert, staging):
            inside = torch._C._current_autograd_node() is not None
            if inside and self.fail is not None:
                raise self.fail
            host = original(expert, staging)
            self.records.append((expert, inside))
            if inside and self.perturb is not None:
                self.perturb(expert, host)
                self.perturbed.append(expert)
            return host

        monkeypatch.setattr(reader, "read_host", read_host)

    def outside(self):
        return [expert for expert, inside in self.records if not inside]

    def inside(self):
        return [expert for expert, inside in self.records if inside]


def _perturb_down(expert, host):
    """Change what this load returns for the expert's down matrix (in the host staging only)."""
    from soup_cli.utils.expert_shard import DOWN, expert_key

    tensor = host[expert_key(expert, DOWN)]
    if tensor.is_floating_point():
        tensor.add_(0.5)
    else:  # NF4: flip nibbles of the packed weights
        tensor.bitwise_xor_(0x55)


@contextmanager
def exact_backward_case(tmp_path, *, device, dtype, nf4=False, nested=False, **routing):
    """Resident eager vs streamed on ``device``: yields ``(want, got, reader, pool, args)``.

    Shared with the CUDA file. ``nf4`` compares against the independently dequantised resident
    copy of the stored weights. Reader and pool are closed on exit.
    """
    import torch

    module, _ = _experts()
    module = module.to(device=device, dtype=getattr(torch, dtype))
    reader, pool = _store(tmp_path, module, nf4=nf4, nested=nested, device=device)
    try:
        if nf4:
            _dequantized_reference(module, reader, pool)
        args = _inputs(device=device, dtype=getattr(torch, dtype), **routing)
        want = _run(module, args)
        got = _run(_streamed(module, reader, pool), args)
        yield want, got, reader, pool, args
    finally:
        _close(reader, pool)


# --------------------------------------------------------------------------
# E1: output, hidden grad and routing-weight grad equal the resident eager module
# --------------------------------------------------------------------------


@pytest.mark.parametrize("routing", _ROUTINGS)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_gradients_equal_the_resident_eager_module(tmp_path, dtype, routing):
    with exact_backward_case(tmp_path, device="cpu", dtype=dtype, **_ROUTINGS[routing]) as case:
        want, got, reader, pool, args = case
        hit = _hit(args)
        if routing in _ALL_HIT:
            assert len(hit) == _EXPERTS
        elif routing == "partial":
            assert hit == [0, 3]
        elif routing == "sparse-top1":
            assert len(hit) < _EXPERTS  # some experts are unused
        _assert_signal(want)
        _assert_exact(want, got)
        assert pool.fetches == reader.fetches == 2 * len(hit)


@pytest.mark.parametrize("amp", ["forward", "both", "backward"])
@pytest.mark.parametrize("routing", ["top2-all-hit", "top1", "partial"])
def test_fp32_experts_under_cpu_bf16_autocast_equal_eager(tmp_path, routing, amp):
    """``forward`` is the training-loop shape: the node rebuilds the expert under the forward's
    autocast state while backward itself runs outside the region. ``both`` and ``backward`` pin
    that the node follows eager wherever the region sits, including a forward that was NOT
    autocast followed by a backward that is."""
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    args = _inputs(**_ROUTINGS[routing])
    try:
        want = _run(module, args, amp=amp)
        got = _run(_streamed(module, reader, pool), args, amp=amp)
        _assert_signal(want)
        _assert_exact(want, got)
        if amp == "forward":  # autocast really changed the arithmetic being matched
            assert not _same(want.out, _run(module, args).out)
    finally:
        _close(reader, pool)


@pytest.mark.parametrize("needs", [("hidden",), ("weights",)])
def test_only_the_inputs_that_need_grad_get_one(tmp_path, needs):
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    args = _inputs()
    try:
        want = _run(module, args, needs=needs)
        got = _run(_streamed(module, reader, pool), args, needs=needs)
        _assert_signal(want)
        _assert_exact(want, got)
        assert (got.hidden_grad is None) == ("hidden" not in needs)
        assert (got.weights_grad is None) == ("weights" not in needs)
        # either input alone makes every hit expert one node that is read again in backward
        assert reader.fetches == pool.fetches == 2 * len(_hit(args))
    finally:
        _close(reader, pool)


def test_empty_routing_matches_eager_and_reads_nothing(tmp_path):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    hidden = torch.zeros(0, 32, requires_grad=True)
    index = torch.zeros(0, 2, dtype=torch.int64)
    weights = torch.zeros(0, 2, requires_grad=True)
    try:
        want, got = module(hidden, index, weights), streamed(hidden, index, weights)
        assert got.shape == want.shape == (0, 32) and torch.equal(got, want)
        assert got.requires_grad == want.requires_grad
        assert reader.fetches == 0 and streamed.last_trace == []
        args = _inputs()  # and the module is unharmed afterwards
        _assert_exact(_run(module, args), _run(streamed, args))
    finally:
        _close(reader, pool)


# --------------------------------------------------------------------------
# NF4 and nested NF4 against an independently dequantised resident reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_nf4_backward_equals_the_independently_dequantized_reference(tmp_path, nested, dtype):
    pytest.importorskip("bitsandbytes")
    with exact_backward_case(
        tmp_path, device="cpu", dtype=dtype, nf4=True, nested=nested, count=24
    ) as case:
        want, got, reader, pool, args = case
        _assert_signal(want)
        _assert_exact(want, got)
        assert pool.plan.packed_and_stats_bytes > 0 and (pool.plan.scratch_bytes > 0) == nested
        assert pool.fetches == reader.fetches == 2 * len(_hit(args))


# --------------------------------------------------------------------------
# the full DeepseekV3MoE block: router weight and shared experts get the native gradients
# --------------------------------------------------------------------------

_BLOCK_GRADS = {
    "gate.weight",
    "shared_experts.gate_proj.weight",
    "shared_experts.up_proj.weight",
    "shared_experts.down_proj.weight",
}
_BIAS = "gate.e_score_correction_bias"


def _run_block(block, hidden, *, amp):
    block.zero_grad(set_to_none=True)
    hidden = hidden.clone().requires_grad_(True)
    with _region(hidden.device.type, amp):
        out = block(hidden)
    out.backward(_upstream(out))
    grads = {name: p.grad for name, p in block.named_parameters() if p.requires_grad}
    return out.detach(), hidden.grad, grads


def _moe_block(device, dtype):
    """A DeepSeek-V3 MoE block with non-degenerate weights; the routed experts stay frozen."""
    import torch
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    _, config = _experts()
    torch.manual_seed(7)
    block = DeepseekV3MoE(config).eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(0, 0.2)
        block.gate.e_score_correction_bias.normal_(0, 0.01)
    # The correction bias is a buffer in transformers. Made a parameter here it is "trainable"
    # for the check: it only steers top-k indices, so no gradient may reach it in either model.
    block.gate.e_score_correction_bias = torch.nn.Parameter(
        block.gate.e_score_correction_bias.detach().clone()
    )
    block = block.to(device=device, dtype=getattr(torch, dtype)).requires_grad_(False)
    block.gate.requires_grad_(True)
    block.shared_experts.requires_grad_(True)
    return block


def check_block_backward(tmp_path, *, device, dtype, nf4=False, nested=False, amp=False):
    """Router weight and shared-expert gradients of the whole block equal the resident block's.

    Shared with the CUDA file. The router (with its correction bias) and the shared experts are
    the original modules, made trainable for the check; only ``experts`` is streamed.
    """
    import torch

    from soup_cli.utils.expert_stream import install_streamed_experts

    block = _moe_block(device, dtype)
    reference = copy.deepcopy(block)
    experts = block.experts
    reader, pool = _store(tmp_path, experts, nf4=nf4, nested=nested, device=device)
    chosen = []
    hook = reference.gate.register_forward_hook(lambda mod, inp, out: chosen.append(out[2]))
    holder = torch.nn.Module()
    holder.mlp = block
    hidden = _inputs(24, device=device, dtype=getattr(torch, dtype))[0].reshape(2, 12, 32)
    try:
        if nf4:
            _dequantized_reference(reference.experts, reader, pool)
        assert install_streamed_experts([holder], {0: reader}, pool) == 1
        want_out, want_hidden, want_grads = _run_block(reference, hidden, amp=amp)
        hook.remove()
        got_out, got_hidden, got_grads = _run_block(block, hidden, amp=amp)
        assert set(want_grads) == set(got_grads) == _BLOCK_GRADS | {_BIAS}
        assert torch.equal(got_out, want_out) and want_out.abs().sum().item() > 0
        assert torch.equal(got_hidden, want_hidden) and got_hidden.dtype == want_hidden.dtype
        assert want_hidden.abs().sum().item() > 0
        for name in _BLOCK_GRADS:
            assert want_grads[name].abs().sum().item() > 0, f"degenerate reference grad {name}"
            assert torch.equal(got_grads[name], want_grads[name]), name
            assert got_grads[name].dtype == want_grads[name].dtype, name
        assert want_grads[_BIAS] is None and got_grads[_BIAS] is None
        # the routed experts contribute no parameter and no gradient to the streamed block
        assert not list(block.experts.parameters())
        assert experts.gate_up_proj.grad is None and experts.down_proj.grad is None
        assert reader.fetches == 2 * len(set(chosen[0].flatten().tolist()))
    finally:
        hook.remove()
        _close(reader, pool)


@pytest.mark.parametrize("mode", ["fp32", "bf16", "fp32-autocast", "nf4", "nf4-nested"])
def test_full_moe_block_input_router_and_shared_expert_grads_equal_eager(tmp_path, mode):
    nf4 = mode.startswith("nf4")
    if nf4:
        pytest.importorskip("bitsandbytes")
    check_block_backward(
        tmp_path, device="cpu", dtype="bfloat16" if mode == "bf16" else "float32",
        nf4=nf4, nested=mode == "nf4-nested", amp=mode == "fp32-autocast",
    )


def test_lora_on_shared_experts_beside_streamed_routed_experts_equals_eager(tmp_path):
    """An adapter outside the routed experts is the supported training surface: its A/B
    gradients through the streamed block equal the resident block's, bit for bit."""
    import torch

    from soup_cli.utils.expert_stream import install_streamed_experts

    peft = pytest.importorskip("peft")
    holder = torch.nn.Module()
    holder.mlp = _moe_block("cpu", "float32").requires_grad_(False)
    peft.get_peft_model(  # wraps the holder's linear layers in place
        holder,
        peft.LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
            target_modules=["gate_proj", "up_proj", "down_proj"],
        ),
    )
    lora = {name for name, _ in holder.named_parameters() if "lora_" in name}
    assert lora and all("shared_experts" in name for name in lora)  # none on routed experts
    assert lora == {n for n, p in holder.named_parameters() if p.requires_grad}
    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for name, parameter in sorted(holder.named_parameters()):
            if name in lora:  # peft starts lora_B at zero: no signal otherwise
                parameter.copy_(torch.randn(parameter.shape, generator=gen) * 0.1)
    reference = copy.deepcopy(holder)
    chosen = []
    hook = reference.mlp.gate.register_forward_hook(lambda mod, inp, out: chosen.append(out[2]))
    reader, pool = _store(tmp_path, holder.mlp.experts)
    hidden = _inputs(24)[0].reshape(2, 12, 32)

    def grads_of(model):
        model.zero_grad(set_to_none=True)
        h = hidden.clone().requires_grad_(True)
        out = model.mlp(h)
        out.backward(_upstream(out))
        named = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
        return out.detach(), h.grad, named

    try:
        assert install_streamed_experts([holder], {0: reader}, pool) == 1
        want_out, want_hidden, want_grads = grads_of(reference)
        hook.remove()
        got_out, got_hidden, got_grads = grads_of(holder)
        assert torch.equal(got_out, want_out) and torch.equal(got_hidden, want_hidden)
        assert set(got_grads) == set(want_grads) == lora
        for name, grad in want_grads.items():
            assert grad.abs().sum().item() > 0, f"degenerate reference gradient {name}"
            assert torch.equal(got_grads[name], grad), name
        assert reader.fetches == 2 * len(set(chosen[0].flatten().tolist()))
    finally:
        hook.remove()
        _close(reader, pool)


# --------------------------------------------------------------------------
# reloads observed at the public reader / pool boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("routing", ["top2-all-hit", "partial", "sparse-top1"])
def test_backward_reads_each_hit_expert_again_and_never_an_unused_one(
    tmp_path, monkeypatch, routing
):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    hidden, index, weights = _inputs(**_ROUTINGS[routing])
    hit = _hit((hidden, index, weights))
    unused = set(range(_EXPERTS)) - set(hit)
    assert bool(unused) == (routing != "top2-all-hit")
    loads = _Loads(monkeypatch, reader)
    try:
        hidden.requires_grad_(True)
        weights.requires_grad_(True)
        out = streamed(hidden, index, weights)
        assert loads.records == [(expert, False) for expert in hit]  # hit only, ascending
        forward_bytes = reader.bytes_read
        assert reader.fetches == pool.fetches == len(hit)
        out.backward(torch.ones_like(out))
        assert sorted(loads.inside()) == hit  # once per hit expert, inside the engine
        assert unused.isdisjoint(expert for expert, _ in loads.records)
        assert reader.fetches == pool.fetches == 2 * len(hit)
        assert pool.logical_bytes_read == reader.bytes_read == 2 * forward_bytes
        # backward loads are counted by the counters; the bounded trace is the forward's
        assert [row["expert"] for row in streamed.last_trace] == hit
        assert all(row["status"] == "ok" for row in streamed.last_trace)
    finally:
        _close(reader, pool)


@pytest.mark.parametrize("amp", [None, "forward"])
@pytest.mark.parametrize("checkpoint", ["non-reentrant", "reentrant"])
def test_checkpointed_run_equals_eager_and_reads_forward_recompute_and_backward(
    tmp_path, monkeypatch, checkpoint, amp
):
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    args = _inputs()
    hit = _hit(args)
    try:
        want = _run(module, args, amp=amp, checkpoint=checkpoint)
        loads = _Loads(monkeypatch, reader)
        got = _run(streamed, args, amp=amp, checkpoint=checkpoint)
        _assert_signal(want)
        _assert_exact(want, got)
        assert loads.outside() == hit  # the original forward
        assert Counter(loads.inside()) == Counter({expert: 2 for expert in hit})  # replay + grad
        assert reader.fetches == pool.fetches == 3 * len(hit)
        # the trace stays bounded: the recompute is a forward call and replaced it
        assert [row["expert"] for row in streamed.last_trace] == hit
        assert all(row["status"] == "ok" for row in streamed.last_trace)
        with pool.lease(reader, 0):  # no lease survived the recompute or the backward
            pass
    finally:
        _close(reader, pool)


def test_a_retained_graph_reloads_again_for_every_backward(tmp_path, monkeypatch):
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    args = _inputs()
    hit = _hit(args)

    def twice(target):
        hidden, index, weights = (t.clone() for t in args)
        hidden.requires_grad_(True)
        weights.requires_grad_(True)
        out = target(hidden, index, weights)
        grad = _upstream(out)
        out.backward(grad, retain_graph=True)
        out.backward(grad)
        return _Run(out.detach(), hidden.grad, weights.grad)

    try:
        want = twice(module)
        loads = _Loads(monkeypatch, reader)
        got = twice(streamed)
        _assert_signal(want)
        _assert_exact(want, got)
        assert loads.outside() == hit
        assert Counter(loads.inside()) == Counter({expert: 2 for expert in hit})
    finally:
        _close(reader, pool)


def test_second_order_differentiation_is_refused_and_frees_the_pool(tmp_path):
    """Backward is first-order only: its gradients are computed against arena weights that the
    next lease overwrites, so no graph over them may be handed back. (The resident eager module
    does support second order; that is a documented difference, not a parity claim.)"""
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    hidden, index, weights = _inputs()
    hidden.requires_grad_(True)
    try:
        out = streamed(hidden, index, weights)
        # a constant upstream gradient: the returned gradient carries no graph at all
        (plain,) = torch.autograd.grad(out, hidden, _upstream(out), create_graph=True)
        assert not plain.requires_grad
        # an upstream gradient that requires grad: torch refuses to differentiate the node twice
        out = streamed(hidden, index, weights)
        upstream = _upstream(out).requires_grad_(True)
        (graphed,) = torch.autograd.grad(out, hidden, upstream, create_graph=True)
        loaded = reader.fetches
        with pytest.raises(RuntimeError, match="twice"):
            graphed.sum().backward()
        assert reader.fetches == loaded  # nothing was differentiated against the arena
        with pool.lease(reader, 0):  # no lease leaked from the refused second-order pass
            pass
    finally:
        _close(reader, pool)


def test_two_layers_sharing_one_pool_reload_each_from_its_own_reader(tmp_path):
    first, _ = _experts(seed=5)
    second, _ = _experts(seed=11)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    reader_a, pool = _store(tmp_path / "a", first)
    reader_b, spare = _store(tmp_path / "b", second)
    spare.close()  # one pool serves both layers
    layer_a, layer_b = _streamed(first, reader_a, pool), _streamed(second, reader_b, pool)
    hidden, index_a, weights_a = _inputs(24, top_k=2)
    _, index_b, weights_b = _inputs(24, top_k=1)

    def chain(one, two):
        h = hidden.clone().requires_grad_(True)
        wa = weights_a.clone().requires_grad_(True)
        wb = weights_b.clone().requires_grad_(True)
        out = two(one(h, index_a, wa), index_b, wb)
        out.backward(_upstream(out))
        return out.detach(), h.grad, wa.grad, wb.grad

    try:
        want, got = chain(first, second), chain(layer_a, layer_b)
        assert all(tensor.abs().sum().item() > 0 for tensor in want)
        assert all(_same(a, b) for a, b in zip(want, got)), "chained layers differ from eager"
        assert reader_a.fetches == 2 * len(set(index_a.flatten().tolist()))
        assert reader_b.fetches == 2 * len(set(index_b.flatten().tolist()))
    finally:
        _close(reader_a, reader_b, pool)


# --------------------------------------------------------------------------
# arena lifetime: nothing saved aliases it, reuse between forward and backward is harmless
# --------------------------------------------------------------------------


def check_arena_churn(tmp_path, *, device, dtype):
    """Other leases and another forward between forward and backward leave the grads alone.

    Shared with the CUDA file.
    """
    import torch

    module, _ = _experts()
    module = module.to(device=device, dtype=getattr(torch, dtype))
    reader, pool = _store(tmp_path, module, device=device)
    streamed = _streamed(module, reader, pool)
    args = _inputs(device=device, dtype=getattr(torch, dtype))
    other = _inputs(12, device=device, dtype=getattr(torch, dtype), top_k=1)
    try:
        want = _run(module, args)
        _assert_signal(want)
        hidden, index, weights = (t.clone() for t in args)
        hidden.requires_grad_(True)
        weights.requires_grad_(True)
        out = streamed(hidden, index, weights)
        for expert in range(_EXPERTS):  # every expert, unused ones included, through the slot
            with pool.lease(reader, expert):
                pass
        with torch.no_grad():
            streamed(*other)
        streamed(other[0].clone().requires_grad_(True), other[1], other[2])  # never backwarded
        out.backward(_upstream(out))
        _assert_exact(want, _Run(out.detach(), hidden.grad, weights.grad))
    finally:
        _close(reader, pool)


def test_arena_churn_between_forward_and_backward_does_not_change_the_gradients(tmp_path):
    check_arena_churn(tmp_path, device="cpu", dtype="float32")


def test_nothing_saved_for_backward_aliases_the_arena_or_the_staging(tmp_path):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    hidden, index, weights = _inputs()
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            out = streamed(hidden, index, weights)
        # at least the gathered hidden and routing slices of every hit expert
        assert len(saved) >= 2 * len(_hit((hidden, index, weights)))
        for owner in (pool._arena, pool._staging_owner):
            low = owner.data_ptr()
            high = low + owner.numel()
            for tensor in saved:
                assert not low <= tensor.data_ptr() < high, "a saved tensor aliases pool memory"
                assert tensor.untyped_storage().data_ptr() != owner.untyped_storage().data_ptr()
        assert out.untyped_storage().data_ptr() != pool._arena.untyped_storage().data_ptr()
    finally:
        _close(reader, pool)


def test_saved_tensors_are_unpacked_before_the_backward_lease(tmp_path):
    """A checkpoint recompute runs inside the unpack and leases experts itself, so the node
    must have unpacked before it holds the one slot. This unpack hook leases an expert like a
    recompute does; it would hit 'active lease' if the node leased first."""
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    args = _inputs()
    hit = _hit(args)
    unpacks = []

    def unpack(tensor):
        with pool.lease(reader, 0):
            unpacks.append(torch._C._current_autograd_node() is not None)
        return tensor

    hidden, index, weights = (t.clone() for t in args)
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    try:
        want = _run(module, args)
        with torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, unpack):
            out = streamed(hidden, index, weights)
        out.backward(_upstream(out))
        assert unpacks and all(unpacks)  # leased from inside autograd's engine
        assert len(unpacks) >= len(hit)  # at least once per node
        _assert_exact(want, _Run(out.detach(), hidden.grad, weights.grad))
    finally:
        _close(reader, pool)


def test_a_backward_lease_while_another_lease_is_active_fails_loudly(tmp_path):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    args = _inputs()
    hidden, index, weights = (t.clone() for t in args)
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    try:
        want = _run(module, args)
        out = streamed(hidden, index, weights)
        loaded = reader.fetches
        with pool.lease(reader, 5) as (gate_up, _down):
            held = gate_up.clone()
            with pytest.raises(RuntimeError, match="active lease"):
                out.backward(_upstream(out))
            assert torch.equal(gate_up, held)  # the lease's weights were not touched
        assert hidden.grad is None and weights.grad is None
        assert reader.fetches == loaded + 1  # nothing but the explicit lease loaded an expert
        _assert_exact(want, _run(streamed, args))  # and the pool is fine afterwards
    finally:
        _close(reader, pool)


def test_no_lease_is_left_active_after_backward_and_frozen_weights_get_no_grad(tmp_path):
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    try:
        _run(streamed, _inputs())
        with pool.lease(reader, 0) as (gate_up, down):  # raises if backward kept the lease
            assert not gate_up.requires_grad and not down.requires_grad
            assert gate_up.grad is None and down.grad is None
        assert not list(streamed.parameters()) and not list(streamed.buffers())
        pool.close()  # refuses while a lease is active
        assert pool.stats()["closed"]
    finally:
        _close(reader, pool)


def test_backward_after_the_pool_was_closed_fails_loudly(tmp_path):
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    hidden, index, weights = _inputs()
    hidden.requires_grad_(True)
    try:
        out = streamed(hidden, index, weights)
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            out.backward(_upstream(out))
        assert hidden.grad is None
    finally:
        _close(reader, pool)


def test_load_failure_inside_backward_propagates_and_leaves_the_pool_usable(
    tmp_path, monkeypatch
):
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    args = _inputs()
    loads = _Loads(monkeypatch, reader)
    hidden, index, weights = (t.clone() for t in args)
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    try:
        want = _run(module, args)
        out = streamed(hidden, index, weights)  # forward loads do not run in the engine
        loads.fail = OSError("deliberate short read in backward")
        with pytest.raises(OSError, match="deliberate"):
            out.backward(_upstream(out))
        assert hidden.grad is None and weights.grad is None
        assert reader._handle is None  # the pool closes the reader that failed, as in forward
        loads.fail = None
        with pool.lease(reader, 0):  # the lock was released, the reader reopens
            pass
        _assert_exact(want, _run(streamed, args))
    finally:
        _close(reader, pool)


# --------------------------------------------------------------------------
# negative control: perturbing only the loads inside backward must show in the gradients
# --------------------------------------------------------------------------


def check_negative_control(tmp_path, monkeypatch, *, device, dtype, nf4=False, nested=False):
    """Expert weights perturbed only on loads inside autograd's engine: the output stays equal
    and both gradients differ; unperturbed on the same inputs everything is equal.

    Shared with the CUDA file.
    """
    import torch

    module, _ = _experts()
    module = module.to(device=device, dtype=getattr(torch, dtype))
    reader, pool = _store(tmp_path, module, nf4=nf4, nested=nested, device=device)
    streamed = _streamed(module, reader, pool)
    args = _inputs(device=device, dtype=getattr(torch, dtype))
    hit = _hit(args)
    loads = _Loads(monkeypatch, reader)
    try:
        if nf4:
            _dequantized_reference(module, reader, pool)
        want = _run(module, args)
        _assert_signal(want)
        _assert_exact(want, _run(streamed, args))  # same inputs, unperturbed: equal
        assert loads.perturbed == []
        loads.perturb = _perturb_down
        perturbed = _run(streamed, args)
        assert sorted(loads.perturbed) == hit  # every backward load, and no forward load
        assert _parts(want, perturbed) == {
            "output": True, "hidden grad": False, "weights grad": False,
        }
        # the perturbation left no trace in the cache file or the arena: exact again after it
        loads.perturb = None
        _assert_exact(want, _run(streamed, args))
    finally:
        _close(reader, pool)


@pytest.mark.parametrize("quant", ["dense", "nf4-nested"])
def test_perturbing_only_backward_reloads_changes_gradients_not_outputs(
    tmp_path, monkeypatch, quant
):
    nf4 = quant != "dense"
    if nf4:
        pytest.importorskip("bitsandbytes")
    check_negative_control(
        tmp_path, monkeypatch, device="cpu", dtype="float32", nf4=nf4, nested=nf4
    )


# --------------------------------------------------------------------------
# the refusals that remain name the B1.3 boundary (the forward-only one is gone, see
# test_b1_expert_forward.py::test_forward_accepts_a_grad_bearing_input_and_reads_only_hit_experts)
# --------------------------------------------------------------------------


def _routed_holder(module):
    import torch

    holder = torch.nn.Module()
    holder.mlp = torch.nn.Module()
    holder.mlp.experts = module
    return holder


@pytest.mark.parametrize("kind", ["peft-target-parameters", "extra-parameter"])
def test_adapters_and_extra_parameters_on_routed_experts_are_refused_before_a_read(
    tmp_path, kind
):
    import torch

    from soup_cli.utils.expert_stream import install_streamed_experts

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    holder = _routed_holder(module)
    if kind == "extra-parameter":
        module.register_parameter("scale", torch.nn.Parameter(torch.ones(1)))
    else:
        peft = pytest.importorskip("peft")
        peft.get_peft_model(
            holder,
            peft.LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.0,
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            ),
        )
    wrapped = holder.mlp.experts
    try:
        # a peft ParamWrapper hides gate_up_proj/down_proj: it must not read as a layout error
        with pytest.raises(ValueError, match="adapter") as excinfo:
            install_streamed_experts([holder], {0: reader}, pool)
        assert "outside routed experts" in str(excinfo.value)
        assert "reshard" not in str(excinfo.value)
        assert holder.mlp.experts is wrapped and reader.fetches == 0
    finally:
        _close(reader, pool)


def test_a_reader_that_does_not_match_the_pool_is_refused_before_a_read(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import install_streamed_experts

    module, _ = _experts()
    (tmp_path / "pool").mkdir()
    (tmp_path / "other").mkdir()
    reader, pool = _store(tmp_path / "pool", module)
    # a cache stored at another dtype than this pool's: the storage contract differs
    other, other_pool = _store(tmp_path / "other", _experts()[0].to(torch.bfloat16))
    other_pool.close()
    holder = _routed_holder(module)
    try:
        with pytest.raises(ValueError, match="does not match the buffer pool"):
            _streamed(module, other, pool)
        with pytest.raises(ValueError, match="does not match the buffer pool"):
            install_streamed_experts([holder], {0: other}, pool)
        with pytest.raises(ValueError, match="does not match the buffer pool"):
            with pool.lease(other, 0):
                pass
        assert holder.mlp.experts is module and other.fetches == 0 and pool.fetches == 0
    finally:
        _close(reader, other, pool)


@pytest.mark.parametrize("trainable", ["gate_up_proj", "down_proj", "both"])
def test_trainable_routed_expert_weights_are_refused_before_a_read(tmp_path, trainable):
    from soup_cli.utils.expert_stream import install_streamed_experts

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    holder = _routed_holder(module)
    if trainable == "both":
        module.requires_grad_(True)
    else:
        getattr(module, trainable).requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="frozen") as excinfo:
            install_streamed_experts([holder], {0: reader}, pool)
        assert "never trains" in str(excinfo.value)
        assert holder.mlp.experts is module and reader.fetches == 0
    finally:
        _close(reader, pool)


@pytest.mark.parametrize("bad", ["negative", "large", "dtype", "shape", "nan"])
def test_malformed_routing_with_grad_bearing_inputs_still_fails_before_a_read(tmp_path, bad):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    hidden, index, weights = _inputs()
    if bad == "negative":
        index[0, 0] = -1
    elif bad == "large":
        index[0, 0] = _EXPERTS
    elif bad == "dtype":
        index = index.to(torch.int32)
    elif bad == "shape":
        weights = weights[:, :1]
    else:
        weights[0, 0] = float("nan")
    try:
        with pytest.raises(ValueError):
            _streamed(module, reader, pool)(
                hidden.requires_grad_(True), index, weights.requires_grad_(True)
            )
        assert reader.fetches == 0
    finally:
        _close(reader, pool)
