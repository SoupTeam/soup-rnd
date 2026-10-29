"""B1.3 ticket 03: the L2L step runs per-expert DeepSeek-V3 models, exact against resident eager GA.

Seam: ``L2LStep.run`` over a per-expert DeepSeek-V3 model built by ``build_streamed_model``
(routed experts streamed one at a time through the bounded arena, LoRA on the MLA projections).
Every expected value comes from an independent RESIDENT eager model with the same weights and
the same LoRA, run as plain gradient accumulation: k forward+backward calls with
``num_items_in_batch`` over the whole window and one generator seeded before the window, as
``tests/test_l2l_rng.py::ga_reference`` runs it. Comparison is ``torch.equal`` (E3: per-micro-batch
losses, every LoRA gradient, generator state after the window); nothing here is a tolerance.

The micro-batches of a window differ, so the router sends them to different experts; the
reference run records what its routers chose, which is the oracle for which experts the step
may read. A load that runs inside autograd's engine (``torch._C._current_autograd_node()``) is an
expert's own backward re-reading its weights; the negative control perturbs only those.

``check_*``, ``make_rig`` and the helpers are shared with ``test_b13_l2l_moe_cuda.py``, which must
set ``CUBLAS_WORKSPACE_CONFIG`` before CUDA starts.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch

from soup_cli.utils import l2l_rng, l2l_step
from tests.test_b13_moe_runtime import (
    N_EXPERTS,
    build_moe_streamed,
    grads_differ,
    lora_parameters,
    moe_batch,
    moe_config,
    moe_resident_reference,
    moe_weights_dir,
    streamed_experts_modules,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR

SEED = 11


# ==========================================================================
# the rig: one streamed per-expert model, its resident twin, a stepper and a store
# ==========================================================================
@dataclasses.dataclass
class Rig:
    config: Any
    moe: Any  # MoeStreamed
    resident: Any
    stepper: Any
    store: Any
    device: str
    rows: int
    seq: int

    @property
    def model(self) -> Any:
        return self.moe.model

    @property
    def runtime(self) -> Any:
        return self.moe.runtime

    @property
    def moe_layers(self) -> range:
        return range(self.config.first_k_dense_replace, self.config.num_hidden_layers)

    def close(self) -> None:
        try:
            self.store.close()
        finally:
            self.moe.close()


@pytest.fixture()
def make_rig(tmp_path):
    rigs: List[Rig] = []

    def make(
        *,
        k: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        config: Any = None,
        rows: int = 2,
        seq: int = 6,
        device: str = "cpu",
        layout: str = "per_expert",
    ) -> Rig:
        from soup_cli.utils.l2l_activations import ActivationStore

        config = config if config is not None else moe_config()
        root = tmp_path / f"rig{len(rigs)}"
        root.mkdir()
        weights = moe_weights_dir(root, config)
        moe = build_moe_streamed(
            weights, root / "shards", lora_dropout=dropout, layout=layout, device=device
        )
        try:
            resident = moe_resident_reference(weights, moe.lora_config, lora_from=moe.model)
            if device != "cpu":
                resident.to(device)
            moe.model.train()
            resident.train()
            if attention_dropout:
                for model in (moe.model, resident):
                    for module in model.modules():
                        if hasattr(module, "attention_dropout"):
                            module.attention_dropout = attention_dropout
                        cfg = getattr(module, "config", None)
                        if cfg is not None:
                            cfg._attn_implementation = "eager"
            first = moe_batch(config, seed=0, rows=rows, seq=seq)["input_ids"].to(device)
            stepper = l2l_step.L2LStep(
                moe.model, moe.runtime, l2l_step.capture_layer_call(moe.model, first)
            )
            store = ActivationStore(
                n_layers=config.num_hidden_layers,
                k=k,
                chunk_shape=(rows, seq, config.hidden_size),
                dtype=torch.float32,
                device=device,
                pin=False,
            )
        except BaseException:
            moe.close()
            raise
        rig = Rig(config, moe, resident, stepper, store, device, rows, seq)
        rigs.append(rig)
        return rig

    yield make
    for rig in rigs:
        rig.close()


def micro_batches(rig: Rig, k: int, *, first: int = 0) -> List[Any]:
    """k micro-batches with different tokens, hence different routing."""
    return [
        moe_batch(rig.config, seed=first + i, rows=rig.rows, seq=rig.seq)["input_ids"].to(
            rig.device
        )
        for i in range(k)
    ]


@dataclasses.dataclass
class Spec:
    """One accumulation window. ``None`` labels / ``n_items`` take the step's defaults."""

    ids: List[Any]
    labels: Optional[List[Any]] = None
    n_items: Optional[int] = None

    def reference(self) -> tuple:
        """(labels, num_items_in_batch) as plain accumulation passes them."""
        labels = self.labels if self.labels is not None else self.ids
        total = self.n_items
        if total is None:
            total = sum(int((lab[:, 1:] != -100).sum()) for lab in labels)
        return labels, total


@dataclasses.dataclass
class Window:
    losses: List[Any]
    grads: Dict[str, Any]
    rng: Any
    #: resident reference only: layer -> router top-k indices of each micro-batch
    routes: Dict[int, List[Any]] = dataclasses.field(default_factory=dict)
    #: resident reference only: layer -> sum |shared experts output| of each micro-batch
    shared: Dict[int, List[float]] = dataclasses.field(default_factory=dict)
    #: L2L only: the step's load counters for this window
    counters: Any = None


def _zero_grads(model: Any) -> None:
    for param in lora_parameters(model).values():
        param.grad = None


def _collect(model: Any) -> Dict[str, Any]:
    return {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in lora_parameters(model).items()
    }


class _ResidentProbe:
    """Records what the resident model's routers chose and what its shared experts produced."""

    def __init__(self, model: Any) -> None:
        self.routes: Dict[int, List[Any]] = {}
        self.shared: Dict[int, List[float]] = {}
        self._handles: List[Any] = []
        for name, module in model.named_modules():
            found = re.search(r"layers\.(\d+)\.mlp\.(gate|shared_experts)$", name)
            if found is None:
                continue
            layer = int(found.group(1))
            if found.group(2) == "gate":
                hook = self._route(layer)
            else:
                hook = self._share(layer)
            self._handles.append(module.register_forward_hook(hook))

    def _route(self, layer: int) -> Callable[..., None]:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            self.routes.setdefault(layer, []).append(output[2].detach().cpu().clone())

        return hook

    def _share(self, layer: int) -> Callable[..., None]:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            self.shared.setdefault(layer, []).append(float(output.detach().abs().sum()))

        return hook

    def reset(self) -> None:
        self.routes, self.shared = {}, {}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def ga_windows(
    rig: Rig, specs: List[Spec], *, seed: int = SEED, amp: bool = False
) -> List[Window]:
    """Resident eager accumulation: forward AMP only, one generator seeded before windows."""
    model = rig.resident
    model.train()
    probe = _ResidentProbe(model)
    torch.manual_seed(seed)
    out: List[Window] = []
    try:
        for spec in specs:
            labels, total = spec.reference()
            _zero_grads(model)
            probe.reset()
            losses = []
            for ids, lab in zip(spec.ids, labels):
                with torch.autocast(
                    torch.device(rig.device).type, dtype=torch.bfloat16, enabled=amp
                ):
                    loss = model(input_ids=ids, labels=lab, num_items_in_batch=total).loss
                loss.backward()
                losses.append(loss.detach())
            out.append(
                Window(losses, _collect(model), l2l_rng.snapshot(), probe.routes, probe.shared)
            )
    finally:
        probe.close()
    return out


def l2l_windows(
    rig: Rig, specs: List[Spec], *, seed: int = SEED, amp: bool = False
) -> List[Window]:
    """The L2L step over the streamed model, windows back to back on one stepper and store."""
    model = rig.model
    model.train()
    torch.manual_seed(seed)
    out: List[Window] = []
    for spec in specs:
        _zero_grads(model)
        with torch.autocast(torch.device(rig.device).type, dtype=torch.bfloat16, enabled=amp):
            losses = rig.stepper.run(spec.ids, rig.store, n_items=spec.n_items, labels=spec.labels)
        out.append(
            Window(
                [loss.detach() for loss in losses],
                _collect(model),
                l2l_rng.snapshot(),
                counters=dataclasses.replace(rig.stepper.counters),
            )
        )
    return out


# ==========================================================================
# comparison helpers
# ==========================================================================
def rng_equal(left: Any, right: Any) -> bool:
    cuda = len(left.cuda) == len(right.cuda) and all(
        a[0] == b[0] and torch.equal(a[1], b[1]) for a, b in zip(left.cuda, right.cuda)
    )
    return bool(torch.equal(left.cpu, right.cpu)) and cuda


def _worst(got: Dict[str, Any], want: Dict[str, Any], limit: int = 3) -> list:
    rows = []
    for name in grads_differ(got, want)[:limit]:
        g, w = got.get(name), want.get(name)
        diff = None if g is None or w is None else float((g.double() - w.double()).abs().max())
        rows.append((name, diff))
    return rows


def losses_equal(want: Window, got: Window) -> bool:
    return len(want.losses) == len(got.losses) and all(
        torch.equal(a.cpu(), b.cpu()) for a, b in zip(want.losses, got.losses)
    )


def assert_same(want: Window, got: Window, *, stream: bool = True) -> None:
    """E3: per-micro-batch losses, every LoRA gradient and (``stream``) the generator state."""
    assert len(got.losses) == len(want.losses)
    for mb, (a, b) in enumerate(zip(want.losses, got.losses)):
        assert torch.equal(a.cpu(), b.cpu()), f"loss of micro-batch {mb}: {a.item()} vs {b.item()}"
    assert set(got.grads) == set(want.grads) and got.grads
    names = grads_differ(got.grads, want.grads)
    assert not names, f"{len(names)} LoRA gradients differ, worst {_worst(got.grads, want.grads)}"
    if stream:
        assert rng_equal(want.rng, got.rng), "the generator state after the window differs"


def max_grad_difference(left: Window, right: Window) -> float:
    return max(
        float((left.grads[name].double() - right.grads[name].double()).abs().max())
        for name in left.grads
        if left.grads[name] is not None and right.grads[name] is not None
    )


def record_comparison(record_property: Callable[..., None], want: Window, got: Window) -> None:
    """Keep numerical evidence in pytest's JUnit report, including non-degeneracy."""
    record_property("lora_grad_tensors", len(want.grads))
    record_property(
        "nonzero_lora_grad_tensors",
        sum(g is not None and bool(g.ne(0).any()) for g in want.grads.values()),
    )
    record_property("max_abs_grad_diff", max_grad_difference(want, got))
    record_property(
        "max_abs_loss_diff", max(float((a - b).abs()) for a, b in zip(want.losses, got.losses))
    )
    record_property("rng_equal", rng_equal(want.rng, got.rng))
    record_property("expert_fetches", got.counters.expert_fetches)
    record_property("expert_bytes_read", got.counters.expert_bytes_read)


def assert_signal(window: Window) -> None:
    """The reference is not degenerate, so equality is not a statement about zeros."""
    assert window.grads
    zero = [n for n, g in window.grads.items() if g is None or not bool(g.ne(0).any())]
    assert not zero, f"all-zero or missing reference gradients: {zero[:4]}"


def hits_of(window: Window) -> Dict[int, List[List[int]]]:
    """layer -> per micro-batch, the experts the reference's router chose (ascending)."""
    return {
        layer: [sorted(set(route.flatten().tolist())) for route in routes]
        for layer, routes in window.routes.items()
    }


def assert_routing_is_varied(window: Window, rig: Rig, k: int) -> None:
    """Different routing per micro-batch, unused experts, shared experts doing work."""
    assert sorted(window.routes) == list(rig.moe_layers)
    for layer, routes in window.routes.items():
        assert len(routes) == k, (
            f"layer {layer}: {len(routes)} routing records for {k} micro-batches"
        )
        for i in range(k):
            for j in range(i + 1, k):
                assert not torch.equal(routes[i], routes[j]), (
                    f"layer {layer}: micro-batches {i} and {j} were routed identically"
                )
        used = [set(route.flatten().tolist()) for route in routes]
        assert any(len(seen) < N_EXPERTS for seen in used), f"layer {layer} used every expert"
        assert len(set().union(*used)) >= 2, f"layer {layer} routed to a single expert"
        assert len(window.shared[layer]) == k and all(s > 0 for s in window.shared[layer]), (
            f"layer {layer}: the shared experts produced nothing"
        )


def check_l2l_equals_accumulation(
    make_rig: Callable[..., Rig],
    *,
    k: int,
    dropout: float,
    device: str = "cpu",
    config: Any = None,
    rows: int = 2,
    seq: int = 6,
) -> tuple:
    """E3 for one window of k different micro-batches; returns (rig, reference, l2l)."""
    rig = make_rig(k=k, dropout=dropout, device=device, config=config, rows=rows, seq=seq)
    specs = [Spec(micro_batches(rig, k))]
    (want,) = ga_windows(rig, specs)
    (got,) = l2l_windows(rig, specs)
    assert_signal(want)
    assert_routing_is_varied(want, rig, k)
    assert_same(want, got)
    assert got.counters.expert_fetches > 0
    return rig, want, got


# ==========================================================================
# E3: the step equals resident eager gradient accumulation, bit for bit
# ==========================================================================
@pytest.mark.parametrize("dropout", [0.0, 0.05])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_e3_l2l_equals_resident_eager_accumulation(make_rig, k, dropout, record_property):
    _rig, want, got = check_l2l_equals_accumulation(make_rig, k=k, dropout=dropout)
    record_comparison(record_property, want, got)


def test_the_fixture_is_not_degenerate_dropout_changes_what_the_reference_computes(make_rig):
    """Control: dropout 0.05 really moves the gradients, so the dropout cases test the replay."""
    plain = make_rig(k=2, dropout=0.0)
    dropped = make_rig(k=2, dropout=0.05)
    (a,) = ga_windows(plain, [Spec(micro_batches(plain, 2))])
    (b,) = ga_windows(dropped, [Spec(micro_batches(dropped, 2))])
    assert grads_differ(a.grads, b.grads)
    assert not rng_equal(a.rng, b.rng)


# ==========================================================================
# router top-k varied independently of k
# ==========================================================================
@pytest.mark.parametrize("dropout", [0.0, 0.05])
@pytest.mark.parametrize(("top_k", "rows", "seq"), [(1, 2, 6), (3, 1, 4)])
def test_router_top_k_is_varied_independently_of_k(make_rig, top_k, rows, seq, dropout):
    """k stays 2; the router picks 1 or 3 experts per token (3 needs few tokens to leave
    experts unused, so that case runs 4 tokens per micro-batch)."""
    config = moe_config(num_experts_per_tok=top_k)
    _rig, want, _got = check_l2l_equals_accumulation(
        make_rig, k=2, dropout=dropout, config=config, rows=rows, seq=seq
    )
    for routes in want.routes.values():
        assert all(tuple(route.shape) == (rows * seq, top_k) for route in routes)


# ==========================================================================
# consecutive windows on one stepper: nothing stale in the store, the routing or the arena
# ==========================================================================
@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_consecutive_windows_on_one_stepper_equal_accumulation(make_rig, dropout, guard_runs):
    """Windows A, B, A on one stepper, one store and one generator, as consecutive optimizer
    steps run them. B routes differently from A, so a routing record, a stored activation or an
    arena slot left behind by a window would show in the next one."""
    k = 2
    rig = make_rig(k=k, dropout=dropout)
    window_a, window_b = micro_batches(rig, k), micro_batches(rig, k, first=10)
    specs = [Spec(window_a), Spec(window_b), Spec(window_a)]
    want = ga_windows(rig, specs)
    got = l2l_windows(rig, specs)
    for expected in want:
        assert_signal(expected)
        assert_routing_is_varied(expected, rig, k)
    assert any(
        not torch.equal(a, b)
        for layer, routes in want[0].routes.items()
        for a, b in zip(routes, want[1].routes[layer])
    )
    assert grads_differ(want[0].grads, want[1].grads)
    for expected, actual in zip(want, got):
        assert_same(expected, actual)
    if dropout == 0.0:  # without dropout the same window gives the same result again
        assert_same(got[0], got[2])
    for guard in guard_runs.guards:
        assert_guard_closed(guard)
    assert all(released for _pair, released in guard_runs.checked)


# ==========================================================================
# SFT labels: -100 on prompt and padding tokens reaches the loss like in accumulation
# ==========================================================================
@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_sft_masked_labels_equal_accumulation(make_rig, dropout):
    k = 2
    rig = make_rig(k=k, dropout=dropout)
    ids = micro_batches(rig, k)
    labels = [row.clone() for row in ids]
    labels[0][:, :3] = -100
    labels[1][:, -2:] = -100
    labels[1][0, :] = -100  # one whole row masked
    n_items = sum(int((lab[:, 1:] != -100).sum()) for lab in labels)
    masked = Spec(ids, labels, n_items)

    (want,) = ga_windows(rig, [masked])
    (got,) = l2l_windows(rig, [masked])
    assert_signal(want)
    assert_same(want, got)
    # control: the masking is real, the same tokens unmasked give another loss
    (plain,) = ga_windows(rig, [Spec(ids)])
    assert not losses_equal(plain, want)


# ==========================================================================
# expert loads: counted per step, and the backward reads them again
# ==========================================================================
class ReadLog:
    """Wraps the PUBLIC ``reader.read_host`` of every streamed layer.

    Notes every expert load and whether it ran inside autograd's engine, which is the case for
    an expert's own backward re-reading its weights (the no-grad forward and the grad recompute
    are called by the step itself, outside the engine). Loads inside the engine can be perturbed.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
        #: layer -> [(expert, ran inside autograd's engine)] in load order
        self.records: Dict[int, List[tuple]] = {}
        #: all layers together, so reverse layer traversal cannot hide in per-layer totals
        self.events: List[tuple] = []
        #: callable(layer, expert, host staging dict) applied to the loads inside the engine
        self.perturb: Optional[Callable[[int, int, Any], None]] = None
        self.perturbed: List[tuple] = []
        for layer, module in streamed_experts_modules(model).items():
            self._wrap(monkeypatch, layer, module.reader)

    def _wrap(self, monkeypatch: pytest.MonkeyPatch, layer: int, reader: Any) -> None:
        original = reader.read_host
        records = self.records.setdefault(layer, [])

        def read_host(expert: int, staging: Any) -> Any:
            inside = torch._C._current_autograd_node() is not None
            host = original(expert, staging)
            records.append((expert, inside))
            self.events.append((layer, expert, inside))
            if inside and self.perturb is not None:
                self.perturb(layer, expert, host)
                self.perturbed.append((layer, expert))
            return host

        monkeypatch.setattr(reader, "read_host", read_host)

    def reads(self, layer: int, *, inside: bool) -> List[int]:
        return [expert for expert, was_inside in self.records[layer] if was_inside == inside]


def perturb_down(_layer: int, expert: int, host: Any) -> None:
    """Change what a load returns for the expert's down matrix (host staging only)."""
    from soup_cli.utils.expert_shard import DOWN, expert_key

    host[expert_key(expert, DOWN)].add_(0.5)


def test_expert_loads_are_counted_and_backward_reads_each_hit_expert_again(
    make_rig, monkeypatch, record_property
):
    """Per hit expert per micro-batch: forward, recompute, backward reads; reverse layers."""
    from collections import Counter

    k = 3
    rig = make_rig(k=k)
    specs = [Spec(micro_batches(rig, k))]
    (want,) = ga_windows(rig, specs)
    log = ReadLog(monkeypatch, rig.model)
    readers = rig.runtime.expert_readers
    before = [(reader.fetches, reader.bytes_read) for reader in readers]
    stats_before = rig.runtime.stats()  # make_rig's capture forward has read experts already
    calls = []
    handles = [
        layer.register_forward_pre_hook(
            lambda _module, _args, idx=idx: calls.append((idx, torch.is_grad_enabled()))
        )
        for idx, layer in enumerate(rig.stepper.layers)
    ]
    try:
        (got,) = l2l_windows(rig, specs)
    finally:
        for handle in handles:
            handle.remove()
    assert_same(want, got)
    assert calls == [
        (layer, grad)
        for grad, layers in (
            (False, range(rig.config.num_hidden_layers)),
            (True, range(rig.config.num_hidden_layers - 1, -1, -1)),
        )
        for layer in layers
        for _mb in range(k)
    ]

    hits = hits_of(want)
    assert sorted(hits) == sorted(log.records) == list(rig.moe_layers)
    total_hits = 0
    for layer, per_mb in hits.items():
        per_expert = Counter(expert for seen in per_mb for expert in seen)
        total_hits += sum(per_expert.values())
        outside, inside = log.reads(layer, inside=False), log.reads(layer, inside=True)
        assert Counter(outside) == Counter({e: 2 * n for e, n in per_expert.items()}), layer
        assert Counter(inside) == Counter(per_expert), f"layer {layer}: backward reloads"
        assert inside, f"layer {layer}: backward re-read nothing"
        # the whole no-grad forward comes first: micro-batch by micro-batch, ascending experts
        forward = [expert for seen in per_mb for expert in seen]
        assert outside[: len(forward)] == forward, f"layer {layer}: forward read order"
        assert set(outside) == set(per_expert), f"layer {layer} read an unused expert"

    counters = got.counters
    assert counters.layer_loads > 0
    assert counters.expert_fetches == 3 * total_hits
    assert counters.expert_fetches == sum(
        reader.fetches - fetched for reader, (fetched, _bytes) in zip(readers, before)
    )
    assert counters.expert_bytes_read == sum(
        reader.bytes_read - read for reader, (_fetched, read) in zip(readers, before)
    )
    assert counters.expert_bytes_read > 0
    stats_after = rig.runtime.stats()
    assert stats_after["expert_fetches"] - stats_before["expert_fetches"] == counters.expert_fetches
    assert stats_after["expert_bytes_read"] - stats_before["expert_bytes_read"] == (
        counters.expert_bytes_read
    )

    # Every forward load precedes the backward walk; each recompute and its real backward
    # finish before the next micro-batch, and the MoE layers are consumed in reverse order.
    forward_events = [
        (layer, expert, False)
        for layer, per_mb in hits.items()
        for seen in per_mb
        for expert in seen
    ]
    assert log.events[:total_hits] == forward_events
    cursor = total_hits
    for layer in reversed(list(rig.moe_layers)):
        for seen in hits[layer]:
            count = len(seen)
            recompute = log.events[cursor : cursor + count]
            backward = log.events[cursor + count : cursor + 2 * count]
            assert recompute == [(layer, expert, False) for expert in seen]
            assert all(idx == layer and inside for idx, _expert, inside in backward)
            assert Counter(expert for _idx, expert, _inside in backward) == Counter(seen)
            cursor += 2 * count
    assert cursor == len(log.events)
    record_property("backward_expert_reads", sum(inside for _idx, _expert, inside in log.events))
    record_property("forward_layer_order", [idx for idx, grad in calls if not grad])
    record_property("recompute_layer_order", [idx for idx, grad in calls if grad])
    record_comparison(record_property, want, got)


# ==========================================================================
# negative controls on the SAME inputs: the comparison is sensitive to what it claims to check
# ==========================================================================
def check_backward_only_perturbation(
    make_rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch, *, device: str = "cpu"
) -> Dict[str, Any]:
    """(a) Expert weights perturbed only on loads inside autograd's engine: the losses stay
    equal and the gradients differ, while the unperturbed run on the same inputs passes."""
    rig = make_rig(k=2, device=device)
    specs = [Spec(micro_batches(rig, 2))]
    (want,) = ga_windows(rig, specs)
    assert_signal(want)
    log = ReadLog(monkeypatch, rig.model)

    (clean,) = l2l_windows(rig, specs)
    assert_same(want, clean)
    assert log.perturbed == []

    log.perturb = perturb_down
    (bad,) = l2l_windows(rig, specs)
    assert log.perturbed, "no load ran inside autograd's engine"
    assert {layer for layer, _expert in log.perturbed} == set(rig.moe_layers)
    assert losses_equal(want, bad)  # the forward and the recompute read the true weights
    assert grads_differ(bad.grads, want.grads)
    with pytest.raises(AssertionError, match="LoRA gradients differ"):
        assert_same(want, bad)

    # the perturbation left no trace in the cache file, the staging or the arena
    log.perturb = None
    (again,) = l2l_windows(rig, specs)
    assert_same(want, again)
    return {
        "max_abs_grad_diff": max_grad_difference(want, bad),
        "different_lora_grad_tensors": len(grads_differ(bad.grads, want.grads)),
        "backward_perturbed_reads": len(log.perturbed),
        "losses_equal": losses_equal(want, bad),
        "rng_equal": rng_equal(want.rng, bad.rng),
        "clean_repeat_exact": True,
    }


def test_negative_control_perturbing_only_backward_reloads_changes_grads_not_losses(
    make_rig, monkeypatch, record_property
):
    for key, value in check_backward_only_perturbation(make_rig, monkeypatch).items():
        record_property(key, value)


def no_routing_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test-only: stand the routing guard down, so a control can see the gradients themselves."""
    for name in ("record", "check"):
        monkeypatch.setattr(l2l_step.RoutingGuard, name, lambda self, layer, mb: nullcontext())


def without_replayed_state_on_the_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    real = l2l_rng.state_at

    def forward_only(state):
        return nullcontext() if torch.is_grad_enabled() else real(state)

    monkeypatch.setattr("soup_cli.utils.l2l_rng.state_at", forward_only)


def check_broken_rng_replay(
    make_rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch, *, device: str = "cpu"
) -> Dict[str, Any]:
    """(b) At dropout 0.05 the recompute without the replayed generator state draws other masks.

    The guard may stop this run when routing changes, so this control disables only that
    check: the gradients must differ while the forward's losses stay equal.
    """
    rig = make_rig(k=2, dropout=0.05, device=device)
    specs = [Spec(micro_batches(rig, 2))]
    (want,) = ga_windows(rig, specs)
    (clean,) = l2l_windows(rig, specs)
    assert_same(want, clean)

    with monkeypatch.context() as broken:
        without_replayed_state_on_the_recompute(broken)
        no_routing_guard(broken)
        (bad,) = l2l_windows(rig, specs)
    assert losses_equal(want, bad)  # the forward still runs at its state
    assert grads_differ(bad.grads, want.grads)
    with pytest.raises(AssertionError, match="LoRA gradients differ"):
        assert_same(want, bad, stream=False)
    (again,) = l2l_windows(rig, specs)
    assert_same(want, again)
    return {
        "max_abs_grad_diff": max_grad_difference(want, bad),
        "different_lora_grad_tensors": len(grads_differ(bad.grads, want.grads)),
        "losses_equal": losses_equal(want, bad),
        "clean_repeat_exact": True,
    }


def test_negative_control_state_not_applied_on_the_recompute_changes_the_grads(
    make_rig, monkeypatch, record_property
):
    for key, value in check_broken_rng_replay(make_rig, monkeypatch).items():
        record_property(key, value)


def assert_guard_closed(guard: Any) -> None:
    assert not guard._records
    assert not guard._handles
    assert guard._visit is None


@pytest.fixture()
def guard_runs(monkeypatch):
    """Observe real per-run guards without changing routing or expert execution."""
    guards, checked = [], []
    real = l2l_step.RoutingGuard

    class ObservedGuard(real):
        def __init__(self, layers):
            super().__init__(layers)
            guards.append(self)

        @contextmanager
        def check(self, layer, mb):
            try:
                with super().check(layer, mb):
                    yield
            finally:
                checked.append(((layer, mb), (layer, mb) not in self._records))

    monkeypatch.setattr(l2l_step, "RoutingGuard", ObservedGuard)
    return SimpleNamespace(guards=guards, checked=checked)


def expert_hooks(rig: Rig) -> Dict[int, tuple]:
    return {
        idx: tuple(module._forward_pre_hooks.items())
        for idx, module in streamed_experts_modules(rig.model).items()
    }


def test_the_routing_guard_stops_a_recompute_that_drew_other_dropout_masks(
    make_rig, monkeypatch, guard_runs
):
    """Broken replay changes routing; failure drops records/hooks, and the next run is exact."""
    rig = make_rig(k=2, dropout=0.05)
    specs = [Spec(micro_batches(rig, 2))]
    (want,) = ga_windows(rig, specs)
    hooks = expert_hooks(rig)
    with monkeypatch.context() as broken:
        without_replayed_state_on_the_recompute(broken)
        mismatch = r"routing guard: layer \d micro-batch \d was routed"
        with pytest.raises(RuntimeError, match=mismatch):
            l2l_windows(rig, specs)
    assert_guard_closed(guard_runs.guards[-1])
    assert guard_runs.checked and all(released for _pair, released in guard_runs.checked)
    assert expert_hooks(rig) == hooks
    assert all(layer.use_checkpoint for layer in rig.stepper.layers)
    (clean,) = l2l_windows(rig, specs)
    assert_same(want, clean)
    assert_guard_closed(guard_runs.guards[-1])
    assert expert_hooks(rig) == hooks


@pytest.mark.parametrize("phase", ["forward", "recompute", "backward"])
def test_guard_cleans_up_arbitrary_expert_exceptions_and_the_stepper_can_run_again(
    make_rig, monkeypatch, guard_runs, phase
):
    rig = make_rig(k=2)
    specs = [Spec(micro_batches(rig, 2))]
    (want,) = ga_windows(rig, specs)
    hooks = expert_hooks(rig)
    reader = rig.runtime.expert_readers[-1]
    original = reader.read_host
    failed = []

    def fail_read(expert, staging):
        inside = torch._C._current_autograd_node() is not None
        # ExpertNode.forward itself disables grad even during the grad recompute; classify
        # the caller from the guard's real visit, not the custom Function's grad mode.
        recording = guard_runs.guards[-1]._visit[2] if not inside else None
        current = "backward" if inside else "forward" if recording else "recompute"
        if current == phase:
            failed.append(expert)
            raise OSError(f"deliberate {phase} expert read failure")
        return original(expert, staging)

    with monkeypatch.context() as broken:
        broken.setattr(reader, "read_host", fail_read)
        with pytest.raises(OSError, match=f"deliberate {phase}"):
            l2l_windows(rig, specs)
    assert failed
    assert_guard_closed(guard_runs.guards[-1])
    assert all(released for _pair, released in guard_runs.checked)
    assert expert_hooks(rig) == hooks
    assert all(layer.use_checkpoint for layer in rig.stepper.layers)
    (again,) = l2l_windows(rig, specs)
    assert_same(want, again)
    assert_guard_closed(guard_runs.guards[-1])
    assert expert_hooks(rig) == hooks


@pytest.mark.parametrize("change", ["choice", "shape", "extra-call", "missing-call"])
def test_guard_checks_actual_routing_before_loading_and_releases_each_pair(make_rig, change):
    rig = make_rig(k=1)
    (want,) = ga_windows(rig, [Spec(micro_batches(rig, 1))])
    layer = next(iter(rig.moe_layers))
    module = streamed_experts_modules(rig.model)[layer]
    route = want.routes[layer][0].clone()
    hidden = torch.randn(route.shape[0], rig.config.hidden_size)
    weights = torch.full(route.shape, 0.5)
    hooks = expert_hooks(rig)
    with l2l_step.RoutingGuard(rig.stepper.layers) as guard:
        with guard.record(layer, 0), torch.no_grad():
            module(hidden, route, weights)
        assert (layer, 0) in guard._records
        bad = route.clone()
        if change == "choice":
            bad[0, 0] = (bad[0, 0] + 1) % rig.config.n_routed_experts
        elif change == "shape":
            bad = bad[:, :1]
        before = module.reader.fetches
        message = "routed differently" if change in ("choice", "shape") else "more often"
        with pytest.raises(RuntimeError, match=message):
            with guard.check(layer, 0):
                if change != "missing-call":
                    module(hidden, bad, weights[:, : bad.shape[1]])
                if change == "extra-call":
                    before = module.reader.fetches
                    module(hidden, route, weights)
        assert module.reader.fetches == before
        assert (layer, 0) not in guard._records
        assert guard._visit is None
    assert_guard_closed(guard)
    assert expert_hooks(rig) == hooks


def test_guard_preserves_differentiable_router_weights(make_rig):
    """Detaching/replacing top-k weights in a guard must not erase their real gradient path."""
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    rig = make_rig(k=1)
    layer = next(iter(rig.moe_layers))
    resident_mlp = decoder_owner(rig.resident).layers[layer].mlp
    streamed_mlp = copy.deepcopy(resident_mlp)
    streamed_mlp.experts = streamed_experts_modules(rig.model)[layer]
    hidden = torch.randn(rig.rows, rig.seq, rig.config.hidden_size)

    def run(mlp):
        inputs = hidden.clone().requires_grad_(True)
        seen = []

        def save_weights(_module, _args, output):
            weights = output[1]
            assert weights.requires_grad and weights.grad_fn is not None
            weights.retain_grad()
            seen.append(weights)

        handle = mlp.gate.register_forward_hook(save_weights)
        try:
            out = mlp(inputs)
            out.backward(torch.ones_like(out))
        finally:
            handle.remove()
        assert len(seen) == 1 and bool(seen[0].grad.ne(0).any())
        return out.detach(), inputs.grad, seen[0].grad

    want = run(resident_mlp)
    with l2l_step.RoutingGuard(rig.stepper.layers) as guard:
        with guard.record(layer, 0), torch.no_grad():
            streamed_mlp(hidden)
        with guard.check(layer, 0):
            got = run(streamed_mlp)
        assert not guard._records
    assert all(torch.equal(a, b) for a, b in zip(want, got))
    assert_guard_closed(guard)


def test_public_launch_still_refuses_deepseek_but_internal_step_accepts_it(make_rig):
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    rig = make_rig(k=1)
    assert l2l_step.model_parts(rig.model)[0].config.model_type == "deepseek_v3"
    with pytest.raises(ValueError, match="deepseek_v3.*llama.*mistral"):
        refuse_unsupported_l2l(rig.config, n_gpu=1)


def test_expert_loads_remain_visible_in_trainer_logs_and_dense_fields_stay_unchanged(make_rig):
    from soup_cli.trainer.l2l_trainer import L2LState

    rig = make_rig(k=2)
    windows = l2l_windows(rig, [Spec(micro_batches(rig, 2)), Spec(micro_batches(rig, 1))])
    state = L2LState(runtime=rig.runtime, dtype=torch.float32)
    for window in windows:
        state.steps += 1
        state.totals.add(window.counters)
    fields = state.log_fields()
    assert fields["l2l_steps"] == 2
    assert fields["l2l_expert_fetches"] == sum(w.counters.expert_fetches for w in windows)
    assert fields["l2l_expert_bytes_read"] == sum(w.counters.expert_bytes_read for w in windows)
    assert fields["l2l_expert_fetches"] > 0 and fields["l2l_expert_bytes_read"] > 0
    state.runtime = SimpleNamespace(expert_pool=None)
    assert state.log_fields() == {
        "l2l_steps": 2,
        "l2l_layer_loads": state.totals.layer_loads,
        "l2l_large_loads": state.totals.large_loads,
        "l2l_weight_restores": state.totals.restores,
    }


def check_bf16_forward_amp(
    make_rig: Callable[..., Rig], *, k: int, device: str = "cpu"
) -> tuple:
    """Ambient AMP around L2L must not change backward reductions relative to HF's loop."""
    rig = make_rig(k=k, device=device)
    specs = [Spec(micro_batches(rig, k)), Spec(micro_batches(rig, k, first=10))]
    want = ga_windows(rig, specs, amp=True)
    got = l2l_windows(rig, specs, amp=True)
    for expected, actual in zip(want, got):
        assert_signal(expected)
        assert set(expected.grads) == set(actual.grads)
        for name, gradient in expected.grads.items():
            torch.testing.assert_close(actual.grads[name], gradient, atol=1e-6, rtol=1e-5)
        for expected_loss, actual_loss in zip(expected.losses, actual.losses):
            torch.testing.assert_close(actual_loss, expected_loss, atol=1e-6, rtol=1e-5)
        assert rng_equal(expected.rng, actual.rng)
    return want, got


@pytest.mark.parametrize("k", [1, 2, 4])
def test_bf16_forward_amp_keeps_backward_outside_autocast_like_resident_ga(
    make_rig, k, record_property
):
    want, got = check_bf16_forward_amp(make_rig, k=k)
    for expected, actual in zip(want, got):
        record_comparison(record_property, expected, actual)


def test_seeded_dropout_runs_repeat_exactly_and_another_seed_changes_gradients(
    make_rig, guard_runs
):
    rig = make_rig(k=2, dropout=0.05)
    specs = [Spec(micro_batches(rig, 2))]
    (want,) = ga_windows(rig, specs)
    (first,) = l2l_windows(rig, specs)
    (again,) = l2l_windows(rig, specs)
    (other,) = l2l_windows(rig, specs, seed=SEED + 1)
    assert_same(want, first)
    assert_same(first, again)
    assert grads_differ(first.grads, other.grads)
    assert not rng_equal(first.rng, other.rng)
    for guard in guard_runs.guards:
        assert_guard_closed(guard)


def check_eager_attention_dropout(make_rig: Callable[..., Rig], *, device: str = "cpu") -> tuple:
    rig = make_rig(k=2, dropout=0.05, attention_dropout=0.1, device=device)
    specs = [Spec(micro_batches(rig, 2)), Spec(micro_batches(rig, 2, first=10))]
    want = ga_windows(rig, specs)
    got = l2l_windows(rig, specs)
    assert l2l_rng.has_draws(rig.stepper.call.draws)
    assert any(draw.p == 0.1 for layer in rig.stepper.call.draws for draw in layer)
    for expected, actual in zip(want, got):
        assert_signal(expected)
        assert_routing_is_varied(expected, rig, 2)
        assert_same(expected, actual)
    return want, got


def test_eager_attention_and_lora_dropout_replay_match_resident_accumulation(
    make_rig, record_property
):
    want, got = check_eager_attention_dropout(make_rig)
    for expected, actual in zip(want, got):
        record_comparison(record_property, expected, actual)
