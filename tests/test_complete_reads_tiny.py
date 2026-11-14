"""Ticket 03: expert reads and exactness of ordinary and complete streaming, tiny DeepSeek, CPU.

Question. Does complete streaming lower the number of expert reads of a window, and do the
reads grow with the number of micro-batches k, on gradient accumulation (GA) and on L2L?

Arms (all on the tiny native DeepSeek-V3 of B1.3, per-expert shards, fp32):
  * ``ordinary``: defaults, three reads of each hit expert (forward, recompute, backward);
  * ``grouped``: ``grouped_expert_routes=True``;
  * ``two_read``: ``two_read_experts=True``;
  * ``complete_cpu``: both levers above. This is every lever that runs on a CPU.
Not run here: ``expert_transfer="packed"`` and ``expert_host_cache_bytes`` need CUDA and NF4
(``test_packed_levers_are_refused_on_cpu`` records the refusal), so the full complete arm is
not measured and nothing below says anything about its speed.

Modes. ``ga``: k forward+backward calls on the streamed model with ``num_items_in_batch`` over
the window. ``l2l``: ``L2LStep.run`` over the same window.

Counted per window: layer loads, expert fetches (``LoadCounters``), expert hits and unique
experts (an independent resident eager router oracle), and physical requests
(``read_range_into`` calls, as in ``test_p3_expert_read_diagnostic.py``). Time is not measured.

Exactness. After the window one AdamW step runs on the LoRA parameters of every arm. Each arm's
gradients, parameters and AdamW state are compared with ``torch.equal`` against ``ordinary`` of
the same mode and k. On CPU the tolerance is zero. Any difference stops the ticket: the
assertion names the mode, k, arm and tensor.

Set ``COMPLETE_READS_JSON`` to a path to get the measured table as JSON from
``test_table``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from soup_cli.utils import l2l_step
from tests.test_b13_l2l_moe import (  # noqa: F401  (make_rig is a fixture)
    Rig,
    Spec,
    ga_windows,
    hits_of,
    make_rig,
    micro_batches,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_p3_expert_read_diagnostic import PhysicalReads

pytestmark = _REQUIRES_TRAINING_FLOOR

K_SWEEP = (1, 4, 16, 64)
MODES = ("ga", "l2l")
ARMS: Dict[str, Dict[str, Any]] = {
    "ordinary": {},
    "grouped": {"grouped_expert_routes": True},
    "two_read": {"two_read_experts": True},
    "complete_cpu": {"grouped_expert_routes": True, "two_read_experts": True},
}
SEED = 11
LEARNING_RATE = 1e-2
READS_PER_HIT_DEFAULT = 3
READS_PER_HIT_TWO_READ = 2


def reads_per_hit(mode: str, arm: str) -> int:
    """Two-read lowers the reads only where a checkpoint recompute exists, which L2L bypasses."""
    two_read = ARMS[arm].get("two_read_experts", False)
    return READS_PER_HIT_TWO_READ if two_read and mode == "ga" else READS_PER_HIT_DEFAULT


def expected_layer_loads(mode: str, k: int) -> int:
    """Three decoder layers: loaded once per micro-batch on GA and once per window on L2L."""
    return 3 * k if mode == "ga" else 3


def routers_patched(model: Any) -> int:
    """Routers whose ``forward`` is the two-read replay wrapper."""
    from soup_cli.utils import expert_stream

    return sum(
        getattr(vars(module).get("forward"), "__func__", None) is expert_stream._routing_forward
        for module in model.modules()
    )


def ga_streamed(rig: Rig, spec: Spec) -> Tuple[List[Any], Any]:
    """Plain accumulation on the streamed model: k forward+backward calls over one window."""
    import torch

    model = rig.model
    model.train()
    torch.manual_seed(SEED)
    for param in fixture.trainables(model).values():
        param.grad = None
    labels, total = spec.reference()
    losses = []
    with l2l_step.count_loads(rig.runtime) as counters:
        for ids, lab in zip(spec.ids, labels):
            loss = model(input_ids=ids, labels=lab, num_items_in_batch=total).loss
            loss.backward()
            losses.append(loss.detach())
    return losses, counters


def l2l_streamed(rig: Rig, spec: Spec) -> Tuple[List[Any], Any]:
    import torch

    model = rig.model
    model.train()
    torch.manual_seed(SEED)
    for param in fixture.trainables(model).values():
        param.grad = None
    losses = rig.stepper.run(spec.ids, rig.store, n_items=spec.n_items, labels=spec.labels)
    return [loss.detach() for loss in losses], rig.stepper.counters


def adamw_update(model: Any) -> Dict[str, Dict[str, Any]]:
    """One AdamW step on the LoRA parameters. Returns the gradients, the parameters after the
    step and the optimizer state, all as detached clones keyed by parameter name."""
    import torch

    params = fixture.trainables(model)
    grads = {n: p.grad.detach().clone() for n, p in params.items() if p.grad is not None}
    optimizer = torch.optim.AdamW(list(params.values()), lr=LEARNING_RATE)
    optimizer.step()
    state: Dict[str, Any] = {}
    for name, param in params.items():
        for key, value in optimizer.state[param].items():
            state[f"{name}::{key}"] = value.detach().clone()
    return {
        "grads": grads,
        "params": {n: p.detach().clone() for n, p in params.items()},
        "state": state,
    }


def measure(
    rig_factory: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch, mode: str, arm: str, k: int,
    oracle: Dict[str, int],
) -> Dict[str, Any]:
    rig = rig_factory(k=k, build_options=ARMS[arm])
    spec = Spec(micro_batches(rig, k))
    physical = PhysicalReads(monkeypatch, list(rig.runtime.expert_readers))
    from soup_cli.utils import expert_stream

    grouped_calls = [0]
    original_grouped = expert_stream._grouped_expert_routes

    def spy(expert_mask: Any) -> Any:
        grouped_calls[0] += 1
        return original_grouped(expert_mask)

    monkeypatch.setattr(expert_stream, "_grouped_expert_routes", spy)
    losses, counters = (ga_streamed if mode == "ga" else l2l_streamed)(rig, spec)
    update = adamw_update(rig.model)
    return {
        "mode": mode,
        "arm": arm,
        "k": k,
        "layer_loads": counters.layer_loads,
        "expert_fetches": counters.expert_fetches,
        "expert_bytes_read": counters.expert_bytes_read,
        "total_hits": oracle["total_hits"],
        "unique_experts": oracle["unique_experts"],
        "physical_calls": physical.calls,
        "physical_bytes": physical.bytes,
        "grouped_calls": grouped_calls[0],
        "routers_patched": routers_patched(rig.model),
        "_losses": losses,
        "_update": update,
    }


def oracle_for(rig: Rig, k: int) -> Dict[str, int]:
    """Hits and unique experts of the window from an independent resident eager model."""
    spec = Spec(micro_batches(rig, k))
    (want,) = ga_windows(rig, [spec])
    hits = hits_of(want)
    return {
        "total_hits": sum(len(seen) for per_mb in hits.values() for seen in per_mb),
        "unique_experts": sum(
            len({expert for seen in per_mb for expert in seen}) for per_mb in hits.values()
        ),
    }


_CELLS: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
_ORACLES: Dict[int, Dict[str, int]] = {}


@pytest.fixture()
def cell(
    make_rig: Callable[..., Rig],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, str, int], Dict[str, Any]]:
    """``cell(mode, arm, k)``: one window measured once per cell for the whole module."""

    def run(mode: str, arm: str, k: int) -> Dict[str, Any]:
        if k not in _ORACLES:
            _ORACLES[k] = oracle_for(make_rig(k=k), k)
        key = (mode, arm, k)
        if key not in _CELLS:
            _CELLS[key] = measure(make_rig, monkeypatch, mode, arm, k, _ORACLES[k])
        return _CELLS[key]

    return run


def _public(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def assert_bit_equal(want: Dict[str, Any], got: Dict[str, Any], where: str) -> None:
    """``torch.equal`` on every tensor; the message names the first tensor that differs."""
    import torch

    assert set(got) == set(want) and want, f"{where}: tensor sets differ"
    for name in sorted(want):
        assert torch.equal(want[name], got[name]), f"{where}: tensor {name!r} differs"


def test_the_comparison_names_a_tensor_that_differs() -> None:
    """Negative control: one changed element fails, and the message names the tensor."""
    import torch

    want = {"a": torch.ones(3), "b": torch.zeros(2)}
    assert_bit_equal(want, {name: tensor.clone() for name, tensor in want.items()}, "same")
    changed = {name: tensor.clone() for name, tensor in want.items()}
    changed["b"][1] = torch.finfo(torch.float32).tiny
    with pytest.raises(AssertionError, match="tensor 'b' differs"):
        assert_bit_equal(want, changed, "changed")


@pytest.mark.parametrize("k", K_SWEEP)
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("mode", MODES)
def test_reads_follow_the_lever_and_the_update_is_bit_equal_to_ordinary(
    cell: Callable[[str, str, int], Dict[str, Any]],
    record_property: Callable[..., None], mode: str, arm: str, k: int,
) -> None:
    import torch

    row, base = cell(mode, arm, k), cell(mode, "ordinary", k)
    for name, value in _public(row).items():
        record_property(name, value)
    where = f"mode={mode} k={k} arm={arm}"
    # Reads against the independent router oracle.
    assert row["total_hits"] >= row["unique_experts"] > 0, where
    assert row["expert_fetches"] == reads_per_hit(mode, arm) * row["total_hits"], (
        f"{where}: {row['expert_fetches']} fetches for {row['total_hits']} hits"
    )
    assert row["layer_loads"] == expected_layer_loads(mode, k), where
    # The arm runs the lever it names, also where the lever changes no count.
    assert (row["grouped_calls"] > 0) == ARMS[arm].get("grouped_expert_routes", False), where
    assert row["routers_patched"] == (
        2 if ARMS[arm].get("two_read_experts", False) else 0
    ), where
    # Two stored tensors per expert, one request each.
    assert row["physical_calls"] == 2 * row["expert_fetches"], where
    assert row["physical_bytes"] >= row["expert_bytes_read"] > 0, where
    # Loss, gradients, parameters and AdamW state equal ordinary, bit for bit.
    assert len(row["_losses"]) == len(base["_losses"]) == k, where
    for mb, (a, b) in enumerate(zip(base["_losses"], row["_losses"])):
        assert torch.equal(a, b), f"{where}: loss of micro-batch {mb} differs"
    for part in ("grads", "params", "state"):
        assert_bit_equal(base["_update"][part], row["_update"][part], f"{where} {part}")
    # The comparison is not about zeros: the reference gradients carry signal.
    assert all(g.ne(0).any() for g in base["_update"]["grads"].values()), where


@pytest.mark.parametrize("mode", MODES)
def test_expert_reads_grow_with_k_on_every_arm(
    cell: Callable[[str, str, int], Dict[str, Any]],
    record_property: Callable[..., None], mode: str,
) -> None:
    """The hypothesis: reads follow k, with ordinary and with complete."""
    for arm in ARMS:
        fetches = [cell(mode, arm, k)["expert_fetches"] for k in K_SWEEP]
        loads = {cell(mode, arm, k)["layer_loads"] for k in K_SWEEP}
        record_property(f"{arm}_fetches_by_k", fetches)
        assert all(after > before for before, after in zip(fetches, fetches[1:])), (
            mode, arm, fetches,
        )
        per_micro_batch = [f / k for f, k in zip(fetches, K_SWEEP)]
        assert min(per_micro_batch) > 0.5 * max(per_micro_batch), (mode, arm, per_micro_batch)
        if mode == "l2l":
            assert len(loads) == 1, (mode, arm, loads)


@pytest.mark.parametrize("k", K_SWEEP)
def test_complete_lowers_the_reads_by_two_thirds_on_ga_and_not_on_l2l(
    cell: Callable[[str, str, int], Dict[str, Any]],
    record_property: Callable[..., None], k: int,
) -> None:
    """Ratio of complete to ordinary reads: 2/3 on GA, 1 on L2L (ticket 01, option A)."""
    ga_ratio = cell("ga", "complete_cpu", k)["expert_fetches"] / cell("ga", "ordinary", k)[
        "expert_fetches"
    ]
    l2l_ratio = cell("l2l", "complete_cpu", k)["expert_fetches"] / cell("l2l", "ordinary", k)[
        "expert_fetches"
    ]
    record_property("ga_ratio", ga_ratio)
    record_property("l2l_ratio", l2l_ratio)
    assert ga_ratio == pytest.approx(2 / 3)
    assert l2l_ratio == 1.0


@pytest.mark.parametrize(
    "options",
    (
        {"expert_transfer": "packed"},
        {"expert_transfer": "packed", "expert_host_cache_bytes": 1 << 20},
    ),
    ids=("packed", "packed+host-cache"),
)
def test_packed_levers_are_refused_on_cpu(
    make_rig: Callable[..., Rig], options: Dict[str, Any],  # noqa: F811
) -> None:
    """The levers that need a card cannot be measured here; they refuse instead of being ignored."""
    with pytest.raises(ValueError):
        make_rig(k=1, build_options=options)


def test_table(
    cell: Callable[[str, str, int], Dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [_public(cell(mode, arm, k)) for mode in MODES for arm in ARMS for k in K_SWEEP]
    header = (  # the spy and patch counters stay in the JSON and in the record_property data
        "mode arm k layer_loads expert_fetches total_hits unique_experts "
        "physical_calls physical_bytes"
    ).split()
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(row[name]) for name in header) + " |" for row in rows]
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    target: Optional[str] = os.environ.get("COMPLETE_READS_JSON")
    if target:
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
    assert len(rows) == len(MODES) * len(ARMS) * len(K_SWEEP)
