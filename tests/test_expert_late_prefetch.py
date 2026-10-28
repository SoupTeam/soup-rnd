"""Late lookahead ordering, replay refusal and complete real-CUDA trajectories."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _case(mode: str, two_reads: bool, grouped: bool = True) -> tuple[Any, ...]:
    import torch
    import torch.nn.functional as functional

    from soup_cli.utils.expert_stream import streamed_experts_class

    generator = torch.Generator().manual_seed(41)

    class Fetcher:
        def __init__(self) -> None:
            self.transfer_mode = mode
            self.events: list[tuple[str, Any]] = []
            self.pending = None
            self.fail_fetch = self.fail_prefetch = self.fail_linear = None
            self.weights = [
                (torch.randn(6, 4, generator=generator),
                 torch.randn(4, 3, generator=generator)) for _ in range(3)
            ]

        def fetch(self, _reader: Any, expert: int) -> tuple[Any, Any]:
            self.events.append(("fetch", expert))
            if expert == self.fail_fetch:
                raise RuntimeError("fetch failure")
            self.pending = None
            return self.weights[expert]

        def prefetch(self, _reader: Any, expert: int) -> None:
            self.events.append(("prefetch", expert))
            self.pending = expert
            if expert == self.fail_prefetch:
                raise RuntimeError("prefetch failure")

        def cancel_prefetch(self) -> None:
            self.events.append(("cancel", None))
            self.pending = None

    fetcher = Fetcher()
    module = streamed_experts_class()(
        SimpleNamespace(count=3), fetcher, functional.silu, 1,
        two_reads=two_reads, grouped_expert_routes=grouped,
    )
    hidden = torch.randn(3, 4, generator=generator)
    indices = torch.tensor([[2, 0], [1, 2], [0, 1]], dtype=torch.int64)
    weights = torch.randn(3, 2, generator=generator).softmax(-1)
    gradient = torch.randn(3, 4, generator=generator)
    return module, fetcher, hidden, indices, weights, gradient


def _observe_linear(monkeypatch: pytest.MonkeyPatch, fetcher: Any) -> None:
    import torch.nn.functional as functional

    original = functional.linear
    identities = {weight.data_ptr(): (expert, part)
                  for expert, pair in enumerate(fetcher.weights)
                  for part, weight in zip(("gate_up", "down"), pair)}

    def linear(hidden: Any, weight: Any, bias: Any = None) -> Any:
        expert, part = identities[weight.data_ptr()]
        fetcher.events.append((part, expert))
        if part == "down" and expert == fetcher.fail_linear:
            raise RuntimeError("linear failure")
        return original(hidden, weight, bias)

    monkeypatch.setattr(functional, "linear", linear)


@pytest.mark.parametrize("two_reads", (False, True))
@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize("mode", ("prefetch", "prefetch_after"))
def test_late_prefetch_follows_current_mlp_and_never_runs_in_backward(
    monkeypatch: pytest.MonkeyPatch, two_reads: bool, grouped: bool, mode: str,
) -> None:
    module, fetcher, hidden, indices, weights, gradient = _case(mode, two_reads, grouped)
    _observe_linear(monkeypatch, fetcher)
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    output = module(hidden, indices, weights)
    expected = []
    for expert in range(3):
        expected.append(("fetch", expert))
        if mode == "prefetch" and expert < 2:
            expected.append(("prefetch", expert + 1))
        expected += [("gate_up", expert), ("down", expert)]
        if mode == "prefetch_after" and expert < 2:
            expected.append(("prefetch", expert + 1))
    expected.append(("cancel", None))
    assert fetcher.events == expected
    assert fetcher.pending is None and module._next_expert is None
    output.backward(gradient)
    assert [value for kind, value in fetcher.events if kind == "prefetch"] == [1, 2]
    assert [value for kind, value in fetcher.events if kind == "fetch"] == [0, 1, 2, 2, 1, 0]


@pytest.mark.parametrize("two_reads", (False, True))
@pytest.mark.parametrize("autocast", (False, True))
def test_late_prefetch_preserves_cpu_outputs_input_and_routing_gradients(
    two_reads: bool, autocast: bool,
) -> None:
    import torch

    captures = []
    for mode in ("sync", "prefetch_after"):
        module, _fetcher, hidden, indices, weights, gradient = _case(mode, two_reads)
        hidden.requires_grad_(True)
        weights.requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            output = module(hidden, indices, weights)
        output.backward(gradient)
        captures.append((output.detach(), hidden.grad, weights.grad))
    assert all(torch.equal(left, right) for left, right in zip(*captures))
    assert all(torch.count_nonzero(value).item() > 0 for value in captures[0])


def test_late_prefetch_skips_two_read_replay_and_backward() -> None:
    import torch

    from soup_cli.utils.expert_stream import two_read_checkpoint_contexts

    module, fetcher, hidden, indices, weights, gradient = _case("prefetch_after", True)
    forward, replay = two_read_checkpoint_contexts()
    with forward, torch.no_grad():
        module(hidden, indices, weights)
    fetcher.events.clear()
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    with replay:
        output = module(hidden, indices, weights)
    assert fetcher.events == []
    assert torch.count_nonzero(output).item() == 0
    assert module._next_expert is None
    output.backward(gradient)
    assert fetcher.events == [("fetch", 2), ("fetch", 1), ("fetch", 0)]
    assert torch.count_nonzero(hidden.grad).item() > 0
    assert torch.count_nonzero(weights.grad).item() > 0


@pytest.mark.parametrize("failure", ("fetch", "prefetch", "linear"))
@pytest.mark.parametrize("two_reads", (False, True))
def test_late_prefetch_cancels_pending_work_on_exceptions(
    monkeypatch: pytest.MonkeyPatch, failure: str, two_reads: bool,
) -> None:
    module, fetcher, hidden, indices, weights, _gradient = _case("prefetch_after", two_reads)
    _observe_linear(monkeypatch, fetcher)
    setattr(fetcher, "fail_" + failure, 0 if failure == "linear" else 1)
    with pytest.raises(RuntimeError, match=failure + " failure"):
        module(hidden, indices, weights)
    assert fetcher.events[-1] == ("cancel", None)
    assert fetcher.pending is None and module._next_expert is None
    if failure == "linear":
        assert not any(kind == "prefetch" for kind, _value in fetcher.events)
    else:
        assert ("prefetch", 1) in fetcher.events


def test_late_prefetch_empty_routes_stage_nothing_and_reset_state() -> None:
    module, fetcher, hidden, indices, weights, _gradient = _case("prefetch_after", True)
    output = module(hidden[:0], indices[:0], weights[:0])
    assert output.shape == (0, 4)
    assert fetcher.events == [("cancel", None)]
    assert fetcher.pending is None and module._next_expert is None


@pytest.mark.gpu(reason="Actual late lookahead NF4/BF16 optimizer parity requires CUDA")
def test_real_cuda_late_prefetch_preserves_kanana_full_optimizer_trajectories(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, record_property: Any,
) -> None:
    from tests.test_expert_transfer import (
        test_real_cuda_transfer_preserves_kanana_full_optimizer_trajectories as trajectories,
    )

    # Same real checkpoint/LoRA/AdamW capture harness and numerical gates as the
    # qualified packed/prefetch modes, including two-read/cache combinations.
    trajectories(tmp_path, monkeypatch, "prefetch_after", record_property)
