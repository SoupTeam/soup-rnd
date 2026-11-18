"""Tests of MBR window boundaries using a deterministic streamed-expert stand-in."""

from contextlib import contextmanager

import pytest
import torch

from soup_cli.utils import l2l_step


class _Experts:
    last_trace = [0, 1]
    _mbr_materialize_recompute = False

    def __init__(self, identity=False):
        self.identity = identity

    def __call__(self, hidden, ids, weights):
        if self.identity:
            return hidden
        factor = ((ids.to(hidden.dtype) + 1) * weights).sum(dim=-1, keepdim=True)
        return hidden * factor


def _run_window(monkeypatch, inputs, group_size, identity=False, bad_routes=False, fixed=False):
    experts = _Experts(identity)
    capture = []
    wrapper_inputs = []

    @contextmanager
    def patched(_module, destination):
        capture.append(destination)
        try:
            yield
        finally:
            capture.pop()

    def layer(_module, hidden, _state, _call):
        wrapper_inputs.append(((hidden,), {}))
        flat = hidden.reshape(-1, hidden.shape[-1])
        ids = torch.stack((flat[:, 0].long() % 2, (flat[:, 0].long() + 1) % 2), -1)
        if bad_routes:
            ids = ids[:-1]
        weights = torch.softmax(torch.stack((flat[:, 1], -flat[:, 1]), -1), -1)
        if fixed:
            weights = weights.detach()
        if bad_routes:
            weights = weights[:-1]
        capture[-1].append(
            {
                "residual": torch.zeros_like(hidden),
                "expert_input": flat,
                "route_ids": ids,
                "route_weights": weights,
                "shared_output": None,
            }
        )

    monkeypatch.setattr(l2l_step, "_streamed_moe_experts", lambda _module: experts)
    monkeypatch.setattr(l2l_step, "_patched_route_preserving_forward", patched)
    step = object.__new__(l2l_step.L2LStep)
    monkeypatch.setattr(step, "_layer", layer)
    return step._route_preserving_layer_window(
        object(), inputs, None, [None] * len(inputs), group_size, ({}, None, wrapper_inputs)
    )


def _inputs():
    return [
        torch.tensor([[[float(10 * i + j), 0.1 * (j + 1)] for j in range(n)]],
                     requires_grad=True)
        for i, n in enumerate((1, 3, 2, 5))
    ]


def test_mbr_ragged_identity_preserves_markers(monkeypatch):
    inputs = _inputs()
    outputs, routes, groups, stats = _run_window(monkeypatch, inputs, 3, identity=True)
    assert groups == [[0, 1, 2], [3]]
    assert stats["routed_tokens"] == 11
    for x, y, route in zip(inputs, outputs, routes):
        assert torch.equal(x, y)
        assert route[0].shape == (x.shape[1], 2)


@pytest.mark.parametrize("fixed", [True, False])
def test_mbr_ragged_outputs_and_router_gradients_match_microbatch(monkeypatch, fixed):
    inputs_a, inputs_b = _inputs(), _inputs()
    a, routes_a, _, _ = _run_window(monkeypatch, inputs_a, 1, fixed=fixed)
    b, routes_b, _, _ = _run_window(monkeypatch, inputs_b, 4, fixed=fixed)
    for i, (ya, yb) in enumerate(zip(a, b)):
        torch.testing.assert_close(ya, yb)
        assert torch.equal(routes_a[i][0], routes_b[i][0])
        torch.testing.assert_close(routes_a[i][1], routes_b[i][1])
    sum((i + 1) * y.square().sum() for i, y in enumerate(a)).backward()
    sum((i + 1) * y.square().sum() for i, y in enumerate(b)).backward()
    for xa, xb in zip(inputs_a, inputs_b):
        torch.testing.assert_close(xa.grad, xb.grad)


def test_mbr_rejects_invalid_ragged_route_count(monkeypatch):
    with pytest.raises(RuntimeError, match="route shape"):
        _run_window(monkeypatch, _inputs(), 4, bad_routes=True)
