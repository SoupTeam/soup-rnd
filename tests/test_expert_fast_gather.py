"""``fast_row_gather``: the expert loop gathers its rows with a light autograd function.

``hidden_states[token_idx]`` and ``top_k_weights[token_idx, top_k_pos, None]`` backpropagate
through ``index_put_(accumulate=True)``, which sorts the indices and launches a dozen kernels
(about 150 us of host time per call, two calls per expert visit). Every token reaches an expert
once, so the indices are unique and the backward is a scatter into zeros: one ``index_add_``. These
tests pin that it gives the same bits, including the sign of zero, and that the option is off by
default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.test_b1_expert_forward import _experts, _inputs, _store, _streamed
from tests.test_b13_expert_backward import _assert_exact, _close, _run

torch = pytest.importorskip("torch")

if TYPE_CHECKING:
    from pathlib import Path


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def graph_names(root: object) -> set[str]:
    seen, names, stack = set(), set(), [root]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.add(type(node).__name__)
        stack.extend(child for child, _ in node.next_functions)
    return names


def row_functions() -> tuple[type, type]:
    from soup_cli.utils.expert_stream import fast_gather_functions

    return fast_gather_functions()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_gather_matches_indexing_forward_and_backward_bit_for_bit(dtype: torch.dtype) -> None:
    gather_rows, _ = row_functions()
    torch.manual_seed(0)
    source = torch.randn(40, 16).to(dtype)
    index = torch.randperm(40)[:17]
    grad = torch.randn(17, 16).to(dtype)
    grad[3, :4] = -0.0                       # a zero gradient keeps its sign through 0 + (-0) = +0
    grad[5, 4:8] = 0.0

    want_source = source.clone().requires_grad_(True)
    want = want_source[index]
    want.backward(grad)
    got_source = source.clone().requires_grad_(True)
    got = gather_rows.apply(got_source, index)
    got.backward(grad)

    assert torch.equal(bits(got.detach()), bits(want.detach()))
    assert torch.equal(bits(got_source.grad), bits(want_source.grad))


def test_row_gather_without_a_grad_source_builds_no_node() -> None:
    gather_rows, _ = row_functions()
    out = gather_rows.apply(torch.randn(6, 4), torch.tensor([1, 3]))
    assert out.grad_fn is None


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_route_weight_gather_matches_indexing_forward_and_backward_bit_for_bit(top_k: int) -> None:
    _, gather_weights = row_functions()
    torch.manual_seed(1)
    tokens = 30
    weights = torch.rand(tokens, top_k)
    flat = torch.randperm(tokens * top_k)[:11]     # unique (token, slot) pairs, like one expert
    token_idx, top_k_pos = flat // top_k, flat % top_k
    grad = torch.randn(11, 1)
    grad[2] = -0.0

    want_w = weights.clone().requires_grad_(True)
    want = want_w[token_idx, top_k_pos, None]
    want.backward(grad)
    got_w = weights.clone().requires_grad_(True)
    got = gather_weights.apply(got_w, token_idx, top_k_pos)
    got.backward(grad)

    assert got.shape == want.shape
    assert torch.equal(bits(got.detach()), bits(want.detach()))
    assert torch.equal(bits(got_w.grad), bits(want_w.grad))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("routing", [dict(count=24, top_k=2), dict(count=24, top_k=1),
                                     dict(count=24, partial=True)])
def test_streamed_experts_with_fast_gather_equal_the_default_and_skip_index_backward(
    tmp_path: Path, dtype: str, routing: dict
) -> None:
    module, _ = _experts()
    module = module.to(dtype=getattr(torch, dtype))
    reader, pool = _store(tmp_path, module, nf4=False, nested=False, device="cpu")
    try:
        args = _inputs(device="cpu", dtype=getattr(torch, dtype), **routing)
        streamed = _streamed(module, reader, pool)
        assert streamed.fast_row_gather is False       # off unless asked for
        want = _run(streamed, args)
        hidden = args[0].clone().requires_grad_(True)
        weights = args[2].clone().requires_grad_(True)
        assert "IndexBackward0" in graph_names(streamed(hidden, args[1], weights).grad_fn)

        streamed.fast_row_gather = True
        got = _run(streamed, args)
        _assert_exact(want, got)
        hidden = args[0].clone().requires_grad_(True)
        weights = args[2].clone().requires_grad_(True)
        assert "IndexBackward0" not in graph_names(streamed(hidden, args[1], weights).grad_fn)
    finally:
        _close(reader, pool)


def test_row_gather_with_a_repeated_index_is_bit_equal_in_fp32_and_only_close_in_bf16() -> None:
    """Indices are unique in the expert loop (a token reaches an expert once), and the docstring
    says so. The repeated case is outside the bit-equality claim: on CPU the sum is bit-equal to
    the indexing backward in fp32 but rounds differently in bf16, and on CUDA the order of atomics
    is not defined either."""
    gather_rows, _ = row_functions()
    torch.manual_seed(2)
    index = torch.tensor([3, 1, 3, 7, 3])
    for dtype in (torch.float32, torch.bfloat16):
        source = torch.randn(10, 8).to(dtype)
        grad = torch.randn(5, 8).to(dtype)
        want_source = source.clone().requires_grad_(True)
        want_source[index].backward(grad)
        got_source = source.clone().requires_grad_(True)
        gather_rows.apply(got_source, index).backward(grad)
        if dtype == torch.float32:
            assert torch.equal(bits(got_source.grad), bits(want_source.grad))
        else:
            assert torch.allclose(got_source.grad.float(), want_source.grad.float(), atol=0.05)
