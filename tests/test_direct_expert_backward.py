"""Direct expert backward (harness only, idea H3 of the DeepSeek P3 ``separate_fast`` report).

The two-read expert node's backward recomputes ``expert_mlp(hidden) * weights`` under autograd
and differentiates it. ``expert_vjp`` computes the same gradients with the same ops, per segment
where the segmented plan cuts the calls, without building a graph. These CPU tests compare it
with autograd over the segmented expression bit for bit; the GPU check is in the micro script.
"""

from __future__ import annotations

from typing import Any, Callable, List

import pytest

torch = pytest.importorskip("torch")
functional = torch.nn.functional

from benchmarks.harness.direct_expert_backward import expert_vjp, segment_mm  # noqa: E402


def segmented_result(x: Any, w: Any, gate_up: Any, down: Any, counts: List[int]) -> Any:
    """What the segmented swap runs inside ``expert_mlp(hidden) * weights``."""
    h = torch.cat([functional.linear(p, gate_up) for p in x.split(counts)])
    gate, up = h.chunk(2, dim=-1)
    a = functional.silu(gate) * up
    y = torch.cat([functional.linear(p, down) for p in a.split(counts)])
    return torch.cat([p * q for p, q in zip(y.split(counts), w.split(counts))])


def autograd_grads(x: Any, w: Any, gate_up: Any, down: Any, g: Any, counts: List[int],
                   need: tuple, autocast: Any = None) -> List[Any]:
    lx = x.detach().requires_grad_(need[0])
    lw = w.detach().requires_grad_(need[1])
    ctx = autocast if autocast is not None else torch.autocast("cpu", enabled=False)
    with torch.enable_grad(), ctx:
        result = segmented_result(lx, lw, gate_up, down, counts)
    wanted = [t for t in (lx, lw) if t.requires_grad]
    return list(torch.autograd.grad(result, wanted, g))


def make(dtype: Any, counts: List[int], hid: int = 16, inter: int = 8) -> tuple:
    torch.manual_seed(sum(counts))
    m = sum(counts)
    x = torch.randn(m, hid).to(dtype)
    w = torch.rand(m, 1)  # routing weights stay fp32
    gate_up = (torch.randn(2 * inter, hid) * 0.3).to(dtype)
    down = (torch.randn(hid, inter) * 0.3).to(dtype)
    result_dtype = torch.promote_types(dtype, w.dtype)
    g = torch.randn(m, hid).to(result_dtype)
    return x, w, gate_up, down, g


def direct(x: Any, w: Any, gate_up: Any, down: Any, g: Any, counts: List[int], need: tuple,
           autocast_dtype: Any = None, mm: Callable[..., Any] = segment_mm) -> List[Any]:
    gx, gw = expert_vjp(x, w, gate_up, down, g, counts, need_hidden=need[0],
                        need_weights=need[1], autocast_dtype=autocast_dtype, mm=mm)
    return [t for t in (gx, gw) if t is not None]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("need", [(True, True), (True, False), (False, True)])
def test_expert_vjp_equals_autograd_over_the_segmented_expression(dtype: Any, need: tuple) -> None:
    counts = [3, 1, 7, 2, 5]
    x, w, gate_up, down, g = make(dtype, counts)
    want = autograd_grads(x, w, gate_up, down, g, counts, need)
    got = direct(x, w, gate_up, down, g, counts, need)
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_expert_vjp_equals_autograd_under_autocast_with_fp32_hidden() -> None:
    counts = [4, 2, 6]
    x, w, gate_up, down, g = make(torch.float32, counts)
    gate_up, down = gate_up.to(torch.bfloat16), down.to(torch.bfloat16)
    g = g.to(torch.float32)
    autocast = torch.autocast("cpu", dtype=torch.bfloat16)
    want = autograd_grads(x, w, gate_up, down, g, counts, (True, True), autocast=autocast)
    got = direct(x, w, gate_up, down, g, counts, (True, True), autocast_dtype=torch.bfloat16)
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_expert_vjp_uses_the_given_segment_matmul_for_all_four_gemms() -> None:
    counts = [2, 3]
    x, w, gate_up, down, g = make(torch.float32, counts)
    seen: List[tuple] = []

    def recording(a: Any, b: Any, transposed: bool, c: List[int]) -> Any:
        seen.append((tuple(b.shape), transposed, list(c)))
        return segment_mm(a, b, transposed, c)

    direct(x, w, gate_up, down, g, counts, (True, True), mm=recording)
    assert seen == [
        ((16, 16), True, counts),  # gate_up forward
        ((16, 8), True, counts),  # down forward (route-weight gradient needs the output)
        ((16, 8), False, counts),  # down input gradient
        ((16, 16), False, counts),  # gate_up input gradient
    ]


def test_expert_vjp_skips_the_input_gradient_gemms_when_only_weights_need_grad() -> None:
    counts = [2, 3]
    x, w, gate_up, down, g = make(torch.float32, counts)
    seen: List[bool] = []

    def recording(a: Any, b: Any, transposed: bool, c: List[int]) -> Any:
        seen.append(transposed)
        return segment_mm(a, b, transposed, c)

    direct(x, w, gate_up, down, g, counts, (False, True), mm=recording)
    assert seen == [True, True]


def test_expert_vjp_casts_a_low_precision_grad_out_to_the_result_dtype_first() -> None:
    """In a two-read recompute the node's output is a bf16 zero, so grad_out arrives in bf16
    while the recomputed product ``y (bf16) * w (fp32)`` is fp32; autograd casts grad_out to
    fp32 before the route-weight product. Multiplying in bf16 first changes low bits."""
    counts = [5, 3, 6]
    x, w, gate_up, down, _ = make(torch.bfloat16, counts)
    g = torch.randn(sum(counts), 16).to(torch.bfloat16)
    want = autograd_grads(x, w, gate_up, down, g, counts, (True, True))
    got = direct(x, w, gate_up, down, g, counts, (True, True))
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_weighted_expert_function_is_found_through_wrappers_of_forward() -> None:
    from benchmarks.harness.direct_expert_backward import weighted_expert_function
    from soup_cli.utils import expert_stream

    forward = expert_stream.streamed_experts_class().forward
    target = weighted_expert_function(forward)

    def clock(original: Any) -> Any:  # what the phase clock does
        def timed(*args: Any, **kwargs: Any) -> Any:
            return original(*args, **kwargs)
        return timed

    def hook(*args: Any, **kwargs: Any) -> Any:  # the segmented hook closes over the original
        return wrapped(*args, **kwargs)

    wrapped = clock(forward)
    assert weighted_expert_function(wrapped) is target
    assert weighted_expert_function(hook) is target
