"""``soup_cli.utils.invariant_gemm``: expert matmuls whose rows do not depend on M (CUDA).

Real expert widths (DeepSeek-V3 7168/4096/2048), a small M up to an MBR window, row windows at
offsets, fp32 and bf16. For each op the rows of one call over ``M`` rows must equal, as raw bits,
the same rows computed in calls over fewer rows: the forward ``x @ W.T``, its input gradient
``g @ W`` and the routing-weight gradient of ``scale_rows``. A native cuBLAS/ATen control is
measured alongside so the test shows it is not vacuous on this GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

#: (k, n): gate_up forward 7168->4096, down forward 2048->7168, down dx 7168->2048.
SHAPES = [(7168, 4096), (2048, 7168), (7168, 2048)]
ROWS = 160
PROBES = [(0, 1), (0, 2), (5, 16), (3, 33), (0, 128), (17, 128)]


def _dtypes():
    out = [torch.float32]
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        out.append(torch.bfloat16)
    return out


def _bits(tensor):
    view = {2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(view)


def _same(a, b):
    return bool(torch.equal(_bits(a), _bits(b)))


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize(("k", "n"), SHAPES)
def test_linear_forward_and_input_grad_rows_do_not_depend_on_m(k, n, dtype):
    from soup_cli.utils import invariant_gemm

    generator = torch.Generator(device="cpu").manual_seed(k + n)
    x = torch.randn(ROWS, k, generator=generator).to("cuda", dtype)
    weight = (torch.randn(n, k, generator=generator) / k**0.5).to("cuda", dtype)
    grad = torch.randn(ROWS, n, generator=generator).to("cuda", dtype)

    def run(rows):
        local = rows.detach().requires_grad_(True)
        out = invariant_gemm.linear(local, weight)
        return out, local

    full, full_x = run(x)
    full.backward(grad)
    for start, m in PROBES:
        part, part_x = run(x[start:start + m])
        assert _same(part, full[start:start + m]), f"forward rows {start}:{start + m}"
        part.backward(grad[start:start + m])
        assert _same(part_x.grad, full_x.grad[start:start + m]), f"dX rows {start}:{start + m}"
    reference = x.double() @ weight.double().t()
    error = (full.double() - reference).norm() / reference.norm()
    assert error < (2e-2 if dtype == torch.bfloat16 else 1e-5)


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize("width", [7168, 2048])
def test_scale_rows_weight_grad_rows_do_not_depend_on_m(width, dtype):
    from soup_cli.utils import invariant_gemm

    generator = torch.Generator(device="cpu").manual_seed(width)
    values = torch.randn(ROWS, width, generator=generator).to("cuda", dtype)
    weights = torch.rand(ROWS, 1, generator=generator).to("cuda", torch.float32)
    grad = torch.randn(ROWS, width, generator=generator).to("cuda", dtype)

    def weight_grad(lo, hi, fn):
        local = weights[lo:hi].detach().requires_grad_(True)
        out = fn(values[lo:hi], local)
        out.backward(grad[lo:hi].to(out.dtype))
        return out.detach(), local.grad

    full_out, full_grad = weight_grad(0, ROWS, invariant_gemm.scale_rows)
    native_differs = False
    native_full = weight_grad(0, ROWS, torch.mul)[1]
    for start, m in PROBES:
        out, grad_w = weight_grad(start, start + m, invariant_gemm.scale_rows)
        assert _same(out, full_out[start:start + m])
        assert _same(grad_w, full_grad[start:start + m]), f"d(weights) rows {start}:{start + m}"
        native = weight_grad(start, start + m, torch.mul)[1]
        native_differs |= not _same(native, native_full[start:start + m])
    reference = (grad.double() * values.double()).sum(-1, keepdim=True)
    torch.testing.assert_close(full_grad.double(), reference, rtol=2e-2, atol=1e-2)
    # Informational: whether ATen's reduction changed with M on this GPU and width.
    print(f"native d(weights) M-dependent at width {width}, {dtype}: {native_differs}")
