"""The elementwise kernels of the fused expert against the ATen chains they replace, bit for bit.

Needs CUDA and a toolkit for the extension build (``pytest -m gpu --no-cov``); the conftest hook
skips it without a card. The remote series script runs it on the rental host.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


def load() -> object:
    from soup_cli.utils import a0_exact_lt

    return a0_exact_lt.load_extension().module


def same(a: object, b: object) -> bool:
    return bool(torch.equal(a.contiguous().view(torch.int32 if a.dtype == torch.float32
                                                else torch.int16),
                            b.contiguous().view(torch.int32 if b.dtype == torch.float32
                                                else torch.int16)))


def sample(shape: tuple, scale: float = 4.0) -> object:
    x = torch.randn(*shape, device="cuda") * scale
    x[..., ::97] = 0.0
    x[..., 1::89] *= 40.0  # large magnitudes, saturating silu
    x[..., 2::83] *= 1e-6  # tiny values
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("rows,inter", [(1, 1408), (7, 1408), (64, 1408), (19, 1408), (5, 64)])
def test_swiglu_forward_equals_silu_times_up(rows: int, inter: int) -> None:
    module = load()
    h = sample((rows, 2 * inter))
    gate, up = h.chunk(2, dim=-1)
    want = torch.mul(torch.nn.functional.silu(gate), up)
    got = module.swiglu_fwd(h)
    assert same(got, want)


@pytest.mark.parametrize("rows,inter", [(1, 1408), (7, 1408), (64, 1408), (19, 1408), (5, 64)])
def test_swiglu_backward_equals_the_atten_chain(rows: int, inter: int) -> None:
    module = load()
    h = sample((rows, 2 * inter))
    grad_a = sample((rows, inter), scale=1.0)
    gate, up = h.chunk(2, dim=-1)
    s = torch.nn.functional.silu(gate)
    grad_gate = torch.ops.aten.silu_backward(torch.mul(grad_a, up), gate)
    want = torch.cat([grad_gate, torch.mul(grad_a, s)], dim=-1)
    got = module.swiglu_bwd(h, grad_a)
    assert same(got, want)


@pytest.mark.parametrize("rows,cols", [(1, 2048), (9, 2048), (64, 2048), (3, 17)])
def test_scale_forward_equals_mul_with_fp32_weights(rows: int, cols: int) -> None:
    module = load()
    y = sample((rows, cols))
    w = torch.rand(rows, 1, device="cuda")
    assert same(module.scale_fwd(y, w), torch.mul(y, w))


@pytest.mark.parametrize("g_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,cols", [(1, 2048), (9, 2048), (64, 2048), (3, 17)])
def test_grad_scale_equals_mul_chain(rows: int, cols: int, g_dtype: object) -> None:
    module = load()
    y = sample((rows, cols))
    g = sample((rows, cols), scale=1.0).to(g_dtype)
    w = torch.rand(rows, 1, device="cuda")
    g32 = g.to(torch.float32)
    grad_y, prod = module.grad_scale(g, y, w, True, True)
    assert same(grad_y, torch.mul(g32, w).to(torch.bfloat16))
    assert same(prod, torch.mul(g32, y))




@pytest.mark.parametrize("cols", [2048, 1408, 2816, 64])
def test_copy_segments_moves_whole_row_blocks_between_layouts(cols: int) -> None:
    module = load()
    counts = [3, 1, 7, 5]
    src_rows = [20, 0, 9, 40]       # where each segment sits in the source
    dst_rows = [0, 3, 4, 11]        # where it goes in the destination
    src = sample((64, cols))
    dst = torch.zeros(24, cols, device="cuda", dtype=torch.bfloat16)
    module.copy_segments(src, dst, src_rows, dst_rows, counts)
    for s, d, c in zip(src_rows, dst_rows, counts):
        assert same(dst[d : d + c], src[s : s + c])
    assert torch.count_nonzero(dst[16:]) == 0, "rows outside the segments must stay untouched"


def test_copy_segments_round_trips_through_a_permutation() -> None:
    module = load()
    counts = [2, 6, 1, 4]
    offsets = [0, 2, 8, 9]
    order = [2, 0, 3, 1]            # groups laid out as segments 2, 0, 3, 1
    perm, pos = [], 0
    for i in order:
        perm.append(pos)
        pos += counts[i]
    x = sample((13, 2048))
    permuted = torch.zeros_like(x)
    module.copy_segments(x, permuted, [offsets[i] for i in order], perm,
                         [counts[i] for i in order])
    back = torch.zeros_like(x)
    module.copy_segments(permuted, back, perm, [offsets[i] for i in order],
                         [counts[i] for i in order])
    assert same(back, x)


def test_spreading_the_groups_over_streams_does_not_change_the_expert_bits() -> None:
    from soup_cli.utils import a0_exact_lt

    backend = a0_exact_lt.load_extension()
    fused = a0_exact_lt.FusedExpert(backend)
    gate_up = (torch.randn(2816, 2048, device="cuda") * 0.02).to(torch.bfloat16)
    down = (torch.randn(2048, 1408, device="cuda") * 0.02).to(torch.bfloat16)
    counts = [3, 1, 7, 20, 5, 2, 12, 9]
    rows = sum(counts)
    assert fused.ensure(counts, gate_up, down)
    hidden = sample((rows, 2048))
    weights = torch.rand(rows, 1, device="cuda")
    grad = sample((rows, 2048), scale=1.0)

    def run(streams: int) -> list:
        backend.set_streams(streams)
        out = backend.expert_forward(hidden, weights, gate_up, down, counts)
        grad_hidden, grad_weights = backend.expert_backward(
            hidden, weights, gate_up, down, grad, counts, True, True
        )
        torch.cuda.synchronize()
        return [out, grad_hidden, grad_weights]

    base = run(1)
    for streams in (2, 4, 1):
        for got, want in zip(run(streams), base):
            assert same(got, want)


def test_expert_calls_from_several_threads_match_the_single_thread_bits() -> None:
    """The layer-range threads call the extension at once: descriptor and handle caches must be safe
    (many new row counts per thread, so every thread inserts into the caches while others read)."""
    import random
    import threading

    from soup_cli.utils import a0_exact_lt

    backend = a0_exact_lt.load_extension()
    fused = a0_exact_lt.FusedExpert(backend)
    gate_up = (torch.randn(2816, 2048, device="cuda") * 0.02).to(torch.bfloat16)
    down = (torch.randn(2048, 1408, device="cuda") * 0.02).to(torch.bfloat16)
    rng = random.Random(3)
    jobs = [[rng.randint(1, 64) for _ in range(rng.randint(3, 8))] for _ in range(48)]
    for counts in jobs:
        assert fused.ensure(counts, gate_up, down)
    data = []
    for counts in jobs:
        rows = sum(counts)
        data.append((counts, sample((rows, 2048)), torch.rand(rows, 1, device="cuda"),
                     sample((rows, 2048), scale=1.0)))

    def run(item: tuple) -> list:
        counts, hidden, weights, grad = item
        out = backend.expert_forward(hidden, weights, gate_up, down, counts)
        grad_hidden, grad_weights = backend.expert_backward(
            hidden, weights, gate_up, down, grad, counts, True, True)
        torch.cuda.synchronize()
        return [out, grad_hidden, grad_weights]

    want = [run(item) for item in data]
    got: list = [None] * len(data)
    errors: list = []

    def work(start: int) -> None:
        try:
            for i in range(start, len(data), 4):
                got[i] = run(data[i])
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for have, expect in zip(got, want):
        for a, b in zip(have, expect):
            assert same(a, b)


@pytest.mark.parametrize("hid", [2048, 7168, 1408, 512, 4096])
def test_segment_row_sum_equals_one_aten_sum_per_segment(hid: int) -> None:
    """One launch for every segment of ``prod.sum(1, keepdim)``, bit for bit with the ATen call
    per segment (ATen picks its reduction tree from the segment's row count)."""
    module = load()
    torch.manual_seed(hid)
    for rows in list(range(1, 65)) + [96, 200]:
        prod = torch.randn(rows, hid, device="cuda") * 3
        prod[::7, ::5] *= 1e4
        prod[::11, 1::13] *= 1e-7
        got = module.segment_row_sum(prod, [rows])
        assert same(got, prod.sum(1, keepdim=True)), rows
    for trial in range(30):
        counts = [int(c) for c in torch.randint(1, 65, (int(torch.randint(2, 9, (1,))),))]
        prod = torch.randn(sum(counts), hid, device="cuda") * 3
        want = torch.cat([part.sum(1, keepdim=True) for part in prod.split(counts)])
        assert same(module.segment_row_sum(prod, counts), want), counts


def test_the_card_runs_the_pytorch_series_whose_aten_reductions_the_kernels_mirror() -> None:
    """Red on any other PyTorch: the bit-equality with A0 is only claimed for this series."""
    from soup_cli.utils.expert_gemm import EXACT_TORCH_SERIES, require_lt_fused

    assert tuple(int(p) for p in torch.__version__.split(".")[:2]) == EXACT_TORCH_SERIES
    require_lt_fused()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", "--no-cov"]))

