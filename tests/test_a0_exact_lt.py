"""A0-exact merged expert GEMM (harness only): segments that cuBLAS serves with one kernel are
merged into one cuBLASLt call with that kernel forced, which keeps every row's A0 bits.

All CPU. The cuBLASLt extension is replaced by a fake backend whose algos are right only for some
row counts, so the tests check the planning: which algo the table keeps, which segments share a
call, and that outputs and input gradients come back in segment order.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

torch = pytest.importorskip("torch")

from benchmarks.harness.a0_exact_lt import (  # noqa: E402
    AlgoTable,
    MergedLinear,
    bucket_segments,
    merged_matmul,
)


def ints(*shape: int, requires_grad: bool = False) -> Any:
    """Small integers in float64: products and sums are exact in any order, so CPU GEMM blocking
    (which does depend on M) cannot change a bit."""
    out = torch.randint(-4, 5, shape).to(torch.float64)
    return out.requires_grad_(requires_grad)


def reference(x: Any, w: Any, transposed: bool) -> Any:
    return x @ w.t() if transposed else x @ w


class FakeBackend:
    """Algos 1, 2, 3. While the table discovers (one segment per call) only the band's algo is
    exact: algo 1 for 2..16 rows, algo 2 for 17 rows and more, none for 1 row. After discovery,
    algos 1 and 2 stay exact on merged rows (rows do not see their neighbours inside one kernel).
    A wrong answer is the exact product plus one."""

    def __init__(self) -> None:
        self.discovering = True
        self.heuristic_calls: List[Tuple[int, bool]] = []
        self.matmul_calls: List[Tuple[int, bool, int]] = []

    @staticmethod
    def supports(x: Any, w: Any) -> bool:
        return True

    @staticmethod
    def band_algo(rows: int) -> int:
        return 0 if rows == 1 else (1 if rows <= 16 else 2)

    def heuristics(self, x: Any, w: Any, transposed: bool, limit: int) -> List[Any]:
        self.heuristic_calls.append((int(x.shape[0]), transposed))
        return [torch.tensor([i, 7, 7]) for i in (1, 2, 3)]

    def matmul(self, x: Any, w: Any, transposed: bool, algo: Any) -> Any:
        rows, algo_id = int(x.shape[0]), int(algo[0])
        self.matmul_calls.append((rows, transposed, algo_id))
        out = reference(x, w, transposed)
        exact = algo_id in (1, 2) and (not self.discovering or algo_id == self.band_algo(rows))
        return out if exact else out + 1


def make_table() -> Tuple[FakeBackend, AlgoTable]:
    backend = FakeBackend()
    return backend, AlgoTable(backend, native=reference)


def discover(table: AlgoTable, backend: FakeBackend, parts: Any, w: Any, transposed: bool) -> None:
    for part in parts:
        table.lookup(part, w, transposed)
    backend.discovering = False
    backend.matmul_calls.clear()


def test_bucket_segments_groups_equal_keys_in_first_seen_order_and_isolates_none() -> None:
    keys = ["a", None, "b", "a", None, "b", "a"]
    assert bucket_segments(keys) == [
        ("a", [0, 3, 6]),
        (None, [1]),
        ("b", [2, 5]),
        (None, [4]),
    ]


def test_table_keeps_the_algo_that_reproduces_native_and_caches_it_per_shape_and_rows() -> None:
    backend, table = make_table()
    w = ints(6, 4)
    first = table.lookup(ints(5, 4), w, True)
    again = table.lookup(ints(5, 4), w, True)
    assert int(first[0]) == 1 and again is first
    assert backend.heuristic_calls == [(5, True)]
    assert int(table.lookup(ints(20, 4), w, True)[0]) == 2
    # the backward form of the same weight is its own entry
    assert table.lookup(ints(5, 6), w, False) is not None
    assert backend.heuristic_calls[-1] == (5, False)


class OverlapBackend(FakeBackend):
    """Algos 1 and 2 are both exact for every row count of 2 and more; the heuristics list them in a
    different order per row count, as cuBLASLt does for neighbouring M."""

    def heuristics(self, x: Any, w: Any, transposed: bool, limit: int) -> List[Any]:
        self.heuristic_calls.append((int(x.shape[0]), transposed))
        order = (1, 2, 3) if int(x.shape[0]) % 2 else (2, 1, 3)
        return [torch.tensor([i, 7, 7]) for i in order]

    def matmul(self, x: Any, w: Any, transposed: bool, algo: Any) -> Any:
        rows, algo_id = int(x.shape[0]), int(algo[0])
        out = reference(x, w, transposed)
        return out if algo_id in (1, 2) and rows >= 2 else out + 1


def test_table_reuses_an_algo_it_already_chose_when_it_is_also_exact_for_the_next_rows() -> None:
    backend = OverlapBackend()
    table = AlgoTable(backend, native=reference)
    w = ints(6, 4)
    # the heuristics would pick 1 for 3 rows, 2 for 4 rows, 1 for 5 rows, 2 for 8 rows
    chosen = [int(table.lookup(ints(rows, 4), w, True)[0]) for rows in (3, 4, 5, 8, 9)]
    assert chosen == [1, 1, 1, 1, 1]


def test_table_does_not_reuse_an_algo_that_is_wrong_for_the_next_rows() -> None:
    backend, table = make_table()
    w = ints(6, 4)
    assert int(table.lookup(ints(5, 4), w, True)[0]) == 1
    # algo 1 stays preferred but is not exact for 20 rows (only algo 2 is): the table must take 2
    assert int(table.lookup(ints(20, 4), w, True)[0]) == 2
    assert int(table.lookup(ints(7, 4), w, True)[0]) == 1


class IntegerOnlyBackend(OverlapBackend):
    """Algo 1 sums in another order: it matches native on integer data but not on fractions."""

    def matmul(self, x: Any, w: Any, transposed: bool, algo: Any) -> Any:
        out = reference(x, w, transposed)
        if int(algo[0]) == 1 and bool((x != x.round()).any()):
            return out + 1
        return out if int(algo[0]) in (1, 2) else out + 1


def test_table_checks_a_reused_algo_on_other_data_than_the_one_segment() -> None:
    backend = IntegerOnlyBackend()
    table = AlgoTable(backend, native=reference)
    w = ints(6, 4)
    assert int(table.lookup(ints(3, 4), w, True)[0]) == 1      # first match, integer sample
    # 4 rows: the heuristics list 2 first; reuse of 1 would match on the integer sample, but 1 is
    # wrong on fractional data, so it must not be reused
    assert int(table.lookup(ints(4, 4), w, True)[0]) == 2


def test_table_returns_none_when_no_algo_reproduces_native() -> None:
    _, table = make_table()
    w = ints(6, 4)
    assert table.lookup(ints(1, 4), w, True) is None
    assert table.stats["no_algo"] == 1


def test_merged_matmul_makes_one_call_per_algo_and_keeps_segment_order() -> None:
    backend, table = make_table()
    w = ints(6, 4)
    counts = [3, 1, 20, 5, 1, 30]
    x = ints(sum(counts), 4)
    discover(table, backend, x.split(counts), w, True)
    out = merged_matmul(x, w, True, counts, table)
    assert torch.equal(out, torch.cat([reference(p, w, True) for p in x.split(counts)]))
    # 3 + 5 rows share algo 1, 20 + 30 rows share algo 2, the two 1-row segments go native
    assert sorted(backend.matmul_calls) == [(8, True, 1), (50, True, 2)]


def test_merged_linear_backward_gives_per_segment_input_gradients() -> None:
    backend, table = make_table()
    w = ints(6, 4)
    counts = [4, 1, 7]
    x = ints(sum(counts), 4, requires_grad=True)
    grad = ints(sum(counts), 6)
    for part, g in zip(x.detach().split(counts), grad.split(counts)):
        table.lookup(part, w, True)
        table.lookup(g, w, False)
    backend.discovering = False
    backend.matmul_calls.clear()
    out = MergedLinear.apply(x, w, counts, table)
    assert torch.equal(out, torch.cat([reference(p, w, True) for p in x.detach().split(counts)]))
    (gx,) = torch.autograd.grad(out, x, grad)
    assert torch.equal(gx, torch.cat([g @ w for g in grad.split(counts)]))
    assert (11, False, 1) in backend.matmul_calls  # 4 + 7 gradient rows in one backward call


def test_merged_matmul_falls_back_to_native_for_a_non_contiguous_weight() -> None:
    backend, table = make_table()
    w = ints(4, 6).t()  # [6, 4] view, not contiguous
    counts = [3, 5]
    x = ints(8, 4)
    out = merged_matmul(x, w, True, counts, table)
    assert torch.equal(out, torch.cat([reference(p, w, True) for p in x.split(counts)]))
    assert backend.matmul_calls == []


def test_merged_matmul_falls_back_to_native_when_the_backend_does_not_support_the_inputs() -> None:
    backend, table = make_table()
    backend.supports = lambda x, w: False  # type: ignore[method-assign]
    w = ints(6, 4)
    counts = [3, 5]
    x = ints(8, 4)
    out = merged_matmul(x, w, True, counts, table)
    assert torch.equal(out, torch.cat([reference(p, w, True) for p in x.split(counts)]))
    assert backend.matmul_calls == [] and backend.heuristic_calls == []


def test_merged_matmul_runs_segments_alone_when_lt_refuses_the_merged_size() -> None:
    backend, table = make_table()
    w = ints(6, 4)
    counts = [3, 5, 4]
    x = ints(sum(counts), 4)
    discover(table, backend, x.split(counts), w, True)
    original = backend.matmul

    def refuse_large(x: Any, w: Any, transposed: bool, algo: Any) -> Any:
        if int(x.shape[0]) > 5:
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")
        return original(x, w, transposed, algo)

    backend.matmul = refuse_large  # type: ignore[method-assign]
    expected = torch.cat([reference(p, w, True) for p in x.split(counts)])
    assert torch.equal(merged_matmul(x, w, True, counts, table), expected)
    assert table.stats["merge_refused"] == 1
    backend.matmul_calls.clear()
    assert torch.equal(merged_matmul(x, w, True, counts, table), expected)
    # the refusal is remembered: no second merged attempt, one call per segment
    assert sorted(r for r, _, _ in backend.matmul_calls) == [3, 4, 5]


class FusedBackend(FakeBackend):
    """Adds ``merged``: the whole planned call in one backend call (the C++ path). Records the
    groups it was given and can refuse groups larger than ``refuse_above`` rows."""

    def __init__(self) -> None:
        super().__init__()
        self.merged_calls: List[Any] = []
        self.refuse_above = 10**9

    def merged(self, x: Any, w: Any, transposed: bool, counts: List[int],
               groups: List[List[int]], algos: List[List[int]]) -> Tuple[Any, List[int]]:
        self.merged_calls.append([(list(g), a[0] if a else None) for g, a in zip(groups, algos)])
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)
        out = torch.full((offsets[-1], w.shape[0] if transposed else w.shape[1]), float("nan"),
                         dtype=x.dtype)
        failed = []
        for index, (segs, algo) in enumerate(zip(groups, algos)):
            rows = sum(counts[i] for i in segs)
            if algo and len(segs) > 1 and rows > self.refuse_above:
                failed.append(index)
                continue
            for i in segs:
                part = x[offsets[i]:offsets[i + 1]]
                out[offsets[i]:offsets[i + 1]] = reference(part, w, transposed)
        return out, failed


def test_merged_matmul_hands_the_whole_plan_to_a_fused_backend_in_one_call() -> None:
    backend = FusedBackend()
    table = AlgoTable(backend, native=reference)
    w = ints(6, 4)
    counts = [3, 1, 20, 5, 1, 30]
    x = ints(sum(counts), 4)
    discover(table, backend, x.split(counts), w, True)
    out = merged_matmul(x, w, True, counts, table)
    assert torch.equal(out, torch.cat([reference(p, w, True) for p in x.split(counts)]))
    assert backend.merged_calls == [[([0, 3], 1), ([1], None), ([2, 5], 2), ([4], None)]]
    assert backend.matmul_calls == []  # nothing went through the per-call path
    assert table.stats["merged_calls"] == 2


def test_a_group_the_fused_backend_refuses_is_rerun_per_segment_and_remembered() -> None:
    backend = FusedBackend()
    table = AlgoTable(backend, native=reference)
    w = ints(6, 4)
    counts = [3, 20, 5, 30]
    x = ints(sum(counts), 4)
    discover(table, backend, x.split(counts), w, True)
    backend.refuse_above = 40  # the 20 + 30 group is refused, the 3 + 5 group is not
    expected = torch.cat([reference(p, w, True) for p in x.split(counts)])
    assert torch.equal(merged_matmul(x, w, True, counts, table), expected)
    assert backend.merged_calls[-1] == [([1], 2), ([3], 2)]  # the rerun, one segment per group
    assert table.stats["merge_refused"] == 1
    backend.merged_calls.clear()
    assert torch.equal(merged_matmul(x, w, True, counts, table), expected)
    assert backend.merged_calls == [[([0, 2], 1), ([1], 2), ([3], 2)]]  # refusal remembered
