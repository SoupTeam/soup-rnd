"""``summarize_kernels``: what a GPU profile of one step says about the GPU's side of the step.

Pure arithmetic on (name, start, duration) rows, so it is checked on CPU with made-up rows.
"""

from __future__ import annotations

from benchmarks.harness.diag_gpu_profile import (
    category_of,
    classify_chrome_trace,
    classify_windows,
    summarize_kernels,
)


def row(name: str, start: float, dur: float) -> dict:
    return {"name": name, "start_us": start, "dur_us": dur}


def test_busy_time_counts_overlapping_kernels_once_and_idle_is_the_rest_of_the_span() -> None:
    rows = [
        row("ampere_bf16_s16816gemm_a", 0, 100),
        row("ampere_bf16_s16816gemm_b", 50, 100),   # overlaps the first on another stream
        row("elementwise_kernel", 300, 50),
    ]
    out = summarize_kernels(rows)
    assert out["kernels"] == 3
    assert out["sum_dur_us"] == 250                  # every kernel's own time
    assert out["busy_us"] == 150 + 50                # the union: 0..150 and 300..350
    assert out["span_us"] == 350
    assert out["idle_us"] == 150
    assert abs(out["busy_fraction"] - 200 / 350) < 1e-9


def test_kernels_are_grouped_by_category_and_the_top_list_is_sorted_by_total_time() -> None:
    rows = [
        row("ampere_bf16_s16816gemm_x", 0, 10),
        row("ampere_bf16_s16816gemm_x", 20, 10),
        row("cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu>", 40, 5),
        row("void at::native::vectorized_elementwise_kernel<4>", 50, 3),
        row("Memcpy HtoD (Pinned -> Device)", 60, 7),
        row("Memset (Device)", 70, 1),
    ]
    out = summarize_kernels(rows)
    cats = out["by_category"]
    assert cats["gemm"] == {"kernels": 3, "us": 25}
    assert cats["elementwise"] == {"kernels": 1, "us": 3}
    assert cats["memcpy"] == {"kernels": 2, "us": 8}
    assert out["top"][0] == {"name": "ampere_bf16_s16816gemm_x", "kernels": 2, "us": 20}


def test_category_of_recognises_gemm_gemv_and_split_k_reduction_names() -> None:
    assert category_of("gemv2T_kernel_val<int, int, __nv_bfloat16>") == "gemm"
    assert category_of("void cublasLt::splitKreduce_kernel<32, 16, int>") == "gemm"
    assert category_of("Memcpy DtoD (Device -> Device)") == "memcpy"
    assert category_of("some_unknown_kernel") == "other"


def test_an_empty_profile_summarises_to_zeros() -> None:
    out = summarize_kernels([])
    assert out["kernels"] == 0 and out["busy_fraction"] == 0.0 and out["top"] == []


def test_idle_gaps_are_bucketed_by_length_and_the_longest_name_their_neighbours() -> None:
    rows = [
        row("kernel_a", 0, 100),
        row("kernel_b", 150, 50),          # gap of 50 us after kernel_a
        row("kernel_c", 5000, 100),        # gap of 4800 us after kernel_b
        row("kernel_d", 5050, 100),        # overlaps kernel_c: no gap
        row("kernel_e", 5160, 10),         # gap of 10 us after kernel_d
    ]
    out = summarize_kernels(rows)
    gaps = out["gaps"]
    assert gaps["<20us"] == {"count": 1, "us": 10}
    assert gaps["20-100us"] == {"count": 1, "us": 50}
    assert gaps["1-10ms"] == {"count": 1, "us": 4800}
    assert gaps["100us-1ms"] == {"count": 0, "us": 0}
    assert out["long_gaps"][0] == {"us": 4800, "before": "kernel_b", "after": "kernel_c"}
    assert out["idle_us"] == 50 + 4800 + 10


def test_windows_are_classed_by_which_side_is_busy() -> None:
    # 100 us windows: 0-100 GPU busy, 100-200 one CPU thread busy, 200-300 both half busy,
    # 300-400 nothing running
    gpu = [(0, 100), (200, 250)]
    cpu = {1: [(100, 200), (250, 300)], 2: [(120, 130)]}
    out = classify_windows(gpu, cpu, window_us=100, start_us=0, end_us=400)
    assert out["windows"] == 4
    assert out["classes"]["gpu"]["windows"] == 1
    assert out["classes"]["cpu"]["windows"] == 1
    assert out["classes"]["mixed"]["windows"] == 1
    assert out["classes"]["stall"]["windows"] == 1
    assert out["classes"]["gpu"]["share"] == 0.25


def test_chrome_trace_lists_the_cpu_ops_by_count_and_time() -> None:
    def event(cat: str, name: str, ts: float, dur: float, tid: int = 1) -> dict:
        return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "tid": tid}

    trace = {"traceEvents": [
        event("kernel", "k", 0, 10),
        event("cpu_op", "aten::add", 0, 5),
        event("cpu_op", "aten::add", 6, 5),
        event("cpu_op", "aten::mm", 12, 30),
        event("user_annotation", "ignored", 0, 100),
    ]}
    out = classify_chrome_trace(trace, window_us=10)
    ops = {row["name"]: row for row in out["cpu_ops"]}
    assert ops["aten::add"] == {"name": "aten::add", "count": 2, "us": 10}
    assert ops["aten::mm"]["count"] == 1 and ops["aten::mm"]["us"] == 30
    assert out["cpu_ops"][0]["name"] == "aten::mm"          # sorted by total time
    assert "ignored" not in ops


def test_chrome_trace_reports_the_busy_share_of_every_named_thread() -> None:
    def event(cat: str, ts: float, dur: float, tid: int) -> dict:
        return {"ph": "X", "cat": cat, "name": "op", "ts": ts, "dur": dur, "tid": tid}

    trace = {"traceEvents": [
        {"ph": "M", "name": "thread_name", "tid": 7,
         "args": {"name": "thread 7 (soup-layer-range-0)"}},
        {"ph": "M", "name": "thread_name", "tid": 9, "args": {"name": "thread 9 (pt_autograd_0)"}},
        event("kernel", 0, 100, 0),
        event("cpu_op", 0, 60, 7),
        event("cpu_op", 10, 20, 7),       # nested in the first: counted once
        event("cpu_op", 50, 40, 9),
    ]}
    threads = {t["tid"]: t for t in classify_chrome_trace(trace, window_us=100)["threads"]}
    assert threads[7]["name"] == "thread 7 (soup-layer-range-0)"
    assert threads[7]["busy_us"] == 60 and threads[7]["busy_share"] == 0.6
    assert threads[9]["busy_us"] == 40
