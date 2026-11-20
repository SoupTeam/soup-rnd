"""Measurement runner for the exact-MBR campaign: modes, OOM rows, MBPP data hash (pure CPU)."""

from __future__ import annotations

import pytest

from benchmarks.harness import mbr_measure as mm


def test_windows_are_the_four_agreed_pairs() -> None:
    assert mm.WINDOWS == ((8, 64), (8, 512), (64, 64), (64, 512))
    assert mm.BIT_LADDER == (1, 2, 4, 8)


def test_speed_in_deterministic_mode_is_refused() -> None:
    with pytest.raises(mm.MixedModeError, match="speed"):
        mm.check_mode("speed", deterministic=True)


def test_bits_in_normal_mode_are_refused() -> None:
    with pytest.raises(mm.MixedModeError, match="bits"):
        mm.check_mode("bits", deterministic=False)


def test_allowed_modes_return_their_label() -> None:
    assert mm.check_mode("speed", deterministic=False) == "normal"
    assert mm.check_mode("bits", deterministic=True) == "deterministic"


def test_a0_is_not_planned_above_k8() -> None:
    assert mm.arms_for(k=8) == ("A0", "B0", "C")
    assert mm.arms_for(k=64) == ("B0", "C")


def test_every_report_row_carries_the_mode() -> None:
    row = mm.make_row(arm="C", k=8, seq=64, kind="speed", deterministic=False,
                      result={"step_seconds": 1.5, "peak_allocated": 1, "peak_reserved": 2,
                              "peak_nvml": 3, "peak_ram": 4, "grad_sha256": "ab"},
                      dataset_sha256="d")
    assert row["mode"] == "normal" and row["status"] == "ok" and row["dataset_sha256"] == "d"
    assert row["peak_nvml"] == 3 and row["grad_sha256"] == "ab"


def test_oom_becomes_a_row_with_the_peak_not_an_exception() -> None:
    class FakeOomError(RuntimeError):
        pass

    def run() -> dict:
        raise FakeOomError("CUDA out of memory. Tried to allocate 2 GiB")

    peaks = {"peak_allocated": 7, "peak_reserved": 8, "peak_nvml": 9, "peak_ram": 10}
    out = mm.run_guarded(run, peaks=lambda: peaks, is_oom=lambda e: "out of memory" in str(e))
    assert out["status"] == "OOM"
    assert out["peak_allocated"] == 7 and out["peak_nvml"] == 9 and out["peak_ram"] == 10
    row = mm.make_row(arm="C", k=64, seq=512, kind="speed", deterministic=False, result=out,
                      dataset_sha256="d")
    assert row["status"] == "OOM" and row["mode"] == "normal"


def test_other_errors_are_not_swallowed() -> None:
    def run() -> dict:
        raise ValueError("bug")

    with pytest.raises(ValueError):
        mm.run_guarded(run, peaks=lambda: {}, is_oom=lambda e: False)


def test_a_successful_run_keeps_its_result_and_peaks() -> None:
    out = mm.run_guarded(lambda: {"step_seconds": 2.0}, peaks=lambda: {"peak_ram": 5},
                         is_oom=lambda e: False)
    assert out == {"step_seconds": 2.0, "peak_ram": 5, "status": "ok"}


def test_mbpp_records_use_real_text_and_the_dataset_hash_is_stable() -> None:
    rows = [{"text": "Write a function.", "code": "def f():\n    return 1", "task_id": 1},
            {"text": "Another task.", "code": "def g():\n    return 2", "task_id": 2}]

    class Tok:
        eos_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
            return {"input_ids": [ord(c) % 250 + 1 for c in text]}

    first = mm.mbpp_records(rows, Tok(), seq=64)
    again = mm.mbpp_records(list(rows), Tok(), seq=64)
    assert first["sha256"] == again["sha256"]
    assert all(len(r) == 64 for r in first["input_ids"])
    edited = [dict(rows[0], code="def f():\n    return 9"), rows[1]]
    changed = mm.mbpp_records(edited, Tok(), seq=64)
    assert changed["sha256"] != first["sha256"]


def test_mbpp_records_fill_k_rows_by_cycling_the_dataset() -> None:
    rows = [{"text": "t" * 5, "code": "c" * 5, "task_id": i} for i in range(3)]

    class Tok:
        eos_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
            return {"input_ids": [3] * len(text)}

    out = mm.mbpp_records(rows, Tok(), seq=16)
    window = mm.take_window(out["input_ids"], k=8)
    assert len(window) == 8 and window[3] == window[0]


def test_block_median_of_three_after_warmup() -> None:
    assert mm.median_after_warmup([9.0, 2.0, 1.0, 3.0]) == 2.0
    with pytest.raises(ValueError):
        mm.median_after_warmup([1.0])
