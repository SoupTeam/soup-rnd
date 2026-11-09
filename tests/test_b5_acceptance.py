"""B5 ticket 03: the acceptance verdict for the Moonlight run, on journals written by hand.

The rule is ``benchmarks/b5-moonlight-acceptance-20261109.md``. These tests feed the verdict a
journal and read the answer. No GPU, no model, no torch.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from benchmarks.harness import b5_acceptance

GIB = 2**30
#: What torch reports for an RTX 3070: 8191 MiB.
CARD = 8191 * 2**20
REVISION = "476b36a473d4467f94469414bef6cee75c9c8172"


def _digest(number: int) -> str:
    return f"{number:064x}"


def good_journal() -> List[Dict[str, Any]]:
    """A journal that passes the judge and the acceptance rule: four steps, nothing odd."""
    rows: List[Dict[str, Any]] = [
        {"event": "signature", "kind": "real", "model": "moonshotai/Moonlight-16B-A3B",
         "revision": REVISION},
        {"event": "environment", "gpu": "NVIDIA GeForce RTX 3070", "device_count": 1,
         "vram_bytes": CARD, "compute_capability": [8, 6], "dtype": "bfloat16",
         "quant": "nf4", "double_quant": False, "expert_layout": "per_expert", "tier": "disk",
         "trust_remote_code": False},
        {"event": "adapter_initial", "sha256": _digest(1), "l2_norm": 3.0, "tensors": 10,
         "elements": 1000},
    ]
    for step in range(4):
        rows.append({
            "event": "step", "step": step, "loss": 2.0 - 0.1 * step, "gradient_norm": 0.5,
            "gradient_tensors": 10, "trainable_tensors": 10, "gradient_nonfinite_tensors": 0,
            "adapter_sha256_before": _digest(step + 1), "adapter_sha256_after": _digest(step + 2),
            "distance_from_initial": 0.01 * (step + 1), "tokens": 512, "seconds": 20.0,
            "tokens_per_second": 25.6, "peak_allocated": 5 * GIB, "peak_reserved": 6 * GIB,
            "expert_fetches": 1500, "expert_bytes_read": 9_000_000_000,
        })
    rows += [{"event": "reload", "parameter_sha256": _digest(5)}, {"event": "complete"}]
    return rows


def _without(journal: List[Dict[str, Any]], event: str) -> List[Dict[str, Any]]:
    return [row for row in journal if row["event"] != event]


def test_a_clean_four_step_journal_passes() -> None:
    result = b5_acceptance.verdict(good_journal())

    assert result["passed"] is True, result["failures"]
    assert result["failures"] == []
    assert result["judge"]["passed"] is True


def _codes(result: Dict[str, Any]) -> List[str]:
    return [failure.split(":", 1)[0] for failure in result["failures"]]


def test_three_steps_are_not_the_four_of_the_rule() -> None:
    journal = [row for row in good_journal() if not (row["event"] == "step" and row["step"] == 3)]
    journal = [dict(row, parameter_sha256=_digest(4)) if row["event"] == "reload" else row
               for row in journal]

    result = b5_acceptance.verdict(journal)

    assert result["passed"] is False
    assert "steps" in _codes(result)


def test_a_judge_fail_fails_the_acceptance_and_keeps_its_reason() -> None:
    journal = good_journal()
    for row in journal:
        if row["event"] == "step" and row["step"] == 2:
            row["gradient_norm"] = 0.0

    result = b5_acceptance.verdict(journal)

    assert result["passed"] is False
    assert result["judge"]["passed"] is False
    assert any(f.startswith("judge: zero_gradient") for f in result["failures"])


def test_a_synthetic_signature_is_not_a_real_moonlight_result() -> None:
    journal = good_journal()
    journal[0] = {"event": "signature", "kind": "synthetic", "model": "tiny",
                  "generator": "fixture"}

    result = b5_acceptance.verdict(journal)

    assert result["passed"] is False
    assert "signature" in _codes(result)


def test_another_model_or_revision_is_not_the_pinned_moonlight() -> None:
    other_model = good_journal()
    other_model[0] = dict(other_model[0], model="deepseek-ai/DeepSeek-V3")
    other_revision = good_journal()
    other_revision[0] = dict(other_revision[0], revision="0" * 40)

    assert "signature" in _codes(b5_acceptance.verdict(other_model))
    assert "signature" in _codes(b5_acceptance.verdict(other_revision))


def _edit(journal: List[Dict[str, Any]], event: str, **values: Any) -> List[Dict[str, Any]]:
    edited = copy.deepcopy(journal)
    for row in edited:
        if row["event"] == event:
            row.update(values)
    return edited


def test_a_card_outside_the_8_gb_class_is_refused() -> None:
    for change in ({"vram_bytes": 24 * GIB}, {"vram_bytes": 6 * GIB}, {"device_count": 2},
                   {"compute_capability": [7, 5]}):
        result = b5_acceptance.verdict(_edit(good_journal(), "environment", **change))

        assert result["passed"] is False, change
        assert "card" in _codes(result), change


def test_a_journal_without_the_environment_event_is_refused() -> None:
    result = b5_acceptance.verdict(_without(good_journal(), "environment"))

    assert result["passed"] is False
    assert "environment" in _codes(result)


def test_the_setup_must_be_nf4_per_expert_on_disk_without_remote_code() -> None:
    for change in ({"quant": "none"}, {"double_quant": True}, {"expert_layout": "fused"},
                   {"tier": "ram"}, {"trust_remote_code": True}):
        result = b5_acceptance.verdict(_edit(good_journal(), "environment", **change))

        assert result["passed"] is False, change
        assert "environment" in _codes(result), change


def test_a_peak_above_the_card_memory_fails_and_names_which_counter() -> None:
    allocated = b5_acceptance.verdict(_edit(good_journal(), "step", peak_allocated=CARD + 1))
    reserved = b5_acceptance.verdict(_edit(good_journal(), "step", peak_reserved=CARD + 1))

    assert "vram_allocated" in _codes(allocated)
    assert "vram_reserved" in _codes(reserved)
    assert "vram_reserved" not in _codes(allocated)


def test_a_peak_exactly_at_the_card_memory_still_fits() -> None:

    result = b5_acceptance.verdict(
        _edit(good_journal(), "step", peak_allocated=CARD, peak_reserved=CARD))

    assert result["passed"] is True, result["failures"]


def test_a_step_without_a_peak_cannot_show_that_memory_fits() -> None:
    journal = good_journal()
    del [row for row in journal if row["event"] == "step"][1]["peak_reserved"]

    result = b5_acceptance.verdict(journal)

    assert result["passed"] is False
    assert "peak_missing" in _codes(result)


def test_a_padded_or_short_step_is_refused() -> None:
    journal = good_journal()
    [row for row in journal if row["event"] == "step"][2]["tokens"] = 511

    result = b5_acceptance.verdict(journal)

    assert result["passed"] is False
    assert "padding" in _codes(result)


def test_a_run_that_did_not_complete_is_refused() -> None:
    result = b5_acceptance.verdict(_without(good_journal(), "complete"))

    assert result["passed"] is False
    assert "incomplete" in _codes(result)


def _with_clock(journal: List[Dict[str, Any]], samples: int, throttled: int
                ) -> List[Dict[str, Any]]:
    clock = {"event": "training_clock", "samples": samples, "throttled_samples": throttled,
             "sm_clock_mhz": {"min": 1500, "median": 1800, "max": 1905}}
    return [*journal[:-1], clock, journal[-1]]


def test_the_measured_block_has_peaks_expert_loads_and_speed_per_step() -> None:
    journal = _with_clock(good_journal(), samples=60, throttled=0)
    steps = [row for row in journal if row["event"] == "step"]
    steps[0].update(tokens_per_second=5.0, expert_fetches=1700)
    steps[1].update(tokens_per_second=24.0, peak_allocated=4 * GIB + 7)
    steps[2].update(tokens_per_second=26.0, peak_reserved=6 * GIB + 9)
    steps[3].update(tokens_per_second=25.0)

    measured = b5_acceptance.verdict(journal)["measured"]

    assert measured["peak_allocated_bytes"] == 5 * GIB
    assert measured["peak_reserved_bytes"] == 6 * GIB + 9
    assert measured["expert_fetches_per_step"] == [1700, 1500, 1500, 1500]
    assert measured["tokens_per_second"] == {
        "status": "measured", "first_step": 5.0, "later_steps_median": 25.0}


def test_speed_is_not_measured_without_clock_samples() -> None:
    measured = b5_acceptance.verdict(good_journal())["measured"]

    assert measured["tokens_per_second"]["status"] == "not measured"
    assert "no clock samples" in measured["tokens_per_second"]["reason"]


def test_speed_from_a_throttled_gpu_is_excluded_not_reported() -> None:
    measured = b5_acceptance.verdict(
        _with_clock(good_journal(), samples=60, throttled=30))["measured"]

    assert measured["tokens_per_second"]["status"] == "not measured"
    assert "throttled" in measured["tokens_per_second"]["reason"]
    assert "first_step" not in measured["tokens_per_second"]


def test_throttling_in_fewer_than_half_the_samples_keeps_the_speed() -> None:
    measured = b5_acceptance.verdict(
        _with_clock(good_journal(), samples=60, throttled=29))["measured"]

    assert measured["tokens_per_second"]["status"] == "measured"
