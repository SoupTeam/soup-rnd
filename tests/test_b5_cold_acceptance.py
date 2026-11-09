"""B5 ticket 06: the acceptance verdict of the synthetic 671B cold-disk run, on hand-made journals.

The rule is ``benchmarks/b5-synthetic-671b-acceptance-20261109.md``. These tests feed the verdict
a journal and the store record and read the answer. No GPU, no model, no torch. Every number in
the journal is invented for the test; none of them describes a run.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from benchmarks.harness import b5_cold_acceptance

GIB = 2**30
CARD = 8191 * 2**20  # what torch reports for an RTX 3070
STORE_BYTES = 380_276_485_539
GENERATOR = "b5_synth_store/1 seed=0 dtype=bfloat16 double_quant=False shape={}"


def _digest(number: int) -> str:
    return f"{number:064x}"


def good_journal() -> List[Dict[str, Any]]:
    """A journal that passes the judge and the rule: three cold-started steps of 128 tokens."""
    rows: List[Dict[str, Any]] = [
        {"event": "signature", "kind": "synthetic", "model": "synthetic MoE store x",
         "generator": GENERATOR},
        {"event": "environment", "gpu": "NVIDIA GeForce RTX 3070", "device_count": 1,
         "vram_bytes": CARD, "compute_capability": [8, 6], "dtype": "bfloat16",
         "quant": "nf4", "double_quant": False, "expert_layout": "per_expert", "tier": "disk",
         "trust_remote_code": False},
        {"event": "shape", "layers": 61, "moe_layers": 58, "experts": 256, "hidden": 7168,
         "expert_size": 2048, "parameters": 671_026_419_200, "store_bytes": STORE_BYTES},
        {"event": "adapter_initial", "sha256": _digest(1), "l2_norm": 3.0, "tensors": 10,
         "elements": 1000},
        {"event": "cold_cache", "unix": 1000.0, "method": "drop_caches",
         "store_bytes": STORE_BYTES, "resident_before_bytes": 90 * GIB,
         "resident_after_bytes": 0, "seconds": 5.0},
        {"event": "training_start", "unix": 1100.0},
        {"event": "training_clock", "samples": 100, "throttled_samples": 0},
    ]
    for step in range(3):
        rows.append({
            "event": "step", "step": step, "loss": 2.0 - 0.1 * step, "gradient_norm": 0.5,
            "gradient_tensors": 10, "trainable_tensors": 10, "gradient_nonfinite_tensors": 0,
            "adapter_sha256_before": _digest(step + 1), "adapter_sha256_after": _digest(step + 2),
            "distance_from_initial": 0.01 * (step + 1), "tokens": 128, "seconds": 400.0,
            "tokens_per_second": 0.32, "peak_allocated": 5 * GIB, "peak_reserved": 6 * GIB,
            "expert_fetches": 14_000, "expert_bytes_read": 600 * 10**9,
            "disk_read_bytes": 610 * 10**9,
        })
    rows += [{"event": "reload", "parameter_sha256": _digest(4)}, {"event": "complete"}]
    return rows


def good_store_record() -> Dict[str, Any]:
    return {"planned_bytes": STORE_BYTES, "actual_bytes": STORE_BYTES + 1_000_000,
            "required_free_bytes": 405_632_188_575, "free_before_bytes": 900 * 10**9,
            "seconds": 1800.0}


def test_a_clean_cold_journal_passes() -> None:
    result = b5_cold_acceptance.verdict(good_journal(), good_store_record())

    assert result["passed"] is True, result["failures"]
    assert result["failures"] == []
    assert result["judge"]["passed"] is True


def _codes(journal: List[Dict[str, Any]], record: Any = "good") -> List[str]:
    record = good_store_record() if record == "good" else record
    result = b5_cold_acceptance.verdict(journal, record)
    assert result["passed"] is False
    return [failure.split(":", 1)[0] for failure in result["failures"]]


def _edit(journal: List[Dict[str, Any]], event: str, **values: Any) -> List[Dict[str, Any]]:
    out = copy.deepcopy(journal)
    for row in out:
        if row["event"] == event:
            row.update(values)
    return out


def _step(journal: List[Dict[str, Any]], number: int) -> Dict[str, Any]:
    return next(row for row in journal if row["event"] == "step" and row["step"] == number)


def _drop(journal: List[Dict[str, Any]], event: str) -> List[Dict[str, Any]]:
    return [row for row in journal if row["event"] != event]


def test_a_real_signature_fails_because_this_report_is_synthetic() -> None:
    journal = _edit(good_journal(), "signature", kind="real", revision="abc")

    assert "signature" in _codes(journal)


def test_a_shape_other_than_deepseek_v3_fails() -> None:
    journal = _edit(good_journal(), "shape", experts=64)

    assert "shape" in _codes(journal)


def test_a_missing_shape_event_fails() -> None:
    assert "shape" in _codes(_drop(good_journal(), "shape"))


def test_the_step_count_is_exactly_three() -> None:
    journal = [row for row in good_journal() if not (row["event"] == "step" and row["step"] == 2)]

    assert "steps" in _codes(journal)


def test_a_padded_step_fails() -> None:
    journal = good_journal()
    _step(journal, 1)["tokens"] = 127

    assert "padding" in _codes(journal)


def test_a_card_outside_the_8_gb_class_fails() -> None:
    journal = _edit(good_journal(), "environment", vram_bytes=24 * GIB)

    assert "card" in _codes(journal)


def test_a_card_without_bf16_fails() -> None:
    journal = _edit(good_journal(), "environment", compute_capability=[7, 5])

    assert "card" in _codes(journal)


def test_a_wrong_setup_field_fails() -> None:
    journal = _edit(good_journal(), "environment", tier="ram")

    assert "environment" in _codes(journal)


def test_a_peak_above_the_card_fails() -> None:
    journal = _edit(good_journal(), "step", peak_reserved=CARD + 1)

    assert "vram_reserved" in _codes(journal)


def test_a_missing_peak_fails() -> None:
    journal = good_journal()
    del _step(journal, 1)["peak_allocated"]

    assert "peak_missing" in _codes(journal)


def test_a_run_without_complete_fails() -> None:
    assert "incomplete" in _codes(_drop(good_journal(), "complete"))


def test_a_missing_store_record_fails() -> None:
    assert "store_record" in _codes(good_journal(), record=None)


def test_a_store_larger_than_its_plan_by_more_than_one_percent_fails() -> None:
    record = good_store_record()
    record["actual_bytes"] = int(STORE_BYTES * 1.011)

    assert "store_size" in _codes(good_journal(), record)


def test_a_store_smaller_than_its_plan_by_more_than_one_percent_fails() -> None:
    record = good_store_record()
    record["actual_bytes"] = int(STORE_BYTES * 0.98)

    assert "store_size" in _codes(good_journal(), record)


def test_free_bytes_below_the_plan_fail() -> None:
    record = good_store_record()
    record["free_before_bytes"] = 200 * 10**9

    assert "disk_free" in _codes(good_journal(), record)


def test_a_run_without_a_cache_drop_fails() -> None:
    assert "cold_cache" in _codes(_drop(good_journal(), "cold_cache"))


def test_a_cache_that_still_holds_the_store_fails() -> None:
    journal = _edit(good_journal(), "cold_cache", resident_after_bytes=STORE_BYTES // 10)

    assert "cold_cache" in _codes(journal)


def test_a_drop_far_before_the_first_step_fails() -> None:
    journal = _edit(good_journal(), "cold_cache", unix=100.0)

    assert "cold_gap" in _codes(journal)


def test_a_drop_after_training_started_fails() -> None:
    journal = _edit(good_journal(), "cold_cache", unix=1200.0)

    assert "cold_gap" in _codes(journal)


def test_a_first_step_that_read_little_from_disk_is_not_cold() -> None:
    journal = good_journal()
    _step(journal, 0)["disk_read_bytes"] = 10**9

    assert "cold_unconfirmed" in _codes(journal)


def test_a_first_step_without_a_disk_read_counter_is_not_confirmed() -> None:
    journal = good_journal()
    del _step(journal, 0)["disk_read_bytes"]

    assert "cold_unconfirmed" in _codes(journal)


def test_a_clean_journal_with_a_warm_label_still_needs_the_drop_method_named() -> None:
    journal = _edit(good_journal(), "cold_cache", method="")

    assert "cold_cache" in _codes(journal)


def test_the_judge_failure_is_passed_through() -> None:
    journal = _edit(good_journal(), "step", gradient_norm=0.0)

    assert "judge" in _codes(journal)


def test_speed_is_measured_from_later_steps_and_the_first_is_reported_apart() -> None:
    journal = good_journal()
    steps = [row for row in journal if row["event"] == "step"]
    for row, speed in zip(steps, (0.1, 0.3, 0.5)):
        row["tokens_per_second"] = speed

    speed = b5_cold_acceptance.verdict(journal, good_store_record())["measured"][
        "tokens_per_second"]

    assert speed == {"status": "measured", "first_step": 0.1, "later_steps_median": 0.4}


def test_speed_of_a_throttled_run_is_not_a_measurement() -> None:
    journal = _edit(good_journal(), "training_clock", throttled_samples=60)

    speed = b5_cold_acceptance.verdict(journal, good_store_record())["measured"][
        "tokens_per_second"]

    assert speed["status"] == "not measured"
    assert "throttled" in speed["reason"]


def test_disk_read_per_step_is_reported() -> None:
    measured = b5_cold_acceptance.verdict(good_journal(), good_store_record())["measured"]

    assert measured["disk_read_bytes_per_step"] == [610 * 10**9] * 3


def test_the_run_config_is_accepted_by_the_schema_and_matches_the_rule() -> None:
    from pathlib import Path

    import yaml

    from soup_cli.config.schema import SoupConfig

    path = (Path(__file__).resolve().parent.parent
            / "benchmarks/protocols/b5-synthetic671b-20261109/soup.yaml")
    config = SoupConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    assert config.data.max_length == b5_cold_acceptance.TOKENS_PER_STEP
    assert config.training.stream_expert_layout == "per_expert"
    assert config.training.stream_source == "disk"
    assert config.training.quantization == "4bit"
    assert config.training.batch_size == 1
    assert config.training.gradient_accumulation_steps == 1
    assert not config.training.l2l


def test_the_rule_shape_is_the_deepseek_v3_preset_of_the_writer() -> None:
    from benchmarks.harness import b5_synth_store as synth

    shape = synth.DEEPSEEK_V3
    expected = {"layers": shape.layers, "moe_layers": shape.moe_layers,
                "experts": shape.experts, "hidden": shape.hidden,
                "expert_size": shape.expert_size, "parameters": synth.total_params(shape)}

    assert b5_cold_acceptance.SHAPE == expected


def test_a_store_record_with_assumed_free_bytes_is_refused() -> None:
    record = good_store_record()
    record["free_assumed"] = True

    assert "store_record" in _codes(good_journal(), record)


def test_a_failed_store_write_is_refused() -> None:
    record = good_store_record()
    record["failed"] = True

    assert "store_record" in _codes(good_journal(), record)


def test_non_expert_disk_reads_are_reported_per_step_next_to_the_expert_bytes() -> None:
    journal = good_journal()
    _step(journal, 0)["disk_read_bytes"] = 650 * 10**9  # 50 GB beyond the 600 GB of experts

    measured = b5_cold_acceptance.verdict(journal, good_store_record())["measured"]

    assert measured["non_expert_disk_read_bytes_per_step"] == [50 * 10**9, 10 * 10**9,
                                                               10 * 10**9]


def _warm_journal() -> List[Dict[str, Any]]:
    journal = _drop(good_journal(), "cold_cache")
    return journal


def test_a_warm_repeat_passes_without_a_cache_drop_and_says_it_is_warm() -> None:
    result = b5_cold_acceptance.verdict(_warm_journal(), good_store_record(), warm=True)

    assert result["passed"] is True, result["failures"]
    assert result["mode"] == "warm"


def test_a_warm_repeat_that_has_a_cache_drop_is_mislabelled_and_fails() -> None:
    result = b5_cold_acceptance.verdict(good_journal(), good_store_record(), warm=True)

    assert result["passed"] is False
    assert [item.split(":", 1)[0] for item in result["failures"]] == ["warm"]


def test_a_warm_repeat_still_needs_the_shape_the_steps_and_the_judge() -> None:
    journal = _edit(_warm_journal(), "shape", experts=64)
    journal = _edit(journal, "step", gradient_norm=0.0)

    result = b5_cold_acceptance.verdict(journal, good_store_record(), warm=True)

    assert {item.split(":", 1)[0] for item in result["failures"]} >= {"shape", "judge"}


def test_a_cold_verdict_names_its_mode() -> None:
    assert b5_cold_acceptance.verdict(good_journal(), good_store_record())["mode"] == "cold"


def test_the_size_tolerance_is_the_one_of_the_writer() -> None:
    from benchmarks.harness import b5_synth_store as synth

    assert b5_cold_acceptance.SIZE_TOLERANCE == synth.SIZE_TOLERANCE


def test_the_committed_run_journals_pass_the_committed_rule() -> None:
    """The journals of the rented run, judged again by the rule file's code, pass."""
    import json
    from pathlib import Path

    from benchmarks.harness import b5_judge

    root = Path(__file__).resolve().parent.parent / (
        "benchmarks/evidence/b5-synthetic671b-20261109/attempt3/host/results")
    if not root.is_dir():
        return
    record = json.loads((root / "store/store-record.json").read_text(encoding="utf-8"))
    cold = b5_judge.read_journal(root / "cold1/run/journal.jsonl")
    warm = b5_judge.read_journal(root / "warm1/run/journal.jsonl")

    assert b5_cold_acceptance.verdict(cold, record)["failures"] == []
    assert b5_cold_acceptance.verdict(warm, record, warm=True)["failures"] == []
    assert b5_cold_acceptance.verdict(warm, record)["passed"] is False  # no cache drop


def test_the_report_is_signed_synthetic_on_every_heading_and_table() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent
            / "benchmarks/b5-synthetic671b-report3-20261109.md").read_text(encoding="utf-8")
    labelled = [line for line in text.splitlines() if line.startswith("#") or line.startswith("**")]

    assert labelled and all(line.rstrip("* ").endswith("синтетические веса") for line in labelled)
    assert text.index("## Размер формы") < text.index("## Запуск и окружение")
