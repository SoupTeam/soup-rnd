"""B5 ticket 02: the shared judge of adapter gradients and updates.

The journals here are written by hand, so the tests check the rule in
``benchmarks/b5-adapter-judge-rule-20261109.md`` and nothing about training. The recorder and a
real tiny-MoE run are covered in ``test_b5_recorder.py``.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

import pytest

from benchmarks.harness import b5_judge

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def good_journal(kind: str = "real") -> List[Dict[str, Any]]:
    """Two steps, finite nonzero gradients, the adapter moves and reloads to the last digest."""
    signature: Dict[str, Any] = {"event": "signature", "kind": kind, "model": "tiny-moe"}
    if kind == "real":
        signature["revision"] = "476b36a473d4467f94469414bef6cee75c9c8172"
    else:
        signature["generator"] = "b13_moe_fixture.moe_weights_dir seed=61"
    return [
        signature,
        {"event": "adapter_initial", "sha256": HEX_A, "l2_norm": 20.0, "tensors": 10,
         "elements": 1000},
        {"event": "step", "step": 0, "loss": 4.2, "gradient_norm": 0.57,
         "gradient_tensors": 10, "trainable_tensors": 10, "gradient_nonfinite_tensors": 0,
         "adapter_sha256_before": HEX_A, "adapter_sha256_after": HEX_B,
         "distance_from_initial": 0.05},
        {"event": "step", "step": 1, "loss": 3.8, "gradient_norm": 0.48,
         "gradient_tensors": 10, "trainable_tensors": 10, "gradient_nonfinite_tensors": 0,
         "adapter_sha256_before": HEX_B, "adapter_sha256_after": HEX_C,
         "distance_from_initial": 0.09},
        {"event": "reload", "parameter_sha256": HEX_C},
        {"event": "complete"},
    ]


def test_a_journal_with_moving_adapters_and_a_signature_passes() -> None:
    result = b5_judge.judge([good_journal()])

    assert result["passed"] is True, result
    assert result["failures"] == []
    assert result["kind"] == "real"


def _steps(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in journal if row["event"] == "step"]


def _drop(journal: List[Dict[str, Any]], event: str) -> List[Dict[str, Any]]:
    return [row for row in journal if row["event"] != event]


def _zero_all(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in _steps(journal):
        row["gradient_norm"] = 0.0
    return journal


def _zero_one(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _steps(journal)[1]["gradient_norm"] = 1e-9
    return journal


def _nan_loss(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _steps(journal)[0]["loss"] = float("nan")
    return journal


def _inf_norm(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _steps(journal)[1]["gradient_norm"] = float("inf")
    return journal


def _nonfinite_tensor(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _steps(journal)[0]["gradient_nonfinite_tensors"] = 1
    return journal


def _missing_grad(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _steps(journal)[0]["gradient_tensors"] = 9
    return journal


def _unchanged(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in _steps(journal):
        row["adapter_sha256_before"] = row["adapter_sha256_after"] = HEX_A
        row["distance_from_initial"] = 0.0
    return journal


def _first_step_only(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The first step moves the adapter, the last one has returned to the start."""
    _steps(journal)[1]["adapter_sha256_after"] = HEX_A
    _steps(journal)[1]["distance_from_initial"] = 0.0
    journal[-2]["parameter_sha256"] = HEX_A
    return journal


def _tiny_move(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 1e-5 against a norm of 20 is a relative 5e-7, under the 1e-6 threshold.
    _steps(journal)[0]["distance_from_initial"] = 1e-5
    return journal


def _digest_distance_clash(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _steps(journal)[0]["distance_from_initial"] = 0.0
    return journal


def _no_initial(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _drop(journal, "adapter_initial")


def _initial_digest_differs(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _steps(journal)[0]["adapter_sha256_before"] = HEX_C
    return journal


def _broken_chain(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _steps(journal)[1]["adapter_sha256_before"] = HEX_A
    return journal


def _one_step(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal.remove(_steps(journal)[1])
    journal[-2]["parameter_sha256"] = HEX_B
    return journal


def _no_reload(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _drop(journal, "reload")


def _reload_differs(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal[-2]["parameter_sha256"] = HEX_A
    return journal


def _no_signature(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _drop(journal, "signature")


def _bad_kind(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal[0]["kind"] = "toy"
    return journal


def _real_without_revision(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    del journal[0]["revision"]
    return journal


def _two_signatures_one_journal(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal.insert(1, {"event": "signature", "kind": "synthetic", "model": "x",
                       "generator": "g"})
    return journal


def _stalled_middle_step(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Three steps; the optimizer did nothing on the middle one, first and last still move."""
    first, last = _steps(journal)
    middle = dict(first, step=1, adapter_sha256_before=HEX_B, adapter_sha256_after=HEX_B,
                  distance_from_initial=first["distance_from_initial"])
    last.update(step=2)
    journal.insert(journal.index(last), middle)
    return journal


def _duplicate_step_index(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _steps(journal)[1]["step"] = 0
    return journal


def _initial_tensor_count_differs(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal[1]["tensors"] = 99
    return journal


def _two_initial_records(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal.insert(2, dict(journal[1], sha256=HEX_C))
    return journal


def _early_bad_reload(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal.insert(-2, {"event": "reload", "parameter_sha256": HEX_A})
    return journal


def _reload_logits_differ(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal[-2]["equal"] = False
    return journal


def _abort_event(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal.insert(-2, {"event": "abort", "reason": "non-finite gradient_norm at step 2"})
    return journal


@pytest.mark.parametrize(("mutate", "code"), [
    pytest.param(_zero_all, "zero_gradient", id="zero-gradients-on-all-steps"),
    pytest.param(_zero_one, "zero_gradient", id="zero-gradient-on-one-step"),
    pytest.param(_nan_loss, "nonfinite", id="nan-loss"),
    pytest.param(_inf_norm, "nonfinite", id="inf-gradient-norm"),
    pytest.param(_nonfinite_tensor, "nonfinite", id="nonfinite-gradient-tensor"),
    pytest.param(_missing_grad, "gradient_missing", id="tensor-without-gradient"),
    pytest.param(_unchanged, "no_update", id="adapter-did-not-change"),
    pytest.param(_first_step_only, "no_update", id="adapter-back-at-start-after-last-step"),
    pytest.param(_tiny_move, "no_update", id="move-below-threshold"),
    pytest.param(_digest_distance_clash, "digest_distance_clash", id="digest-vs-distance"),
    pytest.param(_no_initial, "no_initial", id="no-initial-adapter-record"),
    pytest.param(_initial_digest_differs, "digest_chain", id="step0-base-is-not-initial"),
    pytest.param(_broken_chain, "digest_chain", id="digest-chain-broken"),
    pytest.param(_one_step, "too_few_steps", id="one-step-only"),
    pytest.param(_no_reload, "reload", id="no-reload-record"),
    pytest.param(_reload_differs, "reload", id="reloaded-digest-differs"),
    pytest.param(_no_signature, "signature", id="no-signature"),
    pytest.param(_bad_kind, "signature", id="unknown-signature-kind"),
    pytest.param(_real_without_revision, "signature", id="real-without-revision"),
    pytest.param(_two_signatures_one_journal, "mixed_signatures",
                 id="two-kinds-in-one-journal"),
    pytest.param(_stalled_middle_step, "no_update", id="middle-step-did-not-move-the-adapter"),
    pytest.param(_duplicate_step_index, "step_order", id="duplicate-step-index"),
    pytest.param(_initial_tensor_count_differs, "initial_mismatch", id="initial-tensor-count"),
    pytest.param(_two_initial_records, "initial_mismatch", id="two-initial-records"),
    pytest.param(_early_bad_reload, "reload", id="earlier-reload-differs"),
    pytest.param(_reload_logits_differ, "reload", id="reload-logits-differ"),
    pytest.param(_abort_event, "aborted", id="run-aborted-by-the-recorder"),
])
def test_a_defective_journal_fails_with_its_reason(mutate: Any, code: str) -> None:
    result = b5_judge.judge([mutate(copy.deepcopy(good_journal()))])

    assert result["passed"] is False, result
    assert any(failure.startswith(code + ":") for failure in result["failures"]), result


def test_one_report_cannot_mix_real_and_synthetic_journals() -> None:
    result = b5_judge.judge([good_journal("real"), good_journal("synthetic")])

    assert result["passed"] is False
    assert any(f.startswith("mixed_signatures:") for f in result["failures"]), result
    assert result["kind"] is None


def test_two_journals_of_one_kind_pass_together() -> None:
    result = b5_judge.judge([good_journal("synthetic"), good_journal("synthetic")])

    assert result["passed"] is True, result
    assert result["kind"] == "synthetic"


def test_a_journal_line_that_is_not_json_fails(tmp_path: Any) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(
        [json.dumps(row) for row in good_journal()[:2]] + ["{not json"]
        + [json.dumps(row) for row in good_journal()[2:]]) + "\n", encoding="utf-8")

    result = b5_judge.judge([b5_judge.read_journal(path)])

    assert result["passed"] is False
    assert any(f.startswith("unparsable:") for f in result["failures"]), result


def test_a_nan_written_by_hand_into_the_file_fails(tmp_path: Any) -> None:
    path = tmp_path / "events.jsonl"
    rows = [json.dumps(row) for row in good_journal()]
    rows[2] = rows[2].replace('"loss": 4.2', '"loss": NaN')
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = b5_judge.judge([b5_judge.read_journal(path)])

    assert result["passed"] is False
    assert any(f.startswith("nonfinite:") for f in result["failures"]), result


# --------------------------------------------------------------------------
# report template
# --------------------------------------------------------------------------
def report_record(kind: str = "real") -> Dict[str, Any]:
    return {
        "kind": kind,
        "title": "Малая MoE на 8 GB GPU",
        "command": "python benchmarks/harness/b6_real_moe.py --steps 2",
        "environment": {
            "torch": "2.8.0+cu128", "transformers": "5.0.0", "peft": "0.17.1",
            "bitsandbytes": "0.47.0", "gpu": "NVIDIA GeForce RTX 3070 8 GB", "ram": "32 GiB",
        },
        "logs": ["events.jsonl", "run.log"],
        "verdict": b5_judge.judge([good_journal(kind)]),
        "sections": [
            {"heading": "Градиенты", "tables": [
                {"caption": "Градиентная норма по шагам",
                 "rows": [["шаг 0", "0.57"], ["шаг 1", "0.48"]]}]},
            {"heading": "Скорость", "tables": [
                {"caption": "Токены в секунду", "rows": [["среднее", "1.2"]]}]},
        ],
    }


@pytest.mark.parametrize("kind", ["real", "synthetic"])
def test_every_heading_and_table_caption_carries_the_signature(kind: str) -> None:
    text = b5_judge.render_report(report_record(kind))
    label = b5_judge.KINDS[kind]
    other = b5_judge.KINDS["synthetic" if kind == "real" else "real"]

    lines = text.splitlines()
    headings = [line for line in lines if line.startswith("#")]
    captions = [line for line in lines if line.startswith("**") and line.endswith("**")]
    tables = [i for i, line in enumerate(lines)
              if line.startswith("|") and not lines[i - 1].startswith("|")]
    assert len(headings) >= 4 and len(captions) >= 4
    assert all(line.endswith(" — " + label) for line in headings), headings
    assert all(line.endswith(" — " + label + "**") for line in captions), captions
    # a table is the line after its caption and a blank line, never bare
    assert all(lines[i - 2] in captions for i in tables), tables
    assert other not in text


def test_the_report_carries_command_versions_hardware_and_log_links() -> None:
    text = b5_judge.render_report(report_record())

    for expected in ("python benchmarks/harness/b6_real_moe.py --steps 2", "2.8.0+cu128",
                     "5.0.0", "0.17.1", "0.47.0", "RTX 3070", "32 GiB", "events.jsonl",
                     "run.log", "PASS"):
        assert expected in text, expected


@pytest.mark.parametrize("field", ["torch", "transformers", "peft", "bitsandbytes", "gpu", "ram"])
def test_a_missing_environment_field_is_refused_unless_a_reason_is_given(field: str) -> None:
    record = report_record()
    del record["environment"][field]

    with pytest.raises(ValueError, match=field):
        b5_judge.render_report(record)

    record["unavailable"] = {field: "хост без bitsandbytes"}
    assert "не измерено (хост без bitsandbytes)" in b5_judge.render_report(record)


@pytest.mark.parametrize("field", ["command", "logs"])
def test_a_report_without_command_or_logs_is_refused(field: str) -> None:
    record = report_record()
    record[field] = [] if field == "logs" else ""

    with pytest.raises(ValueError, match=field):
        b5_judge.render_report(record)


def test_a_record_whose_verdict_is_for_the_other_kind_is_refused() -> None:
    record = report_record("real")
    record["verdict"] = b5_judge.judge([good_journal("synthetic")])

    with pytest.raises(b5_judge.MixedRecordsError):
        b5_judge.render_report(record)


def test_a_record_built_from_mixed_journals_is_refused() -> None:
    record = report_record("real")
    record["verdict"] = b5_judge.judge([good_journal("real"), good_journal("synthetic")])

    with pytest.raises(b5_judge.MixedRecordsError):
        b5_judge.render_report(record)


def test_a_section_of_the_other_kind_is_refused() -> None:
    record = report_record("real")
    record["sections"][0]["kind"] = "synthetic"

    with pytest.raises(b5_judge.MixedRecordsError):
        b5_judge.render_report(record)


def test_a_failed_verdict_is_shown_with_its_reasons() -> None:
    record = report_record()
    record["verdict"] = b5_judge.judge([_zero_all(copy.deepcopy(good_journal()))])

    text = b5_judge.render_report(record)

    assert "FAIL" in text and "zero_gradient" in text


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def _write_journal(path: Any, rows: List[Dict[str, Any]]) -> str:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return str(path)


def test_the_judge_command_exits_0_on_pass_and_1_on_fail(tmp_path: Any) -> None:
    good = _write_journal(tmp_path / "good.jsonl", good_journal())
    bad = _write_journal(tmp_path / "bad.jsonl", _zero_all(copy.deepcopy(good_journal())))

    assert b5_judge.main(["judge", "--journal", good]) == 0
    assert b5_judge.main(["judge", "--journal", bad]) == 1
    assert b5_judge.main(["judge", "--journal", good,
                          "--journal", _write_journal(tmp_path / "s.jsonl",
                                                      good_journal("synthetic"))]) == 1


def test_the_report_command_refuses_the_empty_template_and_never_overwrites(
    tmp_path: Any, capsys: Any,
) -> None:
    assert b5_judge.main(["template", "--kind", "real"]) == 0
    template = json.loads(capsys.readouterr().out)
    record = tmp_path / "record.json"
    record.write_text(json.dumps(template), encoding="utf-8")
    out = tmp_path / "report.md"

    assert b5_judge.main(["report", "--record", str(record), "--out", str(out)]) == 3
    assert not out.exists()

    record.write_text(json.dumps(report_record()), encoding="utf-8")
    assert b5_judge.main(["report", "--record", str(record), "--out", str(out)]) == 0
    assert b5_judge.main(["report", "--record", str(record), "--out", str(out)]) == 2


def test_the_template_command_prints_json_that_loads_back(capsys: Any) -> None:
    assert b5_judge.main(["template", "--kind", "synthetic"]) == 0

    assert json.loads(capsys.readouterr().out)["kind"] == "synthetic"
