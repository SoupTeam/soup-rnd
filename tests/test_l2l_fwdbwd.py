"""CPU tests for the A5-03 forward/backward test (benchmarks/harness/l2l_fwdbwd.py).

The 7B NF4 run needs a GPU. These pin what the script decides and writes: how windows are
cut, how the tolerance of A3-07 is applied per window, which counter findings are
discrepancies, the report, and one end-to-end run on the tiny CPU model.
"""

from __future__ import annotations

import json

import pytest

from tests.test_l2l_probe_harness import _load, streamed  # noqa: F401

LENGTHS = [8, 6, 5, 8, 3, 7, 4]  # 7 rows: windows of 4+3 at k=4, 2+2+2+1 at k=2


@pytest.fixture()
def fwdbwd():
    _load("l2l_schedule")
    _load("l2l_mistral_stand")
    return _load("l2l_fwdbwd")


def _rows(fwdbwd, lengths=LENGTHS):
    spans = [[(0, 2)] if i % 2 else [] for i in range(len(lengths))]
    return fwdbwd.build_rows(lengths, spans, vocab=64)


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------
def test_windows_keep_order_and_end_with_an_incomplete_one(fwdbwd):
    rows = _rows(fwdbwd)
    sizes = [len(w) for w in fwdbwd.cut_windows(rows, 4)]
    assert sizes == [4, 3]
    assert [len(w) for w in fwdbwd.cut_windows(rows, 2)] == [2, 2, 2, 1]
    assert [len(w) for w in fwdbwd.cut_windows(rows, 1)] == [1] * 7


def test_rows_have_different_lengths_and_masked_labels(fwdbwd):
    rows = _rows(fwdbwd)
    assert [int(ids.shape[1]) for ids, _ in rows] == LENGTHS
    assert any(int((labels == -100).sum()) for _, labels in rows)


def test_a_case_without_an_incomplete_last_window_is_refused(fwdbwd):
    rows = _rows(fwdbwd, [8, 6, 5, 8])
    with pytest.raises(fwdbwd.FwdBwdRefusedError, match="incomplete"):
        fwdbwd.check_case_shape(rows, 4)


def test_a_case_with_equal_lengths_or_no_masked_labels_is_refused(fwdbwd):
    same = fwdbwd.build_rows([8] * 7, [[(0, 2)]] * 7, vocab=64)
    with pytest.raises(fwdbwd.FwdBwdRefusedError, match="length"):
        fwdbwd.check_case_shape(same, 4)
    bare = fwdbwd.build_rows(LENGTHS, [[]] * 7, vocab=64)
    with pytest.raises(fwdbwd.FwdBwdRefusedError, match="-100"):
        fwdbwd.check_case_shape(bare, 4)


def test_n_items_counts_shifted_labels_that_are_not_masked(fwdbwd):
    import torch

    ids = torch.zeros((1, 5), dtype=torch.long)
    labels = torch.tensor([[-100, 3, -100, 4, 5]])
    assert fwdbwd.n_items([(ids, labels)]) == 3  # positions 1..4 hold 3, 4, 5 -> 3 of 4 count


# --------------------------------------------------------------------------
# counter findings
# --------------------------------------------------------------------------
def _counts(custom=0, dequant=4):
    return {2048: {"custom": custom, "dequant": dequant}}


def _facts(**over):
    base = {
        "gemm": {"ga1": _counts(), "ga2": _counts(), "l2l": _counts()},
        "loads_first_full_window": {1: 32, 2: 32, 4: 32},
        "restores": {1: 0, 2: 0, 4: 0},
        "restores_total": 32,
    }
    base.update(over)
    return base


def test_clean_counters_have_no_findings(fwdbwd):
    assert fwdbwd.counter_findings(_facts(), quant="nf4") == []


def test_an_empty_gemm_counter_on_either_arm_is_a_finding(fwdbwd):
    for arm in ("ga1", "l2l"):
        gemm = {"ga1": _counts(), "ga2": _counts(), "l2l": _counts(), arm: {}}
        found = fwdbwd.counter_findings(_facts(gemm=gemm), quant="nf4")
        assert any(f["kind"] == "gemm_empty" and f["arm"] == arm for f in found)


def test_four_times_the_loads_at_k4_is_a_finding(fwdbwd):
    found = fwdbwd.counter_findings(
        _facts(loads_first_full_window={1: 32, 2: 64, 4: 128}), quant="nf4"
    )
    assert [f["kind"] for f in found] == ["loads_scale_with_k"]


def test_zero_restores_in_the_l2l_arm_alone_is_not_a_finding(fwdbwd):
    # the runtime caches the rebuilt views, so only the first forward counts restores
    assert fwdbwd.counter_findings(_facts(restores={1: 0, 2: 0, 4: 0}), quant="nf4") == []


def test_zero_restores_over_the_whole_run_on_nf4_is_a_discrepancy(fwdbwd):
    found = fwdbwd.counter_findings(_facts(restores_total=0), quant="nf4")
    assert [f["kind"] for f in found] == ["restores_zero"]
    assert "discrepancy" in found[0]["note"]


def test_zero_restores_without_quantization_is_expected(fwdbwd):
    assert fwdbwd.counter_findings(_facts(restores_total=0), quant="none") == []


def test_missing_loads_for_a_k_is_a_finding(fwdbwd):
    found = fwdbwd.counter_findings(
        _facts(loads_first_full_window={1: 32}), quant="nf4"
    )
    assert [f["kind"] for f in found] == ["loads_missing"]


# --------------------------------------------------------------------------
# verdict over the cases
# --------------------------------------------------------------------------
def test_overall_verdict_needs_every_case_and_no_finding(fwdbwd):
    ok = {"pass": True, "status": "within_tolerance"}
    bad = {"pass": False, "status": "l2l_out_of_tolerance"}
    assert fwdbwd.overall({1: ok, 2: ok, 4: ok}, [])["pass"] is True
    assert fwdbwd.overall({1: ok, 2: bad, 4: ok}, [])["pass"] is False
    out = fwdbwd.overall({1: ok, 2: ok, 4: ok}, [{"kind": "restores_zero", "note": "x"}])
    assert out["pass"] is False and out["status"] == "counter_findings"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _payload(**over):
    tol = {"mode": "fp32", "factor": 1.5, "limit_fp32": 1e-4, "grad_rel": [1e-7, 2e-7],
           "grad_limit": [1e-4, 1e-4], "loss_rel": [0.0, 0.0], "loss_limit": [1e-4, 1e-4],
           "floor_grad": None, "floor_loss": None, "judged": True, "failing": [], "pass": True}
    verdict = {"pass": True, "status": "within_tolerance", "note": "",
               "a_vs_a": {"equal": True, "max_abs": 0.0},
               "l2l_vs_ga": {"equal": False, "max_abs": 1e-8, "steps": (2, 2)}, "tolerance": tol}
    base = {
        "data_kind": "synthetic", "model": "mistralai/Mistral-7B-v0.1", "quant": "nf4",
        "card": "8gb", "command": ["python", "l2l_fwdbwd.py"], "env": {"gpu": "RTX 3060"},
        "mode": "fp32", "cases": {1: verdict, 2: verdict, 4: verdict},
        "facts": _facts(), "findings": [],
        "overall": {"pass": True, "status": "within_tolerance", "note": ""},
        "rows": {"lengths": LENGTHS, "masked": 3},
    }
    base.update(over)
    return base


def test_report_prints_the_tolerance_values_and_the_card_label(fwdbwd):
    text = fwdbwd.render_report(_payload())
    assert "1.5" in text and "1e-04" in text
    assert "8gb" in text and "real weights" in text.lower()
    assert "7B" not in text.replace("Mistral-7B", "")  # nothing is extended to bigger models


def test_report_marks_a_finding_as_a_discrepancy(fwdbwd):
    finding = {"kind": "restores_zero", "note": "discrepancy: no NF4 restores"}
    text = fwdbwd.render_report(_payload(
        findings=[finding], overall={"pass": False, "status": "counter_findings", "note": ""}
    ))
    assert "FAIL" in text and "discrepancy: no NF4 restores" in text


def test_report_never_mixes_cards(fwdbwd):
    text = fwdbwd.render_report(_payload(card="24gb"))
    assert "24gb" in text and "8gb" not in text


def test_write_outputs_keeps_report_records_and_never_overwrites(fwdbwd, tmp_path):
    out = tmp_path / "run"
    fwdbwd.write_outputs(out, _payload(), {"ga1.log": "x\n"})
    assert (out / "report.md").read_text()
    assert json.loads((out / "records.json").read_text())["card"] == "8gb"
    assert (out / "raw" / "ga1.log").read_text() == "x\n"
    with pytest.raises(fwdbwd.FwdBwdRefusedError, match="exists"):
        fwdbwd.write_outputs(out, _payload(), {})


# --------------------------------------------------------------------------
# the whole test on the tiny CPU model
# --------------------------------------------------------------------------
def test_tiny_model_passes_for_k_1_2_4(streamed, fwdbwd):  # noqa: F811
    model, runtime = streamed
    result = fwdbwd.run_fwdbwd(
        model, runtime, _rows(fwdbwd), ks=(1, 2, 4), dtype=None, mode="fp32", quant="none",
        device="cpu",
    )
    assert set(result.cases) == {1, 2, 4}
    for k, verdict in result.cases.items():
        assert verdict["status"] == "within_tolerance", (k, verdict["note"])
        assert verdict["a_vs_a"]["equal"] is True
    windows = {k: len(v["tolerance"]["grad_rel"]) for k, v in result.cases.items()}
    assert windows == {1: 7, 2: 4, 4: 2}  # one gradient comparison per window
    loads = result.facts["loads_first_full_window"]
    assert loads[4] < 4 * loads[1]
    assert set(result.logs) >= {"k4_ga1.log", "k4_l2l.log"}
    assert all(f["kind"] != "loads_scale_with_k" for f in result.findings)


def test_tiny_model_fails_when_l2l_gradients_are_off(streamed, fwdbwd, monkeypatch):  # noqa: F811
    model, runtime = streamed
    real = fwdbwd.run_arm

    def broken(*args, **kwargs):
        record = real(*args, **kwargs)
        if kwargs.get("kind") == "l2l":
            record.grads = [{n: g * 1.01 for n, g in step.items()} for step in record.grads]
        return record

    monkeypatch.setattr(fwdbwd, "run_arm", broken)
    result = fwdbwd.run_fwdbwd(
        model, runtime, _rows(fwdbwd), ks=(2,), dtype=None, mode="fp32", quant="none",
        device="cpu",
    )
    assert result.cases[2]["status"] == "l2l_out_of_tolerance"
    assert result.overall["pass"] is False


def test_main_refuses_an_existing_output_directory(fwdbwd, tmp_path, capsys):
    (tmp_path / "taken").mkdir()
    code = fwdbwd.main(["--out", str(tmp_path / "taken"), "--card", "8gb"])
    assert code == 2 and "exists" in capsys.readouterr().out


def test_main_refuses_a_row_count_that_fills_every_window(fwdbwd, tmp_path, capsys):
    code = fwdbwd.main(["--out", str(tmp_path / "x"), "--card", "8gb", "--rows", "8"])
    assert code == 2 and "incomplete" in capsys.readouterr().out


def test_card_label_must_fit_the_gpu(fwdbwd):
    gib = 2**30
    assert fwdbwd.card_matches("8gb", 8 * gib) and not fwdbwd.card_matches("8gb", 24 * gib)
    assert fwdbwd.card_matches("24gb", 24 * gib) and not fwdbwd.card_matches("24gb", 8 * gib)
    assert not fwdbwd.card_matches("8gb", None)


def test_fewer_rows_than_k_is_refused(fwdbwd):
    with pytest.raises(fwdbwd.FwdBwdRefusedError, match="full window"):
        fwdbwd.check_case_shape(_rows(fwdbwd, [8, 6, 5]), 4)
