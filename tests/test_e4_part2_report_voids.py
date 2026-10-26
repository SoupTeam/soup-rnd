"""The report counts voids of every run in the journal, not only the expected stage list, and
an attempt with a start but no end is listed as unclosed (not silently dropped). CPU only."""

# ruff: noqa: I001 (the support import must run before the harness imports)
import tests.e4_part2_support  # noqa: F401  (puts the harness on sys.path)
import e4_part2_common as common  # noqa: E402


def journal(tmp_path):
    return common.AttemptsJournal(tmp_path / "attempts.jsonl")


def test_run_ids_include_every_run_in_the_journal(tmp_path):
    j = journal(tmp_path)
    j.end(j.start("T", "T"), common.OUTCOME_OK)
    j.end(j.start("T_fallback_gateup", "T"), common.OUTCOME_VOID, ["crash: boom"])
    assert j.run_ids() == ["T", "T_fallback_gateup"]


def test_a_void_of_a_targeted_run_is_listed(tmp_path):
    j = journal(tmp_path)
    j.end(j.start("T_fallback_gateup", "T"), common.OUTCOME_VOID, ["crash: M=384"])
    entries = j.void_entries()
    assert [(e["run_id"], e["outcome"]) for e in entries] == [("T_fallback_gateup", "void")]


def test_an_interrupted_start_with_no_end_is_listed_as_unclosed(tmp_path):
    j = journal(tmp_path)
    j.start("T_fallback_gateup", "T")  # the process was killed here: no end event
    entries = j.void_entries()
    assert len(entries) == 1
    assert entries[0]["run_id"] == "T_fallback_gateup" and entries[0]["outcome"] == "unclosed"


def test_ok_attempts_are_not_voids(tmp_path):
    j = journal(tmp_path)
    j.end(j.start("N_4", "N"), common.OUTCOME_OK)
    assert j.void_entries() == []


def test_an_unclosed_attempt_is_a_provenance_gap_even_for_an_extra_run(tmp_path):
    j = journal(tmp_path)
    j.start("T_fallback_gateup", "T")
    assert j.provenance_gap(["T"]) == ["T: no journal entry"]
    j2 = common.AttemptsJournal(tmp_path / "second.jsonl")
    j2.start("T_fallback_gateup", "T")
    gaps = j2.provenance_gap(j2.run_ids())
    assert gaps == ["T_fallback_gateup: attempt without an end"]
