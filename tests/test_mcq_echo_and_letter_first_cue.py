"""MCQ letter extraction: an echo of the option list is not an answer; "A is correct" is.

Found in the E2 gate-suite records (Mistral-7B, top-16 LoRA): the model repeated
"(A) x (B) y (C) z" instead of answering, and the extractor read the last marker as the
choice, so every item keyed on the last option passed. "A is the correct answer." read as
no answer at all.
"""

import re

import pytest

from soup_cli.eval import forgetting
from soup_cli.eval.forgetting import extract_mcq_letter, score_answer


@pytest.mark.parametrize(
    "output",
    [
        "(A) London (B) Berlin (C) Paris",
        "(A) hands \n(B) head \n(C) feet",
        "(A) 3 (B) 4 (C) 5 (D) 6",
        "(a) one (b) two",
    ],
)
def test_a_bare_echo_of_the_options_is_not_a_choice(output):
    assert extract_mcq_letter(output) is None
    assert not score_answer(output, "C") and not score_answer(output, "A")


@pytest.mark.parametrize(
    ("output", "letter"),
    [
        ("(A) x (B) y (C) z\nB", "B"),  # a bare letter after the echo still counts
        ("(A) 3 (B) 4 (C) 5. Answer: (B)", "B"),  # a cue after the echo wins
        ("(A) x (B) y. Answer: \\boxed{A}", "A"),
        ("(B) 4", "B"),  # one marker is a choice, not an echo
        ("(B) 4 (C) 5", "C"),  # not a list from (A): unchanged behaviour
        ("C. Au", "C"),
    ],
)
def test_choices_around_or_without_an_echo_are_kept(output, letter):
    assert extract_mcq_letter(output) == letter


@pytest.mark.parametrize(
    ("output", "letter"),
    [
        ("A is the correct answer.", "A"),
        ("(B) is correct", "B"),
        ("C is right.", "C"),
        ("D is the right option", "D"),
    ],
)
def test_letter_first_statements_are_a_committed_answer(output, letter):
    assert extract_mcq_letter(output) == letter
    assert score_answer(output, letter)


@pytest.mark.parametrize("output", ["A dog is correct to bark", "a is the correct answer"])
def test_prose_and_lower_case_do_not_trigger_the_letter_first_cue(output):
    assert extract_mcq_letter(output) is None


def test_a_model_that_only_echoes_scores_zero_on_the_bundled_mcq_suites():
    for name in ("MINI_MMLU", "MINI_COMMON_SENSE"):
        items = getattr(forgetting, name)
        echoes = [item["question"][item["question"].find("(A)") :] for item in items]
        assert sum(score_answer(e, i["answer"]) for e, i in zip(echoes, items)) == 0
        keyed_last = [
            i for i in items if re.findall(r"\(([A-J])\)", i["question"])[-1] == i["answer"]
        ]
        assert keyed_last, "the regression needs items keyed on the last option"


def test_fingerprint_corpus_detects_a_revert_of_this_fix(monkeypatch):
    from soup_cli.eval import gate_suites

    fixed = gate_suites.bundled_scorer_fingerprint()
    assert fixed == gate_suites.BUNDLED_SCORER_FINGERPRINT
    assert gate_suites.BUNDLED_SCORER_REVISION >= 5

    def last_marker_wins(output):  # the revision-4 reading of an echo
        markers = re.findall(r"\(([A-J])\)", output)
        return markers[-1] if markers else None

    monkeypatch.setattr(gate_suites, "_fingerprint_responses", None, raising=False)
    monkeypatch.setattr(forgetting, "extract_mcq_letter", last_marker_wins)
    assert gate_suites.bundled_scorer_fingerprint() != fixed
