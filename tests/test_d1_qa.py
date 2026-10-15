"""Real-task study guards: held-out QA scoring and answer-only token labels."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


def test_qa_scoring_keeps_answerability_separate():
    from d1_qa import score

    result = score(
        [
            {"prediction": "The Eiffel Tower.", "answers": ["Eiffel Tower"], "unanswerable": False},
            {"prediction": "NO_ANSWER", "answers": [], "unanswerable": True},
            {"prediction": "Paris", "answers": [], "unanswerable": True},
        ]
    )
    assert result["exact_match"] == pytest.approx(200 / 3)
    assert result["f1"] == pytest.approx(200 / 3)
    assert result["false_answer_rate_unanswerable"] == 0.5
    assert result["answerable"]["count"] == 1
    assert result["unanswerable"]["count"] == 2


def test_missing_prediction_is_not_an_abstention():
    from d1_qa import score

    with pytest.raises(ValueError, match="prediction"):
        score([{"answers": [], "unanswerable": True}])


class Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return [1, 2, 3]

    def encode(self, text, **_kwargs):
        return [4, 5] if text != "NO_ANSWER" else [6]


def test_prompt_and_padding_are_masked_but_answer_and_eos_train():
    from d1_qa import tokenize

    row = {
        "context": "context",
        "question": "question",
        "answers": [{"text": "answer"}],
        "unanswerable": False,
    }
    tokens = tokenize(Tokenizer(), row, 8)
    assert tokens["input_ids"] == [1, 2, 3, 4, 5, 99, 0, 0]
    assert tokens["labels"] == [-100, -100, -100, 4, 5, 99, -100, -100]
    assert tokens["attention_mask"] == [1, 1, 1, 1, 1, 1, 0, 0]


def test_real_example_is_never_silently_truncated():
    from d1_qa import tokenize

    with pytest.raises(ValueError, match="truncation"):
        tokenize(
            Tokenizer(), {"context": "x", "question": "x", "answers": [], "unanswerable": True}, 3
        )


def test_tokenizer_batchencoding_result_is_supported():
    from d1_qa import tokenize

    class MappingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **_kwargs):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    row = {"context": "x", "question": "x", "answers": [], "unanswerable": True}
    assert tokenize(MappingTokenizer(), row, 8)["input_ids"] == [1, 2, 3, 6, 99, 0, 0, 0]
