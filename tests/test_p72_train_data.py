"""Training windows and held-out groups for the adapter-equality run (pure Python, no model).

The record format is the one of ``p3_prepare_fresh_metamath_eval.py`` (prompt tokens, answer tokens
and an end-of-text token in ``input_ids``, ``-100`` over the prompt in ``labels``, a group hash of
the normalised original question), so the evaluation tools of PR #72 read the file too.
"""

from __future__ import annotations

import hashlib

import pytest

from benchmarks.harness import p72_train_data as data


class FakeTokenizer:
    """One token per word; the end-of-text id is 2."""

    eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        ids = [10 + (sum(map(ord, word)) % 50) for word in text.split()]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}


def row(question: str, query: str, response: str, kind: str = "GSM_Rephrased") -> dict:
    return {"original_question": question, "query": query, "response": response, "type": kind}


def test_a_record_has_prompt_labels_masked_and_ends_with_the_eos_token() -> None:
    rec = data.build_record(FakeTokenizer(), row("Q one", "what is two plus two", "four"), 64)
    assert rec is not None
    assert len(rec["input_ids"]) == len(rec["labels"]) == rec["length"]
    prompt_len = rec["labels"].index(next(x for x in rec["labels"] if x != -100))
    assert all(x == -100 for x in rec["labels"][:prompt_len])
    assert rec["input_ids"][-1] == 2 and rec["labels"][-1] == 2
    assert rec["answer_targets"] == sum(x != -100 for x in rec["labels"][1:])
    key = " ".join("Q one".split()).casefold()
    assert rec["group_sha256"] == hashlib.sha256(key.encode()).hexdigest()


def test_a_record_longer_than_the_limit_or_without_an_answer_is_dropped() -> None:
    long_query = " ".join(["word"] * 80)
    assert data.build_record(FakeTokenizer(), row("Q", long_query, "four"), 64) is None
    assert data.build_record(FakeTokenizer(), row("Q", "short", ""), 64) is None


def make_rows(count: int) -> list[dict]:
    rows = []
    for i in range(count):
        kind = "GSM_Rephrased" if i % 2 == 0 else "MATH_AnsAug"
        rows.append(row(f"question {i}", f"what is {i} plus {i}", f"it is {2 * i} for sure", kind))
    return rows


def test_selection_gives_disjoint_train_and_heldout_groups_in_the_requested_sizes() -> None:
    out = data.select_records(
        make_rows(400), FakeTokenizer(), train_count=40, train_max_length=64,
        eval_per_family=10, eval_max_length=256, seed=7,
    )
    train, heldout = out["train"], out["heldout"]
    assert len(train) == 40 and len(heldout) == 20
    assert not {r["group_sha256"] for r in train} & {r["group_sha256"] for r in heldout}
    assert sum(r["type"].startswith("GSM") for r in heldout) == 10
    assert out["train"] == data.select_records(
        make_rows(400), FakeTokenizer(), train_count=40, train_max_length=64,
        eval_per_family=10, eval_max_length=256, seed=7)["train"]          # deterministic


def test_selection_refuses_when_there_are_not_enough_examples() -> None:
    with pytest.raises(RuntimeError, match="not enough"):
        data.select_records(
            make_rows(30), FakeTokenizer(), train_count=40, train_max_length=64,
            eval_per_family=10, eval_max_length=256, seed=7)


def test_windows_group_the_train_records_in_order_and_epochs_repeat_them() -> None:
    records = [{"group_sha256": str(i)} for i in range(16)]
    windows = data.make_windows(records, per_window=8)
    assert [[r["group_sha256"] for r in w] for w in windows] == [
        [str(i) for i in range(8)], [str(i) for i in range(8, 16)]]
    assert data.steps_for_epochs(windows, epochs=2) == windows + windows
    with pytest.raises(ValueError):
        data.make_windows(records[:15], per_window=8)


def test_a_window_is_padded_to_the_sequence_length_with_masked_labels() -> None:
    window = [{"input_ids": [1, 5, 6, 2], "labels": [-100, -100, 6, 2]},
              {"input_ids": [1, 7], "labels": [-100, 7]}]
    ids, labels = data.window_arrays(window, seq=6, pad_id=0)
    assert ids == [[1, 5, 6, 2, 0, 0], [1, 7, 0, 0, 0, 0]]
    assert labels == [[-100, -100, 6, 2, -100, -100], [-100, 7, -100, -100, -100, -100]]
    with pytest.raises(ValueError, match="longer"):
        data.window_arrays([{"input_ids": [1] * 7, "labels": [1] * 7}], seq=6, pad_id=0)
