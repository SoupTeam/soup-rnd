"""Synthetic boundary-policy tests; no live generation or model acceptance."""

import importlib.util
from pathlib import Path

import pytest

CODE = Path(__file__).parents[1] / "benchmarks/g6/turn_boundary_candidate.py"
SPEC = importlib.util.spec_from_file_location("g6_boundary_candidate_tests", CODE)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


class ChunkTokenizer:
    def __init__(self, chunks):
        self.chunks = chunks

    def decode(self, ids, **options):
        assert options == {
            "skip_special_tokens": True, "clean_up_tokenization_spaces": False,
        }
        return "".join(self.chunks[index] for index in ids)


@pytest.mark.parametrize("marker", candidate.MARKERS)
def test_marker_split_across_tokens_stops_only_when_complete(marker):
    tokenizer = ChunkTokenizer(["426", *marker])
    for length in range(1, len(tokenizer.chunks)):
        assert not candidate.inspect_prefix(list(range(length)), tokenizer)["stop"]
    ids = list(range(len(tokenizer.chunks)))
    result = candidate.inspect_prefix(ids, tokenizer)
    assert result["stop"] and result["boundary"]["marker"] == marker
    assert result["candidate_scored_text"] == "426"
    assert result["raw_completion_ids"] == ids
    assert result["raw_completion"] == "426" + marker
    assert not result["official_v1_result"]


@pytest.mark.parametrize("text", [
    "user: is a word", "the user: replied", "42\nUser:", "42\nuser", "42\n user:", "",
])
def test_nonmarkers_do_not_stop(text):
    assert candidate.boundary(text) is None


def test_first_marker_wins_and_token_overhang_is_retained():
    tokenizer = ChunkTokenizer(["Sure: 426.\nuser:next\nassistant:other"])
    result = candidate.inspect_prefix([0], tokenizer)
    assert result["candidate_scored_text"] == "Sure: 426."
    assert result["raw_completion"].endswith("other")
    assert result["boundary"] == {"character_offset": 10, "marker": "\nuser:"}


def test_literal_marker_inside_code_is_documented_boundary_not_semantic_parsing():
    assert candidate.boundary("```text\nuser: quoted\n```")["marker"] == "\nuser:"


def test_full_prefix_redecode_can_revise_prior_suffix():
    class RevisingTokenizer:
        def decode(self, ids, **options):
            return "42\nus" if len(ids) == 1 else "42\nuser:"

    tokenizer = RevisingTokenizer()
    assert not candidate.inspect_prefix([1], tokenizer)["stop"]
    assert candidate.inspect_prefix([1, 2], tokenizer)["candidate_scored_text"] == "42"
