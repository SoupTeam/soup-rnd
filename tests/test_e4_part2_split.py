"""E4 part 2, calibration rule section 2 and 2a: split determinism, identity, label masks."""
# ruff: noqa: I001 (the support import must run before the harness imports)

import hashlib
import json
import subprocess
import unicodedata
import zlib

from tests.e4_part2_support import REPO_ROOT  # noqa: F401 (puts the harness on sys.path)
import e4_part2_split as split  # noqa: E402
import pytest



class FakeTokenizer:
    """One token per whitespace-separated word, newlines their own token. Deterministic."""

    eos_token_id = 7
    pad_token_id = 1

    def __call__(self, text, add_special_tokens=False, **_):
        ids = []
        for line_no, line in enumerate(text.split("\n")):
            if line_no:
                ids.append(10)
            ids.extend(100 + zlib.crc32(w.encode()) % 900 for w in line.split())
        return type("Enc", (), {"input_ids": ids})()


TOK = FakeTokenizer()


def row(i, instruction=None, context="", response=None):
    return {"instruction": instruction or f"question number {i}", "context": context,
            "response": response or f"answer number {i}", "category": "x"}


def rows(n, start=0):
    return [row(i) for i in range(start, start + n)]


class TestIdentityEncoding:
    def test_known_answer(self):
        blob = '["Zoë asks","","a b"]'.encode("utf-8")  # compact, non-ASCII raw, one space runs
        assert split.identity("Zoë  asks", "", "a\n b") == hashlib.sha256(blob).hexdigest()

    def test_ensure_ascii_is_off_so_the_bytes_are_utf8_not_escapes(self):
        escaped = hashlib.sha256('["Zo\\u00eb","",""]'.encode()).hexdigest()
        assert split.identity("Zoë", "", "") != escaped

    def test_internal_whitespace_runs_collapse_only_in_the_key(self):
        a = split.identity("a  b\n\nc", "", "r")
        b = split.identity("a b c", "", "r")
        assert a == b
        assert split.normalize_field("  a  b\n\nc \n") == "a  b\n\nc"  # model text keeps them

    def test_nfc_makes_composed_and_decomposed_forms_the_same_text(self):
        composed, decomposed = "café", unicodedata.normalize("NFD", "café")
        assert composed != decomposed
        assert split.normalize_field(decomposed) == composed

    def test_field_boundaries_are_part_of_the_encoding(self):
        assert split.identity("ab", "", "c") != split.identity("a", "", "bc")
        assert split.identity("a", "b", "c") != split.identity("a", "", "bc")


class TestPromptsAndLabels:
    def test_wrapper_with_context(self):
        assert split.build_prompt("Q", "C") == (
            "### Instruction:\nQ\n\n### Context:\nC\n\n### Response:\n")

    def test_wrapper_without_context_is_the_gate_suite_wrapper(self):
        assert split.build_prompt("Q", "") == "### Instruction:\nQ\n\n### Response:\n"
        assert split.gate_wrapper("Q") == split.build_prompt("Q", "")

    @pytest.mark.parametrize("context", ["", "some context"])
    def test_prompt_is_masked_and_response_plus_eos_are_targets(self, context):
        enc = split.encode_example(TOK, "what is it", context, "it is a thing")
        n_prompt = len(enc["prompt_ids"])
        assert enc["labels"][:n_prompt] == [split.IGNORE_INDEX] * n_prompt
        assert enc["labels"][n_prompt:] == enc["response_ids"]
        assert enc["response_ids"][-1] == TOK.eos_token_id
        assert enc["input_ids"] == enc["prompt_ids"] + enc["response_ids"]
        assert len(enc["labels"]) == len(enc["input_ids"])

    def test_response_gets_no_added_leading_space(self):
        enc = split.encode_example(TOK, "q", "", "word")
        assert enc["response_ids"][:-1] == TOK("word").input_ids

    def test_padding_carries_label_minus_100_and_attention_zero(self):
        a = split.encode_example(TOK, "short", "", "x")
        b = split.encode_example(TOK, "a much longer question here", "ctx", "and a long answer")
        batch = split.pad_batch([a, b], pad_id=TOK.pad_token_id)
        width = len(b["input_ids"])
        assert [len(r) for r in batch["input_ids"]] == [width, width]
        fill = width - len(a["input_ids"])
        assert fill > 0
        assert batch["labels"][0][-fill:] == [split.IGNORE_INDEX] * fill
        assert batch["attention_mask"][0][-fill:] == [0] * fill
        assert batch["input_ids"][0][-fill:] == [TOK.pad_token_id] * fill

    def test_wrapper_record_has_text_hashes_ids_and_both_worked_examples(self):
        rec = split.wrapper_record(
            TOK, {"instruction": "i", "context": "c", "response": "r"},
            {"instruction": "i2", "context": "", "response": "r2"})
        assert rec["wrapper_no_context_sha256"] == hashlib.sha256(
            split.WRAPPER_NO_CONTEXT.encode()).hexdigest()
        assert "### Context:" in rec["worked_example_with_context"]["prompt_text"]
        assert "### Context:" not in rec["worked_example_without_context"]["prompt_text"]
        for key in ("worked_example_with_context", "worked_example_without_context"):
            ex = rec[key]
            assert len(ex["input_ids"]) == len(ex["labels"])
            assert ex["labels"][:ex["n_prompt"]] == [split.IGNORE_INDEX] * ex["n_prompt"]


@pytest.fixture(scope="module")
def result():
    return split.build_split(rows(7000), TOK)


class TestSplit:
    def test_sizes_and_disjointness(self, result):
        sizes = {k: len(v) for k, v in result.manifests.items()}
        assert sizes == {"DECISION": 2048, "SMOKE": 256, "TRAIN": 4096}
        ids = [r["identity"] for v in result.manifests.values() for r in v]
        assert len(ids) == len(set(ids))

    def test_sorted_by_identity_and_split_by_position(self, result):
        everything = result.eligible
        assert [r["identity"] for r in everything] == sorted(r["identity"] for r in everything)
        assert result.manifests["DECISION"] == everything[0:2048]
        assert result.manifests["SMOKE"] == everything[2048:2304]
        assert result.manifests["TRAIN"] == everything[2304:6400]

    def test_deterministic_bytes(self, result):
        again = split.build_split(rows(7000), TOK)
        for name in result.manifests:
            assert split.manifest_lines(result.manifests[name]) == split.manifest_lines(
                again.manifests[name])

    def test_dataset_order_does_not_change_which_identities_land_where(self, result):
        shuffled = rows(7000)
        shuffled.reverse()
        other = split.build_split(shuffled, TOK)
        for name in result.manifests:
            assert {r["identity"] for r in result.manifests[name]} == {
                r["identity"] for r in other.manifests[name]}

    def test_unused_rows_remain(self, result):
        assert result.funnel["eligible"] == 7000

    def test_manifest_files_are_lf_with_fixed_keys_and_hashed(self, result, tmp_path):
        written = split.write_manifests(result, tmp_path)
        raw = (tmp_path / "smoke.manifest.jsonl").read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
        first = json.loads(raw.splitlines()[0])
        assert list(first) == ["row_id", "identity", "prompt_tokens", "response_tokens"]
        assert written["SMOKE"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert written["SMOKE"]["rows"] == 256


class TestFilters:
    def test_too_few_rows_stops(self):
        with pytest.raises(split.InsufficientRowsError, match="6400"):
            split.build_split(rows(6399), TOK)

    def _eligible_plus(self, extra):
        return split.build_split(rows(6400) + extra, TOK)

    def test_empty_instruction_or_response_is_excluded(self):
        res = self._eligible_plus([row(9001, instruction="   "), row(9002, response="\n"),
                                   row(9003, context="")])
        assert res.funnel["empty_instruction_or_response"] == 2
        assert res.funnel["eligible"] == 6401

    def test_empty_context_is_allowed(self):
        res = self._eligible_plus([row(9001, context="  ")])
        assert res.funnel["eligible"] == 6401

    def test_dedup_keeps_the_first_in_dataset_order_across_whitespace_variants(self):
        a = row(9001, instruction="same  question", response="same answer")
        b = row(9002, instruction="same question", response="same\nanswer")
        res = self._eligible_plus([a, b])
        assert res.funnel["duplicate"] == 1
        kept = [r for r in res.eligible if r["row_id"] in (6400, 6401)]
        assert [r["row_id"] for r in kept] == [6400]
        assert kept[0]["response"] == "same answer"

    def test_prompt_over_384_tokens_is_excluded_exactly_at_the_boundary(self):
        base = len(split.encode_example(TOK, "w", "", "x")["prompt_ids"])
        at = row(9001, instruction=" ".join(["w"] * (384 - base + 1)), response="ok")
        over = row(9002, instruction=" ".join(["w"] * (384 - base + 2)), response="ok")
        assert len(split.encode_example(TOK, at["instruction"], "", "ok")["prompt_ids"]) == 384
        res = self._eligible_plus([at, over])
        assert res.funnel["prompt_over_384"] == 1 and res.funnel["eligible"] == 6401

    def test_response_plus_eos_over_128_is_excluded_exactly_at_the_boundary(self):
        at = row(9001, response=" ".join(["w"] * 127))  # 127 + EOS = 128
        over = row(9002, response=" ".join(["w"] * 128))  # 128 + EOS = 129
        res = self._eligible_plus([at, over])
        assert res.funnel["response_over_128"] == 1 and res.funnel["eligible"] == 6401

    def test_nothing_is_truncated(self):
        res = self._eligible_plus([row(9001, response=" ".join(["w"] * 127))])
        kept = next(r for r in res.eligible if r["row_id"] == 6400)
        assert kept["response_tokens"] == 128


class TestRuleFiles:
    @pytest.mark.parametrize("name", ["e4-part2-calibration-rule.md", "e4-rule-part2-draft.md",
                                      "e4-rule-part2-decision-skeleton.md"])
    def test_gitattributes_pins_lf_for_both_rule_files(self, name):
        attrs = (REPO_ROOT / ".gitattributes").read_text().splitlines()
        assert f"{name} text eol=lf" in attrs

    def test_harness_and_rule_files_are_checked_with_git_if_present(self):
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "check-attr", "eol", "--",
                              "e4-part2-calibration-rule.md", "e4-rule-part2-draft.md",
                              "e4-rule-part2-decision-skeleton.md"],
                             capture_output=True, text=True, check=False)
        if out.returncode == 0:
            assert out.stdout.count("eol: lf") == 3
