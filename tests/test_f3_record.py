"""The F3 record (benchmarks/gate-f3-benchmark-audit.md) keeps the rule it was committed with
and carries what the ticket 09 acceptance asks for. Text checks only: no network, no GPU.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks" / "gate-f3-benchmark-audit.md"

# sha256 of the text from "## Decision rule" up to "## Out of scope" as committed in 0e17ab6d,
# before the first arm. The rule is not edited afterwards.
RULE_SHA256 = "dd18b783a77b5e4dc18090e44061bc2f6832c83f0422cc26a89b92ee5d8960c2"


@pytest.fixture(scope="module")
def text() -> str:
    return RECORD.read_text(encoding="utf-8")


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    nxt = re.search(r"^## ", text[start + len(heading):], re.MULTILINE)
    end = start + len(heading) + nxt.start() if nxt else len(text)
    return text[start:end]


def test_the_rule_is_byte_identical_to_the_committed_one(text):
    rule = text[text.index("## Decision rule"):text.index("## Out of scope")]
    assert hashlib.sha256(rule.encode()).hexdigest() == RULE_SHA256


def test_the_record_is_no_longer_a_skeleton(text):
    assert "This is the skeleton" not in text
    assert "## Results" in text
    assert "## What was not done and why" in text


def test_conditions_and_refusal_tables_are_filled(text):
    for heading in ("## Conditions table", "## Refusal table"):
        rows = [
            ln for ln in section(text, heading).splitlines()
            if ln.startswith("|") and not ln.startswith("|---")
        ]
        body = rows[1:]
        assert body, heading
        assert all(any(c.strip() for c in r.strip("|").split("|")) for r in body), heading


def test_not_done_section_names_every_cut(text):
    body = section(text, "## What was not done and why")
    for needle in (
        "Chronicals",
        "C2",
        "speed blocks",
        "2026.10.6",
        "destroy",
        "control was not rerun on the A100",
    ):
        assert needle in body, needle


def test_46000_line_says_not_audited_and_not_checked(text):
    assert "46,000" in text
    assert "config not found, not audited" in text
    assert "not checked" in section(text, "## Results")


def test_the_word_invalid_always_carries_its_step(text):
    rule_start = text.index("## Decision rule")
    rule_end = text.index("## Out of scope")
    words_start = text.index("## Words")
    words_end = text.index("## Arms and conditions")
    outside = text[:words_start] + text[words_end:rule_start] + text[rule_end:]
    for line in outside.splitlines():
        for hit in re.finditer(r"\binvalid\b", line, re.IGNORECASE):
            around = line[max(0, hit.start() - 14):hit.end() + 14].lower()
            qualified = "our run invalid" in around or "published run invalid" in around
            assert qualified, line


def test_published_numbers_have_their_own_column_and_card_label(text):
    assert "A100-40GB, not measured by me" in text
    results = section(text, "## Results")
    header = next(ln for ln in results.splitlines() if "Published" in ln and ln.startswith("|"))
    assert "A100-40GB, not measured by me" in header
    assert "A100-PCIE-80GB vast.ai, real" in header
    assert "RTX 3050 Laptop, control" in text


def test_every_cited_result_file_exists(text):
    paths = re.findall(r"benchmarks/results/f3-benchmark-audit/[\w./-]+", text)
    assert paths
    for path in set(p.rstrip(".,)") for p in paths):
        assert (ROOT / path).exists(), path


def test_both_prepared_outcome_statements_name_the_framework(text):
    body = section(text, "## Prepared statements")
    assert "Chronicals" in body
    assert "confirmed" in body.lower() and "not confirmed" in body.lower()
    assert "not peer-reviewed" in body
    assert "inconsistent" in body


def test_unresolved_items_and_pointer_cards(text):
    assert "cc-by-nc-4.0" in text
    assert "2-5x faster" in text
    assert "private" in text and "m1kado/f3-audit-results" in text
