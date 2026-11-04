"""The L2L preference docs match the code and the A6 reports (A6 ticket 07)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.assemble_changelog import validate_fragments
from soup_cli.config.schema import L2L_SUPPORTED_TASKS, TrainingConfig

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "performance-and-quantization.md"
HEADING = "### L2L for preference tasks"
PREFERENCE_METHODS = ("dpo", "orpo", "simpo", "kto")


def _section() -> str:
    text = DOC.read_text(encoding="utf-8")
    assert HEADING in text, f"{DOC.name} has no {HEADING!r} section"
    start = text.index(HEADING)
    following = re.search(r"^### ", text[start + len(HEADING):], re.MULTILINE)
    end = start + len(HEADING) + following.start() if following else len(text)
    return text[start:end]


def _table_rows(section: str) -> dict:
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells and cells[0].lower().strip("`") in PREFERENCE_METHODS:
            rows[cells[0].lower().strip("`")] = cells
    return rows


def test_table_has_one_row_for_every_preference_task_the_schema_accepts():
    accepted = {task for task in L2L_SUPPORTED_TASKS if task != "sft"}
    assert accepted == set(PREFERENCE_METHODS)
    assert set(_table_rows(_section())) == accepted


def test_only_dpo_is_marked_ready():
    rows = _table_rows(_section())
    assert rows["dpo"][1].lower() == "ready"
    for method in ("orpo", "simpo", "kto"):
        assert rows[method][1].lower() == "not ready", method


@pytest.mark.parametrize("method", PREFERENCE_METHODS)
def test_each_row_names_model_card_and_links_an_existing_report(method):
    row = _table_rows(_section())[method]
    joined = " ".join(row)
    assert "SmolLM2-135M" in joined
    assert "RTX 3060" in joined
    link = re.search(r"\]\((\.\./benchmarks/[^)]+)\)", joined)
    assert link, f"{method} row has no benchmarks link"
    assert (DOC.parent / link.group(1)).resolve().is_file()
    assert link.group(1).endswith(f"probe-vast-l2l-{method}-a6-ready.md")


@pytest.mark.parametrize(
    "refused",
    [
        "ref_model",
        "precompute_ref_log_probs",
        "loss_type",
        "dpo_beta_schedule",
        "dpo_beta_end",
        "dpo_ref_regen_epochs",
        "ref_adapter_name",
        "neftune_alpha",
    ],
)
def test_section_names_each_refused_mode(refused):
    assert refused in _section()


def test_section_names_the_ref_adapter_dpo_compares_against():
    assert "`ref` adapter" in _section()


def test_section_states_the_limits():
    section = _section()
    for phrase in (
        "SmolLM2-135M",
        "RTX 3060",
        "trl 0.29.0",
        "transformers 5.16.1",
        "trl 0.29.1",
        "synthetic",
        "float32",
        "MATH",
        "Not measured",
    ):
        assert phrase in section, phrase


def test_section_links_the_four_reports_and_the_rule():
    section = _section()
    for method in PREFERENCE_METHODS:
        assert f"probe-vast-l2l-{method}-a6-ready.md" in section
    assert "l2l-preference-readiness-rule-a6.md" in section


def test_every_benchmarks_link_in_the_section_exists():
    for target in re.findall(r"\]\((\.\./benchmarks/[^)#]+)", _section()):
        assert (DOC.parent / target).resolve().exists(), target


def test_the_sft_limits_no_longer_say_sft_only():
    text = DOC.read_text(encoding="utf-8")
    assert "- `task` other than `sft`." not in text


def test_l2l_field_description_does_not_claim_sft_only():
    description = TrainingConfig.model_fields["l2l"].description
    assert "task='sft'" not in description
    for method in PREFERENCE_METHODS:
        assert method in description


def test_changelog_fragments_are_valid_and_one_names_l2l_preference():
    fragments = validate_fragments(ROOT)
    added = [
        fragment
        for fragment in fragments
        if fragment.section == "added" and "L2L" in fragment.content
        and "preference" in fragment.content.lower()
    ]
    assert added, "no 'added' fragment describes L2L for preference tasks"
