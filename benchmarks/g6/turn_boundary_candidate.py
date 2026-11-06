"""Inactive candidate semantics for a future generation probe, not a v1 scorer patch.

Apply only to generated completion IDs, never to prompt IDs. A line containing a
literal role marker, even inside code or a quotation, is a boundary by this rule.
This ambiguity is retained as a limitation, not resolved using expected answers.
"""

from __future__ import annotations

MARKERS = ("\nuser:", "\nassistant:", "\nsystem:")


def boundary(text: str) -> dict | None:
    """Return the earliest complete, case-sensitive literal newline role marker."""
    found = [(text.find(marker), marker) for marker in MARKERS if marker in text]
    if not found:
        return None
    offset, marker = min(found)
    return {"character_offset": offset, "marker": marker}


def inspect_prefix(completion_ids: list[int], tokenizer: object) -> dict:
    """Inspect one current prefix; suitable for future token-by-token stopping.

    Keep raw token IDs/text, including a boundary token's overhang. Only the
    candidate's scored text excludes the boundary. Never decode only the last
    token: markers can span tokens and decoding can revise a prior suffix.
    """
    text = tokenizer.decode(
        completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    match = boundary(text)
    return {
        "raw_completion_ids": list(completion_ids),
        "raw_completion": text,
        "stop": match is not None,
        "boundary": match,
        "candidate_scored_text": text[:match["character_offset"]] if match else text,
        "policy": "DRAFT_LITERAL_NEWLINE_ROLE_BOUNDARY_V1",
        "official_v1_result": False,
    }
