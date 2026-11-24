"""Deterministic H2-close performance workload (stage P), generation and audit.

Why a new workload: the frozen 128-row correctness fixture holds 24-token arithmetic rows, which
say nothing about training speed at realistic lengths (RUNBOOK "Workload scope"). This module
writes a length-representative SYNTHETIC SFT set in Alpaca format: plain English-like sentences
from a fixed word list, so the token stream is ordinary text for the Llama-3.1 tokenizer. It is a
systems workload only. No quality, convergence or data-domain claim follows from it.

    python benchmarks/harness/h2_perf_workload.py generate   # writes h2/perf_workload.jsonl
    python benchmarks/harness/h2_perf_workload.py check      # byte-compares a fresh generation

Lengths are set in words, not tokens, so generation needs no tokenizer. The actual token counts
come from the real tokenizer and the real Soup formatter and are frozen separately in
benchmarks/h2/perf_workload_receipt.json (see H2_CLOSE_PROTOCOL.txt, section P1).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

SEED = 20261102
ROWS = 96
# Words per row, drawn uniformly. With ~1.05 tokens per word plus the llama3 template this
# lands the formatted rows near 400-960 tokens; the receipt checks the real bounds.
WORDS_MIN, WORDS_MAX = 340, 880
# Share of a row's words in the instruction; the rest is the supervised response.
PROMPT_SHARE = 0.25
OUT = Path(__file__).resolve().parents[1] / "h2" / "perf_workload.jsonl"

WORDS = (
    "the a an this that these those every each some many few several other another same "
    "system model team report result value process change number table record file "
    "method plan step stage window batch layer weight update memory speed device engine "
    "network signal sensor river city market garden library museum station harbor bridge "
    "valley forest mountain island village street school office factory kitchen window "
    "morning evening winter summer spring autumn week month year hour minute moment period "
    "question answer reason example detail summary outline section chapter paragraph line "
    "teacher student doctor driver farmer writer reader worker manager engineer visitor "
    "small large early late quick slow simple careful clear quiet bright heavy light warm "
    "cold fresh old new short long wide narrow strong gentle steady useful common rare "
    "first second third final main local public private central regular normal special "
    "make take give keep find show build move write read check measure compare record "
    "describe explain prepare follow choose change reduce increase open close start finish "
    "carry collect deliver repair visit study learn teach count store load save share "
    "and or but so because while after before when where although unless until since "
    "in on at by for with from into over under between through during without within "
    "is are was were has have had will would can could should may might must does did "
    "very also often rarely always never usually still already almost nearly mostly "
    "water stone paper glass metal wood light sound color shape size price cost level "
    "group pair list set order range limit scale share part whole piece point edge "
    "travel journey path road route map guide note letter message story problem idea"
).split()
OPENERS = (
    "Write a clear explanation of",
    "Summarize the main points about",
    "Describe step by step how to handle",
    "Give a careful account of",
    "Explain to a new colleague the idea of",
    "Prepare a short report on",
)


def _sentence(rng: random.Random, words: int) -> str:
    text = " ".join(rng.choice(WORDS) for _ in range(words))
    return text[0].upper() + text[1:] + "."


def _paragraphs(rng: random.Random, words: int) -> str:
    """About ``words`` words as sentences of 8-20 words in paragraphs of 3-6 sentences."""
    sentences, left = [], words
    while left > 0:
        n = min(left, rng.randint(8, 20))
        sentences.append(_sentence(rng, n))
        left -= n
    out, i = [], 0
    while i < len(sentences):
        k = rng.randint(3, 6)
        out.append(" ".join(sentences[i : i + k]))
        i += k
    return "\n\n".join(out)


def generate_rows(seed: int = SEED, rows: int = ROWS) -> list[dict]:
    rng = random.Random(seed)
    result = []
    for index in range(rows):
        total = rng.randint(WORDS_MIN, WORDS_MAX)
        prompt_words = max(20, round(total * PROMPT_SHARE))
        topic = " ".join(rng.choice(WORDS) for _ in range(3))
        instruction = f"{rng.choice(OPENERS)} the {topic}. Use the notes below.\n\n" + _paragraphs(
            rng, prompt_words
        )
        output = _paragraphs(rng, total - prompt_words)
        result.append({"instruction": instruction, "input": "", "output": output, "row": index})
    return result


def render(rows: list[dict]) -> bytes:
    lines = [
        json.dumps({k: r[k] for k in ("instruction", "input", "output")}, ensure_ascii=True)
        for r in rows
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "check"
    data = render(generate_rows())
    digest = hashlib.sha256(data).hexdigest()
    if command == "generate":
        OUT.write_bytes(data)
        print(json.dumps({"written": str(OUT), "rows": ROWS, "sha256": digest}))
        return 0
    if command == "check":
        same = OUT.exists() and OUT.read_bytes() == data
        print(json.dumps({"rows": ROWS, "sha256": digest, "identical_to_frozen": same}))
        return 0 if same else 1
    print("usage: h2_perf_workload.py generate|check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
