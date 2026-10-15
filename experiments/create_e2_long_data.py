
"""Generate reproducible long-sequence E2 benchmark data."""

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "experiments" / "medium_mistral"
OUTPUT = ROOT / "experiments" / "e2_long_data.jsonl"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

questions = [
    "What is the capital of France?",
    "What color is the sky?",
    "What is 2 + 2?",
    "What is 3 + 5?",
]

targets = [128, 256]
rows = []

for target in targets:
    for index, question in enumerate(questions):
        # Generate long examples from known vocabulary.
        words = ["France", "Paris", "sky", "Blue", "What", "is"]

        instruction = question
        while len(tokenizer(instruction, add_special_tokens=False)["input_ids"]) < target - 20:
            instruction += " " + words[index % len(words)]

        rows.append({
            "instruction": instruction,
            "input": "",
            "output": f"Answer {index + 1}",
        })

with OUTPUT.open("w", encoding="utf-8") as file:
    for row in rows:
        file.write(json.dumps(row) + "\n")

print(f"Generated {len(rows)} examples: {OUTPUT}")

for row in rows:
    length = len(
        tokenizer(
            row["instruction"],
            add_special_tokens=False,
        )["input_ids"]
    )
    print(f"Instruction length: {length}")

