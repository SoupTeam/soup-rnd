
"""Inspect E2 training tokens using Soup's real SFT formatter."""

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from soup_cli.config.loader import load_config
from soup_cli.data.sft_format import build_format_row

CONFIG = "experiments/e2_long_cached.yaml"


def main():
    cfg = load_config(CONFIG)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base)

    format_row = build_format_row(
        tokenizer=tokenizer,
        data_cfg=cfg.data,
        training_cfg=cfg.training,
    )

    rows = [
        json.loads(line)
        for line in Path(cfg.data.train).read_text().splitlines()
        if line.strip()
    ]

    sequences = []

    for index, row in enumerate(rows, start=1):
        messages = []

        instruction = row["instruction"]
        extra_input = row.get("input", "")

        if extra_input:
            instruction = f"{instruction}\n{extra_input}"

        messages.append({
            "role": "user",
            "content": instruction,
        })

        messages.append({
            "role": "assistant",
            "content": row["output"],
        })

        formatted = format_row({"messages": messages})
        if "input_ids" in formatted:
            ids = formatted["input_ids"]
            mask = formatted.get("attention_mask")
            labels = formatted.get("labels")

            assert len(ids) <= cfg.data.max_length

            if mask is not None:
                assert len(mask) == len(ids)
                assert all(value in (0, 1) for value in mask)

            if labels is not None:
                assert len(labels) == len(ids)

        if "input_ids" in formatted:
            ids = formatted["input_ids"]
        elif "text" in formatted:
            ids = tokenizer(
                formatted["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=cfg.data.max_length,
            )["input_ids"]
        else:
            raise RuntimeError(
                f"Unexpected formatter output: {formatted.keys()}"
            )

        sequences.append(tuple(ids))

        unk_count = ids.count(tokenizer.unk_token_id)

        print(
            f"Example {index}: "
            f"length={len(ids)}, "
            f"UNK={unk_count}"
        )

    lengths = [len(ids) for ids in sequences]

    print("\n=== SUMMARY ===")
    print("Examples:", len(sequences))
    print("Unique sequences:", len(set(sequences)))
    print("Minimum length:", min(lengths))
    print("Maximum length:", max(lengths))
    print("Length distribution:", dict(Counter(lengths)))
    print("Configured max_length:", cfg.data.max_length)

    assert len(sequences) == 8
    assert len(set(sequences)) == 8
    assert max(lengths) <= cfg.data.max_length

    print("\nE2 token inspection: PASSED")


if __name__ == "__main__":
    main()
