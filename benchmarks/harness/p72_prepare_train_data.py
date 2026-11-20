"""Write the training windows and held-out groups of the adapter-equality run (harness only).

    python p72_prepare_train_data.py --model /workspace/p3-baseline/source --output train_data.pt

160 short MetaMathQA examples (full prompt, answer and end token of at most 128 tokens; with the
Alpaca-style prompt only 5% of the dataset is that short) become 20
windows of 8; a run of two epochs makes 40 optimizer steps. 128 further groups (64 GSM, 64 MATH, at
most 256 tokens, other original questions) are the held-out set. The file has the schema of
``p3_prepare_fresh_metamath_eval.py``: ``train_steps``, ``heldout_examples``, ``manifest``. Needs
``datasets`` and network access to the pinned MetaMathQA revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from p72_train_data import DATASET, REVISION, make_windows, select_records
from rich.console import Console

console = Console()


def main() -> None:
    import torch
    from datasets import load_dataset

    from soup_cli.utils.kimi_k2 import load_stream_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=160)
    parser.add_argument("--train-max-length", type=int, default=128)
    parser.add_argument("--per-window", type=int, default=8)
    parser.add_argument("--eval-per-family", type=int, default=64)
    parser.add_argument("--eval-max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    tokenizer = load_stream_tokenizer(str(args.model), trust_remote_code=False)
    console.out(f"Loading {DATASET} at {REVISION}")
    rows = load_dataset(DATASET, split="train", revision=REVISION)
    wanted = {"original_question", "query", "response", "type"}
    if not wanted.issubset(rows.column_names):
        raise RuntimeError("dataset schema is missing required fields")
    chosen = select_records(
        rows, tokenizer, train_count=args.train_count, train_max_length=args.train_max_length,
        eval_per_family=args.eval_per_family, eval_max_length=args.eval_max_length, seed=args.seed,
    )
    windows = make_windows(chosen["train"], args.per_window)
    manifest = {
        "dataset": DATASET, "dataset_revision": REVISION, "dataset_fingerprint": rows._fingerprint,
        "seed": args.seed, "train_count": len(chosen["train"]), "windows": len(windows),
        "per_window": args.per_window, "train_max_length": args.train_max_length,
        "heldout_count": len(chosen["heldout"]), "eval_max_length": args.eval_max_length,
        "scope": "own selection of short MetaMathQA examples; not the 160 examples of PR #72",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train_steps": windows, "heldout_examples": chosen["heldout"],
                "manifest": manifest}, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(dict(manifest, data_sha256=digest), indent=2) + "\n", encoding="utf-8")
    console.out("PREPARED " + json.dumps(manifest, indent=2))
    console.out(f"data_sha256 {digest}")


if __name__ == "__main__":
    main()
