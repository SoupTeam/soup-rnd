"""E1 confirmation on questions E1 never used (probe-moe-adapter-placement.md section 11).

Subcommands:
  prepare-c  choose the confirmation set C: 256 rows, one per MetaMathQA source question,
             walking E1's shuffle on from where E1's split stopped and skipping every source
             question an E1 split used, also after normalising it and as a near-duplicate
             (word-set Jaccard >= 0.8)
  run        the measured E1 harness (harness-348f6a48.py.txt, bytes unchanged) trains and
             scores one arm as E1 did; after its leg 2, on the same adapter, this adds NLL
             and accuracy on C and per-row NLL on E1's evaluation set E and on C

The split walk, row filters, encoding, NLL, generation and scoring are the measured
harness's own functions; this file only chooses C and calls them on it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import pathlib
import random
import re
import sys
import time
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

HERE = pathlib.Path(__file__).resolve().parent
MEASURED_HARNESS = (HERE.parent / "results" / "probe-moe-adapter-placement"
                    / "harness-348f6a48.py.txt")
MEASURED_HARNESS_SHA256 = "dd55825c67ccc61d532af150fa798d02e07fe72c97b9e5859669f02c2ecf87b3"
N_CONFIRM = 256
NEAR_DUPLICATE_JACCARD = 0.8
ROLES = ("calibration", "train", "eval")
_WORD = re.compile(r"[a-z0-9]+")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_measured_harness(path: str = str(MEASURED_HARNESS)):
    """Import the measured E1 harness from its published bytes, refusing any other bytes."""
    actual = file_sha256(path)
    if actual != MEASURED_HARNESS_SHA256:
        raise ValueError(f"{path} has SHA-256 {actual}, not the measured {MEASURED_HARNESS_SHA256}")
    name = "_e1_measured_harness"
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def words(text: str) -> List[str]:
    """Lower-cased ASCII letter/digit runs after NFKC: case and punctuation drop out."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).lower())


class UsedQuestions:
    """Source questions E1 used, in the three forms a C row must not match."""

    def __init__(self, keys: Iterable[str]):
        self.keys = set(keys)
        self.normalised = {" ".join(words(key)) for key in self.keys}
        self.word_sets = [frozenset(words(key)) for key in sorted(self.keys)]
        self._index: Dict[str, List[int]] = {}
        self._reasons: Dict[str, Optional[str]] = {}
        for position, word_set in enumerate(self.word_sets):
            for word in word_set:
                self._index.setdefault(word, []).append(position)

    def max_jaccard(self, key: str) -> float:
        mine = frozenset(words(key))
        shared: Dict[int, int] = {}
        for word in mine:
            for position in self._index.get(word, ()):
                shared[position] = shared.get(position, 0) + 1
        return max((n / (len(mine) + len(self.word_sets[p]) - n) for p, n in shared.items()),
                   default=0.0)

    def reason(self, key: str) -> Optional[str]:
        if key not in self._reasons:
            if key in self.keys:
                found = "used_source_question"
            elif " ".join(words(key)) in self.normalised:
                found = "used_after_normalising"
            elif self.max_jaccard(key) >= NEAR_DUPLICATE_JACCARD:
                found = "near_duplicate"
            else:
                found = None
            self._reasons[key] = found
        return self._reasons[key]


def used_source_questions(harness, rows: Sequence[Dict[str, Any]],
                          splits: Sequence[Dict[str, Any]]) -> set:
    return {harness.group_key(rows[i]) for split in splits for role in ROLES for i in split[role]}


def select_confirmation(harness, rows: Sequence[Dict[str, Any]], split: Dict[str, Any],
                        exclude: Sequence[Dict[str, Any]],
                        e1_reject: Callable[[Dict[str, Any]], Optional[str]],
                        wanted: int = N_CONFIRM) -> Dict[str, Any]:
    """C: E1's shuffle walked on from ``split["examined"]``, one row per unused question.

    ``e1_reject`` is E1's own row filter (reference answer, template prefix and length
    for both tokenizers); it runs only on rows whose source question passed the
    used-question checks against ``split`` and every split in ``exclude``.
    """
    order = list(range(len(rows)))
    random.Random(harness.SPLIT_SEED).shuffle(order)
    start = split["examined"]
    position = {index: p for p, index in enumerate(order)}
    if max(position[i] for role in ROLES for i in split[role]) >= start:
        raise ValueError("the split holds a row past its own 'examined' count")
    used = UsedQuestions(used_source_questions(harness, rows, [split, *exclude]))

    def reject(row: Dict[str, Any]) -> Optional[str]:
        return used.reason(harness.group_key(row)) or e1_reject(row)

    accepted, dropped = harness.select_rows(rows, order[start:], wanted, reject)
    chosen = {harness.group_key(rows[i]) for i in accepted}
    if chosen & used.keys or len(chosen) != len(accepted):
        raise ValueError("confirmation rows reuse a source question")
    return {"walk_start": start, "examined": len(accepted) + sum(dropped.values()),
            "dropped": dropped, "excluded_source_questions": len(used.keys), "eval": accepted}


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    harness = load_measured_harness(args.harness)
    rows = harness.load_rows(args.data)
    split = json.loads(pathlib.Path(args.split).read_text(encoding="utf-8"))
    harness.check_group_disjoint(rows, split)
    exclude = [json.loads(pathlib.Path(p).read_text(encoding="utf-8")) for p in args.exclude]
    tokenizers = {key: harness.load_tokenizer(key) for key in harness.MODELS}

    def e1_reject(row: Dict[str, Any]) -> Optional[str]:
        if harness.reference_answer(row["response"]) is None:
            return "no_reference"
        for key, tokenizer in tokenizers.items():
            encoded = harness.encode_row(tokenizer, harness.MODELS[key]["chat"], row["query"],
                                         row["response"])
            if encoded is None:
                return "not_prefix"
            if len(encoded[1]) > harness.MAX_TOKENS:
                return "too_long"
        return None

    chosen = select_confirmation(harness, rows, split, exclude, e1_reject)
    types: Dict[str, Dict[str, int]] = {}
    for name, indices in (("E", split["eval"]), ("C", chosen["eval"])):
        for index in indices:
            kind = rows[index]["type"]
            types.setdefault(name, {})[kind] = types.setdefault(name, {}).get(kind, 0) + 1
    return {
        "set": "C, E1 confirmation (probe-moe-adapter-placement.md section 11)",
        "data_file": harness.DATA_FILE, "data_sha256": harness.DATA_SHA256,
        "split_seed": harness.SPLIT_SEED, "max_tokens": harness.MAX_TOKENS,
        "grouping": "one row per original_question; none used by any E1 split",
        "near_duplicate_jaccard": NEAR_DUPLICATE_JACCARD,
        "after_split_sha256": file_sha256(args.split),
        "excluded_split_sha256": [file_sha256(p) for p in args.exclude],
        "types": {name: dict(sorted(counts.items())) for name, counts in types.items()},
        **chosen,
    }


def check_confirmation(harness, rows: Sequence[Dict[str, Any]], split: Dict[str, Any],
                       confirm: Dict[str, Any]) -> None:
    """Refuse a C that shares a source question with E1's split or repeats one."""
    used = used_source_questions(harness, rows, [split])
    keys = [harness.group_key(rows[i]) for i in confirm["eval"]]
    if len(set(keys)) != len(keys) or set(keys) & used:
        raise ValueError("the confirmation set shares a source question with E1's split or "
                         "repeats one")


def per_row_nll(harness, model, items, pad_id: int, device: str,
                micro: int) -> Dict[str, List[float]]:
    """Summed response-token NLL and target count per row, batched as ``evaluate_nll``."""
    import torch
    from torch.nn import functional as fn

    sums: List[float] = []
    tokens: List[int] = []
    with torch.no_grad():
        for start in range(0, len(items), micro):
            chunk = items[start:start + micro]
            ids, labels, mask = harness._pad_batch(chunk, pad_id, device)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            targets = labels[:, 1:]
            loss = fn.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                    targets.reshape(-1), ignore_index=-100, reduction="none")
            sums.extend(loss.view(len(chunk), -1).sum(dim=1).tolist())
            tokens.extend((targets != -100).sum(dim=1).tolist())
    return {"nll_sum": sums, "tokens": tokens}


def score_confirmation(harness, model, tokenizer, *, rows, chat, e_indices, c_indices,
                       device: str, micro: int, gen_batch: int) -> Dict[str, Any]:
    """On the adapter the measured run just scored: C's NLL and accuracy, per-row E and C."""
    started = time.time()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    e_items, _ = harness.encode_split(rows, e_indices, tokenizer, chat)
    c_items, c_refs = harness.encode_split(rows, c_indices, tokenizer, chat)
    out: Dict[str, Any] = {"eval": harness.evaluate_nll(model, c_items, pad_id, device, micro)}
    out["per_row"] = {"E": per_row_nll(harness, model, e_items, pad_id, device, micro),
                      "C": per_row_nll(harness, model, c_items, pad_id, device, micro)}
    out["task"] = harness.score_task(model, tokenizer, [prompt for prompt, _ in c_items],
                                     c_refs, gen_batch, device)
    out["seconds"] = time.time() - started
    return out


def measured_argv(args: argparse.Namespace) -> List[str]:
    argv = ["run", "--model", args.model, "--data", args.data, "--split", args.split,
            "--out", args.out, "--device", args.device, "--dtype", args.dtype,
            "--arm", args.arm, "--lr", repr(args.lr), "--seed", str(args.seed),
            "--micro", str(args.micro), "--gen-batch", str(args.gen_batch)]
    return argv + (["--hot", args.hot] if args.hot else [])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    harness = load_measured_harness(args.harness)
    measured_args = harness.parse_args(measured_argv(args))
    rows = harness.load_rows(args.data)
    split = json.loads(pathlib.Path(args.split).read_text(encoding="utf-8"))
    confirm = json.loads(pathlib.Path(args.confirm_split).read_text(encoding="utf-8"))
    check_confirmation(harness, rows, split, confirm)
    captured: Dict[str, Any] = {}
    original = harness.score_gate

    def score_gate_then_confirm(model, tokenizer, batch, device):
        gate = original(model, tokenizer, batch, device)
        captured.update(score_confirmation(
            harness, model, tokenizer, rows=rows, chat=harness.MODELS[args.model]["chat"],
            e_indices=split["eval"], c_indices=confirm["eval"], device=device,
            micro=args.micro, gen_batch=args.gen_batch))
        return gate

    harness.score_gate = score_gate_then_confirm
    try:
        record = harness.run(measured_args)
    finally:
        harness.score_gate = original
    if captured:
        record["confirm"] = {"split_sha256": file_sha256(args.confirm_split),
                             "wrapper_sha256": file_sha256(__file__), **captured}
    elif record.get("status") == "ok":
        record["status"] = "void_confirm"
    return record


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """LF line endings on every OS, so the bytes (and the hash a run records) match."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_bytes((json.dumps(payload, indent=1) + "\n").encode("utf-8"))
    partial.replace(target)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-c")
    prep.add_argument("--exclude", action="append", default=[],
                      help="another E1 split whose source questions C must avoid")
    runner = sub.add_parser("run")
    runner.add_argument("--confirm-split", required=True)
    runner.add_argument("--model", required=True)
    runner.add_argument("--arm", required=True)
    runner.add_argument("--lr", type=float, default=0.0)
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--hot", default=None)
    runner.add_argument("--micro", type=int, default=4)
    runner.add_argument("--gen-batch", type=int, default=32)
    runner.add_argument("--device", default="cuda")
    runner.add_argument("--dtype", default="bfloat16")
    for cmd in (prep, runner):
        cmd.add_argument("--data", required=True)
        cmd.add_argument("--split", required=True, help="E1's counted split (split.json)")
        cmd.add_argument("--out", required=True)
        cmd.add_argument("--harness", default=str(MEASURED_HARNESS))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from rich.console import Console

    args = parse_args(argv)
    result = prepare(args) if args.command == "prepare-c" else run(args)
    write_json(args.out, result)
    Console().print(f"{args.command}: wrote {args.out} status={result.get('status', 'ok')}",
                    markup=False)
    return 0 if result.get("status", "ok") in ("ok", "not_applicable") else 1


if __name__ == "__main__":
    sys.exit(main())
