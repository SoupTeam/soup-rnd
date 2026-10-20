#!/usr/bin/env python3
"""Laptop setup for the F3 audit: lock files, foreign code, dataset hash, power-cap check.

Everything here happens before the rental and costs nothing. The pure helpers
(``filter_lock``, ``banned_installed``, ``power_cap_problem``, ``dataset_sha256``,
``fetch_commands``) are tested without a network. The ``main`` subcommands run them for real:

    f3_setup.py power                      refuse if the GPU sits at a power cap
    f3_setup.py fetch NAME                 clone a foreign repo at its pinned commit, outside git
    f3_setup.py data --tokenizer ID        pin and tokenize Alpaca-cleaned, print the sha256
    f3_setup.py verify VENV                fail if a banned package is installed in VENV

Foreign code goes to ``~/f3-foreign``, never into this repository (house rule 6). Cut
Cross-Entropy and xFormers are removed from the lock files and checked absent from the venvs.

Benchmark harness code, not shipped. Heavy imports are lazy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
FOREIGN_DIR = Path.home() / "f3-foreign"
VENV_DIR = Path.home() / ".venvs"

#: Their licences must not touch the artifact (spec, user story 39).
BANNED = ("cut-cross-entropy", "xformers")

#: Pinned foreign sources: name to (url, commit).
FOREIGN = {
    "chronicals": (
        "https://github.com/Ajwebdevs/Chronicals.git",
        "9b33605177a98bc798387c1742b9d36e53ea9712",
    ),
}

DATASET = "yahma/alpaca-cleaned"
#: A GPU current power limit at or below this many watts counts as capped (10 W seen on 2026-10-09).
POWER_CAP_WATTS = 10.0


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def filter_lock(text: str, banned: Sequence[str]) -> str:
    """Drop the banned packages and their ``# via`` comment lines from a compiled lock file."""
    wanted = {_norm(name) for name in banned}
    kept: List[str] = []
    skipping = False
    for line in text.splitlines(keepends=True):
        match = re.match(r"([A-Za-z0-9_.-]+)==", line)
        if match:
            skipping = _norm(match.group(1)) in wanted
        elif not line.startswith(" "):
            skipping = False
        comment = re.match(r"\s+#\s+(?:via\s+)?([A-Za-z0-9_.-]+)\s*$", line)
        if comment and _norm(comment.group(1)) in wanted:
            continue
        if not skipping:
            kept.append(line)
    return "".join(kept)


def banned_installed(installed: Iterable[str], banned: Sequence[str]) -> List[str]:
    """The banned names present among ``installed``, in the order of ``banned``."""
    present = {_norm(name) for name in installed}
    return [name for name in banned if _norm(name) in present]


def power_cap_problem(smi_power_text: str) -> Optional[str]:
    """Why the GPU counts as capped, from ``nvidia-smi -q -d POWER`` text; None if it does not."""
    match = re.search(r"Current Power Limit\s*:\s*([0-9.]+)\s*W", smi_power_text)
    if match is None:
        return "could not read the current power limit from nvidia-smi"
    watts = float(match.group(1))
    if watts <= POWER_CAP_WATTS:
        return f"GPU power limit is {watts:g} W, a cap; reboot before measuring"
    return None


def dataset_sha256(rows: Sequence[Sequence[int]]) -> str:
    """sha256 of the token ids as compact JSON text. Row order and boundaries count."""
    text = json.dumps([list(row) for row in rows], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_commands(url: str, commit: str, dest: Path) -> List[List[str]]:
    """Git commands that put ``url`` at ``commit`` in ``dest``, which must be outside the repo."""
    real = os.path.realpath(dest)
    if os.path.commonpath([real, os.path.realpath(REPO)]) == os.path.realpath(REPO):
        raise ValueError(f"{dest} is inside the repository; foreign code stays outside git")
    return [
        ["git", "clone", "--no-checkout", url, str(dest)],
        ["git", "-C", str(dest), "checkout", "--detach", commit],
    ]


def _run(command: List[str]) -> None:
    subprocess.run(command, check=True)  # noqa: S603


def cmd_power() -> int:
    out = subprocess.run(  # noqa: S603
        ["nvidia-smi", "-q", "-d", "POWER"], capture_output=True, text=True, check=False
    ).stdout
    problem = power_cap_problem(out)
    print(problem or "power limit ok")
    return 1 if problem else 0


def cmd_fetch(name: str) -> int:
    url, commit = FOREIGN[name]
    dest = FOREIGN_DIR / name
    if dest.exists():
        head = subprocess.run(  # noqa: S603
            ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        if head != commit:
            print(f"{dest} is at {head}, not {commit}")
            return 1
        print(f"{name} already at {commit}")
        return 0
    for command in fetch_commands(url, commit, dest):
        _run(command)
    print(f"{name} at {commit} in {dest}")
    return 0


def cmd_data(tokenizer_id: str, out: Path, max_length: int) -> int:
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    revision = HfApi().dataset_info(DATASET).sha
    data = load_dataset(DATASET, split="train", revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    rows = []
    for item in data:
        prompt = item["instruction"] + (("\n" + item["input"]) if item["input"] else "")
        text = prompt + "\n" + item["output"]
        rows.append(tokenizer(text, truncation=True, max_length=max_length)["input_ids"])
    record = {
        "dataset": DATASET,
        "revision": revision,
        "tokenizer": tokenizer_id,
        "max_length": max_length,
        "examples": len(rows),
        "sha256": dataset_sha256(rows),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens.json").write_text(json.dumps(rows), encoding="utf-8")
    (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


def cmd_verify(venv: Path) -> int:
    code = (
        "import importlib.metadata as m, json;"
        "print(json.dumps([d.metadata['Name'] for d in m.distributions()]))"
    )
    out = subprocess.run(  # noqa: S603
        [str(venv / "bin" / "python"), "-I", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    names = json.loads(out.stdout)
    found = banned_installed(names, BANNED)
    print(f"{venv.name}: {len(names)} packages, banned installed: {found or 'none'}")
    return 1 if found else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("power")
    fetch = sub.add_parser("fetch")
    fetch.add_argument("name", choices=sorted(FOREIGN))
    data = sub.add_parser("data")
    data.add_argument("--tokenizer", required=True)
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--max-length", type=int, default=512)
    verify = sub.add_parser("verify")
    verify.add_argument("venv", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "power":
        return cmd_power()
    if args.cmd == "fetch":
        return cmd_fetch(args.name)
    if args.cmd == "data":
        return cmd_data(args.tokenizer, args.out, args.max_length)
    return cmd_verify(args.venv)


if __name__ == "__main__":
    sys.exit(main())
