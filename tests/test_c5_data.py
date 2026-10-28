"""C5 ticket 03: the Dolly train split and held-out panel (benchmarks/harness/c5_data.py).

Spec: .scratch/c5-q2-0-experts/issues/03-rule-and-data.md. Every dataset here is
a few hand-written rows, so nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"

# Six rows whose order by SHA-256 of "<row>\n<instruction>" was computed with
# `printf '%s\n%s' "$row" "$instruction" | shasum -a 256`: rows 5, 3, 4, 1, 0, 2.
INSTRUCTIONS = ["q", "a", "b", "c", "d", "e"]
HASH_ORDER = [5, 3, 4, 1, 0, 2]


@pytest.fixture(scope="module")
def harness() -> Iterator[ModuleType]:
    name = "_c5_data_test_module"
    previous = sys.modules.get(name)
    spec = importlib.util.spec_from_file_location(name, HARNESS / "c5_data.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


@pytest.fixture
def pinned_source(harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A 4600-row source file that the harness accepts as the pinned one."""
    source = tmp_path / "dolly.jsonl"
    monkeypatch.setattr(harness, "SOURCE_SHA256", _write_source(source, 4600))
    return source


def _rows() -> list[dict[str, str]]:
    return [
        {"instruction": text, "context": "", "response": f"answer {text}", "category": "qa"}
        for text in INSTRUCTIONS
    ]


def test_panel_takes_the_first_rows_by_hash_and_train_the_next(harness):
    splits = harness.build_splits(_rows(), heldout_rows=2, train_rows=3)

    assert [row["row"] for row in splits["heldout"]] == HASH_ORDER[:2]
    assert [row["row"] for row in splits["train"]] == HASH_ORDER[2:5]
    assert splits["heldout"][0] == {
        "row": 5, "instruction": "e", "context": "", "response": "answer e", "category": "qa",
    }


def test_shorter_train_split_is_a_prefix_of_the_full_one(harness):
    full = harness.build_splits(_rows(), heldout_rows=2, train_rows=3)
    cut = harness.build_splits(_rows(), heldout_rows=2, train_rows=2)

    assert cut["heldout"] == full["heldout"]
    assert cut["train"] == full["train"][:2]


def test_digest_is_sha256_of_sorted_key_utf8_jsonl(harness):
    split = [
        {"row": 0, "instruction": "é", "context": "", "response": "ответ", "category": "qa"},
        {"response": "r", "row": 7, "category": "qa", "instruction": "b", "context": "c"},
    ]

    # printf '%s\n%s\n' '{"category":"qa","context":"","instruction":"é",...}' ... | shasum -a 256
    assert harness.split_digest(split) == (
        "ddf92c6e6eadd20dd00600302389969b928e49ccd3bbaf69a047fa903394b96d"
    )


def _write_source(path: Path, n_rows: int) -> str:
    lines = [
        json.dumps({"instruction": f"q{i}", "context": "", "response": f"a{i}", "category": "qa"})
        for i in range(n_rows)
    ]
    payload = ("\n".join(lines) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_cli_refuses_a_source_file_other_than_the_pinned_one(harness, tmp_path):
    source = tmp_path / "dolly.jsonl"
    _write_source(source, 4600)
    out_dir = tmp_path / "out"

    assert harness.main(["--source", str(source), "--out-dir", str(out_dir)]) == 1
    assert not out_dir.exists()


def test_cli_writes_splits_whose_file_hashes_match_the_printed_ones(
    harness, pinned_source, tmp_path, capsys,
):
    hashes = []
    for run in ("first", "second"):
        out_dir = tmp_path / run
        assert harness.main(["--source", str(pinned_source), "--out-dir", str(out_dir)]) == 0
        manifest = json.loads((out_dir / "manifest.json").read_text())
        printed = capsys.readouterr().out
        for name, size in (("heldout", 512), ("train", 4000)):
            payload = (out_dir / f"{name}.jsonl").read_bytes()
            assert payload.count(b"\n") == size
            digest = hashlib.sha256(payload).hexdigest()
            assert manifest["splits"][name] == {"rows": size, "sha256": digest}
            assert digest in printed
        hashes.append(manifest["splits"])

    assert hashes[0] == hashes[1]
    assert manifest["source"]["rows"] == 4600


def test_cli_train_rows_cuts_the_train_split_to_its_prefix(harness, pinned_source, tmp_path):

    assert harness.main(
        ["--source", str(pinned_source), "--out-dir", str(tmp_path / "full")]
    ) == 0
    assert harness.main([
        "--source", str(pinned_source), "--out-dir", str(tmp_path / "cut"), "--train-rows", "2000",
    ]) == 0

    full = (tmp_path / "full" / "train.jsonl").read_bytes().splitlines(keepends=True)
    cut = (tmp_path / "cut" / "train.jsonl").read_bytes().splitlines(keepends=True)
    assert cut == full[:2000]
