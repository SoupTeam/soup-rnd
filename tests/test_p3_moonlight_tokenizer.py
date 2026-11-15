"""Moonlight conversion and parity regressions using the retained source bytes."""

from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from benchmarks.harness import p3_moonlight_tokenizer as moonlight

_EVIDENCE = Path(__file__).parent / "fixtures/moonlight_tokenizer_source.zip"


@pytest.fixture(scope="module")
def admitted_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("moonlight-sources") / moonlight.REVISION
    root.mkdir()
    with ZipFile(_EVIDENCE) as archive:
        for name in ("tiktoken.model", "tokenizer_config.json", "tokenization_moonshot.py"):
            member = name + ".txt" if name.endswith(".py") else name
            (root / name).write_bytes(archive.read(member))
    return root


@pytest.fixture(scope="module")
def conversion(tmp_path_factory, admitted_sources):
    root = tmp_path_factory.mktemp("moonlight-conversion")
    output = root / "first" / moonlight.REVISION
    report = moonlight.convert(source_dir=admitted_sources, output=output)
    return output, report


def test_conversion_reproduces_admitted_bytes_and_sparse_special_ids(
    tmp_path, admitted_sources, conversion
):
    output, report = conversion
    repeated = tmp_path / moonlight.REVISION
    report_path = tmp_path / "conversion-report.json"
    assert moonlight.main(["convert", "--source-dir", str(admitted_sources),
                           "--output", str(repeated), "--report", str(report_path)]) == 0
    receipt = json.loads(report_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["two_generations_byte_equal"] is True
    assert receipt["source_files"]["tiktoken.model"]["verified"] is True
    assert receipt["converter_source"]["patches_applied"] is False
    expected = {
        "tokenizer.json": (19592675,
                           "a5b137b65b40e0283a3c646472444078503dd3cb83c64b69379beca74288773d"),
        "tokenizer_config.json": (
            344, "7e0f5761740d72e66d290afdce2cf9d10c598d018503554066ee55d5bf2a30a2"),
        "chat_template.jinja": (527,
                                "0af7f5845651f3ba85a82d70d8af968a1da0a60f4c7bb662a9c305cfc9cc224c"),
    }
    for name, (size, digest) in expected.items():
        assert report["files"][name] == {"bytes": size, "sha256": digest}
        assert (output / name).read_bytes() == (repeated / name).read_bytes()
    tokenizer, metadata = moonlight._admit_converted_tokenizer(output)
    assert len(tokenizer) == 163840
    assert [tokenizer.convert_tokens_to_ids(text) for text in (
        "<|im_system|>", "<|im_middle|>", "<|reserved_token_163589|>", "[PAD]", "[UNK]"
    )] == [163594, 163601, 163589, 163838, 163839]
    assert metadata["eos_id"] == 163585
    assert tokenizer.split_special_tokens is True
    source = moonlight._admit_sources(admitted_sources)
    reference = moonlight._reference(source)
    for text in ("[BOS] [EOS] <|im_system|> <|reserved_token_163840|>",
                 " hello 世界 🌒\n\t café e\u0301 ", "abcXYZ"):
        report = {"coverage": {}, "first_mismatch": None}
        checker = moonlight.CheckedTokenizer(tokenizer, reference, tmp_path, report, io.StringIO())
        checker.bind({"role": "regression", "endpoint": "literal", "split": "test"}, ("text",))
        observed = checker(text, add_special_tokens=False, padding=False, truncation=False)
        assert observed["input_ids"] == reference.encode(text, disallowed_special=())
        assert all(token < 163584 for token in observed["input_ids"])


def test_changed_merge_fails_id_parity(tmp_path, admitted_sources, conversion):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    tokenizer_dir, _ = conversion
    source = moonlight._admit_sources(admitted_sources)
    reference = moonlight._reference(source)
    raw = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    left, right = raw["model"]["merges"].pop(0)

    decoder = {value: key for key, value in bytes_to_unicode().items()}
    text = bytes(decoder[value] for value in left + right).decode("utf-8")
    # BPE ignore_merges accepts whole vocabulary tokens without consulting merges.
    # Exceed every vocabulary entry so the removed whitespace merge must be used.
    text *= max(map(len, raw["model"]["vocab"])) + 1
    fast = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(raw)),
                                  split_special_tokens=True)
    report = {"coverage": {}, "first_mismatch": None}
    checker = moonlight.CheckedTokenizer(fast, reference, tmp_path, report, io.StringIO())
    checker.bind({"role": "regression", "endpoint": "merge", "split": "test"}, ("text",))
    with pytest.raises(moonlight.AdmissionError, match="token_id_parity"):
        checker(text, add_special_tokens=False, padding=False, truncation=False)
    mismatch = json.loads((tmp_path / "first-mismatch.json").read_text(encoding="utf-8"))
    assert mismatch["reference_ids"] != mismatch["fast_ids"]


@pytest.mark.parametrize("change", ["revision", "file_hash"])
def test_conversion_refuses_changed_source_revision_or_hash(tmp_path, admitted_sources, change):
    import shutil

    copied = tmp_path / moonlight.REVISION
    shutil.copytree(admitted_sources, copied)
    if change == "revision":
        copied = copied.rename(tmp_path / "wrong-revision")
    else:
        path = copied / "tiktoken.model"
        raw = path.read_bytes()
        replacement = b"A" if raw[:1] != b"A" else b"B"
        path.write_bytes(replacement + raw[1:])
    output = tmp_path / "generated" / moonlight.REVISION
    with pytest.raises(moonlight.AdmissionError, match="revision|source_identity"):
        moonlight.convert(source_dir=copied, output=output)
    assert not output.exists()
