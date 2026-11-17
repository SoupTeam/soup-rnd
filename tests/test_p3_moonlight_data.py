"""Profile-bound pack receipts and strict tokenizer pins, without model weights."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest

from benchmarks.harness import p3_deepseek_quality_data as data
from benchmarks.harness import p3_moonlight_tokenizer as moonlight
from benchmarks.harness import p3_stage2_dataset as frozen

_ARCHIVE = Path(__file__).parent / "fixtures/moonlight_tokenizer_source.zip"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("moonlight-data-tokenizer") / moonlight.REVISION
    root.mkdir()
    prefix = "generated/"
    with ZipFile(_ARCHIVE) as archive:
        for name in moonlight.GENERATED_FILES:
            (root / name).write_bytes(archive.read(prefix + name))
    return root


@pytest.mark.parametrize("change", ["revision", "file_hash"])
def test_profile_tokenizer_admission_refuses_changed_revision_or_hash(
    tmp_path, tokenizer_dir, change
):
    copied = tmp_path / moonlight.REVISION
    shutil.copytree(tokenizer_dir, copied)
    if change == "revision":
        copied = copied.rename(tmp_path / "foreign-revision")
    else:
        with (copied / "tokenizer.json").open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(ValueError, match="tokenizer_revision|tokenizer_file_hash"):
        data.load_tokenizer(copied, profile_id="moonlight-16b-a3b")


@pytest.mark.parametrize("receipt, requested", [
    ("moonlight-16b-a3b", "deepseek-v3"), ("deepseek-v3", "moonlight-16b-a3b")])
def test_foreign_receipt_cannot_be_loaded_under_selected_profile(tmp_path, receipt, requested):
    # A rehashed receipt must fail by profile before tokenizer/source/window admission.
    root = tmp_path / "foreign-pack"
    root.mkdir()
    manifest = {"schema": data.POLICY, "policy": data.POLICY, "status": data.STATUS,
                "profile_id": receipt, "files": {}}
    frozen._write_manifest(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="pack_profile_mismatch"):
        data.load_pack(root, _sha(root / "manifest.json"), length=1024, steps=1,
                       profile_id=requested)


def test_independent_tokenizer_metadata_matches_strict_profile_admission(tokenizer_dir):
    independent, metadata = moonlight._admit_converted_tokenizer(tokenizer_dir)
    strict, admitted = data.load_tokenizer(tokenizer_dir, profile_id="moonlight-16b-a3b")
    assert metadata == admitted
    text = "[BOS] <|im_system|> math: 1 + 1 = 2\n"
    assert independent(text, add_special_tokens=False)["input_ids"] == strict(
        text, add_special_tokens=False)["input_ids"]


def test_private_assembly_and_public_build_publish_identical_manifest(tmp_path, tokenizer_dir):
    sources = (Path(__file__).parents[1] / moonlight.TRAINING_PATH)
    if not sources.is_file():
        pytest.skip("Frozen training bytes are required on the integration verifier")
    tokenizer, metadata = moonlight._admit_converted_tokenizer(tokenizer_dir)
    candidate, final = tmp_path / "candidate", tmp_path / "final"
    data._assemble_pack(sources=sources, sources_sha256=moonlight.TRAINING_SHA256,
                        tokenizer_dir=tokenizer_dir, tokenizer=tokenizer,
                        tokenizer_spec=metadata, output=candidate, profile_id="moonlight-16b-a3b")
    data.build_pack(sources=sources, sources_sha256=moonlight.TRAINING_SHA256,
                    tokenizer_dir=tokenizer_dir, output=final, profile_id="moonlight-16b-a3b")
    assert (candidate / "manifest.json").read_bytes() == (final / "manifest.json").read_bytes()
    windows, manifest = data.load_pack(final, _sha(final / "manifest.json"),
                                      length=1024, steps=1, profile_id="moonlight-16b-a3b")
    assert manifest["profile_id"] == "moonlight-16b-a3b"
    assert len(windows[0]["micro_batches"][0]["input_ids"][0][:64]) == 64
    assert max(token for window in windows for batch in window["micro_batches"]
               for token in batch["input_ids"][0]) < 163840
