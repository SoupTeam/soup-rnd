"""DeepSeek pack contracts, with a saved fast tokenizer and no network or weights."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.harness import p3_deepseek_quality_data as data
from benchmarks.harness import p3_stage2_dataset as frozen


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")) + "\n", encoding="utf-8")


def _row(domain: str, index: int, response_tokens: int = 4095) -> dict:
    benchmark = "gsm8k" if domain == "math" else "mbpp"
    ref = f"fixture/{benchmark}/train/{index}"
    prompt = "p "
    response = ("a " if domain == "math" else "b ") * (response_tokens - 1) + f"x{index}"
    identity = {"benchmark": benchmark, "task_id": ref, "source_ref": ref,
                "source_row_sha256": hashlib.sha256(ref.encode()).hexdigest()}
    return {**identity, "audit": identity, "domain": domain, "source_domain": benchmark,
            "split": "train", "sft_input_fields": ["prompt", "response"],
            "prompt": prompt, "response": response,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "response_sha256": hashlib.sha256(response.encode()).hexdigest()}


def _sources(path: Path, *, math_rows: int = 33, code_rows: int = 33,
             response_tokens: int = 4095) -> Path:
    rows = [_row(domain, index, response_tokens) for domain, count in
            (("math", math_rows), ("code", code_rows)) for index in range(count)]
    # Deliberately retain spaces and CRLF. Admission must hash bytes, not reserialized rows.
    path.write_bytes(b"".join((json.dumps(row, ensure_ascii=False) + "\r\n").encode("utf-8")
                             for row in rows))
    return path


@pytest.fixture(scope="module")
def tokenizer_fixture(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("quality-tokenizer") / "synthetic-fast-tokenizer"
    backend = Tokenizer(models.WordLevel({"<pad>": 0, "<eos>": 1, "<bos>": 2,
                                         "<unk>": 3, "p": 4, "a": 5, "b": 6,
                                         "abcXYZ": 7}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                       bos_token="<bos>", pad_token="<pad>",
                                       unk_token="<unk>")
    tokenizer.save_pretrained(root)
    names = ("tokenizer.json", "tokenizer_config.json")
    spec = {"repo_id": "test-only/synthetic-fast-tokenizer", "revision": root.name,
            "test_only": True, "files": {name: {"bytes": (root / name).stat().st_size,
                                                  "sha256": _sha(root / name)}
                                         for name in names}}
    return root, spec


@pytest.fixture(scope="module")
def built_pack(tmp_path_factory, tokenizer_fixture):
    root = tmp_path_factory.mktemp("quality-pack")
    sources = _sources(root / "sources.jsonl")
    tokenizer_dir, spec = tokenizer_fixture
    output = root / "pack"
    manifest = data.build_pack(sources=sources, sources_sha256=_sha(sources),
                               tokenizer_dir=tokenizer_dir, output=output,
                               _test_only_tokenizer=spec)
    return output, manifest, sources, tokenizer_dir, spec


@pytest.fixture(scope="module")
def cyclic_pack(tmp_path_factory, tokenizer_fixture):
    root = tmp_path_factory.mktemp("quality-cyclic-pack")
    sources = _sources(root / "sources.jsonl", math_rows=3, code_rows=2, response_tokens=4094)
    tokenizer_dir, spec = tokenizer_fixture
    output = root / "pack"
    manifest = data.build_pack(sources=sources, sources_sha256=_sha(sources),
                               tokenizer_dir=tokenizer_dir, output=output,
                               _test_only_tokenizer=spec)
    return output, manifest, sources, tokenizer_dir, spec


def _flatten(window: dict, key: str) -> list[int]:
    return [token for batch in window["micro_batches"] for token in batch[key][0]]


def test_pack_has_64_balanced_paired_windows_and_exact_shifted_denominators(built_pack):
    output, manifest, _, _, _ = built_pack
    layouts = {length: data.load_pack(output, _sha(output / "manifest.json"),
                                     length=length, steps=64)[0] for length in (1024, 4096)}
    assert len(layouts[1024]) == len(layouts[4096]) == 64
    assert [window["domain"] for window in layouts[1024]] == ["math", "code"] * 32
    for short, long in zip(layouts[1024], layouts[4096]):
        assert len(short["micro_batches"]) == 4
        assert len(long["micro_batches"]) == 1
        for window, length in ((short, 1024), (long, 4096)):
            ids, labels = _flatten(window, "input_ids"), _flatten(window, "labels")
            assert len(ids) == len(labels) == window["input_token_count"] == 4096
            assert all(len(batch["input_ids"]) == len(batch["labels"]) == 1
                       and len(batch["input_ids"][0]) == len(batch["labels"][0]) == length
                       for batch in window["micro_batches"])
            assert [labels[index] for index in (0, 1024, 2048, 3072)] == [-100] * 4
            assert all(label in (-100, token) for token, label in zip(ids, labels))
            assert all(label == -100 for token, label in zip(ids, labels) if token == 4)
            assert window["valid_target_count"] == sum(
                label != -100 for batch in window["micro_batches"]
                for label in batch["labels"][0][1:]) > 0
        assert _flatten(short, "input_ids") == _flatten(long, "input_ids")
        assert _flatten(short, "labels") == _flatten(long, "labels")
        assert short["pairing_sha256"] == long["pairing_sha256"]
    assert manifest["input_token_count"] == 262144
    assert manifest["domain_input_token_counts"] == {"math": 131072, "code": 131072}
    assert manifest["flat_pairing"]["1024"] == manifest["flat_pairing"]["4096"]
    assert manifest["granite_reference_supervised_targets_per_arm"] == 179326
    assert manifest["supervised_target_counts_per_length"] == {
        str(length): sum(window["valid_target_count"] for window in layout)
        for length, layout in layouts.items()}


def test_joint_offsets_mask_straddler_and_append_one_native_eos(tokenizer_fixture):
    tokenizer_dir, spec = tokenizer_fixture
    tokenizer, identity = data.load_tokenizer(tokenizer_dir, _test_only_tokenizer=spec)
    encoded = frozen._encode_document(tokenizer, {"prompt": "abc", "response": "XYZ a"})
    assert encoded["input_ids"] == [7, 5, 1]
    assert encoded["labels"] == [-100, 5, 1]
    assert encoded["boundary_straddling_masked_tokens"] == [0]
    assert identity["eos_id"] == 1
    assert identity["bos_id"] == 2
    assert identity["prepend_bos"] is False
    assert identity["add_special_tokens"] is False
    encoded = frozen._encode_document(tokenizer, {"prompt": "p ", "response": "a b"})
    assert encoded["input_ids"] == [4, 5, 6, 1]
    assert encoded["labels"] == [-100, 5, 6, 1]


def test_nonexhausted_sources_keep_cycle_zero_and_eos_matches_document_spans(built_pack):
    output, manifest, _, _, _ = built_pack
    spans = [
        json.loads(line) for line in (output / "document-spans.jsonl").read_text().splitlines()]
    assert len(spans) == 64
    assert all(span["cycle"] == 0 for span in spans)
    assert len({span["source_ref"] for span in spans}) == len(spans)
    assert sum(span["appended_eos_included"] for span in spans) == 62
    assert all(span["document_token_count"] == 4097 for span in spans)
    windows, _ = data.load_pack(output, _sha(output / "manifest.json"), length=4096, steps=64)
    assert sum(token == 1 for window in windows for token in _flatten(window, "input_ids")) == 62
    usage = [json.loads(line) for line in (output / "row-usage.jsonl").read_text().splitlines()]
    assert len(usage) == 66
    assert sum(row["used_input_token_count"] for row in usage) == 262144
    assert sum(row["used_input_token_count"] == 0 for row in usage) == 2
    assert manifest["exhaustion_policy"] == (
        "cycle_fixed_order_record_identity_cycle_and_final_cutoff")
    assert manifest["unique_source_counts_by_domain"] == {
        "math": {"available": 33, "used": 32}, "code": {"available": 33, "used": 32}}
    assert manifest["cycle_counts_by_domain"] == {
        domain: {"started": 1, "completed": 0, "repeated": 0} for domain in ("math", "code")}


def test_source_and_row_hashes_bind_exact_original_bytes(built_pack):
    output, manifest, sources, _, _ = built_pack
    assert (output / "training-sources.jsonl").read_bytes() == sources.read_bytes()
    descriptor = manifest["files"]["training_sources"]
    assert descriptor["sha256"] == _sha(sources)
    usage = [json.loads(line) for line in (output / "row-usage.jsonl").read_text().splitlines()]
    lines = sources.read_bytes().splitlines(keepends=True)
    for index, (line, row) in enumerate(zip(lines, usage)):
        assert row["physical_row_ordinal"] == index
        assert row["raw_row_sha256"] == hashlib.sha256(line).hexdigest()
        assert row["raw_row_bytes"] == len(line)
    assert usage[0]["raw_row_sha256"] != usage[0]["source_row_sha256"]


def test_two_builds_are_byte_identical_and_existing_output_is_untouched(tmp_path, built_pack):
    output, manifest, sources, tokenizer_dir, spec = built_pack
    second = tmp_path / "again"
    repeated = data.build_pack(sources=sources, sources_sha256=_sha(sources),
                               tokenizer_dir=tokenizer_dir, output=second,
                               _test_only_tokenizer=spec)
    assert repeated == manifest
    names = ["manifest.json", *(value["path"] for value in manifest["files"].values())]
    before = {name: (output / name).read_bytes() for name in names}
    assert before == {name: (second / name).read_bytes() for name in names}
    with pytest.raises(data.AdmissionError, match="existing_output"):
        data.build_pack(sources=sources, sources_sha256=_sha(sources),
                        tokenizer_dir=tokenizer_dir, output=output, _test_only_tokenizer=spec)
    assert before == {name: (output / name).read_bytes() for name in names}


def test_cycling_matches_frozen_granite_windows_spans_and_deterministic_bytes(
    tmp_path, cyclic_pack
):
    output, manifest, sources, tokenizer_dir, spec = cyclic_pack
    tokenizer, _ = data.load_tokenizer(tokenizer_dir, _test_only_tokenizer=spec)
    rows = [json.loads(line) for line in sources.read_bytes().splitlines()]
    expected_windows, expected_spans, expected_metadata = frozen._pack(tokenizer, rows)
    for length in (1024, 4096):
        windows, _ = data.load_pack(output, _sha(output / "manifest.json"), length=length, steps=64)
        assert windows == expected_windows[length]
    spans = [
        json.loads(line) for line in (output / "document-spans.jsonl").read_text().splitlines()]
    assert spans == expected_spans
    assert manifest["flat_pairing"] == expected_metadata["flat_pairing"]
    assert manifest["unique_available_tokens_by_domain"] == {"math": 12288, "code": 8192}
    assert manifest["unique_source_counts_by_domain"] == {
        "math": {"available": 3, "used": 3}, "code": {"available": 2, "used": 2}}
    for domain, count in (("math", 3), ("code", 2)):
        domain_spans = [span for span in spans if span["domain"] == domain]
        ordered_refs = sorted(
            (row["source_ref"] for row in rows if row["domain"] == domain),
            key=lambda ref: (hashlib.sha256(
                ("p3-stage2-data-order-v1\0" + ref).encode()).hexdigest(), ref))
        assert len(domain_spans) == 32
        for number, span in enumerate(domain_spans):
            assert span["source_ref"] == ordered_refs[number % count]
            assert span["cycle"] == number // count
            assert span["source_order_position"] == number % count
            assert span["domain_document_number"] == number
            assert span["stream_start"] == number * 4096
            assert span["stream_end"] == (number + 1) * 4096
            assert span["document_token_start"] == 0
            assert span["document_token_end"] == 4096
            assert span["final_document_cutoff"] is False
            assert span["appended_eos_included"] is True
    assert manifest["cycle_counts_by_domain"] == {
        "math": {"started": 11, "completed": 10, "repeated": 10},
        "code": {"started": 16, "completed": 16, "repeated": 15}}
    assert manifest["row_usage"] == {
        "available_by_domain": {"math": 3, "code": 2}, "used_by_domain": {"math": 3, "code": 2},
        "occurrences_by_domain": {"math": 32, "code": 32}}
    usage = [json.loads(line) for line in (output / "row-usage.jsonl").read_text().splitlines()]
    assert len(usage) == len({(row["source_ref"], row["cycle"]) for row in usage}) == 64
    assert sum(row["used_input_token_count"] for row in usage) == 262144
    original_lines = sources.read_bytes().splitlines(keepends=True)
    for occurrence in usage:
        original = original_lines[occurrence["physical_row_ordinal"]]
        assert occurrence["raw_row_sha256"] == hashlib.sha256(original).hexdigest()
    repeated_dir = tmp_path / "cyclic-again"
    repeated = data.build_pack(sources=sources, sources_sha256=_sha(sources),
                               tokenizer_dir=tokenizer_dir, output=repeated_dir,
                               _test_only_tokenizer=spec)
    assert repeated == manifest
    for name in ["manifest.json", *(value["path"] for value in manifest["files"].values())]:
        assert (repeated_dir / name).read_bytes() == (output / name).read_bytes()


@pytest.mark.parametrize("change", ["revision", "file_hash", "missing_file", "unpinned_file"])
def test_tokenizer_mismatch_is_refused_before_encoding(tmp_path, tokenizer_fixture, change):
    tokenizer_dir, spec = tokenizer_fixture
    copied = tmp_path / tokenizer_dir.name
    shutil.copytree(tokenizer_dir, copied)
    if change == "revision":
        copied = copied.rename(tmp_path / "wrong-revision")
    elif change == "file_hash":
        with (copied / "tokenizer.json").open("ab") as stream:
            stream.write(b" ")
    elif change == "missing_file":
        (copied / "tokenizer_config.json").unlink()
    else:
        (copied / "added_tokens.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="tokenizer"):
        data.load_tokenizer(copied, _test_only_tokenizer=spec)


def test_synthetic_tokenizer_is_never_admitted_as_production(tmp_path, tokenizer_fixture):
    tokenizer_dir, _ = tokenizer_fixture
    with pytest.raises(ValueError, match="revision"):
        data.load_tokenizer(tokenizer_dir)
    copied = tmp_path / data.REVISION
    shutil.copytree(tokenizer_dir, copied)
    with pytest.raises(ValueError, match="tokenizer_file"):
        data.load_tokenizer(copied)


def test_fixture_admission_requires_explicit_test_only_identity(tokenizer_fixture):
    tokenizer_dir, spec = tokenizer_fixture
    with pytest.raises(ValueError, match="test_only"):
        data.load_tokenizer(tokenizer_dir, _test_only_tokenizer={**spec, "test_only": False})


def test_hf_style_snapshot_links_are_admitted_by_pinned_bytes(tmp_path, tokenizer_fixture):
    tokenizer_dir, spec = tokenizer_fixture
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    snapshot = tmp_path / "snapshots" / tokenizer_dir.name
    snapshot.mkdir(parents=True)
    for name in spec["files"]:
        target = blobs / name
        shutil.copyfile(tokenizer_dir / name, target)
        try:
            (snapshot / name).symlink_to(target)
        except OSError:
            pytest.skip("Creating symbolic links is unavailable on this host")
    tokenizer, observed = data.load_tokenizer(snapshot, _test_only_tokenizer=spec)
    assert tokenizer("p a", add_special_tokens=False)["input_ids"] == [4, 5]
    assert observed["pin"] == spec


@pytest.mark.parametrize("change", ["wrong_digest", "trailing_bytes", "duplicate_source"])
def test_source_admission_rejects_changed_bytes_and_duplicate_rows(
    tmp_path, tokenizer_fixture, change
):
    tokenizer_dir, spec = tokenizer_fixture
    sources = _sources(tmp_path / "sources.jsonl")
    expected = _sha(sources)
    if change == "wrong_digest":
        expected = "0" * 64
    elif change == "trailing_bytes":
        with sources.open("ab") as stream:
            stream.write(b" ")
    else:
        lines = sources.read_bytes().splitlines(keepends=True)
        sources.write_bytes(b"".join([lines[0], lines[0], *lines[2:]]))
        expected = _sha(sources)
    output = tmp_path / "failed"
    with pytest.raises(data.AdmissionError):
        data.build_pack(sources=sources, sources_sha256=expected, tokenizer_dir=tokenizer_dir,
                        output=output, _test_only_tokenizer=spec)
    assert not (output / "manifest.json").exists()
    assert (output / "FAILED.json").is_file()


@pytest.mark.parametrize("length,steps", [(1024, 1), (4096, 3), (1024, 64)])
def test_loader_returns_only_requested_prefix_and_verified_manifest(built_pack, length, steps):
    output, manifest, _, _, _ = built_pack
    windows, observed = data.load_pack(output, _sha(output / "manifest.json"),
                                      length=length, steps=steps)
    assert observed == manifest
    assert observed["test_only"] is True
    assert len(windows) == steps
    assert windows == json.loads((output / f"windows-L{length}.json").read_text())[:steps]


@pytest.mark.parametrize("length,steps", [(64, 1), (1024, 0), (4096, 65), (1024, True)])
def test_loader_refuses_unsupported_layout_or_insufficient_windows(built_pack, length, steps):
    output, _, _, _, _ = built_pack
    with pytest.raises(ValueError):
        data.load_pack(output, _sha(output / "manifest.json"), length=length, steps=steps)


def test_loader_refuses_manifest_digest_mismatch(built_pack):
    with pytest.raises(ValueError, match="manifest.*hash"):
        data.load_pack(built_pack[0], "0" * 64, length=1024, steps=1)


@pytest.mark.parametrize("name", ["windows-L1024.json", "windows-L4096.json",
                                  "document-spans.jsonl", "row-usage.jsonl",
                                  "training-sources.jsonl", "tokenizer/tokenizer.json",
                                  "tokenizer/tokenizer_config.json"])
def test_loader_verifies_every_file_even_outside_selected_prefix(tmp_path, built_pack, name):
    output = tmp_path / "corrupt"
    shutil.copytree(built_pack[0], output)
    with (output / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        data.load_pack(output, _sha(output / "manifest.json"), length=1024, steps=1)


@pytest.mark.parametrize("change", ["missing_required_file", "path_escape", "boundary",
                                    "tokenizer_revision", "source_digest", "window_span",
                                    "usage_identity", "internal_manifest_hash"])
def test_rehashed_manifest_cannot_bypass_pack_contract(tmp_path, built_pack, change):
    output = tmp_path / "altered"
    shutil.copytree(built_pack[0], output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "missing_required_file":
        manifest["files"].pop("windows_L4096")
    elif change == "path_escape":
        manifest["files"]["document_spans"]["path"] = "../outside.jsonl"
    elif change == "tokenizer_revision":
        manifest["tokenizer"]["revision"] = "changed"
    elif change == "source_digest":
        manifest["training_sources_sha256"] = "0" * 64
    elif change == "usage_identity":
        path = output / "row-usage.jsonl"
        usage = [json.loads(line) for line in path.read_text().splitlines()]
        usage[-1]["raw_row_sha256"] = "0" * 64
        path.write_text("".join(json.dumps(row) + "\n" for row in usage), encoding="utf-8")
        manifest["files"]["row_usage"].update(bytes=path.stat().st_size, sha256=_sha(path))
    elif change == "internal_manifest_hash":
        manifest["input_token_count"] += 1
        _save(manifest_path, manifest)
        with pytest.raises(ValueError, match="manifest_payload_hash"):
            data.load_pack(output, _sha(manifest_path), length=1024, steps=1)
        return
    else:
        path = output / "windows-L4096.json"
        windows = json.loads(path.read_text())
        if change == "window_span":
            windows[-1]["document_spans"][0]["document_token_start"] += 1
        else:
            windows[-1]["micro_batches"][0]["labels"][0][1024] = 6
        _save(path, windows)
        manifest["files"]["windows_L4096"].update(bytes=path.stat().st_size, sha256=_sha(path))
    manifest.pop("manifest_payload_sha256")
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    manifest["manifest_payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    _save(manifest_path, manifest)
    with pytest.raises(ValueError):
        data.load_pack(output, _sha(manifest_path), length=1024, steps=1)


def test_cli_failure_preserves_evidence_and_has_no_fixture_bypass(tmp_path, tokenizer_fixture):
    tokenizer_dir, _ = tokenizer_fixture
    sources = _sources(tmp_path / "sources.jsonl", math_rows=1, code_rows=1)
    output = tmp_path / "cli-refused"
    assert data.main(["--sources", str(sources), "--sources-sha256", _sha(sources),
                      "--tokenizer-dir", str(tokenizer_dir), "--output", str(output)]) == 1
    assert (output / "FAILED.json").is_file()
    assert not (output / "manifest.json").exists()
    with pytest.raises(SystemExit) as error:
        data.main(["--test-only-tokenizer", "anything"])
    assert error.value.code == 2


def test_module_cli_is_callable_without_heavy_model_setup():
    result = subprocess.run([sys.executable, "-m", "benchmarks.harness.p3_deepseek_quality_data",
                             "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--sources-sha256" in result.stdout
    assert "--tokenizer-dir" in result.stdout


def test_exact_unique_token_floor_completes_one_cycle_without_repetition(
    tmp_path, tokenizer_fixture
):
    tokenizer_dir, spec = tokenizer_fixture
    sources = _sources(tmp_path / "exact.jsonl", math_rows=32, code_rows=32, response_tokens=4094)
    output = tmp_path / "exact-pack"
    manifest = data.build_pack(sources=sources, sources_sha256=_sha(sources),
                               tokenizer_dir=tokenizer_dir, output=output,
                               _test_only_tokenizer=spec)
    assert manifest["unique_available_tokens_by_domain"] == {"math": 131072, "code": 131072}
    assert all(not span["final_document_cutoff"] for span in manifest["final_cutoffs"].values())
    assert manifest["cycle_counts_by_domain"] == {
        domain: {"started": 1, "completed": 1, "repeated": 0} for domain in ("math", "code")}
    windows, _ = data.load_pack(output, _sha(output / "manifest.json"), length=4096, steps=64)
    assert all(_flatten(window, "input_ids")[-1] == 1 for window in windows)


def test_one_token_short_domain_repeats_exactly_one_token_then_cuts_off(
    tmp_path, tokenizer_fixture
):
    tokenizer_dir, spec = tokenizer_fixture
    sources = _sources(tmp_path / "short-by-one.jsonl", math_rows=32, code_rows=32,
                       response_tokens=4094)
    rows = [json.loads(line) for line in sources.read_bytes().splitlines()]
    row = rows[0]
    row["response"] = row["response"].replace("a ", "", 1)
    row["response_sha256"] = hashlib.sha256(row["response"].encode()).hexdigest()
    sources.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows))
    output = tmp_path / "repeat-one-token"
    manifest = data.build_pack(sources=sources, sources_sha256=_sha(sources),
                               tokenizer_dir=tokenizer_dir, output=output,
                               _test_only_tokenizer=spec)
    assert manifest["unique_available_tokens_by_domain"] == {"math": 131071, "code": 131072}
    assert manifest["cycle_counts_by_domain"] == {
        "math": {"started": 2, "completed": 1, "repeated": 1},
        "code": {"started": 1, "completed": 1, "repeated": 0}}
    cutoff = manifest["final_cutoffs"]["math"]
    assert cutoff["cycle"] == 1
    assert cutoff["source_order_position"] == 0
    assert cutoff["domain_document_number"] == 32
    assert cutoff["document_token_end"] == 1
    assert cutoff["final_document_cutoff"] is True
    assert cutoff["appended_eos_included"] is False
    windows, _ = data.load_pack(output, _sha(output / "manifest.json"), length=1024, steps=64)
    assert len(windows) == 64


def test_zero_supervised_window_refuses_manifest(tmp_path, tokenizer_fixture):
    tokenizer_dir, spec = tokenizer_fixture
    rows = [_row(domain, 0) for domain in ("math", "code")]
    for row in rows:
        row["prompt"] = "p " * 131072
        row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
    sources = tmp_path / "prompt-only-windows.jsonl"
    sources.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows))
    output = tmp_path / "zero-target"
    with pytest.raises(data.AdmissionError, match="zero_target_window"):
        data.build_pack(sources=sources, sources_sha256=_sha(sources),
                        tokenizer_dir=tokenizer_dir, output=output, _test_only_tokenizer=spec)
    assert not (output / "manifest.json").exists()
    assert (output / "FAILED.json").is_file()


def test_partial_publication_failure_preserves_files_without_success_manifest(
    tmp_path, built_pack, monkeypatch
):
    _, _, sources, tokenizer_dir, spec = built_pack
    output = tmp_path / "publication-failure"

    def fail_after_one(stage, root, names):
        assert names[-1] == "manifest.json"
        shutil.copyfile(stage / names[0], root / names[0])
        raise OSError("fixture publication interrupted")

    monkeypatch.setattr(frozen, "_publish", fail_after_one)
    with pytest.raises(data.AdmissionError):
        data.build_pack(sources=sources, sources_sha256=_sha(sources),
                        tokenizer_dir=tokenizer_dir, output=output, _test_only_tokenizer=spec)
    assert (output / "training-sources.jsonl").read_bytes() == sources.read_bytes()
    assert (output / "FAILED.json").is_file()
    assert not (output / "manifest.json").exists()


def test_loader_refuses_symlink_escape_even_with_identical_external_bytes(tmp_path, built_pack):
    output = tmp_path / "linked-pack"
    shutil.copytree(built_pack[0], output)
    path = output / "document-spans.jsonl"
    outside = tmp_path / "outside-spans.jsonl"
    shutil.copyfile(path, outside)
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symbolic links is unavailable on this host")
    with pytest.raises(ValueError, match="escapes"):
        data.load_pack(output, _sha(output / "manifest.json"), length=1024, steps=1)


def test_cli_parses_inputs_and_publishes_verifiable_pack(tmp_path, built_pack, monkeypatch):
    _, _, sources, tokenizer_dir, spec = built_pack
    build = data.build_pack

    def fixture_build(**kwargs):
        return build(**kwargs, _test_only_tokenizer=spec)

    # The CLI has no fixture flag. Inject only its production admission boundary here.
    monkeypatch.setattr(data, "build_pack", fixture_build)
    output = tmp_path / "cli-pack"
    assert data.main(["--sources", str(sources), "--sources-sha256", _sha(sources),
                      "--tokenizer-dir", str(tokenizer_dir), "--output", str(output)]) == 0
    windows, manifest = data.load_pack(output, _sha(output / "manifest.json"),
                                       length=1024, steps=3)
    assert len(windows) == 3
    assert manifest["files"]["training_sources"]["sha256"] == _sha(sources)


@pytest.mark.parametrize("change", ["cycle", "position", "document_number", "source_order",
                                    "stream_start", "cutoff", "usage_cycle", "cycle_count",
                                    "unique_source_count", "window_cycle"])
def test_loader_rejects_rehashed_cycle_or_order_corruption(tmp_path, cyclic_pack, change):
    output = tmp_path / "corrupt-cycle"
    shutil.copytree(cyclic_pack[0], output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spans_path = output / "document-spans.jsonl"
    spans = [json.loads(line) for line in spans_path.read_text().splitlines()]
    repeated_span = next(span for span in spans if span["cycle"] == 1)
    if change in {"cycle", "position", "document_number", "stream_start", "cutoff"}:
        if change == "cycle":
            repeated_span["cycle"] += 1
        elif change == "position":
            repeated_span["source_order_position"] += 1
        elif change == "document_number":
            repeated_span["domain_document_number"] += 1
        elif change == "stream_start":
            repeated_span["stream_start"] += 1
        else:
            repeated_span["final_document_cutoff"] = True
        spans_path.write_text("".join(json.dumps(span) + "\n" for span in spans), encoding="utf-8")
        manifest["files"]["document_spans"].update(
            bytes=spans_path.stat().st_size, sha256=_sha(spans_path))
    elif change == "source_order":
        manifest["domain_source_order"]["math"].reverse()
    elif change == "cycle_count":
        manifest["cycle_counts_by_domain"]["math"]["completed"] += 1
    elif change == "unique_source_count":
        manifest["unique_source_counts_by_domain"]["code"]["used"] += 1
    elif change == "window_cycle":
        path = output / "windows-L4096.json"
        windows = json.loads(path.read_text())
        next(span for window in windows for span in window["document_spans"]
             if span["cycle"] == 1)["cycle"] = 0
        _save(path, windows)
        manifest["files"]["windows_L4096"].update(bytes=path.stat().st_size, sha256=_sha(path))
    else:
        usage_path = output / "row-usage.jsonl"
        usage = [json.loads(line) for line in usage_path.read_text().splitlines()]
        next(row for row in usage if row["cycle"] == 1)["cycle"] = 0
        usage_path.write_text("".join(json.dumps(row) + "\n" for row in usage), encoding="utf-8")
        manifest["files"]["row_usage"].update(
            bytes=usage_path.stat().st_size, sha256=_sha(usage_path))
    manifest.pop("manifest_payload_sha256")
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    manifest["manifest_payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    _save(manifest_path, manifest)
    with pytest.raises(ValueError):
        data.load_pack(output, _sha(manifest_path), length=1024, steps=1)


@pytest.mark.parametrize("change", ["empty_domain", "malformed_row"])
def test_invalid_sources_preserve_failed_diagnostics(tmp_path, tokenizer_fixture, change):
    tokenizer_dir, spec = tokenizer_fixture
    sources = _sources(tmp_path / "invalid.jsonl", math_rows=1, code_rows=0)
    if change == "malformed_row":
        with sources.open("ab") as stream:
            stream.write(b'{"broken":\n')
    output = tmp_path / "invalid-pack"
    with pytest.raises(data.AdmissionError):
        data.build_pack(sources=sources, sources_sha256=_sha(sources),
                        tokenizer_dir=tokenizer_dir, output=output, _test_only_tokenizer=spec)
    failure = json.loads((output / "FAILED.json").read_text())
    assert failure["status"] == "FAILED"
    assert failure["training_sources_sha256"] == _sha(sources)
    assert failure["training_source_rows_by_domain"] == {"math": 1, "code": 0}
    assert not (output / "manifest.json").exists()
