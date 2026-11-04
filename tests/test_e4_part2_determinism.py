"""E4 part 2 quantization: reproducible output bytes and an exact output file set.

The earlier output set was retired because 17 codes files had headers that differed between two
runs. The safetensors writer takes ``metadata`` as a hash map, so a header with two metadata keys
is written in an order that changes from process to process. The harness writes one key. The
manifest and SHA256SUMS list exactly the expected output files; a stray or missing file is an
error.
"""
# ruff: noqa: I001 (the support import must run before the harness imports)

import json
import subprocess
import sys
import textwrap

import pytest

from tests.e4_part2_support import HARNESS_DIR, tiny_model, write_legacy_checkpoint

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("transformers")

import e4_part2_common as common  # noqa: E402
import e4_part2_quant as quant  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(common.library_versions()["mismatches"]),
    reason="the E4 part 2 harness is pinned to torch/transformers/bitsandbytes/peft of the rule",
)

LAYERS, EXPERTS = 2, 4
QUIET = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("det")
    return write_legacy_checkpoint(tiny_model(seed=3, dtype=torch.bfloat16), root / "source")


@pytest.fixture(scope="module")
def finished(checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("detout")
    manifest = out / "manifest.json"
    code = quant.run(checkpoint, out / "data", manifest, None, LAYERS, EXPERTS, log=QUIET)
    assert code in (0, 2)  # U2 is not compared without part 1 rows
    return {"out": out / "data", "manifest": json.loads(manifest.read_text()),
            "units": quant.enumerate_units(LAYERS, EXPERTS)}


def header_metadata_keys(path):
    import struct

    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return list(json.loads(handle.read(size)).get("__metadata__", {}))


# --- a: deterministic bytes -------------------------------------------------------------------


class TestDeterministicBytes:
    def test_no_safetensors_file_is_written_with_more_than_one_metadata_key(
            self, checkpoint, tmp_path, monkeypatch):
        import safetensors.torch as st

        seen = []
        real = st.save_file

        def spy(tensors, filename, metadata=None):
            seen.append((str(filename), dict(metadata or {})))
            return real(tensors, filename, metadata=metadata)

        monkeypatch.setattr(st, "save_file", spy)
        quant.quantize_pass(checkpoint, tmp_path, "on", True,
                            quant.enumerate_units(LAYERS, EXPERTS), QUIET)
        assert seen
        assert all(len(md) <= 1 for _, md in seen), [n for n, md in seen if len(md) > 1]

    def test_the_double_quant_setting_is_in_the_directory_name_not_the_header(self, finished):
        for tag in ("off", "on"):
            codes = sorted((finished["out"] / f"codes-dq{tag}").glob("*.safetensors"))
            assert codes
            assert all(header_metadata_keys(p) == ["format"] for p in codes)

    def test_fresh_processes_write_identical_files(self, checkpoint, tmp_path):
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(HARNESS_DIR)!r})
            import e4_part2_quant as q
            from pathlib import Path
            q.quantize_pass(Path(sys.argv[1]), Path(sys.argv[2]), "on", True,
                            q.enumerate_units({LAYERS}, {EXPERTS}), lambda *_: None)
        """)
        outputs = []
        for index in range(5):
            out = tmp_path / f"run{index}"
            subprocess.run([sys.executable, "-c", code, str(checkpoint), str(out)], check=True,
                           capture_output=True)
            outputs.append(out)
        names = sorted(p.relative_to(outputs[0]).as_posix() for p in outputs[0].rglob("*")
                       if p.is_file())
        assert names
        for name in names:
            assert len({(out / name).read_bytes() for out in outputs}) == 1, name

    def test_tensor_data_is_unchanged_by_the_writer(self, finished):
        from safetensors.torch import load_file

        packed = load_file(str(finished["out"] / "codes-dqon" / "layer-00.safetensors"))
        unit = finished["units"][0]
        assert unit.key in packed and packed[unit.key].dtype == torch.uint8


# --- b: the output file set is exact ---------------------------------------------------------


class TestExactOutputSet:
    def test_expected_names_equal_the_files_a_run_writes(self, finished, checkpoint):
        expected = quant.expected_output_names(checkpoint, finished["units"])
        written = sorted(p.relative_to(finished["out"]).as_posix()
                         for p in finished["out"].rglob("*")
                         if p.is_file() and p.name != "SHA256SUMS")
        assert expected == written
        assert set(finished["manifest"]["files"]) == set(expected)

    def test_the_real_layout_has_78_outputs(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        weight_map = {f"model.layers.{i}.self_attn.q_proj.weight": f"s{i}" for i in range(16)}
        weight_map["model.embed_tokens.weight"] = "s0"
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}))
        for name in quant.SIDECAR_FILES:
            (source / name).write_text("{}")
        names = quant.expected_output_names(source, quant.enumerate_units())
        # per setting: 16 codes + 17 shards (16 layers and "other") + index + 5 sidecars
        assert len(names) == 2 * (16 + 17 + 1 + 5) == 78

    def test_a_stray_file_is_an_error_not_a_manifest_entry(self, finished, checkpoint, tmp_path):
        import shutil

        out = tmp_path / "out"
        shutil.copytree(finished["out"], out)
        (out / ".DS_Store").write_bytes(b"\x00\x00")
        expected = quant.expected_output_names(checkpoint, finished["units"])
        with pytest.raises(quant.OutputSetError, match=r"\.DS_Store"):
            quant.hash_outputs(out, QUIET, expected)

    def test_a_missing_file_is_an_error(self, finished, checkpoint, tmp_path):
        import shutil

        out = tmp_path / "out"
        shutil.copytree(finished["out"], out)
        (out / "codes-dqon" / "layer-01.safetensors").unlink()
        expected = quant.expected_output_names(checkpoint, finished["units"])
        with pytest.raises(quant.OutputSetError, match="layer-01"):
            quant.hash_outputs(out, QUIET, expected)

    def test_sha256sums_lists_exactly_the_expected_files(self, finished):
        lines = (finished["out"] / "SHA256SUMS").read_text().splitlines()
        names = [line.split("  ", 1)[1] for line in lines]
        assert names == sorted(finished["manifest"]["files"])
        assert len(lines) == len(finished["manifest"]["files"])

    def test_a_tampered_sha256sums_is_an_error(self, finished, tmp_path):
        files = finished["manifest"]["files"]
        out = tmp_path
        good = "".join(f"{v['sha256']}  {k}\n" for k, v in sorted(files.items()))
        (out / "SHA256SUMS").write_text(good)
        quant.check_sums_file(out, files)  # consistent: no error
        (out / "SHA256SUMS").write_text(good + f"{'0' * 64}  .DS_Store\n")
        with pytest.raises(quant.OutputSetError, match="SHA256SUMS"):
            quant.check_sums_file(out, files)
        first = sorted(files)[0]
        (out / "SHA256SUMS").write_text(good.replace(files[first]["sha256"], "f" * 64))
        with pytest.raises(quant.OutputSetError, match="SHA256SUMS"):
            quant.check_sums_file(out, files)
        (out / "SHA256SUMS").write_text("".join(good.splitlines(True)[1:]))
        with pytest.raises(quant.OutputSetError, match="SHA256SUMS"):
            quant.check_sums_file(out, files)

    def test_a_file_that_appears_during_the_run_stops_it_before_the_manifest(
            self, checkpoint, tmp_path, monkeypatch):
        real_verify = quant.verify

        def verify_then_litter(source, out, units, part1, log):
            result = real_verify(source, out, units, part1, log)
            (out / ".DS_Store").write_bytes(b"finder")  # what the Finder did on the Mac
            return result

        monkeypatch.setattr(quant, "verify", verify_then_litter)
        manifest = tmp_path / "manifest.json"
        with pytest.raises(quant.OutputSetError, match=r"\.DS_Store"):
            quant.run(checkpoint, tmp_path / "data", manifest, None, LAYERS, EXPERTS, log=QUIET)
        assert not manifest.exists()
