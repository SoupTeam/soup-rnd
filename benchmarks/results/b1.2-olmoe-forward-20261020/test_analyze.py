"""CPU archive controls: altered evidence must not retain an accepted verdict."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def analyzer():
    spec = importlib.util.spec_from_file_location("b12_evidence_analysis", HERE / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewrite(tmp_path, mutate):
    with zipfile.ZipFile(HERE / "raw-results.zip") as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    mutate(files)
    manifest = {n: hashlib.sha256(data).hexdigest() for n, data in files.items()
                if n != "SHA256SUMS.json"}
    files["SHA256SUMS.json"] = json.dumps(manifest).encode()
    path = tmp_path / "altered.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def test_retained_archive_regenerates_analysis():
    result = analyzer().analyze(HERE / "raw-results.zip", REPO)
    assert result == json.loads((HERE / "analysis.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(("case", "error"), [
    ("output", "output hashes differ"),
    ("negative", "negative control ineffective"),
    ("budget", "arena exceeds budget"),
    ("trace", "expert load order differs"),
    ("protocol", "protocol differs from committed bytes"),
    ("pid", "child PID reused"),
])
def test_changed_records_rejected_even_with_updated_zip_hashes(tmp_path, case, error):
    def mutate(files):
        name = "layer-00-cuda-bf16/result.json"
        record = json.loads(files[name])
        if case == "output":
            record["result"]["observed_sha256"] = "0" * 64
        elif case == "negative":
            record["result"]["negative_sha256"] = record["result"]["expected_sha256"]
        elif case == "budget":
            record["result"]["pool"]["cuda_allocation_delta"] = 2**30
        elif case == "trace":
            record["result"]["trace"].pop()
        elif case == "protocol":
            protocol = json.loads(files["protocol.json"])
            protocol["replacements"] = 1
            files["protocol.json"] = json.dumps(protocol).encode()
        elif case == "pid":
            series = json.loads(files["series.json"])
            series["cells"][1]["pid"] = series["cells"][0]["pid"]
            files["series.json"] = json.dumps(series).encode()
        files[name] = json.dumps(record).encode()

    with pytest.raises(ValueError, match=error):
        analyzer().analyze(rewrite(tmp_path, mutate), REPO)


def test_changed_source_rejected_even_when_archive_hashes_are_updated(tmp_path):
    def mutate(files):
        stream = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(files["source.zip"])) as original, zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_DEFLATED
        ) as changed:
            changed.comment = original.comment
            for entry in original.infolist():
                data = original.read(entry)
                if entry.filename == "src/soup_cli/utils/expert_stream.py":
                    data += b"\n# altered instrument control\n"
                changed.writestr(entry, data)
        files["source.zip"] = stream.getvalue()
        launch = json.loads(files["launch.json"])
        launch["source_zip_sha256"] = hashlib.sha256(files["source.zip"]).hexdigest()
        files["launch.json"] = json.dumps(launch).encode()

    with pytest.raises(ValueError, match="source Git blob differs"):
        analyzer().analyze(rewrite(tmp_path, mutate), REPO)


def test_missing_attempt_cannot_be_accepted(tmp_path):
    def mutate(files):
        del files["layer-15-cuda-bf16-nf4-dq/result.json"]

    with pytest.raises(ValueError, match="missing or extra attempts"):
        analyzer().analyze(rewrite(tmp_path, mutate), REPO)
