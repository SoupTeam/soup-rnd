"""Prospective execution gates, complete-sample replay and fixed attempt budgets."""

from __future__ import annotations

import copy
import importlib.util
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/d3_grouped_performance.py"
    spec = importlib.util.spec_from_file_location("d3_performance_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm(arm, *, mode="timed", value=None, codec="dense"):
    value = (1.0 if arm == "A" else 0.8) if value is None else value
    return {"arm": arm, "mode": mode, "formal_measurement": True, "status": "ok",
            "codec": codec, "source": {"git_head": "a" * 40}, "protocol_sha256": "b" * 64,
            "recipe_sha256": "c" * 64, "expected_sha256": "d" * 64,
            "F1": {"outcome": "ok"}, "median_seconds": value,
            "samples": [{"wall_seconds": value, "output_sha256": "d" * 64} for _ in range(2)]}


def test_disabled_formal_protocol_cannot_reach_git_or_gpu(monkeypatch):
    module = _module()

    def unexpected(*args, **kwargs):
        raise AssertionError("formal gate reached external work")

    monkeypatch.setattr(module.subprocess, "check_output", unexpected)
    with pytest.raises(ValueError, match="disabled"):
        module.frozen_source({"execution_enabled": False})


def test_dirty_source_refused_before_git_blob_reads(monkeypatch):
    module = _module()

    def git(command, **kwargs):
        assert "status" in command
        return b"?? uncommitted.py\n"

    monkeypatch.setattr(module.subprocess, "check_output", git)
    with pytest.raises(ValueError, match="clean checkout"):
        module.frozen_source({"execution_enabled": True})


def test_incomplete_pins_are_refused_even_in_a_clean_source_tree(monkeypatch):
    module = _module()

    def git(command, **kwargs):
        assert "status" in command
        return b""

    monkeypatch.setattr(module.subprocess, "check_output", git)
    protocol = {"execution_enabled": True, "recipe_sha256": {c: None for c in module.CODECS}}
    with pytest.raises(ValueError, match="incomplete input"):
        module.frozen_source(protocol)


def test_frozen_source_keeps_binary_bytes_and_normalizes_only_text(tmp_path, monkeypatch):
    import hashlib

    module = _module()
    protocol = json.loads(module.PROTOCOL.read_text())
    binary = b"binary\x00payload\r\nunchanged"
    protocol["setup_archive_sha256"] = hashlib.sha256(binary).hexdigest()
    protocol_name = "benchmarks/protocols/d3-grouped-experts/performance-protocol.json"
    blobs = {name: binary if name.endswith(".zip") else b"# pinned source\n"
             for name in protocol["sources"]}
    blobs[protocol_name] = (json.dumps(protocol) + "\n").encode()
    for name, data in blobs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if name.endswith(".zip") else data.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PROTOCOL", tmp_path / protocol_name)

    def git(command, **kwargs):
        if "status" in command:
            return b""
        if "rev-parse" in command:
            return b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        assert command[3] == "show"
        return blobs[command[4].split(":", 1)[1]]

    monkeypatch.setattr(module.subprocess, "check_output", git)
    result = module.frozen_source(protocol)
    assert result["git_blob_sha256"][protocol["setup_archive"]] == protocol["setup_archive_sha256"]
    (tmp_path / protocol["sources"][0]).write_bytes(b"changed source\n")
    with pytest.raises(ValueError, match="differs"):
        module.frozen_source(protocol)


def test_file_containment_refuses_parent_or_absolute_escape(tmp_path):
    module = _module()
    for path in ("../elsewhere", str(tmp_path.parent / "elsewhere")):
        with pytest.raises(ValueError, match="escapes"):
            module.contained(tmp_path, path)


def test_prepared_hash_tamper_is_refused(tmp_path):
    module = _module()
    folder = tmp_path / "layer-07-dense"
    folder.mkdir()
    (folder / "experts.safetensors").write_bytes(b"weight bytes")
    (folder / "inputs.safetensors").write_bytes(b"input bytes")
    recipe = {"codec": "dense", "layer": 7, "dtype": "bfloat16",
              "reference_math": "cublaslt-no-splitk",
              "weights_sha256": module.sha256(folder / "experts.safetensors"),
              "inputs_sha256": module.sha256(folder / "inputs.safetensors")}
    (folder / "recipe.json").write_text(json.dumps(recipe))
    protocol = {"recipe_sha256": {"dense": module.sha256(folder / "recipe.json")}}
    module.validate_prepared(tmp_path, "dense", protocol, formal=True)
    (folder / "inputs.safetensors").write_bytes(b"tampered inputs")
    with pytest.raises(ValueError, match="hash differs"):
        module.validate_prepared(tmp_path, "dense", protocol, formal=True)


def test_all_samples_and_complete_abba_blocks_reproduce():
    module = _module()
    blocks = [[_arm(name) for name in ("A", "B", "B", "A")] for _ in range(5)]
    result = module.analyze(blocks, 2)
    assert result["blocks"] == 5
    assert result["mean_block_ratio"] == pytest.approx(0.8)
    assert result["d3_accepted"] is False


@pytest.mark.parametrize("defect", ["order", "incomplete", "unknown", "void", "source", "data",
                                    "smoke", "profile", "hash", "median", "nan", "zero"])
def test_replay_rejects_contamination_and_identity_or_sample_tampering(defect):
    module = _module()
    block = [_arm(name) for name in ("A", "B", "B", "A")]
    row = block[1]
    if defect == "order":
        row["arm"] = "A"
    elif defect == "incomplete":
        block.pop()
    elif defect in ("unknown", "void"):
        row["F1"]["outcome"] = defect
    elif defect == "source":
        row["source"]["git_head"] = "e" * 40
    elif defect == "data":
        row["recipe_sha256"] = "e" * 64
    elif defect == "smoke":
        row["formal_measurement"] = False
    elif defect == "profile":
        row["mode"] = "profile"
    elif defect == "hash":
        row["samples"][0]["output_sha256"] = "e" * 64
    elif defect == "median":
        row["median_seconds"] += 0.01
    else:
        row["samples"][0]["wall_seconds"] = float("nan") if defect == "nan" else 0
    with pytest.raises(ValueError):
        module.analyze([block], 2)


@pytest.mark.parametrize("failure", [None, "unknown", "budget"])
def test_driver_uses_full_pilot_and_no_replacements_or_budget_overrun(
    tmp_path, monkeypatch, failure,
):
    module = _module()
    monkeypatch.syspath_prepend(str(Path(module.__file__).parent))
    source = {"git_head": "a" * 40}
    monkeypatch.setattr(module, "frozen_source", lambda protocol: copy.deepcopy(source))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    args = SimpleNamespace(prepared=prepared, out=tmp_path / "session", library=Path("unused"))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        arm = command[command.index("--arm") + 1]
        codec = command[command.index("--codec") + 1]
        folder = Path(command[command.index("--out") + 1])
        folder.mkdir()
        row = _arm(arm, mode=command[2], codec=codec)
        if failure == "unknown" and len(calls) == 3:
            row["F1"]["outcome"] = "unknown"
        if failure == "budget":
            value = 0.01 if len(calls) % 8 < 4 else 100.0
            row["samples"] = [{"wall_seconds": value, "output_sha256": "d" * 64}] * 2
            row["median_seconds"] = statistics.median(x["wall_seconds"] for x in row["samples"])
        (folder / "result.json").write_text(json.dumps(row))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    status = module.parent(args, {"max_blocks": 10, "samples": 2})
    if failure is None:
        assert status == 0 and len(calls) == 3 * (5 * 4 + 2)
        assert sum(command[2] == "profile" for command in calls) == 6
    elif failure == "unknown":
        assert status == 1 and len(calls) == 3
    else:
        assert status == 1 and len(calls) == 20
    assert (prepared / ("attempt-" + source["git_head"] + ".json")).is_file()
    duplicate = SimpleNamespace(prepared=prepared, out=tmp_path / "duplicate",
                                library=Path("unused"))
    before = len(calls)
    with pytest.raises(FileExistsError):
        module.parent(duplicate, {"max_blocks": 10, "samples": 2})
    assert len(calls) == before
