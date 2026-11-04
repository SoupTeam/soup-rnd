"""CPU controls for prospective execution gates; no actual commits or GPU runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/harness"))
SPEC = importlib.util.spec_from_file_location(
    "d3_full_model_gate", ROOT / "benchmarks/harness/d3_full_model_gate.py",
)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


@pytest.fixture
def rule():
    value = json.loads(GATE.PROTOCOL.read_bytes())
    value.update(execution_enabled=True, gpu_uuid="GPU-test", software={"torch": "test"},
                 mapping_sha256="0" * 64, controls_sha256="1" * 64,
                 cublas_library_sha256="2" * 64, setup_archive_sha256="3" * 64,
                 bundle_sha256={codec: "4" * 64 for codec in GATE.CODECS})
    return value


def test_disabled_draft_never_consults_git_or_gpu(monkeypatch):
    monkeypatch.setattr(GATE.subprocess, "check_output", lambda *a, **k: pytest.fail("git called"))
    with pytest.raises(ValueError, match="disabled"):
        GATE.frozen({"execution_enabled": False})


@pytest.mark.parametrize("field", ["bundle_sha256", "mapping_sha256", "setup_archive_sha256"])
def test_incomplete_pins_never_reserve_an_attempt(rule, monkeypatch, field):
    rule[field] = {} if field == "bundle_sha256" else None
    monkeypatch.setattr(GATE.subprocess, "check_output", lambda *a, **k: pytest.fail("git called"))
    with pytest.raises(ValueError, match="freeze|incomplete"):
        GATE.frozen(rule)


def test_small_gpu_threshold_cannot_bypass_the_full_oracle_policy(rule):
    rule["minimum_reported_vram_bytes"] = 12 * 2**30
    with pytest.raises(ValueError, match="policy"):
        GATE.frozen(rule)


def test_omitting_executable_source_is_rejected(rule):
    rule["sources"] = [name for name in rule["sources"] if not name.endswith("full_model.py")]
    with pytest.raises(ValueError, match="policy"):
        GATE.frozen(rule)


@pytest.mark.parametrize("field,value", [
    ("batch", 2), ("tokens", 17), ("modes", ["grouped"]),
    ("weight_budget_bytes", 2**30), ("negative_control", "single_scalar"),
    ("native_olmoe_rows_max", 16),
])
def test_unimplemented_recipe_or_budget_change_is_rejected(rule, field, value):
    rule[field] = value
    with pytest.raises(ValueError, match="policy"):
        GATE.frozen(rule)


def test_dirty_source_refused_before_reading_committed_blobs(rule, monkeypatch):
    def git(args, **kwargs):
        assert args[3] == "status"
        return b"?? uncommitted.py\n"

    monkeypatch.setattr(GATE.subprocess, "check_output", git)
    with pytest.raises(ValueError, match="clean checkout"):
        GATE.frozen(rule)


def test_source_identity_normalizes_text_but_preserves_binary(rule, tmp_path, monkeypatch):
    monkeypatch.setattr(GATE, "ROOT", tmp_path)
    protocol = tmp_path / "protocol.json"
    monkeypatch.setattr(GATE, "PROTOCOL", protocol)
    blobs = {}
    for name in GATE.REQUIRED_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = b"binary\r\n\x00" if path.suffix == ".zip" else b"source\n"
        path.write_bytes(blob if path.suffix == ".zip" else blob.replace(b"\n", b"\r\n"))
        blobs[name] = blob
    setup = "benchmarks/protocols/d3-full-model/setup-evidence.zip"
    rule["setup_archive_sha256"] = hashlib.sha256(blobs[setup]).hexdigest()
    protocol.write_bytes(json.dumps(rule).encode())
    blobs["protocol.json"] = protocol.read_bytes()

    def git(args, **kwargs):
        command = args[3]
        if command == "status":
            return b""
        if command == "rev-parse":
            return b"a" * 40
        assert command == "show"
        return blobs[args[4].removeprefix("HEAD:")]

    monkeypatch.setattr(GATE.subprocess, "check_output", git)
    identity = GATE.frozen(rule)
    assert identity["git_blob_sha256"][setup] == hashlib.sha256(blobs[setup]).hexdigest()
    (tmp_path / setup).write_bytes(b"binary\n\x00")
    with pytest.raises(ValueError, match="committed bytes"):
        GATE.frozen(rule)


def traces():
    return [[{"layer": layer, "expert": 2, "token_routes": 3,
              "logical_bytes_read": 1024, "status": "ok"}] for layer in range(16)]


def test_each_layer_dispatch_and_single_read_counts_are_required():
    GATE.check_traces(traces(), mode="grouped", first_fetches=16)
    with pytest.raises(ValueError, match="exactly once"):
        GATE.check_traces(traces(), mode="grouped", first_fetches=17)
    with pytest.raises(ValueError, match="every trained"):
        GATE.check_traces(traces()[:-1], mode="grouped", first_fetches=15)
    bad = traces()
    bad[0].append(copy.deepcopy(bad[0][0]))
    with pytest.raises(ValueError, match="ascending"):
        GATE.check_traces(bad, mode="grouped", first_fetches=17)


@pytest.mark.parametrize("corrupt", [None, "source", "input_sha256", "F1", "mode"])
def test_native_oracle_requires_matching_session_source_input_and_validity(
    rule, tmp_path, corrupt,
):
    oracle = tmp_path / "dense-native"
    oracle.mkdir()
    args = argparse.Namespace(out=tmp_path / "dense-grouped", oracle=oracle, codec="dense")
    source = {"git_head": "a" * 40}
    row = {"status": "ok", "mode": "native", "codec": "dense", "source": source,
           "F1": {"outcome": "ok"}, "protocol_sha256": GATE.file_hash(GATE.PROTOCOL),
           "bundle_sha256": rule["bundle_sha256"]["dense"], "input_sha256": "8" * 64}
    if corrupt is not None:
        row[corrupt] = {"outcome": "void"} if corrupt == "F1" else "tampered"
    (oracle / "result.json").write_text(json.dumps(row), encoding="utf-8")
    if corrupt is None:
        assert GATE.oracle_record(args, source, rule, "8" * 64) == row
        args.oracle = tmp_path / "other-session" / "dense-native"
        with pytest.raises(ValueError, match="same declared session"):
            GATE.oracle_record(args, source, rule, "8" * 64)
    else:
        with pytest.raises(ValueError, match="does not match"):
            GATE.oracle_record(args, source, rule, "8" * 64)


def test_all_twelve_fresh_cells_and_single_attempt_guard(rule, tmp_path, monkeypatch):
    source = {"git_head": "a" * 40, "git_blob_sha256": {"runtime": "0" * 64}}
    monkeypatch.setattr(GATE, "frozen", lambda rule: source)
    args = argparse.Namespace(prepared=tmp_path / "prepared", out=tmp_path / "session",
                              library=tmp_path / "library.so")
    args.prepared.mkdir()
    calls = []

    def execute(command, **kwargs):
        mode = command[2]
        codec = command[command.index("--codec") + 1]
        out = Path(command[command.index("--out") + 1])
        out.mkdir()
        calls.append((codec, mode))
        row = {"status": "ok", "source": source, "F1": {"outcome": "ok"}, "mode": mode,
               "formal_execution": True,
               "codec": codec, "protocol_sha256": GATE.file_hash(GATE.PROTOCOL),
               "bundle_sha256": rule["bundle_sha256"][codec],
               "expected_sha256": "5" * 64, "negative_sha256": "6" * 64,
               "input_sha256": "7" * 64,
               "first_forward_gemm": {"grouped_matrices": 1, "native_olmoe_rows_max": 128}}
        (out / "result.json").write_text(json.dumps(row), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(GATE.subprocess, "run", execute)
    assert GATE.parent(args, rule) == 0
    assert calls == [(codec, mode) for codec in GATE.CODECS for mode in GATE.MODES]
    data = json.loads((args.out / "session.json").read_bytes())
    assert len(data["cells"]) == 12
    assert data["status"] == "full_forward_gate_complete_review_required"
    args.out = tmp_path / "another-name"
    with pytest.raises(FileExistsError):
        GATE.parent(args, rule)
    assert len(calls) == 12


def test_failed_child_stops_without_replacement(rule, tmp_path, monkeypatch):
    monkeypatch.setattr(GATE, "frozen", lambda rule: {"git_head": "b" * 40})
    args = argparse.Namespace(prepared=tmp_path / "prepared", out=tmp_path / "session",
                              library=tmp_path / "library.so")
    args.prepared.mkdir()
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(GATE.subprocess, "run", execute)
    assert GATE.parent(args, rule) == 1
    assert len(calls) == 1
    assert json.loads((args.out / "session.json").read_bytes())["status"] == "incomplete"
    assert (args.out / "dense-native.log").exists()


@pytest.mark.parametrize("stats", [None, {}, {"grouped_matrices": 0},
                                    {"grouped_matrices": 1, "native_olmoe_rows_max": 16}])
def test_native_only_execution_cannot_pass_a_grouped_full_model_cell(stats):
    with pytest.raises(ValueError, match="exercise grouped GEMM"):
        GATE.check_compute_coverage("grouped", stats)
    GATE.check_compute_coverage("native", stats)


def test_declared_positive_grouped_coverage_is_accepted():
    GATE.check_compute_coverage("grouped-resident", {
        "grouped_matrices": 1, "native_olmoe_rows_max": 128,
    })
