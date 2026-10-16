"""Exercise notebook validation seams locally, without installs, checkout, or CUDA."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "d2-fast-lora-kaggle.ipynb"


def _sources() -> dict[str, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


@pytest.fixture
def notebook_helpers(tmp_path: Path) -> dict:
    run = tmp_path / "d2-evidence" / "isolated-run"
    run.mkdir(parents=True)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    namespace = {
        "csv": csv,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "os": os,
        "stat": stat,
        "zipfile": zipfile,
        "re": re,
        "subprocess": subprocess,
        "sys": sys,
        "time": time,
        "uuid": uuid,
        "ET": ET,
        "datetime": datetime,
        "timezone": timezone,
        "Path": Path,
    }
    config = ast.parse(_sources()["configuration"])
    gates = next(
        node.value
        for node in config.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "GATES" for target in node.targets)
    )
    namespace.update(
        {
            "WORK": tmp_path,
            "RUN": run,
            "RUN_ID": run.name,
            "CHECKOUT": checkout,
            "PYTHON": sys.executable,
            "SOURCE_MODE": "git",
            "SEED": 792,
            "STEPS": 50,
            "WARMUP": 5,
            "REPEATS": 5,
            "TOKENS": 16,
            "GATES": ast.literal_eval(gates),
        }
    )
    for name in ("helpers", "evidence-helpers"):
        exec(compile(_sources()[name], name, "exec"), namespace)
    for name in ("cuda-preflight", "cuda-correctness", "timing"):
        tree = ast.parse(_sources()[name])
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(tree, name, "exec"), namespace)
    namespace["say"] = lambda message: None
    return namespace


@pytest.fixture
def source_zip(notebook_helpers: dict) -> tuple[Path, dict, dict[str, bytes]]:
    namespace = notebook_helpers
    kernels = ["fast_lora.py", "fast_lora_qkv.py", "fast_lora_mlp.py"]
    required = namespace.get(
        "REQUIRED_SOURCE_FILES",
        [
            "benchmarks/harness/fast_lora_probe.py",
            "benchmarks/gate-d2-fast-lora-rule.md",
            *("src/soup_cli/utils/" + name for name in kernels),
        ],
    )
    files = {name: b"# isolated transport fixture; never real repo output\n" for name in required}
    files["src/soup_cli/tokenizers.py"] = b"# legitimate source filename, not a credential\n"
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    manifest = {
        "format": "soup-d2-snapshot-v1",
        "base_revision": "9aa43bd71afe12d3724f196202f1140e7dc409dc",
        "source_sha256": hashes,
        "kernel_sha256": {name: hashes["src/soup_cli/utils/" + name] for name in kernels},
        "harness_sha256": hashes["benchmarks/harness/fast_lora_probe.py"],
        "decision_rule_sha256": hashes["benchmarks/gate-d2-fast-lora-rule.md"],
    }
    manifest["snapshot_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    path = namespace["WORK"] / "d2-source-isolated.zip"
    _write_source_zip(path, manifest, files)
    return path, manifest, files


def _write_source_zip(path: Path, manifest: dict, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("D2-SNAPSHOT.json", json.dumps(manifest))
        for name, data in files.items():
            bundle.writestr(name, data)


def test_private_source_zip_verifies_all_bytes_into_a_fresh_directory(
    notebook_helpers: dict,
    source_zip: tuple[Path, dict, dict[str, bytes]],
) -> None:
    path, manifest, files = source_zip
    namespace = notebook_helpers
    assert "extract_source_zip" in namespace, "Private frozen source ZIP support is missing"
    target = namespace["WORK"] / "new-source"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = namespace["extract_source_zip"](path, target, digest)
    assert result["snapshot"] == manifest
    assert result["zip_sha256"] == digest
    assert result["git_state"] == "NO_GIT_SNAPSHOT"
    assert {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} == (
        set(files) | {"D2-SNAPSHOT.json"}
    )
    for name, data in files.items():
        assert (target / name).read_bytes() == data
    assert not (target / ".git").exists()


def _archive(namespace: dict) -> None:
    tree = ast.parse(_sources()["archive"])
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "display"
        )
    ]
    exec(compile(tree, "archive", "exec"), namespace)


@pytest.mark.requires_symlink
@pytest.mark.parametrize("kind", ["external-file", "internal-file", "directory", "broken", "root"])
def test_archive_rejects_all_symlinks_before_hashing(
    notebook_helpers: dict,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    namespace = notebook_helpers
    run = namespace["RUN"]
    target = namespace["WORK"] / "isolated-external"
    target.mkdir()
    fixture = target / "fixture.txt"
    fixture.write_text("isolated fixture; not a credential", encoding="utf-8")
    if kind == "root":
        link = namespace["WORK"] / "linked-run"
        link.symlink_to(run, target_is_directory=True)
        namespace["RUN"] = link
    elif kind == "directory":
        (run / "linked-directory").symlink_to(target, target_is_directory=True)
    else:
        if kind == "internal-file":
            fixture = run / "internal.txt"
            fixture.write_text("internal fixture", encoding="utf-8")
        elif kind == "broken":
            fixture = target / "absent.txt"
        (run / "linked.txt").symlink_to(fixture)
    hashed = []
    original_hash = hashlib.sha256

    def tracked_hash(value):
        hashed.append(value)
        return original_hash(value)

    monkeypatch.setattr(hashlib, "sha256", tracked_hash)
    with pytest.raises(RuntimeError, match="(?i)unsafe|link|contain|traversal"):
        _archive(namespace)
    assert hashed == []
    assert not list(namespace["WORK"].glob("*.zip"))


def test_archive_rejects_reparse_directory_before_hashing(
    notebook_helpers: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = notebook_helpers["RUN"]
    junction = run / "junction"
    junction.mkdir()
    (junction / "fixture.txt").write_text("isolated fixture", encoding="utf-8")
    original = os.lstat

    def reparse_lstat(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if os.fspath(path) == str(junction):
            return SimpleNamespace(st_mode=result.st_mode,
                                   st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return result

    monkeypatch.setattr(os, "lstat", reparse_lstat)
    with pytest.raises(RuntimeError, match="(?i)unsafe|link|reparse"):
        _archive(notebook_helpers)
    assert not list(notebook_helpers["WORK"].glob("*.zip"))


@pytest.mark.requires_hardlink
def test_archive_rejects_hardlinked_artifact(notebook_helpers: dict) -> None:
    external = notebook_helpers["WORK"] / "isolated-fixture.txt"
    external.write_text("isolated fixture", encoding="utf-8")
    os.link(external, notebook_helpers["RUN"] / "linked.txt")
    with pytest.raises(RuntimeError, match="(?i)unsafe|link"):
        _archive(notebook_helpers)
    assert not list(notebook_helpers["WORK"].glob("*.zip"))


def test_archive_rejects_parent_traversal_root(notebook_helpers: dict) -> None:
    child = notebook_helpers["RUN"] / "child"
    child.mkdir()
    notebook_helpers["RUN"] = child / ".."
    with pytest.raises(RuntimeError, match="(?i)unsafe|traversal"):
        _archive(notebook_helpers)
    assert not list(notebook_helpers["WORK"].glob("*.zip"))


def test_archive_rejects_out_of_root_realpath(
    notebook_helpers: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = notebook_helpers["RUN"]
    artifact = run / "fixture.txt"
    artifact.write_text("isolated fixture", encoding="utf-8")
    original = os.path.realpath

    def outside_realpath(path, *args, **kwargs):
        if os.fspath(path) == str(artifact):
            return str(notebook_helpers["WORK"] / "external.txt")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", outside_realpath)
    with pytest.raises(RuntimeError, match="(?i)unsafe|contain|outside"):
        _archive(notebook_helpers)
    assert not list(notebook_helpers["WORK"].glob("*.zip"))


def test_archive_contains_only_current_run_with_verified_hashes(notebook_helpers: dict) -> None:
    run = notebook_helpers["RUN"]
    (run / "nested").mkdir()
    (run / "nested" / "stdout.log").write_text("raw evidence", encoding="utf-8")
    (notebook_helpers["WORK"] / "excluded.txt").write_text("excluded", encoding="utf-8")
    _archive(notebook_helpers)
    with zipfile.ZipFile(notebook_helpers["archive"]) as archive:
        files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    prefix = notebook_helpers["RUN_ID"] + "/"
    assert all(name.startswith(prefix) for name in files)
    manifest_name = prefix + "artifact-sha256.json"
    manifest = json.loads(files[manifest_name])
    assert set(files) == {prefix + name for name in manifest} | {manifest_name}
    for relative, digest in manifest.items():
        assert hashlib.sha256(files[prefix + relative]).hexdigest() == digest
    assert files[prefix + "nested/stdout.log"] == b"raw evidence"
    assert not any("excluded.txt" in name for name in files)


def _parity_report(namespace: dict, request_dx: bool = True) -> dict:
    rows = []
    execution = {}
    for path, projections in namespace["PROJECTIONS"].items():
        outputs = [f"Y/{projection}" for projection in projections] if path == "qkv" else ["Y/Y"]
        quantities = [("forward", output) for output in outputs]
        quantities += [
            ("backward", f"{gradient}/{projection}")
            for gradient in ("dA", "dB")
            for projection in projections
        ]
        if request_dx:
            quantities.append(("backward", "dX"))
        rows += [
            {"path": path, "phase": phase, "quantity": quantity, "finite": True, "passed": True}
            for phase, quantity in quantities
        ]
        execution[path] = [namespace["EXPECTED_FUNCTIONS"][path]] * len(outputs)
    return {
        "rows": rows,
        "request_dX": request_dx,
        "execution": execution,
        "patch_counts": {path: 1 for path in namespace["PROJECTIONS"]},
        "negative_control": {
            path: {"changed_adapter_detected": True} for path in namespace["PROJECTIONS"]
        },
    }


@pytest.mark.parametrize("request_dx", [True, False])
@pytest.mark.parametrize("path", ["single", "qkv", "mlp"])
def test_parity_rejects_wrong_forward_quantity_set(
    notebook_helpers: dict,
    request_dx: bool,
    path: str,
) -> None:
    report = _parity_report(notebook_helpers, request_dx)
    for row in report["rows"]:
        if row["path"] == path and row["phase"] == "forward":
            row["quantity"] = "Y/q_proj" if path == "qkv" else "Y/not-an-output"
    with pytest.raises(RuntimeError, match="(?i)forward|output|quantity"):
        notebook_helpers["validate_parity"](report, request_dx)


@pytest.mark.parametrize("request_dx", [True, False])
def test_parity_accepts_exact_evidence(notebook_helpers: dict, request_dx: bool) -> None:
    notebook_helpers["validate_parity"](_parity_report(notebook_helpers, request_dx), request_dx)


@pytest.mark.parametrize("path", ["single", "qkv", "mlp"])
@pytest.mark.parametrize("request_dx", [True, False])
def test_parity_rejects_empty_custom_execution(
    notebook_helpers: dict,
    path: str,
    request_dx: bool,
) -> None:
    report = _parity_report(notebook_helpers, request_dx)
    report["execution"][path] = []
    with pytest.raises(RuntimeError, match="(?i)execution|fast path|autograd"):
        notebook_helpers["validate_parity"](report, request_dx)


def test_parity_rejects_partial_qkv_custom_execution(notebook_helpers: dict) -> None:
    report = _parity_report(notebook_helpers)
    report["execution"]["qkv"] = report["execution"]["qkv"][:1]
    with pytest.raises(RuntimeError, match="(?i)execution|fast path|autograd"):
        notebook_helpers["validate_parity"](report, True)


@pytest.mark.parametrize("path", ["single", "qkv", "mlp"])
def test_parity_rejects_wrong_custom_node(notebook_helpers: dict, path: str) -> None:
    report = _parity_report(notebook_helpers)
    report["execution"][path][0] = "AddBackward0"
    with pytest.raises(RuntimeError, match="(?i)execution|fast path|autograd"):
        notebook_helpers["validate_parity"](report, True)


@pytest.fixture
def probe_report(notebook_helpers: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    namespace = notebook_helpers
    checkout = namespace["CHECKOUT"]
    namespace["HARNESS"] = checkout / "benchmarks/harness/fast_lora_probe.py"
    namespace["RULE"] = checkout / "benchmarks/gate-d2-fast-lora-rule.md"
    filenames = ["fast_lora.py", "fast_lora_qkv.py", "fast_lora_mlp.py"]
    relatives = ["src/soup_cli/utils/" + name for name in filenames]
    relatives += ["benchmarks/harness/fast_lora_probe.py", "benchmarks/gate-d2-fast-lora-rule.md"]
    hashes = {}
    for relative in relatives:
        fixture = checkout / relative
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text("isolated source fixture: " + relative, encoding="utf-8")
        hashes[relative] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    for source in (namespace["RULE"], namespace["HARNESS"]):
        hashes[str(source.relative_to(checkout))] = hashlib.sha256(source.read_bytes()).hexdigest()
    namespace["MANIFEST"] = {"files_sha256": hashes}
    namespace["REMOTE_HEAD"] = "a" * 40
    report = _parity_report(namespace)
    report.update(
        {
            "passed": True,
            "mode": "parity",
            "device": "cpu",
            "dtype": "fp32",
            "seed": 792,
            "fixture": "SYNTHETIC",
            "environment": {
                "git_head": namespace["REMOTE_HEAD"],
                "git_status": "",
                "decision_rule_sha256": hashes[relatives[-1]],
                "harness_sha256": hashes[relatives[-2]],
                "kernel_sha256": {name: hashes["src/soup_cli/utils/" + name] for name in filenames},
            },
        }
    )

    def emit_fixture(label, argv, **kwargs):
        prefix = Path(argv[argv.index("--output-prefix") + 1])
        Path(str(prefix) + ".json").write_text(json.dumps(report), encoding="utf-8")
        with Path(str(prefix) + ".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
            writer.writeheader()
            writer.writerows(report["rows"])
        return {"returncode": 0}

    monkeypatch.setitem(namespace, "run_logged", emit_fixture)
    return report


@pytest.mark.parametrize(
    "keys",
    [
        [],
        ["fast_lora.py"],
        ["fast_lora.py", "fast_lora_qkv.py"],
        ["fast_lora.py", "fast_lora_qkv.py", "fast_lora_mlp.py", "extra.py"],
    ],
)
def test_probe_rejects_incomplete_kernel_fingerprint_set(
    notebook_helpers: dict,
    probe_report: dict,
    keys: list[str],
) -> None:
    environment = probe_report["environment"]
    environment["kernel_sha256"] = {
        key: environment["kernel_sha256"].get(key, "0" * 64) for key in keys
    }
    with pytest.raises(RuntimeError, match="(?i)kernel.*(hash|fingerprint)|fingerprint.*kernel"):
        notebook_helpers["probe"]("isolated-parity", "parity", "cpu", "fp32")


@pytest.mark.parametrize("value", [None, "0" * 64])
def test_probe_rejects_missing_or_mismatched_harness_fingerprint(
    notebook_helpers: dict,
    probe_report: dict,
    value: str | None,
) -> None:
    if value is None:
        del probe_report["environment"]["harness_sha256"]
    else:
        probe_report["environment"]["harness_sha256"] = value
    with pytest.raises(RuntimeError, match="(?i)harness.*(hash|fingerprint)"):
        notebook_helpers["probe"]("isolated-parity", "parity", "cpu", "fp32")


def test_probe_accepts_exact_fingerprints(notebook_helpers: dict, probe_report: dict) -> None:
    assert notebook_helpers["probe"]("isolated-parity", "parity", "cpu", "fp32") == probe_report


CORRECTNESS_ORDER = [
    "CPU_GRADCHECK",
    "CPU_PARITY",
    "CPU_REGRESSIONS",
    "CPU_NF4_CHECKPOINT",
    "CPU_MIXED_PRECISION",
    "CPU_REFERENCE_PRECISION",
    "CUDA_PREFLIGHT",
    "CUDA_DENSE_PARITY",
    "CUDA_LOSS_50",
    "CUDA_NF4_SINGLE_CHECKPOINT_FP16",
]
TIMING_GATES = ["CUDA_TINY_TIMING", "CUDA_8B_SHAPED_BLOCK"]


def _passed_gates(namespace: dict) -> None:
    for name in CORRECTNESS_ORDER:
        namespace["GATES"][name] = {"status": "PASS"}
    for name in TIMING_GATES:
        namespace["GATES"][name] = {"status": "COLLECTED_NO_VERDICT"}


@pytest.mark.parametrize("upstream", CORRECTNESS_ORDER)
@pytest.mark.parametrize("fails", [False, True])
def test_upstream_retry_invalidates_all_descendants_before_action(
    notebook_helpers: dict,
    upstream: str,
    fails: bool,
) -> None:
    namespace = notebook_helpers
    _passed_gates(namespace)
    descendants = CORRECTNESS_ORDER[CORRECTNESS_ORDER.index(upstream) + 1 :] + TIMING_GATES
    observed = []

    def retry():
        observed.extend(namespace["GATES"][name]["status"] for name in descendants)
        if fails:
            raise RuntimeError("isolated retry failure")
        return "actual retry result"

    if fails:
        with pytest.raises(RuntimeError, match="isolated retry failure"):
            namespace["gate"](upstream, retry)
    else:
        assert namespace["gate"](upstream, retry) == "actual retry result"
    assert observed == ["UNVERIFIED"] * len(descendants)
    for name in descendants:
        assert namespace["GATES"][name]["status"] == "UNVERIFIED"
        assert namespace["GATES"][name]["history"][-1]["status"] in {
            "PASS",
            "COLLECTED_NO_VERDICT",
        }
    saved = json.loads((namespace["RUN"] / "gate_status.json").read_text(encoding="utf-8"))
    assert saved == namespace["GATES"]


@pytest.mark.parametrize("name", CORRECTNESS_ORDER[1:])
def test_gate_rechecks_upstream_before_running_action(notebook_helpers: dict, name: str) -> None:
    namespace = notebook_helpers
    _passed_gates(namespace)
    upstream = CORRECTNESS_ORDER[CORRECTNESS_ORDER.index(name) - 1]
    namespace["GATES"][upstream] = {"status": "FAIL", "error": "isolated upstream failure"}
    calls = []
    with pytest.raises(namespace["UnverifiedGateError"], match=upstream):
        namespace["gate"](name, lambda: calls.append("action"))
    assert calls == []
    assert namespace["GATES"][name]["status"] == "UNVERIFIED"


@pytest.mark.parametrize("cpu", CORRECTNESS_ORDER[:6])
@pytest.mark.parametrize(
    "stage",
    ["cuda_preflight", "cuda_dense_parity", "cuda_loss", "cuda_nf4_checkpoint", "collect_timing"],
)
def test_every_cuda_entry_rechecks_all_cpu_gates_even_with_stale_passes(
    notebook_helpers: dict,
    monkeypatch: pytest.MonkeyPatch,
    cpu: str,
    stage: str,
) -> None:
    namespace = notebook_helpers
    _passed_gates(namespace)
    namespace["GATES"][cpu] = {"status": "FAIL", "error": "isolated CPU failure"}
    calls = []

    def forbidden_cuda(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("CUDA work reached with a failed CPU prerequisite")

    for helper in ("probe", "smi_snapshot", "run_pytest", "run_logged"):
        monkeypatch.setitem(namespace, helper, forbidden_cuda)
    monkeypatch.setitem(namespace, "assert_checkout", lambda: {})
    with pytest.raises(namespace["UnverifiedGateError"], match=cpu):
        if stage == "collect_timing":
            namespace[stage]("CUDA_TINY_TIMING", "tiny", ("fp32", "fp16"))
        else:
            namespace[stage]()
    assert calls == []


def test_failed_gate_history_survives_successful_retry(notebook_helpers: dict) -> None:
    namespace = notebook_helpers

    def failure():
        raise RuntimeError("original raw failure")

    with pytest.raises(RuntimeError, match="original raw failure"):
        namespace["gate"]("CPU_GRADCHECK", failure)
    namespace["gate"]("CPU_GRADCHECK", lambda: "successful retry")
    assert namespace["GATES"]["CPU_GRADCHECK"]["status"] == "PASS"
    history = namespace["GATES"]["CPU_GRADCHECK"]["history"]
    assert any(item.get("error") == "RuntimeError: original raw failure" for item in history)
    assert any(item["status"] == "FAIL" for item in history)


def test_timing_retry_preserves_original_failure(
    notebook_helpers: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = notebook_helpers
    _passed_gates(namespace)
    monkeypatch.setitem(namespace, "assert_checkout", lambda: {})
    monkeypatch.setitem(namespace, "smi_snapshot", lambda label: {})

    def failure(*args, **kwargs):
        raise RuntimeError("original raw timing failure")

    monkeypatch.setitem(namespace, "probe", failure)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="original raw timing failure"):
            namespace["collect_timing"]("CUDA_TINY_TIMING", "tiny", ("fp32",))
    status = namespace["GATES"]["CUDA_TINY_TIMING"]
    assert status["status"] == "FAIL"
    assert any(
        item.get("error") == "RuntimeError: original raw timing failure"
        for item in status["history"]
    )


def test_archive_preserves_interrupted_gate_history(notebook_helpers: dict) -> None:
    history = [{"status": "FAIL", "error": "original raw failure"}]
    notebook_helpers["GATES"]["CPU_GRADCHECK"] = {"status": "RUNNING", "history": history}
    _archive(notebook_helpers)
    status = notebook_helpers["GATES"]["CPU_GRADCHECK"]
    assert status["status"] == "UNVERIFIED"
    assert history[0] in status["history"]
    assert status["history"][-1]["status"] == "RUNNING"


@pytest.mark.parametrize(
    "xml,code,expected_failed",
    [
        (
            '<testsuite><testcase name="isolated"><failure message="raw failure"/>'
            "</testcase></testsuite>",
            1,
            1,
        ),
        (
            '<testsuite><testcase name="isolated"><error message="raw error"/>'
            "</testcase></testsuite>",
            2,
            1,
        ),
        ('<testsuite><testcase name="isolated"/></testsuite>', 7, 0),
    ],
)
def test_failed_pytest_preserves_parsed_summary_before_raising(
    notebook_helpers: dict,
    monkeypatch: pytest.MonkeyPatch,
    xml: str,
    code: int,
    expected_failed: int,
) -> None:
    namespace = notebook_helpers

    def command(argv, **kwargs):
        junit = Path(argv[argv.index("--junitxml") + 1])
        junit.write_text(xml, encoding="utf-8")
        kwargs["stdout"].write(b"isolated raw stdout")
        kwargs["stderr"].write(b"isolated raw stderr")
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(subprocess, "run", command)
    with pytest.raises(RuntimeError):
        namespace["gate"](
            "CPU_GRADCHECK",
            lambda: namespace["run_pytest"](
                "isolated-pytest",
                ["isolated.py"],
                device="cpu",
                expression="not gpu and not smoke",
            ),
        )
    directory = next(namespace["RUN"].glob("isolated-pytest-*"))
    assert (directory / "stdout.log").read_bytes() == b"isolated raw stdout"
    assert (directory / "stderr.log").read_bytes() == b"isolated raw stderr"
    command_record = json.loads((directory / "command.json").read_text(encoding="utf-8"))
    assert command_record["returncode"] == code
    summary = json.loads((directory / "pytest-summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == 1
    assert len(summary["failed"]) == expected_failed
    assert summary["returncode"] == code
    assert namespace["GATES"]["CPU_GRADCHECK"]["status"] == "FAIL"


@pytest.mark.parametrize(
    "xml,code", [(None, 0), (None, 5), ("<testsuite/>", 0), ("<testsuite/>", 5), ("<invalid", 0)]
)
def test_unavailable_or_empty_junit_keeps_diagnostic_summary_and_blocks_pass(
    notebook_helpers: dict,
    monkeypatch: pytest.MonkeyPatch,
    xml: str | None,
    code: int,
) -> None:
    namespace = notebook_helpers

    def command(argv, **kwargs):
        if xml is not None:
            Path(argv[argv.index("--junitxml") + 1]).write_text(xml, encoding="utf-8")
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(subprocess, "run", command)
    with pytest.raises(RuntimeError):
        namespace["gate"](
            "CPU_GRADCHECK",
            lambda: namespace["run_pytest"](
                "isolated-empty",
                ["isolated.py"],
                device="cpu",
                expression="not gpu and not smoke",
            ),
        )
    directory = next(namespace["RUN"].glob("isolated-empty-*"))
    summary = json.loads((directory / "pytest-summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == 0
    assert summary["returncode"] == code
    assert namespace["GATES"]["CPU_GRADCHECK"]["status"] != "PASS"


def test_actual_failed_pytest_keeps_junit_case_summary(notebook_helpers: dict) -> None:
    namespace = notebook_helpers
    fixture = namespace["CHECKOUT"] / "test_isolated_failure.py"
    fixture.write_text(
        'def test_isolated_failure():\n    assert False, "isolated failure"\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError):
        namespace["gate"](
            "CPU_GRADCHECK",
            lambda: namespace["run_pytest"](
                "actual-failed-pytest",
                [str(fixture)],
                device="cpu",
                expression="not gpu and not smoke",
            ),
        )
    directory = next(namespace["RUN"].glob("actual-failed-pytest-*"))
    summary = json.loads((directory / "pytest-summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == 1
    assert summary["failed"][0]["name"] == "test_isolated_failure"
    assert summary["returncode"] == 1
    assert "isolated failure" in (directory / "stdout.log").read_text(encoding="utf-8")
    assert namespace["GATES"]["CPU_GRADCHECK"]["status"] == "FAIL"


@pytest.mark.requires_symlink
def test_archive_rejects_real_directory_redirect(notebook_helpers: dict) -> None:
    namespace = notebook_helpers
    target = namespace["WORK"] / "isolated-junction-target"
    target.mkdir()
    (target / "fixture.txt").write_text("isolated fixture", encoding="utf-8")
    redirect = namespace["RUN"] / "directory-redirect"
    if os.name == "nt":
        result = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(redirect), str(target)],
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
    else:
        redirect.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="(?i)unsafe|link|reparse|contain"):
        _archive(namespace)
    assert not list(namespace["WORK"].glob("*.zip"))


def test_failed_cpu_retry_blocks_previously_passed_cuda_parity(
    notebook_helpers: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = notebook_helpers
    _passed_gates(namespace)
    calls = []
    monkeypatch.setitem(namespace, "assert_checkout", lambda: {})
    monkeypatch.setitem(namespace, "smi_snapshot", lambda label: {})
    monkeypatch.setitem(namespace, "probe", lambda *args, **kwargs: calls.append(args))

    def failure():
        raise RuntimeError("isolated CPU regression failure")

    with pytest.raises(RuntimeError, match="isolated CPU regression failure"):
        namespace["gate"]("CPU_REGRESSIONS", failure)
    with pytest.raises(namespace["UnverifiedGateError"]):
        namespace["cuda_dense_parity"]()
    assert calls == []
    assert namespace["GATES"]["CUDA_PREFLIGHT"]["status"] == "UNVERIFIED"
    assert namespace["GATES"]["CPU_REGRESSIONS"]["error"] == (
        "RuntimeError: isolated CPU regression failure"
    )


@pytest.mark.parametrize(
    "name",
    [
        "../outside.py",
        "/absolute.py",
        "//server/share.py",
        "C:/drive.py",
        "src\\soup_cli\\bad.py",
        "src/soup_cli/../bad.py",
        "src/soup_cli/./bad.py",
        "src/soup_cli/bad.py:stream",
        "source/src/soup_cli/wrapped.py",
        "evidence/raw.json",
        "src/soup_cli/.git/config",
        "src/soup_cli/.env",
        "src/soup_cli/.env.production",
        "src/soup_cli/kaggle.json",
        "src/soup_cli/credentials.json",
        "src/soup_cli/id_rsa",
        "src/soup_cli/hf_token",
        "src/soup_cli/token",
        "src/soup_cli/private.key",
        "src/soup_cli/NUL.py",
        "src/soup_cli/trailing./bad.py",
        "tests/unselected_test.py",
    ],
)
def test_source_zip_rejects_unsafe_or_nonallowlisted_members_before_extracting(
    notebook_helpers: dict,
    source_zip: tuple,
    name: str,
) -> None:
    path, manifest, files = source_zip
    # Include even attacker-consistent manifest entries: allowlist is independent.
    files[name] = b"# isolated attack fixture"
    manifest["source_sha256"][name] = hashlib.sha256(files[name]).hexdigest()
    _write_source_zip(path, manifest, files)
    target = notebook_helpers["WORK"] / "fresh-reject"
    with pytest.raises(RuntimeError, match="(?i)unsafe|inventory|member|source|snapshot"):
        notebook_helpers["extract_source_zip"](path, target)
    assert not target.exists()
    assert not (notebook_helpers["WORK"] / "outside.py").exists()


@pytest.mark.parametrize(
    "stem",
    [
        "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
        "COM1", "COM9", "LPT1", "LPT9",
        "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³",
    ],
)
@pytest.mark.parametrize("suffix", [".py", " .py", ".py/child.py"])
def test_source_member_rejects_portable_windows_device_stems(
    notebook_helpers: dict,
    stem: str,
    suffix: str,
) -> None:
    name = "src/soup_cli/" + stem.swapcase() + suffix
    with pytest.raises(RuntimeError, match="Unsafe source ZIP member"):
        notebook_helpers["source_member_name"](name)


@pytest.mark.parametrize("character", ["<", ">", "?", "*", "|"])
@pytest.mark.parametrize("suffix", [".py", "/child.py"])
def test_source_member_rejects_portable_windows_illegal_characters(
    notebook_helpers: dict,
    character: str,
    suffix: str,
) -> None:
    name = "src/soup_cli/question" + character + suffix
    with pytest.raises(RuntimeError, match="Unsafe source ZIP member"):
        notebook_helpers["source_member_name"](name)


@pytest.mark.parametrize(
    "name", ['src/soup_cli/quote"name.py', 'src/soup_cli/quote"directory/child.py']
)
def test_source_member_rejects_windows_double_quote(
    notebook_helpers: dict,
    name: str,
) -> None:
    with pytest.raises(RuntimeError, match="Unsafe source ZIP member"):
        notebook_helpers["source_member_name"](name)


def _add_source_fixture_member(source_zip: tuple, name: str) -> None:
    path, manifest, files = source_zip
    files[name] = b"# isolated filename-policy fixture; not repository output\n"
    manifest["source_sha256"][name] = hashlib.sha256(files[name]).hexdigest()
    canonical = {key: value for key, value in manifest.items() if key != "snapshot_id"}
    manifest["snapshot_id"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    _write_source_zip(path, manifest, files)


@pytest.mark.parametrize(
    "name",
    [
        "src/soup_cli/COM¹.py",
        "src/soup_cli/CONIN$.py",
        "src/soup_cli/question?.py",
        "src/soup_cli/pipe|name.py",
        "src/soup_cli/CON .py",
        "src/soup_cli/LPT² .py",
        "src/soup_cli/CONOUT$.py",
    ],
)
def test_source_zip_rejects_windows_reserved_names_before_filesystem_access(
    notebook_helpers: dict,
    source_zip: tuple,
    name: str,
) -> None:
    # Fully consistent hashes and snapshot ID must not bypass portable name validation.
    _add_source_fixture_member(source_zip, name)
    target = notebook_helpers["WORK"] / "windows-name-reject"
    with pytest.raises(RuntimeError, match="Unsafe source ZIP member"):
        notebook_helpers["extract_source_zip"](source_zip[0], target)
    assert not target.exists()


@pytest.mark.parametrize(
    "name",
    [
        "src/soup_cli/модель.py",
        "src/soup_cli/café.py",
        "src/soup_cli/模型.py",
        "src/soup_cli/COM10.py",
        "src/soup_cli/COM⁴.py",
        "src/soup_cli/CONIN.py",
        "src/soup_cli/reCON.py",
        "src/soup_cli/question？.py",
        "src/soup_cli/pipe｜name.py",
    ],
)
def test_source_zip_preserves_legitimate_unicode_names_without_normalizing(
    notebook_helpers: dict,
    source_zip: tuple,
    name: str,
) -> None:
    assert notebook_helpers["source_member_name"](name) == name
    _add_source_fixture_member(source_zip, name)
    target = notebook_helpers["WORK"] / "unicode-name-source"
    info = notebook_helpers["extract_source_zip"](source_zip[0], target)
    assert info["snapshot"] == source_zip[1]
    assert (target / name).read_bytes() == source_zip[2][name]
    notebook_helpers["verify_source_tree"](target, info)


@pytest.mark.parametrize(
    "kind", ["exact", "case", "nul", "symlink", "fifo", "reparse", "extra-directory", "file-parent"]
)
def test_source_zip_rejects_duplicate_alias_and_nonregular_members(
    notebook_helpers: dict,
    source_zip: tuple,
    kind: str,
) -> None:
    path, _, _ = source_zip
    name = "src/soup_cli/tokenizers.py"
    with zipfile.ZipFile(path, "a") as bundle:
        if kind in {"exact", "case", "nul"}:
            member = (
                name
                if kind == "exact"
                else name.upper()
                if kind == "case"
                else ("src/soup_cli/evilX.py")
            )
            with pytest.warns(UserWarning) if kind == "exact" else _no_warning_context():
                bundle.writestr(member, b"# alias fixture")
        else:
            entry = zipfile.ZipInfo(
                "unrelated/"
                if kind == "extra-directory"
                else (
                    "src/soup_cli/tokenizers.py/child.py"
                    if kind == "file-parent"
                    else "src/soup_cli/x.py"
                )
            )
            entry.create_system = 3
            entry.external_attr = (
                (
                    stat.S_IFLNK
                    if kind == "symlink"
                    else stat.S_IFIFO
                    if kind == "fifo"
                    else stat.S_IFREG
                )
                | 0o600
            ) << 16
            if kind == "reparse":
                entry.external_attr |= 0x400
            bundle.writestr(entry, b"isolated fixture")
    if kind == "nul":
        path.write_bytes(path.read_bytes().replace(b"evilX.py", b"evil\0.py"))
    target = notebook_helpers["WORK"] / "reject-special"
    with pytest.raises(RuntimeError, match="(?i)unsafe|duplicate|member|inventory|source"):
        notebook_helpers["extract_source_zip"](path, target)
    assert not target.exists()


def _no_warning_context():
    from contextlib import nullcontext

    return nullcontext()


@pytest.mark.parametrize(
    "fault",
    [
        "format",
        "base_revision",
        "snapshot_id",
        "kernel_sha256",
        "harness_sha256",
        "decision_rule_sha256",
        "unknown",
        "source_digest",
        "source_digest_type",
        "missing-required",
    ],
)
def test_source_zip_requires_the_exact_manifest_contract(
    notebook_helpers: dict, source_zip: tuple, fault: str
) -> None:
    path, manifest, files = source_zip
    if fault == "unknown":
        manifest["claimed_gpu_pass"] = True
    elif fault == "source_digest":
        manifest["source_sha256"]["LICENSE"] = "0" * 64
    elif fault == "source_digest_type":
        manifest["source_sha256"]["LICENSE"] = None
    elif fault == "missing-required":
        del manifest["source_sha256"]["tests/conftest.py"]
        del files["tests/conftest.py"]
    else:
        manifest[fault] = "invalid"
    _write_source_zip(path, manifest, files)
    target = notebook_helpers["WORK"] / "reject-manifest"
    with pytest.raises(RuntimeError, match="(?i)manifest|snapshot|SHA256|required"):
        notebook_helpers["extract_source_zip"](path, target)
    assert not target.exists()


@pytest.mark.parametrize("expected", ["1" * 64, "A" * 64, "short", " " + "0" * 64])
def test_source_zip_rejects_wrong_or_malformed_operator_hash(
    notebook_helpers: dict,
    source_zip: tuple,
    expected: str,
) -> None:
    target = notebook_helpers["WORK"] / "reject-sha"
    with pytest.raises(RuntimeError, match="SHA256"):
        notebook_helpers["extract_source_zip"](source_zip[0], target, expected)
    assert not target.exists()


def test_source_zip_never_reuses_or_changes_existing_work(
    notebook_helpers: dict,
    source_zip: tuple,
) -> None:
    target = notebook_helpers["WORK"] / "old-private-work"
    target.mkdir()
    original = target / "old.txt"
    original.write_bytes(b"isolated private old work fixture")
    with pytest.raises((RuntimeError, FileExistsError)):
        notebook_helpers["extract_source_zip"](source_zip[0], target)
    assert original.read_bytes() == b"isolated private old work fixture"
    assert list(target.iterdir()) == [original]


@pytest.mark.parametrize("matches", [0, 2])
def test_source_zip_discovery_never_selects_missing_or_multiple_inputs(
    notebook_helpers: dict, matches: int
) -> None:
    root = notebook_helpers["WORK"] / "private-inputs"
    root.mkdir()
    for index in range(matches):
        (root / f"d2-source-{index}.zip").write_bytes(b"isolated nonexecuted fixture")
    with pytest.raises(RuntimeError, match="exactly one"):
        notebook_helpers["select_source_zip"]("", root)


def test_source_zip_explicit_path_and_unique_discovery_are_contained(notebook_helpers: dict):
    root = notebook_helpers["WORK"] / "private-inputs"
    root.mkdir()
    bundle = root / "d2-source-unique.zip"
    bundle.write_bytes(b"isolated nonexecuted fixture")
    select = notebook_helpers["select_source_zip"]
    assert select("", root) == bundle
    assert select(str(bundle), root) == bundle
    outside = notebook_helpers["WORK"] / "outside.zip"
    outside.write_bytes(b"isolated nonexecuted fixture")
    with pytest.raises(RuntimeError, match="(?i)contain|outside|unsafe"):
        select(str(outside), root)


@pytest.mark.parametrize("fault", ["modified", "extra", "manifest"])
def test_frozen_extracted_source_is_revalidated_before_reuse(
    notebook_helpers: dict, source_zip: tuple, fault: str
) -> None:
    namespace = notebook_helpers
    target = namespace["WORK"] / "frozen-source"
    namespace["CHECKOUT_INFO"] = namespace["extract_source_zip"](source_zip[0], target)
    namespace["CHECKOUT"] = target
    namespace["SOURCE_MODE"] = "zip"
    assert namespace["assert_checkout"]()["git_state"] == "NO_GIT_SNAPSHOT"
    if fault == "modified":
        (target / "LICENSE").write_bytes(b"isolated mutation")
    elif fault == "extra":
        (target / "unmanifested.py").write_bytes(b"isolated mutation")
    else:
        (target / "D2-SNAPSHOT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="(?i)snapshot|source|inventory|manifest"):
        namespace["assert_checkout"]()


@pytest.mark.parametrize("field", ["git_head", "git_status"])
def test_zip_probe_requires_honest_no_git_environment(
    notebook_helpers: dict, probe_report: dict, field: str
) -> None:
    namespace = notebook_helpers
    namespace["SOURCE_MODE"] = "zip"
    probe_report["environment"]["git_head"] = None
    probe_report["environment"]["git_status"] = None
    probe_report["environment"][field] = "invented Git checkout"
    with pytest.raises(RuntimeError, match="(?i)git|snapshot"):
        namespace["probe"]("isolated-zip-parity", "parity", "cpu", "fp32")


@pytest.fixture
def no_git_probe_report(notebook_helpers: dict, probe_report: dict) -> dict:
    namespace = notebook_helpers
    namespace["SOURCE_MODE"] = "zip"
    environment = probe_report["environment"]
    environment.update({"git_head": None, "git_status": None, "collection_errors": {}})
    # Record actual failed queries in this isolated no-repository fixture, like the harness.
    for args in (("rev-parse", "HEAD"), ("status", "--short")):
        command = ["git", "-C", str(namespace["CHECKOUT"]), *args]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        assert result.returncode != 0
        assert result.stderr.strip()
        label = "git " + " ".join(args)
        environment["collection_errors"][label] = "RuntimeError: " + result.stderr.strip()
        namespace["write_json"](
            namespace["RUN"] / ("git-query-" + args[0] + ".json"),
            {
                "argv": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": environment["collection_errors"][label],
            },
        )
    return probe_report


@pytest.mark.parametrize("optional_dependency_error", [False, True])
def test_zip_probe_accepts_none_git_provenance_without_inventing_head(
    notebook_helpers: dict,
    no_git_probe_report: dict,
    optional_dependency_error: bool,
) -> None:
    report = no_git_probe_report
    if optional_dependency_error:
        report["environment"]["collection_errors"]["optional_dependency"] = (
            "ImportError: isolated optional dependency fixture"
        )
    assert notebook_helpers["probe"]("isolated-zip-parity", "parity", "cpu", "fp32") == report
    assert report["environment"]["git_head"] is None
    assert report["environment"]["git_status"] is None


@pytest.mark.parametrize("fault", ["missing", "null", "empty", "list", "text", "wrong-keys"])
def test_zip_probe_rejects_missing_or_malformed_git_query_diagnostics(
    notebook_helpers: dict,
    no_git_probe_report: dict,
    fault: str,
) -> None:
    environment = no_git_probe_report["environment"]
    if fault == "missing":
        del environment["collection_errors"]
    else:
        environment["collection_errors"] = {
            "null": None,
            "empty": {},
            "list": list(environment["collection_errors"].items()),
            "text": "git queries failed",
            "wrong-keys": {
                "git_head": "RuntimeError: fixture",
                "git_status": "RuntimeError: fixture",
            },
        }[fault]
    with pytest.raises(RuntimeError, match="(?i)git.*(quer|diagnostic|error)"):
        notebook_helpers["probe"]("isolated-bad-git-diagnostics", "parity", "cpu", "fp32")


@pytest.mark.parametrize("query", ["git rev-parse HEAD", "git status --short"])
@pytest.mark.parametrize("value", [None, "", " \n\t", False, 7, [], {}])
def test_zip_probe_rejects_empty_or_malformed_each_git_query_error(
    notebook_helpers: dict,
    no_git_probe_report: dict,
    query: str,
    value,
) -> None:
    errors = no_git_probe_report["environment"]["collection_errors"]
    errors[query] = value
    with pytest.raises(RuntimeError, match="(?i)git.*(quer|diagnostic|error)"):
        notebook_helpers["probe"]("isolated-bad-git-query", "parity", "cpu", "fp32")


@pytest.mark.parametrize("query", ["git rev-parse HEAD", "git status --short"])
def test_zip_probe_rejects_missing_or_misspelled_each_git_query_error(
    notebook_helpers: dict,
    no_git_probe_report: dict,
    query: str,
) -> None:
    errors = no_git_probe_report["environment"]["collection_errors"]
    errors[query + " "] = errors.pop(query)
    with pytest.raises(RuntimeError, match="(?i)git.*(quer|diagnostic|error)"):
        notebook_helpers["probe"]("isolated-wrong-git-query-key", "parity", "cpu", "fp32")


def _loss_report(namespace: dict, report: dict, dtype: str = "fp16") -> dict:
    modules = {
        "single": [f"isolated.layers.{layer}.self_attn.o_proj" for layer in range(2)],
        "qkv": [
            f"isolated.layers.{layer}.self_attn.{name}"
            for layer in range(2)
            for name in ("q_proj", "k_proj", "v_proj")
        ],
        "mlp": [f"isolated.layers.{layer}.mlp" for layer in range(2)],
    }
    modes = {path: dict.fromkeys(modules[path], dtype != "fp32") for path in ("qkv", "mlp")}
    report.update(
        {
            "mode": "loss",
            "dtype": dtype,
            "steps": 50,
            "base_unchanged": True,
            "model": {"num_hidden_layers": 2},
            "negative_control": {"unpatched_rejected": True},
            "execution": {
                path: dict.fromkeys(names, namespace["EXPECTED_FUNCTIONS"][path])
                for path, names in modules.items()
            },
            "reference_arithmetic": {
                "ctx_attribute": "grad_fn.reference_order",
                "required_true": dtype != "fp32",
                "expected_modules": {path: modules[path] for path in ("qkv", "mlp")},
            },
            "rows": [
                {
                    "step": step,
                    "baseline_loss": 1.0,
                    "fast_loss": 1.0,
                    "reference_arithmetic_modes": {
                        path: dict(values) for path, values in modes.items()
                    },
                    "reference_arithmetic_status": "observed",
                }
                for step in range(1, 51)
            ],
        }
    )
    return report


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "false",
        "partial",
        "late-step",
        "missing-step",
        "required-true",
        "wrong-modules",
        "status",
    ],
)
def test_low_precision_loss_rejects_unobserved_reference_arithmetic(
    notebook_helpers: dict,
    probe_report: dict,
    fault: str,
) -> None:
    report = _loss_report(notebook_helpers, probe_report)
    if fault == "missing":
        del report["reference_arithmetic"]
    elif fault == "false":
        report["rows"][0]["reference_arithmetic_modes"]["mlp"]["isolated.layers.0.mlp"] = False
    elif fault == "partial":
        del report["reference_arithmetic"]["expected_modules"]["qkv"]
    elif fault == "late-step":
        name = "isolated.layers.0.self_attn.q_proj"
        report["rows"][39]["reference_arithmetic_modes"]["qkv"][name] = False
    elif fault == "required-true":
        report["reference_arithmetic"]["required_true"] = False
    elif fault == "wrong-modules":
        report["reference_arithmetic"]["expected_modules"]["qkv"] = ["unobserved.module"]
    elif fault == "status":
        report["rows"][49]["reference_arithmetic_status"] = "INSUFFICIENT"
    else:
        del report["rows"][49]["reference_arithmetic_modes"]
    with pytest.raises(RuntimeError, match="(?i)reference.*(arithmetic|order|scope)"):
        notebook_helpers["probe"]("isolated-fp16-loss", "loss", "cpu", "fp16")


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_loss_accepts_complete_per_step_mode_evidence(
    notebook_helpers: dict, probe_report: dict, dtype: str
) -> None:
    report = _loss_report(notebook_helpers, probe_report, dtype)
    assert notebook_helpers["probe"]("isolated-loss", "loss", "cpu", dtype) == report


def test_frozen_source_mutation_blocks_a_child_before_launch(
    notebook_helpers: dict,
    source_zip: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = notebook_helpers
    target = namespace["WORK"] / "before-launch-source"
    namespace["CHECKOUT_INFO"] = namespace["extract_source_zip"](source_zip[0], target)
    namespace["CHECKOUT"] = target
    namespace["SOURCE_MODE"] = "zip"
    (target / "LICENSE").write_bytes(b"isolated mutation")
    calls = []

    def command(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", command)
    with pytest.raises(RuntimeError, match="(?i)snapshot|source|SHA256"):
        namespace["run_logged"]("isolated-no-install", [sys.executable, "-c", "pass"])
    assert calls == []


@pytest.mark.requires_symlink
@pytest.mark.parametrize("kind", ["input", "input-parent", "target-parent", "extracted-file"])
def test_source_transport_refuses_real_links_without_touching_old_work(
    notebook_helpers: dict,
    source_zip: tuple,
    kind: str,
) -> None:
    namespace = notebook_helpers
    path = source_zip[0]
    target = namespace["WORK"] / "linked-source"
    original = namespace["WORK"] / "old-fixture"
    original.mkdir()
    old = original / "private.txt"
    old.write_bytes(b"isolated old private fixture")
    if kind == "input":
        linked = namespace["WORK"] / "linked-input.zip"
        linked.symlink_to(path)
        path = linked
    elif kind in {"input-parent", "target-parent"}:
        linked = namespace["WORK"] / "linked-parent"
        linked.symlink_to(original, target_is_directory=True)
        if kind == "input-parent":
            own_zip = original / "d2-source-copy.zip"
            own_zip.write_bytes(path.read_bytes())
            path = linked / own_zip.name
        else:
            target = linked / "new-source"
    else:
        info = namespace["extract_source_zip"](path, target)
        fixture = target / "LICENSE"
        # Replace only our tiny own fixture, never a repo/private work file.
        fixture.unlink()
        fixture.symlink_to(old)
        namespace.update({"CHECKOUT_INFO": info, "CHECKOUT": target, "SOURCE_MODE": "zip"})
    with pytest.raises(RuntimeError, match="(?i)unsafe|link|contain|snapshot"):
        if kind == "extracted-file":
            namespace["assert_checkout"]()
        else:
            namespace["extract_source_zip"](path, target)
    assert old.read_bytes() == b"isolated old private fixture"
    assert not (original / "new-source").exists()


@pytest.mark.requires_hardlink
@pytest.mark.parametrize("kind", ["input", "extracted-file"])
def test_source_transport_rejects_hardlinks(
    notebook_helpers: dict, source_zip: tuple, kind: str
) -> None:
    namespace = notebook_helpers
    target = namespace["WORK"] / "hardlink-source"
    if kind == "input":
        linked = namespace["WORK"] / "linked-input.zip"
        os.link(source_zip[0], linked)
        with pytest.raises(RuntimeError, match="(?i)unsafe|hardlink"):
            namespace["extract_source_zip"](linked, target)
        assert not target.exists()
    else:
        info = namespace["extract_source_zip"](source_zip[0], target)
        own_file = namespace["WORK"] / "hardlink-fixture.txt"
        own_file.write_bytes((target / "LICENSE").read_bytes())
        (target / "LICENSE").unlink()
        os.link(own_file, target / "LICENSE")
        namespace.update({"CHECKOUT_INFO": info, "CHECKOUT": target, "SOURCE_MODE": "zip"})
        with pytest.raises(RuntimeError, match="(?i)unsafe|hardlink"):
            namespace["assert_checkout"]()


def test_source_zip_rejects_duplicate_manifest_json_keys(notebook_helpers: dict, source_zip: tuple):
    path, manifest, files = source_zip
    with zipfile.ZipFile(path, "w") as bundle:
        raw = json.dumps(manifest)
        bundle.writestr("D2-SNAPSHOT.json", '{"format":"soup-d2-snapshot-v1",' + raw[1:])
        for name, data in files.items():
            bundle.writestr(name, data)
    target = notebook_helpers["WORK"] / "duplicate-json-source"
    with pytest.raises(RuntimeError, match="Duplicate.*manifest"):
        notebook_helpers["extract_source_zip"](path, target)
    assert not target.exists()


def test_source_zip_accepts_only_required_ancestor_directory_entries(
    notebook_helpers: dict, source_zip: tuple
) -> None:
    path = source_zip[0]
    with zipfile.ZipFile(path, "a") as bundle:
        bundle.writestr("src/", b"")
        bundle.writestr("src/soup_cli/", b"")
    target = notebook_helpers["WORK"] / "directory-source"
    assert notebook_helpers["extract_source_zip"](path, target)["git_state"] == "NO_GIT_SNAPSHOT"


@pytest.mark.parametrize("fault", ["large-member", "large-total", "encrypted"])
def test_source_zip_enforces_bounded_unencrypted_inventory(
    notebook_helpers: dict,
    source_zip: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    original = zipfile.ZipFile.infolist

    def entries(bundle):
        values = original(bundle)
        if fault == "large-member":
            values[-1].file_size = 65 * 1024 * 1024
        elif fault == "large-total":
            values[-1].file_size = 257 * 1024 * 1024
        else:
            values[-1].flag_bits |= 1
        return values

    monkeypatch.setattr(zipfile.ZipFile, "infolist", entries)
    target = notebook_helpers["WORK"] / "bounded-source"
    with pytest.raises(RuntimeError, match="(?i)source.*(limit|member)|unsafe"):
        notebook_helpers["extract_source_zip"](source_zip[0], target)
    assert not target.exists()


def test_source_changed_by_a_child_blocks_pass_but_retains_junit(
    notebook_helpers: dict,
    source_zip: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = notebook_helpers
    target = namespace["WORK"] / "child-mutation-source"
    namespace["CHECKOUT_INFO"] = namespace["extract_source_zip"](source_zip[0], target)
    namespace.update({"CHECKOUT": target, "SOURCE_MODE": "zip"})

    def command(argv, **kwargs):
        (target / "LICENSE").write_bytes(b"isolated child mutation")
        junit = Path(argv[argv.index("--junitxml") + 1])
        junit.write_text('<testsuite><testcase name="isolated"/></testsuite>', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", command)
    with pytest.raises(RuntimeError, match="(?i)source|snapshot|command"):
        namespace["run_pytest"](
            "isolated-mutating-pytest",
            ["isolated.py"],
            device="cpu",
            expression="not gpu and not smoke",
        )
    directory = next(namespace["RUN"].glob("isolated-mutating-pytest-*"))
    summary = json.loads((directory / "pytest-summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == 1
    assert summary["returncode"] == 0  # Preserve the actual process exit, not a fabricated exit.
    assert summary["command_status"] == "FAILED_SOURCE_SNAPSHOT"


@pytest.mark.parametrize("valid", [False, True])
def test_source_preparation_retains_success_or_failure_without_installing(
    notebook_helpers: dict,
    source_zip: tuple,
    monkeypatch: pytest.MonkeyPatch,
    valid: bool,
) -> None:
    namespace = notebook_helpers
    path, manifest, files = source_zip
    if not valid:
        manifest["format"] = "invalid"
        _write_source_zip(path, manifest, files)
    target = namespace["WORK"] / "stage-source"
    namespace.update(
        {
            "SOURCE_MODE": "zip",
            "SOURCE_ZIP": str(path),
            "EXPECTED_ZIP_SHA256": "",
            "INPUT_ROOT": namespace["WORK"],
            "WORK_SOURCE": target,
            "CHECKOUT": target,
        }
    )
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Preparation tried a subprocess before validated source")

    monkeypatch.setattr(subprocess, "run", forbidden)
    stage = _sources()["checkout-install"].split("try:\n    ORIGINAL_TORCH_VERSION", 1)[0]
    if valid:
        exec(compile(stage, "source-preparation-only", "exec"), namespace)
    else:
        with pytest.raises(RuntimeError, match="(?i)snapshot|manifest"):
            exec(compile(stage, "source-preparation-only", "exec"), namespace)
    artifact = namespace["RUN"] / "source-preparation.json"
    assert artifact.is_file(), "Source preparation did not preserve evidence"
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["status"] == ("PASS" if valid else "FAIL")
    assert calls == []
    if not valid:
        assert not target.exists()
        assert "RuntimeError" in result["error"]


def test_actual_ruff_preparation_preserves_isolated_frozen_source(
    notebook_helpers: dict,
    source_zip: tuple,
    record_property,
) -> None:
    namespace = notebook_helpers
    path, manifest, _ = source_zip
    target = namespace["WORK"] / "ruff-preparation-source"
    namespace.update(
        {
            "SOURCE_MODE": "zip",
            "SOURCE_ZIP": str(path),
            "EXPECTED_ZIP_SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "INPUT_ROOT": namespace["WORK"],
            "WORK_SOURCE": target,
            "CHECKOUT": target,
            "shutil": shutil,
            "ORIGINAL_TORCH": {"fixture": "transport-only; Torch probe not executed"},
        }
    )
    preparation = _sources()["checkout-install"].split("try:\n    ORIGINAL_TORCH_VERSION", 1)[0]
    exec(compile(preparation, "actual-ruff-source-preparation", "exec"), namespace)
    assert json.loads((namespace["RUN"] / "source-preparation.json").read_text())["status"] == (
        "PASS"
    )

    def inventory() -> dict:
        return {
            "files_sha256": {
                item.relative_to(target).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
                for item in target.rglob("*")
                if item.is_file()
            },
            "directories": sorted(
                item.relative_to(target).as_posix() for item in target.rglob("*") if item.is_dir()
            ),
        }

    version_record = namespace["run_logged"](
        "ruff-version", [sys.executable, "-m", "ruff", "--version"]
    )
    record_property(
        "ruff_version", (Path(version_record["directory"]) / "stdout.log").read_text().strip()
    )
    before = inventory()
    namespace["write_json"](namespace["RUN"] / "frozen-inventory-before.json", before)
    # Execute real provenance preparation and its actual Ruff command, not a mocked helper.
    # Only dependency/model import smoke is outside this transport-only fixture's scope.
    tree = ast.parse(_sources()["provenance"])
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "run_logged"
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "import-smoke"
        )
    ]
    try:
        exec(compile(tree, "actual-ruff-provenance-preparation", "exec"), namespace)
    finally:
        after = inventory()
        namespace["write_json"](namespace["RUN"] / "frozen-inventory-after.json", after)
        directories = list(namespace["RUN"].glob("targeted-ruff-*"))
        assert len(directories) == 1
        directory = directories[0]
        command = json.loads((directory / "command.json").read_text(encoding="utf-8"))
        record_property("ruff_argv", json.dumps(command["argv"]))
        record_property("ruff_returncode", command["returncode"])
        record_property("ruff_status", command["status"])
        record_property("frozen_inventory_unchanged", before == after)
        assert command["returncode"] == 0
        assert (directory / "stdout.log").read_text(encoding="utf-8").strip() == (
            "All checks passed!"
        )
        assert (directory / "stderr.log").read_bytes() == b""
        assert command["cwd"] == str(target)
        assert command["argv"][:4] == [sys.executable, "-m", "ruff", "check"]
        assert command["argv"][-len(namespace["TEST_FILES"]) :] == namespace["TEST_FILES"]
        assert before == after, "Ruff added or mutated files/directories inside frozen source"
        assert command["status"] == "EXIT_ZERO"
    assert namespace["assert_checkout"]()["snapshot_id"] == manifest["snapshot_id"]
    assert not (target / ".ruff_cache").exists()
    assert namespace["MANIFEST"]["files_sha256"] == manifest["source_sha256"]
