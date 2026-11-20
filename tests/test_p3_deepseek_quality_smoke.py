"""CPU-only portable-smoke refusal/evidence contracts; no native gate execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.harness import p3_deepseek_quality_smoke as smoke


@pytest.mark.parametrize("value", ["", "   ", None, 123])
def test_invalid_output_is_refused_before_creating_evidence(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="output"):
        smoke.run_smoke(output=value)


def test_existing_output_is_untouched(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("immutable", encoding="utf-8")
    with pytest.raises(FileExistsError):
        smoke.run_smoke(output=output)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "immutable"


def test_existing_file_is_not_an_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "file"
    output.write_text("immutable", encoding="utf-8")
    with pytest.raises(FileExistsError):
        smoke.run_smoke(output=output)
    assert output.read_text(encoding="utf-8") == "immutable"


def test_fixture_directory_factory_refuses_escape(tmp_path: Path) -> None:
    factory = smoke._Directories(tmp_path)
    for name in ("../escape", "a/b", "", ".", ".."):
        with pytest.raises(ValueError, match="fixture directory"):
            factory.mktemp(name)
    assert not list(tmp_path.iterdir())


def test_cpu_refusal_writes_failure_manifest_without_importing_fixture_or_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    before = set(sys.modules)
    output = tmp_path / "cpu-refusal"
    with pytest.raises(RuntimeError, match="smoke FAILED"):
        smoke.run_smoke(output=output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["failed_stage"] == "native_admission"
    assert "available CUDA" in manifest["error"]["message"]
    assert manifest["scientific_admission"] is False
    assert manifest["test_fixture"] is True
    assert manifest["attempts"] == []
    assert (output / "FAILED.json").is_file()
    assert not (output / "fixtures").exists()
    assert not (output / "training").exists()
    imported = set(sys.modules) - before
    assert "benchmarks.harness.p3_deepseek_quality_gate" not in imported
    assert "tests.test_p3_deepseek_quality_model" not in imported
    assert "tests.test_p3_deepseek_quality_train" not in imported
    for name, digest in manifest["files"].items():
        assert smoke._sha256(output / name) == digest


def test_direct_cli_output_only_fails_closed_without_cuda(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    output = tmp_path / "direct-cli"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    process = subprocess.run(
        [sys.executable, "-m", "benchmarks.harness.p3_deepseek_quality_smoke",
         "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["failed_stage"] == "native_admission"
    assert "available CUDA" in manifest["error"]["message"]
    assert manifest["scientific_admission"] is False
    assert manifest["attempts"] == []


@pytest.mark.parametrize("extra", [["--device", "cpu"], ["--length", "64"],
                                    ["--source", "download-me"], ["--steps", "1"]])
def test_cli_cannot_override_fixed_native_workload(tmp_path: Path, extra: list[str]) -> None:
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as error:
        smoke.main(["--output", str(output), *extra])
    assert error.value.code == 2
    assert not output.exists()
