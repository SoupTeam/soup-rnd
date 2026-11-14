"""G3 Windows DirectStorage analysis follows its prospective protocol (CPU only)."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DS = ROOT / "benchmarks" / "g3" / "directstorage"
sys.path.insert(0, str(DS))
import analyze_ds  # noqa: E402

TEMPLATE = json.loads((ROOT / "tests/fixtures/g3/ds_run_template.json").read_text())


def make(tmp_path, rates, *, errors=None, outcomes=None, name="results"):
    """rates: list of blocks, each [A, B, B, A] useful GB/s; errors/outcomes keyed (block, pos)."""
    errors, outcomes = errors or {}, outcomes or {}
    root = tmp_path / name
    (root / "runs").mkdir(parents=True)
    series = []
    for block, values in enumerate(rates):
        for position, (arm, value) in enumerate(zip(analyze_ds.ORDER, values)):
            run = copy.deepcopy(TEMPLATE)
            run["arm"], run["useful_gb_per_s"] = arm, value
            run["explicit_pinned_host_bytes"] = 64 << 20 if arm == "A" else 0
            error = errors.get((block, position))
            run["ok"], run["error"] = error is None, error or ""
            run["validity"]["outcome"] = outcomes.get((block, position), "ok")
            name_ = f"b{block:02d}_{position}_{arm}"
            (root / "runs" / f"{name_}.json").write_text(json.dumps(run))
            series.append(
                {"block": block, "position": position, "arm": arm, "exit": 1 if error else 0}
            )
    # run_g3.ps1 writes UTF-8 with a byte-order mark under Windows PowerShell 5.1.
    (root / "series.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(series).encode())
    (root / "machine.json").write_text(json.dumps({"schema": "g3-ds-machine-v1"}))
    return root


def steady(n, a=3.0, b=3.3):
    return [[a, b, b + 0.01, a + 0.01] for _ in range(n)]


def test_protocol_and_sources_are_present_and_name_the_package_hash():
    protocol = (DS / "PROTOCOL.txt").read_text()
    assert "8ab6c1082537565388b79050d94fdd7f5f437cd0f1d909e7b19f786042b97177" in protocol
    assert (
        "8ab6c1082537565388b79050d94fdd7f5f437cd0f1d909e7b19f786042b97177"
        in (DS / "run_g3.ps1").read_text()
    )
    assert "Never" in protocol or "never claimed" in protocol
    source = (DS / "ds_probe.cpp").read_text()
    assert "CREATE_NEW" in source and "corruption_control_detected" in source


def test_verified_and_faster_b_after_five_clean_blocks(tmp_path):
    result = analyze_ds.analyze([make(tmp_path, steady(5))])
    assert result["capability"]["outcome"] == "DS_TO_CUDA_VERIFIED"
    perf = result["performance"]
    assert perf["label"] == "B_FASTER" and perf["valid_blocks"] == 5
    assert perf["difference_95"][0] > 0
    assert result["descriptive"]["A"]["explicit_pinned_host_bytes"] == 64 << 20


def test_equal_arms_give_no_detectable_difference(tmp_path):
    rates = [
        [3.0, 3.01, 2.99, 3.0],
        [3.0, 2.99, 3.0, 3.01],
        [3.01, 3.0, 3.0, 2.99],
        [2.99, 3.0, 3.01, 3.0],
        [3.0, 3.0, 3.0, 3.0],
    ]
    perf = analyze_ds.analyze([make(tmp_path, rates)])["performance"]
    assert perf["label"] == "NO_DETECTABLE_DIFFERENCE"


def test_noisy_pilot_asks_for_a_top_up_and_a_second_folder_completes_it(tmp_path):
    # Pooled sigma 0.33 against mean A 2.99: the METHOD rule asks for 10 blocks.
    noisy = [
        [3.4, 3.3, 2.8, 2.7],
        [2.6, 3.7, 3.0, 3.2],
        [3.3, 2.9, 3.5, 2.7],
        [2.8, 3.6, 3.0, 3.3],
        [3.1, 3.0, 3.7, 2.8],
    ]
    perf = analyze_ds.analyze([make(tmp_path, noisy)])["performance"]
    assert perf["label"] == "NO_VERDICT" and perf["top_up_blocks"] == 5
    extra = steady(perf["top_up_blocks"])
    second = make(tmp_path, [[0, 0, 0, 0]] * 5 + extra, name="topup")
    # Keep only the top-up blocks in the second folder, as run_g3.ps1 -StartBlock 5 writes them.
    series = json.loads((second / "series.json").read_text(encoding="utf-8-sig"))
    (second / "series.json").write_text(json.dumps([e for e in series if e["block"] >= 5]))
    done = analyze_ds.analyze([make(tmp_path, noisy, name="pilot"), second])["performance"]
    assert done["blocks_run"] == 10 and done["valid_blocks"] == 10
    assert done["label"] in {"B_FASTER", "NO_DETECTABLE_DIFFERENCE", "B_SLOWER"}


def test_void_blocks_are_replaced_and_too_many_give_host_noise(tmp_path):
    void = {(0, 1): "void", (1, 2): "void", (2, 0): "void"}
    perf = analyze_ds.analyze([make(tmp_path, steady(8), outcomes=void)])["performance"]
    assert perf["label"] == "NO_VERDICT_HOST_NOISE" and perf["void_blocks"] == 3
    one = analyze_ds.analyze([make(tmp_path, steady(6), outcomes={(0, 1): "void"}, name="one")])
    assert one["performance"]["valid_blocks"] == 5 and one["performance"]["label"] == "B_FASTER"


def test_unknown_required_check_gives_no_verdict(tmp_path):
    perf = analyze_ds.analyze([make(tmp_path, steady(5), outcomes={(3, 0): "unknown"})])
    assert perf["performance"]["label"] == "NO_VERDICT"
    assert "unknown" in perf["performance"]["reason"]


def test_same_directstorage_error_in_every_b_run_is_a_stack_negative(tmp_path):
    errors = {
        (b, p): "IDStorageFactory::CreateQueue (0x89240008)" for b in range(5) for p in (1, 2)
    }
    result = analyze_ds.analyze([make(tmp_path, steady(5), errors=errors)])
    assert result["capability"]["outcome"] == "DS_UNAVAILABLE_ON_TESTED_STACK"
    assert result["performance"]["label"] == "NO_VERDICT"


def test_missing_runtime_or_failed_control_is_not_a_stack_verdict(tmp_path):
    missing = {
        (b, p): "LoadLibrary dstorage.dll next to ds_probe.exe (0x7e)"
        for b in range(5)
        for p in (1, 2)
    }
    assert (
        analyze_ds.analyze([make(tmp_path, steady(5), errors=missing)])["capability"]["outcome"]
        == "ENVIRONMENT_BLOCKED"
    )
    control = {(0, 0): "byte oracle mismatch (0x1)"}
    assert (
        analyze_ds.analyze([make(tmp_path, steady(5), errors=control, name="c")])["capability"][
            "outcome"
        ]
        == "NO_VERDICT"
    )
    flaky = {(0, 1): "DirectStorage request failed (0x80070005)"}
    assert (
        analyze_ds.analyze([make(tmp_path, steady(5), errors=flaky, name="f")])["capability"][
            "outcome"
        ]
        == "NO_VERDICT"
    )


def test_output_is_never_overwritten(tmp_path):
    root = make(tmp_path, steady(5))
    out = tmp_path / "analysis.json"
    analyze_ds.main([str(root), "--out", str(out)])
    with pytest.raises(FileExistsError):
        analyze_ds.main([str(root), "--out", str(out)])


def test_probe_cross_compiles_without_warnings_when_mingw_is_installed(tmp_path):
    import shutil
    import subprocess

    if shutil.which("x86_64-w64-mingw32-g++-posix") is None:
        pytest.skip("MinGW-w64 cross compiler not installed")
    done = subprocess.run(
        ["sh", str(DS / "build.sh"), str(tmp_path)], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "ds_probe.exe").stat().st_size > 0
    assert "ds_probe.exe sha256:" in (tmp_path / "BUILD.txt").read_text()


def test_retained_rtx4060_series_replays_to_the_published_result():
    evidence = ROOT / "benchmarks/g3/reports/20261104_directstorage/evidence"
    result = analyze_ds.analyze([evidence])
    published = json.loads((evidence.parent / "analysis.json").read_text())
    assert result["capability"]["outcome"] == "DS_TO_CUDA_VERIFIED"
    assert result["performance"]["label"] == "B_SLOWER"
    assert result["performance"] == published["performance"]
    assert result["descriptive"] == published["descriptive"]
