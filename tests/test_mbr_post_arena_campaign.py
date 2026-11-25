"""Ticket 17 measurement acceptance, using CPU evidence and mock launchers only."""

import importlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.harness import mbr_measure as mm

CAMPAIGN = Path(__file__).resolve().parents[1] / "benchmarks/mbr-moonlight-post-arena-20261125"
analysis = importlib.import_module("benchmarks.mbr-moonlight-post-arena-20261125.analyze")
replacement = importlib.import_module("benchmarks.mbr-moonlight-post-arena-20261125.replace_void")


# Known-good dataset hashes of the ticket 03 bits rows, one per sequence length.
SHA_BY_SEQ = {64: "6beb56aef0d35cae86db050e793596605803289a83478962f0a592bfbabbe1a8",
              512: "91048d80d8acbfcc18a999e93deefdf9cbf70bd61de3eecfb39e108ccf76efea"}
VERSIONS = {"torch": "2.14.1+cu130", "triton": "3.8.0", "transformers": "5.20.2",
            "peft": "0.21.2", "bitsandbytes": "0.50.2", "safetensors": "0.8.0"}
IDENTITY = mm.campaign_identity(source="fake", cache="fake", tokenizer=None, lora_r=16,
                                lora_targets=mm.DEFAULT_LORA_TARGETS, steps=4)
LAUNCHER_TOOLS = ("bash", "dirname", "mkdir", "flock", "date", "tee")
needs_launcher = pytest.mark.skipif(
    sys.platform != "linux" or not all(shutil.which(name) for name in LAUNCHER_TOOLS),
    reason="launcher tests need Linux with bash, flock and coreutils",
)
SMALL_SHAPES = {"a.lora_A.weight": (2, 3), "a.lora_B.weight": (3, 2), "b.lora_A.weight": (2, 2)}


@pytest.fixture(autouse=True)
def _refuse_archived_tensor_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ``torch.load`` of a path under the archived campaign folder fails in these tests.

    The check runs on the real path of the argument, or of a file object's ``name``. Plain
    ``open()`` and ``read_bytes()`` calls are not covered; the harness loads tensors only through
    ``torch.load``."""
    import torch

    archive = os.path.realpath(CAMPAIGN)
    real_load = torch.load

    def guarded(source, *args, **kwargs):
        name = source if isinstance(source, (str, os.PathLike)) else getattr(source, "name", None)
        if isinstance(name, (str, os.PathLike)):
            target = os.path.realpath(os.fspath(name))
            if os.path.commonpath([archive, target]) == archive:
                raise AssertionError(f"archived tensor read: {target}")
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)


def _grads(shapes: dict | None = None, seed: int = 0) -> dict:
    """Synthetic LoRA grads, one per name, shaped like the authoritative manifest by default."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    shapes = analysis.GRAD_SHAPES if shapes is None else shapes
    return {name: torch.randn(shape, generator=generator) for name, shape in sorted(shapes.items())}


def _payload(grads: dict, losses: tuple = (1.25, 1.5)) -> dict:
    import torch

    return {"loss": torch.tensor(list(losses), dtype=torch.float64), "grads": grads}


def _row(payload: dict, shapes: dict) -> dict:
    """The row a bits run writes for ``payload``: loss total, digest and the live manifest."""
    flat = {name: grad.float().flatten().tolist() for name, grad in payload["grads"].items()}
    return {"loss": float(sum(payload["loss"].tolist())), "grad_sha256": mm.grad_digest(flat),
            "grad_shapes": {name: list(shape) for name, shape in shapes.items()}}


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _isolate(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Keep bits_report to the synthetic DS cell: no stored or ticket 06 tensor is read."""
    monkeypatch.setattr(analysis, "BIT_FOLDERS", {"ds-k2-s64": ("A0", "DS")})
    monkeypatch.setattr(analysis, "TICKET06_D", root / "no-ticket-06")


def _bit_folder(root: Path, full_grads: dict) -> Path:
    """Synthetic A0 and DS cell at k=2, S=64: tensors, rows and records bound to each other."""
    import torch

    path = root / "bits" / "ds-k2-s64"
    (path / "tensors").mkdir(parents=True)
    rows, records = [], []
    for run, side in enumerate(("A0", "DS"), start=1):
        name = mm.tensor_file_name(side, 2, 64)
        payload = _payload(full_grads)
        torch.save(payload, path / "tensors" / name)
        rows.append(dict(
            _row(payload, analysis.GRAD_SHAPES), arm=side, k=2, seq=64, block=0, kind="bits",
            mode="deterministic", attention="default", status="ok", outcome="ok", run=run,
            tensor_file=name, dataset_sha256=SHA_BY_SEQ[64], step_seconds=1.0,
            campaign_id=IDENTITY,
            environment={"versions": dict(VERSIONS), "git_sha": "0" * 40, "ptxas": "unknown",
                         "alloc_conf": "default"}))
        records.append(dict(arm=side, run=run, round=(run + 1) // 2, outcome="ok",
                            required=list(mm.ARM_CHECKS),
                            checks={name: {"outcome": "ok"} for name in mm.ARM_CHECKS}))
    _write(path / "rows.jsonl", rows)
    _write(path / "arm_records.jsonl", records)
    return path


def _evidence(item: dict) -> tuple[dict, dict]:
    info = {key: item[key] for key in ("arm", "k", "seq", "block", "kind")}
    run = item["index"] + 1
    row = dict(info, mode="normal", outcome="ok", status="ok", run=run,
               step_seconds={"C": 22, "E": 20, "G": 17}[item["arm"]],
               dataset_sha256=SHA_BY_SEQ[item["seq"]], campaign_id=IDENTITY,
               levers=mm.arm_runtime(item["arm"]),
               environment={"versions": dict(VERSIONS), "git_sha": "0" * 40, "ptxas": "unknown",
                            "alloc_conf": "expandable_segments:True"},
               peak_allocated=994472960, peak_reserved=1109393408, peak_nvml=1262485504)
    record = dict(arm=item["arm"], run=run, round=(run - 1) // 2 + 1, outcome="ok",
                  required=list(mm.ARM_CHECKS), checks={name: {"outcome": "ok"}
                                                        for name in mm.ARM_CHECKS},
                  gpu_clock={"outcome": "ok", "sm_clock_mhz": {"max": 1900}},
                  campaign_id=IDENTITY)
    return row, record


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def speed(tmp_path: Path) -> tuple[Path, list[dict], list[dict]]:
    folder = tmp_path / "speed"
    folder.mkdir()
    rows, records = map(list, zip(*map(_evidence, mm.ceg_speed_plan())))
    _write(folder / "rows.jsonl", rows)
    _write(folder / "arm_records.jsonl", records)
    return folder, rows, records


def test_speed_without_provenance_and_memory_has_no_verdict(speed) -> None:
    folder, rows, _ = speed
    for row in rows:
        for field in ("dataset_sha256", "environment", "levers", "peak_allocated",
                      "peak_reserved", "peak_nvml", "campaign_id"):
            row.pop(field, None)
    _write(folder / "rows.jsonl", rows)
    report = analysis.speed_report(folder)
    assert all(window["verdict"] is None for window in report["windows"].values())


@pytest.mark.parametrize("field", [
    "dataset_sha256", "torch", "runtime", "allocator", "levers",
    "peak_allocated", "peak_reserved", "peak_nvml", "campaign_id",
])
def test_speed_rejects_incomplete_provenance(speed, field: str) -> None:
    folder, rows, _ = speed
    if field == "torch":
        rows[0]["environment"]["versions"]["torch"] = "unknown"
    elif field == "runtime":
        rows[0]["environment"]["versions"]["transformers"] = "unknown"
    elif field == "allocator":
        rows[0]["environment"]["alloc_conf"] = "default"
    elif field == "levers":
        rows[0]["levers"]["expert_gemm"] = "lt-fused"
    elif field.startswith("peak_"):
        rows[0][field] = True
    else:
        rows[0][field] = "wrong"
    _write(folder / "rows.jsonl", rows)
    assert analysis.speed_report(folder)["windows"]["(8,64)"]["verdict"] is None


@pytest.mark.parametrize("field", ["peak_allocated", "peak_reserved", "peak_nvml"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, None, "1"])
def test_speed_rejects_invalid_peak_values(speed, field: str, value) -> None:
    folder, rows, _ = speed
    rows[0][field] = value
    _write(folder / "rows.jsonl", rows)
    assert analysis.speed_report(folder)["windows"]["(8,64)"]["verdict"] is None


def test_speed_binds_dataset_sha_to_its_sequence(speed) -> None:
    folder, rows, _ = speed
    for row in rows:
        if row["seq"] == 512:
            row["dataset_sha256"] = SHA_BY_SEQ[64]
    _write(folder / "rows.jsonl", rows)
    assert analysis.speed_report(folder)["windows"]["(8,512)"]["verdict"] is None


def test_speed_rejects_mixed_environment_across_rows(speed) -> None:
    folder, rows, _ = speed
    rows[1]["environment"]["versions"]["torch"] = "2.14.0+cu130"
    _write(folder / "rows.jsonl", rows)
    assert analysis.speed_report(folder)["windows"]["(8,64)"]["verdict"] is None


def test_speed_accepts_the_expected_campaign_identity(speed) -> None:
    folder, _, _ = speed
    report = analysis.speed_report(folder, IDENTITY)
    assert report["windows"]["(8,64)"]["verdict"] == "available"
    assert report["windows"]["(8,512)"]["verdict"] == "available"


def test_speed_refuses_uniformly_foreign_campaign(speed) -> None:
    folder, rows, records = speed
    for row in rows:
        row["campaign_id"] = "foreign"
    _write(folder / "rows.jsonl", rows)
    report = analysis.speed_report(folder, IDENTITY)
    assert all(window["verdict"] is None for window in report["windows"].values())
    assert "campaign_identity" in report["errors"]


def test_speed_rejects_git_sha_change_mid_campaign(speed) -> None:
    folder, rows, _ = speed
    rows[1]["environment"]["git_sha"] = "f" * 40
    _write(folder / "rows.jsonl", rows)
    assert analysis.speed_report(folder)["windows"]["(8,64)"]["verdict"] is None


def test_speed_requires_row_run_ids_without_any_marker_file(speed) -> None:
    folder, rows, _ = speed
    for row in rows:
        row.pop("run", None)
    _write(folder / "rows.jsonl", rows)
    assert not (folder / ".ticket17-active").exists()
    assert analysis.speed_report(folder)["windows"]["(8,64)"]["verdict"] is None


def test_speed_refuses_records_from_another_campaign(speed) -> None:
    folder, _, records = speed
    for record in records:
        record["campaign_id"] = "foreign"
    _write(folder / "arm_records.jsonl", records)
    report = analysis.speed_report(folder, IDENTITY)
    assert all(window["verdict"] is None for window in report["windows"].values())
    assert all(window["invalid_blocks"] for window in report["windows"].values())


def test_legacy_speed_archive_without_campaign_identity_has_no_verdict() -> None:
    report = analysis.speed_report(CAMPAIGN / "speed")
    assert all(window["verdict"] is None for window in report["windows"].values())
    assert any("campaign" in reason for reason in report["errors"])


def test_analyzer_manifest_is_the_harness_manifest() -> None:
    assert analysis.GRAD_SHAPES == mm.MOONLIGHT_GRAD_SHAPES
    assert len(analysis.GRAD_SHAPES) == analysis.GRAD_TOTAL == 216
    lora_a_q0 = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    assert analysis.GRAD_SHAPES[lora_a_q0] == [16, 2048]


def test_metadata_campaign_identity_follows_model_cache_and_recipe() -> None:
    base = dict(source="s", cache="c", tokenizer=None, lora_r=16, lora_targets=("q_proj",),
                steps=4)
    identity = mm.campaign_identity(**base)
    assert mm.campaign_identity(**base) == identity
    assert mm.campaign_identity(**{**base, "tokenizer": "s"}) == identity
    for key, value in (("source", "t"), ("cache", "d"), ("lora_r", 8),
                       ("lora_targets", ("o_proj",)), ("steps", 5)):
        assert mm.campaign_identity(**{**base, key: value}) != identity


@pytest.mark.parametrize(
    "defect", ["slow-clock", "absent-clock", "unknown-clock", "missing-repeat", "missing-record",
               "foreign-record", "missing-check", "unknown-power", "void-suspend", "missing-status",
               "missing-outcome", "crash-ok", "shuffled", "interleaved", "reused-id"],
)
def test_bad_block_cannot_produce_a_speed_verdict(speed, defect: str) -> None:
    folder, rows, records = speed
    if defect == "slow-clock":
        records[0]["gpu_clock"]["sm_clock_mhz"]["max"] = 900
    elif defect == "absent-clock":
        del records[0]["gpu_clock"]["sm_clock_mhz"]["max"]
    elif defect == "unknown-clock":
        records[0]["gpu_clock"]["outcome"] = "unknown"
    elif defect == "missing-repeat":
        del rows[5]
        del records[5]
    elif defect == "missing-record":
        del records[0]
    elif defect == "foreign-record":
        records[0]["arm"] = "E"
    elif defect == "missing-check":
        del records[0]["checks"]["foreign_reads"]
    elif defect == "unknown-power":
        records[0]["checks"]["power"]["outcome"] = "unknown"
    elif defect == "void-suspend":
        records[0]["checks"]["suspend"]["outcome"] = "void"
    elif defect == "missing-status":
        del rows[0]["status"]
    elif defect == "missing-outcome":
        del rows[0]["outcome"]
    elif defect == "crash-ok":
        rows[0]["status"] = "CRASH"
    elif defect == "shuffled":
        rows[1], rows[2] = rows[2], rows[1]
    elif defect == "interleaved":
        rows[5], rows[6] = rows[6], rows[5]
        records[5], records[6] = records[6], records[5]
    else:
        rows.extend(rows[:6])
        records.extend(records[:6])
    _write(folder / "rows.jsonl", rows)
    _write(folder / "arm_records.jsonl", records)
    report = analysis.speed_report(folder)
    window = report["windows"]["(8,64)"]
    assert window.get("verdict") is None and not window.get("deltas")
    assert window["invalid_blocks"]


@pytest.mark.parametrize("failure", ["void", "crash-ok", "missing-repeat", "twice", "interrupted",
                                     "reused-id"])
def test_replacement_runs_whole_new_blocks_and_retains_raw_rows(
    speed, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    folder, rows, records = speed
    if failure == "missing-repeat":
        del rows[5]
        del records[5]
    elif failure == "crash-ok":
        rows[0]["status"] = "CRASH"
    elif failure == "reused-id":
        rows.extend(rows[:6])
        records.extend(records[:6])
    else:
        rows[0]["outcome"] = records[0]["outcome"] = "void"
    rows_path, records_path = folder / "rows.jsonl", folder / "arm_records.jsonl"
    _write(rows_path, rows)
    _write(records_path, records)
    before = rows_path.read_bytes(), records_path.read_bytes()
    launched = []

    def sweep(args, plan: list[dict]) -> list[dict]:
        launched.append(plan)
        if failure == "interrupted" and len(launched) == 1:
            plan = plan[:2]
        output = []
        for index, item in enumerate(plan):
            row, record = _evidence(item)
            if failure == "twice" and len(launched) == 1 and index == 0:
                row["status"] = "CRASH"
            output.append(row)
            mm.append_jsonl(rows_path, row)
            mm.append_jsonl(records_path, record)
        return output

    monkeypatch.setattr(mm, "sweep", sweep)
    argv = ["--out", str(folder), "--source", "fake", "--cache", "fake"]
    if failure == "interrupted":
        assert replacement.main(argv) != 0
    assert replacement.main(argv) == 0
    ids = []
    for batch in launched:
        assert tuple(item["arm"] for item in batch) == mm.CEG_HANDS
        ids.append({item["block"] for item in batch}.pop())
    assert len(ids) == (2 if failure in ("twice", "interrupted") else 1)
    assert len(set(ids)) == len(ids) and all(idx > 4 for idx in ids)
    assert rows_path.read_bytes().startswith(before[0])
    assert records_path.read_bytes().startswith(before[1])
    report = analysis.speed_report(folder)
    assert report["windows"]["(8,64)"]["verdict"] == "available"
    assert report["windows"]["(8,512)"]["verdict"] == "available"
    assert report["runs"] == 60


@pytest.mark.parametrize("case", ["foreign-all", "missing-all", "foreign-one", "other-cache"])
def test_resume_refuses_foreign_campaign_before_gate_sweep_or_write(
    speed, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    folder, rows, _ = speed
    cache = "fake"
    for index, row in enumerate(rows):
        if case == "foreign-all":
            row["campaign_id"] = "foreign"
        elif case == "missing-all":
            del row["campaign_id"]
        elif case == "foreign-one" and index == 0:
            row["campaign_id"] = "foreign"
    if case == "other-cache":
        cache = "other"
    rows_path, records_path = folder / "rows.jsonl", folder / "arm_records.jsonl"
    _write(rows_path, rows)
    before = rows_path.read_bytes(), records_path.read_bytes()
    monkeypatch.setattr(mm, "sweep", lambda *args: pytest.fail("swept a foreign campaign"))
    assert replacement.main(["--out", str(folder), "--source", "fake", "--cache", cache]) != 0
    assert (rows_path.read_bytes(), records_path.read_bytes()) == before
    assert not (folder / "attempts.jsonl").exists()


def test_repeats_uses_original_times_and_reports_speed_metrics(speed) -> None:
    folder, rows, _ = speed
    for row in rows:
        if row["seq"] == 64 and row["arm"] == "C":
            row["step_seconds"] = 10 if (row["run"] - 1) % 12 == 0 else 34
    _write(folder / "rows.jsonl", rows)
    report = analysis.speed_report(folder)
    assert report["blocks_needed"]["(8,64)"]["G_vs_C"] == 130
    assert report["windows"]["(8,64)"]["verdict"] is None


def test_repeats_recomputes_once_after_initial_topup(speed, monkeypatch: pytest.MonkeyPatch):
    folder, rows, _ = speed

    def set_spread(row: dict, half_spread: float) -> None:
        if row["seq"] == 64 and row["arm"] == "C":
            row["step_seconds"] = 22 + (half_spread if (row["run"] - 1) % 12 == 0
                                        else -half_spread)

    for row in rows:
        set_spread(row, 2.7)
    _write(folder / "rows.jsonl", rows)
    recorded = []

    def sweep(args, plan: list[dict]) -> list[dict]:
        output = []
        slot = recorded.count(plan[0]["seq"]) + 5
        recorded.append(plan[0]["seq"])
        for item in plan:
            row, record = _evidence(item)
            # The larger spread in the last top-up must not prompt another one.
            set_spread(row, 4 if slot < 7 else 14)
            output.append(row)
            mm.append_jsonl(folder / "rows.jsonl", row)
            mm.append_jsonl(folder / "arm_records.jsonl", record)
        return output

    monkeypatch.setattr(mm, "sweep", sweep)
    assert replacement.main(["--out", str(folder), "--source", "fake", "--cache", "fake"]) == 0
    report = analysis.speed_report(folder)
    progress = report["repeats"]
    assert (progress["pilot_target"], progress["pilot_blocks_to_add"]) == (7, 2)
    assert (progress["final_target"], progress["topup_blocks_to_add"]) == (9, 2)
    assert recorded.count(64) == recorded.count(512) == 4
    assert all(window["verdict"] == "available" for window in report["windows"].values())


def test_unplanned_reservation_cannot_hide_an_incomplete_block(speed) -> None:
    folder, _, _ = speed
    mm.append_jsonl(folder / "attempts.jsonl", {"k": 8, "seq": 64, "slot": 7, "block": 5})
    window = analysis.speed_report(folder)["windows"]["(8,64)"]
    assert window["unexpected_slots"] == [7]
    assert window["verdict"] is None


def test_valid_comparison_labels_ratio_and_deltas_and_rows(speed) -> None:
    folder, rows, records = speed
    for row in rows:
        if row["seq"] == 64 and row["arm"] == "C" and row["block"] == 1:
            row["step_seconds"] = 26
    _write(folder / "rows.jsonl", rows)
    records[0]["gpu_clock"]["sm_clock_mhz"]["max"] = 1000
    _write(folder / "arm_records.jsonl", records)
    report = analysis.speed_report(folder)
    assert report["gpu_clock_max_min"] == 1000
    window = report["windows"]["(8,64)"]
    delta = window["deltas"]["G_vs_C"]
    assert delta["ratio_of_arm_means"] == pytest.approx(22.8 / 17)
    assert delta["mean_paired_difference_s"] == pytest.approx(5.8)
    assert delta["mean_per_block_time_reduction_pct"] == pytest.approx(25.10, abs=.01)
    lines = analysis._speed_result_lines(report)
    assert "torch 2.14.1+cu130, step 22.80 s" in lines[0]


def test_analyzer_exits_nonzero_and_writes_no_verdict_for_incomplete_acceptance(
    speed, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder, rows, _ = speed
    _write(folder / "rows.jsonl", rows[1:])
    monkeypatch.setattr(analysis, "bits_report", lambda *args: {
        "missing_folders": [], "cells": [], "errors": [], "crosses": [],
    })
    assert analysis.main(["--root", str(folder.parent)]) != 0
    report = json.loads((folder.parent / "analysis.json").read_text())
    assert report["speed"]["windows"]["(8,64)"]["verdict"] is None


def test_ds_refuses_any_above_8_before_sweep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    with pytest.raises(ValueError, match="k.*8"):
        mm.decoupled_segmented_bit_plan(ks=(2, 64))
    monkeypatch.setattr(mm, "sweep", lambda *args: pytest.fail("ran a forbidden A0 sweep"))
    with pytest.raises(SystemExit) as error:
        mm.main(["ds-bits", "--ladder-ks", "2,64", "--source", "s", "--cache", "c",
                 "--out", str(tmp_path)])
    assert error.value.code == 2


@pytest.mark.parametrize("kind", ["ceg-speed", "lever-bits", "ds-bits"])
def test_required_sweep_propagates_gate_refusal_before_first_row(
    tmp_path: Path, kind: str, capsys: pytest.CaptureFixture,
):
    marker = tmp_path / "gate-ran"
    refuse = tmp_path / "refuse_gate.py"
    refuse.write_text(f"from pathlib import Path\n"
                      f"Path({str(marker)!r}).write_text('ran')\n"
                      f"raise SystemExit(1)\n", encoding="utf-8")
    out = tmp_path / kind
    assert mm.main([kind, "--source", "s", "--cache", "c", "--out", str(out),
                    "--gate", sys.executable, str(refuse), "--require-complete"]) == 1
    assert marker.read_text(encoding="utf-8") == "ran"
    assert "gate refused run 1 of" in capsys.readouterr().out
    assert not (out / "rows.jsonl").exists()


def test_required_bits_sweep_does_not_publish_a_verdict_without_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(mm, "sweep", lambda args, plan: [
        dict(item, status="ok", outcome="ok", grad_sha256="same", loss=1.0,
             step_seconds=1.0, tensor_file=None)
        for item in plan
    ])
    assert mm.main(["ds-bits", "--source", "s", "--cache", "c", "--out", str(tmp_path),
                    "--require-complete"]) == 1
    assert "Bit run unsuccessful" in capsys.readouterr().err
    assert not (tmp_path / "bits-verdict.json").exists()


@needs_launcher
@pytest.mark.parametrize(
    "mode,defect",
    [
        ("speed", "no-inhibit"), ("bits", "no-inhibit"),
        ("speed", "gate"), ("bits", "gate"),
        ("speed", "sweep"), ("bits", "sweep"),
        ("speed", "no-output"), ("bits", "no-output"),
        ("speed", "replacement"), ("speed", "archived"), ("bits", "archived"),
    ],
)
def test_launchers_preserve_failures_without_gpu(tmp_path: Path, mode: str, defect: str):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("dirname", "mkdir", "flock", "date", "tee"):
        (bindir / name).symlink_to(shutil.which(name))
    cat = bindir / "cat"
    cat.write_text("#!/bin/bash\necho 1\n")
    cat.chmod(0o755)
    if defect != "no-inhibit":
        inhibitor = bindir / "systemd-inhibit"
        inhibitor.write_text('#!/bin/bash\nshift 2\nexec "$@"\n')
        inhibitor.chmod(0o755)
    interp = tmp_path / "py dir" / "python"
    interp.parent.mkdir()
    interp.symlink_to(sys.executable)
    mock = tmp_path / "mock_python.py"
    mock.write_text(
        """import os, sys
from pathlib import Path
if sys.argv[1] == '-c':
    print('/mock/cu13')
    sys.exit(0)
with open(os.environ['MOCK_LOG'], 'a') as log:
    log.write(sys.argv[2] + '\\n')
defect = os.environ['MOCK_DEFECT']
replacing = sys.argv[2].endswith('replace_void.py')
if defect == 'gate':
    sys.exit(42)
if defect == 'replacement' and replacing:
    sys.exit(43)
if defect == 'no-output' or replacing:
    sys.exit(0)
out = Path(sys.argv[sys.argv.index('--out') + 1])
out.mkdir(parents=True, exist_ok=True)
(out / 'rows.jsonl').write_text('{}\\n')
(out / 'arm_records.jsonl').write_text('{}\\n')
sys.exit(41 if defect == 'sweep' else 0)
""",
    )
    # A bash wrapper keeps the shebang short and free of spaces, whatever the interpreter path.
    python = bindir / "python"
    python.write_text(f'#!/bin/bash\nexec {shlex.quote(str(interp))} '
                      f'{shlex.quote(str(mock))} "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    out, log = tmp_path / "campaign", tmp_path / "mock.log"
    if defect == "archived":
        data = out / ("speed" if mode == "speed" else "bits/s64")
        data.mkdir(parents=True)
        (data / "rows.jsonl").write_text("original\n")
    env = dict(os.environ, PATH=str(bindir), VENV=str(tmp_path), OUT=str(out),
               MOCK_LOG=str(log), MOCK_DEFECT=defect, START_BLOCK="1")
    proc = subprocess.run([shutil.which("bash"), str(CAMPAIGN / f"run_{mode}.sh")], env=env,
                          capture_output=True, text=True)
    assert proc.returncode != 0, proc.stdout
    if defect in ("no-inhibit", "archived"):
        assert not log.exists()
    elif defect != "replacement":
        assert len(log.read_text().splitlines()) == 1
    if defect == "archived":
        assert (data / "rows.jsonl").read_text() == "original\n"


@pytest.fixture(scope="module")
def full_grads() -> dict:
    return _grads()


@pytest.mark.parametrize("gap", [None, "215", "zero", "name", "shape", "dtype", "loss",
                                 "loss-length", "missing-file", "no-digest", "bad-digest"])
def test_bit_equality_requires_authoritative_tensor_coverage(
    tmp_path: Path, full_grads: dict, gap: str | None,
):
    import torch

    grads = dict(full_grads)
    first = next(iter(grads))
    loss = torch.tensor([1.25, 1.5], dtype=torch.float64)
    if gap == "215":
        del grads[first]
    elif gap == "zero":
        grads = {}
    elif gap == "shape":
        grads[first] = grads[first].reshape(-1)
    elif gap == "dtype":
        grads[first] = grads[first].double()
    elif gap == "name":
        grads["unexpected.weight"] = grads.pop(first)
    payload = {"loss": loss, "grads": grads}
    if gap == "loss":
        del payload["loss"]
    elif gap == "loss-length":
        payload["loss"] = loss[:1]
    left, right = tmp_path / "left.pt", tmp_path / "right.pt"
    torch.save(_payload(full_grads), left)
    if gap != "missing-file":
        torch.save(payload, right)
    honest = _row(_payload(full_grads), analysis.GRAD_SHAPES)
    right_row = dict(honest)
    if gap == "no-digest":
        del right_row["grad_sha256"]
    if gap == "bad-digest":
        right_row["grad_sha256"] = "0" * 64
    report = analysis.compare_tensors(left, right, k=2, left_row=honest, right_row=right_row)
    assert report["verdict"] == ("equal" if gap is None else None)
    if gap in ("215", "name"):
        assert first in report["coverage"]["right"]["missing_grad_names"]
    if gap == "name":
        assert report["coverage"]["right"]["unexpected_grad_names"] == ["unexpected.weight"]
    if gap == "missing-file":
        assert "right:missing_tensor_file" in report["reasons"]
    if gap == "no-digest":
        assert "right:missing_digest" in report["reasons"]
    if gap == "bad-digest":
        assert "right:digest_mismatch" in report["reasons"]
    if gap is None:
        assert report["grad_total"] == 216 and report["loss_equal"] is True


@pytest.mark.parametrize("case", ["valid-equal", "valid-differs", "missing-digest",
                                  "digest-mismatch", "full-shape-wrong-values"])
def test_compare_binds_each_payload_to_its_row_digest(
    tmp_path: Path, full_grads: dict, case: str,
):
    import torch

    altered = dict(full_grads)
    first = next(iter(altered))
    altered[first] = altered[first] + 1.0
    honest = _row(_payload(full_grads), analysis.GRAD_SHAPES)
    right_row = dict(honest)
    if case == "valid-differs":
        right_payload = _payload(altered)
        right_row = _row(right_payload, analysis.GRAD_SHAPES)
    elif case == "full-shape-wrong-values":
        right_payload = _payload(altered)
    else:
        right_payload = _payload(full_grads)
        if case == "missing-digest":
            del right_row["grad_sha256"]
        if case == "digest-mismatch":
            right_row["grad_sha256"] = "0" * 64
    left, right = tmp_path / "left.pt", tmp_path / "right.pt"
    torch.save(_payload(full_grads), left)
    torch.save(right_payload, right)
    report = analysis.compare_tensors(left, right, k=2, left_row=honest, right_row=right_row)
    if case == "valid-equal":
        assert report["verdict"] == "equal"
    elif case == "valid-differs":
        assert report["verdict"] == "differs" and report["grads_differing"] == 1
    else:
        assert report["verdict"] is None and report["loss_equal"] is None
        expected = "missing_digest" if case == "missing-digest" else "digest_mismatch"
        assert f"right:{expected}" in report["reasons"]


@pytest.mark.parametrize("case", ["valid-equal", "missing-digest", "digest-mismatch",
                                  "full-shape-wrong-values"])
def test_bit_cell_publishes_equal_only_when_bound_to_its_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, full_grads: dict, case: str,
):
    import torch

    path = _bit_folder(tmp_path, full_grads)
    rows = _read_rows(path / "rows.jsonl")
    if case == "missing-digest":
        del rows[1]["grad_sha256"]
    elif case == "digest-mismatch":
        rows[1]["grad_sha256"] = "0" * 64
    elif case == "full-shape-wrong-values":
        altered = dict(full_grads)
        first = next(iter(altered))
        altered[first] = altered[first] + 1.0
        torch.save(_payload(altered), path / "tensors" / rows[1]["tensor_file"])
    _write(path / "rows.jsonl", rows)
    _isolate(monkeypatch, tmp_path)
    cell = analysis.bits_report(tmp_path, tmp_path / "no-stored", IDENTITY)["cells"][0]
    if case == "valid-equal":
        assert cell["verdict"] == "equal" and cell["reasons"] == []
    else:
        assert cell["verdict"] is None
        expected = "missing_digest" if case == "missing-digest" else "digest_mismatch"
        assert any(expected in reason for reason in cell["reasons"])


@pytest.mark.parametrize("field,reasons", [
    ("dataset_sha256", ["DS:dataset_sha256"]),
    ("tensor_file", ["DS:tensor_file"]),
    ("environment", ["DS:environment.torch", "torch_version_mismatch"]),
    ("loss", ["DS:loss"]),
])
def test_bit_metadata_gaps_are_reported_without_equality(
    tmp_path: Path, field: str, reasons: list[str], monkeypatch: pytest.MonkeyPatch,
    full_grads: dict,
):
    path = _bit_folder(tmp_path, full_grads)
    rows = _read_rows(path / "rows.jsonl")
    del rows[1][field]
    _write(path / "rows.jsonl", rows)
    _isolate(monkeypatch, tmp_path)
    report = analysis.bits_report(tmp_path, tmp_path / "no-stored", IDENTITY)
    assert report["cells"][0]["verdict"] is None
    assert report["cells"][0]["reasons"] == reasons


@pytest.mark.parametrize("index,value,reasons", [
    (1, None, ["DS:campaign_id"]),
    (0, "foreign", ["A0:campaign_id"]),
])
def test_bit_rows_need_the_expected_campaign_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, full_grads: dict,
    index: int, value: str | None, reasons: list[str],
):
    path = _bit_folder(tmp_path, full_grads)
    rows = _read_rows(path / "rows.jsonl")
    if value is None:
        del rows[index]["campaign_id"]
    else:
        rows[index]["campaign_id"] = value
    _write(path / "rows.jsonl", rows)
    _isolate(monkeypatch, tmp_path)
    cell = analysis.bits_report(tmp_path, tmp_path / "no-stored", IDENTITY)["cells"][0]
    assert cell["verdict"] is None
    assert cell["reasons"] == reasons


def test_a_missing_stored_cross_fails_the_analyzer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, full_grads: dict,
) -> None:
    path = _bit_folder(tmp_path, full_grads)
    expected = analysis.expected_campaign_id("fake", "fake")
    rows = _read_rows(path / "rows.jsonl")
    for row in rows:
        row["campaign_id"] = expected
    _write(path / "rows.jsonl", rows)
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(analysis, "speed_report",
                        lambda *args: {"windows": {}, "errors": []})
    empty = tmp_path / "empty-stored"
    empty.mkdir()
    argv = ["--root", str(tmp_path), "--stored-root", str(empty), "--source", "fake",
            "--cache", "fake"]
    assert analysis.main(argv) == 1


@pytest.mark.parametrize("case,verdict", [
    ("complete", "equal"),
    ("missing-one", "no verdict: row:invalid_gradient_coverage"),
    ("empty-grads", "no verdict: row:invalid_gradient_coverage"),
    ("digest-mismatch", "no verdict: row:digest_mismatch"),
    ("missing-digest", "no verdict: row:missing_digest"),
])
def test_launcher_verdict_publishes_equal_only_for_complete_bound_tensors(
    tmp_path: Path, case: str, verdict: str,
):
    import torch

    full = _grads(SMALL_SHAPES)
    arm_grads = dict(full)
    if case == "missing-one":
        del arm_grads["b.lora_A.weight"]
    elif case == "empty-grads":
        arm_grads = {}
    elif case == "digest-mismatch":
        arm_grads = {name: grad + 1 for name, grad in full.items()}
    # The row declares the digest of the grads its file holds, except for a digest mismatch.
    declared = full if case == "digest-mismatch" else arm_grads
    base_row = dict(_row(_payload(full, (1.25,)), SMALL_SHAPES), arm="A0", k=1, seq=64,
                    status="ok", tensor_file="A0-k1-s64.pt")
    arm_row = dict(_row(_payload(declared, (1.25,)), SMALL_SHAPES), arm="E", k=1, seq=64,
                   status="ok", tensor_file="E-k1-s64.pt")
    if case == "missing-digest":
        del arm_row["grad_sha256"]
    torch.save(_payload(full, (1.25,)), tmp_path / "A0-k1-s64.pt")
    torch.save(_payload(arm_grads, (1.25,)), tmp_path / "E-k1-s64.pt")
    result = mm.tensor_ladder_verdict([base_row, arm_row], tensor_dir=tmp_path,
                                      manifest=SMALL_SHAPES)[(1, "E")]
    assert result == verdict


def test_tensor_verdict_without_manifest_ignores_self_declared_partial_shapes(
    tmp_path: Path,
) -> None:
    import torch

    full = _grads(SMALL_SHAPES)
    rows = []
    for arm in ("A0", "E"):
        name = mm.tensor_file_name(arm, 1, 64)
        torch.save(_payload(full, (1.25,)), tmp_path / name)
        rows.append(dict(_row(_payload(full, (1.25,)), SMALL_SHAPES), arm=arm, k=1, seq=64,
                         status="ok", tensor_file=name))
    verdict = mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path)[(1, "E")]
    assert verdict.startswith("no verdict: base:no_expected_grads"), verdict


def test_bits_launcher_never_trusts_a_self_declared_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    full = _grads(SMALL_SHAPES)

    def sweep(args, plan: list[dict]) -> list[dict]:
        bit_dir = mm.bit_tensor_dir(args.out)
        bit_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in plan:
            name = mm.tensor_file_name(item["arm"], item["k"], item["seq"])
            payload = _payload(full)
            torch.save(payload, bit_dir / name)
            rows.append(dict(_row(payload, SMALL_SHAPES), **item, status="ok", outcome="ok",
                             tensor_file=name, step_seconds=1.0))
        return rows

    monkeypatch.setattr(mm, "sweep", sweep)
    assert mm.main(["ds-bits", "--source", "s", "--cache", "c", "--out", str(tmp_path)]) == 1
    verdict = json.loads((tmp_path / "bits-verdict.json").read_text(encoding="utf-8"))
    assert verdict["k2-DS"] == ("no verdict: base:invalid_gradient_coverage, "
                                "row:invalid_gradient_coverage"), verdict


def test_empty_manifest_and_empty_grads_never_publish_equal(tmp_path: Path) -> None:
    import torch

    empty = _payload({}, (1.25,))
    torch.save(empty, tmp_path / "A0-k1-s64.pt")
    torch.save(empty, tmp_path / "E-k1-s64.pt")
    rows = [dict(_row(empty, {}), arm=arm, k=1, seq=64, status="ok", tensor_file=f"{arm}-k1-s64.pt")
            for arm in ("A0", "E")]
    assert mm.tensor_ladder_verdict(rows, tensor_dir=tmp_path)[(1, "E")].startswith("no verdict")


def test_required_bits_sweep_fails_when_a_verdict_has_no_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    def sweep(args, plan: list[dict]) -> list[dict]:
        rows = []
        mm.bit_tensor_dir(args.out).mkdir(parents=True, exist_ok=True)
        for item in plan:
            name = mm.tensor_file_name(item["arm"], item["k"], item["seq"])
            torch.save(_payload({}, (1.25,)), mm.bit_tensor_dir(args.out) / name)
            rows.append(dict(item, status="ok", outcome="ok", tensor_file=name, loss=1.25,
                             grad_sha256=mm.grad_digest({}), grad_shapes={}, step_seconds=1.0))
        return rows

    monkeypatch.setattr(mm, "sweep", sweep)
    assert mm.main(["ds-bits", "--source", "s", "--cache", "c", "--out", str(tmp_path),
                    "--require-complete"]) != 0
    verdict = json.loads((tmp_path / "bits-verdict.json").read_text(encoding="utf-8"))
    assert verdict == {"k2-DS": verdict["k2-DS"]} and verdict["k2-DS"].startswith("no verdict")


@pytest.mark.parametrize("ks,unbound", [("2", "no A0"), ("16", "no C")])
def test_required_bits_sweep_refuses_every_unbound_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ks: str, unbound: str,
) -> None:
    import torch

    full = _grads(SMALL_SHAPES)

    def sweep(args, plan: list[dict]) -> list[dict]:
        bit_dir = mm.bit_tensor_dir(args.out)
        bit_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in plan:
            name = mm.tensor_file_name(item["arm"], item["k"], item["seq"])
            payload = _payload(full, (1.0,) * item["k"])
            torch.save(payload, bit_dir / name)
            rows.append(dict(_row(payload, SMALL_SHAPES), **item, status="ok", outcome="ok",
                             tensor_file=name, step_seconds=1.0))
        return rows

    monkeypatch.setattr(mm, "sweep", sweep)
    argv = ["lever-bits", "--source", "s", "--cache", "c", "--out", str(tmp_path),
            "--seq", "64", "--ladder-ks", ks, "--arms", "E", "--require-complete"]
    assert mm.main(argv) != 0
    verdict = json.loads((tmp_path / "bits-verdict.json").read_text(encoding="utf-8"))
    assert verdict == {f"k{ks}-E": unbound}


def test_grad_digest_serialization_is_pinned() -> None:
    assert mm.grad_digest({"b": [1.0, -0.0], "a": [0.5]}) == (
        "844d398a0c084b3c22ddac0b7a6457fb757b9484d2c411302bb315853bbf4105")


@pytest.mark.parametrize("damage", ["torn-tail", "unterminated-tail", "malformed-middle"])
def test_resume_refuses_torn_or_malformed_jsonl_and_keeps_bytes(
    speed, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    folder, rows, _ = speed
    lines = [json.dumps(row) + "\n" for row in rows]
    if damage == "torn-tail":
        lines[-1] = lines[-1][:-20]
    elif damage == "unterminated-tail":
        lines[-1] = lines[-1].rstrip("\n")
    else:
        lines.insert(len(lines) // 2, '{"arm": "C", "k"\n')
    rows_path, records_path = folder / "rows.jsonl", folder / "arm_records.jsonl"
    rows_path.write_text("".join(lines), encoding="utf-8")
    before = rows_path.read_bytes(), records_path.read_bytes()
    monkeypatch.setattr(mm, "sweep", lambda *args: pytest.fail("resumed over torn evidence"))
    assert replacement.main(["--out", str(folder), "--source", "fake", "--cache", "fake"]) != 0
    assert (rows_path.read_bytes(), records_path.read_bytes()) == before
    assert not (folder / "attempts.jsonl").exists()
    assert "unreadable:rows.jsonl" in analysis.read_speed(folder).errors
    assert analysis.speed_report(folder)["windows"]["(8,64)"]["verdict"] is None


def _readme_section(heading: str) -> str:
    text = (CAMPAIGN / "README.md").read_text(encoding="utf-8")
    assert f"## {heading}" in text
    body = text.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]
    return " ".join(body.split())


def test_resume_contract_documents_the_exact_refusal_and_manual_recovery() -> None:
    section = _readme_section("Resume contract")
    assert ("A final record without its trailing newline (a torn record, or a complete one "
            "that was never terminated) or a malformed record anywhere else makes "
            "`rows.jsonl` or "
            "`arm_records.jsonl` unreadable. `replace_void.py` then exits nonzero before any gate, "
            "sweep or write, and the files keep their original bytes.") in section
    assert "The manual recovery runs on a copy only:" in section


def test_limitations_document_that_archived_bit_rows_carry_no_campaign_id() -> None:
    section = _readme_section("Limitations")
    assert ("The archived bit rows carry no `campaign_id`. `analyze.py` refuses every bits row "
            "whose `campaign_id` is missing or differs from the expected one, so archived bits "
            "cells get no verdict and are diagnostic only.") in section


def test_archived_tensor_reads_are_refused_at_runtime() -> None:
    import torch

    archived = CAMPAIGN / "bits" / "s64" / "tensors" / "A0-k1-s64.pt"
    with pytest.raises(AssertionError, match="archived tensor read"):
        torch.load(archived, weights_only=True)
    handle = io.BytesIO(b"")
    handle.name = str(archived)
    with pytest.raises(AssertionError, match="archived tensor read"):
        torch.load(handle, weights_only=True)


def test_a_crash_row_without_child_json_carries_the_campaign_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import argparse
    import subprocess

    from benchmarks.harness import arm_validity

    class Watch:
        def __init__(self, **kwargs) -> None:
            self.record = {"arm": kwargs["arm"], "run": kwargs["run"], "outcome": "ok"}

        def __enter__(self) -> "Watch":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

    monkeypatch.setattr(arm_validity, "ArmWatch", Watch)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(
        command, 1, stdout="", stderr="child died"))
    args = argparse.Namespace(source="fake", cache="fake", tokenizer=None, lora_r=16,
                              lora_targets=mm.DEFAULT_LORA_TARGETS, steps=4, out=str(tmp_path),
                              device="auto", label=mm.DEFAULT_LABEL, deadline_hours=0,
                              gate=None, require_complete=False)
    rows = mm.sweep(args, mm.ceg_speed_plan(blocks=1)[:2])
    assert [row["status"] for row in rows] == ["CRASH", "CRASH"]
    written = _read_rows(tmp_path / "rows.jsonl")
    assert [row["campaign_id"] for row in written] == [IDENTITY, IDENTITY]
    assert "campaign_identity" not in analysis.read_speed(tmp_path, IDENTITY).errors


@pytest.mark.parametrize("child_id,accepted", [(IDENTITY, True), ("foreign", False),
                                                (None, False)])
def test_a_child_campaign_id_must_equal_the_sweep_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child_id: str | None, accepted: bool,
) -> None:
    import argparse
    import subprocess

    from benchmarks.harness import arm_validity

    class Watch:
        def __init__(self, **kwargs) -> None:
            self.record = {"arm": kwargs["arm"], "run": kwargs["run"], "outcome": "ok"}

        def __enter__(self) -> "Watch":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

    child = {"arm": "C", "k": 8, "seq": 64, "kind": "speed", "mode": "normal", "status": "ok",
             "outcome": "ok", **({} if child_id is None else {"campaign_id": child_id})}
    monkeypatch.setattr(arm_validity, "ArmWatch", Watch)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0, stdout=json.dumps(child) + "\n", stderr=""))
    args = argparse.Namespace(source="fake", cache="fake", tokenizer=None, lora_r=16,
                              lora_targets=mm.DEFAULT_LORA_TARGETS, steps=4, out=str(tmp_path),
                              device="auto", label=mm.DEFAULT_LABEL, deadline_hours=0,
                              gate=None, require_complete=False)
    plan = mm.ceg_speed_plan(blocks=1)[:1]
    if accepted:
        rows = mm.sweep(args, plan)
        assert [row["campaign_id"] for row in rows] == [IDENTITY]
    else:
        with pytest.raises(RuntimeError, match="campaign_id mismatch"):
            mm.sweep(args, plan)
        assert not (tmp_path / "rows.jsonl").exists()


def test_a_foreign_arm_record_refuses_replacement_before_gate_or_write(
    speed, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder, rows, records = speed
    records[3]["campaign_id"] = "foreign"
    rows_path, records_path = folder / "rows.jsonl", folder / "arm_records.jsonl"
    _write(rows_path, rows)
    _write(records_path, records)
    before = rows_path.read_bytes(), records_path.read_bytes()
    monkeypatch.setattr(mm, "sweep", lambda *args: pytest.fail("swept a foreign record"))
    assert replacement.main(["--out", str(folder), "--source", "fake", "--cache", "fake"]) != 0
    assert (rows_path.read_bytes(), records_path.read_bytes()) == before
    assert not (folder / "attempts.jsonl").exists()
