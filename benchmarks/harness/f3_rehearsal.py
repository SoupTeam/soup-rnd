#!/usr/bin/env python3
"""F3 laptop script: the zero-gradient control and the full rehearsal, on SmolLM2-135M.

    f3_rehearsal.py control      the control in the native trainer and in the Unsloth venv
    f3_rehearsal.py rehearsal    the whole flow: power check, control, audit, lifecycle

The flow is the one the rental runs, with a small model and a local folder in place of the
HuggingFace repo and the vast.ai destroy command. Each run is a fresh process of
``f3_arm.py`` in the venv of its framework, under ``ArmWatch``. Results carry the labels
``"RTX 3050 Laptop, control"`` and ``"RTX 3050 Laptop, rehearsal"`` and never enter a verdict.

Arm B (Chronicals) is recorded out of scope: its ``ChronicalsTrainer`` is a custom loop, not
an HF ``Trainer`` (spec: "Out of scope", custom loops).

Benchmark harness code, not shipped. Run it from a venv with ``soup_cli`` importable (the
CUDA venv with ``PYTHONPATH=src``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

import f3_audit  # noqa: E402
import f3_run  # noqa: E402
import f3_setup  # noqa: E402
from arm_validity import ArmWatch  # noqa: E402

REPO = HARNESS.parents[1]
RESULTS = REPO / "benchmarks" / "results" / "f3-benchmark-audit"
VENVS = Path.home() / ".venvs"
DATA = Path.home() / "f3-data" / "smollm2-135m"
MODEL = "HuggingFaceTB/SmolLM2-135M"
LABEL_CONTROL = "RTX 3050 Laptop, control"
LABEL_REHEARSAL = "RTX 3050 Laptop, rehearsal"
STEPS = 12

#: The venv python that runs each training path.
PYTHONS = {
    "native": VENVS / "soup-rnd-cuda" / "bin" / "python",
    "unsloth": VENVS / "f3-unsloth" / "bin" / "python",
}


def run_arm(path: str, out: Path, *, control: bool, steps: int = STEPS) -> Dict[str, Any]:
    """One fresh process of ``f3_arm.py``; returns its report."""
    command = [
        str(PYTHONS[path]),
        str(HARNESS / "f3_arm.py"),
        "--path", path,
        "--model", MODEL,
        "--tokens", str(DATA / "tokens.json"),
        "--out", str(out),
        "--steps", str(steps),
        "--batch", "1",
    ]  # fmt: skip
    if control:
        command.append("--control")
    subprocess.run(command, check=True, cwd=out.parent)  # noqa: S603
    return json.loads(out.read_text(encoding="utf-8"))


def control_runs(out_dir: Path) -> List[f3_audit.ControlRun]:
    runs = []
    for path in f3_audit.CONTROL_PATHS:
        report = run_arm(path, out_dir / f"control-{path}.json", control=True)
        runs.append(
            f3_audit.ControlRun(path, report, float(report["provenance"]["weight_decay"]))
        )
    return runs


def summarize_control(runs: Sequence[f3_audit.ControlRun]) -> Dict[str, Any]:
    problems = f3_audit.control_problems(runs)
    return {
        "label": LABEL_CONTROL,
        "rejected_in_every_path": not problems,
        "problems": problems,
        "paths": {
            run.path: {
                "valid": run.report.get("valid"),
                "refused": f3_audit._failed_checks(run.report),
                "weight_decay": run.weight_decay,
            }
            for run in runs
        },
    }


def cmd_control() -> int:
    out_dir = RESULTS / "control"
    summary = summarize_control(control_runs(out_dir))
    (out_dir / "control.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["rejected_in_every_path"] else 1


def cmd_rehearsal() -> int:
    if f3_setup.power_cap_problem(_smi_power()) is not None:
        print(f3_setup.power_cap_problem(_smi_power()))
        return 1
    out_dir = RESULTS / "rehearsal"
    out_dir.mkdir(parents=True, exist_ok=True)
    control = control_runs(out_dir)

    specs = [
        f3_audit.ArmSpec("C1", "A-unsloth", "unsloth==2026.1.2", versions={"unsloth": "2026.1.2"}),
        f3_audit.ArmSpec(
            "C1", "B-chronicals", f3_setup.FOREIGN["chronicals"][1],
            uses_trainer=False,
            scope_reason="ChronicalsTrainer is a custom loop, not an HF Trainer",
        ),
    ]  # fmt: skip

    def runner(spec: f3_audit.ArmSpec, process: int) -> Dict[str, Any]:
        path = out_dir / f"{spec.config}-{spec.arm}-p{process}.json"
        return run_arm("unsloth", path, control=False)

    def watch(**kwargs: Any) -> ArmWatch:
        return ArmWatch(label=LABEL_REHEARSAL, model_path=str(DATA), **kwargs)

    audit = f3_audit.audit(specs, runner, watch, control)
    f3_audit.write_refusals(audit.refusals, out_dir / "refusals.jsonl")
    summary = {
        "label": LABEL_REHEARSAL,
        "control": summarize_control(control),
        "verdicts": [vars(v) for v in audit.verdicts],
        "note": "rehearsal numbers do not enter a verdict",
    }
    (out_dir / "audit.json").write_text(json.dumps(summary, indent=2, default=str))

    uploaded = out_dir / "uploaded"
    uploaded.mkdir(exist_ok=True)

    def upload(name: str, path: Path) -> None:
        (uploaded / path.name).write_bytes(path.read_bytes())

    def arm_record(name: str) -> Dict[str, Any]:
        return {"arm": name, "label": LABEL_REHEARSAL, "verdicts": summary["verdicts"]}

    lifecycle = f3_run.run_lifecycle(
        [f3_run.ArmPlan(spec.arm, 60.0) for spec in specs],
        arm_record,
        upload,
        lambda: print("destroy: skipped in the rehearsal"),
        out_dir / "lifecycle",
        rate_per_hour=0.65,
        ceiling=2.50,
        time_limit_seconds=3 * 3600,
    )
    print(json.dumps(summary["verdicts"], indent=2, default=str))
    print("lifecycle:", lifecycle.outcomes, "destroyed:", lifecycle.destroyed)
    return 0


def _smi_power() -> str:
    return subprocess.run(  # noqa: S603
        ["nvidia-smi", "-q", "-d", "POWER"], capture_output=True, text=True, check=False
    ).stdout


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("control", "rehearsal"))
    args = parser.parse_args(argv)
    return cmd_control() if args.stage == "control" else cmd_rehearsal()


if __name__ == "__main__":
    sys.exit(main())
