#!/usr/bin/env python3
"""F3 rental script, run on the vast.ai instance: the sleep observation and config C1.

    f3_rental.py observe     ArmWatch around sleep(60), record which checks answer
    f3_rental.py c1          arm A (Unsloth) twice in fresh processes, then upload and destroy

C1 is full fine-tuning of Qwen2.5-0.5B at batch 16, sequence 512, on Alpaca-cleaned. Arm B
(Chronicals) is out of scope: its loop is custom, not an HF ``Trainer``. Results carry the
label ``"A100-PCIE-80GB vast.ai, real"``. The upload goes to the private repo named in
``F3_HF_REPO`` with the token in ``HF_TOKEN``; the destroy uses the per-instance key.

Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

import f3_audit  # noqa: E402
import f3_run  # noqa: E402
import f3_setup  # noqa: E402
from arm_validity import ArmWatch  # noqa: E402

LABEL = "A100-PCIE-80GB vast.ai, real"
MODEL = "Qwen/Qwen2.5-0.5B"
BATCH = 16
STEPS = 20
CEILING = 2.50
TIME_LIMIT_SECONDS = 3 * 3600
VENVS = Path.home() / ".venvs"
DATA = Path.home() / "f3-data" / "qwen2.5-0.5b"
WORKDIR = Path.home() / "f3-results"
#: Written by the bootstrap when the lease begins, so the spend counts from the rental.
LEASE_FILE = Path.home() / "f3-lease-start"
#: Expected seconds of one arm (two fresh processes), for the budget projection.
ARM_SECONDS = {"A-unsloth": 300.0}

PYTHONS = {"unsloth": VENVS / "f3-unsloth" / "bin" / "python"}


def arm_command(path: str, out: Path, steps: int = STEPS, control: bool = False) -> List[str]:
    command = [
        str(PYTHONS[path]),
        str(HARNESS / "f3_arm.py"),
        "--path", path,
        "--model", MODEL,
        "--tokens", str(DATA / "tokens.json"),
        "--out", str(out),
        "--steps", str(steps),
        "--batch", str(BATCH),
    ]  # fmt: skip
    if control:
        command.append("--control")
    return command


def run_arm(path: str, out: Path, steps: int = STEPS) -> Dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(arm_command(path, out, steps), check=True, cwd=out.parent)  # noqa: S603
    return json.loads(out.read_text(encoding="utf-8"))


def c1_specs() -> List[f3_audit.ArmSpec]:
    return [
        f3_audit.ArmSpec("C1", "A-unsloth", "unsloth==2026.1.2", versions={"unsloth": "2026.1.2"}),
        f3_audit.ArmSpec(
            "C1", "B-chronicals", f3_setup.FOREIGN["chronicals"][1],
            uses_trainer=False,
            scope_reason="ChronicalsTrainer is a custom loop, not an HF Trainer",
        ),
    ]  # fmt: skip


def observe_sleep(
    watch: Callable[..., Any], sleep: Callable[[float], None] = time.sleep, seconds: float = 60
) -> Dict[str, Any]:
    """Watch a bare sleep and record which checks answer on this machine."""
    with watch(arm="observe", round=1, run=1, required=()) as watched:
        sleep(seconds)
    record = watched.record
    checks = (record or {}).get("checks", {})
    answering = [n for n, c in checks.items() if c["outcome"] != "unknown"]
    return {
        "label": LABEL,
        "outcome": (record or {}).get("outcome", "unknown"),
        "answering": answering,
        "not_answering": [n for n in checks if n not in answering],
        "record": record,
    }


def _watch(**kwargs: Any) -> ArmWatch:
    return ArmWatch(label=LABEL, model_path=str(DATA), **kwargs)


def cmd_observe() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    result = observe_sleep(_watch)
    (WORKDIR / "observe-sleep60.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "record"}, indent=2))
    return 0


def _load_controls() -> List[f3_audit.ControlRun]:
    control = []
    for path in f3_audit.CONTROL_PATHS:
        file = WORKDIR / f"control-{path}.json"
        if file.exists():
            report = json.loads(file.read_text(encoding="utf-8"))
            control.append(
                f3_audit.ControlRun(path, report, float(report["provenance"]["weight_decay"]))
            )
    return control


def _lease_start() -> Optional[float]:
    """Unix time the bootstrap wrote when the paid lease began, if it ran."""
    if LEASE_FILE.exists():
        return float(LEASE_FILE.read_text(encoding="utf-8").strip())
    return None


def _evidence(specs: Sequence[f3_audit.ArmSpec], logs: Sequence[Path]) -> List[Any]:
    """Write ``audit-c1.json`` and ``refusals.jsonl`` from the per-arm records, then list every
    raw file to upload: reports, watch records, controls, refusals and logs."""
    verdicts: List[Any] = []
    records: List[Any] = []
    refusals: List[Any] = []
    arms: Dict[str, str] = {}
    for spec in specs:
        file = WORKDIR / "lifecycle" / f"{spec.arm}.json"
        if not file.exists():
            arms[spec.arm] = "not run"
            continue
        record = json.loads(file.read_text(encoding="utf-8"))
        arms[spec.arm] = record.get("status", "unknown")
        verdicts += record.get("verdicts", [])
        records += record.get("records", [])
        refusals += record.get("refusals", [])
    (WORKDIR / "audit-c1.json").write_text(
        json.dumps({"label": LABEL, "arms": arms, "verdicts": verdicts, "records": records},
                   indent=2, default=str)
    )  # fmt: skip
    f3_audit.write_refusals(refusals, WORKDIR / "refusals.jsonl")
    files = []
    for path in sorted(WORKDIR.rglob("*")):
        relative = path.relative_to(WORKDIR)
        if path.is_file() and path.suffix in (".json", ".jsonl") and not (
            {"lifecycle", "trainer"} & set(relative.parts)
        ):
            files.append((relative.as_posix(), path))
    files += [(Path(log).name, Path(log)) for log in logs if Path(log).exists()]
    return files


def cmd_c1(
    rate: float, lease_start: Optional[float] = None, logs: Sequence[Path] = ()
) -> int:
    """Run C1 inside the lifecycle: the clock, the budget check and the resume state come first,
    each arm trains under them, and the raw evidence uploads before the instance is destroyed."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    specs = c1_specs()
    by_name = {spec.arm: spec for spec in specs}

    def runner(spec: f3_audit.ArmSpec, process: int) -> Dict[str, Any]:
        return run_arm("unsloth", WORKDIR / "c1" / f"{spec.arm}-p{process}.json")

    def audit_arm(name: str) -> Dict[str, Any]:
        audit = f3_audit.audit([by_name[name]], runner, _watch, _load_controls())
        record = {
            "arm": name, "label": LABEL, "verdicts": [vars(v) for v in audit.verdicts],
            "records": audit.records, "refusals": audit.refusals,
        }  # fmt: skip
        return json.loads(json.dumps(record, default=str))

    lifecycle = f3_run.run_lifecycle(
        [f3_run.ArmPlan(spec.arm, ARM_SECONDS.get(spec.arm, 0.0)) for spec in specs],
        audit_arm,
        f3_run.default_upload_factory(os.environ["F3_HF_REPO"]),
        f3_run.default_destroy,
        WORKDIR / "lifecycle",
        rate_per_hour=rate,
        ceiling=CEILING,
        time_limit_seconds=TIME_LIMIT_SECONDS,
        started=lease_start if lease_start is not None else _lease_start(),
        evidence=lambda: _evidence(specs, logs),
    )
    print("lifecycle:", lifecycle.outcomes, "destroyed:", lifecycle.destroyed)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("observe")
    c1 = sub.add_parser("c1")
    c1.add_argument("--rate", type=float, required=True, help="instance price, $ per hour")
    c1.add_argument("--lease-start", type=float, help="Unix time the paid lease began")
    c1.add_argument("--log", type=Path, action="append", default=[], help="log file to upload")
    args = parser.parse_args(argv)
    if args.cmd == "observe":
        return cmd_observe()
    return cmd_c1(args.rate, args.lease_start, args.log)


if __name__ == "__main__":
    sys.exit(main())
