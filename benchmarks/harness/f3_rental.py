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


def cmd_c1(rate: float) -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    control = [
        f3_audit.ControlRun(
            "unsloth",
            json.loads((WORKDIR / "control-unsloth.json").read_text(encoding="utf-8")),
            0.0,
        )
    ] if (WORKDIR / "control-unsloth.json").exists() else []
    specs = c1_specs()

    def runner(spec: f3_audit.ArmSpec, process: int) -> Dict[str, Any]:
        return run_arm("unsloth", WORKDIR / "c1" / f"{spec.arm}-p{process}.json")

    audit = f3_audit.audit(specs, runner, _watch, control)
    f3_audit.write_refusals(audit.refusals, WORKDIR / "refusals.jsonl")
    verdicts = [vars(v) for v in audit.verdicts]
    (WORKDIR / "audit-c1.json").write_text(
        json.dumps({"label": LABEL, "verdicts": verdicts, "records": audit.records}, indent=2,
                   default=str)
    )

    def arm_record(name: str) -> Dict[str, Any]:
        return {"arm": name, "label": LABEL, "verdicts": verdicts}

    lifecycle = f3_run.run_lifecycle(
        [f3_run.ArmPlan(spec.arm, 0.0) for spec in specs],
        arm_record,
        f3_run.default_upload_factory(os.environ["F3_HF_REPO"]),
        f3_run.default_destroy,
        WORKDIR / "lifecycle",
        rate_per_hour=rate,
        ceiling=CEILING,
        time_limit_seconds=TIME_LIMIT_SECONDS,
    )
    print(json.dumps(verdicts, indent=2, default=str))
    print("lifecycle:", lifecycle.outcomes, "destroyed:", lifecycle.destroyed)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("observe")
    c1 = sub.add_parser("c1")
    c1.add_argument("--rate", type=float, required=True, help="instance price, $ per hour")
    args = parser.parse_args(argv)
    return cmd_observe() if args.cmd == "observe" else cmd_c1(args.rate)


if __name__ == "__main__":
    sys.exit(main())
