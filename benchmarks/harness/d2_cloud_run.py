"""Bounded D2 CUDA evidence run; retains failures and never declares D2 Done.

Invoke with --output-dir outside the source checkout. Does not allocate or stop
cloud resources: the operator must stop the instance before the rental deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seconds", type=int, default=2100)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 2400:
        parser.error("seconds must be in 1..2400; preserve time for downloading and stopping")
    root = Path(__file__).parents[2]
    destination = Path(os.path.realpath(args.output_dir))
    if os.path.commonpath([str(root), str(destination)]) == str(root):
        parser.error("output-dir must be outside the source checkout")
    destination.mkdir(parents=True, exist_ok=False)
    console = Console()
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    started = time.monotonic()
    summary: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": args.seconds,
        "stages": [],
        "D2": "INCOMPLETE pending independent review and full CI",
        "limitations": [
            "synthetic fixtures",
            "no real-checkpoint quality evidence",
            "no Liger/Unsloth timing comparison in this runner",
        ],
    }
    inventory = {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for folder in ("src", "tests", "benchmarks/harness")
        for path in (root / folder).rglob("*.py")
    }
    (destination / "source-manifest.json").write_text(json.dumps(inventory, indent=2))

    def pack() -> None:
        (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        archive = destination.parent / f"{destination.name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in destination.rglob("*"):
                if path.is_file():
                    bundle.write(path, str(path.relative_to(destination)))
        (destination.parent / f"{destination.name}.sha256").write_text(
            hashlib.sha256(archive.read_bytes()).hexdigest() + "\n"
        )

    def stage(name: str, command: list[str], maximum: int = 480) -> int:
        remaining = args.seconds - (time.monotonic() - started)
        if remaining < 30:
            summary["stages"].append({"name": name, "status": "NOT RUN: deadline"})
            pack()
            return -1
        command_file = destination / f"{name}-command.json"
        command_file.write_text(json.dumps({"argv": command, "cwd": str(root)}, indent=2))
        before = time.monotonic()
        console.print(f"Starting {name}")
        with (destination / f"{name}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            try:
                code = child.wait(timeout=min(maximum, remaining - 15))
                status = "EXITED"
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    child.kill()
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
                status = "TIMEOUT"
        summary["stages"].append(
            {
                "name": name,
                "exit_code": code,
                "status": status,
                "elapsed_seconds": time.monotonic() - before,
            }
        )
        pack()
        console.print(f"Finished {name}: {code} ({status})")
        return code

    python = sys.executable
    harness = str(root / "benchmarks/harness/fast_lora_probe.py")
    stage("packages", [python, "-m", "pip", "freeze"], 30)
    stage("nvidia-smi", ["nvidia-smi", "-q"], 30)
    preflight = stage(
        "preflight",
        [
            python,
            "-c",
            "import torch,triton; from rich.console import Console; "
            "c=Console(); c.print(torch.__version__,triton.__version__); "
            "assert torch.cuda.device_count()==1; "
            "assert torch.cuda.is_bf16_supported(including_emulation=False); "
            "x=torch.ones(32,32,device='cuda',dtype=torch.bfloat16); "
            "c.print(torch.cuda.get_device_name(),torch.cuda.get_device_capability(),(x@x).sum())",
        ],
        60,
    )
    if preflight:
        summary["error"] = "GPU preflight failed; no acceptance or timing claimed"
        pack()
        return 1
    try:
        for dtype in ("bf16", "fp16", "fp32"):
            for request_dx in (True, False):
                label = f"parity-{dtype}-dx{int(request_dx)}"
                stage(
                    label,
                    [
                        python,
                        harness,
                        "parity",
                        "--device",
                        "cuda:0",
                        "--dtype",
                        dtype,
                        "--skip-gradcheck",
                        "--tensor-dir",
                        str(destination / label),
                        "--output-prefix",
                        str(destination / label),
                    ]
                    + ([] if request_dx else ["--no-dx"]),
                )
            stage(
                f"strict50-{dtype}",
                [
                    python,
                    harness,
                    "loss",
                    "--device",
                    "cuda:0",
                    "--dtype",
                    dtype,
                    "--steps",
                    "50",
                    "--output-prefix",
                    str(destination / f"strict50-{dtype}"),
                ],
            )
        for dtype in ("bf16", "fp16"):
            for kind in ("nf4", "stream"):
                stage(
                    f"{kind}-{dtype}",
                    [
                        python,
                        str(root / f"benchmarks/harness/d2_{kind}_probe.py"),
                        "--device",
                        "cuda:0",
                        "--dtype",
                        dtype,
                        "--output-dir",
                        str(destination / f"{kind}-{dtype}"),
                    ],
                )
        stage(
            "gpu-regressions",
            [
                python,
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "tests/test_issue839_fast_lora_single_projection.py",
                "tests/test_issue838_fast_lora_qkv.py",
                "tests/test_issue837_fast_lora_mlp.py",
                "tests/test_d2_checkpoint_and_dtypes.py",
                "tests/test_issue842_checkpoint_visible_nf4.py",
                "-m",
                "gpu",
                "-k",
                "not report_only",
                "-q",
                "--junitxml",
                str(destination / "gpu-regressions.xml"),
            ],
        )
        for dtype in ("bf16", "fp16"):
            stage(
                f"timing-{dtype}",
                [
                    python,
                    harness,
                    "timing",
                    "--device",
                    "cuda:0",
                    "--dtype",
                    dtype,
                    "--shapes",
                    "llama3.1-8b",
                    "--tokens",
                    "128",
                    "--warmup",
                    "5",
                    "--repeats",
                    "10",
                    "--output-prefix",
                    str(destination / f"timing-{dtype}"),
                ],
                600,
            )
    finally:
        summary["ended_utc"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_seconds"] = time.monotonic() - started
        pack()
    console.print(f"Evidence: {destination.parent / (destination.name + '.zip')}")
    return 0 if all(row.get("exit_code") == 0 for row in summary["stages"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
