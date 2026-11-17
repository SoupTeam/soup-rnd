"""External wall-time supervisor for one explicit G6 command; never retries."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from run import write
from run_support import contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "cpu-smoke", "cuda-gate", "pilot", "full"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args, rest = parser.parse_known_args()
    spec = contract(args.contract, args.contract_sha)
    if args.out.exists():
        parser.error("Existing output is preserved; no retry or resume")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).with_name("cuda_gate.py" if args.mode == "cuda-gate" else "run.py")
    command = [sys.executable, str(code)]
    if args.mode != "cuda-gate":
        command += [args.mode]
    command += [
        "--contract",
        str(args.contract),
        "--contract-sha",
        args.contract_sha,
        "--out",
        str(args.out),
        *rest,
    ]
    bound = spec["limits"][args.mode + "_total_s"]
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with Path(str(args.out) + ".launcher.log").open("x") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
        timeout = False
        try:
            status = child.wait(timeout=bound + 30)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            status = child.returncode
        write(
            Path(str(args.out) + ".launcher.json"),
            {
                "exitcode": status,
                "hard_timeout": timeout,
                "bound_s": bound,
                "no_retry": True,
                "contract_sha256": args.contract_sha,
            },
        )
        if status or timeout:
            raise SystemExit(status or 1)


if __name__ == "__main__":
    main()
