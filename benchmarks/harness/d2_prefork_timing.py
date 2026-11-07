"""Run isolated timing from a CUDA-free CPU template to avoid repeated imports.

Each arm still has a new OS process and a new CUDA context. The supervisor must
never initialize CUDA; all models, CUDA RNG and tensors are created by the child.
This saves Python import time only, outside the measured CUDA event regions.
Linux fork only; not a supported training launcher.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import traceback
from pathlib import Path

from rich.console import Console

try:
    from . import d2_isolated_timing as timing
except ImportError:
    import d2_isolated_timing as timing


class ForkArm:
    def __init__(self, command: list[str], *, env: dict, stdout: object, **kwargs: object) -> None:
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError("Refuse fork from a CUDA-initialized supervisor")
        case = Path(command[command.index("--case") + 1])
        arm = Path(command[command.index("--arm") + 1])
        implementation = command[command.index("--implementation") + 1]
        self.args = command
        self.process = multiprocessing.get_context("fork").Process(
            target=self.execute, args=(case, arm, implementation, env, stdout.fileno())
        )
        self.process.start()
        self.pid = self.process.pid

    @staticmethod
    def execute(case: Path, arm: Path, implementation: str, env: dict, log_fd: int) -> None:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.environ.update(env)
        timing.write_json(
            arm / "fork-origin.json",
            {
                "start_method": "fork from CUDA-free CPU supervisor",
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "parent_cuda_initialized": False,
            },
        )
        try:
            timing.child(arm, case, json.loads((case / "config.json").read_text()), implementation)
        except Exception:
            Console().print(traceback.format_exc(), markup=False)
            raise SystemExit(1) from None

    def poll(self) -> int | None:
        return self.process.exitcode

    def wait(self, timeout: float | None = None) -> int:
        self.process.join(timeout)
        if self.process.exitcode is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.process.exitcode

    def kill(self) -> None:
        self.process.kill()


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    import peft  # noqa: F401 -- preloaded in the CUDA-free CPU template
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original = subprocess.Popen

    def launch(command: list[str], *args: object, **kwargs: object) -> object:
        if "--worker" in command and command[command.index("--worker") + 1] == "arm":
            return ForkArm(command, **kwargs)
        return original(command, *args, **kwargs)

    subprocess.Popen = launch
    try:
        return timing.main()
    finally:
        subprocess.Popen = original
        if torch.cuda.is_initialized():
            raise RuntimeError("Supervisor unexpectedly created a CUDA context")


if __name__ == "__main__":
    raise SystemExit(main())
