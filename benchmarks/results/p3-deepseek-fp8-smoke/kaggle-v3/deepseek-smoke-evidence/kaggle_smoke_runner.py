"""Bounded Kaggle qualification of the tiny DeepSeek single-candidate smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

from rich.console import Console

BASE = "54be201337cb7ac8ba5e58ba9ff5b55ac7b0bc81"
DEPENDENCIES = (
    "torch==2.11.0", "transformers==5.19.0", "peft==0.21.2", "accelerate==1.15.0",
    "bitsandbytes==0.50.2", "safetensors==0.8.0", "rich==13.9.4", "pydantic==2.14.1",
    "numpy==2.1.3", "ruff==0.17.4", "pytest==8.4.2", "psutil==7.2.2", "pyyaml==6.0.3",
)
FILES = (
    "benchmarks/harness/p3_deepseek_oracle.py",
    "benchmarks/harness/p3_deepseek_fp8_smoke.py",
    "tests/test_p3_deepseek_oracle.py",
    "tests/test_p3_deepseek_fp8_smoke.py",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, name):
    candidate = root / name
    if os.path.commonpath((os.path.realpath(root), os.path.realpath(candidate))) != (
        os.path.realpath(root)
    ):
        raise ValueError("Path escapes owned directory: " + name)
    return candidate


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    console = Console()
    started = time.monotonic()
    deadline = started + 1800
    owned = None
    record = {"status": "STARTED", "commands": [], "cleanup_complete": False,
              "scope": "Synthetic tiny DeepSeek; single NF4/BF16/FP8-Oracle candidate",
              "baseline_arm": False, "full_model_acceptance": False}

    def command(argv, name, env=None, cwd=None, timeout=300):
        remaining = min(timeout, deadline - time.monotonic() - 60)
        if remaining <= 0:
            raise TimeoutError("Controller deadline before " + name)
        process = None
        begun = time.monotonic()
        interrupted = None
        try:
            with (args.output / (name + ".log")).open("x", encoding="utf-8") as log:
                mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGALRM}
                )
                try:
                    launcher = (
                        "import os,signal,sys;"
                        "signal.pthread_sigmask(signal.SIG_SETMASK,[]);"
                        "os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
                    )
                    process = subprocess.Popen(
                        [sys.executable, "-c", launcher, *map(str, argv)], cwd=cwd, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                process.wait(timeout=remaining)
        except BaseException as error:
            interrupted = type(error).__name__
            raise
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            receipt = {"argv": list(map(str, argv)), "name": name,
                       "seconds": time.monotonic() - begun, "interruption": interrupted,
                       "returncode": None if process is None else process.returncode,
                       "direct_child_reaped": (
                           process is not None and process.returncode is not None
                       )}
            record["commands"].append(receipt)
            save(args.output / "run.json", record)
        if receipt["returncode"] != 0:
            raise RuntimeError("Command failed: " + name)

    def terminate(signum, _frame):
        raise KeyboardInterrupt("Controller received signal " + str(signum))

    def timed_out(_signum, _frame):
        record["controller_deadline_triggered"] = True
        signal.setitimer(signal.ITIMER_REAL, 60)
        raise TimeoutError("Controller lifetime deadline reached")

    previous_term = signal.signal(signal.SIGTERM, terminate)
    previous_alarm = signal.signal(signal.SIGALRM, timed_out)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 1740)
    try:
        if digest(args.admission) != args.admission_sha256:
            raise ValueError("Admission checksum mismatch")
        admission = json.loads(args.admission.read_bytes())
        if admission["schema"] != "p3-deepseek-single-smoke-v1" or admission["base"] != BASE:
            raise ValueError("Unexpected admission schema/base")
        if admission["controller_limit_seconds"] != 1800:
            raise ValueError("Expected the short 1800-second controller limit")
        if digest(args.overlay) != admission["source_overlay_sha256"]:
            raise ValueError("Overlay checksum mismatch")
        if digest(Path(__file__)) != admission["controller_sha256"]:
            raise ValueError("Controller checksum mismatch")
        for source, name in ((args.admission, "execution-admission.json"),
                             (args.overlay, "source-overlay.zip"),
                             (Path(__file__), "kaggle_smoke_runner.py")):
            shutil.copyfile(source, args.output / name)
        owned = Path(tempfile.mkdtemp(prefix="deepseek-smoke-", dir="/kaggle/working"))
        root, environment, temporary = owned / "soup", owned / "py312", owned / "tmp"
        temporary.mkdir()
        command(["git", "clone", "--depth", "1", "https://github.com/SoupTeam/soup-rnd.git",
                 root], "clone", timeout=180)
        command(["git", "fetch", "--depth", "1", "origin", BASE], "fetch", cwd=root, timeout=180)
        command(["git", "checkout", "--detach", BASE], "checkout", cwd=root, timeout=60)
        with zipfile.ZipFile(args.overlay) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(admission["source_files_sha256"]):
                raise ValueError("Overlay member inventory mismatch")
            for name in names:
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != admission["source_files_sha256"][name]:
                    raise ValueError("Overlay member checksum mismatch: " + name)
                target = contained(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        record["source_files_sha256"] = admission["source_files_sha256"]
        model_root = Path("/kaggle/input/models/deepseek-ai/deepseek-r1/transformers/deepseek-r1/2")
        index = json.loads((model_root / "model.safetensors.index.json").read_bytes())
        shards = set(index["weight_map"].values())
        if len(shards) != 163 or not all(contained(model_root, name).is_file() for name in shards):
            raise ValueError("Required R1 reference Input is incomplete")
        record["r1_reference"] = {"present_shards": 163, "loaded": False,
                                  "role": "Reference only; measured weights are synthetic tiny V3"}
        command([sys.executable, "-m", "pip", "install", "uv==0.13.7"], "uv", timeout=180)
        uv = [sys.executable, "-m", "uv"]
        command([*uv, "venv", "--python", "3.12.3", "--seed", environment], "venv", timeout=300)
        python = str(environment / "bin/python")
        command([*uv, "pip", "install", "--python", python, *DEPENDENCIES, "-e", root],
                "dependencies", timeout=900)
        env = dict(os.environ, PYTHONPATH=str(root / "src") + ":" + str(root),
                   CUDA_VISIBLE_DEVICES="0", PYTHONHASHSEED="23", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
                   HF_HOME=str(temporary / "hf"), HF_HUB_DISABLE_TELEMETRY="1",
                   HF_HUB_OFFLINE="1", WANDB_DISABLED="true", TMPDIR=str(temporary),
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        for name in ("CUBLAS_LOGINFO_DBG", "CUBLAS_LOGDEST_DBG", "CUBLASLT_LOG_LEVEL",
                     "CUBLASLT_LOG_MASK", "CUBLASLT_LOG_FILE"):
            env.pop(name, None)
        command(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                 "--format=csv,noheader"], "gpu-facts", env, root)
        command([*uv, "pip", "freeze", "--python", python], "installed-packages", env, root)
        controller_source = (
            "benchmarks/results/p3-deepseek-fp8-smoke/orchestration/kaggle_smoke_runner.py"
        )
        command([python, "-m", "ruff", "check", *FILES, controller_source],
                "ruff", env, root, timeout=60)
        command([python, "-m", "pytest", *FILES[2:], "-v", "--tb=short", "-o", "addopts=",
                 "--junitxml=" + str(args.output / "tests.xml")], "tests", env, root, timeout=300)
        command([python, FILES[1], "--device", "cuda", "--output", args.output / "smoke"],
                "single-candidate-smoke", env, root, timeout=300)
        record["status"] = "COMMANDS_COMPLETE"
    except BaseException as error:
        record["status"] = "ERROR"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        (args.output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.clear_frames(error.__traceback__)
    finally:
        try:
            if owned is not None:
                shutil.rmtree(owned)
            record["cleanup_complete"] = True
        except Exception as error:
            record["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
            record["status"] = "ERROR"
        record["wall_seconds"] = time.monotonic() - started
        save(args.output / "run.json", record)
        artifacts = {str(path.relative_to(args.output)): {"bytes": path.stat().st_size,
                     "sha256": digest(path)} for path in sorted(args.output.rglob("*"))
                     if path.is_file()}
        save(args.output / "manifest.json", {"status": record["status"], "artifacts": artifacts})
        console.print({"status": record["status"], "cleanup_complete": record["cleanup_complete"]})
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)
    return 0 if record["status"] == "COMMANDS_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
