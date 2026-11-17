"""Parent-owned Kaggle controller for Stage2 v2 checks, diagnostics and paired evaluation."""

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
import zipfile
from pathlib import Path
from typing import Any

from rich.console import Console

BASE_REVISION = "54be201337cb7ac8ba5e58ba9ff5b55ac7b0bc81"
V1_PROTOCOL = "170c0b700c137c1f6e0c6788376927e4aaa4e99866b990bc83d0afc42a62bc3c"
ARMS = ("bf16_l2l", "oracle_block64_e4m3")
DEPENDENCIES = (
    "torch==2.11.0", "transformers==5.19.0", "peft==0.21.2", "accelerate==1.15.0",
    "safetensors==0.8.0", "rich==13.9.4", "pydantic==2.14.1", "numpy==2.1.3",
    "scipy==1.15.3", "pyarrow==21.0.0", "ruff==0.17.4", "pytest==8.4.2",
    "psutil==7.2.2", "pyyaml==6.0.3",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root: Path, name: str) -> Path:
    path = root / name
    if os.path.commonpath((os.path.realpath(root), os.path.realpath(path))) != (
        os.path.realpath(root)
    ):
        raise ValueError("Path escapes owned root: " + name)
    return path


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def discover(inputs: Path, specification: dict) -> Path:
    pattern = specification["pattern"]
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("Input discovery pattern escapes its root")
    matches = []
    for candidate in inputs.rglob(pattern):
        path = contained(inputs, str(candidate.relative_to(inputs)))
        if path.is_file() and digest(path) == specification["sha256"]:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("Expected one content-bound Input: " + specification["pattern"])
    return matches[0]


def admit_pairs(protocol: dict, requested: list[dict]) -> list[dict]:
    frozen = protocol["execution_pairs"]
    if not requested or not frozen:
        raise ValueError("A nonempty frozen evaluation worklist is required")
    keys = ("seed", "length", "endpoint", "start", "stop")
    frozen_keys = [tuple(pair[key] for key in keys) for pair in frozen]
    if len(set(frozen_keys)) != len(frozen_keys):
        raise ValueError("Frozen worklist contains duplicate pairs")
    positions = []
    for pair in requested:
        key = tuple(pair[field] for field in keys)
        if key not in frozen_keys:
            raise ValueError("Pair is not in the frozen worklist")
        positions.append(frozen_keys.index(key))
        if tuple(pair.get("arms", ARMS)) != ARMS:
            raise ValueError("Evaluation must run baseline then candidate")
        if type(pair["start"]) is not int or type(pair["stop"]) is not int:
            raise ValueError("Pair bounds must be integers")
        if not 0 <= pair["start"] < pair["stop"]:
            raise ValueError("Empty or reversed pair interval")
    if positions != sorted(set(positions)):
        raise ValueError("Requested pairs must be unique and in frozen order")
    return requested


def run(overlay_path: Path, admission_path: Path, admission_sha256: str) -> dict:
    """Execute only hash-admitted work and retain every command receipt and failure."""
    console = Console()
    started = time.monotonic()
    working = Path("/kaggle/working")
    output = working / "evidence"
    output.mkdir(exist_ok=False)
    owned = None
    deadline = started + 60
    work_deadline = started + 30
    record: dict[str, Any] = {
        "schema": "p3-stage2v2-controller-v1", "status": "STARTED", "phase": "ADMISSION",
        "base_revision": BASE_REVISION, "admission_sha256": admission_sha256,
        "commands": [], "pairs": [], "cleanup_complete": False, "quality_admitted": False,
        "v1_decision": "BLOCKED_INCOMPLETE_COVERAGE",
        "outcome_informed_redesign_after_v1_floors": True,
    }
    snapshots = 0

    def save() -> None:
        nonlocal snapshots
        record["controller_wall_seconds"] = time.monotonic() - started
        write_json(output / f"progress-{snapshots:04d}.json", record)
        snapshots += 1

    def command(argv: list, name: str, env: dict | None = None,
                cwd: Path | None = None, timeout: int = 900) -> dict:
        if not name or Path(name).name != name:
            raise ValueError("Unsafe command log name")
        budget = min(timeout, work_deadline - time.monotonic())
        if budget <= 0:
            raise TimeoutError("Controller deadline before " + name)
        begun = time.monotonic()
        process = None
        interruption = None
        cleanup_signaled = False
        try:
            with (output / (name + ".log")).open("x", encoding="utf-8") as log:
                launch_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGINT, signal.SIGTERM}
                )
                try:
                    # Register the child before a raising signal can interrupt us.
                    # The exec wrapper restores its inherited mask without preexec_fn.
                    launcher = (
                        "import os,signal,sys;"
                        "signal.pthread_sigmask(signal.SIG_SETMASK,[]);"
                        "os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
                    )
                    process = subprocess.Popen(
                        [sys.executable, "-c", launcher, *[str(value) for value in argv]],
                        cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, launch_mask)
                try:
                    process.wait(timeout=budget)
                except subprocess.TimeoutExpired:
                    interruption = "declared_deadline"
        except BaseException as error:
            interruption = type(error).__name__
            raise
        finally:
            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                if process is not None:
                    try:
                        if process.poll() is None:
                            try:
                                os.killpg(process.pid, signal.SIGINT)
                            except ProcessLookupError:
                                pass
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                pass
                    finally:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                            cleanup_signaled = True
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            interruption = "direct_child_reap_timeout"
                receipt = {
                    "name": name, "argv": [str(value) for value in argv],
                    "returncode": None if process is None else process.returncode,
                    "seconds": time.monotonic() - begun, "interruption": interruption,
                    "direct_child_reaped": process is not None and process.returncode is not None,
                    "process_group_cleanup_signaled": cleanup_signaled,
                }
                record["commands"].append(receipt)
                write_json(output / (name + ".receipt.json"), receipt)
                save()
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        if interruption or receipt["returncode"] != 0:
            raise RuntimeError("Command failed: " + name)
        return receipt

    def terminate(signum: int, frame: Any) -> None:
        nonlocal deadline
        deadline = min(deadline, time.monotonic() + 60)
        signal.setitimer(signal.ITIMER_REAL, 25)
        raise KeyboardInterrupt("Controller received signal " + str(signum))

    def timed_out(signum: int, frame: Any) -> None:
        record["controller_deadline_triggered"] = True
        # Keep command unwinding, log close and receipt writes bounded too.
        signal.setitimer(signal.ITIMER_REAL, max(0.001, min(25, deadline - time.monotonic())))
        raise TimeoutError("Controller lifetime deadline reached")

    previous_term = signal.signal(signal.SIGTERM, terminate)
    previous_alarm = signal.signal(signal.SIGALRM, timed_out)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        if digest(admission_path) != admission_sha256:
            raise ValueError("Execution admission hash mismatch")
        admission = json.loads(admission_path.read_bytes())
        if admission["schema"] != "p3-stage2v2-execution-admission-v1":
            raise ValueError("Unknown execution admission")
        limit = admission["controller_limit_seconds"]
        if type(limit) is not int or not 60 <= limit <= 43200:
            raise ValueError("Invalid controller deadline")
        deadline = started + limit
        work_deadline = max(started + 1, deadline - 60)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, work_deadline - time.monotonic()))
        phase = admission["phase"]
        phases = {"checks", "data", "tensor", "fixedpoint", "devprobe", "pilot", "evaluate"}
        if phase not in phases:
            raise ValueError("Unknown execution phase")
        if type(admission["gpu"]) is not bool:
            raise ValueError("GPU allocation must be explicit")
        if phase in {"fixedpoint", "devprobe", "pilot", "evaluate"} and not admission["gpu"]:
            raise ValueError("This phase requires a GPU")
        if digest(overlay_path) != admission["source_overlay_sha256"]:
            raise ValueError("Source overlay hash mismatch")
        record.update(phase=phase, source_overlay_sha256=admission["source_overlay_sha256"])
        owned = Path(tempfile.mkdtemp(prefix="stage2v2-", dir=working))
        root, environment, temporary = owned / "soup", owned / "py312", owned / "tmp"
        temporary.mkdir()
        for destination, source in (
            ("execution-admission.json", admission_path),
            ("source-overlay.zip", overlay_path),
            ("stage2v2_eval_runner.py", Path(__file__)),
        ):
            with (output / destination).open("xb") as stream:
                stream.write(source.read_bytes())
        for argv, name, cwd in (
            (["git", "clone", "--depth", "1", "https://github.com/SoupTeam/soup-rnd.git", root],
             "clone", None),
            (["git", "fetch", "--depth", "1", "origin", BASE_REVISION], "fetch", root),
            (["git", "checkout", "--detach", BASE_REVISION], "checkout", root),
        ):
            command(argv, name, cwd=cwd, timeout=180)
        with zipfile.ZipFile(overlay_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(admission["source_files_sha256"]):
                raise ValueError("Overlay inventory mismatch")
            for name in names:
                destination = contained(root, name)
                payload = archive.read(name)
                if hashlib.sha256(payload).hexdigest() != admission["source_files_sha256"][name]:
                    raise ValueError("Overlay member hash mismatch: " + name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
        v1_path = root / "benchmarks/results/p3-stage2-quality/frozen-study-protocol-v1.json"
        if digest(v1_path) != V1_PROTOCOL:
            raise ValueError("Frozen v1 protocol changed")
        v1 = json.loads(v1_path.read_bytes())
        for name, expected in v1["source_files_sha256"].items():
            if digest(contained(root, name)) != expected:
                raise ValueError("Frozen v1 source changed: " + name)
        for name, expected in v1["contract_files_sha256"].items():
            if digest(contained(v1_path.parent, name)) != expected:
                raise ValueError("Frozen v1 contract changed: " + name)
        python = str(environment / "bin/python")
        substitutions = {
            "PYTHON": python, "ROOT": str(root), "OUT": str(output),
            "TMP": str(temporary), "CACHE": str(temporary / "model-cache"),
            "V1_PROTOCOL": str(v1_path),
        }
        record["input_bindings"] = {}
        for name, specification in admission.get("inputs", {}).items():
            if name in substitutions:
                raise ValueError("Input name collides with a reserved substitution")
            path = discover(Path("/kaggle/input"), specification)
            record["input_bindings"][name] = {"path": str(path), **specification}
            substitutions[name] = str(path)
        for name, specification in admission.get("derived_paths", {}).items():
            if name in substitutions:
                raise ValueError("Derived path collides with an existing binding")
            parent = Path(substitutions[specification["binding"]]).parent
            substitutions[name] = str(contained(parent, specification["relative"]))
        if admission["gpu"]:
            model_root = Path(
                "/kaggle/input/models/deepseek-ai/deepseek-r1/transformers/deepseek-r1/2"
            )
            index = json.loads((model_root / "model.safetensors.index.json").read_bytes())
            shards = set(index["weight_map"].values())
            if len(shards) != 163 or not all(contained(model_root, x).is_file() for x in shards):
                raise ValueError("Required R1 reference Input is incomplete")
            record["r1_reference"] = {
                "config_sha256": digest(model_root / "config.json"), "present_shards": 163,
                "loaded": False, "role": "Reference Input only; Granite is the measured model",
            }
        command([sys.executable, "-m", "pip", "install", "uv==0.13.7"], "uv", timeout=180)
        uv = [sys.executable, "-m", "uv"]
        command([*uv, "venv", "--python", "3.12.3", "--seed", environment], "venv", timeout=300)
        if not admission["gpu"]:
            command([*uv, "pip", "install", "--python", python, "torch==2.11.0",
                     "--index-url", "https://download.pytorch.org/whl/cpu"],
                    "cpu-torch", timeout=900)
        command([*uv, "pip", "install", "--python", python, *DEPENDENCIES],
                "dependencies", timeout=900)
        if admission.get("install_dev", False):
            constraints = temporary / "constraints.txt"
            constraints.write_text("\n".join(DEPENDENCIES) + "\n", encoding="utf-8")
            command([*uv, "pip", "install", "--python", python, "-c", constraints,
                     "-e", str(root) + "[dev]"], "project-dev", timeout=1200)
        env = dict(
            os.environ, PYTHONPATH=str(root / "src") + ":" + str(root),
            CUDA_VISIBLE_DEVICES="0" if admission["gpu"] else "", PYTHONHASHSEED="0",
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
            HF_HOME=str(temporary / "hf"), HF_HUB_DISABLE_TELEMETRY="1",
            HF_XET_HIGH_PERFORMANCE="0", WANDB_DISABLED="true",
            CUBLAS_WORKSPACE_CONFIG=":4096:8", TMPDIR=str(temporary),
        )
        for name in ("CUBLAS_LOGINFO_DBG", "CUBLAS_LOGDEST_DBG", "CUBLASLT_LOG_LEVEL",
                     "CUBLASLT_LOG_MASK", "CUBLASLT_LOG_FILE"):
            env.pop(name, None)
        if admission["gpu"]:
            command(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                     "--format=csv,noheader"], "gpu-facts", env, root)
        command([*uv, "pip", "freeze", "--python", python], "installed-packages", env, root)
        if phase == "evaluate":
            if "PROTOCOL" not in admission["inputs"]:
                raise ValueError("Evaluation requires a directly hash-bound frozen protocol")
            expected_protocol = admission["protocol_sha256"]
            if expected_protocol != admission["inputs"]["PROTOCOL"]["sha256"]:
                raise ValueError("Execution admission disagrees with frozen protocol identity")
            protocol_path = Path(substitutions["PROTOCOL"])
            if digest(protocol_path) != expected_protocol:
                raise ValueError("Externally frozen protocol hash mismatch")
            protocol = json.loads(protocol_path.read_bytes())
            with (output / "p3-stage2-v2-protocol.json").open("xb") as stream:
                stream.write(protocol_path.read_bytes())
            for name, expected in protocol["source_files_sha256"].items():
                if digest(contained(root, name)) != expected:
                    raise ValueError("Frozen v2 source changed: " + name)
            pairs = admit_pairs(protocol, admission["execution_pairs"])
            record["protocol_sha256"] = expected_protocol
            per_pair = admission["measured_pair_seconds"]
            if not isinstance(per_pair, (float, int)) or not 0 < per_pair <= limit:
                raise ValueError("Invalid measured pair cost")
            for pair in pairs:
                if work_deadline - time.monotonic() < 2 * per_pair + 30:
                    record["status"] = "STOPPED_RESOURCE_LIMIT"
                    record["next_unstarted_pair"] = pair
                    break
                pair_record = {**pair, "arms": [], "status": "STARTED"}
                record["pairs"].append(pair_record)
                save()
                begun = time.monotonic()
                for arm in ARMS:
                    name = (f"L{pair['length']}-seed{pair['seed']}-{pair['endpoint']}-"
                            f"{pair['start']:05d}-{pair['stop']:05d}-{arm}")
                    shard = contained(output, "shards/" + name)
                    argv = [
                        python, "-m", "benchmarks.harness.p3_stage2v2_loglik", "run-endpoint",
                        "--data-root", substitutions["DATA"], "--endpoint", pair["endpoint"],
                        "--start", str(pair["start"]), "--stop", str(pair["stop"]),
                        "--arm", arm, "--seed", str(pair["seed"]),
                        "--length", str(pair["length"]), "--matrix-root", substitutions["MATRIX"],
                        "--audit-path", substitutions["AUDIT"], "--audit-sha256",
                        admission["inputs"]["AUDIT"]["sha256"], "--v1-protocol-path", v1_path,
                        "--protocol-path", protocol_path, "--protocol-sha256",
                        record["protocol_sha256"], "--cache-dir", substitutions["CACHE"],
                        "--output", shard,
                    ]
                    command(argv, name, env, root, timeout=admission["per_shard_limit_seconds"])
                    manifest = shard / "manifest.json"
                    measured = json.loads(manifest.read_bytes())
                    if measured["status"] != "COMPLETE":
                        raise ValueError("Incomplete endpoint shard")
                    pair_record["arms"].append({
                        "arm": arm, "manifest": str(manifest.relative_to(output)),
                        "sha256": digest(manifest),
                    })
                    save()
                pair_record.update(status="COMPLETE", seconds=time.monotonic() - begun)
                per_pair = max(per_pair, pair_record["seconds"])
                save()
            else:
                record["status"] = "PLANNED_PAIRS_COMPLETE_NOT_GLOBAL_ADMISSION"
        else:
            for specification in admission["commands"]:
                argv = [argument.format_map(substitutions) for argument in specification["argv"]]
                command(argv, specification["name"], env, root, specification["timeout"])
            record["status"] = "COMPLETE"
    except BaseException as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        if not isinstance(error, Exception):
            raise
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, max(1, min(30, deadline - time.monotonic() - 5)))
            try:
                if owned is not None:
                    shutil.rmtree(owned)
                record["cleanup_complete"] = owned is None or not owned.exists()
            except BaseException as error:
                record["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
                record["status"] = "FAILED_CLEANUP"
            signal.setitimer(signal.ITIMER_REAL, max(1, min(5, deadline - time.monotonic())))
            record["finished_at"] = time.time()
            record["controller_wall_seconds"] = time.monotonic() - started
            write_json(output / "run.json", record)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
            signal.signal(signal.SIGTERM, previous_term)
            if previous_timer[0] > 0:
                remaining = max(0.001, previous_timer[0] - (time.monotonic() - started))
                signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])
        console.print({"status": record["status"], "cleanup_complete": record["cleanup_complete"]})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    args = parser.parse_args()
    record = run(args.overlay, args.admission, args.admission_sha256)
    if record["status"] not in {"COMPLETE", "PLANNED_PAIRS_COMPLETE_NOT_GLOBAL_ADMISSION",
                                "STOPPED_RESOURCE_LIMIT"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
