"""Fresh-process synthetic D2 timing with observed CUDA-context birth/death.

The CPU-only supervisor never creates a CUDA context. Every arm must start and
end with an empty compute/graphics process list. One new NVML PID must persist
throughout the arm and disappear after its child exits. A namespace PID mapping
is inferred ONLY from that full lifecycle; an arbitrary sole peer is not trusted.
No old VOID record is reclassified. Liger/Unsloth/full-training are out of scope.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from rich.console import Console

try:
    from . import fast_lora_probe as probe
except ImportError:
    import fast_lora_probe as probe


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_file(path: Path, seconds: float = 45) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"handshake timeout: {path.name}")
        time.sleep(0.01)


def lifecycle_verdict(
    before: list[dict], live: list[dict], after: list[dict], exit_code: int
) -> dict:
    """Fail closed; all telemetry must be complete, quiescent and single-owner."""
    reason = []
    samples = before + live + after
    if not before or not live or not after or any(not row.get("available") for row in samples):
        reason.append("missing telemetry")
    if any(row.get("processes") for row in before + after):
        reason.append("nonempty pre-context/post-exit process list")
    identities = {tuple(row.get("processes", [])) for row in live}
    if len(identities) != 1 or not identities or len(next(iter(identities), ())) != 1:
        reason.append("ambiguous/changed context identity or peer")
    if any(row.get("graphics") for row in samples):
        reason.append("graphics peer")
    if exit_code != 0:
        reason.append("child failed")
    for row in live:
        if not all(isinstance(row.get(key), (int, float)) for key in ("clock_mhz", "power_mw")):
            reason.append("missing clock/power")
        if row.get("pstate") not in (0, 2):
            reason.append("non-performance power state")
        # NVIDIA throttle flags: thermal/hardware brake/unknown exclusions.
        # GPU-idle and software power-cap are recorded but allowed; each arm
        # still requires matched observed clocks before any aggregate verdict.
        if row.get("throttle", 0) & ~0x5:
            reason.append("thermal/hardware/application-clock throttle")
    return {
        "status": "VOID" if reason else "LIFECYCLE_VALID",
        "reasons": sorted(set(reason)),
        "nvml_pid": next(iter(identities))[0]
        if len(identities) == 1 and live and len(next(iter(identities))) == 1
        else None,
    }


class Monitor:
    def __init__(self) -> None:
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        probe._require(pynvml.nvmlDeviceGetCount() == 1, "require one NVML-visible GPU")
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.uuid = pynvml.nvmlDeviceGetUUID(self.handle)

    def sample(self) -> dict:
        nvml, handle = self.nvml, self.handle
        result = {"unix": time.time(), "uuid": self.uuid, "available": False}
        try:
            result.update(
                processes=sorted(
                    {item.pid for item in nvml.nvmlDeviceGetComputeRunningProcesses(handle)}
                ),
                graphics=sorted(
                    {item.pid for item in nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)}
                ),
                clock_mhz=nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
                power_mw=nvml.nvmlDeviceGetPowerUsage(handle),
                pstate=nvml.nvmlDeviceGetPowerState(handle),
                throttle=nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
                temperature=nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                available=True,
            )
        except Exception as exc:
            result["error"] = str(exc)
        return result


def gate(case: Path, config: dict) -> None:
    import torch
    from safetensors.torch import save_file

    dtype, path, dx = config["dtype"], config["path"], config["dx"]
    with probe._deterministic(792, "cuda:0"):
        model = probe._make_fixture(path, getattr(torch, dtype), "cuda:0", config["shapes"])
        fast = copy.deepcopy(model)
        probe._require(probe._patch(fast, path) > 0, "no fast patch")
        x = torch.randn(
            1,
            config["tokens"],
            probe.SHAPES[config["shapes"]]["hidden"],
            device="cuda:0",
            dtype=getattr(torch, dtype),
        ).requires_grad_(dx)
        outputs = probe._outputs(fast, x)
        probe._require(
            all(type(out.grad_fn).__name__ == probe.EXPECTED_GRAD_FN[path] for out in outputs),
            "missing custom path",
        )
        upstream = tuple(torch.randn_like(out) for out in outputs)
        del outputs
        oracle_model = copy.deepcopy(model).double()
        expected = probe._snapshot(model, x, upstream, path)
        actual = probe._snapshot(fast, x.detach().clone().requires_grad_(dx), upstream, path)
        oracle = probe._snapshot(
            oracle_model,
            x.double().detach().requires_grad_(dx),
            tuple(value.double() for value in upstream),
            path,
        )
        probe._validate_snapshot_keys(path, dx, fast=actual, PEFT=expected, oracle=oracle)
        label = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}[dtype]
        checks = [
            {"quantity": key, **probe._comparison(actual[key], expected[key], oracle[key], label)}
            for key in expected
        ]
        control = probe._changed_adapter_control(model, x, expected, oracle, label)
        save_file(
            {
                f"{arm}/{key}": value.detach().cpu().contiguous()
                for arm, values in (("fast", actual), ("PEFT", expected), ("float64", oracle))
                for key, value in values.items()
            },
            str(case / "correctness.safetensors"),
        )
        torch.save(
            {
                "state": {key: value.cpu() for key, value in model.state_dict().items()},
                "x": x.detach().cpu(),
                "upstream": tuple(value.cpu() for value in upstream),
            },
            case / "fixture.pt",
        )
        write_json(case / "gate.json", {"passed": True, "checks": checks, "control": control})


def child(arm: Path, case: Path, config: dict, implementation: str) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    path = config["path"]
    with probe._deterministic(792, "cuda:0"):
        fixture = torch.load(case / "fixture.pt", map_location="cpu", weights_only=True)
        model = probe._make_fixture(path, getattr(torch, config["dtype"]), "cpu", config["shapes"])
        model.load_state_dict(fixture["state"])
        del fixture["state"]
        model.to("cuda:0")
        if implementation == "fast":
            probe._require(probe._patch(model, path) > 0, "no custom patch")
        x = fixture["x"].to("cuda:0").requires_grad_(config["dx"])
        upstream = tuple(value.to("cuda:0") for value in fixture["upstream"])
        del fixture
        warmup_end = time.monotonic() + 2.0
        warmup_count = 0
        while warmup_count < config["warmup"] or time.monotonic() < warmup_end:
            model.zero_grad(set_to_none=True)
            x.grad = None
            torch.autograd.backward(probe._outputs(model, x), upstream)
            warmup_count += 1
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        write_json(
            arm / "ready.json",
            {
                "pid": os.getpid(),
                "uuid": str(torch.cuda.get_device_properties(0).uuid),
                "proc_status": Path("/proc/self/status").read_text() if os.name != "nt" else "",
            },
        )
        wait_file(arm / "go")
        rows = []
        started_unix = time.time()
        minimum_end = time.monotonic() + 1.0
        while True:
            model.zero_grad(set_to_none=True)
            x.grad = None
            outputs, forward = probe._measure(lambda: probe._outputs(model, x), "cuda:0")
            _, backward = probe._measure(
                lambda: torch.autograd.backward(outputs, upstream), "cuda:0"
            )
            rows.append({"forward_ms": forward, "backward_ms": backward})
            # The final graph is checked below, outside event regions.
            if len(rows) >= config["samples"] and time.monotonic() >= minimum_end:
                break
            del outputs
        ended_unix = time.time()
        peaks = {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        }
        write_json(
            arm / "timed.json",
            {
                "rows": rows,
                "peak_bytes": peaks,
                "started_unix": started_unix,
                "ended_unix": ended_unix,
            },
        )
        wait_file(arm / "check")
        if implementation == "fast":
            probe._require(
                all(type(out.grad_fn).__name__ == probe.EXPECTED_GRAD_FN[path] for out in outputs),
                "missing timed custom path",
            )
        actual = {
            key: value.cpu()
            for key, value in probe._collect_snapshot(model, x, outputs, path).items()
        }
        reference = load_file(str(case / "correctness.safetensors"))
        label = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}[config["dtype"]]
        expected = {
            key.removeprefix("PEFT/"): value
            for key, value in reference.items()
            if key.startswith("PEFT/")
        }
        oracle = {
            key.removeprefix("float64/"): value
            for key, value in reference.items()
            if key.startswith("float64/")
        }
        probe._validate_snapshot_keys(
            path, config["dx"], actual=actual, PEFT=expected, oracle=oracle
        )
        checks = [
            {"quantity": key, **probe._comparison(actual[key], expected[key], oracle[key], label)}
            for key in expected
        ]
        save_file(actual, str(arm / "measured.safetensors"))
        write_json(arm / "checks.json", {"passed": True, "rows": checks})


def supervise(args: argparse.Namespace) -> None:
    destination = Path(os.path.realpath(args.output_dir))
    root = Path(__file__).parents[2]
    probe._require(
        os.path.commonpath([str(root), str(destination)]) != str(root),
        "output must be outside source",
    )
    destination.mkdir(parents=True, exist_ok=False)
    monitor = Monitor()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(root / "src"),
    }
    summary = {
        "arms": [],
        "verdict": "NO VERDICT",
        "source": {},
        "started_unix": time.time(),
        "protocol": vars(args),
    }
    for name in ("src", "benchmarks/harness"):
        for entry in (root / name).rglob("*.py"):
            summary["source"][entry.relative_to(root).as_posix()] = hashlib.sha256(
                entry.read_bytes()
            ).hexdigest()
    write_json(destination / "summary.json", summary)
    deadline = time.monotonic() + args.seconds
    for dtype in args.dtypes:
        for path in probe.PROJECTIONS:
            for dx in (True, False):
                if time.monotonic() > deadline - 90:
                    raise TimeoutError("supervisor deadline")
                case = destination / f"{dtype}-{path}-dx{int(dx)}"
                case.mkdir()
                config = {
                    "dtype": dtype,
                    "path": path,
                    "dx": dx,
                    "shapes": args.shapes,
                    "tokens": args.tokens,
                    "warmup": args.warmup,
                    "samples": args.samples,
                }
                write_json(case / "config.json", config)
                command = [
                    sys.executable,
                    str(Path(__file__)),
                    "--worker",
                    "gate",
                    "--case",
                    str(case),
                ]
                write_json(case / "gate-command.json", command)
                with (case / "gate.log").open("w") as log:
                    subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=120,
                    )
                for cycle in range(args.cycles):
                    for position, implementation in enumerate(("PEFT", "fast", "fast", "PEFT")):
                        probe._require(time.monotonic() < deadline - 35, "supervisor deadline")
                        arm = case / f"{cycle:02d}-{position}-{implementation}"
                        arm.mkdir()
                        before = []
                        for _ in range(5):
                            before.append(monitor.sample())
                            time.sleep(0.02)
                        write_json(arm / "preflight.json", before)
                        probe._require(
                            all(
                                row["available"] and not row["processes"] and not row["graphics"]
                                for row in before
                            ),
                            "GPU not idle before child; retained preflight",
                        )
                        command = [
                            sys.executable,
                            str(Path(__file__)),
                            "--worker",
                            "arm",
                            "--case",
                            str(case),
                            "--arm",
                            str(arm),
                            "--implementation",
                            implementation,
                        ]
                        write_json(arm / "command.json", command)
                        live = []
                        observed = []
                        with (arm / "stdout.log").open("w") as log:
                            worker = subprocess.Popen(
                                command, env=env, stdout=log, stderr=subprocess.STDOUT
                            )
                            try:
                                limit = time.monotonic() + 90
                                while not (arm / "ready.json").exists():
                                    observed.append(monitor.sample())
                                    probe._require(
                                        worker.poll() is None, "child failed before ready"
                                    )
                                    probe._require(time.monotonic() < limit, "ready timeout")
                                    time.sleep(0.05)
                                live.append(monitor.sample())
                                (arm / "go").touch()
                                while not (arm / "timed.json").exists():
                                    live.append(monitor.sample())
                                    probe._require(
                                        worker.poll() is None, "child failed during timing"
                                    )
                                    probe._require(time.monotonic() < limit, "timing timeout")
                                    time.sleep(0.02)
                                live.append(monitor.sample())
                                (arm / "check").touch()
                                code = worker.wait(timeout=30)
                            finally:
                                if worker.poll() is None:
                                    worker.kill()
                                    worker.wait()
                                after = []
                                for _ in range(10):
                                    after.append(monitor.sample())
                                    time.sleep(0.05)
                                write_json(
                                    arm / "telemetry.json",
                                    {
                                        "before": before,
                                        "context_birth": observed,
                                        "live": live,
                                        "after": after,
                                    },
                                )
                        validity = lifecycle_verdict(before, live, after, code)
                        checks = (
                            json.loads((arm / "checks.json").read_text())
                            if (arm / "checks.json").exists()
                            else {"passed": False}
                        )
                        if not checks["passed"]:
                            validity = {**validity, "status": "VOID", "correctness_failed": True}
                        row = {
                            "case": case.name,
                            "arm": arm.name,
                            "implementation": implementation,
                            "cycle": cycle,
                            "position": position,
                            "exit_code": code,
                            "validity": validity,
                        }
                        if (arm / "timed.json").exists():
                            row["measurement"] = json.loads((arm / "timed.json").read_text())
                        summary["arms"].append(row)
                        write_json(destination / "summary.json", summary)
                        Console().print(case.name, arm.name, validity)
                        probe._require(
                            validity["status"] == "LIFECYCLE_VALID", "VOID arm; stop run"
                        )
    summary["ended_unix"] = time.time()
    summary["verdict"] = "COLLECTED; clock matching/independent review required"
    summary["descriptive"] = {}
    for case_name in {row["case"] for row in summary["arms"]}:
        arms = [row for row in summary["arms"] if row["case"] == case_name]
        times = {
            implementation: [
                sample["forward_ms"] + sample["backward_ms"]
                for row in arms
                if row["implementation"] == implementation
                for sample in row["measurement"]["rows"]
            ]
            for implementation in ("PEFT", "fast")
        }
        summary["descriptive"][case_name] = {
            implementation + "_median_ms": statistics.median(values)
            for implementation, values in times.items()
        }
    write_json(destination / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("gate", "arm"))
    parser.add_argument("--case")
    parser.add_argument("--arm")
    parser.add_argument("--implementation", choices=("PEFT", "fast"))
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["bfloat16", "float16"],
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--shapes", choices=tuple(probe.SHAPES), default="llama3.1-8b")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    try:
        if args.worker:
            case = Path(args.case)
            config = json.loads((case / "config.json").read_text())
            if args.worker == "gate":
                gate(case, config)
            else:
                child(Path(args.arm), case, config, args.implementation)
        else:
            probe._require(args.output_dir and 1 <= args.seconds <= 2400, "bounded output required")
            probe._require(
                args.samples >= 1 and args.cycles >= 1 and args.warmup >= 1, "invalid sample count"
            )
            supervise(args)
    except Exception:
        Console().print(traceback.format_exc(), markup=False)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
