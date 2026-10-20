#!/usr/bin/env python3
"""Portable A0 session. Planning is the default; --execute runs prepared local fixtures.

Each invocation requires a new output directory. The protocol, source hashes and exact
commands are saved before the first CUDA process. No model downloads occur in the session.
The wall-time limit stops child work; it does NOT stop or destroy a rented server.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import l2l_box
import l2l_rule

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "benchmarks" / "harness"
PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors",
            "pydantic", "numpy", "psutil", "rich")
RULE = {
    "profile": "a0-strict-v1",
    "G1": "F1 k=4 and F2 k=2: nonzero LoRA grads and per-micro-batch losses equal GA",
    "G2": "T_A(16) >= 0.8 * T_B(16)",
    "G3": "T_A(16) >= 1.3 * T_A(1)",
    "G4": "max allocated A16 <= max allocated A1 + 0.5 GB (decimal)",
    "V1": "disk source uses direct I/O",
    "V2": "L2L k1 work equals plain batch1: loads within 0.5, bytes within 1 MB",
    "V3": "pinned activation bytes + 4 GB free RAM/Windows commit; idle GPU <=500 MiB",
    "V4": "two distinct rounds 0 and 1; spread <=5%",
    "V5": "power known for all samples; no battery; sample gap <=30 seconds",
    "V6": "reader snapshots available; no foreign process reads >=1 GB per arm",
    "strict": "unknown telemetry, nonfinite values or GPU peers invalidate dependent gates",
    "batch_control": "plain batches 1..batch_max ascending then descending, fresh processes",
    "fit": "largest tested valid batch; maximum established only with next-batch CUDA OOM",
    "retries": "no automatic retries except the predeclared deterministic G1 fallback",
    "stop": "stop on child failure, missing output, non-pass G1 or session timeout",
    "spill": "pinned vs direct-I/O spill at k16, budget k8; informational",
    "scope": "synthetic dense 70B timing/memory; no quality or giant-MoE throughput claim",
    "cloud": "server power is an operator declaration; peer/reader scope is visible processes",
}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes() -> dict[str, str]:
    files = sorted((ROOT / "src").rglob("*.py")) + sorted(HARNESS.glob("*.py"))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def versions() -> dict[str, str]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def git_head() -> str:
    if not (ROOT / ".git").exists():
        bundle = ROOT / "A0_SOURCE.json"
        return json.loads(bundle.read_text())["base_commit"] if bundle.exists() else "unknown"
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT),
         "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1", required=True, help="local real Mistral-7B checkpoint")
    parser.add_argument("--f2", required=True, help="local synthetic Llama-70B, 32768-row head")
    parser.add_argument("--f1-shards", required=True)
    parser.add_argument("--f2-shards", required=True)
    parser.add_argument("--spill-dir", required=True)
    parser.add_argument("--out", required=True, help="new directory, never an old attempt")
    parser.add_argument("--power-source", choices=("sensor", "server"), default="sensor")
    parser.add_argument("--batch-max", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    return parser


def common_args(args: argparse.Namespace, fixture: str) -> list[str]:
    return ["--weights", getattr(args, fixture), "--shards", getattr(args, f"{fixture}_shards"),
            "--quant", "nf4", "--buffers", "2", "--read-ahead", "2", "--seq", "512",
            "--lora-r", "8", "--seed", "3", "--input-seed", "17"]


def command_plan(args: argparse.Namespace, out: Path) -> list[dict[str, Any]]:
    plan = []

    def add(name: str, script: str, arguments: list[str], **extra: Any) -> None:
        plan.append({"name": name, "output": str(out / f"{name}.json"),
                     "command": [sys.executable, str(HARNESS / script), *arguments,
                                 "--out", str(out / f"{name}.json")], **extra})

    for fixture, tier, k in (("f1", "ram", "4"), ("f2", "disk", "2")):
        base = [*common_args(args, fixture), "--mode", "correctness", "--tier", tier,
                "--k", k, "--label", fixture.upper()]
        add(f"correctness_{fixture}", "l2l_probe.py", base)
        add(f"correctness_{fixture}_det", "l2l_probe.py", [*base, "--deterministic"],
            fallback_for=f"correctness_{fixture}")
    add("timing", "l2l_probe.py", [*common_args(args, "f2"), "--mode", "timing",
        "--tier", "disk", "--ks", "1,2,3,4,8,16", "--steps", "3", "--warmup", "1",
        "--rounds", "2", "--label", "F2"])
    for rnd in (0, 1):
        batches = range(1, args.batch_max + 1) if rnd == 0 else range(args.batch_max, 0, -1)
        for batch in batches:
            add(f"batch_{batch}_r{rnd}", "l2l_probe.py", [*common_args(args, "f2"),
                "--mode", "batch", "--tier", "disk", "--batch", str(batch),
                "--round-index", str(rnd), "--steps", "3", "--warmup", "1",
                "--label", f"A0_batch{batch}_r{rnd}"],
                batch=batch, round=rnd)
    add("spill", "l2l_probe.py", [*common_args(args, "f2"), "--mode", "spill", "--tier",
        "disk", "--spill-k", "16", "--spill-budget-k", "8", "--spill-dir", args.spill_dir,
        "--steps", "3", "--warmup", "1", "--rounds", "2", "--label", "F2"])
    return plan


def read_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list) or not payload["records"]:
        raise ValueError(f"no measured records in {path.name}; CUDA skip is not success")
    return payload


def needs_fallback(path: Path) -> bool:
    row = read_result(path)["records"][0]
    return not (row.get("a_vs_a", {}).get("equal") and row.get("a_vs_a_losses_equal"))


def preflight(args: argparse.Namespace) -> int:
    """Run in a separate process so the session parent never owns a CUDA context."""
    result: dict[str, Any] = {"box_before_cuda": l2l_box.box_stamp(), "versions": versions()}
    errors = []
    try:
        import torch

        import soup_cli

        expected = os.path.realpath(ROOT / "src")
        actual = os.path.realpath(soup_cli.__file__)
        result["soup_cli_file"] = actual
        if os.path.commonpath([expected, actual]) != expected:
            errors.append("soup_cli imported outside this checkout")
        bundle = ROOT / "A0_SOURCE.json"
        if bundle.exists():
            expected_hashes = json.loads(bundle.read_text())["source_sha256"]
            if source_hashes() != expected_hashes:
                errors.append("bundle Python source hashes differ from A0_SOURCE.json")
        result["cuda_available"] = torch.cuda.is_available()
        if not result["cuda_available"]:
            errors.append("CUDA unavailable")
        else:
            gpu = torch.cuda.get_device_properties(0)
            result["gpu"] = {"name": gpu.name, "total_gb": gpu.total_memory / 1e9}
            if not 7.5 <= result["gpu"]["total_gb"] <= 9:
                errors.append("this A0 protocol requires an 8 GB GPU")
        for name, value in result["versions"].items():
            if value == "not installed":
                errors.append(f"missing package: {name}")
        before = result["box_before_cuda"]
        if before["ac"] is not True:
            errors.append("power unknown or battery; declare server power explicitly on Linux")
        if before["gpu_pids"] is None or before["gpu_pids"]:
            errors.append("GPU peer telemetry unavailable or GPU is in use")
        if before["gpu_mem_used_mib"] is None or before["gpu_mem_used_mib"] > 500:
            errors.append("GPU is not idle (<=500 MiB required)")
        if before["avail_phys_gb"] is None or before["avail_phys_gb"] < 17:
            errors.append("less than 17 GB free physical/container RAM")
        if l2l_box.process_read_bytes() is None:
            errors.append("per-process read counters unavailable")
        for fixture in ("f1", "f2"):
            config = json.loads((Path(getattr(args, fixture)) / "config.json").read_text())
            if fixture == "f1" and (config.get("model_type") != "mistral"
                                    or config.get("_soup_synthetic")):
                errors.append("F1 must be a real Mistral checkpoint")
            if fixture == "f2" and not (
                config.get("_soup_synthetic") and config.get("model_type") == "llama"
                and config.get("num_hidden_layers") == 80 and config.get("hidden_size") == 8192
                and config.get("vocab_size") == 32768
            ):
                errors.append("F2 must be the synthetic 80-layer 70B shape with 32768 vocab")
            cache = Path(getattr(args, f"{fixture}_shards"))
            index = json.loads((cache / "index.json").read_text())
            if (index.get("quant") != "nf4" or index.get("double_quant") is not True
                    or index.get("quant_device") != "cuda"):
                errors.append(f"{fixture} needs a prepared double-quant NF4 CUDA cache")
            cache_bytes = sum(path.stat().st_size for path in cache.glob("*.safetensors"))
            result[f"{fixture}_cache_bytes"] = cache_bytes
            if fixture == "f2":
                accessible = before["total_phys_gb"]
                if accessible is None or cache_bytes <= accessible * 1e9:
                    errors.append("F2 NF4 cache must be larger than accessible RAM")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    result["errors"] = errors
    write_json(Path(args.out) / "preflight.json", result)
    return 3 if errors else 0


def run_child(entry: dict[str, Any], out: Path, env: dict[str, str], deadline: float
              ) -> dict[str, Any]:
    target = Path(entry["output"])
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    before = l2l_box.box_stamp()
    reads_before = l2l_box.process_read_bytes()
    watch = l2l_box.SuspendWatch()
    watch.start()
    child = None
    try:
        with (out / "logs" / f"{entry['name']}.log").open("xb") as log:
            child = subprocess.Popen(entry["command"], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            code = child.wait(timeout=max(0.01, deadline - time.monotonic()))
    except BaseException:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        raise
    finally:
        suspend = watch.stop()
    payload = read_result(target)
    oom = any(row.get("kind") == "oom" for row in payload["records"])
    if code != 0 and not ("batch" in entry and code == 3 and oom):
        raise RuntimeError(f"{entry['name']} exited {code}; see its log")
    return {"name": entry["name"], "returncode": code, "oom": oom, "suspend": suspend,
            "box_before": before, "box_after": l2l_box.box_stamp(),
            "foreign_readers": l2l_box.foreign_readers(
                reads_before, l2l_box.process_read_bytes(), own_pids={os.getpid(), child.pid})}


def batch_summary(out: Path, completed: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[int, list[dict[str, Any]]] = {}
    ooms = []
    for observation in completed:
        if not observation["name"].startswith("batch_"):
            continue
        _, batch, round_tag = observation["name"].split("_")
        batch = int(batch)
        payload = read_result(out / f"{observation['name']}.json")
        if observation["oom"]:
            # The runtime has previously surfaced stale page-lock errors as "out of
            # memory". Only a recorded CUDA allocator OOM establishes a batch bound.
            if any(row.get("kind") == "oom" and
                   (row.get("memory") or {}).get("num_ooms", 0) > 0
                   for row in payload["records"]):
                ooms.append(batch)
            continue
        rows = [row for row in payload["records"] if row.get("kind") == "arm"]
        if len(rows) != 1:
            raise ValueError("batch control must contain exactly one plain step result")
        row = {**rows[0], "round": int(round_tag[1:])}
        if payload.get("meta", {}).get("direct_io") is not True:
            row["skipped"] = "batch control direct I/O not confirmed"
        if not payload.get("meta", {}).get("v3", {}).get("ok"):
            row["skipped"] = "batch control V3 not confirmed"
        arms.setdefault(batch, []).append(row)
    judged = {str(batch): l2l_rule.arm_value(rows, strict=True) for batch, rows in arms.items()}
    valid = [int(batch) for batch, value in judged.items() if value["valid"]]
    largest = max(valid) if valid else None
    return {"arms": judged, "oom_batches": sorted(set(ooms)), "largest_valid_tested": largest,
            "maximum_established": (largest is not None and largest + 1 in ooms
                                    and largest + 1 not in arms)}


def main(argv: list[str] | None = None) -> int:
    parser = args_parser()
    args = parser.parse_args(argv)
    if args.batch_max < 3 or args.max_hours <= 0:
        parser.error("batch-max must be >=3 and max-hours must be positive")
    if args.power_source == "server" and sys.platform != "linux":
        parser.error("server power declaration is only for a Linux server")
    for field in ("f1", "f2", "f1_shards", "f2_shards", "spill_dir", "out"):
        setattr(args, field, os.path.abspath(getattr(args, field)))
    if args.power_source == "server":
        os.environ["SOUP_L2L_POWER_SOURCE"] = "server"
    else:
        os.environ.pop("SOUP_L2L_POWER_SOURCE", None)
    if args.preflight:
        return preflight(args)
    out = Path(os.path.abspath(args.out))
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    plan = command_plan(args, out)
    hashes = source_hashes()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "head": git_head(),
                "python": sys.executable, "platform": platform.platform(), "versions": versions(),
                "source_sha256": hashes, "protocol": RULE, "args": vars(args), "plan": plan}
    write_json(out / "protocol.json", manifest)
    if not args.execute:
        return 0
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1"}
    deadline = time.monotonic() + args.max_hours * 3600
    completed: list[dict[str, Any]] = []
    status: dict[str, Any] = {"state": "running", "completed": completed}
    try:
        check = [sys.executable, str(Path(__file__).resolve()), "--preflight",
                 "--power-source", args.power_source]
        for field in ("f1", "f2", "f1_shards", "f2_shards", "spill_dir", "out"):
            check += [f"--{field.replace('_', '-')}", getattr(args, field)]
        with (out / "logs" / "preflight.log").open("xb") as log:
            result = subprocess.run(check, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=120, check=False)
        if result.returncode != 0:
            raise RuntimeError("preflight refused the session; see preflight.json")
        for entry in plan:
            fallback = entry.get("fallback_for")
            if fallback and not needs_fallback(out / f"{fallback}.json"):
                continue
            if entry["name"] == "timing":
                gate = l2l_rule.evaluate(str(out), strict=True)["gates"]["G1"]
                if gate["status"] != "pass":
                    raise RuntimeError(f"G1 {gate['status']}: {gate['detail']}")
            completed.append(run_child(entry, out, env, deadline))
            write_json(out / "session.json", status)
        if source_hashes() != hashes:
            raise RuntimeError("source files changed during the session")
        status["state"] = "finished"
    except (Exception, KeyboardInterrupt) as exc:
        status.update(state="stopped", reason=f"{type(exc).__name__}: {exc}")
    finally:
        write_json(out / "session.json", status)
        try:
            verdict = l2l_rule.evaluate(str(out), strict=True)
            verdict["batch_control"] = batch_summary(out, completed)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            verdict = {"verdict": "NO VERDICT: incomplete or malformed raw data",
                       "analysis_error": f"{type(exc).__name__}: {exc}",
                       "batch_control": {"maximum_established": False, "arms": {}}}
            status.update(state="stopped", reason=verdict["analysis_error"])
            write_json(out / "session.json", status)
        verdict["session"] = status["state"]
        verdict["mechanism_verdict"] = verdict["verdict"]
        control = verdict["batch_control"]
        verdict["control_complete"] = (control["maximum_established"] and all(
            control["arms"].get(str(batch), {}).get("valid") for batch in (1, 2, 3)))
        if status["state"] != "finished":
            verdict["verdict"] = f"SESSION INCOMPLETE: {status.get('reason')}"
        elif not verdict["control_complete"]:
            verdict["verdict"] = f"BATCH CONTROL INCOMPLETE: {verdict['mechanism_verdict']}"
        write_json(out / "verdict.json", verdict)
    return 0 if status["state"] == "finished" else 3


if __name__ == "__main__":
    raise SystemExit(main())
