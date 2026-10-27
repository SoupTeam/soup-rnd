"""Bounded sequential H3 runs. Never provisions or pays for a GPU instance."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from h3_fsdp_gate import scaling
from h3_fsdp_run import read_json, write_json
from rich.console import Console


def run_logged(command: list[str], output: Path, env: dict, timeout: float) -> None:
    """Keep a failing command, exit code and log, then stop the campaign."""
    write_json(output.with_suffix(".command.json"), command)
    started = time.monotonic()
    with output.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=platform.system() == "Linux")
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if platform.system() == "Linux":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if platform.system() == "Linux":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            write_json(output.with_suffix(".exit.json"), {"status": "timeout",
                                                          "timeout_seconds": timeout})
            raise
    write_json(output.with_suffix(".exit.json"), {"returncode": returncode,
                                                  "wall_seconds": time.monotonic() - started})
    if returncode:
        raise RuntimeError(f"command failed ({returncode}); inspect {output}")


def compare_first_step(root: Path, reference: str, arm: str) -> list[dict]:
    """Use the unchanged gate before spending the budget on later pairs."""
    import torch

    if (read_json(root / reference / "rank-0.json")["data_windows"]
            != read_json(root / arm / "rank-0.json")["data_windows"]):
        raise ValueError("actual first global data windows differ across arms")
    comparisons = []
    for filename in ("first-gradients.pt", "first-adapter.pt"):
        expected = torch.load(root / reference / filename, map_location="cpu", weights_only=True)
        actual = torch.load(root / arm / filename, map_location="cpu", weights_only=True)
        if not expected or actual.keys() != expected.keys():
            raise ValueError("cross-world first-step keys differ")
        max_error = 0.0
        for key in expected:
            torch.testing.assert_close(actual[key].float(), expected[key].float(),
                                       atol=1e-5, rtol=0.02)
            max_error = max(max_error, float((actual[key] - expected[key]).abs().max()))
        comparisons.append({"arm": arm, "file": filename, "max_abs_error": max_error,
                            "atol": 1e-5, "rtol": 0.02})
    return comparisons


def summarize(root: Path, prepared_path: Path) -> dict:
    write_json(root / "summary.json", {"status": "invalid_until_all_checks_complete"})
    (root / "summary.md").write_text("Validation pending; no scaling verdict.\n",
                                     encoding="utf-8")
    prepared = read_json(prepared_path / "prepared.json")
    if prepared["cpu_fixture"]:
        raise ValueError("offline CPU fixture cannot produce a scaling verdict")
    runs = []
    dirs = read_json(root / "campaign.json")["arms"]
    for name in dirs:
        arm = root / name
        rank0 = read_json(arm / "rank-0.json")
        world = rank0["world_size"]
        ranks = [read_json(arm / f"rank-{rank}.json") for rank in range(world)]
        reload = read_json(arm / "reload.json")
        checkpoint_reload = read_json(arm / "reload-checkpoint.json")
        if (any(r["status"] != "training_passed_reload_pending" for r in ranks)
                or reload["status"] != "passed" or checkpoint_reload["status"] != "passed"):
            raise ValueError(f"invalid arm {name}")
        if any(r.get("fsdp_policy") != rank0.get("fsdp_policy") for r in ranks):
            raise ValueError("ranks have different FSDP precision policies")
        environments = [read_json(arm / f"environment-rank-{rank}.json")
                        for rank in range(world)]
        if any(e["git_status"] for e in environments):
            raise ValueError("measured checkout must be clean")
        if any(e["versions"] != environments[0]["versions"]
               or e["device"] != environments[0]["device"] for e in environments):
            raise ValueError("ranks have different versions or GPU types")
        config = read_json(arm / "config-rank-0.json")
        config.pop("output")
        config["training"].pop("gradient_accumulation_steps")
        # logging/save_steps are fixed across worlds. Global batch, lengths,
        # model, data, optimizer, LoRA and training schedule must all agree.
        identity = {"prepared": prepared, "config": config,
                    "versions": environments[0]["versions"],
                    "commit": environments[0]["commit"], "gpu": environments[0]["device"],
                    "host": environments[0]["host"], "global_batch": rank0["global_batch"],
                    "steps": rank0["steps"], "warmup_steps": rank0["warmup_steps"],
                    "correctness_steps": rank0["correctness_steps"],
                    "measured_steps": rank0["measured_steps"]}
        identity["precision"] = rank0.get("precision", "bf16")
        identity["float32_matmul_precision"] = rank0.get("float32_matmul_precision")
        runs.append({"status": "passed", "identity": identity, "world_size": world,
                     "tokens_per_second": rank0["tokens_per_second"], "arm": name,
                     "seconds_per_optimizer_step": rank0["seconds_per_optimizer_step"],
                     "data_windows": rank0["data_windows"],
                     "memory_per_rank": [r["training_memory"] for r in ranks],
                     "fsdp_version": rank0["fsdp_version"],
                     "fsdp_policy": rank0.get("fsdp_policy")})
    versions = {run["fsdp_version"] for run in runs if run["world_size"] == 2}
    if len(versions) != 1:
        raise ValueError("two-GPU repeats used different FSDP implementations")
    policies = [run["fsdp_policy"] for run in runs if run["world_size"] == 2]
    if any(policy != policies[0] for policy in policies):
        raise ValueError("two-GPU repeats used different FSDP precision policies")
    reference = next(run for run in runs if run["world_size"] == 1)
    comparisons = []
    for run in runs:
        comparisons.extend(compare_first_step(root, reference["arm"], run["arm"]))
    result = scaling(runs)
    result.update({"status": "passed", "runs": runs, "first_step_comparisons": comparisons,
                   "resume": "unmeasured", "70b": "unmeasured",
                   "tokens_definition": "all nonpadding input tokens, all ranks; loss targets S-1",
                   "timing": "sum of CUDA-synchronized optimizer-step windows; slowest rank",
                   "quality": "unmeasured (synthetic token data)"})
    write_json(root / "summary.json", result)
    lines = ["# H3 measured pilot", "", "| Arm | GPUs | Input tokens/s | Step seconds |",
             "|---|---:|---:|---:|"]
    for run in runs:
        lines.append(f"| {run['arm']} | {run['world_size']} | "
                     f"{run['tokens_per_second']:.2f} | "
                     f"{run['seconds_per_optimizer_step']:.4f} |")
    lines += ["", f"Speedup(2): {result['speedup_2']:.4f}; "
              f"efficiency(2): {result['efficiency_2']:.4f}.", "",
              "4/8 GPUs, 70B, optimizer/scheduler resume and quality: **unmeasured**.",
              "Per-rank VRAM/RAM, first-step deltas and raw timings: summary.json and rank-*.json."]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--campaign-seconds", type=int, default=5400)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    root = Path(os.path.realpath(args.output))
    prepared = Path(os.path.realpath(args.prepared))
    if args.summarize_only:
        summarize(root, prepared)
        return
    if platform.system() != "Linux":
        parser.error("measured campaign requires Linux")
    if min(args.repeats, args.measured_steps, args.timeout_seconds, args.campaign_seconds) < 1:
        parser.error("repeats and time limits must be positive")
    if read_json(prepared / "prepared.json")["cpu_fixture"]:
        parser.error("CPU fixture is excluded from scaling")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                      "--format=csv,noheader"], text=True).strip()
    if active:
        parser.error("foreign GPU compute processes present; use an idle dedicated host")
    import torch

    if torch.cuda.device_count() != 2:
        parser.error("expose exactly the two GPUs reserved for this pilot")
    if torch.cuda.get_device_name(0) != torch.cuda.get_device_name(1):
        parser.error("both GPUs must be the same model")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("commit the harness before a measured campaign")
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("h3_fsdp_run.py")
    arms = [f"repeat-{repeat + 1}-gpu-{world}" for repeat in range(args.repeats)
            for world in ((1, 2) if repeat % 2 == 0 else (2, 1))]
    write_json(root / "campaign.json", {"status": "incomplete", "arms": arms,
                                        "args": vars(args), "argv": sys.argv})
    deadline = time.monotonic() + args.campaign_seconds
    for index, name in enumerate(arms):
        if shutil.disk_usage(root).free < 15 * 1024**3:
            raise RuntimeError("less than 15 GiB disk remains; stop before the next arm")
        world = int(name[-1])
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0" if world == 1 else "0,1",
               "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
        command = [sys.executable]
        if world > 1:
            command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                        f"--nproc-per-node={world}"]
        command += [str(script), "train", "--prepared", str(prepared),
                    "--output", str(root / name), "--measured-steps", str(args.measured_steps),
                    "--precision", args.precision]
        for suffix, cmd in [("train", command), ("reload", [sys.executable, str(script),
                "reload", "--prepared", str(prepared), "--output", str(root / name)]),
                ("reload-checkpoint", [sys.executable, str(script), "reload", "--prepared",
                 str(prepared), "--output", str(root / name),
                 "--checkpoint"])]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("campaign wall-clock limit reached; stop the rented instance")
            run_logged(cmd, root / f"{name}-{suffix}.log", env,
                       timeout=min(args.timeout_seconds, remaining))
        if index == 1:
            verdict = root / "first-pair-control.json"
            write_json(verdict, {"status": "invalid_until_all_checks_complete"})
            comparison = compare_first_step(root, "repeat-1-gpu-1", "repeat-1-gpu-2")
            write_json(verdict, {"status": "passed", "comparisons": comparison})
    result = summarize(root, prepared)
    write_json(root / "campaign.json", {"status": "passed", "arms": arms,
                                        "args": vars(args), "argv": sys.argv})
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file()}
    write_json(root / "sha256.json", manifest)
    Console().print(f"Speedup(2)={result['speedup_2']:.4f}; "
                    f"efficiency(2)={result['efficiency_2']:.4f}")


if __name__ == "__main__":
    main()
