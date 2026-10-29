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
import tarfile
import time
from pathlib import Path

from h3_fsdp_gate import assert_flash_operators, scaling
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


def archive_completed_arm(root: Path, prepared: Path, name: str) -> dict:
    """Seal each arm before the next run; a ready marker follows the final hash.

    Retain numeric references and all adapter saves. Optimizer/RNG and duplicate
    tokenizers are outside this adapter-reload experiment's archive contract.
    The caller must collect/verify these archives independently of a rental timer.
    """
    arm = root / name
    if (os.path.commonpath([os.path.realpath(root), os.path.realpath(arm)])
            != os.path.realpath(root) or arm.is_symlink()):
        raise ValueError("arm must stay inside the campaign root")
    record = read_json(arm / "rank-0.json")
    if record["status"] != "training_passed_reload_pending":
        raise ValueError("cannot seal failed training")
    steps = record["steps"]
    for rank in range(record["world_size"]):
        if read_json(arm / f"rank-{rank}.json")["status"] != record["status"]:
            raise ValueError("cannot seal an incomplete distributed arm")
    for filename in ("reload.json", "reload-checkpoint.json"):
        if read_json(arm / filename)["status"] != "passed":
            raise ValueError("cannot seal a failed adapter reload")
    required = ["first-gradients.pt", "first-adapter.pt", "reload-reference.pt",
                "adapter_model.safetensors", f"checkpoint-{steps}/adapter_model.safetensors"]
    if record.get("bf16_conservative") or record.get("bf16_flash"):
        required += ["initial-adapter.pt", "first-preclip-gradients.pt"]
    if any(not (arm / filename).is_file() for filename in required):
        raise ValueError("cannot seal an arm with missing numeric references")
    archive = root.parent / f"{root.name}-{name}.tar.gz"
    marker = archive.with_suffix(".json")
    if archive.exists() or marker.exists():
        raise ValueError("refusing to overwrite a sealed arm")
    excluded = {"tokenizer.json", "optimizer.bin", "pytorch_model_fsdp.bin",
                "training_args.bin"}
    files = [p for p in sorted(arm.rglob("*")) if p.is_file()
             and p.name not in excluded and not p.name.startswith("rng_state")]
    files += sorted(root.glob(f"{name}-*.log"))
    files += sorted(root.glob(f"{name}-*.command.json"))
    files += sorted(root.glob(f"{name}-*.exit.json"))
    source = prepared / "prepared.json"
    if source.is_symlink():
        raise ValueError("prepared metadata must be a regular file")
    manifest = {}
    partial = archive.with_suffix(".partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as output:
        for path in files:
            if (path.is_symlink() or os.path.commonpath([
                    os.path.realpath(root), os.path.realpath(path)]) != os.path.realpath(root)):
                raise ValueError("archive member escapes the campaign root")
            relative = path.relative_to(root).as_posix()
            manifest[relative] = {"bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            output.add(path, arcname=relative, recursive=False)
        manifest["prepared/prepared.json"] = {
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        output.add(source, arcname="prepared/prepared.json", recursive=False)
    os.replace(partial, archive)
    result = {"status": "sealed", "archive": archive.name,
              "bytes": archive.stat().st_size,
              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "files": manifest, "resume": "unmeasured",
              "excluded_duplicates_or_resume_state": sorted(excluded) + ["rng_state*"]}
    temporary = marker.with_suffix(".partial")
    write_json(temporary, result)
    os.replace(temporary, marker)
    return result


def wait_for_archive_receipt(archive: Path, expected: dict, timeout: float) -> None:
    """Keep artifact traffic outside the next arm's training measurements."""
    deadline = time.monotonic() + timeout
    receipt = archive.with_suffix(".receipt.json")
    while not receipt.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("numeric archive was not acknowledged by the collector")
        time.sleep(0.2)
    actual = read_json(receipt)
    if any(actual.get(key) != expected[key] for key in ("sha256", "bytes")):
        raise ValueError("collector receipt does not match the sealed numeric archive")


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
        if any(r.get("bf16_conservative", False) != rank0.get("bf16_conservative", False)
               for r in ranks):
            raise ValueError("rank conservative precision controls differ")
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
        if rank0.get("bf16_conservative", False):
            identity["bf16_conservative"] = True
            identity["backend_controls"] = rank0["forward_backend_controls"]
            if any(r.get("forward_backend_controls") != rank0["forward_backend_controls"]
                   or not r.get("bf16_conservative") for r in ranks):
                raise ValueError("rank conservative precision controls differ")
        if any(bool(r.get("bf16_flash")) != bool(rank0.get("bf16_flash")) for r in ranks):
            raise ValueError("rank Flash attention modes differ")
        if rank0.get("bf16_flash"):
            identity.update({"bf16_flash": True,
                "flash_deterministic": rank0["flash_deterministic"],
                "backend_controls": rank0["forward_backend_controls"]})
            for item in ranks:
                if (item.get("bf16_conservative")
                        or item.get("flash_deterministic") != rank0["flash_deterministic"]
                        or item.get("forward_backend_controls")
                        != rank0["forward_backend_controls"]):
                    raise ValueError("rank Flash precision controls differ")
                assert_flash_operators(item.get("flash_operator_counts", {}))
            for item in (reload, checkpoint_reload):
                if (not item.get("bf16_flash")
                        or item.get("backend_controls") != rank0["forward_backend_controls"]):
                    raise ValueError("reload Flash precision controls differ")
                assert_flash_operators(item.get("flash_operator_counts", {}), backward=False)
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
    parser.add_argument("--bf16-conservative", action="store_true")
    parser.add_argument("--bf16-flash", action="store_true")
    parser.add_argument("--flash-deterministic", action="store_true")
    parser.add_argument("--archive-arms", action="store_true")
    parser.add_argument("--archive-receipt-timeout", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--campaign-seconds", type=int, default=5400)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.bf16_conservative and args.precision != "bf16":
        parser.error("--bf16-conservative requires --precision bf16")
    if (args.bf16_flash and (args.precision != "bf16" or args.bf16_conservative)
            or args.flash_deterministic and not args.bf16_flash):
        parser.error("Flash requires BF16, excludes conservative math, and owns determinism")
    if (args.archive_receipt_timeout < 0
            or (args.archive_receipt_timeout and not args.archive_arms)):
        parser.error("a positive archive receipt timeout requires --archive-arms")
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
        if args.bf16_conservative:
            command.append("--bf16-conservative")
        if args.bf16_flash:
            command.append("--bf16-flash")
        if args.flash_deterministic:
            command.append("--flash-deterministic")
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
        if args.archive_arms:
            sealed = archive_completed_arm(root, prepared, name)
            if args.archive_receipt_timeout:
                wait_for_archive_receipt(root.parent / sealed["archive"], sealed,
                                         min(args.archive_receipt_timeout,
                                             max(0, deadline - time.monotonic())))
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
