"""Prospective F2 allocation/overflow cohort; never rewrites the earlier study."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import f2_reference as f2
from rich.console import Console

GROUPS = {"ram": ("ram", "4bit"), "overflow": ("nvme", "none"),
          "boundary": ("ram", "none")}
PROTOCOL = f2.ROOT / "benchmarks/f2-capacity-20261020.md"
console = Console()


def memory() -> dict:
    import psutil

    from soup_cli.utils.layer_stream import free_ram_bytes, total_ram_bytes

    root = Path("/sys/fs/cgroup")
    return {"unix_s": time.time(), "host": psutil.virtual_memory()._asdict(),
            "effective_total": total_ram_bytes(), "effective_available": free_ram_bytes(),
            "cgroup": {name: (root / name).read_text() for name in (
                "memory.low", "memory.max", "memory.current", "memory.stat", "memory.events")
                if (root / name).exists()}}


def idle(gpu: str) -> None:
    peers = subprocess.check_output(["nvidia-smi", "--id=" + gpu,
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if peers:
        raise ValueError("assigned GPU has compute peers; leave them untouched")


def prepare(root: Path, checkpoint: Path, deadline: float) -> None:
    if os.path.commonpath([os.path.realpath(root), os.path.realpath(f2.ROOT)]) != os.path.realpath(
        f2.ROOT
    ):
        raise ValueError("campaign must be contained in the measured checkout")
    root.mkdir(parents=True, exist_ok=False)
    for group, (mode, quant) in GROUPS.items():
        cache = root.parent / ("f2-capacity-cache-nf4" if quant == "4bit"
                               else "f2-capacity-cache-bf16")
        directory = root / group
        f2.prepare(directory, checkpoint, cache)
        for length in f2.LENGTHS:
            path = directory / "configs" / f"{mode}-{length}.json"
            config = f2.read_json(path)
            config["training"]["quantization"] = quant
            if group == "overflow":
                config["training"]["stream_source"] = "auto"
            f2.write_json(path, config)
        plan = f2.read_json(directory / "plan.json")
        plan.update(matrix_approved=True, matrix_approval_reference=
            "User explicitly requested completion on allocated 32GB with natural overflow",
            window_ends_unix=deadline, resource_window="six-hour bounded F2 allocation study")
        f2.write_json(directory / "plan.json", plan)
    schedule = []
    for block in range(5):
        schedule.extend(block_schedule(block, len(schedule)))
    f2.write_json(root / "schedule.json", schedule)
    f2.write_json(root / "supplement-schedule.json", block_schedule(5, len(schedule)))
    f2.write_json(root / "allocation-initial.json", memory())
    hardware = Path("/workspace/f2-capacity-hardware-initial.json")
    if hardware.exists():
        f2.write_json(root / "hardware-initial.json", f2.read_json(hardware))


def block_schedule(block: int, offset: int) -> list[dict]:
    rows = []
    for length in f2.LENGTHS:
        for group in ("ram", "overflow", "overflow", "ram", "boundary"):
            kinds = ("train", "bench") if (offset + len(rows)) % 4 == 0 else ("bench", "train")
            for kind in kinds:
                rows.append({"block": block, "group": group, "length": length,
                             "kind": kind, "index": offset + len(rows) + 1})
    return rows


def capacity_verdict(row: dict, log: str) -> tuple[str, str]:
    if row.get("validity", {}).get("outcome") != "ok":
        return "NO_VERDICT", "F1 invalid or unknown"
    physical = row.get("audit", {}).get("physical_vram_bytes", 0)
    if not 7.5 * 1024**3 <= physical <= 8.5 * 1024**3:
        return "NO_VERDICT", "physical 8GB GPU evidence missing"
    text = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", log).split())
    message = ("base exceeds the physical RAM safety ceiling" in text or
               ("stream_source='ram'" in text and "GB of RAM is free" in text
                and "stream_source='auto'" in text))
    code = row.get("returncode")
    if (isinstance(code, int) and code != 0 and not row.get("timeout") and message
            and not row.get("audit", {}).get("steps")):
        return "EXPECTED_CAPACITY_REFUSAL", "native refusal against actual RAM allocation budget"
    return "FAIL", "real capacity refusal absent; unrelated errors never count"


def freeze(root: Path) -> None:
    from soup_cli.config.schema import SoupConfig

    if f2.git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit source/controller/protocol before freeze")
    stamp = memory()
    baseline = int(stamp["cgroup"]["memory.low"])
    if not 32e9 <= baseline <= 36e9 or stamp["effective_total"] != baseline:
        raise ValueError("actual 32GB baseline allocation is not observed by shipped probes")
    for group, (mode, _) in GROUPS.items():
        directory = root / group
        for length in f2.LENGTHS:
            path = directory / "configs" / f"{mode}-{length}.json"
            resolved = SoupConfig.model_validate(f2.read_json(path)).model_dump(mode="json")
            f2.write_json(path, resolved)
        control = "ram" if group == "ram" else "overflow"
        clean = f2.read_json(root / f"clean-{control}.json")
        foreign = f2.read_json(root / f"foreign-{control}.json")
        if clean["outcome"] != "ok" or foreign["checks"]["foreign_reads"]["outcome"] != "void":
            raise ValueError("fresh clean/foreign F1 calibration missing")
    stamp_gpu = f2.preflight(root / "ram")["gpu"]
    if not stamp_gpu or not 7.5 * 1024**3 <= stamp_gpu["memory_bytes"] <= 8.5 * 1024**3:
        raise ValueError("physical 8GB GPU required")
    idle(stamp_gpu["uuid"])
    manifest = {}
    for group in GROUPS:
        plan = f2.read_json(root / group / "plan.json")
        checkpoint = Path(plan["checkpoint"])
        source = f2.read_json(checkpoint / "f2-source.json")
        if source.get("model") != f2.MODEL or source.get("revision") != f2.REVISION:
            raise ValueError("real pinned checkpoint provenance missing")
        cache = Path(plan["cache"])
        mapping = f2.read_json(root / ("mapping-nf4.json" if group == "ram"
                                      else "mapping-bf16.json"))
        files = sorted(cache.rglob("*.safetensors"))
        if len(files) != 51 or mapping["cache_files"] != {str(p): f2.digest(p) for p in files}:
            raise ValueError("mapping does not cover the actual complete cache")
        if not mapping["valid"] or mapping["device"] != "nvme0n1":
            raise ValueError("strict actual-cache physical mapping failed")
        for base in (checkpoint, cache):
            for path in sorted(base.rglob("*")):
                if path.is_file() and ".cache" not in path.parts:
                    manifest[str(path)] = {"sha256": f2.digest(path),
                        "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    inputs = [PROTOCOL, Path(__file__), root / "schedule.json", root / "supplement-schedule.json",
              root / "allocation-initial.json", root / "hardware-initial.json"]
    for group in GROUPS:
        inputs.extend([root / group / "plan.json", root / group / "train.jsonl"])
        inputs.extend(sorted((root / group / "configs").glob("*.json")))
        control = "ram" if group == "ram" else "overflow"
        inputs.extend([root / f"clean-{control}.json", root / f"foreign-{control}.json"])
    inputs.extend([root / "mapping-nf4.json", root / "mapping-bf16.json"])
    initial = f2.read_json(root / "hardware-initial.json")["files"]
    usage = int(initial["/sys/fs/cgroup/memory.current"])
    stat = dict(line.split() for line in initial["/sys/fs/cgroup/memory.stat"].splitlines())
    initial_free = baseline - max(0, usage - int(stat.get("inactive_file", 0)))
    if initial_free < 32e9:
        raise ValueError("initial actual allocation-free RAM was below 32GB")
    frozen = {"commit": f2.git("rev-parse", "HEAD"), "unix_s": time.time(),
        "memory": stamp, "initial_baseline_free_bytes": initial_free,
        "gpu": stamp_gpu, "files": manifest,
        "inputs": {str(p): f2.digest(p) for p in inputs}, "physical_disk": "nvme0n1"}
    if (root / "frozen.json").exists():
        raise FileExistsError("freeze is immutable")
    f2.write_json(root / "frozen.json", frozen)


def validate(root: Path, frozen: dict) -> None:
    if frozen["commit"] != f2.git("rev-parse", "HEAD") or f2.git(
        "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("source changed after freeze")
    for name, digest in frozen["inputs"].items():
        if f2.digest(Path(name)) != digest:
            raise ValueError("frozen input changed: " + name)
    for name, item in frozen["files"].items():
        stamp = Path(name).stat()
        if (stamp.st_size, stamp.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
            raise ValueError("checkpoint/cache metadata changed: " + name)
    if int(memory()["cgroup"]["memory.low"]) != frozen["memory"]["effective_total"]:
        raise ValueError("actual RAM allocation changed")


def run(root: Path) -> None:
    frozen = f2.read_json(root / "frozen.json")
    scheduled = f2.read_json(root / "schedule.json")
    outcomes_path = root / "outcomes.json"
    # A resume never silently repeats a completed command.
    outcomes = f2.read_json(outcomes_path)["outcomes"] if outcomes_path.exists() else []
    done = {r["index"] for r in outcomes}
    if any(index > 150 for index in done):
        scheduled.extend(f2.read_json(root / "supplement-schedule.json"))
    for entry in scheduled:
        if entry["index"] in done:
            continue
        validate(root, frozen)
        group, kind, index = entry["group"], entry["kind"], entry["index"]
        directory = root / group
        plan = f2.read_json(directory / "plan.json")
        deadline = plan["window_ends_unix"]
        if time.time() + 1800 >= deadline:
            break
        idle(frozen["gpu"]["uuid"])
        mode = GROUPS[group][0]
        cell = f"{mode}-{entry['length']}"
        destination = directory / "runs" / f"{index:03d}-{cell}-{kind}"
        if destination.exists():
            raise ValueError("interrupted raw command retained; resolve explicitly before resume")
        before = memory()
        row = f2.run_one(directory, cell, kind, index, frozen["gpu"]["uuid"],
                         deadline=deadline, physical_disk=frozen["physical_disk"])
        f2.write_json(destination / "allocation-before.json", before)
        status, reason = row["status"], row["reason"]
        if group == "boundary":
            status, reason = capacity_verdict(row, (destination / "console.log").read_text())
            f2.write_json(destination / "capacity-classification.json", {
                "status": status, "reason": reason, "raw_generic_f2_status": row["status"]})
        elif group == "overflow":
            config = f2.read_json(directory / "configs" / f"{cell}.json")
            if config["training"]["stream_source"] != "auto":
                raise ValueError("natural overflow must never force disk")
        outcomes.append({**entry, "status": status, "reason": reason,
                         "raw_result": str(destination / "result.json")})
        f2.write_json(outcomes_path, {"outcomes": outcomes,
            "counts": dict(Counter(r["status"] for r in outcomes)),
            "scheduled": len(scheduled), "pooled_with_prior": False})
        console.print(f"{index}/{len(scheduled)} {group} {entry['length']} {kind}: {status}")
        if index == 150 and needs_supplement(outcomes):
            scheduled.extend(f2.read_json(root / "supplement-schedule.json"))
    f2.write_json(root / "allocation-final.json", memory())


def needs_supplement(outcomes: list[dict]) -> bool:
    for group in GROUPS:
        accepted = "EXPECTED_CAPACITY_REFUSAL" if group == "boundary" else "PASS"
        for length in f2.LENGTHS:
            for kind in ("train", "bench"):
                count = sum(r["group"] == group and r["length"] == length
                            and r["kind"] == kind and r["status"] == accepted
                            for r in outcomes)
                if count < 5:
                    return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "freeze", "run"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.checkpoint is None or args.deadline is None:
            parser.error("prepare requires --checkpoint and --deadline")
        prepare(args.directory, args.checkpoint, args.deadline)
    else:
        globals()[args.phase](args.directory)


if __name__ == "__main__":
    main()
