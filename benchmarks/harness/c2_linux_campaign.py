#!/usr/bin/env python3
"""Prepare the frozen cache or run all twelve C2 step arms without replacement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from c2_depth_probe import save


def flags(weights: str, shards: str) -> list[str]:
    return ["--weights", weights, "--shards", shards, "--quant", "nf4", "--tier", "disk",
            "--seq", "256", "--batch", "1", "--read-ahead", "3", "--buffers", "2",
            "--steps", "3", "--warmup", "1", "--step", "--skip-events"]


def cache_identity(shards: str) -> dict:
    return {str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(Path(shards).glob("*.safetensors"))}


def prepare_model(args: argparse.Namespace) -> int:
    import stream_probe

    if Path(args.out).exists():
        raise ValueError("preparation record exists; refusing overwrite")
    sys.argv = [sys.argv[0], *flags(args.weights, args.shards), "--out", args.out]
    standard = stream_probe.parse_args()
    model, runtime, _config, _index, _weights, _shards, shard_s, build_s = stream_probe.build(
        standard, "cuda", "bfloat16"
    )
    runtime.close()
    del model
    for path in [*Path(args.shards).glob("*"), *Path(args.weights).glob("*")]:
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    save(Path(args.out), {"kind": "cache preparation only / no training step",
                          "shard_seconds": shard_s, "build_seconds": build_s,
                          "cache": cache_identity(args.shards)})
    return 0


def campaign(args: argparse.Namespace) -> int:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    primitive_path = Path(args.primitive)
    primitive = json.loads(primitive_path.read_text())
    record = {"kind": "synthetic 70B-shaped real GPU training-step measurement",
              "primitive": primitive["decision"],
              "primitive_sha256": hashlib.sha256(primitive_path.read_bytes()).hexdigest(),
              "rounds": [], "complete": False, "invalid_reasons": [],
              "cache_before": cache_identity(args.shards)}
    save(root / "campaign.json", record)
    env = dict(os.environ, OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
    try:
        for index in range(6):
            entry = {"order": [1, 2] if index % 2 == 0 else [2, 1], "arms": {}}
            record["rounds"].append(entry)
            for depth in entry["order"]:
                name = f"r{index}-d{depth}"
                path = root / f"{name}.json"
                command = [sys.executable, str(Path(__file__).with_name("c2_step_probe.py")),
                           "--c2-depth", str(depth), "--c2-device", args.device,
                           *flags(args.weights, args.shards), "--out", str(path)]
                entry["arms"][str(depth)] = {
                    "file": path.name, "command": command, "status": "started"
                }
                save(root / "campaign.json", record)
                with (root / f"{name}.log").open("w") as log:
                    done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                          env=env, check=False, timeout=600)
                entry["arms"][str(depth)].update(exit_code=done.returncode, status="finished")
                save(root / "campaign.json", record)
                if done.returncode:
                    raise RuntimeError(f"{name} exited {done.returncode}; keep arm and stop")
                data = json.loads(path.read_text())
                if data["c2_validity"]["reasons"]:
                    raise RuntimeError(f"{name} invalid: {data['c2_validity']['reasons']}")
                point = next(point for point in data["records"] if point["label"] == "step_plain")
                entry["arms"][str(depth)].update(
                    step_s=point["step_s_mean"], peak_vram=point["peak_alloc_gb"],
                    staging=data["meta"]["c2_source_admission"]["staging_bytes"],
                    losses=[step["loss"] for step in point["records"]],
                )
                save(root / "campaign.json", record)
        record["complete"] = True
    except BaseException as error:
        record["invalid_reasons"].append(repr(error))
    finally:
        record["cache_after"] = cache_identity(args.shards)
        if record["cache_after"] != record["cache_before"]:
            record["invalid_reasons"].append("cache changed")
        record["decision"] = {"verdict": "NO VERDICT / keep shipped depth 1"}
        if record["complete"] and not record["invalid_reasons"]:
            ratios = [entry["arms"]["1"]["step_s"] / entry["arms"]["2"]["step_s"]
                      for entry in record["rounds"]]
            memory_ok = all(
                entry["arms"]["1"]["staging"] == entry["arms"]["2"]["staging"]
                and entry["arms"]["2"]["peak_vram"] <= entry["arms"]["1"]["peak_vram"]
                + max(entry["arms"]["1"]["peak_vram"] * 0.02, 64 * 2**20 / 1e9)
                for entry in record["rounds"]
            )
            benefit = all(ratio >= 1.05 for ratio in ratios) and statistics.median(ratios) >= 1.10
            record["decision"].update(speedups=ratios, median_speedup=statistics.median(ratios),
                                      memory_ok=memory_ok, repeated_step_benefit=benefit)
            if (primitive["decision"]["verdict"] == "PASS primitive" and benefit and memory_ok):
                record["decision"]["verdict"] = (
                    "PASS candidate; production integration still required"
                )
            else:
                record["decision"]["verdict"] = "DO NOT IMPLEMENT / keep shipped depth 1"
        save(root / "campaign.json", record)
    return 0 if record["complete"] and not record["invalid_reasons"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--device", default="nvme0n1")
    parser.add_argument("--primitive")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepare-model", action="store_true")
    args = parser.parse_args()
    if args.prepare_model:
        return prepare_model(args)
    if not args.primitive:
        parser.error("--primitive is required for a campaign")
    return campaign(args)


if __name__ == "__main__":
    sys.exit(main())
