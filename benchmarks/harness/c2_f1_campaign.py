#!/usr/bin/env python3
"""Fresh-process C2 ABBA blocks, watched by the project's unchanged F1 sensors."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from arm_validity import ArmWatch, Run, block_order, blocks_to_add, repeats
from c2_depth_probe import save
from c2_linux_campaign import cache_identity, flags
from c2_linux_depth_probe import DIAGNOSTIC_PROTOCOL, counters


def evaluate(blocks: list, kind: str) -> dict:
    values = [[Run(arm["value"], arm["watch"]["outcome"]) for arm in block]
              for block in blocks]
    additional = blocks_to_add(values)
    result = {"additional_blocks": additional, "verdict": "NO VERDICT"}
    if len(blocks) >= 2:
        result["repeats"] = dataclasses.asdict(repeats(values))
    if additional or any(arm["watch"]["outcome"] != "ok"
                         for block in blocks for arm in block):
        return result
    ratios = []
    mean_ratios = []
    memory_ok = True
    for block in blocks:
        for left, right in (block[:2], block[2:]):
            a, b = (left, right) if left["depth"] == 1 else (right, left)
            ratios.append(b["value"] / a["value"] if kind == "primitive"
                          else a["value"] / b["value"])
            if kind == "step":
                mean_ratios.append(a["mean_step_s"] / b["mean_step_s"])
                memory_ok &= (a["staging"] == b["staging"] and b["vram"] <= a["vram"]
                              + max(a["vram"] * 0.02, 64 * 2**20 / 1e9))
    result.update(ratios=ratios, median_ratio=statistics.median(ratios), memory_ok=memory_ok)
    if kind == "primitive":
        if all(ratio >= 1.10 for ratio in ratios):
            result["verdict"] = "PASS primitive"
        elif all(0.95 <= ratio <= 1.05 for ratio in ratios):
            result["verdict"] = "FAIL benefit"
        elif all(ratio < 0.95 for ratio in ratios):
            result["verdict"] = "FAIL regression"
        else:
            result["verdict"] = "NO VERDICT / ambiguous"
    else:
        mean_ok = (all(ratio >= 1.05 for ratio in mean_ratios)
                   and statistics.median(mean_ratios) >= 1.10)
        result.update(mean_ratios=mean_ratios, original_mean_rule_ok=mean_ok)
        result["verdict"] = ("PASS step" if memory_ok and all(ratio >= 1.05 for ratio in ratios)
                             and statistics.median(ratios) >= 1.10 and mean_ok
                             else "FAIL step benefit")
    return result


def campaign(args: argparse.Namespace) -> int:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    admission_path = Path(args.admission)
    admission = json.loads(admission_path.read_text())
    if (admission.get("physical_device") != args.device or not admission.get("correctness_ok")
            or not admission.get("direct_io_verified")):
        raise ValueError("a prior direct-I/O mapping and full-byte admission is required")
    record = {"kind": args.kind, "blocks": [], "complete": False, "errors": [],
              "diagnostic_writes": getattr(args, "diagnostic_writes", False),
              "admission_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
              "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "device": args.device, "max_blocks": args.max_blocks,
              "required": ["suspend", "power", "foreign_reads"],
              "cache_before": cache_identity(args.shards) if args.kind == "step" else None}
    record["diagnostic_protocol_sha256"] = (
        hashlib.sha256(DIAGNOSTIC_PROTOCOL.read_bytes()).hexdigest()
        if record["diagnostic_writes"] else None
    )
    save(root / "campaign.json", record)
    try:
        while True:
            decision = evaluate(record["blocks"], args.kind)
            record["decision"] = decision
            if record["diagnostic_writes"]:
                record["effect_decision"] = decision
                record["decision"] = {"verdict": "DIAGNOSTIC ONLY / original quiet gate not met"}
            save(root / "campaign.json", record)
            if decision["additional_blocks"] == 0:
                record["complete"] = True
                break
            if len(record["blocks"]) >= args.max_blocks:
                raise RuntimeError("predeclared block cap reached; incomplete, no verdict")
            block = []
            record["blocks"].append(block)
            for position, name in enumerate(block_order(1)[0]):
                depth = 1 if name == "A" else 2
                stem = f"b{len(record['blocks']) - 1}-p{position}-d{depth}"
                path = root / f"{stem}.json"
                folder = Path(__file__).parent
                if args.kind == "primitive":
                    command = [sys.executable, str(folder / "c2_linux_depth_probe.py"),
                               "--fixture", args.fixture, "--single-depth", str(depth),
                               "--out", str(path)]
                else:
                    command = [sys.executable, str(folder / "c2_step_probe.py"),
                               "--c2-depth", str(depth), "--c2-device", args.device,
                               *flags(args.weights, args.shards), "--out", str(path)]
                if record["diagnostic_writes"]:
                    command.append("--diagnostic-writes" if args.kind == "primitive"
                                   else "--c2-diagnostic-writes")
                arm = {"depth": depth, "command": command, "file": path.name, "status": "started"}
                block.append(arm)
                save(root / "campaign.json", record)
                # Admission independently established the overlay's unique physical leaf.
                # F1's default other_disk_reads covers all physical leaves, excluding stacks.
                with ArmWatch(arm=name, round=len(record["blocks"]), run=position,
                              label=f"C2 {args.kind} synthetic", required=record["required"],
                              model_path=args.fixture if args.kind == "primitive" else args.shards,
                              disk_device=lambda _: args.device,
                              disk_reads=lambda device: counters()[device]["read"]) as watch:
                    with (root / f"{stem}.log").open("w") as log:
                        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                              env=dict(os.environ, OMP_NUM_THREADS="8",
                                                       TOKENIZERS_PARALLELISM="false"),
                                              timeout=600, check=False)
                arm.update(watch=watch.record, exit_code=done.returncode, status="finished")
                save(root / "campaign.json", record)
                if done.returncode or watch.record["outcome"] != "ok":
                    raise RuntimeError(f"{stem} invalid; retain whole sequence and stop")
                data = json.loads(path.read_text())
                if args.kind == "primitive":
                    if not data["complete"] or data["invalid_reasons"]:
                        raise RuntimeError(f"{stem} primitive incomplete/invalid")
                    arm["value"] = data["rounds"][0]["arms"][str(depth)]["gb_s"]
                else:
                    if data["c2_validity"]["reasons"]:
                        raise RuntimeError(f"{stem} step invalid")
                    point = next(row for row in data["records"] if row["label"] == "step_plain")
                    arm.update(value=statistics.median(step["step_s"] for step in point["records"]),
                               mean_step_s=point["step_s_mean"], vram=point["peak_alloc_gb"],
                               staging=data["meta"]["c2_source_admission"]["staging_bytes"])
                save(root / "campaign.json", record)
    except BaseException as error:
        record["errors"].append(repr(error))
        record["decision"] = {"verdict": "NO VERDICT"}
    finally:
        if args.kind == "step":
            record["cache_after"] = cache_identity(args.shards)
            if record["cache_after"] != record["cache_before"]:
                record["errors"].append("immutable cache changed")
                record["decision"] = {"verdict": "NO VERDICT"}
        save(root / "campaign.json", record)
    return 0 if record["complete"] and not record["errors"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["primitive", "step"], required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument("--shards", default="")
    parser.add_argument("--admission", required=True)
    parser.add_argument("--device", default="nvme0n1")
    parser.add_argument("--max-blocks", type=int, default=20)
    parser.add_argument("--diagnostic-writes", action="store_true")
    parser.add_argument("--out", required=True)
    return campaign(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
