"""Verify and summarize a retained D3 trained-model training series."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

from d3_full_model import file_hash
from d3_training_protocol import CODECS, MODES, RELATIVE_PROTOCOL, ROOT, read_rule


def analyze(folder: Path) -> dict:
    rule = read_rule()
    session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    if session.get("status") != "complete_pending_offline_analysis_and_review":
        raise ValueError("only a complete retained series can be analyzed")
    if len(session.get("cells", [])) != rule["expected_cell_count"]:
        raise ValueError("series cell count differs from the frozen attempt budget")
    if (len({cell.get("name") for cell in session["cells"]}) != rule["expected_cell_count"]
            or session.get("protocol_sha256") != file_hash(ROOT / RELATIVE_PROTOCOL)):
        raise ValueError("session has duplicate cells or a mismatched protocol identity")
    expected_names = {
        f"correctness-{codec}-{mode}" for codec, mode in itertools.product(CODECS, MODES)
    }
    expected_names |= {
        f"timing-{codec}-block-{block:02d}-{position}-{mode}"
        for codec in CODECS
        for block in range(1, rule["measurement_blocks_per_codec"] + 1)
        for position, mode in enumerate((*rule["timing_pair"], *rule["timing_pair"][::-1]), 1)
    }
    expected_names |= {
        f"profile-{codec}-{mode}"
        for codec in CODECS for mode in rule["timing_pair"]
    }
    if {cell["name"] for cell in session["cells"]} != expected_names:
        raise ValueError("series cell identities do not match the preregistered design")
    report = {"status": "analysis_complete_review_required", "d3_accepted": False,
              "source": session["source"], "protocol_sha256": session["protocol_sha256"],
              "session_id": session["session_id"], "cells": [], "timing": {},
              "profiles": {}, "scope": rule["scope"]}
    cell_rows = {}
    for cell in session["cells"]:
        result_path = folder / cell["name"] / "result.json"
        row = json.loads(result_path.read_text(encoding="utf-8"))
        if (file_hash(result_path) != cell["result_sha256"] or row.get("status") != "ok"
                or row.get("F1", {}).get("outcome") != "ok"
                or row.get("source") != session["source"]
                or row.get("session_id") != session["session_id"]
                or row.get("protocol_sha256") != session["protocol_sha256"]
                or row.get("kind") != cell["kind"] or row.get("mode") != cell["mode"]
                or row.get("codec") != cell["codec"] or cell.get("returncode") != 0):
            raise ValueError(f"retained cell failed identity or validity: {cell['name']}")
        report["cells"].append({"name": cell["name"], "kind": row["kind"],
                                "codec": row["codec"], "mode": row["mode"],
                                "F1": row["F1"]["outcome"], "status": row["status"]})
        cell_rows[cell["name"]] = row
        if row["kind"] == "correctness":
            if (row.get("A_A_exact") is not True or row.get("restored_exact") is not True
                    or row.get("control_target") is None
                    or not row.get("negative_tensor_sha256")
                    or len(row.get("steps", [])) != rule["steps"]):
                raise ValueError(f"correctness controls are incomplete: {cell['name']}")
            if row["mode"] == "native":
                expected_files = {f"step-{index}.safetensors" for index in range(rule["steps"])}
                if set(row.get("oracle_files", {})) != expected_files:
                    raise ValueError("native training-step oracle files are incomplete")
                for name, digest in row["oracle_files"].items():
                    if Path(name).name != name or file_hash(folder / cell["name"] / name) != digest:
                        raise ValueError(f"native training-step oracle changed: {name}")
        if row["kind"] == "timing":
            if len(row.get("steps", [])) != rule["timed_steps"]:
                raise ValueError(f"timing arm lacks its fixed step count: {cell['name']}")
            durations = [step["seconds"] for step in row["steps"]]
            if any(not math.isfinite(value) or value <= 0 for value in durations):
                raise ValueError(f"timing arm has invalid duration: {cell['name']}")
            report["timing"].setdefault(row["codec"], []).append({
                "block": int(cell["name"].split("-block-")[1].split("-")[0]),
                "mode": row["mode"],
                "step_seconds": durations,
                "mean_step_seconds": statistics.fmean(durations),
                "peak_allocated_bytes": max(step["peak_allocated"] for step in row["steps"]),
                "peak_reserved_bytes": max(step["peak_reserved"] for step in row["steps"]),
                "peak_process_rss_bytes": row["peak_process_rss_bytes"],
            })
        elif row["kind"] == "profile":
            if (len(row.get("steps", [])) != 1
                    or row["steps"][0].get("cuda_kernel_events", 0) < 1
                    or not row["steps"][0].get("tensor_sha256")):
                raise ValueError(f"profile arm lacks a CUDA kernel trace: {cell['name']}")
            report["profiles"].setdefault(row["codec"], {})[row["mode"]] = row["steps"][0]

    for codec in CODECS:
        native = cell_rows[f"correctness-{codec}-native"]
        for mode in MODES:
            candidate = cell_rows[f"correctness-{codec}-{mode}"]
            if len(candidate["steps"]) != rule["steps"]:
                raise ValueError(f"correctness step count differs for {codec}/{mode}")
            for index, step in enumerate(candidate["steps"]):
                if step.get("tensor_sha256") != native["steps"][index].get("tensor_sha256"):
                    raise ValueError(f"native oracle tensor hashes differ for {codec}/{mode}")
        for block in range(1, rule["measurement_blocks_per_codec"] + 1):
            for position, mode in enumerate(
                (*rule["timing_pair"], *rule["timing_pair"][::-1]), 1,
            ):
                name = f"timing-{codec}-block-{block:02d}-{position}-{mode}"
                row = cell_rows[name]
                for offset, step in enumerate(row["steps"]):
                    oracle = native["steps"][rule["warmup_steps"] + offset]
                    if (not set(step.get("tensor_sha256", {})) <= set(oracle["tensor_sha256"])
                            or any(step["tensor_sha256"].get(key) != oracle["tensor_sha256"][key]
                                   for key in step.get("tensor_sha256", {}))):
                        raise ValueError(f"timed step differs from native oracle: {name}")
        for mode in rule["timing_pair"]:
            name = f"profile-{codec}-{mode}"
            hashes = cell_rows[name]["steps"][0]["tensor_sha256"]
            oracle = native["steps"][rule["warmup_steps"]]["tensor_sha256"]
            if (not set(hashes) <= set(oracle)
                    or any(hashes.get(key) != oracle[key] for key in hashes)):
                raise ValueError(f"profile step differs from native oracle: {name}")
        arms = report["timing"].get(codec, [])
        blocks = {}
        for arm in arms:
            blocks.setdefault(arm["block"], {}).setdefault(arm["mode"], []).append(arm)
        if len(blocks) != rule["measurement_blocks_per_codec"]:
            raise ValueError(f"incomplete ABBA timing blocks for {codec}")
        contrasts = []
        for number in sorted(blocks):
            pair = blocks[number]
            if (set(pair) != set(rule["timing_pair"])
                    or any(len(mode_arms) != 2 for mode_arms in pair.values())):
                raise ValueError(f"ABBA block {number} lacks an arm for {codec}")
            control, candidate = (statistics.fmean(arm["mean_step_seconds"] for arm in pair[mode])
                                  for mode in rule["timing_pair"])
            contrasts.append({"block": number, "control_seconds": control,
                              "candidate_seconds": candidate,
                              "relative_change": (candidate - control) / control})
        report["timing"][codec] = {
            "arms": arms, "ABBA_contrasts": contrasts,
            "control_mode": rule["timing_pair"][0],
            "candidate_mode": rule["timing_pair"][1],
            "control_median_seconds": statistics.median(
                statistics.fmean(arm["mean_step_seconds"] for arm in pair[rule["timing_pair"][0]])
                for pair in blocks.values()),
            "candidate_median_seconds": statistics.median(
                statistics.fmean(arm["mean_step_seconds"] for arm in pair[rule["timing_pair"][1]])
                for pair in blocks.values()),
            "median_relative_change": statistics.median(
                pair["relative_change"] for pair in contrasts),
            "scope": "descriptive five-block comparison; no generalized speed claim",
        }
        if set(report["profiles"].get(codec, {})) != set(rule["timing_pair"]):
            raise ValueError(f"profile arms incomplete for {codec}")
    report["correctness_cells"] = sum(cell["kind"] == "correctness" for cell in report["cells"])
    report["timing_cells"] = sum(cell["kind"] == "timing" for cell in report["cells"])
    report["profile_cells"] = sum(cell["kind"] == "profile" for cell in report["cells"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{result['status']}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
