"""Offline packed-fused session audit; no CUDA run and no task sign-off."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

from benchmarks.harness import d3_fused_trainer_protocol as protocol
from benchmarks.harness import d3_trainer_protocol as historical
from benchmarks.harness.d3_trainer_analysis import cuda_kernel_names, replay_native_tensors
from benchmarks.harness.d3_trainer_gate import numerical_view, require_exact, save


def session_rule(folder, session):
    source = session["source"]
    if session["formal_execution"] is True:
        head = source.get("git_head")
        if not isinstance(head, str) or re.fullmatch("[0-9a-f]{40}", head) is None:
            raise ValueError("actual committed source revision required")
        payload = subprocess.check_output(
            ["git", "-C", str(protocol.ROOT), "show", head + ":" + protocol.PROTOCOL]
        )
        for name, expected in source["git_blob_sha256"].items():
            blob = subprocess.check_output(
                ["git", "-C", str(protocol.ROOT), "show", head + ":" + name]
            )
            require_exact(
                hashlib.sha256(blob).hexdigest(), expected, "actual committed source blob"
            )
    else:
        if source.get("kind") != "synthetic packed-fused launcher only":
            raise ValueError("unidentified nonformal source")
        payload = source["protocol_utf8"].encode()
    require_exact(
        hashlib.sha256(payload).hexdigest(),
        session["protocol_sha256"],
        "session/actual source protocol",
    )
    require_exact(
        hashlib.sha256(payload).hexdigest(),
        source["git_blob_sha256"][protocol.PROTOCOL],
        "source receipt protocol",
    )
    rule = json.loads(payload)
    protocol.validate_rule(rule)
    require_exact(rule, json.loads((folder / "protocol.json").read_text()), "saved rule")
    return rule


def audit_record(session, cell, row, result):
    if (
        row.get("returncode") != 0
        or row.get("status") != "ok"
        or result.get("status") != "ok"
        or result.get("cell") != cell
        or result.get("source") != session["source"]
        or result.get("session_id") != session["session_id"]
        or result.get("protocol_sha256") != session["protocol_sha256"]
        or result.get("formal_execution") != session["formal_execution"]
        or result.get("F1", {}).get("outcome") != "ok"
    ):
        raise ValueError("failed/unknown/void or changed child identity")


def audit_performance(rule, result, oracle, mode):
    frames = [
        {key: frame[key] for key in ("parameter_gradients", "parameters")}
        for frame in oracle["steps"]
    ]
    require_exact(result["frames"], frames, "timed actual gradients and updates")
    require_exact(result["windows"], oracle["windows"], "timed delivered inputs/labels")
    samples = result["samples"]
    if [row["step"] for row in samples] != [1, 2, 3] or [
        row["input_tokens"] for row in samples
    ] != ([1024, 1024, 512]):
        raise ValueError("incomplete optimizer windows or wrong tokens")
    for row in samples:
        for key in ("wall_seconds", "cuda_ms"):
            if isinstance(row[key], bool) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("invalid time sample")
    seconds = sum(row["wall_seconds"] for row in samples) / rule["timed_input_tokens"]
    require_exact(seconds, result["seconds_per_input_token"], "timing reduction")
    counter = "fused_matrices" if mode == "fused" else "grouped_matrices"
    if (
        result["runtime"]["expert_gemm"][counter]
        <= (result["runtime_before_timing"]["expert_gemm"][counter])
    ):
        raise ValueError("no positive candidate compute during timed windows")
    return seconds


def audit_correctness(folder, result):
    """Recheck retained replay records and real saved adapter tensors on CPU."""
    from safetensors.torch import load_file

    from benchmarks.harness.d3_trainer_gate import tensor_record

    positive = json.loads((folder / "positive/record.json").read_text())
    require_exact(positive, result["numerical"], "retained positive replay")
    for replay in ("A-A", "restored"):
        record = json.loads((folder / replay / "record.json").read_text())
        require_exact(numerical_view(record), numerical_view(positive), "retained " + replay)
    negative = json.loads((folder / "negative/record.json").read_text())
    if negative["evaluation"] == positive["evaluation"] or all(
        a["parameter_gradients"] == b["parameter_gradients"]
        for a, b in zip(negative["steps"], positive["steps"], strict=True)
    ):
        raise ValueError("retained negative control is ineffective")
    artifact = load_file(str(folder / "positive/adapter/adapter_model.safetensors"))
    require_exact(
        historical.file_hash(folder / "positive/adapter/adapter_model.safetensors"),
        positive["artifact_sha256"],
        "saved adapter file identity",
    )
    require_exact(
        {name: tensor_record(value) for name, value in artifact.items()},
        positive["final"],
        "actual saved adapter bytes",
    )
    require_exact(result["reload"]["factors"], positive["final"], "reloaded factors")
    require_exact(result["reload"]["evaluation"], positive["evaluation"], "reloaded logits")


def analyze(folder):
    session = json.loads((folder / "session.json").read_text())
    rule = session_rule(folder, session)
    summary = dict(
        status="incomplete_no_verdict",
        d3_accepted=False,
        formal_execution=session["formal_execution"],
        source=session["source"],
        retained_cells=len(session["cells"]),
        complete_pretrained_series=False,
    )
    complete = (
        "complete_pending_offline_analysis_and_review"
        if session["formal_execution"]
        else "synthetic_launcher_complete_pending_analysis"
    )
    if session["status"] != complete:
        summary["reason"] = session.get("traceback", "session did not complete")
        return summary
    allowed = {cell["name"]: cell for cell in protocol.cell_plan(rule, blocks=10)}
    if len(session["cells"]) > 192:
        raise ValueError("attempt budget exceeded")
    records, pids = {}, set()
    for row in session["cells"]:
        name = row["name"]
        if name not in allowed or name in records:
            raise ValueError("undeclared or duplicate cell")
        path = historical.contained(folder, name + "/result.json")
        require_exact(historical.file_hash(path), row["result_sha256"], "retained result hash")
        result = json.loads(path.read_text())
        audit_record(session, allowed[name], row, result)
        if result["pid"] in pids:
            raise ValueError("not distinct fresh child processes")
        pids.add(result["pid"])
        records[name] = result
    pilot = protocol.cell_plan(rule)
    required = [c for c in pilot if c["kind"] != "timing" or session["formal_execution"]]
    if not {cell["name"] for cell in required} <= records.keys():
        raise ValueError("incomplete declared correctness/timing/profile grid")
    if not session["formal_execution"] and len(records) != len(required):
        raise ValueError("undeclared additional synthetic cells")
    oracles = {}
    count = 0
    for codec in protocol.CODECS:
        name = f"correctness-{codec}-ordinary-native-resident"
        oracle = records[name]["result"]["numerical"]
        count += replay_native_tensors(folder / name, oracle)
        oracles[codec] = oracle
    cohorts = {}
    profiles = []
    for name, record in records.items():
        cell, result = record["cell"], record["result"]
        oracle = oracles[cell["codec"]]
        if cell["kind"] == "correctness":
            audit_correctness(folder / name, result)
            if not all(
                result[key] is True
                for key in ("A_A", "negative_effective", "restored", "save_reload")
            ):
                raise ValueError("incomplete numerical controls")
            require_exact(
                numerical_view(result["numerical"]),
                numerical_view(oracle),
                "independent native ordinary oracle",
            )
            continue
        audit_performance(rule, result, oracle, cell["mode"])
        if cell["kind"] == "profile":
            path = folder / name / "cuda-profile.json"
            require_exact(historical.file_hash(path), result["profile_sha256"], "actual trace")
            kernels = cuda_kernel_names(path)
            require_exact(kernels, result["kernel_names"], "actual CUPTI kernel names")
            if not kernels or (cell["mode"] == "fused" and "fused" not in kernels):
                raise ValueError("no actual fused candidate dispatch")
            layers, experts, top_k = (16, 64, 8) if session["formal_execution"] else (4, 4, 2)
            routes = result["profile_routes"]
            if {row["layer"] for row in routes} != set(range(layers)) or any(
                row["rows"] != 512
                or row["top_k"] != top_k
                or len(row["counts"]) != experts
                or sum(row["counts"]) != 512 * top_k
                or min(row["counts"]) < 0
                for row in routes
            ):
                raise ValueError("profile did not cover every declared routed layer")
            profiles.append(
                dict(
                    codec=cell["codec"],
                    training=cell["training"],
                    mode=cell["mode"],
                    kernels=len(kernels),
                    fused=kernels.count("fused"),
                    peaks=result["peaks"],
                )
            )
    if session["formal_execution"]:
        for codec in protocol.CODECS:
            for training in protocol.TRAINING:
                matching = [
                    r
                    for r in records.values()
                    if r["cell"]["kind"] == "timing"
                    and r["cell"]["codec"] == codec
                    and r["cell"]["training"] == training
                ]
                indices = sorted({r["cell"]["block"] for r in matching})
                if indices != list(range(1, len(indices) + 1)):
                    raise ValueError("noncontiguous ABBA blocks")
                blocks = []
                for index in indices:
                    rows = sorted(
                        [r for r in matching if r["cell"]["block"] == index],
                        key=lambda r: r["cell"]["slot"],
                    )
                    blocks.append(
                        [
                            dict(
                                mode=r["cell"]["mode"],
                                outcome=r["F1"]["outcome"],
                                seconds_per_input_token=r["result"]["seconds_per_input_token"],
                            )
                            for r in rows
                        ]
                    )
                estimate = protocol.repeat_target(blocks)
                if estimate["status"] != "complete" or len(blocks) != estimate["target"]:
                    raise ValueError("incomplete or undeclared precision top-up")
                key = f"{codec}/{training}"
                require_exact(estimate, session["repeat_calculations"][key], "repeat calculation")
                ratios = [
                    sum(r["seconds_per_input_token"] for r in block[1:3])
                    / (block[0]["seconds_per_input_token"] + block[3]["seconds_per_input_token"])
                    for block in blocks
                ]
                cohorts[key] = dict(
                    blocks=len(blocks),
                    repeat=estimate,
                    mean_paired_time_change_pct=100 * (statistics.fmean(ratios) - 1),
                    block_time_ratios=ratios,
                )
                for mode in ("grouped", "fused"):
                    arms = [r for r in matching if r["cell"]["mode"] == mode]
                    costs = [r["result"]["seconds_per_input_token"] for r in arms]
                    cohorts[key][mode] = {
                        "mean_seconds_per_input_token": statistics.fmean(costs),
                        "input_tokens_per_second": 1 / statistics.fmean(costs),
                        "allocator_peak_bytes": {
                            name: {
                                "min": min(r["result"]["peaks"][name] for r in arms),
                                "median": statistics.median(
                                    r["result"]["peaks"][name] for r in arms
                                ),
                                "max": max(r["result"]["peaks"][name] for r in arms),
                            }
                            for name in ("allocated", "reserved")
                        },
                        "whole_child_peak_rss_bytes": {
                            "min": min(r["process_peak_rss_bytes"] for r in arms),
                            "max": max(r["process_peak_rss_bytes"] for r in arms),
                        },
                    }
    summary.update(
        status="qualification_complete_pending_review"
        if session["formal_execution"]
        else "synthetic_launcher_checks_passed",
        complete_pretrained_series=session["formal_execution"],
        native_tensor_payloads_verified=count,
        fresh_processes=len(pids),
        profiles=profiles,
        cohorts=cohorts,
    )
    return summary


if __name__ == "__main__":
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session)
    save(args.session / "analysis.json", result)
    Console().print(result["status"])
