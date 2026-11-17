"""Separate prospective packed-fused qualification; historical rules stay unchanged."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import zipfile
from pathlib import Path

from benchmarks.harness import d3_trainer_protocol as original

ROOT = original.ROOT
PROTOCOL = "benchmarks/protocols/d3-fused-trainer/protocol.json"
RECEIPT = "FROZEN-D3-FUSED-SOURCE.json"
CODECS = ("nf4", "nf4-dq")
TRAINING = ("ordinary", "l2l")
MODES = ("native-resident", "sequential", "grouped-resident", "grouped", "fused-resident", "fused")
ORDER = ("grouped", "fused", "fused", "grouped")
ENGINES = tuple(
    "benchmarks/harness/" + name + ".py"
    for name in (
        "d3_fused_trainer_protocol",
        "d3_fused_trainer_gate",
        "d3_fused_trainer_series",
        "d3_fused_trainer_analysis",
        "d3_fused_trainer_prepare",
        "d3_trainer_gate",
        "d3_trainer_series",
        "d3_trainer_protocol",
        "d3_trainer_prepare",
        "d3_trainer_selftest",
        "d3_f1_mapping_probe",
        "d3_f1_controls",
        "arm_validity",
    )
)


def read_rule(root=ROOT):
    return json.loads(original.contained(root, PROTOCOL).read_text())


def validate_rule(rule):
    fixed = {
        "task": "D3 packed fused public trainer qualification",
        "variant": "packed-fused-v1",
        "protocol_revision": 1,
        "formats": list(CODECS),
        "correctness_modes": list(MODES),
        "timing_order": list(ORDER),
        "fused_native_rows_from": None,
        "fused_native_rows_max": 128,
        "compute_capability": [8, 6],
        "input_seed": 1729,
        "adapter_seed": 19,
        "data_seed": 29,
        "optimizer": {
            "name": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "scheduler": "constant",
            "warmup": 0,
            "max_grad_norm": 1.0,
            "epochs": 1,
            "gradient_windows": [2, 2, 2, 1],
        },
        "cell_timeout_seconds": 1200,
        "session_timeout_seconds": 129600,
        "attempt_budget": {
            "correctness_cells": 24,
            "timing_pilot_cells": 80,
            "timing_max_cells": 160,
            "profile_cells": 8,
            "minimum_total_cells": 112,
            "maximum_total_cells": 192,
        },
    }
    for key, expected in fixed.items():
        if type(rule.get(key)) is not type(expected) or rule[key] != expected:
            raise ValueError(f"fused prospective contract changed: {key}")
    # Reuse the unchanged full numerical/optimizer/data/precision contract.
    reference = copy.deepcopy(rule)
    reference.update(
        task="D3 public Soup SFT/L2L qualification",
        protocol_revision=3,
        formats=list(original.CODECS),
        correctness_modes=list(original.CORRECTNESS_MODES),
        timing_order=list(original.TIMING_MODES),
        cell_timeout_seconds=900,
        attempt_budget={
            "correctness_cells": 36,
            "timing_pilot_cells": 180,
            "timing_max_cells": 360,
            "profile_cells": 18,
            "minimum_total_cells": 234,
            "maximum_total_cells": 414,
        },
    )
    original.validate_rule(reference)


def cell_plan(rule, *, blocks=None):
    validate_rule(rule)
    count = 5 if blocks is None else blocks
    if type(count) is not int or not 5 <= count <= 10:
        raise ValueError("five through ten finite blocks required")
    cells = []
    for codec in CODECS:
        for training in TRAINING:
            for mode in MODES:
                cells.append(
                    dict(
                        kind="correctness",
                        codec=codec,
                        training=training,
                        mode=mode,
                        name=f"correctness-{codec}-{training}-{mode}",
                    )
                )
    for codec in CODECS:
        for training in TRAINING:
            for block in range(1, count + 1):
                for slot, mode in enumerate(ORDER, 1):
                    cells.append(
                        dict(
                            kind="timing",
                            codec=codec,
                            training=training,
                            mode=mode,
                            block=block,
                            slot=slot,
                            name=f"timing-{codec}-{training}-{block:02d}-{slot}-{mode}",
                        )
                    )
    for codec in CODECS:
        for training in TRAINING:
            for mode in ORDER[:2]:
                cells.append(
                    dict(
                        kind="profile",
                        codec=codec,
                        training=training,
                        mode=mode,
                        name=f"profile-{codec}-{training}-{mode}",
                    )
                )
    return cells


def repeat_target(blocks):
    from benchmarks.harness.arm_validity import Run, repeats

    if not 5 <= len(blocks) <= 10:
        raise ValueError("five complete pilot blocks; ten-block ceiling")
    for block in blocks:
        if len(block) != 4 or tuple(row.get("mode") for row in block) != ORDER:
            raise ValueError("ordered complete staged/fused ABBA required")
        for row in block:
            value = row.get("seconds_per_input_token")
            if (
                row.get("outcome") != "ok"
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("failed/unknown/void or invalid timing")

    def needed(prefix):
        return max(
            5,
            repeats(
                [
                    [Run(row["seconds_per_input_token"], row["outcome"]) for row in block]
                    for block in prefix
                ]
            ).blocks_needed,
        )

    first = needed(blocks[:5])
    if first > 10:
        return {"status": "no_verdict_precision_budget", "target": first}
    if len(blocks) < first:
        return {"status": "additional_blocks_required", "target": first}
    target = max(first, needed(blocks[:first]))
    return {
        "status": "no_verdict_precision_budget"
        if target > 10
        else "additional_blocks_required"
        if len(blocks) < target
        else "complete",
        "target": target,
    }


def launch_blockers(rule, root=ROOT, *, check_git=True):
    validate_rule(rule)
    blockers = []
    if rule["execution_enabled"] is not True:
        blockers.append("formal execution disabled")
    for name, pin in rule["preparation"].items():
        if not isinstance(pin, dict) or not isinstance(pin.get("path"), str):
            blockers.append(f"missing preparation: {name}")
            continue
        path = original.contained(root, pin["path"])
        if not path.is_file() or original.file_hash(path) != pin.get("sha256"):
            blockers.append(f"changed/missing preparation: {name}")
            continue
        data = json.loads(path.read_text())
        if name == "actual_shard_controls":
            if data.get("status") != "controls_passed" or [
                (row.get("case"), row.get("expected"), row.get("observed"))
                for row in data.get("controls", [])
            ] != [("baseline", "ok", "ok"), ("own", "ok", "ok"), ("foreign", "void", "void")]:
                blockers.append("actual-shard controls failed")
        if name == "launcher_self_test":
            if (
                data.get("status") != "synthetic_launcher_checks_passed"
                or data.get("variant") != "packed-fused-v1"
                or data.get("numerical_count") != 24
                or data.get("profile_count") != 8
                or data.get("formal_execution") is not False
                or data.get("pretrained_model_executed") is not False
                or data.get("fixture") != rule["required_launcher_fixture"]
            ):
                blockers.append("fused synthetic launcher gate incomplete")
            cpu = data.get("cpu_validation", {})
            if not isinstance(cpu.get("path"), str):
                blockers.append("current CPU validation receipt missing")
            else:
                cpu_path = original.contained(root, cpu["path"])
                if not cpu_path.is_file() or original.file_hash(cpu_path) != cpu.get("sha256"):
                    blockers.append("CPU validation receipt changed/missing")
                else:
                    checked = json.loads(cpu_path.read_text())
                    if (
                        checked.get("status") != "passed"
                        or checked.get("failures") != 0
                        or checked.get("errors") != 0
                        or checked.get("tests", 0) < 30
                    ):
                        blockers.append("current CPU regressions did not all pass")
                    mandatory = {
                        "benchmarks/harness/d3_fused_trainer_protocol.py",
                        "benchmarks/harness/d3_fused_trainer_analysis.py",
                        "benchmarks/harness/d3_fused_trainer_publication.py",
                        "tests/test_d3_fused_trainer_protocol.py",
                        "tests/test_d3_fused_trainer_publication.py",
                    }
                    hashes = checked.get("source_files", {})
                    if not mandatory <= hashes.keys():
                        blockers.append("incomplete CPU source verification")
                    for item, expected in hashes.items():
                        current = original.contained(root, item)
                        if (
                            not current.is_file()
                            or hashlib.sha256(
                                current.read_bytes().replace(b"\r\n", b"\n")
                            ).hexdigest()
                            != expected
                        ):
                            blockers.append("CPU-tested source changed: " + item)
            # The exporter verifies the bulky publication artifacts once, before
            # creating a canonical execution packet. Children verify its bound
            # receipts/source/data; raw audit ZIPs are not runtime dependencies.
            if check_git:
                artifacts = data.get("artifact_files", [])
                if not artifacts:
                    blockers.append("raw synthetic audit artifacts unbound")
                for artifact in artifacts:
                    current = original.contained(root, artifact["path"])
                    if not current.is_file() or original.file_hash(current) != artifact["sha256"]:
                        blockers.append("audit artifact missing/changed: " + artifact["path"])
            critical = [
                *[
                    name
                    for name in ENGINES
                    if not name.endswith(
                        ("d3_fused_trainer_protocol.py", "d3_fused_trainer_analysis.py")
                    )
                ],
                *[p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")],
            ]
            for item in critical:
                path = original.contained(root, item)
                expected = data.get("source_files", {}).get(item, {}).get("lf_sha256")
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                    != expected
                ):
                    blockers.append(f"synthetic launcher source changed: {item}")
    for name in ENGINES:
        if not original.contained(root, name).is_file():
            blockers.append(f"execution helper missing: {name}")
    if check_git:
        state = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if state.returncode or state.stdout.strip():
            blockers.append("source/rule/preparation must be committed and clean")
        merge = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "-q", "--verify", "MERGE_HEAD"],
            capture_output=True,
            check=False,
        )
        if merge.returncode == 0:
            blockers.append("unfinished merge")
    return blockers


def export_packet(output, root=ROOT):
    rule = read_rule(root)
    blockers = launch_blockers(rule, root)
    if blockers:
        raise ValueError("formal export refused: " + "; ".join(blockers))
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ValueError("canonical commit required")
    names = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "src",
            "benchmarks/harness",
            "benchmarks/protocols/d3-fused-trainer",
            "tests/test_d3_fused_trainer_protocol.py",
            "tests/test_d3_fused_trainer_publication.py",
            "tests/test_d3_fused_group.py",
            "tests/test_d3_fused_evidence.py",
            "tests/test_d3_trainer_gate.py",
            "tests/test_d3_trainer_protocol.py",
            "tests/test_d3_trainer_analysis.py",
            "tests/test_d3_trainer_integration.py",
            "tests/test_issue833_gpu_marker.py",
        ]
    )
    payloads = {}
    for name in names.decode().splitlines():
        data = subprocess.check_output(["git", "-C", str(root), "show", head + ":" + name])
        if original.contained(root, name).read_bytes().replace(b"\r\n", b"\n") != (
            data.replace(b"\r\n", b"\n")
        ):
            raise ValueError("dirty source: " + name)
        payloads[name] = data
    receipt = {
        "kind": "canonical committed packed-fused trainer source",
        "git_head": head,
        "git_blob_sha256": {n: hashlib.sha256(p).hexdigest() for n, p in payloads.items()},
    }
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr(RECEIPT, json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.export:
        Console().print_json(data=export_packet(args.export))
    else:
        Console().print_json(
            data={"cells": len(cell_plan(read_rule())), "blockers": launch_blockers(read_rule())}
        )
