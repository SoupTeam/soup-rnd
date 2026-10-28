"""Prospective public-trainer plan and fail-closed readiness checks; no GPU execution.

This module does not invoke the historical post-embedding training fixture.
The executable trainer gate and its preparation receipts must exist and be
committed before the draft can be promoted or exported for a formal run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "benchmarks/protocols/d3-public-trainer/protocol-draft.json"
CODECS = ("dense", "nf4", "nf4-dq")
TRAINING_MODES = ("ordinary", "l2l")
CORRECTNESS_MODES = (
    "native-resident",
    "sequential",
    "grouped-resident",
    "grouped",
    "triton-resident",
    "triton",
)
TIMING_MODES = ("sequential", "grouped", "triton", "triton", "grouped", "sequential")
PAIR_POSITIONS = {"grouping": (0, 1, 4, 5), "activation": (1, 2, 3, 4)}
PREPARATION_FIELDS = (
    "checkpoint",
    "shards",
    "datasets",
    "initial_adapters",
    "mapping",
    "actual_shard_controls",
    "software",
    "launcher_self_test",
)
ENGINE_FILES = (
    "benchmarks/harness/d3_trainer_prepare.py",
    "benchmarks/harness/d3_trainer_gate.py",
    "benchmarks/harness/d3_trainer_series.py",
    "benchmarks/harness/d3_trainer_analysis.py",
    "benchmarks/harness/d3_trainer_selftest.py",
)
CHECKPOINT_PINS = {
    "config.json": "3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335",
    "model-00001-of-00003.safetensors": (
        "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e"
    ),
    "model-00002-of-00003.safetensors": (
        "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7"
    ),
    "model-00003-of-00003.safetensors": (
        "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04"
    ),
}


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            result.update(block)
    return result.hexdigest()


def contained(root: Path, name: str) -> Path:
    """Require a canonical relative payload name, including on Windows."""
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or any(part in ("", ".", "..") for part in name.split("/"))
        or os.path.isabs(name)
        or ":" in name
    ):
        raise ValueError("canonical relative path required")
    base, target = os.path.realpath(root), os.path.realpath(root / name)
    if os.path.commonpath((base, target)) != base:
        raise ValueError("payload escapes its root")
    return Path(target)


def read_rule(root: Path = ROOT) -> dict:
    return json.loads(contained(root, PROTOCOL).read_text(encoding="utf-8"))


def validate_rule(rule: dict) -> None:
    """Protect numerical gates, independent contrasts and the finite attempt plan."""
    fixed = {
        "schema_version": 1,
        "task": "D3 public Soup SFT/L2L qualification",
        "model_id": "allenai/OLMoE-1B-7B-0924",
        "model_revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        "formats": list(CODECS),
        "training_modes": list(TRAINING_MODES),
        "correctness_modes": list(CORRECTNESS_MODES),
        "timing_order": list(TIMING_MODES),
        "batch": 1,
        "tokens": 512,
        "rows": 7,
        "gradient_accumulation_steps": 2,
        "optimizer_steps": 4,
        "warmup_steps": 1,
        "timed_steps": 3,
        "timed_input_tokens": 2560,
        "adapter_rank": 4,
        "adapter_alpha": 8,
        "adapter_dropout": 0.0,
        "moe_aux_loss_coeff": 0.0,
        "weight_budget_bytes": 64 * 2**20,
        "activation_budget_bytes": 256 * 2**20,
        "call_budget_bytes": 64 * 2**20,
        "native_rows_max": 128,
        "native_rows_from": 481,
        "pilot_blocks": 5,
        "max_blocks": 10,
        "replacements": 0,
        "sessions_per_commit": 1,
        "cell_timeout_seconds": 900,
        "session_timeout_seconds": 10800,
        "exactness": "dtype_shape_and_bytes_including_signed_zero",
        "allow_allclose": False,
        "allow_noise_floor_tolerance": False,
        "allow_value_dependent_fallback": False,
        "disk_counter_source": "guest-diskstats",
        "physical_host_isolation_claim": False,
        "scope_is_full_task_acceptance": False,
    }
    for key, expected in fixed.items():
        value = rule.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"prospective contract changed: {key}")
    if rule.get("entry_point") != {
        "wrapper": "soup_cli.trainer.sft.SFTTrainerWrapper",
        "setup": "setup",
        "train": "train",
        "l2l": "soup_cli.trainer.l2l_trainer",
        "adapter_targets": ["q_proj", "v_proj"],
    }:
        raise ValueError("the actual public SFT wrapper is required")
    if rule.get("attempt_budget") != {
        "correctness_cells": 36,
        "timing_pilot_cells": 180,
        "timing_max_cells": 360,
        "profile_cells": 18,
        "minimum_total_cells": 234,
        "maximum_total_cells": 414,
    }:
        raise ValueError("incorrect finite attempt budget")
    required = {
        "forward",
        "backward",
        "adapter_updates",
        "A_A",
        "negative",
        "restored",
        "save_reload",
        "same_kernel_resident_streamed",
    }
    if set(rule.get("numerical_gates", {})) != required or not all(
        rule["numerical_gates"].values()
    ):
        raise ValueError("all separately stated numerical gates must be retained")
    if set(rule.get("preparation", {})) != set(PREPARATION_FIELDS):
        raise ValueError("incomplete preparation binding schema")
    pins = rule.get("checkpoint_files", {})
    if pins != CHECKPOINT_PINS:
        raise ValueError("all original checkpoint hashes must be pinned")
    if type(rule.get("execution_enabled")) is not bool:
        raise ValueError("explicit execution switch required")


def cell_plan(rule: dict, *, blocks: int | None = None) -> list[dict]:
    validate_rule(rule)
    count = rule["pilot_blocks"] if blocks is None else blocks
    if type(count) is not int or not rule["pilot_blocks"] <= count <= rule["max_blocks"]:
        raise ValueError("block count outside the prospective finite budget")
    cells = []
    for codec in CODECS:
        for training in TRAINING_MODES:
            for mode in CORRECTNESS_MODES:
                cells.append(
                    {
                        "kind": "correctness",
                        "codec": codec,
                        "training": training,
                        "mode": mode,
                        "name": f"correctness-{codec}-{training}-{mode}",
                    }
                )
    for codec in CODECS:
        for training in TRAINING_MODES:
            for block in range(1, count + 1):
                for slot, mode in enumerate(TIMING_MODES, 1):
                    cells.append(
                        {
                            "kind": "timing",
                            "codec": codec,
                            "training": training,
                            "mode": mode,
                            "block": block,
                            "slot": slot,
                            "name": f"timing-{codec}-{training}-{block:02d}-{slot}-{mode}",
                        }
                    )
    for codec in CODECS:
        for training in TRAINING_MODES:
            for mode in TIMING_MODES[:3]:
                cells.append(
                    {
                        "kind": "profile",
                        "codec": codec,
                        "training": training,
                        "mode": mode,
                        "name": f"profile-{codec}-{training}-{mode}",
                    }
                )
    return cells


def pair_projection(block: list[dict], contrast: str) -> list[dict]:
    """Extract two actual ABBA comparisons from the shared symmetric six-arm block."""
    if contrast not in PAIR_POSITIONS:
        raise ValueError("unknown predeclared contrast")
    if len(block) != 6 or tuple(row.get("mode") for row in block) != TIMING_MODES:
        raise ValueError("complete ordered A B C C B A block required")
    for row in block:
        value = row.get("seconds_per_input_token")
        if (
            row.get("outcome") != "ok"
            or isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("stop on failed/unknown/void, missing or nonpositive timing")
    return [block[index] for index in PAIR_POSITIONS[contrast]]


def repeat_target(blocks: list[list[dict]]) -> dict:
    """Unchanged F1 sigma/10% formula, with one declared joint two-contrast top-up."""
    from benchmarks.harness.arm_validity import Run, repeats

    if len(blocks) < 5:
        raise ValueError("five complete prospective pilot blocks required")
    if len(blocks) > 10:
        raise ValueError("attempt budget exceeded")

    def needed(prefix: list[list[dict]]) -> int:
        values = []
        for contrast in PAIR_POSITIONS:
            pairs = [
                [
                    Run(row["seconds_per_input_token"], row["outcome"])
                    for row in pair_projection(block, contrast)
                ]
                for block in prefix
            ]
            values.append(repeats(pairs).blocks_needed)
        return max(5, *values)

    # Validate every retained block, including any final top-up blocks.
    for block in blocks:
        pair_projection(block, "grouping")
    first = needed(blocks[:5])
    if first > 10:
        return {"status": "no_verdict_precision_budget", "target": first}
    if len(blocks) < first:
        return {"status": "additional_blocks_required", "target": first}
    final = max(first, needed(blocks[:first]))
    return {
        "status": "no_verdict_precision_budget"
        if final > 10
        else "additional_blocks_required"
        if len(blocks) < final
        else "complete",
        "target": final,
    }


def launch_blockers(rule: dict, root: Path = ROOT, *, check_git: bool = True) -> list[str]:
    validate_rule(rule)
    blockers = []
    if rule["execution_enabled"] is not True:
        blockers.append("execution is disabled: this is a prospective draft")
    for name in PREPARATION_FIELDS:
        pin = rule["preparation"][name]
        if (
            not isinstance(pin, dict)
            or set(pin) != {"path", "sha256"}
            or not isinstance(pin["sha256"], str)
            or not re.fullmatch("[a-f0-9]{64}", pin["sha256"])
        ):
            blockers.append(f"unbound preparation evidence: {name}")
            continue
        path = contained(root, pin["path"])
        if not pin["path"].startswith("benchmarks/protocols/d3-public-trainer/"):
            blockers.append(f"preparation receipt must be included in the source packet: {name}")
            continue
        if not path.is_file() or file_hash(path) != pin["sha256"]:
            blockers.append(f"preparation evidence missing/changed: {name}")
            continue
        if name == "actual_shard_controls":
            controls = json.loads(path.read_text(encoding="utf-8"))
            observed = [
                (row.get("case"), row.get("expected"), row.get("observed"))
                for row in controls.get("controls", [])
            ]
            if controls.get("status") != "controls_passed" or observed != [
                ("baseline", "ok", "ok"),
                ("own", "ok", "ok"),
                ("foreign", "void", "void"),
            ]:
                blockers.append("actual-shard F1 controls have not all passed")
        if name == "launcher_self_test":
            evidence = json.loads(path.read_text(encoding="utf-8"))
            if (
                evidence.get("status") != "synthetic_launcher_checks_passed"
                or evidence.get("numerical_count") != 36
                or evidence.get("profile_count") != 18
                or evidence.get("formal_execution") is not False
                or evidence.get("pretrained_model_executed") is not False
            ):
                blockers.append("new launcher synthetic controls/profiles have not all passed")
            # Numerical/runtime source must match the tested snapshot. The pure
            # planner/readiness validator and offline analyzer have separate CPU
            # tests; they are not CUDA numerical kernels.
            critical = [
                item for item in ENGINE_FILES
                if not item.endswith("d3_trainer_analysis.py")
            ]
            critical += [item.relative_to(root).as_posix()
                         for item in (root / "src").rglob("*.py")]
            sources = evidence.get("source_files", {})
            for item in critical:
                current = contained(root, item)
                expected = sources.get(item, {}).get("lf_sha256")
                if not current.is_file() or hashlib.sha256(
                    current.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest() != expected:
                    blockers.append(f"launcher self-test runtime source differs: {item}")
    for name in ENGINE_FILES:
        if not contained(root, name).is_file():
            blockers.append(f"new public-trainer execution helper missing: {name}")
    if check_git:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode or result.stdout.strip():
            blockers.append("source/protocol/evidence are not in a clean committed checkout")
        merge = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "-q", "--verify", "MERGE_HEAD"],
            check=False,
            capture_output=True,
        )
        if merge.returncode == 0:
            blockers.append("an unfinished merge must be committed by the operator")
    return blockers


def assert_launch_ready(rule: dict, root: Path = ROOT) -> None:
    blockers = launch_blockers(rule, root)
    if blockers:
        raise ValueError("formal execution refused:\n- " + "\n- ".join(blockers))


def export_packet(output: Path, root: Path = ROOT) -> dict:
    """Export only canonical committed blobs, never uncommitted preflight snapshots."""
    rule = read_rule(root)
    assert_launch_ready(rule, root)
    paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "src",
            "benchmarks/harness",
            "benchmarks/protocols/d3-public-trainer",
            "tests/test_d3_trainer_protocol.py",
            "tests/test_d3_trainer_gate.py",
            "tests/test_d3_trainer_analysis.py",
            "tests/test_d3_trainer_integration.py",
            "tests/test_l2l_step.py",
        ]
    ).decode()
    payloads = {}
    for name in paths.splitlines():
        payload = subprocess.check_output(["git", "-C", str(root), "show", "HEAD:" + name])
        working = contained(root, name).read_bytes()
        # Text checkout normalization is allowed; execution always uses Git blobs.
        if working.replace(b"\r\n", b"\n") != payload.replace(b"\r\n", b"\n"):
            raise ValueError(f"committed/working payload differs: {name}")
        payloads[name] = payload
    receipt = {
        "kind": "committed public-trainer source, not the historical fixture",
        "git_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"])
        .decode()
        .strip(),
        "git_blob_sha256": {
            name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()
        },
    }
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr("FROZEN-D3-TRAINER-SOURCE.json", json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("plan", "check", "export"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    rule = read_rule()
    validate_rule(rule)
    if args.command == "export":
        if args.out is None:
            parser.error("export requires --out")
        output: Any = export_packet(args.out)
    else:
        blockers = launch_blockers(rule)
        plan = cell_plan(rule)
        output = {
            "formal_execution_started": False,
            "launch_blockers": blockers,
            "pilot_plan_cells": len(plan),
            "max_plan_cells": len(cell_plan(rule, blocks=10)),
        }
        if args.command == "plan":
            output["cells"] = plan
    encoded = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.out is not None and args.command != "export":
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    else:
        Console().print_json(encoded)
    return 1 if args.command == "check" and output["launch_blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
