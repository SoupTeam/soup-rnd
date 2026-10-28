"""Offline public-trainer evidence replay; no CUDA experiment and no task sign-off."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmarks.harness.d3_trainer_gate import numerical_view, require_exact, save, tensor_record
from benchmarks.harness.d3_trainer_protocol import (
    CODECS,
    TIMING_MODES,
    TRAINING_MODES,
    cell_plan,
    contained,
    file_hash,
    pair_projection,
    repeat_target,
    validate_rule,
)
from benchmarks.harness.d3_trainer_series import oracle_path


def replay_native_tensors(folder: Path, evidence: dict) -> int:
    """Verify saved native data, not only the candidate's asserted PASS booleans."""
    from safetensors import safe_open

    total = 0
    for index, frame in enumerate(evidence["steps"], 1):
        with safe_open(
            str(folder / "positive/oracle" / f"step-{index}.safetensors"),
            framework="pt",
            device="cpu",
        ) as handle:
            expected = {}
            for name, record in frame["parameter_gradients"].items():
                expected["gradient/" + name] = record
            for name, record in frame["raw_parameter_gradients"].items():
                expected["raw_gradient/" + name] = record
            for name, record in frame["parameters"].items():
                expected["parameter/" + name] = record
            for identity, values in frame["routes"].items():
                for name, record in values.items():
                    expected[f"expert_forward/{identity}/{name}"] = record
            for identity, values in frame["expert_gradients"].items():
                for name, record in values.items():
                    expected[f"expert_gradient/{identity}/{name}"] = record
            for micro, record in enumerate(frame["forward"], 1):
                base = f"logits-{index - 1}-{micro}"
                expected[base] = record["logits"]
                expected[base + "-loss"] = record["loss"]
            if set(handle.keys()) != set(expected):
                raise ValueError("native tensor oracle keys differ from the actual step")
            for name, record in expected.items():
                value = handle.get_tensor(name)
                if name.endswith("-loss"):
                    value = value.reshape(())
                require_exact(tensor_record(value), record, "saved native tensor " + name)
                total += 1
    return total


def analyze(folder: Path) -> dict:
    rule = json.loads((folder / "protocol.json").read_text())
    validate_rule(rule)
    session = json.loads((folder / "session.json").read_text())
    complete = session["status"] == "complete_pending_offline_analysis_and_review"
    summary = {
        "status": "incomplete_no_task_verdict",
        "d3_accepted": False,
        "source": session["source"],
        "protocol_sha256": file_hash(folder / "protocol.json"),
        "session_id": session["session_id"],
        "retained_cells": len(session["cells"]),
        "complete_public_trainer_series": False,
    }
    if summary["protocol_sha256"] != session["protocol_sha256"]:
        raise ValueError("session/protocol identities differ")
    if not complete:
        summary["reason"] = session.get("traceback", "series did not complete")
        return summary
    if len(session["cells"]) > rule["attempt_budget"]["maximum_total_cells"]:
        raise ValueError("attempt budget exceeded")
    permitted = {cell["name"]: cell for cell in cell_plan(rule, blocks=rule["max_blocks"])}
    observed = {}
    for row in session["cells"]:
        name = row["name"]
        if name not in permitted or name in observed:
            raise ValueError("duplicate or undeclared cell")
        path = contained(folder, name + "/result.json")
        if file_hash(path) != row["result_sha256"]:
            raise ValueError("published cell result changed")
        result = json.loads(path.read_text())
        if (
            result["status"] != "ok"
            or result["source"] != session["source"]
            or result["protocol_sha256"] != session["protocol_sha256"]
            or result["session_id"] != session["session_id"]
            or result["formal_execution"] is not True
            or row["returncode"] != 0
            or result["F1"]["outcome"] != "ok"
            or result["cell"] != permitted[name]
        ):
            raise ValueError("cell lacks required identity, process exit or validity")
        observed[name] = result
    required = {cell["name"] for cell in cell_plan(rule) if cell["kind"] != "timing"}
    if not required <= set(observed):
        raise ValueError("missing correctness/profile cells")
    native_tensors = 0
    for codec in CODECS:
        native = json.loads(oracle_path(folder, codec).read_text())["result"]["numerical"]
        if len(native["steps"]) != rule["optimizer_steps"] or (
            len(native["windows"]) != rule["optimizer_steps"]
        ):
            raise ValueError("native oracle lacks all optimizer windows")
        native_tensors += replay_native_tensors(oracle_path(folder, codec).parent, native)
        for row in observed.values():
            cell = row["cell"]
            if cell["codec"] != codec or cell["kind"] != "correctness":
                continue
            result = row["result"]
            if not all(
                result.get(key) is True
                for key in (
                    "A_A",
                    "negative_effective",
                    "restored",
                    "save_reload",
                )
            ):
                raise ValueError("missing effective numerical controls")
            base = contained(folder, cell["name"])
            positive = json.loads((base / "positive/record.json").read_text())
            require_exact(numerical_view(result["numerical"]), numerical_view(positive),
                          "result/retained positive identity")
            require_exact(numerical_view(positive), numerical_view(native), "native GA replay")
            for label in ("A-A", "restored"):
                actual = json.loads((base / label / "record.json").read_text())
                require_exact(numerical_view(actual), numerical_view(positive), label)
            negative = json.loads((base / "negative/record.json").read_text())
            if negative["evaluation"] == positive["evaluation"] or all(
                a["parameter_gradients"] == b["parameter_gradients"]
                for a, b in zip(negative["steps"], positive["steps"], strict=True)
            ):
                raise ValueError("ineffective retained numerical negative")
            from safetensors.torch import load_file

            artifact = base / "positive/adapter/adapter_model.safetensors"
            require_exact(
                {key: tensor_record(value) for key, value in load_file(str(artifact)).items()},
                positive["final"],
                "actual published adapter values/dtypes",
            )
            require_exact(result["reload"]["factors"], positive["final"], "actual reload factors")
            require_exact(
                result["reload"]["evaluation"], positive["evaluation"], "actual reload logits"
            )
    contrasts = {}
    ordered = [cell["name"] for cell in cell_plan(rule) if cell["kind"] == "correctness"]
    for codec in CODECS:
        native = json.loads(oracle_path(folder, codec).read_text())["result"]["numerical"]
        for training in TRAINING_MODES:
            rows = [
                row
                for row in observed.values()
                if row["cell"]["kind"] == "timing"
                and row["cell"]["codec"] == codec
                and row["cell"]["training"] == training
            ]
            if len(rows) % 6:
                raise ValueError("partial symmetric block")
            for row in rows:
                result = row["result"]
                require_exact(result["windows"], native["windows"], "timed input windows")
                require_exact(result["frames"], [
                    {key: frame[key] for key in ("parameter_gradients", "parameters")}
                    for frame in native["steps"]
                ], "timed gradients and updates")
                samples = result["samples"]
                if len(samples) != rule["timed_steps"]:
                    raise ValueError("wrong timed sample count")
                for step, sample in enumerate(samples, rule["warmup_steps"]):
                    if sample["step"] != step or sample["input_tokens"] != (
                        rule["optimizer"]["gradient_windows"][step] * rule["tokens"]
                    ) or any(
                        not math.isfinite(sample[key]) or sample[key] <= 0
                        for key in ("wall_seconds", "cuda_ms")
                    ):
                        raise ValueError("invalid timed step/token record")
                require_exact(result["seconds_per_input_token"],
                              sum(item["wall_seconds"] for item in samples) /
                              rule["timed_input_tokens"], "timing aggregate")
            blocks = []
            for index in range(1, len(rows) // 6 + 1):
                group = sorted(
                    (row for row in rows if row["cell"]["block"] == index),
                    key=lambda row: row["cell"]["slot"],
                )
                block = [
                    {
                        "mode": row["cell"]["mode"],
                        "outcome": row["F1"]["outcome"],
                        "seconds_per_input_token": row["result"]["seconds_per_input_token"],
                    }
                    for row in group
                ]
                if tuple(row["mode"] for row in block) != TIMING_MODES:
                    raise ValueError("symmetric block is missing or reordered")
                blocks.append(block)
            estimate = repeat_target(blocks)
            if estimate["status"] != "complete":
                raise ValueError("the declared repetition rule is not satisfied")
            if len(blocks) != estimate["target"]:
                raise ValueError("extra blocks exceed the declared once-only repetition target")
            ordered.extend(row["cell"]["name"] for row in rows)
            ordered.extend(cell["name"] for cell in cell_plan(rule)
                           if cell["kind"] == "profile" and cell["codec"] == codec
                           and cell["training"] == training)
            values = {}
            for contrast in ("grouping", "activation"):
                changes = []
                for block in blocks:
                    a, b, c, d = pair_projection(block, contrast)
                    left = (a["seconds_per_input_token"] + d["seconds_per_input_token"]) / 2
                    right = (b["seconds_per_input_token"] + c["seconds_per_input_token"]) / 2
                    changes.append(100 * (right / left - 1))
                values[contrast] = {
                    "paired_percent_changes": changes,
                    "median_percent_change": statistics.median(changes),
                }
            contrasts[f"{codec}/{training}"] = {
                "blocks": len(blocks),
                "repeat_calculation": estimate,
                "contrasts": values,
            }
    for row in observed.values():
        if row["cell"]["kind"] != "profile":
            continue
        trace_path = contained(folder, row["cell"]["name"] + "/cuda-profile.json")
        if file_hash(trace_path) != row["result"]["profile_sha256"]:
            raise ValueError("device trace hash changed")
        trace = json.loads(trace_path.read_text())
        names = [item["name"] for item in trace["traceEvents"]
                 if item.get("cat") == "kernel" and item.get("ph") == "X"]
        if not names or names != row["result"]["kernel_names"] or (
            len(names) != row["result"]["cuda_kernel_events"]
        ):
            raise ValueError("missing or mismatched actual device profiler events")
        if row["cell"]["mode"] == "triton" and not any("activate" in name for name in names):
            raise ValueError("device trace lacks executed Triton activation")
    require_exact([row["name"] for row in session["cells"]], ordered, "fresh-process cell order")
    summary.update(
        status="qualification_complete_pending_independent_review",
        d3_accepted=False,
        complete_public_trainer_series=True,
        correctness_cells=36,
        profile_cells=18,
        native_tensor_payloads_verified=native_tensors,
        timing=contrasts,
        scope=("frozen expert / attention-LoRA numerical and descriptive trainer comparison; "
               "no quality or full-task sign-off"),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.session)
    save(args.out, result)
    return 0 if result["complete_public_trainer_series"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
