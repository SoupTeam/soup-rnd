"""Synthetic evidence fixtures exercise complete replay and tampering, without CUDA."""

import copy
import json
import subprocess

import pytest

from benchmarks.harness.d3_trainer_analysis import analyze, committed_protocol
from benchmarks.harness.d3_trainer_gate import save, tensor_record
from benchmarks.harness.d3_trainer_protocol import ROOT, cell_plan, file_hash


@pytest.fixture
def complete_series(tmp_path):
    import torch
    from safetensors.torch import save_file

    source = {
        "git_head": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    }
    rule, protocol_sha256 = committed_protocol(source)
    save(tmp_path / "protocol.json", rule)
    session = dict(
        status="complete_pending_offline_analysis_and_review",
        source=source,
        protocol_sha256=protocol_sha256,
        session_id="synthetic-unit-fixture",
        cells=[],
    )
    value = torch.tensor([1.0])
    stamp = tensor_record(value)
    loss = tensor_record(value.reshape(()))
    frames = []
    windows = []
    for size in rule["optimizer"]["gradient_windows"]:
        frames.append(
            dict(
                forward=[dict(logits=stamp, loss=loss) for _ in range(size)],
                routes={},
                expert_gradients={},
                raw_parameter_gradients={"test": stamp},
                parameter_gradients={"test": stamp},
                parameters={"test": stamp},
            )
        )
        windows.append(dict(samples=[{"input_ids": stamp} for _ in range(size)], num_items=size))
    native = dict(
        steps=frames,
        windows=windows,
        final={"test": stamp},
        evaluation=stamp,
        attention="synthetic",
    )
    permitted = cell_plan(rule)
    order = [cell for cell in permitted if cell["kind"] == "correctness"]
    for codec in rule["formats"]:
        for training in rule["training_modes"]:
            order.extend(
                cell
                for cell in permitted
                if cell["kind"] == "timing"
                and cell["codec"] == codec
                and cell["training"] == training
            )
            order.extend(
                cell
                for cell in permitted
                if cell["kind"] == "profile"
                and cell["codec"] == codec
                and cell["training"] == training
            )
    for cell in order:
        folder = tmp_path / cell["name"]
        folder.mkdir()
        if cell["kind"] == "correctness":
            result = dict(
                numerical=native,
                A_A=True,
                negative_effective=True,
                restored=True,
                save_reload=True,
                reload={"factors": native["final"], "evaluation": stamp},
            )
            for label in ("positive", "A-A", "restored", "negative"):
                target = folder / label
                target.mkdir()
                record = copy.deepcopy(native)
                if label == "negative":
                    record["evaluation"] = {"sha256": "synthetic-effective-negative"}
                    for frame in record["steps"]:
                        frame["parameter_gradients"] = {"test": {"sha256": "different"}}
                save(target / "record.json", record)
            adapter = folder / "positive/adapter"
            adapter.mkdir()
            save_file({"test": value}, str(adapter / "adapter_model.safetensors"))
            if cell["mode"] == "native-resident" and cell["training"] == "ordinary":
                oracle = folder / "positive/oracle"
                oracle.mkdir()
                for index, frame in enumerate(frames, 1):
                    tensors = {
                        key + "/test": value.clone()
                        for key in ("gradient", "raw_gradient", "parameter")
                    }
                    for micro in range(1, len(frame["forward"]) + 1):
                        tensors[f"logits-{index - 1}-{micro}"] = value.clone()
                        tensors[f"logits-{index - 1}-{micro}-loss"] = value.clone()
                    save_file(tensors, str(oracle / f"step-{index}.safetensors"))
        else:
            samples = [
                dict(
                    step=i,
                    wall_seconds=float(n),
                    cuda_ms=float(n * 1000),
                    input_tokens=n * rule["tokens"],
                )
                for i, n in enumerate(rule["optimizer"]["gradient_windows"])
                if i
            ]
            result = dict(
                samples=samples,
                windows=windows,
                frames=[
                    {key: frame[key] for key in ("parameter_gradients", "parameters")}
                    for frame in frames
                ],
                seconds_per_input_token=5 / 2560,
            )
            if cell["kind"] == "profile":
                names = ["unit_activate_kernel"]
                save(
                    folder / "cuda-profile.json",
                    {"traceEvents": [{"name": names[0], "cat": "kernel", "ph": "X"}]},
                )
                result.update(
                    cuda_kernel_events=1,
                    kernel_names=names,
                    profile_sha256=file_hash(folder / "cuda-profile.json"),
                )
        payload = dict(
            status="ok",
            source=source,
            protocol_sha256=session["protocol_sha256"],
            session_id=session["session_id"],
            formal_execution=True,
            cell=cell,
            F1={"outcome": "ok"},
            result=result,
        )
        save(folder / "result.json", payload)
        session["cells"].append(
            dict(**cell, returncode=0, status="ok", result_sha256=file_hash(folder / "result.json"))
        )
    save(tmp_path / "session.json", session)
    return tmp_path


def test_replay_accepts_serialized_protocol_copy_with_committed_identity(complete_series):
    session = json.loads((complete_series / "session.json").read_text())
    assert session["protocol_sha256"] != file_hash(complete_series / "protocol.json")
    assert analyze(complete_series)["complete_public_trainer_series"]


def test_complete_offline_replay_never_claims_full_task_acceptance(complete_series):
    result = analyze(complete_series)
    assert result["complete_public_trainer_series"]
    assert not result["d3_accepted"]
    assert result["correctness_cells"] == 36
    assert result["profile_cells"] == 18
    assert result["native_tensor_payloads_verified"] > 50
    assert len(result["timing"]) == 6
    assert set(result["timing"]["dense/ordinary"]["modes"]) == {
        "sequential",
        "grouped",
        "triton",
    }
    assert result["timing"]["dense/ordinary"]["modes"]["grouped"][
        "median_input_tokens_per_second"
    ] > 0


@pytest.mark.parametrize(
    "tamper",
    ["order", "foreign_session", "void", "partial_tokens", "trace", "gradient", "missing_payload"],
)
def test_replay_rejects_plausible_but_invalid_published_sessions(complete_series, tamper):
    root = complete_series
    session = json.loads((root / "session.json").read_text())
    if tamper == "order":
        session["cells"][0], session["cells"][1] = session["cells"][1], session["cells"][0]
    else:
        cell = next(
            row
            for row in session["cells"]
            if row["kind"] == ("profile" if tamper == "trace" else "timing")
        )
        path = root / cell["name"] / "result.json"
        result = json.loads(path.read_text())
        if tamper == "foreign_session":
            result["session_id"] = "other"
        elif tamper == "void":
            result["F1"]["outcome"] = "void"
        elif tamper == "partial_tokens":
            result["result"]["samples"][-1]["input_tokens"] = 1024
        elif tamper == "gradient":
            result["result"]["frames"][0]["parameter_gradients"] = {}
        elif tamper == "missing_payload":
            path.unlink()
        else:
            trace = path.parent / "cuda-profile.json"
            save(trace, {"traceEvents": []})
            result["result"]["profile_sha256"] = file_hash(trace)
        if tamper != "missing_payload":
            save(path, result)
            cell["result_sha256"] = file_hash(path)
    save(root / "session.json", session)
    with pytest.raises((ValueError, FileNotFoundError)):
        analyze(root)
