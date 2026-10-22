"""Audit the frozen D3 optimizer-step analysis contract without CUDA."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pytest
from d3_training_analysis import analyze
from d3_training_protocol import CODECS, MODES, ROOT, read_rule


def _write(path: Path, value: dict) -> str:
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _session(folder: Path) -> None:
    rule = read_rule()
    protocol = ROOT / "benchmarks/protocols/d3-trained-step/protocol.json"
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    source = {"git_head": "a" * 40, "git_blob_sha256": {"src/example.py": "b" * 64}}
    rows = []

    names = [(f"correctness-{codec}-{mode}", "correctness", codec, mode)
             for codec, mode in itertools.product(CODECS, MODES)]
    names += [
        (f"timing-{codec}-block-{block:02d}-{position}-{mode}", "timing", codec, mode)
        for codec in CODECS
        for block in range(1, rule["measurement_blocks_per_codec"] + 1)
        for position, mode in enumerate((*rule["timing_pair"], *rule["timing_pair"][::-1]), 1)
    ]
    names += [(f"profile-{codec}-{mode}", "profile", codec, mode)
              for codec in CODECS for mode in rule["timing_pair"]]

    for name, kind, codec, mode in names:
        result = {"status": "ok", "F1": {"outcome": "ok"}, "source": source,
                  "session_id": "s" * 32, "protocol_sha256": protocol_hash,
                  "kind": kind, "codec": codec, "mode": mode}
        tensors = {"logits": "a" * 64, "loss": "b" * 64,
                   "adapter/a": "c" * 64, "gradient/a": "d" * 64}
        if kind == "correctness":
            result.update(A_A_exact=True, restored_exact=True, control_target=0,
                          negative_tensor_sha256={"logits": "e" * 64},
                          steps=[{"tensor_sha256": {**tensors, "derivative/0/input": "f" * 64}}
                                 for _ in range(rule["steps"])], oracle_files={})
            if mode == "native":
                result["oracle_files"] = {}
                for index in range(rule["steps"]):
                    filename = f"step-{index}.safetensors"
                    oracle_path = folder / name / filename
                    oracle_path.parent.mkdir(parents=True, exist_ok=True)
                    oracle_path.write_bytes(f"oracle-{codec}-{index}".encode())
                    result["oracle_files"][filename] = hashlib.sha256(
                        oracle_path.read_bytes()
                    ).hexdigest()
                    result["steps"][index]["tensor_sha256"] = {
                        **tensors, "derivative/0/input": "f" * 64,
                    }
            else:
                result["steps"] = [{"tensor_sha256": {
                    **tensors, "derivative/0/input": "f" * 64,
                }} for _ in range(rule["steps"])]
        elif kind == "timing":
            steps = []
            for offset in range(rule["timed_steps"]):
                index = rule["warmup_steps"] + offset
                steps.append({"seconds": 1.0, "peak_allocated": 10,
                              "peak_reserved": 12,
                              "tensor_sha256": tensors})
            result.update(steps=steps, peak_process_rss_bytes=20)
        elif kind == "profile":
            result["steps"] = [{"cuda_kernel_events": 1, "tensor_sha256": tensors}]
        result_path = folder / name / "result.json"
        digest = _write(result_path, result)
        rows.append({"name": name, "returncode": 0, "result_sha256": digest,
                     "status": "ok", "kind": kind, "codec": codec, "mode": mode})

    _write(folder / "session.json", {
        "status": "complete_pending_offline_analysis_and_review", "source": source,
        "protocol_sha256": protocol_hash, "session_id": "s" * 32, "cells": rows,
    })


def test_complete_design_replays_into_scoped_analysis(tmp_path: Path) -> None:
    _session(tmp_path)

    result = analyze(tmp_path)

    assert result["correctness_cells"] == 18
    assert result["timing_cells"] == 60
    assert result["profile_cells"] == 6
    assert result["timing"]["nf4-dq"]["control_mode"] == "grouped"
    assert result["timing"]["nf4-dq"]["candidate_mode"] == "triton"
    assert result["d3_accepted"] is False


def test_incomplete_or_corrupted_cell_cannot_be_analyzed(tmp_path: Path) -> None:
    _session(tmp_path)
    session_path = tmp_path / "session.json"
    session = json.loads(session_path.read_text())
    session["cells"].pop()
    _write(session_path, session)

    with pytest.raises(ValueError, match="cell count"):
        analyze(tmp_path)
