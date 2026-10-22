"""Run the frozen full-model training correctness and timing series."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

from d3_full_model import file_hash
from d3_training_protocol import CODECS, MODES, frozen_source


def _save(path: Path, value: dict) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def _cell(args, source: dict, rule: dict, session: dict, *, codec: str,
          mode: str, kind: str, name: str, oracle: Path | None = None) -> dict:
    folder = args.out / name
    command = [sys.executable, str(Path(__file__).with_name("d3_training_gate.py")),
               "child", "--prepared", str(args.prepared), "--out", str(folder),
               "--library", str(args.library), "--session", str(args.out),
               "--mode", mode, "--codec", codec, "--kind", kind]
    if oracle is not None:
        command += ["--oracle", str(oracle)]
    with folder.with_suffix(".log").open("x") as log:
        proc = subprocess.run(command, check=False, cwd=Path(__file__).resolve().parents[2],
                              stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    if not folder.is_dir() or not (folder / "result.json").is_file():
        raise RuntimeError(f"cell produced no retained result: {folder}")
    row = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    session["cells"].append({"name": name, "returncode": proc.returncode,
                             "status": row.get("status"), "mode": mode,
                             "codec": codec, "kind": kind,
                             "result_sha256": file_hash(folder / "result.json")})
    _save(args.out / "session.json", session)
    if (proc.returncode or row.get("status") != "ok" or row.get("source") != source
            or row.get("session_id") != session["session_id"]
            or row.get("protocol_sha256") != session["protocol_sha256"]
            or row.get("formal_execution") is not True
            or row.get("F1", {}).get("outcome") != "ok"):
        raise RuntimeError(f"training cell failed or lacks required validity: {folder}")
    if kind == "correctness":
        if (row.get("A_A_exact") is not True or row.get("restored_exact") is not True
                or not row.get("steps") or row.get("control_target") is None):
            raise RuntimeError(f"training correctness controls are incomplete: {folder}")
        # Candidate cells must match the native step hashes; the native oracle
        # cell alone writes the safetensors files.
        if (mode != "native" and not row.get("oracle_files")
                and any(record.get("tensor_sha256") is None for record in row["steps"])):
            raise RuntimeError(f"candidate step hashes are missing: {folder}")
    elif kind == "profile":
        if not row.get("steps") or not row["steps"][0].get("cuda_kernel_events"):
            raise RuntimeError(f"CUDA profiler retained no kernels: {folder}")
    return row


def parent(args, rule: dict) -> int:
    source = frozen_source()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol_sha = file_hash(Path(__file__).resolve().parents[2] /
                             "benchmarks/protocols/d3-trained-step/protocol.json")
    session = {"status": "running", "kind": "trained-model optimizer-step series",
               "session_id": uuid.uuid4().hex, "source": source,
               "protocol_sha256": protocol_sha, "cells": [], "d3_accepted": False,
               "attempt_budget": rule["attempt_budget"]}
    _save(args.out / "session.json", session)
    try:
        for codec in CODECS:
            oracle = None
            for mode in MODES:
                row = _cell(args, source, rule, session, codec=codec, mode=mode,
                            kind="correctness", name=f"correctness-{codec}-{mode}",
                            oracle=oracle)
                if mode == "native":
                    oracle = args.out / f"correctness-{codec}-native"
                    if set(row.get("oracle_files", {})) != {
                        f"step-{index}.safetensors" for index in range(rule["steps"])
                    }:
                        raise RuntimeError(f"native step oracle is incomplete: {oracle}")
            if oracle is None:
                raise RuntimeError(f"native oracle missing for {codec}")

        left, right = rule["timing_pair"]
        for codec in CODECS:
            oracle = args.out / f"correctness-{codec}-native"
            for block in range(rule["measurement_blocks_per_codec"]):
                for position, mode in enumerate((left, right, right, left)):
                    _cell(args, source, rule, session, codec=codec, mode=mode,
                          kind="timing", name=f"timing-{codec}-block-{block + 1:02d}-"
                          f"{position + 1}-{mode}", oracle=oracle)
        for codec in CODECS:
            oracle = args.out / f"correctness-{codec}-native"
            for mode in rule["timing_pair"]:
                _cell(args, source, rule, session, codec=codec, mode=mode,
                      kind="profile", name=f"profile-{codec}-{mode}", oracle=oracle)
        if len(session["cells"]) != rule["expected_cell_count"]:
            raise RuntimeError("series ended without consuming its exact frozen cell budget")
        session["status"] = "complete_pending_offline_analysis_and_review"
    except BaseException:  # noqa: BLE001 - persist incomplete status on every abort
        session["status"] = "incomplete"
        session["traceback"] = traceback.format_exc()
    finally:
        _save(args.out / "session.json", session)
    return 0 if session["status"] == "complete_pending_offline_analysis_and_review" else 1
