"""Verify the exported D1 packet on CPU; no GPU allocation or model downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = Path(__file__).parent
    index = json.loads((packet / "index.json").read_text(encoding="utf-8"))
    source = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if source != index["source_sha"]:
        raise ValueError("Use a separate checkout of the measured source SHA from index.json")
    sys.path.insert(0, str(args.repo / "benchmarks/harness"))
    from d1_rules import evidence_path

    for row in json.loads((packet / "packet-manifest.json").read_text(encoding="utf-8")):
        path = evidence_path(packet, row["file"])
        if (
            path.stat().st_size != row["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise ValueError("Curated packet SHA-256 mismatch: " + row["file"])
    args.output.mkdir(parents=True, exist_ok=False)
    archive_input = args.output / "archive-input"
    archive_input.mkdir()
    proof_path = packet / "full-proof-manifest.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    shutil.copyfile(proof_path, archive_input / proof_path.name)
    for part in proof["parts"]:
        name = part["file"]
        if Path(name).name != name:
            raise ValueError("Expected adjacent archive filename")
        archive = packet.parent / name
        target = archive_input / name
        try:
            os.link(archive, target)
        except OSError:
            shutil.copyfile(archive, target)
    sys.path.insert(0, str(args.repo / "benchmarks/harness"))
    from argparse import Namespace

    from d1_full_report import build_index, qa_contract
    from d1_full_study import records
    from d1_materialize import materialize
    from d1_nf4_study import replay
    from d1_projection_replay import audit as projection_audit
    from d1_qa import load_split
    from d1_qa_replay import audit as qa_audit

    raw = args.output / "raw"

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    save("materialization.json", materialize(archive_input / proof_path.name, raw, True))
    study = raw / "results/full-k"
    rebuilt = json.loads(json.dumps(build_index(study)))
    if rebuilt != index:
        raise ValueError("Recomputed index differs from the supplied index")
    save("index.json", rebuilt)
    if (packet / "mechanism.json").exists():
        from mechanism_audit import audit as mechanism_audit

        mechanism = mechanism_audit(study, rebuilt, args.repo)
        if mechanism != json.loads((packet / "mechanism.json").read_text(encoding="utf-8")):
            raise ValueError("Recomputed mechanism audit differs from the supplied record")
        save("mechanism.json", mechanism)
    if (packet / "projection-coverage.png").exists():
        from coverage_figure import draw

        draw(rebuilt, args.output / "projection-coverage.png")
    for dtype in ("float16", "bfloat16"):
        result = replay(Namespace(input=study / ("correctness-" + dtype)))
        save("correctness-" + dtype + ".json", result)
        if result["status"] != "PASS":
            raise ValueError("Correctness replay failed")
    result = projection_audit(study)
    save("projection-replay.json", result)
    if result["status"] != "PASS":
        raise ValueError("Projection replay failed")
    qa = next(case for case in rebuilt["cases"] if case["plan"]["kind"] == "train")
    comparison = qa["comparisons"][0] if qa.get("comparisons") else {}
    qa_ids = comparison.get("accepted_block_ids", [])
    if (packet / "qa-replay.json").exists():
        qa_ids = json.loads((packet / "qa-replay.json").read_text(encoding="utf-8"))["block_ids"]
    if qa_ids:
        for row in records(study / "qa"):
            if row["block"] in qa_ids:
                qa_contract(qa["plan"], row)
        result = qa_audit(study / "qa", gold=load_split(raw / "data", "test"), block_ids=qa_ids)
        save("qa-replay.json", result)
        if result["status"] != "PASS":
            raise ValueError("QA replay failed")
    status = {
        "stored_evidence_replay": "PASS",
        "measured_source_sha": source,
        "measurements": rebuilt["status"],
        "completed_cases": rebuilt["completed_cases"],
        "qa_replayed_block_ids": qa_ids,
        "cuda_contexts_created": 0,
        "model_downloads": 0,
        "independent_acceptance": False,
    }
    save("review-status.json", status)
    Console().print({**status, "output": str(args.output)})


if __name__ == "__main__":
    main()
