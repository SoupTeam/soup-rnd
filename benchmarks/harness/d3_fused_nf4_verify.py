"""CPU-only evidence audit; does not execute or certify a CUDA numerical experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

from benchmarks.harness.d3_fused_nf4_gate import evaluate_cells
from benchmarks.harness.d3_trainer_protocol import contained, file_hash


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def zipped(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "ZIP CRC failure")
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate ZIP member")
        return {name: archive.read(name) for name in names}


def verify_matrix(files: dict[str, bytes], prefix: str) -> int:
    """Compare actual retained safetensors bytes, not only PASS strings."""
    matrix = json.loads(files[prefix + "matrix-01/status.json"])
    require(matrix["status"] == "passed" and len(matrix["cells"]) == 8,
            "incomplete trainer matrix")
    seen = {row["case"] for row in matrix["cells"] if row["status"] == "passed"}
    for codec in ("nf4", "nf4-dq"):
        for l2l in (0, 1):
            names = [f"matrix-01/{codec}-{l2l}-{mode}"
                     for mode in ("sequential", "fused_nf4")]
            require(set(names) <= seen, "missing declared trainer child")
            rows = [json.loads(files[prefix + name + "/status.json"]) for name in names]
            require(all(row["status"] == "passed" for row in rows), "trainer failure")
            require(rows[0]["final_parameters"] == rows[1]["final_parameters"],
                    "adapter update mismatch")
            require(rows[1]["runtime"]["expert_gemm"]["fused_matrices"] > 0,
                    "fused path was inactive")
            for payload in ("gradients-0.safetensors", "gradients-1.safetensors",
                            "trained-logits.safetensors"):
                require(files[prefix + names[0] + "/" + payload]
                        == files[prefix + names[1] + "/" + payload],
                        "actual gradient/logit payload mismatch")
            for row in rows:
                reload = row["reload_comparison"]
                require(reload["actual_dtype"] == reload["wanted_dtype"]
                        and reload["actual_sha256"] == reload["wanted_sha256"],
                        "same-recipe reload mismatch")
    return 8


def verify(folder: Path) -> dict:
    manifest = json.loads((folder / "SHA256SUMS.json").read_text())
    require(bool(manifest), "empty publication manifest")
    for name, expected in manifest.items():
        require(file_hash(contained(folder, name)) == expected, f"publication hash: {name}")
    raw = zipped(folder / "development-results.zip")
    sources = zipped(folder / "development-sources.zip")
    require(not set(raw) & set(sources), "ambiguous raw/source member")
    files = {**raw, **sources}
    prefix = "d3-fused-screen/"
    original = json.loads(files[prefix + "raw-file-manifest.json"])
    receipt = json.loads((folder / "verification.json").read_text())
    missing = receipt["excluded_synthetic_fixture_payloads"]
    for name, expected in original.items():
        key = prefix + name
        if key not in files:
            require(key in missing and missing[key]["sha256"] == expected["sha256"]
                    and missing[key]["bytes"] == expected["bytes"],
                    "undeclared omitted original payload")
            continue
        data = files[key]
        require(len(data) == expected["bytes"]
                and hashlib.sha256(data).hexdigest() == expected["sha256"],
                f"original payload hash: {name}")
    arithmetic = json.loads(files[prefix + "attempt-04/result.json"])
    require(not evaluate_cells(arithmetic["cells"]), "incomplete/failed arithmetic grid")
    for attempt in ("attempt-01", "attempt-02", "attempt-03"):
        require(bool(evaluate_cells(json.loads(files[prefix + attempt + "/result.json"])["cells"])),
                "unsuccessful raw candidate no longer retained as such")
    dispatch = json.loads(files[prefix + "dispatch-02/result.json"])
    require(len(dispatch["cells"]) == 8, "incomplete dispatch grid")
    require(len({row["name"] for row in dispatch["cells"]}) == 8, "duplicate dispatch cell")
    for row in dispatch["cells"]:
        require(row["pass"] and row["kernels"] == ["fused"] and row["input_state_unchanged"]
                and row["allocator_delta_bytes"] <= row["plan"]["call_bytes"],
                "dispatch/budget/state failure")
        trace = files[prefix + "dispatch-02/" + row["name"] + ".json"]
        require(hashlib.sha256(trace).hexdigest() == row["trace_sha256"], "trace hash mismatch")
    mlp = json.loads(files[prefix + "mlp-02/result.json"])
    require(len(mlp["cells"]) == 8 and all(row["pass"] for row in mlp["cells"]),
            "incomplete/failed MLP grid")
    require(json.loads(files[prefix + "mlp-01/result.json"])["status"] == "error",
            "allocator refusal missing")
    first = verify_matrix(files, prefix + "trainer-development-01/")
    with tarfile.open(folder / "allocator-followup.tar.gz") as archive:
        followup = {member.name: archive.extractfile(member).read()
                    for member in archive if member.isfile()}
    second = verify_matrix(followup, "d3-fused-screen-08/trainer-development-01/")
    source = json.loads(followup["d3-fused-screen-08/DEVELOPMENT-SOURCE.json"])
    for name, expected in source["files"].items():
        require(hashlib.sha256(followup["d3-fused-screen-08/" + name]).hexdigest() == expected,
                "follow-up executed source mismatch")
    # The original source packs retain earlier failed versions as well. Reading
    # them verifies their CRC/container integrity without executing source code.
    packs = 0
    for name, data in sources.items():
        if name.endswith(".tar.gz"):
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                for member in archive:
                    if member.isfile():
                        archive.extractfile(member).read()
            packs += 1
    return {"kind": "offline audit of synthetic development evidence; no CUDA rerun",
            "publication_hashes": len(manifest), "original_manifest_entries": len(original),
            "original_payloads_omitted_and_disclosed": len(missing),
            "arithmetic_cells": 20, "dispatch_profiles": 8, "mlp_cells": 8,
            "synthetic_trainer_children": [first, second], "retained_source_packs": packs,
            "formal_pretrained_verdict": False}


if __name__ == "__main__":
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    Console().print_json(data=verify(parser.parse_args().folder))
