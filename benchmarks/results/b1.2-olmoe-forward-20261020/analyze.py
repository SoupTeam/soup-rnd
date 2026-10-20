"""Verify retained B1.2 gate records against the prospective Git source.

This checks recorded hashes, routing, traces and memory; it does not rerun CUDA
or reconstruct model outputs. Requires the frozen commit in a local Git repo.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath

from rich.console import Console

COMMIT = "99664b09ad0b3238dc0a8a790f207f140239ea31"
PROTOCOL = "benchmarks/protocols/b1.2-expert-forward/protocol.json"
EXPORTS = (
    "src/soup_cli", "pyproject.toml", "tests/conftest.py", "tests/__init__.py",
    "tests/test_b1_expert_shard.py", "tests/test_b1_expert_forward.py",
    "tests/test_b1_expert_forward_gate.py", "benchmarks/harness/b1_expert_forward_gate.py",
    PROTOCOL,
)
HASH = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    require(result.utcoffset() is not None, "timestamp must include timezone")
    return result


def zip_files(bundle: zipfile.ZipFile) -> dict[str, bytes]:
    entries = bundle.infolist()
    require(len(entries) < 3000, "unexpected archive entry count")
    require(sum(e.file_size for e in entries) < 128 * 2**20, "archive exceeds audit limit")
    require(len({e.filename for e in entries}) == len(entries), "duplicate archive paths")
    for entry in entries:
        name = PurePosixPath(entry.filename)
        require(not name.is_absolute() and ".." not in name.parts
                and "\\" not in entry.filename, "unsafe archive path")
    require(bundle.testzip() is None, "archive CRC mismatch")
    return {e.filename: bundle.read(e) for e in entries if not e.is_dir()}


def analyze(archive: Path, repo: Path) -> dict:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)

    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        files = zip_files(bundle)
    manifest = json.loads(files["SHA256SUMS.json"])
    require(set(files) == set(manifest) | {"SHA256SUMS.json"}, "raw manifest coverage differs")
    for name, expected in manifest.items():
        require(sha(files[name]) == expected, f"raw hash mismatch: {name}")
    source_bytes = files["source.zip"]
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as bundle:
        sources = zip_files(bundle)
        require(bundle.comment.decode() == COMMIT, "Git archive comment differs")
    tree = git("ls-tree", "-r", "-z", COMMIT, "--", *EXPORTS)
    objects = {}
    for row in tree.rstrip(b"\0").split(b"\0"):
        metadata, name = row.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        require(kind == b"blob" and mode in (b"100644", b"100755"), "unexpected Git object")
        objects[name.decode()] = oid.decode()
    require(set(sources) == set(objects), "source snapshot file set differs from Git")
    for name, data in sources.items():
        oid = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(oid == objects[name], f"source Git blob differs: {name}")
    require(files["protocol.json"] == sources[PROTOCOL], "protocol differs from committed bytes")
    rule = json.loads(sources[PROTOCOL])
    require(rule["attempts_per_cell"] == 1 and rule["replacements"] == 0, "attempt budget differs")
    prepared = json.loads(files["b12-prepared-99664b09.json"])
    identity = prepared["source"]
    require(identity["commit"] == COMMIT and prepared["status"] == "prepared",
            "preparation differs")
    for name, expected in identity["git_blob_sha256"].items():
        require(sha(sources[name]) == expected, f"runtime source SHA differs: {name}")
    require(prepared["checkpoint_sha256"] == rule["checkpoint_files"], "prepared weights differ")
    require(prepared["model_id"] == rule["model_id"]
            and prepared["model_revision"] == rule["model_revision"], "model identity differs")
    require(prepared["native_bf16"] is True and prepared["compute_capability"][0] >= 8,
            "native BF16 preparation is missing")
    launch = json.loads(files["launch.json"])
    require(launch["commit"] == COMMIT and launch["planned_cells"] == 12, "launch identity differs")
    require(sha(files["launcher.py"]) == launch["launcher_sha256"], "launcher hash differs")
    require(sha(source_bytes) == launch["source_zip_sha256"], "source archive hash differs")
    series = json.loads(files["series.json"])
    plan = [(layer, cell) for layer in rule["layers"] for cell in rule["cells"]]
    require(len(plan) == 12 and len(series["cells"]) == 12, "incomplete 12-cell series")
    require(series["commit"] == COMMIT and series["status"] == "all_12_cells_passed",
            "series did not complete successfully")
    require(series["attempts_per_cell"] == 1 and series["replacements"] == 0,
            "series attempt budget differs")
    expected_results = {f"layer-{layer:02d}-{cell}/result.json" for layer, cell in plan}
    require({n for n in files if n.endswith("/result.json")} == expected_results,
            "missing or extra attempts")
    commit_time = timestamp(git("show", "-s", "--format=%cI", COMMIT).decode().strip())
    previous = timestamp(series["started_utc"])
    require(commit_time <= timestamp(prepared["finished_utc"]) <= timestamp(launch["started_utc"])
            <= previous, "source/preparation/launch timestamp order differs")
    rows, pids = [], set()
    for number, ((layer, cell), invocation) in enumerate(zip(plan, series["cells"]), 1):
        folder = f"layer-{layer:02d}-{cell}"
        record = json.loads(files[f"{folder}/result.json"])
        result = record["result"]
        require(invocation["number"] == number and invocation["layer"] == layer
                and invocation["cell"] == cell, "invocation order differs")
        require(invocation["verdict"] == "PASS" and invocation["exit_code"] == 0,
                "child process did not succeed")
        require(invocation["pid"] not in pids, "child PID reused")
        pids.add(invocation["pid"])
        require(previous <= timestamp(invocation["started_utc"])
                <= timestamp(record["started_utc"]) <= timestamp(record["finished_utc"])
                <= timestamp(invocation["finished_utc"]), "cell timestamp order differs")
        previous = timestamp(invocation["finished_utc"])
        require(record["layer"] == layer and record["cell"] == cell
                and record["verdict"] == "PASS", "cell verdict or identity differs")
        require(record["source"] == identity and record["protocol"] == rule,
                "cell source or protocol differs")
        require(record["checkpoint_sha256"] == rule["checkpoint_files"], "cell weights differ")
        for name, expected in rule["package_versions"].items():
            require(record["box"][name].split("+")[0] == expected, "cell package differs")
        require(record["box"]["gpu"] == prepared["gpu"], "GPU differs between cells")
        require(all(result.get(key) is True for key in
                    ("equal", "negative_differs", "finite", "loads_ok", "budget_ok")),
                "required acceptance flag failed")
        require(result["max_abs_diff"] == 0, "nonzero forward difference")
        for key in ("expected_sha256", "observed_sha256", "negative_sha256",
                    "experts_file_sha256"):
            require(bool(HASH.fullmatch(result[key])), "invalid tensor/file hash")
        require(result["expected_sha256"] == result["observed_sha256"], "output hashes differ")
        require(result["negative_sha256"] != result["expected_sha256"],
                "negative control ineffective")
        indices, weights = result["routing_indices"], result["routing_weights"]
        require(len(indices) == len(weights) == rule["tokens"], "token count differs")
        require(all(len(i) == len(w) == 8 for i, w in zip(indices, weights)), "top-k differs")
        require(all(type(i) is int and 0 <= i < 64 for row in indices for i in row),
                "invalid routed expert")
        require(all(math.isfinite(w) and w >= 0 for row in weights for w in row),
                "invalid routing weights")
        counts = Counter(i for row in indices for i in row)
        trace = result["trace"]
        require([e["expert"] for e in trace] == sorted(counts), "expert load order differs")
        require(all(e["layer"] == layer and e["status"] == "ok"
                    and e["token_routes"] == counts[e["expert"]] for e in trace),
                "trace does not agree with routing")
        pool = result["pool"]
        require(pool["slot_count"] == 1 and pool["fetches"] == len(trace), "load count differs")
        logical = sum(e["logical_bytes_read"] for e in trace)
        require(pool["logical_bytes_read"] == logical, "logical read accounting differs")
        budget = pool["weight_budget_bytes"]
        allocation, layout = pool["cuda_allocation_delta"], pool["plan"]
        components = ("dense_bytes", "packed_and_stats_bytes", "code_bytes",
                      "scratch_bytes", "padding_bytes")
        require(all(layout[k] >= 0 for k in components)
                and sum(layout[k] for k in components) == layout["weight_bytes"],
                "arena component accounting differs")
        require(budget == rule["weight_budget_bytes"] and 0 < layout["weight_bytes"] <= budget
                and 0 <= allocation <= budget, "arena exceeds budget")
        progress = json.loads(files[f"{folder}/progress.json"])
        require(progress == {"reader_fetches": pool["fetches"], "logical_bytes_read": logical,
                             "trace": trace, "pool": pool}, "final progress differs")
        require(f"PASS: /workspace/b1.2-forward-session/{folder}/result.json" in
                files[invocation["log"]].decode(), "child PASS log missing")
        row = {"layer": layer, "cell": cell, "verdict": "PASS", "max_abs_diff": 0.0,
               "routed_experts": len(trace), "token_routes": sum(counts.values()),
               "arena_planned_bytes": layout["weight_bytes"], "arena_allocated_bytes": allocation,
               "host_staging_bytes": layout["host_staging_bytes"], "logical_bytes_read": logical,
               "direct_io": result["reader_direct_io"], "negative_control_differs": True,
               "expected_sha256": result["expected_sha256"],
               "experts_file_sha256": result["experts_file_sha256"]}
        if cell.startswith("cuda"):
            require(allocation >= layout["weight_bytes"], "CUDA allocation below arena storage")
            for arm in ("resident_memory", "streamed_memory"):
                before, after = result[arm]["before"], result[arm]["after"]
                require(after["peak_allocated"] >= max(before["allocated"], after["allocated"])
                        and after["peak_reserved"] >= max(before["reserved"], after["reserved"],
                                                           after["peak_allocated"]),
                        "invalid CUDA memory accounting")
            streamed = result["streamed_memory"]
            require(streamed["after"]["allocated"] - streamed["before"]["allocated"] == allocation,
                    "streamed allocation delta differs from pool")
            row.update(resident_peak_allocated_bytes=result["resident_memory"]["after"]["peak_allocated"],
                       streamed_peak_allocated_bytes=streamed["after"]["peak_allocated"],
                       streamed_peak_reserved_bytes=streamed["after"]["peak_reserved"])
        rows.append(row)
    require(previous <= timestamp(series["finished_utc"]), "series ended before final cell")
    return {
        "scope": rule["scope"], "verdict": "PASS", "cells_passed": 12,
        "failed_or_no_verdict": 0, "replacements": 0, "commit": COMMIT,
        "protocol_sha256": sha(files["protocol.json"]), "raw_archive_sha256": sha(raw),
        "verification": {"raw_crc": "passed", "raw_manifest_hashes": len(manifest),
                         "source_crc": "passed", "source_git_blob_identities": len(objects),
                         "runtime_source_sha256": len(identity["git_blob_sha256"]),
                         "unique_child_pids": len(pids)},
        "checkpoint_sha256": rule["checkpoint_files"], "model_revision": rule["model_revision"],
        "gpu": prepared["gpu"], "packages": prepared["packages"],
        "started_utc": series["started_utc"], "finished_utc": series["finished_utc"],
        "cells": rows,
        "limits": [
            "Archive checks reconstruct the verdict from recorded evidence; no new CUDA execution.",
            "Trained routed-expert layers use synthetic activations, "
            "not a whole-model training run.",
            "NF4 equality uses independently dequantized reference weights "
            "from the same stored bytes.",
            "CUDA allocator peaks exclude driver/context memory; arena budget is not total VRAM.",
            "Reconstructed expert weight files stay on server; hashes and rebuilding sources "
            "are retained.",
            "No physical-host isolation, throughput, disk-read savings, "
            "backward or convergence claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.archive, args.repo)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    Console().print(f"Verified: {result['cells_passed']}/12 PASS; saved {args.output}")


if __name__ == "__main__":
    main()
