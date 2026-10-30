"""Audit the retained prospective full-forward gate, not a new CUDA experiment."""

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

HEAD = "c94214d397ffbe609916221e43fb2efd8c43ad4d"
CODECS = ("dense", "nf4", "nf4-dq")
MODES = ("native", "sequential", "grouped-resident", "grouped")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(folder, checkout):
    def blob(name):
        return subprocess.check_output(["git", "-C", str(checkout), "show", HEAD + ":" + name])

    protocol_name = "benchmarks/protocols/d3-full-model/protocol.json"
    rule = json.loads(blob(protocol_name))
    assert rule["control_revision"] == 2
    assert rule["negative_control"] == "zero_entire_active_expert_down_projection"
    assert rule["attempts"] == 1 and rule["replacements"] == 0
    assert rule["execution_enabled"] is True
    assert (
        digest(blob("benchmarks/protocols/d3-full-model/setup-evidence.zip"))
        == rule["setup_archive_sha256"]
    )
    session = json.loads((folder / "session.json").read_bytes())
    assert session["status"] == "full_forward_gate_complete_review_required", session.get(
        "traceback"
    )
    source = session["source"]
    assert source["git_head"] == HEAD
    assert set(source["git_blob_sha256"]) == {*rule["sources"], protocol_name}
    for name, want in source["git_blob_sha256"].items():
        assert digest(blob(name)) == want, name
    protocol_hash = digest(blob(protocol_name))
    with zipfile.ZipFile(
        io.BytesIO(blob("benchmarks/protocols/d3-full-model/setup-evidence.zip"))
    ) as z:
        assert z.testzip() is None
        with zipfile.ZipFile(io.BytesIO(z.read("raw/new-host-setup.zip"))) as raw:
            vocab = json.loads(raw.read("prepared/dense/config.json"))["vocab_size"]
    pairs = [(r["codec"], r["mode"]) for r in session["cells"]]
    assert pairs == [(c, m) for c in CODECS for m in MODES]
    pids = set()
    result = {
        "scope": (
            "Trained OLMoE complete forward with synthetic token IDs; "
            "not training, quality, throughput or full D3 sign-off"
        ),
        "source": source,
        "protocol_sha256": protocol_hash,
        "cells": 12,
        "formats": {},
        "d3_accepted": False,
    }
    import torch
    from safetensors.torch import load_file

    for codec in CODECS:
        cells = {}
        expected = load_file(str(folder / (codec + "-native") / "logits.safetensors"))["logits"]
        assert tuple(expected.shape) == (1, 512, vocab) and torch.isfinite(expected).all()
        expected_hash = digest(expected.contiguous().view(torch.uint8).numpy())
        for mode in MODES:
            r = json.loads((folder / (codec + "-" + mode) / "result.json").read_bytes())
            assert r == session["cells"][pairs.index((codec, mode))]
            assert r["status"] == "ok" and r["formal_execution"] is True
            assert r["source"] == source and r["protocol_sha256"] == protocol_hash
            assert r["F1"]["outcome"] == "ok"
            assert r["bundle_sha256"] == rule["bundle_sha256"][codec]
            assert r["software"] == rule["software"] and r["gpu_uuid"] == rule["gpu_uuid"]
            assert r["vram_bytes"] >= rule["minimum_reported_vram_bytes"]
            assert r["expected_sha256"] == expected_hash and r["negative_sha256"] != expected_hash
            assert r["negative_control"] == rule["negative_control"]
            assert r["control_target"] == {"layer": 0, "expert": r["traces"][0][0]["expert"]}
            assert len(r["visible_workload"]) == 1
            pid = r["visible_workload"][0]["pid"]
            assert pid not in pids
            pids.add(pid)
            assert len(r["traces"]) == 16
            for layer, rows in enumerate(r["traces"]):
                experts = [t["expert"] for t in rows]
                assert experts and experts == sorted(set(experts))
                assert sum(t["token_routes"] for t in rows) == rule["tokens"] * 8
                assert all(
                    t["layer"] == layer
                    and 0 <= t["expert"] < 64
                    and t["token_routes"] > 0
                    and t["status"] == ("routed" if mode == "native" else "ok")
                    for t in rows
                )
            if mode in ("sequential", "grouped"):
                pool = r["pool"]
                assert pool["weight_budget_bytes"] == rule["weight_budget_bytes"]
                assert pool["plan"]["weight_bytes"] <= rule["weight_budget_bytes"]
                assert pool["cuda_allocation_delta"] <= rule["weight_budget_bytes"]
            if mode in ("grouped-resident", "grouped"):
                assert r["first_forward_gemm"]["grouped_matrices"] > 0
                assert r["first_forward_gemm"]["native_olmoe_rows_max"] == 128
                assert r["gemm"]["workspace_bytes"] == rule["workspace_bytes"]
            cells[mode] = {
                "peak_allocated_bytes": r["peak_allocated"],
                "peak_reserved_bytes": r["peak_reserved"],
                "peak_rss_bytes": r["peak_rss_bytes"],
                "pool": r["pool"],
                "first_forward_gemm": r.get("first_forward_gemm"),
                "expected_sha256": r["expected_sha256"],
                "negative_sha256": r["negative_sha256"],
                "input_sha256": r["input_sha256"],
                "F1": r["F1"]["outcome"],
            }
        assert len({r["input_sha256"] for r in cells.values()}) == 1
        result["formats"][codec] = cells
    assert len(pids) == 12
    assert {p.parent.name for p in folder.glob("*/result.json")} == {
        c + "-" + m for c in CODECS for m in MODES
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.session, args.checkout)
    args.out.write_bytes((json.dumps(value, indent=2) + "\n").encode())
    from rich.console import Console

    Console().print("Verified 12 fresh-process full-forward cells; not task sign-off:", args.out)
