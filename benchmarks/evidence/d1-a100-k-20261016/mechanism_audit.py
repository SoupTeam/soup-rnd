"""Describe an accepted fallback operating point without inventing a profiler."""
from __future__ import annotations

import hashlib
import json
import subprocess


def audit(study, index, repo):
    from d1_full_study import records

    name = "projection-float16-512x4096x4096-native-forced"
    case = next(row for row in index["cases"] if row["plan"]["name"] == name)
    comp = case["comparisons"][0] if case["comparisons"] else {}
    if comp.get("status") not in ("MEASURED", "SAME_PATH"):
        return {"status": "UNAVAILABLE", "case": name, "reason": "timing not accepted"}
    rows = [row for row in records(study / name) if row["block"] in comp["accepted_block_ids"]]
    source = index["source_sha"]
    path = "src/soup_cli/utils/layer_stream_runtime.py"
    blob = subprocess.check_output(["git", "-C", str(repo), "show", source + ":" + path])
    blob_sha = hashlib.sha256(blob).hexdigest()
    assert all(row["runtime_sha256"] == blob_sha for row in rows)
    result = {
        "status": "PASS",
        "case": name,
        "source_sha": source,
        "source_file": path,
        "source_blob_sha256": blob_sha,
        "accepted_block_ids": comp["accepted_block_ids"],
        "arms": len(rows),
        "routes": {},
        "scope": "Accepted timing rows and an untimed candidate-forward backend control. "
        "The control does not count checkpoint-step dequantizations or profile their cost.",
        "static_source_observation": "The Function backward explicitly dequantizes dense W "
        "and uses torch.matmul for dX; forcing its forward does not fuse that backward.",
        "dense_ceiling_scope": "Auxiliary same-session forward CUDA-event reference, "
        "taken after peak capture. A separate ceiling telemetry window was not retained; "
        "no frequency-matched ceiling multiplier is established.",
    }
    for arm, route in (("A", "native"), ("B", "forced")):
        selected = [row for row in rows if row["route"] == route]
        assert selected and all(not row["native_fused"] for row in selected)
        controls = sorted(
            {json.dumps(row["backend_counters_untimed"], sort_keys=True) for row in selected}
        )
        expected = {"custom": 0, "fallback": 0 if route == "native" else 1}
        assert [json.loads(value) for value in controls] == [expected]
        result["routes"][route] = {
            "paths": comp["paths"][arm],
            "untimed_forward_backend_controls": [expected],
            "median_wall_ms": comp["median_ms"][arm],
            "median_forward_cuda_ms": case["descriptive_metrics"][arm]["forward_cuda_ms"],
            "median_backward_cuda_ms": case["descriptive_metrics"][arm]["backward_cuda_ms"],
        }
    return result
