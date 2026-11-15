"""CPU-only: run the real pairing collector and frozen-task checks on shards that exist now.

Read-only; no model, no CUDA, nothing written. An unfinished endpoint is expected to stop
only at the final coverage check ("incomplete or reordered endpoint coverage").
"""
import json
import time
from pathlib import Path

from benchmarks.harness import p3_deepseek_quality_eval as ev
from benchmarks.harness import p3_stage2_generation as generation

PROFILE = "moonlight-16b-a3b"
ROOT = Path("/data/p3-moonlight-quality")
E = ROOT / "evidence/moonlight-resident-20261111t201548z-whole3"
DATA = ROOT / "source/benchmarks/results/p3-stage2-v2/data-build-kaggle-v1/evidence/data"
study = json.loads((E / "frozen-study-protocol.json").read_bytes())
groups = {}
for path in sorted(E.glob("eval-*")):
    if path.name.endswith(("-process", "-observation")) or ".interrupted-" in path.name:
        continue
    manifest = path / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_bytes()).get("status") == "COMPLETE":
        m = json.loads(manifest.read_bytes())
        groups.setdefault((m["arm"], m["length"], m["endpoint"]), []).append(path)

passed = {}
for (arm, length, endpoint), paths in sorted(groups.items()):
    began = time.time()
    try:
        cells = ev._collect(paths, arm, profile_id=PROFILE)
        passed[(arm, length, endpoint)] = paths
        outcome = f"COLLECT PASS rows={sum(len(rows) for _m, rows in cells.values())}"
    except Exception as error:  # noqa: BLE001 - report the exact refusal text
        outcome = f"COLLECT STOP: {type(error).__name__}: {error}"
    # pair() lines 1033-1039: every scored row must match the frozen task content.
    count = study["loglik"]["hellaswag_n"] if endpoint == "hellaswag" else None
    tasks, _binding = ev.admit_tasks(DATA, endpoint, 0, count or len(
        ev.frozen._task_rows(DATA, ev.frozen._read_manifest(DATA, ev.DATA_MANIFEST_SHA256),
                             endpoint, development=False)), hellaswag_n=count)
    checked = mismatched = 0
    for path in paths:
        for line in (path / "rows.jsonl").read_text("utf-8").splitlines():
            row = json.loads(line)
            task = tasks[row["task_index"]]
            scoring = {name: task[name] for name in ("endpoint", "metric", "choices", "gold",
                                                     "target") if name in task}
            checked += 1
            if not (row["task_id"] == task["task_id"]
                    and row.get("task_sha256") == generation._object_hash(task)
                    and row.get("task_scoring") == scoring):
                mismatched += 1
    print(f"{arm} L{length} {endpoint} ({len(paths)} shard(s)): {outcome}; "
          f"frozen task content checked={checked} mismatched={mismatched}; "
          f"{time.time() - began:.0f}s", flush=True)


# Cross-arm pairing (the real pair_shards) for every cell that is complete in both arms.
for length in (1024, 4096):
    both = sorted(e for (a, l, e) in passed if a == ev.ARMS[0] and l == length
                  and (ev.ARMS[1], length, e) in passed)
    if not both:
        continue
    try:
        primary, secondary = ev.pair_shards(
            [p for e in both for p in passed[(ev.ARMS[0], length, e)]],
            [p for e in both for p in passed[(ev.ARMS[1], length, e)]], profile_id=PROFILE)
        print(f"PAIR L{length} {both}: PASS primary_rows={len(primary)} "
              f"secondary_rows={len(secondary)}", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"PAIR L{length} {both}: REFUSED {type(error).__name__}: {error}", flush=True)