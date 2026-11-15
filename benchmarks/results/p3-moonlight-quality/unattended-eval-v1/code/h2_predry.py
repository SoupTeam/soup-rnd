"""Read-only pre-check of pair()/analysis requirements on the shards that already exist.

No model, no CUDA, no project evaluator import; writes nothing outside stdout.
Part A is stdlib only. Part B imports the pure statistics modules the analysis step uses.
"""
import hashlib
import json
import sys
from pathlib import Path

E = Path("/data/p3-moonlight-quality/evidence/moonlight-resident-20261111t201548z-whole3")
SCORING = Path("/data/p3-moonlight-quality/source/benchmarks/results/p3-stage2-v2/"
               "p3-stage2-v2-protocol.json")
study = json.loads((E / "frozen-study-protocol.json").read_bytes())
PROTO = hashlib.sha256((E / "frozen-study-protocol.json").read_bytes()).hexdigest()
bad = []


def check(ok, label):
    if not ok:
        bad.append(label)


shards = {}
for path in sorted(E.glob("eval-*")):
    if path.name.endswith(("-process", "-observation")) or ".interrupted-" in path.name:
        continue
    manifest = path / "manifest.json"
    if not manifest.exists():
        continue
    value = json.loads(manifest.read_bytes())
    if value.get("status") != "COMPLETE":
        continue
    shards[path.name] = (path, value)
print("complete shards:", len(shards))

cells = {}
for name, (path, m) in shards.items():
    ident = m["adapter_identity"]
    # pair() lines 1015-1030
    check(m["protocol_sha256"] == PROTO, f"{name}: protocol sha")
    check(m["loglik"] == study["loglik"], f"{name}: loglik != study")
    check(m["scoring"] == study["scoring"], f"{name}: scoring != study")
    check(ident["completed_steps"] == study["training"]["steps"] == 64, f"{name}: steps")
    for key in ("initial_adapter_sha256", "pack_sha256", "source", "cache", "tokenizer"):
        check(ident[key] == study[key], f"{name}: adapter identity {key} != study")
    check(m["tokenizer"] == study["tokenizer"], f"{name}: shard tokenizer != study")
    # _collect per-shard identity (lines 806-867, the parts that need no project import)
    check(ident.get("mode") == "final" and ident.get("arm") == m["arm"]
          and ident.get("length") == m["length"] and ident.get("seed") == 23
          and ident.get("protocol_sha256") == m["protocol_sha256"], f"{name}: identity cell")
    check(ident["source"] == m["model"]["source"] and ident["cache"] == m["model"]["cache"],
          f"{name}: adapter/model source-cache")
    check(ident.get("attention_policy") == m.get("attention_policy"), f"{name}: attention policy")
    check(m.get("evaluation_storage") == "resident_nf4"
          and m.get("runtime", {}).get("tier") == "resident_nf4", f"{name}: storage tier")
    check(ident.get("tokenizer") == m.get("tokenizer"), f"{name}: adapter/eval tokenizer")
    check("batches.jsonl" in m["files"] and "rows.jsonl" in m["files"], f"{name}: files")
    cells.setdefault((m["arm"], m["length"], m["endpoint"]), []).append((name, path, m))

# adapters_by_length (pair line 1031): one adapter identity per arm/length across endpoints,
# including the two shards reused from the first host.
by_arm_length = {}
for (arm, length, endpoint), items in cells.items():
    for name, _path, m in items:
        first = by_arm_length.setdefault((arm, length), (name, m["adapter_identity"]))
        check(first[1] == m["adapter_identity"],
              f"{name}: adapter identity differs from {first[0]} (adapter changed between endpoints)")
for key, (name, _ident) in sorted(by_arm_length.items()):
    members = sorted(n for (a, l, _e), items in cells.items() if (a, l) == key for n, _, _ in items)
    print("one adapter identity for", key, "over", len(members), "shards:",
          sorted({n.split("-L")[1].split("-")[1] for n in members}))

# cross-arm fields pair_shards compares (line 944), old host vs new host and bf16 vs oracle
ref_name, (_p, ref) = next(iter(shards.items()))
for name, (_path, m) in shards.items():
    for key in ("model", "protocol_sha256", "loglik", "scoring", "tokenizer",
                "source_files_sha256"):
        check(m[key] == ref[key], f"{name}: {key} differs from {ref_name}")

# multi-shard join (_collect lines 870-928)
for (arm, length, endpoint), items in sorted(cells.items()):
    items.sort(key=lambda item: item[2]["binding"]["start"])
    first = items[0][2]
    cursor = 0
    contiguous = True
    for name, path, m in items:
        b = m["binding"]
        for key in ("adapter_identity", "model", "protocol_sha256", "loglik", "scoring",
                    "tokenizer", "source_files_sha256"):
            check(m[key] == first[key], f"{name}: join field {key}")
        for key in ("complete_endpoint_row_count", "endpoint_ordered_task_ids_sha256",
                    "data_manifest_sha256", "tasks_file"):
            check(b[key] == first["binding"][key], f"{name}: binding {key}")
        rows = [json.loads(line) for line in (path / "rows.jsonl").read_text("utf-8").splitlines()]
        if b["start"] != cursor:
            contiguous = False
        check(type(b["start"]) is int and type(b["stop"]) is int and b["stop"] > b["start"]
              and len(rows) == b["stop"] - b["start"]
              and [r["task_id"] for r in rows] == b["ordered_task_ids"], f"{name}: coverage")
        for index, r in enumerate(rows, start=b["start"]):
            ok = (r["endpoint"] == endpoint and r["length"] == length and r["seed"] == 23
                  and r["task_index"] == index and r["adapter_identity"] == m["adapter_identity"]
                  and r["task_scoring"]["endpoint"] == endpoint
                  and r.get("profile_id") == "moonlight-16b-a3b")
            if endpoint in ("hellaswag", "arc", "boolq", "lambada"):
                ok = ok and type(r.get("pass")) is bool
            if not ok:
                check(False, f"{name}: row {index} identity")
                break
        cursor = b["stop"]
    total = first["binding"]["complete_endpoint_row_count"]
    print(f"{arm} L{length} {endpoint}: {len(items)} shard(s), covered [0,{cursor}) of {total}, "
          f"contiguous so far={contiguous}")

print("PART A failures:", bad if bad else "none")

if "--stats" in sys.argv:
    from benchmarks.harness.p3_stage2_statistics import summarize_pairs
    from benchmarks.harness.p3_stage2v2_decision import apply_informativeness_gate
    from benchmarks.harness.p3_stage2v2_results import ARMS, summarize_nll
    scoring = json.loads(SCORING.read_bytes())
    assert (scoring["margin"], scoring["one_sided_alpha"], scoring["floor_offset"]) == (
        0.01, 0.025, 0.05)
    assert set(scoring["chance"]) >= {"hellaswag", "arc", "boolq", "lambada"}

    def rows_of(name):
        return [json.loads(line) for line in
                (E / name / "rows.jsonl").read_text("utf-8").splitlines()]
    left = rows_of("eval-bf16_l2l-L1024-boolq-0-3270")
    right = rows_of("eval-oracle_block64_e4m3-L1024-boolq-0-3270")
    assert [r["task_id"] for r in left] == [r["task_id"] for r in right]
    pairs = [{"benchmark": "boolq", "length": 1024, "seed": 23, "task_id": a["task_id"],
              "profile_id": "moonlight-16b-a3b", "baseline_pass": a["pass"],
              "candidate_pass": b["pass"]} for a, b in zip(left, right, strict=True)]
    raw = summarize_pairs(pairs, expected_tasks={"boolq": [r["task_id"] for r in left]},
                          seeds=[23], lengths=[1024], margin=scoring["margin"],
                          one_sided_alpha=scoring["one_sided_alpha"])
    gated = apply_informativeness_gate(raw, chance=scoring["chance"],
                                       floor_offset=scoring["floor_offset"])
    assert "decision" in gated and "statistical_decision" in gated
    nll = []
    for task, (x, y) in enumerate(((1.5, 1.6), (2.0, 1.9), (0.7, 0.75))):
        for arm, mean in zip(ARMS, (x, y), strict=True):
            nll.append({"endpoint": "gsm8k", "task_id": f"t{task}", "length": 1024, "seed": 23,
                        "arm": arm, "n_targets": 10, "nll_sum": mean * 10, "nll_mean": mean})
    secondary = summarize_nll(nll, expected_tasks={"gsm8k": ["t0", "t1", "t2"]}, seeds=[23],
                              lengths=[1024])
    assert all(cell["status"] == "COMPLETE" for cell in secondary)
    json.dumps({"quality": gated, "secondary_nll": secondary}, sort_keys=True, allow_nan=False)
    cell = gated["cells"][0]
    print("PART B ok: statistics, gate, NLL summary and strict JSON all run on real BoolQ rows;",
          "n =", cell["n"], "keys:", sorted(gated)[:8])
