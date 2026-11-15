"""CPU-only exact forward counts per planned shard (no model, no CUDA, no scoring).

Same call sequence run_endpoint uses (eval.py 699-711 -> loglik _batch_plan/_position_chunks).
Usage: plan_shards.py OUT.json LIMIT TOKENIZER DATA SHARDS_JSON
Exit 1 if any shard's forwards (= torch.profiler sessions per process) exceed LIMIT.
"""
import json
import sys
from pathlib import Path

from benchmarks.harness import p3_deepseek_quality_eval as ev
from benchmarks.harness import p3_stage2v2_loglik as fz

out, limit = Path(sys.argv[1]), int(sys.argv[2])
tokenizer_dir, data = Path(sys.argv[3]), Path(sys.argv[4])
shards = json.loads(sys.argv[5])
V, LIM, CH = 163840, 536870912, 128
tokenizer, _ = ev._load_tokenizer(tokenizer_dir, profile_id="moonlight-16b-a3b")
result, worst = {}, 0
for endpoint, intervals in shards.items():
    for start, stop in intervals:
        rows, _ = ev.admit_tasks(data, endpoint, start, stop,
                                 hellaswag_n=9000 if endpoint == "hellaswag" else None)
        if endpoint in ev.SECONDARY:
            encoded = fz._document_tokens(tokenizer, [(r["context"], r["target"]) for r in rows])
        else:
            requests, _ = fz._requests(rows)
            encoded = fz._request_tokens(tokenizer, requests)
        plan = fz._batch_plan(encoded, max_tokens_per_batch=8192, vocabulary=V,
                              logits_memory_limit_bytes=LIM, reduction_chunk_tokens=CH)
        forwards = sum(len(fz._position_chunks(
            sorted({p for i in batch for p in encoded[i]["positions"]}), batch_size=len(batch),
            vocabulary=V, logits_memory_limit_bytes=LIM, reduction_chunk_tokens=CH))
            for batch in plan)
        result[f"{endpoint}-{start}-{stop}"] = {"batches": len(plan), "forwards": forwards,
                                                "tasks": stop - start}
        worst = max(worst, forwards)
        print(endpoint, start, stop, "batches", len(plan), "forwards", forwards, flush=True)
result["_summary"] = {"limit": limit, "max_forwards": worst,
                      "total_forwards": sum(v["forwards"] for k, v in result.items()
                                            if not k.startswith("_"))}
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
sys.exit(0 if worst <= limit else 1)
