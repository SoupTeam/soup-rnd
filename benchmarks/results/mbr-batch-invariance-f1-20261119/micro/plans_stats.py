"""Diagnosis 4: how many distinct segment-count tuples does a real segmented step have?

Runs the bridge under the segmented context and, when the context exits, writes the statistics of
``state.plans`` (the latest count list of each (layer, expert)). A CUDA graph for a whole expert
call is keyed by that tuple, so the number of distinct tuples is the number of graphs.

Usage: plans_stats.py <bridge.py> <out.json> [bridge args...]
"""
import collections
import json
import runpy
import sys
from contextlib import contextmanager

bridge, out = sys.argv[1], sys.argv[2]
sys.argv = [bridge] + sys.argv[3:]
import p72_invariant_expert_gemm as module  # noqa: E402

original = module.segmented_expert_gemm


@contextmanager
def wrapped():
    with original() as state:
        yield state
    plans = [tuple(counts) for counts in state.plans.values()]
    tuples = collections.Counter(plans)
    sizes = collections.Counter(size for counts in plans for size in counts)
    summary = {
        "leases": len(plans),
        "distinct_tuples": len(tuples),
        "tuples_seen_once": sum(1 for count in tuples.values() if count == 1),
        "mean_segments_per_lease": sum(len(p) for p in plans) / max(len(plans), 1),
        "max_segments": max((len(p) for p in plans), default=0),
        "distinct_segment_sizes": len(sizes),
        "max_segment_size": max(sizes, default=0),
        "top_tuples": [[list(t), c] for t, c in tuples.most_common(5)],
        "rows_total": sum(sum(p) for p in plans),
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)


module.segmented_expert_gemm = wrapped
try:
    runpy.run_path(bridge, run_name="__main__")
except SystemExit:
    pass
