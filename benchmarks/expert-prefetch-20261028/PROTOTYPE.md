# Packed NF4 expert transfers and one-expert lookahead

Experimental direct API; defaults, CLI and config schema are unchanged.
GPU accuracy, speed and capacity are **not yet qualified**.

`runtime.expert_fetcher.configure_transfer("packed")` groups an expert's packed
weights and NF4 sidecars into one pinned byte arena and one H2D operation.
`configure_transfer("prefetch")` additionally stages the next routed expert before
computing the current expert. The existing `"sync"` path remains the default.
All three modes use the same dequantization and expert arithmetic.

Two slots have independent transfer-complete and post-dequantization consumer
events. CPU storage is not overwritten until its transfer finishes; GPU packed
storage is not overwritten until dequantization has finished reading it. Dense
decoded weights have their own storage and retain the existing autograd lifetime.
Lookahead is skipped during two-read checkpoint replay and cleared on exceptions.
Backward uses packed demand transfers without changing the two-read graph.

The prototype requires CUDA and two NF4 matrices; unsupported inputs fail rather
than silently switch mode. Each slot is bounded to 16 MiB and allocated only to
the actual payload size. Extra pinned RAM and GPU staging are reported separately
from the existing cache ceilings. This is not whole-model automatic placement.

Cache admission and logical visit counters are unchanged. Non-consuming cache
peeks do not affect LRU order or hit/miss counts. Speculative real reads still
count as I/O, including a canceled lookahead. Actual fetch validates file identity
before consuming prefetched bytes. Changing cache settings drains pending work.

Local tests cover byte layout, scalar-sidecar alignment, API refusals, default
streaming, complete CPU optimizer trajectories and two-read contracts. Prepared
real-CUDA controls exercise delayed consumers, mutated host inputs, ring wrap,
missing-consumer-fence detection, cancellation/reset/file changes and complete
Kanana-shaped NF4/BF16 optimizer captures under the original numerical tolerances.
Those CUDA cells must actually pass before downloading the real 30B checkpoint.

Next: fresh same-source sync/packed/prefetch comparisons on physical 8 GB VRAM,
native RAM at most 32 GiB with workload swap disabled. Accuracy precedes speed and
memory. Report failed/negative outcomes, setup cost and complete-entry time; do
not infer whole-job gains or downstream quality from short random-token captures.
