# Expert-cache allocation follow-up

The previous physical 8 GB Moonlight-16B-A3B experiment found two-read plus
2 GiB packed host LRU / 0.25 GiB dense GPU caching **6.11% slower** than two-read
alone. Its numerical comparisons passed, including bitwise capture equality.
See [the retained GPU report](https://github.com/SoupTeam/soup-rnd/blob/0beacac93b3184c2b04584c9ab9fc9e71d267852/benchmarks/two-read-expert-cache-20261025/GPU-RESULTS.md).

## What was wasting work

The host LRU stays bounded at 481 complete experts, 2,146,541,384 payload bytes.
It nevertheless evicts 16,809 entries during six timed optimizer updates, copying
69.86 GiB across approximately 134,000 CPU tensor allocations. The 15-entry
dense GPU hotset removes only 1.16% of NF4 materializations. Avoiding file reads
does not automatically outweigh this CPU work or eliminate host-to-GPU copies
and dequantization.

The reported 10.31 GiB RSS is the lifetime peak during setup/warmup. During
timing the combined arm peaks at 8.58–8.62 GiB and ends at 7.77–7.79 GiB,
compared with 3.77–3.80 GiB for two-read. Payload accounting does not measure
native allocator overhead or retained pages. Source inspection and stable
entry/RSS counts do not show an unbounded entry leak. Allocator retention is a
hypothesis until independently instrumented on Linux; it is not inferred just
from the difference between payload and RSS. The audit JSON retains all four
arms and stage boundaries.

## Candidate

The opt-in `host_cache_policy="static"` owns the first complete expert payloads
that fit and bypasses subsequent misses without cloning, eviction or overwrite.
The default LRU and the dense device hotset remain unchanged. All reader identity
checks, NF4 sidecars and numerical operations remain intact. Setup resamples live
host memory after the candidate probe; static admission then clears synthetic
probe entries so real training chooses the hotset. Payload ceilings stay reserved.

This is a workload-specific candidate, not a universal LRU replacement. On the
captured two-read trace, first-fit admission nearly preserves file-read counts;
on ordinary three-read streaming it loses substantial immediate replay reuse.

## Local CPU allocation control

The frozen route comes from the previous real accuracy capture: four accumulated
microbatches / two AdamW updates, 11,026 two-read visits. Exact order is qualified
against actual native `ExpertFetcher.fetch` calls in a tiny-model test. The CPU
experiment uses synthetic bytes with all eight real packed Moonlight NF4 tensor
shapes (4,462,664 bytes/expert), one reusable staging allocation, and a metadata
filter for the same 15 device entries. It performs no GPU/SSD/dequantization work.

Uninstrumented fresh-process runs repeat this fixed trace three times; these
are not reconstructed future training routes. One local Windows / CPU Torch
2.14.0 control per policy, with one Torch thread:

| Policy | Host insertions | Host hits | Modeled file reads | Cloned payload | CPU cache-operation wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 22,074 | 10,656 | 22,089 | 91.74 GiB | 7.586 s |
| Static | 481 | 10,655 | 22,090 | 2.00 GiB | 0.544 s |

`counts.hits` excludes the separate ownership-verification lookup. Times measure
cache operations only; they are neither end-to-end training times nor a GPU
speedup. Local process peaks are approximately 2.23 GiB for both policies and do
not reproduce the old Linux CUDA-process RSS gap.

A separate instrumented 2,200-visit prefix observes exactly 13,632 CPU clone calls
/ 7,604,379,456 copied bytes for LRU, and 3,848 / 2,146,541,384 for static.
The prefix deliberately lacks the full trace's reuse pattern, so its hit rate
must not replace the complete-trace comparison. Profiler timing is excluded
from throughput claims. Stored JSON includes source/route hashes, allocation
counts, ownership checks, RSS, after-clear/GC snapshots and limitations.
All original JSON, both complete CPU profiler traces and validation logs/XML
are retained byte-for-byte in `cpu-evidence.zip`; `evidence-manifest.json` lists
every original length and SHA-256. This keeps raw routing arrays out of the
review diff without changing or dropping their contents.

Candidate-scoped local validation: **301 passed, four CUDA skips**, Ruff and
applicable staged pre-commit clean. An additional runtime/residency/RAM/SSD
integration selection produced **838 passed, 21 CUDA skips, one failure**.
That failure is the existing issue-971 AST scan expecting `load_async` instead
of `_copy_layer`; it reproduces unchanged on isolated source `55137de2`, with
identical runtime/test bytes after newline normalization. The actual borrower
release control passes. This is not a green full-repository or hosted-CI claim.

## Requalification

Run unchanged reference-relative `rtol=.01, atol=1e-5`, finite and three-decimal
loss gates before timing. Compare each candidate separately against fresh
two-read, retaining initial adapters, routed inputs, router/LoRA gradients and
two accumulated AdamW updates. Bitwise equality is reported separately.
Balanced fresh-process timing must use the same inputs, optimizer, context and
resource limits; the existing practical gate requires the lower paired-bootstrap
confidence bound to exceed 2% step-time reduction. No GPU gain is established
by the CPU allocation control.
