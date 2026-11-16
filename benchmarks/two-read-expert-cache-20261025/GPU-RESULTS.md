# Moonlight-16B: two-read with expert RAM/GPU caches

The current caches slow two-read by **6.110%** in this workload; there is no incremental speed benefit.
All four short numerical captures pass the original gates and independently
recompare **bitwise** after lossless transfer. A negative speed result is retained.

## Fixed experiment

- Source [`55137de2`](https://github.com/SoupTeam/soup-rnd/tree/55137de277895c57c68ebe7cffe8ca3527fa0ff8);
  upstream two-read PR [#2158](https://github.com/MakazhanAlpamys/Soup/pull/2158)
  at `f2599ddc` is preserved by merge, with its authorship. The integration and runner
  remain on the companion branch; PR #5's older production diff is separate.
  The executed archive/per-file hashes are retained; the provenance audit records
  the CRLF/LF normalization bridge between archive bytes and the Git source.
- Real `moonshotai/Moonlight-16B-A3B`, revision `476b36a473d4467f94469414bef6cee75c9c8172`.
  Frozen NF4 base, BF16 compute, native FP32 attention LoRA rank 4 / alpha 8 on
  `q_proj`, `kv_b_proj`, `o_proj`; 162 trainable adapter tensors.
- Physical RTX 4060 Ti: 8188 MiB VRAM; native RAM ceiling **29.756 GiB**,
  `memory.swap.max=0`, no observed process/container swap. Disk tier, two decoder
  buffers, decoder recompute on, no retained activations or compilation.
- Batch 1, context 512, accumulation 2: **1024 processed tokens per optimizer update**.
  Identical frozen random IDs and adapter initialization in every arm.
  Cache arms both request **2 GiB packed RAM + .25 GiB dense GPU** ceilings.
  This is an explicit placement test, not a policy that fills all free memory.

## Speed and memory

| Arm | Median step, s | Tokens/s | Peak allocated VRAM, GiB | Peak reserved VRAM, GiB | Peak process RSS, GiB | Median setup, s | Fresh process wall, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary streaming | 10.7907 | 94.90 | 1.9681 | 2.0430 | 3.7895 | 7.092 | 99.977 |
| Two-read | 7.9076 | 129.50 | 1.9695 | 2.0234 | 3.7986 | 7.026 | 75.717 |
| RAM/GPU caches | 10.1201 | 101.18 | 2.2059 | 2.2930 | 10.8356 | 21.407 | 109.087 |
| Two-read + caches | 8.4017 | 121.88 | 2.2073 | 2.2734 | 10.3081 | 18.598 | 91.581 |

Four balanced rounds, **16 fresh processes**, two warmup plus six measured updates
each: **96 timed + 32 warmup updates**. No times or processes were excluded.
RSS is the native process lifetime high-water mark; step allocator peaks come
from unprofiled measured updates. Setup/warmup, scalar/state checks and file writes
are retained separately. Fresh process wall includes interpreter/build/setup costs.

- `cache` vs `ordinary`: paired step-time reduction **6.387%**, 95% interval [5.636, 6.962]%; >=2% gate: **PASS**.
- `two_read` vs `ordinary`: paired step-time reduction **26.780%**, 95% interval [25.882, 27.563]%; >=2% gate: **PASS**.
- `two_read_cache` vs `ordinary`: paired step-time reduction **22.307%**, 95% interval [21.227, 23.064]%; >=2% gate: **PASS**.
- `two_read_cache_vs_cache` vs `cache`: paired step-time reduction **17.006%**, 95% interval [16.543, 17.434]%; >=2% gate: **PASS**.
- `two_read_cache_vs_two_read` vs `two_read`: paired step-time reduction **-6.110%**, 95% interval [-6.705, -5.337]%; >=2% gate: **FAIL**.

Intervals use the frozen 5,000-sample paired round bootstrap of block medians.
These are small-sample, fixed-model/host/input results. Independent improvements
are never added, and percentages refer to step time, not a full training job.

## Accuracy and mechanism

Two actual CUDA preflight trajectory tests passed, without skips, before downloading
the real weights. Then each real-model arm completed two AdamW updates with four
accumulated microbatches. Initial adapters/IDs/top-k indices are exact, floating
values retain reference-relative `rtol=.01, atol=1e-5`, finiteness and three-decimal
loss gates. The controller compared full loss/logits/differentiable-input gradients,
router logits/weights and gradients, every LoRA gradient, final adapters and AdamW
moments. All 162 adapters actually changed and baseline gradients are nonzero.

All four original captures were reconstructed byte-for-byte and independently
compared again locally; every candidate is **bitwise** equal to ordinary. All
retained unprofiled loss sequences also match their paired ordinary controls.
This is short **same-NF4 numerical parity**, not BF16-vs-NF4 quality equivalence,
downstream task quality, convergence or full-model fine-tuning.

Accuracy-phase counter deltas, excluding setup probes:

| Arm | Logical expert visits | Expert file reads | RAM hits | GPU hits |
| --- | ---: | ---: | ---: | ---: |
| RAM/GPU caches | 16539 | 7768 | 8612 | 159 |
| Ordinary streaming | 16539 | 16539 | 0 | 0 |
| Two-read | 11026 | 11026 | 0 | 0 |
| Two-read + caches | 11026 | 7669 | 3256 | 101 |

The 3:2 visit ratio is exact and both cache tiers actually hit. Expert file counters
count tensor payload, not physical SSD traffic. Linux process read-byte counters
and direct-I/O status are retained separately; the container uses loopback-backed
storage and **physical NVMe backing is unverified**.

The mechanisms overlap: cache-only and combined already perform nearly the same
number of file reads. A RAM hit still transfers/dequantizes the expert; a miss
adds owned CPU clones and LRU eviction/allocation. This source inspection offers
a plausible overhead explanation, not a causal profile or a proven speed ceiling.
A static immutable RAM hotset is a prospective next A/B, **not implemented or
measured by this result**. Async/pinned transfer changes need separate lifetime tests.

## Retention, cost and cleanup

Complete source/model/spec hashes, raw timings, allocator/RSS/swap/I/O snapshots,
gate reports, logs and the exact input fixture are in the
[compact public record](gpu-results-evidence.zip), indexed by its internal `files.json`.
The [summary](gpu-results-summary.json) includes the original full-file hashes.
All large original result tensors and the self-contained lossless archive remain locally
retained; large real-model `capture.pt` files and base weights are not embedded in Git.
Base weights remain reproducibly identified by their public revision and full hashes;
they were removed with the rented disk rather than copied to the local result record.
The decoder verified every encoded chunk, reconstructed chunk and original file,
followed by independent outer-manifest SHA checks. Compression changes no tensor bytes.

The large-file SFTP collector slowed after its small transport control. An
authenticated four-connection SSH range fallback recovered the **same immutable
archive**, with exact range lengths and final whole-archive SHA, before decoding
and verifying all originals. Resources were removed only after that retention
proof; any interrupted original SFTP-controller trace is retained separately and
is not treated as a failed training/timing arm.

Conservative aggregate accounting, **including both earlier failed attempts**:
**$1.316854** from the $7.026460
remaining allowance; **$5.709607 remains**. This is an
upper accounting estimate with network/time reserves, not an invoice.

Earlier attempt A stopped before setup due to a controller-file freeze race;
attempt B stopped before installing/downloading the model because plain full-raw
retention could not fit the transport deadline. Both are retained and accounted for,
not included as timing arms. Immutable handoff and the existing lossless codec
were qualified before attempt C. The Moonlight full-rank-query target fix passed
81 local tests with one expected CUDA skip; two real CUDA controls later passed.

Every owned instance and persistent volume from all three attempts was deleted
and fresh exact-ID/label listings confirmed absence. Unrelated group resources
were untouched. Hosted CI was not awaited or counted as GPU verification.
