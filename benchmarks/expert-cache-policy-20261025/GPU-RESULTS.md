# Expert-cache policy: physical 8 GB GPU results

Static RAM admission reduces paired step time **10.311%** versus two-read alone;
expanding the dense GPU hotset gives **14.026%** overall. Full numerical captures
match bitwise. These gains apply to the fixed MoE recipe below; short entries
remain slower because cache setup must be amortized.

Opt-in static host admission avoids repeated CPU cache replacement: admitted NF4 expert tensors stay immutable; later misses bypass a full host hotset without cloning. The existing LRU default is unchanged.

Executed source: [`70392afb78dd`](https://github.com/SoupTeam/soup-rnd/tree/70392afb78dde4098c73843a224aa8bab17d7df3). This is an experimental setup API and research harness. [CPU investigation](README.md), [machine-readable results](gpu-results-summary.json), [compact evidence](gpu-evidence.zip).

Moonlight-16B-A3B, frozen NF4 base / BF16 compute, LoRA rank 4 and alpha 8, sequence 512, batch 1, accumulation 2. Physical GPU: NVIDIA GeForce RTX 4060 Ti, 8188 MiB VRAM; native workload RAM ceiling 29.756 GiB, with swap unavailable. All candidates use the same frozen inputs/adapters and numerical gates.

## Complete paired GPU measurement

Four balanced rounds put every arm in each position once, using 16 fresh timing processes. Each entry has two warmup and six measured optimizer updates. Shared checkpoint download and NF4 sharding are excluded from whole-entry time.

| Arm | Median update, s | Median setup, s | Median whole entry (8 updates), s |
| --- | ---: | ---: | ---: |
| Two-read, no expert caches | 7.8848 | 7.175 | 75.290 |
| Two-read + LRU RAM / 0.25 GiB GPU | 8.2921 | 18.523 | 90.555 |
| Two-read + static RAM / 0.25 GiB GPU | 7.0679 | 17.379 | 79.227 |
| Two-read + static RAM / 1.5 GiB GPU | 6.7791 | 17.476 | 76.847 |

Recorded shared stages: checkpoint download 161.573 s; preparation 60.018 s, including NF4 sharding 21.999 s. These are outside the paired whole-entry table.

| Arm | Timed peak allocated VRAM, GiB | Timed peak reserved VRAM, GiB | Timed sampled peak RSS, GiB | Process lifetime peak RSS, GiB |
| --- | ---: | ---: | ---: | ---: |
| Two-read, no expert caches | 1.969 | 2.023 | 3.776 | 3.776 |
| Two-read + LRU RAM / 0.25 GiB GPU | 2.207 | 2.273 | 8.624 | 10.303 |
| Two-read + static RAM / 0.25 GiB GPU | 2.207 | 2.273 | 5.788 | 5.788 |
| Two-read + static RAM / 1.5 GiB GPU | 3.463 | 3.770 | 5.807 | 5.807 |

Sampled timed-stage RSS and the process lifetime high-water mark measure different periods. Allocated/reserved VRAM and cache payload are different quantities; RSS sampling can miss brief peaks. Both host policies have the same 2 GiB payload ceiling; observed payload maxima are retained in the JSON.

| Paired comparison | Step-time reduction | Paired 95% interval | Fixed ≥2% lower-bound gate |
| --- | ---: | ---: | --- |
| two_read_lru_vs_two_read | -5.111% | [-5.839%, -4.643%] | NO PRACTICAL WIN |
| two_read_static_vs_two_read | 10.311% | [9.932%, 10.741%] | PASS |
| two_read_static_large_gpu_vs_two_read | 14.026% | [13.162%, 14.622%] | PASS |
| static_vs_lru | 14.673% | [14.225%, 15.062%] | PASS |
| large_device_vs_static | 4.142% | [3.537%, 4.576%] | PASS |

Comparisons use paired per-round medians and the geometric mean ratio with 5,000 round-bootstrap samples. Combined savings are measured directly. These are fixed-recipe step/entry measurements, with descriptive setup amortization estimates in the JSON; they do not forecast a complete training run.

The complete eight-update entry remains slower: **79.227 s** with the small
static/GPU combination and **76.847 s** with the larger GPU hotset, versus
**75.290 s** without caches. Dividing the extra measured setup by steady-step
savings estimates break-even at **12.49** and **9.32 updates**, respectively.
These estimates exclude shared download/sharding and do not establish long-run
throughput or an automatic placement policy.

## Numerical controls

All four full accuracy captures have the same SHA-256. The unchanged reference-relative `rtol=0.01`, `atol=1e-5`, finite and three-decimal loss gates pass, with bitwise agreement of the captured inputs, initial adapters, logits/loss, decoder-input gradients, router values/gradients/top-k indices, all adapter gradients, updated adapters and AdamW state. Every one of 162 adapters is nonzero, receives a nonzero gradient and changes over two accumulated optimizer updates (four microbatches).

Complete capture SHA-256: `2b9f7411137e18f3990a30a920f2afd79d7c55c550c8abe64a0e5521688927c3`.

Four cheap native CUDA policy cells actually passed, with zero skips, before the checkpoint download. Every saved unprofiled warmup/timed last-microbatch loss was independently compared. The first microbatch loss in each timed accumulated update was not persisted: **NO VERDICT for that unrecorded value**. Complete numerical trajectories cover the separate accuracy phase. Frozen random token IDs test numerical/throughput parity; downstream quality and convergence were not evaluated.

## Why the host policy helps

The earlier LRU run is retained in the [original negative cohort](../two-read-expert-cache-20261025/GPU-RESULTS.md). Repeated expert visits can cause the LRU to clone/free complete eight-tensor NF4 entries even when the bounded payload remains stable. Static admission keeps the first entries that fit and bypasses later misses. Selection depends on cold visit order; warmup fills the hotset and this policy does not adapt to a changed routing distribution.

The Linux CPU control replays the same captured route three times in one fresh CPU-only process per policy, with the real expert shapes, a 2 GiB host ceiling and a simulated 0.25 GiB device filter. CPU cache-operation time excludes synthetic fill time and cannot be converted into GPU training speedup.

| CPU policy | Cache-operation time, s | Derived cloned payload, GiB | Process peak RSS, GiB |
| --- | ---: | ---: | ---: |
| lru | 5.02095 | 91.744 | 4.205 |
| static | 0.38649 | 1.999 | 2.485 |

After the isolated LRU CPU probe clears entries and runs GC, its RSS is 1.962 GiB; after diagnostic-only `malloc_trim(0)` it is 0.478 GiB. RSS fell after isolated post-run trim, consistent with allocator-retained pages; this does not identify all prior RSS as live/free tensors or prove a training leak. No allocator trim runs in GPU training. This CPU control does not classify all excess RAM in the GPU experiment.

## Retention and budget

All 159 original files (3,903,407,315 bytes) were restored and hash-verified from the lossless archive `09e8a08ab4edf03a071c5b9f7eaa5549d81e4bc67fb364b68ce8102ca96016eb` (610,652,160 encoded bytes). The compact packet includes JSON/XML/logs/text, frozen token input and small CUDA fixtures; large full captures remain locally retained and are hash-indexed. The independent audit consumed those full originals. Exact duplicate files share a ZIP member through `files.json`; every logical path is retained and its alias is byte-compared and hash-verified.

Owned instance `57707481` and volume `57707480` were deleted and fresh listings independently confirmed their absence. This stage conservatively cost **$0.540119**, leaving **$5.169488** from its $5.709607 cap. The earlier three owned attempts cost $1.316854 and were already deducted before this cap; they are counted once.

The result supports an opt-in two-read expert-cache policy for this measured configuration. Automatic placement, optimal use of all spare VRAM, verified physical NVMe performance, other models/routing distributions and long-run training quality remain unmeasured. Model/store identities are retained remote fingerprints cross-checked against public pinned LFS metadata; the large stores were not copied back for an independent local rehash.
