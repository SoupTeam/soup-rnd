# Kanana 30B-A3B: two-read and bounded RAM/GPU expert caches

Measured source: [`97404397dd38`](https://github.com/SoupTeam/soup-rnd/tree/97404397dd38487f6daf44ebd26bee81fdaf132c). [Preparation and CPU controls](README.md), [complete summary](gpu-results-summary.json), [compact evidence](gpu-evidence.zip).

Pinned model: `kakaocorp/kanana-2-30b-a3b-base-2601` at `0d31365acf656e5f9a593cacc4913e9c0d36cb1c`. This is a sparse MoE with 30,670,809,088 parameters, plus 6,016 frozen router-bias buffer values; it is not a dense 32B-model result.

Frozen NF4 base weights, BF16 compute, FP32 LoRA adapters, rank 4 / alpha 8 on `q_proj`, `kv_b_proj`, `o_proj`; sequence 512, batch 1, accumulation 2. Token IDs and initial adapters are frozen across arms. The numerical inputs are random tokens.

Physical GPU: NVIDIA GeForce RTX 4060 Ti, 8188 MiB in nvidia-smi; Torch reports 7.625 GiB. The native workload RAM ceiling is 29.756 GiB; workload swap is disabled. The RAM ceiling is checked before checkpoint download.

## What was compared

| Arm | Expert loading schedule | Cache payload ceilings |
| --- | --- | --- |
| Ordinary | Original three-read schedule | No expert cache |
| Two-read only | Two-read schedule | No expert cache |
| Caches only | Original three-read schedule | Static 8 GiB host NF4 + 2 GiB GPU BF16 |
| Both | Two-read schedule | Static 8 GiB host NF4 + 2 GiB GPU BF16 |

These cache ceilings are bounded payload limits, not promised occupancy or the total process memory. Static host admission keeps the first complete immutable expert entries that fit, then bypasses later misses without replacement cloning. The GPU cache admits the first complete dense entries that fit, then bypasses later misses without capacity eviction. Selection depends on cold visit order; it does not automatically optimize tensor placement. This is an opt-in setup API and companion benchmark. Existing CLI/default host policy remains LRU.

## Complete paired GPU measurement

Four balanced Latin rounds place every arm in each position once: 16 fresh timing processes, two warmup plus six measured optimizer updates each (32 warmup and 96 measured updates total). Whole entry is the controller's fresh-process wall timer including interpreter/import bootstrap, setup/admission, all eight updates and result validation. Shared checkpoint download and NF4 sharding are excluded. A fresh worker does not imply cold physical storage.

| Arm | Median update, s | Median setup, s | Median whole entry (8 updates), s |
| --- | ---: | ---: | ---: |
| Ordinary streaming (three-read) | 27.6969 | 12.664 | 250.310 |
| Two-read only | 20.5802 | 12.207 | 190.295 |
| RAM/GPU caches (static host) | 22.2806 | 47.258 | 241.816 |
| Two-read + RAM/GPU caches | 17.0114 | 40.032 | 189.532 |

Recorded shared stages: checkpoint download 313.289 s; preparation 114.355 s, including sharding 47.282 s.

| Arm | Timed peak allocated VRAM, GiB | Timed peak reserved VRAM, GiB | Timed sampled peak RSS, GiB | Process lifetime peak RSS, GiB |
| --- | ---: | ---: | ---: | ---: |
| Ordinary streaming (three-read) | 1.658 | 1.971 | 3.850 | 3.850 |
| Two-read only | 1.660 | 1.953 | 3.849 | 3.849 |
| RAM/GPU caches (static host) | 3.641 | 3.854 | 12.036 | 12.036 |
| Two-read + RAM/GPU caches | 3.643 | 3.838 | 12.029 | 12.029 |

Timed-stage sampled RSS and lifetime RSS high-water mark cover different periods. Allocated/reserved VRAM and expert-cache payload also measure different quantities. RSS sampling can miss brief peaks. Process RSS omits additional kernel/filesystem page-cache and other cgroup memory charged under the native 29.756 GiB cap, so these results do not establish fit on a 12 or 16 GiB host.

| Arm | Maximum observed host payload, GiB | Maximum observed GPU payload, GiB |
| --- | ---: | ---: |
| Ordinary streaming (three-read) | 0.000 | 0.000 |
| Two-read only | 0.000 | 0.000 |
| RAM/GPU caches (static host) | 7.998 | 1.995 |
| Two-read + RAM/GPU caches | 7.998 | 1.995 |

| Candidate versus reference | Step-time reduction | Paired 95% interval | Fixed ≥2% lower-bound gate |
| --- | ---: | ---: | --- |
| Two-read only versus Ordinary streaming (three-read) | 25.753% | [25.160%, 26.342%] | PASS |
| RAM/GPU caches (static host) versus Ordinary streaming (three-read) | 19.668% | [18.637%, 20.750%] | PASS |
| Two-read + RAM/GPU caches versus Ordinary streaming (three-read) | 38.789% | [38.273%, 39.300%] | PASS |
| Two-read + RAM/GPU caches versus Two-read only | 17.557% | [16.586%, 18.512%] | PASS |
| Two-read + RAM/GPU caches versus RAM/GPU caches (static host) | 23.802% | [23.020%, 24.542%] | PASS |

Comparisons use paired per-round medians, the geometric mean ratio, and 5,000 round-bootstrap samples. The fixed practical gate requires the interval lower bound to reach 2%. Combined effects and incremental gains are measured in the same cohort; percentages are not added.

| Candidate versus reference | Extra median setup, s | Median seconds saved per update | Descriptive updates to amortize extra setup |
| --- | ---: | ---: | ---: |
| Two-read only versus Ordinary streaming (three-read) | -0.457 | 7.1167 | 0.00 |
| RAM/GPU caches (static host) versus Ordinary streaming (three-read) | 34.593 | 5.4163 | 6.39 |
| Two-read + RAM/GPU caches versus Ordinary streaming (three-read) | 27.368 | 10.6855 | 2.56 |
| Two-read + RAM/GPU caches versus Two-read only | 27.825 | 3.5688 | 7.80 |
| Two-read + RAM/GPU caches versus RAM/GPU caches (static host) | -7.225 | 5.2692 | 0.00 |

Amortization divides positive extra median setup by median seconds saved per update. A zero means no extra median setup; no positive saving has no amortization estimate. These are descriptive estimates, not confidence bounds or forecasts of full training. The whole-entry table reports the actual measured eight-update cost.

## Numerical controls and scope

Every candidate passed independently against ordinary streaming using the unchanged reference-relative `rtol=0.01`, `atol=1e-5`, finite-value and three-decimal-loss gates. Input IDs, initial adapters and integer expert-routing indices require exact equality. Full captures include loss/logits, decoder-input gradients, router values/gradients, adapter gradients, updated adapters and AdamW state. All 288 adapter tensors were independently verified nonzero, receiving a nonzero gradient, and changing over two accumulated optimizer updates (four microbatches). This compares optimizations against ordinary NF4 streaming; it does not compare NF4 model quality against an unquantized BF16 baseline.

All four full accuracy trajectories also agree bitwise on the captured tensors and scalars. Bitwise agreement is reported as additional evidence; the numerical tolerances were not changed.

All full accuracy files share SHA-256 `e00fa8f90bc33098158ae33180f78d7158d247c7ebe9ab27f0446be353b08d01`.

Exactly three cheap CUDA cases executed successfully with zero skips before the model download: the Kanana four-arm NF4/BF16 trajectory and the existing LRU/static NF4 trajectory controls.

Every recorded unprofiled warmup/timing last-microbatch loss was independently compared. The first microbatch loss in each accumulated timed update was not persisted: **NO VERDICT for that unrecorded value**. Full numerical trajectories cover the separate accuracy phase, not every value in all timed updates. Random-token controls establish short numerical parity and throughput; downstream task quality, convergence and full-model tuning were not evaluated.

## Model/store evidence and limits

The byte-exact executed model config and actual NF4 store index are included. Their SHA-256 values are `a284853f01be28fb1ca71b25b36f49e3eb4a0737e98105bb3350e501b51a6ea8` and `01b57f21365f9dc6ce88abc67ecfc7e2fa2cf4976556a93c8851b9264c1d83f8`. The audited store has 6,016 expert entries: 2,434,184 packed NF4 bytes per expert (including sidecars), compared with 9,437,184 dense BF16 bytes. Payload-only capacity is at most 3,528 host entries and 227 GPU entries at these ceilings; other allocations and admission checks still apply.

The visible data volume is a container loopback mount. Physical NVMe backing is unverified. Expert file-payload reads and cache counters must not be interpreted as physical SSD traffic. The 13 original BF16 checkpoint files passed the pinned public LFS SHA checks. Generated NF4 store fingerprints were checked against the prepared experiment spec; these large checkpoint/store files were not transferred back for an independent local rehash. The actual config/index originals were retained and verified locally.

The [earlier 16B policy experiment](https://github.com/SoupTeam/soup-rnd/blob/4bfccb15a3eb646acf0118b353386b2773c83138/benchmarks/expert-cache-policy-20261025/GPU-RESULTS.md) is a separate model and cache-capacity cohort. Its performance percentages are not mixed into these comparisons. No CPU allocator-trim diagnostic or active allocator trim is part of this Kanana training run.

## Retention, cleanup and budget

All 130 original files (4,695,959,402 bytes) were restored and hash-verified from lossless archive `f5e488dfc689cf909e2dad4dcb8bb855784a21877ad5d7f954cb6209ce82deff` (626,247,680 encoded bytes). The independent audit consumed those full originals. Large accuracy captures remain locally retained and hash-indexed; the compact packet includes all allowed JSON/XML/log/text, frozen token inputs and small CUDA fixtures. Exact duplicate files share ZIP members through `files.json`; every allowed logical path remains indexed, byte-compared and hash-verified.

Owned instance `57717449` and volume `57717448` were deleted and independently confirmed absent by fresh listings. This stage conservatively cost **$0.830547**, leaving **$4.338941** from its $5.169488 cap. Previous stage D cost $0.540119; the older three attempts cost $1.316854. Both were already deducted before the Kanana cap and are counted once, not charged again.

This cohort measures one fixed MoE model, recipe and static cache allocation. Automatic placement, optimal use of all free VRAM, other routing distributions, dense-model scaling, physical NVMe performance and long-run training quality remain unmeasured.
