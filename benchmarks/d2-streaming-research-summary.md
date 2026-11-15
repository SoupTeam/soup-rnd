# Streaming memory research overview

The work uses spare GPU and host memory to reduce repeated weight reads and
transfers during streamed LoRA training. PR #5 adds optional persistent decoder
weights, a bounded SSD-tier RAM hotset, coordination between those caches, and
selection from measured placement profiles using setup cost and job length.
The latest profile also compares activation retention against recomputation.

The expert-streaming experiments build on Batyrkhan's
[two-read replay](https://github.com/MakazhanAlpamys/Soup/pull/2158).
The retained-activation path builds on Alpamys's
[stream recompute work](https://github.com/MakazhanAlpamys/Soup/pull/2152).
The cache work combines these mechanisms, preserves their numerical gates and
measures their effect together. Companion expert kernels and transfer prototypes
remain distinct from the production features included in PR #5.

| Experiment | Measured result | Memory and scope |
|---|---|---|
| [Persistent weights, dense 7B BF16](streaming-residency-20261020.md) | Step time reduced 15.83–31.25% at batch 1 and 6.04–11.60% at batch 2 | RTX 4090 with a 10 GiB allocator cap. Eight/sixteen resident layers add about 3/6 GiB VRAM. |
| [Physical 8 GiB GPU, dense 8B BF16](stream8-compile-20261022-resume/RESULTS.md) | Four resident layers reduced step time by 2.39% | RTX 3060 Ti. Allocated VRAM 4.14 → 5.58 GiB. This is a separate workload from the 7B result. |
| [Coordinated caches, dense 32B BF16](joint-cache-20261024/README.md) | 71.72 → 58.33 s/update, a paired 18.69% reduction | RAM cache supplies most of the gain. Allocated VRAM 5.03 → 5.94 GiB, RSS 7.91 → 21.97 GiB. |
| [Coordinated caches, dense 32B NF4](nf4-ssd-32b-20261024/README.md) | 9.26 → 8.60 s/update, a paired 7.15% reduction | Physical RTX 4060 Ti 8 GB, native RAM bound 29.76 GiB. VRAM 3.64 → 5.41 GiB, RSS 5.83 → 19.90 GiB. |
| [Two-read, caches and grouped routes, MoE 30B](expert-streaming-final-20261026/README.md) | 27.80 → 15.99 s/update, a paired 42.47% reduction. Whole eight-update entry reduced 28.04%. | Direct ordinary-versus-complete comparison. VRAM 1.66 → 3.64 GiB, RSS 3.86 → 12.05 GiB. |
| [Measured placement profile, dense 32B NF4](d2-stream-cache-profile-20261104.md) | Whole 18-update job 188.30 → 180.16 s, a paired 4.47% reduction | Auto chose ordinary streaming because cache setup did not amortize. VRAM 5.68 → 4.04 GiB, RSS 19.50 → 5.42 GiB. |
| [Profile and retention, synthetic 329M NF4 on Kaggle](stream-cache-profile-kaggle-20261107/README.md) | Whole eight-update job 12.33 → 9.88 s, 19.9% shorter. Whole 64-update job 78.38 → 59.30 s, 24.3% shorter. | T4 under a software 8 GiB allocator cap. Auto selected the same weight caches plus activation retention for all 24 layers. VRAM 219 → 577 MiB. |

Short jobs expose setup costs. In the dense 32B BF16 study, the combined caches
were 2% slower over the complete six-update entry despite faster steady steps.
The corresponding NF4 entry grew from 82.52 to 94.15 seconds. The profile feature
uses this tradeoff when choosing among measured configurations. It retains the
fresh memory admission check and the configured RAM/GPU ceilings.

The Kaggle follow-up corrects the reuse cost estimate by subtracting measured
planning probes that run only while recording. Mandatory candidate-memory probes
remain charged. The measured deduction for its joint configuration was 1.754 s.
Both formulas selected the same configuration for the measured eight- and
64-update jobs. Their measured speed difference comes from retaining activations.

Additional expert-transfer research measured improvements over successively
optimized baselines. [Packed NF4 transfers](expert-prefetch-20261028/README.md)
reduced step time by another 17.75% (16.53 → 13.62 s).
[Prepacked host entries](expert-followup-20261029/README.md) reduced the next
baseline by 3.43% (13.53 → 13.08 s), with additional process RAM.
These experiments use separate cohorts. The directly measured total against
ordinary streaming remains the 42.47% MoE comparison above. A total for all later
prototypes together requires its own ordinary-versus-complete measurement.

The studies also retained negative results. External FlashAttention with resident
layers produced no practical win in the
[tested configurations](streaming-residency-flashattention-20261020.md).
Both compiled Inductor variants in the physical-8-GiB study failed the original
numerical gates and were excluded from timing. Earlier dynamic expert-placement
trials supplied useful correctness evidence but no confirmed extra speed gain.

Accuracy gates precede timing and preserve the original reference-relative
tolerances. Successful captures compare losses, gradients, adapter updates and
optimizer state, with exact agreement in the cited qualifying trajectories.
The Kaggle study records 26,912 exact checks before its timing comparisons.
The experiments cover controlled, short LoRA trajectories. Real-data quality,
convergence and universal performance across models remain separate evaluations.

The resulting feature makes available memory useful when its measured savings
outweigh preparation, while allowing smaller placements for short jobs. Each
linked report retains its own hardware, precision, inputs, baseline, timing
scope, memory measurements and evidence limits.
