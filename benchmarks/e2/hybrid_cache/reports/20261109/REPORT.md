# E2 hybrid K=8 with the activation cache: bit-identical to no cache, 3.2x faster in the warm epoch

Protocol: `benchmarks/e2/hybrid_cache/PROTOCOL.txt`, committed in `c16c7e45` before any run of this
series; addendum `ADDENDUM_20261109.txt` (`7f5f8b78`). Analysis: `analysis.json`, reproduced by
`python benchmarks/e2/hybrid_cache/analyze_speed.py benchmarks/e2/hybrid_cache/reports/20261109/evidence/20261109_070805`.
Evidence: the Kaggle zip unpacked under `evidence/` (identical base-tokenizer copies omitted);
`MANIFEST.json` has every SHA256 and the zip's. One Kaggle session (Save & Run All, T4 x2),
13 runs in protocol order, none failed or restarted in this series.

**Result: WARM_SPEEDUP** (primary) and **TRAINING_SPEEDUP** (secondary). Hybrid K=8 with the
cache against all-layer LoRA, fresh processes, A-B-B-A, three cycles:

| Cycle | Epoch 3 (warm) A/B | Whole training A/B | Epoch 2 A/B |
|---|---|---|---|
| 1 | 3.20x | 1.46x | 1.42x |
| 2 | 3.27x | 1.52x | 1.49x |
| 3 | 3.09x | 1.46x | 1.41x |
| median | **3.20x** (rule: >= 1.10, each >= 1.00) | **1.46x** | 1.42x |

Epoch step time: A about 535-560 s in every epoch; B 534-569 s in epoch 1 (all layers, same as
A), 365-389 s in epoch 2 (build the cache while training), 163-176 s in epoch 3 (read it).
Cache install at step 2,000 (fingerprinting the 24 frozen layers): 33-35 s, inside the timed
first step of epoch 2. Cache: 2,000 files, 1.58 GB per run.

## Validity (all passed)

| Check | Result |
|---|---|
| V1 | every run 6,000 steps, finite CE, 2,000/200 rows, deterministic |
| V2 parity | all six B runs equal C bit for bit in final LoRA fingerprint, training-batch hash and validation CE (0.8775755167007446); all six A runs equal each other (0.8966562151908875) |
| V3 cache use | installed at step 2,000 over layers 0-23; epoch 2: 0 hits + 2,000 misses; epoch 3: 2,000 hits, 0 misses; 2,000 files = misses, in every B run |
| V4 hybrid | frozen layers 0-23 in B and C, bottom bit-identical after the freeze |
| V5 configs | W32 template apart from base and output |

## Descriptive, not decided

Validation CE after three epochs: hybrid K=8 0.8776, all-layer 0.8967 (the hybrid is lower). This
series does not claim quality (settled over seeds in `hybrid_seeds`, at two epochs). Process wall
time ratio median 1.44x (includes model load). GPU temperature and clocks before and after each run
are in `arm.json`; no run was voided on them (Kaggle is a shared host).
The first session of this series (2026-11-08) lost its archive and was rerun unchanged, see the
addendum; its process-time ratios (1.45, 1.48, 1.50) agree with this run's (1.44, 1.50, 1.44).

## What this means for E2

- The frozen-prefix cache can be installed in the middle of a run, at the freeze, without changing
  the result by a single bit.
- With three epochs the hybrid at K=8 trains 1.46x faster over the whole run and 3.2x faster in an
  epoch whose cache is already built, on two T4s with a disk-backed cache. In a short run the cache
  build (epoch 2) is the cost: with two epochs the earlier preflight showed 1.10x at K=8.
- Scope: Mistral-7B, Alpaca 2,000 rows, sequence 128, batch 1, LoRA r4 on q/v, K=8, seed 42, T4 x2.
  Not shown: other K (the 2-epoch preflight was slower at K=16 and K=24), models, MoE, data, GPUs.
