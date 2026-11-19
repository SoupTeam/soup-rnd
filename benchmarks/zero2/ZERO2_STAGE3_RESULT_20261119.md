# ZERO-2 stage 3 result: which Qwen3.5-9B layers to train (run 2026-11-19, Vast RTX 4090, Texas)

Protocol: `ZERO2_STAGE3_PROTOCOL_20261119.md`. Runner sha256 `a3d8ae92…59f4` (matches), Soup `b3bca25c`.
Runner wall time 7.6 h; instance 61433243 destroyed after the tgz was copied (~8 h rented, about $3.7).
Every layer-select arm recorded `backward_cut: cut` (the resident backward stopped at the lowest chosen layer).

## Decisions (as fixed)

### Speed and fit (part S, 24 steps, GSM8K rows)

| Block | A s/step | B s/step | Speed-up A/B | Verdict |
| --- | --- | --- | --- | --- |
| r: no cap, all 32 vs top 8 | 2.133, 2.144 | 0.995, 0.977 | **2.17x** | **FASTER** |
| s: 8 GiB stream, all vs top 8 | 2.899, 2.615 | 1.825, 1.641 | **1.59x** | **FASTER** |
| q: 4 GiB stream + L2L, all vs top 8 | 4.121, 4.000 | 2.246, 2.245 | **1.81x** | **FASTER** |
| l: no cap, without vs with `use_liger` | 2.125, 2.159 | 2.988, 2.863 | 0.73x | **SLOWER** |

`fits_8gib_plain_host`: **true**. h8 (8 GiB, plain, `kbit_keep_16bit` + `embed_on_host`) ran 24 steps,
2.28 s/step, torch peak 7.19 GiB, 1.89 GiB of embedding on the host, no host warning.

### Quality (part Q, 300 steps, k 8; delta vs full, verdict at 2 se)

| Dataset (metric, n) | base | full | top8 | random | grad | angular | lisa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K (strict EM, 200) | 0.800 | **0.870** | 0.620 −25.0 **WORSE** | 0.810 −6.0 SAME | 0.840 −3.0 SAME | 0.785 −8.5 **WORSE** | 0.805 −6.5 SAME |
| ARC-Challenge (acc_norm, 1172) | 0.555 | **0.672** | 0.567 −10.5 **WORSE** | 0.634 −3.8 SAME | 0.610 −6.2 **WORSE** | 0.586 −8.6 **WORSE** | 0.613 −6.0 **WORSE** |
| HumanEval (pass@1, 164) | 0.640 | 0.598 | 0.573 −2.4 SAME | 0.616 +1.8 SAME | 0.585 −1.2 SAME | 0.591 −0.6 SAME | 0.561 −3.7 SAME |
| PubMedQA | — | — | — | — | — | — | — |

Deltas in points. SAME means no difference this run can resolve (one seed; HumanEval's se ~5.5
points resolves almost nothing).

**PubMedQA: no result.** All 7 evals failed with CUDA OOM in lm-eval's `log_softmax` (batch 16 x
long abstracts x 248k vocabulary asked 4.29 GiB with 20 GiB already held). Training of all 6 arms
was OK. The adapters were deleted after their eval, so a fix (smaller eval batch) needs a retrain;
not rerun under this protocol.

## Layers each method chose (same on every dataset unless noted)

| Method | Layers | Lowest | Step s/step (part Q, GSM8K) |
| --- | --- | --- | --- |
| full | 0–31 | 0 | 2.16 |
| top8 | 24–31 | 24 | 0.98 |
| random (seed 0) | 1, 8, 13, 15, 16, 24, 28, 31 | 1 | 2.02 |
| grad | **0–7 on all four datasets** | 0 | 2.13 |
| angular | 0 + mostly 18–31 (e.g. GSM8K 0, 18, 19, 22, 23, 27, 30, 31) | 0 | 2.15 |
| lisa | 8 random, re-drawn every 20 steps (16 draws) | varies | 1.95 |

## Reading (not gated)

1. **The trade-off is sharp: what makes training fast hurts quality.** The speed-up comes only
   from not running the backward below the lowest trained layer, so only top-k methods are fast
   (top8: 2.2x resident, 1.6x at 8 GiB, 1.8x at 4 GiB). But top8 lost 25 points on GSM8K and 10.5
   on ARC. The methods that keep quality (random, grad, lisa on GSM8K; random on ARC) all train a
   low layer and run at ~1.0–1.1x.
2. **The gradient picks the bottom layers, every time.** grad chose layers 0–7 on all four
   datasets, the opposite of the "train the last layers" intuition; on GSM8K it was the best k-8
   method (−3, SAME). Lower layers carry the largest LoRA gradients on this model; it is not a
   dataset-specific signal at k 8. angular always includes layer 0 too.
3. **Fine-tuning helped where the task matches the data** (GSM8K +7, ARC +11.7 over base) and
   did not on code (CodeAlpaca → HumanEval: full 0.598 < base 0.640, within noise): a code SFT
   set unlike HumanEval's style. The code row says little about layer methods.
4. **Liger is slower here** (0.73x) on Qwen3.5 NF4 QLoRA, with torch peak 8.79 → 7.74 GiB. Not
   turned on for Qwen3.5 recipes.
5. **Plain 8 GiB now works** with `kbit_keep_16bit` + `embed_on_host` (2.28 s/step vs 2.6–2.9 for
   streaming at 8 GiB in the same session; descriptive, different configurations).

## What this means for the lead's question

- "Train only a couple of layers" works for speed only if they are the top layers, and on GSM8K /
  ARC that costs 10–25 points at k 8 out of 32. A random or gradient-picked 8 keeps quality
  (within 3–6 points, SAME on GSM8K) but saves almost no time on this model.
- The useful middle is still unmeasured: top-k with larger k (16 → cut at layer 16, expected
  ~1.4–1.5x) and "top-k plus a few low layers". That is stage 4, together with PubMedQA (eval
  batch fixed) and more seeds for the SAME verdicts.

## For stage 4

- k ∈ {12, 16, 24} for top-k; a hybrid (top-k + grad's lowest layers); 2–3 seeds on GSM8K and ARC.
- PubMedQA with eval batch 2 (or the chunked log-softmax in lm-eval off); retrain its arms.
- Code: an SFT set closer to HumanEval (or MBPP) if code stays in the matrix.
- Keep `kbit_keep_16bit`; Liger off; Triton NF4 off.
