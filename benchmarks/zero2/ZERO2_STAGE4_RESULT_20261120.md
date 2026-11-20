# ZERO-2 stage 4 result: the trade-off map around full LoRA (run 2026-11-20, Vast RTX 4090, Texas)

Protocol: `ZERO2_STAGE4_PROTOCOL_20261120.md`. Runner sha256 `7762245a…d9ec` (matches), Soup `b3bca25c`.
Same host as stage 3. Runner wall time 9.9 h, every arm OK (PubMedQA's eval now ran at batch 2).
About $4.6 rented.

## Decisions (as fixed)

### Speed (A-B-B-A against full in the same setup, 24 steps)

| Variant | Setup | full s/step | variant s/step | Speed-up | Verdict |
| --- | --- | --- | --- | --- | --- |
| top24 | no cap | 2.30, 2.13 | 1.73, 1.73 | **1.28x** | FASTER |
| top16 | no cap | 2.15, 2.12 | 1.34, 1.34 | **1.59x** | FASTER |
| top12 | no cap | 2.14, 2.14 | 1.16, 1.15 | **1.86x** | FASTER |
| top16 | 8 GiB stream | 3.16, 3.11 | 2.40, 2.35 | **1.32x** | FASTER |
| top16 | 4 GiB L2L | 4.42, 4.47 | 3.25, 3.12 | **1.40x** | FASTER |
| r32 | no cap | 2.15, 2.17 | 2.09, 2.12 | 1.03x | FASTER (by ~0.05 s) |
| imp | no cap | 2.16, 2.16 | 2.18, 2.17 | 0.99x | SLOWER (by ~0.01 s) |
| dora | no cap | 2.16, 2.13 | 3.16, 3.18 | **0.68x** | SLOWER |

### Quality (delta vs full in points; gsm8k and arc: mean of seeds 0 and 1, pubmedqa: seed 0)

| Variant | GSM8K (full 0.845) | ARC (full 0.669) | PubMedQA (full 0.797) |
| --- | --- | --- | --- |
| top24 | −1.8 SAME (−4.0, +0.5) | −1.8 SAME (−3.2, −0.3) | −0.5 SAME |
| top16 | **−9.5 WORSE** (−10.5, −8.5) | **−4.0 WORSE** (−5.1, −2.9) | −3.0 SAME |
| top12 | **−13.5 WORSE** | **−10.8 WORSE** | **−4.3 WORSE** |
| r32 | −0.5 SAME | +0.8 SAME | −1.8 SAME |
| imp | −1.5 SAME | +0.8 SAME | −0.2 SAME |
| dora | −1.0 SAME | −0.6 SAME | 0.0 SAME |

Base model: GSM8K 0.800, ARC 0.555, PubMedQA 0.669 (full LoRA adds +4.5, +11.4, +12.8 points).

## The map

| Point | Speed | Quality vs full | Reading |
| --- | --- | --- | --- |
| top12 | 1.9x | −4 to −14 | fast, clear loss everywhere |
| top16 | 1.6x (1.3x at 8 GiB, 1.4x at 4 GiB) | −3 to −10 (WORSE on GSM8K, ARC) | dataset-dependent: fine for PubMedQA-like classification, not for reasoning |
| **top24** | **1.3x** | **−0.5 to −1.8, SAME on all three** | **the safe speed-up** |
| full | 1.0x | reference | |
| r32, imp | ~1.0x | −1.8 to +0.8, SAME | no gain at this budget |
| dora | 0.7x | −1 to 0, SAME | slower, no gain |

## Reading (not gated)

1. **Faster side: top24 is the free lunch.** Training the top 24 of 32 layers (skipping the
   bottom 8, which the backward then never reaches) is 1.28x faster with no loss this run can see
   on any dataset. top16 doubles the saving but costs 4–10 points on reasoning (GSM8K, ARC) and is
   borderline on PubMedQA; top12 loses everywhere. Loss grows faster than the speed-up below 24.
2. **Slower side: nothing beats full at 300 steps.** More capacity (r 32, rank 64 on layers 0–7)
   or DoRA lands within ±2 points of full on every dataset; DoRA costs 32 % more time for it. The
   "better than the reference, a bit slower" point does not exist at this training length; full
   LoRA r 16 is not capacity-bound here. (Longer training or harder data could change this; not
   tested.)
3. **Speed-ups shrink under memory budgets** (top16: 1.59x resident, 1.32x at 8 GiB, 1.40x at
   4 GiB), because streaming adds per-layer work the backward cut does not remove.
4. r32 "FASTER" and imp "SLOWER" are differences of 0.01–0.05 s/step: real by the rule, too small
   to matter.

## What to recommend in Soup (Qwen3.5-9B QLoRA)

- Default: full LoRA r 16 with `kbit_keep_16bit`.
- "Fast" option: `lora_layers: {method: fixed, layers: [8..31]}` (top 24): ~1.3x, no measurable loss.
- "Fastest" option: top 16, ~1.6x, with a warning: −4 to −10 points on reasoning tasks.
- Not recommended on this model: DoRA (0.7x, no gain), larger r (no gain at 300 steps).

## Not tested

Other k (20, 28), longer training, other learning rates, other models, more seeds on PubMedQA.
