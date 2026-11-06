# E2 hybrid over three seeds: K = 24, 16 and 8 all within the margin

Protocol: `benchmarks/e2/hybrid_seeds/PROTOCOL.txt`, committed in `cc414064` before any arm of this
series ran. Analysis: `analysis.json`, reproduced by
`python benchmarks/e2/hybrid_seeds/analyze_seeds.py benchmarks/e2/hybrid_seeds/reports/20261106/evidence/20261105_201228 benchmarks/e2/hybrid_seeds/reports/20261106/evidence/20261105_230802 benchmarks/e2/hybrid_seeds/reports/20261106/evidence/20261106_020313`
(seed-42 W32/H24/H16 are read from `benchmarks/e2/hybrid/reports/20261105`). Evidence: the three
Kaggle zips unpacked under `evidence/`, every file kept except identical base-tokenizer copies;
`MANIFEST.json` has every SHA256 and the zips'.

**Result.** Training LoRA on all 32 layers in epoch 1 and only on the top K from epoch 2 stays within
the E2 quality margin at every K tested, over seeds 42, 901 and 1234. The smallest such K, and the
one this series recommends for the speed check with the cache on, is **K = 8** (K/L = 1/4).

| K | Label | Mean CE gap | Gap per seed (42 / 901 / 1234) | Mean MMLU-mini delta | Mean tool-call delta |
|---|---|---|---|---|---|
| 24 | WITHIN_MARGIN | -0.09% | -0.21 / -0.08 / +0.02 | +0.01 | +0.05 |
| 16 | WITHIN_MARGIN | -0.05% | -0.22 / +0.01 / +0.08 | -0.01 | +0.03 |
| 8 | WITHIN_MARGIN | +0.00% | -0.21 / +0.03 / +0.18 | -0.04 | +0.03 |

Common sense, instruction, arithmetic and JSON deltas are 0.00 to -0.01 on average at every K;
over-refusal is 0.975-1.00 in every arm. For comparison, top-K training from step 1 at K=24 cost
+1.54% CE and -0.23 MMLU-mini (`recipe_check/reports/20261104`). The K=16 MMLU-mini miss in the
one-seed check (two questions) does not repeat in the other seeds; at K=8 the mean MMLU-mini delta,
-0.04 (about one question of 26), is the closest any quantity comes to its margin.

## Validity (all passed)

| Check | Result |
|---|---|
| V1 arms complete | 10 new arms: 4,000 steps, 2,000/200 rows, deterministic; 274 outputs per gated arm; re-scored = notebook |
| V2 anchor | ANCHOR_s42 (W32 through the extended runner) CE 0.8757693767547607, bit-identical to the 20261105 W32 |
| V3 freeze | in every H arm the frozen tensors are exactly the q/v LoRA of layers 0..31-K, changed in epoch 1, bit-identical after step 2,000 |
| V4 configs | W32 template apart from base, output, seed and data_seed; seeds match the arm |
| V5 versions | torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2 in all four sessions; kit `cc414064` |

No arm failed or was restarted.

## Training time without the cache (descriptive, not a speed result)

Median epoch-2 time over all runs of each schedule (W32: 4 runs; H: 3 each), Kaggle T4 x2:
W32 547 s, H24 486 s, H16 414 s, H8 336 s; epoch 1 is 549-553 s for all. The epoch-2 ratios
against W32 (1.13x, 1.32x, 1.63x) come from not back-propagating into the frozen bottom; the cache
would also remove its forward pass (bound L/K against all-layer training: 1.33x, 2x, 4x). These
times are from single runs on a shared host, not from an A-B-B-A series, and are not a claim.

## What this means for E2

- The quality cost of top-K is removed by training all layers for one epoch before freezing the
  bottom: at K/L = 1/4 the hybrid matches all-layer LoRA on CE and the suites, over three seeds.
- Next, per protocol section 1: a speed check with the activation cache on at K = 8, in A-B-B-A
  fresh processes (all-layer W32 against hybrid H8 with the cache), with cache parity across the
  freeze, under its own prospective rule.
- Scope of this result: Mistral-7B, Alpaca 2,000 rows, sequence 128, two epochs, LoRA r4 on q/v,
  three seeds, small suites (24-40 items). Not shown: other models (including MoE), data, recipes,
  more epochs.
