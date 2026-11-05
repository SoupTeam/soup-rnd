# E2 hybrid: all layers in epoch 1, top K from epoch 2

Protocol: `benchmarks/e2/hybrid/PROTOCOL.txt`, committed in `90adc1a2` before any hybrid arm ran.
Analysis: `analysis.json`, reproduced by
`python benchmarks/e2/hybrid/analyze_hybrid.py benchmarks/e2/hybrid/reports/20261105/evidence/20261105_082550`.
Evidence: the Kaggle result zip unpacked under `evidence/`, every file kept except three identical
copies of the base tokenizer; `MANIFEST.json` lists every SHA256, the zip and the kit
(`2b7badc4`, KIT.json SHA256 `427cd66d...a0c4`, which the notebook verified).

**Primary label (H24): WITHIN_MARGIN.** Training LoRA on all 32 layers in epoch 1 and only on the
top 24 from epoch 2 gives validation CE 0.8739 against 0.8758 for all-layer LoRA in the same
session (-0.21%); every general suite is within 0.05 (MMLU-mini -0.04, one question of 26; the others
equal). The same K with top-K training from step 1 cost +1.54% CE and -0.23 MMLU-mini
(`recipe_check/reports/20261104`).

**Secondary (H16): FAILS, on one suite.** CE 0.8738 (-0.22%), but MMLU-mini 0.92 against 1.00
(-0.077, two questions of 26) is outside the 0.05 margin. All other suites are equal to all-layer.

## Validity (all passed)

| Check | Result |
|---|---|
| V1 arms complete | H24, H16, W32: 4,000 steps, 2,000/200 rows, deterministic, 274 outputs each; re-scored = notebook |
| V2 anchor | CE(W32) 0.8757694, bit-identical to the recipe-check session's W32 and 0.12% from the record |
| V3 freeze | frozen tensors are exactly the q/v LoRA of layers 0-7 (H24: 32 tensors) and 0-15 (H16: 64); they changed in epoch 1, are bit-identical from step 2,000 to the end, requires_grad False; the top layers kept changing |
| V4 configs | every arm uses `recipe_check/configs/W32.yaml` with only base/output replaced |

No restart was needed (the notebook removes Kaggle's torchao before installing).

## Results (seed 42, Kaggle T4 x2, torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2)

| Arm | Schedule | Val CE | vs W32 | MMLU | Common sense | Instruction | Arithmetic | Tool call | JSON | Over-refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| W32 | all 32 layers, 2 epochs | 0.87577 | - | 1.00 | 0.96 | 1.00 | 0.63 | 0.70 | 1.00 | 0.98 |
| H24 | all 32, then top 24 | 0.87394 | **-0.21%** | 0.96 | 0.96 | 1.00 | 0.63 | 0.70 | 1.00 | 0.98 |
| H16 | all 32, then top 16 | 0.87380 | **-0.22%** | 0.92 | 0.96 | 1.00 | 0.63 | 0.70 | 1.00 | 0.98 |
| W24 (recipe check) | top 24 from step 1 | 0.88930 | +1.54% | 0.77 | 0.79 | 0.92 | 0.60 | 0.63 | 1.00 | 0.98 |

Epoch-1 training losses are identical across the three arms (same deterministic run up to the
freeze). mini_safety is 0/40 except H16 1/40 (base model) and is not in the rule. One seed, no
interval; the negative CE differences are not claimed as improvements.

## What this means for E2

- The quality cost of top-K came from never adapting the bottom layers, not from freezing them
  after one epoch. Freezing at the epoch boundary keeps all-layer quality at K=24 and, on CE, at K=16.
- This is the schedule the activation cache needs: the bottom layers are frozen from epoch 2, so
  their outputs can be cached once at the boundary and reused in every later epoch.
- The speed gain depends on K and is not measured here. Counting a backward as two forwards, a
  top-K step without the cache costs about L + 2K layer-forwards and with the cache about 3K, so
  the bound per later epoch is (L + 2K) / 3K: about 1.1x at K=24, 1.3x at K=16
  and 2x at K=8 (the measured warm epoch-2 ratio at K=8 was 1.84x, E2 close-out section 4). Epoch 1 runs at full cost, and the boundary adds one cache build.
  The useful operating point is therefore small K, where this check has only H16 (one suite out).

Next checks worth doing: K=8 and K=16 hybrid on more seeds or larger suites (the H16 MMLU miss is
two questions); end-to-end time of a hybrid run with the cache on over 3+ epochs; cache parity
across the freeze (not run here).
