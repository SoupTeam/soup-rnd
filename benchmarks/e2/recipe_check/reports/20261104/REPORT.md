# E2 recipe check: does more adapter capacity close the top-K gap?

Protocol: `benchmarks/e2/recipe_check/PROTOCOL.txt`, committed in `9e5846cf` before any arm ran.
Analysis: `analysis.json`, reproduced by
`python benchmarks/e2/recipe_check/analyze_recipe.py benchmarks/e2/recipe_check/reports/20261104/evidence/20261104_182324 benchmarks/e2/recipe_check/reports/20261104/evidence/20261104_231206`.
Evidence: both Kaggle result zips unpacked under `evidence/`, every file kept except four identical
copies of the base tokenizer; `MANIFEST.json` lists the SHA256 of every file, the excluded ones and
both zips.

**Label: GAP_PERSISTS.** With LoRA r=16 on all seven linear modules, top-24 training is 2.31% worse
in validation CE than all-layer training, against 1.54% with the recorded r=4 q/v recipe. More
adapter capacity, with the data and schedule unchanged, does not close the gap.

**Caveat that limits the reading.** The strong recipe overfits this data: its epoch-2 training loss
falls to about 0.45-0.50 while its validation CE (0.983 at K=32) is worse than the weak recipe's
(0.876). The comparison therefore tests "more capacity on 2,000 rows for 2 epochs", not "a better
recipe". Whether top-K would hold up with a recipe that has both more capacity and enough data not
to memorise is not answered here.

## Validity (all passed)

| Check | Result |
|---|---|
| V1 arms complete | W32, W24, S32, S24: 4,000 steps, 2,000/200 rows, deterministic, 274 outputs each; re-scored suites equal the notebook's |
| V2 platform anchor | CE(W32) 0.875769 vs record 0.876791: 0.12% (limit 0.5%) |
| V3 weak failure reproduces | gap(W) +1.54% (record +1.46%; limit >= 1.0) |
| V4 configs and kit | every config equals its template apart from base/output; kit `5f79206f` verified on Kaggle |

Two arms (W32 in session 1, S32 in session 2) first failed before training started: peft refuses
Kaggle's preinstalled torchao 0.10.0 while wrapping the model. torchao was uninstalled and each arm
was started once more, as section 7 allows; both sessions carry `RETRY.txt`, and the failed starts'
zips are named with their SHA256 in `MANIFEST.json`. S16 (secondary, optional) was started and
stopped by the owner before it finished; no S16 result exists or is reported.

## Results (seed 42, Kaggle T4 x2, Python 3.13, torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2)

| Arm | Recipe | K | Val CE | MMLU | Common sense | Instruction | Arithmetic | Tool call | JSON | Over-refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| W32 | r4 q/v | 32 | 0.8758 | 1.00 | 0.96 | 1.00 | 0.63 | 0.70 | 1.00 | 0.98 |
| W24 | r4 q/v | 24 | 0.8893 | 0.77 | 0.79 | 0.92 | 0.60 | 0.63 | 1.00 | 0.98 |
| S32 | r16 all 7 | 32 | 0.9828 | 0.96 | 0.96 | 1.00 | 0.60 | 0.83 | 1.00 | 0.98 |
| S24 | r16 all 7 | 24 | 1.0055 | 0.88 | 0.96 | 0.96 | 0.58 | 0.70 | 1.00 | 1.00 |

| Pair | CE gap | MMLU | Common sense | Instruction | Arithmetic | Tool call |
|---|---|---|---|---|---|---|
| weak, W24 vs W32 | **+1.54%** | -0.23 | -0.17 | -0.08 | -0.03 | -0.08 |
| strong, S24 vs S32 | **+2.31%** | -0.08 | 0.00 | -0.04 | -0.03 | -0.13 |

mini_safety is 0/40 in every arm (base model), as in the E2 close-out, and is not in the rule.
Suites are small (24-40 items): 0.08 on MMLU-mini is two questions. One seed, no interval.

Wall time per arm on T4 x2: training 17-31 min, gate suites 19-30 min.

## What this means for E2

- The pre-registered reading: adding adapter capacity does not remove the cost of freezing the
  bottom layers; on CE it grew. The giant-MoE plan should not assume top-K is free.
- The general-suite drop at K=24 is smaller with the strong recipe (MMLU -0.08 vs -0.23, common sense
  0 vs -0.17), but tool calling dropped more; at these suite sizes and one seed this is not a finding.
- The next check that can change the E2 decision is a schedule, not a recipe: train all layers in
  epoch 1 and freeze the bottom only from epoch 2, which is all the activation cache needs
  (`benchmarks/e2/hybrid/PROTOCOL.txt`).

Not measured: other seeds, more data, other learning rates, K=16 with the strong recipe, speed.
