**Quality gate.** min_learned_nat 0.05, material_nat 0.02, sd_multiple 2, forgetting 0.05. Status: proposed by the AI assistant before the first run; approved by the record owner after the results, unchanged (2026-10-11). Scope: this recipe only (section 3): rank 16, 256 steps on the group-disjoint MetaMathQA split, soup ship's leg 2 at 256 new tokens.

Frozen rates: olmoe `A` 0.0002, `AT` 0.0001, `AE` 0.0001; qwen35 `A` 0.0002, `AS` 0.0002, `AT` 0.0001, `AE` 0.0001. V3 not re-decided: learning rates are frozen at E1's section 8 winners.

## olmoe

Scope: learning rates swept on this model.

- E: **MEASURED**; validity: V1 pass, V2 pass, V3 pass, V4 pass
- C: **MEASURED**; validity: V1 pass, V2 pass, V3 pass, V4 pass

Arm order by mean NLL: E AE < AT < A; C AE < AT < A.

| Arm | E verdict | C verdict | Holds | E gap | C gap | Gap holds | Per-seed gaps E / C | Accuracy E / C |
|---|---|---|---|---|---|---|---|---|
| `A` | REGRESSES | REGRESSES | yes | +0.0083 (not material) | +0.0072 (not material) | yes | s0: +0.0081 / +0.0072; s1: +0.0085 / +0.0079; s2: +0.0083 / +0.0066 | 0.574, 0.586, 0.609 / 0.582, 0.613, 0.594 |
| `AT` | REGRESSES | REGRESSES | yes | +0.0016 (not material) | +0.0009 (not material) | yes | s0: +0.0011 / +0.0012; s1: +0.0020 / +0.0013; s2: +0.0018 / +0.0001 | 0.645, 0.582, 0.645 / 0.629, 0.617, 0.605 |
| `AE` | REGRESSES | REGRESSES | yes | +0.0000 (not material) | +0.0000 (not material) | yes | s0: +0.0000 / +0.0000; s1: +0.0000 / +0.0000; s2: +0.0000 / +0.0000 | 0.621, 0.586, 0.656 / 0.637, 0.637, 0.633 |

Base-corrected E/C generalisation contrast: (E - C) of the arm minus (E - C) of the base; the 95% interval reflects question-sampling noise only (E and C rows resampled independently), not a cause. Bootstrap 10000 draws, seed 0; E rows 256, C rows 256. Gap difference = (X - AE) on E minus (X - AE) on C.

| Arm | NLL E | NLL C | E-C | Contrast [95% CI] | Accuracy E | Accuracy C | Contrast [95% CI] | Gap difference NLL [95% CI] |
|---|---:|---:|---:|---|---:|---:|---|---|
| base | 0.9360 | 0.9028 | +0.0332 | - | 0.512 | 0.512 | - | - |
| `A` | 0.2508 | 0.2396 | +0.0112 | -0.0220 [-0.0594, +0.0151] | 0.590 | 0.596 | -0.007 [-0.095, +0.081] | +0.0011 [-0.0009, +0.0031] |
| `AT` | 0.2442 | 0.2333 | +0.0109 | -0.0223 [-0.0599, +0.0151] | 0.624 | 0.617 | +0.007 [-0.081, +0.094] | +0.0008 [-0.0003, +0.0019] |
| `AE` | 0.2426 | 0.2324 | +0.0102 | -0.0230 [-0.0607, +0.0145] | 0.621 | 0.635 | -0.014 [-0.102, +0.072] | - |

Published E1 verdict (historical: published E1 records scored on E, from rule.analyse): **MEASURED**

| Arm | Published verdict | Published E gap |
|---|---|---|
| `A` | REGRESSES | +0.0083 (not material) |
| `AT` | REGRESSES | +0.0016 (not material) |
| `AE` | REGRESSES | +0.0000 (not material) |

## qwen35

Scope: learning rates transferred from OLMoE, not swept on this model.

- E: **MEASURED**; validity: V1 pass, V2 pass, V3 pass, V4 pass
- C: **MEASURED**; validity: V1 pass, V2 pass, V3 pass, V4 pass

Arm order by mean NLL: E A < AS < AT < AE; C A < AS < AT < AE.

| Arm | E verdict | C verdict | Holds | E gap | C gap | Gap holds | Per-seed gaps E / C | Accuracy E / C |
|---|---|---|---|---|---|---|---|---|
| `A` | ACCEPTABLE | ACCEPTABLE | yes | -0.0038 (not material) | -0.0029 (not material) | yes | s0: -0.0038 / -0.0030; s1: -0.0038 / -0.0027 | 0.910, 0.898 / 0.891, 0.875 |
| `AS` | ACCEPTABLE | ACCEPTABLE | yes | -0.0031 (not material) | -0.0024 (not material) | yes | s0: -0.0030 / -0.0026; s1: -0.0032 / -0.0022 | 0.898, 0.906 / 0.883, 0.875 |
| `AT` | REGRESSES | REGRESSES | yes | -0.0020 (not material) | -0.0017 (not material) | yes | s0: -0.0017 / -0.0017; s1: -0.0023 / -0.0018 | 0.863, 0.883 / 0.875, 0.871 |
| `AE` | REGRESSES | REGRESSES | yes | +0.0000 (not material) | +0.0000 (not material) | yes | s0: +0.0000 / +0.0000; s1: +0.0000 / +0.0000 | 0.879, 0.883 / 0.879, 0.871 |

Base-corrected E/C generalisation contrast: (E - C) of the arm minus (E - C) of the base; the 95% interval reflects question-sampling noise only (E and C rows resampled independently), not a cause. Bootstrap 10000 draws, seed 0; E rows 256, C rows 256. Gap difference = (X - AE) on E minus (X - AE) on C.

| Arm | NLL E | NLL C | E-C | Contrast [95% CI] | Accuracy E | Accuracy C | Contrast [95% CI] | Gap difference NLL [95% CI] |
|---|---:|---:|---:|---|---:|---:|---|---|
| base | 0.6341 | 0.6143 | +0.0198 | - | 0.477 | 0.488 | - | - |
| `A` | 0.1348 | 0.1275 | +0.0073 | -0.0125 [-0.0398, +0.0142] | 0.904 | 0.883 | +0.033 [-0.055, +0.121] | -0.0010 [-0.0022, +0.0003] |
| `AS` | 0.1355 | 0.1279 | +0.0076 | -0.0123 [-0.0396, +0.0145] | 0.902 | 0.879 | +0.035 [-0.053, +0.123] | -0.0007 [-0.0019, +0.0004] |
| `AT` | 0.1366 | 0.1286 | +0.0080 | -0.0118 [-0.0390, +0.0150] | 0.873 | 0.873 | +0.012 [-0.074, +0.098] | -0.0003 [-0.0010, +0.0005] |
| `AE` | 0.1386 | 0.1303 | +0.0083 | -0.0115 [-0.0386, +0.0150] | 0.881 | 0.875 | +0.018 [-0.066, +0.104] | - |

Published E1 verdict (historical: published E1 records scored on E, from rule.analyse): **MEASURED**

| Arm | Published verdict | Published E gap |
|---|---|---|
| `A` | ACCEPTABLE | -0.0038 (not material) |
| `AS` | ACCEPTABLE | -0.0031 (not material) |
| `AT` | REGRESSES | -0.0020 (not material) |
| `AE` | REGRESSES | +0.0000 (not material) |

## Reproduction of the published E1 records (historical check)

| Model | Arm | lr | Seed | Reproduced | Differs | Peak CUDA equal |
|---|---|---:|---:|---|---|---|
| olmoe | `A` | 0.0002 | 0 | yes | - | True |
| olmoe | `A` | 0.0002 | 1 | yes | - | True |
| olmoe | `AE` | 0.0001 | 0 | yes | - | True |
| olmoe | `AE` | 0.0001 | 1 | yes | - | True |
| olmoe | `AT` | 0.0001 | 0 | yes | - | True |
| olmoe | `AT` | 0.0001 | 1 | yes | - | True |
| olmoe | `base` | 0 | 0 | yes | - | True |
| olmoe | `A` | 0.0002 | 2 | yes | - | True |
| olmoe | `AE` | 0.0001 | 2 | yes | - | True |
| olmoe | `AT` | 0.0001 | 2 | yes | - | True |
| qwen35 | `A` | 0.0002 | 0 | yes | - | True |
| qwen35 | `AS` | 0.0002 | 0 | yes | - | True |
| qwen35 | `AT` | 0.0001 | 0 | yes | - | True |
| qwen35 | `base` | 0 | 0 | yes | - | True |
| qwen35 | `AS` | 0.0002 | 1 | yes | - | True |
| qwen35 | `AT` | 0.0001 | 1 | yes | - | True |
| qwen35 | `A` | 0.0002 | 1 | yes | - | True |
| qwen35 | `AE` | 0.0001 | 0 | yes | - | True |
| qwen35 | `AE` | 0.0001 | 1 | yes | - | True |

## Learning-rate sweep on the new records (olmoe, seed 0)

Note: a C winner that differs from E1's rate is a diagnostic, not proof of a cause.

| Arm | Set | 5e-05 | 0.0001 | 0.0002 | 0.0004 | Winner | Status | Frozen | C winner differs |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| `A` | E | - | 0.2548 | 0.2515 | 0.2529 | 0.0002 | ok | 0.0002 |  |
| `A` | C | - | 0.2432 | 0.2401 | 0.2420 | 0.0002 | ok | 0.0002 | False |
| `AE` | E | 0.2446 | 0.2435 | 0.2469 | 0.2691 | 0.0001 | ok | 0.0001 |  |
| `AE` | C | 0.2338 | 0.2330 | 0.2380 | 0.2595 | 0.0001 | ok | 0.0001 | False |
| `AT` | E | 0.2475 | 0.2446 | 0.2448 | 0.2576 | 0.0001 | ok | 0.0001 |  |
| `AT` | C | 0.2365 | 0.2341 | 0.2349 | 0.2468 | 0.0001 | ok | 0.0001 | False |

## Box checks (second-box base, never counted)

- `e1-c-olmoe2/e1-results/confirmcheck-olmoe-base-box2.json` (olmoe): identical True
- `e1-c3-olmoe/e1-results/confirmcheck-olmoe-base-box3.json` (olmoe): identical True
- `e1-c3-qwen/e1-results/confirmcheck-qwen35-base-box3.json` (qwen35): identical True

## RAM tier (qwen35)

Parity pilot: NOT PASSED (parity pilot did not pass (status not_passed, passed False)), `e1-results/parity-qwen35.json`.

| Step | Streamed loss | Resident loss | Relative |
|---:|---:|---:|---:|
| 0 | 0.708621 | 0.708621 | 0.000000 |
| 1 | 0.485761 | 0.480979 | 0.009942 |
| 2 | 0.260612 | 0.264705 | 0.015461 |
| 3 | 0.152288 | 0.154201 | 0.012409 |

Loss rule: relative <= 0.001.

Step-0 gradient: global relative L2 0.010718 (rule <= 0.01), min cosine 0.999748 (rule >= 0.99), max abs 0.003021; bit exact {'losses': False, 'gradients_step0': False}.

E: **NOT COUNTED** (parity pilot did not pass (status not_passed, passed False)); descriptive verdict if counted: NO VERDICT; missing: `RAM-tier A seed 1`, `RAM-tier AS seed 1`

vs resident (not attributable to streaming or NF4: it mixes the NF4 base, warmup, data order and initialisation):

| Arm | NLL RAM - resident | Accuracy RAM - resident |
|---|---:|---:|
| `A` | +0.0012 | +0.010 |
| `AS` | +0.0011 | +0.008 |

C: **NOT COUNTED** (parity pilot did not pass (status not_passed, passed False)); descriptive verdict if counted: NO VERDICT; missing: `RAM-tier A seed 1`, `RAM-tier AS seed 1`

vs resident (not attributable to streaming or NF4: it mixes the NF4 base, warmup, data order and initialisation):

| Arm | NLL RAM - resident | Accuracy RAM - resident |
|---|---:|---:|
| `A` | +0.0006 | +0.000 |
| `AS` | +0.0011 | +0.012 |

Memory (memory is a property of the regime and is reported whether or not the parity pilot passed; §11.3's strict reading counts nothing from these runs; AS - A peak, seeds [0]): extra parameters 4915200, bytes per parameter 8, tolerance +13.2 MiB.

| Rule | Predicted | Residual per seed | Residual max-over-seeds | Holds |
|---|---:|---|---:|---|
| e1 | 75.0 MiB | s0: -46.9 MiB | -46.9 MiB | NO |
| dtype | 37.5 MiB | s0: -9.4 MiB | -9.4 MiB | yes |

## Rejected published records

- `e1-olmoe/e1-results/run-olmoe-AE-lr2e-4-s0.json`: harness 900a067703169e66dddff50a5b401e133ffb0c72955c6c575dbfbaafe11a4296
- `e1-olmoe/e1-results/run-olmoe-base.json`: harness 900a067703169e66dddff50a5b401e133ffb0c72955c6c575dbfbaafe11a4296
