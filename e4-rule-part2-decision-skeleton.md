# E4 part 2, main rule: decision skeleton

Card: [E4]. Follows the calibration rule (`e4-part2-calibration-rule.md`, with its 2026-10-20 card-class clarification) and extends the draft `e4-rule-part2-draft.md` (v4.2).
Status: SKELETON, committed 2026-10-21, before any calibration data exists. Nothing in part 2 has been run or measured. Every `{{CAL.*}}` below is empty on purpose and is filled in a later commit, before the first training arm (house rule 1; `benchmarks/METHOD.md:275-277`). The `.gitattributes` line `<this file> text eol=lf` is added with this commit. Planned dates (plans, not outcomes): calibration report Oct 24, this rule with values filled in Oct 26, both conditional on the accesses requested in the PR thread. Review happens before any training.

**Frozen as of this commit.** Only `{{CAL.*}}` placeholders may change in the Oct 26 commit. The comparison table, the metric, the sign convention, delta, the outcome table and the labels are frozen now. Any other change is a new rule version, not an edit.

This file fixes the shape of the decision: the metric, the comparison, and when the answer is PASS, FAIL or NO VERDICT. It carries no number that comes from calibration. Where the draft v4.2 already fixed a number by agreement (delta, bootstrap, row counts), it is repeated here and marked AGREED.

## 1. Question

Does NF4 double-quant (DQ) on the frozen base change what LoRA learns on OLMoE-1B-7B, measured by task loss on DECISION? Path: fake-quant + peft ParamWrapper. The answer applies to this path only and is not a statement about Soup's production 4-bit QLoRA (draft v4.2 section 0).

## 2. Comparison

| Arm | Base | Role |
|---|---|---|
| A_off | fake-quant NF4, DQ off | comparator |
| A_on | fake-quant NF4, DQ on | tested |
| A_bf16 | unquantized bf16 | scale reference |

Primary contrast: **A_on vs A_off**, same seeds (42, 7, 123), same data order within a seed, same recipe (draft v4.2 sections 5 and 6, unchanged). A_bf16 is not part of the primary contrast. It sets the scale: how much quantization itself costs (M3), so a DQ effect can be read against it. A_bf16 is reference only and does not gate the outcome (section 7).

## 3. Metric

M1, per arm and seed: response-target NLL on the 2,048 DECISION rows, nat per supervised target, token-weighted (per-sequence sums in fp64, stored per row, arm and seed with the target count n_i).

**Sign convention.** D = NLL(A_on) - NLL(A_off) on DECISION. Positive D means DQ on is worse (higher loss); negative D means DQ on is better. Every use of D, its CI and delta below follows this sign.

Primary statistic D: the DQ effect on M1, `D = mean over seeds s of [ sum_i (sum_on,s,i - sum_off,s,i) / sum_i n_i ]`, computed on a paired bootstrap, 2,000 resamples, seed 20261015, one multiset of row indices per resample shared by every arm and seed. 90% percentile CI. AGREED.

Reported with D, not decided on: M0 (base NLL before training), M3 = A_off - A_bf16 and A_on - A_bf16, L = (M1_on - M0_on) - (M1_off - M0_off), the per-seed D_s. The CI is conditional on these three seeds and does not cover seed-to-seed or GPU nondeterminism; calibration P is one paired observation and is reported only.

## 4. Outcome

The outcome is one of PASS, FAIL, NO VERDICT, with the sub-case labels below. Evaluate in this order; the first row that matches decides. CI is the 90% CI of D.

| # | Condition | Outcome |
|---|---|---|
| 1 | Any validity row in section 5 fails | **NO VERDICT (invalid)**, reason named |
| 2 | CI lower bound > +delta | **FAIL**: DQ harms NLL |
| 3 | CI entirely within [-delta, +delta] | **PASS**: no material NLL effect |
| 4 | CI upper bound < -delta | **PASS (DQ helps)** |
| 5 | otherwise | **NO VERDICT (inconclusive)**: CI straddles a bound |

delta = **0.005 nat** (AGREED; about 0.5% in perplexity ratio; a margin for this experiment, not an established threshold). It is not calibration-derived and is not changed after calibration.

**Suites** (M2, `DEFAULT_GENERAL_SUITE`, scorer revision 4) are exploratory flags and never change the outcome above. Per suite, flagged when A_on is below A_off by more than `{{CAL.epsilon[suite]}}` (strict `>`) on every seed. The record states the NLL outcome and the suite flags as two separate results and uses no combined phrase ("quality preserved", "gate passed").

**Wording.** PASS means "no NLL effect larger than delta was found on this path with these seeds". It does not mean equivalence of anything else.

## 5. Validity (any failure gives NO VERDICT)

| Row | Check | Uses |
|---|---|---|
| U1-U4 | quantization outputs, mapping, hashes | none |
| S | strong control: a 1.5x absmax error is detected on every tested unit (max abs output difference above 10 x tau) | `{{CAL.tau[cell,output]}}` |
| X1-X4 | wrapper vs the intended expert operation | `{{CAL.tol_W[output]}}`, `{{CAL.tol_W[input_grad]}}`, `{{CAL.tol_W[A_grad]}}`, `{{CAL.tol_W[B_grad]}}`, `{{CAL.tol_W[update]}}` |
| X5, E5 | frozen base unchanged | none |
| P1, P2 | adapter placement and data order identical across arms | none |
| Finiteness | every loss finite | none |
| Peers, Provenance | per run | none |
| Card | every run on `{{CAL.final_card}}`, instance `{{CAL.instance_id}}`, GPU UUID `{{CAL.gpu_uuid}}` | the calibration decision |

E1-E4 (base-weight checks, tau per cell and quantity) do not give NO VERDICT: a cell that fails is reported as not shown equivalent to real 4-bit, and the outcome stays scoped to fake-quant + wrapper. A cell with no tau gets no claim (calibration section 4).

A void run (crash, OOM, interruption, peer, hash mismatch) keeps its record and is rerun once in its position with the same seed; a second void of the same run stops the study and is reported as NO VERDICT (invalid).

## 6. Placeholders

All come from `benchmarks/results/e4-part2-calibration-<date>/report.json` on the final card. None is replaced at run time.

| Placeholder | Report key | Used in |
|---|---|---|
| `{{CAL.final_card}}`, `{{CAL.instance_id}}`, `{{CAL.gpu_uuid}}` | `final_card` | section 5 Card |
| `{{CAL.tau[cell,quantity]}}` | `tau` | S, E1-E4 |
| `{{CAL.tol_W[quantity]}}` | `tol_W` | X1-X4 |
| `{{CAL.epsilon[suite]}}`, `{{CAL.floor[suite]}}`, `{{CAL.n_items[suite]}}` | `noise_floor` | section 4 suites |
| `{{CAL.projected_hours}}` | `projected_main_plan.total_hours` | schedule only |
| `{{CAL.calibration_commit}}`, `{{CAL.record_sha256sums}}` | the record's `SHA256SUMS` | section 8 |

A kind or cell without a tau (fewer than 4 contributing units, or never seen on SMOKE) is written as `NONE` with the reason, not left blank.

## 7. Decisions on the open points

Proposed by the author for the reviewer to accept or reject; in force as written unless rejected before the Oct 26 commit.

1. A_bf16 is reference only and does not gate. M3 is reported next to D and decides nothing.
2. An inconclusive CI is a NO VERDICT sub-case, labelled "NO VERDICT (inconclusive)", distinct from "NO VERDICT (invalid)" (a failed validity row).
3. A CI entirely below -delta is a PASS sub-case, labelled "PASS (DQ helps)".

## 8. Before the Oct 26 commit

- Fill every `{{CAL.*}}` from `report.json`; `grep -c '{{' <this file>` must print 0.
- Reviewed before any training (planned Oct 26). Only `{{CAL.*}}` placeholders change.
- Commit message names the calibration commit and the SHA-256 of the record's `SHA256SUMS`.
- Sections 3 to 5 of the draft v4.2 (checks, coverage, arms) are copied unchanged; if the owner prefers one document, the filled skeleton replaces draft sections 7 and 8 and nothing else.
