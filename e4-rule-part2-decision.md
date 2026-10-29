# E4 part 2, main rule: decision skeleton

Card: [E4]. Follows the calibration rule (`e4-part2-calibration-rule.md`, with its 2026-10-20 card-class clarification) and extends the draft `e4-rule-part2-draft.md` (v4.2).
Status: SKELETON, committed 2026-10-21, before any calibration data exists. Filled draft (2026-10-25) in section 9, pending review; not in force until reviewed and merged. Nothing in part 2 has been run or measured. Every `CAL.*` value below was empty in the skeleton and is filled here from the calibration record (section 9 gives each value and its source), before the first training arm (house rule 1; `benchmarks/METHOD.md:275-277`). The `.gitattributes` line `<this file> text eol=lf` is added with this commit. Planned dates (plans, not outcomes): calibration report Oct 24, this rule with values filled in Oct 26, both conditional on the accesses requested in the PR thread. Review happens before any training.

**Frozen as of this commit.** Only `CAL.*` placeholders may change in the Oct 26 commit. The comparison table, the metric, the sign convention, delta, the outcome table and the labels are frozen now. Any other change is a new rule version, not an edit.

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

**Suites** (M2, `DEFAULT_GENERAL_SUITE`, scorer revision 4) are exploratory flags and never change the outcome above. Per suite, flagged when A_on is below A_off by more than the suite's epsilon in section 9.4 (strict `>`) on every seed. The record states the NLL outcome and the suite flags as two separate results and uses no combined phrase ("quality preserved", "gate passed").

**Wording.** PASS means "no NLL effect larger than delta was found on this path with these seeds". It does not mean equivalence of anything else.

## 4a. Gate/up fallback cells (owner's engineering decision, 2026-10-25)

nurik delegated engineering decisions to me on 2026-10-25 via Telegram. This section is the owner's engineering decision under that delegation.

- The four uncalibrated output cells `gate_proj` and `up_proj` on the fallback kernel, with needs_grad False and True (section 9.15), are to be calibrated with concatenated SMOKE inputs in the fallback range (M > 128).
- Targets, decided 2026-10-25: M = 129 and 256, then the full distinct pool when the pool is above 256 (capped at 512). Cells are keyed on (kind, needs_grad, kernel), not on M. No row is repeated. Source unit ids and the pool size are recorded on every call.
- Observations on this series (instance 57707996, folder `e4-calibration-2026-10-25-b`):
  - Attempt `fc0f5e1b` (`5ab78624`, the earlier targets 129, 256, 384, 512): void, `cannot reach M=384 from 286 distinct SMOKE rows`. The pool is 286 distinct rows.
  - Attempt `ca44d213`: started 2026-10-26 00:43:54 UTC and interrupted (no end event). The journal closed it as void at 01:21:46 UTC. No record file; commit not recorded.
  - After these two voids the harness stopped the run (calibration section 8). No gate/up fallback tau exists in this record.
- Disclosure: F, T, W, N_1 to N_5 and P ran on `5ab78624`. No recorded attempt ran on `bbe772db`, which carries the pool-based targets. The fallback targets in the code are the only change between them.
- **OPEN, pending nurik:** whether `T_fallback_gateup` is rerun, and on which commit. The calibration rule says a failed T attempt stops calibration and is not rerun (calibration section 8). A rerun is an owner's decision, not an allowance in the rule.
- When measured, these taus are merged only through this section. They are not added to the section 9.2 table, and inherited lines, including calibration section 4, are unchanged.
- Card: the values, when they exist, come from the same instance as the rest of this series (57707996; card-change clause, calibration rule line 155; section 9.16).
- Status: OPEN pending nurik. Section 9.15 keeps the four cells uncovered.

## 5. Validity (any failure gives NO VERDICT)

| Row | Check | Uses |
|---|---|---|
| U1-U4 | quantization outputs, mapping, hashes | none |
| S | strong control: a 1.5x absmax error is detected on every tested unit (max abs output difference above 10 x tau) | output tau per cell, section 9.2 |
| X1-X4 | wrapper vs the intended expert operation | tol_W, section 9.3: output 0.2197437286376953, input_grad 0.017418980598449707, A_grad 0.6594619750976562, B_grad 0.08243513107299805, update 0.0004000067710876465 |
| X5, E5 | frozen base unchanged | none |
| P1, P2 | adapter placement and data order identical across arms | none |
| Finiteness | every loss finite | none |
| Peers, Provenance | per run | none |
| Card | every run on A100 40 GB (`NVIDIA A100-PCIE-40GB`), instance `57707996`, GPU UUID `GPU-b296fdd0-1f0b-1bb9-ad20-66fb670a37f0` | the calibration decision |

E1-E4 (base-weight checks, tau per cell and quantity) do not give NO VERDICT: a cell that fails is reported as not shown equivalent to real 4-bit, and the outcome stays scoped to fake-quant + wrapper. A cell with no tau gets no claim (calibration section 4).

A void run (crash, OOM, interruption, peer, hash mismatch) keeps its record and is rerun once in its position with the same seed; a second void of the same run stops the study and is reported as NO VERDICT (invalid).

## 6. Placeholders

All come from `benchmarks/results/e4-part2-calibration-<date>/report.json` on the final card. None is replaced at run time.

| Value | Report key | Used in |
|---|---|---|
| final card, instance, GPU UUID (section 9.1) | `final_card` | section 5 Card |
| tau per cell and quantity (section 9.2) | `tau` | S, E1-E4 |
| tol_W per quantity (section 9.3) | `tol_W` | X1-X4 |
| epsilon, floor, n_items per suite (section 9.4) | `noise_floor` | section 4 suites |
| projected hours (section 9.1) | `projected_main_plan.total_hours` | schedule only |
| calibration commit and record SHA-256SUMS (section 9.1) | the record's `SHA256SUMS` | section 8 |

A kind or cell without a tau (fewer than 4 contributing units, or never seen on SMOKE) is written as `NONE` with the reason, not left blank.

## 7. Decisions on the open points

Proposed by the author for the reviewer to accept or reject; in force as written unless rejected before the Oct 26 commit.

1. A_bf16 is reference only and does not gate. M3 is reported next to D and decides nothing.
2. An inconclusive CI is a NO VERDICT sub-case, labelled "NO VERDICT (inconclusive)", distinct from "NO VERDICT (invalid)" (a failed validity row).
3. A CI entirely below -delta is a PASS sub-case, labelled "PASS (DQ helps)".

## 8. Before the Oct 26 commit

- Fill every `CAL.*` value from `report.json` (done; sources in section 9). The placeholder check `grep -c '\{\{' <this file>` must print 0.
- Reviewed before any training (planned Oct 26). Only `CAL.*` placeholders change.
- Commit message names the calibration commit and the SHA-256 of the record's `SHA256SUMS`.
- Sections 3 to 5 of the draft v4.2 (checks, coverage, arms) are copied unchanged; if the owner prefers one document, the filled skeleton replaces draft sections 7 and 8 and nothing else.

## 9. Filled values, sources and open items

Status of this section: a filled draft for review. Values are copied from the calibration folder `~/e4-cal-records/e4-cal-3/e4-calibration-2026-10-25-b` (instance 57707996, `report.json`) unless a row says otherwise. Citations to the earlier #36 record (`d1a3ddf1`, instance 57596627) are superseded: that instance is destroyed. No value was chosen by the author. The three section 7 proposals are unchanged (review 5643944224). delta = 0.005 nat, the sign convention, the CI conditions, the metric and the comparison are unchanged from the skeleton.

Scope: fake-quant bf16 write-back plus the PEFT ParamWrapper path, as in section 1. Nothing here is a statement about production 4-bit QLoRA.

### 9.1 Placeholder values

Source: `report.json` in the calibration folder `~/e4-cal-records/e4-cal-3/e4-calibration-2026-10-25-b` (copied from instance 57707996), unless a row says otherwise. This calibration replaces the one on instance 57596627 (#36 record `d1a3ddf1`), which was destroyed. Values are copied as stored; no value was chosen by the author.

| Item | Value | Source |
|---|---|---|
| Final card | A100 40 GB (`NVIDIA A100-PCIE-40GB`), decision `FINAL_A100_40GB` | report.json `final_card`, `final_card_decision`; decision.json |
| Instance | `57707996` | report.json `final_card` |
| GPU UUID | `GPU-b296fdd0-1f0b-1bb9-ad20-66fb670a37f0` | report.json `final_card` |
| Run commit (code that produced F, T, W, N_1-5, P and the one T_fallback record) | `5ab78624` | run records `provenance.soup_commit` and `harness_commit` |
| Record SHA-256SUMS | `7e92b131496cc37a0eb259fbd0ca9a338f3d798f8d581a3a9aebbe0d7905e91e` | shasum -a 256 SHA256SUMS in the folder; 28 of 28 entries OK |
| Record location | not in the repo: `~/e4-cal-records/e4-cal-3/e4-calibration-2026-10-25-b` (Mac copy) | section 9.11, item 1 |
| Projected main plan, training updates only | 5.722281638913684 h (9 runs + 1 rerun of A_on, 256 updates each) | report.json `projected_main_plan` |

### 9.2 tau (T)

Cell names follow calibration section 4: `kind/needs_grad=.../kernel=...` for forward cells and `kind/backward` for the input-gradient cell. Skeleton kind names map to report names as `q` = `q_proj` and so on. The report has no `gate_proj` or `up_proj` fallback cell: those four cells are not calibrated in this record (section 4a, OPEN). LoRA quantities exist for attention kinds only.

| Cell | Quantity | tau |
|---|---|---|
| down_proj/backward | input_grad | 0.023301124572753906 |
| down_proj/needs_grad=False/kernel=fallback | output | 0.02446269989013672 |
| down_proj/needs_grad=False/kernel=fused | output | 0.05623054504394531 |
| down_proj/needs_grad=True/kernel=fallback | output | 0.02446269989013672 |
| down_proj/needs_grad=True/kernel=fused | output | 0.05623054504394531 |
| gate_proj/backward | input_grad | 0.019277572631835938 |
| gate_proj/needs_grad=False/kernel=fused | output | 0.042357444763183594 |
| gate_proj/needs_grad=True/kernel=fused | output | 0.042357444763183594 |
| k_proj/backward | input_grad | 0.02360677719116211 |
| k_proj/needs_grad=False/kernel=fallback | output | 0.00781559944152832 |
| k_proj/needs_grad=False/kernel=fused | A_grad | 0.1528167724609375 |
| k_proj/needs_grad=False/kernel=fused | B_grad | 0.03586554527282715 |
| k_proj/needs_grad=False/kernel=fused | output | 0.0077970027923583984 |
| k_proj/needs_grad=False/kernel=fused | update | 0.00039999932050704956 |
| k_proj/needs_grad=True/kernel=fallback | output | 0.00781559944152832 |
| k_proj/needs_grad=True/kernel=fused | A_grad | 0.1528167724609375 |
| k_proj/needs_grad=True/kernel=fused | B_grad | 0.03586554527282715 |
| k_proj/needs_grad=True/kernel=fused | output | 0.0077970027923583984 |
| k_proj/needs_grad=True/kernel=fused | update | 0.00039999932050704956 |
| o_proj/backward | input_grad | 0.032341957092285156 |
| o_proj/needs_grad=False/kernel=fallback | output | 0.0034580230712890625 |
| o_proj/needs_grad=False/kernel=fused | A_grad | 0.01310586929321289 |
| o_proj/needs_grad=False/kernel=fused | B_grad | 0.0032613277435302734 |
| o_proj/needs_grad=False/kernel=fused | output | 0.0008977353572845459 |
| o_proj/needs_grad=False/kernel=fused | update | 0.00039999373257160187 |
| o_proj/needs_grad=True/kernel=fallback | output | 0.0034580230712890625 |
| o_proj/needs_grad=True/kernel=fused | A_grad | 0.01310586929321289 |
| o_proj/needs_grad=True/kernel=fused | B_grad | 0.0032613277435302734 |
| o_proj/needs_grad=True/kernel=fused | output | 0.0008977353572845459 |
| o_proj/needs_grad=True/kernel=fused | update | 0.00039999373257160187 |
| q_proj/backward | input_grad | 0.02473592758178711 |
| q_proj/needs_grad=False/kernel=fallback | output | 0.007790803909301758 |
| q_proj/needs_grad=False/kernel=fused | A_grad | 0.13753509521484375 |
| q_proj/needs_grad=False/kernel=fused | B_grad | 0.0281793475151062 |
| q_proj/needs_grad=False/kernel=fused | output | 0.007513523101806641 |
| q_proj/needs_grad=False/kernel=fused | update | 0.00039999932050704956 |
| q_proj/needs_grad=True/kernel=fallback | output | 0.007790803909301758 |
| q_proj/needs_grad=True/kernel=fused | A_grad | 0.13753509521484375 |
| q_proj/needs_grad=True/kernel=fused | B_grad | 0.0281793475151062 |
| q_proj/needs_grad=True/kernel=fused | output | 0.007513523101806641 |
| q_proj/needs_grad=True/kernel=fused | update | 0.00039999932050704956 |
| up_proj/backward | input_grad | 0.02380847930908203 |
| up_proj/needs_grad=False/kernel=fused | output | 0.09822463989257812 |
| up_proj/needs_grad=True/kernel=fused | output | 0.09822463989257812 |
| v_proj/backward | input_grad | 0.034206390380859375 |
| v_proj/needs_grad=False/kernel=fallback | output | 0.003907561302185059 |
| v_proj/needs_grad=False/kernel=fused | A_grad | 0.1528167724609375 |
| v_proj/needs_grad=False/kernel=fused | B_grad | 0.03586554527282715 |
| v_proj/needs_grad=False/kernel=fused | output | 0.006474494934082031 |
| v_proj/needs_grad=False/kernel=fused | update | 0.00039999932050704956 |
| v_proj/needs_grad=True/kernel=fallback | output | 0.003907561302185059 |
| v_proj/needs_grad=True/kernel=fused | A_grad | 0.1528167724609375 |
| v_proj/needs_grad=True/kernel=fused | B_grad | 0.03586554527282715 |
| v_proj/needs_grad=True/kernel=fused | output | 0.006474494934082031 |
| v_proj/needs_grad=True/kernel=fused | update | 0.00039999932050704956 |

### 9.3 tol_W (W)

Source: `tol_W` in report.json, 2 x max over calls, layers and arms (calibration section 5).

| Quantity | tol_W |
|---|---|
| output | 0.2197437286376953 |
| input_grad | 0.017418980598449707 |
| A_grad | 0.6594619750976562 |
| B_grad | 0.08243513107299805 |
| update | 0.0004000067710876465 |

### 9.4 Suite noise floor and epsilon (N)

Five greedy runs of the seed-42 16-update adapter, identical scores in all five. Floor 0.0 everywhere, so epsilon = 1/n_items. Comparison is strict `>` (calibration section 6).

| Suite | n_items | Score (each of 5 runs) | Floor | Epsilon |
|---|---|---|---|---|
| mini_arithmetic | 40 | 0.625 (x5, equal) | 0.0 | 0.025 |
| mini_common_sense | 24 | 0.7916666666666666 (x5, equal) | 0.0 | 0.041666666666666664 |
| mini_format_json | 40 | 0.9 (x5, equal) | 0.0 | 0.025 |
| mini_instruction | 24 | 0.9583333333333334 (x5, equal) | 0.0 | 0.041666666666666664 |
| mini_mmlu | 26 | 0.6538461538461539 (x5, equal) | 0.0 | 0.038461538461538464 |
| mini_over_refusal | 40 | 0.975 (x5, equal) | 0.0 | 0.025 |
| mini_safety | 40 | 0.0 (x5, equal) | 0.0 | 0.025 |
| mini_tool_call | 40 | 0.45 (x5, equal) | 0.0 | 0.025 |

### 9.5 F memory and time (F)

Peak reserved threshold: 36,000,000,000 bytes (calibration section 3). Every arm is below it, so the final card is the 40 GB instance.

| Arm | Peak reserved bytes | Peak allocated bytes | Median s per update | Device max used (NVML) bytes | Device min free (NVML) bytes |
|---|---|---|---|---|---|
| A_off | 17846763520 | 17076301312 | 8.030015620053746 | 18940887040 | 24008785920 |
| A_on | 17846763520 | 17076301312 | 7.98066275799647 | 18940887040 | 24008785920 |
| A_bf16 | 17846763520 | 17076301312 | 8.152295885025524 | 18940887040 | 24008785920 |

These are the 16-update F runs at the calibration learning-rate schedule (calibration section 3), not main-plan peaks.

### 9.6 Repeatability (P)

A_on, seed 42, repeated once: NLL 1.626105685835482 vs 1.6260342313352905, difference -7.145450019163668e-05; adapters bit-identical: False. One paired observation, reported only; it is not a decision input.

### 9.7 Budget (not a schedule for the whole study)

Section 9.14b gives the projected budget from this record's timings. Loading, evaluation, controls and retries are projected there from observed times, and DECISION NLL time stays UNKNOWN.

### 9.8 Retry limit

A new output directory does not reset the section 5 and calibration section 8 void count. This series (folder `e4-calibration-2026-10-25-b`) is a restart on the same instance as the first attempt; section 9.17 records it. The void table in 9.17 lists every void in this series' journal.

### 9.9 mini_safety

mini_safety is 0.0 on every calibration run, so it cannot flag a drop from zero. The absence of a flag is not safety evidence. Any refusal-behaviour reading is unverified without generated answers. Suites remain exploratory and do not decide the outcome (section 4).

### 9.10 Edits to the skeleton that are not placeholder fills

Compared with `e4-rule-part2-decision-skeleton.md` at `main`, these lines changed and are not a value fill:

1. Status line: the sentence "Every brace-wrapped CAL placeholder below is empty on purpose and is filled in a later commit" became "Every `CAL.*` value below was empty in the skeleton and is filled here from the calibration record"; a sentence pointing to section 9 and stating "not in force until reviewed and merged" was added.
2. Frozen paragraph: "Only brace-wrapped CAL placeholders" became "Only `CAL.*` placeholders" (braces removed).
3. Section 6 table header: "Placeholder" became "Value"; the first column's placeholder names became plain keys with section references.
4. Section 8: "Only brace-wrapped CAL placeholders change" became "Only `CAL.*` placeholders change" (braces removed). The fill bullet was reworded, and its check line now reads `grep -c '\{\{' <this file>`, so the line itself holds no placeholder token.

Placeholder fills (values or references to section 9): section 4 suite epsilon; section 5 S, X1-X4 and Card rows; section 6 value column. Sections 1-3, 4 (apart from the epsilon reference), 5 (other rows), 7 and the outcome table are unchanged. Diff against the skeleton to confirm.

### 9.11 OPEN items (must be resolved before the first decision training run)

1. **Record not in the repo.** The values come from `~/e4-cal-records/e4-cal-3/e4-calibration-2026-10-25-b` (SHA-256SUMS `7e92b131496cc37a0eb259fbd0ca9a338f3d798f8d581a3a9aebbe0d7905e91e`). The #36 record (`d1a3ddf1`, instance 57596627) is superseded for these values. The owner decides how this record is committed and cited. Until then the rule is not in force.
2. **Gate/up fallback taus: OPEN, pending nurik.** This record has no gate_proj or up_proj fallback tau: both T_fallback attempts were voided (9.17). Section 4a holds the decision; no values are entered.
3. **Resolved: cell of each main-run call** (9.12).
4. **Split pin.** The three hashes in 9.13 are those of the split rebuilt on the box (runbook step 3). The box-side check output is not in the record folder. Confirm before the main series.
5. **Budget.** Projected table in 9.14b: 17.5611 h, excluding DECISION NLL time (UNKNOWN). **OPEN for nurik: the main-run spending cap.**
6. **S coverage gap.** Still 4 of 28 reachable output cells uncalibrated (9.15), until item 2 is resolved.
7. **Resolved: main-run counts** (9.14c).

### 9.12 Cell of each main-run call

Check result: the calibration harness records (kind, needs_grad, kernel) per call, and the cell is assigned in one place:

- Kernel: `GemmProbe` (`benchmarks/harness/e4_part2_gemm_probe.py:40`) records each `gemm_4bit` dispatch; `take()` (`:102`) returns `fused` or `fallback`.
- `linear_forward_backward` (`benchmarks/harness/e4_part2_calibration.py:564`) runs one call with the probe; `info = probe.take(1)` at `:577`.
- `run_t_unit_calls` (`:588`) assigns per call: `kernel = info["kernels"][0]` (`:609`); `cell_name(kind, needs_grad, kernel)` (`:616`, function at `:514`); `backward_cell(kind)` for input gradients (`:622`, function at `:518`). Each record stores `kind`, `needs_grad`, `kernel` and `cell`.
- LoRA quantities use the same probe and `cell_name` (`:693-703`).

No main-run module calls these functions yet (grep: only the calibration harness and the probe module). The main-run harness must import them; this is a requirement, not a completed step.

**Proposed rule text (not in force; for review):** For every main-run call the harness assigns the cell with the functions of section 9.12 (kernel from the GEMM probe, cell from `cell_name`, input gradients from `backward_cell`) and logs the cell on that call's record. The main-run record reports, per cell, the count of calls with no calibrated tau (no claim). Reporting only; no claim does not count as pass.

**Resolution.** The instruction "a call whose cell has no calibrated tau is void" conflicts with inherited text (calibration section 4 and skeleton section 5: no claim). The inherited "no claim" is kept and no void clause is written. Inherited lines are not changed.

### 9.13 Split pin

| Manifest | File | Rows | SHA-256 |
|---|---|---|---|
| DECISION | `split/decision.manifest.jsonl` | 2048 | `90787bbf0557d2194f65a7abe4260c2096f4c0237d534040d4e4d7830723453f` |
| SMOKE | `split/smoke.manifest.jsonl` | 256 | `8e13970ff802550a0367ebb94bc5ae88b27a9ad1b2cb24f0bcb70b4d4a03b55e` |
| TRAIN | `split/train.manifest.jsonl` | 4096 | `ba33e9dc09a4b37bf85f9d10c05071e5a03ccf412c917bc3ccfd3446d6e7c047` |

Dataset `databricks/databricks-dolly-15k` at revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`. Eligible rows: 11,844, above the 6,400 floor. The earlier pre-commit count of 11,952 is not explained; it does not affect validity, because the current split follows the rule. The reviewer rebuilt the split and reproduced 11,844 eligible rows and all three manifests byte-for-byte (nurik's review on #36, 5643944224). This series used the split rebuilt on the box (runbook step 3); the box-side hash check is OPEN (9.11, item 4).

### 9.14 Observed wall-clock on this calibration (OBSERVED ON CALIBRATION, not a prediction)

Source: `attempts.jsonl` start and end timestamps (UTC) of this series (folder `e4-calibration-2026-10-25-b`) and each run's `seconds_per_update`. Stage labels follow the calibration stages. Loading and evaluation are not timed separately, so they sit inside the attempt wall-clock columns.

| Stage | Runs (ok) | Attempt wall s (each) | Update time s |
|---|---|---|---|
| Training: F_A_off, F_A_on, F_A_bf16, P (16 updates each) | 4 | F_A_off 370, F_A_on 368, F_A_bf16 369, P 375 | F_A_off 130.7, F_A_on 130.1, F_A_bf16 132.6, P 131.7 |
| Control: T (full selection) | 1 | 187 | - |
| Control: W | 1 | 1040 | - |
| Evaluation: N_1 to N_5 (one scoring each) | 5 | 852, 855, 857, 859, 856 | - |

Totals: first attempt start 2026-10-25 22:41:01 UTC to last attempt end 2026-10-26 01:21:46 UTC, including the voids in 9.17. The 16-update rows are calibration runs, not main-plan runs.

### 9.14b Projected budget (PROJECTED from this calibration's timings, not a prediction)

Counts: arms and seeds from the skeleton (section 2); 256 updates from calibration section 3 (linear decay to 0 at update 256); the rerun count of 1 from calibration section 3 ("9 runs + 1 rerun"), with arm A_on from report.json `projected_main_plan.rerun_arm`. Per-update times and attempt overhead are observed in this series. Nothing is estimated beyond these.

| Component | Count | Per-unit time (observed) | Formula | Source | Hours |
|---|---|---|---|---|---|
| Training updates | 10 runs (3 A_off, 4 A_on incl. rerun, 3 A_bf16), 256 updates each | A_off 8.030015620053746 s; A_on 7.98066275799647 s; A_bf16 8.152295885025524 s per update | 256 x (3 x A_off + 4 x A_on + 3 x A_bf16) = 20600.2 s | counts: skeleton section 2, calibration section 3; times: report.json `projected_main_plan.median_seconds_per_update` | 5.7223 |
| Loading (per training attempt) | 10 attempts | 239.22 s: mean over F_A_off, F_A_on, F_A_bf16, P of (attempt wall - update time); includes model load and other non-training time | 10 x 239.22 s = 2392.2 s | journal `attempts.jsonl`; run records `seconds_per_update` | 0.6645 |
| Evaluation: suite scorings (M2) | 6 scorings (A_on and A_off, 3 seeds): DERIVED from skeleton section 4 ("on every seed") | 855.8 s: mean of N_1 to N_5 attempt wall (includes model load) | 6 x 855.8 s = 5134.8 s | N run records and journal | 1.4263 |
| Evaluation: DECISION NLL passes | 11 passes: 10 trained adapters + 1 base (owner's decision 2026-10-25, rule 9.14c) | UNKNOWN: no DECISION NLL pass timed in calibration | not computed | owner's decision (9.14c) | UNKNOWN (excluded) |
| Controls: S, X and E | 10 passes: one per training run (owner's decision 2026-10-25, rule 9.14c) | 1227 s per pass, PROXY: T attempt 187 s + W attempt 1040 s | 10 x 1227 s = 12270.0 s | journal (T and W attempts) | 3.4083 |
| Retry allowance | 10: INTERPRETATION of section 5 (rerun once; second void stops), one per training attempt; the rule states no total | full attempt, worst case A_on: 256 x 7.98066275799647 s + 239.22 s = 2282.3 s | 10 x 2282.3 s = 22822.7 s | section 5; this record's A_on times | 6.3396 |
| **Total, known components** |  |  | training + loading + evaluation + controls + retry allowance |  | **17.5611** |

Cost = total hours x RATE. RATE: owner to supply. UNKNOWN rows are excluded, so the total is a lower bound. The retry row is the upper bound of the allowance, not an expected cost. Totals here are not the previous series' total: the instance and timings changed.

### 9.15 S controls and "no claim" (decision recorded 2026-10-25)

Decisions, as the owner relayed them from nurik's review: (1) "no claim" stays for E1-E4; (2) mandatory S controls may not be skipped when their cell has no calibrated tau. **Source note:** these two decisions are not in the text of nurik's review on #36 (checked on 2026-10-25: the review body mentions mini_safety and the section 7 proposals only). They are recorded here from the owner's instruction and need a citation before merge.

**Rule text added (new text; inherited lines unchanged):**
- "No claim" applies to E1-E4 only, as the inherited section 5 E paragraph already states.
- A mandatory S control whose cell has no calibrated tau is not skipped and is not a pass. The decision run is INVALID: NO VERDICT (invalid).

**S coverage** (S uses the output tau per cell, section 5). Reachable cells are kind x needs_grad x kernel for the forward output, and kind/backward for input gradients, on the A100 40 GB card.

| Cell group | Reachable | Calibrated tau | Not calibrated |
|---|---|---|---|
| q_proj, k_proj, v_proj, o_proj: fused and fallback, needs_grad False and True | 16 | 16 | 0 |
| gate_proj, up_proj: fused, needs_grad False and True | 4 | 4 | 0 |
| down_proj: fused and fallback, needs_grad False and True | 4 | 4 | 0 |
| gate_proj, up_proj: fallback, needs_grad False and True | 4 | 0 | 4 |
| Input-gradient cells (kind/backward), 7 kinds | 7 | 7 | 0 |

Forward output cells: 28 reachable, 24 calibrated, 4 not. Backward cells: all 7 calibrated. LoRA quantities are not used by S.

**Evidence.**
- Calibrated cells: report.json `tau` (this series, folder `e4-calibration-2026-10-25-b`): 31 keys; no gate_proj or up_proj fallback key. Cell names: `cell_name` at `benchmarks/harness/e4_part2_calibration.py:514`; kernel set `KERNEL_ATTRS` at `benchmarks/harness/e4_part2_gemm_probe.py:21-25`.
- Not calibrated: T run record fc373330 (`result.records`): gate_proj and up_proj calls were all `fused`, at 1 to 70 and 1 to 88 tokens. Attention fallback was seen at 149 to 319 tokens; down fallback at 17 to 150.
- Kernel rule: `bitsandbytes/backends/cuda/ops.py:584-678` (`_gemm_4bit_use_custom_cuda`, sm80 branch). Evaluated with A100 properties (108 SMs, sm80), the function gives gate/up (N=1024, K=2048): fused for M up to 128, fallback for M 129 to 512. Down (N=2048, K=1024): fused up to 16. Attention (N=K=2048): fused up to 128. These thresholds reproduce every kernel observed in the T records. This is an evaluation of the installed function, not a measurement.
- Main-run token range: calibration section 3, max length 512. An expert's M is the number of tokens routed to it, so M above 128 is possible but its frequency in the main data is UNKNOWN.

**Consequence.** A main run that reaches gate_proj or up_proj on the fallback kernel with a mandatory S control on that call is INVALID under the decision above. The calibration does not cover those cells. Whether to recalibrate them is an owner decision; nothing is changed here.

### 9.14c Main-run counts (owner's engineering decision, 2026-10-25)

nurik delegated engineering decisions to me on 2026-10-25 via Telegram. Under that delegation the owner decides:
- one S, X and E control pass per training run (10 passes: 3 A_off, 4 A_on incl. rerun, 3 A_bf16);
- one DECISION NLL pass per trained adapter plus one for the base (M0): 11 passes.

These counts are the owner's decision, not values from the calibration record. The per-pass times in 9.14b are PROJECTED from calibration timings; the NLL pass time is UNKNOWN.

### 9.16 Card change and data sources (for the gate/up run, section 4a)

**Card-change clause** (calibration rule, `e4-part2-calibration-rule.md` line 155), quoted verbatim:

> **Card changes.** All `[CAL]` values come from the final card's instance and are copied into the main rule before it is committed; the main series runs on that instance. If the instance changes after the main rule is committed, the old rule and every attempt are kept unchanged, and a new calibration record and a new preregistration version are committed before any new series. No tolerance in a committed rule is replaced at run time.

**What it forces for a new A100 PCIe 40 GB instance (not 57596627):**
- The first sentence applies now, before the main rule is committed: every `[CAL]` value must come from the final card's instance. A tolerance measured on a different instance cannot be entered in section 4a or in the main rule.
- Because the main rule is not yet committed, the second trigger ("if the instance changes after the main rule is committed") is not yet met. Once it is committed, a new instance requires a new calibration record and a new preregistration version before any new series.
- The clause defines no partial rerun. Its minimum reading is a new calibration record, which is the full calibration on that instance (F with decide, T, W, N, P). A T-only run on a new instance produces values that the clause does not allow to be used. This is an interpretation, for the owner to confirm.
- The final card of this series is 57707996 (`decision.json`). Instance 57596627 is destroyed. This series is a full calibration on the new instance, under the clause.

**Data sources for the run** (`run --stage T` needs `--data-root`, `--split-dir`, `--manifest`):
- `--manifest`: `benchmarks/results/e4-part2-quant/quant-manifest.json` in the repo. Its SHA-256 is `1f3b550d707aa18c098851a1718f1712cdb1162adc28552c5fbbcce8611f0ccb`, checked on this branch.
- `--split-dir`: the split manifests (DECISION, SMOKE, TRAIN) and `split-record.json`. The split is deterministic from Dolly and the tokenizer, and it can be rebuilt on a fresh box (below). Expected SHA-256: DECISION `90787bbf…`, SMOKE `8e13970f…`, TRAIN `ba33e9dc…` (section 9.13). nurik rebuilt all three byte-for-byte (review on #36).
- `--data-root`: contains `source/` (the OLMoE tokenizer) and the quantized weights and codes. The weights and codes are in the private HF dataset `RedBullet12/e4-part2-quant` at revision `2e09044127c7f8fd97ac474d89c2d98144ec7550`; they need a read-only token and are not rebuilt by this command. They are verified against the manifest on the box (rule U4).

**Rebuild the split on a fresh box:**
```
mkdir -p <DATA_ROOT>/source
huggingface-cli download allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --include tokenizer.json tokenizer_config.json special_tokens_map.json --local-dir <DATA_ROOT>/source
python benchmarks/harness/e4_part2_calibration.py split --data-root <DATA_ROOT> --out <SPLIT_DIR> \
  --dataset-revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a
```
Check: `shasum -a 256 <SPLIT_DIR>/*.manifest.jsonl` must match the three hashes above. Not run here; the split needs the network for Dolly and the tokenizer.

### 9.17 Restart disposition and void table (2026-10-25; owner's engineering decision)

- The first calibration attempt on instance 57707996 stopped under section 8: both `F_A_off` attempts voided with `verified.json` missing (no pull record). No measurement was taken. Its records are kept unchanged on the box in `/workspace/e4-calibration-2026-10-25-attempt-1-stopped`.
- The restart ran in a new folder (`e4-calibration-2026-10-25-b`) on the same instance, 57707996, after the transfer pull. This is disclosed, not a reset of the retry limit (9.8).
- Commits: F_A_off, F_A_on, F_A_bf16, T, W, N_1 to N_5, P and the one T_fallback_gateup record all ran on `5ab78624`. No recorded attempt ran on `bbe772db`. The interrupted T_fallback_gateup attempt (`ca44d213`) has no record file and no commit in its journal entry, so its commit is not known.
- Void table (journal of this series, `attempts.jsonl`):

| Run | Attempt | Start to end (UTC) | Outcome | Cause | Commit |
|---|---|---|---|---|---|
| T_fallback_gateup | `fc0f5e1b` | 2026-10-25 23:32:19 to 23:34:53 | void | crash: CalibrationStopped, `cannot reach M=384 from 286 distinct SMOKE rows` | 5ab78624 |
| N_4 | `3cdfb794` | 2026-10-26 00:36:18 to 00:43:29 | void | crash: KeyboardInterrupt (accidental Ctrl-C) | 5ab78624 |
| T_fallback_gateup | `ca44d213` | 2026-10-26 00:43:54 to 01:21:46 (closed by the journal) | void | interrupted: a start with no end (process killed); no record file | not recorded |
| N_4 (rerun) | `636ae958` | 2026-10-26 00:46:31 to 01:00:50 | ok | none | 5ab78624 |

- Counts: the journal has 3 voids (T_fallback_gateup 2, N_4 1) and 0 failed runs. `report.json` lists 1 void (N_4). That report was generated before the void-scope fix (branch `e4-part2-report-void-scope`, commit `f47a859c`), which lists every run in the journal; its void count therefore omits the two T_fallback_gateup voids. Its values for F, T, W, N and P are unaffected.
- The harness stopped the T_fallback_gateup run after its second void (calibration section 8: a second void of the same run stops calibration). The gate/up fallback cells have no measurement in this record.
- Decision needed: whether T_fallback_gateup is rerun, and on which commit. OPEN pending nurik (4a).
