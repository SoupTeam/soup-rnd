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
| Card | every run on A100 40 GB (`NVIDIA A100-PCIE-40GB`), instance `57596627`, GPU UUID `GPU-cc9d0159-449a-85d6-42b8-1515e454e9fc` | the calibration decision |

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

Status of this section: a filled draft for review. Values are copied from the calibration record on PR #36 (branch `e4-part2-calibration-record`, head `d1a3ddf1079ec5473dbe64056d109230895d1b33`, file `benchmarks/results/e4-part2-calibration-2026-10-25/report.json`) unless a row says otherwise. Citations move from `d1a3ddf1` to #36's merge commit when it lands. No value was chosen by the author. The conditions in the reviewer's recommendation on #36 (restart disposition and adapter archive) are recorded on #36. The three section 7 proposals are kept exactly as the reviewer accepted them (review 5643944224), with bf16 reference only, inconclusive and invalid as distinct NO VERDICT cases, and a CI entirely below -delta labelled PASS (DQ helps). delta = 0.005 nat, the sign convention, the CI conditions, the metric and the comparison are unchanged from the skeleton.

Scope: fake-quant bf16 write-back plus the PEFT ParamWrapper path, as in section 1. Nothing here is a statement about production 4-bit QLoRA.

### 9.1 Placeholder values

| Item | Value | Source |
|---|---|---|
| Final card | A100 40 GB (`NVIDIA A100-PCIE-40GB`), decision `FINAL_A100_40GB` | report.json `final_card`, `final_card_decision` |
| Instance | `57596627` | report.json `final_card` |
| GPU UUID | `GPU-cc9d0159-449a-85d6-42b8-1515e454e9fc` | report.json `final_card` |
| Run commit: code that produced the values (harness and soup source) | `fe0575329428a9af4e643a70a928174c90d190ae` | record README; run records `provenance` |
| Record commit: source of the values | `d1a3ddf1079ec5473dbe64056d109230895d1b33` | PR #36 head; moves to the merge commit when #36 lands |
| Calibration record commit | `d1a3ddf1079ec5473dbe64056d109230895d1b33` | PR #36 head |
| SHA-256 of the record's `SHA256SUMS` | `21725b8b6fa8692a3a2f3aa262c79c6c89b55d6c1472dbd95b9b8f9284e6d419` | `SHA256SUMS` at `d1a3ddf1079ec5473dbe64056d109230895d1b33` |
| Projected main plan, training updates only | 5.314368934846587 h (9 runs + 1 rerun of A_on, 256 updates each) | report.json `projected_main_plan` |

### 9.2 tau (T)

Cell names follow calibration section 4: `kind/needs_grad=…/kernel=…` for forward cells and `kind/backward` for the input-gradient cell. Skeleton kind names map to report names as `q` = `q_proj`, `k` = `k_proj`, `v` = `v_proj`, `o` = `o_proj`, `gate` = `gate_proj`, `up` = `up_proj`, `down` = `down_proj`. Every kind has 8 contributing units (`contributing_per_kind` in the T run record) and `kinds_without_tau` is empty. LoRA quantities (A_grad, B_grad, update) exist only for attention kinds, as the rule specifies. E0 (CUDA dequantize vs transferred weight) is identical on all 112 comparisons.

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
| mini_common_sense | 24 | 0.8333333333333334 (x5, equal) | 0.0 | 0.041666666666666664 |
| mini_format_json | 40 | 0.925 (x5, equal) | 0.0 | 0.025 |
| mini_instruction | 24 | 0.9583333333333334 (x5, equal) | 0.0 | 0.041666666666666664 |
| mini_mmlu | 26 | 0.7307692307692307 (x5, equal) | 0.0 | 0.038461538461538464 |
| mini_over_refusal | 40 | 0.975 (x5, equal) | 0.0 | 0.025 |
| mini_safety | 40 | 0.0 (x5, equal) | 0.0 | 0.025 |
| mini_tool_call | 40 | 0.45 (x5, equal) | 0.0 | 0.025 |

### 9.5 F memory and time (F)

Peak reserved threshold: 36,000,000,000 bytes (calibration section 3). Every arm is below it, so the final card is the 40 GB instance; no F80 repeat.

| Arm | Peak reserved bytes | Peak allocated bytes | Median s per update | Device max used (NVML) bytes | Device min free (NVML) bytes |
|---|---|---|---|---|---|
| A_off | 17846763520 | 17076301312 | 7.45720755495131 | 18940887040 | 24008785920 |
| A_on | 17846763520 | 17076301312 | 7.577096295077354 | 18940887040 | 24008785920 |
| A_bf16 | 17846763520 | 17076301312 | 7.3511017670389265 | 18940887040 | 24008785920 |

These are the 16-update F runs at the calibration learning-rate schedule (calibration section 3), not main-plan peaks.

### 9.6 Repeatability (P)

A_on, seed 42, repeated once: NLL 1.6256667041687507 vs 1.625419037791071, difference -0.00024766637767958954; adapters bit-identical: False. One paired observation, reported only; it is not a decision input.

### 9.7 Budget (not a schedule for the whole study)

- Training updates only: 5.314368934846587 h, from the median seconds per update of the F runs (A_bf16 7.3511017670389265, A_off 7.45720755495131, A_on 7.577096295077354).
- Not estimated, and no number is given for any of them: model loading; evaluation (M0, M1, M2 suites, M3); controls and validity checks (S, X, E); retries of void runs.

### 9.8 Retry limit

A new output directory does not reset the section 5 and calibration section 8 void count. The first calibration series produced two `F_A_off` voids (`attempt-1-stopped/`, record README). A second series in a new directory does not give those runs a new retry. The second series is accepted only as a disclosed, infrastructure-only deviation, recorded in the record README (restart disposition). It is not an allowance in the rule.

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

1. **Blocked on #36.** The record is not merged. The values cite #36's branch (`d1a3ddf1`). Citations move to #36's merge commit when it lands; the rule is not in force until then.
2. **Resolved: both commits cited** (section 9.1). The run commit is the code that produced the values; the record commit is their source.
3. **Partly open: cell of each main-run call.** The calibration harness records (kind, needs_grad, kernel) per call and assigns the cell with the functions in section 9.12. Proposed rule text is in 9.12. One clause conflicts with inherited text and stays OPEN there.
4. **Resolved: split pinned** by the three manifest hashes (section 9.13). The 11,952 count is an unexplained earlier count that does not affect validity; the reviewer reproduced 11,844 eligible rows and all three manifests byte-for-byte.
5. **Budget.** Observed wall-clock per stage is in section 9.14, labelled observed, not a prediction. Loading, evaluation, controls and retries have no estimate for the main run. **OPEN for nurik: the main-run spending cap.** No cap is proposed here.

### 9.12 Cell of each main-run call (OPEN 3)

Check result: the calibration harness records (kind, needs_grad, kernel) per call, and the cell is assigned in one place:

- Kernel: `GemmProbe` (`benchmarks/harness/e4_part2_gemm_probe.py:40`) records each `gemm_4bit` dispatch; `take()` (`:102`) returns `fused` or `fallback`.
- `linear_forward_backward` (`benchmarks/harness/e4_part2_calibration.py:564`) runs one call with the probe; `info = probe.take(1)` at `:577`.
- `run_t_unit_calls` (`:588`) assigns per call: `kernel = info["kernels"][0]` (`:609`); `cell_name(kind, needs_grad, kernel)` (`:616`, function at `:514`); `backward_cell(kind)` for input gradients (`:622`, function at `:518`). Each record stores `kind`, `needs_grad`, `kernel` and `cell`.
- LoRA quantities use the same probe and `cell_name` (`:693-703`).

No main-run module calls these functions yet (grep: only the calibration harness and the probe module). The main-run harness must import them; this is a requirement, not a completed step.

**Proposed rule text (not in force; for review):** For every main-run call the harness assigns the cell with the functions of section 9.12 (kernel from the GEMM probe, cell from `cell_name`, input gradients from `backward_cell`) and logs the cell on that call's record.

**CONFLICT, not written as rule text.** The instruction "a call whose cell has no calibrated tau is void" conflicts with inherited text: calibration section 4 and skeleton section 5 both say a cell with no tau gets no claim. Inherited text is not changed here. Reviewer to choose: keep "no claim", or make such a call void, which is a new rule version.

### 9.13 Split pin (OPEN 4)

| Manifest | File | Rows | SHA-256 |
|---|---|---|---|
| DECISION | `split/decision.manifest.jsonl` | 2048 | `90787bbf0557d2194f65a7abe4260c2096f4c0237d534040d4e4d7830723453f` |
| SMOKE | `split/smoke.manifest.jsonl` | 256 | `8e13970ff802550a0367ebb94bc5ae88b27a9ad1b2cb24f0bcb70b4d4a03b55e` |
| TRAIN | `split/train.manifest.jsonl` | 4096 | `ba33e9dc09a4b37bf85f9d10c05071e5a03ccf412c917bc3ccfd3446d6e7c047` |

Dataset `databricks/databricks-dolly-15k` at revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`. Eligible rows: 11,844 (`split-record.json` `funnel.eligible`), above the 6,400 floor. The earlier pre-commit count of 11,952 (`pre_commit_eligible`) is not explained. It does not affect validity, because the current split follows the rule. The reviewer rebuilt the split from the Dolly revision and tokenizer and reproduced 11,844 eligible rows and all three manifests byte-for-byte (nurik's review on #36, 5643944224). Source: record split folder at `d1a3ddf1`.

### 9.14 Observed wall-clock on calibration (OBSERVED ON CALIBRATION, not a prediction)

Source: `attempts.jsonl` start and end timestamps (UTC) and each run's `seconds_per_update`, at record commit `d1a3ddf1`. Stage labels follow the calibration stages. Loading and evaluation are not timed separately in the record, so they sit inside the attempt wall-clock columns.

| Run | Stage | Start (UTC) | End (UTC) | Attempt wall s | Training updates | Training s (sum of per-update) | Wall minus training s |
|---|---|---|---|---|---|---|---|
| F_A_off | Training, 16-update F arm | 02:46:53 | 02:51:27 | 274 | 16 | 118.9 | 155.1 |
| F_A_on | Training, 16-update F arm | 02:51:34 | 02:56:13 | 279 | 16 | 121.8 | 157.2 |
| F_A_bf16 | Training, 16-update F arm | 02:56:20 | 03:00:49 | 269 | 16 | 117.3 | 151.7 |
| T | Control: base-weight tolerances (T) | 03:00:57 | 03:02:49 | 112 |  |  |  |
| W | Control: expert wrapper (W) | 03:02:57 | 03:15:06 | 729 |  |  |  |
| N_1 | Evaluation: suite noise floor (N, one of 5 scorings) | 03:15:14 | 03:27:55 | 761 |  |  |  |
| N_2 | Evaluation: suite noise floor (N, one of 5 scorings) | 03:28:03 | 03:40:51 | 768 |  |  |  |
| N_3 | Evaluation: suite noise floor (N, one of 5 scorings) | 03:40:59 | 03:53:40 | 761 |  |  |  |
| N_4 | Evaluation: suite noise floor (N, one of 5 scorings) | 03:53:48 | 04:06:25 | 757 |  |  |  |
| N_5 | Evaluation: suite noise floor (N, one of 5 scorings) | 04:06:32 | 04:19:08 | 756 |  |  |  |
| P | Training, 16-update repeat (P) | 04:19:14 | 04:23:32 | 258 | 16 | 112.1 | 145.9 |

| Aggregate (observed) | Attempt wall s |
|---|---|
| Training: F x3 and P (16 updates each) | 1080 (training part 470.1 s; rest 609.9 s) |
| Controls: T and W | 841 |
| Evaluation: N x5 | 3803 |
| All attempts (journal sum) | 5724 |

Totals: first attempt start to last attempt end is about 1 h 38 min on the calibration box. The F and P rows are the 16-update runs at the calibration schedule, not main-plan runs; nothing here is a projection.
