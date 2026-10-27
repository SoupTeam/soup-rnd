# E4 main-series harness: tickets

Scope: the harness for the E4 main series (10-run plan, controls, DECISION NLL, suite scoring, report) under `e4-rule-part2-draft.md` §3d, §5-§8, built on the calibration harness committed in `benchmarks/harness/e4_part2_*.py` (calibration record on instance 57707996).

Source: gap table from an earlier session, plus one ticket from nurik (fault tolerance). Hours are rough estimates for one engineer, including tests for that ticket.

## Summary

| # | Ticket | Hours | Blocked by |
|---|---|---|---|
| 1 | Foundation: 256-update training + stage auto-upload + journal resume | 9 | none |
| 2 | DECISION NLL (M0, M1), token-weighted, fp64 sums | 5 | 1 |
| 3 | Strong control S (forward, absmax x 1.5) | 5 | 1 |
| 4 | Wrapper controls X1-X5 (expert ParamWrapper) | 5 | 1 |
| 5 | Controls E1-E5 and per-cell reporting | 4 | 1 |
| 6 | Suite scoring, 6 scorings (A_on and A_off x seeds 42, 7, 123) | 2 | 1 |
| 7 | Paired bootstrap, D, CI, D_s, M3, L, outcome table with sub-labels | 6 | 2, 6 |
| 8 | Orchestration, journal and report for the 10-run plan with controls and NLL | 4 | 1, 2, 3, 4, 5, 6, 7 |
| 9 | Tests, CPU dry run, review | 5 | 8 |
| | **Total** | **45** | |

Order is topological: each ticket's blockers come earlier in the list.

Dependency graph:

```
1 ──┬── 2 ──────────────┐
    ├── 3 ──────────────┤
    ├── 4 ──────────────┤
    ├── 5 ──────────────┤
    └── 6 ──┬── 7 ──────┤
            │           └── 8 ── 9
    (2 also feeds 7)
```

Hard rules that apply to every ticket:
- Calibration behaviour does not change. Calibration record hashes must still reproduce after each ticket.
- The DECISION split (`e4_part2_split.py`, `DECISION = (0, 2048)`) is never tokenized or scored by any calibration path.
- A void is journaled and kept with its record. A second void of the same run stops the study (`AttemptsJournal.assert_may_start`).

---

## Ticket 1: Foundation. 256-update training, stage auto-upload, journal resume

**Hours:** 9 (training 3, fault tolerance 6)
**Blocked by:** none. Calibration record and `[CAL]` tau values are already committed.

**Scope**

Part A, 256-update training (3 h):
- Add a main-series recipe object with `f_updates=256`. Recipe values otherwise as draft §6 (one pass over TRAIN 4,096, microbatch 1 x accumulation 16, lr 1e-4, warmup 16, clip 1.0, gradient checkpointing).
- Do not change the `Recipe.f_updates` default (16, `e4_part2_calibration.py:52`). `run_f` reads `recipe.f_updates` (`:411`, `:426`), so the main series passes the 256 recipe explicitly. The calibration record must still reproduce.
- Run arms A_off, A_on, A_bf16 for seeds 42, 7, 123, plus the A_on/42 rerun. Each run is one process and one journaled attempt.
- Record `updates_requested` and the number of updates actually performed.

Part B, fault tolerance (nurik request, 6 h):
- After each finished stage (its attempt record is written and its journal entry closed), upload immediately to a private HF repo: the attempt record JSON, the full `attempts.jsonl`, and the adapter safetensors. The adapter upload is my choice, not in the source: suite scoring (ticket 6) needs the adapter after a box loss.
- Reuse `transfer.push`/`pull` (`benchmarks/harness/e4_part2_transfer.py`). Upload refuses a non-private repo, as it does now.
- On restart, read the journal. Stages with a completed attempt are skipped via the existing `completed()` (`e4_part2_calibration.py:1389`). Nothing is re-run or re-journaled.
- If the local journal is missing, pull it from the private repo before any attempt. Void counts come from the pulled journal, so a restart cannot reset the §8 void count.
- A stage interrupted mid-run is closed as a void by `close_interrupted`, counted, and re-run in the same position with the same seed.

**Acceptance test**
- A 256-update run logs `updates_requested == 256` and performs 256 updates. Calibration's default recipe still gives 16, and the calibration record hash is unchanged.
- Kill the process after stage 3 finishes but before stage 4 starts. On restart: stages 1-3 are skipped with no new journal lines; stage 4 starts; the void count for every run equals its pre-kill value.
- Delete the local output directory and restart. The journal is pulled from the private repo, and the void count matches the pre-delete value. The uploaded attempt record's SHA-256 matches the local file.
- Kill the process after a stage's journal close but before its upload. On restart, that stage's record is uploaded before any new work begins.
- A push to a non-private repo is refused with no upload.
- Dry-run with `--inject` (`kill`, `oom`, `void`) on one stage: the journal shows the expected event sequence, and a second void of the same run stops the study.

---

## Ticket 2: DECISION NLL (M0, M1)

**Hours:** 5
**Blocked by:** Ticket 1 (adapters and stage records).

**Scope**
- M0: NLL of each arm's base (no adapter) on the 2,048 DECISION rows. Computed after this rule's commit and the §3 checks, before run 1.
- M1: response-target NLL after training, per adapter, on the same rows.
- Per-target loss in fp32 cross-entropy, summed in fp64 per sequence. Store per DECISION row, arm and seed: NLL sum and target count n_i.
- Token-weighted: the reported value is total NLL over total targets.
- DECISION rows are tokenized only in this path. Calibration never tokenizes them (add a test that asserts this).

**Acceptance test**
- fp64 per-row sums match a naive reference loop within 1e-12 relative error on 64 rows.
- n_i equals the count of response targets for each row, checked against the split manifest.
- M0 record timestamp is earlier than the first training run's `box_stamp_before.unix_s`. A mismatch is a run-order error and fails the check.
- Calibration code path: grep and unit test show no DECISION index is tokenized.

---

## Ticket 3: Strong control S (forward)

**Hours:** 5
**Blocked by:** Ticket 1 (stage runner and records).

**Scope**
- Draft §3d. Build a `Linear4bit` from R's packed codes with a plain (non-nested) QuantState whose per-block absmax is R's decoded absmax times **1.5**. With DQ on, decode as `dequantize_blockwise(absmax, state2) + offset` (`functional.py:746-748`).
- Run on the same inputs as 3a, in the same process.
- Detected on a unit if max abs output difference > **10 x tau(cell, output)** on every cell the unit ran.
- Multiplier, factor and all tau values are fixed before S runs. Tau comes from the calibration record.
- S is mandatory. A cell with no calibrated tau gives verdict **INVALID** for the whole study, not a pass.

**Acceptance test**
- Unit test: S on a synthetic unit with a known 1.5x absmax change is detected. S on an unmodified R is not detected.
- A cell with no tau in the calibration record marks the run INVALID (test with a deleted tau entry).
- S passing on all tested units is written to the stage record. Any non-detected unit fails S, and the verdict is "no verdict".

---

## Ticket 4: Wrapper controls X1-X5 (expert ParamWrapper)

**Hours:** 5
**Blocked by:** Ticket 1.

**Scope**
- Draft §3b. Layers 0 and 8, A_off and A_on weights, calibration W inputs (first 32 SMOKE rows), fp32 factors, seed 20261016.
- X1 forward, X2 input gradient, X3 A and B gradients, X4 one AdamW update: each compared to the plain-torch bf16 reference (`baddbmm` with A and B cast to bf16 first, autocast off) against tol_W `[CAL]`.
- X5: SHA-256 of `gate_up_proj` and `down_proj` unchanged after the step, in both arms.
- Report whether wrapper and reference are bit-identical, as well as the pass/fail.

**Acceptance test**
- X1-X5 run on both layers for both arms, and each produces a pass/fail record.
- Unit test: perturbing the wrapper's fold order (fp32 instead of bf16 `baddbmm`) fails X1 or X4. This shows the check can see the known error.
- Any X fail produces "no verdict" in the stage record, not a silent continue.

---

## Ticket 5: Controls E1-E5 and per-cell reporting

**Hours:** 4
**Blocked by:** Ticket 1.

**Scope**
- E0-E4 (draft §3a): CUDA `dequantize_4bit` byte-identical check, forward, input gradient, LoRA gradients, one AdamW update, per cell (kind, needs_grad, kernel).
- E5 (draft §3c): after one full training update in each arm, fused base `.grad` is None and SHA-256 of every base parameter is unchanged.
- Cells with a failing E1-E4 are reported as "not shown equivalent to real 4-bit". They do not void the study.
- Cells with no calibrated tau get no equivalence claim and are reported, not judged.

**Acceptance test**
- Synthetic: a cell with max diff just above tau reports "not shown equivalent" and the study still runs.
- E5 fails (test: force a grad on the base) and the stage gets "no verdict".
- Every call record includes `needs_grad` and `kernel`, both from the CUDA backend, not from inference.

---

## Ticket 6: Suite scoring, 6 scorings

**Hours:** 2
**Blocked by:** Ticket 1 (adapters from every seed).

**Scope**
- Today N scores only seed 42. Extend to A_on and A_off for seeds 42, 7 and 123: 6 scorings.
- Scoring: `DEFAULT_GENERAL_SUITE` through `score_bundled_suite` (`commands/ship.py:782-802, :973-977`), scorer revision 4, greedy, 256 new tokens, no chat template, MCQ cue kept.
- A_bf16 suite scores are not in the 6. Flagged as an open question below.

**Acceptance test**
- Exactly 6 suite result files are produced, one per (arm, seed) pair, with scorer revision 4 recorded.
- Item counts match the fixtures: 40 for each behavioural suite, 26/24/24/40 for the MCQ suites.
- Re-scoring one adapter gives identical per-item outputs (greedy determinism).

---

## Ticket 7: Paired bootstrap and outcome table

**Hours:** 6
**Blocked by:** Ticket 2 (NLL sums), Ticket 6 (suite scores for sub-labels).

**Scope**
- Paired percentile bootstrap, 2,000 resamples, seed 20261015. Each resample draws one multiset I of DECISION row indices, applied to every arm and every seed.
- Per seed: D_s = sum_{i in I}(sum_on,s,i - sum_off,s,i) / sum_{i in I} n_i. D = mean of D_1, D_2, D_3. Point estimate uses all 2,048 rows the same way. 90% CI from the resample distribution.
- M3 (A_off - A_bf16, A_on - A_bf16 on M1) and L (descriptive) per seed and pooled.
- Outcome table, per draft §8: NLL verdict from CI against delta = 0.005 nat, and suite flags (A_on below A_off by more than epsilon(suite) on every seed, strict `>`). The two are separate fields with separate sub-labels, and the table never combines them into "quality preserved".

**Acceptance test**
- Bootstrap with a fixed seed reproduces the same CI on two runs.
- Hand-computed toy example with 4 rows and 2 resamples matches D and CI exactly.
- Each of the four NLL verdict cases is reached by a synthetic input (CI inside, lower bound above, upper bound below, straddling).
- Suite flag set and NLL verdict appear in separate fields. Test asserts no combined "passed" string.

---

## Ticket 8: Orchestration, journal and report

**Hours:** 4
**Blocked by:** Tickets 1-7.

**Scope**
- Run plan from draft §5, positions 1-10 with the 3 x 3 Latin square, each run in a fresh process. Run 10 (A_on/42 rerun) is recorded and excluded from decision statistics.
- Ordering gate: §3 checks (tickets 3-5) and M0 (ticket 2) complete before run 1. Otherwise the orchestrator refuses to start.
- Per run: GPU UUID, instance id, driver and CUDA, SM clock start and end, free host RAM and commit charge, GPU peers and foreign processes, `box_stamp_before.unix_s`, attempt journal. Use the existing `box_stamp` and `validate_run_stamps`.
- Report: `analysis.json`, `report.json`, `SHA256SUMS`, void and rejected runs kept, with the rule's commit hash and calibration values recorded.

**Acceptance test**
- Full dry run (`e4_part2_calibration.py --dry-run`) completes all 10 positions in the Latin-square order. The journal's order matches draft §5.
- Injected void on run 3: rerun in the same position with the same seed. Injected second void of the same run: study stops and is reported.
- Ordering gate: removing M0 from the output blocks run 1.
- `provenance_gaps` is empty in the report. A result with no journal entry produces a gap.

---

## Ticket 9: Tests, CPU dry run, review

**Hours:** 5
**Blocked by:** Ticket 8.

**Scope**
- Unit tests for each new module. Integration test that runs the whole plan on CPU with the fake card (`--fake-card`) and injected failures.
- Lint: `ruff check benchmarks/harness/ tests/` clean (line length 100, rules E, F, I, N, W).
- Code review against draft §3-§8 and the calibration rule. Every `[CAL]` reference resolves to a value in the calibration record.
- Verify the calibration record reproduces (hash check).

**Acceptance test**
- `pytest tests/ -v --tb=short` green on Python 3.10, 3.11 and 3.12 (CI matrix).
- CPU dry run completes with exit 0 and writes a complete report with every stage present.
- Review checklist: each ticket's acceptance test is linked to a passing test in the PR description.

---

## Open items (need an owner decision)

1. **Adapter upload size.** Ticket 1 uploads adapter safetensors per stage (about 0.5 GB fp32 per run, about 5 GB for 10 runs). Chosen so suite scoring survives a box loss. Confirm, or limit upload to records only.
2. **A_bf16 suite scores.** The 6 scorings in ticket 6 leave out A_bf16. Include or exclude?
3. **Ticket 1 is two things.** Per the request, the foundation ticket includes both 256-update training and fault tolerance, so it is 9 h. Splitting it would let training start before the upload path exists.
4. **Hours are estimates.** Ticket 1 fault tolerance (6 h) reuses the existing journal, `drive_run`, `completed()` and `transfer.push`. If that reuse holds, it may be lower.
