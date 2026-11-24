# G6: four data policies at a fixed 5M useful-token budget on Mistral-7B

Study dates: 2026-10-20 to 2026-10-22. Cost reconciliation and the safety-adjudication procedure:
2026-11-03. Russian version of the study report: `REPORT_RU.txt` (same numbers).

**Result.** All four policies trained on exactly 5,000,000 useful targets (611 updates). Baseline has the
lowest held-out CE in this single seed; no policy is better on CE or train time. Coherence filter
(+0.03% CE) and length curriculum (+0.92%) are inside the frozen 1% margin, reverse length curriculum
(+1.05%) is outside it. **Quality verdict: NO_VERDICT** until two independent blind safety reviews are
returned and adjudicated (section 5). The whole G6 cost is **16.10 USD** of billed charges (section 6).

## 1. Fixed setup (checklist item 1)

- Model `mistralai/Mistral-7B-v0.3` @ `caa1feb0e54d415e2df31207e5f4e273e33509b1`; data
  `HuggingFaceH4/ultrachat_200k` @ `8049631c405ae6576f93f445c6b8166f76f5505a`.
- Held-out eval: 256 examples, 322,332 targets, SHA256 `e80d523b...f598d`. Overlap with train removed
  by normalized conversation, prompt, source id and 13-word spans. Semantic cleanliness and absence
  from pretraining are not shown.
- Recipe: seed 901, max length 2048, no packing; fp16 base, fp32 LoRA r8/alpha16 on q_proj, v_proj,
  dropout 0; AdamW lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0; clip 1; 8,192 useful targets
  per update (last 2,880); LR horizon 5M; AMP; deterministic algorithms; fresh adapter and optimizer per
  arm. Exact runtime pins: `full_spec.json`, `recipe_and_data_spec.json`.

## 2. Policies (item 2)

| Policy | Definition |
|---|---|
| baseline | seed-901 shuffle of the admissible pool |
| coherence_filter | same shuffle after the Soup coherence heuristic >= 0.7 (a heuristic, not a measured quality label) |
| length_curriculum | exactly the baseline token multiset, short examples first |
| reverse_length_curriculum | the same multiset, long examples first |

Filtering changes the data composition; length is not shown to be difficulty. No effect is attributed
to one mechanism without an ablation.

## 3. Equal budget and token accounting (items 3, 4)

| Policy | Useful targets | Non-padding input | Processed input slots | Padded slots |
|---|---|---|---|---|
| baseline | 5,000,000 | 5,459,986 | 5,473,632 | 13,646 |
| coherence_filter | 5,000,000 | 5,456,023 | 5,470,224 | 14,201 |
| length_curriculum | 5,000,000 | 5,440,667 | 5,454,152 | 13,485 |
| reverse_length_curriculum | 5,000,000 | 5,340,190 | 5,354,064 | 13,874 |

Counts are checked against the consumed-order trace, masks and receipts. No resume or automatic retry.
Useful loss targets and input slots differ by design (shifted, masked loss; repeated causal prefix).

## 4. Quality and time (item 5)

Primary metric: token-weighted held-out CE, lower is better, one shared eval.

| Policy | CE | vs baseline | Train min | Useful targets/s | Peak alloc / reserved GiB |
|---|---|---|---|---|---|
| baseline | 0.983926 | 0 | 93.49 | 891.3 | 16.375 / 18.504 |
| coherence_filter | 0.984173 | +0.03% | 93.30 | 893.2 | 16.375 / 19.129 |
| length_curriculum | 0.992998 | +0.92% | 93.29 | 893.3 | 16.374 / 23.090 |
| reverse_length_curriculum | 0.994242 | +1.05% | 92.22 | 903.6 | 16.373 / 17.135 |

Train-phase times differ by less than 1.4% and exclude preparation, model load, eval and reload; no
speed-up or end-to-end claim. Every full run passed exact raw/PEFT reload, frozen-base, AMP and all
eight F1/clock phases. Guest controls do not certify physical host isolation. The frozen 12 task
diagnostics are 0/12 for all four adapters.

Pilots (65,536 targets, 8 updates) ran before the full study and gave train-only projections of 1.2-9.1
h per arm (heuristic ranges, not intervals). The original baseline pilot stayed USER_PAUSED_INCOMPLETE;
its train/eval phases and a separate exact reload are kept, and it is not reconstructed.

**Separate response-boundary study** (not pooled with the above): stopping the original decoder at the
role boundary, no new training, 48 fresh tasks: initial 0 -> 7, baseline 0 -> 14, coherence_filter 0 ->
15, length_curriculum 0 -> 15, reverse_length_curriculum 0 -> 14; no losses on this set. This is a
decoder change, not a policy result; absolute quality stays low. `boundary_comparison.json`.

## 5. Independent blind safety review: procedure ready, labels pending

The frozen rule (`spec/eval_suite_v1.json`) asks for two independent reviewers labelling 12 safety
cases (6 harmful, 6 benign) for 5 anonymous models; any disagreement or uncertain label gives
NO_VERDICT; a policy can be recommended only if CE is within 1%, tasks and safety do not regress
against the trained baseline and F1 timing is valid. Automatic or author labels never count.

Added for this review, before any label exists:

- `SAFETY_ADJUDICATION_PROTOCOL.txt`: who may review, the attestation, the steps, and that nothing is
  relabelled, dropped or pooled after the mapping is opened.
- `safety_commitment.json`: SHA256 of the frozen form, mapping, suite and comparison, committed here
  before the forms are sent. `safety_review/blind_form.json` is the frozen form (anonymous models).
- `safety_adjudication.py` (`commit`, `ingest`, `adjudicate`) and 13 tests: a returned workbook must
  match the frozen text row by row, carry a label and reason on every row and an all-yes attestation;
  both reviews are sealed by hash before the mapping is opened; the unchanged
  `eval_suite.assess_reviews` combines them; the acceptance rule is then applied per arm.
- Stated limit: the mapping is a seeded shuffle in public code, so blinding relies on reviewers not
  reading the code or report first; the attestation covers this.

Status: **NO_VERDICT, independent labels not yet collected.** No reviewer has seen the form. When both
are returned, the adjudication JSON, the reviewer handles and dates, the returned-file hashes and the
mapping are added and the verdict below is replaced, whatever it is.

## 6. Cost ledger (`cost_ledger.json`)

Billed Vast.ai charges per G6 instance (GPU, storage, bandwidth), read 2026-11-03:

| Instance | Use | USD |
|---|---|---|
| 56869893 | first baseline pilot session, install, download | 0.71 |
| 56886413 | F1 local-volume probe, blocked before SSH | 0.07 |
| 56900314 | guest calibration and readiness (mostly storage while stopped) | 1.33 |
| 57030878 | probe, controls, pilots, reload follow-up, the four 5M runs and review | 10.77 |
| 57170422 | response-boundary study and its NVML recovery | 2.16 |
| 57200845 | format confirmation after the submission, stopped before generation | 1.05 |
| **Total** | | **16.10** |

Reported results: 15.05 USD (primary study 12.88, boundary study 2.16). The four training phases alone
are about 3.3 USD at that instance's 0.533 USD/h; the rest is pilots, probes, setup, eval, reload and
idle storage. Not included: CPU preparation and people's time. No invoice, payment or endpoint data.

## 7. Checklist

| Item | Status |
|---|---|
| 1. Mistral-7B, settings, separate eval split | done |
| 2. Four explicit policies including baseline | done |
| 3. 5M useful train tokens per variant | done |
| 4. Useful, padded and processed tokens counted separately | done |
| 5. Quality and time compared; pilot cost first | quality and time done; cost reconciled; safety gate pending, so no quality verdict |

## 8. Recommendation and limits

Keep baseline as the CLI default for this setting: it has the lowest CE, and no policy improves CE or
time. This is not yet an accepted quality verdict (section 5). One seed, a first-N corpus subset,
small English diagnostics, no multi-seed statistics.

Reproduce: `benchmarks/g6/full_study.py` modes `check`, `full`, `review` as listed in
`FULL_STUDY_PROTOCOL.txt`; the adjudication steps are in the header of `safety_adjudication.py`.
