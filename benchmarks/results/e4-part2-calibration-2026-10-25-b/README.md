# E4 part 2: calibration record on A100 40 GB, instance 57707996 (2026-10-25)

Calibration under `e4-part2-calibration-rule.md`, on one vast.ai A100 PCIe 40 GB (instance `57707996`, GPU UUID `GPU-b296fdd0-1f0b-1bb9-ad20-66fb670a37f0`). Decision: `FINAL_A100_40GB`. Values are in `report.json`; this README gives the facts a reader needs to use them.

**This record supersedes #36 (`d1a3ddf1`, instance 57596627) for the main run.** That instance is destroyed. Under the calibration rule's card-change clause (line 155), every `[CAL]` value must come from the final card's instance, so the main rule cites this record.

**Status: the gate/up fallback taus are OPEN, pending nurik.** No `gate_proj` or `up_proj` fallback cell is measured here (see Voids). Main rule section 4a holds the decision.

## Result summary (from `report.json`)

| Item | Value |
|---|---|
| F peak reserved, every arm | 17,846,763,520 bytes (threshold 36,000,000,000) |
| Median s per update (F) | A_off 8.030015620053746; A_on 7.98066275799647; A_bf16 8.152295885025524 |
| tol_W | output 0.2197437286376953; input_grad 0.017418980598449707; A_grad 0.6594619750976562; B_grad 0.08243513107299805; update 0.0004000067710876465 |
| tau | 31 cells (see `report.json` `tau`); gate/up fallback cells absent |
| Noise floor | floor 0.0 on all 8 suites; epsilon = 1/n_items; mini_safety 0.0 on every run |
| P | NLL 1.626105685835482 vs 1.6260342313352905; difference -7.145450019163668e-05; adapters not bit-identical |
| Projected main plan | 5.722281638913684 h, training updates only (9 runs + 1 rerun of A_on, 256 updates) |

## Voids and attempts (journal `attempts.jsonl`, the authority)

| Run | Attempt | Start to end (UTC) | Outcome | Cause | Commit |
|---|---|---|---|---|---|
| F_A_off | `75034fef` | 2026-10-25 21:02:39 to 21:02:39 | void | `inputs: verified.json is missing; the instance was not verified by the transfer pull` (setup, attempt 1) | 5d3d61db |
| F_A_off | `cf497e3a` | 2026-10-25 21:02:46 to 21:02:47 | void | same as above (setup, attempt 1) | 5d3d61db |
| T_fallback_gateup | `fc0f5e1b` | 2026-10-25 23:32:19 to 23:34:53 | void | `cannot reach M=384 from 286 distinct SMOKE rows` | 5ab78624 |
| N_4 | `3cdfb794` | 2026-10-26 00:36:18 to 00:43:29 | void | KeyboardInterrupt (accidental Ctrl-C) | 5ab78624 |
| T_fallback_gateup | `ca44d213` | 2026-10-26 00:43:54 to 01:21:46 (closed by the journal) | void | interrupted: start with no end event; no record file | not recorded |
| N_4 (rerun) | `636ae958` | 2026-10-26 00:46:31 to 01:00:50 | ok | none | 5ab78624 |

- The first two rows are from the attempt-1 journal in `e4-calibration-2026-10-25-attempt-1-stopped/attempts.jsonl`. Both are setup voids with no measurements. The journal for this series has 3 voids (T_fallback_gateup 2, N_4 1) and no failed runs. No provenance gaps.
- T_fallback_gateup stopped after its second void. Calibration section 8: a second void of the same run stops calibration. The gate/up fallback cells have no measurement here.
- Under §8 the two F_A_off voids of attempt 1 stop calibration. The restart is a disclosed infrastructure-only deviation, not a reset of the void count: both voids were a missing `verified.json`, no measurement was taken, and from `5ab78624` the harness refuses that case before any attempt (exit 6). Same basis as the pynvml restart disposition on #36. See #38 §9.17.

**Report undercount.** `report.json` lists 1 void (N_4). It was generated before the void-scope fix (branch `e4-part2-report-void-scope`, commit `f47a859c`), which lists every run in the journal. The report's void list therefore omits the two T_fallback_gateup voids. Its values for F, T, W, N and P are not affected. Regenerate the report with `f47a859c` or later to get a void list that matches the journal; the records in this folder were not changed.

## Commit facts

- F_A_off, F_A_on, F_A_bf16, T, W, N_1 to N_5, P and the one T_fallback_gateup record: harness and soup commit `5ab78624` (checked in each run record's `provenance`).
- Attempt 1 (F_A_off x2, setup voids, `75034fef` and `cf497e3a`): harness and soup commit `5d3d61db`. All restart runs above are on `5ab78624`.
- No recorded attempt ran on `bbe772db`, which carries the pool-based fallback targets.
- The interrupted attempt `ca44d213` has no record file, and its journal entry has no commit, so its commit is not known.

## Attempt 1 on the box (not in this folder)

The first attempt on instance 57707996 stopped under section 8. Both `F_A_off` attempts voided with `verified.json` missing, which means the transfer pull had not run. No measurement was taken. Its records are in `benchmarks/results/e4-calibration-2026-10-25-attempt-1-stopped/` (`attempts.jsonl` and two run records), unchanged from the box. The restart (this folder) ran on the same instance after the pull. A new output folder does not reset the section 8 void count.

## Files

| Path | What |
|---|---|
| `attempts.jsonl` | start and end of every attempt in this series |
| `runs/<attempt_id>.json` | one record per attempt, with box stamps and provenance |
| `decision.json` | the final-card decision (`FINAL_A100_40GB`) |
| `report.json` | every value the main rule copies (see the undercount note above) |
| `SHA256SUMS` | as written on the instance; 28 entries, 12 of them adapter files |
| `rest-of-calibration.log` | the driver output of the restart |
| `../e4-calibration-2026-10-25-attempt-1-stopped/` | attempt 1 on the box: two setup voids, no measurements |

The `adapters/` folder (12 files) is not in the repo. `SHA256SUMS` lists those files, and the archive is kept outside the repo.

**Adapter archive:** https://huggingface.co/RedBullet12/e4-part2-calibration-adapters-b (gated, manual approval), revision `7cd1cf2be5e8491c9e6842879ce493f605fcfa82`; 12/12 hashes verified against `SHA256SUMS` from a fresh download on 2026-10-25.

## Not in this folder

- The split manifests and `split-record.json` used by this run. They were rebuilt on the box (runbook step 3), and their hashes should be the ones in `e4-rule-part2-decision.md` section 9.13. The box-side check output is not here. Confirm before the main series.

## Verification

`shasum -a 256 -c SHA256SUMS` in this folder: all 16 non-adapter entries pass. The 12 adapter entries fail here because `adapters/` is not in the repo; they verify against the archive once it is uploaded.
