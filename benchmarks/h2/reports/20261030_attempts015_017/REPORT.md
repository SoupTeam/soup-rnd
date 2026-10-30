# H2 local work report: Liger CE-only path tests, real-tokenizer check, runner hardening

Date: 2026-10-30. Branch `research/h2-kernel-comparison` @ `13f3ed39ddbfd07d676a4890624ca201edd21e12` (GitHub remote head, confirmed with `git ls-remote`). The source for the local work, attempt015, attempt016 and attempt017 is already committed and pushed (5d0cd345, ddb1995d, da5b7629, c4fd72f3, 0f64caef, 6846ca51, 13f3ed39); only this report is new.
Scope: local H2 work (tests, real-tokenizer check, runner hardening), GPU attempts 015 and 016, and the attempt017 CE micro-test.
Verdict: **attempts 015 and 016 are BLOCKED; performance remains NO_VERDICT; H2 is NOT accepted.** Attempt017 completed its synthetic CE micro-test, but it failed the representativeness gate and does not explain the model-gradient mismatch. Recommendation: stop H2 GPU work here; any fp64-reference model-gradient experiment needs a separate decision and budget.

Costs (displayed Vast dashboard values, rounded, not invoices; recent activity may lag): attempt015 instance 58207641 $0.31; attempt016 instance 58221727 $0.30; attempt017 replacement instance 58360938 $0.32 plus volume 58360936 under $0.01. The earlier attempt017 rental (instance 58302213, volume 58302212) displayed $0.53 + $0.37 = $0.90, reported separately. Total H2 spend is not reconciled: older billing rows (for example 58104557, 58135664) are unverified, and other protected resources still bill at an account-wide displayed rate of about $0.09/hour. Do not read this as the whole account having stopped billing.

## 1. What changed and why (plain language)

**Problem.** GPU attempt014 stopped before its CE-only probes with `KeyError: 'attention_mask'`, after 0 optimizer updates. The cause is in pinned TRL 0.29.1: when Liger is on, it drops `labels`/`attention_mask` from the prepared dataset, and the collator rebuilds them later. Commit `5d0cd345` already fixed the audit so it checks the batches the trainer actually builds.

**Gap that remained.** No test had ever run TRL's *real* Liger-mode dataset preparation; the existing test imitated it by hand. The CE-only *training* path, where TRL adds extra arguments to the model call, had never run anywhere. As things stood, the next paid GPU attempt would have been its first execution, and a failure there would come after most of the money was spent, as in attempt014.

**Change (three pieces, now committed and pushed).**
- **H2-L1:** new `tests/test_h2_ce_real_liger_path.py` (final sha256 `94330d4aa95b7162acec93587c75e834e349626e52f08c1263a30c5aa5494ecf`, 513 lines; it was `daa8348a…`, 506 lines, when first reviewed and was later updated for Stage B).
- **Stage A (H2-L2):** new `tests/test_h2_ce_real_tokenizer_rows.py` (sha256 `20686a65bc9d2dcec249d2d5e2885ca94149f2414d74e97ea90ebf29a582c024`, 151 lines). It checks all 128 frozen rows with the real Llama-3.1 tokenizer files, for both arms.
- **Stage B (runner hardening, D3):** edits to `benchmarks/harness/h2_ce_confirmation.py` and `h2_ce_training.py`, a new prospective addendum `benchmarks/h2/CE_HARDENING_ADDENDUM.txt`, new `tests/test_h2_ce_hardening.py` (41 tests), and one fixture line in `tests/test_h2_ce_confirmation.py`.
- Protected files are byte-identical to `5d0cd345` (`git diff HEAD` is empty): `h2_gpu_correctness.py`, `numerical_policy.json`, `ce_confirmation_workset.json`, `prepared_fixture_candidate.jsonl`, `ce_confirmation_plan.json` and everything else under `benchmarks/h2` that was tracked. Thresholds, workset, fixture, the 2400 s / 600 s / 40 s limits and the $5 check in `validate_contract` are unchanged.

**How it works.**
- Each arm (baseline, CE-only) runs in its own fresh CPU subprocess through Soup's real SFT wrapper and TRL 0.29.1.
- The runner's real `PatchController` drives a stand-in `liger_kernel` module that exists only inside that subprocess.
- In the CE arm, the stand-in replaces `torch.nn.functional.cross_entropy` with a counting function that calls the stock op. That is the same function real Liger 0.8.5 replaces.

## 2. What the tests check, and what they found

| Check | Result |
|---|---|
| Real Liger-mode dataset columns | Exactly `{input_ids, assistant_masks}`; the baseline keeps `labels`/`attention_mask` |
| 5d0cd345 collated-batch audit | Passes on the real rows of both arms (tiny local model and tokenizer) |
| attempt014 bug reproduced on real code | The pre-fix expression raises `KeyError: 'attention_mask'` on the real pruned dataset |
| Audit refusals | Corrupted mask, label or token; reordered, missing or extra row; wrong dtype; shuffled order: all rejected |
| TRL/Soup settings at runtime | Shuffle off, padding-free off, completion-only and assistant-only loss off, `loss_type` nll, `pad_to_multiple_of` None |
| CE-arm training (16 microbatches, 4 optimizer updates, gradient checkpointing ON) | Patch counts 1/1; stand-in hit on 16/16 training calls with `reduction='sum'` and `ignore_index=-100`; TRL's extra arguments reach the model 16/16 |
| Baseline arm | Patch counts 0/0; stand-in hit 0 times; cross-entropy still the stock function |
| Loss denominators (`num_items_in_batch`) | Integers 8, 12, 16, 20 per window, in both arms, equal to independently counted targets |
| Parity | Losses, gradients, initial and final LoRA weights and optimizer state are bit-identical (`torch.equal`) between the arms |
| Negative controls | A stand-in that forces `'mean'` is rejected by the pair review; a patch applied without the stand-in being reached fails the 16/16 guard; a no-op optimizer step and a dropped batch are detected |

Note: the CE arm also produces 16 extra `reduction='none'` calls with no gradients, from Soup's own rewind diagnostic. The guard counts only the gradient-enabled training calls and asserts that the others are all `'none'`.

### Stage A: real tokenizer, 128 frozen rows
- The three tokenizer files were copied read-only from the USB snapshot (no weights, about 17 MB). Their sha256 values equal `H2_SNAPSHOT_RECEIPT.json` (`tokenizer.json` `6b9e4e7f…`, `tokenizer_config.json` `590b9837…`, `special_tokens_map.json` `32f404d6…`); the test re-checks them itself.
- A tiny random Llama with the real 128,256-token vocabulary stands in for the 8B model. The `baseline.json` recipe and the real Soup data path run on the frozen `synthetic_fixture.jsonl`. The audit is compared with the frozen `prepared_fixture_candidate.jsonl` (128 rows).
- Result: both arms pass the audit on 128 rows; CE-arm columns are exactly `{assistant_masks, input_ids}`; a corrupted label (row 64), a corrupted token (row 7) and a dropped row are all rejected.
- The test is skipped unless `H2_TOKENIZER_DIR` is set. It ran with 0 skipped only with the variable set.
- Declared deviations from the recipe: tiny random weights, CPU, quantization none, automatic mixed precision off, `bnb_4bit_quant_storage` dropped, a spy that only makes Soup set `use_liger_kernel=True`, no `PatchController`.

### Stage B: runner hardening (prospective addendum, fresh attempts only)
| Item | What it does now |
|---|---|
| R7 | The child alarm is `min(max_seconds, deadline - now - 40 - 15)` seconds and the child refuses to start if that is 2 s or less. The alarm is cancelled first in both the error path and the final block, before the controls stop and every evidence write, so a timeout can no longer cut off the evidence. A pending alarm becomes a no-op. A tiny residual window (a few bytecodes) is stated in the addendum. |
| R2 | `num_items_in_batch` must be a real Python integer equal to the shifted useful-target count of its four-microbatch window. A missing value, a bool, a float or a string is refused. On the frozen workset every window is 24. |
| R3b | The audit receipt also records the dataset columns and the collator settings (class, pad token, padding-free, completion-only loss, `pad_to_multiple_of`); a missing field fails the audit. |
| R4 | Before training, each of the 16 fixed training batches is checked against the audited prepared row it claims to come from (`validate_workset_link`); the result is stored as `workset_link`. |
Fail-closed negative tests cover each item (`tests/test_h2_ce_hardening.py`, 41 tests). R6 (the clean-tree check ignores untracked files) was handled outside the runner: the GPU box must record `git status --porcelain --untracked-files=all` as empty.
Pinning: no repository file that code reads stores the runner hashes; the per-attempt contract pins them. The next contract must carry the new sha256 of `h2_ce_confirmation.py`, `h2_ce_training.py` and `CE_HARDENING_ADDENDUM.txt`, taken from the new commit. Older local-review receipts record earlier hashes; they are historical and must not be edited.

## 3. Test results

| Run | Coder | Reviewer (independent) |
|---|---|---|
| H2-L1 file, first review | 14 passed, 0 skipped | 14 passed, 0 skipped (36.5 s) |
| All H2 tests, first review | 320 passed | 320 passed, 0 skipped (89 s); baseline before the change 306 |
| Stage A file (`H2_TOKENIZER_DIR` set) | 1 passed, 0 skipped (7.8 s) | 1 passed, 0 skipped (7.6 s); 1 skipped when unset |
| Stage B hardening file | 38 passed (round 1) | 41 passed (round 2) |
| All H2 tests, final, `H2_TOKENIZER_DIR` set | 359 passed (round 1) | **362 passed, 0 skipped** (round 2) |
| `ruff check` / `ruff format --check` | clean | clean; `git diff --check` clean |

Environment: `g3-h2-preparation/h2_cpu_transformers_517/bin/python` (Python 3.11.14), pinned torch 2.14.1, transformers 5.17.0, TRL 0.29.1, PEFT 0.21.2, datasets 4.8.5, tokenizers 0.23.3. `soup_cli` loaded from the H2 checkout's `src`. Runs were offline, with coverage, pytest cache and bytecode off, and all temporary files in a scratch folder. liger-kernel, triton and unsloth are not installed, so **no Liger code ran**.

The earlier "211 passed" figure for 5d0cd345 has no recorded environment or command and is cited as historical only.

## 4. Independent review

- **Mutation checks**, run by the reviewer on a scratch copy; the checkout was never touched. Each one made a test fail:
  - the stand-in never installed;
  - `'mean'` forced;
  - the audit removed;
  - the pre-fix code path;
  - gradient checkpointing off;
  - Liger switched off.
- **Upstream checks**, by the researcher. The stand-in signature and swap points match liger-kernel v0.8.5 `monkey_patch.py:259-300` (fetched from the tag, sha `51a5a0c8…`, the same as the attempt013 GPU readback). The installed TRL `sft_trainer.py` (sha `0c3a1a67…`) and transformers `trainer.py` (sha `1bb62572…`) match their release tags.

| ID | Severity | Finding | Status |
|---|---|---|---|
| L1-1 | Low | The module-level `skipif` silently skips the whole file if the TRL/transformers pins differ | Open, accepted. The claim holds only in the pinned environment; check for "0 skipped" |
| L1-2 | Low | `match="tolerance"` in the `'mean'` negative is broad | Open, accepted. The rejection is shown by the run and mutation M2 |
| L1-3 | Low | Diagnostic calls are told apart from training calls by the gradient flag | Closed, informational |
| L1-4 | Info | `PatchController.verify`, `prepare_capture` and `kernel_audit` are not exercised. An earlier claim that the order/kernel_config guards were not exercised was wrong; they run | Resolved for the stale claim; the rest stays a GPU/probe gate |

**Stage A review (PASS).** The reviewer re-hashed the tokenizer files against the receipt, checked the frozen fixture hash (`b8d9d98d…`, equal to the pins in `ce_confirmation_plan.json` and `manifest.json`), and ran mutations: an audit that always passes makes the test fail, and a one-byte change to a tokenizer file fails the in-test hash check. The deviations declared in the docstring match the code.

| ID | Severity | Finding | Status |
|---|---|---|---|
| A-1 | Low | Without `H2_TOKENIZER_DIR` the 128-row check is skipped; it ran with 0 skipped only with the variable set | Open, accepted |
| A-2 | Low | The recipe is `baseline.json` with quantization overridden; the independence of the data path from weights and NF4 is not proven for the real 8B load | Open, accepted, declared |
| A-3 | Info | Does not test the GPU box's `tokenizers` build beyond the pinned 0.23.3 here | Informational |

**Stage B review (PASS in round 2; round 1 was CONDITIONAL).** The reviewer read the runner diff in full and ran mutations on a scratch copy. Each one made a test fail: R7 alarm ignoring the deadline; R7 no disarm in the final block; R7 disarm leaving the handler armed; R2 accepting a missing value (recorder and reader); R3b collator not recorded; R4 comparison disabled; R4 call deleted from the child; R4 call moved after the dataset replacement. The unchanged-files claim was verified byte for byte.

| ID | Severity | Finding | Status |
|---|---|---|---|
| R-B1 | Medium-low | No test pinned that the child actually calls the R4 link check (deleting the call kept every test green) | Resolved: a new test checks the call and its position |
| R-B2 | Low | The dropped-batch case in the H2-L1 file is now rejected earlier by R2, so that real-trainer test no longer reaches the accumulation-boundary check (still covered by existing unit tests) | Open, accepted |
| R-B3 | Low | R2 is strict: on GPU any non-integer or different `num_items_in_batch` BLOCKS the run (intended and written in the addendum). CPU ints matched; GPU behaviour has not been observed | Accepted risk |
| R-B4 | Low | R7, R3b and R4 receipts run only in the Linux/CUDA child; off-GPU they are covered by helper, signal and code-structure tests only | Accepted, stated |
| R-B5 | Info | Audit assertions in both older tests were updated consistently | Closed |
| R-B6 | Low | The addendum hash changed after hand-off | Resolved: final `d1585d25…` |

No high or medium findings. No runner bug exposed. The reviewer is an independent code reviewer, not a G6 human safety reviewer.

## 5. What this does NOT show (limitations)

- **Wiring only.** Liger's kernels are replaced by a stand-in, so this says nothing about Liger/Triton numerics, CUDA, NF4 or bf16 on the 8B model, kernel attribution, or speed.
- **Tiny model.** The 8B weights are not involved. The 128 frozen rows are checked with the real tokenizer only when `H2_TOKENIZER_DIR` is set (Stage A); the data path is assumed independent of quantization and weights.
- **Only the CE-only arm** is modelled. The other Liger operators failed the earlier attempt013 diagnostic and are out of scope.
- **Fixed training batches.** The frozen GPU training set has a uniform target count per microbatch, so on GPU it still cannot catch a denominator-weighting bug. This CPU test covers that locally only.
- **Not run end to end off-GPU:** the R7 alarm, and the R3b and R4 receipts, are written only in the Linux/CUDA child. Off-GPU they are covered by helper, signal and code-structure tests.
- **Strict integer denominator (R2) has not been observed on GPU.** If the real run passes anything other than the expected integer (24 per window), the run blocks by design.
- **R6** (clean tree including untracked files) is left to the box receipt, not the runner.

## 5b. Attempt015 (GPU, 2026-10-29): BLOCKED, NO_VERDICT

Outcome: attempt015 ran only the two probe phases (baseline and CE-only, 0 optimizer updates) and stopped with `Probe comparison failed: Gradient differs beyond prospective tolerance` on 9 LoRA-B tensors. The training phases never ran. There is no retry or resume inside this cap, no threshold was changed, and the box is destroyed. Cost (displayed Vast dashboard values read by the user, rounded, not invoices): instance 58207641 = $0.31; its volume 58207640 was displayed as under $0.01 earlier. Cap $5.00. Older H2 spend (for example rows 58104557 and 58135664) is still unreconciled. Evidence: `/Volumes/Downloads/Soup-RND-evidence/20261029_h2_attempt015/preserved.tar.gz` (sha256 `2d2c4224…2426`). Full numbers and the fact/inference split: (private analysis notes, not part of this folder). The analysis was read-only and streamed from the archive; nothing was re-run.

What the captured evidence shows (facts):
- Probes 00 and 01 (24 and 48 positions) are bit-identical between arms. Probe 02 (96 positions) fails 9 of 448 tensors (worst: max element 5.9 times its allowance, relative L2 up to 3.3%, lowest cosine 0.99946). Probe 03 (384 positions) fails 1 tensor, on the max-element rule only (1.17 times; relative L2 1.5%). Policy limits: max element 1e-4 + 2% of the tensor's maximum, relative L2 2%, cosine 0.999.
- The differences are diffuse: about 65-76% of elements differ in nearly every LoRA-B tensor of probes 02 and 03, they grow toward the early layers (median relative L2 about 2% at layer 0, about 0.05% at layer 31), have no net bias (about 3e-6), and the CE arm's total gradient norm is higher by only 0.16-0.27%. Losses differ by one fp32 step (9.5e-7) or not at all.
- Arm setup looks right: identical inputs and initial state; patch bindings as designed (CE rebound to `liger_cross_entropy` only in the CE arm); one `liger_cross_entropy_kernel` launch in the CE arm's traced probe and none in baseline, no rope, RMSNorm or SwiGLU kernels; the 128-row audit passed with the new R3b receipt fields; GPU ownership controls passed.

Reading (inference, not proven): bf16-scale noise entering through the CE gradient and amplified by the bf16 backward pass, rather than a semantic CE error (a wrong reduction or denominator would show a scale error, which is absent) or an arm-setup fault. The evidence has no baseline-versus-baseline repeat, so the noise floor, and whether these thresholds are tighter than the noise, cannot be established from it. The policy calls its limits unvalidated screening thresholds and forbids loosening them after a failed arm.

What to say: CE-only correctness was NOT established; performance is NO_VERDICT; do not claim either that Liger CE is wrong or that it passes. Kernel attribution, patch bindings, controls and the Stage B receipts worked on a real GPU. The strict integer denominator and the training phases remain unobserved on GPU. A further experiment would need a separately declared design (noise-floor repeat first) and its own budget decision.

## 5c. Attempt016 (GPU noise-floor series, 2026-10-29): BLOCKED, NO_VERDICT

Outcome: attempt016 ran to `COMPLETE_CE_NOISE_FLOOR_SERIES` (four probe-only children on one box, all exit 0) and the recorded classification is **BLOCKED**. No retry or resume, no threshold change (`numerical_policy.json` unchanged), performance NO_VERDICT. Source commit `0f64caef70b6508fa4b07488ff9b55305e9d7e62`; instance 58221727 and volume 58221726 destroyed by the user; protected resources (58096232, 58096231, 58145733) untouched. Cap 2.00 USD (separate authority). Cost (displayed Vast dashboard values read by the user, rounded, not invoices): instance 58221727 = $0.30; the volume line for 58221726 was not given. Cap $2.00. Older H2 spend (for example rows 58104557 and 58135664) is still unreconciled. Evidence: `/Volumes/Downloads/Soup-RND-evidence/20261029_h2_attempt016/preserved.tar.gz`, sha256 `e3e27a9cc2ed217bc1f09dfa76190b618455119e175541fd8fb038e74a529e9e` (134 files, 583 MB archived). Full numbers: (private analysis notes, not part of this folder).

What the evidence shows (facts):
- Baseline vs baseline repeat and CE-only vs CE-only repeat are bit-identical in all four probes (and pass the unchanged policy). So in these repeats (one repeat per arm, deterministic algorithms off) the run-to-run noise of the probe gradients was zero; that does not show noise cannot occur.
- Every baseline-vs-CE pair fails the policy on probe 02 (9 of 448 tensors) and probe 03 (1 tensor), with the same tensors and the same numbers as attempt015 (global relative L2 1.81% and 1.28%); probes 00 and 01 are bit-identical across arms. Minimum tensor cosine is 0.999456 (about 0.9994). A-vs-C losses are not bit-identical on probes 01-03: they differ in the 7th digit, by one fp32 step (9.5e-7); within-arm losses are identical. And the CE arm's gradient norm is 0.16-0.27% higher.
- The attempt015 and attempt016 gradients are bit-identical to each other for both arms in all four probes (same GPU UUID `GPU-4a68399c…` and boot ID `417aa71d…`, different container; nothing is shown for other hosts).
- NOISE_FLOOR_EQUIVALENT (K = 3, F = 2^-8, fixed before the run, not blind): not equivalent. With zero within-arm noise the threshold is 1.17e-2; probe 02 is at 1.54 times it (not marginal) and probe 03 at 1.10 times it (marginal, within 10%; only probe 03 is marginal). The label can never produce a PASS.
- Setup and controls were clean: patch bindings as designed (one Liger CE kernel launch in each CE child, none in baseline; kernel traces exist for probe_00 only), 128-row audit passed, GPU ownership and F1 controls passed, determinism receipts recorded (SDPA flash/mem-efficient/math/cudnn on, cuBLAS workspace `:4096:8`, deterministic algorithms off, TF32 off).

Reading (inference): the earlier "bf16 noise" explanation (section 5b) is replaced: the difference is deterministic (it did not change between repeats) and most plausibly comes from the one difference between the arms, the cross-entropy function and its gradient, amplified by the bf16 backward pass; the attribution is inferred, not shown. No scale, sign or masking error appears. This is consistent with, but does not prove, a rounding-level difference in the CE gradient.

Not established: semantic correctness over a training run (the training phases and the denominator of 24 were never run on GPU); which arm is closer to the true gradient; any speed or memory claim; behaviour on other hardware. The decision rule was applied as fixed; an alternative K was not substituted after seeing the result.

## 5d. Attempt017 (CE-function micro-test, 2026-10-30): explanatory only, 015/016 stay BLOCKED

- **What ran:** `h2_ce_microtest.py` at source commit 13f3ed39 on a replacement RTX 4090 rental (instance 58360938; same host, GPU UUID and boot ID as attempts 015/016), with its own cap of $0.75. It ran 200 synthetic-logit cases (4 shapes x 5 regimes x 5 seeds x 2 paths) with a PyTorch phase first, then one Liger CE rebind and a Liger phase, against an fp64 reference. Setup took 169 s, the throughput gate said GO, and the series (about 2 min) was COMPLETE: 200 of 200 cases, no VOID reason. Dispatch was at 06:56 UTC, before the latest allowed dispatch time; the series window was 27 min (1,619 s) because 37 min of lease remained. Archive sha256 888d4f01...8881, verified on the Mac before the instance was destroyed. Displayed cost $0.32 (instance 58360938) plus under $0.01 (volume 58360936); rounded dashboard values, not an invoice. The instance and volume are absent from the current inventories (confirmed by the user). The earlier attempt017 rental (58302213 / 58302212) displayed $0.53 + $0.37 = $0.90 and is reported separately; both rentals together are about $1.22 plus under $0.01.
- **Result (facts):**
  - CE-level label: LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC, not marginal. No case has both errors above the A threshold (max Liger error 5.6e-8, about 0.055 A).
  - Representativeness gate: NOT_REPRESENTATIVE, so the explanatory reading is withheld by the pre-registered rule. The high-loss cell showed P-vs-L flips at the 24- and 48-row shapes.
  - bf16 level: 43,548 PyTorch-vs-Liger flips in 178 of 200 cases (about 1.2e-5 of elements); downstream dh relative difference at most 7.07e-5.
  - Repeats: all 200 repeats were bit-identical for both paths (the output fields `p_repeat_equal` and `l_repeat_equal` count non-equal repeats; 0 means none). Torch deterministic algorithms were off, so this is observed repeatability only.
  - The real Liger kernel modified its input in place in all 200 cases (clones made this harmless). Profiler: 8 Liger captures show one `liger_cross_entropy_kernel` event each; 2 PyTorch captures show none.
- **What may be said:** on this GPU with synthetic logits, Liger's CE gradient is comparable to PyTorch's against an fp64 reference at fp32 rounding level, while the bf16-propagated gradients still differ in about 1.2e-5 of elements.
- **What may NOT be said:** that attempts 015/016 changed (both stay BLOCKED; there is no PASS and the policy is unchanged); that the 1.3-1.8% model-level difference is explained (the CE-level dh difference is two to three orders smaller); that Liger is more accurate (the ratio is floor-driven because the PyTorch error is below u in all 200 cases; unfloored E_L/E_P is 0.50-1.63, median 1.04); that this reproduces the real pattern (probes 00/01 are bit-identical); that it holds beyond this GPU and boot.
- **Status:** performance NO_VERDICT; H2 not accepted. Reviewer PASS (private review notes, not part of this folder).

## 6. Remaining gates

1. **Done:** local tests, real-tokenizer check, runner hardening, attempts 015, 016 and 017 (all committed and pushed).
2. **Spend:** per-attempt caps were $5.00 (015), $2.00 (016) and $0.75 (017). Total historical H2 spend is unreconciled (older rows such as 58104557 and 58135664 are unverified).
3. **Attempts 015 and 016 are BLOCKED / NO_VERDICT.** No retry or resume. The thresholds and `numerical_policy.json` were never changed.
4. **Stop recommendation:** stop H2 GPU work here. Open items that would need a separate decision and budget: an fp64-reference gradient for the model probes, and the training phases (never run on GPU).

The retained H2 instance 58096232, volume 58096231 and instance 58145733 were not touched.

## 7. Reproduce

```
cd <H2 checkout>
S=$(mktemp -d)
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=$S/hf COVERAGE_FILE=$S/.coverage \
    H2_TOKENIZER_DIR=<dir with the 3 verified tokenizer files> \
  <H2 CPU env>/bin/python \
  -B -m pytest --no-cov -p no:cacheprovider --basetemp=$S/pt tests/test_h2_*.py -q -rs
# expect: 497 passed, 0 skipped at 13f3ed39 with H2_TOKENIZER_DIR set (without it the Stage A file is skipped)
git status --porcelain   # expect a clean tree
```
`--no-cov` matters: the project's pytest config otherwise writes `.coverage` into the checkout. Use the H2 interpreter explicitly; the Desktop venv's own Python would load transformers 5.19.0 / datasets 5.1.0. The first H2-L1 review figures (14 passed; 320 passed) are from before Stage A and B.

## 8. Evidence and git

Evidence archives (external drive, not in git): `/Volumes/Downloads/Soup-RND-evidence/20261029_h2_attempt015/preserved.tar.gz` sha256 `2d2c4224c93bb4bdf96b96463f89b27a88d8edd171e8ae50d552290676922426`; `.../20261029_h2_attempt016/preserved.tar.gz` sha256 `e3e27a9cc2ed217bc1f09dfa76190b618455119e175541fd8fb038e74a529e9e`; `.../20261030_h2_attempt017_replacement001/preserved.tar.gz` sha256 `888d4f015d2e2968d3a75192f8c31d56e030a5955a4dac0b5f1fe941cfba8881`.

This report is added under `benchmarks/h2/reports/20261030_attempts015_017/` together with `evidence_references.json`.
