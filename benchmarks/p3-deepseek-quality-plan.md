# P3 DeepSeek-V3 quality pair: BF16 baseline vs FP8 block64 + Oracle (Implementation Plan)

> Implementation history and phase-3 design. Current prelaunch authority, commands and acceptance are in the [review guide](p3-deepseek-quality-prelaunch-review.md) and [pilot runbook](p3-deepseek-quality-pilot-runbook.md). Historical task checkboxes below are not permission to rerun scientific qualification.

**Goal:** Train one BF16 baseline LoRA and one FP8-candidate LoRA on the real DeepSeek-V3 671B with the Granite study's training data, then compare their quality on the Granite v2 evaluation tasks with the Granite v2 decision rule.

**Architecture:** The two arms differ only in how L2L stores layer-boundary activations. The baseline keeps them in BF16 (`l2l_activation_codec: none`). The candidate stores block64 E4M3 and replays its original forward routes in backward with the DeepSeek Oracle. Checkpoint, NF4 cache, initial adapter, data order, optimizer, machine and software are shared. Both arms and both lengths use strict EFFICIENT_ATTENTION. New BF16 gradient and memory qualification runs on approved Ampere-or-newer Vast hardware, not Kaggle T4. Full-model training still requires a measured pilot and a frozen complete protocol.

**Tech stack:** Python 3.12 and the pinned torch/transformers/peft/bitsandbytes stack, recorded again by each executed environment's pip freeze. Kaggle remains available for data work and historical MATH evidence. New native BF16 efficient-attention qualification uses the authenticated Vast API on Ampere+. A Vast CLI dependency is not required.

**Spec:** `C:\Users\sydyk\Downloads\P3_×50_к_скорости_обучения_гигантских_моделей_на_8_ГБ.md` (cited below as P3:line). Study design source: `benchmarks/p3-stage2-v2-plan.md`, `benchmarks/results/p3-stage2-v2/p3-stage2-v2-protocol.json`, `benchmarks/results/p3-stage2-quality/frozen-study-protocol-v1.json`. Full-model path source: `benchmarks/b6-giant-moe-20261028.md`.

## Current prelaunch state

The full-model pilot is a separate diagnostic, pending team-lead code review and
the user's price/cap/deadline approval. The USD90/96-hour envelope is a proposal,
not an authorization. Previous small-smoke and USD40 approvals do not extend it.
Preparation, strict attention, data, gate, paired training, resource timing and
scoring code are now present. This revision has no full-model execution receipt.

The historical tiny E3 remains **FAIL: 0.05028628354277727 >0.05**. Do not rerun
its gate or replace its result. Stage c runs only the portable technical smoke.
The approved order is a→b→c→d; any gate refusal stops the pilot, preserves/uploads
evidence and deletes the instance. There is no BF16-only fallback or retry for PASS.
Only a full-model PASS admits e/f. Phase3 remains blocked on explicit approval of
the later frozen protocol SHA256 and its separate budget.

## Historical handoff snapshot (2026-11-10, before implementation)

| Item | State | Evidence |
| --- | --- | --- |
| Tiny native DeepSeek, NF4 noDQ, BF16 compute, block64 E4M3 + Oracle | Passed on Kaggle: 121 tests, k=1/8, 2 SGD updates each, 36/36 Oracle VJPs | `benchmarks/results/p3-deepseek-fp8-smoke/verification-v3.json` |
| DeepSeek selected-sigmoid Oracle | Research harness, tiny models only | `benchmarks/harness/p3_deepseek_oracle.py` (`DeepseekOracleProbe`) |
| Full-checkpoint runner | **Missing.** The smoke CLI accepts only `--device`, `--output`, `--learning-rate` and builds a synthetic 3-layer model | `benchmarks/harness/p3_deepseek_fp8_smoke.py:877-893` |
| Real DeepSeek-V3 load, NF4 cache, 2 LoRA steps | Done once (B6) on RTX 5060 8 GB, 64 tokens per step, 486 s and 379 s | `benchmarks/b6-giant-moe-20261028.md:188-191` |
| FP8 on tiny DeepSeek without Oracle | Failed: 75/96 windows refused for routing changes, 18 of the other 21 failed G4 | `docs/performance-and-quantization.md:1030-1040` |
| Granite quality study (v2) | `INCONCLUSIVE_UNINFORMATIVE_FLOOR`. All 10 BoolQ cells below floor, ARC L1024 seed113 U=0.0106 > 0.01 | `benchmarks/results/p3-stage2-v2/results-v2.md` |
| Branch | `feat/fp8-activation-storage` at `54be2013`, 7 ahead and 30 behind `origin/main` (B1.4 #58/#59 merged there). The smoke and Oracle harnesses are untracked | `git status`, `git rev-list` |
| Kaggle GPU quota | 2.42 h left, refresh 2026-11-14T00:00Z | `benchmarks/results/p3-deepseek-fp8-smoke/resource-closeout.json` |

## Decisions already made (do not reopen)

1. **Precision.** Both arms use BF16 compute. Baseline boundary storage is BF16, candidate is block64 E4M3 + Oracle. FP16 is not part of this study. P3 never mentions FP16, and P3:179 names the comparison "FP8 vs BF16". The trainer also refuses FP16 with L2L (`src/soup_cli/trainer/l2l_trainer.py:132-136`) and FP8 storage accepts only FP32/BF16 compute (`:185-186`).
2. **Checkpoint.** `deepseek-ai/DeepSeek-V3` at revision `e815299b0bcbac849fa540c768ef21845365c9eb`, the P3 target and the B6 baseline. The Kaggle R1 Input (`deepseek-ai/deepseek-r1/transformers/deepseek-r1/2`) is used only for code checks on Kaggle.
3. **Weights and adapter.** NF4 without double quantization, `expert_layout="per_expert"`, disk tier, as in `benchmarks/harness/b6_real_moe.py:144-204`. LoRA on the five MLA projections (`MOE_TEXT_LORA_TARGETS["deepseek_v3"]`: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`), rank 2, alpha 4, dropout 0, FP32, nonzero-B init with scale 0.02 as in the Granite protocol. One initial adapter file is written once and loaded by both arms.
4. **Seeds and lengths.** One seed, 23. Two context lengths, L=1024 and L=4096, with N=4096 input tokens per optimizer step (B=1, k=4 and k=1), as in Granite. N is the tokens-per-step quantity of P3:134. L is the sequence length of one micro-batch. Keep the two separate in every record. The P3 speed sweep (64-token micro-batches, N from 512 to 8192) is a different experiment and is out of scope.
5. **Training data.** The same 7801 Granite source rows (`benchmarks/results/p3-stage2-quality/admission-kaggle-v3/evidence/dataset/training-sources.jsonl`, SHA-256 `8ab0bdfdefde056387835fcaf38914fe2c6879cde1cf89a8f6169629630675a7`, GSM8K-train + MBPP-train), re-tokenized with DeepSeek. Use the frozen Granite `_domain_stream` fixed-order cycling: record `cycle`, `source_order_position`, `domain_document_number`, and cut the final document at 131072 tokens per domain. Set `exhaustion_policy="cycle_fixed_order_record_identity_cycle_and_final_cutoff"`. No BOS, `add_special_tokens=False`, one appended native EOS per document. The user explicitly approved this correction after Kaggle measured 33116 unique code tokens: roughly four MBPP-train passes, identical in both arms.
6. **Optimizer.** As Granite: AdamW lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0.01, clip 1.0, HF cosine with 2 warmup steps, horizon equal to the frozen step count.
7. **Evaluation.** The v2 tasks and scoring unchanged: primary HellaSwag, ARC-Easy+Challenge, BoolQ, LAMBADA; secondary GSM8K-v2 and MBPP-test NLL. Task data from `benchmarks/results/p3-stage2-v2/data-build-kaggle-v1/evidence/data/manifest.json` (file SHA-256 `07f3c65cce76d0b209ce20aff7b096857077ac906156ac4ecf2c7955141b2b81`). These tasks are text-level, so the DeepSeek tokenizer applies at scoring time.
8. **Decision rule.** The v2 rule unchanged: per-cell paired Clopper-Pearson via `p3_stage2_statistics.summarize_pairs`, one-sided alpha 0.025, union bound, margin 0.01 absolute, then `p3_stage2v2_decision.apply_informativeness_gate` with floor offset 0.05 and the v2 chance values (HellaSwag 0.25, ARC 0.2501597, BoolQ 0.6217125, LAMBADA 0). Required primary cells: 4 endpoints × kept lengths × seed 23.
9. **Where code runs and rental approval.** Local edits, packaging and hashing only; no local project execution. Remote CPU verification is separate from native BF16 qualification, which requires approved Ampere+ Vast hardware. T4's EFFICIENT_ATTENTION refusal is established evidence, not a reason to rerun it. The small RTX3060 smoke authorization is historical and does not authorize a new rental or 671B download. The full-pilot offer, spending limit and deadline require separate approval. Track costs manually; do not implement rental/budget/instance-protection infrastructure.
10. **No base-model evaluation.** Compare only the two trained arms, as in Granite v2. Do not execute Task3.2 Step3 or compute base-model development accuracy/NLL. Record `include_base_model_evaluation: false` in the frozen protocol. Report both step-loss trajectories and the Frobenius norm of the adapter change from the shared initial file to each final adapter.
11. **Approved attention correction.** Main GA/repeat, BF16 L2L, FP8/Oracle and int4 control use `sdpa_kernel([EFFICIENT_ATTENTION])` without fallback. Task1.3 uses MATH only for its alternative noise floor on8×64 tokens. Enforce `torch.use_deterministic_algorithms(True, warn_only=False)` and record actual forward/backward operators, CUDA kernel names, Q/K/V shapes and flags. Confirm MLA q/k192, v128 for both lengths and arms. Backend protocol is `benchmarks/results/p3-deepseek-quality/attention-protocol-v1.json`, SHA256`fc887c5a996bd58536ef0325b61405cd73a2a83cf363fcc4156204ab180e1ed6`, fixed before new-backend outcomes. Bind it into the later complete study protocol and resume/gate identities. Data, L/B/k, seed, optimizer and quality criteria do not change.
12. **Fit interpretation.** This quality pair has no mandatory8GB target. Its L4096/B1 microbatch is not P3's64-token microbatch configuration. An OOM preserves the measured peak and unchanged L/B/k; propose the next cheapest native-BF16 card, never shrink the workload to make it fit.

## Global constraints

- AGENTS.md: Pydantic schema is the config source of truth, heavy imports inside functions, `rich` Console output only, path containment with `os.path.realpath` + `os.path.commonpath`, line length 100, ruff `E, F, I, N, W` clean.
- New behavior goes in new modules `benchmarks/harness/p3_deepseek_quality_*.py` with tests `tests/test_p3_deepseek_quality_*.py`. Do not edit files hash-pinned by the Granite v1/v2 protocols, any existing evidence, or the smoke harnesses after Task 0.1 commits them. If a reused private helper is Granite-bound, wrap it in the new module instead of editing it.
- Evidence is immutable: fresh output directories, exclusive file creation, manifests written last, failures preserved, no retry-until-success, no answer or checkpoint selection.
- No quality outcome on a trained DeepSeek checkpoint before Task 2.2 freezes the protocol and records its SHA-256. Pilot measurements use development splits and the untrained base model only.
- The RoutingGuard stays enabled and G4 thresholds stay unchanged. Oracle-forced route agreement is never reported as natural routing stability.
- A routing refusal, missing Oracle VJP, non-finite loss or gradient, OOM, spend cap or deadline stops that arm with evidence. Only an infrastructure loss (instance death) may resume, from the last complete checkpoint, with the event recorded.
- Memory is reported per resource: sampled RSS, cgroup `memory.peak` when present, CUDA allocated and reserved peaks, boundary-store host bytes. These are never summed.
- Speed figures from this study are descriptive. They are not the ×50 verdict, which needs its own pre-committed rule (P3:143-167).
- Implement backend selection and bounded executed-kernel evidence in the shared `benchmarks/harness/p3_deepseek_quality_attention.py`. Training and gate consume that same implementation. Preserve frozen Oracle/codec/smoke helpers. Move native gradient, memory and resume integration tests to Ampere CUDA; pure data/serialization/arithmetic tests may remain CPU. Reject old MATH-main receipts for the new policy.

---

## Phase 0: branch

### Task 0.1: Research branch on a verified base

**Files:**
- Create branch `research/p3-deepseek-quality` in its own worktree.
- Commit as-is: `benchmarks/harness/p3_deepseek_fp8_smoke.py`, `benchmarks/harness/p3_deepseek_oracle.py`, their tests, `benchmarks/results/p3-deepseek-fp8-smoke/`.

**Interfaces:**
- Consumes: `verification-v3.json` `source_files_sha256`.
- Produces: the base commit every later task builds on, recorded in the ledger.

- [ ] **Step 1:** Create the branch from `feat/fp8-activation-storage` HEAD. Commit the untracked smoke files. Their SHA-256 must equal `source_files_sha256` in `verification-v3.json`.
- [ ] **Step 2:** List the files changed by `origin/main` since `54be2013` that the runner will import: `src/soup_cli/trainer/l2l_trainer.py`, `src/soup_cli/trainer/stream_setup.py`, `src/soup_cli/utils/l2l_step.py`, `src/soup_cli/utils/layer_stream_runtime.py`, `src/soup_cli/config/schema.py` and their dependencies.
- [ ] **Step 3:** If none overlap, keep the base and reuse the v3 receipt. If any overlap, merge `origin/main`, resolve conflicts, then rerun the tiny smoke and the 121 tests listed in `benchmarks/results/p3-deepseek-fp8-smoke/kaggle-v3/deepseek-smoke-evidence/tests.xml` on Kaggle into a fresh evidence directory.
- [ ] **Step 4:** Start the ledger `.superpowers/sdd/p3-deepseek-quality-plan/progress.md` with the base commit and the decision from Step 3.

**Acceptance:** a committed base with a passing tiny smoke receipt bound to it.

---

## Phase 1: code, native BF16 qualification on approved Ampere+ Vast

### Task 1.1: DeepSeek training pack

**Files:**
- Create: `benchmarks/harness/p3_deepseek_quality_data.py`
- Test: `tests/test_p3_deepseek_quality_data.py`

**Interfaces:**
- Consumes: `training-sources.jsonl` (hash above), the document-encoding rules of `p3_stage2_dataset._encode_document(tokenizer, row)` and the window rules of `p3_stage2_dataset._pack`. The tokenizer files of DeepSeek-V3 at the pinned revision, hashed. No weights.
- Produces: CLI `python -m benchmarks.harness.p3_deepseek_quality_data --sources PATH --sources-sha256 HEX --tokenizer-dir PATH --output NEW_DIR`. Output: windows for L=1024 and L=4096 and `manifest.json` written last with tokenizer identity, row usage, supervised-target counts per length and file hashes.

- [ ] **Step 1:** Write tests on a small synthetic source file with a real fast tokenizer. Cover: 64 windows of exactly4096 input tokens per length; alternating math/code windows; identical flat streams; response-only labels; one EOS per document; every1024-token boundary ignored; deterministic repeated builds and cycling parity with frozen Granite on the same inputs; refusal of mismatched tokenizer bytes/revision, malformed or empty domains, and corrupted cycle/order/document numbers. Usage keys are `(source_ref,cycle)`; record per-domain cycle counts and available/used unique-source counts. Validate `cycle == domain_document_number // len(order)`, the order position modulo its length, contiguous spans and exactly131072 covered tokens per domain.
- [ ] **Step 2:** Run the tests on Kaggle CPU. Expected: FAIL, module missing.
- [ ] **Step 3:** Reuse the existing pure encoding and cyclic window logic. Do not copy Granite's tokenizer pin (`p3_stage2_dataset._load_tokenizer` requires Granite ids). Do not reject a domain merely because it has fewer unique tokens than the allocation.
- [ ] **Step 4:** Tests and ruff pass on Kaggle.
- [ ] **Step 5:** Build the real pack on Kaggle CPU and publish it as a private Kaggle dataset. Record the manifest hash in the ledger.

**Acceptance:** real pack manifest hash recorded. Supervised-target counts per length are reported next to the Granite counts (179,326 per arm) for reference.

### Task 1.2: Full-model two-arm training runner

**Files:**
- Create: `benchmarks/harness/p3_deepseek_quality_train.py`
- Test: `tests/test_p3_deepseek_quality_train.py`

**Interfaces:**
- Consumes: the B6 build path (`build_meta_skeleton`, `shard_checkpoint(..., quant="nf4", double_quant=False, expert_layout="per_expert")`, `inspect_shard_cache`, `build_streamed_model(..., tier="disk")`) as used in `b6_real_moe.py:144-204`. The smoke's wiring of the production L2L step, FP8 store, `DeepseekOracleProbe` and RoutingGuard (`p3_deepseek_fp8_smoke.py`, `_build_candidate`, `_stepper`, `_apply_update`, `_run_candidate`). The pack from Task 1.1.
- Produces: CLI `python -m benchmarks.harness.p3_deepseek_quality_train --source DIR --cache DIR --pack DIR --pack-sha256 HEX --initial-adapter FILE --arm {bf16_l2l,oracle_block64_e4m3} --length {1024,4096} --steps N --output NEW_DIR [--resume DIR]` and a subcommand that writes the shared initial adapter once from seed 23. Output per run: `steps.jsonl`, step-1 gradients, per-step checkpoints, final adapter saved through the production save guard, `manifest.json` last.

Per-step record: actual input and target tokens, loss, gradient norm before clipping, learning rate used, adapter digest chain, step seconds, tokens/s, sampled RSS peak, cgroup `memory.peak` if present, CUDA allocated and reserved peaks, boundary-store host bytes, expert fetch counts and logical bytes, routing-guard outcome, and for the candidate the Oracle forward/recompute/VJP counts.

- [ ] **Step 1:** Write failing tests on a tiny V3 checkpoint saved with `save_pretrained` and sharded through the same path, as `tests/test_b2_deepseek_v3_stream.py` does. Cover: both arms run 3 steps at k=4 and k=1; both load the same initial adapter hash; the candidate's boundary store holds about half the baseline's bytes plus scales; interrupting after step 2 and resuming gives the same final digest as an uninterrupted run under deterministic flags; a routing refusal, a missing Oracle VJP and a non-finite loss each stop the arm with `FAILED` evidence; a resume with a different pack, arm, length or initial adapter is refused.
- [ ] **Step 2:** Run native numerical tests on approved Ampere+ Vast. Pure data/arithmetic tests may run on CPU remotely. Preserve failing-before evidence.
- [ ] **Step 3:** Implement. Checkpoint after every step (adapter, optimizer, scheduler, RNG, digest chain, pack cursor). Follow the resume semantics of merged #55 where they apply.
- [ ] **Step 4:** Tests and ruff pass on approved remote hardware. Publish one tiny end-to-end evidence directory per arm.

**Acceptance:** Ampere test receipt plus tiny evidence for both arms, both lengths and exact resume.

### Task 1.3: One-step gradient gate (P3 accuracy requirement)

P3 requires two checks on the real model before any quality claim. P3:165: one L2L step at k=8 matches ordinary accumulation within the A3 tolerance. P3:179: FP8 and BF16 gradients on one step agree by G4 E1-E3. This task builds that gate. It runs in the pilot.

**Files:**
- Create: `benchmarks/harness/p3_deepseek_quality_gate.py`
- Test: `tests/test_p3_deepseek_quality_gate.py`

**Interfaces:**
- Consumes: the model build of Task 1.2; the first 512 tokens of pack window 0 at L=1024, cut into k=8 micro-batches of 64 tokens (the B6 micro-batch size), boundary masks kept; the A3 L2L-versus-accumulation tolerance rule of PR #34, as rerun in `benchmarks/results/p3-bf16-bounded-investigation/README.md`; G4 thresholds from `benchmarks/probe-g4-l2l-activation-compression.md:291-293` and `g4_rule.py`.
- Produces: CLI with the same source/cache/adapter arguments. Three passes from the shared initial adapter, no optimizer step: (a) ordinary streamed accumulation, (b) BF16 L2L, (c) FP8 + Oracle L2L. Verdicts: A3 for (b) vs (a), with the BF16 floor measured on this GPU; E1 (every layer rms row error ≤ 0.05 and worst row ≤ 0.25), E2 (one-recompute rms ≤ 0.10) and E3 (global relative L2 ≤ 0.05 and minimum per-tensor cosine ≥ 0.99) for (c) vs (b).

- [ ] **Step 1:** Write failing tests on the tiny V3 checkpoint. Cover: (b) equals (a) bitwise in FP32; every E metric is finite and present for every layer and tensor; an int4 control fails E1, E2 and E3 (the G4 V5 discrimination check); a missing layer or NaN metric reads no verdict, never pass.
- [ ] **Step 2:** Run on approved Ampere+ Vast. Preserve failing-before evidence.
- [ ] **Step 3:** Implement by reusing the G4 metric code, not a new metric.
- [ ] **Step 4:** Tests and ruff pass on approved Ampere+ Vast.

**Acceptance:** Ampere receipt with main EFFICIENT and alternative MATH kernel evidence. The real-model verdict comes from Task 2.1.

### Task 1.4: Streamed-model scorer

**Files:**
- Create: `benchmarks/harness/p3_deepseek_quality_eval.py`
- Test: `tests/test_p3_deepseek_quality_eval.py`

**Interfaces:**
- Consumes: the frozen v2 tasks and scoring rules, the DeepSeek tokenizer, and a hash-admitted final trained adapter. Production base-model evaluation is forbidden.
- Produces: `run-endpoint --data-root DIR --endpoint NAME --start I --stop J --source DIR --cache DIR --adapter DIR --adapter-sha256 HEX --protocol PATH --protocol-sha256 HEX --output NEW_DIR`. Pair final trained BF16 and FP8+Oracle rows with identical task IDs.

The existing `score_requests` refuses hooks and replaced forwards, and the streamed runtime replaces forwards. Admit the streamed runtime explicitly, record its identity, and keep every scoring rule identical. How batches are scheduled through streamed layers is a performance choice for the pilot, not part of this task.

- [ ] **Step 1:** Qualify tiny streamed scores against the native resident scorer, shard coverage, wrong-adapter rejection and production base-evaluation refusal. Fixture-only comparisons may use untrained models; production may not.
- [ ] **Step 2:** Run native GPU qualification on approved Ampere+ Vast. Preserve failing-before evidence.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Tests and ruff pass on approved remote hardware.

**Acceptance:** Remote receipt with scorer parity, frozen attention-policy admission and strict per-forward EFFICIENT evidence. Reset bounded observation per forward so variable-length shards do not exceed a shard-wide signature limit.

### Task 1.5: Rental controller cancelled by the user

Do not implement automated budget enforcement, instance-ownership infrastructure, or provider supervision. Use direct Vast API calls for this experiment only after the new pilot approval. The earlier USD40 budget is not a current full-pilot grant. Never touch instance60034978 or any unrelated instance.

Record the selected offer, created instance ID, executed commands, evidence and final teardown. Do not build a reusable rental subsystem. The user explicitly prioritizes correct activation-memory measurements.

### Task 1.6: Real-weight check on Kaggle (team-lead request)

The bounded Kaggle check uses mounted R1 weights before the full Vast experiment. The working mount has about20.86GB free, but `/tmp` has about1TB. Full-cache conversion alone took2568.94seconds in the B6 reference, exceeding this check's1200-second stage budget.

- [ ] **Step 1:** In the first Kaggle GPU job of this plan, record `df` for `/kaggle/working` and `/tmp` and the R1 Input mount (NFS, read-only, 163 shards, config SHA-256 `4422016f…` in `benchmarks/results/p3-fp8-kaggle-20261105/`).
- [ ] **Step 2:** Record disk and time admission separately. Use a byte-identical prefix through the first native MoE layer, with every expert and the original hidden width retained, for one step per arm. Record that this checks the implementation, not full-model fit or quality. Full671B execution remains onVast.

**Acceptance:** one of the three outcomes recorded with evidence.

---

## Phase 2: measured pilot and protocol freeze

### Task 2.1: Pilot rental

**Requires:** Tasks 1.1-1.4 accepted, and the user's approval of the full-pilot offer and spending cap. Task1.5 was cancelled; the small Ampere smoke approval does not authorize this pilot.

- [ ] **Step 1:** Show the user the selected offer, hourly price, pilot cap and stage list. Start only after approval.
- [ ] **Step 2:** Machine admission: GPU model, VRAM, BF16 support, driver, host RAM, cgroup limits, disk physical and size.
- [ ] **Step 3:** Download DeepSeek-V3 at the pinned revision. Check 163 files and 688,586,727,753 bytes, and record ETags.
- [ ] **Step 4:** Build and validate the NF4 cache with 8 threads. B6 reference: 2568.94 s, 380,276,447,824 bytes.
- [ ] **Step 5:** Run the portable stage-c technical smoke. Do not execute the historical tiny gradient gate or recalculate its E3.
- [ ] **Step 6:** Run the full-model gate. A3 or any E1-E3 FAIL, NO_VERDICT, OOM, limit or deadline stops the pilot immediately. Preserve/upload evidence, delete the instance and report. No BF16-only fallback and no retry for PASS.
- [ ] **Step 7:** One measured training step per arm and length from the shared initial adapter. Record time and every memory peak. An OOM at a length is recorded as a refusal. Do not change L, B or k to make it fit.
- [ ] **Step 8:** Measure resource-only forward throughput and logits memory at development-input shapes, without computing or publishing base-model accuracy, NLL, pass/fail or other quality outcomes. Freeze the evaluation cost forecast from timing/resource evidence only; quality outcomes remain limited to the two trained arms after protocol approval.
- [ ] **Step 9:** Destroy the instance unless the user pre-approved continuing into Phase 3. Record billing and the absence check.

**Acceptance:** `benchmarks/results/p3-deepseek-quality/pilot-v1/README.md` with every measurement, the gate verdict and the receipts.

### Task 2.2: Protocol freeze

**Files:**
- Create: `benchmarks/results/p3-deepseek-quality/protocol-v1.json` and `protocol-v1.md`, following `benchmarks/harness/p3_stage2v2_protocol.py`.

- [ ] **Step 1:** Record every fixed decision above with input hashes: checkpoint revision, cache index, pack manifest, task data manifest, initial adapter, source files of every module used, and the immutable attention-policy identity. Bind the passed same-machine gate receipt and strict EFFICIENT/MATH backend roles before any trained quality outcome.
- [ ] **Step 2:** Choose the measured items from pilot data only: step count S (Granite used 64), scheduler horizon S, lengths kept, HellaSwag N, evaluation batch tokens, shard and controller deadlines, hardware, forecast hours and cost marked as inference, spending cap.
- [ ] **Step 3:** If the forecast exceeds what the user wants to spend, present the options with their cost and statistical headroom (`p3_stage2_statistics.precision_scenarios`): fewer steps, a frozen-order prefix of HellaSwag, dropping the L=4096 pair, a different GPU class, or adopting an expert-reuse branch for both arms. The user chooses. Record the choice before any outcome.
- [ ] **Step 4:** Record the decision rule (Decision 8), the required primary cells, the secondary analyses (seed-free paired NLL difference with a paired t interval, per-endpoint accuracy table, step-1 E3 between arms per length) and the stop rules.
- [ ] **Step 5:** Compute the file SHA-256 and get the user's approval of the protocol hash and the Phase 3 budget.

**Acceptance:** frozen protocol hash in the ledger and the user's recorded approval.

---

## Phase 3: execution

### Task 3.1: Training

- [ ] **Step 1:** Run on one machine in this order: L1024 baseline, L1024 candidate, L4096 baseline, L4096 candidate. A budget stop then leaves complete pairs.
- [ ] **Step 2:** After each run, verify the digest chain, the final adapter files and the manifest. Compute the step-1 E3 diagnostic between the two arms of each length from the saved gradients.
- [ ] **Step 3:** On instance loss, resume from the last complete checkpoint into a fresh output directory and record the event.

### Task 3.2: Evaluation

- [ ] **Step 1:** Score each final adapter on all primary and secondary endpoints, in shards from the frozen plan. Finish both arms of a pair before the next pair.
- [ ] **Step 2:** Verify each shard's coverage and hashes. Count only whole pairs.
- [x] **Step 3: Do not execute.** The user excluded untrained base-model evaluation. The study compares only the trained BF16 and FP8+Oracle arms.

### Task 3.3: Assembly and report

**Files:**
- Create: `benchmarks/results/p3-deepseek-quality/results-v1.md`
- Modify: `docs/performance-and-quantization.md` (one paragraph next to the tiny smoke paragraph), a changelog fragment per `changelog.d/README.md`.

- [ ] **Step 1:** Pair the rows and call `summarize_pairs(rows, expected_tasks=..., seeds=[23], lengths=<kept lengths>)`, then `apply_informativeness_gate(summary, chance=<frozen values>)`.
- [ ] **Step 2:** Report the gated decision next to the statistical decision, every primary cell (baseline and candidate accuracy, harms, gains, U, result), the NLL table, the gate verdicts from the pilot, and a resource table with step seconds, tokens/s and each memory resource separately.
- [ ] **Step 3:** State limits: one seed, rank-2 attention-only adapter, S steps, one machine; MBPP-train code repeats approximately four times, identically in both arms. Include both per-step loss trajectories and hash-bound initial-to-final adapter Frobenius norms. Adapter movement is not a claim of improvement over an unmeasured base model.

---

## Stop rules

| Event | Action |
| --- | --- |
| Gradient gate fails (Task 2.1 Step 6) | Stop. Report to the user. No training. |
| Routing refusal, missing Oracle VJP, non-finite value | Stop that arm, preserve evidence, report. No skip, no retry. |
| OOM | Record a refusal at that configuration. No silent change of L, B or k. |
| Spend cap or deadline | Stop, upload evidence, destroy, report completed pairs. |
| Instance loss | Resume from the last complete checkpoint, record the event. |

## Rough cost picture (all [INFERENCE] until the pilot)

- **Training.** P3:95 estimates about 9 minutes per 4096-token step. At 64 steps that is about 9.6 hours per arm and about 38 hours for four runs. That estimate assumes one expert read per window. PR #58 measured that an expert is read again for every micro-batch that hits it, so the real step time may be several times longer. Pilot Step 7 measures it.
- **Evaluation.** Scaling Granite's development token rates to the full v2 task counts gives about 1.95M input tokens for HellaSwag, 0.49M for ARC and 1.11M for BoolQ. LAMBADA was not measured. That is at least 3.55M tokens per checkpoint and at least 14.2M for four checkpoints. The DeepSeek tokenizer gives different counts. Evaluation is likely the largest cost. Pilot Step 8 measures throughput.

## Risks

| Risk | Effect | Mitigation in this plan |
| --- | --- | --- |
| Expert re-reads per micro-batch | Steps far slower than P3's estimate | Measured in the pilot. Budget options at the freeze |
| Rank-2 attention-only LoRA changes the model little | Both arms may remain close and the test can be weak | Report both loss trajectories and hash-bound adapter change norms; no base-quality evaluation |
| FP8 fails E1-E3 on the real model | No candidate arm | Gate runs before any training spend |
| BF16 A3 floor does not execute on the rented GPU | A3 verdict missing | Record `no_floor`. Do not substitute another tolerance |
| Instance dies during long runs | Lost hours | Per-step checkpoints, per-stage upload, resume |
| Kaggle disk too small for the real cache | Task 1.6 limited | Bounded subset or recorded limit |

## What the result can and cannot claim

- **Can claim.** Whether FP8 block64 + Oracle boundary storage degrades zero-shot quality of a LoRA-trained DeepSeek-V3 by more than 1 pp on the measured endpoints, under one seed and the frozen training budget. Measured boundary-store bytes of both arms.
- **Cannot claim.** The ×50 speed target, fit under 8 GB and 32 GiB unless the run used those limits, natural routing stability, FP16 behavior, generalization to other seeds, adapters or models.
