# E2 performance decision rule — approved before final measurements

Approval source: team lead's message on 2026-10-13. The lead explicitly
approved **1.10×** as an experimental threshold and required each valid cycle
to be at least **1.00×**. This document records that decision and the
conditions to be used for the next final benchmark; it does not claim that
any experiment has passed.

## Comparison and scope

- A: Top-K LoRA without frozen-prefix activation caching.
- B: the same Top-K LoRA with frozen-prefix caching enabled.
- Model: pinned Mistral-7B-v0.3 revision
  `caa1feb0e54d415e2df31207e5f4e273e33509b1`.
- Initial performance scope: K=8 of L=32 decoder layers, same A/B configs
  except cache flag and output, 2 epochs, 2,000 train-only examples,
  batch size 1, grad accumulation 1, max length 128, `--warmup 8`.
- This is only one K/L condition. K=16 and K=24, and the Small-MoE matrix,
  must be reported separately. The K=L control has no E2 caching.

## Repeats and acceptance calculation

- Three **complete** A–B–B–A cycles (12 fresh processes), with independent
  cold caches for every B run. Do not append runs from a different session
  to an incomplete cycle or select favorable runs after measurement.
- Primary performance measure: the sum of optimizer-step wall times in
  epoch 2, as recorded by `BenchCollector.epoch_timing[1].duration_seconds`.
  This is **not** the entire epoch's elapsed wall time.
- For cycle `i`, let `A_i` be the median of that cycle's two A epoch-2
  measurements and `B_i` the median of its two B measurements. Compute
  `R_i = A_i / B_i`.
- Compute `R_final = median(R_1, R_2, R_3)`.
- Primary warm-epoch performance threshold is met if and only if
  `R_final >= 1.10` **and** every valid cycle has `R_i >= 1.00`.
- A noisy, invalid or incomplete experiment remains **NO VERDICT**; do not
  report PASS or FAIL by cherry-picking cycles. A reproducible slowdown under
  valid comparable conditions may be reported as FAIL after review.

## Preconditions and run validity

- Lock this rule, model/dataset identities, configurations, git commit and
  software environment **before** conducting the final 12-run experiment.
- All 12 subprocesses must run on the same GPU model/count and stable
  hardware setup; capture GPU UUIDs, utilization, temperature and clocks,
  CPU, RAM, PyTorch/CUDA/Transformers versions and timestamps. No comparisons
  across different GPU models. Investigate substantial runtime drift.
- The training dataset and its hash, model revision, seed, optimizer,
  scheduler, LoRA rank/alpha/target modules, dtype, batch size and sequence
  settings must match across A and B. Validation data is not used for training.
- Require each report to be valid, show precisely 4,000 optimizer steps and
  exactly two epochs of 2,000 steps, nonempty supervised token counts, and
  identical supervised work between comparable arms.
- For each B run, verify cache misses when building in epoch 1 and cache hits
  in epoch 2. Keep each raw report, subprocess log, config and machine record.
- Record failed or incomplete runs as such; reruns require an explicitly
  documented fresh experiment, not replacement of selected bad outcomes.

## Required additional cost and payback analysis

Report **separately**, for each run and each ABBA cycle:

1. **End-to-end** elapsed benchmark subprocess time, including loading/setup,
   cache construction, training and report generation, with A/B ratios and
   seconds saved or lost. Do not present it as GPU kernel-only time.
2. **Cold-cache construction** time: independently instrument frozen-prefix
   computation plus serialized activation write on cache misses; also report
   total load/read time on hits and counts. Use CUDA synchronization/events
   when timing asynchronous GPU work; document synchronization overhead.
3. First-epoch and second-epoch step-time sums separately. First-epoch B-A
   differences are only an overhead **proxy**, not an isolated build timer.
4. **Amortization:** using measured cold investment `C` and per-warm-epoch
   benefit `S`, show `C/S` warm epochs if `S > 0`; otherwise payback is not
   reached. Compare the calculated estimate to the **observed two-epoch
   end-to-end** difference. Explain assumptions; do not treat a proxy as
   direct measurement.

## Scope of verdicts

Passing the warm-epoch 1.10× threshold does **not** mean overall E2 PASS.
Overall E2 acceptance requires correctness/invalidation, approved quality
comparison to **all-layer LoRA** on required gate suites and task loss using
predeclared quality tolerances, pretrained Small-MoE evaluation, and complete
performance evidence. Until these are complete, **overall: NO VERDICT**.

The quality tolerance and suite acceptance rules are separate and are **not**
created or implicitly approved by this performance threshold.
