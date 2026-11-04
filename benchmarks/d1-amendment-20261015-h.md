# D1 amendment H: complete A100 cohort after valid clock diagnostics

Dated 2026-10-15, committed before the complete A100 cohort. Diagnostic G,
source 49e91d7d, ran on NVIDIA A100-SXM4-40GB, physical UUID
GPU-9194385f-8e62-4a62-a0e7-90cb10f10c31, driver 580.178.04,
Python 3.12.3, torch 2.11.0+cu126, bitsandbytes 0.50.2,
transformers 5.17.0 and peft 0.20.0. Retain the full resolved environment.
There is exactly one assigned CUDA device; its UUID is exported explicitly
as CUDA_VISIBLE_DEVICES and matched against NVML. Cleanup addresses only that
GPU and this exact rental, using local and in-rental independent guards.

Both CUDA alias/streaming fixtures passed, and the existing GPU regressions
reported 12 passed, 17 deselected, no skips. G's one-block BF16 clock screens
met every required validity row: M=1 dense/native stayed at 1095 MHz across
all four arms; M=2048 native/forced stayed at 1410 MHz. They remain one-block
diagnostics with NO_VERDICT, not performance findings or counted H pilot blocks.

The `-lgc 1200,1200` setup attempt was denied. No software clock lock is
claimed. Use the unchanged provider clock/power policy and retain ALL actual
timed-work samples. The original <=5% clock spread applies independently
within every comparison; the valid clock values can differ BETWEEN shapes/M.
A failed later case still stops the controller; the diagnostic is not a blanket
pass or permission to remove outlier samples.

## Complete plan fixed before H timings

Start a new source/output cohort on this same physical GPU; do not pool any
G, E/F or earlier RTX 3090 timing with H. Re-run the two correctness fixtures
and GPU regressions. Run the five-valid-block target-card ablation first,
then ALL 96 projection cases, then five valid ABBA blocks of real QA training.
This changes only the order from F, so a projection validity failure is found
before spending on another set of QA repeats. Every case remains required.

Projections use >=20 steps AND 5 seconds of synchronized warm-up and 200
timed steps, exactly G's screened settings. Preserve separate checkpoint
forward and loss/backward CUDA events, combined events/wall arrays, allocated
and reserved peaks, native oracle witnesses, dispatch counters, same-session
dense GEMM ceilings, SM clocks and memory-controller utilization. SAME_PATH
cases stay in the grid and are never called fused speedups.

QA retains the fixed 256/128/64 SQuAD selection, pinned TinyLlama revision,
seed 842, rank 16, lr 2e-4, batch 1, length 1024 and 276 updates (20 warm-up
plus 256 timed). No independent synthetic thermal prelude is used on H.
Its processes are fresh and share no model/optimizer/CUDA allocator state.
The cached-zero ablation remains timing-only garbage math. Native-reference
equality, negative controls, F1 sensor checks and pilot/N/one-top-up sizing
are unchanged; every failed/void/incomplete attempt is retained.

## Budget and handoff

The diagnostic rental's two-hour lease may be extended only after this rule
is committed. Its immutable absolute ceiling is 15 hours from allocation,
including setup and diagnostics, with two hours reserved for export. At the
fixed $0.5448148148/hour including 100 GB disk, maximum compute/storage is
$8.17222222. Add the conservative $1.20 prior-spend ceiling and $0.60 transfer
reserve: $9.97222222, within the user's cumulative $10 authorization.
Independent guards also stop this rental at the $8.20 cost threshold.
No other user's/team's rental is modified. A deadline/budget stop means
incomplete measurements, never acceptance.

Rebuild the report from every watched record, audit every accepted QA repeat
against its gold and stored tensors, verify every archived file hash, and
release the exact rental after a verified local export. Keep the earlier
RTX 3090 pilot and E/F attempts as separate historical evidence. A100 results
do not establish consumer-card, sm_120 or full-model 8B/70B speed multipliers.
Independent review and the team lead's final acceptance are still required.
