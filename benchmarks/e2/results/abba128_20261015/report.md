# Soup E2 - Mistral-7B ABBA128 Performance Evidence Review
**Date:** 15 October 2026  |  **Source:** archived final benchmark evidence  |  **Status:** Completed; awaiting research review

> **Executive finding.** The preapproved warm-epoch criterion was met: median second-epoch step-time speedup **1.844x** over three ABBA cycles, with every cycle above 1.00x. However, median end-to-end subprocess ratio was **0.726x** (cache runs slower). This is not an overall E2 research PASS: independent cold-cache accounting, comparable validation quality, and Small-MoE evidence are outstanding.

## 1. Experiment contract and provenance

| Field | Captured value |
|---|---|
| Model | Mistral-7B-v0.3, pinned snapshot `caa1feb0e54d415e2df31207e5f4e273e33509b1` |
| Method | Top-K LoRA, K=8; cache-disabled A vs cache-enabled B |
| GPU / dtype | NVIDIA GeForce RTX 2080 Ti (GPU 0); CUDA FP16, AMP initial scale 1024 |
| Inputs | 128 fixed JSONL records; 2 epochs; max length 128 |
| Steps | 256 measured / requested each run; 8 warmup steps excluded from overall step summary |
| ABBA design | A-B-B-A repeated three times; 12 independent subprocess runs |
| Seed | 42, reported in every training report |
| Git commit | `2390142a487aae0ee2f5d1fdf103107afd623b32` |
| Dataset SHA256 | `5d6e80c64defd24e759136f8d88e58d98da89958ec861e772ef2b02568df1c3c` |
| Runner SHA256 | `7d89b8c2093de9da770c6660b64972cce6d6b1f39c4fe88d1681c57758661efa` |
| Approved decision rule SHA256 | `bf8bf3156b1ad354662c9222b1a4953d6997ea71ee4db5c595366661157d7fde` |
| Runner status | `COMPLETE_AWAITING_REVIEW`; benchmark exit code 0 |

The cached configuration differs from the baseline in `frozen_prefix_cache` and in the output location; the relevant inputs/configuration hashes were frozen before the benchmark. The runner captures the exact decision-rule content and SHA256 in its manifest.

## 2. Predeclared scoring rule

For each A-B-B-A cycle, compute **mean of the two baseline second-epoch step-time sums / mean of the two cached second-epoch step-time sums**. Because there are two observations per mode, the mean equals the median used by the runner. The predeclared gate requires **median of three cycle ratios >= 1.10x**, and **every cycle >= 1.00x**. These are warm-step criteria, not total-runtime acceptance criteria.

## 3. Verified results

| Cycle | A epoch-2 (s) | B epoch-2 (s) | Warm ratio | A full run (s) | B full run (s) | E2E ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.915 | 7.538 | 1.846x | 47.517 | 66.996 | 0.709x |
| 2 | 13.795 | 7.519 | 1.835x | 48.667 | 65.786 | 0.740x |
| 3 | 13.894 | 7.534 | 1.844x | 47.798 | 65.809 | 0.726x |
| **Median of cycles** | - | - | **1.844x** | - | - | **0.726x** |

![Cycle ratios](e2_abba128_cycle_ratios.png)

### Mean timings across six A and six B runs

| Measurement | Baseline A (s) | Cached B (s) | B minus A (s) |
|---|---:|---:|---:|
| Epoch 1, step-time sum | 14.538 | 15.105 | +0.567 |
| Epoch 2, step-time sum | 13.868 | 7.530 | -6.338 |
| Full subprocess wall time | 47.994 | 66.197 | +18.203 |
| Wall time not inside epoch step-time sums | 19.589 | 43.562 | +23.974 |

![Mean durations](e2_abba128_time_comparison.png)

![Time decomposition](e2_abba128_time_breakdown.png)

The cached mode reduced second-epoch step time by **45.7%**, or approximately **6.34 seconds per 128-example warm epoch**. Yet its full subprocess duration rose by **37.9%**, an increase of **18.20 seconds per run**.

**Measured decomposition of the mean full-run increase:**
- Epoch-1 step-time difference: **+0.567 s** (cached slower).
- Epoch-2 step-time difference: **-6.338 s** (cached faster).
- Remaining process time outside epoch step-time sums: **+23.974 s** (cached slower).
- Net subprocess difference: **+18.203 s**.

**Interpretation:** The most important unresolved overhead is *outside recorded optimizer-step timing*. This residual includes model initialization, preprocessing, cache setup/IO, and potentially other work; the files do **not** attribute it to any particular operation. The first-epoch difference is **not** an independent cache-construction timer.

## 4. All 12 run-level results

| Arm | Cycle | Mode | Epoch 1 (s) | Epoch 2 (s) | Full (s) | Unattributed (s) | E1 misses | E2 hits | Valid |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 1 | A | 14.462 | 13.924 | 47.493 | 19.108 | - | - | Yes |
| 2 | 1 | B | 15.211 | 7.591 | 68.699 | 45.897 | 128 | 128 | Yes |
| 3 | 1 | B | 15.090 | 7.484 | 65.294 | 42.720 | 128 | 128 | Yes |
| 4 | 1 | A | 14.596 | 13.907 | 47.540 | 19.038 | - | - | Yes |
| 5 | 2 | A | 14.641 | 13.789 | 49.156 | 20.727 | - | - | Yes |
| 6 | 2 | B | 15.113 | 7.513 | 65.984 | 43.358 | 128 | 128 | Yes |
| 7 | 2 | B | 14.958 | 7.524 | 65.588 | 43.107 | 128 | 128 | Yes |
| 8 | 2 | A | 14.445 | 13.800 | 48.178 | 19.933 | - | - | Yes |
| 9 | 3 | A | 14.631 | 13.951 | 48.230 | 19.649 | - | - | Yes |
| 10 | 3 | B | 15.130 | 7.542 | 65.407 | 42.735 | 128 | 128 | Yes |
| 11 | 3 | B | 15.128 | 7.526 | 66.212 | 43.558 | 128 | 128 | Yes |
| 12 | 3 | A | 14.451 | 13.838 | 47.367 | 19.078 | - | - | Yes |

All 12 reports mark `valid: true`, list no failures, report `steps_measured: 256` and have report SHA256 values matching their manifest entries. In all six cached runs, the first epoch recorded **128 misses / 0 hits** and the second recorded **128 hits / 0 misses**.

## 5. GPU telemetry and execution sanity checks

- **331** samples, with median spacing **2.078 s**, maximum gap **2.783 s**.
- Zero GPU-query failures; no sampled instance with two or more concurrent GPU-0 compute PIDs.
- One distinct observed GPU-0 PID per arm; no PID repeated in another arm; no GPU-0 compute PIDs observed outside recorded arm windows.
- Per-arm sampled median GPU utilization: baseline **86-89%**, cached **11-17%**; this is consistent with substantial non-GPU or waiting time during cached subprocesses, but cannot identify the cause.
- Caveat: NVML host PID differs from container PID, temporal alignment is not proof of process ownership, and approximately two-second sampling can miss short concurrent activity.

## 6. Measurement and scientific limitations

1. **No isolated cold-cache build/read/write times.** The reported `cold_cache_build_seconds` is `null` and `cache_profile` is absent; runner payback values are labeled proxy estimates only.
2. **Epoch times are sums of measured step durations**, not full wall-clock epoch times. Full subprocess duration separately includes launch and setup overhead.
3. **Quality cannot be assessed from this archive.** The benchmark used `val_split: 0.0`; no comparable validation-loss evidence is provided. The proposed <=1% quality threshold is not approved as a completed gate.
4. **Model coverage is incomplete.** This evidence tests Mistral-7B, K=8 only; Small-MoE and other K/L fractions require separate evidence.
5. **Only the warm-performance gate is met.** The manifest explicitly reports performance `NO VERDICT` pending independent overhead review, and research acceptance `NO VERDICT` pending quality and Small-MoE.

## 7. Recommended follow-up plan

**P0 - preserve and present evidence (no GPU run required):** attach this review, the original checksum-verified tarball and run-level CSV to the existing PR; include approved-rule SHA256, pinned Git commit, exact cycle metrics and `NO VERDICT` limitations. Confirm team access to backup artifacts.

**P1 - isolate the +23.97 s residual:** add low-overhead wall timers around config/tokenizer/model initialization, dataset materialization, frozen-prefix forward, cache serialization, cache reads, host-device transfers and teardown. Separate CPU timers and CUDA synchronized operation timers. Store aggregate cache setup/read/write time per arm, then use a new predeclared protocol for profiling experiments; do not edit the completed ABBA artifact.

**P2 - test end-to-end payback:** after profiling, test controlled runs across multiple repeated epochs / larger reuse horizons with identical input order and architecture. Count one-time cache construction and all load/IO expenses; report full wall-clock, steps, hit/miss counts, and amortization together. Check whether prep work is repeated unnecessarily on each run.

**P3 - quality parity and cache correctness:** compare cached vs no-cache against an approved validation set with matched seeds and inputs; check loss degradation, output parity, attention masks, position IDs and cache invalidation upon input/model revision change.

**P4 - research matrix:** test additional K/L fractions (1/4, 1/2, 3/4, 1 as control) and a small supported MoE; separate timing and quality findings. Do not make an overall performance or quality PASS until all gates are satisfied.

## 8. Suggested technical update for the team lead

> The Mistral-7B FP16 ABBA128 benchmark completed successfully (12/12 valid arms). The predeclared warm-performance criterion was met: median second-epoch step-time speedup was **1.844x**, and each cycle exceeded **1.00x**. Cache behavior was consistent (128 first-epoch misses followed by 128 second-epoch hits in every B arm). However, median end-to-end ratio was **0.726x**, indicating that full cached runs remain slower. Decomposing subprocess time suggests approximately **24 s** of additional time outside measured steps in cached runs, which needs independent profiling before attribution. GPU telemetry captured 331 samples without query failures or concurrent GPU-0 PIDs in observed samples. The runner correctly retains **NO VERDICT** for complete performance/research acceptance; quality and Small-MoE remain separate. Next step: instrument and investigate cache build/read/write and setup overhead without altering the completed evidence.

## 9. Evidence reference

- Original user-uploaded file: `e2_abba128_final_20261015_evidence.tar.gz`.
- Raw manifest: `mistral_fp16_abba128_final_v1/manifest.json`.
- Run reports: `mistral_fp16_abba128_final_v1/01_A.json` ... `12_A.json`.
- GPU observations: `mistral_fp16_abba128_final_v1_monitor/gpu_telemetry.jsonl`.
- Generated audit table: `e2_abba128_runs.csv`.

*Derived calculations independently recomputed from archived JSON. Ratios use baseline time divided by cached time. No new model training was performed for this review.*
