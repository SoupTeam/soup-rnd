# H2 close-out: actual updates, Unsloth, and interleaved throughput on RTX 4090

Date: 2026-11-02. Branch `research/h2-kernel-comparison`. Protocol `benchmarks/h2/H2_CLOSE_PROTOCOL.txt`
(committed before the first arm) and `H2_CLOSE_ADDENDUM_1.txt`. Measured source: stages U and S at
`fc2880f9`, stage P at `c5d1ddbb`.

**Result.** Baseline Transformers (A) and Liger CE-only (L) both train with real adapter updates
(UPDATES_PASS). Unsloth refused this recipe (UNSLOTH_INCOMPATIBLE). In interleaved full-training runs
L/A useful throughput is **1.013 (95% interval 0.998-1.029), NO_DETECTABLE_DIFFERENCE**, and L reserves
**33% less** CUDA memory (11.23 vs 16.82 GB; allocated 10.96 vs 11.20 GB). The multiplier holds for the
measured configuration only (section 4).

Unchanged: `NUMERICAL_POLICY.txt` and every threshold; attempts 015/016 stay BLOCKED and attempt019
stays NOT_CALIBRATABLE. Stage U measures the L-vs-A difference but does not gate on it.

## 1. What was asked (card checklist)

| Item | Evidence | Status |
|---|---|---|
| Same 8B recipe for every variant | One snapshot (unsloth/Meta-Llama-3.1-8B `e9a141a2`), NF4 QLoRA r16/alpha32, AdamW 2e-4, batch 1, GA 4, seed 42, llama3 template; configs differ only in backend/Liger flag/output (test) | Done for A and L; Unsloth refused the recipe (stage S) |
| Versions, targets, numerics, token accounting | `runtime_candidate.lock`; LoRA on 7 projections x 32 layers; bf16 autocast; per-step token counts frozen in `perf_workload_receipt.json` and matched in every run | Done |
| Actual adapter updates | Stage U, 4 optimizer updates per arm on GPU | Done (A, L) |
| Full tok/s and memory, interleaved | Stage P, 5 ABBA blocks, 20 fresh processes | Done |
| Multiplier only for the measured configuration | Section 4 | Done |

Dependencies D1/D2/D3 and H1 were not re-checked here; their acceptance is the reviewer's decision.

## 2. Stage U: actual updates (A and L)

Series `updates` of `h2_ce_confirmation.py`: the existing training phase, 16 frozen microbatches, 4
updates, one fresh supervised process per arm. Review `h2_close_updates.review_updates`.

| Row | A baseline | L Liger CE-only |
|---|---|---|
| U1 16 microbatches, 4 updates, `num_items_in_batch` = 24 in every window | pass | pass |
| U2 every adapter tensor changed (448/448); frozen, buffer and NF4 state unchanged | pass | pass |
| U3 finite losses and gradients | pass | pass |
| U4 identical ordered work, initial state and optimizer | pass | pass |
| U5 kernels: Liger CE kernel only in L, none in A; patch calls | 0/0, no Liger family | class 1 / instance 1, CE family only |
| Outcome | **UPDATES_PASS** | **UPDATES_PASS** |

Correction (addendum 2, 2026-11-04): the review code as measured only required at least one
adapter tensor to change per update, not every declared tensor as U2 says. The corrected review
fails any unchanged declared tensor; re-run over the retained GPU tensors it gives the same
outcome, 448 of 448 tensors changed in both arms (`evidence/stage_u_reaudit_20261104/`).

Measured, not gated: per-microbatch loss L-A is 0 for the first window and at most 0.0175 after; after
4 updates the adapter deltas differ (relative L2 median 0.093, max 0.551; minimum cosine 0.851). The
offline review on the Mac matched the box review.

## 3. Stage S: Unsloth

One audited process, Unsloth 2026.10.7 imported first, pinned stack. Install, import and model load
worked; Unsloth's own trainer setup then refused the recipe:

> ValueError: Unsloth: Untrained tokens ... `<|eot_id|><|start_header_id|><|end_header_id|>` ... add
> `embed_tokens` & `lm_head` to `target_modules` ... use the instruct version

The llama3 template's special-token embeddings are untrained in the base model, and Unsloth requires
training `embed_tokens` and `lm_head`, which is a different recipe. Outcome **UNSLOTH_INCOMPATIBLE**
(S1). Arm U was not admitted to stage P. No other Unsloth version or recipe was tried.

## 4. Stage P: interleaved throughput and peak VRAM

Workload `perf_workload.jsonl`: 96 synthetic English-like rows, 404-956 tokens, 72.8% supervised. Each
run is a fresh process: 20 optimizer steps through `soup bench train`'s trainer path, 4 warm-up and 16
counted (32,105 supervised of 44,084 tokens). Block A L L A. Every run passed the F1 checks (suspend,
power, foreign reads on the model disk, 1 GB), had an exclusive GPU and the frozen per-step token
counts.

| | A baseline | L Liger CE-only |
|---|---|---|
| Useful tokens/s, mean of 10 runs | 934.4 | 946.7 |
| Median step seconds, median of 10 runs (range) | 2.128 (2.110-2.149) | 2.094 (2.074-2.146) |
| max_memory_allocated | 11.20 GB | 10.96 GB (-2.2%) |
| max_memory_reserved | 16.82 GB | 11.23 GB (-33%) |

Block differences L-A (tok/s): 27.7, 4.1, 0.0, 8.8, 20.8. Mean difference +12.3, 95% t interval
[-2.1, +26.7], p = 0.077. The repeats rule gave N = 5, so the 5 pilot blocks were the whole series.
**Ratio L/A = 1.013 [0.998, 1.029], label NO_DETECTABLE_DIFFERENCE.** A speed-up above about 3% is
excluded at 95%. A smaller one cannot be told apart from zero. This is not an equivalence claim.

Configuration of this number, and nothing else: 1x RTX 4090 24 GB (driver 580.173.02, CUDA 13
runtime), whole machine; torch 2.14.1, transformers 5.17.0, TRL 0.29.1, PEFT 0.21.2, bitsandbytes
0.50.2, liger-kernel 0.8.5; Llama-3.1-8B NF4 QLoRA r16 on seven projections, bf16 autocast, gradient
checkpointing, batch 1 x GA 4, this workload. Clock record: SM median 2670 MHz in every run;
`sw_power_cap` in 1-7 of about 34 samples per process in both arms (the window covers the whole
process, including model load).

## 5. Attempts, cost and procedure

| Rental | Outcome | Displayed cost |
|---|---|---|
| U+S, instance 58810258 | A, L UPDATES_PASS; Unsloth refused | ~$0.71 |
| P attempt 1, instance 58823768 | FAILED_RUN on the first run, nothing measured (see below) | ~$0.29 |
| P attempt 2, instance 58843853 | COMPLETE, section 4 | ~$0.45 |

Total about $1.45 of the $6.00 H2-close budget. These are rounded dashboard values, not invoices.
Every instance and its volume was destroyed after the archive SHA matched on the Mac.

Defects found by the GPU runs, and fixed in `c5d1ddbb` (addendum 1):
- `soup bench train` fingerprinted trainable weights through numpy, which has no bfloat16, so bf16
  adapters failed before training. It now hashes raw bytes; fp32 fingerprints are unchanged.
  Regression test: `tests/test_issue836_bench_collector.py`.
- The stage P analysis imported scipy, which the pinned box runtime lacks. It now uses a stdlib
  Student t (matches scipy to 2e-14), and a test keeps box-side harness imports inside the lock.

Procedure deviations, none affecting a result: the U+S preservation script had an operator-precedence
bug and was rerun by hand; the box's direct SSH port then refused new connections (later autopilots hold
one multiplexed connection).

## 6. Evidence and replay

`evidence/` holds one packet per rental, built by `benchmarks/harness/h2_close_packet.py` from the
preserved archive after checking every member against its manifest:

| Packet | Archive SHA256 | Members included |
|---|---|---|
| `stage_u_s_58810258` (stages U and S) | `7c20a5ff...205e` | 99 of 163 |
| `stage_p_attempt1_58823768` (FAILED_RUN) | `46378636...21be` | 38 of 39 |
| `stage_p_58843853` (stage P) | `8e2339d1...968c` | 77 of 78 |

Each `PACKET.json` lists the excluded members with bytes and SHA256: the tensor and optimizer
payloads (2.5 GB), the public tokenizer, Unsloth's generated trainer cache, and `rental.json`
(SSH endpoint, budget, paths; the rest is in `rental_public.json`). Kernel traces are gzipped.
The per-tensor digests that the stage U gates read are in the included `training.json` files.

Replay without a GPU:

```bash
python benchmarks/harness/h2_perf_abba.py review benchmarks/h2/reports/20261102_h2_close/evidence/stage_p_series.json
python benchmarks/harness/h2_close_reaudit.py runs --series-dir benchmarks/h2/reports/20261102_h2_close/evidence/stage_p_58843853/evidence/h2close_P
```

The second command rechecks each of the 20 `run.json` reports with the frozen per-run rules and
recomputes the labels (0 mismatches). The stage U re-audit with tensors
(`h2_close_reaudit.py updates`) needs the full archive; its outputs are in
`evidence/stage_u_reaudit_20261104/`. Billing and SSH endpoint data are not published.

## 7. Not shown

- Speed or memory on any other GPU, model, sequence length, batch size, quantization or Liger kernel.
  The 33% reservation saving was not turned into a larger batch here. That would be a separate,
  pre-registered question.
- Model quality. The data is synthetic.
- Unsloth with `embed_tokens`/`lm_head` trainable (a different recipe).
- Correctness of the other Liger operators. They failed attempt013's gradient thresholds and were not
  measured.
