# P3 FP8 / BF16 bounded investigation

## Outcome: blocked with the exact missing evidence

FP8 is **not accepted**. A formally verified BF16 fallback is **not yet proposed for acceptance**, because the unchanged A3 BF16 noise-floor measurement could not execute on the assigned hardware. The observed BF16 controls are encouraging, but bitwise agreement does not replace the required floor. This does not close all phase-0 requirements.

All execution, experiments and lint ran through Kaggle CLI. No Vast.ai resources were used. Work continued in `soup-rnd-p3-fp8`; this investigation added harnesses and evidence without changing the production FP8 codec, trainer, memory planner or expert-read implementation. Earlier evidence in `../p3-fp8-kaggle-20261105/` remains untouched.

- [Kaggle kernel](https://www.kaggle.com/code/batyrkhan2131123/soup-p3-bf16-bounded-investigation)
- [Predeclared protocol](protocol.json): numerical rules, separate input seeds, a 25% minimum useful host-reservation saving, and at most one optional candidate.
- [Complete matrix summary](matrix-summary.json), [decoder attribution](decoder-summary.json), [whole-step observations](whole-step-observations.csv), [coverage checks](evidence-coverage.json).
- [Submitted sources and Input metadata](kaggle-submissions.zip), [artifact hashes and source identity](manifest.json).

## Scope and precision

The official Input `deepseek-ai/deepseek-r1/transformers/deepseek-r1/2` was mounted. Its index referenced **163 present shards**. Its config declares BF16 compute and FP8 E4M3 weight quantization with 128-by-128 blocks. It is **not an already-converted P3 NF4 cache**. See `v5-verification/run.json` and `v4-holdout/policy-screen.json`.

Numerical training used a synthetic, native Transformers DeepSeek-V3 fixture: 3 decoder layers, hidden size 32, 8 routed experts, top-2 routing. **No 671B model training step or full R1-to-NF4 conversion was performed.** The full-width experiment below exercised storage at R1 dimensions, not the giant model's computation.

Two weight formats were tested separately:

- Dense BF16, converted from the unchanged FP32 fixture checkpoint, seed 61.
- NF4, block size 64, **double quantization disabled**, BF16 dequantization/compute. The resident control used the same packed projection values and production dequantization forward, plus native frozen experts containing the exact stored BF16 dequantizations. Quant-state hashes and arithmetic module classes are in each matrix JSON.

Decoder computation/activation storage input was BF16; native router arithmetic and PEFT trainables were FP32. FP8 changed **storage**, not the decoder compute dtype. Autocast and TF32 were disabled, dropout was zero, deterministic execution enabled, backward outside autocast. Observed dtypes and SDPA calls were recorded rather than inferred from flags.

Exact P3 LoRA: rank 2, alpha 4, targets `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`; routed experts frozen. The retained broader fixture additionally targeted `lm_head`, the three shared-expert projections, and `embed_tokens`. Existing adapter initialization and randomization were preserved: materialization seed 71, randomization seed 67.

Environment: PyTorch 2.11.0+cu130, Transformers 5.19.0, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0. Assigned GPU: Tesla T4, capability 7.5; BF16 emulation reported supported, native BF16 reported unsupported. An explicit `NvidiaL4` CLI request was accepted as kernel version 4, but its actual GPU receipts still show T4. No L4 measurement is claimed.

## Matched controls and unchanged numerical criteria

Both adapter scopes and both weight formats ran at **k=1/4/16/64**, with two full windows and a final `max(1, k//2)` window. Micro-batch row counts and sequence lengths varied; labels included masking. Arms started each window from the same adapters; only GA1 advanced the common next-window start. This matrix is not an independent multi-step trajectory comparison; HF Trainer trajectories were checked separately.

Discovery and held-out inputs used offsets 0 and 10000. Recorded input-hash sets had no overlap within any paired case. There were **32 cases / 96 matched windows** across both sets.

Resident GA versus uncompressed production L2L was **bitwise equal in loss, gradients and update deltas in all 96 windows**. A3 nevertheless returned `no_floor` throughout: BF16 EFFICIENT_ATTENTION failed with `No available kernel. Aborting execution.` Its diagnostic accepted Half/Float, not BF16, on this T4. CPU did not supply the required floor either. No absolute tolerance, alternative floor or forced route was substituted.

FP8 was compared against matched uncompressed L2L, not directly against GA. G4 remained: E1 RMS row error <=0.05 and worst row <=0.25; E2 branch RMS row error <=0.10; E3 global gradient relative L2 <=0.05 and minimum tensor cosine >=0.99. Update deltas were also checked. Native **ordered** expert IDs had to match exactly.

| Weight format / adapters | Discovery routing refusals, out of 12 windows | Holdout routing refusals, out of 12 windows | Accepted FP8 cases |
|---|---:|---:|---:|
| Dense BF16 / exact P3 | 9 | 8 | 0 |
| NF4 / exact P3 | 10 | 10 | 0 |
| Dense BF16 / broader | 10 | 9 | 0 |
| NF4 / broader | 10 | 9 | 0 |

In total, **75/96 FP8 windows refused routing and did not apply SGD**. Of the 21 diagnostic windows that reached provisional SGD, 18 failed the gradient/update G4 gate; three isolated broader-fixture passes did not rescue any case. Failed provisional updates were never carried into another arm or the next common window. E1 passed all first-window decoder diagnostics, but E2 frequently failed. E1/E2 coverage is the first full window, including its maximum length, not every window.

For the exact P3 adapters, discovery k=1/window 0 already failed without a routing change: gradient relative L2 was **0.056459** with dense weights and **0.061355** with NF4, above 0.05. Every full k=4/16/64 FP8 window failed the routing guard in discovery and holdout. Some short held-out dense windows completed but still failed G4.

## Decoder diagnosis and bounded policy decision

Diagnostics capture real decoder boundaries, attention outputs, residual branches, native router inputs/IDs and route-stable local vector-Jacobian products. They do not use embedding-only router probes and do not replace the production gradient gate.

- Dense P3, discovery k=4: layer 1, micro-batch 2, token 8 changed `[3, 2]` to `[3, 1]`. The original eligible-expert margin was 0.00182843; router-input relative L2 error was 0.0204672.
- NF4 P3, discovery k=4: layer 1, micro-batch 1, token 13 changed ordered IDs `[4, 6]` to `[6, 4]`. Permutations remain failures under the existing guard; other captured tokens also changed expert membership.
- At route-stable dense P3 k=1 boundaries, local adapter-VJP relative L2 errors were about 0.0389, 0.0474 and 0.0410 across layers. These are attribution measurements with a deterministic upstream vector, not loss-gradient acceptance tests. They show that preserving routes alone would not establish G4.

Exactly one policy was screened: retain every MoE decoder boundary in BF16 and compress only dense boundaries. R1 has 58 MoE layers and 3 dense layers. Even an impossible implementation making all three dense boundaries **free**, with zero candidate metadata/scratch/compute, can save at most **4.8998%** of the charged baseline reservation. Its optimistic lower bound is **3,907,522,560 bytes**, against 4,108,849,152 bytes for BF16. Real FP8 payload, scales and additional costs can only worsen that bound.

The policy therefore fails the predeclared **25%** usefulness requirement and was rejected **before implementation**. No replacement codec, residual-correction scheme or second candidate was introduced. There is no selected policy to validate on holdout or to benchmark for whole-step cost; the independent holdout instead checked the unchanged BF16/FP8 comparison. See `v4-holdout/policy-screen.json`.

## Memory accounting

The isolated-process storage experiment exercised actual production `L2LState` / `_store_for` allocation and put/get paths at 61 layers, hidden size 7168, 64 tokens per micro-batch, and k=1/4/16/64. The reservation ceiling was 5 GiB. All eight cases retained their requested k, matched the planned reservation, passed their round-trip criterion and released all leases.

At k=64 (4096 aggregate tokens):

| Measurement, bytes | BF16 storage | FP8 storage |
|---|---:|---:|
| Charged activation + routing reservation | 4,108,849,152 | 2,063,077,441 |
| Owned payload, including FP8 scales | 3,581,935,616 | 1,791,967,232 |
| Sampled process RSS peak | 5,105,061,888 | 3,059,687,424 |
| CUDA allocated peak during storage probe | 13,797,376 | 13,797,376 |

The charged reduction is **49.7894%**, so plain FP8 clears the storage-usefulness gate but fails numerical acceptance. Each codec/k ran in a fresh subprocess to avoid comparing against retained pinned-allocation caches. Version 1's sequential-process RSS observations are preserved but superseded by version 2 for that comparison.

Charges include pinned rounding, codec scales/scratch and routing reservation. The full-width probe **leased** 15,208,448 routing bytes but did not fill physical route tensors. RSS includes process baseline and probe scratch; sampled peaks are lower bounds. Weights, optimizer, expert graphs, caller-retained hidden states/gradients and allocator caches are not covered by the activation reservation. This is not an 8 GB whole-model fit result. At k=64 each codec completed 3904 puts and gets; FP8 E1 RMS row error was 0.026549, worst row 0.030325.

The numerical matrix recorded zero remaining activation/routing lease usage in **480/480 arm cleanup records**, with no missing cleanup fields.

## Whole-step observations

Timing spans synchronized forward, backward and provisional SGD. Decoder diagnostics and raw tensor export are outside the interval. The fixture twins and instrumentation are included in observed process/device memory. These are single tiny-model observations, not throughput estimates for R1 or native-BF16 hardware.

Selected completed exact-P3 windows; **all failed numerical acceptance**:

| Inputs / weights / capacity k / actual micro-batches | BF16 step, seconds | FP8 step, seconds | Observed overhead |
|---|---:|---:|---:|
| Discovery / dense / 1 / 1, window 2 | 0.093370 | 0.096438 | 3.29% |
| Discovery / NF4 / 1 / 1, window 2 | 0.105560 | 0.113572 | 7.59% |
| Holdout / dense / 4 / 2, short window | 0.181923 | 0.197626 | 8.63% |
| Holdout / dense / 16 / 8, short window | 0.763834 | 0.781850 | 2.36% |

Aborted FP8 intervals are retained separately in the CSV. Their shorter times are **not speedups**. No valid full-window FP8 overhead exists for the exact-P3 k=4/16/64 cases because those windows refused routing.

## Production trainer and B1.4 handoff

The internal native DeepSeek BF16 fixture ran the actual HF Trainer/L2L training path at every k, with three independent optimizer/scheduler steps, masked-label normalization, variable sequence lengths, store growth and final short-window reuse. Per-step gradients matched resident GA, and final adapter differences were zero. Trainer logging had tiny non-bitwise loss differences at k=4/64 (up to approximately 1.16e-7 relative); these are not silently declared A3 passes. Internal DeepSeek execution did **not** remove its public launch refusal.

A separate supported tiny-Llama public builder run on Kaggle CPU/FP32 verified requested k=4 -> resolved effective_k=2 **before trainer/scheduler construction**, three steps, growth, short-window reuse, C3 weight/activation separation, rejected mismatched HF k, and lease cleanup. This is a CPU caller-contract result, not a CUDA BF16 builder result. On T4 the ordinary stream-dtype resolver selected FP16, which the MoE planner rejects; that GPU builder failure is preserved, not bypassed.

Negative checks covered missing C3 routing reservations, excess window k/shape, dtype/codec/device mismatch, exhausted host budget and unsupported public DeepSeek launch.

The B1.4 contract at the pinned revision remains:

1. Bound fixed consumers and maximum activation shape; resolve k once against the remaining host envelope.
2. Reserve the C3 activation **and routing** partition before optional weight allocation. Do not resolve k against an already requested-k-sized partition or subtract the same activations twice.
3. Construct HF arguments, trainer and scheduler with `memory_plan.effective_k`, then pass the matching plan and shared budget through `_build_l2l_trainer` / `attach_l2l_state`.

The receiving seams enforce this contract. The public caller is still missing: `sft.py:956,1210` uses configured k and passes no resolved plan/budget; `stream_setup.py:1173-1181` sizes C3 using configured k and omits MoE routing bytes. This is a handoff finding, not an implementation or agreement by the B1.4 owner. No B1.4 caller changes were made. Repeated expert reads remain a separate P3 performance dependency; they were not optimized here.

## Receipts and verification

- `v1-accounting/`: initial sequential storage experiment, preserved with its RSS limitation.
- `v2-accounting-isolated/`: definitive isolated storage accounting, eight passing cases.
- `v3-discovery/`: full discovery matrix and GPU trainer contract, including initial harness lint findings.
- `v4-holdout/`: full independent-input matrix, retention-policy screen, CPU/GPU caller contracts; actual GPU remained T4 despite the L4 request.
- `v5-verification/`: final source smoke after lint correction, CPU builder rerun and unchanged numerical refusals. **Full repository Ruff passed on Kaggle.** The numerical command deliberately retains exit code 1; the launcher remains `FAIL`, not falsely green.

Each run includes commands, exit codes, dependencies, source hashes and Input metadata. Tensor ZIPs contain canonical adapters, gradients, update deltas and decoder diagnostics. `kaggle-submissions.zip` preserves submitted scripts/metadata for versions 2-5. The eight production overlay files are byte-identical between accounting and final verification. No new permanent tests or unverified production numerical policy were added.

## Next steps / exact missing evidence

1. On Kaggle, first obtain and **observe** a GPU/backend combination that actually executes the unchanged BF16 EFFICIENT_ATTENTION floor. Then rerun A3 for both adapter scopes, dense/NF4 and all k, including the HF Trainer trajectories. An accelerator request string is not hardware evidence.
2. Until that evidence exists, retain the specification's uncompressed BF16 fallback as the conservative direction, **not an accepted fallback or FP8 completion**. Do not weaken G4 or the routing guard. This bounded investigation does not authorize another compression search.
3. Have B1.4 integrate the resolved-plan caller contract separately. CUDA public-builder coverage with a supported compute dtype is still missing.
4. Full R1 FP8-checkpoint-to-NF4 conversion, real 671B training/memory behavior and remaining phase-0 acceptance are outside the verified tiny-model/storage scope. Preserve that distinction before claiming DeepSeek readiness or proceeding beyond Kaggle.
