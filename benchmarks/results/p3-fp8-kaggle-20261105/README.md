# P3 FP8 activation storage: Kaggle verification, 2026-11-05

**Storage and trainer integration verified; DeepSeek FP8 acceptance failed.**
Uncompressed tiny DeepSeek matches resident gradient accumulation at k=1/4/16/64.
FP8 changes routing and can exceed the unchanged G4 gradient tolerances even when routes match.
Keep `training.l2l_activation_codec: none` for the MoE acceptance path. No Vast.ai was used.

## Implementation and ownership

Worktree: `soup-rnd-p3-fp8`, branch `feat/fp8-activation-storage`, based on
`e94dda7ac6e4895549dea6e4d70af99eabe3b81e`. No commit or push was made.
The existing #53 planner change (`8f6e01e4`) was imported, not independently reimplemented.
#47 is already merged. B1.4's public DeepSeek launch and automatic-k policy remain with its owner.

- `training.l2l_activation_codec: fp8_e4m3` selects product `FP8ActivationStore` in the SFT trainer. Default: `none`.
- G4 numerical operations are shared through `activation_codec.py`; product code does not import benchmarks.
- Variable microbatch capacities preserve token rows/hidden width, grow per position, and reuse storage for short windows.
- The shared estimator charges payload, per-token FP32 scales, pinned rounding and host scratch; CUDA scratch is separate. Leases precede allocation and survive failed drains.
- Resolved `memory_plan.effective_k` must match HF accumulation. C3 reserves explicitly supplied routing bytes before allocating weights. Neither path silently changes k or falls back to another codec/spill.
- The backward consumer releases its previous restored boundary before the next restore.
- Recipe override checks remain; the blanket schema-default snapshot assertion was removed rather than re-pinned for an unused opt-in field.

## Execution and provenance

Private Kaggle kernel: [production validation](https://www.kaggle.com/code/batyrkhan2131123/soup-p3-fp8-production-validation-20261105).
Version 2 contains the experiment matrix; version 3 is the final regression-only run.
The 12 production/harness file hashes are identical between those versions (`v2/run.json`, `v3/run.json`).
`manifest.json` lists evidence hashes and kernel references. Raw JSON is authoritative.

Official Input: `deepseek-ai/deepseek-r1/transformers/deepseek-r1/2`, mounted at
`/kaggle/input/models/deepseek-ai/deepseek-r1/transformers/deepseek-r1/2`.
All 163 indexed shards were present. Config: 61 layers, hidden 7168, 256 routed experts, top-k 8.
Bounded reads used actual embedding rows, layer-3 router weight/bias and an expert slice.

Environment: Tesla T4, one visible CUDA device; Python 3.13.16, PyTorch 2.11.0+cu130,
Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0, Safetensors 0.8.0.
Deterministic algorithms enabled, TF32 disabled, experiment Torch threads set to 1.
Kaggle's optional `torchao 0.10.0` preinstall was removed because PEFT requires >=0.16.0;
these storage/FP32/BF16/NF4 paths do not use TorchAO. Tests used `/kaggle/working` for temporary
files, avoiding `/tmp`'s overlay-filesystem spill refusal. The C3 setup test declares its disk
admission input explicitly; it is not an NVMe performance measurement.

Verification used Kaggle CLI submission/output retrieval, not local execution:

| Final check | Result |
|---|---|
| CPU regression selection | 1857 passed, 64 skipped |
| CUDA-marked selection | 55 passed, 1 skipped |
| Ruff: src, scripts, tests, benchmarks | All checks passed |
| Actual SFT trainer with FP8 | Mixed lengths and short windows at k=1/4; public C3 setup/train also exercised |
| Failing-before feature check on base | Missing `soup_cli.utils.fp8_activations`, as expected |

The CUDA skip requires two visible GPUs. These are selected regression suites, not the entire repository suite.
See `v3/cpu-tests.log`, `v3/gpu-tests.log`, JUnit XML and `v3/ruff.log`.
Initial environment and lint failures are retained separately; they are not counted as passes.

## Tiny DeepSeek matrix

Production `L2LState`, `run_window`, `_store_for` and `L2LStep`; native tiny DeepSeek-V3,
46 trainable adapter tensors, FP32 compute, standalone momentum-free SGD. Two full windows
plus a half-size final window for k>1; variable row counts, sequence lengths and masked labels.
The reference starts each comparison from the current candidate adapters: this establishes
per-window agreement, not independently diverging multi-step trajectories.

CPU and CUDA had the same window verdicts:

| k | Uncompressed | FP8 windows 0 / 1 / 2 | FP8 accepted updates |
|---|---|---|---:|
| 1 | PASS, 3 updates | gradient FAIL / routing REFUSED / PASS | 1 |
| 4 | PASS, 3 updates | REFUSED / REFUSED / REFUSED | 0 |
| 16 | PASS, 3 updates | REFUSED / REFUSED / REFUSED | 0 |
| 64 | PASS, 3 updates | REFUSED / REFUSED / REFUSED | 0 |

All uncompressed losses, gradients and parameter updates passed bitwise comparison;
gradient relative L2 was 0 and minimum cosine was 1. Injected routing-corruption controls passed.
Every refused window discarded partial gradients without an optimizer update. All store/routing
reservations returned to zero after cleanup.

FP8 k=1/window 0 retained identical routes and forward losses, but CUDA gradient relative L2
was **0.051908965** (limit 0.05), with minimum tensor cosine **0.949918628** (minimum 0.99).
CPU produced the same failure. E1 passed in every FP8 arm; E2 failed at k=16 and k=64 under
the unchanged 0.10 residual-branch RMS bound. No tolerance or routing guard was relaxed.
See `v2/p3-cpu.json` and `v2/p3-cuda.json`.

The bounded attached-R1 router probe changed the selected expert set for **1 of 8 embedding rows**
after FP8 restoration; E1 RMS row-relative error was 0.026120859. These are embedding rows fed
directly to the router, not hidden states produced by the preceding decoder layers.
This probe is not an estimate of the model's production routing-mismatch rate.

## Full-dimension activation storage

Synthetic BF16 boundaries with the attached model's 61-layer, 7168-hidden shape and 64 tokens
per microbatch. The product store performed every put/get: 61, 244, 976 and 3904 chunks per
direction respectively. All four cases passed E1 and released their budget on close.

| k | Tokens/window | Owned payload + scales, bytes | Host reservation, bytes | Sampled process RSS peak, bytes |
|---|---:|---:|---:|---:|
| 1 | 64 | 27,999,488 | 31,998,017 | 1,031,913,472 |
| 4 | 256 | 111,997,952 | 127,991,873 | 1,131,577,344 |
| 16 | 1024 | 447,991,808 | 511,967,297 | 1,516,646,400 |
| 64 | 4096 | 1,791,967,232 | 2,047,868,993 | 3,064,684,544 |

Each reservation matched the shared estimator exactly. Pinned payloads round independently;
the scale slab rounds once. The CUDA scratch estimate was 15,599,617 bytes, separate from host
reservation. Observed CUDA allocated peak was 13,764,608 bytes for this storage-only probe,
including input/restored chunks and diagnostic scratch. Worst layer RMS row-relative error
was 0.027575 or less. RSS is sampled and includes process baseline/caches; it is not a hard limit.

These figures exclude model weights, optimizer state, a complete training step and the MoE
routing reservation. They do not establish an 8 GB whole-model fit or a training speedup.

## Remaining blockers and next steps

1. **DeepSeek FP8 is not accepted.** Keep the uncompressed path while a separately reviewed
   numerical/routing-preservation design is evaluated. Forcing saved routes or disabling the
   guard would change the existing correctness contract; this implementation does neither.
2. **B1.4 integration remains separate.** Pass the resolved planner/budget into trainer construction
   before scheduling; retain `effective_k` and the explicit C3 routing reservation contract.
3. **BF16-compute k=1/4/16/64 matrix is not run.** BF16 storage conversion was tested, and existing
   BF16/autocast regressions ran, but the new complete matrix used FP32 compute.
4. **No 671B optimizer step, 8 GB run, or ×50 claim.** Full model Inputs and bounded tensor probes
   are not giant-model training. Further verification stays on Kaggle until explicitly authorized otherwise.
5. Expert fetches still scale with microbatches: the first uncompressed windows recorded
   30/147/597/2421 fetches at k=1/4/16/64. These are runtime counters, not physical SSD reads.
   FP8 storage does not solve expert-weight I/O amortization.

Reproduce the experiment inside Kaggle with the Input attached and the recorded source:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python benchmarks/harness/p3_fp8_smoke.py --device cuda \
  --model-input /kaggle/input/models/deepseek-ai/deepseek-r1/transformers/deepseek-r1/2 \
  --output /kaggle/working/p3-cuda.json
```

Exit 1 is the recorded scientific gate failure, not a successful training acceptance.
