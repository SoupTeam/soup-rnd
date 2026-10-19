# Streamed Liger 7B: arithmetic repair and measurement

**Verdict: strict numerical parity passes; no streaming performance benefit was observed.**
All three component selections pass the original numerical gate. The measured combined
integration is 0.48% slower in input-token
throughput on this fixture, with unchanged peak Torch allocated/reserved memory.
This is a compatibility integration: Liger supplies the BF16 SwiGLU forward, Soup
supplies the width-4096 RMS forward, and backwards preserve HF arithmetic in ATen.

## Frozen scope

- Executable source: `94d1870dddd951d0f0ce78edd7026519303f3bc6`, branch
  `koko/d2-liger-streaming`. The saved source archive predates the commit but its
  executable files match the committed checkout after LF normalization.
- Source archive SHA-256:
  `31270d3ac0ee6cc6ff7c9d1d47422ab2a71683e9f7edadef17b96243ff875e77`.
- NVIDIA RTX 3090, native BF16; Python 3.12.3, Torch 2.11.0+cu126,
  Triton 3.6.0, Liger 0.8.1, Transformers 5.17.0, PEFT 0.20.0, TRL 0.29.1.
  Exact packages, driver, GPU UUID and telemetry are retained with the results.
- `openlm-research/open_llama_7b_v2`, revision
  `e5961def23172a2384543940e773ab676033c963`; unchanged tensor names/dtypes in
  the conversion from original `.bin` shards to safetensors. Both hashes retained.
- Actual Soup SFT with RAM layer streaming, two slots, pinned RAM, per-layer
  checkpoints, dense frozen BF16 base, ordinary PEFT LoRA on all seven projections,
  rank/alpha 4, dropout zero. `fast_lora=false`.
- Seed 792; identical explicitly initialized nonzero LoRA A/B, generator 123.
  Two optimizer updates, four microbatches, batch 1, accumulation 2, actual length 1024.
- The same 32 frozen synthetic arithmetic rows as the earlier report, data SHA-256
  `313d17f90f4b62004a1d8232ba9bc0b19388a65e933c930ddc9aecfe7ffbf0f9`.
  These rows are a numerical fixture, not task-quality evidence.
- Identical deterministic SDPA/CUBLAS policy in all correctness and timing arms.
  The original 24-case test function and the pretrained capture/comparison functions
  have unchanged ASTs. The gate remains `rtol=1e-2, atol=1e-5`, plus losses matching
  to three decimal places. Initial adapters and batches must match exactly.

## Arithmetic repair

The earlier [7B failure](liger-streaming-7b-20261019.md) is preserved with its
original source and raw archive. Merely retaining cast boundaries and disabling
RMS backward FMA was insufficient. The compatibility implementation now uses:

| Operation | Qualified implementation |
|---|---|
| RMS forward, width 4096 | Soup Triton kernel: four sequential vector accumulators and explicit descending cross-warp folds matching ATen |
| RMS forward, other widths | Liger's unweighted FP32 forward, with HF cast/affine outside |
| RMS backward | ATen inverse RMS and HF derivative operation order; out of place |
| BF16 SwiGLU forward | Liger fused SiLU/product, preserving the rounded BF16 SiLU boundary |
| SwiGLU backward | Rounded product gradient followed by ATen SiLU backward; rounded SiLU for the other input gradient |
| FP16 SwiGLU | FP32 Liger SiLU core and external cast/product boundary |

The width-4096 RMS forward is a Soup compatibility kernel; the backwards use ATen.
These results do not qualify upstream Liger's complete fused forward/backward stack.
Projection modules, canonical parameter names, checkpoint ownership, streamed head,
RoPE and loss implementation remain the existing Soup/HF/TRL paths.

Implementation references: [Liger RMSNorm](https://github.com/linkedin/Liger-Kernel/blob/v0.8.1/src/liger_kernel/ops/rms_norm.py)
and [Liger SwiGLU](https://github.com/linkedin/Liger-Kernel/blob/v0.8.1/src/liger_kernel/ops/swiglu.py).
The repaired RMS source expresses the required ATen accumulation order explicitly.

## Numerical results

The unchanged native CUDA matrix passes **24/24, no skips or failures**:
components separately/together, FP16/BF16, RAM/disk, dense/NF4, on the small model.
An additional **12/12 zero-tolerance GPU regressions** pass: the captured FP16
boundary, two real 7B BF16 boundaries, every finite BF16 SiLU gate value with
nonunit up values and both gradients, and width-4096 RMS rows 1/4/8/16 in both dtypes.
These stricter regressions add coverage; they do not replace the original gate.

| Pretrained 7B selection | Gradients / 896 | Final adapters / 448 | Logits / 4 | Losses to 3 decimals |
|---|---|---|---|---|
| RMSNorm | Pass | Pass | Pass | Pass |
| SwiGLU | Pass | Pass | Pass | Pass |
| RMSNorm + SwiGLU | Pass | Pass | Pass | Pass |

The saved GPU comparison summaries report all 448 initial tensors and 16 batch
tensors matching, all 1,813 tensors per comparison passing, and every group's
maximum absolute difference equal to zero. No NaN occurred. The four independently
captured trace files also had identical SHA-256 on the GPU host before export.
Native small-model tests retain public PEFT reload checks. **Independent local
tensor replay is unavailable:** raw tensor and adapter payloads did not finish
transferring before the financial guard deleted the instance. The retained JSON,
JUnit and logs establish the GPU-run results but do not replace those missing bytes.

## Streaming speed and allocator memory

One synchronized **ABBA** cycle: two independent processes per arm, three warmup
and five measured optimizer steps per process. Each step processes 2 x 1024 input
tokens. The measurement excludes initial batch fetch, setup, saving and evaluation.
Initial adapter and dataset hashes match across all four timing processes. The
table aggregates per-process step medians and per-process allocator peaks.

| Quantity | Ordinary streamed PEFT | Combined compatibility integration |
|---|---:|---:|
| Median seconds / optimizer step | 4.0637 | 4.0832 |
| Processed input tokens / second | 503.98 | 501.57 |
| Peak Torch allocated GiB | 1.8448 | 1.8448 |
| Peak Torch reserved GiB | 2.1504 | 2.1504 |

Throughput change: **-0.48%**.
Allocated-memory reduction: **0.00%**;
reserved-memory reduction: **0.00%**.
Torch allocator peaks exclude CUDA context/driver allocations. GPU clock/power,
temperature and visible process samples are retained in `gpu-telemetry.jsonl`.
One ABBA cycle supplies a bounded observation, not a broad speedup or an uncertainty
estimate across workloads. The resident 1.5B/3B VRAM reductions are different experiments.

## Evidence, budget and cleanup

Retained results live outside Git under `output/liger-7b-final3-20261019/`:
JUnit, comparison/capture summaries, logs, environment, model/data manifests,
complete timing samples, immutable executable source archive, local GPU telemetry
and provider receipts. A compact machine-readable summary is committed beside
this report as `liger-streaming-7b-repair-20261019.json`.

- Verified local metadata archive: **8,650,907 bytes**, SHA-256
  `657474a0fb716d7900a6011707976bde764d1f01d128b0769e0ed01ec42aac14`.
- **Artifact-retention blocker:** neither the 2,286,714,413-byte full archive nor
  its 505,916,742-byte losslessly deduplicated export finished transferring.
  The first packing operation also hit its timeout and was retried. Only
  36,536,320 bytes of the compact export reached local partial chunks; these
  chunks are not a verified archive. The financial guard then deleted the host.
  Raw gradients, adapter weights and complete traces from this fresh run are
  lost. No independent local numerical replay is claimed. A fresh GPU capture
  would be needed to close this retention gap; no further instance was rented.
- Provider-observed cumulative cost across all task instances and setup attempts:
  **USD 1.919 / USD 2 maximum**.
  Settlement can lag; the conservative all-in guard bound was USD 1.990,
  including network/storage reserves. Guard and deletion receipts are retained.
- All task instances are verified absent. No separate persistent volume was created.
- An interrupted A40 follow-up lost its unexported raw archive during the pause.
  Its displayed timing figures are excluded from this retained measurement verdict.
  The earlier complete A40 archive remains available.

Local validation: **170 passed, 36 CUDA skips**, plus **52 passed** for README
translation synchronization and changelog checks, with full-path Ruff clean.
The fresh native GPU tests above establish execution beyond those CPU skips.
Full CI is dispatched on the final documentation commit; completion is not awaited
at the user's request. See the
[branch CI history](https://github.com/SoupTeam/soup-rnd/actions/workflows/ci.yml?query=branch%3Akoko%2Fd2-liger-streaming).
7B FP16, disk/NF4, other model families, Fast-LoRA + Liger, and task quality remain
outside the qualified scope. The option remains experimental and opt-in.
