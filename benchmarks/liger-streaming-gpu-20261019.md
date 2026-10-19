# Streamed Liger: A40 correctness correction and minimal 7B run

**2026-10-19 result:** the unchanged 24-case CUDA matrix passed, a separate
zero-tolerance FP16 regression passed, and all four pretrained 7B pilots completed
finite training and exact public PEFT adapter reload. Both rented instances were
deleted and verified absent. Current provider charges total **USD 0.361** against
the user's cumulative **USD 2** ceiling; billing settlement may lag.

## Correction

The original `f7649c72` integration failed all 24 CUDA cases. Preserving HF's
low-precision cast boundaries around FP32 Liger cores reduced this to four
FP16/dense RMSNorm failures. This keeps the real upstream Liger operations and
existing PEFT projections, but deliberately gives up full fusion and allocates
FP32 intermediates.

The remaining failure was isolated on an actual retained norm input: identical
forward output and inverse RMS statistics, but CUDA fused multiply/add in the
Liger backward rounded one scaled FP16 input gradient to `-3092` instead of HF's
`-3094`. Compiling the same upstream Triton backward with
`enable_fp_fusion=False` made that gradient match exactly. The compatibility
autograd class inherits Liger's forward and launches its actual backward kernel;
it does not substitute a Torch backward or change Liger globally.

The captured synthetic input is now
`tests/fixtures/liger_fp16_rms_rounding.safetensors` (3,208 bytes). Its CUDA
regression requires **rtol=0, atol=0** for both output and input gradient.

## Gates and environment

| Check | Result |
| --- | --- |
| Native CUDA matrix, all three component selections | 24 passed, zero skips/failures |
| FP16/BF16 × RAM/disk × dense/NF4 | All cells passed |
| Original tensor tolerances | Unchanged: rtol=1e-2, atol=1e-5 |
| Original per-step loss criterion | Unchanged: match to three decimals |
| Captured FP16 rounding boundary | 1 passed, zero tolerance |
| Independent CPU replay of the final 24 raw arm pairs | All tensor groups and losses passed |
| Focused local CPU checks | 67 passed, 25 CUDA cases skipped |
| Ruff across src/scripts/tests/benchmarks | Clean |

The full original GPU test function, including its input data and numerical
assertions, was unchanged. Its source SHA-256 is
`fcee6f07591ccae1305543805751213eb1b3328fd20e57b4e9b94260f779768d`.
Actual Liger autograd execution is still observed in the training graph.

Hardware: Vast.ai NVIDIA A40, compute capability 8.6, driver 595.91.07.
Environment: Python 3.12.3, torch 2.11.0+cu126, Triton 3.6.0, Liger 0.8.1,
transformers 5.17.0, PEFT 0.20.0, TRL 0.29.1, bitsandbytes 0.50.1.
The 7B tokenizer additionally required protobuf 7.36.2; its initial missing-dependency
failure and the resumed run are both retained. The full repository suite was not
rerun in this follow-up; the local count above is the focused scope.

## Pretrained 7B pilot

Model: `openlm-research/open_llama_7b_v2`, revision
`e5961def23172a2384543940e773ab676033c963`, Apache-2.0, ungated. Upstream `.bin`
shards were converted without arithmetic to safetensors for Soup streaming;
original and converted file hashes are retained.

Each arm ran five optimizer steps (two warmup, three recorded), batch 1,
accumulation 2, seed 792, LoRA rank/alpha 4, dropout 0, all seven attention/MLP
projections, dense weights, two decoder buffers, pinned RAM source, BF16.
The input is 32 explicitly synthetic arithmetic chat rows. `max_length=1024` is
a cap; the training examples were shorter. This is not a fixed-length-1024
throughput experiment.

| Arm | Installed patches | Steps | Finite loss/gradient/final adapters | Public PEFT reload |
| --- | --- | ---: | --- | --- |
| Baseline | None | 5 | Pass | 448 tensors exact |
| RMSNorm | 65 norms | 5 | Pass | 448 tensors exact |
| SwiGLU | 32 MLPs | 5 | Pass | 448 tensors exact |
| Both | 65 norms, 32 MLPs | 5 | Pass | 448 tensors exact |

The 7B loss histories are not identical. This pilot establishes finite execution
and export/reload only; it does **not** qualify real-7B tensor parity or
three-decimal loss equality. Real-7B disk/NF4, other checkpoints/architectures,
final task quality, hallucinations, speedup and memory improvement are not
established. Three timing samples per arm are retained without an acceleration
verdict.

## Evidence and cleanup

Local raw evidence is intentionally ignored by Git:

- Original failure: `output/liger-gpu-minimal-20261019/`.
- Correction and final run: `output/liger-gpu-fix-20261019/`.
- Final `retained/results/`: JUnit XML, full logs, raw tensors, model/data/source
  manifests, all four pilot metrics/adapters/reload receipts, and harness sources.
- Final `retained/attempt1-results/`: the intermediate 20-pass/4-fail run and
  retained-input, inverse-RMS and CUDA FMA diagnostic evidence.
- `offline-replay.json`, `local-code-identity.json`, `cumulative-billing.json`,
  and `vast-deletion.json`: independent local checks, source identity, exact-instance
  billing and deletion receipts.

Measured source archive SHA-256:
`7d6d58cc145bba82566ae1064a5b44249bc1335c0ae890588473373fd3a7943d`.
Only a production docstring clarification followed GPU execution; an executable
AST comparison against that archive passed. The final result archive is
596,614,082 bytes, SHA-256:
`76e56f9d359399f311d2586812323e344b44980fd6b42f7d528ce484e6d5fbf7`.

Results were downloaded and hash-verified before deletion. Exact instances
`56660857` and `56662809` were verified absent through the provider API. No
separate persistent volume was created, and no unrelated instance or NoRA
worktree was modified. No push or merge was performed.
