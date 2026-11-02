# G1 ticket 03: LoRA on a frozen NVFP4 base, small model

All numbers on this page are **synthetic**: a small random model, random tokens and a fixed random
token mapping as the task. They say whether gradients reach the adapters through native FP4 GEMM.
They are not a quality or speed result, and not a Mistral-7B result.

## What runs

- Base: `te.Linear` layers built under `te.quantized_model_init` with `NVFP4BlockScaling`. Weights
  are packed (`NVFP4Tensor`) and `requires_grad` is off. Forward runs under `te.autocast`.
- Adapter: bf16 LoRA on every linear (`LoRALinear`: `base(x) + (alpha / rank) * x A^T B^T`, B starts
  at zero). Rank 8, alpha 16.
- Model: embedding (frozen, bf16), 2 residual feed-forward blocks (256 -> 512 -> 256), head
  (256 -> 64). 5 adapted linears. Batch 8, sequence 32 (256 tokens).
- Task: predict `mapping[token]` for a fixed random permutation of 64 tokens. Fresh random tokens
  every step.
- Training: AdamW, lr 3e-3, no weight decay, 60 steps, seed 0. Harness:
  `benchmarks/harness/g1_lora_nvfp4.py`. CPU tests: `tests/test_g1_lora_nvfp4.py`.

## The pass rule, written before the GPU run

| check | rule |
|---|---|
| `grad_norm_positive` | total gradient norm of the first step is finite and above 0 |
| `adapters_changed` | every adapter's A and B differ from their initial values |
| `loss_decreased` | mean of the last 5 losses is below the mean of the first 5 |
| `finite` | no NaN or Inf in any loss, in the first gradient norm, or in the adapters |
| `base_frozen` | no parameter outside the adapters has `requires_grad` |

`lora_a` has gradient 0 at step one because `lora_b` starts at zero. That is expected, so the
first-step check reads `lora_b` per adapter and the total norm. A reaches a nonzero value from step 2.

## Result (synthetic, RTX 5090, 2026-10-22)

Machine: Vast.ai instance 57232401, RTX 5090 (compute capability 12.0), torch 2.14.1+cu130,
TransformerEngine 2.20.2. Raw record: `benchmarks/results/g1-lora-nvfp4/report.json`,
environment: `environment.txt`.

**All five checks passed on the first run.** Base weight storage was `NVFP4Tensor` for every layer.

| quantity | value |
|---|---|
| first-step total gradient norm | 0.4377 |
| first-step `lora_b` gradient norm, head | 0.4302 |
| first-step `lora_b` gradient norm, `down.1` / `down.0` | 0.0296 / 0.0338 |
| first-step `lora_b` gradient norm, `up.1` / `up.0` | 0.0469 / 0.0473 |
| loss, first 5 steps | 4.307, 4.279, 4.250, 4.141, 4.061 |
| loss, last 5 steps | 0.013, 0.013, 0.011, 0.010, 0.010 |
| random-guess loss (ln 64) | 4.159 |

The gradient reaches `up.0`, the adapter furthest from the loss along the residual branch, so it
crossed the frozen NVFP4 layers and not only the head. This does not show that the backward
GEMMs ran in FP4: the harness has no hook or comparison against a bf16 backward, and the backward
runs outside `te.autocast`. TransformerEngine reads the recipe saved at forward time, which is
unchecked here. Nonzero gradients rule out the zero-gradient failure of TransformerEngine before 2.19.

## Limits

- The harness as committed computes the loss inside the autocast context. The GPU run above used
  the earlier version, which computed it outside. The loss is not a TransformerEngine op, so the
  numbers are not expected to differ, but the GPU run was not repeated.
- One run, one seed, one small model. It shows the path works, not how well it trains.
- The task is a lookup table, so a near-zero loss shows the adapters learn, not that NVFP4 preserves
  model quality. That is the quality gate on a real model (tickets 04 and 05).
- No timing or memory was measured. The adapter is a plain bf16 matmul around `te.Linear`, with no
  fusion.
- Shapes follow ticket 02: in_features a multiple of 32, tokens and out_features a multiple of 16.
  `check_model_shape` refuses other shapes with `UnsupportedShapeError`; there is no NF4 fallback.
