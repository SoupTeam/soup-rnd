# ZERO-2 stage 2b result (run 2026-11-18, Vast RTX 4090, Estonia)

Protocol: `ZERO2_2B_PROTOCOL_20261119.md`. Runner sha256 `cec3c0d0…c5ad` (matches), Soup
`6ac80238`, triton 3.8.0. Model download 63 s, sha256 check passed. Instance 61318260 destroyed
after the tgz was copied (~25 min rented, about $0.2).

## Decisions (as fixed)

| Decision | Result |
| --- | --- |
| `nf4_bit_exact` | **true**: all 5 shapes bit-identical, dequant and forward linear |
| `nf4_train_verdict` | **NO_CLEAR_DIFFERENCE**, speed-up 0.997 (m 2.138 / 2.087, t 2.148 / 2.092 s/step) |
| `fits_8gib_plain_keep16` | **false**: m8k OOM at 7.90 GiB allocated (asked 122 MiB more) |
| `fits_8gib_plain_keep16_host` | **false**: m8h ERROR, a bug of `embed_on_host` (below), not a memory result |
| `eval_ok` | **true** |

So the Triton NF4 kernel gives no gain in training and stays opt-in, off for stage 3.

## NF4 kernel

Bit-exact on the GPU in both entry points; the `t` arms' counters show it served every dequant
(23 808 backward, 47 616 forward + recompute calls). One dequant, mean of 50 (CUDA events):

| Shape | bitsandbytes ms | Triton ms |
| --- | --- | --- |
| 8192 x 4096 | 0.097 | 0.094 |
| 1024 x 4096 | 0.064 | 0.041 |
| 4096 x 4096 | 0.064 | 0.042 |
| 12288 x 4096 | 0.148 | 0.142 |
| 4096 x 12288 | 0.147 | 0.142 |

Reading (not gated): on the small matrices one launch instead of two saves ~0.02 ms; on the large
ones both are bound by memory bandwidth and equal. Per step that is well under 1 % of ~2.1 s, so
it disappears in run noise. On a T4 (the hackathon's GPU, ~300 GB/s, launch-bound) the same
kernel measured 1.85x; on a 4090 the dequant is not where the time is. The next kernel has to
remove memory traffic (fused head + CE, or dequant fused into the GEMM), not launches.

## Arms

| Arm | Setup | Status | s/step | tok/s | torch peak alloc | nvidia-smi peak |
| --- | --- | --- | --- | --- | --- | --- |
| m24a | plain, fp32 upcast | OK | 2.138 | 377 | 14.58 GiB | 17 898 MiB |
| t24a | + Triton NF4 | OK | 2.148 | 375 | 14.58 GiB | 15 962 MiB |
| t24b | + Triton NF4 | OK | 2.092 | 385 | 14.58 GiB | 15 962 MiB |
| m24b | plain, fp32 upcast | OK | 2.087 | 386 | 14.58 GiB | 15 962 MiB |
| k24 | plain, `kbit_keep_16bit` | OK | 2.018 | 399 | 8.79 GiB | 11 488 MiB |
| s8 | 8 GiB, stream RAM + chunked CE | OK | 2.443 | 330 | 3.39 GiB | 4 304 MiB |
| m8k | 8 GiB, plain, keep16 | OOM | — | — | 7.90 GiB | 8 624 MiB |
| m8h | 8 GiB, plain, keep16 + embed_on_host | ERROR | — | — | 7.19 GiB | 7 872 MiB |

Speeds across configurations are descriptive. 24 steps, ~830 tokens per step.

## Loss gap explained

Largest relative per-step loss difference against the streamed arm s8: **k24 0.9 %**, m24a 6.2 %.
With the fp32 upcast removed the plain path matches the streamed one; stage 2's gap came from
PEFT's fp32 upcast, not from streaming. (m/t arms agree with each other within run
nondeterminism, as the bit-exact kernel should.) `kbit_keep_16bit` also cut the plain path's
peak from 14.6 to 8.8 GiB.

Consequence for stage 3: with `kbit_keep_16bit` on the resident path, the resident and streamed
setups compute the same thing and can be compared.

## m8k and m8h

- m8k: keep16 alone peaks at 8.79 GiB uncapped (k24); 8 GiB is ~0.8 GiB short.
- m8h failed at step 1: `F.embedding` got CPU indices but a CUDA weight. `embed_on_host` moved the
  weight to pinned host memory (log: 1.89 GiB off the GPU), but something later moved the model's
  parameters back to the GPU (the weight is still a registered parameter, so a `model.to(cuda)`
  takes it along). Fix: keep the host copy outside the module's parameters. With the 1.89 GiB off,
  k24's 8.79 GiB becomes ~6.9 GiB, under the cap; that is an estimate until measured.

## Eval

lm-eval GSM8K, 50 questions, NF4 model passed in as a loaded `HFLM`: base 0.80 strict / 0.78
flexible, m24a's adapter 0.82 / 0.82 (24 steps, 50 questions: not a quality result).
**2.95 s per question** (base), 3.25 s with the adapter, batch 8, on a 4090.

## For stage 3

- 8 GiB: streamed (s8). 4 GiB: streamed + L2L. Resident runs use `kbit_keep_16bit`.
- Triton NF4: off (no gain on a 4090).
- Eval budget: ~3 s per GSM8K question; 500 questions ≈ 25 min per model.
- Plain 8 GiB: only after the `embed_on_host` fix is measured (a 1-arm check).
