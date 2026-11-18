# ZERO-2 stage 2b protocol: the stage-2 fixes on Qwen3.5-9B (fixed 2026-11-19, before the run)

Author: Nurseiit. Follows `ZERO2_BASE_RESULT_20261118.md` (stage 2): plain 8 GiB failed at setup
(PEFT's fp32 upcast of a 248k x 4096 vocabulary matrix), the eval smoke failed at model load
(`load_in_4bit` no longer accepted), streamed and plain losses differed by up to 9 %, and the
profile put NF4 dequantization first among non-GEMM kernels. This run checks the fixes.

## Code under test

Soup `research/zero2-fixes` at `6ac802385acba2250f7a09d84d5bf04a81356d6a` = `research/layer-select`
(`d82cd14a`) + `a3f43247` (`training.kbit_keep_16bit`, `training.embed_on_host`, Triton NF4
dequant behind `SOUP_NF4_TRITON=1`) + `6ac80238` (the Triton dequant also serves the 4-bit
forward). The NF4 kernel is adapted from the Soup Kernel Challenge entry of AG TEAM (Apache-2.0).

## Fixed setup

| Item | Value |
| --- | --- |
| Machine | Vast, one RTX 4090 or 3090 (24 GB), >= 64 GB RAM, >= 120 GB disk; recorded in `env.json` |
| Model, data | as stage 2: `Qwen/Qwen3.5-9B` @ `c2022362…`, `openai/gsm8k` @ `740312ad…`, first **96** train rows in file order, alpaca |
| Soup config, all training arms | as stage 2 (sft, max_length 1024, lr 1e-4, batch 1, k = 4, seed 0, gradient checkpointing, LoRA r 16 / alpha 32 / dropout 0 on all 10 projection types of all 32 layers, NF4) but **24** optimizer steps |
| Step time | a TrainerCallback records `perf_counter` at each step end; s/step = mean of the intervals after step 3 |
| VRAM cap | `set_per_process_memory_fraction(cap GiB / total)`; the streamed arm also `stream_vram_override` |
| Runner | `zero2_2b.py`, sha256 `cec3c0d0a0e898027932c55b705138e120ad65e9f8e8035d72bf79064b8bc5ad`, via `run_zero2_2b.sh` |

Arms, fixed order, one process each:

1. `nf4x`: on the GPU, for Qwen3.5-9B's weight shapes [8192, 4096], [1024, 4096], [4096, 4096],
   [12288, 4096], [4096, 12288] (random bf16 weights, NF4, blocksize 64, double quantization):
   the Triton dequant against `bitsandbytes.functional.dequantize_4bit`, and the Triton forward
   (`dequant_linear`) against bitsandbytes' `_dequant_linear_fallback` on a [256, K] input,
   both compared bit for bit; plus the time of one dequant each (CUDA events, mean of 50).
2. `m24a`, `t24a`, `t24b`, `m24b`: no cap, plain QLoRA (stage 2's m24 config); `t` arms with
   `SOUP_NF4_TRITON=1`. A-B-B-A in one session on one machine.
3. `k24`: no cap, plain, `kbit_keep_16bit: true`.
4. `s8`: 8 GiB, stream RAM + chunked CE (stage 2's m8s config).
5. `m8k`: 8 GiB, plain, `kbit_keep_16bit: true`.
6. `m8h`: 8 GiB, plain, `kbit_keep_16bit: true` + `embed_on_host: true`.
7. `e_base`, `e_m24`: lm-eval `gsm8k` (its default few-shot), limit 50, batch 8; the model
   loaded by the runner with `BitsAndBytesConfig` (NF4, double quant, bf16 compute) and passed
   to lm-eval as a loaded `HFLM`; `e_m24` with `m24a`'s adapter through `PeftModel`.

## What is decided

1. `nf4_bit_exact`: every `nf4x` shape is bit-identical in both comparisons.
2. `nf4_train_verdict` from the A-B-B-A s/step: **FASTER** if both `t` arms are faster than both
   `m` arms, **SLOWER** if both are slower, otherwise **NO_CLEAR_DIFFERENCE**. Speed-up reported
   as (m24a + m24b) / (t24a + t24b). The kernel counts as giving a gain only with
   `nf4_bit_exact` and FASTER.
3. `fits_8gib_plain_keep16`: `m8k` runs 24 steps without error or OOM.
4. `fits_8gib_plain_keep16_host`: `m8h` does.
5. `eval_ok`: both eval smokes return a GSM8K score.

Reported, not gated: per-arm s/step, tok/s, torch and nvidia-smi peaks, host RAM minimum, the
NF4 Triton call counts per entry point in the `t` arms, the `nf4x` times, eval time per question.
Loss diagnosis: the largest relative per-step loss difference of `k24` vs `s8` and of `m24a` vs
`s8`; if `k24` sits near `s8` and `m24a` does not, stage 2's difference came from the fp32
upcast, not from streaming. The 24-step adapters and the GSM8K scores of 50 questions are not a
quality result. Speeds of different configurations (k24, s8, m8k, m8h) are descriptive only.

## How the result is used

The memory setup per budget for stage 3 (a plain 8 GiB path if `m8k`/`m8h` pass), whether
`SOUP_NF4_TRITON` is on in stage 3 (only with decision 2 = gain), the eval harness and its
time per question, and whether stage 3 can compare methods across memory setups (loss diagnosis).
A failed arm is reported with its error and not rerun under this protocol.

## Not tested

Layer selection, quality, other datasets, other kernels, two GPUs, the streamed path with the
Triton kernel (the RAM tier dequantizes through the same function, but no A-B-B-A is run there).
