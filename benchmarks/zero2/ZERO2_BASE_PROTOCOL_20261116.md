# ZERO-2 stage 2 protocol: Qwen3.5-9B baseline in 8 and 4 GiB (fixed 2026-11-16, before the run)

Author: Nurseiit. Plan: ZERO-2 doc (https://claude.ai/artifact/3GRxLb8h4FZkYnZpLRNtAA). A
feasibility and measurement run, no method comparison: does 9B QLoRA fit 8 GiB without
streaming, does it fit 4 GiB with streaming + L2L, how fast is each, where the time goes
(for the Triton-kernel work), and does the GSM8K eval pipeline work and how long it takes.

## Code under test

Soup `research/layer-select` at `63443f153a9fe5accaed0490df63571a01d3cfd7` =
`research/l2l-two-gpu` (`2c0b1ffc`: chunked CE, qwen3_5 in L2L, fp16 in L2L,
`fp16_init_scale`, L2L split) + `63443f15` (`training.lora_layers`, not used in this run).

## Fixed setup

| Item | Value |
| --- | --- |
| Machine | Vast, one 24 GB GPU (RTX 3090 or 4090), >= 64 GB RAM, >= 120 GB disk; GPU, CPU and RAM recorded in `env.json` |
| Model | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (32 layers, vocab 248 320) |
| Data | `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866`, `main` train, first 48 rows in file order, alpaca (question -> worked answer) |
| Soup config, all arms | sft, max_length 1024, epochs 1, lr 1e-4, batch 1, k = 4 (gradient accumulation), 12 optimizer steps, seed 0, gradient checkpointing, LoRA r 16 / alpha 32 / dropout 0 on q,k,v,o,in_proj_qkv,in_proj_z,out_proj,gate,up,down of all 32 layers, quantization 4bit |
| VRAM cap | `torch.cuda.set_per_process_memory_fraction(cap GiB / total)` (allocator only; the CUDA context comes on top), streamed arms also `stream_vram_override` = cap |
| Arms, fixed order, one process each | m24 no cap, plain · m8 8 GiB, plain · m8s 8 GiB, streaming (RAM tier) + chunked CE · m4 4 GiB, streaming (RAM tier) + L2L + chunked CE · p24 = m24 + torch.profiler over optimizer steps 3–4 |
| Eval smoke | lm-eval `gsm8k` (its default few-shot), `limit` 50, batch 8, model NF4 bf16 compute: e_base (no adapter), e_m24 (m24's adapter) |
| Runner | `zero2_base.py`, sha256 `d1511826bfbc9b8bed29ae8cab11decb14dd3cf134bf62a202b17ad963f486f2`, started by `run_zero2_base.sh` (checks the hash) |

## What is decided

1. `fits_8gib_plain`: m8 runs 12 steps without error or OOM.
2. `fits_8gib_streamed`: m8s does.
3. `fits_4gib_streamed_l2l`: m4 does.
4. `eval_ok`: both eval smokes return a GSM8K score.

Reported, not gated: s/step over steps 2–12 and tok/s per arm, torch and nvidia-smi peaks,
host RAM minimum, the top CUDA kernels of p24 by self CUDA time, eval wall time per question.
Speed between arms is descriptive (one process each, same machine, fixed order, no A-B-B-A);
no arm is called faster than another from this run. The 12-step adapters are not a quality
result: e_m24 tests the adapter path only.

## How the result is used

The stage-3/4 protocols take from it: which memory setup runs each budget, the per-step time
(to size training subsets so a run lasts 1–2 h), the eval time per question (to size the test
subsets), and the profile (which kernels to try first). A failed arm is reported with its
error and not rerun under this protocol.

## Not tested

Layer selection, LISA, quality, kernels, other datasets, two GPUs.
