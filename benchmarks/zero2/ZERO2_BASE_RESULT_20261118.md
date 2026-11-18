# ZERO-2 stage 2 result: Qwen3.5-9B QLoRA in 24 / 8 / 4 GiB (run 2026-11-18, Vast RTX 4090)

Protocol: `ZERO2_BASE_PROTOCOL_20261116.md` + `ZERO2_BASE_ADDENDUM_A_20261119.md` (download only).
Runner sha256 `d1511826…86f2` (matches), Soup `63443f15`. Machine: Vast 61313031, RTX 4090 24 GB,
EPYC 9654 (32 vCPU), 116 GB RAM, torch 2.14.1+cu132, transformers 5.22.2, bitsandbytes 0.50.2,
fla 0.6.0. Model download 111 s (curl, token), sha256 check passed. All arms + evals: 393 s wall.
Instance destroyed after the tgz was copied (~35 min rented, about $0.35).

## Decisions (as fixed)

| Decision | Result |
| --- | --- |
| `fits_8gib_plain` | **false**: m8 OOM before step 1 |
| `fits_8gib_streamed` | **true**: m8s ran 12 steps |
| `fits_4gib_streamed_l2l` | **true**: m4 ran 12 steps |
| `eval_ok` | **false**: both eval smokes failed at model load (runner bug, below) |

## Arms

| Arm | Setup | Status | s/step (steps 2–12) | tok/s | nvidia-smi peak | torch peak alloc |
| --- | --- | --- | --- | --- | --- | --- |
| m24 | no cap, plain | OK | 1.73 | 480 | 17 902 MiB | 14.58 GB |
| m8 | 8 GiB, plain | **OOM** | — | — | 7 794 MiB | 7.19 GB |
| m8s | 8 GiB, stream RAM + chunked CE | OK | 2.18 | 380 | 5 968 MiB | 3.39 GB |
| m4 | 4 GiB, stream RAM + L2L + chunked CE | OK | 2.00 | (not logged) | 4 250 MiB | 3.39 GB |
| p24 | m24 + torch.profiler on steps 3–4 | OK | 5.55 (profiler on) | 149 | 15 978 MiB | 14.58 GB |

s/step comes from whole-second timestamps over 11 steps (about ±0.1 s). Steps hold ~830 tokens
(4 GSM8K rows of ~200 tokens), far below max_length 1024. Speeds between arms are descriptive only
(one process each, fixed order). Host RAM never below 94 GB free, so the RAM tier never paged.

## Readings (not gated)

1. **m8 fails at setup, not in training.** `prepare_model_for_kbit_training` upcasts the
   non-quantized weights to fp32; the 3.79 GiB allocation that broke the cap is one of the two
   untied 248 320 × 4096 embedding matrices (input embedding / lm_head). The 4-bit layers fit;
   the fp32 vocabulary matrices do not. Keeping them in bf16/fp16 (what all four hackathon
   entries do for Qwen2's tied embedding) is the obvious thing to test for a plain 8 GiB path.
2. **On this host streaming is cheap**: 8 GiB streamed 2.18 s/step and 4 GiB + L2L 2.00 s/step
   against 1.73 s/step unbounded (descriptive). With 116 GB RAM and PCIe 4.0 the RAM tier is not
   the bound it was on Kaggle's disk. Both streamed arms peaked at the same torch 3.39 GB.
3. **Streamed arms' losses differ from m24's.** p24 (same path as m24) matches m24 within 0.5%
   on every step; m8s and m4 match each other but differ from m24 by up to 9% (step 3:
   0.826 / 0.824 vs 0.758), converging by step 12 (0.526 / 0.530 vs 0.522). The streamed path
   (stream tier, chunked CE, fp16) is numerically different. Consequence for stage 3: compare
   layer-selection methods within one memory setup, never across setups.
4. **Profile (p24, steps 3–4, self CUDA time):** `aten::mm` 686 ms (cuBLAS/cutlass bf16 GEMMs);
   NF4 `kDequantizeBlockwise` **368 ms** in 5 952 calls; bf16 copy/cast kernels ~130 ms;
   elementwise mul/add ~100 ms; the DeltaNet kernels (fla `ChunkGatedDeltaRule` fwd + bwd) only
   ~57 ms; `log_softmax` 17 ms. The rows mix op-level and kernel-level entries and are not added up.
   First kernel target: **NF4 dequantization** (the largest non-GEMM item, ~11% of the step),
   then the casts/copies. The vocabulary head is small here because rows are short.

## Eval smoke: failed, not rerun

Both `e_base` and `e_m24` stopped in 7 s at model load:
`TypeError: Qwen3_5ForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'`.
The runner passed `load_in_4bit=True` in lm-eval's `model_args`; transformers 5.x no longer takes
it and wants a `BitsAndBytesConfig`. My runner bug. `s_per_question` in the json is meaningless.
Per the protocol a failed arm is reported, not rerun. The stage-3 eval harness has to load NF4
through `quantization_config` (or its own generation loop) and be smoke-tested before stage 3.

## What stage 3 takes from this

- 8 GiB budget: streamed setup (m8s); 4 GiB: streamed + L2L (m4). Plain 8 GiB only after the
  embedding-dtype fix, under its own protocol.
- Step time ~2 s at ~830 tokens/step on a 4090: a 1–2 h run is ~2 000–3 500 steps at this size.
- Eval cost per question: unknown (smoke failed); must be measured in the fixed harness first.
- Kernels: NF4 dequant first, then casts; vocabulary head once rows are long.
