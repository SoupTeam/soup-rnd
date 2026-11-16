# P3: one L2L step on tiny DeepSeek-V3, BF16 vs FP8 vs FP8 + Oracle

Harness: [`benchmarks/harness/p3_fp8_deepseek_step.py`](../../harness/p3_fp8_deepseek_step.py)
at commit `e7fc9db04c9bd5a43d73bdd6a88310ea7486a116`.

```bash
PYTHONPATH=src:. python benchmarks/harness/p3_fp8_deepseek_step.py --device cuda --output out --ks 1 4
```

- **Model:** synthetic native DeepSeek-V3. It has 3 layers (1 dense and 2 MoE), 8 experts, top-2 routing
  and hidden size 128. Weights are NF4 per-expert shards, randomly initialized with seed 23.
  LoRA covers the five attention projections (30 tensors). This is not DeepSeek-R1 671B.
- **Step:** one optimizer step with `k=1` and `k=4`, micro-batch `(1, 64)`. All three arms use the same
  weights, inputs, labels and initial LoRA, and all compute in BF16. Only the boundary
  storage differs.
- **Comparison:** E1 to E3 are measured against the BF16 arm, using the thresholds in
  `benchmarks/harness/g4_rule.py`.
- **Routing guard:** the production guard is used unchanged.
- **Processes:** each (arm, k) runs in its own process.

## GPU: Tesla T4 (`kaggle-t4/`)

Run on Kaggle kernel `batyrkhan2131123/soup-p3-fp8-deepseek-step` v3, using `kaggle-t4/kaggle_runner.py`.
Tesla T4 15 GiB (compute 7.5), driver 580.178.04. Python 3.12.3, torch 2.11.0+cu130,
transformers 5.19.0, peft 0.21.2.

| k | arm | routing | window loss | E1 rms / worst | E2 rms | E3 rel L2 / min cos | store host B | Oracle host B | RSS peak B | CUDA alloc peak B |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | BF16 | ok | 4.196337 | 0 / 0 | 0 | ref | 49,152 | 0 | 1,689,919,488 | 68,242,432 |
| 1 | FP8 | **refused**: layer 2, micro-batch 0, 3 of 64 tokens | none | none | none | none | 25,665 | 0 | 1,659,789,312 | 68,034,560 |
| 1 | FP8 + Oracle | ok (replayed) | 4.196337 | 0.0259 / 0.0310 | 0.0415 | **0.0537** / 0.9928 | 25,665 | 36,352 | 1,687,564,288 | 68,242,432 |
| 4 | BF16 | ok | 4.200341 | 0 / 0 | 0 | ref | 196,608 | 0 | 1,689,686,016 | 68,317,696 |
| 4 | FP8 | **refused**: layer 2, micro-batch 0, 3 of 64 tokens | none | none | none | none | 102,465 | 0 | 1,660,219,392 | 68,102,144 |
| 4 | FP8 + Oracle | ok (replayed) | 4.200341 | 0.0258 / 0.0316 | 0.0406 | **0.0586** / 0.9956 | 102,465 | 47,104 | 1,689,022,464 | 68,317,696 |

On CUDA, the FP8 codec scratch is on the device, so host storage is the payload plus scales:
−47.8% at `k=1` and −47.9% at `k=4` against BF16. The process RSS peak (about 1.69 GB) is dominated by the
CUDA runtime and model, and differs by less than 2% across arms.

## CPU (`cpu/`)

Intel Xeon Platinum 8558 (16 vCPU). Python 3.12.3, torch 2.15.0, transformers 5.23.0, peft 0.21.3.
No VRAM numbers.

| k | arm | routing | window loss | E1 rms / worst | E2 rms | E3 rel L2 / min cos | store host B | Oracle host B | step RSS peak B |
|---|---|---|---|---|---|---|---|---|---|
| 1 | BF16 | ok | 4.184646 | 0 / 0 | 0 | ref | 49,152 | 0 | 932,904,960 |
| 1 | FP8 | **refused**: layer 2, micro-batch 0, 3 of 64 tokens | none | none | none | none | 305,921 | 0 | 910,278,656 |
| 1 | FP8 + Oracle | ok (replayed) | 4.184646 | 0.0260 / 0.0296 | 0.0403 | **0.0534** / 0.9967 | 305,921 | 155,408 | 933,175,296 |
| 4 | BF16 | ok | 4.189848 | 0 / 0 | 0 | ref | 196,608 | 0 | 983,433,216 |
| 4 | FP8 | **refused**: layer 2, micro-batch 0, 3 of 64 tokens | none | none | none | none | 381,953 | 0 | 935,653,376 |
| 4 | FP8 + Oracle | ok (replayed) | 4.189848 | 0.0258 / 0.0300 | 0.0398 | **0.0551** / 0.9961 | 381,953 | 166,160 | 983,449,600 |

Bounds: E1 rms at most 0.05 and worst row at most 0.25. E2 rms at most 0.10. E3 relative L2 at most 0.05
and minimum cosine at least 0.99.

## Reading

- **Natural routing.** FP8 without the Oracle is refused at the first MoE recompute
  (layer 2, micro-batch 0, 3 of 64 tokens) at both `k` values, on both CPU and T4. The guard
  stops the step, and no gradient or update is produced.
- **FP8 + Oracle.** It completes, and its loss equals BF16 exactly. That is expected: the forward
  reads BF16 boundaries before they are stored. E1, E2 and the E3 cosine pass. The E3 relative
  L2 is 5.3–5.9%, just over the 5% bound. This is the known gradient divergence. It
  reflects FP8 rounding of the saved boundaries during recompute, and it does not change the forward
  or the loss.
- **Memory on CUDA.** The codec scratch is on the device, so host storage falls by about 48% even on
  the tiny model (25,665 vs 49,152 B at `k=1`; 102,465 vs 196,608 B at `k=4`). The Oracle adds
  36–47 KB of route records. CUDA allocated peaks are within 0.3% across arms.
- **Memory on CPU.** The codec scratch (about 0.26 MB) is host memory, so on the tiny model FP8 is
  larger than BF16. It pays off at `k` ≥ 16.
- **Memory at DeepSeek-V3 dimensions.** The shared `fp8_activation_allocation` estimator gives these
  host bytes for 61 layers, hidden size 7168 and micro-batch `(1, 64)`. These are estimates from the
  same function the store and planner use, not measurements.

  | k | CUDA (pinned) BF16 B | CUDA FP8 B | saving | CPU BF16 B | CPU FP8 B | saving |
  |---|---|---|---|---|---|---|
  | 1 | 63,963,136 | 31,998,017 | 50.0% | 55,967,744 | 43,599,105 | 22.1% |
  | 4 | 255,852,544 | 127,991,873 | 50.0% | 223,870,976 | 127,597,569 | 43.0% |
  | 16 | 1,023,410,176 | 511,967,297 | 50.0% | 895,483,904 | 463,591,425 | 48.2% |
  | 64 | 4,093,640,704 | 2,047,868,993 | 50.0% | 3,581,935,616 | 1,807,566,849 | 49.5% |

  The earlier measured storage probe in #57 (k=64) gave 49.79%.

Full DeepSeek measurement on the real model is deferred to phase 2.
