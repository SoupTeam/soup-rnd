# P3: one L2L step on tiny DeepSeek-V3, BF16 vs FP8 vs FP8 + Oracle

Harness: [`benchmarks/harness/p3_fp8_deepseek_step.py`](../../harness/p3_fp8_deepseek_step.py)
at commit `e7fc9db04c9bd5a43d73bdd6a88310ea7486a116`.

```bash
PYTHONPATH=src:. python benchmarks/harness/p3_fp8_deepseek_step.py --device cpu --output cpu --ks 1 4
```

- **Model:** synthetic native DeepSeek-V3. It has 3 layers (1 dense and 2 MoE), 8 experts, top-2 routing
  and hidden size 128. Weights are NF4 per-expert shards, randomly initialized with seed 23.
  LoRA covers the five attention projections (30 tensors). This is not DeepSeek-R1 671B.
- **Step:** one optimizer step with `k=1` and `k=4`, micro-batch `(1, 64)`. All three arms use the same
  weights, inputs, labels and initial LoRA, and all compute in BF16. Only the boundary
  storage differs.
- **Hardware:** CPU, Intel Xeon Platinum 8558 (16 vCPU). Python 3.12.3, torch 2.15.0, transformers 5.23.0,
  peft 0.21.3. There was no GPU, so no VRAM numbers.
- **Comparison:** E1 to E3 are measured against the BF16 arm, using the thresholds in
  `benchmarks/harness/g4_rule.py`.
- **Routing guard:** the production guard is used unchanged.

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
  (layer 2, micro-batch 0) at both `k` values. The guard stops the step, and no gradient or
  update is produced.
- **FP8 + Oracle.** It completes, and its loss equals BF16 exactly. That is expected: the
  forward reads BF16 boundaries before they are stored. E1, E2 and the E3 cosine pass. The E3
  relative L2 is 5.3–5.5%, just over the 5% bound. This is the known gradient
  divergence. It reflects FP8 rounding of the saved boundaries during recompute, and it does not
  change the forward or the loss.
- **Memory at this size.** FP8 has a fixed host codec scratch cost (about 0.26 MB), so on this tiny
  model FP8 storage is larger than BF16 at both `k` values. Process RSS peaks are dominated
  by the runtime and differ by less than 3%. The tiny model cannot show the saving.
- **Memory at DeepSeek-V3 dimensions.** The planner's `fp8_activation_allocation` gives these
  host bytes for 61 layers, hidden size 7168 and `(1, 64)` micro-batches:

  | k | BF16 store B | FP8 store B | saving |
  |---|---|---|---|
  | 1 | 55,967,744 | 43,599,105 | 22.1% |
  | 4 | 223,870,976 | 127,597,569 | 43.0% |
  | 16 | 895,483,904 | 463,591,425 | 48.2% |
  | 64 | 3,581,935,616 | 1,807,566,849 | 49.5% |

  These are allocation estimates from the same function the store and planner use, not measurements.
  At `k=64` the estimate matches the earlier measured storage probe in #57 (49.79%).

Full DeepSeek measurement on the real model is deferred to phase 2.
