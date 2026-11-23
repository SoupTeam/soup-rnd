# Routing Oracle on CPU: router calls, top-k work and the step win (ticket 07, part 1)

Date: 2026-11-23. Scope: CPU only, no GPU run and no bit ladder; those are part 2 after
review. The lever is `training.l2l_routing_oracle` over route-preserving MBR.

## Setup

- CPU: 12th Gen Intel Core i5-12500H, 16 threads exposed, `torch.set_num_threads(1)`.
- `torch` 2.14.1+cpu, `transformers` 5.19.0, Python 3.12.
- Tiny DeepSeek-V3 fixture from `benchmarks/harness/b13_moe_fixture.py`: 3 decoder layers,
  2 MoE layers, 8 routed experts, top-2, one shared expert, hidden 32. BF16 compute, dense
  storage, per-expert streaming, `two_read_experts=True`, `grouped_expert_routes=True`.
- One step = route-preserving MBR over k micro-batches of one row, S tokens, AdamW-free
  (losses and gradients only). Median of 5 measured steps after 1 warmup step, one process.
- Harness: `benchmarks/harness/mbr_oracle_cpu_profile.py`.

Reproduce (from the repository root, with the local CPU venv):

```bash
python benchmarks/harness/mbr_oracle_cpu_profile.py --k 8 --seq 16 --steps 7 --warmup 2 \
  --dtype bfloat16 --json /tmp/oracle-cpu.json
```

## Counts and step time

The baseline is the lever's own toggle: MBR without the Oracle. "Router calls" are native
`DeepseekV3TopkRouter.forward` executions. "Top-k dispatches" are router top-k decisions,
executed or replayed; two-read's `TopkDecisions` already replays the recompute's decisions
below `aten`, so the Oracle removes the dispatch entries, not the aten executions.

| k | S | router calls/step | top-k dispatches/step | `aten::topk` executions | step median, s | win |
|---|---|---|---|---|---|---|
| 4 | 32 | 16 -> 8 (50.0%) | 48 -> 24 (50.0%) | 24 -> 24 | 0.0644 -> 0.0500 | **+22.3%** |
| 8 | 16 | 32 -> 16 (50.0%) | 96 -> 48 (50.0%) | 48 -> 48 | 0.0904 -> 0.0822 | **+9.1%** |
| 8 | 16 repeat | 32 -> 16 (50.0%) | 96 -> 48 (50.0%) | 48 -> 48 | 0.0911 -> 0.0829 | **+9.0%** |
| 8 | 64 | 32 -> 16 (50.0%) | 96 -> 48 (50.0%) | 48 -> 48 | 0.1396 -> 0.1260 | **+9.7%** |

Each step with the Oracle records `native_forward = 2 layers x k`,
`oracle_recompute = 2 x k`, `oracle_vjp = 2 x k`, `oracle_peek = 0` (this MBR path has no
decoupled peek phase yet). The recompute therefore runs no router, no top-k selection and no
`TopkDecisions` dispatch; the forward router stays native.

The win is small in absolute time (8-13 ms per step on this fixture) and comes from skipping
the router linear, sigmoid/group selection and the replayed top-k bookkeeping in the backward
recompute. It does not include the pinned host copies, which cost on CUDA. This table is not a
GPU speed claim; the A-B-C-C-B-A ladder on the RTX 3050 is part 2.

## Neutrality on CPU

Direct VJP of one native router against the Oracle's saved original-forward Jacobian
(upstream gradient on the gate's routing weights; both arms have the router parameter frozen,
as the Oracle requires). Fixed seed 0, hidden `[5, 32]`, forward weights and routes are
bit-equal in every row.

| dtype | `norm_topk_prob` | `routed_scaling_factor` | `torch.equal` | max abs | relative L2 |
|---|---|---|---|---|---|
| FP32 | True | 1.0 | no | 2.98e-8 | 7.29e-8 |
| FP32 | True | 2.5 | no | 1.49e-8 | 2.35e-8 |
| FP32 | False | 1.0 | yes | 0 | 0 |
| FP32 | False | 2.5 | yes | 0 | 0 |
| BF16 | True | 2.5 | yes | 0 | 0 |

The FP32 normalize branch is the known dense-FP32 fact from the main ticket 06 (~1e-7
relative L2): the divide-and-sum backward runs in a different FP32 order. Without
normalization the two paths are bit-equal. BF16, Moonlight's compute dtype, is bit-equal
with normalization on.

Whole tiny-MoE MBR step, Oracle on against Oracle off, same weights and batches, `torch.equal`
over losses and all 30 LoRA gradients:

| dtype | k=1 | k=2 | k=4 |
|---|---|---|---|
| BF16 | equal, max abs 0 | equal, max abs 0 | equal, max abs 0 |
| dense FP32 | losses equal, 26/30 grads differ, max abs 1.19e-7, rel L2 1.34e-7 | losses equal, 30/30 differ, max abs 5.96e-8, rel L2 1.42e-7 | losses equal, 30/30 differ, max abs 2.24e-8, rel L2 1.38e-7 |

So the lever is neutral for the target NF4/BF16 arm on CPU, and is not neutral in dense FP32
at rounding level. Per D6 and the ticket, it stays an opt-in, off-by-default lever; a GPU
ladder decides whether it is neutral there.

## Tests

- `tests/test_p3_routing_oracle.py`: the direct VJP (logits, weights, ids, input gradient, four
  normalize/scaling combinations), the BF16 VJP, the whole tiny-MoE MBR pair at k = 1, 2, 4 in
  BF16 (bit-equal) and dense FP32 (rounding recorded), the trainer state switch, and the
  refusal without MBR or with a trainable router.
- `tests/test_p3_l2l_recipe.py`: the recipe switch, its MBR requirement and resume refusal.
- MBR and L2L regression files re-run unchanged: `tests/test_p72_mbr_cpu_bits.py`,
  `tests/test_p3_mbr_route_summary.py`, `tests/test_p3_mbr_window_boundaries.py`,
  `tests/test_b13_l2l_moe.py`, `tests/test_l2l_trainer_mode.py`.
