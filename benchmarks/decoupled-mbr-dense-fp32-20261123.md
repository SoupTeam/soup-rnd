# Decoupled MBR on dense FP32: the recorded gradient gap (ticket 06, part 1)

Date: 2026-11-23. CPU only; no GPU run. This note pins the number the tests call
`KNOWN_DENSE_FP32_GAP`, the model and the window it was measured on, and the threshold
that holds it.

## Setup

- CPU: 12th Gen Intel Core i5-12500H, 16 threads exposed, `torch.set_num_threads(1)`.
- `torch` 2.14.1+cpu, `transformers` 5.19.0, Python 3.12.
- Model: the tiny DeepSeek-V3 fixture from `benchmarks/harness/b13_moe_fixture.py` — 3
  decoder layers, 2 MoE layers, 8 routed experts, top-2, one shared expert, hidden 32,
  dense storage (`storage="dense"`), `dtype="float32"`, per-expert streaming,
  `two_read_experts=True`, `grouped_expert_routes=True`.
- Window: k micro-batches of `rows=2`, `seq=6` each; the A0 arm is plain L2L, the candidate
  is `mbr_decoupled_subwindow=True` over the whole window. Seed 42 for every row.
- Measured quantity: the largest absolute difference over the per-micro-batch losses and
  every LoRA gradient, after `torch.equal` reports the row not bit-equal.
- Reproduce (repository root, CPU venv):

  ```bash
  python -m pytest tests/test_decoupled_mbr_cpu.py --no-cov -s \
    -k "dense_fp32_records_the_known_gap or dropout_rng_replay"
  ```

## Measured rows

`DECOUPLED_MBR` lines from `record()` in `tests/test_decoupled_mbr_cpu.py`:

| row | tensors compared | tensors not `torch.equal` | max abs |
|---|---|---|---|
| dense FP32 k=1 | 45 | 0 | 0 |
| dense FP32 k=2 | 46 | 44 | **2.682e-07** |
| dense FP32 k=4 | 48 | 44 | 1.341e-07 |
| dense FP32 dropout k=4 | 48 | 44 | 8.941e-08 |

The largest measured gap is **2.682e-07** at k=2. The tests hold it with
`KNOWN_DENSE_FP32_GAP = 1e-6`, about 3.7x the recorded value. The earlier "~9e-8" in the
module docstring was the dropout row alone, and "~1e-7" was a rounded restatement of the
whole set; this note is the source of truth.

## Why there is a gap

The losses are exact. The gap comes from the R2 multi-root backward adding the routed and
the non-routed contribution at the residual in another order than one engine call. BF16
rounds it away (the NF4 + BF16 rows in the same file are bit-equal), FP32 keeps it at
rounding level. It is not a correctness boundary and does not gate the BF16 target.
