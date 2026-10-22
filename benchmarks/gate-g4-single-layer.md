# G4 ticket 02: one layer, native FP4 GEMM against NF4 plus a bf16 matmul

All numbers on this page are **synthetic**: random bf16 data, one linear layer, forward only. They
are not training and not a model result.

## Comparison

- Arm A, `fp4`: `te.Linear` built under `te.quantized_model_init` with the `NVFP4BlockScaling`
  recipe, forward under `te.autocast`. The weight is held packed.
- Arm B, `nf4`: bitsandbytes NF4 weight, restored to bf16 inside every timed step, then
  `F.linear`. The restore is part of the NF4 path's cost.
- Reference: the bf16 product `F.linear(x, W)` of the same data. Each arm reports the relative
  Frobenius error and the cosine of its output against it, measured once per run, before timing.
- Time: median of 200 timed steps after 50 warm-up steps, CUDA events, one synchronize per step.
- Order and repeats: `benchmarks/METHOD.md`. Blocks of A, B, B, A, each run a fresh process, a
  5-block pilot, then N from `repeats`, one top-up at most. Harness: `benchmarks/harness/g4_fp4_layer.py`.
- Padding cost (separate): arm A is `fp4` at a shape that is a multiple of 16 but not of 128,
  arm B is `fp4+pad`: the same layer padded to a multiple of 128 (weight zero-padded once, outside the
  timed step; the activation is padded and the output sliced inside every step). Same A-B-B-A
  procedure. Also reported analytically for every shape: extra FLOPs and extra weight elements.

## The decision rule, written before the run

| validity row | check | required | if it fails |
|---|---|---|---|
| `suspend` is ok on every arm the verdict reads | `suspend` | yes | no verdict |
| `power` is ok on every arm the verdict reads | `power` | yes | no verdict |
| `foreign_reads` is ok on every arm the verdict reads | `foreign_reads` | no | recorded only |

`foreign_reads` is not required because no arm reads a model from disk: the data is generated on
the GPU. On a rented container the disk under test may also have no block device, which makes the
check unknown whatever the arms do. A void `foreign_reads` still voids the arm. A void arm's block is
replaced and never used. The GPU clock record is read for every arm; a run whose median SM clock
shows a power or thermal cap in every sample is treated as void by hand and its block replaced.

A refusal row: a shape with a dimension that is not a multiple of 16 must raise
`UnsupportedShapeError` before any layer is built, and TransformerEngine's own error for the same
shape is recorded next to it. No NF4 fallback exists in the harness.

The verdict is not a pass or fail on speed. The page reports the numbers: error and time of both
paths, and the cost of padding to 128.
