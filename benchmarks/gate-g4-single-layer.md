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

## Results (synthetic, RTX 5090, 2026-10-22)

Machine: Vast.ai instance 57227559, RTX 5090 (UUID GPU-ef2be508-ab9a-3253-3f2c-617b895842ae),
driver 580.119.02, torch 2.14.1+cu130, TransformerEngine 2.20.2, bitsandbytes 0.50.2.
Raw records: `benchmarks/results/g4-single-layer/`. Every comparison ran 5 blocks (20 runs); the
repeats rule gave N = 2 each time (sigma 1.5 to 6 microseconds), so the pilot was enough.
No arm was void. `suspend` and `power` were ok in every arm. `foreign_reads` was unknown in every
arm (the container's disk has no block device); the rule does not require it. The GPU clock record
saw `sw_power_cap` in 8 of 398 samples (at most 25% of one run's samples) and a median SM clock of
at least 2520 MHz in every run, so no run was voided.

### Native FP4 against NF4 plus bf16 matmul (forward, 4096 tokens, mean of 10 runs per arm)

| shape (tokens x in x out) | FP4 step | NF4 step | NF4 / FP4 | FP4 error vs bf16 | NF4 error vs bf16 |
|---|---:|---:|---:|---|---|
| 4096 x 4096 x 4096 | 256.5 us | 730.1 us | 2.85 | rel 0.1462, cos 0.9893 | rel 0.0920, cos 0.9958 |
| 4096 x 4096 x 11008 | 461.9 us | 2012.0 us | 4.36 | rel 0.1462, cos 0.9893 | rel 0.0920, cos 0.9958 |

Error is the relative Frobenius norm and the cosine of the layer output against `F.linear` in bf16,
on weights drawn from N(0, 0.02) and unit-normal inputs. The FP4 path quantizes weights and
activations, the NF4 path quantizes only the weights, which is one reason its error is lower.
FP4 holds its weight packed (`NVFP4Tensor`). The NF4 step includes restoring the weight to bf16.
Forward only: no backward, no training.

### Refusal on shapes the FP4 GEMM cannot run

- The harness raises `UnsupportedShapeError` before it builds a layer and never calls the NF4 path.
  Text for tokens=250: "native FP4 GEMM needs every dimension divisible by its multiple; got
  tokens=250 (needs a multiple of 16). There is no fallback to NF4: it would mix two formats in
  one arm. Change the shape or pad it."
- TransformerEngine's own refusals, measured (`refusal-250x512x512.json`): tokens=250 or
  in_features=520 give `ValueError: FP8 execution requires the product of all dimensions except the
  last to be divisible by 8 and the last dimension to be divisible by 16`. tokens=248 and
  out_features=1000 give `RuntimeError ... First dim for NVFP4 must be divisible by 16`.
- **TransformerEngine's check of 16 is not enough for in_features.** At 4096 for the other two
  dimensions, in_features=4112 passes both TransformerEngine checks and then fails in cuBLAS with
  `Unable to find suitable cuBLAS GEMM algorithm`. 4096, 4128, 4160, 4192 and 4224 ran. tokens and
  out_features ran at 4112, 4128, 4160, 4192 and 4224. Only one value with in_features = 16 mod 32
  was tried. The harness therefore requires a multiple of 32 for in_features and 16 for the rest
  (`align_probe.json`; the first padding attempt, `status-first-attempt.log`, failed on this).

### Cost of padding to a multiple of 128

Arm A is FP4 at the shape, arm B is FP4 at the shape padded to 128 in every dimension. The weight is
padded once outside the timed step. The activation pad and the output slice are inside it.

| shape | extra FLOPs | extra weight elements | step unpadded | step padded | padded / unpadded |
|---|---:|---:|---:|---:|---:|
| 1040 x 1056 x 1040 | +33.9% | +20.8% | 174.9 us | 219.6 us | 1.256 |
| 2064 x 4128 x 4112 | +10.8% | +5.1% | 220.2 us | 243.3 us | 1.105 |
| 1040 x 4128 x 11024 | +14.5% | +3.4% | 245.1 us | 263.8 us | 1.076 |
| 4112 x 4128 x 4112 | +8.0% | +5.1% | 291.0 us | 302.2 us | 1.039 |

The padded output matches the unpadded one: the error against bf16 is identical to four digits in
every pair. The cost shrinks as the shape grows, from +25.6% at about 1000 to +3.9% at about 4100.
These are numbers for a later decision. They do not recommend padding.
