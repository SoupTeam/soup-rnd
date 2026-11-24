# Ticket 14: C against A0, the operation behind the divergence

Moonlight-16B-A3B (`2a9a177f`), RTX 3050 Laptop (sm_86, 4 GiB, one card, exclusively used here),
torch 2.14.1+cu130, CUDA 13.0, venv `~/.venvs/soup-rnd-cuda`, LoRA targets
`q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj` (rank 16). Mode: bits, **strict** determinism
(`torch.use_deterministic_algorithms(True)`; the ticket 12 non-determinism raises instead of
drifting). Every run is a fresh process, on AC power (the driver refuses otherwise), inside
`systemd-inhibit --what=sleep`. Each row in `rows.md` names model, card, window, arm, mode, expert
GEMM and torch.

## Answer

**First divergent tensor:** layer 1 (the first MoE layer), routed expert 1, the **`down`
projection** (`2048x1408`, bf16), forward pass. A0 called that expert twice in the group, with
M = 15 and M = 16 rows; C called it once, with M = 31. All 31 rows differ; in the first row
803 of 2048 elements differ, max abs 1.22e-4 against a row max of 2.92e-2 (relative 4.2e-3).

**Operation and cause:** cuBLAS picks the GEMM kernel *and its K reduction* from the row count.
For this shape M <= 16 runs `ampere_bf16_s16816gemm_64x64_sliced1x2_stages_64x6` (one
reduction); M = 17..31 runs `cutlass_80_tensorop_bf16_s16816gemm_relu_64x64_64x6_align8` **plus
`cublasLt::splitKreduce_kernel`** (two-stage split-K sum). Same formula, different summation tree.
Classification: **algorithmic kernel selection by row count (split-K reduction), not a dtype
change, not non-determinism, not the backward order**. Strict mode keeps repeated identical calls
identical; it says nothing about two different row counts.

**Everything in front of it is bitwise equal:** the 16 module boundaries of layers 0 and 1 (both
layer norms, attention, router gate, shared expert; layers are matched by input digest, so C's
reverse group order does not matter) agree, the router decisions agree, and all 768 gate_up input
rows of layer 1 are the same rows. The gate_up output of that expert is also equal (same kernel
bucket at M = 15/16/31); the first difference is the `down` GEMM that consumes it. From layer 2 on
the inputs already carry the difference; those boundaries cannot be compared and are propagation,
not a second source. The routed-expert node of layer 1 differs in 124 of 128 output rows, after the
projection.

**Hypothesis of ticket 06 is confirmed in a narrower form.** The divergence is in the grouped
native expert forward, yes, but not in "grouping" as such: it is the cuBLAS/cuBLASLt row-count
decision inside it. D inherits it because D's forward is C's grouped native GEMM (ticket 06: loss
D bit-identical to C), and E escapes it because `lt-fused` forces the reference kernel per segment.

## Reproduction: C against A0, k=2, (8,64)

`run_repro.sh` (two fresh processes; `torch.equal` on the loss and all 216 LoRA gradients):

| pair | loss equal | grads differing | max abs | `grad_rel_l2` | min cos |
|---|---|---|---|---|---|
| A0 vs C, k=2, S=64 | no (1.928908 vs 1.918482) | 108 / 216 | 4.680e-2 | 1.286e-1 | 0.982401 |

This is the same number the ticket 03/06 rows recorded; `repro/bits.json` holds one A0/C pair
from two fresh processes.

## B1: the direct chain, A0 plain against C-seg

The ladder compared C-seg with A0+seg in every window. Only k=2, S=64 also had a direct
A0-plain control. `compare_cross_day.py` closes that gap on the CPU from stored tensors: A0 plain
is ticket 03's tensor (20261122, `~/moonlight-local/results/bits`), C-seg is the ticket 14 ladder
tensor, and the ladder's A0+seg against C-seg verdict is read from `ladder/*/bits.json`. The
report is `cross-day/bits.json` and `cross-day/bits.md`.

| k, S | A0 plain vs A0+seg | ladder A0+seg vs C-seg | A0 plain vs C-seg | covered |
|---|---|---|---|---|
| k=1, S=64 | equal, 0 / 216 | equal, 0 / 216 | equal, 0 / 216 | yes |
| k=2, S=64 | equal, 0 / 216 | equal, 0 / 216 | differs, 108 / 216 | no |
| k=4, S=64 | equal, 0 / 216 | equal, 0 / 216 | equal, 0 / 216 | yes |
| k=8, S=64 | equal, 0 / 216 | equal, 0 / 216 | equal, 0 / 216 | yes |
| k=1, S=512 | equal, 0 / 216 | equal, 0 / 216 | equal, 0 / 216 | yes |
| k=2, S=512 | equal, 0 / 216 | equal, 0 / 216 | equal, 0 / 216 | yes |
| k=4, S=512 | equal, 0 / 216 | equal, 0 / 216 | equal, 0 / 216 | yes |
| k=8, S=512 | equal, 0 / 216 | equal, 0 / 216 | equal, 0 / 216 | yes |

**Covered: 7 of 8 cells directly, plus the rung the ladder lacked.** Two facts changed against the
first report:

- A0 plain equals A0+seg in **all 8 cells**, not only k=2, S=64. The no-op control is no longer
  one cell: the 20261122 A0 tensors and the 20261124 A0+seg tensors are bit-equal everywhere.
- C-seg equals plain A0 directly in 7 cells. At **k=2, S=64 the direct comparison is not
  covered**: `tensors/C-k2-s64.pt` was overwritten by `run_repro.sh` at 07:21 with the plain-C run
  (loss 1.918482, 108 / 216 against A0), after the ladder wrote C-seg to the same filename at
  07:13. The ladder record `ladder/k2-s64/bits.json` still holds the A0+seg against C-seg verdict
  (loss equal, 0 / 216), and the A0-plain against A0+seg control holds in this cell, so the chain
  there is the recorded pair plus transitivity, not a fresh direct comparison. No GPU rerun was
  made for it.

**Dataset and configuration.** The bits rows of ticket 03 carry
`dataset_sha256 = 6beb56ae...` for S=64 and `91048d80...` for S=512; every row of a window shares
one value (`dataset_sha256` in `cross-day/bits.json`). Recomputing `mbpp_records(seq=...)` with the
same tokenizer reproduces exactly these two hashes, and the capture drives that same function; the
ticket 14 rows only failed to store the field, because `_write_row` selected a subset of the harness
row. The capture now writes `dataset_sha256` and the segment counters, so rows captured from here on
carry them; the historical rows are unchanged.
Both campaigns used Moonlight rev `2a9a177f...`, the same source and cache, LoRA targets
`q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj` at rank 16, three steps, torch 2.14.1+cu130, the
default SDPA backend at S=64 and the math backend at S=512. One deliberate difference: ticket 03
ran deterministic with `warn_only=True`, ticket 14 strict; strictness changes whether a
nondeterministic op raises, not the bits of a deterministic op. The capture now takes an explicit
`--model-revision` (the run scripts pass `2a9a177f...`); the historical rows carry the revision the
old fallback read from the cache record, and a source without a `_commit_hash` now records
`unknown` instead of guessing.

## Boundaries (`compare-k2-s64/`)

`capture_boundaries.py` hooks the layer's submodules (not the decoder or experts, which the MBR
path refuses) and wraps `StreamedExperts.forward` at class level plus every expert `F.linear`, for
step 0. `compare_boundaries.py` matches module boundaries by the *input* digest, expert rows by
the input row digest, and the verdict is the lowest layer with a difference (modules of a layer
before its expert calls).

| boundary | before the first difference |
|---|---|
| module boundaries, layers 0-1 | 16 / 16 equal; 0 mismatched |
| layer-1 gate_up input rows | 768 / 768 identical |
| layer-1 router gate, shared expert | equal |
| first differing tensor | layer 1, expert 1, `down`, forward, M 15+16 vs 31, 31/31 rows differ |
| routed-expert node, layer 1 | 124 / 128 output rows differ (downstream of the projection) |
| layers >= 2 boundaries | inputs already differ; 250/266 unmatched, propagation |
| forward calls with a differing row | 31 of 112 at layer 1 (same-kernel-bucket pairs stay equal) |

The full record is `compare-k2-s64/compare.json` (unmatched rows counted, first 12 kept) and
`compare-k2-s64/compare.md`.

## Op-level attribution (`probe/`)

The probe replays both expert projections with the row counts of the capture. Kernel buckets on
this card, from `probe/probe.md`:

| projection | M | kernel |
|---|---|---|
| gate_up 2816x2048 | 1 | `gemv2T_kernel_val` (GEMV) |
| | 2-16 | `ampere_bf16_s16816gemm_64x64_sliced1x2_stages_64x6` |
| | 17-31 | same `sliced1x2`, `stages_64x5` (same reduction tree) |
| | 32-39 | `ampere_bf16_s16816gemm_128x64_stages_64x3` |
| | >= 40 | `ampere_s16816gemm_256x128_stages_32x3` + `splitKreduce_kernel` |
| down 2048x1408 | 1 | `gemv2T_kernel_val` (GEMV) |
| | 2-16 | `ampere_bf16_s16816gemm_64x64_sliced1x2_stages_64x6` |
| | 17-31 | `cutlass_80_tensorop_bf16_s16816gemm_relu_64x64_64x6_align8` + `splitKreduce_kernel` |
| | 32-39 | `ampere_bf16_s16816gemm_128x64_stages_64x3` |
| | 40-64 | `cutlass_80_tensorop_bf16_s16816gemm_relu_128x64_64x3_align8` |

Measured pairs (first rows of a small call against the same rows inside a larger call):

| projection | M small vs large | equal | n_diff / numel | max abs diff |
|---|---|---|---|---|
| gate_up | 15 vs 31 | yes | 0 / 42240 | 0 |
| gate_up | 1 vs 2 | no | 2 / 2816 | 2.44e-4 |
| down | 15 vs 31 | no | 11928 / 30720 | 7.81e-3 |
| down | 1 vs 2 | no | 2 / 2048 | 6.10e-5 |

The kernel name of the down bucket contains `relu`, but the epilogue does not clamp (differences
are rounding-sized, not sign-sized); the extra `splitKreduce_kernel` is the math difference.

## Exact alternatives

| alternative | exists | fact |
|---|---|---|
| `segmented` (new shipped mode) | **yes** | native cuBLAS called once per micro-batch segment, so every call keeps the reference row count. No extension. Ladder below: A0+seg against C-seg bit-equal at k = 1,2,4,8 in both windows, and C-seg against plain A0 directly in 7 of 8 cells (B1). Cost: C k=2, S=64 17.16 s against 17.00 s native (one deterministic step, same rows). The harness row now records `linear_rows`, `mul_rows` and `passthrough_calls`; rows captured before this change have no counters (the k=8 row names only its step time). |
| `lt-fused` (arm E) | yes | forces the reference kernel per segment in one C++ call (`a0_exact_lt`); already the exact lever, measured in tickets 02/03. Needs the CUDA extension. |
| `padded` (harness `padded_expert_gemm`, plan D1) | yes, other meaning | pads every call to one row count; with pad 32 and 512 a 1-row and a 2-row call give the same bits (`probe/`), so bits stop depending on M. It changes the reference too: A0 must be padded with the same pad, so it is a "both arms" invariant, not a way to keep plain-A0 bits. |
| row-wise (`rowwise_expert_linear`, `invariant`) | yes, other meaning | one call per row (M=1 always); needs both arms at M=1. Does not reproduce plain A0, whose calls have M up to 63 here. |
| `allow_bf16_reduced_precision_reduction = False` | no | M 15 vs 31 on down still differs (15 / 30720, max 1.95e-3); M 1 vs 2 still differs; the kernels do not change. |
| strict determinism / `CUBLAS_WORKSPACE_CONFIG=:4096:8` | no | already on; repeated calls with the same M are bit-identical (`tests`, ticket 12). The difference is between different M, not between repeats. |
| cuBLASLt forced algorithm on the joined call | not equal | `lt-fused` forces the algorithm *per segment*; forcing one algorithm on the joined rows would still reduce K in that algorithm's order, not in the reference calls' order, and no table for every M exists. Not implemented. |

The op has no torch flag that makes one row count compute like another; the only exact fixes are
shaping the calls like the reference (segmented / lt-fused) or shaping both arms alike (padded /
row-wise).

## Fix and ladder

The fix is the shipped mode `soup_cli.utils.expert_gemm.expert_gemm_context("segmented")`
(`EXPERT_GEMM_CHOICES` now `native`, `segmented`, `lt-fused`). The mode and its refusals are pinned
by `tests/test_expert_gemm_select.py`; in the commit history the feature is `862042f8` and the tests
are `6e0b4481`, so this is a test-pinned change, not a red-first sequence visible in history. It
reuses `segmented_expert_gemm`, which splits both the forward GEMM and the recompute's backward at
the micro-batch boundaries.

`run_ladder.sh`: A0 and C both with `--expert-gemm segmented`, k = 1,2,4,8, windows S=64 and
S=512 math, strict mode, three steps, fresh process per cell. For A0 the plan is empty (one
segment per call), so it is a passthrough — the cross-day control proves that in every cell. The
verdict below is the ladder's A0+seg against C-seg pair.

| window | loss bitwise equal | grads differing |
|---|---|---|
| k=1, S=64 | yes | 0 / 216 |
| k=2, S=64 | yes | 0 / 216 |
| k=4, S=64 | yes | 0 / 216 |
| k=8, S=64 | yes | 0 / 216 |
| k=1, S=512 math | yes | 0 / 216 |
| k=2, S=512 math | yes | 0 / 216 |
| k=4, S=512 math | yes | 0 / 216 |
| k=8, S=512 math | yes | 0 / 216 |

Controls (`ladder/`):

- A0 plain against A0 with `segmented`: loss and all 216 gradients equal in **all 8 cells** from the
  stored tensors of both campaigns (`cross-day/`); the k=2, S=64 control is
  `ladder/noop-a0-k2-s64/`. The mode is a no-op on a plain L2L step.
- C native against C with `segmented` at k=2: 108 / 216 differ, max abs 4.68e-2, `grad_rel_l2`
  1.352e-1, min cos 0.982401 — the same divergence as against A0, so the mode is what changes the
  bits (`ladder/plain-vs-seg-c-k2-s64/`).
- Boundary capture of the fixed pair (`compare-k2-s64-segmented/`): 266 / 266 module boundaries
  equal, 79872 / 79872 expert rows matched and equal, 6656 / 6656 node rows equal, verdict
  `equal` — no first divergent tensor remains at k=2.

The default is **not** switched to `segmented`: arm C is the base of the measured tables in
tickets 02, 03, 06 and 13, and changing its definition silently would invalidate them. The exact
lever stays E (`lt-fused`); `segmented` is the extension-free exact path. It held in 7 of 8 cells
by a fresh direct comparison and in the eighth by the pair recorded when the ladder ran; the
one-cell gap is named in B1 rather than papered over.

## What this fixes in the other tickets

- **02 (need for E):** E is not a speed or toolchain curiosity. The default grouped native path
  changes bits because cuBLASLt picks a split-K kernel by row count; E is the fast exact path and
  `segmented` is the dependency-free one.
- **06 (kill D):** D's forward is C's grouped native GEMM, so D == C in loss bits and diverges
  from A0 above k=1 for this reason. The decoupled backward is not the cause; the verdict stands.
- **08 (Triton A1):** a "batch-invariant" Triton row must reproduce the reference's per-call row
  counts, not a mathematical invariant of the GEMM: the reference bits are cuBLAS's choice for the
  reference M. Compare at the same segment shapes, or pad/row-wise both arms.

## Files

- `cdiag.py`, `capture_boundaries.py`, `compare_boundaries.py` — boundary capture and comparison.
- `probe_gemm_rows.py` — the op-level kernel and row-count matrix (`probe/`).
- `compare_bits.py`, `summarize_rows.py` — the `torch.equal` verdict and `rows.md`.
- `compare_cross_day.py`, `cross-day/` — B1: A0 plain (20261122) against ladder C-seg (20261124).
- `run_repro.sh`, `run_ladder.sh` — the two campaigns (AC check, `systemd-inhibit`, lock file).
- `rows.md` — every result row with model, card, window, arm, mode, expert GEMM, torch.
- Raw tensors, captures and logs are git-ignored.

## Commit types

`CONTRIBUTING.md` lists `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`. The
earlier commits of this campaign use `bench`, which is not in that list. History is not rewritten;
every new commit of the campaign uses `chore(bench)` for campaign data and the type that matches
the change otherwise (`fix`, `test`, `docs`).
