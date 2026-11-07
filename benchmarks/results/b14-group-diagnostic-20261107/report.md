# B1.4 diagnostic: grouped expert compute (D3) against sequential, by routed-row bucket

Answer to the D3 owner's proposal in the PR #28 thread
(<https://github.com/SoupTeam/soup-rnd/pull/28#issuecomment-6709450031>): compare native and
grouped dispatch at 1 to 128 routed rows per expert, attribute kernels by phase, and measure
interleaved wall time without the profiler. This is a layer-level kernel-count and device-time
diagnostic. It is not a speed claim (contract #52) and it is not a result for the full B1.4 path.

## Result

- On an RTX 3070 8 GB, D3 grouped GEMM removes 12 of 385 kernels (3.1 percent) of one layer's
  forward plus backward, and only at 32 rows per expert and above. At 16 rows and below D3 runs
  the native path by its own policy, so the counts are identical.
- Backward is 262 of the 385 kernels (68 percent) and D3 leaves it unchanged by design: each
  expert is recomputed one at a time natively.
- Grouped GEMM adds 16 device copies (pointer tables, 47 to 63) and is 0.1 to 0.6 ms (0.2 to 1.3
  percent) slower in wall time than sequential, inside the repeat-to-repeat spread. Forward
  device time is 3 percent lower at 32 and 64 rows and 1 percent higher at 128 rows, but those
  rows are not bitwise equal to the resident layer (see below), so by the spec's own order
  (validate, then accept timing) they are not accepted yet. The 2x small-matrix cost of P3 was
  not tested: D3 never groups below 32 rows here.
- At 32 and 64 rows the grouped GEMM output is not bitwise equal to the resident layer (1 bf16
  unit in the last place on the output, 0.0625 on the route-weight gradient).
- A second run added 256 and 512 rows (below). Kernel counts do not change (373 against 385).
  Grouped GEMM forward device time is 28 percent lower at 256 rows and 1 percent higher at 512,
  both bitwise equal to the resident layer, and wall time stays within about 1 ms of
  sequential. 256 rows is the mean for length 2048.
- Not measured: the real checkpoint weights, real routing, the full model and the L2L trainer
  with grouping. D3 refuses this geometry as shipped; the harness widens its shape list locally.

## What was measured

One MoE expert layer with the geometry of `ByteDance-Seed/academic-ds-9B` (hidden 2048, expert
width 1408, bfloat16; the checkpoint has 64 routed experts and `num_experts_per_tok` 8, read from
its `config.json` at revision `2a205727f616995ad4aba57806f8e2d0d2f28743`). The layer has 8
experts with random weights (`hidden ** -0.5` and `intermediate ** -0.5` scale), written to one
expert shard file and also kept resident as the reference. A bucket is the number of routed rows
every visited expert receives: 1, 2, 4, 8, 16, 32, 64, 128. Routing sends exactly that many
tokens to each of the 8 experts (top-k 1, route weight 0.5, `route_for_bucket`).

Three arms run the same inputs, one forward and one backward:

| Arm | What it is |
|---|---|
| `sequential` | B1.3 `StreamedExperts` with one `ExpertBufferPool` lease per expert. The path B1.4 uses today. |
| `grouped_restore` | D3 `GroupedAutogradExperts`, group width 4, no grouped GEMM. Groups the weight restore only. |
| `grouped_gemm` | The same with D3 `GroupedGemm` (cuBLAS grouped batched GEMM, native path up to 16 rows). |

Phases are `torch.profiler.record_function` ranges. A kernel belongs to the innermost range that
was open when it was launched (`attribute_kernels`; the backward runs on the autograd thread, so
a range of another thread may claim a launch when none of its own thread does):

| Phase | Covers |
|---|---|
| `fwd` | forward remainder: gather, route multiply, scatter. For `sequential` it also holds the GEMMs and the activation, which cannot be split without editing B1.3 |
| `gemm`, `act` | the grouped arms only: the two GEMM calls and the activation of a group |
| `restore_fwd`, `restore_bwd` | weight restore (the lease entry). Copies only, no kernels |
| `bwd` | the whole backward: per-expert recompute and input gradient. Not split further, both are inline in D3 and B1.3 |
| `unattributed` | 4 kernels outside any range (building the probe tensor between forward and backward) |

Kernels, copies and device time come from one profiled run per arm and bucket (after one
warm-up). Wall time is the median of 20 repeats, arms run in rotating order, no profiler, with a
device synchronisation after the forward and after the backward. The table sums `fwd`, `gemm` and
`act` into "forward".

Correctness: output, input gradient and route-weight gradient of every arm are compared with the
resident layer (`bitwise_equal_to_resident`, `max_abs_diff_vs_resident` in the JSON).

## Results, RTX 3070 8 GB

Host: vast.ai offer 56413546, GeForce RTX 3070 (8196653056 bytes, sm_86), driver 610.43.02,
Ryzen 5 5600X, image `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` with `torch==2.11.0`
(`2.11.0+cu130`, cuBLAS 13.1.0.3) installed over it and `torchvision`, `torchaudio` removed
(`vast-rtx3070/pip-versions.txt`). BLAS policy for every arm: cuBLASLt, bf16 reduced-precision
reduction off, split-K off. Files: `vast-rtx3070/result.json`, `run.log`.

| Rows per expert | Arm | Kernels | Forward kernels | Backward kernels | Copies | Forward device us | Backward device us | Wall ms (median) | Bitwise equal to resident |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sequential | 385 | 119 | 262 | 54 | 565 | 1118 | 49.85 | yes |
| 1 | grouped_restore | 385 | 119 | 262 | 47 | 546 | 1114 | 48.98 | yes |
| 1 | grouped_gemm | 385 | 119 | 262 | 47 | 543 | 1113 | 50.13 | yes |
| 2 | sequential | 385 | 119 | 262 | 54 | 568 | 1121 | 50.00 | yes |
| 2 | grouped_restore | 385 | 119 | 262 | 47 | 561 | 1124 | 49.32 | yes |
| 2 | grouped_gemm | 385 | 119 | 262 | 47 | 563 | 1122 | 50.20 | yes |
| 4 | sequential | 385 | 119 | 262 | 54 | 592 | 1132 | 50.08 | yes |
| 4 | grouped_restore | 385 | 119 | 262 | 47 | 591 | 1136 | 49.27 | yes |
| 4 | grouped_gemm | 385 | 119 | 262 | 47 | 593 | 1139 | 50.29 | yes |
| 8 | sequential | 385 | 119 | 262 | 54 | 634 | 1164 | 50.04 | yes |
| 8 | grouped_restore | 385 | 119 | 262 | 47 | 630 | 1158 | 49.35 | yes |
| 8 | grouped_gemm | 385 | 119 | 262 | 47 | 631 | 1159 | 50.15 | yes |
| 16 | sequential | 385 | 119 | 262 | 54 | 722 | 1243 | 50.19 | yes |
| 16 | grouped_restore | 385 | 119 | 262 | 47 | 716 | 1233 | 49.15 | yes |
| 16 | grouped_gemm | 385 | 119 | 262 | 47 | 717 | 1235 | 50.31 | yes |
| 32 | sequential | 385 | 119 | 262 | 54 | 599 | 1232 | 49.99 | yes |
| 32 | grouped_restore | 385 | 119 | 262 | 47 | 588 | 1210 | 49.35 | yes |
| 32 | grouped_gemm | 373 | 107 | 262 | 63 | 580 | 1215 | 50.45 | NO ['output', 'route_weight_grad'] |
| 64 | sequential | 385 | 119 | 262 | 54 | 637 | 1329 | 50.05 | yes |
| 64 | grouped_restore | 385 | 119 | 262 | 47 | 633 | 1323 | 49.45 | yes |
| 64 | grouped_gemm | 373 | 107 | 262 | 63 | 617 | 1320 | 50.68 | NO ['output', 'route_weight_grad'] |
| 128 | sequential | 385 | 119 | 262 | 54 | 797 | 1787 | 50.56 | yes |
| 128 | grouped_restore | 385 | 119 | 262 | 47 | 812 | 1785 | 49.40 | yes |
| 128 | grouped_gemm | 373 | 107 | 262 | 63 | 804 | 1787 | 51.00 | yes |

The Tesla T4 run (Kaggle, 16 GB, torch 2.11.0+cu130) has the same counts as the 3070 for
`sequential` and `grouped_restore` in every bucket (385 kernels, backward 262; 54 and 47 copies)
and wall time medians of 118 to 145 ms. The `grouped_gemm` arm is refused there:
`ValueError: native BF16 sm_80+ required`. Table: `python make_tables.py kaggle-t4/result.json`.
The T4 has a different capacity and does not speak for the 8 GB card.

## Second run: 128, 256 and 512 rows

The first run stopped at 128 rows, while the mean for length 2048 is about 256. A second rental
of the same vast.ai offer (the same machine) ran buckets 128, 256 and 512 with the commit that sizes the activation
budget from the largest bucket (`7aa1c4ac`); everything else is unchanged. Files:
`vast-rtx3070-256-512/result.json`, `run.log`, `remote-group2.sh`. Bucket 128 is a repeat of the
first run.

| Rows per expert | Arm | Kernels | Forward kernels | Backward kernels | Copies | Forward device us | Backward device us | Wall ms (median) | Bitwise equal to resident |
|---|---|---|---|---|---|---|---|---|---|
| 128 | sequential | 385 | 119 | 262 | 54 | 914 | 1968 | 50.42 | yes |
| 128 | grouped_restore | 385 | 119 | 262 | 47 | 893 | 1974 | 49.58 | yes |
| 128 | grouped_gemm | 373 | 107 | 262 | 63 | 860 | 1975 | 50.68 | yes |
| 256 | sequential | 385 | 119 | 262 | 54 | 2005 | 3839 | 52.13 | yes |
| 256 | grouped_restore | 385 | 119 | 262 | 47 | 2017 | 3853 | 51.26 | yes |
| 256 | grouped_gemm | 373 | 107 | 262 | 63 | 1448 | 3852 | 52.07 | yes |
| 512 | sequential | 385 | 119 | 262 | 54 | 2731 | 6437 | 54.40 | yes |
| 512 | grouped_restore | 385 | 119 | 262 | 47 | 2745 | 6401 | 53.91 | yes |
| 512 | grouped_gemm | 373 | 107 | 262 | 63 | 2763 | 6404 | 55.13 | yes |

- **Counts do not change.** 385 kernels (119 forward, 262 backward) for `sequential` and
  `grouped_restore`, 373 (107 and 262) for `grouped_gemm`, copies 54, 47 and 63, as at 128 rows.
  Every arm and bucket is bitwise equal to the resident layer, including `grouped_gemm` at 128,
  256 and 512 rows.
- **Forward device time of `grouped_gemm` against `sequential`:** 860 against 914 us at 128 rows
  (-5.9 percent), 1448 against 2005 at 256 (-27.8 percent), 2763 against 2731 at 512 (+1.2
  percent). The 256-row gain sits in the GEMM phase: 1001 us grouped against 1562 us for the
  native calls of `grouped_restore`, while at 128 and 512 rows the two are within 6 percent
  (548 against 582, 1973 against 1939). Native device time grows 2.7 times from 128 to 256 rows
  and 1.2 times from 256 to 512, so the native path looks slow at 256 rows rather than the
  grouped path fast. This is one observation on one card and one cuBLAS build; the cause (kernel
  choice for that shape) was not examined.
- **Run-to-run variation is large.** `sequential` at 128 rows has forward device time 797 us in
  the first run and 914 us in the second, backward 1787 and 1968 us. The 3 percent differences
  of the first run (32 and 64 rows) are inside that variation.
- **Wall time stays within about 1 ms.** `grouped_gemm` minus `sequential`: +0.26 ms at 128,
  -0.06 at 256, +0.73 at 512. `grouped_gemm` minus `grouped_restore`: +1.10, +0.81, +1.22 ms, the
  same constant offset as in the first run, which supports the per-call synchronisation as the
  cause. `grouped_restore` minus `sequential`: -0.85, -0.87, -0.49 ms. Even where the grouped
  GEMM saves 557 us of forward device time (256 rows), that is about 1 percent of the 52 ms.
- **The device share of the wall time grows with rows.** Forward plus backward device time of
  `sequential` is 2.9 ms at 128 rows, 5.8 ms at 256 and 9.2 ms at 512, against 50.4, 52.1 and
  54.4 ms of wall time (6, 11 and 17 percent). Restore still dominates at these sizes.

## What it shows

1. **Kernel count does not depend on rows.** `sequential` launches 385 kernels in every bucket,
   about 48 per visited expert for forward plus backward of one layer. The row count changes
   device time (forward 565 to 797 us, backward 1118 to 1787 us), not the number of launches.
2. **The backward is two thirds of the launches and D3 does not touch it on this path.** 262
   of 385 kernels are in `bwd` in every arm and bucket, and the backward device times of the
   three arms agree within 1.8 percent. On the non-packed pool measured here,
   `ReloadExpert.backward` in `expert_group_autograd.py` restores and recomputes one expert at a
   time with native operations. D3 also has a packed-only pool whose backward calls
   `packed_expert_backward` and `gemm.input_gradient`; that path was not measured. For the path
   measured, any saving from grouping can only come from the forward (119 kernels, 31 percent).
3. **Grouping the restore changes no kernels.** `grouped_restore` has 385 kernels like
   `sequential` and 7 fewer device copies (54 to 47, all in the forward: 20 to 13). Its wall
   time is 0.6 to 1.2 ms below `sequential` in all eight buckets (1.2 to 2.3 percent). The
   min-to-max spread of `sequential` over its 20 repeats is 1.2 to 12.4 ms (bucket 1: 49.5 to
   61.9), so the direction is consistent and the size is near the noise.
4. **Grouped GEMM removes kernels only from 32 rows.** Up to 16 rows D3 calls `torch.mm` per
   matrix (`native_olmoe_rows_max`, only 16 or 128 are accepted), so kernels (385) and copies
   (47) equal `grouped_restore`. From 32 rows the 16 GEMM kernels of the forward become 4, the
   total goes 385 to 373 (3.1 percent) and the forward 119 to 107 (10.1 percent). The cost is
   16 more copies (47 to 63, the pointer tables of `cublasGemmGroupedBatchedEx`).
5. **No clear gain, and the small-matrix risk is untested.** Forward device time of
   `grouped_gemm` against `sequential`: 580 against 599 us at 32 rows (-3.2 percent), 617
   against 637 at 64 (-3.1 percent), 804 against 797 at 128 (+0.9 percent). Grouped GEMM never
   ran below 32 rows, because D3 goes native at 16 rows and below, so P3's 2x cost for small
   matrix products was not exercised. Wall time of `grouped_gemm` is 0.1 to 0.3 ms above
   `sequential` at 1 to 16 rows and 0.45 to 0.63 ms above it at 32 to 128 rows (0.2 to 1.3
   percent). It is 0.8 to 1.6 ms above `grouped_restore` in every bucket, including 1 to 16
   rows where kernels and copies are identical. So the extra cost is not tied to grouping a
   matrix product: `GroupedGemm` synchronises the device in `launch` after every call, native
   calls included, which is a likely cause. Not isolated, and an arm-order effect is not
   excluded: the arms run in rotating order but only medians are kept, no paired differences.
6. **The device does almost nothing in this layer.** Forward device time is 0.54 to 0.81 ms and
   backward 1.1 to 1.8 ms for all 8 experts, against about 50 ms of wall time (forward about 24
   ms, backward about 26 ms). The rest is weight restore (file read, staging, copies) and
   launch overhead on the host. At this layer size, 12 fewer kernels are below 1 percent of the
   wall time. The shard file (about 138 MB) stays in the page cache here, so a real tier reads
   more slowly.
7. **Not bitwise at 32 and 64 rows.** `grouped_gemm` differs from the resident layer at 32 and
   64 rows: output by 0.0039 (one bf16 unit at values in [0.5, 1)) and route-weight gradient by
   0.0625. The input gradient is equal, and every other arm and bucket is bitwise equal,
   including `grouped_gemm` at 128 rows. D3 asserts bitwise equality in its tests for OLMoE
   shapes; this geometry was not validated by D3 and no tolerance is set here. The harness
   records the comparison but does not gate the timing on it, so the table shows timing for
   these two rows anyway; treat them as not accepted until a tolerance is agreed.

## What it means for the path of ticket 09

`academic-ds-9B` routes 8 of 64 experts per token, so the mean number of routed rows per visited
expert is about `tokens / 8` if routing were balanced: about 2 at length 16, 64 at 512 and 256 at
2048 for one row of tokens. This is arithmetic from the config, not observed routing: ticket 09
recorded visited experts, not rows per expert.

- Length 16 falls in D3's native zone: no kernels removed.
- Length 512 falls near the 64-row bucket: 12 fewer kernels per 8 experts per grouped forward,
  1.5 per expert.
- Length 2048 falls at the 256-row bucket of the second run: the same 12 fewer kernels, forward
  device time 28 percent lower for grouped GEMM on this card, wall time within about 1 ms of
  sequential. Uneven rows per expert in the real model were not measured.

The step of ticket 09 runs the expert forward twice per layer (forward and recompute) and the
backward once, and its figure of about 82 kernels per visited expert is a two-point estimate on
the full model with attention and LoRA. The D3 owner asked not to reuse that slope as a grouped
prediction, and this report does not: it gives the layer figures above only (12 fewer kernels
per 8 experts per grouped forward at 32 rows and above, 1.5 per expert, phase-normalized).
Whether the recompute forward of the L2L step goes through the grouped path was not checked, so
no saving for the full path is stated.

## What was not done

- Real checkpoint weights and real routing. Weights are random, routing is uniform with top-k 1.
  No histogram of routed rows per expert from the real model exists yet.
- The full model and the L2L trainer with grouping. D3 (PR #28) conflicts with the B1.4 branch
  in `l2l_trainer.py` and `l2l_step.py`, so this diagnostic is built on the D3 head `a2eed33c`
  and does not touch the B1.4 branch.
- Buckets between the powers of two (17 to 31 rows is where the native cutoff ends), buckets
  above 512, top-k above 1, uneven rows per expert, other codecs (NF4), more than 8 experts.
- The split of `bwd` into recompute and input gradient, and of `fwd` into gather, scatter and
  route multiply. Both are inline in D3 and B1.3 and splitting them means editing that source.
- Grouped backward and the packed-only pool. The measured pool recomputes natively. Artix6034's
  FP16 T4 prototype figure (16 to 24 percent slower backward at k=8) is a different code path
  and is not tested here.
- Expert visits and kernel events per kernel. The JSON keeps counts and device time per phase,
  `expert_fetches` per arm (16 in every bucket: each expert is read once forward and once
  backward), not kernel names or individual events.
- A test that rows up to 16 take the native path. The policy is D3's (`native_olmoe_rows_max`);
  here it is visible only in the GPU data (identical kernel and copy counts at 1 to 16 rows).
  Rows 17 to 31 were not measured.
- Attention-LoRA gradients and optimizer updates: the layer has no attention.
- Any speed statement. Wall time is a diagnostic of this layer on this host.

## Deviations from D3 as shipped

- `expert_group_gemm.SUPPORTED_SHAPES` accepts only OLMoE and tiny-control shapes. The harness
  adds `(2048, 2816)` and `(1408, 2048)` at run time (`geometry.supported_shapes_added` in the
  JSON). The geometry is not validated by D3; the only check is the comparison with the
  resident layer above.
- D3's `require_reference_policy` reads `allow_bf16_reduced_precision_reduction_split_k`. That
  flag does not exist on torch 2.7.1 (the image of ticket 09), so the grouped GEMM cannot run
  there. On torch 2.11 it is set by assigning `(False, False)`, as D3's own harness does.

## How it was run

Harness `benchmarks/harness/b14_group_diag.py` (`route_for_bucket`, `attribute_kernels`,
`attribute_memcpy`, `attribute_kernel_time_us`, `build_layer`, `run_bucket`), tests
`tests/test_b14_group_diag.py` (30, CPU). Source: branch `b14-d3-grouped-diagnostic`, D3 head
`a2eed33c` plus the harness. Defaults: 8 experts, hidden 2048, width 1408, bfloat16, 20 repeats. The source pin is the
commit names in this report: `environment.source_sha` is null in both JSON files because the
hosts held a source tarball, not a git checkout.

```bash
# vast.ai, RTX 3070: source tarball of src/ and the harness uploaded to /root
pip install -q "transformers==5.20.2" safetensors psutil rich pydantic pyyaml typer
pip install -q torch==2.11.0 && pip uninstall -y -q torchvision torchaudio
PYTHONPATH=/root/soup/src B14G_OUT=/root/out B14G_WORK=/root B14G_TAG=vast-rtx3070 \
  python3 /root/soup/benchmarks/harness/b14_group_diag.py
# Kaggle T4: kaggle-t4/run.py with dataset aldiyarsagat/soup-b14-grouped-source (private)
# tables: python make_tables.py vast-rtx3070/result.json
```

The vast run used `b14_group_diag.py` from commit `c103d6d1`. The Kaggle run used commit
`0545ae41`; the two later commits only change the policy code for torch 2.7.

The second run (128, 256, 512 rows) used `b14_group_diag.py` from commit `7aa1c4ac` and
`remote-group2.sh`, the same install steps in one script.

Attempts not used: Kaggle run 1 failed setting the split-K flag; run 2 was cancelled by Kaggle
with no output; run 3 set the policy as `False` instead of `(False, False)` and every
`grouped_gemm` arm was refused by the policy check. Vast runs 1 to 3 failed on torch 2.7.1 (the
tuple form is rejected, then the split-K flag is missing) and on `torchvision` after the torch
upgrade (`operator torchvision::nms does not exist`). Their results are not kept.

Rental of the first run: instance 59596767, offer 56413546, 0.130 USD/h, `vastai show instance` reported 335 s of
running time; with loading, the four launches and the upload the instance lived a few minutes
longer, under 0.1 USD against the ceiling of 1 USD. Destroyed after the files were copied.
Rental of the second run: instance 59599544, same offer, `vastai show instance` reported 210 s
of running time, again under 0.1 USD; destroyed after the files were copied. Three RTX 4090
instances of other users were running on the shared team account and were not touched. The
Kaggle runs are free. Left on the Kaggle account (private): dataset
`soup-b14-grouped-source` and kernel `soup-b14-group-diag`.

Files and hashes: `SHA256SUMS`.
