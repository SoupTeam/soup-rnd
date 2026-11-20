# P3 MBR group-size sweep — 2026-11-17

## Result

The fixed-window sweep completed with all four diagnostic gates passing. Increasing the
route-preserving MBR group size reduces expert reads and step time on this Moonlight/Soup
bridge. At the same time, grouping changes later-layer hidden states/routes and the resulting
raw LoRA gradient relative to current L2L. This is a promising systems result with a measurable
numerical trade-off; it is not a training-quality or P3 acceptance result.

## Protocol

- Model: `moonshotai/Moonlight-16B-A3B`, revision
  `2a9a177f4841649f849ba941e41d2ee73c042314`.
- Soup source revision: `5f02e9c57d4eae26fb697a7e5f63686d2d1b2c96`, with the experimental
  route-preserving MBR and corrected per-phase read accounting.
- Host: one RTX 3060 Ti 8 GiB, BF16, NF4 per-expert cache, 27 decoder layers / 26 MoE layers.
- Every run used the same fixed 8 × 64 token input window (`input_sha256` in each result), the
  same initial LoRA state, current L2L with two-read experts and grouped routes, and one AdamW
  update per arm. MBR group size was 1, 2, 4, or 8 microbatches per routed-FFN dispatch. Attention
  and routing remained microbatch-local.
- Each group-size result used an out-of-timing diagnostic pass followed by timed A-B-B-A
  (`L2L, MBR, MBR, L2L`); timing reports the median of two samples per arm. A/A and B/B raw
  gradient repeatability were exactly zero in all four runs.
- The temporary RAM ballast occupied 49.68 GiB and was stopped after the sweep. It left about
  32.5 GiB cgroup headroom before model work; the later run preflights observed 26.9–31.7 GiB
  headroom as the cgroup file cache filled. It did not change `memory.max`.

## Measurements

| MBR group size | Grouped dispatches / MoE layer | L2L median (s) | MBR median (s) | Speedup | Expert fetches / step | Logical expert bytes / step | Route-slot mismatch | Raw-gradient rel. L2 (cosine) | Loss delta MBR−L2L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 98.60 | 91.09 | 1.08× | 19,814 → 19,814 (0%) | 89.79 → 89.79 GiB (0%) | 0.000% | 0.000% (1.000000) | 0.000000 |
| 2 | 4 | 100.80 | 52.92 | 1.90× | 19,814 → 11,146 (−43.75%) | 89.79 → 50.50 GiB (−43.75%) | 2.666% | 6.095% (0.998144) | +0.002878 |
| 4 | 2 | 100.22 | 31.40 | 3.19× | 19,814 → 6,014 (−69.65%) | 89.79 → 27.25 GiB (−69.65%) | 3.012% | 10.465% (0.994658) | +0.005937 |
| 8 | 1 | 98.27 | 18.37 | 5.35× | 19,814 → 3,144 (−84.13%) | 89.79 → 14.25 GiB (−84.13%) | 2.985% | 9.948% (0.995198) | +0.004878 |

The MBR counter audit passed all 26 routed layers at each group size: forward and
`backward_dX` fetched exactly the sum of unique experts in the corresponding groups;
`recompute_forward` fetched zero experts; dispatch counts were 8/4/2/1 per layer as configured;
route IDs and gate weights matched between each branch's original forward and recompute. The
group-size-1 branch exactly matched the L2L branch's routes, loss, raw gradients, and adapter
update on this input.

## Interpretation and limits

1. Cross-microbatch grouping caused the expected reuse curve: k=2/4/8 cut logical expert reads
   by 43.8/69.7/84.1% and reduced median step time by 1.90/3.19/5.35× on this run.
2. The k=1 path is a useful implementation control: it did not reduce expert reads and showed
   exact route/gradient parity. Its 1.08× timing difference is not evidence of a speedup with
   only two timing samples.
3. Gradient and route drift begins when groups combine microbatches. It is not monotonic from
   k=4 to k=8 in this one-step sample, so these measurements do not establish a drift law or
   predict multi-step quality.
4. The reported expert-byte counters are logical runtime reads; process `read_bytes` is a
   separate process-level counter. The host reports `/workspace` on `overlay`; no physical
   NVMe/block-device read claim is made.
5. The container's `memory.max` was 91,658,125,312 bytes (85.4 GiB), not ≤32 GiB. Ballast
   reserved a similar *initial remaining budget*, but it does not emulate a hard 32-GiB cgroup
   limit and the P3 RAM acceptance criterion therefore failed. This is also not the target
   DeepSeek-V3 run and includes no heldout quality evaluation.

## Next steps

1. Keep the implementation opt-in. Do not change Soup's default path based on this screen.
2. Run the production `l2l_auto_k` capacity path only on a properly qualified host; record
   requested and resolved `effective_k` from the recipe in both arms. If those values differ,
   report a capacity/scheduler result separately from a matched-k causal comparison.
3. Repeat the fixed-k bridge on target DeepSeek-V3 and a host with `memory.max ≤ 32 GiB`, swap
   disabled, verified backing storage, and the target model/cache. Then run the paired target
   fine-tune and heldout quality gate before treating MBR as accepted.

## Artifacts

- Per-k `result.json` files and summary: `results/mbr_group_size_sweep_20261117/`.
  The larger route captures, gradient tensors, progress files and raw logs are local
  experiment artifacts and are not included in this change.
- Sweep summary: `results/mbr_group_size_sweep_20261117/summary.json`
- Harness: `benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py`
- Driver: `benchmarks/harness/run_p3_mbr_group_size_sweep.sh`
- Harness SHA-256: `e85b501da7a670c1edd8c90f39f5a3628115398f95503f0cb64537089eca4186`
- `l2l_step.py` SHA-256: `1f817cc19433d8852272e38f2c1dca00571c050ca94643cb942e622cb4e70209`

These hashes identify the historical versions used for this experiment. Both files changed
afterward; the current worktree is not expected to match these hashes.
