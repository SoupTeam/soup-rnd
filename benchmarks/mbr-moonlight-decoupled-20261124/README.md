# Ticket 06, part 2: decoupled MBR on Moonlight-16B-A3B

Moonlight-16B-A3B, RTX 3050 Laptop (sm_86, 4 GiB VRAM, 14 GiB RAM), torch 2.14.1+cu130,
CUDA 13.0, venv `~/.venvs/soup-rnd-cuda`, model revision `2a9a177f`, LoRA targets
`q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj` (rank 16). Arms: A0 (plain L2L, the bit
reference at k <= 8), C (route-preserving MBR with the ordered backward; at k = 64 the
reference), D (C's decoupled sibling: `mbr_decoupled_subwindow=True` over the whole group).
Bit and memory rows are deterministic mode; speed rows are normal mode.

Result: D is bit-equal to A0 only at k = 1. From k = 2 it differs exactly as C does, because
its forward is C's grouped native expert GEMM; its own decoupled backward is what it adds, and
that changes the gradients against C at the 1e-3 to 1e-2 relative level. It never uses more
VRAM, but it does not remove C's (64,512) OOM either. The speed gate decides whether the
opt-in has a reason to exist.

## Bits (`run_bits.sh`)

`torch.equal` on the saved loss and every one of the 216 LoRA gradients, against the reference
arm of the k (A0 at k <= 8, C at k = 64). Every cell was repeated in a fresh folder
(`run_repeats.sh`); all 18 cells reproduced bit-for-bit (`repeats.json`), so the verdicts below
are arm properties, not run noise.

| window | arm vs reference | verdict | loss equal | grads differing | max abs | `grad_rel_l2` | `min_cos` |
|---|---|---|---|---|---|---|---|
| (1,64) | D vs A0 | equal | yes | 0/216 | 0 | 0 | 1.000000 |
| (2,64) | D vs A0 | differs | no | 108/216 | 4.51e-2 | 1.303e-1 | 0.978251 |
| (4,64) | D vs A0 | differs | no | 108/216 | 4.54e-2 | 1.021e-1 | 0.983504 |
| (8,64) | D vs A0 | differs | no | 108/216 | 4.26e-2 | 7.859e-2 | 0.985339 |
| (8,512) math | D vs A0 | differs | no | 108/216 | 2.77e-2 | 3.298e-2 | 0.997029 |
| (64,64) | D vs C | differs | yes | 108/216 | 1.54e-3 | 3.833e-3 | 0.999990 |

The first four rows repeat ticket 03's C rows: C differs from A0 at k = 2, 4 and 8 with the
same count and size (k = 2: 108/216, max 4.68e-2, `grad_rel_l2` 1.286e-1). D's loss is
bit-identical to C at every window that produced one, including (64,64); only the backward
differs. At (64,512) neither arm produced a loss: both OOM.
D against C, the isolation of the decoupling itself:

| window | loss equal | grads differing | max abs | `grad_rel_l2` | `min_cos` |
|---|---|---|---|---|---|
| (2,64) | yes | 108/216 | 5.28e-3 | 1.558e-2 | 0.999724 |
| (4,64) | yes | 108/216 | 3.91e-3 | 9.058e-3 | 0.999932 |
| (8,64) | yes | 108/216 | 2.99e-3 | 6.914e-3 | 0.999927 |
| (8,512) math | yes | 108/216 | 1.67e-3 | 2.171e-3 | 0.999980 |
| (64,64) | yes | 108/216 | 1.54e-3 | 3.833e-3 | 0.999990 |

### What a bits verdict may read

- A row whose `outcome` is not `ok`, or is missing, reads as `void`. Its pair gets no verdict.
- When `arm_records.jsonl` exists, row N binds to record N: the arm must match, and `k`, `seq` and
  `run` must match wherever both carry the field. The record's `outcome` must be `ok`.
- A folder without `arm_records.jsonl` binds nothing beyond its own row outcome. This is a limit:
  such a folder cannot prove its rows belong to the arm they name. Every folder this analysis reads
  has its records.
- Duplicate rows: the last row of each (k, arm) decides, as in `mbr_measure.py`. An earlier
  duplicate is overwritten, so it cannot yield `equal`. The test
  `test_a_duplicate_row_gives_the_verdict_and_the_diff_the_same_row` pins this rule.

### Why the CPU equality does not carry to this card

Part 1 pinned NF4+BF16 equality to A0 at k = 1, 2, 4 on CPU. That equality is real, but it is a
property of the CPU matmul: there the grouped expert call and the per-micro-batch calls round
to the same result. On this card the forward expert call of both C and D runs the group's
rows in one native cuBLAS GEMM, and cuBLAS picks a different algorithm than it does for A0's
per-micro-batch row counts. The loss therefore changes at k >= 2 before any backward runs, and
no backward ordering can restore it. `lt-fused` (arm E) is what forces the per-micro-batch
algorithm groups in the forward and restores A0's bits (ticket 03); D refuses it by design,
because its own backward does not go through the fused node
(`docs/performance-and-quantization.md`). The honest consequence: D is not an exact lever on
CUDA sm_86 above k = 1, and the ticket's D-against-A0 ladder is not met at k = 2, 4, 8.

## Window (64,512): the local ace does not open

| arm | status | alloc GiB | reserved GiB | NVML GiB | error |
|---|---|---|---|---|---|
| C | OOM | 3.45 | 3.52 | 3.68 | `Tried to allocate 20.00 MiB` |
| D | OOM | 2.85 | 3.52 | 3.68 | `Tried to allocate 148.00 MiB` |

D's live set is 0.6 GiB smaller, but both arms hit the same reserved ceiling and neither fits.
No OOM relief, so the VRAM gate does not open for D. The switch itself is pinned by a unit
test (`large_window_enabled`: C fits -> decoupled is not enabled); its runtime event here is
"measured, no relief" in `analysis.json` (`large_window.c_oom_relieved` is false).

## Memory (`alloc`, `reserved`, NVML, RAM) per window, deterministic rows

| window | arm | alloc GiB | reserved GiB | NVML GiB | RAM GiB |
|---|---|---|---|---|---|
| (1,64) | A0 / C / D | 0.96 / 0.96 / 0.96 | 1.04 / 1.05 / 1.04 | 1.23 / 1.24 / 1.23 | 4.58 / 4.58 / 4.57 |
| (8,64) | A0 / C / D | 0.97 / 0.98 / 0.97 | 1.09 / 1.15 / 1.10 | 1.28 / 1.34 / 1.29 | 4.62 / 4.63 / 4.63 |
| (64,64) | C / D | 1.73 / 1.15 | 1.93 / 1.35 | 2.12 / 1.54 | 5.02 / 5.02 |
| (8,512) math | A0 / C / D | 1.81 / 2.23 / 1.84 | 2.09 / 2.42 / 2.40 | 2.28 / 2.61 / 2.59 | 5.00 / 5.00 / 5.01 |
| (64,512) | C / D | 3.45 / 2.85 | 3.52 / 3.52 | 3.68 / 3.68 | 7.09 / 7.10 |

Peak VRAM is never worse for D: at (64,64) it saves 0.58 GiB allocated and 0.58 GiB NVML, at
(8,512) 0.39 GiB allocated, and at (8,64) it is within 0.05 GiB of C.

## Speed (`run_speed.sh`)

C-D-D-C, the two-hand A-B-B-A form of the ticket's A-B-C-C-B-A, five blocks, normal mode, one
fresh process per run, `ArmWatch` with suspend, power and foreign reads required, the whole
campaign under `systemd-inhibit --what=sleep`, AC online checked before every pass. All 60
runs are valid: 0 void, 0 OOM, every arm record `ok`, the lowest per-run SM clock maximum
1920 MHz, and no check failure (`analysis.json`).

| window | C step s | D step s | paired gain | per-block gains % | t (5 pairs) |
|---|---|---|---|---|---|
| (8,64) | 20.32 | 21.21 | **-4.38%** | -3.92, -5.45, -3.91, -4.36, -4.23 | -15.43 |
| (8,512) | 27.15 | 28.02 | **-3.19%** | -2.51, -3.35, -2.96, -3.84, -3.29 | -14.48 |
| (64,64) | 35.91 | 53.28 | **-48.39%** | -47.33, -49.77, -51.68, -47.36, -45.80 | -46.52 |

The gate wants +3% and t > 2.78 in at least two of the three windows. **0 of 3**; the sign is
negative in all five blocks of every window. The (64,64) loss is structural: the sub-windowed
recompute re-runs attention, router and shared FFN per micro-batch, 64 times, where C's ordered
backward keeps one engine call over the group.

Memory in the normal-mode speed rows (peaks, GiB):

| window | C alloc / reserved / NVML | D alloc / reserved / NVML |
|---|---|---|
| (8,64) | 0.93 / 1.12 / 1.26 | 0.92 / 1.06 / 1.21 |
| (8,512) | 2.02 / 2.11 / 2.25 | 1.80 / 2.07 / 2.21 |
| (64,64) | 1.69 / 1.89 / 2.03 | 1.11 / 1.29 / 1.43 |

## Verdict

**The lever is not taken.** D fails the bits gate above k = 1, is slower in all three fitting
windows (0 of 3), and does not remove the one OOM it was kept alive for. The numbers:

- Bits: 108 of 216 gradients differ from A0 at k = 2, 4 and 8, `grad_rel_l2` 0.079 to 0.130,
  `min_cos` 0.978 to 0.985; the same class and size as C's known rows. Against C, the loss is
  exact everywhere and the gradients differ at 0.7% to 1.6% relative, `min_cos` >= 0.9997. This
  is the D1 opt-in row, not an exact lever.
- Speed: -4.38% / -3.19% / -48.39%, t -15.4 / -14.5 / -46.5; 0 of 3 windows at +3%.
- VRAM: never worse, up to 0.58 GiB allocated saved at (64,64), but (64,512) is OOM in both
  arms. `large_window_enabled` is true by the switch's condition (C does not fit), yet
  `c_oom_relieved` is false because D does not fit either. The card's ace does not open.
- k = 64 against C: loss equal, 108/216 gradients differ at `grad_rel_l2` 3.8e-3,
  `min_cos` 0.99999; recorded, not a blocker, as the ticket says.

Where the lever stands: `mbr_decoupled_subwindow` stays off by default (part 1), and nothing in
this campaign is a reason to promote it or to buy a Vast run for it. Whether the already
reviewed opt-in code stays on the branch or is reverted is a separate PR decision; this ticket
only records that the lever earns no place in the default stack.

The one unmeasured path to A0 bits above k = 1 is `lt-fused` forward with the decoupled
backward. The decoupled path refuses a patched expert node by design (part 1), so that
combination needs its own ticket and its own bit ladder before any claim.


## Reproduce

```bash
bash benchmarks/mbr-moonlight-decoupled-20261124/run_bits.sh
bash benchmarks/mbr-moonlight-decoupled-20261124/run_repeats.sh
bash benchmarks/mbr-moonlight-decoupled-20261124/run_speed.sh
python benchmarks/mbr-moonlight-decoupled-20261124/compare_repeats.py \
  --root benchmarks/mbr-moonlight-decoupled-20261124
python benchmarks/mbr-moonlight-decoupled-20261124/analyze.py \
  --root benchmarks/mbr-moonlight-decoupled-20261124
```

Result lines: `rows.md`. Machine-readable: `analysis.json`, `repeats.json`, and the per-pass
`rows.jsonl`. Raw tensors stay in `*/tensors/`, git-ignored (about 60 MB per cell).

## Not done here

- No Kanana run: this is the local card; the Kanana gates stay in `../mbr-unified/issues/05`.
- No retry of (64,512) with allocator tricks (`expandable_segments`, forced `empty_cache`): the
  ticket records the OOM; forcing it would change the comparison.
- The `lt-fused` forward combined with the decoupled backward is not measured: the decoupled
  path refuses a patched expert node by design (part 1). It is the only candidate seen here
  that could restore A0's bits above k = 1, and it would need its own ticket.
