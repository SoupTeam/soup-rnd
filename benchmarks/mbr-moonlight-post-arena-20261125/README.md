# Ticket 17: E's bits after the arena, G's speed, and the D+segmented cell

Moonlight-16B-A3B (`2a9a177f`), RTX 3050 Laptop (sm_86, 4 GiB, one card, used exclusively here),
torch 2.14.1+cu130, CUDA 13.0, venv `~/.venvs/soup-rnd-cuda`, LoRA targets
`q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj` (rank 16). Every run is a fresh process, on AC power,
inside `systemd-inhibit --what=sleep`, wrapped in `ArmWatch` with `suspend`, `power` and
`foreign_reads` required (`speed-rule.md`). The ticket 13 health gate (`gate.py`: AC online, SM
clock at least 1500 MHz under a 10 s load, enforced power limit at least 50 W) ran **before every
single run**: 60 gate checks in the speed pass, 22 in the bits pass, zero refusals, SM 1740-1957
MHz, limit 89.7-90.0 W. All hands ran under the one memory recipe of ticket 13 phase 2:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A0_SCRATCH_PREALLOC_ROWS=4096 A0_SCRATCH_PREALLOC_HID=2048 A0_SCRATCH_PREALLOC_INTER=1408
A0_SCRATCH_WHEN=lazy A0_SCRATCH_PREALLOC_GATHER=1
```

`A0_SCRATCH_PREALLOC_ROWS=4096` is the largest visit of the campaign (`k*S = 64*64`). The arena
options are inert for C, D and segmented; they only size the `lt-fused` scratch arena.

## Answer

**E's bits survive the arena, G's bits survive the new memory path, and G is the fastest exact
hand on both windows: 8.6% and 13.6% above C, with E's memory.**

| window | C step s | E step s | G step s | G vs C | E vs C | G vs E |
|---|---|---|---|---|---|---|
| (8,64) | 20.46 | 20.70 | **18.68** | **+8.60%** (t 5.62) | -1.28% (t -0.67) | **+9.74%** (t 26.31) |
| (8,512) | 27.40 | 27.77 | **23.65** | **+13.63%** (t 12.61) | -1.40% (t -1.01) | **+14.81%** (t 57.97) |

A gain is read as "the second hand against the first": positive means faster. The per-block gains
of G against C are `[7.85, 7.57, 14.62, 6.23, 6.74]` at (8,64) and
`[15.34, 17.00, 12.13, 12.21, 11.44]` at (8,512): G won all five blocks on both windows.

- **G is bit-exact at the cell measured after the arena** (k=2, (8,64): 0 of 216 gradients
  differ, loss equal). Ticket 03 proved G equal to A0 at every k before the arena; this campaign
  re-proves the cell under the new memory path.
- **G costs E's memory, not more**: NVML +194 MiB over C at (8,512), +198 MiB at (8,64) — the
  same numbers E pays (ticket 16: +194/+198), and G's reserved equals E's to the MiB.
- **E is not slower than C here**, but the difference is inside the card's noise: -1.3% and -1.4%
  with t below the 2.78 threshold, against -3.5%/-3.4% (t -6.9/-8.9) in ticket 16. The direction
  matches; the magnitude is this day's.
- **The D drift at k=2 was the forward.** D plain differs from A0 (108/216 gradients, max abs
  4.51e-2); D with `expert_gemm=segmented` (DS) is bit-equal to A0 (0/216, loss equal). D's own
  decoupled backward is exact in this cell; the grouped native forward was the whole drift.
  The decision on D does not change: it is slower than C in every window (ticket 06) and this
  cell only names the mechanism. Comment added to ticket 06.

## Summary: window x hand x speed x memory x bits

| window | hand | step s | NVML peak GiB | bits against A0 up to k=8 |
|---|---|---|---|---|
| (8,64) | C | 20.46 | 1.18 | differs at k=2,4,8 (ticket 03) |
| (8,64) | E | 20.70 | 1.37 | equal at k=1,2,4,8 (this campaign) |
| (8,64) | **G** | **18.68** | 1.37 | equal at k=2 (this campaign); equal at all k (ticket 03) |
| (8,512) | C | 27.40 | 2.23 | differs at k=2,4,8 (ticket 03) |
| (8,512) | E | 27.77 | 2.42 | equal at k=1,2,4,8 (this campaign) |
| (8,512) | **G** | **23.65** | 2.42 | equal at all k (ticket 03) |

## Bits

Deterministic mode (`torch.use_deterministic_algorithms(True)`), `torch.equal` on the loss and
all 216 LoRA gradients of fresh runs, plus a CPU cross against the stored tensors of tickets 03
and 06.

| cell | arm | vs A0 | loss equal | grads differing | max abs | dataset_sha |
|---|---|---|---|---|---|---|
| k=1, S=64 | E | equal | yes | 0 / 216 | 0 | agrees (6beb56ae) |
| k=2, S=64 | E | equal | yes | 0 / 216 | 0 | agrees |
| k=4, S=64 | E | equal | yes | 0 / 216 | 0 | agrees |
| k=8, S=64 | E | equal | yes | 0 / 216 | 0 | agrees |
| k=1, S=512 math | E | equal | yes | 0 / 216 | 0 | agrees (91048d80) |
| k=2, S=512 math | E | equal | yes | 0 / 216 | 0 | agrees |
| k=4, S=512 math | E | equal | yes | 0 / 216 | 0 | agrees |
| k=8, S=512 math | E | equal | yes | 0 / 216 | 0 | agrees |
| k=2, S=64 | G | equal | yes | 0 / 216 | 0 | agrees |
| k=2, S=64 | D | **differs** | no | 108 / 216 | 4.51e-2 | agrees |
| k=2, S=64 | DS | equal | yes | 0 / 216 | 0 | agrees |

Stored-tensor cross (CPU, `torch.equal` on loss and all gradients):

| fresh | stored | result |
|---|---|---|
| A0, every E cell | ticket 03 A0 (20261122) | 0 differences |
| E, every E cell | ticket 03 E (20261122) | 0 differences |
| A0, G cell | ticket 03 A0 | 0 differences |
| G, G cell | ticket 03 G | 0 differences |
| A0, D cell | ticket 03 A0 | 0 differences |
| D, D cell | ticket 06 D (20261124) | 0 differences |
| A0, DS cell | ticket 03 A0 | 0 differences |
| DS vs D (same day) | fresh D | **108 / 216**, max abs 4.51e-2 |

The window (64,64) shares the S=64 row, so the ladder is counted once for (8,64) and (64,64)
(ticket 03 convention). A0 does not run at k=64, so no A0 reference exists there.

## Memory

Peaks of the speed runs, max over the ten runs of the cell, GiB (allocated / reserved / NVML /
host RAM):

| window | C | E | G |
|---|---|---|---|
| (8,64) | 0.93 / 1.03 / 1.18 / 4.59 | 1.10 / 1.23 / 1.37 / 4.63 | 1.10 / 1.23 / 1.37 / 4.58 |
| (8,512) | 1.99 / 2.09 / 2.23 / 4.97 | 2.05 / 2.28 / 2.42 / 5.01 | 2.05 / 2.28 / 2.42 / 4.97 |

G spends E's arena and nothing besides: at (8,512) G's NVML peak is 2 MiB below E's and reserved
is equal; against C the premium is **+194 MiB NVML / +194 MiB reserved** at (8,512) and
**+198 / +198** at (8,64), the ticket 13/16 price.

## Method

- Speed: `mbr_measure.py ceg-speed`, 5 blocks of **C-E-G-G-E-C** per window, block-major, one
  fresh process per run: 60 runs, 60 ok, no void replacement needed. The order is the ticket's
  `A-B-C-C-B-A` with the three hands in the roles C, E, G, so every hand sits at the block centre
  and a linear drift of the host cancels. Per-block number: the median of an arm's two runs;
  window number: the mean of the five block medians; paired t on the per-block gains with the
  5-block critical value 2.78. `blocks_needed` is 2 in every pair and window, so the five blocks
  are more than the rule asks.
- Bits: `mbr_measure.py lever-bits` (A0 and E at k=1,2,4,8 on S=64 and S=512 math; A0 and G, A0
  and D at k=2, S=64) and `ds-bits` (A0 and DS at k=2, S=64), deterministic mode. `--arms` keeps
  the reference and the one hand under test. Every run wrote its loss and gradients to
  `bits/<label>/tensors/` (git-ignored, about 28 MB each).
- The DS arm is new wiring: D's step options with `expert_gemm="segmented"`. Under the decoupled
  backward the segmented hook used to refuse the joined group, because phase R1 forwards a layer
  with the routed experts skipped and its route captures were never consumed
  (`micro-batch order [64, 64] does not match ... [64 x 118]`). The hook now reads the group's own
  trailing captures (`c3c61b8d`, red test in `tests/test_p72_segmented_expert_gemm.py`).

## Reproduce

```bash
bash benchmarks/mbr-moonlight-post-arena-20261125/run_bits.sh
bash benchmarks/mbr-moonlight-post-arena-20261125/run_speed.sh
python benchmarks/mbr-moonlight-post-arena-20261125/analyze.py \
    --root benchmarks/mbr-moonlight-post-arena-20261125
```

`analysis.json` holds the per-block numbers, the deltas with their t, the memory peaks, the
blocks-needed values and the CPU crosses; `rows.md` has one line per measured cell with model,
card, window, arm, mode and torch.

## What this changes for the default

The candidate list of ticket 16 was C (native, drifts), S (exact, slower) and E (exact, +194 MiB).
On both windows measured here **G dominates E**: same memory, 9.7-14.8% faster, bits equal at the
cell measured, and 8.6-13.6% faster than C while keeping bits. So the live question is no longer
"E or the hybrid" but "G becomes the default or not", and that is a decision for the user. Two
gaps before switching, both of them measurements, not code:

- G's speed at (64,64) is not measured in this campaign; ticket 16 measured E there (42.18 s
  against C 35.50) and ticket 03 measured G's bits at k=64 against nothing (no A0 above k=8).
  `fast_row_gather` pays most where expert calls are many, so (64,64) is where G could lose its
  gain.
- G's bits after the arena are re-proved only at k=2. Ticket 03's all-k proof predates the arena;
  the arena changed E's allocation, not its algorithm, and E's all-k ladder still matches A0 bit
  for bit, which is why the one G cell is read as enough for the memory question but not for a
  default switch.
