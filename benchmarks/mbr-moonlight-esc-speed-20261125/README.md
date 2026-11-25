# Ticket 16: C-S-E-E-S-C, the three candidate default hands in one campaign

Moonlight-16B-A3B (`2a9a177f`), RTX 3050 Laptop (sm_86, 4 GiB, one card, exclusively used here),
torch 2.14.1+cu130, CUDA 13.0, venv `~/.venvs/soup-rnd-cuda`, LoRA targets
`q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj` (rank 16). Mode: speed, **normal** (`check_mode`
refuses speed in deterministic mode). Every run is a fresh process, on AC power, inside
`systemd-inhibit --what=sleep`, wrapped in `ArmWatch` with `suspend`, `power` and `foreign_reads`
required (`speed-rule.md`). The ticket 13 health gate (`gate.py`: AC online, SM clock at least
1500 MHz under a 10 s load, enforced power limit at least 50 W) runs **before every single run**;
a refusal stops the sweep instead of writing a row. Each line in `rows.md` names model, card,
window, arm, mode and torch.

Arms: **C** route-preserving MBR with the ordered backward (the default, native expert GEMM),
**S** C with `expert_gemm="segmented"`, **E** C with `expert_gemm="lt-fused"` (the exact fused
visit, `a4f3326b`). All three hands ran under **one** allocator recipe, so the memory is
comparable and the ticket's E recipe is the one measured:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A0_SCRATCH_PREALLOC_ROWS=4096 A0_SCRATCH_PREALLOC_HID=2048 A0_SCRATCH_PREALLOC_INTER=1408
A0_SCRATCH_WHEN=lazy A0_SCRATCH_PREALLOC_GATHER=1
```

`A0_SCRATCH_PREALLOC_ROWS=4096` is the largest visit of the campaign (`k*S = 64*64`): the arena
is sized once for the widest window, so the (8,64) cell carries the same 167 MiB-class block as
(8,512) and (64,64). The arena options are inert for C and S; neither loads the extension.

## Answer

**E is the fastest exact hand on every window, and the only one that stays usable at k=64.**

| window | C step s | S step s | E step s | S vs C | E vs C | E vs S |
|---|---|---|---|---|---|---|
| (8,64) | 19.99 | 22.67 | 20.69 | **-13.42%** (t -20.50) | **-3.50%** (t -6.89) | **+8.74%** (t 19.80) |
| (8,512) | 26.81 | 29.88 | 27.71 | **-11.45%** (t -36.30) | **-3.36%** (t -8.88) | **+7.26%** (t 19.14) |
| (64,64) | 35.50 | 58.38 | 42.18 | **-64.59%** (t -30.81) | **-18.93%** (t -11.45) | **+27.75%** (t 148.15) |

A gain is read as "the second hand against the first": negative means slower, positive faster.

- Against C, the exact hands pay **3.4-3.5% at the k=8 windows for E** and **11.5-13.4% for S**;
  at (64,64) the price is **18.9% for E** and **64.6% for S** (1.64x, a cliff).
- E is **7.3-8.7% faster than S at the k=8 windows and 27.8% at (64,64)**, with a paired t of 19.1
  to 148.2 over the five blocks: the difference is far outside the noise of this card.
- **The bits are the same up to k=8** (8 of 8 cells, 216 of 216 gradients each, section below),
  so on the fitting windows the choice is speed and memory only, not bits.
- Memory: **E costs +196 MiB NVML at (8,512)** and +198 MiB at (8,64) against C under the one
  recipe of ticket 13; **S is 20 MiB below C** at (8,512). E leaves 1614 MiB of the 4096 MiB NVML
  free at its heaviest window.
- On the equal-weight sum of one step per window, C is 82.30 s, **E everywhere 90.58 s (1.101x C)**,
  **S on k<=8 + E on k=64 94.73 s (1.151x)** and **S everywhere 110.93 s (1.348x)**.

Three combinations, same numbers, in the last section. No default is switched and none is
recommended here: S everywhere is slower than the hybrid at the same memory peak, so the live
choice is between the hybrid and E everywhere, and it is a VRAM budget decision.

## Method

`mbr_measure.py esc-speed`: 5 blocks of **C-S-E-E-S-C** per window, block-major, one fresh
process per run. The order is the ticket's `A-B-C-C-B-A` with the three hands in the roles
C, S, E, E, S, C: every hand sits at the block centre (mean position 8.5 of 18), so a linear
drift of the host cancels between hands. Windows: (8,64), (8,512), (64,64) — the three that fit
4 GiB. **90 runs**: A0 has no role in this campaign and does not exist above k=8, while C, S and
E all do, so the palindrome is whole on every window.

- The number of a run is the median of its timed steps after the warm-up (`steps = 4`).
- Per block: the mean of each arm's two runs.
- Over the blocks: the mean of the per-block gains of the pair, and the paired t of the five
  gains. `t = 2.78` is the two-sided 5% critical value with 4 degrees of freedom.
- `blocks_needed` in `analysis.json` is the METHOD repeats rule read on each pair: a hand's
  per-block median stands in for its runs, so one block of C-S (or E-C, E-S) becomes one A-B-B-A
  block. Sigma is pooled over the two hands, delta is 10% of the reference hand's mean.

### Validity

97 arm records for 90 planned runs: **90 `ok`, 7 `void`**, every void one on `foreign_reads`
(1.09 to 31.2 GB read on `nvme0n1` by host processes of the session while the arm was live; the
model folder is the disk under test). The cells were re-run by `replace_void.py` under the same
gate and the replacements are the rows the table reads: **each of the 45 cells has exactly two
`ok` runs**, no cell was dropped.

Every one of the 97 runs passed the ticket 13 gate (AC online, SM clock 1740-1957 MHz under a 10 s
load, enforced limit 81.6-90.0 W), and **no gate refused a run**. The lowest per-run SM clock
maximum is **1927 MHz** against the speed-rule threshold of 1000, so no window loses its verdict to
a validity row. `blocks_needed` is **2 for every pair and every window** (the floor of
`MIN_BLOCKS`): the 5-block pilot is above N, so the repeats rule asks for no top-up and no second
computation.

## Speed by window

Absolute step seconds are the mean of the five per-block medians; a pair's gain is
`1 - second/first` per block, its t is computed on the five blocks (`t = 2.78` critical, 4 dof).

| window | C step s | S step s | E step s | S vs C | E vs C | E vs S |
|---|---|---|---|---|---|---|
| (8,64) | 19.99 | 22.67 | 20.69 | -13.42% | -3.50% | +8.74% |
| (8,512) | 26.81 | 29.88 | 27.71 | -11.45% | -3.36% | +7.26% |
| (64,64) | 35.50 | 58.38 | 42.18 | -64.59% | -18.93% | +27.75% |

| window | pair | per-block gains % | mean % | t |
|---|---|---|---|---|
| (8,64) | S-C | -11.80, -14.78, -14.26, -11.87, -14.41 | -13.42 | -20.50 |
| (8,64) | E-C | -1.71, -4.49, -4.35, -3.84, -3.10 | -3.50 | -6.89 |
| (8,64) | E-S | 9.02, 8.97, 8.67, 7.17, 9.89 | 8.74 | 19.80 |
| (8,512) | S-C | -10.37, -11.62, -12.32, -11.60, -11.36 | -11.45 | -36.30 |
| (8,512) | E-C | -2.97, -2.09, -3.78, -3.74, -4.24 | -3.36 | -8.88 |
| (8,512) | E-S | 6.70, 8.54, 7.61, 7.04, 6.39 | 7.26 | 19.14 |
| (64,64) | S-C | -56.47, -65.80, -68.31, -65.32, -67.08 | -64.59 | -30.81 |
| (64,64) | E-C | -12.73, -19.47, -22.69, -19.75, -20.02 | -18.93 | -11.45 |
| (64,64) | E-S | 27.95, 27.94, 27.10, 27.56, 28.16 | 27.75 | 148.15 |

- **The emem2 one-block ABBA is confirmed**: there E measured 27.72 s against C's 26.82 s
  (+3.34%) at (8,512); here the same pair over five blocks is 27.71 against 26.81, **-3.36%**
  (t -8.88). The single block was not a fluke.
- The k=8 pairs are tight. The two loose blocks are block 0 of (8,64) (S -11.80% against -14.3 to
  -14.8% later, the same noisy block ticket 14 saw) and the C cell of (64,64) (37.35 s in block 0
  against 34.6-35.4 s later, so E-C reads -12.73% there instead of -19.5 to -22.7%). No run was
  dropped: the rule reports the mean of the per-block gains and the list.
- S at (64,64) is the known structural cost of cutting a 64-row group into eight micro-batch
  segments (ticket 14: -65.05%, here -64.59%). E keeps the width of the grouped call and pays
  much less (-18.93%), which is the whole reason the ticket asks about a hybrid.

## Memory per arm and window

Peaks of the normal-mode runs, MiB (max over the ten runs of each cell):

| window | arm | allocated | reserved | NVML | RAM |
|---|---|---|---|---|---|
| (8,64) | C | 948 | 1058 | 1204 | 4703 |
| (8,64) | S | 942 | 1058 | 1270 | 4716 |
| (8,64) | E | 1131 | 1256 | 1402 | 4740 |
| (8,512) | C | 2042 | 2140 | 2286 | 5086 |
| (8,512) | S | 1990 | 2120 | 2266 | 5107 |
| (8,512) | E | 2094 | **2334** | **2482** | 5133 |
| (64,64) | C | 1712 | 1822 | 1968 | 5107 |
| (64,64) | S | 1661 | 1770 | 1916 | 5126 |
| (64,64) | E | 1787 | 1862 | 2010 | 5151 |

- **E against C**: +198 MiB reserved and NVML at (8,64), **+194 reserved / +196 NVML at (8,512)**,
  +40 / +42 at (64,64). The (8,512) premium reproduces ticket 13 phase 2 exactly (+194.0 / +196.0)
  under the same recipe.
- **S against C**: reserved equal at (8,64) but **NVML +66 MiB** there (1270 vs 1204: the cell's
  reserved is identical, so this is the process-side count, not a new block), -20 reserved /
  -20 NVML at (8,512), -52 / -52 at (64,64). On the two heavier windows segmentation is never
  worse; at (8,64) its NVML peak sits 66 MiB above C's.
- The premium is live working set, not trapped space: the arena is sized once for the campaign's
  largest visit (4096 rows) and the rest is E's 16.2 MiB of extension workspaces. At (64,64) the
  premium is only +40 MiB because the arena already fits the step's own staging.
- All three hands ran the same `PYTORCH_CUDA_ALLOC_CONF` (`expandable_segments:True`), so the
  comparison is allocator-for-allocator; the arena variables are inert for C and S.

## Bits: E against S on the CPU

The two exact paths were each proven against A0 in separate ladders — E at k = 1,2,4,8 in
ticket 03 (`benchmarks/mbr-moonlight-bits-20261122/`), S in ticket 14
(`benchmarks/mbr-moonlight-cdiag-20261124/`). `bits_cross.py` closes the transitive gap directly,
on the CPU, from the stored tensors: `torch.equal` on the loss and all 216 LoRA gradients, plus
the recorded `dataset_sha256` of each side.

| k | S | loss equal | grads differing | max abs | E == A0 | S == A0 | dataset sha |
|---|---|---|---|---|---|---|---|
| 1 | 64 | yes | 0 / 216 | 0.0 | yes | yes | agrees |
| 2 | 64 | yes | 0 / 216 | 0.0 | yes | yes | agrees |
| 4 | 64 | yes | 0 / 216 | 0.0 | yes | yes | agrees |
| 8 | 64 | yes | 0 / 216 | 0.0 | yes | yes | agrees |
| 1 | 512 | yes | 0 / 216 | 0.0 | yes | yes | agrees |
| 2 | 512 | yes | 0 / 216 | 0.0 | yes | yes | agrees |
| 4 | 512 | yes | 0 / 216 | 0.0 | yes | yes | agrees |
| 8 | 512 | yes | 0 / 216 | 0.0 | yes | yes | agrees |

**8 of 8 cells bit-equal, 216 of 216 gradients each**, and both sides equal to A0 in the same
cell: at k <= 8 the two exact paths are the same bits, so on the fitting windows the choice
between S and E is speed and memory only, not bits. The dataset hashes agree on both sides
(`6beb56ae...` for S=64, `91048d80...` for S=512; the E rows carry them, the S side's are in
`mbr-moonlight-cdiag-20261124/cross-day/bits.json`, and the S tensors were the ones that campaign
proved equal to A0). Above k=8 there is no A0 reference to compare against: both paths are exact
by construction (S keeps the reference row count per segment, E forces the reference kernel), so
k=64 is priced in speed and memory only. Tensors are not committed (about 28 MB each); the E side
lives under `~/moonlight-local/results/bits/`, the S side under ticket 14's `tensors/` folder.

One detail of provenance: `mbr-moonlight-cdiag-20261124/tensors/C-k2-s64.pt` was overwritten by
a plain-C reproduction run (ticket 14 README), so that cell's S tensor is read from
`direct-proof/tensors/C-k2-s64.pt` — the direct B1 proof run, `expert_gemm=segmented`, same
dataset hash. It is bit-equal to E and to A0 like the other seven.

## Combinations

One step per window, each window weighted once (the sum is a configuration's price for a step at
each of the three sizes, not a throughput number). Peaks are the max of the chosen arm's cell over
the three windows.

| candidate | (8,64) | (8,512) | (64,64) | step sum s | vs C | alloc / reserved / NVML MiB | bits |
|---|---|---|---|---|---|---|---|
| C everywhere | C 19.99 | C 26.81 | C 35.50 | **82.30** | 1.000x | 2042 / 2140 / 2286 | inexact vs A0 |
| S everywhere | S 22.67 | S 29.88 | S 58.38 | **110.93** | 1.348x | 1990 / 2120 / 2266 | exact k<=8; k=64 by construction |
| S on k<=8 + E on k=64 | S 22.67 | S 29.88 | E 42.18 | **94.73** | 1.151x | 1990 / 2120 / 2266 | exact k<=8; k=64 by construction |
| E everywhere | E 20.69 | E 27.71 | E 42.18 | **90.58** | 1.101x | 2094 / 2334 / 2482 | exact k<=8; k=64 by construction |

- **S everywhere is dominated**: the hybrid has the same three peaks (the (8,512) S cell is the
  ceiling in both) and needs **14.6% less time** on the summed step. Its one non-numeric merit is
  that it needs no compiled extension and no arena environment.
- **The live choice is the hybrid against E everywhere**: the hybrid costs **4.6% more summed step
  time (94.73 vs 90.58 s) for 216 MiB less NVML peak** (2266 vs 2482), which is 196 MiB below C's
  2286 at the hybrid's heaviest window. At (8,64) the hybrid is slower in absolute terms too
  (22.67 vs 20.69 s, +9.6%) while taking 132 MiB less NVML (1270 vs 1402).
- **C is the fastest and not exact**: E's summed step is 10.1% longer and S's 34.8% longer. The
  exactness premium of the best exact hand is therefore about a tenth of one step at this mix.
- The combination table in `analysis.json` carries the same numbers per window (`per_window`,
  `steps_rel_C`, `peaks`, `bits`), so a different window mix can be read from it without rerunning.

### The ticket's two outcomes

E is locally faster than S on all three windows (+7.3%, +8.7%, +27.8%), so the campaign lands on
the ticket's second branch: **the gain over S is paid in memory**, +196 MiB NVML at (8,512) and
+198 at (8,64), and it is cheap at (64,64) (+42). If the VRAM budget is the binding constraint, the
hybrid keeps S's peak and most of the gain; if it is not, E everywhere is the fastest exact
candidate. **The default decision is the user's.**

## What this does and does not change

- It measures the three candidate hands on one card in one day, with one allocator recipe and one
  gate. No default is switched: the recommendation is the user's.
- C drifts (5.40x/4.61x over A0 on 20261121); it is the baseline of this campaign, not a stable
  constant. The campaign's C is measured inside the same blocks as S and E, so the deltas do not
  depend on that drift.
- `lt-fused` stays opt-in and needs the arena options above on a 4 GiB card.
- (64,64) has no bit reference; the S and E numbers there are speed and memory only.

## Reproduce

```bash
# the speed campaign: gate before every run, AC check, lock, systemd-inhibit, then replacements
bash benchmarks/mbr-moonlight-esc-speed-20261125/run_speed.sh
python benchmarks/mbr-moonlight-esc-speed-20261125/analyze.py \
  --root benchmarks/mbr-moonlight-esc-speed-20261125 \
  --bits benchmarks/mbr-moonlight-esc-speed-20261125/bits-cross.json

# the CPU bit cross (no GPU, about a minute)
python benchmarks/mbr-moonlight-esc-speed-20261125/bits_cross.py \
  --e-root $HOME/moonlight-local/results/bits \
  --s-dir benchmarks/mbr-moonlight-cdiag-20261124/tensors \
  --a0-dir benchmarks/mbr-moonlight-cdiag-20261124/tensors \
  --s-cell 2,64=benchmarks/mbr-moonlight-cdiag-20261124/direct-proof/tensors/C-k2-s64.pt \
  --e-rows benchmarks/mbr-moonlight-bits-20261122/s64/rows.jsonl \
           benchmarks/mbr-moonlight-bits-20261122/s512-math/rows.jsonl \
  --s-sha benchmarks/mbr-moonlight-cdiag-20261124/cross-day/bits.json \
  --out benchmarks/mbr-moonlight-esc-speed-20261125/bits-cross.json
```

`run_speed.sh` refuses to run over existing rows (unless `START_BLOCK` resumes a stopped pass),
holds `speed/.speed.lock`, checks AC power before the pass and before the replacements, and wraps
both in `systemd-inhibit --what=sleep`. `replace_void.py` re-runs only the slots without a valid
run, under the same gate, appending them under the same block number.

## Acceptance

- [x] Campaign: hands C-S-E-E-S-C (the ticket's `A-B-C-C-B-A` with three hands), 5 blocks,
      normal mode, windows (8,64), (8,512), (64,64), **90 runs**; ArmWatch on every run with
      `suspend`, `power`, `foreign_reads` required; ticket 13 health gate before every run;
      `systemd-inhibit`; 7 void runs replaced by the rule.
- [x] Table by window: absolute step seconds and the paired deltas S-C, E-C, E-S with t over the
      five blocks; every line of `rows.md` names model, card, window, arm, mode and torch.
- [x] Memory per arm and window (allocated, reserved, NVML, plus RAM) from the same runs.
- [x] CPU bit cross E against S: dataset hashes agree, `torch.equal` on the loss and 216 gradients
      in all 8 cells (k=1,2,4,8 on (8,64) and S=512 math), both sides equal to A0.
- [x] Summary of the three candidates (S everywhere / S on k<=8 + E on k=64 / E everywhere) with
      speed, memory and bits, and the anti-recommendation the numbers support (S everywhere is
      dominated on speed at equal memory). No default recommendation is made.
- [x] Both outcomes of the ticket: E is faster than S, so the price is stated in VRAM (hybrid
      keeps S's peak, E everywhere costs +216 MiB of peak) and the decision goes to the user.
- [x] No full suite was run (only the touched test files and the harness tests they load).

## Files

- `run_speed.sh` — the campaign (health gate before every run, AC check, lock,
  `systemd-inhibit`, then the void replacements).
- `analyze.py` — `analysis.json` and `rows.md` from the raw rows: per window the arm means, the
  three deltas with t, per-block steps, memory peaks, the repeats-rule block count and the three
  combinations.
- `replace_void.py` — re-runs the slots with no valid run of the `esc-speed` plan.
- `bits_cross.py`, `bits-cross.json`, `bits-cross.md` — the CPU cross of E against S.
- `speed/rows.jsonl`, `speed/arm_records.jsonl` — every run, void ones included.
- `speed/summary.md` — the harness table of the cells.
- `environment.txt` — card, driver, torch, model revision, allocator recipe.
- `rows.md` — one result line per measured cell.
- `analysis.json` — the numbers behind the tables above.
- `*.log` — campaign stdout/stderr, git-ignored.
