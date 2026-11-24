# Ticket 14 tail: the price of the segmented expert GEMM

Moonlight-16B-A3B (`2a9a177f`), RTX 3050 Laptop (sm_86, 4 GiB, one card, exclusively used here),
torch 2.14.1+cu130, CUDA 13.0, venv `~/.venvs/soup-rnd-cuda`, LoRA targets
`q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj` (rank 16). Mode: speed, **normal**
(`check_mode` refuses speed in deterministic mode). Every run is a fresh process, on AC power,
inside `systemd-inhibit --what=sleep`, wrapped in `ArmWatch` with `suspend`, `power` and
`foreign_reads` required (`speed-rule.md`). Each line in `rows.md` names model, card, window,
arm, mode and torch.

Arms: **A0** plain L2L (the bit reference at k <= 8), **C** route-preserving MBR with the
ordered backward (the default), **S** C with `expert_gemm="segmented"` — the mode from ticket 14
that keeps every expert GEMM at the reference row count and is bit-equal to A0 at k = 1,2,4,8 in
(8,64) and (8,512) math (`../mbr-moonlight-cdiag-20261124/README.md`). This campaign prices that
mode; it does not re-measure its bits.

## Answer

**Segmentation is not free at the fitting windows, and it has a second, much larger price at
k = 64.**

| window | A0 step s | C step s | S step s | S vs C | t (5 pairs) | S vs A0 | C/A0 | S/A0 |
|---|---|---|---|---|---|---|---|---|
| (8,64) | 110.74 | 20.51 | 23.10 | **-12.65%** | -11.52 | +79.14% | 5.40x | 4.79x |
| (8,512) | 125.36 | 27.22 | 30.45 | **-11.89%** | -142.24 | +75.71% | 4.61x | 4.12x |
| (64,64) | - (no A0 at k=64) | 35.67 | 58.87 | **-65.05%** | -317.38 | - | - | - |

- Against C: **-12.65% at (8,64), -11.89% at (8,512), -65.05% at (64,64)**. On the ticket's
  scale the two k=8 windows are a visible price (above 3%, below 2x); (64,64) is a multiple,
  2.87x slower than C, a separate point on the scale.
- Against A0: at the k=8 windows S keeps **4.79x** and **4.12x**, against C's **5.40x** and
  **4.61x**, i.e. **89% of C's speed advantage over A0 survives** segmentation. The exact path
  stays several times faster than the plain step; it does not stay as fast as the inexact one.
- Memory: S is never worse than C. Allocated/reserved/NVML are equal at (8,64) (0.93/1.12/1.26
  GiB), 0.04/0.02/0.02 GiB lower at (8,512), and 0.05/0.07/0.07 GiB lower at (64,64).
- No default switch: this campaign only prices the mode. Whether C's default moves to
  `segmented` is the user's decision; the numbers above are the input to it.

The per-block gains show the shape of the cost; at (64,64) it is flat and large, at the k=8
windows it is flat except block 0 of (8,64):

| window | per-block S-vs-C gains % | per-block S-vs-A0 gains % |
|---|---|---|
| (8,64) | -8.30, -13.70, -14.28, -13.68, -13.27 | 79.38, 79.15, 78.95, 79.10, 79.13 |
| (8,512) | -11.93, -11.79, -12.20, -11.76, -11.76 | 75.77, 75.68, 75.65, 75.72, 75.71 |
| (64,64) | -64.25, -65.32, -65.28, -65.09, -65.32 | - |

Block 0 of (8,64) is the one noisy block: its second C run measured 22.32 s against 19.91-20.43 s
for the other nine C runs, so that block's mean gain is -8.30% instead of the -13.3 to -14.3% of
blocks 1-4. No run was dropped: the method reports the mean of the five per-block gains and the
per-block list, and the t is computed on those five values.

## Memory per arm and window

Peaks of the normal-mode runs, GiB (max over the ten runs of each cell):

| window | arm | allocated | reserved | NVML | RAM |
|---|---|---|---|---|---|
| (8,64) | A0 / C / S | 0.92 / 0.93 / 0.93 | 1.06 / 1.12 / 1.12 | 1.20 / 1.26 / 1.26 | 4.57 / 4.57 / 4.59 |
| (8,512) | A0 / C / S | 1.77 / 2.02 / 1.97 | 2.07 / 2.11 / 2.09 | 2.21 / 2.25 / 2.23 | 4.95 / 4.95 / 4.97 |
| (64,64) | C / S | 1.69 / 1.63 | 1.89 / 1.82 | 2.03 / 1.96 | 4.97 / 4.99 |

S against C: 0 / -6 MiB allocated at (8,64), -43 / -16 / -16 MiB allocated/reserved/NVML at
(8,512), -52 / -70 / -70 MiB at (64,64). The segmented mode does not buy memory back and does not
cost any.

## Method

`mbr_measure.py seg-speed`: 5 blocks of **A0-C-S-S-C-A0** per window, block-major, one fresh
process per run. The order is the ticket's `A-B-C-C-B-A` with three hands; every hand sits at mean
position 3.5, so a drift of the host cancels between hands.

Windows: (8,64), (8,512), (64,64) — the three that fit 4 GiB. **80 runs, not 90**: A0 is not
planned above k = 8 (`A0_MAX_K`; its micro-batches no longer fit), so (64,64) keeps the
C-S-S-C core, 6 + 6 + 4 = 16 runs per block. No A0 verdict is claimed at (64,64).

- The number of a run is the median of its timed steps after the warm-up (`steps = 4`).
- Per block: the mean of each arm's two runs (block 0 of (8,64) and (8,512) has an A0 hand; the
  two void runs there were replaced, see below).
- Over the blocks: the mean of the per-block gains, and the paired t of the five gains.
  `t = 2.78` is the two-sided 5% critical value with 4 degrees of freedom.
- Check: `ArmWatch` requires `suspend`, `power` and `foreign_reads` on every run; the model folder
  is `source/`, the disk under test.

### Validity

82 arm records for 80 planned runs: 80 `ok`, 2 `void` — both in block 0 of (8,64), A0 and S,
`foreign_reads` (3.7 GB read on `nvme0n1` by host processes of the agent session, mostly
`opencode-pty` running an unrelated test pass while the arm was live). Both cells were re-run by
`replace_void.py` and the ok replacements are the runs the table reads. Every other record is
`ok`; the lowest per-run SM clock maximum is 1942 MHz (threshold 1000), so no window loses its
verdict to a validity row.

`replace_void.py` settles one run per planned slot (each arm runs twice per block): a cell with
one ok row and one void row still gets its second run scheduled.

## What this does and does not change

- It prices `segmented` on this card. The bits were already proven bit-equal to A0 at k = 1,2,4,8
  in (8,64) and (8,512) math (ticket 14), so the 12% at the k=8 windows is the cost of exact
  bits through native cuBLAS per micro-batch.
- At (64,64) the same exactness costs 65% (2.87x) against C. A 64-row group cut into eight
  micro-batch segments loses the batched GEMM's width; that is structural, not noise (t -317).
- The comparison to `lt-fused` (arm E) is not re-run here. E is the other exact path, with its
  own memory premium measured in ticket 13; this campaign does not claim which exact path is
  cheaper.
- No default is switched and no switch is recommended. C stays `native`; `segmented` stays an
  opt-in mode with a measured price.

## Reproduce

```bash
bash benchmarks/mbr-moonlight-seg-speed-20261124/run_speed.sh
python benchmarks/mbr-moonlight-seg-speed-20261124/analyze.py \
  --root benchmarks/mbr-moonlight-seg-speed-20261124
```

`run_speed.sh` refuses to run over existing rows, holds `speed/.speed.lock`, checks AC power
before the pass and before the replacements, and wraps both in `systemd-inhibit --what=sleep`.
`replace_void.py` re-runs only the slots without a valid run, appending them under the same block
number.

## Acceptance

- [x] Three hands measured: A0 (context), C `native`, S `segmented`; A-B-C-C-B-A per window,
      5 blocks, normal mode, windows (8,64), (8,512), (64,64).
- [x] 80 runs of the 90 in the ticket's count: A0 does not exist at k = 64 (`A0_MAX_K = 8`), so
      (64,64) runs C-S-S-C and gets no A0 column.
- [x] Void runs replaced per `speed-rule.md`: 2 of 82 records, both in block 0 of (8,64); the
      replacements are ok and are the rows the verdict reads.
- [x] Memory for every arm and window: allocated, reserved, NVML (plus RAM).
- [x] Deltas S vs C (price of segmentation) and S vs A0 (what is left of C's advantage), with
      per-block gains and a paired t over the five blocks.
- [x] Verdict on the ticket's scale: k=8 windows are a visible price (~12%, above 3%, below 2x);
      (64,64) is a multiple (65%, 2.87x). No default recommendation.
- [x] Every result line names model, card, window, arm, mode and torch (`rows.md`).
- [x] Harness change under TDD: red tests first for the `S` arm and the `seg-speed` plan
      (`tests/test_mbr_measure.py`), then the campaign tests (`tests/test_mbr_seg_speed_campaign.py`,
      including the two-run replacement slot and the `deadline_hours` field of the replacement
      namespace).

## Files

- `analyze.py` — `analysis.json` and `rows.md` from the raw rows; per window the arm means, both
  deltas with t, per-block steps, memory peaks.
- `replace_void.py` — re-runs the slots with no valid run of the `seg-speed` plan.
- `run_speed.sh` — the campaign (AC check, lock, `systemd-inhibit`, then replacements).
- `speed/rows.jsonl`, `speed/arm_records.jsonl` — every run, void ones included.
- `speed/summary.md` — the harness table of the cells.
- `environment.txt` — card, driver, torch and model revision.
- `rows.md` — one result line per measured cell.
- `analysis.json` — the numbers behind the tables above.
- `*.log` — campaign stdout/stderr, git-ignored.
