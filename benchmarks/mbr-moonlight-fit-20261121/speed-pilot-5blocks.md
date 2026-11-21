# Moonlight speed, A0 against C: five-block pilot

Pilot of `speed-rule.md`: A0 B C C A0 blocks on the windows (8,64) and (8,512), normal mode, one
fresh process per run, `ArmWatch` with suspend, power and foreign_reads required. The number of a
run is the median of its timed steps.

## How the pilot ran

1. Block 1 ran first, with `abba --blocks 5`. It was stopped by hand after its 8 records, on
   2026-11-21 at 23:30, because the user asked to stop after one block. The records are in
   `speed-block1/`, and the reading is in `speed-block1-preliminary.md`.
2. Blocks 2 to 5 resumed with `abba --start-block 1 --blocks 4`, into the same output folder. The
   plan keeps block numbers and run indexes continuous, so run numbers go from 1 to 40.
3. `speed-pilot/rows.jsonl` and `arm_records.jsonl` hold all 40 runs. `speed-pilot/summary.md`
   is the harness summary over all 40 rows, with 10 runs per arm and window.

Block numbers in the data are 0-based: block 0 is the first block.

## Result

| window | A0 step s | C step s | ratio A0 / C of means | sd of block ratios |
|---|---|---|---|---|
| k=8, S=64 | 69.06 (sd 0.79) | 13.56 (sd 0.09) | 5.09 | 0.03 |
| k=8, S=512 | 80.69 (sd 0.27) | 20.32 (sd 0.06) | 3.97 | 0.01 |

Per block, the mean of the A0 runs minus the mean of the C runs, in seconds, and the ratio of the
block means:

| block | S=64, A0 minus C (s) | S=64 ratio | S=512, A0 minus C (s) | S=512 ratio |
|---|---|---|---|---|
| 0 | 56.57 | 5.15 | 60.64 | 3.98 |
| 1 | 55.34 | 5.08 | 60.55 | 3.97 |
| 2 | 55.33 | 5.08 | 60.15 | 3.97 |
| 3 | 55.14 | 5.08 | 60.30 | 3.97 |
| 4 | 55.09 | 5.06 | 60.20 | 3.96 |
| mean | 55.49 | 5.09 | 60.37 | 3.97 |

## N from the repeats rule

`repeats()` on the pilot gives sigma 0.56 s at S=64 and 0.19 s at S=512, with `blocks_needed = 2`
for both windows. The pilot already has 5 blocks, so `blocks_to_add` is 0 and no top-up runs.

## Checks

- All 40 arm records are `ok`. Suspend, power and foreign_reads are `ok` on every arm.
- The SM clock maximum per arm is between 1950 and 1972 MHz. No arm ran in the capped state.
- The GPU clock row of `speed-rule.md` is evaluated by hand from these maxima. No code evaluates
  its 1000 MHz threshold. The rule itself is unchanged since it was committed before the first arm.

## Reading

- C is faster than A0 in both windows. The ratio is about 5.1 at S=64 and about 4.0 at S=512.
- The block differences vary by at most 1.5 s at S=64 and 0.5 s at S=512, so the spread is much
  smaller than the gap between the arms.
- The pilot does not test the D1 gate. That gate runs on Kanana on Vast.
- The ratio at S=512 is lower than at S=64. This pilot does not explain why.
