# Moonlight speed, A0 against C: first block only (preliminary)

This is one block of the ABBA design in `speed-rule.md`, A0 B C C A0 per window, stopped by hand
after block 1. It gives a ratio and no spread. It sets no verdict. The pilot of five blocks and
the N from `repeats()` are still to run.

Setup: RTX 3050 Laptop (sm_86), Moonlight-16B-A3B, revision 2a9a177f, mains power, one process per
run, `ArmWatch` with suspend, power and foreign_reads required. Records: `arm_records.jsonl`, rows:
`rows.jsonl`. The rows and records come from `results/speed-abba/` in the local run.

## Result

| window | A0 step s (2 runs) | C step s (2 runs) | ratio A0 / C of means |
|---|---|---|---|
| k=8, S=64 | 69.3, 71.1 (mean 70.2) | 13.5, 13.8 (mean 13.6) | 5.15 |
| k=8, S=512 | 80.7, 81.2 (mean 81.0) | 20.4, 20.3 (mean 20.3) | 3.98 |

Each step value is the median of the timed steps of one run, as the harness writes it.

## Checks

- All 8 arm records have outcome `ok`. Suspend, power and foreign_reads are `ok` on each.
- The SM clock maximum per arm is between 1950 and 1965 MHz. The GPU ran at full clock during
  every arm. The 210 MHz capped state did not appear.

## Limits

- One block, so there is no spread and no N. The ratios are point estimates from one A0 and one C
  pair per window.
- The ratio is 5.15 at S=64 and 3.98 at S=512. The gap between the windows is not explained here.
- The pilot stopped before the N computation. Five blocks are needed for a verdict.
