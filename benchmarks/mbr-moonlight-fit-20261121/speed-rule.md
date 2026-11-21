# Speed rule: A0 against C on Moonlight-16B-A3B, RTX 3050 Laptop

Written and committed before the first speed arm. Do not edit after the first arm.

## Question

How much faster is C than A0 in normal mode, on the windows (8,64) and (8,512)?
Only the ratio is reported. This ticket sets no pass or fail gate; the D1 gate runs on Kanana
on Vast.

## Design

- Arms: A0 in the A role, C in the B role.
- Order: blocks of A, B, B, A, each run in a fresh process. `arm_validity.block_order` gives it.
- Windows: (8,64) and (8,512). The window (64,64) is left out, because A0 is not planned above k=8.
- Number of a run: the median of its timed steps, as `mbr_measure.py` writes it.
- Pilot: 5 blocks. Then N from `repeats()`, and blocks up to N, as `METHOD.md` says. N is computed
  once from the pilot, once more from all blocks, and then the run stops.
- With two windows, N is the larger of the two window values.

## Checks

Every run is wrapped in `ArmWatch` with `required = (suspend, power, foreign_reads)`. The label is
`RTX 3050 Laptop, real`. The model folder is `source/`, the disk under test.

## Validity rows

| validity row | check | required | if it fails |
|---|---|---|---|
| `suspend` is ok on every arm the verdict reads | `suspend` | yes | no verdict for the window |
| `power` is ok on every arm the verdict reads | `power` | yes | no verdict for the window |
| `foreign_reads` is ok on every arm the verdict reads | `foreign_reads` | yes | no verdict for the window |
| `gpu_clock.sm_clock_mhz.max` is at least 1000 MHz on every arm the verdict reads | `gpu_clock` | yes | no verdict for the window |

Why 1000 MHz. The capped state from `METHOD.md` holds the SM clock at 210 MHz. A healthy load on
this card measured 1927 to 1965 MHz. The threshold sits between them.

Why the maximum and not the median. The clock record covers the whole arm, model load included.
The GPU is idle during the load, so the median of a healthy arm can be low. The maximum shows that
the GPU ran at full clock during the arm.

## Reporting

For each window:

1. Per block: the mean of the A0 runs minus the mean of the C runs.
2. Over the blocks: the mean of those differences, and the ratio A0 over C of the means.
3. The value of N and `blocks_to_add` after the pilot and after the top-up.
4. Each arm record, void ones included, in `arm_records.jsonl`.

A block with a void run is not counted. It is replaced, as `METHOD.md` says. A window where a
validity row fails on any arm it reads gets no verdict, and the report says so.

## Manual check

Ran on this laptop before the pilot. Records are in `manual-check/`.

| step | arm | outcome | what it shows |
|---|---|---|---|
| 1 clean | clean | ok | nothing voids a quiet arm |
| 2 sleep | sleep | void | `suspend` catches a 36.3 s boot-time jump |
| 3 battery | battery | void | `power` catches 16 of 32 samples off mains |
| 4 foreign read | foreign-read | void | `foreign_reads` catches 2.02 GB of foreign reads |
