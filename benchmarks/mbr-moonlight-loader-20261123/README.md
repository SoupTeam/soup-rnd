# Ticket 05, part 2: `cpp_async_2slot` against `async_lease`

Moonlight-16B-A3B, RTX 3050 Laptop (sm_86, 4 GiB VRAM, 14 GiB RAM), torch 2.14.1+cu130,
venv `~/.venvs/soup-rnd-cuda`, extension built from the vendored #77 source with
`--with-cuda-nf4`. Bit and memory rows are deterministic; speed rows are normal mode.

Result: the bits hold and the speed does not. `async_lease` stays the default.

## Bits (`run_bits_h.sh`)

`torch.equal` on the saved loss and every LoRA gradient, H against the reference arm of its k
(A0 at k <= 8, C above it).

| window | arm | k | verdict |
|---|---|---|---|
| (1,64) | H vs A0 | 1 | equal |
| (2,64) | H vs A0 | 2 | equal |
| (4,64) | H vs A0 | 4 | equal |
| (8,64) | H vs A0 | 8 | equal |
| (8,512) math | H vs A0 | 8 | equal |
| (64,64) | H vs C | 64 | differs, recorded, not a gate |

The kill ladder of the ticket is k = 1, 2, 4, 8: equal everywhere. The k = 64 cell differs from
C and the ticket records it without blocking. The two C repeats and the H repeat
(`k64-counter-*`) are the counter runs.

What a bits verdict may read, as in the decoupled campaign: a row whose `outcome` is not `ok` or
is missing reads as `void`; with `arm_records.jsonl`, row N must bind to record N by arm, `k`,
`seq` and `run`. The verdict and the diff detail read the same row, the last one per (k, arm).

## py-spy (`run_pyspy.sh`)

Profile of one k = 8, S = 64 step, `--native`, on both arms. The Python controller leaf
(`CppPackedStream.read`, `PackedStreamPlan.entry_for`) is 1.29 s of a 285 s profiled step:
**0.45%**. The extension holds 17.4% of the step. The >10% condition for moving the controller
into C++ is not met, so no C++ controller.

## Speed (`run_speed_h.sh`)

G-H-H-G, the two-arm A-B-B-A of `METHOD.md`, five blocks over the three windows that fit, one
fresh process per run, `ArmWatch` with suspend, power and foreign reads required, the whole pass
under `systemd-inhibit sleep`. A block with a void run is not counted and runs again under its own
block number (`speed/excluded_blocks.json`).

| window | G (async_lease) step s | H (cpp_async_2slot) step s | paired gain | per-block gains % | t (5 pairs) |
|---|---|---|---|---|---|
| (8,64) | 18.19 | 17.84 | **+1.90%** | +2.04, -0.83, +0.88, +1.25, +6.24 | 1.62 |
| (8,512) | 23.00 | 23.47 | **-2.02%** | -5.17, -4.57, -6.37, +1.94, +4.09 | -0.96 |
| (64,64) | 33.65 | 33.92 | **-0.81%** | -4.46, -1.50, +3.51, +1.13, -2.72 | -0.57 |

The gate wants at least 3% and t > 2.78 in at least two of the three windows. **0 of 3.** The
window (8,64) is the only one pointing up, at 1.90% its t is 1.62, and its largest block gain
comes last in the campaign.

Memory, peaks per window, G and H identical to three digits:

| window | allocated GiB | reserved GiB | NVML GiB | RAM GiB |
|---|---|---|---|---|
| (8,64) | 0.97 | 1.12 | 1.26 | 4.58 / 4.59 |
| (8,512) | 2.15 | 3.23 | 3.38 | 4.97 / 4.98 |
| (64,64) | 1.82 | 2.05 | 2.20 | 4.99 / 5.00 |

### Validity

68 rows, 66 ok, 2 void. Checks: suspend ok on every run, GPU clock maximum at least 1942 MHz on
every run, one power void (2 of 80 samples off mains) and one foreign-read void (1.30 GB read by
another process on the model disk, the same process class that ran this campaign).

Block 0 of (64,64) is excluded and replaced: its first pass voided one G run on power, the
full re-run voided one G run on foreign reads, and the replacement pass is block 5. Excluding or
counting that block changes no window's sign or significance.

### Why the gain is not the ring's

The two-slot ring does not overlap the read: one expert read on this disk (about 3 ms) is longer
than the compute it could hide behind, so the ring saves a stall only when the read lands early.
Nothing is credited to the ring here: there is no gain to credit.

## Result lines

`rows.md` holds one line per measured cell with model, card, window, arm, mode and torch;
`analysis.json` holds the verdicts, the gate and the memory cells.

## Not done here

- The full test suite (only the ticket's own files and the loader tests were run).
- Vast. The local kill acts as the main one for this lever (D13).
