# Ticket 15: the price of the segmented expert GEMM, phase profile

**Status: instruments built and verified; the measurement pass is blocked on the card's power
state.** `run_profile.sh` runs the whole campaign in one go; it has not produced its rows yet.
What follows separates what is measured (pilot, valid clocks) from what is waiting on the card.

## Answer so far, from the pilot (valid)

Pilot window 2026-11-24 18:46–19:05, before the driver lost its power budget (see Blocked):
Moonlight-16B-A3B (`2a9a177f`), RTX 3050 Laptop (sm_86, 4 GiB), torch 2.14.1+cu130, LoRA targets
`q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj` (rank 16), mode speed (normal), SM clock 1942–2100 MHz.
Window (8,64), k=8, a three-step run per arm; the profiled step is traced with CUDA activities
only.

| arm | step s | expert calls/step | kernel s | busy s | exposed s | launches |
|---|---|---|---|---|---|---|
| C `native` | 21.49 | 6 863 | 2.41 | 4.54 | 17.21 | 158 681 |
| S `segmented` | 23.80 | 34 979 | 4.90 | 7.03 | 18.08 | 265 149 |

- **The price is in the kernels, not in the Python dispatch.** S's step is +2.31 s. Its device-busy
  time grows by **+2.49 s** while its time outside the kernels (Python dispatch, plan lookups,
  `split`/`cat`, expert streaming) **shrinks by 0.17 s**. `expert_gemm.py` never appears as a
  measurable phase.
- **Cost is per call, and the per-call cost is the same in both arms.** The `M<=16` cuBLAS kernel
  carries 2 092 calls / 0.128 s in C and 19 732 calls / 1.175 s in S — **61 µs against 60 µs per
  call**. S simply makes 5.1× more of them at k=8 (`M=1` calls alone: 8 120 of 34 979, at 50 µs
  each in the `gemv2T` kernel). The dequantisation and elementwise kernels are flat between arms
  (0.503 against 0.504 s, 0.075 against 0.076 s).
- **Per row, not per call, is where S loses.** 60 µs per call is 3.8 µs/row at M=16 and 10 µs/row
  at M=6, while C's grouped call at M≈50 pays the same 61 µs for 1.2 µs/row. Each call streams the
  expert's whole weight matrix again, so µs/row falls as `1/M` up to the tile width and then
  stops: the cuBLAS tile pads every call to its M bucket.
- **Kernel identity: S is A0's shapes, not "M<=16".** S's row counts run 1…63 at (8,64) — one
  micro-batch's rows never exceed 64 — with `M=1` at 24% of the calls. So the kernel set is A0's
  full set (gemv for M=1, `64x6` for M<=16, `64x5` for 17–31, `128x64`/cutlass for 40–64), not the
  single `M<=16` form the ticket expected. The bit gate of ticket 14 is what makes that set
  correct; the "differs from expectation" is in the shape of the set, not in its bits.
- **k=64 scales the same way.** The counters at (64,64) are clock-independent and were taken in
  the pilot window: C 7 499 expert calls per step, S **269 367** (35.9×), step 36.63 s against
  59.35 s. The k=8 trace priced one extra call at ~55 µs of device time; 262k extra calls at that
  price is the +22.7 s. The whole-step trace at (64,64) is not available on this host (see
  Instruments).
- **The ticket's "2.87x" is a slip.** `S/C` at (64,64) is 58.87/35.67 = **1.65×**; 65.05% is
  `(S-C)/C`, and 2.87 is its reciprocal (the fraction of C's time S keeps). The step is 1.65 times
  as long, not 2.87.

## Blocked (2026-11-24 19:24)

The card is pinned at **210 MHz, P8, power limit 15 W → N/A**, `SW Power Cap` and `SW Thermal
Slowdown` active; a synthetic 4096³ bf16 GEMM measures 1.68 TFLOPS instead of ~12, and a (8,64) C
step takes 58 s instead of 21.5 s. The driver log names the cause at the moment a whole-step trace
of (64,64) was OOM-killed:

```
19:24:09 NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x51)
         returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:1336
19:24:09 NVRM: RmHandleDNotifierEvent: Failed to handle ACPI D-Notifier event, status=0x11
19:24:08 nvidia-powerd: Client (presumably SBIOS) has requested to disable Dynamic Boost DC controller
19:24:08 nvidia-powerd: CRITICAL! Failed to get SysPwrLimitGetInfo
```

The power controller never got its budget back, so no speed number taken after 19:24 is comparable
with `benchmarks/mbr-moonlight-seg-speed-20261124/`. Restoring it needs root:

```bash
sudo nvidia-smi -pl 80            # if that is refused:
sudo systemctl restart nvidia-powerd
# if the driver keeps the state: a reboot
```

Check before re-running the campaign: `nvidia-smi --query-gpu=clocks.sm --format=csv` under load
must read ~1900 MHz, and the 4096³ bf16 GEMM must beat 10 TFLOPS. `run_profile.sh` itself only
checks AC power; the clock check is manual.

## What is built

- `sweep_rows.py` — the per-M cost and kernel identity of the two expert projections, isolated from
  the model: every row count 1…512 plus 640, `--reps` calls after `--warmup`, one CUDA-only
  profile per point. Writes `sweep.json` (every M, the prices the model uses) and `sweep.md` (the
  bucket edges).
- `profile_step.py` — one normal-mode run of one cell in one process under `ArmWatch`
  (`suspend`, `power`, `foreign_reads`), appending the row to `<out>/runs/rows.jsonl` and, with
  `--trace`, the traced step to `<out>/trace/`.
- `run_profile.sh` — the pass: sweep, then `--count` runs of the four cells, then traced steps of
  (8,64), then py-spy snapshots of all four, then `analyze.py`. Refuses to run over existing rows,
  off AC, or while another campaign holds the lock; `systemd-inhibit --what=sleep` around
  everything.
- `analyze.py` — `analysis.json`, `layout.md` (the C/S row per window, the M histograms, the sweep
  price with each arm's calls, the py-spy shares, the attribution verdict) and `rows.md` (one line
  per cell naming model, card, window, arm, mode and torch).

Harness side: `--count` (a `torch.nn.functional.linear` counter of the routed-expert calls, exact
and memory-flat) and `--trace` (a CUDA-only profiled step) in `benchmarks/harness/mbr_measure.py`,
the summarizers in `benchmarks/harness/mbr_trace.py`, the counter in
`benchmarks/harness/mbr_count.py`.

## Instruments that were rejected, and why

- **A CPU-activity `torch.profiler` trace of a whole step** (the ticket 05 method) does not fit this
  14 GiB host: 7.9 GiB RSS at (8,64) and OOM-killed; a CUDA-only profile of the same step is
  5.6 GiB. Consequence: the expert-GEMM counters come from the `F.linear` wrapper, not from
  `aten::mm` shapes, and kernel identity comes from the sweep's own profiles plus the traced
  kernel list.
- **A whole-step CUDA-only trace at (64,64)** was OOM-killed at about 1.1M kernel events — that is
  the kill that took the driver's power state down. (64,64) is therefore measured by counters (clock
  independent), the isolated per-M sweep and py-spy, and `analyze.py` marks the cell
  `measured: false`.
- **`nvidia-smi` polling as an in-process GPU-utilisation sampler**: one query blocks for seconds
  while the card is loaded, which inflated a (8,64) step from 24 s to 115 s and pushed the card
  into the GPU-bound regime it was supposed to measure. Removed.

## Reproduce

```bash
bash benchmarks/mbr-moonlight-segprof-20261124/run_profile.sh   # ~40 min, one card
```

Raw rows, traces, profiles and the sweep stay in this folder; logs are git-ignored. `pilot/` holds
the two valid traced steps quoted above (18:46 and 19:00), produced with the same harness calls
before the driver wedge; they are evidence for the phase split, not campaign rows.
