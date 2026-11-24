# Ticket 13 phase 2: reducing the `lt-fused` memory premium

Moonlight-16B-A3B revision `2a9a177f4841649f849ba941e41d2ee73c042314`, NVIDIA GeForce RTX 3050
Laptop GPU (sm_86, 4096 MiB), torch 2.14.1+cu130, venv `~/.venvs/soup-rnd-cuda`. Mode **normal**,
default SDPA backend, 3 steps per fresh process (the first is the warm-up), LoRA rank 16 on
`q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj`. Code commit of the runs `91085f28` plus the arena in
`src/soup_cli/utils/a0_exact_lt_ext.cpp`. Every row of `rows_canonical.jsonl` names the model,
card, window `k=8,S=512`, arm (C or E), mode `normal`, torch version, lever and the experiment env.

The phase-1 harness is reused unchanged (`run_mem.py`, `capture_snapshot.py`,
`analyze_snapshots.py`, `emem_analysis.py`); `run_mem2.py` adds a `lever` label and the env to the
canonical row, `run_cell.sh` gates and runs one cell, `gate.py` is the power/clock gate. The gate
reads `/sys/class/power_supply/ACAD/online`, loads the card for 10 s and fails when the SM clock
drops below 1500 MHz or the enforced power limit falls below 50 W. One gate run during the pass
reported `SW Power Cap` at 1912 MHz; `nvidia-smi` then showed the normal 80 W limit (not the 15 W
collapse), and two clean gates at 1740-1950 MHz followed before the pass continued. No memory
number was taken from a low-clock run.

## Result

The premium at (8,512) falls from **+1152.0 MiB reserved / +1154.0 MiB NVML** to **+194.0 MiB
reserved / +196.0 MiB NVML** with one shared scratch arena in the extension plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. E takes 2334.0 reserved and 2482.0 NVML
against C's 2140.0 and 2286.0 under the same flag, so E leaves 1614 MiB of the 4096 MiB NVML
free where the phase-1 baseline left 638 MiB. The premium is now live working set, not trapped
free space. Speed does not regress: one ABBA block puts E at 27.72 s per step against C's 26.82 s
(+3.3 %), while the phase-1 baseline E was +4.5 % against C.

## Premium after each lever

Peaks of the whole run, MiB. C is the ordered-backward arm (no fused staging); a lever that
changes the allocator is measured against C under the same allocator config. "Premium" is E
minus that C.

| lever | arm | allocated | reserved | NVML | premium reserved | verdict |
|---|---|---|---|---|---|---|
| phase-1 baseline | C | 2066.2 | 2158.0 | 2304.0 | - | reference |
| phase-1 baseline | E | 2204.0 | 3310.0 | 3458.0 | **+1152.0** | reference |
| 1. order: arena at load, before weights | E | 2122.0 | 2624.0 | 2772.0 | +466.0 | no change (lazy arena is the same 2624.0; with the gather slots left out the early arena was +8.0 worse) |
| 2. reuse: one arena, all 9 slots, lazy | E | 2122.0 | 2624.0 | 2772.0 | **+466.0** | **-686.0**; the win |
| 2b. arena without the two gather slots | E | 2155.5 | 3054.0 | 3202.0 | +896.0 | worse: the gather buffers fall back to the old pool and pin 320 MiB segments |
| 3. MemPool staging-only | - | - | - | - | - | not built: after the arena staging is one 167.2 MiB live block and no fragmented staging segment is left to isolate; a pool keeps the same live bytes and adds a second allocator. The class delta of the snapshot is all live |
| 4. `max_split_size_mb:20` | E | 2141.3 | 3628.0 | 3766.0 | +1470.0 | worse by +1004.0 |
| 4. `max_split_size_mb:32` | E | 2138.5 | 3584.0 | 3732.0 | +1426.0 | worse by +960.0 |
| 5. `CUBLASLT_WORKSPACE_SIZE=1048576` | E | 2122.0 | 2624.0 | 2772.0 | +466.0 | 0.0: the retained 2 x 8.125 MiB are the extension's own `shared_workspace`, not torch's cuBLAS handle workspace |
| 6. `expandable_segments:True` alone | E | 2168.1 | 3214.0 | 3362.0 | +1056.0 | -96.0 only; not enough without the arena |
| 6. arena + `expandable_segments:True` | C | 2042.4 | 2140.0 | 2286.0 | - | same flag for the reference |
| 6. arena + `expandable_segments:True` | E | 2087.5 | 2334.0 | 2482.0 | **+194.0** | **best combined; -958.0** |

`max_split_size_mb` values below 20 are refused by torch (`CachingAllocator option
max_split_size_mb too small, must be >= 20`). The sweep is closed: splitting is not the cause of
the premium, preventing splits opens new segments and grows reserved. The order lever is real but
empty: the arena is a single block, and placing it before the weights changes nothing.

## Where the premium went

Allocator snapshots of one step after the last warm-up, pickles in `raw/snapshots/` (about 220 MB
each, outside git), committed diffs in `emem-diff-*.json` and `.md`.

- Phase-1 source: 265.4 MiB live staging (14 buffers from the nine slot kinds in three thread
  pools, plus two cuBLASLt workspaces) and 1112.9 MiB of free space trapped in 160/320 MiB
  segments that 8-30 MiB live buffers pinned. The arena collapses the 14 buffers of three pools
  into one 167.2 MiB block (the nine slots at the largest visit, 3937 rows; the two gather slots
  take the largest row bytes of a visit, 5632 B, each).
- After the arena, the remaining trap was one 320 MiB segment that held only an 8.125 MiB
  `shared_workspace` (311.9 MiB free). That is the split-large-segment case of the research, but
  `max_split_size_mb` made it worse, not better.
- `expandable_segments:True` on top of the arena leaves two expandable segments and no fully
  cached segments; the snapshot records `expandable_segments: True`. The remaining premium is
  live: `staging` +167.2 MiB (the arena) and `cublaslt_workspace` +16.2 MiB (two 8.125 MiB
  workspaces, one per calling thread); `other` is -5.9 MiB. The "trapped" metric of
  `analyze_snapshots.py` does not apply to expandable segments: their free tail is available, not
  pinned. The flag helps only after the arena; with 14 separate buffers expandable alone still
  cost +1056.0 MiB because the trap was live-block pinning, not slivers.

## Speed, one ABBA block

Same env for both arms (`expandable_segments:True`, arena lazy, all slots), fresh process per
cell, median of the two steps after the warm-up:

| cell | arm | step seconds |
|---|---|---|
| A1 | E | 27.44 |
| B1 | C | 26.54 |
| B2 | C | 27.10 |
| A2 | E | 27.99 |

E mean 27.72 s, C mean 26.82 s, E/C = **+3.34 %**. The phase-1 baseline pair was 26.95 against
25.80 (+4.46 %); the memory variant does not cost speed within this one block.

## Bits

E was already `torch.equal` to A0 at k <= 8 (tickets 03 and 04); the arena changes addresses and
allocation order, not kernels. The new GPU test
`test_one_scratch_arena_serves_forward_and_backward` compares the forward to the ATen chain of one
native GEMM per segment bit for bit, with two row counts in one arena; the whole
`tests/test_a0_exact_lt_kernels_gpu.py` file (36 tests, including the four-thread and side-stream
checks) passes with the arena preallocated. The bits were not re-measured as a full step ladder.

## Replication

```bash
# best combined variant, one cell (the script gates first and refuses a bad card):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
A0_SCRATCH_PREALLOC_ROWS=3937 A0_SCRATCH_PREALLOC_HID=2048 A0_SCRATCH_PREALLOC_INTER=1408 \
A0_SCRATCH_WHEN=lazy A0_SCRATCH_PREALLOC_GATHER=1 \
benchmarks/mbr-moonlight-emem2-20261124/run_cell.sh --lever best --arm E --k 8 --seq 512

# snapshot one step and compare it with the committed C snapshot:
VENV=~/.venvs/soup-rnd-cuda python benchmarks/mbr-moonlight-emem-20261123/capture_snapshot.py \
  --source ~/moonlight-local/source --cache ~/moonlight-local/cache \
  --out benchmarks/mbr-moonlight-emem2-20261124/raw/snapshots --arm E --k 8 --seq 512 --steps 2
python benchmarks/mbr-moonlight-emem-20261123/analyze_snapshots.py \
  --c .../snapshot-C-k8-s512.pickle --e .../snapshot-E-k8-s512.pickle \
  --json emem-diff-best.json --markdown emem-diff-best.md
```

## Not done here

- MemPool staging-only was not built, for the reason in the table; it cannot remove live bytes.
- Bucketing slots by visit size is what the preallocated arena already does at its limit: it sizes
  every slot once to the largest visit and never grows. Rounding up to a bucket adds bytes; the
  measured 3937-row arena is the smallest one that serves every visit of this window.
- `expandable_segments` is an opt-in launch option, not a default. The arena itself is in `src`
  and activates only when `A0_SCRATCH_PREALLOC_ROWS` (with HID and INTER) is set; without it the
  phase-1 per-slot pool runs unchanged.
- Speed is one ABBA block on a laptop card; no temperature-controlled sweep.
