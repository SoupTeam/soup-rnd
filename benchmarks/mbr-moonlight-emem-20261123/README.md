# Ticket 13: the memory premium of E (`lt-fused`) over C

Moonlight-16B-A3B revision `2a9a177f4841649f849ba941e41d2ee73c042314`, NVIDIA GeForce RTX 3050
Laptop GPU (sm_86, 4096 MiB), torch 2.14.1+cu130, venv `~/.venvs/soup-rnd-cuda`. Mode **normal**,
default SDPA backend, 3 steps per run (first step is the warm-up), one fresh process per cell, AC on
mains (`/sys/class/power_supply/ACAD/online` = 1) and the whole pass under
`systemd-inhibit --what=sleep`. LoRA targets `q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj`, rank 16.
Repo commit of the runs `8abae417`.

Result: the premium is real and reproducible, it is the fused path's retained working set plus the
allocator free space that working set pins, and it is **not** the `73d72016` class of retention.
No code change. E stays opt-in for cards with headroom. At (8,512) E needs 3458 of 4096 MiB NVML
against 2304 for C.

Bits are not re-measured here: E is already `torch.equal` to A0 at k <= 8 and differs from C at
k = 64 by design (tickets 03 and 04). This ticket measures memory and holdings only.

## Memory, C against E (`run_mem.sh`, `rows.md`, `rows_canonical.jsonl`)

Peaks of the whole run. The ticket asks for the ordinary mode, so these are normal-mode numbers on
the default SDPA backend; they are **not** comparable with the deterministic math-backend table of
ticket 03 (different mode and backend give different absolute peaks).

| window | arm | allocated MiB | reserved MiB | NVML MiB | RAM MiB |
|---|---|---|---|---|---|
| (8,64) | C | 954.6 | 1146.0 | 1292.0 | 4682.4 |
| (8,64) | E | 994.6 | 1144.0 | 1290.0 | 4719.1 |
| (8,512) | C | 2066.2 | 2158.0 | 2304.0 | 5063.8 |
| (8,512) | E | 2204.0 | 3310.0 | 3458.0 | 5112.2 |
| (64,64) | C | 1725.8 | 1932.0 | 2078.0 | 5086.0 |
| (64,64) | E | 1865.3 | 2102.0 | 2250.0 | 5130.0 |

Deltas, E minus C:

| window | allocated | reserved | NVML |
|---|---|---|---|
| (8,64) | +40.0 | -2.0 | -2.0 |
| (8,512) | +137.8 | **+1152.0** | **+1154.0** |
| (64,64) | +139.5 | +170.0 | +172.0 |

Two independent passes agree byte for byte on all six cells (`raw/first-pass-rows/` is the first).
The normal mode at S=512 is not bit-deterministic (ticket 12), but the allocator peaks reproduced
exactly.

The live part of the delta is the same at (8,512) and (64,64): +137.8 and +139.5 MiB allocated.
What changes by window is the reserved part: zero at S=64 and +1152 MiB at S=512.

## Allocator snapshots of one step, C against E at (8,512)

`capture_snapshot.py` runs one C and one E process with `torch.cuda.memory._record_memory_history(
enabled="all", context="all", stacks="all", max_entries=1000000)` from before the model build, writes
`_dump_snapshot` inside the last step and only then runs the copy probe, so the pickle is the state
after the step and before anything else touches the allocator. `analyze_snapshots.py` compares the
two pickles; the committed output is `emem-diff-k8-s512.json` and `.md`, the pickles stay in
`raw/snapshots/` (about 230 MB each, outside git).

| arm | live MiB | reserved MiB | trapped in pinned segments MiB | fully cached (releasable) MiB |
|---|---|---|---|---|
| C | 845.8 | 2158.0 | 40.2 | 1272.0 |
| E | 1110.8 | 3310.0 | 1153.2 | 1046.0 |
| delta | +265.1 | +1152.0 | **+1112.9** | -226.0 |

"Trapped" is free space inside a segment that also holds a live block, so `empty_cache()` cannot
release it; "releasable" is free space in segments with no live block. 845.8 + 40.2 + 1272.0 =
2158.0 and 1110.8 + 1153.2 + 1046.0 = 3310.0, and the deltas add up: +265.1 + 1112.9 - 226.0 =
+1152.0.

The copy probe in the same capture: C allocates its 1312 MiB of free reserved space and fills it;
E cannot allocate its 2.15 GiB of free reserved space (CUDA OOM, driver reports 1.33 GiB free). The
extra reserved memory of E is therefore not usable memory, it is mapping that cannot be packed.

### The 265.1 MiB that are live

Sixteen live buffers, all allocated through the extension's `scratch()` or its cuBLASLt workspace
pool, each stack ending at `a0_exact_lt_ext.cpp` and called from `a0_exact_lt.py:310`
(`expert_forward`) or `:314` (`expert_backward`):

| requested bytes | MiB | source | slot decode at `requested_size` |
|---|---|---|---|
| 8519680 | 8.12 | cuBLASLt workspace | 8.125 MiB workspace |
| 8519680 | 8.12 | cuBLASLt workspace | 8.125 MiB workspace |
| 11086592 | 10.57 | `scratch()` | kA/gradA, rows x 2816, rows = 3937 |
| 11086592 | 10.57 | `scratch()` | kA/gradA, rows x 2816, rows = 3937 |
| 11086592 | 10.57 | `scratch()` | kA/gradA, rows x 2816, rows = 3937 |
| 15507456 | 14.79 | `scratch()` | kY/gradY, rows x 4096, rows = 3786 |
| 16125952 | 15.38 | `scratch()` | kY/gradY, rows x 4096, rows = 3937 |
| 16125952 | 15.38 | `scratch()` | kY/gradY, rows x 4096, rows = 3937 |
| 16125952 | 15.38 | `scratch()` | kY/gradY, rows x 4096, rows = 3937 |
| 21322752 | 20.33 | `scratch()` | kH/gradH, rows x 5632, rows = 3786 |
| 21322752 | 20.33 | `scratch()` | kH/gradH, rows x 5632, rows = 3786 |
| 21322752 | 20.33 | `scratch()` | kH/gradH, rows x 5632, rows = 3786 |
| 22173184 | 21.15 | `scratch()` | kH/gradH, rows x 5632, rows = 3937 |
| 22173184 | 21.15 | `scratch()` | kH/gradH, rows x 5632, rows = 3937 |
| 22173184 | 21.15 | `scratch()` | kH/gradH, rows x 5632, rows = 3937 |
| 32251904 | 30.76 | `scratch()` | kProd, rows x 8192, rows = 3937 |

264.1 MiB requested, 265.1 MiB allocated (block rounding). The sizes are exact multiples of the
slot strides in `a0_exact_lt_kernels.cu`: `kH`/`kGradH` rows x 2 x inter x 2 = rows x 5632,
`kA`/`kGradA` rows x inter x 2 = rows x 2816, `kY`/`kGradY` rows x hid x 2 = rows x 4096,
`kProd` rows x hid x 4 = rows x 8192, `kGatherIn`/`kGatherOut` are the x 4096 / x 5632 cases;
inter = 1408, hid = 2048. The largest expert visit at (8,512) routes 3937 of the 4096 tokens to one
expert, and three (thread, stream) pools each keep their buffer for the largest count they saw
(3937 or 3786). The pool is deliberate: `scratch()` keeps each slot per thread, device and stream
(`a0_exact_lt_ext.cpp`, the comment above `enum Slot`) and reuses it call after call.

### The 886.6 MiB that are not live

E holds 1112.9 MiB more trapped free space than C, and 226.0 MiB less releasable. The big items:

| segment MiB | live MiB | free MiB | live block classes |
|---|---|---|---|
| 320 | 21.1 | 298.9 | fused_scratch |
| 320 | 40.7 | 279.3 | fused_scratch |
| 320 | 44.7 | 275.3 | fused_scratch |
| 160 | 55.2 | 104.8 | fused_scratch, other |
| 160 | 67.3 | 92.7 | fused_scratch |

Every large pinned segment of E keeps only one to four live scratch buffers (8.1 to 30.8 MiB) and
leaves the rest of the 160 or 320 MiB segment unusable. C's trapped space is 40.2 MiB, mostly in
20 MiB and 2 MiB segments; none of it is a 160 or 320 MiB block. By trapped bytes per holder, E
minus C is +1104.8 MiB for `fused_scratch` and +112.9 MiB for `other` (a segment can list more than
one holder, so the two sums exceed the 1112.9 MiB delta). This is the main source of the premium.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` does not remove it: one extra run of each arm in
the same window gives C 2140.0 MiB reserved and E 3214.0 MiB, still +1074.0 MiB.

The window dependence fits: the live scratch is the same at (64,64) (allocated delta +139.5 against
+137.8 MiB at (8,512)), but the trap appears only at S=512. The peaks say the same scratch does not
raise reserved at S=64 (0 and +170 MiB), so the trap belongs to the S=512 segment layout; the
S=64 allocator state was not snapshotted, so the layout itself is not claimed here.

## Candidates, each checked by name

| candidate | found | size in the snapshot | verdict |
|---|---|---|---|
| cuBLASLt workspace | yes, but small | 2 x 8.125 MiB retained in E (16.25 MiB); the 8.125 MiB per call inside the algo search is transient | minor; not the premium |
| segment staging buffers | yes, inside the same scratch pool | gather slots are the rows x 4096 and rows x 5632 cases, up to 21.2 MiB | part of the 265.1 MiB live set, not a separate retention |
| per-row-count input retention (`73d72016`) | no | none; `FusedExpert.ensure` drops its samples after the search (`a0_exact_lt.py:359-367`) and the snapshot has no tensor per row count, only host-side descriptors and 8-int algo records | not present |

## Verdict

- **The premium is a fundamental price of the fused path, not a retention bug.** At (8,512) E costs
  +1152.0 MiB reserved and +1154.0 MiB NVML over C. +265.1 MiB are live scratch buffers that the
  kernels need at the peak; the other +886.6 MiB are free space that those buffers pin in large
  segments and that `empty_cache()` cannot release. The `73d72016` class, the cuBLASLt workspace and
  the staging buffers are checked and are not it.
- **E stays opt-in for cards with headroom.** On 4 GiB at (8,512) E leaves 638 MiB of NVML free
  against C's 1792 MiB; at (8,64) the premium is zero and at (64,64) it is +172 MiB, so the price is
  paid only at the long-sequence window.
- **No fix is applied.** An untested candidate would be one arena per (thread, stream) for the
  fourteen scratch slots instead of separate tensors, which should collapse the pinned segments; it
  needs a CUDA kernel test and a full remeasure before any claim, so it is not recorded as a fix.
- The recommendation for `lt-fused` in `docs/expert-forward.md` is updated with these numbers; it
  stays opt-in.

## Replication

```bash
# Six memory cells, AC-gated and under systemd-inhibit:
benchmarks/mbr-moonlight-emem-20261123/run_mem.sh
# Snapshot one step of each arm at (8,512); pickles go to raw/snapshots:
VENV=~/.venvs/soup-rnd-cuda python benchmarks/mbr-moonlight-emem-20261123/capture_snapshot.py \
  --source ~/moonlight-local/source --cache ~/moonlight-local/cache \
  --out benchmarks/mbr-moonlight-emem-20261123/raw/snapshots --arm E --k 8 --seq 512 --steps 2
# Compare them:
python benchmarks/mbr-moonlight-emem-20261123/analyze_snapshots.py \
  --c .../snapshot-C-k8-s512.pickle --e .../snapshot-E-k8-s512.pickle \
  --json emem-diff-k8-s512.json --markdown emem-diff-k8-s512.md
```

Tests of the CPU toolkit: `pytest tests/test_emem_analysis.py tests/test_emem_scripts.py` (44
tests), `ruff check` on the same files. The full suite was not run (local rule).

## Not done here

- Bits and speed: not re-measured; ticket 03 settled the bits, and this ticket measures memory only.
- Deterministic mode and the math backend: deliberately not used, the ticket asks for the ordinary
  mode. Absolute peaks therefore differ from ticket 03's deterministic math-backend table.
- Snapshots of (8,64) and (64,64): the ticket asks the one step comparison at (8,512); the other two
  windows are covered by peaks only.
- No attempt to make the trap smaller by tuning `PYTORCH_CUDA_ALLOC_CONF`; only the
  `expandable_segments` default alternative was measured once and did not help.
