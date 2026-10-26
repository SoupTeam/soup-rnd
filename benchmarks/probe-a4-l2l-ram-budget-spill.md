# Probe A4 - L2L activations under a host-RAM budget, spilled to a drive past it

**Status: protocol v3 committed before its GPU session; no v3 verdict yet.** v1 (the rule as
first written) judged one RTX 3090 session on 2026-10-25: `SPILL NOT SAFE` on G2 alone (§9).
Protocol v2 (§3.1: G2 bounds only what spill adds, 24-block cap, a machine admission) was
committed at `efdd1a76` before any v2 arm; none of three rented machines passed its
admission and no v2 session ran (§9). The v2 admission was wrongly specified (§3.2): no
pause, and a criterion relative to the volume's own median; the refusals say nothing about
the platform. Protocol v3 replaces the admission with a schedule at twice v1's spill demand,
runs only on the GPU class that demand was measured on (RTX 3090 / 3090 Ti), and keeps rule
v2 (§3.2). v1's raw records are kept unchanged and its verdict stays its own.

Track A, research question 4 of the program brief (`rndTASK`, Track A): "Host-RAM
budget as a planner term: boundary activations compete with pinned staging and arenas
(#901 rounding) and, on 40-70B stores, with pinned residency (Track C). Who gets the
RAM, by what rule?" Trello card [A4] "Cap the RAM and spill intermediate results to
SSD" (P1, depends on A1, coordinate with C3). Prior art: L2L is Pudipeddi et al. 2020
(arXiv 2002.05645); FlexGen 2023 (arXiv 2303.06865) offloads activations the same
way for inference. Nothing here is claimed as new.

Harness (benchmark code, not shipped): `harness/a4_activation_budget.py` (pure planner),
`harness/a4_spill_store.py` (the store that obeys a plan), `harness/a4_box.py` (spill
volume preflight, cgroup limit and swap, RSS), `harness/a4_admission.py` (the v3 machine
admission, §3.2), `harness/a4_probe.py` (one process per arm), `harness/a4_session.py`
(A-B-B-A blocks), `harness/a4_rule.py` (verdict), `harness/a4_g4_compose.py` (G4's
compressed store planned by A4 and spilled through the real unbuffered file, §2). Reused
unchanged: the shipped `soup_cli.utils.l2l_step` / `l2l_activations` (through the harness
views `l2l_schedule.py` / `l2l_activations.py`), `stream_probe.py`, `arm_validity.py`,
`l2l_session.py`. No file in `src/soup_cli/` changes.

## 1. What exists, and what was missing

What the shipped runtime and the L2L harness already account for:

- **Weight source, RAM tier.** `RamSource` packs the store into power-of-two pinned
  arenas (`plan_pinned_arenas`, `src/soup_cli/utils/layer_stream_runtime.py:556`,
  #901); `pinned_bytes` is what is really page-locked, `nbytes` the tensors' own
  bytes. `StreamRuntime.stats()["pinned_bytes"]` (`layer_stream_runtime.py:2966`)
  reports it.
- **Weight source, disk tier.** `AsyncDiskSource._allocate_staging`
  (`src/soup_cli/utils/async_disk_source.py:673`) packs the read-ahead regions into
  the same arenas and sets `pinned_bytes` (`:702`).
- **RAM-tier decision.** `trainer/stream_setup.py:784-811` and `:856` read
  `free_ram_bytes()` (`utils/layer_stream.py:984`, psutil `available`) against
  `RAM_TIER_HEADROOM` (0.7, `layer_stream.py:54`).
- **#901 recovery.** `release_cached_pinned_memory` (`layer_stream_runtime.py:3351`)
  returns the blocks torch's caching host allocator keeps after a free;
  `drain_stale_cuda_error` (`:3308`) clears the stale error a failed page-lock leaves.
- **L2L harness.** `ActivationStore` (`benchmarks/harness/l2l_activations.py:268`):
  `n_layers x k` chunks, the lowest `spill_layers` through a `DirectSpillFile`, two
  rings of `ring` staging buffers of `chunk + 4096` B (`aligned_buffer`, `:41`, #1531).
  `l2l_probe.spill_layers_for` picks `spill_layers` from a budget counted in
  micro-batches; `fits_in_ram` (V3) checks the NOMINAL `n_layers x k x chunk`
  against free RAM and Windows commit minus 4 GB.

What was missing, and what this track adds:

| Missing | Added |
|---|---|
| The activations' REAL host bytes: power-of-two rounding of every pinned chunk, the rings (an 8 MiB chunk's staging buffer page-locks 16 MiB), and the ring cost appearing only when something spills | `StoreLayout.host_bytes` mirrors `ActivationStore`'s allocations one for one; a test builds the real store and compares bytes (`test_the_layout_matches_what_the_real_store_allocates`) |
| A budget in bytes rather than micro-batches, a priority rule between activations, weight staging/arenas and C3 residency | `plan_activations`, `split_host_ram` (§2) |
| A named refusal when the minimum cannot fit or the drive is short | `BudgetRefusalError` codes `BUDGET_BELOW_MINIMUM`, `RAM_OVER_BUDGET`, `SPILL_DISK_SHORT`, `UNALIGNED_SPILL`, each with the arithmetic |
| A full drive mid-step: the writer thread's `OSError` surfaced as a generic `RuntimeError`, and nothing removed the file | `SpillDiskFullError` (`SPILL_ENOSPC`) with the arithmetic, raised after the file is removed and pinned memory returned |
| Release on Ctrl-C or an exception mid-step; a constructor that fails after opening the spill file (`ActivationStore.__init__` opens it at `:318` before pinning the rings at `:320-321`) left the file behind | `BudgetedStore` closes on every exit, clears the traceback frames that still hold a buffer (the exception leaving the `with` block, a failed writer/reader thread) so their buffers can go, removes a file its opener left behind, releases the host cache; on POSIX the file is unlinked as soon as it is open, so even a cgroup OOM kill leaves no file. Tests: `test_a_ring_that_fails_to_pin_after_the_file_is_open_leaves_nothing`, `test_no_store_buffer_is_reachable_when_the_pinned_cache_is_released` |
| A spill-volume preflight (tmpfs, overlay, network volume, `O_DIRECT`) and the cgroup limit the run is really under (v2 and v1, from the process's own cgroup up, swap included) | `a4_box.spill_volume`, `a4_box.ram_limit` |
| Spill write/read GB/s on the drive and the hidden share of the spill I/O | `TimedSpill`, `summarise` |
| Whether spill writes slow the weight reads that share the drive | `WeightReadMeter` on the shipped `AsyncDiskSource` |

## 2. The planner and the priority rule

**Inputs.** A host-RAM total (the tighter of `--host-total-bytes` and the cgroup limit
the process is under; a declaration cannot raise the cgroup's), `n_layers`, `k`,
`chunk_bytes`, ring depth, pinning, and the other consumers taken from the runtime's own
accounting: the weight source's page-locked bytes are `runtime.stats()["pinned_bytes"]`
(`layer_stream_runtime.py:2966`), not re-derived. The rest of the process
(`a4_box.non_activation_bytes`) is its `RssAnon` plus any `RssShmem` (page-locked,
gate-655) beyond that figure, so a pinned buffer the runtime does not report is still
charged and the weights are not charged twice; off Linux it is RSS minus the weights.
When the process's memory cannot be read, a run with a host total is refused.

**Who gets the RAM, in order:**

1. the reserve (OS, page cache for the shard reads, the margin): never lent out
   (`--reserve-bytes`, default 4 GB, V3's margin);
2. everything the process already holds that is not activations: the weight source's
   arenas or staging as the runtime reports them, and the pageable rest (CUDA context,
   skeleton, adapters, optimizer);
3. the activation minimum: both staging rings plus any per-chunk floor that never
   spills, so that a step at this k can run at all;
4. activation residency, a layer at a time from the TOP, up to all-pinned;
5. what is left is spare: the term C3's pinned-residency planner reads.

Activations rank above C3 weight residency. Both save traffic per byte held, but only
activation traffic WRITES to the drive, and those writes cost endurance and can slow
the weight reads on a shared drive (gate G5 measures this).

**Granularity: the layer, the lowest layers first.** `ActivationStore(spill_layers=)`
is unchanged. The forward stores layer `l`'s input at visit `l`; the backward reads it
at visit `L - 1 - l`. The lowest layers are written first and read last, so they have
the most compute between write and read to hide their I/O behind. A whole layer at a
time wastes at most `k - 1` chunks of the budget against a per-chunk planner; every
plan reports that as `granularity_waste_bytes`.

**The minimum is not zero, and it can exceed all-pinned.** When any layer spills, the
store pins `2 x ring` staging buffers. When all-pinned costs less than that minimum,
the planner keeps everything pinned: a budget below all-pinned is then refused, not
planned into a larger store. Mistral-7B at seq 512 in bf16 has a 4 MiB chunk and an
8 MiB staging buffer (4 MiB + 4096 B, rounded):

| k | all-pinned | minimum, ring 4 | 0.5 budget | 0.25 budget |
|---|---|---|---|---|
| 1 | 128 MiB | 64 MiB | 32 layers spilled, 64 MiB | refused, `BUDGET_BELOW_MINIMUM` |
| 2 | 256 MiB | 64 MiB | 24 spilled, 128 MiB | 32 spilled, 64 MiB |
| 3 | 384 MiB | 64 MiB | 22 spilled, 184 MiB | 30 spilled, 88 MiB |
| 4 | 512 MiB | 64 MiB | 20 spilled, 256 MiB | 28 spilled, 128 MiB |
| 16 | 2048 MiB | 64 MiB | 17 spilled, 1024 MiB | 25 spilled, 512 MiB |

(Computed by `plan_activations`. With the store's default ring of 8, the minimum is
128 MiB and k = 2's half budget would spill everything; the probe uses ring 4.)

**`chunk_bytes` for G4.** The planner never assumes bf16. A store passes the
UNROUNDED bytes of one stored chunk as `chunk_bytes`: what a resident chunk pins and
what the spill file gets per spilled chunk. G4's compressed store passes its payload
there and one chunk's per-row scales as `pinned_floor_bytes`. G4 holds all `n_layers x
k` scale chunks in ONE pinned slab (`g4_activation_codec.py:461`), spilled or not, so
the planner rounds the slab once (`floor_slab`, `n_layers x k x pinned_floor_bytes`), not
each chunk's scales. If anything spills, `chunk_bytes` must be a multiple of 4096 B
(`UNALIGNED_SPILL` otherwise), because the unbuffered file is addressed in whole
sectors; G4's store refuses the same payload for spill. Neither module imports the other.

Worked example (Mistral-7B, seq 512, G4 int8, k = 4, ring 4, pinned;
`test_worked_mistral_7b_int8_example_by_hand`): payload 2,097,152 B, scales 2,048 B per
chunk. Resident payload chunk 2 MiB (a power of two already); ring buffer 2 MiB + 4096 B,
page-locked 4 MiB; scale slab 32 x 4 x 2,048 B = 256 KiB, rounded once. All-pinned
268,697,600 B, minimum 33,816,576 B; the 0.5 budget (134,348,800 B) keeps 11 layers
(8 MiB each), spills 21, holds 126,091,264 B and writes a 176,160,768 B spill file.

**Composition check** (`a4_g4_compose.py`, §9): real G4 sizes for (1, seq, H) shapes in
int8, fp8_e4m3, identity bf16, identity float32 and int4 at odd hidden sizes go into
`plan_activations` unrounded; a real G4 `CodecActivationStore` is built with the plan's
`spill_layers` over `DirectSpillFile`, every (layer, micro-batch) is put in forward order
and got in backward order, and the storages, spill file and bytes moved are compared
with the plan row by row. Rerun by the integration owner against the final G4 code
(`g4_activation_codec.py` sha256 `2403acfb...`, G4 branch commit `a7e5ced7`): 8 rows x 4
budgets, every row equal to the plan (raw output kept locally, not published).

## 3. The decision rule, written before the run

Reference for every comparison: L2L with all activations pinned (budget `1`), in the
same build, at the same k, on the same token rows. Spill is lossless by construction,
so correctness is **bit-exact**, forward and backward stated separately; it is not
quality-gated.

| Gate | Pass | Fail | No verdict |
|---|---|---|---|
| G1 bit-exact | for k = 2, 3, 4 and every planned budget (0.5, 0.25, min): per-micro-batch losses (forward) and nonzero LoRA grads (backward) `torch.equal` to all-pinned L2L AND to GA | any spilled budget differs in either direction | references not reproducible (GA vs GA, pinned vs pinned) even with `--deterministic`; the corrupted-read control did NOT differ; no budget spilled |
| G2 RAM within budget (v2) | every arm: store bytes == plan bytes <= budget (real sizes, rounding and rings included); every B arm's **excess** (CUDA host allocator peak during the timed steps − its baseline − the store's held bytes) <= the excess its block's A arms share; with a declared host total, peak RSS <= total | any arm's store over, any B excess over the reference, RSS over | a counter unavailable; a block whose A arms are missing, void, spill, or whose two excesses differ (that block's B arms) |
| G3 full disk and interruption | Ctrl-C mid-backward, exception mid-forward and ENOSPC mid-step each raise their named error, leave no spill file and no spill thread, and the host allocator returns to its baseline; ENOSPC names `SPILL_ENOSPC` with the arithmetic; no timed arm leaves a file | any of these | a scenario not run |
| G4 cost at k = 2 and 4 | `ram_k2_half` and `ram_k4_half`: mean per-block step-time overhead of spill over all-pinned, two-sided 95% t interval, upper bound < 10% | lower bound >= 10% | the interval straddles 10%, or a validity row fails |
| G5 weight-read interference | `disk_k4_same_drive`: per-block ratio of weight read GB/s (while the reader is busy) with spill writes on the weights' drive over without, 95% t interval, lower bound >= 0.90 | upper bound < 0.90 | straddles 0.90, or a validity row fails |

Validity rows (METHOD.md form):

| validity row | check | required | if it fails |
|---|---|---|---|
| V1 `suspend`, `power`, `foreign_reads` ok on every arm a gate reads (`foreign_reads` watches the SPILL drive on RAM-tier arms, the weights' drive on disk-tier arms) | F1 `ArmWatch` | yes | a void block is replaced; an unknown arm takes the gate away |
| V2 spill volume scope `local` or `virtual` (never tmpfs, overlay or network); unbuffered round trip ok; every spilling arm's store is direct | `a4_box.spill_volume`, `store.direct` | yes | that arm is invalid |
| V3 device is CUDA | meta | yes | a CPU pilot never gets a verdict |
| V4 A and B of a block train on the same token rows | ids fingerprint | yes | the block is invalid |
| V5 finite losses on every timed step | records | yes | the arm is invalid |
| V6 the plan happened: budget `1` spills nothing; any other budget spills > 0 layers and writes and reads exactly `spill_file_bytes` per step | records | yes | the arm is invalid |
| V7 the F1 repeats rule has no block left to add | `arm_validity.blocks_to_add` | yes | no verdict on that gate |
| V8 disk-tier arms read the weights with direct I/O | runtime | yes | the arm is invalid |
| V9 `disk_k4_same_drive`: every spilling arm's OS placement (resolved path, mount, whole physical disk) shows the spill file and the weight shards on one physical disk | `a4_box.placement` in each arm record, recomputed by the rule | yes | no verdict on G5 (unknown placement is not "same") |

Verdict, in order:

1. a CPU pilot: `NO VERDICT: SYNTHETIC/CPU PILOT`;
2. G1, G2 or G3 fail: `SPILL NOT SAFE` (A3 must not ship the spill path as built);
3. G1, G2 or G3 without a verdict: `NO VERDICT`;
4. G4 without a verdict: `SPILL SAFE, COST NO VERDICT`;
5. G4 pass: `SPILL READY FOR A3` (an automatic fallback past the RAM budget);
6. G4 fail: `SPILL SAFE BUT COSTLY` (an explicit capacity tier, not a fallback);
7. G5 fail appends: "on the disk tier spill to a drive the weights are not read from".

Informational, never gating: `ram_k2_min`, `ram_k3_min`, `ram_k4_quarter`, `ram_k4_min`
(overhead intervals); spill write and read GB/s; the hidden share `1 - (put_wait +
get_wait) / (write busy + read busy)`; the drive's sequential rates (`drive.json`);
an overhead upper bound <= 5% is reported as "hidden".

Thresholds, and why: 10% is the effect the F1 repeats rule is powered to detect
(METHOD.md, `EFFECT_FRACTION`); a smaller threshold would need blocks the rule does not
plan for. 0.90 for G5 is the same 10% on the weight read rate. k = 2 and 4 are the
card's requirement and research question 3's region where compute meets the read.

### 3.1 Protocol v2 (fixed after the v1 session, before the v2 session)

What v1 (2026-10-25, one RTX 3090 VM, §9) showed, and what v2 changes because of it.
Nothing below was tuned to make a v1 number pass; v1 is not re-judged.

| | v1 | v2 | why |
|---|---|---|---|
| G2 reference | allocator growth (peak − baseline) <= the activation budget, every arm | B's excess outside its store (growth − held bytes) <= the excess both A arms of its block share, which must be equal; plus store == plan <= budget on every arm, unchanged | in v1 every one of 280 arms grew by budget + **4 B**: 140 no-spill A arms and 140 B arms alike, excess difference B − A = 0 in all 70 blocks, store == plan in all 280. The 4 B are a pinned float32 buffer torch allocates for `float(loss)` on the timed steps (confirmed locally on torch 2.14.1; `CUDAScalar.cu`). v1's rule charged a cost both arms share to the spill; v2 bounds only what the spill adds |
| block cap | `--max-blocks 12` | `--max-blocks 24` | v1 stopped at 12 blocks in 5 of 7 comparisons; the repeats rule wanted 321 more (`ram_k2_half`), 20, 34, 13 and 6 |
| machine admission | none | `a4_admission.py` before the session (below); not a gate | in v1, 17 of 140 spill arms wrote below 0.5 GB/s (0.04-0.44): 12 of 116 RAM-tier spill arms (median 2.02 GB/s) and 5 of 24 same-drive arms (median 0.54 GB/s; that drive also serves the weight reads, by design). Each of the five comparisons that hit the cap had such an arm inside its 5-block pilot, from which the repeats rule sizes N |
| rule version | `a4-ram-budget-spill-v1` | `a4-ram-budget-spill-v2`; `verdict.json` carries `rule_version` | |

Unchanged: G1, G3, G4 and G5 and their thresholds (10%, 0.90), the repeats rule (5-block
pilot, N from it, one top-up, `arm_validity.blocks_to_add`), validity rows V1-V9, the seven
comparisons, steps per arm, the verdict order, `--max-hours 8` (a session that reaches it
is recorded as stopped, not extended), the 24 GiB / swap 0 scope, no FP8, no second drive.

**Machine admission, v2** (`a4_admission.py` at `efdd1a76`; replaced in v3, §3.2): 600 s of
unbuffered I/O on the spill volume in 4 MiB chunks, each cycle writing one k = 4
minimum-budget spill file (32 layers x 4 chunks) in the store's forward order and reading
it back in the backward order (layers descending), from 16 distinct random chunks, with
no pause between cycles. Each operation's bytes and busy time are shared over the 10 s
windows it overlaps. The machine was admitted iff every window's write rate AND read rate
was >= 50% of that direction's median window rate, no read returned other bytes, and no
file was left.

**What the v1 data looks like under v2** (illustration, not a verdict:
`results/a4-rtx3090-20261025/v1-under-rule-v2-ILLUSTRATION.json`): G2 would pass (A
reference 4 B in all 70 blocks, every B excess 4 B), and the verdict would read `SPILL
SAFE, COST NO VERDICT` with v1's 12 blocks.

**Limits of v2.** The cause of v1's dips is not established (a host neighbour, the virtual
disk, guest write-back or the host SSD: none is shown). `foreign_reads` sees only I/O
inside the VM. The physical topology and the host's cache are not controlled. G2's v2
reference no longer catches an allocation both arms make; such an excess is still
recorded per arm.

### 3.2 Protocol v3: why the admission changed (fixed after the v2 refusals, before v3)

v3 replaces the machine admission and restricts the GPU class (below). The decision rule is
unchanged: `a4_rule.py` is byte-identical to v2's, so a v3 session is "protocol v3, rule v2"
and `verdict.json` reads `rule_version` v2. G1-G5, their thresholds (10%, 0.90), the repeats
rule, V1-V9, the seven comparisons, `--max-blocks 24`, `--max-hours 8`, the 24 GiB / swap 0
scope, no FP8 and no second drive stay as §3.1 has them.

**The v2 admission was wrongly specified, in two ways.**

1. It wrote with no pause, at the volume's full speed. The whole v1 session asked for at
   most 0.233 GB/s of spill writes (and the same volume of reads); the v2 admissions wrote
   at 1.1-4.0 GB/s, 5-17 times that, and 0.6-2.2 TiB in 10 minutes against 0.34 TiB of
   spill in the whole 2.5 h v1 session.
2. Its criterion (`bins` / `judge`) divided bytes by the volume's busy time and compared
   each window with the median of the same volume. That tests whether the volume is steady
   against itself, not whether it keeps up with the session: a volume that dips from 10
   to 3 GB/s is refused although 3 GB/s covers the demand many times over, and a volume
   steady at 0.3 GB/s is admitted although it cannot keep the session's pace with margin.

So the three v2 refusals say nothing about whether those machines would have run a session
steadily; there is no conclusion about the platform. The v2 admission data are not re-judged
by the v3 criterion: they were taken without the schedule.

| load | write rate | written | duration |
|---|---|---|---|
| v1 session, largest spill arm (`ram_k3_min/b05_p1_B`) | 0.233 GB/s (written / step) | 384 MiB per 1.73 s step | |
| v1 session, all 280 arms | | 0.34 TiB of spill | 2 h 29 min |
| v2 admission, machine 1 (RTX 5090) | 4.04 GB/s (wall clock) | 2.21 TiB | 600 s |
| v2 admission, machine 2 (RTX 3090 Ti) | 1.14 GB/s (wall clock) | 0.62 TiB | 600 s |
| v2 admission, machine 3 (RTX 4090) | 1.49 GB/s (wall clock) | 0.81 TiB | 600 s |
| v3 admission (schedule) | 0.466 GB/s (twice the demand) | 0.76 TiB | 1800 s |

**Machine admission, v3** (`harness/a4_admission.py`; every number fixed here, before any
v3 box):

- **Demand** D = 402,653,184 B / 1.729781 s = 0.2328 GB/s: the largest, over v1's 140
  spill arms, of the arm's median over its timed steps of `spill_written / step_s`
  (`ram_k3_min/b05_p1_B.json`; `a4_admission.v1_demand` recomputes it from
  `results/a4-rtx3090-20261025/`). The spill reads per step equal the writes.
- **Cycle**: one k = 4 minimum-budget spill file (32 layers x 4 chunks x 4 MiB = 512 MiB)
  written in the forward's order, then read back in the backward's, through the store's own
  `DirectSpillFile`, from 16 distinct random chunks. Every 4 KiB sector carries its cycle
  and file offset, so a read that returns stale or misplaced bytes counts as a mismatch.
- **Schedule**: period P = 512 MiB / (2 D) = 1.1532 s, i.e. writes at twice the demand and
  reads at twice the demand. Cycle i starts at max(t0 + i P, the end of cycle i - 1), so a
  volume that falls behind runs cycles back to back; lateness_i = end_i - (t0 + (i + 1) P).
  1,561 cycles, 0.76 TiB written, over 30 minutes.
- **Admitted** iff the largest lateness over the 30 minutes is <= 5 s, every chunk read
  back is the bytes its cycle wrote there, and no test file is left. 5 s is about four
  periods of slack: a short stall passes; a sustained fall below the tested rate, the kind
  v1's slow arms showed (0.04-0.44 GB/s), builds lateness fast and is refused.
- **Recorded**: per cycle its planned start, start, end and lateness
  (`admission-cycles.csv`); per second the bytes and busy time (`admission-per-second.csv`);
  the 10 s windows against their median, as description only. The windows play no part in
  the decision.

On the controlled-time model in `tests/test_a4_admission.py` (exact fake volumes): a
volume at 10 GB/s that dips to 3 GB/s for ten minutes never falls behind and is admitted,
where v2 refused it; a volume steady at 0.3 GB/s ends 746 s behind and is refused, where
v2 admitted it; five minutes at 0.1 GB/s, the rate of v1's slowest arms, end 237 s behind;
one 3 s stall is absorbed, one 8 s stall is not.

**What passing means.** The admission is a screen, not a guarantee: it shows the spill
volume kept twice v1's spill pace for 30 minutes without falling more than 5 s behind. It
does not show the volume cannot dip during the session, and it does not explain v1's dips.
After a session, `a4_admission.describe_arms` lists, for description only, each spill
arm's write rate against the demand and the arms below it; it drops no arm and changes no
verdict. `a4_rule` reads nothing of the admission.

**GPU class.** The demand was measured on an RTX 3090 (Mistral-7B NF4, seq 512). A faster
GPU shortens the step and so raises the rate spill must sustain. The v1 step was set by
compute: the all-pinned A arms took 0.577-0.591 s per micro-batch at k = 2, 3 and 4 alike
(RAM tier, medians over arms), so the weight stream was hidden. NVIDIA's published dense
BF16 tensor rate with FP32 accumulate is 71 TFLOPS on the RTX 3090, 80 on the RTX 3090 Ti
and 165.2 on the RTX 4090 (the GA102 and Ada whitepapers). A 4090 step could therefore be
up to about 2.3x shorter and its spill demand up to about 0.54 GB/s, above the 0.466 GB/s
the admission holds; a 3090 Ti's, up to about 0.26 GB/s, stays inside it. So v3 runs only
on an RTX 3090 or RTX 3090 Ti. The owner's instruction also allowed the RTX 4090; it is
excluded for this reason, fixed before any v3 box.

**The repeats rule** (unchanged) sizes the blocks to detect a 10% effect with 80% power
(`ceil(7.85 (sigma / delta)^2)`, delta = 10% of A's mean); it is not sized to estimate the
overhead to +-10%. G4 and G5 still decide on the 95% interval against 10% and 0.90, so a
comparison can reach its N with an interval that still straddles the threshold.

**The cause of v1's slow arms is not established**; v3 changes which machine is used, not
what is known about why a volume dips.

## 4. Fixtures, arms and order

- **F1**: `unsloth/mistral-7b-instruct-v0.3` at revision
  `60b99dcbc7f853b7cd18472a607e8d045b96f03e` (the A0 fixture,
  `results/a0-vast-rtx3070/preparation-0*/preparation.json`), NF4 double-quant CUDA
  cache, seq 512, LoRA r 8 on q/k/v/o, seeds 3 (model) and 17 (inputs). 32 layers,
  hidden 4096, chunk 4 MiB (bf16). k = 2, 3, 4: all-pinned at k = 2 is 256 MiB,
  above the 0.5 and 0.25 budgets.
- **Tiers**: RAM tier for G1, G3, G4 and the informational arms; disk tier for G5, with
  the spill on the drive that holds the shards.
- **Optional F2**, the synthetic dense 70B (`synth_checkpoint.py --shape llama-70b
  --vocab 32768`): 80 layers, chunk 8 MiB, all-pinned 2.5 GiB at k = 4. Needs about
  140 GB of source plus 40 GB of NF4 cache plus the spill, so about 205 GB free (A0's
  preparation check). Not part of the verdict.
- **Order**: preflight, drive, correctness k = 2, 3, 4 (with `_det` only when GA or
  all-pinned is not reproducible), cleanup, then for each comparison A-B-B-A blocks in
  fresh processes until `blocks_to_add` is 0 (5-block pilot, at most one top-up,
  `--max-blocks 24`; v1 ran with 12). Comparisons in this order: `ram_k2_half`, `ram_k2_min`,
  `ram_k3_min`, `ram_k4_half`, `ram_k4_quarter`, `ram_k4_min`, `disk_k4_same_drive`,
  and `disk_k4_other_drive` when a second physical drive exists. A is budget `1`
  except in `disk_k4_other_drive`, where A spills to the other drive and B to the
  weights' drive.
- **Per arm**: 2 warm-up steps and 5 timed steps; the arm's value is the median step.
  Recorded per step: step time, losses, put/get wait, spill write/read seconds and
  bytes, and weight read bytes and busy time on the disk tier. Per arm: the plan, the
  cap check, host allocator baseline and peak, `RssShmem` growth from the store,
  peak RSS, cgroup limit/current/peak, the F1 record (box stamps with free RAM and
  commit, power, foreign reads, GPU peers, SM clock), and whether a spill file or a
  spill thread is left.

## 5. What each outcome means for A3 and G4

- **SPILL READY FOR A3**: A3 can make spill the automatic fallback when the activations
  outgrow the budget: same numbers, no step-time cost past 10% at k = 2-4, RAM honoured.
- **SPILL SAFE BUT COSTLY**: spill is correct and bounded but slower. A3 keeps it as an
  explicit capacity tier (opt in, with the overhead printed) and picks k from the
  all-pinned budget by default. G4's compression then has a second motive: fewer bytes
  to spill.
- **SPILL NOT SAFE**: A3 must not ship this path. The failing gate names what to fix
  (numerics, accounting, or cleanup).
- **G5 fail**: on the disk tier the spill must go to a different drive from the
  weights, or the reader's throughput is shared. The runbook's `disk_k4_other_drive`
  arm says whether a second drive recovers it.
- **For G4**: G4's compressed chunk size plugs into the same planner (`chunk_bytes`,
  `pinned_floor_bytes`); the measured spill GB/s and hidden share say how much the
  smaller spill is worth.

## 6. What this does NOT measure

- Model quality: the arms train a few steps on fixed random token rows. Spill is
  lossless, so G1 is the whole quality argument.
- Giant-MoE throughput, or any 70B number unless F2 is run (and then only as
  information).
- A physical-host view on a virtualised box: `foreign_reads` sees only I/O inside the VM;
  the physical topology under a virtual disk and the host's cache are not controlled.
- Whether the spill volume stays as steady during the session as during the admission, or
  why a volume dips.
- Endurance (TBW) of the spill drive.
- Pageable (non-pinned) activations on CUDA; the planner handles `pin=False`, the
  GPU arms do not use it.
- C3's residency planner: §2 defines the spare term it reads; C3 is not run here.
- A RAM limit enforced by anything other than the cgroup the run is placed in.

## 7. Box state to record

Per arm, by the F1 `ArmWatch`: power source, suspend, foreign reads on the watched
drive, GPU processes and memory, SM clock and clock-event reasons, free RAM and commit
before and after. Per session (`protocol.json`): base commit, package versions, SHA-256
of every source file, the rule (`a4_rule.RULE`). Per arm in addition: cgroup
`memory.max`/`memory.current`/`memory.peak`, `RssAnon`/`RssShmem`/`VmHWM`, the spill
volume's filesystem, device, model and free space.

## 8. GPU runbook

Target: a Linux CUDA box with an RTX 3090 or RTX 3090 Ti (the GPU class v1's spill demand
was measured on, §3.2), one local NVMe that is NOT the container's overlay root, about
60 GB free, and `systemd-run` or `docker` to impose the RAM limit.

### Needs

| Resource | Amount | Why |
|---|---|---|
| VRAM | estimate under 8 GB: not measured for F1 here; A0 measured 4.46 GB peak allocated on the 70B shape at batch 1 on its 8 GB card | Mistral-7B NF4 + LoRA + one layer's activations; L2L's VRAM does not grow with k (A0's G4) |
| Host RAM limit | 24 GiB (`MemoryMax`) | F1 RAM-tier store ~4.1 GB pinned + activations <= 512 MiB + ~6 GB process + reserve |
| Disk | ~15 GB source + ~4.5 GB NF4 cache + <= 1 GiB spill + 4 GiB drive test; 60 GB free is ample | the `snapshot_download` below fetches the source |
| Second physical drive | optional | `disk_k4_other_drive` only |
| Time | v1 measured ~32 s per child (289 children in ~2 h 35 min on one RTX 3090 VM); v3 estimate: 30 min admission, then up to ~6 h if every comparison runs to the 24-block cap (`--max-hours 8` stops it) | 7 comparisons x 5-24 blocks x 4 arms; correctness 3 x ~3 min; cleanup ~3 min |

### How the RAM limit is imposed

Pinned (page-locked) host memory counts against the cgroup limit: gate-655 measured a
pinned 4 GiB allocation adding exactly 4 GiB to the cgroup's `memory.stat.shmem`. So
`MemoryMax` caps the weight arenas, the activation store and the rings together.

```bash
# systemd (a host or a VM with a user session):
systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 -- <command>
# docker:
docker run --gpus all --memory 24g --memory-swap 24g -v /mnt/nvme:/spill ... <command>
```

The probe reads the tightest `memory.max` from its own cgroup up to the root (v1: the
memory controller's `memory.limit_in_bytes`, likewise) and plans inside it
(`split_host_ram`); `ram_limit` records `memory.swap.max` and `SwapTotal` beside it, so
a run whose swap was not set to 0 is visible in every arm record. Do not run inside a
Vast container whose only writable volume is the overlay root: the preflight refuses an
overlay spill directory. Bind-mount a local NVMe volume and spill there.

### Commands

```bash
git clone https://github.com/SoupTeam/soup-rnd && cd soup-rnd
git checkout research/a4-l2l-ram-budget-spill   # the commit that is to carry this rule
A0_PYTHON=python3.12 bash benchmarks/a0-bootstrap.sh   # venv, CUDA torch, pinned deps, A0 tests
.venv/bin/python -m pytest tests/test_a4_activation_budget.py tests/test_a4_spill_store.py \
  tests/test_a4_box.py tests/test_a4_rule.py tests/test_a4_g4_compose.py \
  tests/test_a4_admission.py -q -o addopts=''
# Real ENOSPC volume for the cleanup child (root; 64 MiB, below the 512 MiB spill file
# the cleanup's k = 4 minimum plan writes):
sudo sh -c 'truncate -s 64M /root/a4-small.img && mkfs.ext4 -q -F -m 0 /root/a4-small.img \
  && mkdir -p /mnt/a4-small && mount -o loop /root/a4-small.img /mnt/a4-small \
  && chmod 777 /mnt/a4-small'
export PYTHONPATH="$PWD/src:$PWD/benchmarks/harness" HF_HUB_OFFLINE=0
# Fixture: Mistral-7B at the pinned revision, NF4 cache prepared by a dry run.
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("unsloth/mistral-7b-instruct-v0.3",
                  revision="60b99dcbc7f853b7cd18472a607e8d045b96f03e",
                  local_dir=".soup/models/mistral-7b", token=False,
                  allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"])
EOF
export HF_HUB_OFFLINE=1
SPILL=/mnt/nvme/a4-spill            # a local NVMe directory, not overlay, not tmpfs
WSPILL=/mnt/nvme-weights/a4-spill   # on the SAME physical disk as .soup/f1-nf4 (V9 checks)
# 0. Machine admission (30 min, §3.2); exit 4 = refused: delete the box, take the next one.
.venv/bin/python benchmarks/harness/a4_admission.py --spill-dir "$SPILL" \
  --out .soup/admission.json --log .soup/admission-per-second.csv \
  --cycles .soup/admission-cycles.csv
# 1. Plan (prints every command, writes nothing):
.venv/bin/python benchmarks/harness/a4_session.py \
  --weights .soup/models/mistral-7b --shards .soup/f1-nf4 --spill-dir "$SPILL" \
  --weights-spill-dir "$WSPILL" \
  --out .soup/results/a4-01
# 2. Run under the RAM limit:
systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 -- \
  .venv/bin/python benchmarks/harness/a4_session.py \
  --weights .soup/models/mistral-7b --shards .soup/f1-nf4 --spill-dir "$SPILL" \
  --weights-spill-dir "$WSPILL" \
  --enospc-dir /mnt/a4-small \
  --out .soup/results/a4-01 --execute
# Optional, second physical drive:
#   add --other-spill-dir /mnt/nvme2/a4-spill --comparisons <default list>,disk_k4_other_drive
# 3. Verdict, recomputed from the raw JSON alone, and the arms against the admission's
#    demand (description only):
.venv/bin/python benchmarks/harness/a4_rule.py .soup/results/a4-01 --out verdict.json
.venv/bin/python benchmarks/harness/a4_admission.py --describe .soup/results/a4-01 \
  --out arms-vs-demand.json
```

The first `correctness` child shards the checkpoint into `.soup/f1-nf4` (about two
minutes) if the cache is absent; every later child reuses it. `--enospc-dir` adds the
`enospc_real` cleanup case: the store's spill file is created on that volume and the
filesystem itself must refuse it with ENOSPC. The probe refuses a `--enospc-dir` that is
not the root of its own filesystem, that sits on a volume over 1 GiB (a wrong path must
not fill a real drive), or whose free space would hold the spill file. Without it the
cleanup runs the injected ENOSPC only, and G3 then rests on the injected case.
`--weights-spill-dir` has no default. Before any child the session records the OS's
placement of the shards and of each spill directory (resolved path, mount, whole disk from
`/sys/dev/block`), refuses a same-drive comparison the OS does not show on one physical
disk, and refuses `--other-spill-dir` unless the OS shows it on another disk. Every arm
records the same placement again; V9 removes G5 when any spilling same-drive arm lacks it.
`--accept-network` runs on a network volume, labelled `network`, and V2 then removes
every gate that reads a spilling arm.

### Files that come back

`.soup/results/a4-01/`: `protocol.json`, `preflight.json`, `drive.json`,
`correctness_k{2,3,4}.json` (and `_det` if run), `cleanup.json`,
`arms/<comparison>/b<NN>_p<P>_<A|B>.json`, `logs/*.log`, `session.json`,
`verdict.json`; beside it `.soup/admission.json`, `.soup/admission-per-second.csv` and
`.soup/admission-cycles.csv`.

## 9. Results

### CPU pilots and acceptance, 2026-10-22 and 2026-10-23 (no verdict)

The raw evidence of these CPU runs is kept locally and is not published; none of it is
CUDA or GPU evidence, and the rule refuses to judge any of it (V3).

- SYNTHETIC / CPU PILOT: SmolLM2-135M, end to end through a real unbuffered spill file on
  a Windows laptop; shows the harness runs, not evidence for any gate.
- Pre-GPU acceptance (2026-10-22): the G4+A4 composition smoke (8 rows x 4 budgets agree
  with the plan; direct spill), a real ENOSPC on an 8 MiB ext4 loop filesystem in WSL2
  through `open_spill("direct")` -> `DirectSpillFile` (raised by the zero fill while the
  file is created; not a mid-step write, not through `BudgetedStore`), and the cgroup
  reading inside a real `systemd-run -p MemoryMax=512M` scope.
- Second review (2026-10-23): audit reproductions fixed, each with a regression that fails
  on the previous code: arm records are keyed by `<comparison>/<basename>` (a one-byte
  budget violation in `ram_k2_half/b00_p1_B.json` was hidden by a clean namesake in
  another comparison); the spill tests read the open file through its handle (`fstat`) and
  check, on Linux, that the name is gone while the handle is open; `--weights-spill-dir`
  has no default, the session refuses a same-drive comparison the OS does not place on the
  shards' disk (and an `--other-spill-dir` it does not place on another disk), and V9
  checks every arm's own placement.
- Protocol v2 (2026-10-25, at `efdd1a76`): a SmolLM2-135M CPU session through the v2
  harness (22 children, all exit 0, `rule_version` v2 in `verdict.json`, `NO VERDICT:
  SYNTHETIC/CPU PILOT`), and a 30 s admission smoke on the laptop's NTFS volume through the
  real `DirectSpillFile` (refused, as a smoke, by its own non-protocol settings).
- Protocol v3 (2026-10-26): a 60 s admission smoke on the laptop's NTFS volume through the
  real `DirectSpillFile`: 53 of 53 scheduled cycles, largest lateness 0.116 s, 0 of 6,784
  chunks read back wrong, wall write 0.471 GB/s, no file left (refused, as a smoke, by its
  non-protocol duration); `--describe` on the published v1 records runs.

### GPU, v1: one RTX 3090 VM, 2026-10-25 (rule v1)

**`SPILL NOT SAFE: A3 must not ship the spill path as built`**, by rule v1, at commit
`e123b5ff` (the v1 harness; its L2L step and store were the harness's own copies, hashed in
`protocol.json`). Compact record: `results/a4-rtx3090-20261025/`.

| gate | v1 result | detail |
|---|---|---|
| G1 | pass | k = 2, 3, 4 at spill layers 20-32: forward and backward `torch.equal` to all-pinned and GA, under the predeclared `--deterministic` rerun (the default runs were not reproducible GA vs GA at any k); the corrupted-read control differs |
| G2 | **fail** | all 280 arms: allocator growth = budget + 4 B (§3.1) |
| G3 | pass | Ctrl-C mid-backward, exception mid-forward, injected ENOSPC and a real ENOSPC on a 64 MiB ext4 loop volume: named, no file, no thread, pinned memory returned |
| G4 | no verdict | `ram_k4_half` −0.61% [−5.56%, +4.33%] (5 blocks, hidden); `ram_k2_half` 12 blocks, interval [−13.7%, +64.2%], the repeats rule wanted 321 more |
| G5 | no verdict | weight read rate ratio 0.838 [0.758, 0.918], 12 blocks; the rule wanted 6 more |

Box: vast.ai RTX 3090 24 GB, KVM VM, 98.8 GiB RAM, one virtio disk for the spill and the
shards, session under `systemd-run --scope -p MemoryMax=24G -p MemorySwapMax=0` (read
back per PID from inside, in every arm record). After the run, not changing the verdict:
the 4 B source (§3.1) and the v1 data under rule v2 (illustration).

### GPU, v2: no session, the admission was wrongly specified (2026-10-25, `efdd1a76`)

**No v2 verdict.** Three machines were rented, the owner's limit for the admission; each
passed the preflight (MemTotal > 24 GiB, VRAM >= 23 GB, CUDA, cgroup 24 GiB / swap 0 read
per PID inside the scope, spill ext4 local with direct I/O, suspend / power / foreign_reads
ok) and none passed the v2 admission (§3.1, protocol settings, unchanged); each was then
destroyed and confirmed gone. No A4 arm ran on any of them, and the stop rule ended v2.

| machine | GPU, host / machine | write median, floor (GB/s) | low write windows | read median, min (GB/s) | first slow second, written before it |
|---|---|---|---|---|---|
| 1 | RTX 5090, 702330 / 152098 | 9.82, 4.91 | 5 (2.43-2.77) | 7.39, 5.28 | 171 s, 0.71 TiB |
| 2 | RTX 3090 Ti, 79466 / 18065 | 2.47, 1.23 | 7 (0.31-1.20) | 2.53, 2.33 | 433 s, 0.49 TiB |
| 3 | RTX 4090, 59399 / 11740 | 3.52, 1.76 | 12 (1.70-1.76) | 3.35, 3.23 | 350 s, 0.55 TiB |

All three: KVM VM, one virtio disk `vda` for the spill and the weights, 0 read-back
mismatches, no read window low. Rental cost: $0.59 of the $15 v2 budget (invoices).

What this does and does not show:

- None of the three machines passed the v2 admission, and the v2 admission was wrongly
  specified (§3.2): it wrote with no pause, at 5-17 times v1's spill rate, and judged each
  window against the volume's own median, not against what the session needs. The lowest
  write window on any of the three (0.31 GB/s busy rate, machine 2) is above v1's largest
  spill demand (0.233 GB/s). The refusals say nothing about whether these machines would
  have run a session steadily: there is no conclusion about the platform.
- Why the writes dipped after 0.5-0.7 TiB is not established (a host write cache filling,
  a neighbour, the virtual disk, the host SSD: none is shown).
- G4 and G5 WERE measured in v1, and stayed inconclusive (§9, v1 table): `ram_k4_half`
  −0.61% [−5.56%, +4.33%] and `ram_k2_half` [−13.7%, +64.2%] at the 12-block cap; the
  weight-read ratio 0.838 [0.758, 0.918]. v2 adds no measurement to them.
- G1 and G3 (spill bit-exact; cleanup leaves nothing) and the store's exact accounting
  stand on the v1 session; v2 re-judged nothing. The v1 verdict stays v1's.
- **No shipping clearance was established.** v1 is `SPILL NOT SAFE` (G2) and v2 has no
  verdict, so A4 clears no spill mode for A3, neither an automatic fallback nor an explicit
  one.

Raw records stay outside the repository: the three admissions and preflights are published
compactly in `results/a4-v2-admission-20261025/`.
