# C2 Linux physical-NVMe protocol v1

Frozen before any Linux read-rate or training arm. This is a new experiment;
the Windows protocol, old R3' rule and all earlier results remain unchanged.
User authorized finding/renting a suitable Vast machine and completing C2,
then opening a PR (2026-10-18). Rental 56566608 is dedicated to this task,
whole-GPU offer, Samsung 990 PRO 1 TB advertised, RTX PRO 4000, approximately
32 GB RAM, 300 GB disk, actual quoted rate $0.308889/hour. These are provider
claims until verified inside the machine. Destroy the rental after downloading
evidence; this attempt's budget is $1. No existing D2 instance is used.

## Hardware admission, before performance measurements

Record OS/kernel, Python/packages/import path and Git commit, GPU/driver/clocks,
host `/proc/meminfo`, cgroup memory limits, mounts and every NVMe device's
sysfs model and statistics. Admit only a real local NVMe leaf block device:
no network, QEMU or hardware RAID device presented as proof of an NVMe.
No system settings, cache dropping, peer termination or clock changes.

Ordinary filesystem-to-block mapping is preferred. When a container exposes an
overlay root without its host mount, use the following predeclared admission
check: after a five-second idle window (each NVMe <1 MiB/s reads AND writes),
read exactly 1 GiB of an allocated, non-sparse fixture with the shipped
`open_direct`/`read_range_into`; exactly ONE whole NVMe must report a read-byte
increment within 5% of 1 GiB, and no other NVMe may report >=10% of that amount.
Record every counter before/after. This is an untimed mapping/validity check,
not a throughput result. If no unique physical NVMe can be established, stop
with NO VERDICT rather than accepting an advertised disk name.

The immutable fixture must be larger than host MemTotal, not merely larger
than currently free RAM or a container allocation. No sparse or compressed
synthetic file is accepted. Record sizes, allocated blocks and mtimes before
and after. Synthetic random weights/bytes are labelled as such; they measure
real hardware behavior, not model quality or real-data training convergence.

## Primitive arms

- Four workers and four sector-aligned ranges per 400 MiB span, unchanged.
  Same ordered 24 spans at both depths. All reads use the shipped Linux
  `O_DIRECT` path; reject fallback. Two touched, pinned 400 MiB arenas in BOTH
  arms. Depth 1 cycles slot 0, depth 2 slots 0/1. No extra workers or read-ahead.
- Untimed full-byte verification of every span at each depth before reuse;
  after timed arms fully verify every retained final slot. Queue is bounded
  at depth spans / `depth * 4` range tasks, with at most four executing calls.
  All writers drain before slot reuse/return, including errors.
- Six paired rounds, orders 1/2, 2/1, 1/2, 2/1, 1/2, 2/1. One untimed span
  warm-up before each arm. Throughput = total submitted bytes / monotonic
  submission-to-final-completion wall time / 1e9. Never sum overlapping spans.
- Record protocol/harness hashes, exact fixture spans, allocated and pinned
  bytes, raw wall times/rates, physical NVMe counters at 0.25 s, temperature
  if exposed, memory, CPU load, GPU clients/state and all checks/exceptions.

### Validity and decision

Any byte mismatch is FAIL correctness. Any missing/invalid physical evidence,
changed fixture, fallback, I/O error, memory admission failure, incomplete
round or unknown foreign workload is NO VERDICT. Preserve partial/void arms;
stop that sequence, never selectively replace a slow arm. A later window is
a new separately named sequence, with this same rule.

At every arm boundary: >=4 GB MemAvailable; no other GPU client. Before each
arm: five seconds with NVMe reads and writes each <=1 MiB/s. During an arm:
>=4 samples, no sample gap >2 s; physical NVMe read-byte delta must be within
[0.80, 1.20] times submitted bytes, and writes <=1 MiB/s. Negative/missing
counters or unavailable device statistics void the sequence. This does not
prove global absence of peers; a visible peer workload also voids it. Capture
GPU state, disk temperature and CPU state rather than silently changing them.

Use UNROUNDED paired depth2/depth1 rates after validity:

| Six valid ratios | Primitive decision |
| --- | --- |
| All >=1.10 | PASS primitive |
| All within [0.95, 1.05] | FAIL benefit |
| All <0.95 | FAIL regression |
| Anything else | NO VERDICT / ambiguous |

## Training-step diagnostic and shipping gate

Record step time independently of the primitive. A benchmark-only source
adapter may admit depth 1/2 without changing shipped runtime or config. It must
preserve staging ownership, release events, direction/replans, oldest-read
liveness, per-root worker counts, errors and teardown. Test bytes, queue bounds,
held slots and asynchronous CUDA copies before using its timing results.

Fixture: synthetic Llama-70B shape, 80 layers, vocab 32768, BF16 source / NF4
cache, same cache at both depths, cache larger than physical host RAM. If this
condition cannot be met on the rented host, freeze a new fixture protocol
BEFORE running steps; do not silently weaken it. Existing `stream_probe.py`
build/step path, batch 1, seq 256, read-ahead **3 in both arms**, two device
buffers, four ranges, pinned staging, LoRA r8 q/k/v/o, adapter seed 3, input
seed 17, PagedAdamW8bit lr 1e-4. Fresh process/model/optimizer per arm. Each
arm: one warm-up and three timed uninstrumented forward/backward/optimizer
steps. Six rounds with the same alternating orders. Record all losses, peak
VRAM, clocks, physical reads, byte checks and imported source class.

The benchmark adapter is experimental and does not itself ship depth 2.
Production implementation requires BOTH primitive PASS and valid repeated
step benefit: all six step-time speedups >=1.05 AND median speedup >=1.10,
finite losses, all byte/DMA checks passing, identical staging/device budgets,
and no >2% peak-VRAM increase (or >64 MiB, whichever is larger). Otherwise
keep shipped depth 1 and report the negative/ambiguous result. Losses are
diagnostic; timings are not evidence of convergence or gradient parity.
Striping is outside a one-physical-disk result and must be gated separately
before enabling a changed default for striped stores.

No run exists at protocol creation. Results, amendments and commands are
appended below, leaving this frozen section verbatim.

## Admission history (before performance arms)

- Rental A, 56566608: rejected before a throughput arm. Sysfs/lsblk reveal
  two Samsung 990 PRO drives behind RAID1 (`md0`/`md1`), so its advertised
  6.694 GB/s is not a one-physical-drive ceiling. Raw inventory:
  `evidence/c2-linux-nvme-20261018/admission-rental-a.json`. Deleted via API;
  no model install, fixture creation or read-rate result from this rental.
- Rental B, 56567227 (offer 56206478): RTX 5070 12 GB, i7-14700K, advertised
  32 GB RAM, 300 GB allocated disk, quoted $0.323333/hour including storage.
  Sysfs identifies a single `nvme0n1`, Lexar SSD NQ780 1 TB; no md RAID.
  Overlay storage is backed by XFS/LVM; the preregistered direct-read counter
  mapping check remains required. Raw inventory: `admission-rental-b.json`.
  The same frozen decision/validity/fixture rules apply. No performance arm
  exists at the time this admission note is written.

## Prospective F1 extension (before sequence 2 or any training-step arm)

Sequence 1 is void: its third pair failed the pre-arm idle-writes threshold.
Its two completed pairs are retained, excluded from all decisions, and are not
a pilot. No training-step arm has run. The original section above is unchanged.
The six-pair preparation alone did not implement METHOD.md's adaptive repeat
count. The following additional requirement is frozen before new measurements;
it can only withhold implementation, never rescue a failed original gate.

Use `c2_f1_campaign.py` for **both** primitive and step comparisons. Each run
is a fresh process, in `arm_validity.block_order` A-B-B-A blocks (A=depth 1,
B=depth 2). First five valid blocks form the pilot. Use the unchanged
`blocks_to_add` and `repeats` functions for the pilot estimate, initial target,
and single top-up. For steps the F1 value is the median of three timed steps;
retain the original means as well. For the primitive it is submitted GB/s.
No result is accepted before this repeat rule returns zero. Cap at 20 blocks
or this rental attempt's $1 budget; reaching either limit before completion is
NO VERDICT. A void/unknown run stops the entire separately named sequence;
retain it and restart a whole sequence in a new window, without selecting or
replacing individual runs. This is stricter than F1's block replacement.

Require all three unchanged F1 `ArmWatch` checks: suspend, power, foreign_reads.
Watch the child from its parent and wait for its exit before ending the watch.
Because the host mount is hidden by overlay, inject the **already admitted**
unique physical NVMe name and its sysfs read counter into ArmWatch; other-disk
reads sum only the other physical NVMe leaves (not duplicate LVM counters).
The 1 GiB direct-read mapping and full-byte correctness must already have
passed in a preserved admission record and are repeated in every primitive
child. No fabricated successful sensor values or ignored unknown checks.
Keep all existing per-arm quiet, byte, physical-I/O, memory and GPU checks.
Record F1 sensor evidence, clocks, commands, exits and raw child results.

Apply the original thresholds conservatively to **every** consecutive paired
round in the completed F1 sequence, rather than choosing six favorable pairs.
Primitive PASS requires all paired rate ratios >=1.10. Step PASS additionally
requires every paired median-step speedup >=1.05, median speedup >=1.10 and the
same staging/VRAM limits; retain/check the original mean-step rule too before
any production integration. Both independent gates must pass to implement.
OMP_NUM_THREADS=8 and TOKENIZERS_PARALLELISM=false remain fixed in all children.
All weights and losses remain synthetic; no quality or convergence claim.

## Host B rejected; prospective host C admission

Host B's first six-pair sequence stopped before pair 3 on idle writes
1,680,616.6 B/s. Its four completed arms are all void. The first fresh-process
F1 sequence stopped before its first timed arm: idle writes 1,301,333.7 B/s;
the parent's three F1 checks were ok. A separate 180-second idle observation
then found six contaminated five-second windows and at most 11 consecutive
clean windows (55 seconds). This host did not provide a stable exclusive disk
window. Preserve every record in `evidence/c2-linux-nvme-20261018/`, including
the unknown/negative outcomes. Host B was destroyed after downloading evidence.
No training-step timing ran there. No host-B rates enter another host's result.

Host C, instance 56572675 / offer 56566107 / machine 152543, is a new independent
attempt under the **unchanged** fixture, F1, validity and decision rules above.
Provider claims: whole-GPU host (`gpu_frac=1`), RTX 5080 16 GB, i5-14400F,
31,872 MB RAM, unnamed NVMe at 5.016 GB/s, 300 GB allocated disk, quoted total
$0.404444/hour. All remain subject to native inventory and physical admission.
Rebuild the same seeded synthetic checkpoint/cache; never mix the two hosts.
The overall rental cap remains $1; host C is limited to 90 minutes from its
reported start (1792336337.477 UTC Unix seconds), including setup and evidence
download. No host-C performance arm exists when this note is committed.

## Host C startup failure; prospective host D admission

Host C never launched: the provider's Docker daemon returned
`unknown or invalid runtime name: nvidia`. Destroyed without SSH, package/model
installation or measurements; API absence confirmed 2026-10-18 15:15:49 UTC.

Host D, instance 56573182 / offer 54705057 / machine 143798, is independently
subject to the same native admission and unchanged experimental rules. Claims:
whole GPU (`gpu_frac=1`), RTX 3080 Ti 12 GB, Xeon E5-2640 v4, 31,936 MB RAM,
WDS100T3X0C-00SJG0 NVMe, 300 GB allocated storage, $0.456667/hour. Pin the eleven
critical package versions recorded in host B's environment via the accompanying
`host-d-constraints.txt`; capture the resolved environment as well. Same seeded
checkpoint, cache shape and per-arm settings, rebuilt locally on this host.
Do not pool any results across hosts. Maximum host-D rental: 80 minutes from
its provider start 1792336527.509 UTC Unix seconds, including setup and download,
within the overall $1 attempt cap. No host-D performance arm exists at commit.

Prospective sensor clarification for D: use F1's unchanged default
`other_disk_reads`, which already excludes partition/RAID/LVM duplication and
also covers physical SATA disks if present. Only the uniquely admitted target
name and its read counter need injecting for overlay. Host B exposed only one
physical NVMe, so its NVMe-only sum gave the same other-disk counter (zero).
Its void outcomes remain untouched; no host-D data exist before this change.

## Explicit user pause — 2026-10-18

The user requested stopping work before shutting down the laptop. Host D's pip
installation was terminated, its partial log and native inventory downloaded,
and instance 56573182 destroyed. API confirmation at 15:22:09 UTC showed no C2
instances remaining (`pause-cleanup.json`). No host-D fixture, model or timing
was produced. No task PR was opened. Work stays paused until an explicit resume;
the durable checkpoint and remaining steps are in `c2-resume-20261018.md`.

## User resume and prospective host E — 2026-10-18

The user explicitly resumed work after the pause. Native D was no longer
available. Host E is a new independent attempt: instance 56584163, offer
56410758, machine 137733. Provider claims: whole GPU (`gpu_frac=1`), RTX 3090,
Ryzen 5 7600X, 31,189 MB RAM, WD_BLACK SN850X 2000GB, 300 GB allocated disk,
$0.611111/hour. Verify natively before admitting it. Use the same eleven pinned
versions, fixture, F1 repeat rule, sensor accounting and decision thresholds.
Maximum E rental: 55 minutes from provider start 1792341456.849 UTC Unix
seconds, including setup and evidence download. The overall attempt cap remains
$1 including earlier rentals. No host-E performance arm exists at commit.

Host E's later idle-only record showed ~13 MB writes at one-minute intervals.
A separate, prospective diagnostic study is frozen in
`c2-linux-nvme-diagnostic-20261018.md`. It retains canonical F1 and bounded write
accounting but cannot authorize production changes or satisfy strict v1.
No v1 observation is reclassified. See that protocol for exact validity rows.
