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
