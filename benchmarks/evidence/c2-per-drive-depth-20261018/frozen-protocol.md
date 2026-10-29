# C2: two queued layers per physical NVMe

Status: **PREPARED / NO HARDWARE VERDICT**. Runtime remains at one pending layer
per root. Base: `94981a6078054af6069cc8bc635c47e709413f3f` (`origin/main`, fetched
2026-10-18). Work is isolated on `koko/c2-per-drive-depth`; the dirty D2 checkout
is outside this worktree. No GPU or sustained disk benchmark is authorized by
the availability of this document alone.

## Frozen protocol v1 (before measurements)

Question: can removing the per-layer range barrier improve one physical NVMe's
direct-read throughput with **four workers unchanged**? This is independent of
`stream_read_ahead`, disk count, and training speed. Current `_pending[root]`
in `AsyncDiskSource._claim_reads` allows one layer per root. Roots are logical
placement labels, not proof of distinct physical disks.

The old R3' rule and record stay unchanged. Its ratios 1.106 / 1.101 / 0.963
are ambiguous. This is a new, balanced six-round experiment; no old arm counts
toward it. Run each target NVMe separately in a user-agreed quiet window.

- Windows, AC throughout; pinned, sector-aligned buffers; the existing #974
  unbuffered primitive. No cache flush, power-plan change, or peer termination.
- One fixed ordered set of 24 x 400 MiB spans from an immutable store **larger
  than physical RAM**. Both depths read these exact spans; direct I/O bypasses
  the page cache. Four sector-aligned ranges/span, four workers in BOTH arms.
- Two allocated, touched 400 MiB pinned arenas in BOTH arms (800 MiB total).
  Depth 1 uses only slot 0; depth 2 cycles slots 0/1. This keeps allocation and
  locked memory fixed. At most `depth * 4` submitted unfinished range tasks;
  at most four executing I/O calls, never eight workers.
- Before timing: complete full-byte verification of EVERY span at both depths,
  before its slot is reused. This is a separate correctness pass and excluded
  from throughput. A one-span untimed warm-up precedes each timed arm.
- Six rounds, orders 1/2, 2/1, 1/2, 2/1, 1/2, 2/1. No adaptive stopping for a
  favorable answer. One session per physical disk, fixed configuration.
- Wall time brackets submission through completion of ALL range tasks. Rate is
  total bytes / wall seconds / 1e9; queue residence times are not summed.
  After each timed arm, fully compare the last span retained in each used slot.
  Staging memory lives until all futures settle, including after exceptions.
- Save protocol SHA256, harness SHA256, commit, Python/torch/import paths,
  exact spans, physical disk identity, start/end box state, per-arm wall times,
  raw counters, byte checks, all void/partial arms and exceptions in JSON.

### Validity (evaluated before the throughput rule)

Missing evidence is **NO VERDICT**, not a zero or a pass. A byte mismatch is
**FAIL correctness**, regardless of rate. Stop and preserve the attempted
sequence on any invalid arm; do not replace individual slow arms. A fresh
window starts a new, separately named sequence, keeping the old JSON.

- Recorded user-agreed window identifier; all fixture paths map to one verified
  physical NVMe; ordinary local drive paths only (no junction/symlink/UNC).
  Record disk model/bus/number and fixture sizes/mtimes before and after.
- Fixture files unchanged; total fixture bytes exceed physical RAM; pinned and
  direct I/O remain enabled; exact workers/spans/allocations match.
- AC sampled continuously; no suspend (counter sample gap >2 s); >=4 GB free
  host RAM, commit <85% of limit at each arm boundary. GPU unused by this probe
  except initialization for pinned allocation; shared GPU work requires a
  different window. Record GPU state for human review.
- Physical-disk PDH samples every 0.25 s, >=4 valid samples inside each arm.
  Missing counters/errors void the sequence. Mean writes >1 MiB/s, or measured
  disk read rate >1.20 times the harness read rate, void the sequence. Counter
  boundary noise is why read contamination has a 20% allowance. Counter rates
  cannot prove absence of peers: the agreed window and host-state review are
  also required. Unknown peer activity => NO VERDICT.
- Any I/O error, abnormal exit, incomplete six rounds, or unreviewed host state
  => NO VERDICT. In-flight calls are drained before buffers can be released;
  a kernel I/O call that never returns requires operator diagnosis, not unsafe
  buffer disposal. This harness does not cancel native synchronous I/O.

### Decision rule (six valid paired ratios, unrounded)

| Condition | Verdict / next action |
| --- | --- |
| All six depth2/depth1 rates >=1.10 | PASS primitive; implement a bounded runtime candidate and gate it separately |
| All six within [0.95, 1.05] | FAIL benefit; keep depth 1 |
| All six <0.95 | FAIL regression; keep depth 1 |
| Anything else | NO VERDICT / ambiguous; keep depth 1 |

No training-speed claim follows from primitive PASS. Before shipping a runtime
candidate, freeze a new end-to-end gate with fixed `stream_read_ahead`, workers,
memory/slot budget, same cache and forward/backward numerical checks, then run
the cold 70B-shaped NF4 step on single and striped layouts. Slot availability
may cap useful per-drive depth; do not silently increase read-ahead to hide it.
Runtime tests must cover held pinned slots, release events, turnaround/replans,
out-of-order completion, bounded per-disk pending reads, errors, close and wedge
detection. Existing source lifetime tests remain required. Primitive CPU tests
below do not establish CUDA DMA safety for a runtime that has not been built.

## Reproduction and resource required

CPU validation needs no CUDA or model download. Commands and results are added
below without changing the frozen protocol above. Physical execution uses
`benchmarks/harness/c2_depth_probe.py`; its `--help` does not import torch or the
Windows reader. The operator supplies a reviewed window identifier and confirms
host-state review separately when evaluating saved results.

Required resource: an agreed quiet Windows/CUDA window, a pinned-memory-capable
CUDA stack, and immutable >RAM fixtures on each target physical NVMe (test one
at a time). This host's available environment is Python 3.10 / torch 2.13.0+cpu;
no hardware throughput or end-to-end result has been collected.
