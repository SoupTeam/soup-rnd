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

## Implementation audit and CPU evidence

The research brief's historical explanation predates R4. On this base,
`_read_layer` already submits asynchronously; `_claim_reads` limits each root
through `if root in self._pending`. `_pending` maps each root to **one** read.
Changing that condition alone would overwrite its bookkeeping: completion,
busy-slot exclusion, oldest-read liveness and pool checks all consume that map.
A depth-2 runtime candidate therefore needs a collection per root (or a
separate per-layer map), as well as tests. It is not a safe one-line change.

No runtime candidate was enabled without evidence of benefit. The new standalone
`c2_depth_probe.py` keeps the old R3' harness and record intact. Its portable
queue core owns a fixed-size executor, holds at most two spans, checks every
range before publishing a span and drains outstanding writes on errors.
The full-byte verification callback runs before that slot can be reused.
The physical harness is **prepared, not hardware-validated**; platform calls,
PDH instance naming and DMA behavior remain hardware acceptance items.

Evidence directory: [`evidence/c2-per-drive-depth-20261018/`](evidence/c2-per-drive-depth-20261018/).
`frozen-protocol.md` preserves the exact pre-measurement document bytes;
`cpu-correctness.json` records that SHA256 and the tested harness SHA256.
The JSON's `source_sha` is the protocol commit; the new harness was uncommitted
then, so its explicit file hash identifies the implementation actually run.

- TDD: the first collection failed because the new harness did not exist, before
  implementation. Then the new tests passed.
- New harness tests: **20 passed** (`c2-tests.log`). Includes a blocked slow
  range with an idle worker, full-byte checks of all spans, bounded outstanding
  ranges, slot reuse, read/submission/verification errors and their drain,
  invalid limits, missing telemetry, contaminated arms and frozen verdicts.
- Existing async source + striped-reader tests, plus the new tests:
  **93 passed, 11 skipped in 28.29 s**. Command below; the skipped cases require
  capabilities unavailable in this CPU environment. No CUDA lifetime claim.
- Synthetic CPU exercise: **38/38 full-span byte checks**, 19 spans/depth,
  four workers and 128 KiB allocated at either depth. Maximum pending spans
  1/2; maximum pending ranges 4/8. Raw JSON deliberately has **NO VERDICT**;
  its elapsed times are scheduler diagnostics, not disk-throughput results.
- `ruff check src/soup_cli/ scripts/ tests/ benchmarks/`: **passed** (`ruff.log`).

From the isolated worktree in PowerShell (the executable is an existing CPU
environment; `PYTHONPATH` selects this worktree, confirmed in `environment.json`):

```powershell
$env:PYTHONPATH = "$PWD\src"
$py = 'C:\Users\esenb\Desktop\soup-rnd-d2\.venv\Scripts\python.exe'
& $py -m pytest tests/test_c2_depth_probe.py tests/test_issue971_async_disk_source.py tests/test_disk_tier_stripe_reader.py --no-cov -q
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
& $py benchmarks/harness/c2_depth_probe.py --cpu-check --out cpu-new-sequence.json
& $py -m pytest tests/ -v --tb=short
```

Only inside an agreed quiet window on a suitable Windows stand (replace the
directory and identifier with actual agreed resources):

```powershell
& $py benchmarks/harness/c2_depth_probe.py --drive 'D:\immutable-large-store' --window 'agreed-window-id' --out c2-disk-D-sequence-1.json
& $py benchmarks/harness/c2_depth_probe.py --evaluate c2-disk-D-sequence-1.json
# After independently reviewing the saved host state and peer activity:
& $py benchmarks/harness/c2_depth_probe.py --evaluate c2-disk-D-sequence-1.json --host-reviewed
```

No hardware sequence has run. Access to a cloud account does not establish an
idle physical disk or make a Windows direct-I/O result transferable to Linux.
The available cloud instance is assigned to another campaign; it was not used
for C2. A Linux follow-up would require a separately frozen protocol and the
shipped Linux direct-I/O primitive, physical-device visibility, a >RAM fixture
and an agreed quiet window. No paid resource was rented or started for C2.

### Full-suite attempt history

The first `pytest tests/ -v --tb=short` process ended before a final summary or
exit status could be captured. Its raw journal is preserved as
`full-suite-interrupted.log.gz`; `interrupted-attempt.json` retains the last
line, 1,071 passed lines and four failed test IDs in the `.can` tests. Failure
tracebacks were not emitted before interruption, so their cause is unknown.
All four passed in the second full-suite attempt without code changes. Neither
the incomplete attempt nor a later passing rerun erases those failed lines.

During the second attempt six failures reproduced in isolation, without loading
the new C2 tests: **6 failed, 82 passed** across `test_issue1265_ab_nig.py` and
`test_issue1267_auth_status_classification.py` (`full-suite-failure-recheck.log`).
Four fail because this environment's NumPy 1.26.4 has no `np.trapezoid`; two
reach `KeyError: 'file'` in Python 3.10.0's `tempfile` while the existing error
formatter probes `urllib.error.HTTPError.response`. The affected tests and
`src/soup_cli/utils/errors.py` have identical Git blob hashes at the base and
C2 HEAD (`base-file-hashes.txt`). No dependency upgrade or unrelated repair
was made in the shared D2 environment. The full suite is therefore **not green**.

The second full-suite attempt then exited natively with **-1073740791
(0xC0000409)**, near 27%, before a pytest summary or coverage result. The log
contains **9,515 passed, 154 skipped and six failed test lines**; these are
partial counts, not a completed-suite summary. See `full-suite.log.gz`,
`full-suite-exit.txt` and `full-suite-summary.json`. The last test belonged to
`test_issue1344_sequence_distill_dropped_row_numbering.py`; that entire file
passed separately (**10 passed**, `native-exit-recheck.log`, exit 0) with
faulthandler enabled. The native-exit cause remains unknown: the last printed
test name does not establish which component crashed. The file and its distill
trainer also match the base by Git blob hash. No full-suite pass or coverage
acceptance is claimed, and another unbounded rerun is not substituted for this
failed attempt.

Windows preparation decision: **NO HARDWARE VERDICT; keep runtime depth 1**. The prepared
primitive harness and portable CPU checks are reviewable, but Windows hardware,
CUDA staging lifetime and any end-to-end training benefit remain unmeasured.

Later independent Linux follow-up: see [C2 results](c2-results-20261018.md) for
physical NVMe diagnostics and real CUDA tests. This does not retroactively
establish a Windows hardware result or turn these failed suite attempts green.
