# C3: partial dense weights in RAM

Status (2026-10-31): dense Llama 3.1 8B CUDA numerical parity is confirmed for
completed arms; **task acceptance incomplete / performance NO_VERDICT**.
Attempt019 passed all six CUDA controls, including the public RAM-only C3+L2L
Trainer control. It completed 22/80 planned arms with exact loss/gradient/final-adapter
parity, then stopped early: arms14 and19 failed the unchanged F1 foreign-read gate.
Both resources are deleted and all186 evidence files verified on Mac and external
storage. See `reports/20261031_attempt019/REPORT_RU.txt`.
Raw per-arm, calibration, foreign-reader and CUDA-control evidence for attempts 017 and 019,
with the frozen source manifests, is in `evidence/`; `python benchmarks/c3/replay_evidence.py
benchmarks/c3/evidence/<packet>` recomputes the gates without a GPU (see `evidence/README.md`).
Attempt017 retains NO_VERDICT_INSUFFICIENT_REPETITIONS; attempt018 remains a
supervisor-admission failure with zero measured arms. Neither history is rewritten.
Branch: `research/c3-partial-ram`. Pilot source commit:
`5bcd74b4792ec6b3580babf9aaf0266d30dff2c0` (attempt010). Earlier pilot report published in `7c061bc84`;
no PR creation or merge is claimed here.

## Implemented

- Opt-in `training.stream_partial_ram` on the explicit disk path for unquantized
  dense Llama Transformers SFT. Default configurations retain their prior behavior.
- Whole decoder layers selected under an explicit cache allocation cap. Cached
  paths are absent from the async reader, including readahead. Large boundary
  weights keep their existing streaming behavior.
- Host envelope checks reservations, resident allocation, staging allocation and
  loading transient before allocation. Pinned arena rounding is charged. Unsupported
  combinations are rejected; the envelope is not a process-RSS cap.
- Source close waits for recorded cached transfer events and releases RAM/disk
  resources. Construction failure also releases an already-created source.
- CLI and runtime statistics distinguish resident tensors, staging and allocation
  accounting. Logical RAM-served tensor bytes are not physical disk-byte savings.

## Local evidence

`local_validation.json` identifies the base commit, commands and source SHA256s.
The combined C3 and regression selection passes **642 tests, 24 skipped** on macOS.
CUDA cases are among the skips; this is not GPU correctness evidence. Changed
Python files pass Ruff and `git diff --check` passes. Full repository CI was not run.

A synthetic four-layer dense Llama, fp32 CPU, LoRA q/v projections and AdamW runs
three updates in disk-only and partial-resident modes from the same seed. Losses,
every trainable gradient and final trainable tensors are bit-identical; weights
actually change. File-backed source tests exercise forward/backward order,
turnarounds and zero/partial/full residency. An instrumented reader never opens
cached layer paths during those source requests. A real SFT trainer setup test
confirms the configuration reaches the new source and preserves reservations.

Injected cache-read and device-buffer-construction failures verify cleanup. Pinned
allocation arithmetic and fake-event lifetime tests run without a CUDA device;
they do not establish real pinned H2D correctness. An intermediate combined test
run aborted in the test-injected destructor recorder during teardown. That recorder
was changed to retain booleans rather than resurrecting the closing source object;
the focused and regression runs reported above then completed. No failed run is
counted as passing.

## Trello acceptance mapping

Card: https://trello.com/c/h3rVZVsO

| Requirement | Current evidence / remaining work |
|---|---|
| Choose a dense model | Frozen pretrained Llama 3.1 8B BF16 completed 20 CUDA arms on controlled text in attempt017. |
| Budget after buffers and L2L | Allocation envelope and shared A4 low-level budget/lifetime bridge pass CUDA zero/partial/full spill controls. Public RAM-only trainer integration passes local tests and attempt019's actual tiny-model CUDA Trainer parity/accounting control. This is not an 8B L2L performance claim. |
| Keep selected weights resident | Attempt017 retained 8/32 real 8B decoder layers with pinned H2D; exact loss/gradient/final-adapter parity across 20 arms. |
| Free resources and enforce limits | Local failure/overbooking tests plus all five CUDA controls pass; low-level shared leases return on close. Public trainer cleanup also passes the new sixth CUDA control in attempt019; failed-drain retention is covered locally. |
| Compare training, disk bytes and speed | Attempt017: 8B numerical parity, NVMe calibration, foreign-reader rejection and all F1/clock gates pass; process physical reads decrease 25%. Observed 42.53 to 57.49 tokens/s is descriptive only: repetitions require 20 blocks, but 5 ran. Performance NO_VERDICT. Attempt019 confirmation was aborted after F1 rejected foreign reads in two arms. |

Do not mark the entire card complete based on CPU evidence. No dynamic eviction,
NF4 validation, accepted A4 integration or throughput claim is included.
CUDA correctness now includes the frozen pretrained dense 8B comparison in attempt017;
its controlled text is not a model-quality evaluation.

A proposed `HostMemoryBudget` now atomically leases bytes to actual weight and
resident/spilled activation consumers. Tests cover overbooking, concurrent requests and
release. `BudgetedActivationStore` now accounts for both aligned staging rings and drains
workers/transfers before releasing its lease. The runtime accepts the same budget
owner. The public RAM-only L2L path now reserves a maximum activation partition before C3 weights;
its end-to-end Trainer parity test passes locally and on CUDA in attempt019 (tiny model).

## Remaining acceptance work

1. Obtain a valid separate dense-8B confirmation series on a sufficiently quiet
   physical NVMe host. Keep 20 pre-registered ABBA blocks and all numerical/F1/clock
   rules unchanged. Attempt019 cannot be repaired by dropping its void arms or
   appending runs. A frozen count is not a guaranteed performance pass.
2. Preserve and review the new evidence, publish only the measured configuration,
   and prepare the PR with full validation limitations. Public RAM-only C3+L2L
   functionality now has a real CUDA control; public spill/compression and 8B L2L
   throughput remain outside this change.

The current cumulative cap is USD6. After attempt019, conservative planning spend
is USD4.14, leaving USD1.86; dashboard values are not invoices. No active C3 lease.
`CONFIRMATION_PROTOCOL.txt` is the preserved pre-run attempt019 plan, not a claim
that another run has started. Its historical budget figures must not be reused.

The original authorization was USD5 for C3 (subsequently raised to USD6 as above). The frozen `GPU_PROTOCOL.md` defines one
bounded synthetic CUDA pilot; its outcome must be recorded separately. The 2026-10-26 series has separate task budgets. Attempt010 instance58131635 and volume58131634 are deleted; provider absence is confirmed in report closure_status.json. A changelog draft is provided separately; use an actual PR number when
creating the required release fragment, rather than inventing one now.

## Paid attempt 001

Container startup exceeded 600 seconds; GPU controls/arms: 0. After Stop, both
instance 57508623 and volume 57508622 were destroyed with user confirmation;
provider lists confirmed zero instances and zero volumes.
See `reports/20261024/REPORT_RU.txt` and `attempt_001.json` for evidence and closure.
No GPU or performance pass is claimed.

## Preparation recovery, 2026-10-26

Attempts002/003 are preserved in `reports/20261026/`; both instances and volumes
were deleted. Attempt003 reached a working RTX4090/NVMe container but setup exited141
before CUDA controls. The exact SIGPIPE trigger remains unresolved.
`SETUP_RECOVERY.txt` describes the prospective bounded supervisor and its local
process-lifecycle tests. Source freeze/review and a new attempt are required;
no failed run is resumed. See `commit_review.json` for the latest local checks.

## Completed synthetic CUDA pilot, 2026-10-26

Attempt007 ran on RTX4090 with actual NVMe/XFS and O_DIRECT, without buffered
fallback. Two CUDA controls passed without skips. Five ABBA blocks produced
20 runs, 120 total optimizer updates and 80 measured steps. Losses, gradient
hashes and final adapter hashes matched exactly; adapters actually changed.
Keeping two of eight layers resident reduced successful application read bytes
from 729917472 to 547438104 per measured step (25%). Initial RAM fill and headers
are excluded. These are not physical device sector counts.

Observed step medians were 0.1870s and 0.1582s. No accepted speedup or VRAM
saving follows: physical-read calibration and foreign-reader injection are
missing, and process-lifetime RSS is not an isolated per-arm peak. All 47 files
were preserved and SHA-verified. Instance57853606 and volume57853605 were
deleted with provider absence confirmed. See `reports/20261026_attempt007/`.

Prospective measurement instrumentation is now implemented in `isolated_compare.py`,
`isolated_arm.py`, `io_calibration.py` and `foreign_read_control.py`, under
`MEASUREMENT_PROTOCOL.txt`. Each arm runs in a fresh supervised process. Physical
NVMe counters must pass direct-read calibration and controlled foreign-reader
rejection before any CUDA arms. Setup/cache-fill counters are separate; RSS is
explicitly the whole process lifetime. The local focused suite passes 72 tests
with 2 CUDA skips. Measurement source was published in `7c061bc84`; the nested-deadline
fix was published in `238577099`. Attempt008 passed remote physical-read calibration
and foreign-reader rejection but blocked before measurement updates on the launch
layout. Attempt009 failed during setup while installing uv; zero GPU measurement
runs. Both attempts were preserved and their new resources were deleted. See the
20261028 attempt008/attempt009 report packets. No performance pass is claimed.

A prospective offline setup now has all 108 hash-pinned Linux wheels and the pinned
CPython3.11.16 archive locally verified. Its 225 active dependency edges passed the
metadata closure check. On 2026-10-28, a complete copy to the reconnected USB drive
passed full per-file SHA readback. Published offline setup installed successfully on Linux in attempt010, followed
by two CUDA controls and the bounded measurement pilot. See `OFFLINE_SETUP_PROTOCOL.txt` and
`offline_preparation_review.json`.

At the time of attempt010, representative workload and shared A4 reservation/lifetime
integration remained open. Attempt017 subsequently validated the dense workload and
low-level bridge; public integration and performance acceptance remain open.


## Measurement pilot010, 2026-10-28

The pinned offline Linux setup and two CUDA controls passed. Physical NVMe
read-counter calibration and deliberate foreign-reader rejection passed before
the workload. Five ABBA blocks ran in 20 isolated processes: 120 optimizer
updates, 80 after warmup. Losses, gradients and final adapters matched exactly.
All 20 arms passed exclusive GPU ownership, F1 and complete clock checks.
Local archive/member/restored-file SHA verification and frozen numerical review
passed. Full observations and measurement boundaries are in
`reports/20261028_attempt010/REPORT_RU.txt` and `local_review.json`.

This is a synthetic 122M FP32 pilot. Descriptive step times exclude numerical
hashing, whereas I/O includes it; RSS includes model construction/cache fill.
Do not generalize these observations to a representative model, claim an isolated
RSS reduction, or declare the C3/A4 integration complete. Performance NO_VERDICT
and the remaining representative-model/shared-reservation acceptance gates stay
open. Resource deletion status is recorded separately in closure_status.json.

## Prospective shared-budget and dense-8B validation, 2026-10-30

The uncompressed bridge matches 128 captured layouts from the published A4
planner at d6f7f733e09599327c61e67a3cf5434bfaa78164. Tests exercise actual
layer-major backward with partial cached weights and zero/partial/full activation
spill, exact loss/gradient/adapter parity, and cleanup failures. CUDA variants
require pinned buffers and direct I/O; CPU results do not establish those properties.

`REPRESENTATIVE_PROTOCOL.txt` and the frozen model/token manifests define the next
comparison. The prepared-input runner passes a real fresh-process CPU fixture,
rejects changed inputs, and preserves the synthetic pilot's default behavior.
This was the pre-run status on October30. Attempt017 below subsequently completed
the 8B workload. Public C3+L2L remains disabled.
See `shared_budget_local_review.json` for exact validation/source hashes, including
known baseline failures in the broader regression selection.

## Attempt013 and local alignment recovery, 2026-10-30

On a physical Samsung990PRO NVMe/XFS host, offline setup passed and CUDA controls
returned 2 passed / 3 failed. All three new L2L controls rejected buffered fallback:
the pinned staging base address was not sector-aligned. No representative arm or
8B download started; no paid retry was made. All25evidence files and their archive
passed SHA verification. Instance58432891 and volume58432890 were destroyed.
Displayed cost was USD0.15 +0.01; recent billing may be delayed.

The subsequent local correction reserves worst-case base-alignment slack in each
pinned staging arena before allocation, then aligns the absolute address. Existing
default RAM-store planning is unchanged. CPU regression tests exercise deliberately
misaligned bases and preallocation budget refusal. This correction was subsequently validated by all five CUDA controls in attempt017;
it does not change attempt013's BLOCKED result.
See `reports/20261030_attempt013/REPORT_RU.txt`. Performance remains NO_VERDICT.

## Attempt014 path admission (2026-10-30)

Physical NVMe and offline setup passed. CUDA controls returned 1 passed / 4 failed:
the new basetemp was outside HOME/CWD/TMPDIR. The existing shard containment guard
correctly rejected it before alignment validation. No model download or measurement
ran. All 26 evidence files and external archive passed SHA verification; instance
58437082 and volume 58437081 were destroyed. Performance remains NO_VERDICT.
See reports/20261030_attempt014/REPORT_RU.txt. A prospective local path correction
keeps control fixtures under repository CWD; it does not amend this failed run.

## Attempt015 upload timeout, 2026-10-30

Physical Samsung990PRO NVMe/XFS and SSH preflight passed. The one-shot offline
bundle upload exceeded1200s after2429982720 of3288719360bytes. No setup, CUDA
controls, model download or representative arms ran; no retry was made. All16
evidence files and archive passed local SHA and external readback verification.
Instance58440604 and volume58440603 were destroyed with user confirmation;
both are absent from provider lists. Displayed cost USD0.17 +0.01 =0.18,
subject to billing delay. The lease guard was deleted. See
`reports/20261030_attempt015/`. Performance remains NO_VERDICT.

## Attempt016 startup failure and attempt017 dense CUDA validation, 2026-10-31

Attempt016 failed host container image extraction (no space left on device), before
any setup/GPU work. All17diagnostic files were preserved; instance58444781 and
volume58444780 are deleted. See `reports/20261031_attempt016/`.

Attempt017 used a new RTX4090/64GB host with physical NVMe/XFS. Direct fetching of
109 pinned artifacts and setup passed, followed by five CUDA controls without skips.
The shared-budget L2L controls cover zero/partial/full activation spill with pinned
buffers/O_DIRECT, exact gradients/updates and lease cleanup. This validates the
low-level bridge, not public C3+L2L trainer integration.

The frozen pretrained dense Llama3.1 8B BF16 controlled-text comparison completed
five ABBA blocks,20fresh processes,120optimizer updates and80measured steps. All
losses, gradients and final adapters matched exactly; adapters changed. NVMe
calibration, injected-reader rejection, all20F1/clock/ownership gates passed. Eight
of32layers stayed in RAM. Physical process reads fell from26,173,685,760 to
19,630,264,320bytes/step (25%). Instrumented useful tokens/s were42.53 and57.49.
The observed ratio1.3518 is descriptive only: unchanged repetitions require20blocks,
not the5available. Performance remains NO_VERDICT_INSUFFICIENT_REPETITIONS; no
post-hoc extension or accepted speedup is claimed. RSS includes setup/cache fill;
VRAM peaks were unchanged. See the report for allocation rounding and timing scope.

All167evidence files are verified internally and in a second external copy. The
external archive SHA matches; a first ExFAT extraction produced a calibration
file with a mismatching hash (cause undetermined) and is retained separately with an incident record. Instance58496976 and volume58496975 were deleted after exact-ID confirmation;
provider absence is confirmed and the lease guard deleted. Displayed usage is
USD0.26+0.03=0.29 (recent billing may be delayed). Conservative series spending
including rounding margin isUSD3.60 of5.00.
See `reports/20261031_attempt017/closure_status.json` for current cleanup status.
No commit/push/PR or independent review is claimed. Whole-card acceptance remains
open on public integration and sufficient performance repetitions.
