# C3: partial dense weights in RAM

Status: local implementation tested; **task acceptance incomplete / NO_VERDICT on performance**.
Branch: `research/c3-partial-ram`. No commit or PR has been created for this change.

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
| Choose a dense model | Synthetic dense Llama is the CPU control; representative model/run still needed. |
| Budget after buffers and L2L | Explicit static reservations and staging checks implemented; shared A4 contract unresolved. L2L combination is refused. |
| Keep selected weights resident | Implemented and tested with real CPU tensors and safetensors. |
| Free resources and enforce limits | Allocation refusal, close and injected failures tested locally; actual CUDA transfer lifetime pending. |
| Compare training, disk bytes and speed | CPU training parity passes; physical SSD accounting and paired GPU throughput not measured. |

Do not mark the entire card complete based on CPU evidence. No dynamic eviction,
NF4 validation, accepted A4 integration, CUDA correctness or throughput claim is included.

A proposed `HostMemoryBudget` now atomically leases bytes to actual weight and
RAM-only activation consumers. Tests cover overbooking, concurrent requests and
release. `BudgetedActivationStore` is an experimental bridge, not A4 spill-scheduler
integration; the public L2L combination remains refused.

## Next validation, requiring a separate agreed run

1. Connect A4's published reservation ownership/lifetime contract before enabling
   L2L; test simultaneous consumers and failure cleanup.
2. Freeze a representative dense model, data/order, dtype, seed and short step
   budget. Compare disk-only and partial residency on the same GPU/NVMe host.
3. Check losses/gradients/updates under declared tolerances, pinned allocation,
   peak RAM/VRAM, shutdown and allocation-failure behavior.
4. Measure startup/cache-fill separately from steady-state training; capture F1
   power/suspend/foreign-I/O controls, physical device counters, logical reader
   bytes and useful tokens/time. Report all attempts and exclusions.
5. Review evidence before declaring speedup or updating the Trello checklist.

User separately authorized USD5 for C3. The frozen `GPU_PROTOCOL.md` defines one
bounded synthetic CUDA pilot; its outcome must be recorded separately. The 2026-10-26 series has separate task budgets; no live C3 resources remain. A changelog draft is provided separately; use an actual PR number when
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
