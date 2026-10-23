# G3: direct NVMe to CUDA device memory

**Status: prospective recovery revision, awaiting a new owner commit.** The inventory, native primitive,
sensor controls, one-device comparison, raw-byte audit and safe exporter are
implemented and CPU-tested. Native SDK/hardware execution is still pending.
No model training is needed. No production streaming integration is included.

The first preparation at source commit `c611499fa064310028e764b346497fa7e605f9d3`
failed compilation on CUDA 12.8.1: `cudaDeviceGetUuid` is not a CUDA Runtime API.
No transfer arm or sensor calibration ran. Its raw diagnostics and source snapshot
are preserved; the outcome is ENVIRONMENT_BLOCKED / NO_VERDICT. This revision
uses `cudaGetDeviceProperties(...).uuid` and adds an optional real-SDK syntax
compilation test. CPU substitutes alone do not establish SDK compatibility.
Any subsequent attempt must declare its new source commit, retain the prior
failure and account for prior G3 spending within the original $3 authorization.

## Scope and references

Source: `rndTASK.pdf`, page 14, as transcribed in the owner's
`rndTASK_agent_context(1).md`; house rules on pages 17–18. Trello G3 is the
working decomposition: <https://trello.com/c/tK9xPzWP>.

The source asks for a primitive probe on one and two drives against the #974
unbuffered primitive. Two files on one drive are not two independent drives.
If only one physical NVMe is available, report that scope and leave the second
configuration pending. Do not silently mark it tested.

Official references checked 2026-10-22:

- [NVIDIA GDS configuration and benchmarking](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html).
- [cuFile API and compatibility mode](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html).
- [GDS troubleshooting](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html).
- [CUDA Runtime device properties](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html).

cuFile calls can fall back to POSIX host staging. An installed API or a correct
read therefore does not prove direct DMA. Newer supported configurations can
use PCI P2PDMA without nvidia-fs; absence of that module is not an unsupported
verdict. No kernel unload, reboot, driver installation, system configuration
edit, IOMMU/security change or new access grant is part of the collector.

## Phase 1: non-measuring inventory

Run from a source checkout on the intended Linux host, with an existing data
directory on the actual target filesystem:

```sh
python benchmarks/harness/g3_capability.py \
  --data-path /path/on/target/nvme \
  --out /path/to/new/g3-inventory-001
```

The collector queries GPU identity, PCI topology, the target mount, block
devices, and installed `gdscheck -p`. It records bounded raw diagnostics,
return codes, timeouts, selected cuFile environment settings, the selected
config file, boot ID, modules and available nvidia-fs statistics. It does not
load torch/CUDA, read fixtures, install dependencies or perform transfers.
Every command has a 15-second limit; output is capped and truncation is marked.
Existing output directories are refused. SHA256 covers stored JSON and source.

Inventory states are deliberately **not** support verdicts:

| State | Meaning |
| --- | --- |
| OUT_OF_SCOPE_PLATFORM | This collector targets Linux. A Mac run says nothing about Windows DirectStorage or a future Linux host. |
| ENVIRONMENT_BLOCKED | A required diagnostic is absent, failed, timed out, empty or truncated. Missing packages do not prove unsupported hardware. |
| CONTROLLED_PROBE_REQUIRED | Diagnostics were collected; review them and run the native, controlled probe. This is not a positive capability gate. |

All three retain `support_verdict=NO_VERDICT`, `direct_path_verified=false` and
`performance_measured=false`. Even a gdscheck line saying supported/unsupported
is retained for review rather than promoted to an end-to-end verdict.

## Phase 2: native capability gate (hardware execution pending)

Before execution, freeze exact source, toolchain, cuFile/CUDA versions, device
UUID, filesystem/device topology, fixture size/alignment/seed, byte oracle,
control commands, diagnostic interpretation and timeout. Commit the rule before
any transfer arm, following `benchmarks/METHOD.md`. The discovery inventory may
precede that commit because it is explicitly not a transfer measurement.

1. Generate nontrivial deterministic bytes into an exclusive, regular fixture
   file on the target mount. Retain the fixture hash and exact byte count.
2. Positive control: Soup's existing #974 direct-I/O read into correctly aligned
   pinned CPU staging, followed by H2D and completed CUDA synchronization. The
   C++ primitive follows the aligned POSIX read semantics of
   `safetensors_reader.open_direct` / `read_range_into`; it never permits a
   buffered fallback.
3. Candidate: same aligned file/range into a CUDA **device** allocation, with
   POSIX compatibility fallback disabled in a per-process cuFile config. Capture
   the effective config and the native return codes. No managed/host pointer
   may be passed off as VRAM. Preserve allocator and buffer registration facts.
4. Copy the result back **outside the timed interval** and compare every byte
   against the independent fixture oracle, for both paths. Include a deliberately
   corrupted byte control to prove the integrity check fails.
5. Prove which I/O path actually ran using version-appropriate driver/cuFile
   diagnostics or counters. Missing counters, unsupported diagnostic formats,
   compatibility activity, or ambiguous transport attribution are NO_VERDICT.
   The parser accepts exact per-process level-3 read-byte/path counts for the
   selected UUID; it rejects POSIX, dynamic-route, unaligned and error activity.
6. Preserve failed native attempts and successful controls. Do not automatically
   repair a failed stack or rerun a timed arm under changed settings.

| Capability outcome | Required evidence |
| --- | --- |
| DIRECT_PATH_VERIFIED | Correct device bytes, positive control passes, direct transport is attributable, compatibility fallback excluded. |
| UNSUPPORTED_ON_TESTED_STACK | Reproducible native rejection attributable to this specific GPU/filesystem/driver stack, with the ordinary transfer control passing. Scope the conclusion to the tested stack. |
| ENVIRONMENT_BLOCKED | Missing dependency, permission, installation, inaccessible device or unresolved setup problem. Not a hardware verdict. |
| NO_VERDICT | Failed integrity/positive control, ambiguous route, invalid evidence or contaminated required checks. |

A demonstrated stack-specific negative is a legitimate G3 deliverable. A Mac
inventory or a missing `gdscheck` executable is not that deliverable.

## Phase 3: performance only after a positive capability gate

A = #974 unbuffered NVMe → pinned host → device; B = verified direct device read.
Use matched byte counts, offsets, request geometry, concurrency and cache policy.
Time completed reads plus completed H2D for A, and completed device reads for B;
do not compare CPU enqueue latency with GPU completion time. Keep allocation,
fixture generation, integrity readback, instrument setup and teardown out of the
steady-state interval. Process/arm wall time and native bracket time are distinct;
their difference is not an isolated setup microbenchmark.

Required validity rows, before **each** arm:

| Row | Required evidence | Invalid result |
| --- | --- | --- |
| Same stack | GPU UUID, boot ID, source, runtime and target mount/device match the frozen authority | NO_VERDICT |
| Byte validity | Exact requested/completed bytes; independent full oracle and corruption control | NO_VERDICT |
| Native transport | No compatibility fallback; correct device buffer and attributable direct I/O | NO_VERDICT |
| Storage identity | Actual local NVMe topology; single vs two physical devices distinguished | NO_VERDICT |
| F1 | suspend, power and foreign_reads all ok; sensor controls accepted on this host | NO_VERDICT |
| GPU conditions | GPU peers, free VRAM, SM clocks/event reasons recorded per arm | NO_VERDICT on missing evidence or unapproved peers |
| Host memory | Free RAM/commit, process CPU time, explicit pinned-host bytes, and raw VmLck/VmPin samples | Do not claim total pinned-host savings: internal driver allocation is unknown |

Use fresh-process A-B-B-A blocks, five pilot blocks and the repeats rule in
`METHOD.md`, preserving void arms. Each performance process performs two warm-up
and three timed/byte-checked reads, returning the median; capability uses one
read per process. The maximum is ten total blocks, with a 900-second phase
watchdog and 120-second child limit. Run no extra blocks after the one-top-up
stopping rule completes. Unknown required sensors produce NO_VERDICT. Set absolute
process/run/lease limits and a cumulative cost cap before any paid window. If
additional required repeats cannot fit, retain the pilot and report NO_VERDICT;
never lower the validity bar to fit a deadline.

Report measured primitive time, derived GB/s, CPU seconds and explicit pinned
host bytes separately. Retain raw kernel VmLck/VmPin samples; they do not prove
total internal pinned allocation or its peak. Resident GPU pages registered for
DMA are not pinned **host** RAM. No total-RAM saving is claimed by this version.
No unconditional speedup, training improvement or giant-model claim follows
from this primitive. No integration into the shipped source before the positive
gate. Do not assume availability of a shared GPU or stop other people's jobs.

## Execution scope and current completion boundary

See [RUNBOOK.md](RUNBOOK.md) for exact prospective commands and authority format.
Topology resolution must establish one visible local NVMe namespace and an
ext4/xfs target, not merely a provider's SSD label. Missing tools/configuration
are environment gaps. A failed/missing gdscheck alone is not a prohibition on
our native probe once the selected GPU, native build and target topology are
established; only actual controlled evidence can determine the capability result.
No driver/security settings are changed by this package.

Native statistics API/header support is build-checked on the target SDK. CPU
substitutes validate branches and bytes but are explicitly synthetic and rejected
from hardware verdicts. Rules and source must match the owner-committed Git
objects before a native transfer. The binary and every evidence member are
SHA-pinned and locally read back. Capability uses one control/direct/direct/control
block; performance is allowed only after a same-host positive raw-byte audit.

The one-device path is implemented, not measured. Two-drive concurrent performance
and complete internal pinned-host accounting remain outside this implementation;
a positive single-device run cannot close those items. A reproducible native
unsupported outcome on the tested stack is the negative-deliverable branch
explicitly allowed by G3. No broad unsupported-GPU claim follows.

One lease and a failed native SDK compilation have occurred at the original
commit above; no GPU transfer or PR has been performed. The recovery source is
not yet committed or hardware-validated. A PR-numbered changelog fragment is prepared only once the PR
number exists. Old G6 evidence and the paused E2 work are unaffected.
