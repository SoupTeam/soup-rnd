# G3 status after review (2026-11-03)

**Status: INCOMPLETE_ENVIRONMENT_BLOCKED, support NO_VERDICT.** No direct NVMe-to-VRAM read has
completed (0), no performance was measured, and no outcome in `PROTOCOL.md` (DIRECT_PATH_VERIFIED or
UNSUPPORTED_ON_TESTED_STACK) has been reached. This update adds evidence and states what can unblock
the task; it changes no frozen protocol, gate or threshold. No rental was made for it.

## 1. All rented attempts

| # | Instance | GPU | Phase | Result | Direct reads |
|---|---|---|---|---|---|
| 1 | 57253071 | RTX 3090 | preparation | native build failed (UUID API, fixed in `50cbebd`) | 0 |
| 2 | 57256165 | RTX 3090 | preparation | SDK compile pass; target topology blocked | 0 |
| 3 | 57258175 | GeForce RTX 4090 | controlled capability | both staged-copy controls pass; cuFile 5001 (driver not initialized) | 0 |
| 4 | 57264221 | A40 | read-only discovery | md RAID data mount; no nvidia_fs, gdscheck, stats header | 0 |
| 5 | 57268242 | A100-SXM4-40GB | read-only discovery | local NVMe, but no nvidia_fs, gdscheck, stats header | 0 |
| 6 | 57268755 | L40S | read-only discovery | md RAID data mount; no nvidia_fs | 0 |
| 7 | 57325630 | DGX A100 | read-only discovery | nvidia_fs loaded with counters, but data on XFS over /dev/loop0; no gdscheck | 0 |
| 8 | 57830497 | A100 | read-only discovery | local NVMe; readiness gate rejects: nvidia_fs route, counters and gdscheck GPU diagnostic missing | 0 |

Attempts 1-6: `reports/20261023/`. Attempt 7 and a P2PDMA review: `reports/20261024/`. Attempt 8:
`reports/20261026/attempt_result.json` (readiness gate `REJECT_CANDIDATE`, 0 CUDA allocations). Every
instance and volume was preserved locally and its deletion verified. All eight are environment gaps; none
is a stack-attributable native rejection, so none is the preregistered unsupported outcome.

## 2. Why rented containers are unlikely to close G3

From the NVIDIA GPUDirect Storage release notes (checked 2026-11-03,
<https://docs.nvidia.com/gpudirect-storage/release-notes/index.html>). These are vendor statements, not
measurements, and are not used as a verdict:

- GDS mode is supported on Data Center and Quadro (desktop) cards; all other cards, which by the note's
  wording includes GeForce, are supported only in compatibility mode (POSIX host staging).
- cuFile APIs in cloud-service-provider environments are described as compatibility mode.
- The nvidia-fs-free NVMe P2PDMA mode needs the open kernel module, kernel >= 6.2 (Ubuntu 22.04) on
  x86_64, upstream NVMe without RAID0 or multipath, and may need `nokaslr`. These are host settings that a
  container tenant cannot change, and the protocol forbids changing them as an automatic repair.

The consumer target in the source brief (an RTX laptop GPU) is therefore expected to have no native GDS
route on Linux, but the protocol requires a reproducible native rejection on a real stack to say so.

## 3. What would reach a preregistered outcome

Both paths need a machine whose owner has administrator rights; no new rental is proposed.

1. **Linux, GDS installed by the machine's owner** (official `nvidia-gds` packages, local single NVMe with
   ext4/xfs). The frozen `direct_probe.cpp`, controls and readiness gate run unchanged (`RUNBOOK.md`).
   On a GeForce card a driver-initialised cuFile that rejects the device buffer, with both staged-copy
   controls passing, is UNSUPPORTED_ON_TESTED_STACK scoped to that stack; on a Data Center or workstation
   card the same run can reach DIRECT_PATH_VERIFIED and then the A-B-B-A comparison against #974.
2. **Windows DirectStorage**, named in the source brief for the consumer box. This is a different API and
   transport (DirectStorage reads into a GPU-visible destination through runtime-managed staging, then a
   D3D12-to-CUDA import); it needs its own prospective protocol, committed before any run, with the same
   staged-copy positive control, full byte oracle and separate CPU / explicit host-memory accounting. It
   cannot by itself establish a peer-to-peer route.

Two physical drives (`TWO_DRIVE_TOPOLOGY.txt`, `g3_pair_topology.py`): offline gate only; it rejects two
files, partitions or namespaces on one device. The concurrent runner is not implemented.

## 4. Checklist

| Item | Status |
|---|---|
| Check DirectStorage (Windows) or GDS (Linux) for compute buffers | GDS: 8 Linux leases, all environment-blocked; DirectStorage: not run |
| Short NVMe-to-VRAM test if available | 0 direct reads |
| Compare with the unbuffered (#974) read | staged-copy controls pass (attempt 3); no comparison |
| Measure bandwidth, CPU, pinned RAM | not measured |
| Reproducible unsupported verdict if no support | not reached: setup gaps are not hardware non-support |

Machine-readable: `checklist_status.json` in this folder.
