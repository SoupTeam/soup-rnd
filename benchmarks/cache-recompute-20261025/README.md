# Cached disk streaming with native retained activations

Status: full 32B numerical qualification complete; balanced timing retry in progress.

The source is upstream Soup 0.76.0 at
`51777d12077df022db8ee8a57cf46ed2f76909ef`, plus candidate commit
`c22e5f90` in the attached research checkout. The portable production patch is
[candidate-production.patch](candidate-production.patch.zip); its exact identity is
[candidate-production-patch.json](candidate-production-patch.json).
The older PR #5 implementation and historical timings are separate frozen sources.

## Candidate

Port immutable GPU weights and a bounded RAM hotset onto current upstream. With
an actual cache installed, native `stream_recompute: auto` can probe disk-tier
activation retention. The planners share the physical/free/override VRAM cap,
64 MiB setup reserve and 12 bytes per trainable parameter for future AdamW state
and workspace. A complete forward/backward validates the selected retained
suffix after the one-layer estimate. Automatic capacity failure restores
recomputation; other execution errors propagate. Probes preserve adapter values,
RNG and module modes. This is capacity admission, not globally optimal placement.

## Frozen comparison

Qwen3-32B, fixed NF4 double quantization, physical RTX 4060 Ti 8 GB and a real
native-cgroup RAM bound below 32 GiB. Context 512, batch 1, accumulation 2,
ordinary PEFT LoRA rank/alpha 4 on seven decoder projections. No compile, Liger,
FastLoRA or explicit FlashAttention change. Every arm has a 14336 MiB RAM-cache
allocation ceiling; the GPU arms keep four centred decoder layers.

| Internal arm | Actual configuration |
| --- | --- |
| baseline | RAM hotset, ordinary checkpoint recomputation |
| ram | RAM hotset and native auto retained activations |
| vram | RAM hotset and four persistent GPU layers |
| both | RAM hotset, four persistent GPU layers and auto retained activations |

The names `ram` and `vram` are inherited harness identifiers; they do not mean
that the other three arms lack a RAM cache. GPU layers are excluded from RAM
membership. Selected membership/counts are frozen after capacity qualification.

Complete primary and padded-semantics captures precede timing. Every tensor is
independently replayed with the original reference-first `rtol=.01, atol=1e-5`,
finite checks, matching initial adapters/inputs and three-decimal loss gates.
Four balanced fresh-process blocks retain all 96 updates, final adapters, losses
and memory observations. Setup and complete-entry times are separate from step time.
The >=2% practical speed gate is unchanged.

Deterministic arithmetic is enabled in all captures and timings. Deterministic
empty-allocation poison fills are explicitly disabled in every arm, before
setup, so debug allocation fills are not credited as production speed gains.
This differs from the older controlled studies and requires a fresh baseline.

## Storage and setup scope

Vast advertises NVMe, but the container exposes an XFS loopback image. Physical
NVMe backing is unverified. Current upstream correctly detects `ssd` rather
than inferring NVMe from an unrelated device; it refused the first 32B setup.
The next matched run uses the documented `training.stream_disk_kind: nvme` VM
override in every arm. This changes media admission, not arithmetic or the RAM
headroom policy. No physical-NVMe throughput or service-time claim is justified.

The source passes 479 local checks, with seven CUDA skips. The first GPU-host
control set passed all 271 checks, including CUDA ownership/promotion cases.
This is not a full-suite or hosted-CI success statement. The same-source 32B captures now pass independent replay bitwise. All 896 adapter tensors change in each arm. See [accuracy-qualification.json](accuracy-qualification.json) and the exact independent report. Balanced speed/memory timing remains pending.

Earlier setup attempts and cleanup receipts are preserved: the first deadline
could not guarantee full raw retention; a later SSH calibration stream closed;
the third attempt hit the native disk-kind policy after download. The transfer
calibration now uses hash-verified, resumable blocks with negative controls.
Every earlier instance and volume was deleted. Their conservative aggregate
charge is $1.054928; these are not 32B training or performance results.

The first complete numerical lease hit two consumer/orchestration errors after capture: configs had already been exported by the capacity bundles, and the approval filename became visible before its JSON upload completed. A separately hash-verified config view qualified the original captures without modifying any canonical archive or numerical gate. The next run fixes the export layout and uses atomic SFTP replacement; earlier immutable encoded objects are transport donors only. Every calculation is repeated on the GPU.

The exact executed config-location consumer and its four negative/positive controls are preserved byte-for-byte in [replay-consumer.zip](replay-consumer.zip), for extraction into the retained study root.
