# Cached disk streaming with native retained activations

Status: completed; full numerical and balanced timing replay passed.

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
The matched runs use the documented `training.stream_disk_kind: nvme` VM
override in every arm. This changes media admission, not arithmetic or the RAM
headroom policy. No physical-NVMe throughput or service-time claim is justified.

The source passes 479 local checks, with seven CUDA skips. The first GPU-host
control set passed all 271 checks, including CUDA ownership/promotion cases.
This is not a full-suite or hosted-CI success statement. The same-source 32B captures now pass independent replay bitwise. All 896 adapter tensors change in each arm. See [accuracy-qualification.json](accuracy-qualification.json) and the exact independent report. Balanced speed and memory timing is independently replayed below.

Earlier setup attempts and cleanup receipts are preserved: the first deadline
could not guarantee full raw retention; a later SSH calibration stream closed;
the third attempt hit the native disk-kind policy after download. The transfer
calibration now uses hash-verified, resumable blocks with negative controls.
Every earlier instance and volume was deleted. Their conservative aggregate
charge is $1.054928; these are not 32B training or performance results.

The first complete numerical lease hit two consumer/orchestration errors after capture: configs had already been exported by the capacity bundles, and the approval filename became visible before its JSON upload completed. A separately hash-verified config view qualified the original captures without modifying any canonical archive or numerical gate. The completed timing run fixes the export layout and uses atomic SFTP replacement; earlier immutable encoded objects are transport donors only. Every calculation is repeated on the GPU.

The exact executed config-location consumer and its four negative/positive controls are preserved byte-for-byte in [replay-consumer.zip](replay-consumer.zip), for extraction into the retained study root.

Full numerical replay, original control XMLs and all adapter-update audits are retained byte-for-byte in [accuracy-evidence.zip](accuracy-evidence.zip), indexed by [accuracy-evidence-manifest.json](accuracy-evidence-manifest.json). Compact qualification stays directly reviewable above.

## Results

| Metric | RAM cache | RAM + activations | RAM + GPU weights | All three |
| --- | ---: | ---: | ---: | ---: |
| Optimizer step, s | 7.5863 | 7.3974 | 7.5453 | 7.4613 |
| Native setup, s | 7.8855 | 18.1393 | 21.2067 | 29.7367 |
| Fresh six-update entry, s | 57.7495 | 65.5199 | 69.3144 | 77.2552 |
| Peak allocated VRAM, GiB | 3.6444 | 5.0998 | 4.7049 | 5.0925 |
| Peak reserved VRAM, GiB | 3.8184 | 5.5469 | 5.0371 | 5.5020 |
| Peak process RSS, GiB | 19.8231 | 19.8989 | 19.8949 | 19.8855 |

`baseline_to_ram`: paired step-time reduction +2.512%, 95% interval [+2.404, +2.579]%; practical gate passed: True.

`baseline_to_vram`: paired step-time reduction +0.537%, 95% interval [+0.445, +0.637]%; practical gate passed: False.

`baseline_to_both`: paired step-time reduction +1.682%, 95% interval [+1.551, +1.706]%; practical gate passed: False.

`ram_to_both`: paired step-time reduction -0.849%, 95% interval [-1.055, -0.720]%; practical gate passed: False.

The fastest arm passing the prospective practical gate is `ram`. This is a winner among these fixed configurations, not a globally tuned placement policy.

Intervals are pointwise descriptive whole-block bootstrap intervals. Short complete entries include setup and may be slower. These are same-NF4 numerical results, not task-quality equivalence, convergence or a universal speedup.

This current-upstream study uses native BF16 adapter storage (896 tensors), rather than the explicitly FP32 adapters from the earlier 4B BF16/NF4 quality pilot. Automatic retained-activation probing is upstream functionality; the candidate connects it to the cached disk tier and shares memory admission with persistent GPU weights.

All 30 model stages were independently checked against the enforced 29.756 GiB cgroup RAM bound, with zero observed container swap. Host swap was available; it is not described as disabled.

## Separate diagnostic profile

One instrumented baseline update spans 7.613 s. Source-get brackets cover 0.953 s, of which 0.0327 s has no observed GPU activity. GPU-copy and kernel interval unions are 5.212 s and 7.016 s; name-matched NF4 dequantization covers 1.281 s. These intervals overlap and are not additive costs or an estimate of removable time. This cache baseline does not demonstrate disk starvation; host-to-device transfer and NF4 dequantization are candidates for a controlled follow-up. Profiled elapsed time is not used in the unprofiled performance verdict.

Conservative new-study charge including every retry: $2.390093; remaining allowance $7.026460. Every owned instance and volume is deleted. Full raw tensors/archives remain in the retention directory indexed in [evidence-manifest.json](evidence-manifest.json).
