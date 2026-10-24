# Streaming research plan

Updated 2026-10-24. Destination: SoupTeam/soup-rnd PR #5.

## Goal

Reduce real LoRA training time for models exceeding GPU memory by coordinating GPU,
RAM and SSD storage. Select allocations for measured speed with memory headroom and
measured task quality. A later automatic placement policy must be supported by profiles
and comparisons, rather than assume that maximum cache occupancy means maximum speed.

## Agreed order and decision criteria

- [x] **First: Qwen3-4B BF16 versus NF4 on BoolQ.** Freeze model/data revisions,
  token arrays, training budget, seeds, optimizer and evaluation before rental.
  Evaluate before/after matched adapter training. Retain logits, adapters and batch
  identities; replay accuracy/NLL independently. Report uncertainty and practical
  limits. A small pilot cannot prove universal equivalence or decide 32B quality.
  Numerical correctness is separate from comparing quantization task quality.
- [ ] **Then: 32B on physical 8 GB VRAM and a real 32 GiB RAM upper bound.**
  Verify actual hardware/kernel limits before model download. Compare residency
  off/on at the same precision/configuration. Correctness precedes timing; report
  optimizer-step time, startup/complete-entry time and VRAM/RAM independently.
- [ ] **Then: SSD when weights exceed available RAM.** Compare ordinary streaming,
  RAM cache, GPU residency and both together in one matched configuration. Record
  actual backing storage and I/O; do not label loopback or SATA verified NVMe.
- [ ] **Then: automatic placement.** Use measured capacity, transfer/compute waits,
  setup overhead and steady-state savings to choose cache allocations with headroom.
  Larger models and candidate Triton kernels require their own supporting profiles.

## Boundaries

- NF4 will become the main research configuration only if its measured quality tradeoff
  is acceptable. Keep BF16 as a control; no acceptable degradation margin was agreed.
- All numerical comparisons within a fixed precision keep the existing reference-first
  `rtol=.01`, `atol=1e-5`, finite checks and three-decimal loss predicates.
- No full fine-tuning, convergence, 70B/120B capacity or universal speedup claim.
- Repeated timing processes cannot substitute for independent training seeds.
- Publish complete/failing outcomes and scoped conclusions to PR #5. Preserve exact
  raw outputs locally. Always delete instance and persistent volume and verify absence.

## Starting state and funding

The prior [32B BoolQ attempt](nf4-quality-20261024/README.md) was blocked before
model download. No quality, speed or peak training-memory result was obtained.
Two instances and two persistent volumes were deleted; independent listings showed 0/0.

Remaining conservative allowance was $3.343605273574782. The user authorized up to
$10 more if needed, making $13.343605273574782 available in total. Begin the 4B pilot
within the existing $3.34; keep the additional allowance for the later stages.
Charge all retries, setup, disks and transfers; never deduct prior studies twice.
This is a spending ceiling, not a commitment to spend it or a promise all stages fit.

## Active work

The [4B pilot](nf4-boolq-4b-20261024/README.md) is complete and independently replayed:
post-training accuracy BF16 86.719% versus NF4 85.547%, difference -1.172 pp with a
paired 95% interval [-3.906, +1.172] pp. NF4 reduced paired optimizer-step time by
17.230%; peak process RSS was 3.626 versus 9.045 GiB. This is a precision tradeoff,
not an equivalence or larger-model quality result. Keep BF16 as a control.

All four pilot instances and volumes were deleted. Conservative pilot accounting,
including retries: $1.842130, leaving $11.501476 from the combined authorization.
No additional $10 has been consumed yet. Exact bytes are retained and hash-indexed.

The [32B NF4 RAM-source attempt](nf4-capacity-32b-20261024/README.md) verified the
physical 8 GB GPU and a real 29.756 GiB cgroup RAM bound on a 30.997 GiB host.
NF4 preparation completed; native RAM-source setup refused its 19.2 GB base/extras
under the unchanged 55% physical-RAM policy, before any training. This is not OOM
or a larger-model task-quality result. Instance and persistent volume were deleted.

That stage cost $0.492097 conservatively. Combined accounting since the 4B pilot is
$2.334227, leaving $11.009379. The extra $10 allowance remains unused so far.

Next: keep the same physical RAM/GPU boundary, use the supported SSD tier, profile,
then qualify ordinary streaming / bounded RAM cache / GPU residency / both before
matched timing. Preserve the existing source headroom and numerical tolerances.
An automatic speed-aware placement policy remains separate from the existing
startup capacity planner; maximum cache occupancy is not the acceptance target.
