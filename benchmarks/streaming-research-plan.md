# Streaming research plan

Updated 2026-10-24. Destination: SoupTeam/soup-rnd PR #5.

## Goal

Reduce real LoRA training time for models exceeding GPU memory by coordinating GPU,
RAM and SSD storage. Select allocations for measured speed with memory headroom and
measured task quality. A later automatic placement policy must be supported by profiles
and comparisons, rather than assume that maximum cache occupancy means maximum speed.

## Agreed order and decision criteria

- [ ] **First: Qwen3-4B BF16 versus NF4 on BoolQ.** Freeze model/data revisions,
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

Preparing the 4B quality pilot. No new rental or GPU result is implied by this checklist.
The quality pilot does not require claiming an enforced 32 GiB host-memory limit;
that hard limit remains a separate acceptance condition for the 32B capacity study.
