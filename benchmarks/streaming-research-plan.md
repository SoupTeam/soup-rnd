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

Next: keep the same physical RAM/GPU boundary, use the supported SSD tier and qualify
ordinary streaming / bounded RAM cache / GPU residency / both. Then profile and run
matched timing. Preserve the existing source headroom and numerical tolerances.
An automatic speed-aware placement policy remains separate from the existing
startup capacity planner; maximum cache occupancy is not the acceptance target.

## Latest SSD outcome and concrete next actions

The [bounded 32B NF4 disk attempt](nf4-ssd-32b-20261024-blocked/README.md) completed
preparation and a preliminary GPU-only capacity probe with seven resident layers.
The combined 14 GiB RAM / GPU cache then hit the pinned staging-loan guard before
independent accuracy qualification or accepted timing. A stale rotating-slot mapping
was incorrectly borrowed again after the same layer entered the immutable RAM cache.
The fix separates those ownership paths and preserves rejection of genuine cold loans.

The CPU regression fails on the old behavior; a mutation bypassing the remaining
cold-loan guard also fails. Focused validation: 200 passed, 18 CUDA skips; final changed
module: 19 passed, 2 CUDA skips. The new real-CUDA regression still needs the retry.
No performance benefit is inferred from the preliminary capacity probe. A separate
prospective profiler warmup error was also repaired and checked against the native
benchmark API; explicit stop signals now have retention-aware handlers in the helper.

- [ ] Retry real-CUDA source/regression and native helper controls before model download.
- [ ] Qualify all eight primary/semantics captures for the four cache variants; replay
  complete tensors and actual parameter updates independently before subsequent phases.
- [ ] Retain a separate CUDA diagnostic profile; ordinary speed runs remain unprofiled.
- [ ] Retain all 16 fresh-process timing runs and 96 warmup/measured updates, then replay
  all final adapters, losses, RAM/swap boundaries and batch identities.
- [ ] Use the measured bottleneck and startup costs to choose a speed-aware placement
  feature. Treat the existing auto-residency and RAM-hotset planners as capacity policies.

Current task accounting since the 4B pilot is $2.842538 conservatively, leaving
$10.501068 from the combined authorization. All instances and persistent volumes
created by this task are deleted and independently absent by exact IDs. The user
confirmed that the API key is shared by the group; unrelated colleague rentals stay
untouched and outside this task's ledger. Lease/cleanup guards now use exact owned
IDs and stage labels rather than demand an empty account; unresolved new resources
on the selected machine remain a hard stop rather than being guessed or deleted.

[10Cache](https://arxiv.org/html/2511.14124v1) motivates profiling tensor reuse/order
and reusable pinned buffers. Soup already has asynchronous reads, prefetch and pinned
arenas, so copying those concepts alone is not a new feature. After the NF4 profile,
compare exposed source waits, large vocabulary loads and repeated dequantization before
choosing a cache or kernel change. The paper's DeepSpeed training gains are not Soup
LoRA measurements. Preserve memory headroom and measure benefit rather than target 100%
occupancy or extrapolate the paper's speedups.

The [first fixed-source GPU check](nf4-ssd-32b-20261024-fixture-check/README.md)
stopped before model download: 148/149 source/helper controls passed; the new CUDA
regression's hotset was empty because its two eligible decoder layers already fit
ordinary read-ahead and its tiny budget did not cover a native pinned arena.
The fixture now keeps three eligible decoder layers plus a distinct cold vocabulary
group, and derives its ceiling from the actual arena plan. A new CPU check proves
that the same pinned plan is nonempty. Final changed-module checks: 20 passed,
2 CUDA skips. The next GPU-host control count is 150; model download remains gated.

This check's instance and volume were deleted. Conservative own-study accounting
is now $3.155377, leaving $10.188229 from the combined authorization; colleague
rentals remain outside this ledger. The numerical contract is unchanged.
