# CPU threads on top of two-read and expert caches

**Result: no confirmed additional speedup.** All five separate full numerical
captures match bitwise under the original tolerances. More PyTorch CPU intra-op
threads did not beat the fresh native baseline, so no candidate entered held-out
confirmation. This does not establish that every form of CPU parallelism or
expert prefetch is ineffective.

## Same-cohort experiment

Companion source [`43cf0b34`](https://github.com/SoupTeam/soup-rnd/commit/43cf0b34afe0d8850d76bf86d60663ab5c85cdb2),
Kanana-2-30B-A3B, revision `0d31365acf656e5f9a593cacc4913e9c0d36cb1c`:
30.67B total sparse-model parameters, frozen NF4 base / BF16 compute / FP32
attention LoRA rank 4, alpha 8, context 512, batch 1, accumulation 2. All variants
already use grouped integer routes, two-read and static **8 GiB RAM / 2 GiB GPU
expert-weight cache ceilings**. No production, CLI, schema, default, attention,
precision or cache-policy change.

Physical RTX 4060 Ti **8188 MiB**, Ryzen 5 9600X (6 cores / 12 logical CPUs), native
cgroup RAM ceiling **31,950,110,720 bytes (29.756 GiB)**, workload swap disabled.
Separate runtime cohort: Torch 2.10.0+cu130, CUDA 13.0, Triton 3.6.0,
bitsandbytes 0.50.2, actual `libbitsandbytes_cuda130.so` verified before model
download. All 20 coupled runtime wheels use exact official URL/hash locks.
Earlier F15 and failed A timings are not denominators.

Only CPU **intra-op** thread count changes, after identical native model/cache
construction and adapter initialization. Native is actually 1 intra-op / 12
inter-op threads, with affinity to 12 logical CPUs; `t1` is a duplicate control.
Inter-op, affinity, OMP/MKL environment, GPU kernels and training inputs stay fixed.
This is not a test of a background expert-read or transfer pipeline.

## Exploratory timing and memory

One fresh process per policy, two warmup and six measured accumulated AdamW
updates: **30 measured + 10 warmup updates**. Fixed exploration order is
`t4, native, t8, t2, t1`, with no discarded updates.

Step is the median; RSS is the sampled timing peak.

| Policy | Threads | Step, s | Eight-update entry, s | Allocated VRAM, GiB | RSS, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 1 | 16.3243 | 180.580 | 3.6443 | 11.9631 |
| t1 | 1 | 16.5433 | 182.879 | 3.6443 | 11.9687 |
| t2 | 2 | 16.6033 | 182.631 | 3.6443 | 11.9753 |
| t4 | 4 | 16.4912 | 183.781 | 3.6443 | 11.9665 |
| t8 | 8 | 16.5039 | 182.323 | 3.6443 | 11.9754 |

Peak reserved VRAM is **3.8320 GiB in all five arms**. RSS is sampled every 50 ms;
it excludes filesystem/kernel charges and is not total system RAM. No observed
process/cgroup swap use. Whole entry includes setup, warmup, measurement and
worker startup/validation, while common model download/NF4 preparation is separate.

Every distinct candidate median is slower than native. The original selection
therefore yields **NO_CONFIRMED_WIN**, with zero confirmation pairs. These five
exploratory medians are not a statistically confirmed slowdown or a general
thread-count recommendation. The unchanged >=2% lower-confidence-bound gate was
not bypassed, and no bootstrap confidence interval is manufactured without the
preregistered four NEW balanced confirmation pairs.

## Numerical verification

Accuracy ran before profiling/timing. All five complete **two-AdamW-update /
four-microbatch** captures independently recompare **bitwise**: full recorded
gradients, LoRA parameters and AdamW states. All **288 trainable LoRA tensors /
144 adapted attention projections** receive nonzero gradients and actually
update. Original reference-first `rtol=.01`, `atol=1e-5`, finite and three-decimal
loss gates are unchanged. Four actual small CUDA controls passed first.

Deterministic algorithms, TF32 off, eval mode with differentiable LoRA updates and
dropout disabled. Short random-token numerical parity is not downstream quality,
convergence or full fine-tuning evidence. Saved last-micro warmup/timed losses are
checked; unrecorded first-micro scalar losses and full final timed tensors retain
NO VERDICT. Full tensor parity applies to the separate captures.

## Profile: a narrower, concrete next candidate

The original native forward diagnostic skips 1024 actual expert fetches and records
16. All 16 are **RAM-cache hits**, with zero GPU-cache hits and zero storage reads
in that window. Actual trace contains 678 CUDA kernels and 128 H2D events.

Within the **18.131 ms first-to-last GPU activity span**, H2D union is **3.064 ms**,
all GPU-kernel union **1.480 ms**, and direct NF4 dequantization kernel durations
sum to about **0.392 ms**. CPU self-time rankings include 128
`cudaStreamSynchronize` calls (~2.992 ms), 128 `cudaMemcpyAsync` calls (~2.111 ms),
and kernel launches. Profiler category counts come from the raw trace, rather
than treating GPU user annotations as kernels.

This supports investigating **transfer batching and event-protected next-expert
preparation**. It does not prove
a whole-step CPU bottleneck, disk starvation, expected speedup, or that all gaps
are avoidable: this is an overhead-bearing forward window, not backward/full-step
causal attribution. Nested/concurrent durations must not be added together.

The current expert path has demand-driven fetches, pageable mutable staging and
synchronizing `Tensor.to`; NF4 dequantization runs on GPU. A bounded pinned staging
pipeline with separate CUDA stream and protected buffer reuse is a future
candidate, not implemented or measured here. The [official PyTorch transfer
guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
describes the overlap requirements and warns about premature mutation of pinned
source buffers. Physical NVMe backing remains unverified; logical reads are not
physical SSD service times.

## Failures, recovery and audit provenance

Attempt A on an RTX 5060 Ti / Torch 2.8 cu128 passed four small CUDA numerical
cases, but its mandatory tiny profile had `CUPTI_ERROR_INVALID_DEVICE`, zero
actual CUDA/H2D trace events, and stopped **before** the 61 GB download. All 108
originals remain retained. It is FAILED/NO 30B VERDICT, not relabeled successful.
B changes both host and software; success is not attributed to one change alone.

B completed on the original remote process without a compute restart. The local
controller interrupted and its status file became all NUL bytes. Same-lease
recovery restored hidden local guards without resetting the original deadline or
budget. A corrupted mutable API-throttle file was preserved and reinitialized
under its OS lock with the same two-second floor to permit exact cleanup. The
conservative transfer-reserve check rejected after the interruption; evidence-only
recovery retrieved the normal completed immutable archive. The recovery monitor's
exit 1 remains recorded. All original bytes were verified before deletion.

The pre-data independent auditor `a0b434b3...` initially rejected the actual
fragmentless `pip freeze` URL format. A separate, reviewed post-retention parser
correction accepts only the exact registered URL without a fragment; mandatory
installation hashes and all scientific gates remain unchanged. The original
failure, exact diff, RED control, 13 passing focused controls and provenance are
retained. Corrected full independent audit `14a10faf...` passed. Prelaunch evidence:
203 operator/design controls, 18 independent runtime controls and 90 pre-data
auditor controls; these overlapping groups are not summed into a unique count.

## Retention, budget and cleanup

All **195 B originals / 5,875,436,481 bytes** were restored and SHA-verified from the
**627,015,680-byte** lossless archive. A's 108 originals are retained separately.
Compact public packets use exact-byte aliases and restore checks; large captures,
model payloads and source archives are explicitly hash-indexed when local-only.
Both packets pass actual API/private-key material scans. The full 61 GB checkpoint
is not retained; immutable expected/observed SHA receipts are, with this audit limit
explicitly recorded. See [summary](summary.json), [evidence manifest](evidence-manifest.json),
[A packet](evidence-a.zip) and [B packet](evidence-b.zip).

Exact A instance/volume **58086698 / 58086695**, and B **58097253 / 58097252**,
were deleted. Fresh full-account B absence: **2026-10-28T14:56:50Z**. Restored local
guard/monitor processes exited. Colleague resources were not modified.

Conservative costs, including failed A and B interruption idle time/reserves:
**A $0.507217 + B $0.769376 = $1.276594**, leaving **$4.909114** from the prior
$3.185708 remainder plus the newly authorized $3. These are ledgers, not invoices.
No paid resources remain. Full-suite and hosted-CI success are not inferred.
