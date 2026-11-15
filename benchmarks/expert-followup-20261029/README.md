# Host packing, usage placement and late expert lookahead

**Verdict: CONFIRMED_PRACTICAL_WIN.** Four NEW balanced AB/BA pairs for **hostpacked** give **3.428%** paired step-time reduction (95% bootstrap **[3.034%, 3.820%]**) and **3.550%** throughput increase. The unchanged >=2% lower-bound gate passes.

## Scope and six arms

Same Kanana-2-30B-A3B (30.67B total parameters), revision
`0d31365acf656e5f9a593cacc4913e9c0d36cb1c`, frozen NF4 base / BF16 compute /
FP32 attention LoRA rank 4, alpha 8, context 512, batch 1, accumulation 2.
Physical **NVIDIA GeForce RTX 4060 Ti / 8188 MiB**;
native RAM ceiling **29.756 GiB**, workload swap disabled.
Frozen Torch 2.10.0+cu130 / CUDA 13 / Triton 3.6.0 / bitsandbytes 0.50.2 stack.
All arms already have two-read, grouped routes, packed H2D and an 8 GiB static RAM
expert-weight cache. Defaults, CLI and model arithmetic remain unchanged.

- **baseline:** existing packed transfer, first-fit 2 GiB dense GPU hotset.
- **hostpacked:** immutable prepacked pageable RAM entries, including alignment
  padding charged inside the SAME 8 GiB RAM ceiling; one CPU staging copy per hit.
- **prefetch_after:** next expert preparation after current MLP/index-add enqueue.
- **usage2:** frozen usage-selected GPU entries under the SAME 2 GiB payload ceiling.
- **larger:** first-fit GPU hotset at 2.125 GiB.
- **all:** host packing, late lookahead and usage-selected 2.125 GiB GPU entries.

The 2.125 GiB larger ceiling was fixed **before rental**, based on the existing
conservative admission policy, which reserves the full lazy cache ceiling on top
of measured peaks. This stage did not try 3 GiB, observe a 3 GiB failure, weaken
admission checks or search for a maximum/optimal cache size.

Tested companion source [9873f323](https://github.com/SoupTeam/soup-rnd/commit/9873f3239db47ee6c5539cea8b125261cdba82c1),
[portable patch ZIP](prototype.patch.zip) against `f5670242`. This remains
an opt-in experimental source; the PR's older production implementation is separate.
The baseline here is already optimized packed streaming, not ordinary three-read.
Old results are not denominators and percentages are not added across cohorts.

## Accuracy before speed; exploratory timing and memory

The table has one fresh process per arm, two warmup and six measured AdamW
updates. Every measured update is retained. It is an exploratory comparison,
not a statistically confirmed ranking of all six candidates.

| Arm | GPU cache GiB | Step s | Entry s | Alloc GiB | Reserved GiB | RSS GiB | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 2 | 13.5957 | 158.846 | 3.6496 | 3.8574 | 11.9768 | bitwise |
| hostpacked | 2 | 13.1175 | 170.013 | 3.6496 | 3.8574 | 13.4971 | bitwise |
| prefetch_after | 2 | 13.7856 | 158.199 | 3.6496 | 3.8574 | 11.9670 | bitwise |
| usage2 | 2 | 13.5302 | 157.750 | 3.6486 | 3.8555 | 11.9371 | bitwise |
| larger | 2.125 | 13.5707 | 156.040 | 3.7726 | 3.9941 | 11.9591 | bitwise |
| all | 2.125 | 13.1417 | 154.195 | 3.7716 | 3.9824 | 13.1621 | bitwise |

Held-out confirmation only: step and complete entry are medians of four worker summaries per arm; memory is their observed maximum.

| Arm | Step, s | Eight-update entry, s | Allocated GiB | Reserved GiB | RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 13.5301 | 157.646 | 3.6496 | 3.8574 | 11.9779 |
| hostpacked | 13.0792 | 154.450 | 3.6496 | 3.8574 | 13.9874 |

Entry includes cache/model setup, warmup, timing, process startup and validation.
Common checkpoint download/NF4 preparation is separate. RSS is sampled every
50 ms, excludes kernel/filesystem charges and is not total system memory.
Only the best distinct positive exploratory candidate enters four NEW balanced
AB/BA confirmation pairs; selection samples are excluded. Bootstrap: 5000 draws,
seed 29; unchanged practical gate: lower 95% bound >=2% step-time reduction.
Negative and NO_CONFIRMED_WIN outcomes are retained without changing gates.

Six separate full 2-AdamW-update / 4-microbatch captures passed before any timing;
comparison result: **all bitwise**.
All 288 LoRA tensors / 144 attention projections receive gradients and update.
Original rtol=.01, atol=1e-5, finite and three-decimal loss checks are unchanged;
all recorded exploratory and paired warmup/timed losses are also checked.
Full final timed tensors and unrecorded first-microbatch losses retain NO VERDICT.
This is same-NF4 numerical/resource qualification, not task quality/convergence.

## Frozen usage profile and placement

Plans use ONLY actual logical expert requests from two baseline warmup updates,
including cache hits; speculative prefetches are excluded. All six workers use
the same frozen inputs and initial adapters. Plans are frozen before candidate
accuracy/timing. These input-specific frequencies are not a held-out dataset study.

| Learning request source | Count | Share |
| --- | ---: | ---: |
| device_hits | 1553 | 3.81% |
| host_hits | 23064 | 56.55% |
| storage_reads | 16169 | 39.64% |

| GPU ceiling GiB | Experts | Payload GiB | Recorded request coverage |
| --- | ---: | ---: | ---: |
| 2 | 227 | 1.9951 | 4.45% |
| 2.125 | 241 | 2.1182 | 4.73% |

Ranking is frequency times packed-transfer bytes per dense-GPU byte, with exact
integer fractions, stable first-observed ties and whole-entry budget accounting.
It is a greedy traffic proxy, not measured latency or an optimal placement claim.
Plan membership, sums, raw request counts, per-arm budgets and cache conservation
equations independently reproduce. Different plans may change hit/read counts;
logical expert work and numerical gates remain the same.

## Current packed baseline diagnostic

This is a separate instrumented diagnostic, never a speed denominator. CPU spans
cover a complete update's forward/backward; CUDA trace covers ONLY 16 forward
fetches after 1024 skipped fetches in the first microbatch. The scopes differ;
nested/overlapping durations must not be added or turned into whole-step percentages.

| Complete-update CPU wall span | Calls | Inclusive seconds |
| --- | ---: | ---: |
| fetch | 17624 | 8.305270 |
| storage_read | 4406 | 1.469308 |
| packing_layout | 16734 | 0.278948 |
| transfer_stage | 16734 | 2.909810 |
| nf4_dequant | 34812 | 2.437762 |

| CUDA device | First-to-last recorded activity, us | H2D union, us | Kernel union, us |
| --- | ---: | ---: | ---: |
| 0 | 15520.115 | 2940.833 | 1347.049 |

Within that bounded CUDA window, **64 dequant-named kernels** have a summed
duration **391.708 us**. Exact names/counts are in [diagnostic.json](diagnostic.json).
These are kernel-category events, not user annotations; their sum is not a
separate additive part of H2D/whole-step time. The profile cannot by itself prove
physical disk starvation or the current whole-training critical path.

## Source, evidence, cleanup and budget

Pre-data scoped source controls: **314 CPU passes**.
The **14 declared actual CUDA cases** subsequently
passed without skips before the full checkpoint download. Actual runtime library
identity and immutable official wheel hashes were checked. Full-suite/hosted-CI
success is not inferred. The frozen standalone auditor independently reloads all
six captures, reconstructs plans, reparses original CUDA trace, recomputes selection
and paired timing, and checks source, inputs, no-swap resources, budgets and cleanup.

All **321 originals / 7,061,120,837 bytes** retained
and hash-verified from **626,984,960 encoded bytes**. Compact public
[evidence.zip](evidence.zip) uses exact-byte aliases; large captures/model payloads
are explicitly hash-indexed as local-only. Runnable original auditors are in the
packet; `.py.txt` review copies normalize line endings only. The full BF16 source
checkpoint is not retained; official expected/observed SHA receipts are retained.
Physical NVMe backing remains unverified.

Owned instance **58147079** and persistent volume **58147078**
deleted and freshly verified absent. Shared-account colleagues were untouched.
This follow-up conservative cost: **$0.610643 / $4.394630**;
remaining: **$3.783987**. Prior packed-transfer study cost
**$0.605370** is separate, already deducted from the same $5 allowance;
aggregate against that $5 is **$1.216013**.
These are conservative ledgers, not invoices. Older unused $4.909114 is not added.

[Full audited summary](summary.json) · [all timing arrays](tables.json) ·
[diagnostic and placement tables](diagnostic.json) · [evidence manifest](evidence-manifest.json).
