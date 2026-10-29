# Kanana 30B: ordinary streaming versus the complete expert-streaming variant

Direct same-host comparison at source [`43cf0b34`](https://github.com/SoupTeam/soup-rnd/commit/43cf0b34afe0d8850d76bf86d60663ab5c85cdb2).
The candidate combines two-read expert replay, static RAM/GPU expert caches and grouped
integer route preparation. Experimental opt-in APIs; defaults/CLI are unchanged.

## Accuracy before timing

Pinned `kakaocorp/kanana-2-30b-a3b-base-2601` at `0d31365acf656e5f9a593cacc4913e9c0d36cb1c` (30.67B MoE parameters).
Frozen NF4 base, BF16 compute, FP32 attention LoRA rank 4 / alpha 8;
512 tokens, batch 1, accumulation 2, frozen random inputs and initial adapters.
Physical **NVIDIA GeForce RTX 4060 Ti 8188 MiB**;
native workload RAM ceiling **29.755859 GiB**, swap disabled.
Candidate cache payload ceilings: static host **8 GiB**, device **2 GiB**.
Preregistered storage uses provider ROOT **32 GB** and data volume **112 GB**.
Actual free-space floors are **16 GiB ROOT / 100 GiB data** before download;
the separate before-NF4 check retains the root floor and requires 29,387,287,853 data
bytes free. The unchanged 12 GiB raw-output reserve and 1.5 GB encoded archive
reserve are budgeted on ROOT. Admission uses measured filesystem free bytes;
provider GB are not assumed to equal GiB. RAM, VRAM and no-swap gates are unchanged.
All eleven workers record deterministic algorithms enabled, TF32 disabled and eval
mode with differentiable LoRA updates and dropout disabled. The frozen worker enables
`torch.use_deterministic_algorithms(True)`; the runtime allocation debug flag
`fill_uninitialized_memory` was not persisted: **NO VERDICT** on its runtime value.
These timings apply to this controlled recipe. They do not establish a universal
gain for default nondeterministic CLI training. Allocation-fill overhead may affect
the contrast, but no causal profile identifies its contribution or a kernel cause.

All three complete accuracy trajectories (ordinary, legacy two-read+caches, complete
candidate) independently passed the original `rtol=.01`, `atol=1e-5`, finite and
three-decimal loss gates. Each capture covers **2 AdamW updates / 4 microbatches**;
all **288 trainable LoRA tensors** (A/B for 144 adapted attention projections)
receive nonzero gradients and change. Captured comparison
bitwise agreement: **True**. This does not establish downstream task quality
or convergence equivalence.

## Fresh unprofiled comparison

| Variant | Median step, s | Setup, s | Whole eight-update entry, s | Step allocated / reserved VRAM, GiB | Sampled timing RSS, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary streaming (three-read) | 27.8047 | 11.802 | 250.966 | 1.658 / 1.971 | 3.860 |
| Two-read + RAM/GPU caches + grouped routes | 15.9873 | 38.948 | 180.605 | 3.643 / 3.838 | 12.048 |

Paired step-time reduction: **42.466%**;
throughput increase: **73.811%**.
Descriptive paired-bootstrap 95% interval for time reduction:
**[41.934%, 42.931%]**. Original lower-bound >=2% gate: **PASS**.

Four balanced AB/BA pairs, eight fresh processes, two warmup and six measured
optimizer updates per process: 16 warmup and 48 measured updates total.
Percentages use geometric means of per-pair median ratios, so they need not equal
the ratio of the two pooled medians shown above. Whole entry includes process/import
bootstrap, setup/admission, all eight updates and validation; shared model download
and NF4 preparation are excluded. Setup cost must be amortized by longer training.
RSS sampling can miss brief peaks and excludes kernel/filesystem page-cache charges;
the RSS column is not a claim that this fits on a host with that much RAM.

This cohort is a direct ordinary-versus-complete comparison on the hardware above.
Earlier 16B/30B studies remain separate; their percentages are not added or used
as timing denominators here. No physical NVMe claim follows from a container mount.

## Recording and retention limits

Recorded timed last-microbatch losses pass the original gates. The first timed
microbatch scalar and final timed adapter/AdamW tensors were not persisted:
**NO VERDICT** for their numerical parity. Full tensor parity applies to the
separate accuracy captures, not every timed update or full convergence.

Both independent audits passed: numerical/timing and infrastructure/retention/cleanup.
The v5 infrastructure audit checks captured SHA256/size receipts for all thirteen
official shards, bounded fresh input repair, the imported helper and native prepared
fingerprints before accuracy. Full 61 GB model weights and rejected payloads are not
retained; the local audit verifies those recorded receipts and fingerprints rather
than rehashing a retained full checkpoint.
`gpu-evidence.zip` contains exact selected originals and audit/control sources;
`files.json` maps all logical originals to stored members, sizes and SHA256 values.
`F15/ops/` contains the exact frozen private launch/transport helpers and CPU/mutation
controls, bound to the launch manifest and independent prelease operational review.
That operational review establishes no GPU accuracy or speed result by itself.
Aliases are accepted only after full byte comparison, and every logical original
is restored and verified after ZIP creation. Complete tensor payloads larger than 64 KiB remain
locally retained and hash indexed in `evidence-manifest.json`; they are explicitly
omitted from Git, not discarded. Frozen production input archives are also retained
locally and indexed separately. The compact ZIP alone cannot replay full numerical
captures. [Summary](gpu-results-summary.json).

## Budget and cleanup

F9's rejected offer created no identified GPU or disk. Its raw prospective ledger
and no-allocation reconciliation are preserved, without claiming a verified invoice
or refund. F10 stopped at its profiler prerequisite: four numerical CUDA cases
passed, the CUDA/CUPTI profiler-event case failed; full 30B accuracy and timing
were not run. F11 passed four numerical CUDA controls but was stopped during
checkpoint download before full 30B accuracy/timing. F12 failed at provider API 429
during boot, before authenticated SSH or GPU work. All failed attempts remain
**FAILED / NO VERDICT** for full 30B accuracy and speed, with originals retained.
F13 passed four numerical CUDA controls but failed the unchanged official SHA256
checkpoint gate for shard 12 before NF4 preparation or full 30B accuracy/timing.
All thirteen expected official shard pins equal the earlier frozen reference. The
actual rejected shard digest was not recorded by that original code; its failure
cause remains unknown. F13 is also **FAILED / NO VERDICT** for full 30B results.
F14 verified the official inputs and produced only the ordinary 2-update/4-microbatch
capture before the unchanged no-swap gate failed. Cgroup swap was observed while
process swap remained zero; policy/remapping cause is unknown. That capture is
retained from an invalid run. No optimized capture, timing or full comparative
accuracy verdict exists for F14: **FAILED / NO VERDICT**.
F15 uses four actual
numerical CUDA controls and unprofiled timing; it makes no passed-profile claim.
Numerical tolerances and the complete 30B accuracy/timing recipe are unchanged.

Failed F10 conservative cost: **$0.324850**.
Failed F11 conservative cost: **$0.375204**.
Failed F12 conservative fallback: **$0.468727**.
Failed F13 conservative cost: **$0.597840**.
Failed F14 conservative cost: **$0.649493**.
F15 conservative cost: **$0.601058**;
remaining task authorization: **$3.185708** from
$3.786766 before F15.
Initial task authorization was $6.202880; each paid
failed attempt is deducted exactly once. F9's prospective allowance is not spend.
These are conservative estimates with reserves, **not invoices**.
Exact owned F10, F11, F12, F13, F14 and F15 instances and persistent volumes were independently
confirmed absent.
Shared-account colleagues' resources were untouched.
