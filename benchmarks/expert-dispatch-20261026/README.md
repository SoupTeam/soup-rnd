# Kanana 30B: grouped expert dispatch after two-read and RAM/GPU caches

Experimental source: [`43cf0b34`](https://github.com/SoupTeam/soup-rnd/commit/43cf0b34afe0d8850d76bf86d60663ab5c85cdb2).
`grouped_expert_routes` defaults to **False**; no CLI/schema/default change.
It builds immutable integer routes in bulk while preserving expert, top-k-slot,
token and floating-point accumulation order. No new attention, precision or Triton kernel.

## Accuracy first

Physical RTX 4060 Ti **8188 MiB**, native RAM ceiling **29.755859 GiB**, workload swap disabled.
Pinned `kakaocorp/kanana-2-30b-a3b-base-2601` at
`0d31365acf656e5f9a593cacc4913e9c0d36cb1c`: 30,670,809,088 parameters, 47 MoE layers,
128 experts/layer, top-6. NF4 base, BF16 compute, FP32 attention LoRA rank 4 / alpha 8;
512 input tokens, batch 1, accumulation 2; static host 8 GiB / device 2 GiB expert caches.

Five actual CUDA controls passed without skips before download. The standalone F8
qualifier captured ordinary streaming, legacy two-read+caches, and grouped
two-read+caches in full **before timing**. All 288 adapters received nonzero gradients
and changed over 2 AdamW updates / 4 microbatches per capture. Each candidate independently
passed the original reference-relative `rtol=.01`, `atol=1e-5`, finite and three-decimal
loss gates, with exact inputs/initial adapters/integer routes. Grouped vs legacy also
passed. **All new numerical comparisons bitwise: True.** The extra F7 ordinary
reference is an additional numerical comparison, never a timing denominator.

## Fresh unprofiled timing

| Variant, same two-read+caches | Median step, s | Setup, s | Whole eight-update entry, s | Step allocated / reserved VRAM, GiB | Sampled timing RSS, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 17.220160 | 40.863 | 192.216 | 3.6433 / 3.8379 | 12.0374 |
| grouped | 16.164656 | 39.243 | 181.051 | 3.6433 / 3.8379 | 12.0290 |

Paired step-time reduction: **6.249%**,
descriptive 95% whole-pair bootstrap **[5.716%, 7.151%]**.
Original lower-bound>=2% practical gate: **PASS**.
Four balanced AB/BA pairs, eight fresh workers, two warmup+six measured updates each:
16 warmup and 48 measured updates. Geometric paired-median ratios, 5,000 draws, seed 29.
All recorded losses, full cache ceilings, actual expert flags and equal timed cache
work were independently rechecked. Whole entry includes process/import bootstrap,
admission/setup, all eight updates and validation; shared download/NF4 sharding is excluded.

The previous RTX 4060 Ti four-arm study is [preserved separately](../kanana-30b-cache-20261026/GPU-RESULTS.md).
Its absolute times are not a controlled denominator for this source/cohort. Percentages
from separate studies are not added or multiplied into a new ordinary-streaming claim.

## Diagnostic profile and preserved refusal

F7 completed native ordinary/two-read/two-read+caches numerical captures first.
The subsequent two-read trace captured 5,127,782 events. Full raw trace
**1,884,940,630 bytes** and gzip **136,032,043 bytes** were retained and independently
decoded/hash verified. The original 128 MiB encoded-size gate **FAILED**; it was not
relaxed and the run is not called a successful combined-profile cohort. The cached
large profile and F7 dispatch timing were not executed. F8 is a separate qualification.

The trace records `26833` item calls,
`8944` where calls and
`9038` nonzero calls in the captured microbatch.
This supports testing bulk dispatch; counts alone do not prove end-to-end speed.
CPU/CUDA inclusive times and concurrent activities are not additive. The independently
parsed GPU interval unions/gaps are inside the observed GPU span, not a whole-step idle
percentage. All profile durations contain profiling overhead. F7 final profile-state
hashes/postprocess qualification were not recorded after the size refusal: **NO VERDICT**.

## Scope and recording limits

Controlled deterministic numerical mode and TF32-off are identical across variants.
The profile includes many allocation/fill events; those are not a verified production
Triton opportunity. Physical NVMe backing is unverified (container loopback mount).
Payload counters do not measure physical SSD service time.

- First scalar loss of an accumulated timed update: **NO VERDICT: not persisted by native worker**.
- Timed final adapter/AdamW tensor parity: **NO VERDICT: final timing tensors not persisted; full tensors cover accuracy phase**.
- Full numerical trajectories cover the separate accuracy phase; recorded timed losses
  and per-update finite checks do not establish every unrecorded intermediate.
- Random-token, short LoRA qualification; downstream task quality, convergence, full
  fine-tuning, other models/hardware and globally optimal automatic placement unmeasured.

Candidate CPU controls 127 passed / one CUDA skip; public qualifier 52 CPU; F8 ops 139 CPU;
independent auditor/foundation 73 CPU; source Ruff and applicable pre-commit passed.
These are separate scopes, not one summed suite. Full-suite/hosted-CI success is not inferred.

## Retention, budget and cleanup

All original large captures/traces are fully retained locally and hash indexed in
`evidence-manifest.json`; the compact packet explicitly omits those large payloads.
Every selected packet original and duplicate alias was restored/byte-compared and hashed.
Executed independent audit: **passed_standalone_qualification_audit**; profile refusal stays failed.
All owned instances and persistent volumes, including failed attempts, are deleted
and exact IDs independently confirmed absent. Shared-account peers were untouched.

Conservative follow-up cost since completed stage E: **$3.136061**,
including every failed attempt; remaining authorization **$1.202880**.
This resumed turn (F5-F8): **$1.760702**.
These include conservative reserves and are **not invoices**. Prior E/D costs were
already deducted before this authorization and are not charged again.
