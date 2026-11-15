# Persistent decoder layers in streamed 7B training — 2026-10-20

Research prototype over current upstream Soup; ordinary training configuration and arithmetic
are unchanged. This record is one dense-BF16 workload on one rented RTX 4090, not a
general throughput guarantee or downstream task-quality evaluation.

## Frozen method

- Upstream: `d500ae0dca7c4da905f8d59c9866af2f8aaa824b`; 602 production files are
  byte-identical to the GitHub snapshot. This includes existing head-prefetch and prime fixes.
- GPU stack: Torch `2.8.0+cu128`, Transformers `5.17.0`,
  PEFT `0.20.0`, TRL `0.29.1`, Accelerate `1.14.0`.
  Independent CPU replay uses a separate local Torch version, recorded in the JSON.
- Model: `openlm-research/open_llama_7b_v2`, revision
  `e5961def23172a2384543940e773ab676033c963`; dense BF16, ordinary PEFT LoRA rank/alpha 4,
  all seven projections, dropout 0, nonzero initial adapters, seed 792.
- Same frozen 32-row synthetic arithmetic data and order, seq 1024, batches 1/2, accumulation 2.
- Baseline: two rotating slots; resident8: layers 12–19; resident16: layers 8–23.
  Selected layers get private slots; other layers preserve upstream pool/event ordering.
- Every process has the same 10 GiB CUDA allocator limit; driver context is additional.
  This is an allocation-budget experiment on a 24 GB card, not a physical 8 GB card validation.
- The CPU pinned store remains intact in every arm. The added storage is GPU memory.
- Existing capture/comparator code is unchanged: reference-first allclose at
  `rtol=1e-2`, `atol=1e-5`, exact input/initial tensors, finite tensors, loss to three decimals.
- Accuracy before timing. Four fresh-process pairs per arm/batch, AB/BA/BA/AB,
  six warmup and eight measured optimizer updates. Final adapters/losses are checked
  again after all 14 updates. Model/data identity is checked for each pair.
- Practical gate: at least 2% median paired reduction, at least 75% positive pairs,
  and 95% pair-bootstrap lower bound above 0; 20,000 resamples, seed 792.

## Independently replayed accuracy

| Comparison | Tensors | Original gate | Bitwise equal |
| --- | ---: | --- | --- |
| baseline-repeat-b1 | 1813 | True | True |
| resident8-b1 | 1813 | True | True |
| resident8-b2 | 1813 | True | True |
| resident16-b1 | 1813 | True | True |
| resident16-b2 | 1813 | True | True |
| resident8, causal/padding semantics | 903 | True | True |
| resident16, causal/padding semantics | 903 | True | True |

Semantic capture preserves the existing causal-prefix and masked-padding isolation
assertions. Cross-shape unpadded logits remain a diagnostic, with no relaxed replacement gate.

## Paired time and GPU memory

Time columns are medians of process medians. Reduction is the median of within-pair
ratios; these are different aggregations. Allocated/reserved columns are medians of
per-process measured-window peaks. All timing samples are unprofiled.

| Arm | Batch | Baseline step, s | Candidate step, s | Paired reduction | 95% bootstrap | Allocated GiB, base/candidate | Reserved GiB, base/candidate | Practical gate |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| resident8 | 1 | 1.974721 | 1.662359 | 15.826% | [15.795%,15.840%] | 1.8448/4.8606 | 2.1230/5.1387 | True |
| resident8 | 2 | 2.197845 | 2.064987 | 6.038% | [5.887%,6.058%] | 2.4739/5.4896 | 3.0078/6.0234 | True |
| resident16 | 1 | 1.974758 | 1.357570 | 31.246% | [31.191%,31.332%] | 1.8448/7.8763 | 2.1230/8.1543 | True |
| resident16 | 2 | 2.197580 | 1.942621 | 11.596% | [11.570%,11.624%] | 2.4739/8.5053 | 3.0078/9.0391 | True |

## Weight movement, RAM and preload

Logical decoder bytes count actual executed pool copies, including the existing owner-skip
behavior. Vocabulary-boundary loads are separate. Counts are not a disjoint time profile.

| Arm | Batch | Decoder loads/update, base/candidate | Decoder GiB/update, base/candidate | Resident GiB | Pinned host GiB, base/candidate | Preload seconds |
| --- | ---: | --- | --- | ---: | --- | ---: |
| resident8 | 1 | 120.0/88.0 | 45.2362/33.1732 | 3.0157 | 13.0000/13.0000 | 0.1314 |
| resident8 | 2 | 120.0/88.0 | 45.2362/33.1732 | 3.0157 | 13.0000/13.0000 | 0.1313 |
| resident16 | 1 | 120.0/56.0 | 45.2362/21.1102 | 6.0315 | 13.0000/13.0000 | 0.2622 |
| resident16 | 2 | 120.0/56.0 | 45.2362/21.1102 | 6.0315 | 13.0000/13.0000 | 0.2619 |

The result JSON additionally records process RSS, peak RSS, setup time, whole-training
wall time including final saving, throughput, every pair, and all 14-update trajectory checks.
The preload-only break-even estimate excludes download, sharding and general setup;
it is descriptive and is not a measured cold-start break-even. RSS includes Python,
file-backed mappings and other allocations; identical pinned-store bytes are the direct
evidence that this prototype has not freed the host copies.

## Scope and remaining integration work

- CUDA allocator peaks exclude driver context and CPU RAM. Compare configurations at a
  fixed card budget; a larger cache can crowd out batch/context or cause OOM.
- These comparisons hold microbatch size fixed. They do not establish that caching is
  faster than spending the same free VRAM on a larger native microbatch at the same
  effective batch. That control needs its own unchanged-tolerance accuracy comparison.
- Four pairs bound uncertainty coarsely. No cross-host or previous-lease timing is pooled.
- Baseline and candidate share research accounting around the unchanged copy routine.
  Per-copy Python accounting is part of this prototype measurement; a production integration
  should validate its counter-free execution as well.
- The installer is a dedicated research entry, off the production path. There is no new
  shipping config switch, automatic selection policy or production-ready claim.
- A product follow-up needs schema validation, a VRAM-aware residency planner, prediction
  accounting, safe teardown, resume/eval coverage, and architecture/NF4 qualification.
  All arms use the same ordinary LoRA targets and dense BF16 arithmetic.
- Local current-upstream regressions: 62 passed, 4 skipped. Real-GPU prototype checks:
 13 passed, including two CUDA async-recycling tests. This is not a green full CI matrix.

## Retention and cleanup

Instance `56895623` was deleted and independently verified absent. Provider-observed
charges are USD 0.787000; the conservative estimate is
USD 0.947442, below the new USD 5 cap. Provider settlement can lag.
Other tasks and previous experiments are excluded from this new allocation budget.

Every export was authenticated, SHA256 checked and its original-file manifest verified
before acknowledgement. Canonical archives, original tensor paths and receipts remain
under `output/residency-vast-20261020/`. Verified duplicate transport chunks were removed;
byte-identical retained tensors may share a hard link. All original paths remain available.

A laptop-network outage interrupted the first controller during qualification export.
Deletion could not then be confirmed. Once connectivity returned, the same live instance
and verified fragments were resumed, with no source, workload or tolerance changes.
The original controller error and network-recovery amendment remain in the evidence.

[Machine-readable results](streaming-residency-results-20261020.json).
Frozen protocol/source and independent analysis are retained with the private evidence.

## Replay

The committed [frozen harness and inputs](evidence/streaming-residency-20261020/README.md)
include byte-identical tested helpers and per-file hashes for reproducing the study
over the recorded upstream commit, independently of this research branch's other work.
The commands below replay the larger locally retained tensor evidence.

```powershell
python output/residency-vast-20261020/analyze.py
python output/residency-vast-20261020/audit_source.py
python output/residency-vast-20261020/reconcile_cost.py --final
```

These commands do not rent a GPU. Reconciliation requires the already configured local
provider credential and performs read-only requests.
