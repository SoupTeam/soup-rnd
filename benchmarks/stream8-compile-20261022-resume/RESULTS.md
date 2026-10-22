# Physical 8 GiB residency and Inductor: resumed evidence

The [original blocked attempt](../stream8-compile-20261022.md) remains historical.
This continuation uses the same USD 5 compilation share, including the earlier
USD 0.144429. It changes no production source or numerical tolerance.

## Qualified resident timing

Four fresh-process pairs in AB/BA/BA/AB order each execute six warmup and eight
measured optimizer updates. Positive reduction means less time per update.

| Pair | Baseline median, seconds | Resident median, seconds | Time reduction |
| --- | ---: | ---: | ---: |
| AB | 5.970576 | 5.837229 | 2.233% |
| BA | 5.989684 | 5.842737 | 2.453% |
| BA | 5.991754 | 5.850807 | 2.352% |
| AB | 5.995708 | 5.850630 | 2.420% |

Median paired reduction is **2.386%**, with **4/4 positive pairs** and a bootstrap
95% interval of **[2.233%, 2.453%]**. The prespecified practical gate passes:
median at least 2%, at least 75% positive pairs and a positive lower interval bound.
The bootstrap resamples whole process pairs 20,000 times with `random.Random(792)`;
individual optimizer updates are not treated as independent replicates.

Every pair has identical initial-state/data hashes, all 14 losses equal and all
504 final adapter tensors bitwise equal. Independent replay verifies eight unique
process IDs, physical GPU identity, affinity, 28 fixed-shape microbatches per
process, BF16 AMP, deterministic algorithms and the exact executed benchmark SHA.
The timing archive contains 186 verified original files, is 116,070,400 bytes and
has SHA-256 `f2bd436a94242b57d330e66b6bc159949442dd19e80b5919efa743d020daa809`.

This gain costs memory: whole-run peak allocated rises **4.142 → 5.579 GiB** and
reserved rises **4.709 → 6.162 GiB**. At the final measured update, decoder loads
fall **136 → 120** and logical streamed bytes **52,481,429,504 → 46,307,143,680**.
These are logical copy counters, not measured PCIe wire traffic. Timings cover
warm steady-state optimizer updates; setup/preload/process durations are retained
separately. Four pairs on this single GPU do not establish broader performance.

## Independently replayed numerical result

Qwen/Qwen3-8B on the physical RTX 3060 Ti qualified four resident decoder layers,
indices 16–19. Every saved training tensor and semantic tensor matched ordinary
streaming bitwise. Both actual Inductor variants failed the original numerical
gate and were excluded from performance timing.

| Arm | Training capture | Semantic capture | Performance timing |
| --- | --- | --- | --- |
| Resident weights | 2,037 tensors, bitwise equal | 1,015 tensors, bitwise equal | Qualified |
| Resident + default Inductor | Failed | Not attempted after primary failure | Excluded |
| Resident + precise Inductor | Failed | Not attempted after primary failure | Excluded |

Both compiled variants executed four compiled decoder wrappers with one unique
graph and 112 captured calls. Default compilation failed 504 final adapter tensors,
694 gradient tensors and all four logits tensors. Precise compilation failed 504,
656 and four respectively. Both failed the separate three-decimal loss gate,
although the losses passed the elementwise relative tolerance. All candidate
elements were finite. Inputs and all 504 initial adapter tensors remained exact.
Independent bounded-memory replay of all saved elements reproduced these counts.

The original comparator remains reference-first `torch.allclose`, `rtol=.01`,
`atol=1e-5`, with exact input/initial tensors, finite values, three-decimal loss,
gradients captured before reset and final adapters. A failed optimization is
reported as failed even when a narrower subgroup passes.

## Fixed workload and implementation

Upstream production is frozen at `d500ae0dca7c4da905f8d59c9866af2f8aaa824b`.
All 602 production files match independently. The research source archive has
SHA-256 `2b3ce103921a5713849d120b6bd3d74fc5c895124b65bbea93f06a2658bc3f6a`.
The [public benchmark](bench.py) is byte-identical to the executed GPU benchmark,
SHA-256 `080b4de75b8f228f367d7f9de2e4b7be7793b2b3c1e6c06c27ec0b0638b10f1b`.

The base is frozen dense BF16, ordinary PEFT rank/alpha 4 on all seven projections,
dropout zero, seed 792, nonzero adapter initialization using CPU generator 123.
Context is 1,024, batch one and accumulation two. The common 32-row training fixture
has SHA-256 `313d17f90f4b62004a1d8232ba9bc0b19388a65e933c930ddc9aecfe7ffbf0f9`.
The pinned Qwen3-8B revision is `b968826d9c46dd6066d109eabc6255188de91218`.

The RTX 3060 Ti is UUID `GPU-72f1f8b8-6875-712e-9ed7-97bd231370ef`.
The host preflight established 62,411,112,448 usable bytes of RAM. Three pre-staging
GPU samples were idle. The allocator is capped at 7 GiB with a prospective 0.75 GiB
workspace reserve. CPU affinity is 0–3; this is not an enforced NUMA memory binding.
Torch is 2.8.0+cu128, Transformers 5.17.0, PEFT 0.20.0 and TRL 0.29.1.

Compilation changes only selected resident decoder `inner.forward` functions.
Inductor uses fullgraph, disabled CUDA graphs and disabled error suppression;
scheduling, buffer ownership and checkpointing stay eager. No compilation-failure
fallback is used. Inclusive compiler metrics and process/setup times are retained
separately; nested compiler durations must not be added together. These captures
do not establish a cold-start or steady-state compilation speedup.

The semantic path executes fixed-shape causal and masked-padding perturbations
with exact internal checks. Its saved initial/gradient/batch/loss/logits tensors
are independently replayed. Counterfactual arrays are not saved separately;
provenance records successful execution of the internal checks. The separate
unpadded cross-shape comparison is diagnostic because SDPA kernel choice can
change with shape; it is not the padding acceptance gate.

## Exact raw retention and earlier attempts

The complete fresh accuracy archive is 2,382,039,040 bytes, SHA-256
`aa08cf54f3ba7c5e55a34890429138ece90b219e45ab254ce8162317168477ef`.
All 103 original files, totaling 11,645,294,945 logical bytes, were reconstructed
and individually SHA-verified before ACK. The separate first baseline archive
is 1,046,620,160 bytes, SHA-256
`75b0d07a713abaf2ce2d77ed100295c57f01334b9744727a0a0eeaa63d0ae477`.
All its 50 original files were verified. The baseline alone has 2,037 tensors and
2,664,310,056 bytes: complete logits, inputs, losses, initial/final adapters and
both pre-reset gradient states. Every initial B tensor is nonzero and all 504
final adapter tensors changed during actual training.

Canonical archives are self-contained. Exact encoded-object reuse retrieves only
missing immutable byte spans; reconstructed canonical bytes must match the remote
whole-archive SHA before the unchanged codec verifies every original file. The
codec SHA is `98ba251412e4d8d42781d0778b2f60a037d1a3434146cf2df6deb885e992579c`.
There is no dtype conversion, sampling or substitution of metrics for raw data.
Derived decoded traces can be evicted after verification and replay to conserve
local disk; their original bytes remain restorable from retained canonical archives.
Eviction ledgers identify exactly which decoded copies were removed.

Earlier transport failures are preserved and are not counted as complete evidence:

| Attempt | Complete evidence retained | Incomplete archive retained |
| --- | --- | --- |
| A, RTX 3070 | Baseline and resident training captures, independently bitwise equal | Accuracy archive: 1,032,847,360 / 2,389,975,040 bytes |
| B, RTX 3050 | Small diagnostics | Baseline: 1,021,509,632 / 1,046,999,040 bytes |
| C, preflight only | Diagnostics; no model work | None |
| D, RTX 3050 | Small diagnostics | Baseline: 677,380,096 / 1,046,999,040 bytes |
| E, preflight only | Diagnostics; no model work | None |

A completed remote gate does not make an incomplete local archive complete.
Historical failed compilation captures therefore have a narrower retention claim
than the complete fresh F cohort. Useful-progress retries, negative SSH exit
classification and immutable range resume have regression coverage. Synthetic
transport tests and metadata-only inventory are not model-performance evidence.

The later F 14B export failed after receiving about 102.5 MB of encoded objects:
an incomplete SSH response with status zero was incorrectly treated as fatal.
The controller deleted that lease; its 14B canonical archive was not completed.
The fixed receiver treats a short successful response as resumable EOF, preserves
useful byte progress and still rejects ten minutes without progress. Both short
successful responses and empty successful responses have regression coverage.

A final G lease repeats 14B under the same USD 5 share, with prior conservative
spend rounded up to USD 3.95 and a two-hour deadline. It does not repeat 8B.
It uses the same physical GPU UUID, frozen source, benchmark and model revision.
Its complete 14B scientific trace has SHA-256
`bb49a3cbeaa165d7c6822e076b8b938d92c751273b069c648efcce2e25596d12`,
exactly matching the expected original SHA in the failed F manifest. These
2,746,657,872 original bytes were therefore recovered exactly. Some other F14
run files remain missing; its original archive is still classified as incomplete.

## Observed training capacity and stop boundaries

The largest independently retained native model trained here is **Qwen3-14B for
two optimizer updates / four microbatches**. The largest model with qualified
optimization timing and complete 14-update trajectories is **8B**. A universal
maximum is not established.

| Model | Actual result | Optimization/performance boundary |
| --- | --- | --- |
| 8B | Complete training and semantic captures; four fresh 14-update process pairs | Resident passes accuracy and practical timing; both compiled arms fail accuracy |
| 14B | Two native optimizer updates, all 2,261 tensors finite; all 560 final adapters changed, 280 initial B tensors and 1,120 gradient tensors nonzero | No resident layers selected under the fixed reserve; no optimized timing |
| 32B | Not staged | 63,534,301,184 usable host bytes vs 74,114,181,120 required for one pinned store plus 8 GiB reserve |
| 24B fallback | Not staged | Conservative staging estimate USD 4.8761 exceeded the USD 4.8 admission ceiling |

The 14B run also executes the native causal/padding checks and retains all 1,127
semantic tensors. Independent local replay checks every saved element for finite
values; provenance records successful execution of the exact internal invariants. These are short-run
baseline sanity checks, not optimized parity, long-run stability or task-quality
evidence.

The 14B allocated/reserved peaks are **5.132 / 5.674 GiB**, with CPU high-water
RSS **33.054 GiB** on the host that passed a 64,212,733,952-byte available-RAM
preflight. A decoder costs 660,623,872 bytes, about 0.615 GiB. After the 0.75 GiB
reserve only about 0.576 GiB remains below the 7 GiB cap, so the fixed planner
selects zero resident decoders. Baseline training fits; residency is excluded by
the prospective reserve. Neither the untrained 32B nor 24B establishes a GPU
capacity failure.

The complete G baseline archive is 1,194,967,040 bytes, SHA-256
`65947a24c1200fca52556f7c04b1ee7bde7ca505f073d32bd148d7de4b4cea99`,
with 50 original files verified before ACK. The semantic/completion archive is
85,299,200 bytes, SHA-256
`61be047b4778e82c5e01229a72e4fae93613167c177ad01e696b961baacf27d4`,
with seven originals verified. Complete stop/final metadata archives are also
retained. [All eight F/G canonical archives](evidence/raw-local-archives.json)
are self-contained and were independently rehashed before publication.

## Accounting, cleanup and validation

The final conservative estimate is **USD 4.395906 / 5** for this entire compile
series, including the initial USD 0.144429 and all intervening attempts. The
separate cache series totals **USD 3.036059 / 5**, giving **USD 7.431965 / 10**.
This includes observed network, disk-inclusive lease time, extra time and money
margin; it is not invoice reconciliation. The cache share was not transferred.
No October 21 spending is added to this October 22 authorization.

Unconditional exact-ID cleanup and independent scoped time/money guards protect
each lease. A fresh independent paginated provider listing verifies all **15**
instance IDs from the two studies absent. Unrelated instances are never modified.
See the [final accounting and cleanup proof](evidence/final-cost-and-cleanup.json).
The four complete F and four complete G archives were collected and all original
files verified before their corresponding lease cleanup.

The combined publication checkout passes **64 CPU tests with 10 CUDA skips** and
full Ruff over production, scripts, tests and benchmarks. The compile GPU separately
passes 16 target regressions; the cache GPU separately passes 17. CPU checks do not
replace the pretrained-model evidence. Full hosted CI is not awaited.

The public benchmark Git object preserves the exact executed bytes, including
line endings. Evidence snapshots disable Git newline conversion. The published
legacy runtime modules preserve the executed AST, and exact original source
snapshots are retained alongside their hashes.

## Relationship to the cache study

The [separate cache study](../stream8-cache-20261022/resume/README.md) uses another
physical GPU. Its latest cohort and this F cohort have identical pinned model
files, all 504 initial adapter tensors and all 16 actual training input tensors.
Cross-study input identity does not permit pooling performance measurements across
these GPUs or leases. Historical RTX 3070 initialization differences are retained
as historical facts and do not describe this fresh F cohort.

The [harness guide](README.md) documents the code and scoped CPU/GPU validation.
Machine-readable receipts, exact execution inputs, small original logs/configs,
independent replays and archive manifests accompany the final results. Multi-GB
tensor archives remain in the documented ignored local output directories.
