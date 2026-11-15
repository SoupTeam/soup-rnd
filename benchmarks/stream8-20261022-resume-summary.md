# Physical 8 GiB streaming studies: combined resumed results

Resident weights passed the fixed practical timing gate on both tested GPUs.
Activation retention and the combined one-layer selection passed numerical
qualification but failed the practical timing gate. Both resident-only Inductor
variants failed the original accuracy gate and were excluded from timing.
Production source and numerical tolerances remain unchanged.

Positive reduction means less time per optimizer update. Each timing row uses
four independent fresh-process pairs, six warmup and eight measured updates per
process, AB/BA/BA/AB order and 20,000 whole-pair bootstrap resamples with seed 792.

| Physical GPU | Candidate | Layers | Median paired time reduction | Bootstrap 95% | Positive pairs | Practical gate |
| --- | --- | --- | ---: | --- | --- | --- |
| RTX 3060 Ti | Resident weights | 16–19 | 2.386% | [2.233%, 2.453%] | 4/4 | Pass |
| RTX 4060 | Resident weights | 16–19 | 5.347% | [5.025%, 5.685%] | 4/4 | Pass |
| RTX 4060 | Retained activations | 17 | -0.431% | [-0.794%, -0.305%] | 0/4 | Fail |
| RTX 4060 | Both | 17 | 0.788% | [0.072%, 1.047%] | 4/4 | Fail: below 2% |

The gate requires a median reduction of at least 2%, at least 75% positive pairs
and a positive lower confidence bound. All 16 pairs have equal 14-update loss
trajectories and bitwise-equal final adapters within each pair. Training and
causal/padding captures for the eligible arms match ordinary streaming bitwise.
The fixed workload is dense BF16 Qwen3-8B, ordinary PEFT rank/alpha 4 on all seven
projections, context 1,024, batch one and accumulation two.

Resident weights increase allocated peak from 4.142 to 5.579 GiB and reserved
peak from 4.709 to 6.162 GiB. In the cache cohort, retained/both allocated peaks
are 4.705 GiB; reserved peaks are 5.150/5.143 GiB. Different selected layer counts
under the common 7 GiB cap and 0.75 GiB reserve prevent interpreting these rows
as an isolated additive interaction. These four-pair results are conditional on
the tested workloads and devices, not general performance guarantees.

The latest 8B cohorts have identical pinned model files, all 504 initial adapter
tensors and all 16 actual training input tensors. Performance is reported
separately for each physical GPU and lease. Earlier RTX 3070 input-initialization
differences are historical and do not describe these latest cohorts.

## Native capacity and excluded compilation

Qwen3-14B completed two native optimizer updates, with all 2,261 saved tensors
finite, all 560 final adapter tensors changed and all 1,120 gradients nonzero.
Its 1,127 semantic tensors and executed causal/padding checks were also retained
and independently reviewed. GPU allocated/reserved peaks are 5.132/5.674 GiB;
CPU high-water RSS is 33.054 GiB. The fixed planner selects no resident decoder
after charging the 0.75 GiB reserve, so no optimized 14B timing is claimed.

32B was not staged because 63,534,301,184 usable host bytes were below the
74,114,181,120-byte pinned-store/reserve requirement. The 24B fallback passed
the RAM check but its USD 4.8761 conservative staging estimate exceeded the USD
4.8 admission ceiling. Neither untrained model establishes a GPU capacity limit.
Thus **14B is the largest verified short native training run; 8B is the largest
model with qualified optimization timing and full 14-update trajectories**.

On the RTX 3060 Ti, default/precise Inductor each executed four selected resident
decoder wrappers with a real compiled graph. Both failed all 504 final adapter
tensors and four logits tensors; gradient failures were 694/656, respectively.
Both also failed three-decimal loss agreement. All values remained finite and
inputs/initial adapters exact. The failed compiled raw captures are complete;
there are no compiled speed claims and no relaxed accuracy predicates.

## Evidence, cost and publication

- [Compile/residency report](stream8-compile-20261022-resume/RESULTS.md) and
  [machine-readable results](stream8-compile-20261022-resume/results.json).
- [Cache report](stream8-cache-20261022/resume/README.md) and
  [machine-readable results](stream8-cache-20261022/resume/results.json).
- [Independent integrated evidence](evidence/stream8-20261022-resume-integrated/),
  including all three cache trajectory replays, input/source audits, a fresh
  rehash of 13 complete canonical archives and verification of 167 published
  original Git blobs against canonical file hashes.

Full raw tensors remain in the documented ignored local canonical archives.
Every qualifying result's original files were SHA-verified before lease cleanup.
Earlier partial transfers remain explicitly recorded. The failed F14 archive
remains incomplete, while its full scientific trace was recovered exactly by
matching its expected original SHA to the fresh retained G trace. No historical
partial archive is represented as a complete one.

Conservative series estimates are **USD 4.395906 / 5** for compile/residency and
**USD 3.036059 / 5** for cache: **USD 7.431965 / 10 total**. These include prior
attempts, the initial preflight spending, observed network and cleanup margin;
they are estimates rather than invoice reconciliation. No cache allowance was
transferred. A fresh paginated provider check confirms all **15 study instance
IDs absent**. Unrelated instances were never modified.

Combined local checks pass **64 tests, 10 CUDA skips**, plus full Ruff. Target
GPU regressions separately pass 16 compile and 17 cache tests. Hosted CI is not
awaited. This change publishes research harnesses, results and evidence; it adds
no production configuration or release.
