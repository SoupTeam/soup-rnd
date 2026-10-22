# Physical 8 GiB cache study: resumed measurements

The previous [blocked report](../README.md) is historical. This resume preserves the
same frozen numerical gates and uses an exact-byte artifact transport supplied by
the coordinator. Production code, tolerances, model, initialization and data are unchanged.

## Accepted paired results

Positive reduction means a faster candidate. Each row has four paired process
medians; intervals use the prescribed bootstrap, not individual steps as independent samples.

| Candidate | Median paired time reduction | Bootstrap 95% | Positive pairs | Practical gate |
| --- | ---: | ---: | ---: | --- |
| Resident weights | 5.347% | [5.025%, 5.685%] | 4/4 | Pass |
| Retained activations | -0.431% | [-0.794%, -0.305%] | 0/4 | Fail |
| Both | 0.788% | [0.072%, 1.047%] | 4/4 | Fail |

Resident weights passed the gate in this fixed workload. The combined one-layer selection returned
0.788% paired reduction: all pairs were positive and its interval stayed above zero, but it missed
the 2% minimum. Retaining one layer's
activations independently was numerically valid but slightly slower in every pair.
The selections differ under the same memory cap, so these rows do not isolate an
additive interaction between the two mechanisms. Four pairs on one GPU are limited
evidence; they do not establish general model or hardware performance.

Machine-readable [results](results.json) contain every pair, eight measured step
times per process, tokens/s, allocator peaks, CPU RSS/high-water mark, sampled
device memory, setup/training/process times and final-adapter/loss checks. Small
original metrics/provenance/config files and device timelines are in [evidence](evidence/).
The full original tensors remain in the ignored archives listed in
[raw-local-archives.json](evidence/raw-local-archives.json); their SHA/size and
all-file decode receipts are published alongside the compact results.

## Numerical qualification obtained

On one physical RTX 4060, **all three candidates matched original streaming bitwise**
for every saved training tensor and every causal/padding semantic tensor. Independent
local replay checked every original element and reproduced the remote verdicts.
This is relative numerical parity on the fixed fixture, not a task-quality claim.

| Arm | Persistent decoder layers | Decoder checkpoints skipped |
| --- | --- | --- |
| Baseline | None | None |
| Resident weights | 16, 17, 18, 19 | None |
| Retained activations | None | 17 |
| Both | 17 | 17 |

The independent retained arm snapshots every autograd-saved alias of rotating base
weight storage. Its copies live through backward and are charged against the cap.
The combined arm borrows the immutable private slot instead. Ordinary prefetch/copy
events remain in force; missing backward lookahead is repaired at the next body's entry.
This prototype does not claim to be an optimal retention schedule or a production planner.

All arms use the same 7 GiB allocator cap and a prospective 0.75 GiB workspace reserve.
Selection uses baseline's measured reserved peak, charging 385,892,864 bytes per
resident decoder and an additional 512 MiB activation allowance for retained/both.
The different layer distributions are deliberate consequences of that common cap.

Frozen source: `d500ae0dca7c4da905f8d59c9866af2f8aaa824b`, with all 602 production
files independently verified byte-identical. Research source archive SHA-256:
`12063cfc2a3326b0c6cb4020758733b3a02168e8b1d8e3b3dc041f4b60784968`.
The original comparator remains unchanged: reference first, `rtol=.01`, `atol=1e-5`,
finite tensors, exact inputs/initial adapters, loss to three decimals, gradients
captured before reset and final adapters. Architecture count is 36 × 14 = 504.

Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, frozen dense BF16,
ordinary PEFT rank/alpha 4 on all seven projections, dropout 0, seed 792, nonzero
adapter generator 123, context 1024, batch 1, accumulation 2, common 32-row fixture.
Training data SHA-256: `313d17f90f4b62004a1d8232ba9bc0b19388a65e933c930ddc9aecfe7ffbf0f9`.
Timing initializer SHA-256: `4b0bffec36ae011fe0aab9779d2ec9d0c8c63a13770ac048a0be46550dc64d89`.
Pinned stack includes Torch 2.8.0+cu128, Transformers 5.17.0, PEFT 0.20.0 and TRL 0.29.1.
The card reports 8,188 MiB through `nvidia-smi`; CUDA reports 8,184,397,824 usable
bytes, with the allocator limited to 7,516,192,768 bytes. Native BF16 and idle
pre-staging state were checked. Processes were pinned to four GPU-local CPUs
22–25, with no enforced NUMA memory binding. Device timelines preserve clocks,
temperature and visible compute processes; power readings were unavailable.

The frozen upstream reports Liger status as `_liger_applied`, whereas the inherited
capture expects `_streaming_liger_patches`. `baseline_patch_metadata()` rejects
actual Liger/Fast-LoRA use and supplies only the empty report field; it changes no
production tensor arithmetic. A regression test covers the absent-field case.

## Raw evidence retained

The actual baseline capture has **2,037 tensors, 2,664,310,056 bytes**. It includes
four complete FP32 logits of shape `[1,1024,151936]`, 504 initial adapters, 1,008
gradient tensors across two optimizer steps, 504 final adapters, all 16 tensor-valued
forward inputs and all four FP64 losses. No logits or gradients were sampled or removed.

The full accuracy export contains **140 files / 11,840,913,568 original bytes**.
The standalone lossless archive is **1,090,017,280 bytes**, SHA-256
`ae05ff6e6c969df21e962a9ee3eba4f8bb32fec56dc48cfcf76c134bb83cf5ec`.
Every original file and chunk was reconstructed and verified before ACK. Complete
duplicate files share hardlinks; 8,935,520,874 duplicate decoded bytes were avoided.
The encoded transfer took 4,167.43 s; decode/verification took 20.56 s.

The coordinator's unmodified `lossless_results.py` SHA-256 is
`98ba251412e4d8d42781d0778b2f60a037d1a3434146cf2df6deb885e992579c`.
It uses chunk deduplication, reversible byte-plane shuffling, optional XOR against
an archived object and zlib. Each archive is self-contained and every original byte
survives. No tensor precision conversion, predicate changes or metrics-only substitution.
The compile chat owns the [common codec](../../harness/lossless_results.py) and its
tests; this folder records the executed hash and does not fork that implementation.

The separate transport receiver resumes a disconnected immutable archive from its
exact local byte offset. It never substitutes hashes for missing original data:
the completed archive must match authenticated size/SHA before decode, and every
decoded original must pass the codec checks before ACK. Seven fault-injection
tests cover repeated resets/timeouts, existing partial files, no-progress bounds,
corruption, oversized prefixes and reset after a complete payload. The accepted
timing rental also passed an actual SSH disconnect/resume preflight with offsets
`0 → 10,158,080`; the final 17,233,920-byte archive and all synthetic originals
were restored exactly. This validates transport, not model accuracy.

Numerical proof was obtained on instance 57170199. After successful retention a
control-channel timeout ended that rental. Timing resumes only on the **same exact
physical GPU UUID**, driver 580.95.05, source hash, package versions, pinned model
file hashes, data hash and initial adapter hash. `timing_worker.py` enforces these
identities before proceeding; this is not pooling accuracy across different cards.
Every timing pair independently checks all 14-update losses and final adapters again.
An earlier timing cohort on instance 57181254 lost its archive during a connection
reset and is void: none of its timings enter the accepted paired statistics. The
accepted series restarts all four pairs per candidate on instance 57186464 with
the tested resumable receiver. Failed startup/rental costs remain in the same USD5 share.

## Timing protocol

Each candidate has its own four fresh AB/BA/BA/AB pairs versus original streaming.
Every fresh process runs six warmup and eight measured optimizer updates with CUDA
synchronization. No compilation, external FlashAttention, Liger, Fast-LoRA or NF4 arm.
Timing archives are retained and ACKed after each candidate's four pairs.
The [run-order audit](evidence/run-order-audit.json) independently confirms the
required order, 24 distinct process IDs, successful exits and nonoverlapping
benchmark bodies from the retained raw provenance.

Practical gate: median paired reduction ≥2%, ≥75% positive pairs and bootstrap 95%
lower bound >0; 20,000 resamples using `random.Random(792)`. Pair medians, not pooled
optimizer steps, are the resampling units. No outlier removal or tolerance relaxation.

All arms carry the same lightweight copy/call counters. Reported decoder-copy
bytes are logical decoder payload, excluding vocabulary/head traffic, setup and
driver transfers; they are not a measured PCIe throughput or copy-wait duration.
Inner-forward calls are recomputation counts, not measured FLOPs. Retention's
saved-byte/snapshot counters accumulate over all 14 updates and can count aliased
nonpool views; they must not be read as simultaneously live or peak memory.

`analyze.py` replays original numerical predicates with bounded temporary chunks,
then checks every final adapter/loss and computes the original paired gate. Tests
compare that local replay against the original full comparator on FP32/BF16 tensors,
including a deliberately non-finite mutation. The original full comparator ran on
the GPU host before any timing qualification.

## Reproduction

Use `../prepare.py` to reconstruct the exact upstream plus preserved legacy overlay
in an isolated directory, then copy this folder's `candidate.py`, `bench.py`,
`worker.py` and `timing_worker.py` to `/root/stream8-cache-resume` on an already
qualified Linux CUDA host. Keep its checkout at `/root/stream8-cache-resume/repo`.
Use the recorded constraints, affinity, deterministic environment and shared codec.
No script here rents hardware. `timing_worker.py` requires authenticated retained
qualification on that exact GPU; otherwise run the full `worker.py` qualification.
Use the exact model preparation, pinning, telemetry, setup and target-test snapshots
in [executed-inputs](evidence/executed-inputs/) (remove their `.txt` suffixes). Their
paths already use `/root/stream8-cache-resume`; the legacy preparation helper's
generated helpers instead use `/root/stream8-cache`. Supply the
coordinator-owned codec at the recorded hash and the bounded lifecycle/ACK driver.
Rebuilding a tarball can change its metadata hash: it requires full qualification,
whereas timing-only reuse requires the original archived source bytes and identities.

To independently replay the retained original tensors, final adapters and paired
statistics from the recorded local artifact roots:

```powershell
python benchmarks/stream8-cache-20261022/resume/analyze.py `
  --root output/stream8-cache-resume-20261022-f `
  --reference output/stream8-cache-resume-20261022-d `
  --output output/cache-independent-replay.json
```

Authenticate encoded size/SHA against the published bundle indices before using
the shared codec to restore missing decoded directories. Preserve the verification
receipts; analysis requires all originals. CPU replay uses bounded chunks, preserving
the original FP32/BF16 dtypes and reference-first predicate. It does not rerun training.

Raw/private material stays in ignored `output/stream8-cache-resume-20261022*`.
The initial blocked record and each failed startup are retained as history.
The lead's Mistral7B NF4/context512 quote remains unverified; this dense-BF16 Qwen
experiment neither independently confirms nor directly compares with that claim.

## Resource and process accounting

These are maxima across fresh runs, except the explicitly labelled medians. Baseline resources
include twelve controls; candidate resources include four runs. Only the four paired reductions per
candidate determine its speed gate.

| Arm | Allocated GiB | Reserved GiB | Sampled device MiB | CPU RSS high-water GiB |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 4.142 | 4.709 | 5061 | 18.436 |
| Resident weights | 5.579 | 6.162 | 6549 | 18.428 |
| Retained activations | 4.705 | 5.150 | 5513 | 18.429 |
| Both | 4.705 | 5.143 | 5505 | 18.435 |

Allocator peaks above cover measured updates; whole-run allocator peaks also appear in JSON and stay
below the cap. Device samples cover benchmark-body setup/training/save at approximately five-second
intervals and can miss peaks. No sample showed multiple compute processes. CPU high-water values
come from `/proc/self/status`, not a device-memory estimate.

| Candidate | Median step A → B, s | Median setup A → B, s | Median full process A → B, s |
| --- | ---: | ---: | ---: |
| Resident weights | 9.973 → 9.440 | 37.858 → 38.617 | 191.886 → 185.573 |
| Retained activations | 9.950 → 10.007 | 38.752 → 38.686 | 192.971 → 193.241 |
| Both | 9.967 → 9.892 | 39.347 → 39.659 | 193.816 → 193.539 |

A is the row's matched original-streaming control. These descriptive medians need not reproduce the
median of paired ratios. Full-process wall time includes Python imports and teardown; setup is a
subset. Per-process training wall time includes warmup and final checkpoint saving. The step gate
excludes setup, artifact transfer, imports and saving. Compilation was disabled in every arm.

| Arm | Decoder loads/update | Logical decoder GiB/update | Inner calls/update | Private weight GiB |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 136 | 48.877 | 144 | 0.000 |
| Resident weights | 120 | 43.127 | 144 | 1.438 |
| Retained activations | 136 | 48.877 | 142 | 0.000 |
| Both | 132 | 47.440 | 142 | 0.359 |

The independent retained arm copied 10,805,000,192 saved-weight bytes cumulatively over 14 updates;
the combined arm copied 0, borrowing its immutable resident weights instead. This supports the
measured mechanism but does not isolate copy-wait time or provide a general retention cost model.
All 504 adapters changed from the exact initializer in every timing run; every matched final adapter
tensor and all 14 losses were bitwise equal.

## Cost, cleanup and checks

Conservative cumulative upper estimate: **USD 3.0361 / 5** for this share of the original USD10
stage, including all failed attempts. This is not a final settled invoice. The last attempt is
charged its full 45GB ingress / 30GB egress / USD0.40 miscellaneous reserve, a storage-margin hourly
rate, and teardown time, rather than only observed transfer bytes.

| Instance | Outcome | Conservative additional USD |
| --- | --- | ---: |
| 57114865 | initial transport preflight; no model staged | 0.1500 |
| 57167617 | provider port collision; no model staged | 0.1500 |
| 57168037 | metadata-only harness failure before training | 0.7500 |
| 57169760 | local console encoding failure during setup | 0.1500 |
| 57170199 | accepted full numerical qualification | 0.5233 |
| 57181254 | void timing: incomplete archive; no statistics used | 0.3905 |
| 57186464 | accepted twelve timing pairs; all archives retained | 0.9223 |

All seven owned instances were independently confirmed absent in a fresh paginated provider listing
after controller cleanup. See [cost and cleanup](evidence/cost-and-cleanup.json).

Validation: 46 local targeted tests passed, ten CUDA tests skipped on the CPU host; 18
actual-Qwen/PEFT target-stack CPU/CUDA tests passed without skips before timing (two inherited
unknown-mark warnings). Seven local transport fault-injection cases and the intentional live
disconnect/resume passed. Full repository Ruff and Git whitespace checks passed. Local source
imports used the frozen upstream tree. Hosted CI was not awaited. See
[checks](evidence/checks.json), [local JUnit](evidence/local-tests.xml), [GPU test
log](evidence/target-regressions.log) and [executed-code audit](evidence/execution-code-audit.json).
