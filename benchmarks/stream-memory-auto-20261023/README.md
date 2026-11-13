# Automatic streamed-weight residency: implementation and physical 8 GiB qualification

Status: **8B numerical qualification passed bitwise; a small 1.919% step-time gain
failed the unchanged 2% practical threshold. Both instances deleted and independently
verified absent.** This is a working opt-in capacity feature, not a general acceleration
claim or an optimal joint memory allocator.

Machine records and exact helper sources are in [evidence.zip](evidence.zip), keeping raw
measurements out of the review diff. [The archive manifest](evidence-manifest.json) records
every member's size/hash and the ZIP hash. Paths below shown as code are inside that ZIP.
Public [summary data](result-summary.json) and replay programs remain directly reviewable.

The ordinary decoder pool cycles frozen weights through rotating slots even when spare
VRAM could retain selected layers. This change adds native private slots and the opt-in
`training.stream_resident_layers: auto` setting. Its setup probes measure the configured
forward/backward shape, reserve future AdamW state/workspace and validate a candidate.
The host store and arithmetic are unchanged; ordinary streaming remains the default.

This is conservative **capacity selection**, not an optimal performance allocator. It
does not implement a general VRAM/RAM/NVMe scheduler or a persistent disk-tier RAM cache.
The first supported production scope is dense text SFT, CUDA, RAM tier, an admitted AdamW
optimizer, with dense or packed NF4 weights. Large-model NF4 throughput is not established
by the tiny-model correctness tests below. The new tiny NF4 composition/trajectory checks
are CPU checks; this paid large-model cohort qualifies dense BF16.

## Research basis

The inspected [Alpamys PR #2152](https://github.com/MakazhanAlpamys/Soup/pull/2152)
retains activations to avoid checkpoint recomputation. It is a different allocation from
the immutable weights retained here. Its published Mistral NF4 results are author results;
this experiment does not repeat or refute that full study. The compatibility check below
is enough to establish a bounded starting point for a later shared budget.

Soup already has RAM/NVMe tiers, asynchronous prefetch and an opt-in NF4 scale cache. This
change adds native weight residency and automatic measured capacity selection. A joint
time-based choice among weights, activations and scales, and a partial persistent RAM cache
for the disk tier, remain separate work. The profiling/reuse ideas in
[10Cache](https://arxiv.org/abs/2511.14124) inform that next step; no paper speedup is assumed.

## Source and controls

- Base: `f8833454c9a7e60af5c7d4b038600616f8979b25`, SoupTeam/soup-rnd PR #5.
- Frozen GPU source archive: `adfc15617878f3592f3bd49c37fd69ef7b94f8e4e2d5b177e409a2e42f7e1bb7`.
- CPU source hashes: `cpu-source-manifest.json`; [prospective protocol](protocol.json),
  [executable GPU protocol](gpu-protocol.json).
- [Exact executed helpers and manual replay](executed/README.md), including the source
  newline mapping, and [independent retained-results replay](replay_results.py).
- Baseline `stream_resident_layers: 0` versus production `auto` on the same physical GPU,
  weights, tokens, nonzero adapters, optimizer and attention backend.
- Original reference-first numerical predicates retained: `rtol=0.01`, `atol=1e-5`,
  exact inputs/initial adapters, finite values, loss to three decimals, pre-reset
  gradients, updated adapters, causal/padding checks before timing.
- Four fresh-process AB/BA/BA/AB pairs, six warmup and eight measured optimizer updates;
  fourteen-update final adapters and losses checked for each pair. Practical gate:
  median paired time reduction at least 2%, at least 75% positive pairs, whole-pair
  bootstrap 95% lower bound above zero. Setup and planner overhead are separate.
- New stage has a total USD10 cap, including attempts, setup, transfer and cleanup.
  Historical unspent allowances are not added. Raw tensors use lossless verified retention.

Reported step VRAM comes from the eight individually reset measured-update arrays in
`metrics.json`. Setup/whole-run provenance fields sample allocator peaks and are not a
guaranteed maximum over every allocation during setup. Input-token throughput counts
2048 input positions per update; it is not a count restricted to supervised labels.
Resident selections are recorded for each process and may vary with measured headroom.
These synthetic arithmetic rows establish numerical equivalence and bounded performance,
not convergence or downstream task quality.

## CPU evidence

`cpu-validation.xml`: **346 passed, 24 hardware skips** in the combined scoped run.
Full-scope Ruff and 317 changelog-fragment validation pass. Two pre-existing worker import
ordering errors were corrected without changing arithmetic; historical executed `.py.txt`
records and their evidence were not altered.
The bounded-memory replay verifier has [eight passing controls](replay-validation.xml),
including identical NaN/Inf rejection and mutations routed through the original tolerances.

[Independent review evidence](review-evidence/README.md): three deliberately broken pool
variants killed after their unmodified controls passed. Transactional allocation cleanup
is checked while the failure traceback is retained, and resident no-op loads must not
read the source or increment load/generation counters.

An in-memory port to exact Alpamys PR #2152 head
`aa025afbb0adcd322c1fa72f31268b8e7af24bc2` preserved its original generation constructor.
Dense/NF4 tiny models with full/partial activation retention matched ordinary recompute
bitwise in loss and every nonzero adapter gradient (four CPU combinations). This proves a
bounded composition check, not the author's GPU speedup or an integrated joint auto policy.
An activation-auto policy must also subtract the explicit optimizer/setup reserve from
its remaining cap; merely running two independent planners is insufficient.

## Independent retained-result replay

From the feature checkout with its dependencies available and `PYTHONPATH=src:.`
on Linux (`src;.` on Windows), provide every incremental retained bundle needed:

```bash
python benchmarks/stream-memory-auto-20261023/replay_results.py \
  /path/bundles/impact-baseline-8/retained \
  /path/bundles/impact-accuracy-8/retained \
  /path/bundles/impact-timings-8/retained \
  --output /path/independent-replay.json
```

The utility reads tensors in place and SHA-checks the archived comparator helpers.
Four complete timing pairs are required for a performance verdict; partial timing
remains pending. An explicit saved auto-capacity fallback with empty actual resident
indices completes an accuracy-only verdict and reports no timing. Absent optional
models remain not run. The original comparator may differ only in CRLF/LF source
representation when exact CRLF reconstitution recovers its frozen size and SHA-256;
all numerical tolerances remain unchanged.

When both complete trace files have identical SHA-256 and size, replay independently
checks every tensor for finiteness in bounded slices and derives the original comparison
predicates from exact equality. Equal NaN/Inf files still fail. Small unequal traces use
the unchanged archived comparator; larger unequal traces explicitly refuse low-memory
admission instead of receiving a numerical verdict. No full trace is converted to double
by the identical-file path.

## Paid attempts

Attempt A (`57386126`) was deleted before package/model download or GPU training. The
exact-byte network calibration passed, but its measured throughput made the prospective
first-cohort retention estimate exceed the initial four-hour deadline. The controller
refused and independently verified deletion. Its conservative cost accounting is
USD0.266485, including an unobserved-traffic allowance; this is not a settled invoice.
Records: `attempt-a-stage-ledger.json`, `attempt-a-network-preflight.json`.

Attempt B uses a prospective eight-hour maximum deadline within the same dollar cap,
restores four GPU-local physical-core pinning, and reuses current-stage donor objects only
after exact encoded-byte verification. The canonical archive and every original file are
still reconstructed and hashed before acknowledgment. No numerical or raw-retention gate
was weakened. The original 96 GB RTX3060Ti offer disappeared; the fresh matched comparison
is on a physical 8 GB RTX3070. Hardware results will not be pooled with earlier cards.

## Physical 8 GB GPU result

Qwen3-8B dense BF16, context 1024, batch 1, accumulation 2, ordinary PEFT rank/alpha 4
on all seven projections. The physical card is RTX3070, with 8192 MiB reported by
`nvidia-smi` and 8,222,998,528 usable bytes reported by CUDA. Four distinct GPU-local
physical CPU cores were pinned; NUMA memory binding was not enforced. This is the RAM
tier. The container exposes an NVMe device but its root mount is overlay, so this run
does not establish an NVMe backing device or SSD performance.

| Measurement | Ordinary streaming | Production `auto` |
| --- | ---: | ---: |
| Median of process-median update times | 5.4372 s | 5.3329 s |
| Processed input positions/s | 376.665 | 384.035 |
| Maximum measured allocated VRAM | 4.142 GiB | 5.579 GiB |
| Maximum measured reserved VRAM | 4.709 GiB | 6.189 GiB |
| Maximum host high-water RSS in timing processes | 20.431 GiB | 20.430 GiB |
| Actual private decoder indices, every timing process | none | 16, 17, 18, 19 |
| Steady decoder loads per optimizer update | 138 | 122 |

Median paired time reduction is **1.919119%**, whole-pair bootstrap 95% interval
**[1.875267%, 1.956550%]**. All four pairs are positive, with reductions
1.956550%, 1.939416%, 1.875267%, 1.898823%. The practical gate remains **failed**:
rounding this result to 2% does not make it pass. The observed small reduction is real
for this bounded workload; it does not establish a material gain on other GPUs/models.

The plan consistently retains 1,543,571,456 bytes (1.438 GiB) of decoder weights. Its
forward/backward probes plus allocation take 6.446–6.495 s in timing processes, additional
to ordinary setup. At the observed approximately 0.104 s/update saving, about 63 updates
would amortize this planner cost alone; this is an extrapolation, not a measured
end-to-end long-training result. Short runs can therefore take longer overall.

Both original numerical gates pass bitwise: **2037 primary tensors** and **1015
causal/padding tensors**, including exact inputs/initialization, logits, pre-reset
gradients and updated adapters. Each of the four full fourteen-update pairs has identical
losses and all 504 final adapter tensors equal bitwise. No tolerance changed.

The independent CPU replay (`independent-replay.json`) recomputes the timing verdict
exactly, verifies eight distinct process IDs and the prescribed order, and repeats all
fourteen-update comparisons. For the two large trace comparisons, it independently
hashes both complete serialized files and scans every tensor for finiteness in bounded
slices. Identical finite serialized traces establish a stronger equality proof than the
original tolerance predicates. Unequal large files are refused by its local-memory
admission; they are never accepted by a looser predicate. The GPU worker also ran the
original comparator unchanged.

The scoped CUDA-environment run passed **92 tests**. Original result files (`gpu/`),
[summary](result-summary.json), worker decision (`gpu/worker-summary.json`) and
GPU JUnit (`gpu/scoped-regressions.xml`) are retained. Optional 14B was **not run** because
8B failed the predeclared practical speed gate, despite sufficient money/time. No new
27B fit, NF4 GPU throughput, compile result or downstream quality claim follows.

## Retention, spending and cleanup

All five bundles were reconstructed exactly and every original file hash was checked
before acknowledgment. The retention index (`raw-retention-index.json`) publishes the
original paths/sizes/hashes and exact-byte receipts. Full raw tensors and canonical
archives remain in the local `soup-memory-impact-20261023-b/bundles` store; they are not
hosted as public multi-gigabyte Git blobs. Public files include configurations, input
data, raw timing arrays, provenance, model-file hashes, environment, logs and JUnit.

Current-stage deduplication reused 1,029,643,341 encoded bytes for the accuracy bundle;
59,913,127 new bytes crossed the network. All original bytes were still reconstructed.
The local replay uses bounded slices because the laptop had about 4.4 GiB free RAM.
This changes neither the stored captures nor any numerical tolerance.

The final ledger (`lease/stage-ledger.json`) conservatively accounts for **USD0.765158
of the new USD10 cap**, leaving **USD9.234842**. This includes both attempts, setup,
telemetry-based transfer and unobserved-traffic/cleanup allowances; it is not a settled
Vast invoice. Attempt B elapsed 2898 s at USD0.153333/hour including its requested disk.
The deletion receipt (`lease/vast-deletion.json`) and a separate full-account
API check (`lease/independent-cleanup-check.json`) confirm both stage instances absent
and **zero account instances**. No paid process remains for this experiment.

## What this establishes and what comes next

Native private slots and opt-in automatic capacity selection work on the qualified 8B
workload without arithmetic changes. They remove manual layer-count selection and reduce
transfers. This RTX3070 result adds a small measured benefit; the larger historical gains
on other configurations must not be substituted for it.

The next selection policy should profile exposed transfer waits versus recomputation,
then share one cap among weights, activations and scale cache. It should optimize time
with a reserve, rather than automatically spend every spare byte on weights. A narrower
candidate is resident-aware prefetch: only while the current layer occupies a private
slot, look ahead to the next dynamic layer. Naive skipping from a rotating slot can
overwrite its current weights. This remains an unmeasured hypothesis requiring profiling,
owner/generation regression checks and a separately qualified GPU cohort. No Triton kernel,
resident-aware prefetch change, joint allocator or RAM/SSD cache is implemented here.
