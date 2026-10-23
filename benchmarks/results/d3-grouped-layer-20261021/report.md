# D3: prospective grouped-expert layer comparison, 2026-10-21

## Scope and outcome

This is a forward-only, single trained OLMoE routed-expert layer comparison on an
RTX 3060 12 GB, using 512 synthetic hidden states with trained weights and real
router assignments. It completes the committed **layer-performance series**;
**it does not close D3, validate whole-model training, or change production defaults**.

The source and executable protocol were committed and pushed as
`0aef6904af8b6cbbc1d9119fcf687d1d0c170f04` before the first formal arm. The clean
Linux checkout and all 19 pinned source/protocol Git-blob hashes match the raw
records. The implementation base is the reviewed B1.2 head `3f034963`; integration
with subsequently merged B1 code/current main is outside this measurement.

A uses unchanged sequential B1.2 arithmetic. B uses bounded grouped restoration
and a restricted **hybrid** GEMM: native calls for at most 16 routed rows, grouped
cuBLAS calls above that threshold. Both arms explicitly select cuBLASLt and disable
BF16 reduced-precision/split-K reductions and TF32. This is **not** a bit-exact
replacement for the old default cuBLAS dispatch, arbitrary shapes, or all inputs.

## Complete prospective series

- Three formats: BF16 (`dense`), NF4, and nested NF4 (`nf4-dq`). Layer 7, group width 4.
- Five complete ABBA blocks per format, fresh process per arm, five warmups and
  20 retained samples per timed arm: **60 arms / 1,200 samples**.
- Six separate A/B CUDA profiles. All 66 processes returned OK and F1 OK.
- Every retained output hash matches the exact reference; each process exercised
  an effective active-weight corruption negative control and restored positive.
- The unchanged ADR 0003 calculation gives N=2 for each format and zero additions
  after the five-block pilot. No timed/profile failure, replacement or early favorable stop.
- Functional smoke and preparation/instrument failures are retained separately
  in `development-evidence.zip`; they are never counted as formal timings.

## Wall time and actual launches

| Format | Mean of A arm medians, ms | Mean of B arm medians, ms | Observed paired wall reduction | Block B/A range | Profile CUDA kernel events A → B |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 172.900 | 164.774 | 4.629% | 0.914717–1.000798 | 852 → 756 |
| nf4 | 70.804 | 69.293 | 2.124% | 0.956580–1.009764 | 980 → 788 |
| nf4-dq | 77.243 | 72.682 | 5.898% | 0.925351–0.965111 | 1236 → 788 |

The reported reduction is `100 × (1 - mean(block B/A))`, where each block ratio
uses the average of its two B medians divided by the average of its two A medians.
It is not the ratio of the two descriptive columns. Every sample and block counts.

These are modest observed differences in this recipe, **not a demonstrated 10%
effect, a universal speedup or a guaranteed gain**. BF16 and NF4 each include a
block with B slower than A. Launch-count reductions do not predict wall-time
reductions by themselves. No post-hoc significance claim or throughput threshold
is used to choose extra repeats.

Wall time includes buffered reader I/O, decode, routing/gather, activation,
GEMMs, scatter and synchronization. The CUDA-event interval retained in each
sample is a dispatch span including host gaps, not pure kernel compute. Hashing
is after the timer. Profiles count actual CUPTI kernel events around one exact
forward including its exactness-check wrapper; this adds five events to each
arm versus the earlier development forward-only traces. No modeled launch counts
are substituted for profiler events.

Clocks were observed rather than locked; all clock diagnostics returned OK:
dense/A: 1792–1957 MHz; dense/B: 1792–1957 MHz; nf4/A: 1792–1957 MHz; nf4/B: 1792–1807 MHz; nf4-dq/A: 1792–1792 MHz; nf4-dq/B: 1792–1792 MHz. The raw per-arm telemetry remains available.

## Memory trade-off

| Format | A peak allocated MiB | A peak reserved MiB | A peak RSS MiB | B peak allocated MiB | B peak reserved MiB | B peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 30.043 | 44.000 | 1213.211 | 63.103 | 72.000 | 1028.473 |
| nf4 | 34.043 | 44.000 | 1261.137 | 77.103 | 86.000 | 1019.070 |
| nf4-dq | 34.043 | 44.000 | 1279.293 | 77.103 | 86.000 | 1020.129 |

Each cell is the maximum across that arm's ten timed processes. GPU peaks start
before pool/backend creation and include correctness/negative controls and warmup;
they are not a steady-state-only forward delta. RSS is the process high-water mark,
not the complete host/container RAM budget. Host-level box counters are retained
but must not be interpreted as container-only memory readings.

B uses **more allocated and reserved GPU memory**. Its logical weight arena is
bounded by 64 MiB, separately from the 4 MiB cuBLAS workspace and the conservative
256 MiB activation plan. CUDA contexts, library internals and allocator reservations
are outside the weight-arena bound. There is no total-VRAM or 8 GB model-fit claim.

## Validity and retained development history

Real F1 controls retained in the prospectively pinned setup show baseline OK,
own 1 GiB read OK and independent-process 1 GiB read VOID. The selector binds the
immutable files and exposed device identities; formal arms require suspend,
power and foreign-read checks. Exposed cgroup/block counters and visible GPU peer
checks do **not** certify physical-host isolation or visibility into other tenants.
The shards are warmed/buffered; this is not a cold physical-NVMe throughput test.

All ten development archives are preserved, including rejected Triton/direct
GEMM candidates, the failed expanded one-row hybrid, the corrected 2,304-case
restricted-shape exactness check, 110 integration tests on the CUDA host, resident/streamed
controls, and the initial unknown/ambiguous storage-discovery attempts. One early
helper did not retain per-read counters for an ambiguous association; the corrected
helper and successful immutable-file association receipts are retained. No formal
verdict uses that failed association. Before the operator commit two trailing
empty source lines were removed; archived/current executable AST equality is
recorded. Historical sources remain unchanged.

The selected local CPU suite recorded 279 passed / 99 hardware skips, with four
existing PEFT warnings. This is not a full-repository test or current-main/platform
matrix claim. Formal GPU processes ran on the committed source, not on a dirty
copy of the preparation checkout.

## Reproduction and archive verification

`raw-results.zip` retains all process logs, progress/results, the complete session,
launch receipt and six raw traces. `reproduction.zip` includes the exact pinned
source, D3 tests, preparation identities, environment/commit metadata and the
post-run replay verifier. Its verifier recomputes each analysis, repetition decision,
negative-control/hash consistency and trace kernel count. `development-evidence.zip`
retains all earlier versions and the selected test log. Inner manifests and outer
`SHA256SUMS.json` cover the published bytes; ZIPs are binary and text artifacts use LF.

On a checkout containing the measured source, extract the raw and reproduction
archives into separate folders. The offline replay requires no GPU:

```bash
python reproduction/verify.py --session raw-results/session \
  --checkout /path/to/soup-rnd --out replay.json
cmp replay.json analysis.json
```

The pinned immutable weights/hidden-state tensors are identified by full hashes
in the recipes but are not duplicated in these archives. Recreating them requires
the pinned public OLMoE checkpoint and the archived preparation code/environment.
An independent new measurement needs its own prospectively fixed protocol and
instrument checks. The runner refuses a second session under this source commit;
do not bypass that guard or relabel another host as this run.

## Remaining D3 acceptance

- Coordinate/review the interface before final B1.4 measurements and reconcile
  the final B1 dependencies/current-main integration.
- Prepare and freeze the task's separate **16 GB+ whole-model/end-to-end** CUDA
  gate. No such execution or training/backward validation was performed here.
- Obtain implementation and evidence review/sign-off. Public trainer integration,
  general numerical equivalence and production defaults are not claimed.

The layer-performance question has measured evidence. The separate full-scenario
and coordination/review requirements remain open.
