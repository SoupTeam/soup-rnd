# FlashAttention with persistent streamed layers — 2026-10-20

**No tested configuration passed the practical speed gate for adding external FlashAttention to persistent-layer streaming.**

The accuracy-qualified hybrid residency configurations match the native reference
bitwise in their primary and padded/causal captures. Pure external attention matches
the main training captures but fails the unchanged cross-backend padded-input numerical
comparison. It is excluded from timing. The hybrid keeps native SDPA for explicit
masks and uses the pinned external kernel when unmasked.

The separate 16-token semantic profiler observes native SDPA flash kernels. The
additional native kernel profile at the actual 1024-token training shape was not
obtained; this limits causal attribution, while the full-shape timing results remain valid.
The comparison measures the external build at the same persistent-layer count.

## Controlled method

- Clean upstream `d500ae0dca7c4da905f8d59c9866af2f8aaa824b`: all 602 production files unchanged.
- OpenLLaMA-7B-v2 revision `e5961def23172a2384543940e773ab676033c963`, dense BF16,
  ordinary rank/alpha-4 LoRA on all seven projections, dropout 0, nonzero initial B, seed 792.
- Same frozen 32-row synthetic chat data/order, sequence cap 1024, microbatch 1/2, accumulation 2.
- Resident8 layers 12–19 / resident16 layers 8–23; two rotating slots remain in both arms.
- Same 10 GiB allocator cap. The full 13 GiB pinned CPU base store remains; driver context is additional.
- Byte-identical original capture/comparator: reference first, `rtol=1e-2`, `atol=1e-5`,
  exact input/initial tensors, finite values, loss to three decimals. No tolerances relaxed.
- Original causal/padding assertions retained, plus the same profiled unmasked inference
  logits in every semantic capture. Semantic profiling is excluded from timing.
- Only qualified configurations timed: four fresh-process pairs, AB/BA/BA/AB, six warmup
  plus eight measured optimizer updates. Final adapters/losses checked after all 14 updates.
- Practical gate: at least 2% median paired time reduction, at least 75% positive pairs,
  and positive pair-bootstrap 95% lower bound (20,000 resamples, seed 792).
- External FA 2.7.4 wheel SHA `f7f5d839264e24505604c3d50bc2b0d6d00c001cfa596edf7601ca9e060a5582`.
  Native Torch arithmetic macros and no `--use_fast_math`; this does not qualify a stock wheel.
- No Fast-LoRA, Liger, NoRA, compile, quantization, production switch or automatic VRAM planner.

Frozen GPU source archive SHA-256 (identical on all four model-running leases):
`fc843cff5c56bfd875b5f8f3808f9d18972a3583cffdc92c152715722a85e6c7`.

## Separate hardware observations

| Lease | CPU | Provider PCIe bandwidth | Measured separate pinned H2D qualification | Timed configurations |
| --- | --- | ---: | --- | --- |
| A | AMD EPYC 9474F 48-Core Processor | 6.3 GB/s | not part of A protocol | resident8/16, batch1/2 |
| B | Core™ i9-14900KS | 22.8 GB/s | 23.98 GB/s (requires >=16); VOID environmental audit | resident16, batch1/2; timings VOID (external compute peers) |
| C | AMD EPYC 9654 96-Core Processor | 25.5 GB/s | 26.81 GB/s (requires >=16) | resident16, batch1/2; accuracy only, no timing processes started |
| D | AMD EPYC 9654 96-Core Processor | 25.5 GB/s | 26.79 GB/s (requires >=16) | resident16, batch1/2 |
| E | AMD Ryzen Threadripper PRO 7985WX 64-Cores | 25.5 GB/s | not run | additional native profile blocked before GPU work |

A exposed a slow 6.3 GB/s advertised PCIe link. B was registered as a separate
confirmation after A timing had started, selecting >=20 GB/s advertised PCIe and the
highest residency count to reduce transfer masking. A was completed, retained and
deleted before B was rented. B repeated numerical qualification before timing.
The choice was based on hardware bandwidth; all A outcomes remain in the record.
Final telemetry audit found up to three GPU compute processes in B, including
pre-existing activity before its training. **All B speed/VRAM comparisons are VOID**
for performance interpretation, regardless of their observed values. Raw observations,
numerical captures and the failed environmental audit are retained without relabelling
the run as clean. Its separate JSON/archive explicitly records this status.
C was registered only after B was deleted and this failed audit was identified.
C repeats the same source, numerical gates and resident16 batch1/2 protocol, with
three idle-GPU samples before setup, an actual H2D qualification, and an abort if
sampled compute peers exceed one during a stage. No B timing is used as a result.
C passed the complete numerical qualification, but local disk filled during capture
archive extraction before any timing process started. Automatic cleanup deleted C.
Both canonical capture archives had already downloaded; every original file was
recovered and SHA-256 verified locally. The interrupted state and recovery remain
explicitly recorded. Its affected package received no remote acknowledgement;
the recovery receipt records verification after deletion. No timing run is claimed for C.
D was registered after C deletion and complete capture recovery. Verified identical
files were coalesced into hardlinks, preserving every original path and byte. D uses
streaming extraction with full payload hashing, >=8 GiB free before allocation, and
faster lossless archive compression. These are transport changes outside measured
training; GPU source, numerical gates, warmup and timing protocol remain identical.
**No timing is pooled across leases or with the preceding residency/compiler studies.**

## Incremental time and allocator VRAM

SDPA and hybrid always have the same number of persistent layers within a pair.
Times are medians of per-process median step times; reduction is the median of four
within-pair ratios, so the columns use different aggregations. Memory is the median
of per-process maximum peaks over the eight measured updates; full process records
remain in the detailed observation archives.
Positive reduction means a shorter hybrid step; negative means a longer step.

| Lease | Layers | Batch | SDPA/hybrid step, s | Paired reduction | 95% bootstrap | Allocated GiB, SDPA/hybrid | Reserved GiB, SDPA/hybrid | Practical gate |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- |
| A | 8 | 1 | 5.867194/5.879058 | -0.784% | [-1.543%, -0.152%] | 4.8606/4.8606 | 5.1387/5.1387 | False |
| A | 8 | 2 | 6.182374/6.158324 | 0.355% | [-0.281%, 0.849%] | 5.4896/5.4896 | 6.0234/6.0234 | False |
| A | 16 | 1 | 4.096665/4.114799 | -0.433% | [-0.511%, -0.306%] | 7.8763/7.8763 | 8.1543/8.1543 | False |
| A | 16 | 2 | 4.671641/4.617346 | -0.038% | [-0.439%, 2.576%] | 8.5053/8.5053 | 9.0391/9.0391 | False |
| D | 16 | 1 | 1.338522/1.384998 | -3.464% | [-3.501%, -3.416%] | 7.8763/7.8763 | 8.1543/8.1543 | False |
| D | 16 | 2 | 1.958509/1.959084 | -0.053% | [-0.369%, 0.003%] | 8.5053/8.5053 | 9.0391/9.0391 | False |

Total: **48 fresh timing processes / 24 pairs**, eight measured updates per process.
All final adapter tensors and logged losses after 14 updates bitwise equal: **True**.
Additionally, B retains 16 timing processes / eight pairs with matching adapters/losses,
but those performance measurements are VOID because the environmental audit failed.
C contributes accuracy captures only; it started zero timing processes.
The 448 adapter tensor names/counts, input shapes, seeds, data/initial-adapter hashes,
BF16 autocast and allocator caps were checked in every pair. Four pairs per case give
limited uncertainty evidence. Timings exclude setup, preload, downloads and final saving.

## Numerical detail

| Lease | Comparison | Tensors | Original gate | Bitwise equal |
| --- | --- | ---: | --- | --- |
| A | primary/native-repeat | 1813 | True | True |
| A | primary/external-r8-b1 | 1813 | True | True |
| A | primary/external-r8-b2 | 1813 | True | True |
| A | primary/external-r16-b1 | 1813 | True | True |
| A | primary/external-r16-b2 | 1813 | True | True |
| A | primary/hybrid-r8-b1 | 1813 | True | True |
| A | primary/hybrid-r8-b2 | 1813 | True | True |
| A | primary/hybrid-r16-b1 | 1813 | True | True |
| A | primary/hybrid-r16-b2 | 1813 | True | True |
| A | semantics/external-r8 | 904 | False | False |
| A | semantics/external-r16 | 904 | False | False |
| A | semantics/hybrid-r8 | 904 | True | True |
| A | semantics/hybrid-r16 | 904 | True | True |
| B | primary/native-repeat | 1813 | True | True |
| B | primary/hybrid-r16-b1 | 1813 | True | True |
| B | primary/hybrid-r16-b2 | 1813 | True | True |
| B | semantics/hybrid-r16 | 904 | True | True |
| C | primary/native-repeat | 1813 | True | True |
| C | primary/hybrid-r16-b1 | 1813 | True | True |
| C | primary/hybrid-r16-b2 | 1813 | True | True |
| C | semantics/hybrid-r16 | 904 | True | True |
| D | primary/native-repeat | 1813 | True | True |
| D | primary/hybrid-r16-b1 | 1813 | True | True |
| D | primary/hybrid-r16-b2 | 1813 | True | True |
| D | semantics/hybrid-r16 | 904 | True | True |

The pure external padded-input comparison fails loss agreement to three decimals,
445 gradient tensors and one logits tensor on each residency count in A.
Its within-arm causal/padding invariance checks are a separate assertion and passed;
this is a failed cross-backend numerical gate, not proof that every external mask is wrong.
The hybrid captures preserve all original checks and match all 904 tensors bitwise.
Profiler events and route counters confirm actual external CUDA execution, including
the added unmasked forward; the explicitly masked route retains native SDPA.

Every hybrid timing process records 1,792 external-unmasked dispatches and zero
native-mask dispatches. The training timings therefore exercise the external kernel
rather than measuring a hybrid that always falls back to native attention.

After the D matrix, an optional kernel profile was rejected before model setup:
its caller requested one measured step, while the unchanged harness requires >=2.
The failed caller, guard exception and exclusion timestamps remain archived.
E prepared the corrected two-measured-step diagnostic after D deletion, but its
artifact-transfer preflight failed DNS lookup before GPU model work. E was deleted
and independently verified absent. No native 1024-token kernel or Chrome trace is
claimed. The corrected planned caller and both failures remain retained.
The native flash observation is restricted to the separate 16-token semantic
fixture. Actual full-shape bottleneck/kernel attribution remains unverified; the
48-process speed/VRAM matrix and its numerical gates are unaffected.

## Limits and practical interpretation

- Native SDPA executes flash kernels in the 16-token semantic fixture. Its actual
  1024-token training kernel was not profiled successfully. Do not treat the short
  diagnostic as direct proof of the timed training kernel or a measured bottleneck.
- Interpretable performance observations concern one pinned model/stack and two
  RTX 4090 leases. One additional lease is retained as VOID for performance and
  another as an accuracy-only interrupted run. These observations do
  not establish downstream task quality or a universal result for other shapes/GPUs.
- All configurations hold microbatch fixed. No comparison with spending cache memory
  on a larger native microbatch, physical 8 GB-card qualification, or NF4 coverage is claimed.
- The earlier residency speedup remains a separate experiment; these percentages are
  incremental attention effects and are not added to its percentages.
- Research accounting/route counters are present. A production implementation needs
  counter-free validation, configuration/planning, lifecycle and broader architecture tests.
- GPU samples cover complete processes, not just measured step windows. Affinity and
  sampled compute peers are audited; a one-GPU offer does not prove whole-host exclusivity.
- Allocator VRAM excludes driver context and CPU RAM. Full raw observations are retained.
- This does not establish a Fast-LoRA speedup or D2/T00/F1 completion.

## Budget, retention and validation

Instances `56920445`, `56932192`, `56936157`, `56939591` and `56942924` were deleted and independently verified absent.
Cumulative conservative cost, including the preceding residency study: **USD 4.276317 / USD 5**.
Visible prior charges USD 3.283000; final lease observed USD 0.025000.
Provider settlement can lag. No top-up or unrelated-instance deletion was performed.

Every encrypted export was authenticated and SHA-256 checked, with every original-file
manifest verified before acknowledgement. Canonical archives, tensors, receipts and
failed-gate evidence remain under the five local `output/residency-fa*-vast-20261020/`
study directories. Only verified duplicate transport chunks were removed; identical
original files may use verified hardlinks, preserving every path and byte. The small
committed frozen archives preserve executable sources, exact inputs and build provenance;
they do not embed the larger raw training tensor archives.

Scoped local checks: 11 residency, 5 constructor and 5 routing tests passed; two CUDA
cases skipped locally. Each of the four model-running leases ran 18 passing tests,
including two actual CUDA asynchronous recycling tests. Ruff, pre-commit and diff checks are reported
separately at publication; no complete CI matrix is claimed.

[Lease A machine observations](streaming-residency-flashattention-results-20261020.json),
[lease B VOID machine observations](streaming-residency-flashattention-fast-pcie-results-20261020.json),
[lease C machine observations](streaming-residency-flashattention-clean-pcie-results-20261020.json).

[lease D machine observations](streaming-residency-flashattention-final-pcie-results-20261020.json).

[E blocked native-profile attempt](streaming-residency-native-kernel-profile-results-20261020.json).

[A frozen harness/inputs](evidence/streaming-residency-flashattention-20261020/README.md),
[B VOID frozen harness/inputs](evidence/streaming-residency-flashattention-fast-pcie-20261020/README.md),
[C interrupted frozen harness/inputs](evidence/streaming-residency-flashattention-clean-pcie-20261020/README.md),
[D frozen harness/inputs](evidence/streaming-residency-flashattention-final-pcie-20261020/README.md),
[E corrected planned caller and blocker](evidence/streaming-residency-native-kernel-profile-20261020/README.md).
