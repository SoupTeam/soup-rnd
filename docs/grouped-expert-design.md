# D3: bounded grouped expert compute — layer evidence and whole-model preparation

Status: an internal bounded grouped NF4 restore and restricted hybrid GEMM
candidate is implemented and development-tested. The [prospective layer series](../benchmarks/results/d3-grouped-layer-20261021/report.md)
is complete: 60 valid arms / 1,200 samples and six CUDA profiles. **D3 end-to-end
acceptance and coordination/review remain open.** Public flags and training defaults do not change.
The [Trello card](https://trello.com/c/G6t5xoIj) is the task; source context is
`rndTASK.pdf`, pages 6/10, with the team's decomposed acceptance checklist.

## Dependencies and existing work

The working base is B1.2 `3f03496397adf41479a4a8e8bd39c70ca052e6f7`, including
B1.1. PRs [24](https://github.com/SoupTeam/soup-rnd/pull/24) and
[25](https://github.com/SoupTeam/soup-rnd/pull/25) awaited merge when initial
preparation was written; both are now merged. The 2026-10-21 dependency audit
verified that all five reused B1 runtime files match current `main` at `fdf067fc`
byte-for-byte. The isolated layer series still pins its original reviewed source;
no full current-main/platform validation follows from that file comparison.
A1/A2/A3 are not prerequisites for the isolated D3 forward kernel.
Coordinate the interface with B1 before final B1.4 measurements. B1.3 remains the
dependency for expert-streamed backward; B1.2 cannot be used as a complete SFT path.

Reuse these audited components without changing their reference arithmetic:

| Existing component | Use |
| --- | --- |
| `expert_shard.py` | Independent expert reads, layout, NF4/nested metadata |
| `expert_stream.py` | B1.2 eager baseline, slot accounting, ownership/error semantics |
| `b1_expert_forward_gate.py` | Native OLMoE expert construction, independent NF4 reference and checkpoint/hash verification |
| B1.2 protocol | Pinned OLMoE revision and full checkpoint hashes; reuse existing downloaded files |
| F4 report and replay | Shapes, historical launch-cost hypothesis and disclosed limitations |

The original Soup `upstream/main` inspected at
`12a19f55d038c90295f8284ef71d0da5e2be2762` has a sequential expert path, including
backward. It is useful prior implementation, but no grouped NF4 expert kernel was
found there. RND's native Transformers `grouped_mm` B2 tests do not establish a
bounded grouped NF4 implementation.

F4 [PR 11](https://github.com/SoupTeam/soup-rnd/pull/11) is merged and publishes descriptive
curves, not a formal baseline validity verdict. Its historical A/B/C/C2/D arms
change routing/weights/workloads, and the old loop discards router weights.
Do not copy their causal subtraction, best-of samples or modeled dense reference
into D3. Run new matched A/B measurements after freezing their rules.

## Proposed implementation boundary

1. Keep the native router and shared expert unchanged. Construct one routing map
   from actual `(token, top-k position, expert, weight)` assignments and use the
   identical map and hidden states in both arms. Cover skew, unused experts,
   all-expert routing, one token, empty input and partial groups/tiles.
2. Select groups of 1/2/4 active experts in ascending ID order. Use a bounded
   multi-slot arena derived from B1.2 accounting. The preparation planner counts
   full replicated slot bytes conservatively, including repeated code tables;
   it is a design upper bound. The new internal pool implements full slots plus
   padded pointer tables separately, with a fixed group width. Plan activations,
   gathered rows, scatter outputs and backend workspaces separately.
3. Hold all slot leases until every consumer finishes. Keep exact input/output
   shapes for each expert; padding tokens must never reach the reduction. Budget
   refusal happens before allocation/read. Handle error, concurrent use, close,
   nested NF4 statistics and stale/missing shard files explicitly.
4. Implement grouped dequant/GEMM using one declared primitive for resident and
   streamed comparisons. Never materialize the full dense MoE stack. First
   compare the grouped primitive at width 1 to the unchanged native eager loop;
   then compare widths 2/4. A failed comparison blocks timing and integration.
5. Preserve activation rounding, route weighting and expert scatter accumulation
   order. A change to GEMM reduction order, fused SiLU/product or atomic scatter
   can break exactness. No `allclose`, relaxed tolerance or value-dependent
   compare-and-recompute can turn a mismatch into a D3 PASS. The later hybrid
   candidate declares native small-row dispatch explicitly, before its broader
   validation/performance protocol; those operations are never counted as fused.

The [Triton grouped GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
is a scheduling reference, not an acceptance-ready kernel. Its example checks
approximate equality and is not an NF4 expert implementation. No tutorial code
was copied here. Review licensing before copying any third-party implementation.

## Correctness before performance

First run tiny synthetic controls and the future candidate's CPU scheduling /
budget / failure tests. Then run CUDA BF16, NF4 and nested NF4 comparisons,
including irregular sizes and routing. Reuse B1.2's independently reconstructed
weights for a pinned trained OLMoE gate at layers 0/7/15. Synthetic activations
remain labeled synthetic even when weights are trained. Check nonzero finite
outputs, exact equality, stable A/A controls, and an altered-active-weight
negative control in the same process. Empty input is a separate zero-load test.

Forward equality does not establish gradients, LoRA updates or whole-model
training. If the final D3 path includes backward, add its own exact gradient and
negative controls and coordinate B1.3. Do not claim full training using a
forward-only primitive or infer its speed from an isolated layer.

## Prospective timing — currently disabled

The [design draft](../benchmarks/protocols/d3-grouped-experts/protocol-draft.json)
is not executable. The original preparation tool has no formal-run mode.
The separate performance runner/protocol described below is ready for an
operator commit, but refuses a dirty/uncommitted checkout. Before any formal arm,
test the candidate and runner, choose
the actual recipe, freeze device/software/source hashes, thresholds, validity
rows and budgets, and commit that executable rule. The draft is not a frozen
protocol or completed preregistration.

The current layer protocol proposes width 4, layer 7 and three formats, five
fresh-process ABBA pilot blocks each and the unchanged ADR 0003 deterministic
additions, with a ten-block cap and zero replacements. Each arm has 5 warmup
+20 measured invocations. Keep every sample, failed arm and traceback. A/B include identical weighted
routing, load/dequant, gather, scatter and synchronization. Record wall time and
CUDA event time separately. Measure fresh total allocated/reserved peaks with
resident references released; record staging/RSS and logical reads too.

Use a separate profiler invocation. Count actual CUDA kernel events exported by
CUPTI and retain the trace. CPU operators or missing device traces cannot prove
fewer GPU launches. Record clocks, temperatures, power and visible peers per arm
with F1 controls, explicit missing-telemetry handling and the same physical GPU.
No cold-disk, giant-model, generalized speedup or optimizer-storage claim follows.

## Hardware and runnable preparation

Two SSDs are **not required**. One local filesystem for shards/checkpoint is
enough; disk striping is outside D3. Local CPU is enough for this preparation.
Linux sm_80+ with native BF16 and Triton is needed to compile/test the candidate.
An RTX 3060 12 GB is suitable for small kernel and trained-layer checks; the card
requires at least 16 GB for end-to-end measurements (e.g. RTX 3090 24 GB).
Do not rent a second machine until the candidate and CPU controls are ready.

Run in an environment already configured for B1.2, from the checkout root with
`src` on `PYTHONPATH`. Every output path must be new; failures are retained:

```bash
PYTHONPATH=src python benchmarks/harness/d3_expert_prepare.py plan --out /tmp/d3-plan-01
PYTHONPATH=src python benchmarks/harness/d3_expert_prepare.py cpu-control --out /tmp/d3-cpu-01
PYTHONPATH=src python -m pytest tests/test_d3_expert_prepare.py --no-cov -v
```

Only when a Linux CUDA window is ready, the following tests the unchanged B1.2
synthetic baseline, native BF16, NF4/nested codec and profiler visibility. It
neither compiles a grouped candidate nor performs formal timing:

```bash
PYTHONPATH=src python benchmarks/harness/d3_expert_prepare.py cuda-preflight --out /tmp/d3-cuda-01
```

`instrument_controls_passed` means the preparation controls passed.
`d3_accepted` remains false. No model downloads, new rentals, commits or pushes are automatic.

## Development checks on RTX 3060, 2026-10-20

These are uncommitted development diagnostics, not formal protocol arms or task
acceptance. The existing Vast instance was reached over SSH; no new server was
rented. Remote runtime is the retained B1.2 `99664b09` checkout cloned to a new
development directory. Its five relevant runtime Git-blob hashes match the
local B1.2 base. New preparation/probe sources are identified by their actual
working-file SHA-256 values, not described as a new Git commit.

| Diagnostic | Observed result | Consequence |
| --- | --- | --- |
| Synthetic B1.2 readiness | All 12 format/routing controls pass; native BF16 and CUDA kernel tracing available | RTX 3060 is usable for development |
| Original masked Triton BF16 GEMM | 6/32 checks differ from native eager, max absolute difference 0.03125; group width 1/4 match each other | Cannot use as an exact replacement |
| Reduced-precision flag diagnostic | Disabling reduced-precision BF16 reduction does not remove those six mismatches; no-split-K attempt errors on the current backend | Keep both attempts, no acceptance claim |
| PyTorch public grouped GEMM | Corrected probe: 24/24 aligned-shape checks exact; the 33-by-47 shape is explicitly unsupported | Exactness on supported shapes only |
| PyTorch grouped wrapper profiling | Four-GEMM recipe: loop has 5 kernel events, wrapper has 12, including metadata/packing; cuBLASLt preference did not change this observation | This wrapper does not demonstrate fewer device launches |
| Grouped NF4 decode | 64/64 decode comparisons exact, 64/64 changed-packed-byte negative controls differ | Useful isolated component, not complete D3 |
| Decode-only profiling for four weights | NF4: bitsandbytes loop 4 events, grouped decode 1; nested NF4: loop 12, grouped decode 1 | Launch reduction for this decode-only invocation, not measured whole-model speedup |

The grouped NF4 prototype validates metadata, shapes, dtypes, code tables,
nibble/statistics bounds and a logical tensor-storage budget before allocating
outputs. It retains input ownership through the launch closure. It has no disk
reader integration, pool lease integration, routed-expert execution, backward
or total-VRAM certification yet. Its metadata preparation is outside the
decode-only profiler invocation. No throughput ratio follows from event counts.

An early public-backend probe also had a dispatch-name shadowing defect and
stopped at a stride error. No arithmetic result is taken from that attempt.
The corrected dispatch and precision restoration have CPU regressions. That
failed source/report is preserved alongside the subsequent diagnostics.

Five evidence ZIPs, including source versions, raw traces, logs and failed
attempts, have been copied back to `C:/ml/soup/work/d3-development-evidence-01.zip`
through `-05.zip`. A local integrity receipt accompanies them. These archives
are development evidence and have not been published as a formal benchmark.
Earlier preparation checks: 102 passed, 5 CUDA skips; full repository Ruff clean.
The remote preparation/control subset passed 37 tests. Later integration checks
are recorded below; none are a full repository pytest or completed D3 verdict.

## Reader and arena integration

`src/soup_cli/utils/expert_group.py` adds an internal `GroupedExpertPool` and
forward-only `grouped_experts_class()`. Widths 1 through 4 use one byte arena and
one blocking host-staging buffer. The arena counts full B1.2 slots (including
their scratch/code/padding) and, for NF4, ten padded 512-byte pointer tables.
This deliberately conservative accounting is independent of the total number
of experts. Routing/activation tensors, GEMM workspace, CPU reference-decoder
temporaries and allocator reservations are separate.

`expert_group_decode.py` restores each matrix part across the current group in
one Triton launch into the arena views. The caller retains its lease through
the eager computations and device synchronization. Reads remain one per active
expert in ascending order. This groups weight restoration only: GEMMs and
expert accumulation use the unchanged B1.2 operations. Dense BF16 needs no decoder.

Budget refusal precedes arena allocation and expert reads. Concurrent/reentrant
leases and close-during-lease fail explicitly. Read/decode errors close the
reader without yielding a partly restored group. If device synchronization
fails, the pool refuses further leases because safe reuse is unknown. Explicit
CUDA-device selection also scopes the Triton launch to the pool's device.

Local selected B1/B2/D3 tests: **227 passed, 62 skipped**, with four existing PEFT
pattern warnings; this includes the trained-check runner's four CPU controls.
On the existing RTX 3060, the final integration suite passes **65/65 with no skips**,
including 27 CUDA combinations of BF16/NF4/nested NF4, widths 1/2/4 and empty,
one-token/partial or all-expert routing. It checks exact native and unchanged
B1.2 outputs, repeatability, effective same-process negative controls and loads.
These use synthetic small weights and do not replace a trained-layer gate.

The final-source trained-layer development checks also pass: all nine
layer/format cells and all **27 width comparisons** have exact output hashes,
repeat equality, effective changed-weight negative controls and correct loads.
The 64 MiB weight-arena budget is checked by the allocator as well as the plan.
No full-model capacity or speed claim follows. An intermediate 64-test version
and its trained checks remain in evidence archive `-06.zip`; the final version
and added busy-pool trace regression are in `-07.zip`, alongside source snapshots.
Both are copied to `C:/ml/soup/work/` and covered by the local integrity receipt.

The additional development runner `benchmarks/harness/d3_grouped_restore_check.py`
reuses B1.2's full checkpoint hashes, resident reconstruction and trained router
at layers 0/7/15. It uses 17 synthetic tokens and all three formats/widths,
releases the resident expert stack before testing the bounded reader path and
retains progress on failure. It is **not** B1.2's original 512-token frozen gate
or a frozen D3 benchmark; formal timing stays disabled. Run only with the
already downloaded, hash-matching local checkpoint:

```bash
PYTHONPATH=src python benchmarks/harness/d3_grouped_restore_check.py \
  --model-dir /path/to/pinned-olmoe --out /tmp/d3-trained-development-01
```

## Restricted hybrid GEMM and current validation

`expert_group_gemm.py` uses the documented
[cuBLAS grouped API](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex)
with device pointer arrays into the leased slots, avoiding a packed dense expert
stack. `expert_group_compute.py` groups both matrix operations and preserves
the native activation, route weighting and ascending scatter order.

The supported `(K,N)` shapes are `(2048,2048)` and `(1024,2048)` for the pinned
OLMoE recipe, and `(32,32)` / `(16,32)` for tiny controls. Other geometries are
refused. Zero rows do no compute; 1–16 rows use explicit native `torch.mm(out=)`;
larger row counts use grouped cuBLAS. The decision depends only on the declared
geometry/row count, never on comparing values and silently retrying a mismatch.

The caller must select cuBLASLt with BF16 reduced-precision/split-K both false
for **both** arms; the runtime checks this policy without changing globals.
See [the pinned PyTorch BF16 reduction documentation](https://github.com/pytorch/pytorch/blob/v2.14.1/docs/source/notes/cuda.md).
This is not a claim of bit-exact substitution under the old default cuBLAS
dispatch. Earlier pure-grouped/default-policy mismatches remain failures.

| Development check | Result |
| --- | --- |
| Direct group-all cuBLAS, default policy | 6/32 native comparisons fail; 5-to-1 launch reduction alone does not pass exactness |
| One-row native hybrid under cuBLASLt | Initial 32 pass, broader check fails 154/800; rejected |
| Final OLMoE-shaped native-through-16 hybrid | 2304/2304 expanded native comparisons exact, covering eight seeds, rows 0–139 and 255/256/257/512 |
| Integrated restore/compute + previous restore controls | 110/110 tests pass on RTX 3060, no skips |
| Trained layers 0/7/15, 512 synthetic tokens, all formats and widths 1/2/4 | 27/27 streamed comparisons exact, real grouped operations confirmed |
| Resident oracle using the same new compute class/backend and identical stored weights | All 27 controls exact |

Separate full expert-forward CUPTI traces for layer 7 / width 4 contain:

| Format | A: native loop kernel events | B: hybrid grouped kernel events |
| --- | ---: | ---: |
| BF16 | 847 | 751 |
| NF4 | 975 | 783 |
| Nested NF4 | 1231 | 783 |

These count actual CUDA kernel events, including native small groups and the
surrounding routing/gather/activation/scatter operations. They do not imply a
throughput ratio. Pointer preparation, I/O, synchronization and allocations
remain part of the prospective wall-time comparison.

The weight arena remains separately limited to 64 MiB. cuBLAS has a separately
owned 4 MiB workspace; each call bounds its outputs/pointer allocation bins.
Forward planning conservatively bounds masks, route indices, output and
intermediates within 256 MiB, before expert reads. These logical tensor limits
exclude CUDA/cuBLAS contexts and allocator reservations; total peaks are recorded
independently. Neither the reader nor compute path enables autograd or a trainer.

Archives `d3-development-evidence-08.zip` and `-09.zip` retain rejected versions,
raw traces, exact source versions and the later controls in `C:/ml/soup/work/`.
The local integrity receipt covers their CRCs and inner hashes.

## Prospective layer performance: commit required

[performance-protocol.json](../benchmarks/protocols/d3-grouped-experts/performance-protocol.json)
pins the current GPU UUID, software, cuBLAS binary, immutable inputs and
instrument receipts. Its configuration is enabled for the next source commit;
**no formal timed arm had run at that commit**. The resulting [2026-10-21 series](../benchmarks/results/d3-grouped-layer-20261021/report.md)
completed from clean commit `0aef6904`, with all arms and memory peaks retained.
The runner refuses incomplete pins or any
dirty/uncommitted source, verifies committed bytes and permits one session per
source commit. Only the operator performs the commit/push.

[setup-evidence.zip](../benchmarks/protocols/d3-grouped-experts/setup-evidence.zip)
retains the preparation sources, identities and instrument evidence before
timing. Weight/input tensors remain on the instance, with full hashes in the
recipes. There are no model downloads in the performance runner.

Default F1 discovery initially returned `unknown` for overlay storage. It is
preserved. Controlled direct reads then associated each named immutable shard
with an exposed cgroup/block device; metadata is warmed before two exact 1 MiB
association reads. A fresh-data association attempt was ambiguous and is retained;
its first helper did not retain per-read counters, so no verdict uses that call.
The corrected helper retains ambiguity details and refuses non-exact deltas.

Real instrument controls produced baseline OK, own 1 GiB OK, independent-process
1 GiB VOID. These establish exposed-counter behavior, not physical-host isolation.
The committed selector binds exact file/device identities. Formal arms still
require all F1 checks OK, no visible GPU peers before setup and the same visible
workload set afterwards. Unknown/void stops the session; it is never relabeled OK.

Six functional A/B smoke arms cover all formats, with same-process active-weight
corruption, restored positive controls and real F1. Smoke records **no timing
samples**. CPU controls test commit/data gates, complete ABBA replay, output hashes,
the full five-block pilot, budget exhaustion and zero replacements.

After the source/protocol commit, from its clean Linux checkout on the same instance:

```bash
PYTHONPATH=src /workspace/b12-env/bin/python benchmarks/harness/d3_grouped_performance.py run \
  --prepared /workspace/d3-performance-inputs-20261020-01/prepared \
  --library /workspace/b12-env/lib/python3.12/site-packages/nvidia/cu13/lib/libcublas.so.13 \
  --out /workspace/d3-layer-performance-NEW-COMMIT
```

Do not run this command from the development checkout or invent a frozen commit.
The existing 12 GB GPU is enough for this layer scope. The separate end-to-end
gate still needs the task's 16 GB+ setup and its own prospectively fixed recipe.

## Remaining task checklist

- Preparation: reused references, synthetic controls, conservative slot plan and draft rule.
- Implemented and development-tested: grouped NF4 restoration and bounded multi-slot pool.
- Implemented and development-tested: restricted hybrid grouped GEMM, same-policy native
  exactness, same-kernel resident/streamed controls and actual full-forward launch traces.
- Completed: committed paired layer protocol, immutable setup, real F1 controls,
  functional smoke, attempt guard and deterministic replay.
- Completed for the pinned layer scope: 15 ABBA blocks, 60 valid timed arms / 1,200
  samples, six actual CUDA profiles and local byte-identical analysis replay.
  All raw/development evidence and GPU/RAM peaks are [published together](../benchmarks/results/d3-grouped-layer-20261021/report.md).
  Observed wall reductions are 4.63% / 2.12% / 5.90%; grouped GPU memory is higher.
- Open: design coordination before B1.4, 16 GB+ end-to-end gate and review/sign-off.
- Backward/derivative validation remains separate; no training claim follows from forward.

## Whole-model follow-up preparation, 2026-10-21

The [new preparation and disabled protocol](../benchmarks/protocols/d3-full-model/README.md)
assemble every native OLMoE block and share one streaming weight pool across
layers. There is a separate independent full resident oracle. Selected CPU tests
exercise whole-model logits, NF4/nested references, pool ownership, negative
controls, source/oracle identity, and one-attempt/zero-replacement execution.
The subsequent RTX 3090 preparation passed the three tiny CUDA smoke cases and
48 grouped-compute tests, including the new large-shape dispatch regression.
The original cutoff-16 large-shape probe failed on this host and is preserved;
the prospective full-model recipe uses an explicit OLMoE cutoff of 128, without
altering the archived RTX3060 study or claiming its speed for the new policy.
The first trained series at `c4900bc9` passed five cells, then stopped when
NF4 sequential's one-byte negative control was ineffective. The failed session
is [retained](../benchmarks/results/d3-full-forward-c4900bc9/report.md).
The revised complete-matrix zero control passed CPU and tiny CUDA development
checks; its twelve-cell series awaits a new prospective source/protocol commit.
No public training path or backward is enabled. The RTX3090 has already executed
both full resident formats attempted so far; a second SSD is unnecessary.
