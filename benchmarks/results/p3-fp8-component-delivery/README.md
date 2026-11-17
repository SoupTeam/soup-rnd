# P3 FP8 component delivery

This report covers two components: opt-in FP8 boundary activation storage in the
production trainer, and phase-0 MoE activation-memory budgeting. The component
checks pass. **DeepSeek FP8 numerical acceptance still fails.** This delivery does
not close phase 0, enable public DeepSeek L2L launch, or validate the full R1 model.
No Stage2 prototype or result is part of this report.

## Sources and retained runs

The definitive component run is version 4 of the private
[Kaggle kernel](https://www.kaggle.com/code/batyrkhan2131123/soup-p3-fp8-component-delivery),
identified locally as `v4-stacked`. The checked-in receipts are the reviewable
record; [submission metadata and the runner](reproduction-v4-stacked.zip) retain
the submitted configuration and source overlay.

- [Run manifest](v4-stacked/evidence/run.json): base revision
  `f0f040cd1db4c61e904ede24dde4ca492cedc9ba`, exact command arguments, exit codes,
  overlay SHA-256 hashes and mounted Input metadata. The measured sources are the
  pinned checkout plus that overlay, not an unqualified claim about every later edit.
- [Environment](v4-stacked/evidence/environment.json): Python 3.12.3,
  PyTorch 2.11.0+cu130, CUDA 13.0, Tesla T4, Transformers 5.19.0, PEFT 0.21.2,
  Accelerate 1.15.0, bitsandbytes 0.50.2, TRL 0.29.1 and Ruff 0.17.4.
  The runner exposes CUDA device 0 to GPU checks and hides CUDA for the CPU group.
- [Production smoke JSON](v4-stacked/evidence/production-smoke.json) and
  [log](v4-stacked/evidence/production-smoke.log),
  [storage accounting JSON](v4-stacked/evidence/storage-accounting.json), and
  [complete launcher log](v4-stacked/soup-p3-fp8-component-delivery.log).
- [Historical evidence integrity receipt](historical-evidence-integrity.json):
  108 bounded-investigation artifacts and 24 earlier FP32 artifacts checked,
  with no recorded mismatches. This is an archived integrity check, not a new run.

Earlier attempts remain intact. Their failures are not relabelled as successes:

| Run | Recorded outcome |
|---|---|
| [v1-red](v1-red/evidence/run.json) | `FAIL`, 11 routing-contract regressions failed, 37 deselected. This red run used Python 3.13, outside the declared supported range. |
| [v2-python312](v2-python312/evidence/run.json) | `FAIL`, Ruff failed; focused CPU had 555 passed and 4 failed; CUDA, production and storage passed; the G4 shell check exited 2. |
| [v3-final](v3-final/evidence/run.json) | `FAIL`, Ruff, focused CPU/CUDA, production and storage passed; the G4 shell check still exited 2 with `set: pipefail: invalid option name`. |
| [v4-stacked](v4-stacked/evidence/run.json) | `PASS`, every recorded command exited 0 on the pinned stacked revision and overlay. |

Each earlier directory has its launcher log and evidence; the corresponding
`reproduction-v1-red.zip`, `reproduction-v2-python312.zip` and
`reproduction-v3-final.zip` retain the submitted runners. The v2 CPU failures were
the two public reserved-store cases and two resume cases, not numerical FP8 passes.

## Implemented and exercised behavior

`training.l2l_activation_codec: fp8_e4m3` selects lossy E4M3 boundary storage with
one FP32 scale per token. `none` remains the uncompressed default. The shared
`activation_codec` preserves the G4 operation order; this change does not introduce
FP8 GEMM or quantize model weights. Supported trainer inputs are FP32/BF16.
The low-level storage estimator's FP16 support does not enable FP16 trainer use.

The configuration requires streamed Transformers text SFT with L2L, without RAFT
or multipack. Public launch still admits only Llama/Mistral and at most one visible
GPU. The existing FP16, padding, mixed-shape dropout and NEFTune restrictions still
apply. See the [feature guide](../../../docs/performance-and-quantization.md)
for configuration and the complete launch limits.

### Production builder and HF loop

The production smoke called `StreamingSetupMixin._build_l2l_trainer` and ran the
HF Trainer loop on a tiny three-layer dense Llama on **CPU/FP32**, separately for
`none` and `fp8_e4m3`. It did not run a DeepSeek public builder, CUDA BF16 builder,
NF4 model or full-model training. Its one-MoE-layer routing allowance is a
conservative synthetic reservation for dense Llama, not observed Llama routing.

For each codec, a predeclared activation/routing quota resolved requested k=4 to
`effective_k=2` before trainer and scheduler construction. This is a caller-contract
check, not an automatic global-k policy for the 8,388,608-byte C3 host envelope.
The smoke refused HF accumulation k=4 against the resolved k=2 without silently
rescheduling or trimming batches. It then observed:

- Three optimizer/L2L steps and a scheduler constructed for three steps, with
  finite nonzero gradients and changed weights for every adapter at every step.
- Variable batch and sequence shapes across the three windows:
  `[(2,4), (2,5)]`, `[(2,8), (2,9)]`, then the short final `[(1,3)]` window.
  Masked-label token counts were 9, 23 and 2.
- Per-position hidden-element capacities grew from `[256,320]` to `[512,576]`.
  Store identities were `0,1,1`: one replacement for growth, then reuse for the
  short window, without shrinking the k=2 capacity.
- Maximum-shape activation/routing quotas were 14,688 bytes for `none` and 26,065
  for FP8. The larger FP8 quota on this tiny CPU case includes CPU codec scratch;
  it is not evidence that FP8 always saves host memory. The grown FP8 store used
  25,849 bytes including the 864-byte route allowance, below its quota.
- Activation/routing usage returned to zero and stores closed. The C3 weight
  source had zero active leases and removed its prefetch hook. Its remaining
  18,784-byte or 30,161-byte fixed reservation was recorded separately, not called
  an activation leak or zero total process memory.

Three-window timings are retained in the JSON, but are not a controlled speed
comparison. This smoke does not establish DeepSeek FP8 loss/gradient parity.

### Budgets, refusals and cleanup

`plan_moe_l2l` selects the largest fitting integer `effective_k` up to requested k,
using a fixed maximum `(batch, sequence, hidden)` envelope and the remaining host
budget. It refuses a budget that cannot fit k=1. Both the store and planner use
`fp8_activation_allocation`, including its nonlinear scale-slab rounding and shared
scratch. A trainer consumes a resolved plan; it does not select k after setup.

Host charges include persistent FP8 payloads, FP32 scales, independently
power-of-two-rounded pinned payload allocations, one rounded scale slab, and
bounded host codec scratch. Codec operations use at most 256 rows per tile. The
scratch bound includes a decoded output while `get` constructs it; returned
outputs and source activations belong to the caller. CPU scratch is host memory.
CUDA codec scratch is a separate device estimate, never another host charge.

The routing contract charges pageable int64 IDs for all retained micro-batch/MoE
layer records plus one recompute-comparison record:

```text
comparison_bytes = max_batch * max_length * experts_per_token * 8
routing_record_bytes = effective_k * n_moe_layers * comparison_bytes
host_bytes = activation_bytes + routing_record_bytes + comparison_bytes
```

The [focused CPU log](v4-stacked/evidence/focused-cpu.log) records exercised
refusals for unknown/insufficient RAM, malformed or oversized windows, changed
shape/dtype/pinning/codec/device, incompatible routing topology, absent host/C3
routing budgets, replaced plans with undersized leases and oversized live route
records. It also records native resolved-routing full and short windows, C3
activation/weight partition separation, failed allocation and training-exit cleanup.
Legacy unbudgeted uncompressed B1.3 execution is not a host-budget guarantee.

The [CUDA log](v4-stacked/evidence/focused-cuda.log) records pinned rounding,
cross-stream close, slot reuse waiting for a prior reader, scratch/event lifetime,
retained-exception cleanup and routing-refusal cleanup. Drain-failure tests in the
CPU group inject failures; they are not GPU hardware-failure measurements. The
required behavior is to retain storage and its activation/routing leases if a drain
fails, allow close to retry, and block weight-runtime cleanup until users drain.
Routing changes still abort the optimizer window. No tolerance or guard was relaxed.

`saved_expert_input_bytes`, when `expert_input_dtype` is supplied, estimates the
gathered expert inputs for one recomputed layer and micro-batch. It has no layer or
k multiplier and is reported separately from host boundary/routing charges. The
standalone production smoke and full-width storage probes leave that optional
field null. Focused planner tests cover the independent dtype calculation, not a
whole-model expert-workspace measurement. CPU callers must include those inputs
among other host consumers; CUDA callers must admit their device workspace.

### Full-width storage measurements

Eight isolated subprocesses exercised production `L2LState` / `_store_for` and
put/get storage at 61 layers, hidden size 7168, shape `(1,64,7168)`, BF16 input,
pinned host storage, k=1/4/16/64 and a 5-GiB host-reservation ceiling. Every case
kept requested k, matched the planned reservation, kept it unchanged through the
round trip and returned budget usage to zero after close. There were 61*k puts
and gets per case, including 3904 each at k=64.

All numbers below are bytes from the **v4 component run**, not substituted RSS
measurements from the earlier bounded investigation:

| k | BF16 charged host | FP8 charged host | Route lease included in each | BF16 sampled RSS peak | FP8 sampled RSS peak | CUDA allocated peak, either codec | CUDA reserved peak, either codec |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64,204,800 | 32,239,681 | 241,664 | 1,078,407,168 | 1,049,120,768 | 13,765,120 | 25,165,824 |
| 4 | 256,806,912 | 128,946,241 | 954,368 | 1,272,012,800 | 1,146,310,656 | 13,766,656 | 25,165,824 |
| 16 | 1,027,215,360 | 515,772,481 | 3,805,184 | 2,043,002,880 | 1,533,161,472 | 13,772,800 | 25,165,824 |
| 64 | 4,108,849,152 | 2,063,077,441 | 15,208,448 | 5,126,172,672 | 3,081,224,192 | 13,797,376 | 25,165,824 |

At k=64 the charged host reduction was **49.7894%**, exceeding the unchanged 25%
storage-usefulness criterion. FP8's activation charge of 2,047,868,993 comprises
1,790,967,808 payload bytes, 999,424 scale bytes, 255,901,696 pinned-rounding bytes
and 65 host scratch bytes. The route lease adds 15,204,352 retained-record bytes
plus 4096 comparison bytes. Owned persistent host tensor storage was
1,791,967,232 bytes for FP8 versus 3,581,935,616 for BF16. These are different
quantities from the conservative reservation and process RSS.

The FP8 CUDA transient estimate was 15,599,617 bytes, independent of k in this
shape, and owned persistent device storage was zero. That conservative scratch
estimate is not the measured CUDA allocated peak, the CUDA allocator's reserved
peak, or a VRAM saving. RSS includes process baseline and probe scratch; 10-ms
sampling gives a lower bound and can miss short peaks. Allocated/reserved CUDA
peaks include the storage probe and G4 comparison, not model training.

BF16 round trips were bitwise equal. FP8 E1 RMS row error at k=64 was
0.0265486982, worst row 0.0303247095. A passing storage round trip does not imply
passing loss, gradient, update or routing criteria.

The mounted R1 Input had all 163 indexed shards. Its config declares
`torch_dtype: bfloat16` and FP8 E4M3 128-by-128-block weight storage.
Those metadata declarations do not establish runtime arithmetic or identify a
P3 NF4 artifact. The probes did not load its weights or train a 671B step.
They leased routing bytes without filling physical routing tensors.
Model/optimizer memory, expert arenas/graphs, caller-retained tensors, framework
workspaces and allocator caches are outside the owned-tensor reservation. No
8-GB whole-model fit, 32-GiB cgroup/no-swap fit or physical-disk result follows.

## Recorded verification commands

These commands ran on Kaggle in v4, not during preparation of this report.
The exact arguments and return codes are in `run.json`. With
`PY=/kaggle/working/py312/bin/python`, `PYTHONPATH=/kaggle/working/soup/src`, and
`CUDA_VISIBLE_DEVICES=0`, the focused file list was:

```bash
focused=(
  tests/test_fp8_activations.py tests/test_fp8_l2l_trainer.py
  tests/test_moe_l2l_budget.py tests/test_c3_public_l2l.py
  tests/test_c3_l2l_shared_runtime.py tests/test_b13_l2l_moe.py
  tests/test_b13_l2l_moe_cuda.py tests/test_b13_moe_trainer.py
  tests/test_l2l_trainer_mode.py tests/test_l2l_trainer_window.py
  tests/test_l2l_trainer_labels.py tests/test_l2l_spill_trainer.py
  tests/test_l2l_activation_failures.py tests/test_l2l_step.py
  tests/test_issue621_recipe_snapshot.py tests/test_a4_g4_compose.py
  tests/test_a4_activation_budget.py
)
CUDA_VISIBLE_DEVICES='' "$PY" -m pytest "${focused[@]}" --no-cov -v --tb=short -m 'not gpu and not smoke'
"$PY" -m pytest "${focused[@]}" --no-cov -v --tb=short -m 'gpu and not smoke'
"$PY" -m ruff check src/soup_cli/ scripts/ tests/ benchmarks/
"$PY" benchmarks/harness/p3_fp8_delivery_smoke.py --output /kaggle/working/evidence/production-smoke.json
G4_PYTHON="$PY" bash benchmarks/g4-run.sh check
"$PY" benchmarks/harness/p3_bf16_accounting.py --device cuda \
  --model-input /kaggle/input/models/deepseek-ai/deepseek-r1/transformers/deepseek-r1/2 \
  --output /kaggle/working/evidence/storage-accounting.json
```

The runner also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, offline HF/Transformers
flags, `TOKENIZERS_PARALLELISM=false`, and temporary-directory/G4-home paths.
See the reproduction runner for the complete environment and dependency install.

| Check | Recorded result |
|---|---|
| Focused CPU | 562 passed, 36 deselected, 115.05 s |
| Focused CUDA | 36 passed, 562 deselected, 34.84 s |
| Ruff over source/scripts/tests/benchmarks | `All checks passed!` |
| Production smoke | Both codecs passed the CPU/FP32 builder/HF-loop checks above |
| G4 shared-codec check | 362 passed in 22.96 s; CUDA round-trip RMS row errors 0.00892 int8, 0.02652 FP8, 0.15722 int4 |
| Storage accounting | Eight passing isolated storage cases, with the exclusions above |

The G4 `check` command runs `test_g4_codec.py`, `test_g4_harness.py`,
`test_g4_rule.py`, `test_l2l_probe_harness.py` and `test_l2l_session.py`, then CUDA
codec round trips. It does not run the G4 model-quality arms. The full repository
pytest suite was not run by this component launcher.

## Known numerical failures and external handoffs

The [earlier FP32 report](../p3-fp8-kaggle-20261105/README.md) and
[matched BF16 investigation](../p3-bf16-bounded-investigation/README.md) remain the
numerical record. In the latter, all 96 uncompressed L2L windows matched resident
GA bitwise. FP8 refused 75 windows for routing changes; 18 of the remaining 21
failed G4. **Zero complete FP8 cases were accepted.** The old 25% criterion and
rejected dense-only retention policy remain unchanged. A changed research
criterion or forced oracle routing cannot turn those results into natural-route
FP8 acceptance. There is no production oracle mode or new codec in this delivery.

Those numerical runs used Python 3.13 and are preserved as measurements, not
supported-interpreter certification. The component v4 Python 3.12 checks do not
rerun or overturn them. General BF16 A3 acceptance still needs the independent
reference floor unavailable on the tested T4 backend. A public builder selecting
FP16 there still refuses FP8 trainer storage; internal FP32 CUDA tests do not
change that policy.

B1.4 owns the public caller integration and DeepSeek launch admission. Its exact
receiving contract is:

1. Bound fixed consumers and maximum batch/sequence shape. Supply the expert
   input's compute dtype and separately admit its workspace. Resolve
   `memory_plan = plan_moe_l2l(...)` once against the remaining host envelope.
2. Before optional weight allocation, call `plan_c3_l2l` with
   `microbatches=memory_plan.effective_k` and
   `routing_bytes=memory_plan.routing_record_bytes + memory_plan.routing_comparison_bytes`,
   using matching shape, dtype, pinning, codec and device. Do not select k against
   a C3 partition already sized for requested k or subtract activation bytes twice.
3. Construct HF arguments, trainer and scheduler with
   `gradient_accumulation_steps=memory_plan.effective_k`. Pass
   `memory_plan=memory_plan` and `host_budget=c3_plan.activations` through
   `_build_l2l_trainer` to `attach_l2l_state`, retaining the same shared activation
   budget object. The receiving seam checks consistency; it does not perform the
   public caller's admission or rescheduling.

The general BF16 acceptance owner still owns A3. The P3 performance owner still
owns repeated expert reads. Full-model validation still needs the P3 NF4
artifact, real-model k=8 comparison, expert/attention workspace and total host/device
peaks under the specified hardware, cgroup, swap and disk conditions. None of those
handoffs is closed by publishing these two components.
