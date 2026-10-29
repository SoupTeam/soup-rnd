# A3 criterion 1b: RTX 4090 causal diagnostic, 2026-10-29

**Cause established for this pinned fixture, not a product fix.** Padding policy changes
its forward loss endpoint. With padding matched, disabling ambient CUDA autocast only at
backward entry eliminates the residual gradient discrepancy over the three measured steps,
without changing the measured forward execution or L2L's schedule and accumulation order.
The exact internal cast/kernel and earliest backward divergence remain unresolved.

**The original criterion 1b remains `FAIL / l2l_out_of_tolerance`.** This diagnostic
reproduces its four failures; the later aligned comparison is a new diagnostic result, not
permission to relabel sessions [1](../a3-spill-rtx4090-20261027/README.md),
[2](../a3-spill-v2-rtx4090-20261027/README.md), or
[3](../a3-spill-v3-rtx4090-20261028/README.md). Spill-versus-no-spill equality, RAM, cleanup
and cost are not re-gated here. No product behavior or tolerance was changed by this study.

## Fixture and execution identity

- One vast.ai KVM VM, instance **58293613**, RTX 4090, Australia host **370354** /
  machine **137610**, the same host/machine as spill session 2. Driver **580.95.05**,
  PCIe gen 4 x16. Offered combined rate **$0.5826666667/h**; an offer rate is not an invoice.
- The measured product and original stand were imported from baseline
  **`ec795ef175c222b60ff91b9ba44f7f2bfad55dd5`**, not the current product tree. Diagnostic
  runner/analyzer source: **`cf0a74900617e3140d79b9770c1f617a6a174049`**. The initial
  nine-arm phase ran **19:07:21Z–19:11:57Z**, exited 0, and recorded `state: finished`.
  Adaptive records are separate from that original session, not rewritten children.
- Historical `unsloth/mistral-7b-instruct-v0.3` revision
  `60b99dcbc7f853b7cd18472a607e8d045b96f03e`; NF4, bf16 compute, LoRA r 8,
  k 4, seq 512. The first fetched window uses rows **[5, 6, 1, 2]**, valid lengths
  **[307, 512, 384, 256]**, and **N = 1215 causal supervised tokens**. Padding uses the
  actual fetched window, not a substituted synthetic batch.
- Python **3.12.16**, torch **2.14.1+cu130**, transformers **5.19.0**, peft **0.21.2**,
  accelerate **1.15.0**, bitsandbytes **0.50.2**. Correctness arms use deterministic
  execution and SDPA **MATH**; floor arms use **EFFICIENT_ATTENTION**.
- Every selected original and adaptive arm has identical saved baseline-source fingerprints,
  model-file hashes, initial adapter hashes, NF4 shard-cache fingerprints and dependency
  fingerprints. They are deduplicated without changing their values in
  [`common-fingerprints.compact.json`](common-fingerprints.compact.json). The integration
  owner and independent evidence reviewer inspected these identities; this publication
  does not claim a new source or GPU replay.

## Original 1b failure reproduced

The unchanged pinned stand's `decide`, using the original native-GA EFFICIENT floor,
reports the same four failures as sessions 2 and 3:

| Check (zero-based step) | Relative error | Original limit |
|---|---:|---:|
| Gradient 0 | 0.018090733100463656 | 0.016452461646560432 |
| Gradient 1 | 0.02133434290248132 | 0.020615278589764058 |
| Gradient 2 | 0.17721940196011468 | 0.07667679771063814 |
| Logged loss 0 | 0.00022173059391030002 | 0.000010254296903358378 |

Native GA repeats bit for bit. Its first logged loss is **12.834332466125488**, versus
**12.831486701965332** for original L2L. The full original decision and numeric per-parameter
gradient metrics are retained under `initial_analysis.statistical_1b` and `pairs` in the
[derived compact causal summary](compact-causal-summary.json). Its initial analyzer
`NO CAUSAL VERDICT` is preserved under
`initial_analysis_causal_mechanism_before_adaptive`: that report preceded the interventions
below, and is not silently relabeled as a causal PASS.

## Sequential causal decomposition

These are two controlled contrasts, **not additive components of the original error**.

### 1. Padding changes the forward endpoint

The three observer controls are `BITMATCH`: native GA with/without tracing, padded GA
with/without tracing, and the first window of original L2L with/without tracing. The latter
compares a one-step trace arm with the initial window of the three-step arm, not whole-run
post-step scheduler state. The controls support interpreting the observed initial-window
losses and gradients rather than an observer-induced change.

The original native-versus-padded GA trace contrast changes padding **and** mask policy.
Its earliest observed valid-token difference is **decoder 0, window 0, microbatch 0**:
max absolute **0.001953125**, relative L2 **0.004390999502744277**. This is localization of
that combined contrast, not proof that padding alone first differs at decoder 0.

A subsequent native-GA no-mask control, **`ga_native_no_mask_verified_v3`**, keeps the
native shapes and drops only `attention_mask`. It records actual effective inputs once,
after mask removal. It matches native GA's loss and **all 256 gradient descriptors** bit
for bit, with the same rows, lengths, initial adapters and N=1215. Comparing that native
no-mask endpoint with padded no-mask GA therefore isolates a padding-policy effect on
this fixture: padded GA's first loss is **12.831486701965332**, equal to initial L2L's.
The no-mask arm is untraced, so it does not supply a pure-padding layer localization.

The earlier `ga_native_no_mask` attempt recorded incorrect effective metadata, and
`ga_native_no_mask_verified` failed on an exclusive metadata-file collision. Neither is
published as valid evidence; **only verified v3** supplies the no-mask control.

### 2. Backward-entry autocast explains the matched-input residual

Padded GA and original L2L have matching effective inputs. Their **144 aligned
initial-forward tensors bitmatch**: 140 valid-token/output tensors and four independent
CE numerator scalars. Actual per-row losses, denominators and independently computed
CE numerators match as well. Nonetheless, their first-window gradients differ:
relative L2 **0.007499608746411462**, max absolute **0.2308349609375**.

The adaptive `l2l_backward_outside` wrapper changes only ambient CUDA autocast at backward
entry. It patches `torch.autograd.backward` to call the original callable, forwarding
arguments unchanged, inside `torch.autocast("cuda", enabled=False)`, then restores the
callable in `finally`. Direct `child` execution, recorded in `argv`, keeps the patch in the
measured process. This covers both `loss.backward()` delegation and direct
`torch.autograd.backward(out, grad_tensors=...)` in the pinned L2L path; it does not wrap
explicit forward or reverse-layer forward recomputation.

All **144 original-versus-adaptive L2L initial trace files have identical serialized-file
SHA-256 values**. `layer_calls`, `per_microbatch_losses`, `windows`, `dispatch`, and
`training_step_losses` also remain exactly identical. With those forwards preserved, all
**256 gradient descriptors** now match padded GA bit for bit. The original and adaptive
manifests and `adaptive.original_vs_outside_forward` expose the supporting hashes.

Three-step confirmation uses an independent padded-GA repeat and a newly matched padded
EFFICIENT floor:

| Evidence | Result |
|---|---|
| Padded GA vs L2L-outside gradients | All 256 tensors bitmatch at each of 3 steps; gradient relative errors **[0, 0, 0]** |
| Independent padded-GA repeat | Gradients, losses, rates and step count bitmatch |
| Applied learning rates in both compared arms | **[0, 0.001, 0.0005]** |
| Aggregate logged loss relative errors | **[0, 7.445811247195722e-8, 0]** |
| Unchanged `decide` with newly matched padded EFFICIENT floor | **`within_tolerance`, `pass: true`**, diagnostic only |

**Aggregate logged losses are not all bit-equal.** At zero-based step 1 they are
**12.808198928833008** for padded GA and **12.808199882507324** for L2L-outside.
[`aligned-comparator-confirmed.json`](aligned-comparator-confirmed.json) explicitly
records `losses_equal: false`, alongside exact gradients and a tolerance pass.
[`confirmation-summary.json`](confirmation-summary.json) retains all three-step values.

Backward-scope alignment alone is **not a native-GA fix**: native GA versus L2L-outside
still has gradient relative errors **[0.015837362726258193, 0.014676687301150623,
0.12116253884098395]**. Matching both execution conditions is what produced the aligned
result, and the original comparison remains failed.

## What is and is not explained

**Supported:** on this pinned fixture, padding policy changes the forward endpoint, and
disabling ambient CUDA autocast at backward entry is sufficient to remove the measured
matched-input residual gradient discrepancy. Because L2L's schedule and accumulation
order stay unchanged, an accumulation-order change is **not necessary** for that residual
correction. This is a reviewed counterfactual result, not an inference from aggregate
norms alone.

**Unresolved:** the exact internal cast or CUDA kernel, and the earliest backward
divergence. Residual adjoint coverage is `INCOMPLETE`: four decoder-0 **input** boundaries
are unmatched. Partial boundary alignment cannot rule out an unobserved earlier
boundary. Nor does disabling ambient autocast imply that **all backward work is FP32**:
saved tensor dtypes and explicit custom-backward/checkpoint autocast restoration can still
matter.

The initial analyzer's generic decoder-0 “mask/position/other kwargs differ” warning is
not evidence of a residual mask mismatch. In all four captured calls, arguments and the
remaining kwargs match; the sole difference is GA's extra `num_items_in_batch` kwarg,
which L2L's capture drops. Actual loss denominators match. The summary retains the
original warning and captured calls, plus the explicit key-by-key explanation.

This does not establish a production-safe cutover, generalize to other models/stacks/shapes,
or claim model quality, convergence, a backward-wide precision policy, or a new spill result.
No automatic product change is part of this request.

## Published records, provenance and replay limits

This folder is a **compact evidence publication**, not a complete tensor archive. It
contains nine original and six valid adaptive arm projections, all 15 original manifests,
three original trace projections, common fingerprints, the original session record, the
aligned comparator and confirmation summary, and five small execution/fixture/package
records. No tensors, logs, operational/rental scripts, keys, API snapshots or failed control
attempts are copied. The adaptive wrapper sources are retained privately: their raw hashes
and exact behavioral descriptions are in
[`provenance.json`](provenance.json), not substituted source files.

Files ending `.compact.json` and `compact-causal-summary.json` are explicitly **DERIVED**:

- Common source/model/cache/dependency/initial-adapter fields are retained once, after
  checking equality across all 15 source records; arm projections reference that file.
- Arm projections retain every other original field, including all gradient SHA-256/dtype/
  shape/finite metadata, row/window inputs, per-row losses, layer calls, rates and artifact
  manifests, except the large dispatch payload. Its canonical sorted-key compact-JSON
  SHA-256 is retained. The forward-preservation summary separately records exact dispatch
  equality between original and adaptive L2L.
- Trace projections remove only repeated `valid_positions` arrays; their other original
  fields remain. The causal summary retains all initial per-parameter numeric gradient
  metrics and forward-comparison metrics, but omits duplicate full arm records and replaces
  dispatch rows and adjoint-boundary detail with counts and canonical JSON digests.
- The summary identifies its source files and raw hashes, retains the pre-adaptive report
  separately from the reviewed interpretation, and derives adaptive equality checks from
  saved descriptors/manifests. It does **not** recompute tensor norms or the comparator.

The original compressed analysis is private: **6,794,136 B compressed**, **57,439,661 B
uncompressed** (uncompressed SHA-256
`6a5cde7c311d8643a15cfe924d241ae3c6622e041f9a8d37abb416b815e376da`). It is not committed.
Public projections permit inspection of equality metadata, inputs, numeric results and
controls, but **cannot recompute numeric tensor comparisons and are not drop-in analyzer
inputs**. Replay needs the retained raw arm/tensor/trace layout and the pinned baseline.
The diagnostic runner and analyzer are
[`a3_1b_diagnose.py`](../../harness/a3_1b_diagnose.py) and
[`a3_1b_analyze.py`](../../harness/a3_1b_analyze.py); exact recorded original and adaptive
commands are in the published session and arm `argv` fields. To repeat the original phase
with those prerequisites, use a fresh output directory:

```bash
python benchmarks/harness/a3_1b_diagnose.py session \
  --baseline-path /work/baseline-ec795ef1 --weights /work/mistral-7b \
  --shards /work/nf4-shards --out /work/new-diagnostic --execute
python benchmarks/harness/a3_1b_analyze.py /work/new-diagnostic \
  --baseline /work/baseline-ec795ef1 --out /work/new-analysis.json
```

The adaptive counterfactuals additionally require the exact wrapper behavior described
above and hashed in provenance; an ordinary original-phase replay does not apply them.

For path-only copies, **only** these substitutions change source bytes, longest first:
`/root/a3/baseline` → `<baseline>`, `/root/a3/repo` → `<diagnostic-work>`, then
`/root/a3` → `<box>`. For derived files, provenance explicitly describes the additional
field removal, deduplication, serialization and interpretation; it never labels them as
path-only sanitization. Each evidence entry records raw source and published SHA-256.
The LF [`SHA256SUMS`](SHA256SUMS) covers **44 evidence/provenance files**, not this authored
README or the manifest itself. Root `.gitattributes` marks the entire result directory
`-text` to preserve those bytes.

The integration owner verified the complete private initial tensor archive,
`diagnostic-evidence.tar.gz`: **3,910,733,600 B**, **1,599 entries**, SHA-256
`1f765ff706a904fdfc4fdc16909ce73b9f12c2eab75a92dad03cca988d346a39`.
The complete private adaptive archive, **`adaptive-evidence-complete.tar.gz`**, was also
verified locally before deletion: **1,737,457,063 B**, **724 entries**, SHA-256
`fb10633a24821e519d40b7aa10e15b0934a43bf4bdb4dbf76eeb241068d92969`.
The earlier `adaptive-evidence.tar.gz` transfer was partial; it is not the verified archive.
Private `metadata.tgz` SHA-256:
`7fcd19daeed18ba0246f83a4d1b5409639849705fbf3abcbbe254b7d629fc60c`;
private `final-metadata.tgz` SHA-256:
`7c4e1640e72ad0512fdd2a86d63372ed4b8899b403f630e175816967f5728750`.
These archive attestations are separate from the publisher's metadata-only projections;
publication did not reload their tensors or rerun GPU work, tests, lint, or builds.

## Rental closeout and posted charges

After both complete tensor archives were hash-matched locally, the integration owner
explicitly ran **`vastai destroy instance 58293613 --raw --yes`**. At
**2026-10-29T20:19:30.040875+00:00**, the API returned **`instances: null`** and the
instance was absent from the listing. This is the absence-check time, not an exact
deletion-completion timestamp. An earlier destroy attempt without `--yes` aborted and
had no effect; its return code 0 is not evidence of successful deletion.

At **2026-10-29T20:19:33.358764+00:00**, instance-scoped **posted** charges were GPU
**$0.700** (1.313 h at $0.5333/h) and storage **$0.066** (1.339 h at $0.0493/h), total
**$0.766**. These are the posted invoice amounts, not calculations from rounded durations
or the offer rate. **Immediate post-deletion billing can lag; this is not a final-bill
claim.** Adding the prior A3 sessions' $1.290 gives **$2.056 posted**, leaving **$7.944**
of the $10 budget on that posted basis. Provenance retains hashes and whitelisted values
from private `confirm-gone.json`, `billing-posted.json` and the aborted first response;
the API/billing snapshots and unrelated instance IDs are not published.
