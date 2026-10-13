# D2 complete LoRA step: PEFT, Soup, Liger, Unsloth — 2026-10-14

**Decision: external-baseline numerical gates failed; full-step speed is unproven.**
All four real GPU stacks completed 50 training updates in FP16 and BF16 on one
RTX 3090. In the deterministic execution control, Soup matched PEFT exactly in
the retained adapter weights and gradients, and all 50 losses matched at `.3f`.
Liger and Unsloth did not meet the predeclared additional full-model numerical
criterion. The performance stage was blocked: **0 accepted timing arms / 72 planned**.
D2 is not declared complete by this report.

## Scope and results

Complete pretrained TinyLlama 1.1B, revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`; dense FP16/BF16 frozen base,
FP32 rank-16 adapters at all seven projection types, batch 1, sequence 512,
fixed synthetic tokens, dropout/checkpoint/cache off. The step includes forward,
loss, backward, scaling when enabled, and AdamW. This is full-model **LoRA**,
not full-weight training, an 8B/NF4 model, real-data quality, or Soup CLI throughput.
See the [frozen protocol](d2-training-baselines-protocol.md) for optimizer,
warmup, fixture identity, execution proof and lifecycle requirements.

The additional full-model gate requires all 50 losses to match the same-dtype
PEFT reference at `.3f`, and each retained adapter tensor to have relative L2
error <=1%. It is an experimental criterion declared before the rental, not a
claim that every upstream library promises bitwise-equivalent training.
There are 1,232 tensor quantities per case: 308 adapter parameters, weights and
unscaled gradients, at steps 1 and 50. PEFT compared with itself is the reference,
not an independent correctness result.

| Deterministic case vs PEFT | Losses matching `.3f` | Tensor quantities failing / 1,232 | Result |
| --- | ---: | ---: | --- |
| Soup FP16 | 50/50 | 0, all retained tensors exact | PASS |
| Liger FP16 | 1/50 | 1,232 | FAIL |
| Unsloth FP16 | 1/50 | 1,232 | FAIL |
| Soup BF16 | 50/50 | 0, all retained tensors exact | PASS |
| Liger BF16 | 1/50 | 1,226 | FAIL |
| Unsloth BF16 | 0/50 | 1,225 | FAIL |

All eight final GPU process lifecycle checks passed: empty before/after,
single persistent owner and matching GPU UUID, observed clock spread <=5%,
valid power/state/throttle telemetry. This supports workload isolation; it does
not turn a failed numerical comparison into an accepted performance comparison.
Raw gate records contain diagnostic latency/memory fields, explicitly pending
supervisor/replay. Do not quote their ratios or tokens/sec as accepted speedups.

The original seed-792 strict50 probe also passed **50/50 in FP16 and BF16** in
this CUDA environment. Earlier projection timing and NF4/streaming evidence
retain their previous narrower scope; this run does not revalidate NF4 on a
complete pretrained model.

## Controls, failures and changes to the experiment

1. The initial packaged source omitted the decision-rule file and stopped in
   provenance preflight. The corrected source retained the failed output.
2. Unsloth's initial constructor altered the frozen embedding tensor, so its
   input-identity check failed before training. The harness now restores altered
   frozen parameters from the identical pinned checkpoint and requires their
   original SHA-256 exactly. The adjustment and before/after digests are recorded.
   No changed input is admitted with a numerical tolerance.
3. The first complete matrix used default CUDA execution and failed the numerical
   gate across candidates. Same-source, fresh-process PEFT controls also failed:
   FP16 matched 1/50 losses, BF16 8/50. Therefore those initial cross-library
   failures cannot establish a library-specific defect or rank implementations.
4. A separate execution control sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
   imports and enables PyTorch deterministic algorithms. Seeds, tokens, base,
   adapters, optimizer, scaler, step count and criteria did not change. Both
   fresh-process PEFT reference repeats then passed 50/50 and all 1,232 tensor
   quantities exactly. The final four-stack matrix used this control.
5. The final `--gates-only` run collected both precision matrices and rejected
   the external comparisons. No thresholds were relaxed and no performance
   campaign was launched after those failures.

The root cause of the Liger/Unsloth differences is **not yet established**.
Their stacks change normalization, attention, loss and fused backward paths;
the result cannot be attributed solely to a LoRA kernel. Do not describe this
as a general correctness failure of either upstream project.

CPU replay of the selected first-step `layers.0.mlp.down_proj.lora_A` witnesses
narrows the diagnostic boundary: using the saved unscaled gradient and identical
initial parameter in AdamW reproduces each saved GPU post-update weight within
`1.87e-9` maximum absolute error. The selected gradients already differ from
PEFT by about 9.3% relative L2 in FP16 and 4.9–5.1% in BF16. This is evidence of
a difference upstream of that first optimizer update for those four witnesses;
it does not locate a particular attention/norm/loss/backward operation or prove
the same explanation for every parameter.

## Environment and reproducible evidence

RTX 3090 24 GB (sm86), driver 595.71.05, Linux/Python 3.12.3; torch 2.11.0+cu126,
Triton 3.6.0, Transformers 5.17.0, PEFT 0.20.0, Liger 0.8.4,
Unsloth/Unsloth Zoo 2026.10.4, xformers 0.0.35, bitsandbytes 0.50.1.
The pinned environment passed `pip check`; a compiled Triton GPU smoke kernel
executed and matched its expected output. Metadata resolution alone is not the
basis for claiming those kernels executed: per-arm hooks/profiler records are
retained for the actual Soup, Liger and Unsloth paths.

[Evidence index and replay instructions](evidence/d2-baselines-20261014/README.md)
include every 50-step loss curve, original strict50 outputs, GPU lifecycle
records, frozen source archives, environment and fixture manifests, CPU audit
summaries, and raw first-step tensor counterexamples. The compact public replay
independently verifies **81 final payload hashes, eight loss curves, four failed
loss curves and eight failing weight/gradient pairs**. The earlier default-mode
bundle independently verifies 85 hashes, eight curves and 12 witnesses; retain
it as a failed-reference-control experiment, not a library verdict.

Public compact witnesses establish the reported failures. They **do not** replay
all omitted tensor quantities or independently establish every positive Soup
tensor match. Complete tensors and reference repeats are retained locally in
two losslessly deduplicated raw archives; their hashes and sizes are indexed.
Full CPU replay of those archives is recorded separately from compact replay.
Model checkpoint bytes are omitted; the immutable model revision and base-tensor
SHA-256 digests identify the required download.

Source transport history: final full matrix used archive SHA-256
`a7dff39f466d354399bc2a87ae4d93502b2caa98cc6f0ea0fed5d33ba6e27bbb`.
The deterministic reference worker preceded the gates-only argument. Its original
outer ZIP was overwritten during preparation; it is **not claimed preserved**.
Its worker was reconstructed from the final source by removing that argument
and early-return branch, then verified against the recorded worker SHA-256
`1a8d344b9ab4429ce432a569de3d2e6c9a7d1feb3e8ff647ea25ffd9313634b9`.
The reconstruction and method are included and labelled accordingly.

## Validation and remaining work

59 selected CPU integration/rejection/audit/timing tests passed; repository
Ruff passed. These checks validate the harness and evidence handling, not an
unmeasured GPU speedup. Production Fast-LoRA arithmetic was unchanged by this
comparison work. Final-head full CI and team review remain separate acceptance
requirements.

Before requesting another paid run, diagnose the retained first-step discrepancies
on CPU and review whether the declared whole-stack criterion is the intended
acceptance criterion. Any proposed change to backend configuration or criterion
must be explicit and independently justified, with the failed experiments kept.
Only after numerical validity is established should a newly authorized rental
collect the full ABBA timing matrix. Additional GPU time alone cannot guarantee
that D2 closes or that a speedup exists.

Instance `55797036` was destroyed and verified absent through the provider API
after 5,442.5 seconds (1 h 31 min), within the authorized two-hour/$1 ceiling.
Observed instance-specific provider charges after deletion were **$0.449**,
including compute, disk and traffic; this is an observed charge record and may
settle later, not an aggregate team-balance difference. Compute/storage estimate
from the accepted rate and local wall time is $0.420. The sanitized
[rental receipt](evidence/d2-baselines-20261014/rental-summary.json) records the
accounting boundary. Both full ZIPs passed local SHA-256/CRC verification and
334 materialized payloads passed their manifests before publication.
