# D3 public trainer attempt 01: failed evaluation control

**No qualification verdict.** The first formal cell at committed source
`cd6515ac503b15a111050b35db5c33768d506a9b` is retained as **FAIL**. There are
zero accepted cells, no completed A/A or negative/restored control, no
save/reload qualification and no timing/profile comparison from this attempt.

The exact source export contains 767 committed files and passed source,
software, GPU/boot and all 157 prepared input-file identity checks before run.
The native-resident dense ordinary-SFT cell exercised all sixteen pretrained
OLMoE decoder layers, frozen experts and attention LoRA through the public
SFT wrapper. Four real SGD steps reported finite loss, gradients and parameter
states. The callback's final evaluation produced nonfinite logits and raised
`NumericalFailureError`. Unchanged F1 recorded **OK**, including mains power,
no suspend and no foreign disk reads. Neither memory exhaustion nor a changed
GPU/source identity was the reported stopping condition.

## Diagnosis and corrective scope

The evaluation callback selected `model.get_input_embeddings().weight.device`
for its token IDs. Untied streamed embedding/head weights keep meta parameters
while real weight bytes arrive through the runtime's buffers. The parameter's
`meta` device is therefore not an execution device for actual token IDs.
The original small synthetic fixture tied embedding/head and did not exercise
this boundary. This was a gap in the preparation coverage.

A new CPU regression with an untied public streaming model fails before any
embedding calculation: its hook observes a meta token-ID tensor. After the
callback chooses `wrapper._stream_runtime.device`, the same regression passes.
Additional four-layer CPU ordinary-SFT/L2L controls compare against native
ordinary accumulation and retain exact replay/reload checks. These demonstrate
the instrumentation defect and synthetic correction; they do not retrospectively
pass this failed pretrained cell or prove the next pretrained run will pass.
Production training kernels, optimizer/data recipe and numerical criteria are
not changed by this one-line callback correction.

## Retention

`records.zip` retains the raw session/analysis, child result and F1 record,
traceback, child and operational logs, canonical-source identity and operational
launch/freeze scripts. Every included entry is SHA-256 listed; the ZIP contains
an explicit full-archive retention receipt.

The original full archive is retained locally and on the server:

- Filename: `d3-public-trainer-results-cd6515ac.zip`
- Size: **897,498,866 bytes**
- SHA-256: `08b731f604368dc03a21d7def54be095617428d03753aea4504ef65aa5309bcb`

All 26 full-archive payload hashes were checked after download. Four large
native step-tensor files and the redundant 46 MB committed-source ZIP remain
in that full archive, with identities in `failure.json`; they are not claimed
as Git-published payloads. The source commit and frozen Git-blob manifest remain
available independently. No raw native tensor, failed record or session marker
was removed on the server.

The parent stopped after this first cell, the operational wrapper produced
the incomplete-session analysis and retained the archive, and no replacement
or automatic retry was started. A new formal attempt requires its own corrected
prospective source/rule commit. The four instrumented training-step durations
are not used as throughput evidence or a substitute for the declared timing arms.
