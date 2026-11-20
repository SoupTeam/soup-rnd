# Qwen3-32B NF4: physical 8 GB GPU / bounded RAM

Outcome: **RAM source refused by the existing safety policy, before training**.

This is a resource and same-precision numerical study. It does not measure
32B task quality or establish 70B/120B capacity. The separate 4B BoolQ result
is not extrapolated to larger models.

## Setup and memory boundary

Pinned Qwen/Qwen3-32B; native Soup RAM-backed NF4 streaming, double quantization,
batch 1, accumulation 2, context 512, rank/alpha 4, seven LoRA projections.
No compile, FastLoRA, Liger or explicit FlashAttention was requested. The planned
numerical/timing arms would use identical nonzero initialized native adapters,
not the 4B-trained adapter; those captures and timing did not execute.

Physical GPU: **NVIDIA GeForce RTX 4060 Ti 8 GB**.
Physical RAM: **30.9970 GiB**;
effective physical/native-cgroup bound: **29.7559 GiB**.
Host swap may exist. Model processes abort on observed container swap use;
this is not described as kernel-disabled swap.

No completed accuracy-qualified timing matrix is claimed.
Worker error: `CohortError: Capacity screen did not complete: baseline-capacity`.

NF4 cache preparation completed, but the native setup then rejected
`stream_source='ram'`: base plus resident extras needs 19.2 GB, exceeding the
unchanged 55% physical-RAM safety limit on this 33.3 GB host. The exact ValueError,
zero-training provenance and completed preparation are retained. This is a policy
refusal, not a demonstrated CUDA OOM or proof that 32B cannot train on this machine.
The next matched study will use the supported SSD source with bounded partial RAM
and GPU caches; the source policy and numerical tolerances are not relaxed.

## Retention, controls and spending

- local_protocol_and_retention_controls: 14 tests; 0 failures, 0 errors, 0 skips.
- gpu_host_controls: 144 tests; 0 failures, 0 errors, 0 skips.

No full-repository or hosted-CI success is inferred from these controls.
Current-stage conservative accounting **$0.492097**;
aggregate since the 4B pilot **$2.334227**, combined remainder **$11.009379**.
All earlier 4B attempts are included exactly once; earlier unrelated studies
were already deducted from the starting allowance. This is not an invoice.
Exact instance and persistent volume deleted; independent fresh listings
confirmed zero instances and zero volumes at completion.

[Machine-readable result](result-summary.json), [evidence manifest](evidence-manifest.json),
[full raw-file index](raw-artifact-index.json.gz). Concatenate numbered evidence
parts in manifest order; the complete ZIP round-tripped byte-for-byte.
Full original tensors/logs/archives remain at
`C:/Users/esenb/Documents/SoupResearch/nf4-capacity-32b-20261024-a`.

BoolQ data: Clark et al., NAACL 2019, CC BY-SA 3.0 (data only).
Model weights and credentials are not committed. Actual frozen code, configuration,
all outcomes, cleanup proofs and accounting are included in the evidence.
