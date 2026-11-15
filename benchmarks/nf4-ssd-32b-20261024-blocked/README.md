# Qwen3-32B NF4: bounded-RAM disk streaming

Outcome: **FAILED**.

Physical 8 GB GPU; real physical/native-cgroup RAM upper bound <=32 GiB.
Pinned Qwen3-32B, batch 1, accumulation 2, context 512, ordinary PEFT LoRA
rank/alpha 4 on seven decoder projections. Fixed double-quantized NF4 weights
and identical nonzero native adapters across cache arms. No compile, Liger,
FastLoRA or explicit FlashAttention change. This is numerical/resource research,
not a 32B task-quality, convergence or 70B/120B capacity result.

Actual GPU: NVIDIA GeForce RTX 4060 Ti.
Physical RAM 30.9970 GiB; effective bound
29.7559 GiB.
Host swap may exist; model stages reject observed container swap use.
No kernel-disabled-swap claim is made.

No completed accuracy-qualified timing matrix is claimed.
Worker error: `CohortError: Non-capacity failure in host-cache screen`.

NF4 preparation and the GPU-only capacity screen completed. The combined
14 GiB RAM hotset / seven GPU-layer screen then stopped on the existing
pinned-staging loan guard (layer 0 still borrowed when large layer 64 was requested).
It did not reach independent accuracy qualification or accepted timing.

The root also discovered an unrelated prospective diagnostic warmup=0 API
error and requested a stop. SIGINT was inherited as ignored; the native
capacity guard subsequently ended the attempt. Stop intent is not described
as confirmation that the job stopped before its preliminary capacity work.

The cache-lifetime defect was reproduced locally, then fixed without bypassing
the cold-loan guard. Exact patch bytes, a real-CUDA regression awaiting the retry,
red/green local evidence and two killed mutations are retained separately from
the original GPU snapshot. No GPU qualification of that fix is claimed here.

## Storage, controls, retention and spending

The actual container disk mount and loop-device details are retained. Physical
NVMe backing is not assumed; disk-source or Linux I/O counters do not establish
physical NVMe service time or eliminate host filesystem caching effects.

- local_protocol_and_retention_controls: 32 tests; 0 failures, 0 errors, 0 skips.
- context_refusal_control: 1 tests; 0 failures, 0 errors, 0 skips.
- profile_interval_control: 1 tests; 0 failures, 0 errors, 0 skips.
- gpu_host_controls: 144 tests; 0 failures, 0 errors, 0 skips.

No full-repository or hosted-CI success is inferred from these controls.
Current-stage conservative accounting **$0.508311**;
aggregate since the 4B pilot **$2.842538**;
combined remainder **$10.501068**. Not an invoice.
All prior pilot/capacity retries are included once; earlier unrelated studies
were already deducted from the starting allowance. Exact instance and persistent
volume independently confirmed absent by exact IDs. Unrelated resources were preserved;
the full account inventory is retained in the cleanup evidence.

[Machine-readable results](result-summary.json), [evidence manifest](evidence-manifest.json),
[full raw-file index](raw-artifact-index.json.gz). Concatenate numbered evidence
parts in manifest order; the ZIP round-tripped byte-for-byte. Complete original
outputs remain in the stable SoupResearch directory identified in the raw index.

BoolQ: Clark et al., NAACL 2019, CC BY-SA 3.0 (data only). Model weights and
credentials are not committed. All outcomes and original tolerances are preserved.
