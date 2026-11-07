# Qwen3-32B NF4: bounded-RAM disk streaming

Outcome: **GPU-regression fixture precondition failed before model download**.

148 of 149 GPU-host source/helper controls passed. The new promotion regression
stopped because its RAM hotset was empty: after excluding GPU layers, the two
remaining decoder layers were already covered by read-ahead=2, and the 8 KiB
ceiling also did not represent native pinned-arena allocation. This did not
exercise the production fix's intended cached-read path and is not a 32B accuracy
or performance failure. The corrected fixture keeps three eligible decoder layers,
a distinct uncached singleton vocabulary group, and budgets the actual pinned arena.
A CPU precondition check verifies that exact nonempty pinned plan before retrying.

All original failed output is preserved; only this publication directory was
renamed to distinguish the fixture-check attempt from the subsequent measurement.

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
Worker error: `CohortError: Actual GPU source/regression controls failed`.

## Storage, controls, retention and spending

The actual container disk mount and loop-device details are retained. Physical
NVMe backing is not assumed; disk-source or Linux I/O counters do not establish
physical NVMe service time or eliminate host filesystem caching effects.

- local_protocol_and_retention_controls: 32 tests; 0 failures, 0 errors, 0 skips.
- context_refusal_control: 1 tests; 0 failures, 0 errors, 0 skips.
- profile_interval_control: 1 tests; 0 failures, 0 errors, 0 skips.
- native_profile_and_stop_protocol_controls: 3 tests; 0 failures, 0 errors, 0 skips.
- shared_account_ownership_controls: 4 tests; 0 failures, 0 errors, 0 skips.
- gpu_host_controls: 149 tests; 1 failures, 0 errors, 0 skips.

No full-repository or hosted-CI success is inferred from these controls.
Current-stage conservative accounting **$0.312839**;
aggregate since the 4B pilot **$3.155377**;
combined remainder **$10.188229**. Not an invoice.
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
