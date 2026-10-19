# D1 RTX 3090 VM F1, 2026-10-19

Created through the authenticated user's team context with a separately approved
personal SSH key and the official Ubuntu VM template (741094). No personal-credit
top-up or team-credit transfer was needed. The earlier rejected Docker cohorts
and their NO_VERDICT are unchanged and never pooled into this cohort.

The private editor copy was not used: VM type did not persist in that copy.
The executed resource is the official VM, with guest cleanup installed over SSH.
resource-amendment-before-cuda.json extends only the time/cost allowance under
the user's new $15 cumulative budget, before CUDA/timing; all numerical F1 gates,
five ABBA blocks, sample counts, clock guard and sizing rule remain unchanged.

The initial clock request reported success, but a later check showed 1845 MHz
with persistence disabled. Before CUDA gates, persistence-before-cuda.sh enabled
persistence and reapplied 1200 MHz on
the leased UUID only. Actual primary/reference telemetry decides acceptance.
clock-restore.json records reset clocks and disabled persistence before deletion.

raw.tgz contains source.tar, run.sh, run.log and all saved results, CUDA witnesses,
per-step timings, GEMM references and telemetry. audit.json is an independent CPU
replay of saved tensors/source/medians/the unchanged F1 validator, not a CUDA run.
Read its actual verdict, not this README, to determine acceptance. No additional
blocks or long training were added. Full D1 still needs an owner coverage decision.

Budget snapshots include D1-owned IDs only; invoices can lag. The VM is deleted.
No private SSH key, account API key, browser session or account screenshot is
included. All earlier archive blobs remain unchanged.
