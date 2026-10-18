# D1 F1 finish session, 2026-10-19

Independent hardware/clock cohorts, never pooled. The existing rule and historical
NO_VERDICT are unchanged. Each raw.tgz contains exact source.tar, run.sh, run.log
and all results/tensors/telemetry. CPU audit.json verifies saved tensors, source
provenance, wall medians, dispatch and the unchanged formal validator. It is not
a new CUDA run. Read each attempt's booking-plan.json and actual clock telemetry;
a requested clock policy does not by itself prove frequency stability.

The VM allocation error is preserved separately: team accounts do not support
account SSH-key registration and VM creation required such a key. No VM lease
was created and no account-wide SSH key was added. Any startup failures likewise
have no CUDA/performance claim. Never treat failed controls as passing coverage.

manifest.json lists each artifact's size and SHA-256. budget.json contains only
D1-owned-instance charges. All session rentals are deleted; final billing can lag.
The original evidence archive and earlier follow-ups remain unchanged.
