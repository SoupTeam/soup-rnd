# Retained evidence: stream8-compile-20261022

This is a preflight-only record. No CUDA numerical/compile/timing run occurred.
`manifest.json` records SHA-256 and byte sizes of the retained diagnostics and
`attempt-inputs.zip`. No account credentials, transfer-secret values, connection
addresses, private provider responses or model weights are included.

The ZIP preserves the exact attempted controllers and intended worker scripts.
The archive is an audit input, **not a command to restart the historical lease**.
Its `cloud.py` contains the historical exact label/offer and local key-file path.
Any authorized replay must use fresh ownership identifiers, a fresh independent
budget policy, private connection/receipt files, new transfer secrets and an input
hash manifest. It must reconcile uncertain creation outcomes before retrying and
delete only its own matching label/ID in `finally`, with a separate absence check.
The Paramiko binary transport requires Paramiko and a previously recorded trusted
SSH host key; it does not disable host-key verification.

The frozen model harness already lives in
[`../streaming-residency-compile-8gb-20261021/planned-harness.zip`](../streaming-residency-compile-8gb-20261021/planned-harness.zip).
That overlay plus upstream `d500ae0dca7c4da905f8d59c9866af2f8aaa824b` defines the
planned executable source. `current-source-audit.json` verifies the 602 production
files and records the comparator AST/source-archive hashes. Only the architectural
adapter-count assertion differs from the older 32-layer capture; tolerances do not.

Local checks from the repository root:

```bash
python -m pytest tests/test_streaming_residency_candidate.py \
  tests/test_streaming_residency_compile_bench.py --no-cov -q
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
git diff --check
```

The decisive records are `network-preflight.json` (completed SHA-verified binary
probe), `capacity-preflight.json`, `final-budget.json`, `vast-deletion.json` and
`final-instance-absence.json`. The latter is the independent deletion check.
`post-transfer-guard.json` contains the last observed container traffic and budget
telemetry; `remote-guard-before-setup.json` proves the independent remote guard ran.
