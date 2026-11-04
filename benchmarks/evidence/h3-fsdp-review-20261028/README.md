# H3 RAM-efficient loading regression evidence

See the [review follow-up](../../h3-fsdp-review-20261028.md) for the trigger,
implementation and limits.

- `pytest.log`: expanded local selection, 337 passed (15 upstream warnings).
- `ruff.log`: full repository Ruff scope, passed.
- `mutation.log`: intentionally restored premature sharding, expected failure.
  This negative run is excluded from passing-test counts. The source was
  restored byte-for-byte immediately afterward.
- `environment.json`: exact commands, versions, base commit, worktree SHA256
  and normalized Git blob hashes for the implementation/test files.

These tests use real two-rank CPU/Gloo DTensors and loading collectives, with
lightweight accelerator/plugin state. No CUDA training, performance or resume
claim is derived from them. Existing GPU evidence remains separate and immutable.
