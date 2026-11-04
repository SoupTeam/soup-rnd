# Local pipeline preparation evidence

**CPU ONLY. No GPU rental or pilot has occurred.**

- `pytest.log`: 350 tests in the expanded H3/FSDP/pipeline selection passed.
- `pipeline-final.log`: all 13 pipeline tests passed on the final harness.
  These overlap the expanded selection and must not be added to its count.
- `ruff.log`: full repository Ruff passed.
- `validation.json`: final fixture identity, source/environment hashes, actual
  stage call counts, parity results, fresh-process reload and file SHA256.
- `cpu-fixture/`: retained fixture from the expanded selection.
- `cpu-fixture-final/`: final-harness fixture; use this for the final parity
  references. Each capture retains its own executed harness SHA in its output
  environment record. Prepared configs retain original absolute capture paths.
- `budget-proposal.json`: read-only offer and prior task-charge snapshots;
  proposed new allowance up to $1 is **not approved**.

Both captures use a random four-layer Qwen2 and synthetic tokens, actual two-rank
CPU/Gloo `PipelineStage` / `ScheduleGPipe` P2P/autograd, two optimizer steps,
microbatch2 × two microbatches × two accumulated schedule calls = global8.
Every stage executes eight actual forwards and backwards. Gradients/updated
adapters match the ordinary control exactly and independent reload has zero
logit error. These results establish fixture logic, not pretrained/GPU training
or performance. Full input/base/adapter/eval references are retained.
