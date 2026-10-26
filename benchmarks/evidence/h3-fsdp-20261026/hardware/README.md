# Retained Linux GPU evidence — partial H3

Read [the report](../../../h3-fsdp-lora-20261026.md) for controls, definitions,
commands, acceptance ledger and limits. `result.json` deliberately leaves speedup
and efficiency null. None of these differing runs form an accepted scaling pair.

- `campaign/`: initial 34b99878 1-GPU training/reloads and failed 2-GPU count gate.
- `campaign-v2/`: 491ea26a 1-GPU training/reloads and failed 2-GPU first forward.
- `diagnostic-storage/` and `support/storage-rank-*.json`: pre-fix actual DTensor
  storage readings, not proof that the failed model could train.
- `tied-fix-smoke/`: e3ea457f real 2-GPU rank readings, environment, first-step
  tensors, root and checkpoint-4 adapters. Four optimizer steps completed.
- `support/h3-fixed-train.log`: successful four-step training.
- `support/h3-fixed-reload.log`: both fresh-base reload checks passed before the
  supplemental first-step control raised. The combined worker exit code is 1;
  it must not be turned into an overall campaign success.
- `support/first-step-control-analysis.json`: all-tensor offline analysis with
  unchanged atol=1e-5, rtol=0.02. Both gradient and adapter comparisons fail.
  Matching raw tensor pairs are kept under campaign-v2/repeat-1-gpu-1 and
  tied-fix-smoke. This compares first-step behavior only, across different source
  commits/run horizons, and establishes neither scaling nor full trajectory parity.
- `support/*.capture.txt`: actual launcher/validation/diagnostic source captures.
  These are preserved as data, not imported by the benchmark.
- `support/destroy.json` and `deletion-verification.json`: successful deletion of
  the new pilot resource only; estimate is not a provider invoice.
- `artifact-manifest.json`: SHA256 and byte sizes of retained files and the two
  larger archives stored outside Git in the user's Codex output directory.

**Retention gap:** the watchdog deleted the server at the original deadline
before the approved extension applied. The final smoke's 40 MB pre-save numeric
reference and completed reload JSON files were not copied. Its validation log
records both completed checks, but the exact max logit error is unknown and
cannot be independently recalculated from these artifacts. The earlier 1-GPU
arms retain their full references in the external archive and report zero error.

The rank status `training_passed_reload_pending` is the training process's own
status, unchanged after its exit; the separate validation log supplies reload
evidence. Root/checkpoint adapter safetensors also independently match exactly
(96 keys). Optimizer/scheduler resume is unmeasured regardless of checkpoint
file presence. Existing #373 probe source/tests were not changed.
