# Retained conservative BF16 GPU evidence

Four sequential real Qwen2.5-0.5B BF16 arms on one Linux host with two RTX 4090s.
See [the report](../../../h3-bf16-result-20261027.md) and
the protocol committed before rental. This scope uses deterministic math SDPA,
FP32 intermediate reductions and explicit matching BF16 autocast. It does not
admit the earlier failed ordinary BF16 or lost-artifact observations.

`campaign/` keeps raw commands, exit codes, logs, all six rank records,
resolved Soup configs, versions, actual parameter placements and memory/timings.
It also keeps all first-step/init/preclip tensor snapshots and eight adapters.
`prepared/prepared.json` identifies immutable base files and the fixed dataset.
`support/` contains collection/offline acceptance checks and cleanup/budget proof.

The five full archives listed in `artifact-manifest.json` remain at exact local
paths with byte counts, SHA256 and per-file hashes. They retain every pre-save
logit/adapter reference and the prepared Arrow dataset. Those larger reference
files are excluded from this Git subset. To replay adapter admission/scaling on
CPU from public records (this is analysis of GPU evidence, not new GPU execution):

```bash
python benchmarks/harness/h3_fsdp_campaign.py --summarize-only \
  --prepared benchmarks/evidence/h3-bf16-20261027/hardware/prepared \
  --output benchmarks/evidence/h3-bf16-20261027/hardware/campaign
```

`campaign/sha256.json` is the original full remote ledger and includes omitted
files; `artifact-manifest.json` is the ledger for this public subset. Raw files
are copied byte-for-byte, including the remote summary. Offline replay saves and
restores those raw summary bytes to avoid Windows newline changes.
Optimizer/scheduler/RNG resume is unmeasured and excluded from the archive contract.
