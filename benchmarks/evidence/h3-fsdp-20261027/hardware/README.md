# H3 retained GPU evidence — accepted FP32 first stage

See [the report](../../../h3-fsdp-followup-20261027.md) for controls, measured
results, commands, acceptance and limits.

- `campaign-fp32/`: four complete primary arms (two pairs, 1/2 then 2/1), all
  rank/config/environment records, raw commands/exits/logs, first-step tensors,
  final and periodic adapters, reload reports, summary and original hash ledger.
- `campaign/`: failed c5ac8463 BF16 follow-up (2-GPU reload mismatch).
- `campaign-amp/`: c6bd159e BF16 train/reload passed, strict first-step gate failed.
  Both directories are diagnostic only and excluded from the accepted summary.
- `prepared/prepared.json`: immutable base revision/file hashes and dataset hash.
- `support/`: launch captures, bootstrap logs, source/deletion/budget/preservation
  verification, local revalidation and cumulative rental ledger.
- `artifact-manifest.json`: public file hashes plus the complete external archive
  path/size/SHA256. All four pre-save numeric references survived this run.

First-step tensors and all eight primary adapter safetensors are committed.
Large reference logits/optimizer/tokenizer duplicates are retained in the full
archive outside Git. CPU `--summarize-only` can replay all numerical/report gates
from this public subset; it is analysis of real GPU records, not hardware execution.
Adapter reload is not optimizer/scheduler resume. BF16 strict cross-world parity,
4/8 GPUs, 70B and model quality remain unmeasured or explicitly unresolved.

`campaign-fp32/sha256.json` is the original full-tree ledger and includes
large files kept only in the archive. `artifact-manifest.json` lists the
committed subset and the complete archive hash.
