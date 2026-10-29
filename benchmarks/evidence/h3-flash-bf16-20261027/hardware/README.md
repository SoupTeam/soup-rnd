# Native Flash BF16 GPU evidence

The accepted case is `campaign-deterministic/`: four sequential 18-step arms,
eight independent root/checkpoint reloads, actual CUDA Flash operators on every
rank, ordinary BF16 GEMM reduction enabled, deterministic algorithms enabled.
The default nondeterministic `campaign/` failed the original strict first-step
gate after two complete arms. Its four reloads passed; its timing is excluded.
See [the result report](../../../h3-flash-bf16-result-20261027.md) and the protocol.

All numeric references were downloaded/hash-verified before cleanup. The eight
full archives in `artifact-manifest.json` retain every init/preclip/first-step
and eval reference, all twelve adapter saves, raw records and prepared Arrow
data. Their exact local paths, bytes, SHA256 and per-member hashes are recorded.
This Git subset includes raw commands/logs/configs/versions, nine physical rank
records, first-step/init/preclip tensors and all twelve adapters. Larger eval
references and the complete dataset remain in the full archives.

CPU replay of the accepted GPU records (this is not new GPU execution):

```bash
python benchmarks/harness/h3_fsdp_campaign.py --summarize-only \
  --prepared benchmarks/evidence/h3-flash-bf16-20261027/hardware/prepared \
  --output benchmarks/evidence/h3-flash-bf16-20261027/hardware/campaign-deterministic
```

`campaign-deterministic/sha256.json` is the original full remote ledger,
including files omitted from Git. `artifact-manifest.json` describes this
public subset. Raw summary bytes are restored after offline replay so Windows
line endings do not alter captured hashes. Optimizer/scheduler resume is unmeasured.
