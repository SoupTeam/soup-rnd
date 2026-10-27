# H3 follow-up protocol — 2026-10-27

The first pilot's strict one/two-GPU first-step comparison failed. Its results
remain unchanged and excluded from scaling. See the 2026-10-26 report.

Owning Torch 2.13 `MixedPrecisionPolicy` documentation/code distinguishes the
original optimizer/master dtype from the dtype of the materialized forward
parameter. The previous harness configured BF16 materialization for FSDP2, while
ordinary PEFT kept LoRA FP32 under BF16 autocast. It also forced all FSDP group
outputs, potentially including FP32 loss, to BF16. FP32 reduction alone does not
make these policies identical. This is a concrete control mismatch; whether it
explains all observed discrepancies requires a fresh same-commit GPU comparison.

Before any new measurement, the follow-up protocol is fixed:

- Same immutable Qwen2.5-0.5B revision, synthetic data/seed/seq512, LoRA,
  BF16 base/autocast, FP32 adapter parameters/gradients, optimizer and global8
  as the initial report. One GPU micro1/GAS8; two GPUs micro1/GAS4.
- FSDP2 `param_dtype=None`, `output_dtype=None`, `reduce_dtype=float32`:
  preserve the original parameter/output dtypes; let the same autocast context
  select compute dtypes in both arms. Frozen tied embedding group remains BF16.
- Read actual trainable parameter dtypes in the first forward after FSDP unshard
  hooks. Every LoRA tensor must be observed FP32. Remove observation hooks before
  timing. Master dtype alone cannot pass this admission check.
- Keep the existing first-step comparison unchanged: `atol=1e-5, rtol=0.02`,
  all keys and elements, both gradients and updated adapter. Do not loosen it in
  response to the outcome. Two alternating pairs; 2 correctness, 5 warmup,
  10 measured and 1 final inspection steps per fresh arm (18 total).
- Root/checkpoint adapters must reload in separate processes with exact keys and
  tensor values, fixed eval input and unchanged BF16 output tolerance. Resume
  remains separate and unmeasured.
- No failed/incomplete arms enter scaling. Identical commit, GPU/server/model/
  data/config/precision in all arms; only world size and GAS differ. Retain raw
  durations, per-rank allocated/reserved VRAM and sampled host RAM; use existing
  tokens/s, speedup and efficiency definitions. Two pairs are budget-sized repeats,
  not evidence for a narrow confidence interval.

Commands after the same pinned environment/model preparation:

```bash
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 python benchmarks/harness/h3_fsdp_campaign.py \
  --prepared benchmarks/evidence/h3-fsdp-20261027/prepared \
  --output benchmarks/evidence/h3-fsdp-20261027/live/campaign \
  --repeats 2 --measured-steps 10 --timeout-seconds 600 --campaign-seconds 2400
```

Stop on the first correctness/reload failure, nonfinite state, OOM, missing rank,
dependency conflict, low disk or subprocess timeout. Continuously copy completed
artifacts, including pre-save numeric references and reload JSON, off the server.
The command-group deadline and rental watchdog are independent; collect and
delete only the newly created pilot instance, with final archival margin.

The existing $2 total ceiling includes the earlier estimated $1.0093. A refreshed
offer must fit a 75-minute maximum and $0.20 traffic/adjustment reserve within the
remaining ceiling. Existing instance/machine IDs are snapshotted read-only and
excluded from all mutations. No budget increase is assumed.

Hardware outcome: pending. H3 is not closed by a prepared protocol.
