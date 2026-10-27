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
- Explicitly enter CUDA BF16 autocast around forward/loss in both arms and the
  pre-save eval reference, leaving backward/optimizer outside. Accelerate's
  FSDP2 preparation bypasses its ordinary `prepare_model` AMP forward wrapper;
  a requested BF16 training flag does not prove actual autocast. Record and
  require enabled autocast inside every observed trainable module's forward.
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

Reproduce on an already-approved idle Linux server from the repository root.
Use the exact experiment commit recorded with the result, a fresh output path,
Python 3.10–3.12, CUDA 13 compatible driver and at least 15 GiB free disk:

```bash
python3 -m venv benchmarks/evidence/h3-fsdp-20261027/venv
source benchmarks/evidence/h3-fsdp-20261027/venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[train]' psutil -c benchmarks/h3-fsdp-constraints.txt
python -m pip check
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
python benchmarks/harness/h3_fsdp_run.py prepare \
  --prepared benchmarks/evidence/h3-fsdp-20261027/prepared \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
python benchmarks/harness/h3_fsdp_campaign.py \
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

The first follow-up at `c5ac8463` trained on both worlds but failed 2-GPU output
reload (max absolute error 0.484375). It preserved FP32 LoRA yet did not explicitly
establish the same autocast/eval contexts. This run is excluded from scaling.
The correction above implements the declared common autocast control; numeric
tolerances and matching/repeat rules remain unchanged.

Hardware outcome of the corrected protocol: pending.
