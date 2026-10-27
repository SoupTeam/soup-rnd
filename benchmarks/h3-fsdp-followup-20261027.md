# H3 first-stage result: FSDP2 + LoRA, save/reload and measured 1→2 scaling

2026-10-27. **The bounded small-model FP32 first stage passed.** Real training,
updates, frozen-base preservation, data/batch checks, actual parameter shards,
strict first-step agreement and eight fresh-process adapter reloads passed.
On this server, **two GPUs are slower**: median 4216.4633 → 2075.1697 input
tokens/s; speedup(2) **0.492159**, efficiency(2)
**0.246079**. No acceleration is claimed.

This completes the selected first-stage FP32 experiment, not all H3. BF16's
strict first-step comparison remains unresolved; resume, 4/8 GPUs, 70B, quality
and the wider recipe/backend matrix remain unmeasured. Failed BF16 arms are
preserved separately and never pooled with FP32 measurements.

Raw [evidence index](evidence/h3-fsdp-20261027/hardware/README.md),
[summary JSON](evidence/h3-fsdp-20261027/hardware/campaign-fp32/summary.json),
[artifact manifest](evidence/h3-fsdp-20261027/hardware/artifact-manifest.json).
All pre-save numeric references survived this run. The earlier pilot's retention
gap and failed checks remain documented in [the initial audit](h3-fsdp-lora-20261026.md).

## Configuration and admission

Measured code: `0c7fddf720e39710e921e59e47d93f91128d99ee`, clean checkout in every arm. The later report/evidence
commits do not alter the measured source. Actual implementation: FSDP2/DTensor
on two GPUs; the one-GPU control holds the full model and has zero FSDP units.

| Control | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Immutable revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Precision | Full FP32 base, LoRA, gradients and reduction; no AMP; IEEE `highest` matmul, no TF32 |
| Parameters | 494,573,440 unique total; 540,672 trainable, 96 LoRA tensors |
| LoRA | r=8, alpha=16, q_proj/v_proj, dropout=0 |
| Data | Same 512 seeded synthetic token rows, seq512, labels=input_ids, attention mask all ones |
| Global batch | 8 sequences = 4096 input tokens/optimizer step |
| One GPU | micro1, GAS8, GPU0 only |
| Two GPUs | micro1/rank, GAS4, GPU0+GPU1 |
| Optimizer | AdamW torch, lr0.0002, weight_decay0, constant scheduler, clip norm1 |
| Seeds | train/data42; token generator314159 |
| Steps/arm | 2 correctness + 5 warmup + 10 timed + 1 final inspection = 18 |
| Repeats/order | Two fresh pairs: 1/2 then 2/1; same base/init/data and commit |
| Disabled | compile, quantization, streaming, packing, gradient checkpointing |

Base files and the full prepared dataset are hashed and verified before training.
Prepared data SHA256: `9042a5e2cc6049064ea7ee35b00b4d5d65b2c000719df2443ba20a889295fcc9`.
Resolved SoupConfig is retained; benchmark precision/schedule controls and actual
dtypes are recorded separately. FP32 uses the existing SFT load-dtype/precision
resolution seams without changing production defaults, then the real
`SFTTrainerWrapper.setup()`/`.train()`/optimizer/checkpoint/final save paths.

Physical host: two NVIDIA GeForce RTX 3090, driver 580.167.08, CUDA 13.0,
NCCL 2.29.7; topology `NODE`, no NVLink.
CPU: AMD EPYC 7742 64-Core Processor. The provider listed 64 GB allocated RAM and 60 GB disk; system-wide
RAM samples below describe the host, not a unique allocation to the job.
Python 3.12.3, Torch 2.13.0+cu130, Transformers
5.17.0, Accelerate 1.14.0,
PEFT 0.20.0, TRL 0.29.1.
Full inventory/topology, package freeze and commands are in the raw files.

## Correctness and sharding

Every recorded microstep loss is finite. All LoRA gradients were checked at
optimizer steps 1, 2 and 18: finite, some nonzero, no frozen-base gradients;
reconstructed reduced-gradient hashes agree across ranks. First-step B updates
are required, while unchanged A at initialization is allowed. At least one
adapter tensor updates and all final adapter tensors are finite. Each rank's
frozen-base local hash is identical before/after training in all four arms.

Every optimizer step checks actual microbatch shape/count and accumulation.
The first two global windows each contain exactly eight distinct sample hashes,
without cross-rank duplicates, matching across worlds and repeats. No rounded
accumulation or inferred effective batch is used.

Both ranks have **27 wrapped FSDP modules** (the tied input/output modules share
one group). Actual `DTensor.to_local().numel()` readings are **247,286,720
elements/rank**, sum 494,573,440; no extra tied allocation. Each named global
shape, actual local shard size, dtype and placement is retained and checked.
Global numel divided by world size is not used as a measurement. Local shards
themselves are not expected to equal each other.

All first-step gradient and updated-adapter tensors pass the unchanged
`atol=1e-5, rtol=0.02` comparison against one GPU. Maximum absolute differences
over all repeats: gradients **1.49011611938e-08**, adapter **1.27030943986e-07**.
All 96 trainable parameters are observed FP32 *during forward*, with autocast
disabled. `highest` matmul and all-FP32 parameter storage are asserted; a master
dtype or requested training flag alone cannot pass admission.

## Adapter save/reload

Every arm saves the final adapter and checkpoint-18. Each is reloaded in its own
fresh Python process over the same immutable base loaded FP32. All eight checks
verify exact keys and tensor values against the pre-save adapter snapshot, then
compare a fixed 1×64-token eval input at predeclared `atol=rtol=1e-5`.
**All eight maximum absolute logit errors are 0.0.** There are 96 adapter
tensors per save. Training/reload PIDs, byte inventories, logs and the numeric
references are retained. Offline revalidation also independently matches every
root/checkpoint adapter against its reference.

Optimizer/scheduler/RNG files are inventoried, but were not restored and training
was not resumed. Adapter reload is not evidence of resume or closure of #352.

## Training throughput and memory

| Arm | GPUs | Timed optimizer steps | Global input tokens/s | Step seconds |
|---|---:|---:|---:|---:|
| repeat-1-gpu-1 | 1 | 10 | 4276.8180 | 0.957721 |
| repeat-1-gpu-2 | 2 | 10 | 2070.6530 | 1.978120 |
| repeat-2-gpu-2 | 2 | 10 | 2079.6863 | 1.969528 |
| repeat-2-gpu-1 | 1 | 10 | 4156.1086 | 0.985537 |

Median throughput(1)=4216.463278; throughput(2)=2075.169674 input tokens/s.
Speedup(N)=throughput(N)/throughput(1); efficiency(N)=speedup(N)/N.
Thus speedup(2)=0.492159, efficiency(2)=0.246079.
Two pairs are budget-sized repeats; no narrow confidence interval is claimed.
These numbers apply to this small FP32/global8/PCIe experiment, not BF16 or 70B.

Input tokens/s = actual nonpadding input tokens summed across ranks divided by
the maximum rank sum of CUDA-synchronized timed optimizer-step durations.
Each sequence contributes 512 input tokens and 511 causal targets. Step seconds
uses the same denominator / 10. Timing is Observer `on_step_begin`→`on_step_end`:
forward/backward, synchronization, optimizer/scheduler and intervening callback
work. CPU batch fetch before the callback, setup/loading, heavy probes, warmup,
eval, checkpoint/final save and reload are excluded. No compilation is enabled.
This is training-loop compute throughput, not whole-job/input-pipeline throughput.

| Arm | Rank | Peak allocated MiB | Peak reserved MiB | Sampled RSS MiB | Sampled host used MiB |
|---|---:|---:|---:|---:|---:|
| repeat-1-gpu-1 | 0 | 4123.03 | 4350.00 | 1868.39 | 24373.07 |
| repeat-1-gpu-2 | 0 | 3679.88 | 4586.00 | 2141.23 | 27208.41 |
| repeat-1-gpu-2 | 1 | 3677.13 | 4586.00 | 2141.64 | 27208.41 |
| repeat-2-gpu-2 | 0 | 3677.13 | 4586.00 | 2157.47 | 27036.16 |
| repeat-2-gpu-2 | 1 | 3677.13 | 4586.00 | 2157.46 | 27036.16 |
| repeat-2-gpu-1 | 0 | 4123.03 | 4350.00 | 1870.48 | 24859.05 |

MiB=2^20 bytes; exact bytes and ten raw per-rank durations are retained in JSON.
CUDA peaks reset after warmup; the allocator cache remains warm. Prior correctness
gathers can affect reserved memory, so these are not minimal-VRAM estimates.
Parameter shards halve, but peak allocated falls only from about 4.323 GB to
3.859 GB/rank; reserved rises from 4.561 GB to 4.809 GB. Do not equate sharding
with a halving of total VRAM. RSS/host used RAM are sampled every 50 ms; spikes
can be missed and summed RSS can double-count shared pages. Host used RAM includes
other system state. No NCCL profiling or broader bottleneck attribution was done.

## BF16 controls and scope

The earlier #373 probe repair is unchanged; see the initial issue audit for
reference ownership, #352/#373/#39–#41, existing fixes and unknown H1 card.
The production change jointly places frozen tied embeddings/head in one public
FSDP2 group to avoid the reproduced first-forward shared-parameter rejection.
Untied/trainable embedding and FSDP1 routes are no-ops; no backend migration.

The initial BF16 comparison failed. Follow-up `c5ac8463` preserved FP32 LoRA
materialization but did not enforce matching actual AMP/eval contexts; its 2-GPU
reload failed. Accelerate's FSDP2 preparation bypasses the ordinary `prepare_model`
AMP wrapper. Explicit forward/loss/eval autocast at `c6bd159e` made both-world
reloads pass, but the strict first-step gate still failed. Those failures remain
preserved and excluded. Cause of the remaining BF16 per-element disagreement is
not claimed resolved. FP32 was declared as a separate control *before* its timing,
with the same strict threshold; no tolerance was loosened or failed arm relabelled.

| Acceptance criterion | Status |
|---|---|
| Real FSDP2 + LoRA forward/backward/update, frozen base, actual batch/data, rank synchronization | measured, FP32 first stage passed |
| Actual local shards on every rank | measured via DTensor API |
| Root/checkpoint adapter keys/values and separate-process output parity | measured; all eight passed, references retained |
| Same-server 1→2 throughput, warmup separation, repeats, per-rank VRAM and host RAM | measured; negative scaling |
| BF16 strict cross-world first-step control | failed/unresolved; BF16 scaling not accepted |
| Optimizer/scheduler resume | unmeasured |
| 4/8 GPU, 70B, full #41 recipes, multi-node/pipeline, quality | unmeasured |
| H1 authoritative card/dependency completion | unknown; pinned environment/pip check measured independently |

## Reproduce and stop

Use the measured commit above on an authorized idle two-GPU Linux host, Python
3.10–3.12 and CUDA 13 compatible driver, at least 32 GB RAM/60 GB disk, >15 GiB
free before arms. Source stays clean; use fresh prepared/output directories.

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
  --output benchmarks/evidence/h3-fsdp-20261027/live/campaign-fp32 \
  --precision fp32 --repeats 2 --measured-steps 10 \
  --timeout-seconds 600 --campaign-seconds 2400
```

The campaign emits all individual train/reload commands and exits. It stops on
the first failure, low disk, nonfinite/update/base/data/shard/reload assertion,
missing rank or timeout; its process-group timeout kills torchrun workers.
The first pair must pass the strict numerical gate before later pairs run.
No silent retries/upgrades. A separate rental watchdog caps lifetime; continuously
copy results, verify complete archives, then destroy only the new instance/disk.

CPU-only report replay from the committed GPU readings/tensor snapshots:

```bash
python benchmarks/harness/h3_fsdp_campaign.py --summarize-only \
  --prepared benchmarks/evidence/h3-fsdp-20261027/hardware/prepared \
  --output benchmarks/evidence/h3-fsdp-20261027/hardware/campaign-fp32
```

This rechecks admission/numerical comparisons and derives the table from real
GPU records; it does not execute new GPU training. Larger reload references and
optimizer files are in the verified external archive named in the manifest.

## Preservation, budget and local validation

Original D2 checkout remains on its original HEAD with all 129 dirty/untracked
snapshot files byte-identical; today's ZIP is outside Git. Existing rented
machines were untouched. A new inaccessible instance was deleted without training;
its cost and failed runs are not benchmark data. The replacement had a scoped
75-minute watchdog and remained inside the original **$2 total** ceiling.

All three new pilot instances across both days are deleted; the final archive
was SHA256-verified with all four complete arms and numeric references before
the last deletion, then the API confirmed absence. Conservative cumulative
compute/storage estimate **$1.3930**; with $0.20 traffic/adjustment reserve
**$1.5930**, below $2. This is not a provider invoice. Raw ledger and
deletion confirmations are under hardware/support.

**334 selected pytest passed**, 15 warnings; ruff clean. The selection covers H3,
existing #373/#352/#350, distributed device-map, #1204, multi-GPU and changelog
checks. The CPU tiny-model fixture separately trained/saved/reloaded; its results
are not hardware proof. Full repository pytest was not run; hosted CI was not
monitored. Offline replay exactly matches the remote summary, source hashes match
the measured Git commit, and every saved adapter is independently rechecked.

This delivers the selected first stage. The PR stays draft as requested; it does
not claim all H3 or upstream #352/#373/#41 closed.
