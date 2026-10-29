# H3 extra direction: two-stage pipeline smoke protocol

Prepared before paid execution, 2026-10-29. The user subsequently authorized
up to$1 for this pilot, within the cumulative$4 cap. Original proposal snapshots
below retain their historical pending status; they are not current authorization.
Selected direction: a bounded executable two-GPU Qwen2 + PEFT/LoRA pipeline
recipe associated with [upstream #39](https://github.com/MakazhanAlpamys/Soup/issues/39).
The general Soup CLI pipeline path remains wiring only; use the explicit harness
below. No 70B, DeepSeek recipe or four-stage implementation is claimed.

The [PyTorch pipeline API](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
supports manual `PipelineStage` construction and native GPipe schedules. Pin the
validated Torch 2.13.0 stack; the API is experimental. This experiment uses
manual Qwen2 block partitioning instead of requiring export tracing of PEFT.

## Predeclared configuration and admission

- Linux, one new server with exactly two equal RTX3090/4090 GPUs. Existing
  instances and machines are excluded. No NVLink requirement.
- Immutable Qwen/Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`; LoRA q/v r8 alpha16 dropout0.
- IEEE FP32 base/adapters/gradients, eager attention, highest matmul precision,
  TF32/AMP/compile/quantization disabled. No BF16 backend comparison is added.
- Fixed synthetic random tokens, seed314159, seq128; train/init seed42.
  Pydantic `SoupConfig` is materialized in `prepared/config.json`, with data and
  immutable base file hashes in `prepared.json`. No quality claim.
- Two equal layer partitions. Stage0 owns embedding plus decoder0..11; stage1
  decoder12..23 plus final norm/head. Frozen tied embedding/head has a copy on
  each stage; trainable adapter ownership must be disjoint and complete.
- Schedule batch4, two microbatches of2, two schedule calls per optimizer step:
  **global batch8**, not multiplied by two stages. Four optimizer steps.
- Native `ScheduleGPipe(scale_grads=False)`; each causal mean micro-loss is
  divided by microbatches × accumulation exactly once. Automatic schedule
  gradient rescaling must not repeatedly scale already-accumulated gradients.
- AdamW lr0.0002, betas0.9/0.999, eps1e-8, weight_decay0, global clip norm1.
  Stage gradient norms are combined across ranks for global clipping.
- A fresh ordinary full-model control runs on each rank before partitioning,
  using identical initial adapters/data and microbatch order. Afterward adapters
  are reset before the pipeline run. Compare every step's unclipped gradients
  and updated adapter values against that control at **atol1e-5, rtol0.02**.
  Compare mean loss at atol=rtol1e-5. Do not widen thresholds after a failure.
- Require finite loss/gradients, some nonzero stage gradients, actual adapter
  updates and unchanged frozen stage weights. Zero initial LoRA A updates are
  allowed. Verify actual stage ownership, microbatch/accumulation and sample IDs.
- Merge rank adapter shards with strict disjoint/exact key checks. Evaluate a
  fixed1x32 input through both actual stage modules, passing activations by P2P.
  Save that reference before serialization. A separate process reloads the
  merged adapter over the same base; eval logits must match at **atol=rtol1e-5**.
  The tolerance covers ordinary FP32 implementation/roundoff differences;
  storage/key identity is checked separately. No resume claim.
- Retain both rank records, first/control error metrics, adapter shards/merged
  safetensors, eval input/reference, versions/source/config/data hashes, raw
  commands/logs and peak allocated/reserved VRAM. The wall time includes probes;
  it is diagnostic and must not be called training throughput or scaling.

## Commands

Local offline preparation and real two-process CPU regression:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python -m pytest \
  tests/test_h3_pipeline_smoke.py --no-cov -q --tb=short
python -m ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

The CPU test uses a random four-layer Qwen2 fixture. It exercises native
pipeline P2P/autograd, two accumulated schedule calls, parameter updates and a
fresh reload. It is not pretrained-model or CUDA evidence.
The expanded H3/FSDP/pipeline selection passed350 tests; the final standalone
pipeline selection passed13 (overlapping), full-scope Ruff and shell syntax
checks passed. [Local logs and complete fixture tensors](evidence/h3-pipeline-20261029/local/README.md)
record eight actual forwards/backwards per stage, zero control gradient/update
differences and zero reload output error. No GPU result is inferred from this.

After explicit budget approval, on the new two-GPU Linux server:

```bash
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python benchmarks/harness/h3_pipeline_smoke.py prepare --prepared /tmp/h3-pp/prepared
python benchmarks/harness/h3_pipeline_smoke.py launch \
  --prepared /tmp/h3-pp/prepared --output /tmp/h3-pp/repeat-1 --device cuda --steps 4
python benchmarks/harness/h3_pipeline_smoke.py reload \
  --prepared /tmp/h3-pp/prepared --output /tmp/h3-pp/repeat-1 --device cuda
# Fresh run and fresh reload, without overwriting repeat-1.
python benchmarks/harness/h3_pipeline_smoke.py launch \
  --prepared /tmp/h3-pp/prepared --output /tmp/h3-pp/repeat-2 --device cuda --steps 4
python benchmarks/harness/h3_pipeline_smoke.py reload \
  --prepared /tmp/h3-pp/prepared --output /tmp/h3-pp/repeat-2 --device cuda
```

Pin Python3.12, Torch2.13.0+cu130, Transformers5.17.0, PEFT0.20.0,
Accelerate1.14.0, Pydantic2 and Rich. Record actual versions and `pip check`.
Seal, download and hash-verify repeat1 before starting repeat2. Preserve failed
attempts for diagnosis, without admitting them as passing hardware results.
The [Linux runner](h3-pipeline-linux.sh) wraps this sequence with command logs,
timeouts, version checks and sealed per-repeat archives. A collector must
download/verify each archive and write `repeat-N.received.json` with its
`sha256` before the runner proceeds; receipts time out after180 seconds. Collect
the metadata archive and completed campaign log as well. Rental lifecycle and
the independent60-minute watchdog stay outside this script.

### Interpreter regression discovered before the first GPU training step

On the disposable runtime, a system `torchrun` entry point selected
`/usr/bin/python3` even though the runner's active venv contained the pinned
dependencies. Both workers failed to import Rich before loading the model.
That attempt is excluded. The `launch` action now uses the active
`sys.executable -m torch.distributed.run`; it does not rely on a console script
being installed inside a venv that inherits system Torch.

A real-subprocess regression shadows `torchrun` on PATH and supplies a probe
Torch module, verifying that the active interpreter and two-rank arguments are
used. All14 pipeline tests and full-scope Ruff pass after this correction. The
training algorithm, inputs and numerical tolerances are unchanged. The failed
capture is retained separately; a new clean source revision and output directory
are required for the subsequent attempt. The original rental deadline remains.

## Original budget proposal and stop rules

The following is the pre-rental proposal. Subsequent authorization, failed
preparation, comparable replacement, measured results and final cost/cleanup
are recorded in the [result report](h3-pipeline-result-20261029.md).

A read-only Vast.ai search found offer55801599, machine48612, North Macedonia,
2xRTX3090, Threadripper3960X, 64GB allocated RAM, reliability0.99018. Quoted
compute is $0.53556/hour; 30GB storage adds approximately $0.00833/hour.
The offer is transient and must be rechecked immediately before an approved
rental. A comparable replacement must fit the same explicit ceiling.

Request **up to $1 additional spending**, at most60 minutes including setup,
execution and collection. Expected compute/storage is at most about$0.55;
transfer costs and a reserve fit inside $1. This is a new proposed allowance,
not an authorization inferred from the earlier $4 cap. Refresh task-owned prior
charges before giving the user a cumulative estimate. Do not access colleagues'
instances or use them as replacements.
The refreshed provider snapshot still reports$2.348 for the seven prior task
rentals (no further pages). Keeping the earlier conservative$2.60 upper estimate
and reserving$1 for this pilot gives **$3.60 cumulative**, within the prior$4 cap.
At proposal time no rental had started; the user then explicitly approved the
limit. [Original offer/charge arithmetic](evidence/h3-pipeline-20261029/local/budget-proposal.json)
is a historical snapshot, not a final invoice. A replacement offer must remain below
$0.60/hour including30GB storage and avoid every existing machine.

- Preflight a4MiB binary upload/download with SHA before training. Abort a
  stalled transfer; do not spend the run waiting for artifact collection.
- Check both CUDA devices, NCCL P2P and enough disk/RAM before model download.
- Stop on first failed numerical/ownership/loading gate, timeout or OOM. Keep
  diagnostic logs; do not spend time widening tolerances or switching models.
- Limit bootstrap to15 minutes, each train/reload command to10 minutes, total
  rental to60 minutes with an independent watchdog. No unapproved extension.
- Download/hash-verify all artifacts before deleting the exact newly created
  ID+label. On failure, prioritize records/config/logs/adapter and eval references.
- Delete only this task's new rental and verify API absence. Final acceptance
  stays pending if required artifacts are lost.
