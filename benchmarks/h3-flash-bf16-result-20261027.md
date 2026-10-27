# H3 native Flash BF16: reproducible configuration and measured 1→2 scaling

2026-10-27. **Native CUDA Flash BF16 passes the original strict validation when
deterministic algorithms are enabled.** Four fresh training arms, actual Flash
forward/backward on every rank, actual parameter shards and eight separate-process
adapter reloads passed. All first-step gradient/adapter differences and all eight
eval-logit reload errors are **0.0**. This is FlashAttention with ordinary BF16
GEMM reduction enabled; no math-SDPA substitution or tolerance relaxation.

The default nondeterministic Flash case also trained and reloaded, but **failed
the unchanged strict cross-world first-step criterion**. Its two arms/four
reloads are preserved as diagnostics and excluded from scaling. The resolution
is the verified reproducible Flash configuration below. Nondeterministic defaults
are not relabelled as passing; this is not a claim that every Flash configuration
is numerically interchangeable.

Accepted median input throughput: **8178.1305 → 8102.4617 tokens/s**;
speedup(2) **0.990747**, efficiency(2)
**0.495374**. Scaling is approximately flat on this small
model/global batch; no acceleration is claimed.

[Pre-rental protocol](h3-flash-bf16-protocol-20261027.md),
[raw evidence](evidence/h3-flash-bf16-20261027/hardware/README.md),
[accepted original summary](evidence/h3-flash-bf16-20261027/hardware/campaign-deterministic/summary.json),
[full archive and public subset manifests](evidence/h3-flash-bf16-20261027/hardware/artifact-manifest.json).
The [FP32](h3-fsdp-followup-20261027.md) and
[conservative math BF16](h3-bf16-result-20261027.md) experiments retain their own
results. Their timings are not pooled with this experiment.

## Same model, data, software and arithmetic

Measured clean source: `81b9a56a40999bf1cd64cb0a6b3fdee78233a001` in both cases and every rank.
The real Soup SFT setup/train/optimizer/checkpoint/final-save paths run with
benchmark instrumentation and explicit attention/reproducibility controls.
Product defaults and Pydantic config fields were not changed by this follow-up.

| Control | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Immutable revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Parameters | 494,573,440 unique total; 540,672 trainable, 96 LoRA tensors |
| LoRA | q_proj/v_proj, r8, alpha16, dropout0 |
| Precision | BF16 base/actual AMP compute/output; FP32 LoRA masters, gradients and FSDP reduction |
| Attention | native PyTorch CUDA Flash only; math/efficient/cuDNN backends disabled |
| GEMM | ordinary BF16 reduced-precision reduction enabled; highest FP32 matmul |
| Data | same 512 synthetic random-token rows, seq512, labels=input_ids, all-one mask |
| Data SHA256 | `9042a5e2cc6049064ea7ee35b00b4d5d65b2c000719df2443ba20a889295fcc9` |
| Batch | global8, micro1/rank; GAS8 on one GPU, GAS4 on two |
| Optimizer | AdamW torch, lr0.0002, betas(0.9,0.999), eps1e-8, weight_decay0, clip1; constant scheduler |
| Seeds | train/data42; synthetic token generator314159 |
| Steps/arm | 2 correctness + 5 warmup + 10 timed + 1 final inspection = 18 |
| Accepted repeats | two fresh pairs, 1/2 then 2/1, sequential |
| Disabled | compile, quantization, streaming, packing, gradient checkpointing |

Single physical Linux host: two NVIDIA GeForce RTX 4090, AMD EPYC 9354 32-Core Processor;
driver 580.178.04, CUDA 13.0, NCCL 2.29.7,
no NVLink. Provider allocation about 51.5 GB RAM/30 GB disk. Python
3.12.3, Torch 2.13.0+cu130,
Transformers 5.17.0, Accelerate 1.14.0,
PEFT 0.20.0, TRL 0.29.1.
Full inventory, topology, package freezes and source hashes are retained.

## Actual Flash execution and reproducibility result

The first correctness optimizer step records CPU dispatcher events for actual
CUDA-specific `aten::_scaled_dot_product_flash_attention` and
`aten::_scaled_dot_product_flash_attention_backward`: **192 forward + 192
backward calls on one GPU, 96 + 96 per rank on two GPUs** (24 layers × actual
accumulation). Every independent reload records 24 Flash forwards. Required
operators must be present, and math/efficient/cuDNN/CPU fallback is rejected.
Backend-enable flags alone cannot pass. Profiling ends during the first
correctness step, before warmup and measured training.

Actual forward hooks observe 96 FP32 LoRA parameters, CUDA autocast enabled,
96 BF16 LoRA outputs and the expected backend flags. Offline checks verify all
initial adapters/data windows equal and resolved Soup configs equal between
cases at each world size. The only observed backend-control differences are:

| Setting/result | Default Flash diagnostic | Accepted reproducible Flash |
|---|---|---|
| Deterministic algorithms | false | true |
| CUBLAS workspace | unset | `:4096:8` |
| BF16 GEMM reduced-precision reduction | true | true |
| Executed attention | native Flash forward/backward | native Flash forward/backward |
| Strict first-step gradients/adapter | failed | passed, maximum errors 0.0/0.0 |
| Final/checkpoint reloads | four passed, output error 0.0 | eight passed, output error 0.0 |
| Scaling admission | excluded | four accepted arms |

Default first-gradient maximum absolute difference is
**0.000199826667085**;
updated-adapter difference **0.000399882730562**.
Global gradient relative L2 difference is
0.00232588796; there are
361 gradient sign changes.
Each arm's own saved gradients replay its update with CPU AdamW to maximum
absolute error 4.3656e-11. That diagnostic supports locating the discrepancy
before the optimizer; it does not establish a unique kernel-level cause.

PyTorch 2.13's [native backward implementation](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L112-L122)
selects deterministic Flash backward from the public deterministic-algorithm
setting. That setting also affects other CUDA operations and the CUBLAS workspace
changes here. Therefore this result establishes a working combined reproducibility
configuration, rather than attributing the entire mismatch exclusively to Flash
backward. No PyTorch kernel was patched and no optimizer defect is claimed.

## Training, rank synchronization, shards and saves

Every microstep loss is finite; gradients inspected at steps 1, 2 and 18 are
finite and some are nonzero. No frozen-base gradients appear, and every rank's
local frozen-base hash remains unchanged. First-step B updates change 48 tensors;
zero A at initialization is allowed. Final updates change all 96 tensors.
Actual microbatch counts/shape and accumulation are checked on every step.
The first two global data windows each contain eight distinct hashes without
rank duplication, equal across worlds, repeats and both cases. Reconstructed
adapter/reduced-gradient hashes agree across ranks, including preclip gradients.

Both distributed ranks have 27 FSDP2 units. Actual DTensor `to_local().numel()`
readings are **247,286,720 parameter elements per rank**, reconstructing
494,573,440 without a duplicate tied embedding. Each parameter's local size,
global shape, dtype and placement is retained. The one-GPU control holds the
full unique total and has zero FSDP units. Global numel/world_size is not used
as a physical measurement, and different local shards need not equal each other.

The original cross-world per-element threshold remains **atol=1e-5, rtol=0.02**.
All accepted first-step tensors match exactly across all four arms. Each final
adapter and checkpoint-18 reloads in a new process over the same immutable BF16
base, fixed 1×64-token input, eval mode, same Flash/reproducibility settings.
Exact 96 keys/tensor values and output **atol=1/64, rtol=1/128** pass; all eight
accepted maximum logit errors are zero. The predeclared output tolerance is
two absolute BF16 ULPs near one plus one relative ULP. Offline validation also
matches all twelve default/control saves to their retained pre-save references
and checks distinct reload/training PIDs. No optimizer/scheduler state is restored;
adapter reload is not resume evidence.

## Measured throughput and memory: deterministic Flash only

| Arm | GPUs | Timed steps | Input tokens/s | Step seconds |
|---|---:|---:|---:|---:|
| repeat-1-gpu-1 | 1 | 10 | 8235.8173 | 0.497340 |
| repeat-1-gpu-2 | 2 | 10 | 8139.6881 | 0.503213 |
| repeat-2-gpu-2 | 2 | 10 | 8065.2354 | 0.507859 |
| repeat-2-gpu-1 | 1 | 10 | 8120.4437 | 0.504406 |

Median throughput(1)=8178.130491; throughput(2)=8102.461741 input tokens/s.
Speedup(N)=throughput(N)/throughput(1), efficiency(N)=speedup(N)/N.
Two budget-sized pairs do not establish a narrow confidence interval; this
small difference should be treated as approximately flat scaling. Failed
default timings are not admitted or mixed into these medians.

Input tokens/s counts all actual nonpadding input tokens across ranks divided
by the slowest rank's sum of CUDA-synchronized optimizer-step durations. Each
optimizer step processes 4096 input tokens; each 512-token sequence has 511
causal loss targets. Step seconds uses that same duration divided by ten.
The measured callback window on_step_begin→on_step_end includes forward/backward,
rank synchronization, optimizer/scheduler and intervening callback work. It
excludes CPU batch fetch before the callback, loading, profiling/correctness
probes, five warmup steps, eval, checkpoint/final save, archive transfers and
reload. No compilation is enabled. This is training-loop compute throughput,
not end-to-end job/input-pipeline throughput.

| Arm | Rank | Peak allocated MiB | Peak reserved MiB | Sampled RSS MiB | Sampled host used MiB |
|---|---:|---:|---:|---:|---:|
| repeat-1-gpu-1 | 0 | 2437.11 | 2654.00 | 2661.19 | 21682.61 |
| repeat-1-gpu-2 | 0 | 2211.49 | 2682.00 | 2771.20 | 24468.08 |
| repeat-1-gpu-2 | 1 | 2211.49 | 2684.00 | 2770.57 | 24468.08 |
| repeat-2-gpu-2 | 0 | 2211.49 | 2682.00 | 2771.05 | 24895.65 |
| repeat-2-gpu-2 | 1 | 2211.49 | 2682.00 | 2770.96 | 24895.65 |
| repeat-2-gpu-1 | 0 | 2437.11 | 2654.00 | 2660.60 | 21818.59 |

MiB=2^20 bytes. CUDA peaks reset after warmup with a warm allocator; previous
probes can affect reserved memory, so this is not minimal-VRAM sizing. Exact
bytes and ten raw durations/rank are retained. RSS/host used RAM are sampled
every 50 ms and can miss spikes; summed RSS may double-count shared pages.
Host used RAM covers the system, not only this container. Parameter sharding
does not imply halving total VRAM. No bottleneck attribution is claimed.

## Retention, acceptance and remaining scope

A 4 MiB random binary transfer passed byte-count/SHA256 verification before
training. Every arm's references/adapters/logs were sealed, downloaded and
hash-verified before its receipt released the next arm; no bulk transfer
overlapped measured training. Eight complete local archives retain all six
arms, both metadata/data captures and every numeric reference. Their exact
paths, byte counts, archive/member SHA256 are in the manifest. Public evidence
includes both cases' raw commands/configs/versions, nine physical rank records,
all first-step/init/preclip tensors and twelve saved adapters. Larger eval
references and complete Arrow data remain in the full archives.

| Criterion | Status |
|---|---|
| Actual native CUDA Flash forward/backward, ordinary BF16 compute/reduction | measured, passed in both cases |
| Strict one/two-GPU first-step agreement | passed with deterministic algorithms; nondeterministic defaults failed |
| Update/frozen base, actual batch/data, synchronized ranks, physical shards | measured, passed for accepted case |
| Separate-process final/checkpoint reload | eight accepted checks passed with zero output error; references retained |
| Same-host 1→2 training throughput and per-rank VRAM/host RAM | measured, approximately flat scaling |
| Optimizer/scheduler resume, 70B, 4/8 GPUs, multi-node/pipeline, quality | unmeasured |
| H1 authoritative card | unknown; pinned environment/pip check verified independently |

This completes the selected native Flash BF16 validation with an explicit
reproducible recipe. It does not claim strict agreement for nondeterministic
defaults, all attention backends or the entire H3 matrix. The original #352/
#373/#39–#41 audit remains in the initial report, and the existing #373 probe
repair is unchanged. The original D2 HEAD/status and 129 dirty/untracked files
remain byte-for-byte preserved. Hosted CI was not monitored.

## Reproduce and budget

Use the measured source and pinned stack on an authorized idle two-GPU Linux
host. Require clean `pip check` and the same immutable prepared model/data.
Captured bootstrap scripts show the disposable image setup; its existing Torch
is reused via a venv and its unrelated conflicting global `spin` utility is
removed with the actual global interpreter.

```bash
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
ROOT=benchmarks/evidence/h3-flash-bf16-20261027
python benchmarks/harness/h3_fsdp_run.py prepare --prepared "$ROOT/prepared" \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
python benchmarks/harness/h3_fsdp_campaign.py --prepared "$ROOT/prepared" \
  --output "$ROOT/live/campaign-deterministic" --precision bf16 \
  --bf16-flash --flash-deterministic --archive-arms \
  --repeats 2 --measured-steps 10 --timeout-seconds 600 --campaign-seconds 1800
```

Omit `--flash-deterministic` for the default negative-control case and use a
fresh output directory. The actual rental added `--archive-receipt-timeout 180`
with its external collector; omit that receipt option for archives retained
directly on your own host. Stop on failed correctness, transfer, environment,
foreign GPU processes, insufficient disk, command timeout or rental deadline.
Keep failures diagnostic. Never relabel their timing as accepted scaling.

Prior provider-reported charges were $2.142; the latest seven-owned-instance
snapshot reports **$2.348**. The frozen prior conservative upper $2.26 plus this
rental's compute/storage estimate $0.1899
and $0.15 reserve gives a cumulative upper estimate **$2.5999**,
below the user's **$4 cumulative cap**. Provider snapshots are not a fully
reconciled final invoice. The independent 45-minute watchdog was not extended;
the new instance was deleted after about ten minutes, after evidence verification.
All seven task-created instances are absent from the API. Three existing
instances were excluded before selection and were not mutated.

Local verification: **97 targeted pytest passed**, full repository Ruff scope
passed, and a tiny CPU train/save/reload smoke passed. Those CPU checks are
logic checks only. Independent offline analysis exactly reproduces the accepted
GPU summary, verifies source and dataset hashes, raw timings, all six accepted
rank records and all twelve saved adapters. Full repository pytest was not run.
