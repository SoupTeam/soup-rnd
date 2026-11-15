# D2: prepared Liger/Unsloth and complete training-step comparison

Status on 2026-10-14: **GPU CORRECTNESS COLLECTED / EXTERNAL BASELINE GATES FAILED**.
The authorized RTX 3090 run executed all four stacks in FP16 and BF16.
Deterministic Soup/PEFT comparisons passed; Liger/Unsloth comparisons failed the
additional full-model criterion. No accepted full-step performance comparison was
collected. See [results and limitations](d2-training-baselines-20261014.md).

## Question and scope

Compare four **complete LoRA training stacks** in separate Python interpreters:

| Arm | Implementation |
| --- | --- |
| PEFT | Unpatched Transformers + PEFT reference |
| Soup | Same model with Soup single/QKV/MLP Fast-LoRA patches |
| Liger | PEFT + Liger RoPE, RMSNorm, SwiGLU and fused linear cross-entropy |
| Unsloth | `FastLanguageModel` + its optimized LoRA model paths |

Liger is not a replacement LoRA matrix kernel. Unsloth also changes attention,
normalization and loss implementations. These are comparisons of whole stacks;
their differences cannot be attributed solely to LoRA fusion. Do not combine
Liger and Unsloth in one process or import Unsloth into a PEFT reference process.
Official interfaces checked during preparation:
[Liger patch API](https://linkedin.github.io/Liger-Kernel/High-Level-APIs/),
[Unsloth API](https://huggingface.co/docs/transformers/community_integrations/unsloth),
[Unsloth installation](https://unsloth.ai/docs/get-started/install/pip-install).

Main fixture: complete pretrained `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`, dense FP16 and BF16 bases, FP32
adapters, rank 16 / alpha 32, all seven projection types, nonzero B, dropout 0,
batch 1, 512 fixed synthetic tokens, cache and checkpointing disabled.
The same frozen base tensors, adapter tensors and tokens are verified across
arms. Tokenization, packing, quantization, dataset loading and network transfers
are outside the experiment. No independent re-quantization is compared.

This measures a complete **1.1B-model LoRA training step**, including attention,
loss and optimizer. It does not establish 8B/NF4 full-model performance, real-data
quality, convergence, a long training run or Soup CLI/dataloader throughput.
The previous 8B-shaped projection and v7 NF4 evidence retain their own scope.
Team review must confirm whether this full-model comparison satisfies the D2
acceptance scope; a specifically required full 8B/NF4 run needs a separate matrix
and estimate, not a relabelled 1.1B result.

## Frozen numerical and execution gates

The 2026-10-14 loader preflight found that Unsloth rewrote the embedding tensor.
Its constructor output failed the byte-identity check before any Unsloth training.
The loader now restores any altered frozen parameter from the same pinned
checkpoint, verifies the original SHA-256 exactly and records the adjustment.
No changed input is accepted through a tolerance. Both precision correctness
matrices are collected before admitting any timing; a numerical failure blocks
the performance stage. The original failed loader and numerical pilot remain
retained, with their own source hashes. Seed/data/optimizer/scaler/criteria stay fixed.

Same-source fresh-process PEFT repeats then failed the full-model criterion too
(FP16 1/50 matching losses, BF16 8/50). Cross-library failures from that regime
must not be attributed solely to the library. A separately recorded execution
control uses `--deterministic`: CUBLAS_WORKSPACE_CONFIG=:4096:8 before imports,
and `torch.use_deterministic_algorithms(True)`. It preserves all original inputs,
optimizer/scaler and thresholds, and first checks reference repeatability.
It does not claim that arbitrary third-party custom kernels become deterministic.
Use `--gates-only` to complete numerical validation without launching the 72-arm
timing campaign. Performance remains unmeasured in that staged run, even if all
numerical gates pass; provider time authorization is checked separately.

1. Rerun the original seed-792 strict50 FP16/BF16 probe in the resolved CUDA
   environment first. Its original fixture, arithmetic telemetry and `.3f`
   requirement are unchanged.
2. Run each new full-model arm for 50 actual AdamW updates. Settings: lr 0.003,
   betas (0.9, 0.999), eps 1e-8, weight decay 0, foreach/fused False. FP16 uses
   the same GradScaler (initial scale 128, growth interval 2000); BF16 disables
   scaling. Scaler work is included in the step. Skipped updates fail validation.
3. Every full-model loss must be finite and equal to the matching PEFT step at
   `.3f`. Save every loss plus all adapter weights and unscaled gradients after
   steps 1 and 50. Each saved tensor must be finite, with identical shape/dtype
   and relative L2 error <= 1% versus PEFT (zero-reference denominator floor
   1e-12). This additional full-model criterion was declared before a GPU run;
   it does not replace or relax the original v7 float64/NF4/strict50 criteria.
4. Refuse missing adapter parameters, gradients, first/final snapshots, altered
   fixtures, differing common package versions and changed frozen bases.
5. Verify Soup custom autograd at every declared projection during preflight;
   verify Liger patched MLP/norm and executed fused-loss autograd; verify Unsloth
   patched QKV/O/MLP and actual `LoRA_QKV`, `LoRA_W`, `LoRA_MLP` autograd events.
   Function signatures alone do not count as execution proof. Profiler/hooks are
   removed before timing. No package-missing fallback is allowed.
6. A failed gate stops performance collection. Preserve failures and diagnostics;
   do not tune the seed, tokens, optimizer, scaler or threshold after failures.

All arms use the same source and interpreter stack. The requested Transformers
attention is SDPA; Unsloth's own attention path is part of its stack and must be
reported rather than called an identical attention implementation. TF32 is off.

## Timing and evidence

The worker performs >=20 warmup updates and >=2 seconds of warmup, then restores
the frozen adapters and resets AdamW/GradScaler. Warmup never advances the
measured trajectory. Every measurement starts at the same step zero. Each worker
owns one model and exits before the next worker starts.

Measure 50 steps per arm with synchronized CUDA events and host wall time.
Each region includes `zero_grad(set_to_none=True)`, forward/loss, backward,
scaler work when enabled and `AdamW.step()`. Wall time is the primary full-step
comparison; CUDA event latency is retained separately. Loading, compilation,
preloaded token transfer, profiler, CPU tensor copies and validation are outside
these timers. Tokens/sec uses **511 predicted tokens** per step divided by the
sum of measured step wall times; it is not overall job throughput. Initial
optimizer-state allocation is included in the first measured step for every arm.
Peak allocated/reserved memory is reset after warmup and retained separately.

For each dtype and candidate, collect three PEFT/candidate/candidate/PEFT cycles:
2 dtypes x 3 candidates x 3 cycles x 4 arms = **72 timing processes**, plus
8 full-model correctness processes. Report each cycle and median PEFT/candidate
ratio, first-step latency, steady-step distributions, tokens/sec and memory.
No minimum speedup is required. A regression or no gain is a valid outcome.

A CPU/NVML supervisor records empty pre-context/post-exit process lists, context
birth, a single persistent owner, child GPU UUID, clocks, power, pstate, throttle
and graphics peers. Host and container PIDs can differ: retain the child PID but
link the sole NVML owner through lifecycle and UUID, not assumed PID equality.
Require >=3 clock observations and <=5% spread within each arm and ABBA cycle.
Ambiguous owners, peers, missing telemetry, disallowed throttling, clock drift,
incomplete rows or replay failures make the run VOID/INCOMPLETE. Keep raw logs,
per-step rows, snapshots, telemetry, environment and fixture SHA-256 manifests.

## Prepare before renting

Use an isolated **Linux/Python 3.12** environment. Do not modify the working
Windows environment or install optional packages into an existing research run.
The candidate stack is pinned in `d2-training-baselines.in`; the archived generated lock
contains exact transitive versions and distribution hashes. Its metadata was
resolved locally for Linux x86_64 / Python 3.12. The original TRL 1.9.2 /
Datasets 4.3.0 conflict was resolved to Datasets 4.7.0 before any GPU run.

```bash
# Restore the exact recorded Linux lock from the evidence snapshot.
git show b2923981b7cb938852999459b19c968e5fbeb37f:benchmarks/d2-training-baselines.lock \
  > benchmarks/d2-training-baselines.lock

python3.12 -m venv /tmp/d2-baselines
source /tmp/d2-baselines/bin/activate
python -m pip install --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  -r benchmarks/d2-training-baselines.lock
python -m pip check
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /tmp/d2-fixtures /tmp/d2-results
python benchmarks/harness/d2_training_baselines.py prepare \
  --profile tinyllama-1.1b --dtype float16 --output /tmp/d2-fixtures/float16
python benchmarks/harness/d2_training_baselines.py prepare \
  --profile tinyllama-1.1b --dtype bfloat16 --output /tmp/d2-fixtures/bfloat16
```

Prepare/cache model and wheel downloads on unpaid CPU capacity when available.
Each dtype fixture contains a checkpoint copy; allow roughly 5 GB for fixtures
and use at least 40 GB disk for the environment, caches and evidence. Freeze the
actual source, including newly prepared files, in a checksummed archive. Do not
use a bare `git archive HEAD` while the new harness is still uncommitted. Record
the source SHA, installation logs, `pip freeze`, lock SHA and fixture manifests.
Do not upgrade packages automatically after a failure.

## Paid GPU run, only after new authorization

Candidate: one verified RTX 3090 24 GB, sm86; RTX 4090 is an alternative. Require
no peer workload, native BF16, accessible complete NVML telemetry and working
Triton. Select the host by total compute/storage/traffic cost and compatible
driver, not the displayed compute rate alone. Check `python -m xformers.info`
and preserve extension/ABI diagnostics. The first Liger/Unsloth gate must compile
and execute their real kernels; successful dependency resolution is insufficient.

```bash
python -m xformers.info > /tmp/d2-results/xformers-info.txt
python benchmarks/harness/fast_lora_probe.py loss --device cuda --dtype fp16 \
  --seed 792 --steps 50 --output-prefix /tmp/d2-results/original-strict50-fp16
python benchmarks/harness/fast_lora_probe.py loss --device cuda --dtype bf16 \
  --seed 792 --steps 50 --output-prefix /tmp/d2-results/original-strict50-bf16
python benchmarks/harness/d2_training_baselines.py run \
  --fixtures /tmp/d2-fixtures/float16 /tmp/d2-fixtures/bfloat16 \
  --steps 50 --warmup 20 --cycles 3 --seconds 4500 --deterministic --gates-only \
  --output /tmp/d2-results/comparison

# CPU replay of any measured arm against the frozen same-dtype PEFT gate:
python benchmarks/harness/d2_training_baselines.py compare \
  --reference /tmp/d2-results/comparison/float16-gate-peft \
  --candidate /tmp/d2-results/comparison/float16-soup-0-1-soup \
  --output /tmp/d2-results/replay-example.json
```

Execute sequentially with fail-on-error (`set -e` in a Bash controller); do not
continue after failed original strict50. A fresh run output directory is required.
All measured arms replay against their dtype's PEFT gate before aggregation.
Keep independent CPU replay results and review telemetry before publishing ratios.

`--seconds` bounds benchmark processes, **not provider billing**. At launch,
install both an external local and provider-side destruction guard for the exact
task instance. The rental deadline includes installation, preflight, downloads,
collection and evidence transfer. Reserve at least 10 minutes for download and
destroy; always verify provider destruction with the API, including failures.
Do not stop unrelated team instances or reuse an expired rental authorization.
Reduce `--seconds` further if installation used more of the rental window:
use the lesser of 4500 seconds and the remaining authorized rental time minus
600 seconds reserved for transfer/destruction. Do not extend the rental to
finish the matrix automatically.

## Local preparation validation

The final captured Windows CPU invocation passed **59 tests**, including actual
CPU training integration, independent numerical replay, failure-witness replay
and the existing timing/audit/materializer cases:

```bash
python -m pytest tests/test_d2_training_baselines.py \
  tests/test_d2_audit_training.py tests/test_d2_baseline_witnesses.py \
  tests/test_d2_isolated_timing.py tests/test_d2_timing_audit.py \
  tests/test_d2_timing_materialize.py --no-cov -q --tb=short
python -m ruff check src/soup_cli/ scripts/ tests/ benchmarks/harness/
```

Ruff is clean. The text-only CPU integration fixture disables availability probes
for unused sklearn/audio/vision extensions; it executes actual Llama, PEFT,
Soup autograd, AdamW updates and saved-tensor replay without mocking arithmetic.
The GPU worker has no such import overrides. These CPU tests alone do not
constitute GPU validation of Liger/Unsloth; the separate GPU evidence and its
failed full-model gates are reported with their scope.

## Estimate and remaining acceptance

Read-only offer search on 2026-10-13, 40 GB disk: RTX 3090 examples around
$0.158/hour and $0.178/hour total compute/storage; the latter advertised about
$0.00260/GB download and $0.00391/GB upload. These offers change and are not
reservations. Prefer low traffic charges when installing CUDA wheels. The
previous isolated projection run consumed about $0.15, but full-model/baseline
setup is a new workload and that cost is not extrapolated as a guarantee.

| Work in the rental window | Planning allowance |
| --- | --- |
| Provision, install cached pinned stack, ABI/Triton/BF16 checks | 15–30 min |
| Original strict50 and new full-model correctness gates | 10–20 min |
| 72 timed fresh processes, imports/model loads, ABBA repeats | 30–50 min |
| Transfer evidence and verify instance destruction | 10–15 min |

Allow **1–2 GPU-hours**, expected **$0.40–0.70 total**, with a proposed hard
authorization ceiling of **2 hours / $1** including traffic/storage. This is a
planning estimate, not permission to rent and not a promise that every numerical
or compatibility gate will pass. Stop on a blocker, retain evidence, and repair
locally before requesting another run; do not spend the ceiling automatically.

If the gates pass, the GPU part of this declared comparison can be finished in
that window. Finishing D2 also requires final-head full CI, independent team
review of timing/scope and D2/T00/F1 sign-off. More GPU time cannot guarantee
those outcomes. Do not claim D2 Done from prepared scripts or passing CPU tests.
