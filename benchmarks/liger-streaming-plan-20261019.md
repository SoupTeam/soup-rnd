# Liger inside Soup streaming: status and GPU plan (2026-10-19)

The [strict 7B follow-up](liger-streaming-7b-20261019.md) supersedes the 7B and
performance status below: all three Liger choices fail the unchanged tensor
tolerances against a bitwise-repeatable ordinary streaming control. The historical
plan and earlier local checks remain here for provenance.

The original plan below is retained with its historical local-validation record.
The 2026-10-19 A40 follow-up passed the unchanged 24-case CUDA matrix and four
short pretrained 7B save/reload pilots after the precision corrections described
in [the GPU report](liger-streaming-gpu-20261019.md). It establishes no speed or
quality improvement.
Base: published PR #5 head `859b2df33453e33b8546dabb97e4a878fbed1237`, checked live
before creating the isolated `koko/d2-liger-streaming` worktree. The subsequent
user-authorized Vast.ai run had a USD 2 cumulative ceiling and verified instance
deletion. No push, merge or changes to the NoRA worktrees were made.

## What is connected

The actual `SFTTrainerWrapper` streaming setup installs selected operation
forwards on the existing model instances before its VRAM probe. It does not
call Liger's global model patch or set TRL/HF `use_liger_kernel`, so the streamed
head, logits and loss path remain the Soup/HF ones. The default is RMSNorm only;
SwiGLU is independently selectable. Neither choice implies numerical qualification.

| Model / source / weights | CPU integration with substitute ops | Actual CUDA Liger | Performance |
| --- | --- | --- | --- |
| Tiny local Llama / RAM / dense | Pass | FP16/BF16 pass on A40 | Unmeasured |
| Tiny local Llama / disk / dense | Pass | FP16/BF16 pass on A40 | Unmeasured |
| Tiny local Llama / RAM / NF4 | Pass | FP16/BF16 pass on A40 | Unmeasured |
| Tiny local Llama / disk / NF4 | Pass | FP16/BF16 pass on A40 | Unmeasured |
| OpenLLaMA 7B v2 / RAM / dense | Not run | Four BF16 finite/save/reload pilots | Unmeasured |
| Real 7B / disk or NF4 | Not run | Not qualified by these pilots | Unmeasured |
| Other architectures, non-SFT, CPU/MPS Liger | Refused | Unsupported | No claim |
| Fast-LoRA + Liger | Refused | Unsupported | No claim |
| Liger RoPE, CE, fused-linear-CE in streaming | Not enabled / selector refused | Unsupported | No claim |

The CPU tests run four layers through two slots with non-unit, layer-specific
norm weights, nonzero LoRA A/B gradients, two optimizer updates, accumulation=2,
and periodic plus final PEFT saves. They compare logits, gradients and updated
adapters with ordinary streamed LoRA; canonical tensors reload on a resident
base with the public PEFT loader. The separate prediction round trip covers
dense/RAM FP32; loading NF4-trained adapter tensors on a dense base is not proof
of dense-versus-NF4 inference equivalence. This is useful integration evidence, not a
test of Triton, a GPU lifetime guarantee, a 7B result or a quality evaluation.

RMSNorm calls Liger with Llama casting and `in_place=False`, reading `self.weight`
inside `functional_call`. Saved tensors remain visible to checkpoint recomputation;
no closure retains a pooled base weight. SwiGLU keeps PEFT linears intact and
fuses only SiLU-times-up. Model parameter identities and state-dict names do not
change. Installation failure closes the disk-reader; instance patches do not
change newly constructed ordinary models.

Upstream APIs inspected:
[RMSNorm](https://github.com/linkedin/Liger-Kernel/blob/v0.8.1/src/liger_kernel/ops/rms_norm.py),
[SwiGLU](https://github.com/linkedin/Liger-Kernel/blob/v0.8.1/src/liger_kernel/ops/swiglu.py),
[model patching](https://github.com/linkedin/Liger-Kernel/blob/v0.8.1/src/liger_kernel/transformers/monkey_patch.py).
Their actual CUDA behavior still needs the gates below. The out-of-place RMSNorm
choice avoids a shared-gradient hazard; it does not establish a cause for a
third party's 7B NaN report. That report and the claim that RMSNorm alone worked
are not reproduced here. A diagnosis requires the exact model/revision, component
flags, seed/data, versions, dtype/scaler settings, and the first non-finite tensor.

## 1. Freeze the source and environment before paid time

Use Linux, one Ampere-or-newer NVIDIA device, CUDA-enabled PyTorch, working
Triton and Liger 0.8.1. An A40 48 GB gives more room for a 7B diagnosis;
a 3090 24 GB is a lower-memory candidate, subject to Soup's measured fit gate.
Request at least 48 GB host RAM and 100 GB free local NVMe for the 7B stage.
Actual RAM pinning, disk tier and free VRAM must be checked, not inferred from
the GPU name. Do not use a T4 to claim native BF16 qualification.

In the prepared source checkout (the archive must include uncommitted new files):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,liger]' 'liger-kernel==0.8.1'
mkdir -p output/liger-gpu
git rev-parse HEAD > output/liger-gpu/base-head.txt
git diff --binary HEAD > output/liger-gpu/tracked.patch
git ls-files -co --exclude-standard -z | tar --null -T - -czf output/liger-gpu/source.tgz
sha256sum output/liger-gpu/source.tgz > output/liger-gpu/source.sha256
python -m pip freeze > output/liger-gpu/pip-freeze.txt
python -m torch.utils.collect_env > output/liger-gpu/torch-env.txt
nvidia-smi -q > output/liger-gpu/nvidia-smi.txt
```

Resolve the model to an immutable local HF snapshot and the data to one frozen
local JSONL before measurements. Save the HF revision and SHA-256 manifest of
all snapshot files. Use a Llama 7B model you already have permission to download;
this plan does not assume access to a gated repository. Use the agreed real-data
study's frozen training subset when available; a substitute subset must be named
and hashed and cannot answer that study's quality question. Keep source archives
and raw results under ignored `output/`, out of the review diff.

## 2. Correctness first, all three component arms separately

```bash
export CUDA_VISIBLE_DEVICES=0
export SOUP_LIGER_STREAM_EVIDENCE="$PWD/output/liger-gpu/tensors"
python -m pytest tests/test_liger_streaming.py -m gpu --no-cov -v --tb=short \
  --junitxml=output/liger-gpu/correctness.xml \
  > output/liger-gpu/correctness.log 2>&1
```

Require **24 passed, zero skips**, not merely exit 0. These are
3 component sets × 2 precisions × 2 sources × 2 weight formats, each paired with
ordinary streamed PEFT LoRA. FP16 and BF16 override only the Soup dtype policy
to exercise both on one suitable card; GPU kernels/capability reports are real.
Predeclared gates: finite logits and gradients, identical initial adapters,
per-tensor gradient/updated-adapter rtol=1e-2 and atol=1e-5, logits with the same
tolerances, and loss equal at three decimal places. CPU tolerances are stricter
and do not replace these GPU gates. Do not loosen thresholds after a failure.

The tensor files are written per completed arm before cross-arm assertions, so a
parity failure retains its inputs/outputs. `ARM_CAPTURED_NOT_A_PARITY_VERDICT`
in JSON is not a pass; use the JUnit result and retain the nonzero exit code.
Earlier setup/NaN failures may produce only the failure log. Replay safetensors
on CPU to compare both arms independently; archive failed runs too.

If 7B produces NaN, rerun a small fixed batch through `baseline`, RMSNorm-only,
SwiGLU-only, then their union, in separate processes. Inspect the first differing
loss/logits/adapter gradient before timing; compare the dtype and norm-weight
mapping and enable anomaly detection only in this diagnostic arm. Test residual
branches with out-of-place RMSNorm. Exclude RoPE and fused loss in every arm.
Do not label a partial component pass as support for the full combination.

## 3. Real Soup 7B pilot and timing after qualification

Create `output/liger-gpu/soup.yaml` using absolute paths to the frozen inputs:

```yaml
base: /data/frozen-llama-7b-snapshot
task: sft
backend: transformers
data:
  train: /data/frozen-train.jsonl
  format: chatml
  chat_template: chatml
  val_split: 0
  max_length: 1024
training:
  seed: 792
  batch_size: 1
  gradient_accumulation_steps: 2
  epochs: 1
  lr: 0.00002
  warmup_ratio: 0
  quantization: none
  stream_layers: true
  stream_source: ram
  stream_buffers: 2
  stream_read_ahead: 2
  stream_pin: true
  stream_vram_probe: true
  gradient_checkpointing: true
  fast_lora: false
  lora:
    r: 4
    alpha: 4
    dropout: 0
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

Start with a short `--warmup 2 --steps 3` pilot per qualified component in the
real model, inspecting saved adapter reloads and the first finite loss/gradients.
The runner intentionally refuses remote model IDs/datasets and existing output
directories. It overrides eval/periodic saves for measurement and records the
resolved config, seed, actual decoder dtype, AMP flags, patch counts, raw step
durations, allocated/reserved CUDA peaks, GPU snapshots and failure status.
It uses Soup's data loader and SFT wrapper, not a separate HF training stand.
Its own CUDA execution is pending; only argument/config behavior is CPU-tested.

For each qualified component B, run five ABBA blocks against A=baseline, using
the same config, GPU, model/data, seed, batch and sequence limit. Example for
one RMSNorm block (repeat blocks with distinct output paths):

```bash
for arm in baseline rms_norm rms_norm baseline; do
  index=$(( ${index:-0} + 1 ))
  python -m benchmarks.harness.liger_streaming_bench \
    --config output/liger-gpu/soup.yaml --arm "$arm" \
    --warmup 10 --steps 30 --output "output/liger-gpu/block1-$index-$arm"
done
```

Run the same blocks for SwiGLU and the union only if their correctness gates
passed. Compare paired block medians, spread and confidence intervals, not one
fastest step. The synchronized callback window measures optimizer steps with
accumulation; it excludes initial batch fetch, logging/save/eval and setup.
The separate training wall time includes final saving. Neither is automatically
end-to-end CLI throughput. CUDA peak counters describe the warm measured window,
not the full setup peak. Report both allocated and reserved values.

The runner uses the production card-aware dtype policy (normally BF16 here),
not the tests' FP16 policy override. Report its recorded dtype, not a filename's
label. The first performance stage is dense/RAM at seq=1024; do not promise the
whole matrix. Expand one cell at a time to NF4 and NVMe disk, then seq=2048 and
4096 only when fit and correctness are established. For disk, distinguish warm
OS-cache throughput from actual storage reads; never silently drop system caches.
Monitor clocks, power, thermals and other GPU processes externally for all arms;
missing telemetry or contention invalidates the corresponding timing block.
Keep measurement hooks and correctness diagnostics out of the same timed arm.

A fixed number of finite steps proves neither final quality nor time-to-target.
Those require the independent agreed real-data evaluation budget and seeds.

## Budget estimate, not authorization

Planning allowances: environment/download/setup 20–40 min; tiny GPU correctness
and failure triage 15–30 min; real 7B pilots 15–30 min; one qualified component's
five ABBA blocks 30–90 min. Thus the first useful qualification + dense/RAM
comparison is roughly **1.5–3 hours**, conditional on the measured step time;
all three components and wider cells can take **4–8 hours**. Recompute from the
pilot as `20 runs × (warmup + measured_steps) × seconds_per_step`, plus repeated
setup/sharding and storage overhead. These are allowances, not observed timings.

The [Runpod pricing page](https://www.runpod.io/pricing), checked 2026-10-19,
displayed A40 **$0.59/hour**, RTX 3090 **$0.50/hour**, RTX 4090 **$0.89/hour**.
At that displayed A40 rate, 1.5–3 hours is about **$0.89–$1.77 GPU compute**;
4–8 hours is **$2.36–$4.72**. Rates, availability, host RAM and storage vary;
storage/network/minimum deposits are additional. Recheck the actual offer before
any spend. Nothing was rented or paid for in this task.

## Local validation record

Python 3.10.0, torch 2.13.0+cpu, transformers 5.17.0, peft 0.20.0, trl 0.29.1,
bitsandbytes 0.50.1. Liger/Triton are not installed in this CPU environment.
The existing environment was read without installation or modification;
`PYTHONPATH` points to this isolated worktree and bytecode writes are disabled.

The final expanded regression invocation passed **963 tests / 61 skips**, covering
the new streaming tests and benchmark config, Fast-LoRA resident wiring, schema
recipe snapshots, task scoping, GPU marker policy, streamed save guards, multi-GPU
refusal, changelog validation, README translation parity, install-aware kernel
selection and the existing v0.72.0–v0.72.4 streaming suites. Five additional edge
cases verify refusal before tokenizer/weight downloads, independent activation
requirements, and matching streamed/resident predictions after PEFT reload with
the public checkpointing flag disabled.
The 24 new real-CUDA cases are among those skips. This is not full repository CI.
Full-path Ruff and `git diff --check` passed. Separately, all 12 CPU component ×
storage × quantization cells saved raw tensors; an independent reload compared
**2,748 tensor pairs** successfully with the declared CPU tolerances. The final
capture re-ran all 12 cells and pins source file hashes in
`final-source-manifest.json`; `replay-final.json` records matching source hashes,
JUnit and artifact SHA-256 values. This is
explicitly substitute-op integration evidence. Logs, JUnit, tensors and replay
JSON are under ignored `output/liger-local-checks/`; raw data is not added to Git.

Reproduce the final scoped suite from this checkout:

```bash
python -m pytest --no-cov -q --tb=short \
  tests/test_liger_streaming.py tests/test_liger_streaming_edges.py \
  tests/test_liger_streaming_bench.py \
  tests/test_fast_lora_training.py tests/test_issue621_recipe_snapshot.py \
  tests/test_issue806_kernel_flags_task_scoped.py tests/test_issue833_gpu_marker.py \
  tests/test_issue1011_streamed_save_guard.py tests/test_stream_multi_gpu_guard.py \
  tests/test_readme_translation_sync.py tests/test_issue1212_autopilot_install_aware_kernels.py \
  tests/test_issue487_changelog_fragments.py tests/test_v07200.py \
  tests/test_v07201.py tests/test_v07202.py tests/test_v07203.py tests/test_v07204.py
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
git diff --check
```

An initial new test used a full Kineto trace and failed at profiler shutdown with
`MemoryError: bad allocation`, after training. The trace was replaced with direct
autograd backward-node hooks, then the integration matrix passed. A setup-only
test initially passed an empty dataset into full trainer setup; it now invokes
the specific model-setup method it is testing. Neither failed invocation is a
Liger GPU or 7B numerical result.

The continuation also tested a disposable Linux/Python-3.12 image with CPU torch
2.13.0, Liger 0.8.1 and Triton 3.8.0 under `TRITON_INTERPRET=1`. The unmodified
RMSNorm forward failed when the CPU interpreter produced `None` for the external
reciprocal-square-root call; this did not qualify actual Liger numerics. A later
SwiGLU-only probe produced no usable result because Docker/WSL failed with EOF
and a bootstrap error. No kernel was patched to manufacture a pass. These logs
are under ignored `output/liger-interpreter/`. Production still requires CUDA;
no CPU fallback or interpreter mode was added. No Docker/WSL or Windows memory
configuration was changed, and this attempt does not diagnose the third-party
7B NaN report.

### Full Windows suite and clean-baseline controls

The full repository run completed on the current Python source with
`PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
`pytest tests/ -n 4 --dist worksteal -v --tb=short`. Raw result:
**35,287 passed / 12 failed / 13 setup errors / 261 skipped / 2 xfailed** in
26m47s, **86.55% coverage** (77% coverage threshold reached). Its exit code is
nonzero; this is **not** a full-suite pass or a CI waiver.

The two README stamp/structure failures were caused by the new README paragraph.
The English text, Turkish/Arabic/Japanese translations and stamps were corrected,
then all 31 README checks passed. Those checks also pass in the final 963-test
targeted invocation. The remaining **10 failures and 13 setup errors** reproduce
in an isolated source/test/fixture snapshot extracted from the original published
`859b2df3`, without the new Liger code or config field:

| Existing test family | Reproduced result / boundary |
| --- | --- |
| #1265 A/B integrated likelihood | Four failures: NumPy 1.26.4 has no `numpy.trapezoid` |
| #1267 and #1361 HTTP error classification | Three failures: `KeyError('file')` from the Python-3.10 error wrapper |
| #1550 numeric hostname limits | Thirteen setup errors: this Python 3.10.0 lacks `sys.set_int_max_str_digits` |
| #204 and #885 ANSI control assertions | Two failures: the colour-control output contains no ANSI sequence; also failed after removing `NO_COLOR`, so no sole cause is claimed |
| #819 fresh-process plugin discovery | One failure: the subprocess reader decodes UTF-8 CLI bytes as cp1251, raises `UnicodeDecodeError`, then the test sees `stdout=None` |

The four baseline control invocations produced 469 passes, 10 failures and 13
setup errors across those families. `failure-classification.json` matches every
raw full-run failure/setup-error ID against those controls or the corrected
README checks: **23 reproduced baseline cases, 2 corrected README cases, zero
unclassified cases**. These unrelated modules and tests were not changed.

The earlier sequential run was interrupted before completing, and the first
parallel attempt stopped during collection because existing v0.43 tests
parameterize unordered sets. Their collection agrees with a fixed hash seed;
those tests were not edited. Partial runs and their logs are retained, rather
than presented as clean invocations. All logs, XML and the extracted baseline
remain under ignored `output/liger-local-checks/` and are excluded from Git.
