# Two-read + expert-cache GPU qualification

This is a prepared runner, not a GPU result. It never rents a machine, downloads
a model, changes a memory limit, drops the system page cache, or deletes storage.
Keep the full output outside an ephemeral instance before its owner removes it.

## What is compared

| Arm | Decoder recompute | Two-read | Expert RAM/GPU caches |
| --- | --- | --- | --- |
| `ordinary` | on | off | both disabled |
| `two_read` | on | on | both disabled |
| `cache` | on | off | same fixed byte ceilings |
| `two_read_cache` | on | on | same fixed byte ceilings |

The native `shard_checkpoint` and `build_streamed_model` paths are used, with an
NF4 frozen base, BF16 compute, disk tier, two decoder buffers, decoder RAM hotset
disabled, and rank-4 attention LoRA (alpha 8). The query target follows the
checkpoint: `q_proj` when `q_lora_rank` is null, as in Moonlight-16B-A3B, otherwise
`q_b_proj`; the other targets are `kv_b_proj` and `o_proj`. Every requested target
must exist on the actual meta skeleton before checkpoint sharding starts.
Routed-expert adapters, patched routers, forced uniform routing, retained decoder
activations and compilation are excluded from this comparison. Each arm runs
the same differentiable eval-mode graph with dropout disabled, AdamW lr .002 /
weight decay .01 / foreach and fused off. LoRA B starts at a reproducible small
nonzero value so both LoRA factors can actually update in this short control.
The resolved target list is frozen once in the spec; base weights, initial
adapters, inputs and numerical gates remain identical across all four arms.

The current two-read implementation admits DeepSeek-V3 decoder layers only.
Supply a materialized local supported real checkpoint, such as the native
DeepSeek-V3-family checkpoint selected with the two-read author. A tiny random
model validates the harness locally; it is not a real-checkpoint quality result.
The GPU input fixture is also frozen random token IDs: this tests numerical
preservation and throughput, not downstream task quality or convergence.

## Resource boundary and preparation

Use Linux, native BF16 CUDA, a **physical** 7.5–8.5 GiB card, sufficient local NVMe
for both source weights and the sharded store, and an externally enforced RAM
ceiling of at most 32 GiB with swap disabled for the workload. A 24 GiB GPU with
an allocator cap does not qualify. The script reads native cgroup-v2 mount and
ancestor limits; v1 is refused. For a larger host, set real cgroup/container
limits before starting it. The runner records the bound; it never configures it.

Use an isolated environment and the integration branch containing these files:

```bash
python -m pip install -e '.[dev]'
export PYTHONPATH="$PWD/src:$PWD"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python benchmarks/harness/two_read_cache.py --help
```

Install/transfer time, the package lock (`pip freeze`), source checkpoint ID and
immutable Hugging Face revision, disk model/mount and rental cost must be retained
by the operator as well. A directory using symlinks outside its root is refused
by the fingerprint containment check; materialize the local checkpoint first.

Before transferring the real checkpoint, run the tiny CUDA NF4 controls on the
chosen GPU. The second control uses Moonlight's full-rank query, BF16 autocast,
the worker's complete four-arm updates and unchanged numerical gates:

```bash
python -m pytest \
  tests/test_stream_expert_cache_nf4.py::test_real_cuda_nf4_expert_cache_preserves_complete_optimizer_trajectories \
  tests/test_two_read_cache_full_query.py::test_real_cuda_nf4_full_rank_query_bf16_worker_trajectories \
  -q -o addopts= --tb=short --junitxml=/results/preflight.xml
```

Require two actual passes, not skips. Retain the full-query test's `gpu-evidence`
directory (its location is also recorded in JUnit properties): it includes
numerical captures, hashes, comparisons and both cache-tier counters. CUDA
bitwise equality is informational; CPU controls additionally require it.

Choose explicit cache ceilings for a **qualification attempt**, not a claimed
optimal policy. The example requests 2 GiB packed RAM and .25 GiB dense GPU cache;
admission may refuse it on the chosen model. Both cache arms use identical values.
Do not adjust them for just one arm or lower a numerical gate after a failure.
To change budgets, prepare a new spec in a new directory and repeat all arms.

```bash
python benchmarks/harness/two_read_cache.py prepare \
  --weights /data/moe-checkpoint --shards /data/moe-nf4-expert-store \
  --out /results/two-read-prepared \
  --host-cache-gib 2 --device-cache-gib .25 --max-host-gib 32 \
  --seq-len 512 --batch-size 1 --accumulation 2 \
  --accuracy-updates 2 --warmup-updates 2 --timed-updates 6 --rounds 4
```

Preparation validates parameters and hardware before sharding. It binds the
source code, dependencies, GPU UUID, source/config/store contents and token
fixture with SHA-256, and records separate sharding/preparation times. Unsupported
or incompatible store layouts are refused. This first runner accepts only an
unstriped self-contained store. Cached expert entries are cleared after setup
probes so the actual workload starts cold in every arm.

## Accuracy before timing

```bash
python benchmarks/harness/two_read_cache.py run \
  --spec /results/two-read-prepared/spec.json --out /results/two-read-full
```

`--accuracy-only` stops before timing. A later full run must use a new output
directory and repeats accuracy; it does not trust an old green report.

All four fresh-process accuracy arms execute two AdamW updates, each with two
accumulated microbatches by default. Full CPU tensor captures retain initial
adapters, actual IDs, logits/loss, differentiable decoder-input gradients,
router logits and gradients, routing weights and gradients, exact top-k indices,
all LoRA gradients, every updated adapter, AdamW moments and step counts. Frozen
decoder inputs with no autograd edge are not given invented zero gradients.

Each candidate is compared independently with `ordinary`. Initial adapters,
inputs and integer indices must match bitwise. Floating values must be finite,
pass the existing reference-relative `rtol=.01, atol=1e-5` pointwise gate, and
loss must match at three decimals. Bitwise equality is additionally reported;
it is not assumed for CUDA. Every adapter must actually change. Missing captures,
updates, gradients or optimizer state fail closed.

Before timing, counters must also show the expected three-to-two logical expert
visits and actual **host and device hits** in both cache arms. A cache ceiling
that fits every routed expert on the GPU leaves the RAM arm unexercised and is
refused for this combined-cache protocol. Payload read counters are not SSD
traffic. Content hashes are checked again at phase transitions and after timing;
source/input hashes and model/store size/mtime identities are rechecked per arm.

## Timing, memory and interpreting the result

Timing uses 16 additional fresh processes by default: four rounds, every arm
occupies each execution position once. Each process runs two warmup updates and
six measured optimizer updates. No diagnostic autograd hooks run in timing.
CUDA is synchronized around each complete accumulated optimizer update. Outside
that interval, every microbatch loss, gradient, adapter and AdamW moment is
checked for finiteness; progress writes and these checks are included in stage
and whole-entry wall time. Setup probes, warmup and measured steps are separate.
The controller probes hardware in a child so its own CUDA context does not spend
the measured card's VRAM.

Outputs include sampled process RSS/swap (50 ms, not an exact instantaneous RSS
peak), native process lifetime RSS high-water mark, cgroup RAM/swap evidence,
allocated/reserved CUDA peaks, setup/entry wall times, per-update times,
tokens/second, logical expert requests, file-payload reads, materializations,
cache hits and Linux per-process physical I/O accounting (`/proc/self/io`).
Unavailable counters stay unavailable. Direct-I/O flags are retained. A warm
OS cache can make physical reads zero; the script does not force a cold SSD
claim or count another tenant's disk traffic as this process's work.

Step-time reductions use paired round block medians and a fixed-seed 5,000-sample
round bootstrap. The fixed practical gate requires the lower 95% interval to be
at least 2%. Four rounds remain a small-sample, host/model/input-specific result.
The report compares the combined arm both with `ordinary` and incrementally with
`two_read` and `cache`; independent savings are never added. A shorter step is
not automatically a shorter full job: inspect setup and whole-entry times too.

`run.json` is written atomically after each completed arm/block and on exceptions.
Each worker retains its own log, partial JSON and, for accuracy, `capture.pt`.
An OOM, missing CUDA, failed gate, timeout or cleanup failure is a nonzero failure,
not a skip or a speed result. Existing output directories with data are refused.
The complete capture files may be large; keep them and their SHA-256 manifest in
durable storage, and link the compact final report from PR #5.

## Local preparation verification

```bash
python -m pytest tests/test_two_read_cache_evidence.py \
  tests/test_two_read_cache_runner.py tests/test_two_read_cache_worker.py \
  tests/test_two_read_cache_preflight.py tests/test_two_read_cache_full_query.py \
  -q -o addopts= --tb=short
python -m ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

These tests exercise the real tiny-DeepSeek CPU training capture across four
arms, fixed numerical gates, mutation-sensitive controller refusals, strong
identity checks and memory/NaN guards. The full-rank query path is checked with
both dense BF16 and real CPU NF4 weights. They do not qualify real CUDA allocation,
NF4 kernels, physical SSD traffic, real-model speed or downstream task quality.
