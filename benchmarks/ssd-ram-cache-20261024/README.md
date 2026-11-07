# Bounded RAM hotset for SSD streaming on a physical 8 GB GPU

Qwen3-32B **BF16 LoRA** on RTX 3070: caching 14 decoder layers in RAM reduced
optimizer-step time by **17.18%** (four paired fresh-process runs, 95% paired
bootstrap interval **[17.11%, 17.28%]**). Original numerical tolerances were
unchanged; all 517,663,625 captured elements were finite and **bitwise equal**.
The prospective practical speed gate passed: median reduction >=2%, at least 75%
positive pairs, and bootstrap lower bound >0. This is an SSD-source result, separate
from the earlier GPU-residency study and FastLoRA research on this PR.

| Metric | Ordinary SSD streaming | SSD + RAM hotset |
| --- | ---: | ---: |
| Optimizer step | 71.732 s | 59.415 s |
| Complete fresh six-update entry | 485.876 s | 427.333 s |
| Setup inside that entry | 45.029 s | 58.836 s |
| Peak allocated VRAM | 5.030 GiB | 5.030 GiB |
| Peak reserved VRAM | 5.492 GiB | 5.492 GiB |
| Peak process RSS | 7.911 GiB | 21.912 GiB |
| Pinned RAM including ordinary staging | 6.000 GiB | 20.000 GiB |
| Physical SSD reads per measured update | 225.240 GiB | 174.379 GiB |

The complete six-update entries include imports, setup, cache preload, training
and saving. Their median paired time reduction is
12.21%. The shared initial
model download/sharding preparation took 1155.594 s once, before both arms, and is
excluded from both entry times. These are short fixed-fixture runs, not full-epoch
or convergence measurements. Throughput is
14.275 -> 17.235
processed input positions/s, including padding (1024 per optimizer update).

## Profile, cause and implementation

Profiling preceded implementation. One diagnostic CPU step spanned 71.891 s;
the union of `AsyncDiskSource.get` intervals was 59.543 s, of which 47.663 s had
no observed GPU activity. Kernel/copy union was 23.842 s. These are overlapping
interval unions, not additive time buckets; exposed `get` time includes source
and host overhead and is not a pure SSD service-time measurement.

The opt-in production source now loads a fixed, immutable hotset through the
existing verified range reader, skips its subsequent disk reads, and reads ahead
across cached indices. Shard identity checks and DMA lifetime protection remain.
The same rotating GPU buffers and training math execute in both arms. Decoder
GPU loads remain 250 per measured update, plus four large embedding/head loads.
Physical Linux process reads fall from 241,849,663,488 to 187,238,449,152 bytes
per update (**22.58%**). This corresponds to 56 avoided aligned decoder reads
(14 cached layers, forward/recompute, two microbatches); literal read-call counts
were not instrumented in the unprofiled timing runs.

The cache ceiling is 14 GiB allocated, holding 12.715 GiB payload in layers 25..38.
Total pinned RAM includes the unchanged 6 GiB staging allocation. Caching uses
only part of RAM when the complete base does not fit; it cannot eliminate SSD
reads or make an arbitrary model fit in the machine. It does not consume extra
VRAM and does not increase GPU residency.

```yaml
training:
  stream_layers: true
  stream_source: disk
  stream_ram_cache_mb: 14336  # MiB allocation ceiling; choose for your machine
```

Default 0 retains the existing path. The source automatically selects how many
repeated layers fit the explicit ceiling and current RAM headroom, accounting
for actual arena rounding, visible process/ancestor cgroups, 30% slack and
512 MiB reserve. No swap is admitted. Allocation failure is explicit. This is
a startup hotset planner, not a joint VRAM/RAM/SSD speed autotuner, dynamic LRU,
or implementation of the whole [10Cache paper](https://arxiv.org/abs/2511.14124).
The paper informs profiling and reuse; none of its advertised speedups is claimed.

## Qualification and hardware

- Qwen/Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`,
  64 decoder layers, frozen BF16 base 65,524,246,528 bytes (61.03 GiB).
  No NF4, FastLoRA, Liger, compile, custom Triton or FlashAttention change here.
- Physical RTX 3070 8192 MiB; CUDA usable 8,222,998,528 bytes (7.658 GiB).
  Physical host RAM 62.63 GiB, native provider cgroup hard limit 42.5166 GiB,
  positive soft planning hint 29.75 GiB. The full base exceeds the hard limit.
  An additional child-cgroup limit was unavailable; this does **not** mean the
  existing provider hard limit was absent. No observed process swap in timing.
- Attached `/data` volume maps to NVMe partition `259:3` (WD WDBGMP0010BNC-WRSN).
  Both arms actually used direct I/O; cache-ceiling selection was frozen before timing.
- SFT LoRA rank/alpha 4, seven projections, batch 1, accumulation 2, context 512,
  fixed 32 ChatML examples, identical nonzero initialization, 896 adapter tensors.
  GPU Torch 2.8.0+cu128; independent replay Torch 2.13.0+cpu on Python 3.10.0.
- Before timing: 3605 primary tensors / 445,388,808 elements plus 1799 separate
  semantic tensors / 72,274,817 elements, all finite and bitwise equal. Original
  reference-first `torch.allclose(reference,candidate,rtol=.01,atol=1e-5)` and
  losses rounded to three decimals passed. The oracle is matched ordinary SSD
  streaming, not a fully resident 32B training run or downstream task-quality oracle.
- Four fresh pairs AB/BA/BA/AB, eight distinct processes, two warmup plus four
  measured optimizer updates each. All 48 updates, all six losses per process,
  and every final adapter element retained and independently replayed; final
  adapters/losses bitwise equal in every pair. No timing profiler or excluded updates.
  Bootstrap resamples whole pairs 20,000 times, seed 792. The reported reduction
  is the median of paired ratios, not a ratio of the displayed role medians.

## Validation, costs and cleanup

All **48 source controls passed on actual CUDA**, including async buffer lifetime
checks. Final project-marker/cache CPU controls: 80 passed, two CUDA skips.
Full default CPU suite: 35,429 passed, 16 failures, 13 errors, 252 skips and
two expected failures; **that invocation is not green**. Coverage was 86.56%
against the unchanged 77% requirement.
Six failures from incomplete fake config objects / the intentional default-snapshot
addition were fixed without changing production code; the complete consumer
modules then passed (248 passed, three hardware skips). Remaining failed
cases were independently reproduced against unchanged head `20124f08` in this
environment: old NumPy lacks `trapezoid`, Python 3.10.0 lacks the integer-string
limit API and exposes an urllib/tempfile issue, plus existing Rich rendering
expectations and a Windows subprocess encoding failure. See the exact failure
comparison in `evidence.zip`; no tests or
tolerances were disabled. Hosted CI is separate and is not awaited.

Conservative stage cost **$2.5300 / $7.65**, including failed hardware
preflights and a retained forecast reserve for one refused offer; this is an
upper estimate, not an invoice. The earlier separate study's $2.34585 is not
charged to this remaining cap again. Instance **57424635** and persistent volume
**57424634** were deleted; a fresh independent API query confirmed both absent.

## Evidence and reproduction

- [result-summary.json](result-summary.json): derived metrics and qualification.
- [results-tables.md](results-tables.md): all pairs and full timing arrays.
- [effective-protocol.json](effective-protocol.json): actual case plus explicit
  inherited descriptive-label corrections; the original protocol is unchanged.
- [Split evidence archive](evidence-manifest.json), joined by [join_evidence.py](join_evidence.py):
  exact baseline/candidate source archives, all released helpers, configs/data,
  eight process metrics/provenance files, independent replays, source hashes,
  exact compressed profiler trace, hardware/cleanup/cost receipts and validation.
- [raw-artifact-index.json.gz](raw-artifact-index.json.gz): hashes and paths of all
  750 full raw retained files. Large tensors,
  optimizer states and original archives remain under
  `C:/Users/esenb/Documents/SoupResearch/ssd-ram-cache-20261024-h`.
  Reviewers can inspect compact evidence directly; full tensor replay requires
  these retained raw artifacts. The public index is not a substitute for the tensors.

The repository caps new files at 5 MB. Six parts preserve the entire verified ZIP
without changing any payload. Verify and join them to a new destination, then extract:

```bash
python join_evidence.py --output /path/to/new/evidence.zip
```

The reconstructed ZIP's `manifest.json` records each exact payload hash;
baseline and candidate tarballs are under `executed/`. `executed/feature-ready.json`
binds all 19 released candidate/helper payloads. Exact configs and dataset are
under `inputs/`; check archive paths before replay. On a suitably provisioned
GPU host, install the source and declared training dependencies, prepare the
pinned checkpoint with the retained preparation helper, then use the retained
`cache_bench.py`/`feature_phase.py` sequence. Check retained hardware, budgets and
source hashes; do not relaunch cloud-controller scripts or reuse historical IDs.
The complete historical cloud controller is retained for provenance, not as a
safe one-command rental recipe.

For CPU verification with the full retained raw directory, place the exact
released helpers and `analysis/` helpers alongside its metadata and run:

```bash
python audit_cache_timing.py --root <retained-root> --output <new-timing-replay.json>
```

The accuracy replay helper and original numeric gates are also retained. The
profiling entry has an explicitly recorded pre-execution v2 normalization from
unsupported warmup 0 / measured 4 to warmup 1 / measured 3, still four diagnostic
updates. Offline analysis selects the CPU profiler step instead of its duplicate
GPU annotation. The committed GPU test file subsequently adopts the required
project `gpu` marker; all executed test bodies and all five production source
files still match the tested candidate archive. The exact executed test source
and adjustment proof are included. No source/archive provenance was rewritten.
