# Coordinated RAM and GPU weight caches for SSD streaming

Qwen3-32B **BF16 LoRA**, physical RTX 3070 8 GB, context 512:
the combined cache reduces optimizer-step time by
**18.69%** versus ordinary
SSD streaming and **1.75%** versus the RAM
cache alone. The incremental comparison **does not pass** the original practical speed gate.
All original numerical gates passed; captured tensors were
**bitwise equal**.
These are measurements of the exact fixture below, not a universal speed guarantee.

| Metric | Ordinary SSD | RAM cache | GPU residency | Both caches |
| --- | ---: | ---: | ---: | ---: |
| Optimizer step (s) | 71.721 | 59.379 | 70.636 | 58.329 |
| Input positions/s, including padding | 14.278 | 17.245 | 14.497 | 17.555 |
| Fresh complete six-update entry (s) | 486.562 | 429.702 | 555.214 | 497.629 |
| Setup inside that entry (s) | 44.977 | 59.753 | 120.771 | 136.234 |
| Peak allocated VRAM (GiB) | 5.030 | 5.030 | 5.940 | 5.940 |
| Peak reserved VRAM (GiB) | 5.492 | 5.492 | 6.430 | 6.430 |
| Peak process RSS (GiB) | 7.912 | 21.912 | 7.950 | 21.969 |
| Physical SSD reads/update (GiB) | 225.240 | 174.379 | 221.607 | 170.746 |
| Additional RAM cache allocation (GiB) | 0.000 | 14.000 | 0.000 | 14.000 |
| Persistent GPU weight payload (GiB) | 0.000 | 0.000 | 0.908 | 0.908 |
| Timed decoder GPU loads/update | 250 | 250 | 246 | 246 |

| Comparison | Paired step-time cut | Pointwise 95% interval | Positive blocks | Original speed gate |
| --- | ---: | ---: | ---: | --- |
| Ordinary SSD → RAM cache | 17.23% | [17.18%, 17.28%] | 4/4 | PASS |
| Ordinary SSD → GPU residency | 1.51% | [1.42%, 1.67%] | 4/4 | FAIL |
| Ordinary SSD → Both caches | 18.69% | [18.62%, 18.76%] | 4/4 | PASS |
| RAM cache → Both caches | 1.75% | [1.74%, 1.78%] | 4/4 | FAIL |

The displayed role times are medians of four process medians. Percentage reductions
are medians of paired ratios, so they need not equal a ratio of displayed role times.
Positive values mean less time. A time reduction is not the same percentage as a
throughput increase. Confidence intervals are descriptive and pointwise, not familywise.

## What changed and why

The previous [profile and RAM-only study](../ssd-ram-cache-20261024/README.md)
found exposed SSD waiting. This study adds no new profiler trace. It tests whether
keeping weights on GPU complements the already implemented RAM hotset.

The SSD source now coordinates the two caches. GPU-resident layers are removed
from speculative disk queues and RAM-hotset planning. Lookahead crosses both
hotsets to fetch cold layers. GPU selection and validation happen first, then
the RAM hotset fills its own ceiling with different layers. A rejected candidate
restores the source; cache replacement drains pending copies and frees the old
host arenas before allocating new ones. Defaults remain disabled.

The native GPU capacity screen selected layer [31] before
accuracy or timing. Held-out runs use that fixed count. The RAM ceiling was
14336 MiB in both host-cache arms; both hold 14 decoder layers,
with identical actual allocated bytes and payload. The combined arm moves one
formerly RAM-cached layer to GPU and uses the freed RAM space for a different
layer. The 64-layer model's complete BF16 base is 61.03 GiB, larger than the
provider's native 42.5166 GiB RAM hard limit. Physical host RAM is 62.63 GiB;
the positive v1 soft planning hint is 29.75 GiB. No child limit or system setting
was changed. The actual `/data` volume maps to NVMe partition 259:3,
WD WDBGMP0010BNC-WRSN; every measured arm used direct I/O.

The GPU is physically 8192 MiB; CUDA usable memory is 7.658 GiB.
The original planner's 85% ceiling, 64 MiB reserve and future AdamW reservation
remain unchanged. More occupancy alone is not a performance objective, and
there must still be memory for activations, gradients, optimizer state and buffers.

```yaml
training:
  stream_layers: true
  stream_source: disk
  stream_resident_layers: auto
  stream_ram_cache_mb: 14336  # MiB ceiling; choose for actual RAM headroom
```

This is coordinated startup capacity planning, not a joint speed optimizer,
dynamic LRU, activation retention or full implementation of 10Cache. Tensor
bytes and arithmetic kernels are unchanged. No NF4, compile, FastLoRA, Liger,
custom Triton or external FlashAttention change is included in this comparison.

## Whole-run cost and setup

Complete fresh entries include imports, capacity probes, preload, six optimizer
updates and saving. They exclude the shared one-time model download and sharding.
Paired complete-entry time reductions are ordinary→RAM **11.90%**,
ordinary→GPU **-14.09%**, ordinary→both
**-2.00%**, RAM→both **-15.27%**.
Thus the combined cache was 2.00% slower than ordinary SSD streaming and 15.27%
slower than RAM alone in these complete six-update entries (paired medians).
GPU setup can cost more than a short run saves. Measured incremental setup
overhead versus RAM alone is 74.237 s; the steady-step estimate repays
that overhead after **71 updates**.
This extrapolation is not an observed full-epoch or convergence measurement.

## Accuracy, design and validation

- Pinned model Qwen/Qwen3-32B revision
  `9216db5781bf21249d130ec9da846c4624c16137`, dense BF16 frozen base.
  LoRA rank/alpha 4, seven decoder projections, 896 trainable adapter tensors,
  batch 1, accumulation 2, context 512; 1024 input positions/update
  including padding. Same fixed 32-example data and nonzero initialization.
- Eight full two-update captures completed before any timed run: primary
  3605 tensors / 445,388,808 elements, separate semantics
  1799 tensors / 72,274,817 elements per arm. All three candidates were checked
  against ordinary SSD streaming, including finite scans, exact inputs/init,
  original reference-first `torch.allclose(reference,candidate,rtol=.01,atol=1e-5)`
  and losses rounded to three decimals. Full raw tensors were retained and
  independently replayed on CPU before releasing timing. All four arms also
  have independently verified actual initial→final adapter updates.
- Four prospectively balanced blocks, 16 distinct fresh processes. Each process
  performs two warmup and four measured optimizer updates; all 96 updates,
  all six losses and every final adapter element are retained. Full trajectories
  pass the same numerical gates. No timing profiler or excluded updates.
  Whole-block bootstrap: 20,000 resamples, seed 792. Gate: median cut >=2%,
  at least 75% positive blocks, 95% lower bound >0. Primary incremental contrast:
  RAM→both. See [results-tables.md](results-tables.md) for every timing array.
- Actual GPU-host source controls: 144 passed, no skips or failures;
  six marked tests exercise real CUDA pools and async copy lifetimes.
  Full CPU suite: 35,454 passed, 10 failed, 13 errors, 253 skipped,
  10 deselected, two expected failures; coverage 86.57% against unchanged 77%.
  **The full CPU invocation is not green.** All 23 failing/error cases had
  already reproduced against unchanged head `20124f08` in this environment:
  old NumPy/Python APIs, urllib/tempfile, Rich rendering and Windows encoding.
  Exact messages are retained; the Rich folding-message difference is explicitly
  reported rather than normalized away. No tolerances/tests were disabled.
  Hosted CI is separate and is not awaited.

## Costs, cleanup and evidence

Conservative current-stage cost **$1.2299 / $5.12**, including its retained
reserve; this is an upper estimate, not an invoice. Prior-stage $2.529972 is
separate and was not deducted again. Instance **57478767** and persistent
volume **57478766** were deleted, then independently confirmed absent through
fresh account API listings.

- [result-summary.json](result-summary.json): derived metrics, original gates,
  setup estimate, source hashes, validation, cost and cleanup.
- [effective-protocol.json](effective-protocol.json): actual four-arm contract,
  explicitly separated from stale descriptive fields inherited in the immutable
  original protocol. Original bytes are preserved in the evidence archive.
- [evidence-manifest.json](evidence-manifest.json): exact ZIP identities and
  repository-sized parts. [join_evidence.py](join_evidence.py) verifies every
  part and reconstructed archive before creating a new output file:

  ```bash
  python join_evidence.py --output /path/to/new/evidence.zip
  ```

- The ZIP contains all executed frozen helpers and source snapshot, exact
  configs/data, all 16 timing metrics/provenance files, original validation
  receipts, independent replays and cleanup/cost proofs. It is review evidence,
  not a one-command cloud rental recipe; never relaunch historical controllers.
- [raw-artifact-index.json.gz](raw-artifact-index.json.gz) indexes all
  940 retained raw files
  (28,766,229,427 logical bytes). Full traces, adapters,
  optimizer states and canonical original archives remain under
  `C:/Users/esenb/Documents/SoupResearch/joint-cache-20261024-a`.
  Public hashes are not a substitute for the raw tensors needed for full replay.

After installing the tested Soup source snapshot and CPU replay dependencies
(Torch, safetensors, YAML and Rich), use the retained raw root and released helpers:
`replay_joint.py timing --root <retained-root>`. The original `accuracy` command
is intentionally tied to the pre-timing approval checkpoint; its completed report
and all eight original traces remain available for inspection. For a fresh
post-run numerical replay, run this from the retained helper directory:

```python
import json
from pathlib import Path

from rich.console import Console
from prepare_model import digest
from replay_cache_core import check_trace_schema, compare_files

root = Path.cwd()
qualification = json.loads((root / "independent-accuracy-replay.json").read_bytes())
paths = dict()
for arm in ("baseline", "ram", "vram", "both"):
    for phase in ("primary", "semantics"):
        key = arm + "-" + phase
        path = (root / "bundles" / ("sweep-joint-" + phase + "-" + arm)
                / "retained/results/cases" / arm / phase / "trace.safetensors")
        assert digest(path) == qualification["traces"][key]
        schema = check_trace_schema(path, phase, 896, 151936, 512)
        assert schema["initial_adapter_sha256"] == qualification["initial_adapter_sha256"]
        paths[key] = path
for arm in ("ram", "vram", "both"):
    for phase in ("primary", "semantics"):
        gate = compare_files(paths["baseline-" + phase], paths[arm + "-" + phase])
        assert gate["passed"]
        Console().print(arm, phase, gate)
```

This exact rendered code was also executed successfully on the retained captures;
its script, log and hash receipt are in the archive's `validation/` directory.
It calls the same frozen full-tensor checker and verifies the retained trace
identities and schemas; it cannot issue an approval or start cloud work. GPU runtime was
Torch 2.8+cu128; independent CPU replay used Torch 2.13 on Python 3.10.0.
No full fine-tuning, downstream quality, convergence, longer-context fit or
other-hardware performance claim is made.
