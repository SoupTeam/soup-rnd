# Streaming weight residency on a physical 8 GB GPU

This Qwen3-8B study found **no practical speedup under the unchanged 2% gate** from
adding private decoder slots on RTX 3070, despite reducing weight transfers substantially.
All seven same-format correctness qualifications passed with bitwise-equal saved tensors.
The separate fourteen-update held-out trajectories also passed independent replay.

| Comparison | Seconds/update, baseline -> candidate | Paired time cut, 95% CI | Candidate allocated / reserved GiB |
| --- | ---: | --- | ---: |
| BF16 RAM, 0 -> 6 private layers | 4.0735 -> 4.0596 | +0.25% [-0.21%, +0.52%] | 6.298 / 6.889 |
| NF4 RAM, 0 -> 34 private layers | 4.0528 -> 4.0076 | +0.92% [-0.15%, +2.63%] | 6.679 / 7.150 |
| NF4, RAM -> NVMe, no private layers | 4.0281 -> 14.0073 | -248.03% [-340.59%, -3.21%] | 3.527 / 4.000 |

Positive time cut means less time per optimizer update. Seconds summarize process medians;
the time cut is the median of four paired reductions, not a ratio of the displayed summaries.
These are measured windows, not full-dataset training times. The four-layer BF16 control,
per-process timings, complete short-run phases, CPU memory and I/O are in
[results-tables.md](results-tables.md); [result-summary.json](result-summary.json) is a derived
view. Full original summaries and process records remain in the exact evidence archive.

## Capacity and transfers

BF16 RAM counts 0/4/5/6 passed full numerical qualification. Seven completed one update
but was refused when sampled driver-free memory fell to 99.125 MiB, below the original
128 MiB floor; this was a policy refusal, not an OOM. Higher BF16 counts were not attempted.
NF4 capacity screens reached 34 private layers; 35 and 36 were refused at the same floor.
Only NF4 counts 0 and 34 received full accuracy qualification. These are configuration-specific
boundaries, not global maximum layer counts.

At six BF16 private layers, decoder loads fell from **138 to 114 per update**. At NF4 count 34
they fell from **138 to 2**. Its private indices are 1–34; layers 0 and 35 can retain distinct
rotating slots, but the current `prime()` still refills layer 0 each microbatch. All cases
retain four large embedding/head loads per measured update. Logical copy counters do not
measure PCIe bandwidth. RAM source copies remain allocated when weights are duplicated in VRAM.

NF4 cache plus index occupies **5.656 GiB**, versus **15.256 GiB** for BF16. These are file
sizes, not GPU memory or required machine RAM. Native dense adapters are FP32 whereas
TRL 0.29.1 casts this NF4 path's adapters to BF16. Thus the direct cross-format strict
diagnostic is **not admitted because dtypes differ**; no BF16-versus-NF4 quality equivalence
or quantization-only memory saving is claimed. Every residency and RAM/disk comparison
uses a matched format and adapter dtype.

## NVMe result and its limit

Soup already supports disk streaming. This comparison uses the same NF4 shard bytes,
data, adapters and backend, with **zero private layers in both RAM and disk arms**.
The cache was on a separately mounted, verified KINGSTON NVMe partition, and each disk
process actually used direct I/O. Every measured disk update reports 13,539,246,080
Linux process `read_bytes` (12.61 GiB); matched RAM updates report zero.

The measured disk/RAM multiplier is **3.4803x**, but disk updates switch from roughly
4 to 17–18 seconds inside the original windows. Six warmups did not ensure stationarity;
the multiplier describes these mixed-regime windows, not universal steady NVMe performance.
No updates were excluded or reclassified. A later read-only cgroup/temperature snapshot
does not establish the cause. The archive includes both original timer arrays and the
SSD investigation. Disk streaming with private GPU slots was **not** tested here.

## Protocol and reproducibility

- Physical RTX 3070 8192 MiB; CUDA usable total 8,222,998,528 bytes (7.658 GiB).
- Qwen/Qwen3-8B at `b968826d9c46dd6066d109eabc6255188de91218`, 36 decoder layers.
- SFT LoRA, batch 1, context 1024, accumulation 2, rank/alpha 4, all seven decoder projections;
  BF16 autocast, dropout 0, fixed data/order and nonzero adapter initialization.
- Research wrapper installs centered private slots after normal mode-0 setup. It leaves
  the production auto planner's 85% cap unchanged. Research allocation reserves 256 MiB
  against CUDA total and checks a sampled 128 MiB free floor at setup/update/completion.
- Primary capture compares 2,037 tensors; the separate semantic capture compares 1,015.
  Exact inputs/initialization, finite values, original reference-first `rtol=0.01, atol=1e-5`,
  losses to three decimals. No tolerances were relaxed. Saved cross-arm semantic tensors
  replay independently; perturbed future/padding outputs were not retained, so isolation
  itself relies on the frozen harness's assertions and completion status.
- Accuracy precedes selection screens (2 warmup + 3 timed updates). Held-out timing uses
  four fresh process pairs, AB/BA/BA/AB, each 6 warmup + 8 measured updates. All fourteen
  losses and 504 final adapters per process are checked. Throughput counts 2,048 processed
  input positions per update, including padding.
- Original speed gate: median paired time cut >=2%, >=75% positive pairs, 95% whole-pair
  bootstrap lower bound >0; 20,000 resamples, seed 792. Selection times cannot satisfy it.
  Separate setup, all-14-updates-plus-save and process-wall summaries are descriptive,
  overlapping windows; do not add them or interpret fourteen updates as a full epoch.
- 92 scoped regression tests passed on the GPU host. The local replay/summary/packaging
  controls passed 77 tests. These counts describe different scopes; hosted CI was not awaited.

Executed source SHA256 is
`adfc15617878f3592f3bd49c37fd69ef7b94f8e4e2d5b177e409a2e42f7e1bb7`.
The archive derives from source baseline `f8833454c9a7e60af5c7d4b038600616f8979b25`;
four production feature files match PR #5 commit `e0d47e0dfb76a932e436a7034d2f655b46d86788`
under the recorded CRLF/LF-only mapping. This is not an assertion of full Git-tree equality.
`executed/launch-manifest.json` pins the separate frozen experiment helpers.

The [earlier 16–31% result](../streaming-residency-20261020.md) used RTX 4090 24 GB,
OpenLLaMA-7B, 8/16 private layers and a 10 GiB allocator cap. Its 16-layer arm also reduced
fourteen-updates-plus-save time, 28.55 to 19.91 seconds. It does not establish the same gain
on physical 8 GB/Qwen3-8B. Hardware, model, vocabulary and resident fraction differ;
no operator profile establishes which difference explains the changed result.

## Evidence, spending and cleanup

Aggregate conservative ledger estimate **$2.345850 / $10.00**, including prior leases $0.765158; current lease $1.580692; remaining **$7.654150**. This is not a provider invoice.

Instance `57401620` and persistent volume `57401619` were deleted. A fresh read-only account listing independently confirms both absent.

[evidence.zip](evidence.zip) preserves each included original byte-for-byte. The compact
[manifest](evidence-manifest.json) identifies the archive and losslessly compressed
[full manifest](evidence-manifest.full.json.gz). The
[relative raw index](raw-artifact-index.json.gz) contains retained tensor/archive hashes;
large captures and canonical raw archives remain local, outside the metadata ZIP.
The private absolute-path index is not published. Compact statistics copy existing values;
the full original `local/result-summary.json` is inside the ZIP. No implementation changed
in this research commit.

For offline replay, extract the ZIP and use CPU Python with torch, safetensors and rich.
Pass each retained bundle root separately; the metadata archive alone lacks the raw tensors:

```powershell
$replayArgs = @('--helpers', 'executed',
    '--root', 'D:\study\bundles\bundle1\retained',
    '--root', 'D:\study\bundles\bundle2\retained', '--output', 'replay.json')
python audit-tools/replay_results.py @replayArgs
```

The archive also includes source-pinned notes on 10Cache, native adapter precision, CPU
memory counters and the residency/load mapping. This study does not test torch.compile,
activation retention, their joint policy, or 14B/27B. A next candidate is resident-aware
GPU/RAM caching for the disk path to reduce its observed reads; its benefit remains unmeasured.
Filling VRAM alone is not an optimization objective: a placement policy must minimize measured
step time while keeping memory headroom.
