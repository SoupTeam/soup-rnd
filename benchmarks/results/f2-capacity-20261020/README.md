# F2: allocated 32 GB and natural NVMe fallback on a physical 8 GB GPU

New independent cohort: **180 commands; 142 PASS, 35 EXPECTED_CAPACITY_REFUSAL, 3 NO_VERDICT**.
The [preregistered protocol](../../f2-capacity-20261020.md) and source
`fed4c760bf7d8619bc265636c33e80ba7901f868` identify the measurements.
The earlier [RTX 3060 Ti forced-tier study](../f2-reference-20261019/README.md)
is retained separately and contributes no samples to this result.

This demonstrates real **Qwen2.5-14B LoRA fine-tuning**, automatic NVMe selection
when the BF16 resident store exceeds the native container-aware RAM budget, and native
RAM-only capacity refusal. It does not demonstrate physical OOM/exhaustion,
model quality, full-parameter fine-tuning or forward/backward exactness.
Reviewer acceptance of the complete F2 task remains separate from these facts.
Observed completed working commands with 12 steps and changed counted-window
parameters: **144**.
F1 admission is separate: only **70** benchmark
commands enter the throughput summaries.

## Nine combinations

- **primary: 150 commands**, 119 PASS, 29 EXPECTED_CAPACITY_REFUSAL, 2 NO_VERDICT.
- **supplement: 30 commands**, 23 PASS, 1 NO_VERDICT, 6 EXPECTED_CAPACITY_REFUSAL.

The single whole supplementary block was frozen before the first command;
it is included only when the primary schedule counts do not meet the preregistered
minimum. No failed/void run is relabelled or replaced. Per-cell phase ledgers
are in [analysis.json](analysis.json); every command's phase is in the CSV.

Counts below mean PASS for working conditions and EXPECTED_CAPACITY_REFUSAL
for the boundary. Native bench rates use eight counted steps after four
warmup steps, median across valid native reports from the completed primary
and supplementary phases; ordinary train contributes
update/save evidence and no throughput samples.

| Condition | Token cap | Ordinary train | Native bench | Step s | Useful tok/s | Total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| RAM / NF4 | 128 | 12/12 | 12/12 | 1.186 | 78.17 | 107.88 |
| RAM / NF4 | 384 | 12/12 | 12/12 | 1.287 | 270.76 | 298.13 |
| RAM / NF4 | 1024 | 12/12 | 12/12 | 2.356 | 419.65 | 434.61 |
| Auto NVMe / BF16 | 128 | 12/12 | 10/12 | 8.628 | 10.75 | 14.83 |
| Auto NVMe / BF16 | 384 | 12/12 | 12/12 | 8.646 | 40.32 | 44.40 |
| Auto NVMe / BF16 | 1024 | 12/12 | 12/12 | 8.706 | 113.60 | 117.65 |
| RAM-only BF16 capacity refusal | 128 | 6/6 | 5/6 | n/a | n/a | n/a |
| RAM-only BF16 capacity refusal | 384 | 6/6 | 6/6 | n/a | n/a | n/a |
| RAM-only BF16 capacity refusal | 1024 | 6/6 | 6/6 | n/a | n/a | n/a |

All raw outcomes, including invalid commands, are in [runs.csv](runs.csv).
All 18 CLI/condition/length counts meet the preregistered minimum of five.
Invalid commands: **3**. They are retained in the raw packet,
with their original F1 readings and command outcomes, and do not enter rates.

- `boundary/070-ram-128-bench` (primary): NO_VERDICT, F1 void; at least 4262514688 bytes of foreign reads on nvme0n1 (limit 1000000000).
- `overflow/095-nvme-128-bench` (primary): NO_VERDICT, F1 void; at least 2379173888 bytes of foreign reads on nvme0n1 (limit 1000000000).
- `overflow/155-nvme-128-bench` (supplement): NO_VERDICT, F1 void; at least 2420858880 bytes of foreign reads on nvme0n1 (limit 1000000000).

No ten-percent-effect or causal RAM-versus-disk speedup claim is made: the
RAM condition uses NF4 and the capacity conditions use BF16. The format is
part of the condition. Useful tokens exclude prompt/padding tokens.

## Actual resources and capacity semantics

The board is **RTX 5060 Ti**, NVML 8,546,942,976 B (8151 MiB), CUDA available
8,122,859,520 B, UUID `GPU-a1b34640-a100-ca11-5f9c-06d248ed645a`, driver
610.57.04. This is neither an RTX 5070 baseline nor the older RTX 3050/3060 Ti
measurement. No GPU memory cap was used to simulate an 8 GB board.

The provider assigned a protected baseline of **32,028,753,920 B**
(32.03 decimal GB), observed as `memory.low`; the initial allocation-free
headroom was **32,003,039,232 B** (32.003 GB), before
dependency/model preparation. `memory.low` is protection, **not a kernel hard
limit**. The actual burst ceiling `memory.max` was **56,060,018,688 B**
(56.06 GB); the physical shared host had approximately 101.15 GB RAM. Host
MemAvailable, baseline-free headroom and burst capacity are distinct records.
No cgroup/pagefile setting, cache flush, memory-pressure process or clock/power
setting was changed. The shipped planner now observes the conservative real
container baseline, bounded by any smaller hard/ancestor limit and host memory.

The exact NF4 cache is 9,931,805,043 B; BF16 is 29,540,125,622 B. With
`stream_source: auto`, the native planner selects disk for BF16 without a
forced tier. The same BF16 checkpoint with `stream_source: ram` refuses before
an optimizer step because resident loading exceeds its actual free-RAM/safety
budget (the first refusal reports 29.5 GB on disk and 30.7 GB free). This is a
natural allocation-budget fallback and refusal, not a claim that Linux reached
physical OOM at a hard 32 GB limit. No artificial HDD policy flag is involved.

The sole physical device is `nvme0n1`, **SAMSUNG MZVL21T0HCLR-00BH1**.
`stream_disk_kind: nvme` identifies verified media; it does not force the tier.
Full O_DIRECT mapping covers all 51 actual safetensors files of each cache,
with each payload accounted on that NVMe and zero reads on another device.
Fresh F1 controls for both actual caches pass clean and void a separate SSH
foreign read of at least 1 GB. Per-arm F1 checks and counted GPU traces remain
under their original acceptance rules.

| Condition | Tokens | Peak CUDA reserved, GiB | Max sampled RSS tree, GB | Mean GPU util, % | Median SM MHz | Median W | Sampled physical read MB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ram | 128 | 2.465 | 26.27 | 99.2 | 2820 | 77.7 | 0.0 |
| ram | 384 | 3.037 | 26.32 | 99.4 | 2794 | 117.6 | 0.0 |
| ram | 1024 | 4.697 | 26.28 | 99.7 | 2790 | 138.1 | 0.0 |
| overflow | 128 | 3.217 | 21.95 | 60.0 | 2835 | 36.7 | 5877.0 |
| overflow | 384 | 3.898 | 22.01 | 60.2 | 2820 | 44.4 | 5869.2 |
| overflow | 1024 | 5.521 | 21.97 | 60.9 | 2820 | 61.9 | 5834.8 |

Allocator peaks include setup and training. Process-tree RSS and GPU/device
samples are retained separately; RSS is sampled within counted steps and can
double-count shared pages. It is not a kernel allocation-peak measurement.
Clock events cover the entire execution; throughput telemetry covers counted
steps. See [analysis.json](analysis.json) for all events and invalid outcomes.

## Recipe, artifacts and repairs

Base: `Qwen/Qwen2.5-14B-Instruct`, revision
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
The exact revision in `checkpoint-metadata/f2-source.json` inside the raw
packet and the frozen manifest is authoritative.
The real base has 14,770,033,664 parameters. q/v LoRA uses r=16, alpha=16,
dropout=0, seed/data_seed=42, batch/accumulation=1, AdamW 2e-5, weight decay
0.01, two streaming buffers/read-ahead=2, 12 optimizer steps per working
command. The dataset and all resolved configs are included in the raw packet.
The dataset is **synthetic benchmark input**: twelve fixed chat examples with
repeated explanatory text, truncated to each token cap. The GPU execution,
checkpoint tensors, gradients, updates and timings are real; this corpus does
not support a downstream model-quality claim. Every command starts a fresh
12-step run, rather than continuing one long training trajectory.

The six deterministically first F1-valid ordinary-train adapters were saved
before subsequent commands could overwrite their outputs. Each has 192 finite
tensors, 12,582,912 parameters and 96 nonzero LoRA-B tensors; saved tensor bytes
reproduce the originating runtime fingerprint. See
[adapter-verification.json](adapter-verification.json). The six trained weight
files and original archives are SHA-verified in the task's durable local
artifact store. This lightweight public packet excludes model/cache and adapter
tensor blobs; it includes their exact hashes and recorded verification.

The first new attempt (`b9156e31`) is retained under `interrupted/` in the
packet: two RAM/128 false refusals, one successful overflow benchmark and one
interrupted overflow training command. It is excluded from this cohort. The
memory probe originally failed to credit clean active checkpoint page cache;
the correction credits clean active/inactive file LRU while excluding shared,
dirty/writeback and anonymous pages. Red regression cases preceded the repair.
A fresh committed source, fresh F1 controls and new freeze preceded this full
new schedule. The earlier native-cache preparation error is also retained.

Rental validation on the corrected source: **412 passed, 14 skipped**, Ruff
clean. This is the focused CPU regression set, not a new full-suite/CUDA
exactness verdict. Earlier disclosed rental test failures remain in the old
report. Package versions and exact measured source archive are in the packet.
The hosted [Ubuntu/Python 3.12 unit job and lint at the measured commit](https://github.com/SoupTeam/soup-rnd/actions/runs/39483098386)
also passed; optional jobs were skipped. This hosted result is separate from
the focused rental tests and the old rental failures.

## Independent replay

Verify the published file bytes with [CHECKSUMS.sha256](CHECKSUMS.sha256)
(`sha256sum --check CHECKSUMS.sha256` from this directory).

[records.tar.gz](records.tar.gz): 4,958,025 B;
SHA256 `da81dd3c24b9e77c8b3842021acb4192d37784458846a76d65c83630c8a5c1b1`. Its internal member manifest verifies every
original record, config, dataset, control and source snapshot. All frozen
inputs have preserved byte-for-byte copies, so later repository changes do
not alter their provenance. Full checkpoint/cache hashes were rechecked after
timing on the rental before export.

Extract the packet, then run the adjacent audit tools from the PR checkout
(with Soup and safetensors/torch installed):

```bash
PYTHONPATH=src python benchmarks/harness/f2_capacity_audit.py /path/to/extracted/corrected \
  --harness benchmarks/harness
# Actual six adapter tensor files are required for saved-weight verification:
python benchmarks/harness/f2_capacity_verify_adapters.py /path/to/valid-adapters \
  /path/to/extracted/corrected --output adapter-verification.json
```

The replay rebuilt **72 native bench reports** completely
from original step/snapshot evidence and recomputed foreign-read bounds for
**180 commands**. It rejudged every raw outcome and recomputed
counted trace summaries and the nine-cell descriptive table. It does not load
the excluded base/cache tensor blobs. To repeat GPU measurements, use the
prospective protocol with a new resource window, the pinned checkpoint and
the same real resource checks before freezing; do not reuse the expired rental
deadline or reinterpret the burst ceiling as hard 32 GB RAM.

## Rental cleanup and budget

After local archive/member SHA verification and saved-adapter fingerprint
verification, only this task's rental was deleted and its absence confirmed
through the provider API. See [cleanup-receipt.json](cleanup-receipt.json).
The conservative combined F2 cost bound, including earlier F2 rentals and
the new transfer reserve, was **$4.303**
within the authorised **$5**. This is a reserved upper-bound estimate, not a
settled provider invoice. Unrelated account rentals were not modified.
