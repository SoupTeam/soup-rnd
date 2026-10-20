# F2: real Qwen2.5-14B LoRA on a physical 8 GiB GPU

The selected NF4 matrix completed on 2026-10-19/20 UTC: **150 commands, 118 PASS,
2 NO_VERDICT, 30 EXPECTED_REFUSAL**. All **60 working native bench reports are
valid**, ten per RAM/NVMe/length cell. All 120 working CLI processes completed
12 steps, with changed trainable parameters before/after the counted window.
This is a functional and throughput result, not a model-quality evaluation.

**Automatic RAM overflow remains unverified.** NVMe was forced in this matrix;
the HDD override is an artificial refusal-policy control. The separate BF16
capacity experiment was NOT_RUN because its physical-mapping preparation gate
failed. These limits prevent treating this record as full closure of upstream
[#375](https://github.com/MakazhanAlpamys/Soup/issues/375).

Measured Soup source: `e6829f159c24e29a7496f4516bd19a94d0ca8f80`, based on `ca9be3cc9f11b91cba76d26f8a0d7ad18787c4ff`.
The [protocol](../../f2-14b-reference.md), frozen inputs and raw records identify
the measurement commit independently of this later results document.

## Native throughput and outcomes

Each working process observes 12 steps and excludes four warmup steps. Throughput below
is the median of the **native** bench useful/total-token rates across ten valid
bench processes. Step time is the median of their native per-run medians; it is
not substituted into the native throughput formula. Ordinary train provides a
separate update/save check and contributes no throughput samples to this table.

| Tier | Token cap | Ordinary train | Native bench | Step s | Useful tok/s | Total tok/s |
|---|---:|---|---|---:|---:|---:|
| RAM | 128 | 10/10 PASS | 10/10 PASS | 1.149 | 79.16 | 109.24 |
| RAM | 384 | 10/10 PASS | 10/10 PASS | 1.542 | 221.27 | 243.64 |
| RAM | 1024 | 10/10 PASS | 10/10 PASS | 3.211 | 304.21 | 315.06 |
| NVMe | 128 | 9/10 PASS; 1 NO_VERDICT | 10/10 PASS | 17.527 | 5.10 | 7.04 |
| NVMe | 384 | 10/10 PASS | 10/10 PASS | 17.634 | 19.48 | 21.45 |
| NVMe | 1024 | 9/10 PASS; 1 NO_VERDICT | 10/10 PASS | 18.459 | 52.08 | 53.94 |
| HDD policy override (artificial) | 128 | 5/5 expected refusal | 5/5 expected refusal | n/a | n/a | n/a |
| HDD policy override (artificial) | 384 | 5/5 expected refusal | 5/5 expected refusal | n/a | n/a | n/a |
| HDD policy override (artificial) | 1024 | 5/5 expected refusal | 5/5 expected refusal | n/a | n/a | n/a |

Actual total tokens per step are 128, 384 and 1024, respectively. Prompt tokens
are excluded from useful throughput. The fixed 12-row human-readable dataset
is a throughput fixture. There is no convergence or language-quality claim.

## Resources and clocks

Allocator/RSS maxima cover accepted train and bench commands, including setup.
GPU utilisation, SM clocks and process I/O rates describe valid bench counted
windows. RSS sums the process tree and can double-count shared pages; it is not
unique physical RAM consumption. I/O is the process-tree `read_bytes` rate,
not a separate sustained-drive benchmark.

| Tier | Cap | CUDA allocated / reserved peak GiB | Sampled RSS-tree peak GiB | GPU util % | SM MHz | Read MB/s |
|---|---:|---:|---:|---:|---:|---:|
| RAM | 128 | 2.23 / 2.46 | 19.94 | 99.6 | 1950.0 | 0.0 |
| RAM | 384 | 2.64 / 3.04 | 19.96 | 99.3 | 1942.5 | 0.0 |
| RAM | 1024 | 4.04 / 4.70 | 19.98 | 99.7 | 1935.0 | 0.0 |
| NVMe | 128 | 2.23 / 2.46 | 13.94 | 8.9 | 1665.0 | 718.0 |
| NVMe | 384 | 2.64 / 3.04 | 13.96 | 11.1 | 1665.0 | 729.6 |
| NVMe | 1024 | 4.04 / 4.70 | 13.98 | 20.6 | 1665.0 | 688.5 |

F1 recorded 323 `sw_power_cap` and five `sw_thermal_slowdown` samples over whole
executions. Thermal flags occurred in working bench records 018, 051, 091 and
118, plus refusal record 019. Their timing relative to counted steps is not
established by the whole-execution flag. F1 records these flags without
automatically voiding runs; all accepted rows remain in the primary summary.
Power, fan and clock settings were unchanged. These are descriptive results
under the observed default clock/load regimes, not a controlled clock comparison.
See [clock-events-by-run.json](clock-events-by-run.json).

All three lengths have five complete valid **bench** ABBA blocks. The unchanged
F1 ten-percent-effect planner nevertheless requests 975 / 355 / 110 additional
blocks for 128 / 384 / 1024. That small-effect claim is **not made**, and no such
unbounded top-up was run. Full calculations and individual step times are in
[analysis.json](analysis.json); every scheduled outcome is in [runs.csv](runs.csv).

## Hardware and recipe

- RTX 3060 Ti, physical 8,589,934,592 VRAM bytes, UUID
  `GPU-b1248e58-f621-afa5-d5f4-b27c16d75b40`, driver 610.57.04. CUDA reports
  8,219,852,800 available total bytes; no 24 GiB card/process-limit substitute.
- Physical host RAM 33,364,897,792 bytes, **container limit 22,648,193,024 bytes
  (21.1 GiB)**. The roughly 32 GB host must not be described as 32 GB usable
  inside this container. The successful NF4 RAM store fits this actual allocation.
- SK hynix `HFS256GD9TNG-62A0A`; this host has two NVMe devices. Before the NF4
  cohort, every one of the 51 actual cache safetensors files was fully O_DIRECT
  read: 9,931,939,840 device bytes on `nvme0n1`, zero on `nvme1n1`. Cache payload
  is 9,931,800,632 bytes. The default unmapped overlay resolver stays unknown.
- Fresh F1 clean calibration returned ok. A separate SSH session directly read
  1,207,959,552 bytes from the actual cache; foreign-read detection returned void.
  The one-GB F1 threshold and four-second counted-trace gap rule were unchanged.
- Public checkpoint `Qwen/Qwen2.5-14B-Instruct`, revision
  `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, 14,770,033,664 base parameters.
  SFT/Transformers, NF4 streaming, pinned RAM or forced verified NVMe, batch 1,
  accumulation 1, seed/data seed 42, LoRA r=16/alpha=16 on q/v, dropout 0,
  AdamW LR 2e-5, weight decay .01. Resolved defaults/config hashes are frozen.
- Actual timing stack: Python 3.11.13, torch 2.8.0+cu128, transformers 5.17.0,
  trl 0.29.1, peft 0.20.0, bitsandbytes 0.50.2. The separate CPU test environment
  was not used for measurement. The observer delegates to the normal CLI,
  model, loss, trainer and optimiser; synchronisation/hashing are observation
  overhead shared by the two CLI paths. No backward-exactness claim is made.

## Invalids, saved weights and the capacity boundary

Both NO_VERDICTs are **ordinary train**, records 005 (NVMe/128) and 145
(NVMe/1024). They completed training, but F1 could not place all own reads on
the selected device while another disk read 1,001,037,824 and 1,797,414,912 bytes,
respectively. Suspend/power checks were ok. The records remain excluded;
unknown is not rounded into clean, and neither record is replaced or pooled.

All six final saved adapters contain 192 finite tensors / 12,582,912 parameters,
with all 96 LoRA-B tensors nonzero. Actual Windows file hashes match the server
files, and each file matches its last train runtime fingerprint. Five also
match at least one F1-valid train fingerprint. **The final NVMe/1024 adapter is
from F1-unknown record 145 and does not match a retained F1-valid train hash.**
It is verified as an actually updated/saved artifact, not a valid performance
sample or a quality-qualified model. Other F1-valid NVMe/1024 train records are
retained in the raw data. See [saved-adapters-valid-run-matches.json](saved-adapters-valid-run-matches.json).

For the separate natural-capacity recipe, the shipped sharder prepared a real
BF16 store of 29,540,125,474 bytes in 120.89 s, without OOM or process kills.
Its first full-file mapping attempt also saw 4,096 other-device read bytes; the
one declared preparation retry saw 282,624. Both violate the preregistered
zero-other-device criterion. Both proofs are retained, the criterion was not
relaxed, and **zero auxiliary CLI measurements were run**. There is no claim of
automatic NVMe overflow, planner capacity refusal, physical OOM, or physical HDD
performance from that preparation. See [auxiliary.json](auxiliary.json).

## Validation and provenance

The full server CPU suite is **not green**: 35,045 passed, 384 skipped,
12 deselected, 2 xfailed, **2 failed**, coverage 86.57%. Failures are the nested
MLX-worker venv missing pydantic, and CPU BF16 fast-MLP tolerance `[4]`. Both
reproduce at pre-telemetry source `a03be3b3`; the relevant source/tests were
unchanged. The F2-specific 35 tests passed.

The actual CUDA-runtime subset is **166 passed, 1 skipped, 4 failed**. The same
four failures reproduce at `a03be3b3`: two tests assume `active_bytes.current`
host-allocator stats absent in torch 2.8, and two characterise a stale CUDA error
that this driver/runtime already clears. Functional pinned multi-arena identity
and page-lock fallback/recovery tests pass. No test was hidden or patched to
obtain green. These validation contexts were pinned before measurement.
Full source/scripts/tests/benchmarks ruff and validation of 325 changelog
fragments passed. See [validation.json](validation.json) and the retained logs.

Evidence exposed three narrow repairs: BF16 raw-byte fingerprints (`1203a9f7`),
forced-disk media validation even when the planner initially fits RAM (`a03be3b3`),
and exited-child process-census handling (`e6829f15`). Missing roots or inaccessible
live processes still invalidate telemetry. Model/loss/optimiser arithmetic and
F1 acceptance rules were unchanged. Regression red/green records are retained.

Earlier cohorts are excluded: the original machine's paused 32 records use a
different GPU/driver; the replacement `f2-campaign-relocated` has an invalid
checkpoint-vs-cache mapping; `f2-campaign-cache-mapped` is the old sampler pilot.
Raw original statuses/partial attempts are preserved rather than rehabilitated.
Only the fresh 150-command `f2-campaign-telemetry-fixed` cohort enters these tables.

## Evidence, replay and cleanup

[records.tar.gz](records.tar.gz) retains raw logs/traces/native reports/configs,
all failed attempts and controls, source bundle and postprocessing scripts.
Tensor blobs are excluded from this lightweight repository archive. Full local
packets with tensor blobs are hash-indexed in [artifact-manifest.json](artifact-manifest.json).
Public model/tokenizer metadata and cache indices are included as a nested
metadata archive, checked against frozen hashes where applicable.

Records archive: 7,399,404 bytes, SHA-256
`ca2215cc50d48e2c06cd4cfeb53bece6d0f145cc04cb7ff9dfdec2438d29fd08`. [CHECKSUMS.sha256](CHECKSUMS.sha256) covers the report files.

After extraction, from an environment containing the measured source/dependencies:

```bash
python reproduce/audit_collected.py phases/f2-campaign-telemetry-fixed --harness /path/to/measured/benchmarks/harness
python reproduce/analyse_collected.py phases/f2-campaign-telemetry-fixed --harness /path/to/measured/benchmarks/harness
```

Both task rentals were deleted **after** actual local packet/metadata/output
SHA verification; API absence was confirmed. Conservative accounting is
**$3.30**, including $0.75 transfer reserve
and an extra minute of deallocation margin, within the authorised $5. It
overprices stopped intervals and is **not a settled invoice**. Pricing units
follow the [official Vast CLI](https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py).
See [budget-and-cleanup.json](budget-and-cleanup.json).

Automatic approval review blocked optional deletion of the local task-only
SSH private key with `blocked by policy`. It remains in ignored `out/f2-rental`;
the two associated rentals are deleted. No key is included in the evidence archive.
No PR, package publication or release was created by this experiment.

The legacy root-level `f2-clean.json`, `f2-foreign.json`, `f2-overlay-default.json`,
`f2-disk-mapping/` and `f2-pip-freeze.txt` describe the original machine. The fresh
cohort's controls/hardware are explicitly named inside the records archive.
