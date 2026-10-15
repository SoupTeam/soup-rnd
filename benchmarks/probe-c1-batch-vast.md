# C1: batch control on a synthetic 70B-shaped store, Vast RTX 3090

## Result and scope

Batch 3 is the fastest of batches 1, 2 and 3 on this measured cloud instance:
124.735 input tokens/s, versus 48.046 for batch 1 and 95.446 for batch 2.
All three fit without OOM. This compares ordinary Soup layer streaming without L2L.
It does not establish model quality, the largest possible batch, or performance on an 8 GB GPU.

This is a synthetic cloud measurement with guest-visible disk validity controls.
**Physical NVMe, cold physical-host reads and host isolation are not certified.**
The task's reference NVMe/8 GB GPU regime therefore remains a separate acceptance decision.

## Protocol and sources

The decision rule and runner patch were committed before this new Vast training series:
`4c2fcf8f326a764047fe0871a3d1da8a331f1894`.
The verbatim rule is [protocol.json](results/c1-vast-20261015/protocol.json).
Local verification compared that Git object with every block's protocol and checked that
its commit timestamp precedes the first training arm. Git timestamps are not an independent
timestamp authority. Earlier Colab results were known when designing this new protocol,
but are excluded from all statistics below. Preparation and sensor controls precede the series.

The packaged historical Soup snapshot is `8387417e9167494f5c0f864568b536f74482e489`;
the F1 module is pinned to `614229d66d62522984f36996fd3771cf663015ce`.
Source hashes, not a current-main checkout, identify the executed code. No production code
or training default is changed by this documentation submission.

Soup's sharder prepares the NF4 layers, `build_streamed_model` builds the streamed model,
and the existing `stream_probe.py` instruments synchronized training steps.
The external wrapper schedules fresh processes and injects only the calibrated disk resolver
into F1; it preserves F1's counters, thresholds and stop-on-invalid behavior.

## Fixture, recipe and machine

- Synthetic Llama-shaped fixture: 80 layers, hidden 8192, intermediate 28672,
  64 attention heads, 8 KV heads, vocabulary 32768; 68,989,231,104 parameters.
- NF4 with double quantization; cache size 36,388,404,614 bytes, larger than guest RAM.
  Preparation took 654.866 seconds. Tiny full-versus-chunked NF4 cache gate: 95 tensors equal.
  This is a cache-equivalence check, not an independent forward/backward parity experiment.
- Sequence length 512; batches 1/2/3; attention q/k/v/o targets; rank 8, alpha 16,
  dropout 0; PagedAdamW8bit, learning rate 1e-4; model seed 3, input seed 17.
- Disk tier, two buffers, read-ahead 2, pinned staging; direct I/O true in all accepted arms.
- RTX 3090, 24,576 MiB; GPU UUID recorded in raw plans. Driver 595.91.07.
  Guest MemTotal 30,728,568 KiB; cgroup memory limit 30,206,328,832 bytes; no guest swap.
- PyTorch 2.11.0+cu128. Full installed versions are in raw `work/packages.txt`.
  The exposed disk is QEMU `sdb`, reached through overlayfs, not an exposed physical NVMe.

## Design and measurements

Five rounds each compare 1/2, 1/3, then 2/3; every comparison is A-B-B-A in fresh processes.
Each arm has one warmup and three timed steps. The arm statistic is the median timed step;
throughput is `batch * 512 / median_step_seconds` (input tokens, not shifted-label targets).
Timing includes synchronized forward/backward/optimizer work, excluding initial setup.
The table averages arm statistics over 20 accepted arms per batch.

| Batch | Mean arm-median step, s | Mean input tokens/s | Throughput range | Max allocated GB | Max reserved GB | Max sampled process RSS GB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 10.656479 | 48.045973 | 47.840378–48.073774 | 4.377839 | 4.798284 | 3.959722 |
| 2 | 10.728971 | 95.445515 | 93.565663–95.659981 | 5.344831 | 5.511315 | 3.929571 |
| 3 | 12.314186 | 124.735267 | 124.047405–125.653569 | 6.311823 | 6.608126 | 3.925434 |

GB means decimal 10^9 bytes. Allocated/reserved are PyTorch allocator peaks, not total VRAM.
The maximum observed whole-device GPU usage was 4980 / 5660 / 6706 MiB respectively,
sampled roughly once per second across the process. Process RSS was sampled every 0.1 s;
it is not an exact peak or peak whole-host RAM. F1 also records available RAM and commit
charge before/after each arm. No unsampled system-wide peak is claimed.
Copy/stall event instrumentation was disabled: zero event counters mean unmeasured, not zero cost.

## Repetitions and operational recommendation

The committed F1 rule uses pooled within-arm sample standard deviation and
`N = max(2, ceil(7.85 * (sigma / (0.10 * mean_A))**2))`, with at least five pilot blocks.
The pilot counts; N is recomputed once at the required count. For all pairs N is 2,
so the five completed pilot blocks also supply the recomputation and no additions are required.

| Pair A/B | Valid ABBA blocks | Pooled sigma, s | Formula N | Additional blocks | Mean block B/A throughput ratio | Ratio range |
|---|---:|---:|---:|---:|---:|---|
| 1/2 | 5 | 0.03696617 | 2 | 0 | 1.987884 | 1.980051–1.990059 |
| 1/3 | 5 | 0.03076268 | 2 | 0 | 2.600188 | 2.592443–2.608646 |
| 2/3 | 5 | 0.05446335 | 2 | 0 | 1.305635 | 1.302043–1.316549 |

Batch 3 wins every accepted pairwise block involving it. Use batch 3 among these three
choices for this recipe on this instance. The repetition calculation is not a p-value,
a multiple-comparison significance claim, or a guarantee on another machine.

## Validity and exclusions

The path-specific direct-read calibration mapped 64/192/64 MiB exactly to `sdb` with
zero idle reads. F1 controls returned baseline `ok`, own 128 MiB `ok`, and foreign
1280 MiB `void`. This validates the guest-visible instrument, not physical-host isolation.

All 60 accepted arms have successful process exits, matching result hashes, direct I/O,
finite measured losses and 640 changed adapter tensors. Required F1 suspend, power and
foreign-read checks are `ok`; before/after GPU-peer snapshots are empty.
`sw_power_cap` appears throughout the accepted series, alongside occasional `gpu_idle`.
This is retained power-limited operation, not a fixed-clock or throttle-free measurement;
no other clock-event reason was found by the archive audit. Guest power classification
uses F1's no-battery rule and does not independently certify physical mains state.

`c1-13-block05` is excluded in full, including its successful first arm. Its second arm
finished training but F1 found at least 5,131,763,712 bytes of foreign reads, exceeding
the 1,000,000,000-byte threshold. The reader was not attributable to a visible process.
The remaining arms did not run. All records remain in the raw archive. A fresh full block,
`c1-13-block05-replacement01`, passed and replaces it. No OOM was observed.

## Reproduction and evidence

Files are under [results/c1-vast-20261015](results/c1-vast-20261015/):

- `raw-results.zip`: C1 raw JSON/logs, including the excluded block, controls, preparation,
  setup, resource probe and installed versions. Unrelated E3 training records are omitted.
- `analysis.json`: audited arm records, aggregates, repetition calculation and exclusions.
- `protocol.json` and `runner.zip`: the precommitted rule and C1 wrapper patch.
- `reproduction.zip`: historical base runner with its source manifest, C1 patch and analyzer.
- `provenance.json` and `SHA256SUMS.json`: source/commit evidence and artifact integrity.

To independently analyze the retained evidence (standard-library Python):

```bash
python analyze_vast_c1.py raw-results.zip --output checked-analysis.json
```

To reconstruct the execution directory, extract `reproduction.zip`, then extract its
`vast_runner.zip`, then `c1_measurement_patch.zip` inside the resulting `vast_runner/`.
The original Linux preparation commands were:

```bash
python3 run.py probe
python3 run.py setup
python3 run.py c1-gate
python3 run.py c1-prepare --exploratory-c1
.venv/bin/python diagnose_c1_overlay.py
.venv/bin/python c1_overlay_control.py
```

Following the prospective commit, each block used:

```bash
.venv/bin/python run_overlay_c1.py --pair 1,2 --id c1-12-block01 --protocol-commit 4c2fcf8f326a764047fe0871a3d1da8a331f1894
python3 run.py bundle
```

Repeat the pairs and rounds in the protocol, preserving all failed attempts. The fixture
certificate includes metadata that can differ on regeneration. On a new machine, prepare
and calibrate first, then prospectively commit its new certificate/source identities;
do not bypass the refusal or reuse the historical SHA for modified protocol contents.
The 36.4 GB payload is not included, so payload hashes cannot be recomputed locally from
this submission alone; generation hashes, certificates and source code are retained.

## Acceptance boundaries

The comparison and prescribed repetitions are complete for this cloud recipe. It is not
a reproduction of the reference 8 GB GPU / 32 GB RAM / physical NVMe system. Guest direct I/O
does not rule out host/hypervisor caching or other physical-disk tenants. Model weights and
tokens are synthetic; loss is a sanity check, not quality evidence. No current-main GPU run,
L2L performance gain, production change or generalized hardware recommendation is claimed.
Reviewer acceptance of this hardware scope is separate from the completed measurements.
