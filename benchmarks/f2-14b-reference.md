# F2: 14B on a physical 8 GB GPU

Status on 2026-10-20: the selected NF4 matrix completed on a physical RTX 3060 Ti
8 GiB: **150 commands, 118 PASS, 2 NO_VERDICT, 30 EXPECTED_REFUSAL**. All 60
working native bench reports are valid, ten per RAM/NVMe/length cell. See the
[results, raw evidence and validation limits](results/f2-reference-20261019/README.md).
Automatic RAM overflow remains unverified: the separate BF16 preparation failed
its physical-mapping gate and no auxiliary CLI measurement was run. The user
delegated the operational matrix choice on 2026-10-19; its limitations remain.

## Requirements and dependency

- R&D source page 12 asks for the shipped `soup train`, through `soup bench train`,
  three disk conditions by three sequence lengths, GPU utilisation and clocks.
- [Upstream #375](https://github.com/MakazhanAlpamys/Soup/issues/375) specifies
  RAM, NVMe overflow and the refusal boundary, and at least five runs per cell.
  Neither source specifies the three lengths or how to create the boundary.
- [Upstream #901](https://github.com/MakazhanAlpamys/Soup/issues/901) is closed.
  Its reporter confirmed real Qwen2.5-14B training on an RTX 3070 8 GB after
  `614b7f64`. The local [#901 gate](gate-901-14b-on-8gb.md) primarily uses a
  random-weight shape; that fixture is not the checkpoint for this study.
- F1 [PR #3](https://github.com/SoupTeam/soup-rnd/pull/3) is merged into the
  selected base, `ca9be3cc9f11b91cba76d26f8a0d7ad18787c4ff`. The method is
  [METHOD.md](METHOD.md), implemented in `harness/arm_validity.py`. Availability
  of the code is confirmed; calibration on this rental remains a separate gate.
  F1 explicitly lists containers among its untested regimes.

Work is isolated from the dirty D2 checkout on `koko/f2-14b-reference`.
No D2/C2 source changes are included. Model/loss/optimizer arithmetic is unchanged;
the evidence work exposed repairs needed in BF16 fingerprints and forced-disk validation.

## Selected matrix, not requirements inferred from the PDF

| Disk condition | 128 tokens | 384 tokens | 1024 tokens |
|---|---|---|---|
| `stream_source: ram`, confirmed pinned store | 10 bench + 10 train PASS | 10 bench + 10 train PASS | 10 bench + 10 train PASS |
| `stream_source: disk`, verified real NVMe, explicit `stream_disk_kind: nvme` | 10 bench + 9 train PASS; 1 train NO_VERDICT | 10 bench + 10 train PASS | 10 bench + 9 train PASS; 1 train NO_VERDICT |
| `stream_source: disk`, explicit `stream_disk_kind: hdd` policy control | 5 bench + 5 train expected refusal | 5 bench + 5 train expected refusal | 5 bench + 5 train expected refusal |

128 is a short-shape control, 384 reproduces the #901 report's cap, and 1024
exercises a larger logits/activation footprint. These are selected caps; the
collector must record the actual supervised and total token counts.

Forced disk uses the shipped disk tier but **does not demonstrate automatic
RAM overflow**. The HDD override tests a named policy refusal, **not an actual
HDD or a physical capacity boundary**. A real capacity check needs a separate,
preregistered recipe whose model store naturally exceeds the available RAM;
its results cannot enter this NF4 matrix. Do not allocate a RAM-filling process or change host
memory, pagefile, power, clocks or cache settings to manufacture that condition.

## Recipe and immutable inputs

Checkpoint: `Qwen/Qwen2.5-14B-Instruct`, Hub revision
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` (resolved from the Hub API on
2026-10-19), 14,770,033,664 parameters. Use real safetensors, config and tokenizer,
not `synth_checkpoint.py`. `f2_stage_checkpoint.py` checks 60 GB free before
downloading, pins the revision and records provenance. About 29.6 GB of source
weights, roughly 10 GB of NF4 cache and at least 15 GB scratch headroom are
budgeted. Recheck actual cache size and free space on the target filesystem.

Prepared configs resolve through `SoupConfig`, the project's schema. SFT,
transformers backend, NF4 streaming, two buffers, two read-ahead stages, batch
1, accumulation 1, one epoch / 12 rows, seed/data seed 42, LoRA r=16,
alpha=16, q/v targets, dropout=0, LR 2e-5, AdamW, weight decay .01, no packing,
evaluation or warmup schedule. Each row has a brief prompt and a long,
human-readable assistant explanation of layer streaming. It is deterministic
throughput data, not a quality or convergence benchmark. Record its SHA-256.
The compute dtype and all remaining defaults must be frozen from the resolved
schema and the actual installed stack. No cross-dtype exactness claim is made.

`freeze` requires the protocol and observer committed in a clean tracked tree,
the approval reference, a reserved resource window with an expiry, physical
NVMe evidence and an `ok` clean F1 record. It hashes every checkpoint file,
dataset and resolved config, and records commit, Python, versions, driver,
GPU UUID and physical VRAM. `run` verifies these hashes before starting.
The public checkpoint revision is independent of the Soup code commit.

## Linking the two CLI paths

Every scheduled execution has a fresh `soup train` process and a fresh
`soup bench train --steps 12 --warmup 4` process using the same prepared config,
seed, data, device and cache. The command order alternates between executions.
`bench train` itself replaces output with scratch storage, disables evaluation,
logs each step and sets `save_steps=13`; record that resolved config/hash from
the native report. Train has the same logging/save interval, one epoch of 12
rows, and retains its final adapter. Accept only the observed 12 steps.

`f2_cli_observe.py` enters `soup_cli.cli` through its normal `run()` entry
point. For train it attaches Soup's existing `BenchCollector` after setup;
for bench it exposes the collector that the command already installs. It
does not replace the model, trainer, loss, optimizer or data. It records
before/after trainable-tensor SHA-256 fingerprints, each step's grad norm and
token counts, counted-window timestamps, setup peaks and training allocator
peaks. Its batch observation delegates to the original training step unchanged.
CUDA boundary synchronisation and fingerprints are observation overhead, shared
by both measured paths. Direct unobserved CLI commands are also recorded in
the run's `command.json` arguments after the observer separator.

Only native bench reports supply table throughput. Train is a separate
end-to-end update/adapter check. Never replace either with a forward/backward
microbenchmark. Adapter existence alone and nonzero gradients alone do not
prove updates. A fingerprint must change; a missing final fingerprint fails.

## Order and validity rule, written before measurements

Each of five blocks visits all three lengths. At each length run RAM, NVMe,
NVMe, RAM, then the refusal control. Thus RAM and NVMe each have ten executions
per length and refusal has five, each with both CLI commands: 150 processes.
This applies F1's five counted ABBA pilot blocks to RAM versus NVMe. Keep the
block means, medians, individual step times and pooled spread in seconds.
No 10% effect or superiority claim is allowed until `arm_validity.repeats`
has planned any required top-up on this machine. The initial matrix is
descriptive; an incomplete block is retained and excluded from a comparison.
There is no automatic replacement/cherry picking of failed or void arms.

| Validity row | Required evidence | Invalid outcome |
|---|---|---|
| Physical target | GPU UUID, torch total VRAM and nvidia-smi; 7.5–8.5 GiB physical; ~32 GB host RAM | NO_VERDICT; a process limit on 24 GB never qualifies |
| Suspend | F1 `suspend: ok` | NO_VERDICT |
| Mains | F1 `power: ok`; power/clock-event record retained | NO_VERDICT |
| Foreign reads | F1 `foreign_reads: ok`; 1 GB threshold, host counters and bounds retained | NO_VERDICT |
| Real storage tier | Ready line confirms pinned RAM or NVMe disk; no silent pageable fallback | NO_VERDICT |
| GPU/load telemetry | Samples inside the counted window, SM clock, utilisation, power, state, temperature, memory, process-tree RSS and I/O | NO_VERDICT if required evidence absent |
| Actual training | 12 observed steps, supervised tokens, finite positive counted grad norms, nonempty differing parameter fingerprints; native bench valid | FAIL if absent or unchanged |
| Fit | Setup and training allocated/reserved peaks <= physical VRAM | FAIL for spill or actual OOM |
| Ordinary train output | Newly saved adapter from this execution | NO_VERDICT if absent |
| Refusal control | Nonzero exit plus `needs NVMe or more RAM` in raw log | EXPECTED_REFUSAL; unrelated errors FAIL |

All three F1 checks are required. A void or unknown outcome is never treated
as slower performance. GPU throttling is recorded in F1's clock-event record;
F1 does not automatically void it. State that limitation and inspect it before
any between-arm claim. Do not transplant the RTX 3050 or battery-run noise
numbers into this study. Unknown sensor fields remain unknown.

In a container, `/workspace` may be overlayfs. The default F1 disk resolver
then returns unknown. An explicit `physical_disk` mapping is allowed only
after recording mountinfo, the single backing NVMe and diskstats. It changes
device resolution only; the original host-wide counters and F1 bounds stay
active. Containers' PID namespace may hide readers; disk-wide unattributed
reads still count. A clean calibration and a controlled foreign-read check are
needed before the mapping can support a performance verdict. Do not call
"no visible peer processes" proof of exclusive host disk use.

## Reproduction

### Amendment before the replacement campaign

The first campaign at `1203a9f7` demonstrated that a forced disk tier skipped
media validation whenever the planner first selected RAM. The HDD override
therefore trained instead of refusing. Four setup-path regressions reproduce
the bypass (HDD, SSD, unknown, and explicit HDD override); the RAM control passes.
The replacement source validates the forced tier before any shard writes.
The verified overlay-backed NVMe arm explicitly sets `stream_disk_kind: nvme`.
Lengths, data, seed, 12/4 steps, order and validity thresholds are unchanged.
The old campaign is retained separately and is never pooled with replacement
measurements. Its last child completed before the controller was stopped; that
arm has no finalised parent validity/result and cannot support a verdict.

### Commands and hardware

The user paused the first repaired campaign before closing the laptop. It kept
32 completed records (26 PASS, 6 EXPECTED_REFUSAL) and an interrupted arm. On
resume, Vast could not reacquire that GPU. Those records and the incomplete
arm remain separate; none is pooled with the replacement machine.

Replacement rental 56748309 has a physical RTX 3060 Ti 8192 MiB, driver
610.57.04, GPU UUID `GPU-b1248e58-f621-afa5-d5f4-b27c16d75b40`, host RAM
33,364,897,792 bytes and a **22,648,193,024-byte container memory limit**.
The allocation therefore has less usable host memory than the original rental.
Its 100 GiB writable volume is backed by SK hynix `HFS256GD9TNG-62A0A` NVMe.
Two NVMe devices are visible. The first replacement phase incorrectly used an
O_DIRECT **checkpoint** probe (`nvme1n1`) as evidence for the NF4 **cache**.
F1 later showed the workload's reads on the other device. That whole phase is
`INVALID_MAPPING`: keep all its records, including originally recorded PASS
statuses, and exclude all of them from the final matrix.

With training stopped, a full O_DIRECT read of all 51 actual cache safetensors
files (9,931,800,632 bytes) increased `nvme0n1` by 9,931,870,208 bytes and
`nvme1n1` by zero. Every individual file was on `nvme0n1`. The corrected clean
calibration returned ok, and a separate SSH read of 1.2 GB from the cache's
`large_embed_tokens.safetensors` made `foreign_reads` void. The unmapped overlay
result remains unknown. Identical overlay path prefixes are not evidence that
two datasets use the same physical disk.

This pre-telemetry pilot used Soup source `a03be3b3229ff25f640c6f6d9c37145111857de5`.
All 67 checkpoint/cache content hashes matched the prior frozen manifest;
the fresh download has its own provenance timestamp. Only after verifying
checkpoint bytes, eight source mtimes were restored from the verified cache
manifest, as an ordinary backup would preserve them. Neither the cache content
nor its fingerprint was rewritten. The largest RAM shape (1024) completed a
separate 12-step ordinary-train recipe check with counted updates and a saved
adapter. Its former storage mapping is not accepted as a final F2 verdict.

The corrected campaign starts all five blocks afresh (150 CLI processes). Its
controller requires the mapping to cover the exact set of all 51 cache files;
the plan pins that proof, the correction reason and the controller hash. The
frozen plan's "Normal OS caching" means shipped I/O defaults, with no manual
flush or artificial memory pressure. On Linux the shipped disk reader uses
O_DIRECT where supported; it is not a buffered page-cache benchmark. A forced
disk tier on verified NVMe does not establish natural RAM overflow or isolated
device bandwidth. Physical read counters are reported rather than assuming
every logical read reaches the device.

Use Python 3.10–3.12 with a CUDA training install. The rental selected for
the original preparation is RTX 3060 Ti 8192 MiB, driver 580.173.02, 33,537,638,400 host RAM
bytes, cgroup memory limit 32,194,428,928 bytes, WD `WDS500G3X0C-00SJG0` NVMe,
Linux/container. Rental id 56717237; only this task's instance may be stopped.
The $5 authorisation is shared by both F2 rentals and includes storage/transfers.
On resume the watchdog conservatively prices all pre-resume wall time at the
old active rate, then adds the new active rate, retained old storage and a $0.75
transfer reserve. It stops the task's running rentals at a $4.50 bound. A second
guard on the new rental now stops it twelve hours after creation even if SSH closes;
the measurement window ends ten minutes earlier. These guards preserve disks.
They are conservative estimates, not final settled billing. The original
destroying watchdog was cancelled when the user paused. Copy and verify all
artifacts locally before deleting either rental. Unrelated instances belong to
other work and must not be interrupted. After this completed campaign, both
F2 rentals were deleted following local SHA verification; the API confirmed
their absence. The guard records remain as provenance, not active rentals.

### Telemetry repair before the fresh complete cohort

The correctly mapped `a03be3b3` pilot on rental 56748309 retained 32 completed
records (21 PASS, 5 NO_VERDICT, 6 EXPECTED_REFUSAL), plus an interrupted arm.
It is separate from the earlier 32-record prefix on rental 56717237. The user
paused again, then explicitly resumed. Both disks were preserved during that pause.

The sampler previously discarded every process's RSS/I/O whenever one transient
child exited between the process census and its sensor reads. Two regressions
reproduce the loss at the memory and I/O reads. The repaired sampler captures
both readings per process, omits only a child known to have exited, and records
its PID. A missing root or an inaccessible live process still leaves the whole
sample unknown. Those three negative controls stay green. The four-second gap,
F1 thresholds and full counted-window evidence requirement are unchanged.

The fixed observer ran a fresh five-block, 150-command cohort. Neither the
pilot's accepted records nor its missing samples are reclassified into this
cohort. Model, loss and optimizer arithmetic are unchanged. Tests ran on the
rental before timing, with no concurrent acceptance suite or bulk collection
during the measurement campaign. The new commit, controller and twelve-hour
resource expiry were frozen before that cohort; the older frozen plans remain intact.

```bash
python benchmarks/harness/f2_stage_checkpoint.py --destination /workspace/qwen14b
python benchmarks/harness/f2_reference.py prepare --directory out/f2 \
  --checkpoint /workspace/qwen14b --cache "$PWD/out/f2-layer-cache"
python benchmarks/harness/f2_reference.py preflight --directory out/f2
# After human matrix approval and on-box calibration, fill plan.json's
# approval reference, NVMe evidence path, clean record and reserved window.
# Commit this protocol and all source changes BEFORE freeze/measurement.
python benchmarks/harness/f2_reference.py freeze --directory out/f2
python benchmarks/harness/f2_reference.py run --directory out/f2
python benchmarks/harness/f2_reference.py report --directory out/f2
```

Keep `plan.json`, `frozen.json`, preflight, package freeze, configs and dataset;
each execution's command/env, full stdout/stderr, native bench JSON, parameter
audit JSON, F1 JSON, one-second trace JSONL and result JSON; and `matrix.md`.
There is a 30-minute per-process ceiling and the reserved window is enforced.
Timeouts, OOMs, crashes and incomplete arms remain recorded; unexecuted cells
are `NOT_RUN`, with no invented errors or throughput. No evidence here yet
establishes quality, forward exactness or backward exactness at 14B.
