# F2: 14B on a physical 8 GB GPU

Status on 2026-10-19: preparation and recipe diagnostics. **No completed F2
matrix or throughput claim.** The user delegated the matrix choice after receiving the proposal on 2026-10-19.
The operational matrix below is selected with the stated limitations.

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
| `stream_source: ram`, pinned store must be confirmed | NOT_RUN | NOT_RUN | NOT_RUN |
| `stream_source: disk`, verified real NVMe, explicit `stream_disk_kind: nvme` | NOT_RUN | NOT_RUN | NOT_RUN |
| `stream_source: disk`, explicit `stream_disk_kind: hdd` policy control | NOT_RUN | NOT_RUN | NOT_RUN |

128 is a short-shape control, 384 reproduces the #901 report's cap, and 1024
exercises a larger logits/activation footprint. These are selected caps; the
collector must record the actual supervised and total token counts.

Forced disk uses the shipped disk tier but **does not demonstrate automatic
RAM overflow**. The HDD override tests a named policy refusal, **not an actual
HDD or a physical capacity boundary**. If the user requires natural overflow
and an actual media/capacity boundary, amend and approve the protocol before
running the campaign. Do not allocate a RAM-filling process or change host
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

Use Python 3.10–3.12 with a CUDA training install. The rental selected for
preparation is RTX 3060 Ti 8192 MiB, driver 580.173.02, 33,537,638,400 host RAM
bytes, cgroup memory limit 32,194,428,928 bytes, WD `WDS500G3X0C-00SJG0` NVMe,
Linux/container. Rental id 56717237; only this task's instance may be stopped.
The $5 authorisation includes storage/transfers. A separate local watchdog
destroys this instance at 12 hours or an estimated $4.25 including a $0.75
transfer reserve; this is a conservative guard, not final settled billing.
Copy all artifacts locally before destroying it. The other running Liger
instance belongs to other work and must not be interrupted.

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
