# G2 GPU sessions, 2026-10-31 to 2026-11-01

**Verdict: `DO NOT BUILD: recompute is as fast or faster on both tiers`** (session 2, rule
v2). Storing each decoder layer's inner activations is bit-exact but slower than
recomputing them, in pinned RAM and on the spill drive. Rule:
`../../probe-g2-recompute-vs-spill.md` §3; record: §9 there.

Fixture: `unsloth/mistral-7b-instruct-v0.3` at `60b99dcb…`, NF4 streamed, LoRA r = 8
q/k/v/o, seq 512, timed arms at k = 4, SDPA. Both boxes are vast.ai KVM VMs. The measured
weight and spill paths resolve to virtio `vda`, vendor 0x1af4; the box inventory also
contains `vdb`. G4 and the break-even describe the measured guest disk at that time,
not a physical NVMe. V12 records this scope without changing a gate.

## Gates

| gate | session 2: RTX 3090 VM, rule v2 | session 1: RTX 4090 VM, rule v1 |
|---|---|---|
| G1 exact | **pass** (k = 1, 2, 4, deterministic reruns) | pass |
| G2 bytes | **pass** | pass |
| G3 RAM tier, B/A | **fail**: 1.244, CI [1.243, 1.245], 5 blocks | fail: 1.860, CI [1.858, 1.862] |
| G4 disk tier, B/A | **fail**: 1.705, CI [1.637, 1.772], 5 blocks | no verdict: 5.99, CI [5.29, 6.69], 20 blocks; V7 asked for 2,021 more |
| verdict | **DO NOT BUILD** | NO VERDICT |

- **G1**: on both boxes the default runs' GA vs GA was not reproducible (all 250 LoRA
  gradients differ, losses equal), so G1 read the predeclared `--deterministic` reruns:
  GA vs GA, L2L vs GA and both stores (pinned RAM, direct spill) vs GA and vs L2L all
  `torch.equal`. The identity-hook control raised "found a recycled weight slot" at k = 2, 4
  and gave different gradients at k = 1; the corrupted record changed the gradients, not the
  losses.
- **G2**: record capacity 104,996,864 B, 205,064 B/token/layer saved on CUDA SDPA (the CPU
  shape estimate's number); k = 4 spill wrote = read = plan 13,439,598,592 B per step,
  direct, 0 weight bytes, no file left.
- **Rule v2** changes only V7 (each arm's spread over its own mean). It was written after
  session 1's data and committed before session 2 (`5027d2ad`); session 1 stays NO VERDICT
  under v1. B running about 5x A with about 25% relative noise made v1 request 2,041
  blocks in total, beyond session 1's cap of 20, not mathematically unreachable.

## Why the store loses (device-synchronised profile, k = 4)

| | recompute saved (s/step) | store packing + copies (s/step) | drive waits put + get (s) |
|---|---:|---:|---:|
| RTX 4090, RAM tier | 0.434 | 1.036 | 0 |
| RTX 4090, disk tier | 0.487 | 1.050 | 1.235 + 1.534 |
| RTX 3090, RAM tier | 0.808 | 1.102 | 0 |
| RTX 3090, disk tier | 0.848 | 1.135 | 0.015 + 0.149 |

The store's own work exceeds the recompute it removes before any drive I/O is counted.

## Break-even, measured (RTX 4090 VM, V9 and V11 hold)

Sustained I/O session on the same box and disk (`disk-rtx4090/`): 10 arms, each 600 s warmup
+ 1800 s measured, fixtures 63.6 GB > 50.9 GB RAM, all `candidate`. Under contention: spill
write 2.155 GB/s, read 3.110 GB/s; weights 5.767 GB/s alone, 1.798 GB/s beside the spill.
Applied to session 1's profile (`session1-rtx4090-breakeven/verdict.json`):

- RAM tier: serial margin **-0.60 s/step**.
- Disk tier: write 6.24 s + read 4.32 s for 13.44 GB + weight contention 2.63 s; serial
  margin **-13.75 s/step**, overlapped **-12.71 s**. Holding the measured compute and
  packing/copy costs fixed, increasing drive bandwidth alone cannot break even: the
  store overhead already exceeds the saved recompute. This is not a claim about other
  implementations, hardware, or profiles.

## Machines and cost

| | machine 1 | machine 2 |
|---|---|---|
| instance | 58578846, RTX 4090, gpu_frac 1.0 | 58588456, RTX 3090, gpu_frac 0.25 |
| guest | 47.4 GiB RAM, 15 CPUs | 49.3 GiB RAM, 30 CPUs |
| code | `40aea9b9` (rule v1) | `5027d2ad` (rule v2) |
| work | session 1 + sustained I/O | session 2 |
| deleted (API confirmed) | 2026-11-01T05:15Z | 2026-11-01T02:14Z |
| posted charges | $4.494 | $1.513 |

Total **$6.007** (instance-scoped invoices, posted right after deletion; not claimed final).
Session 2's first child (`shapes`, CPU only) took 2 h 19 min against < 30 s on machine 1;
the cause is not established and no gate reads it.

## Publication selection, frozen source and replay

The publication transfers selected files from research snapshot `b315e8dc` onto current
main. It includes every session JSON record in `session1-rtx4090/` and
`session2-rtx3090/`, the sustained-I/O manifest/rates/arm records in `disk-rtx4090/`, and
the historical break-even verdict and exported rates in `session1-rtx4090-breakeven/`.
All 178 numerical evidence JSON files and `provenance.json` retain their published bytes;
`SHA256SUMS` is unchanged. That manifest excludes itself and this README.

Two directories added after review carry their own `MANIFEST.json`:

- `frozen-source/`: the 7 harness files (`g2_disk`, `g2_inner_activations`, `g2_probe`,
  `g2_rule`, `g2_session`, `l2l_schedule`, `rental_guard`) as git blobs at each measured
  commit, stored as `*.frozen`. They are the only files of the 634 that `protocol.json`
  hashes (`source_sha256`) which differ from public commit `fdf067fc`; the other 627 are
  read from that commit.
- `raw-records/`: per-run `logs/`, `preflight/`, the runlog (kernel journal, session
  start/exit, fixture hashes, `pip freeze`, test log, spill-left-at-end) for both
  sessions, and the ten sustained-I/O `samples.jsonl.gz` (per-second counters, gzip with
  mtime 0). Only the box prefixes `/root/g2/soup-rnd` and `/root/g2` (to `<work>`,
  `<box>`) and one `PARTUUID` were changed; `MANIFEST.json` lists the raw
  and published SHA-256 of every file and marks the changed ones. Session 2 has no
  per-second samples; it ran no sustained-I/O arm.

Still omitted: rental scripts, API/invoice records, SSH details, the raw tar archives
and the CPU sessions. `provenance.json` lists the private archive hashes; the
`raw_sha256` fields in `raw-records/MANIFEST.json` tie each published file to the private
bytes. `provenance.json` is unchanged (`SHA256SUMS` hashes it), so its `excluded` list
still names the logs, runlogs, preflights, kernel journal and `samples.jsonl` that
`raw-records/` now publishes in sanitized form.

### Replay

```bash
python benchmarks/harness/g2_replay.py
```

Needs `git` and this repository's history, no GPU, no network. For each session it

1. rebuilds the measured `src/` + `benchmarks/harness/` tree and checks all 634
   `source_sha256` hashes of that session's `protocol.json`;
2. runs that tree's own `g2_rule.py` on the published records and requires the result to
   equal the published `verdict.json`: session 1 `NO VERDICT` (rule v1) and session 2
   `DO NOT BUILD` (rule v2);
3. for session 1, recomputes the ten sustained-I/O summaries with that tree's
   `g2_disk.summarize` from `samples.jsonl.gz` and requires equality with each `arm.json`
   summary;
4. verifies both manifests.

It exits 1 and names the file if a frozen file or a samples file is altered (checked by
tampering one of each). It does not rerun kernels, I/O windows or the F1 sensors, and it
does not recompute the break-even verdict in `session1-rtx4090-breakeven/`. The raw logs
and journal are published for audit but are not inputs of the replay.

The measured commits themselves are still not on public history; this PR does not publish
that ancestry. The frozen files stand in for them and are checked by hash, not by fetching.

### Current main is not the measured runtime

The harness files elsewhere in this PR are an adapted implementation, not the measured
tree. Current main can select fused NF4 GEMM instead of the measured dequantize-plus-linear
path. This publication does not establish CUDA exactness, saved-byte counts, or timing for
that newer dispatch. The current rule-v2 evaluator recalculates session 2 from the
published records without CUDA:

```bash
python benchmarks/harness/g2_rule.py benchmarks/results/g2-gpu-20261031/session2-rtx3090
```

It rejects a missing or different protocol rule instead of silently judging it under v2, so
it refuses session 1; use `g2_replay.py` for that. Rule v2 was chosen after session 1's
results and before session 2, so session 1 must not be presented as a prospective v2
result.

### Scope of the disk result

Both machines' disks are virtio (`vda`, vendor `0x1af4`). The G4 and sustained-I/O numbers
describe that virtual disk at that time, not the physical dual-NVMe write rate of the card.

