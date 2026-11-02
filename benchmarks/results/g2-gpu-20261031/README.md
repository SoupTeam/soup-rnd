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

## Publication selection and replay limits

The publication transfers selected files from research snapshot `b315e8dc` onto current
main. It includes every session JSON record in `session1-rtx4090/` and
`session2-rtx3090/`, the sustained-I/O manifest/rates/arm records in `disk-rtx4090/`, and
the historical break-even verdict and exported rates in `session1-rtx4090-breakeven/`.
All 178 numerical evidence JSON files and `provenance.json` retain their published bytes;
`SHA256SUMS` is unchanged. That manifest excludes itself and this README.

Private raw logs, per-second samples, runlogs, preflights, the kernel journal, rental
scripts, API records, and operational pilots are omitted. They remain in the local
research archives and are not dependencies of record recalculation. `provenance.json`
records their archive hashes, raw/published file hashes and the original path
substitutions. Raw-versus-published provenance therefore cannot be independently checked
without those private archives. The only CPU capture selected is
`../g2-cpu-pilot/shapes_seq512.json`, which supports the protocol's shape-estimate table;
the CPU sessions and operational smoke logs are not published.

Measured source commits are pinned in `provenance.json`:

- Session 1 and sustained I/O used `40aea9b92b6611735a05bec2862fcca0fa31f497`, rule v1.
- Session 2 used `5027d2ada3392593d064bbebf4e442da68c4b1c7`, rule v2.

Those commits exist in the local private research history. This PR does not publish that
ancestry or establish that the commits can be fetched from a remote. Source hashes in the
records identify measured files, but hashes do not supply their contents. A full replay
of the measured runtime or historical v1 output requires access to the pinned source and
environment. The current-main harness is an adapted implementation, not the measured tree.
Current main can select fused NF4 GEMM instead of the measured dequantize-plus-linear
path. This publication does not establish CUDA exactness, saved-byte counts, or timing
for that newer dispatch.

The current rule-v2 evaluator can recalculate the selected session-2 records without CUDA:

```bash
python benchmarks/harness/g2_rule.py benchmarks/results/g2-gpu-20261031/session2-rtx3090
```

This checks recorded values; it does not rerun kernels, I/O windows or F1 sensors. The
publication evaluator rejects a missing or different protocol rule instead of silently
judging it under v2. Session 1's historical rule-v1 verdict remains `NO VERDICT`, and its
break-even output remains as recorded; the current CLI does not reproduce their full v1
output. Rule v2 was chosen after session 1's results and before session 2, so session 1
must not be presented as a prospective v2 result.

