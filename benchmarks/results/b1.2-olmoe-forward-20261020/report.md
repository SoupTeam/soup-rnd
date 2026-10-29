# B1.2 trained-layer forward gate, 2026-10-20

**12/12 cells PASS, with zero failures, unknown outcomes or replacements.**
Streamed forward matches the resident eager reference exactly on trained OLMoE
layers 0, 7 and 15 in CPU fp32, CUDA BF16, CUDA NF4 and CUDA nested NF4 modes.
All same-process different-weight negative controls produce different hashes.
Every cell loads each routed expert exactly once, in ascending expert-ID order,
and stays within the prospectively committed 64 MiB expert-weight budget.

## Frozen source and execution

- Source and protocol commit: [`99664b09`](https://github.com/SoupTeam/soup-rnd/commit/99664b09ad0b3238dc0a8a790f207f140239ea31).
- Model: `allenai/OLMoE-1B-7B-0924`, revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
- The committed protocol pins all four configuration/weight-file SHA-256 values,
  package versions, 512 input tokens, seed 1234, three layers, four modes,
  exact-equality rules, one attempt per cell and zero replacements.
- Preparation verified the complete checkpoint hashes before launch. Each child
  verified those hashes again and recorded the same source identity and protocol.
- Hardware: Vast instance 56902453, one RTX 3060 12 GB (native BF16, sm_86).
  Linux, Python 3.12.13, torch 2.14.1+cu130, transformers 5.19.0,
  bitsandbytes 0.50.2 and safetensors 0.8.0.
- The series ran from 15:02:02 to 15:05:17 UTC (20:02:02 to 20:05:17
  Asia/Qyzylorda). These are execution timestamps, not throughput measurements.
- Twelve distinct child PIDs executed sequentially. No failed or excluded arm is
  present in this recorded series; no replacement or automatic formal retry ran.

The real trained router selected top-8 routes for each of 512 synthetic normal
hidden states scaled by the trained layer norm. All 64 experts were hit in every
cell (4,096 token/expert routes); all were therefore fetched. Logical read bytes
and direct-I/O flags are recorded, without a physical-host disk-read claim.
For NF4, the resident reference is independently reconstructed from the same
stored quantized bytes and sidecars. Equality is to that reference, not a claim
that NF4 preserves the original BF16 checkpoint's outputs or model quality.

## CUDA memory and correctness

MiB means 2^20 bytes. Arena allocation includes allocator rounding. Peak allocated
and reserved values describe this isolated layer primitive; they exclude CUDA
driver/context memory and do not describe a complete model or training process.

| Layer | Mode | Result | Max abs difference | Arena allocated MiB | Streamed peak allocated MiB | Streamed peak reserved MiB | Resident peak allocated MiB |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | BF16 | PASS | 0 | 12 | 28.339 | 36 | 784.621 |
| 0 | NF4 | PASS | 0 | 16 | 32.339 | 40 | 784.621 |
| 0 | NF4 nested | PASS | 0 | 16 | 32.340 | 40 | 784.622 |
| 7 | BF16 | PASS | 0 | 12 | 28.110 | 36 | 784.370 |
| 7 | NF4 | PASS | 0 | 16 | 32.110 | 40 | 784.370 |
| 7 | NF4 nested | PASS | 0 | 16 | 32.111 | 40 | 784.371 |
| 15 | BF16 | PASS | 0 | 12 | 28.145 | 36 | 784.339 |
| 15 | NF4 | PASS | 0 | 16 | 32.145 | 40 | 784.339 |
| 15 | NF4 nested | PASS | 0 | 16 | 32.146 | 40 | 784.340 |

The three CPU fp32 cells also pass exact equality and negative controls; their
single weight arenas are 24 MiB. CPU RSS was not measured by this gate. The
planned CUDA NF4 arenas are approximately 15.375 and 15.347 MiB, rounded by
PyTorch's allocator to 16 MiB; both the plans and measured allocations are
below 64 MiB. Dense and packed arrays, sidecars, code tables, nested-statistics
scratch and padding are included. Activations and backend workspaces are
accounted for separately by the recorded peaks.

## Archive verification and replay

[`raw-results.zip`](raw-results.zip) retains the unmodified downloaded archive:
the committed source snapshot, protocol, preparation receipt, launch receipts,
launcher, all 12 child logs, result records, final progress records and raw hash
manifest. The reconstructed `experts.safetensors` outputs remain on the server;
their SHA-256 values are recorded and their reconstruction is reproducible from
the pinned source/checkpoint. They are not included in the download archive.

The CPU-only [`analyze.py`](analyze.py) verifies both ZIP CRCs, all 44 raw manifest
hashes, all 603 archived source files against their Git blob identities at the
frozen commit, seven runtime source SHA-256 values, the committed protocol,
checkpoint identities, fresh-process records, routing/trace agreement, negative
control hashes, output hashes, logical reads, progress and memory accounting.
It regenerates [`analysis.json`](analysis.json) byte-for-byte. This is an audit of
retained evidence; no second CUDA execution or reconstructed output calculation
is claimed by the analyzer. The four checksum claims concern the recorded
checkpoint hashes; the 14 GB checkpoint is not bundled for a second local hash.

From a repository checkout containing commit `99664b09`:

```bash
python benchmarks/results/b1.2-olmoe-forward-20261020/analyze.py --archive benchmarks/results/b1.2-olmoe-forward-20261020/raw-results.zip --output /tmp/b12-analysis.json
python -m pytest -o addopts= benchmarks/results/b1.2-olmoe-forward-20261020/test_analyze.py -q
```

All nine archive-control tests pass locally. They include rejecting changed
outputs, ineffective negative controls, excess memory, missing expert loads,
changed protocol, PID reuse, changed source and a missing cell, even when the
altered ZIP's own hashes are recomputed. Numerical results and raw records are
preserved unchanged. `SHA256SUMS.json` covers the published files except itself;
local attributes fix text files to LF for consistent fresh-checkout hashes.

## Scope and next acceptance step

The prospective forward/arena gate is satisfied for these trained layers on
this GPU. The separately retained seven-test CUDA preflight covers B1.1's codec
and B1.2's tiny DeepSeek/Kimi native-block, shared-expert and forward cases.
The preceding CPU regression run passed 199 tests with 48 hardware skips and
eight existing PEFT pattern warnings. Publication adds CPU archive controls;
it does not rerun the complete repository or claim current-head CI success.

This remains forward-only. It does not establish backward correctness, training,
convergence, throughput, fewer disk reads, physical-host isolation or whole-model
fit on an 8 GB GPU. OLMoE layer extraction exercises the generic expert primitive;
it does not add OLMoE to the RND checkpoint sharder. B1.3 backward and B1.4 public
planner/CLI integration remain separate tasks. B1.1 dependency review, the B1.2
PR/changelog, required CI and reviewer acceptance remain separate from the
successful measured gate; this report does not move the task board to Done.
