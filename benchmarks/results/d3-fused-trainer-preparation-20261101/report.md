# D3 packed fused trainer: preparation and launcher qualification

Preparation date: 2026-11-01. Candidate implementation is committed at
`d1e0e2ca523242c5aad7e7e7a3137f0bfa4d7892`. New qualification helpers and their
receipts were developed afterwards; they are not relabeled as that commit.
The operator must commit the new prospective packet before formal execution.

**No pretrained fused forward/backward or timing cell has run.** Trained-model
work so far consists of downloading/hash verification, creating caches/data and
initial adapters, and instrument controls. CUDA numerical qualification in this
packet uses a pinned random four-layer fixture, not trained OLMoE.

## Completed preparation

- Downloaded `allenai/OLMoE-1B-7B-0924` at
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`; all four original SHA-256 pins match.
- Prepared all sixteen trained layers in NF4 and nested NF4 through the existing
  public SFT sharder/setup, without invoking model forward. Retained exact initial
  attention-LoRA factors, seven 512-token rows, labels and Arrow metadata. Shifted
  nonignored labels are 465 per row and 930/930/930/465 per accumulation window.
- Discovered the actual weight-file mapping on guest disk `vda`. Baseline, complete
  own-read and independent sibling foreign-read controls produce OK/OK/VOID.
  The record concerns exposed guest counters, not physical-host isolation.
- Regenerated the original pinned random fixture: four untied OLMoE layers,
  H=2048/I=1024, four experts, top-k two, vocabulary 64, seed 71; weight SHA-256
  `541a84fa8513d02da3381a8c2fbc83efe4e3ff89975d958f0cc3505dc91b550d`.
  Its prepared NF4/nested-NF4 files also pass actual-shard controls.
- The third synthetic launcher session completes **32/32 distinct child
  processes**, all with F1 OK: 24 numerical cells and eight separate profiles.
  Both codecs and ordinary/L2L training cover native-resident/sequential,
  staged-resident/staged and packed-fused-resident/packed-fused pairs.
- Each numerical child runs the public trainer, A/A, the fixed effective negative
  perturbation, restored replay and same-recipe save/reload. Packed residents
  clone original packed bytes/statistics and use the same fused forward/dX and
  row policy as streaming. They are correctness-only, unbounded references.
- CPU replay independently verifies **748 native tensor payloads**, retained
  replay records, actual saved adapters, route/profile coverage and CUDA traces.
  The analyzer regenerates `analysis.json` **byte-for-byte** on Windows CPU.
  Its auditor source is retained separately from the original executed launcher.
- **141 selected CPU regressions pass**; the JUnit and current source hashes are
  bound in the prospective packet. Full-tree Ruff passes. This does not claim
  the full repository pytest/platform matrix or a new CI result.

Pre-hook visit counts can include checkpoint recomputations that stop before
all operations complete. The retained simple visit-to-launch diagnostic is not
an acceptance threshold; launch evidence uses actual CUPTI events. Complete
route geometry and positive fused coverage are checked separately.

The host is Vast instance 58638000, RTX 3090 24 GiB (sm_86), driver 580.95.05,
Python 3.12.16, torch 2.14.1+cu130, Triton 3.8.0, transformers 5.19.0,
bitsandbytes 0.50.2, PEFT 0.20.0 and TRL 0.29.1. GPU UUID, boot identity,
installed cuBLAS hash and software identities are bound in the preparation.
`software-inventory.json` also retains all 82 installed distribution versions
captured after synthetic validation and before any formal trained execution.

## Retained unsuccessful launcher development

1. Launcher 01 stops in parent validation before creating a session or starting
   any model cell: a checkpoint-loop variable shadowed the prepared-manifest path.
   Preserve its exit/log/source; the correction has a CPU regression.
2. Launcher 02 retains four successful numerical children and the fifth rejected
   fused-resident child. That new control lacked the `_shapes` interface required
   by the real routing validator. Its F1 outcome is OK, but the numerical child
   has no verdict. Correct the control and exercise the actual validator in CPU
   tests. Production kernel sources were unchanged.
3. Launcher 03 is a new development source/session and completes the full grid.

These are synthetic development iterations, not replacements in a formal
pretrained session. The prospective trained rule still allows zero replacements
and one session per committed source. The outer synthetic guard was four hours;
the session exited successfully under it.

## Publication and reproduction

Five self-contained ZIP parts contain content-addressed objects. `manifest.json`
maps **3,273 logical files to 1,209 unique payloads** and records all part hashes.
The parts total 309,983,204 bytes; each is below 100 MB. Identical bytes are
deduplicated, while logical names retain every run output, raw tensor, adapter,
trace, log, source snapshot, command and failed attempt.

The manifest explicitly lists **104 omitted model/checkpoint/cache payloads**.
Original trained weights are available through their pinned upstream checkpoint;
generated caches/fixture weights have their complete identities and generators
retained. Experimental outputs are retained. The omitted weight files are not
advertised as publicly downloadable archive contents.

From the checkout root:

```sh
PYTHONPATH=.:src python -m benchmarks.harness.d3_fused_trainer_publication --folder benchmarks/results/d3-fused-trainer-preparation-20261101
```

This reconstructs logical files, verifies every object/part/CRC, replays the
synthetic analyzer and checks current numerical/launcher source identities.
It performs no CUDA experiment. The materializer uses hard links for duplicate
payloads where supported.

## Remaining formal experiment

The [executable prospective packet](../../protocols/d3-fused-trainer/README.md)
is ready for operator commit and canonical export. Its pilot is 24 correctness
children, 80 fresh-process ABBA timing arms and eight profiles: **112 children**,
with a prospective ten-block ceiling of 192, no replacements and fixed time caps.
The complete staged and packed-hybrid backends have explicitly different high-row
dispatch profiles; timing does not isolate fusion alone.

After the operator commits/pushes, export Git blobs, verify the live host/data
identities, execute the finite trained-model series, replay/publish its results,
and wait for required CI/review. No speedup, whole-trainer VRAM reduction,
convergence result or final D3 acceptance is asserted by this preparation.
