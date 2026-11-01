# D3 packed fused trainer qualification: preparation plan

Status on 2026-11-01: executable helpers and all preparation bindings are ready
for **operator commit and canonical export**. The execution switch is enabled
in the prospective rule; dirty/uncommitted source and missing canonical receipts
still prevent formal launch. No pretrained fused cell has run.

The [preparation record](../../results/d3-fused-trainer-preparation-20261101/report.md)
retains the two launcher development errors, completed 32-child synthetic gate,
748 native tensor checks, byte-identical analyzer replay and 141 CPU passes.

The completed [234-cell staged public-trainer study](../../results/d3-public-trainer-attempt-03-f781e927-20261030/report.md)
and new [synthetic fused development](../../results/d3-fused-nf4-development-20261101/report.md)
remain distinct. The existing `d3_trainer_series.py` accepts the old staged rule;
it must not be launched with a renamed JSON to claim fused qualification.

## Preparation before the next freeze

Items 1–5 below are complete. Item 6 has populated receipts/rule; operator
commit and canonical export are the remaining prerequisite to formal launch.

1. Commit the candidate implementation and development evidence. Export canonical
   Git bytes; do not execute a dirty working-tree overlay as that commit.
2. Reuse the existing public-trainer preparation, F1 mapping/control, capture,
   negative-control, replay and archive helpers as the foundation. Add explicit
   fused modes and a packed resident control with the same kernel/row policy;
   retain historical mode/rule validation and archive replay compatibility.
3. Download/verify the pinned `allenai/OLMoE-1B-7B-0924` revision
   `6d84c48581ece794365f2b8e9cfb043c68ade9c5` against the existing four checkpoint
   SHA-256 pins. Prepare NF4 and nested-NF4 expert caches, tokenized inputs,
   attention-LoRA initial factors and actual delivered-label denominators.
   Preparation does not execute the pretrained forward/backward.
4. Discover all actual data/shard devices, record guest boot/GPU/software/cuBLAS
   identities and run actual-shard baseline/own/foreign controls. Preserve any
   rejection; guest counters do not certify physical-host isolation.
5. Exercise the complete new parent/child launcher and offline analyzer on a
   synthetic fixture, including timeout, incomplete-grid, ineffective-control,
   numerical-mismatch, source-change and no-fused-dispatch refusals. Verify the
   exported packet can launch children using both checkout root and `src`.
6. Populate the final rule/receipts, publish the finite attempt decision and
   commit them **before** any new pretrained correctness/timing cell.

## Proposed bounded comparison

These parameters are implemented in `protocol.json` and the separate
`d3_fused_trainer_*` helpers. Commit them before any pretrained measurement.

- One RTX 3090 sm_86 with 24 GiB VRAM, at least 64 GiB host RAM and 100 GiB disk
  suffices for preparation. Check actual free capacity before construction.
  A second SSD is not a D3 prerequisite.
- Public Soup `SFTTrainerWrapper.setup/train`, ordinary SFT and A3 L2L; frozen
  routed experts and attention q/v LoRA. Both NF4 and nested-NF4 are separate
  cohorts. Preserve native attention, SiLU, routing, scatter and loss semantics.
- Preserve the previous seven 512-token rows, accumulation windows 2/2/2/1,
  four optimizer steps and the one-step warmup/three-step timing definition.
  Reuse the existing seeds, SGD, exact ignored-label denominator and initial
  factors after validating them against the installed software.
- Six correctness modes per cohort: native-resident/sequential,
  staged-resident/staged, fused-resident/fused. Resident means routed experts
  resident, with other layers still using the same streaming setup. Fused
  residents preserve original packed bytes; they do not decode all experts to
  a dense stack and call a different kernel. Native and same-kernel pairs must
  agree exactly, including signed zero, forward/input/router/LoRA derivatives,
  every optimizer update and saved/reloaded logits.
- Each correctness cell retains same-process A/A, the original fixed effective
  negative perturbation and restored replay. No allclose/noise tolerance,
  comparison-dependent fallback or retry on a new kernel is permitted.
- Timing compares the previous staged implementation against the new packed
  hybrid on **this same host**, in fresh-process ABBA blocks. Five pilot blocks
  per cohort, a prospective precision calculation, at most ten blocks, zero
  replacements and one session per frozen revision. The existing native-loop
  timing study remains historical context, not a new-host measurement.
- The pilot contains 24 correctness cells, 80 timing arms and eight separate
  CUDA profiles: **112 children**, excluding synthetic preparation. The ten-block
  ceiling is 192. The child cap is 1,200 seconds; the session cap is 129,600
  seconds. These are abort ceilings, not predicted completion times.
- Require positive fused work in candidate runs and actual fused CUDA kernel
  events in profiles. Count the native <=128-row fallback separately. Record
  wall/CUDA step time, input tokens/s, allocated/reserved VRAM, process RSS,
  pool/call budgets, actual restores/GEMMs/launches and route counts.
- Fail closed on numerical mismatch, incomplete controls, unknown/VOID F1,
  changed source/data/host identities, time/cell budget exhaustion or absent
  fused coverage. Preserve all attempted cells, successful or otherwise.

After execution, independently replay the analyzer, verify archive and canonical
publication hashes, update the acceptance map and run required repository CI.
Faster performance, less VRAM, convergence or final reviewer acceptance is not
inferred from passing the synthetic development screens.

## Canonical freeze and launch

From a clean operator-committed checkout:

```sh
PYTHONPATH=.:src python -m benchmarks.harness.d3_fused_trainer_protocol --export /path/to/new-fused-source.zip
```

The exporter reads actual Git blobs and includes the tested source, CPU tests,
rule and bound preparation receipts. It verifies the raw publication artifacts
before export. The bulky preparation ZIPs are audit artifacts, not dependencies
of every model child; canonical receipts/source and live prepared data remain
verified by each child.

Extract the packet to a fresh directory on the prepared host and use its root
and `src` in `PYTHONPATH`. After checking the source receipt and unchanged host:

```sh
python -m benchmarks.harness.d3_fused_trainer_series parent --prepared /root/d3-fused-data/inputs-01/prepared.json --out /root/d3-fused-data/inputs-01/formal-session-01
```

There is no resume/replacement command. A failed or unknown cell preserves an
incomplete session and stops. The operator, not this preparation phase, performs
Git commit/push; no formal process was launched during preparation.
