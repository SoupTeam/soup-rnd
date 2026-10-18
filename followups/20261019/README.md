# D1 follow-up, 2026-10-19

Independent cohorts. Where present, raw.tgz contains exact executed source.tar,
run.sh, run.log and the complete results directory, including failed gates/tensors.
A startup-only failure instead has startup status/deadline receipts and no CUDA
execution claim; its prepared source was never treated as an executed source.
Neither cohort is pooled with the historical rejected RTX block. Read audit.json
and the study checkout's D1_ACCEPTANCE.md before interpreting any timing.

SHA-256 and sizes are in manifest.json. Both rentals are released. Cost summaries
are instance-scoped provider snapshots; billing can lag. Only the D1 instances
were managed. The original 265-file archive remains unchanged at repository root.

The source is Soup (Apache-2.0). Root LICENSE is included. Raw data do not grant a
broader support/throughput claim or close D1/#842. To replay, restore raw.tgz to a
new directory and use the pinned source and recorded package versions. CPU tensor
replay does not substitute for CUDA execution.
