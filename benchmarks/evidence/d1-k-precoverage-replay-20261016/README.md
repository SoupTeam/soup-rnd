# K historical snapshot: real consumer replay

This is the closed snapshot **before** the coverage continuation, not the final
K report. Numeric source: `87b3bc1747ff78ff0fe16b8bdd2fa2eb0105d77c`.
Physical GPU: A100-SXM4-40GB, `GPU-b0245e84-f202-902b-9331-f732ab021e57`.

Archive `d1-a100-k-results.tgz`: 177,094,108 bytes, SHA-256
`3ed056fd904cd44926a1a0cc369a33ccf6e7f53d4244c2d0c5a9cab85133caf8`.
The retained local copy is named `d1-a100-k-before-coverage.tgz`.
`full-proof-manifest.json` lists all 4,929 raw files and the archive part.
The archive is retained locally; it is not included in this directory.

On 2026-10-16 the archive was restored twice, including through the packet
consumer script with a separate checkout of the measured commit. Every file
size and SHA-256 passed. Both stored FP16/BF16 correctness replays passed,
including detection of the deliberately broken finite-gradient control.
All 760 projection witnesses were replayed: 688 exact native comparisons and
72 explicitly declared finite dense/fused arithmetic differences. The latter
are dense-baseline witnesses, not a relaxation for native/forced candidates.

`mechanism.json` verifies the source blob and untimed forward backend control
for the accepted q/o FP16 M=512 native/forced cell. Its counters do not profile
checkpoint-step dequantization cost. The auxiliary dense event reference has
no separate ceiling telemetry window and establishes no frequency-matched
ceiling multiplier.

**Measurement status remains INCOMPLETE:** 26 accepted comparisons, one
clock-invalid cell and 71 not yet run in this historical snapshot. No stopped
rental or final K completion is claimed. CPU replay verifies stored evidence;
it neither repairs invalid GPU timing nor substitutes for independent review.
The running continuation and later results must be assessed separately.

Reproduction uses the measured-source `d1_materialize.py` with the listed
archive filename beside the manifest, then `d1_full_report.py`,
`d1_nf4_study.py replay` and `d1_projection_replay.py`. Choose fresh output
directories and use `--hardlink-duplicates` to avoid duplicating identical bytes.
