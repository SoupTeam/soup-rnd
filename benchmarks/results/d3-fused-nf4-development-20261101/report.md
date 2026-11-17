# D3 packed fused NF4: synthetic development, 2026-11-01

This packet records development on Vast instance 58638000, one RTX 3090 24 GiB
(sm_86), driver 580.95.05, Linux, torch 2.14.1+cu130, Triton 3.8.0,
bitsandbytes 0.50.2, transformers 5.19.0, PEFT 0.20.0 and TRL 0.29.1.
Sources were uncommitted development snapshots based on `55d5f9b7`.
Original launch packs are retained, including unsuccessful versions.
No pretrained fused correctness/performance session has started.

## Implementation and observations

The candidate reads original B1 NF4/nested-NF4 packed bytes, decodes tiles to
BF16 inside a grouped CUDA WMMA mainloop and accumulates in FP32. It implements
forward and transposed dX without a full dense weight tensor on that mainloop.
The explicit hybrid policy uses native restoration/MM for 1–128 rows, fused
WMMA for 129–512 and skips empty experts. Fused/native counts stay separate.
Native SiLU, route multiplication and ordered scatter remain unchanged.

| Development screen | Retained result |
| --- | --- |
| Raw Triton attempts 01/02 | Group/singleton comparisons pass, but native bitwise comparisons fail. These are unsuccessful candidates. |
| Raw WMMA attempt 03 | Some small-row native comparisons fail. No universal exactness claim. |
| Hybrid attempt 04 | All 20 finite arithmetic/control cells pass: both projection geometries, both codecs, forward/dX, singleton/grouped and independent native bits, empty experts and effective/restored corruption controls. |
| Dispatch attempt 01 | Instrument incorrectly counted a GPU memcpy as a second compute kernel. Preserve the original rejection and trace. |
| Dispatch attempt 02 | Eight profiles pass: one actual `fused` compute kernel per eligible four-expert group, unchanged inputs/NF4 state and allocator delta within the output/descriptor bound. The descriptor HtoD transfer remains visible. |
| MLP attempt 01 | Four plain-NF4 cases pass before nested-NF4 width-one construction refuses an allocator delta above its exact requested budget. This attempt remains an error, not a complete MLP pass. |
| MLP attempt 02 | Eight cases pass with the declared 64 MiB weight ceiling: plain/nested NF4, widths one/four, balanced/mixed routes, exact outputs and input/router gradients, nonzero derivatives and expired-lease refusal. |
| Public synthetic trainer matrix 01 | Eight fresh children pass: sequential/fused x plain/nested NF4 x ordinary/L2L. Actual saved gradient and trained-logit files agree byte-for-byte, final adapter records agree, and same-recipe reload succeeds. |
| Public synthetic trainer follow-up 08 | The same eight-child matrix passes after correcting preflight to reserve the configured allocator ceiling. Its executed runtime files match this publication's current runtime source; actual tensor bytes are independently rechecked. |

The trainer fixture is a random two-layer OLMoE with H=2048/I=1024, four experts,
vocabulary 64 and attention q/v LoRA. Three 512-token rows with accumulation two
produce two optimizer steps, including a one-row final window. It is not the
sixteen-layer trained OLMoE checkpoint. These short development runs do not
exercise the full prospective A/A/negative/restored/F1 trained-model gate.

In the MLP screen, the four-slot pool actually allocates 14,157,824 bytes for
plain NF4 and 12,990,464 bytes for nested NF4. A nested width-one request of
3,247,616 bytes occupies 3,276,800 allocator bytes. The constructor checks the
actual configured ceiling; public preflight now reserves that whole ceiling
before construction. Packed pool, temporary small-row dense restoration,
activations, BLAS workspace and total trainer peaks are distinct quantities.
No whole-trainer memory saving or speedup is inferred from these numbers.

Selected CPU regressions on Windows pass: **154 tests**, 44 GPU deselections,
plus **four offline-verifier fault checks**. The selected set includes the packed
pool, lease/budget validation, existing grouped/derivative/public trainer
integration and the repository GPU-marker regression. Full-tree Ruff and
`git diff --check` pass. This is not a full repository pytest or new CI result.

## Retention and verification

- `development-results.zip` retains raw statuses, logs, native/split-K diagnostic
  attempts, actual trainer tensors/adapters, all dispatch traces and the original
  full-file manifest. It contains 255 entries.
- `development-sources.zip` retains the actual workspace sources and launch
  snapshots, including earlier versions; it contains 624 entries. Later cosmetic
  driver/assertion changes are not relabeled as those earlier executed sources.
- `allocator-followup.tar.gz` retains the corrected reservation runtime and the
  repeated eight-child matrix. Its manifest has 931 entries and its source
  receipt verifies 721 Python files. The synthetic fixture matches the original
  full archive byte-for-byte.
- The complete first archive, including synthetic fixture weights, is downloaded
  locally as `d3-fused-development-58638000-20261101.tar.gz` (195,925,831 bytes),
  SHA-256 `f4bc770291ac8a336a81a6a7d5f5deb3d293d108a82c80205648edcc0390fd46`.
  All 875 original manifest entries were verified before creating this compact
  publication. The full binary archive is not committed here. Omitted synthetic
  fixture payload names/hashes are explicit in `verification.json`; their
  deterministic generators and all result tensors remain published. Omitted
  fixtures are not advertised as publicly downloadable bytes.

From the checkout root, with Python and Rich installed:

```sh
PYTHONPATH=.:src python -m benchmarks.harness.d3_fused_nf4_verify --folder benchmarks/results/d3-fused-nf4-development-20261101
```

This CPU audit checks publication hashes, archive integrity, original retained
payload identities, unsuccessful-candidate retention, primitive result grids,
dispatch trace hashes, actual trainer tensor-byte comparisons, reload records
and the follow-up executed source receipt. It does not rerun CUDA arithmetic.

## Remaining delivery

The existing [234-cell staged trainer study](../d3-public-trainer-attempt-03-f781e927-20261030/report.md)
is complete in its own scope. This new fused implementation still needs its own
prepared actual-shard controls, committed executable prospective rule, trained
native/same-kernel resident/streamed correctness and paired staged/fused timing,
whole-trainer memory and actual-launch profiles. See the
[preparation plan](../../protocols/d3-fused-trainer/README.md).
No default switch, quality/convergence result, hardware purchase, two-SSD
requirement or final D3 acceptance is implied by this development record.
