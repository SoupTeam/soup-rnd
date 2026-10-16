# D1: the decision this research enables in Soup

Status: measured product decision, incomplete measurement acceptance. This
document does not claim a new speedup in production or closure of upstream #842.
The shipped checkpoint-visible NF4 implementation was contributed by Amix29
and merged before this study; D1 measures and checks it.

## Decision supported now

Retain bitsandbytes' native dispatch for checkpointed frozen-base NF4 training.
Do not add a product option that forces the Function at all training shapes
based on a claim of lower memory. In the valid fixed A100 RAM-tier TinyLlama
QA comparison, the forced Function took 368.821 ms/update versus 325.090 ms
for native dispatch: 13.45% longer, with identical measured allocated and
reserved peaks. All 20 fresh-process runs agreed exactly in final adapters,
losses and held-out predictions. The Function's large-M backward still
dequantises the dense weight to calculate dX.

This result helps avoid a concrete performance regression. It does not mean
the existing Soup release just became faster. The fixed tiny-data study is
neither an 8B/70B result nor evidence of general training-quality improvement.
See [the report and raw evidence references](d1-final-research-20261016.md).

## What the remaining measurements can change

| Evidence | Product decision | Acceptance limit |
|---|---|---|
| Native/forced comparisons at missing gate/up and down shapes | Check whether any measured operating region justifies a dispatch change | Native exact forward/dX, valid fresh-process blocks, actual kernel counters, lower step time or memory with stated tradeoff |
| Same-process sampled dense reference | Quantify remaining projection headroom on that GPU | Same UUID/power and <=5% combined clock spread; reference measures forward only |
| RAM-tier baseline/cached-zero ablation | Estimate whether removing dequantisation is worth engineering next | Cached-zero work is garbage mathematics, a ceiling probe only |
| Whole-training comparison | Check whether a local kernel change helps Soup's actual training workflow | Same model/data/update budget, full-step time, both VRAM peaks, held-out outputs |

The research changes the product only when these findings support an actual
patch or a tested decision to preserve the existing path. A complete table by
itself is not acceleration. Alias correctness protects training from silently
wrong gradients; the existing large-alias negative control is essential even
when loss appears normal.

## Next optimization to justify, not promise

Use the valid RAM-tier measurements and an independently labelled profiler
trace to separate dequantisation, forward multiplication, dX multiplication
and checkpoint recomputation. If dequantisation remains material, evaluate a
transposed fused dX path against native exactness and full-step time. Do not
start by forcing the existing Function globally: its dX still materializes W.
Any kernel implementation is a separate change needing numerical, memory and
whole-training evidence before switching Soup's dispatch.

For a cold disk tier, reducing dequantisation cannot be assumed to help when
layer reads dominate. Upstream explicitly distinguishes that bottleneck from
RAM-tier compute headroom; its historical 9.8% / 23% ceilings are not D1's
measured end-to-end speedups. See [upstream #842 and its discussion](
https://github.com/MakazhanAlpamys/Soup/issues/842).

## Handoff and remaining resource boundary

The original K cohort remains 81/98 valid cases, with 17 clock-related
NO_VERDICT cases and no sampled dense-reference windows. Keep the immutable
K archive/packet and their checksums. The new L amendment and CPU-tested
telemetry close the harness gap for future executions, not the old evidence gap.

The $8.061 confirmed cost leaves $1.939. K's 17 missing cases consumed about
1.57 hours of fresh-arm execution; setup, new-card correctness, reference
windows, repeats and export increase that. A selected follow-up is estimated
at 3-5 hours including preparation, conditional on fixed-clock access. A full
new cohort is a different undertaking: K used 9.35 hours in fresh arms alone.
Budget and time must be estimated on the chosen GPU before starting it; do not
spend the final reserve on an uncontrolled-clock repeat of the same failure.
