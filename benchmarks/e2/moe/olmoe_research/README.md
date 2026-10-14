# E2 Pretrained OLMoE — Exploratory Research

Status: NO VERDICT

## Model

- Model: allenai/OLMoE-1B-7B-0924
- Revision: 6d84c48581ece794365f2b8e9cfb043c68ade9c5
- Hardware: RTX 2080 Ti, 22 GiB VRAM
- Precision: FP16 with GradScaler
- LoRA: r=4, alpha=8, q_proj/v_proj
- Seed: 42

This directory contains research-only experiments.
It does not add production OLMoE cache support.

## Correctness

Pretrained OLMoE cache correctness was tested for:

- K=4: 12 frozen decoder layers
- K=8: 8 frozen decoder layers
- K=12: 4 frozen decoder layers
- K=16: no-cache control

Deterministic Alpaca smoke tests used 8 training and
8 validation records for 2 epochs.

K=4/8/12 cached and baseline runs produced identical
training loss, validation loss and final LoRA adapters.

Strict CUDA deterministic execution was required for
reproducibility across independent processes.

## Exploratory FP16 Performance

128 synthetic examples per epoch, 2 epochs:

| K | Warm speedup | Two-epoch end-to-end ratio |
|---|---:|---:|
| 4 | 1.692x | 0.614x |
| 8 | 1.218x | 0.747x |
| 12 | 1.087x | 0.886x |

Warm-cache execution is faster in these pilots, but
total two-epoch time remains slower due primarily
to initial frozen-prefix SHA256 fingerprinting.

These measurements are not ABBA and have no
continuous GPU-peer validity monitor.

## Preliminary Quality Smoke

Validation task loss on 8 validation examples:

| K | Task loss |
|---|---:|
| 4 | 2.611490 |
| 8 | 2.562600 |
| 12 | 2.522681 |
| 16 | 2.441741 |

These are technical smoke results, not quality
acceptance. The research runner uses constant LR
and approximate Soup data formatting.

## Known Limitations

- Research-only OLMoE cache, not production E2.
- Guard relies on PyTorch tensor versions and identity.
- Raw storage or .data modifications require explicit invalidation.
- No full 2,000/200 quality evaluation completed.
- No predefined quality/safety acceptance thresholds.
- No final Mistral-7B 12-run ABBA result.
- No final research acceptance verdict.

## Next Steps

1. Align OLMoE data formatting and LR scheduler with Soup SFT.
2. Fix and approve quality/safety thresholds before confirmatory runs.
3. Train and evaluate K=4/8/12/16 on the full dataset.
4. Complete the approved Mistral-7B 12-run ABBA benchmark.
5. Publish validity evidence, cache costs and payback analysis.

Final research acceptance: NO VERDICT.
