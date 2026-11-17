# D3 CPU review follow-up

The Apple Silicon exactness finding is addressed by refusing the unqualified CPU recipe before cache access or expert arena allocation. No native logits/gradient/reload comparison was replaced by a tolerance. This is an explicit support restriction, not a claim to have repaired or qualified Apple arithmetic.

The supported CPU development boundary is Linux/Windows x86-64, float32 model weights and one CPU thread. Explicit CPU selection reaches Trainer and adapter reload. CUDA validation and the historical source-pinned measurements are unchanged.

Windows/Python 3.12.14/torch 2.15.0+cpu/TRL 0.29.1: 52 CPU tests pass with transformers 5.21.1 / PEFT 0.21.2; a separate 130-case analysis/SFT selection passes on that stack; the 52-case CPU selection also passes with transformers 5.19.0 / PEFT 0.20.0. No Mac hardware run or full repository/platform CI is claimed. See summary.json for source hashes and logs.

The separate portable replay command and operator-supplied raw archives are documented in [the evidence guide](../../../docs/d3-evidence-replay.md). Reviewer receipt/access is not implied by preparing these files.

The final all-D3 CPU selection on the newer stack passes **481 tests**, with 94 GPU cases deselected and no failures/errors/skips. This is a selected CPU suite, not the full repository CI or new CUDA execution.
