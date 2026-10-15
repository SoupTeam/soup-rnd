# E2 Mistral-7B ABBA128 — Predeclared Warm Performance Protocol

## Approval
Team lead @nurikw confirmed the warm-performance thresholds
on 2026-10-15 at 19:18 (local chat time).

Approval applies ONLY to Mistral-7B warm-performance evaluation.

## Fixed experiment
- Model: Mistral-7B-v0.3
- Model revision: caa1feb0e54d415e2df31207e5f4e273e33509b1
- Backend: Transformers
- Precision: CUDA FP16
- AMP initial scale: 1024
- Training: LoRA, top K=8 layers
- LoRA rank: 4, alpha: 8, dropout: 0
- Target modules: q_proj, v_proj
- Dataset: experiments/e2_abba_128_fixed_v1.jsonl
- Examples: 128
- Epochs: 2
- Batch size: 1
- Gradient accumulation: 1
- Sequence max length: 128
- Learning rate: 0.0001
- Optimizer steps per run: 256
- Warmup steps excluded from step summary: 8
- Cycles: 3
- Order: A-B-B-A repeated 3 times
- Total subprocess runs: 12
- Total planned optimizer steps: 3072

## Dataset SHA256
5d6e80c64defd24e759136f8d88e58d98da89958ec861e772ef2b02568df1c3c

## Configuration SHA256
Baseline A:
39304f7670c5f017368a54760d63ab43a495249e4b573c31fc3b35457519b684

Cached B:
0d76bf0d392f6781de0e9a2d1d080d351e323b62260e73035565b13dbf466c80

The configurations differ only in frozen_prefix_cache
and their output paths.

## Predeclared warm-performance rule
For each ABBA cycle:
- A = baseline, B = cached
- Use second-epoch summed step wall time
- Cycle speedup = mean(A times) / mean(B times)
- Median speedup across 3 cycles must be >= 1.10
- Every individual cycle speedup must be >= 1.00

## Required evidence
Preserve each arm's:
- Per-epoch step-wall timing, including epoch 1 and epoch 2
- Full subprocess wall-clock duration
- Cache hits and misses
- Training report, provenance, logs and exit code
- GPU telemetry and evidence of comparable execution conditions

## Limitations
- Summed step-wall time is not full epoch wall-clock time
- Epoch 1 is not an independent cold-cache construction timer
- Full subprocess wall time includes startup and other overhead
- The earlier ~1.83x pilot is separate from final measurements
- No validation-loss claim is made from val_split=0.0
- Quality and Small-MoE require separate evaluations
- No overall E2 research PASS is authorized by this protocol
- This runner does not automatically issue a performance verdict

The final verdict must be based on recorded evidence,
not merely successful execution.
