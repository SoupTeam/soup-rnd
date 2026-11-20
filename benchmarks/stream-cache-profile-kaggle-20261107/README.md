# Kaggle T4 memory-profile qualification

A synthetic Llama with 329,303,040 parameters and 24 decoder layers was trained
through the native Soup streaming runtime and profile callback. The benchmark
used frozen NF4 weights, FP16 compute, FP32 AdamW adapter states, 256 tokens per
microbatch, accumulation of two, and one T4. The PyTorch allocator was capped to
8 GiB on the physical 16 GB card. Kaggle supplied Python 3.13, Torch 2.11.0+cu130,
Transformers 5.20.2, PEFT 0.21.2 and bitsandbytes 0.50.2.

The fixed configuration used eight GPU-resident layers and a 96 MiB RAM-cache
ceiling, admitting nine host-cached layers. Profile reuse selected those same
weight caches plus retained activations for all 24 layers.

| Median of two balanced pairs | Fixed weight caches | Profile selection |
|---|---:|---:|
| Eight updates with setup | 12.328 s | 9.878 s |
| 64 updates with setup | 78.380 s | 59.302 s |
| Step after the first two, 64-update jobs | 1.180 s | 0.889 s |
| Peak allocated VRAM | 219.5 MiB | 576.9 MiB |
| Peak process RSS | 1.535 GiB | 1.538 GiB |

Whole-job median reductions were 19.9% and 24.3%. Each job runs in a fresh worker,
with the paired order reversed on the second repetition. Profile recording and
creation of the synthetic checkpoint occur separately. The gain here comes from
the selected activation-retention placement. Two pairs establish a scoped
observation, with no large-model or downstream-quality conclusion.

Four comparisons against ordinary streaming, each covering four AdamW updates,
gave 26,912 exact tensor comparisons. The gate checked loss, logits, all 336
adapter gradients/parameters and AdamW states using original rtol=0.01,
atol=1e-5 and the three-decimal loss rule before confirmation timing.

## Reuse-startup correction

Recording formerly charged the baseline residency probe and activation sizing
probe to every future reuse. Reuse skips those planning passes and still runs
the mandatory complete memory checks. New profiles store their measured
`record_only_seconds` and subtract it when predicting reuse time. Old profiles
default this field to zero.

The combined candidate measured 1.754 seconds of record-only work. The corrected
prediction first changes the placement for a three-update job in this history.
The measured eight- and 64-update jobs select the same placement under both
formulas. The three-update prediction has no direct timing measurement.

## Reproduction and evidence

The executed base is PR revision `2aab18d7611fe003930f15ac86f62e131b21a981`.
[bootstrap.json](bootstrap.json) binds the exact source patch
and probe by SHA-256. The runner stages that base and
patch, preserves Kaggle's Torch distribution and installs bitsandbytes 0.50.2.
The [frozen source packet](frozen-source.zip) contains the byte-exact
`probe.py`, `runner.py` and `source.patch`. Extract it to reproduce the run. Outputs are retained in
[evidence/summary.json](evidence/summary.json), with individual worker JSON,
profile history and fixture metadata alongside it.

The first attempt stopped before measurements because the shard output was
outside Soup's allowed roots. The second attempt used the experiment's current
directory and completed. [Preflight](preflight.json) verifies real T4 NF4/FP16
execution. The kernel completed and released its runtime.
