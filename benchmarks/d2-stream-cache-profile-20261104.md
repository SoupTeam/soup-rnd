# Profiled memory placement on Qwen3-32B NF4

The profile selected ordinary disk streaming for an 18-update job on a physical
RTX 4060 Ti (8,188 MiB VRAM) with a 31,950,110,720-byte host RAM limit and no swap.
The fixed comparison uses a 14 GiB RAM cache ceiling and seven GPU-resident
decoder layers. Four balanced fresh-process pairs used the same frozen NF4
checkpoint, BF16 compute, rank-4 LoRA, 512-token microbatches and accumulation of
two. The measured job contains two warmup and 16 timed optimizer updates.
Common model download, conversion, and the separate runs that recorded the
profile are outside the measured job time.

| Median | Fixed caches | Profile selection |
|---|---:|---:|
| Whole 18-update job | 188.30 s | 180.16 s |
| Setup | 30.17 s | 11.83 s |
| Optimizer update | 8.59 s | 9.12 s |
| Tokens/s | 119.24 | 112.32 |
| Peak allocated VRAM | 5.68 GiB | 4.04 GiB |
| Peak reserved VRAM | 5.95 GiB | 4.32 GiB |
| Peak process RSS | 19.50 GiB | 5.42 GiB |

The paired whole-job time reduction was 4.47% (bootstrap 95% interval
4.30–4.78%). Profile selection took 6.17% longer per update. Setup savings
paid off within this short job. Longer jobs can make a different choice.

| Pair | Order | Fixed job (s) | Profile job (s) | Fixed step (s) | Profile step (s) |
|---|---|---:|---:|---:|---:|
| 1 | Fixed → profile | 189.305 | 181.159 | 8.580 | 9.066 |
| 2 | Profile → fixed | 187.290 | 178.046 | 8.588 | 9.079 |
| 3 | Fixed → profile | 190.661 | 182.453 | 8.588 | 9.161 |
| 4 | Profile → fixed | 187.253 | 179.152 | 8.587 | 9.155 |

Before confirmation timing, four eight-update trajectories matched the fixed
reference exactly, for 179,904 comparisons. They covered ordinary streaming,
three resident layers plus four retained activation layers, seven retained
activation layers without GPU-resident weights, and the selected replay. The
gate covered loss, logits, gradients including all 64 decoder-layer inputs,
LoRA parameters and AdamW state. Original tolerances were `rtol=0.01`,
`atol=1e-5`, and loss rounded to three decimals.

The GPU measurements are from research revision
`0e585bdca3b37887f3f84adec846b1282e973379` with deterministic settings,
fixed controlled text inputs and dropout disabled. The port in PR #5 is tested
separately. These measurements qualify the short-run policy on this machine.
Both activation placements passed the numerical gate before recording.
