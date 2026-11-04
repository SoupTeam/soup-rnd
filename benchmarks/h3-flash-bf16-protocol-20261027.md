# H3 native Flash BF16: protocol fixed before rental

2026-10-27. Requested follow-up to the accepted FP32/conservative-math BF16
experiments. No old arm is relabelled as Flash: the older ordinary BF16 run
used automatic SDPA selection and did not retain executed backend operators.

Subsequent outcome: the ordinary nondeterministic Flash case failed the first
strict pair. The predeclared deterministic Flash control passed all four arms
with zero first-step/reload errors. Both cases and all numeric references were
retained; see [the result](h3-flash-bf16-result-20261027.md). The original protocol
below is preserved, and failed timing is not admitted.

## Target and controls

Use the same immutable Qwen2.5-0.5B base revision, dataset SHA256, sequence512,
LoRA q/v r8 alpha16 dropout0, seeds, AdamW and global batch8 as the accepted
BF16 report. Native PyTorch CUDA FlashAttention through
`sdpa_kernel([SDPBackend.FLASH_ATTENTION])`; all fallback backends disabled.
BF16 base/AMP/LoRA outputs; FP32 LoRA masters, gradients and FSDP reduction.
Keep ordinary BF16 GEMM reduced-precision reduction enabled. Do not use math
SDPA or the earlier conservative arithmetic controls as a Flash result.

First campaign: `--bf16-flash`, ordinary nondeterministic Flash backward.
Four fresh sequential arms (1/2, 2/1), micro1/rank and GAS8/GAS4, each with
2 correctness + 5 warmup + 10 timed + 1 final inspection optimizer steps.
The first correctness step uses CPU dispatcher profiling to retain actual
`aten::_scaled_dot_product_flash_attention` and its backward on every rank.
Require both operators and no math/efficient/cuDNN/CPU attention fallback.
Profile each separate reload's actual Flash forward too. The profiler is
closed before warmup and timed training; no profiled timing is admitted.

Original admission thresholds remain unchanged: first reduced gradients and
updated adapter, per-element atol=1e-5/rtol=0.02 across worlds/repeats. Final
and checkpoint-18 reload on the same BF16 base in separate processes, exact
96 keys/tensor values and fixed eval-output atol=1/64/rtol=1/128. All finite,
optimizer/frozen-base, actual batch/data, synchronization and actual DTensor
local-shard checks remain required. Save init/preclip/first-step snapshots.

Stop that campaign after the first pair if strict agreement fails. Retain
both arm archives and diagnostic metadata; exclude all failed timing. A
predeclared causal control may then repeat the same Flash campaign with
`--flash-deterministic`, changing only PyTorch's public deterministic-algorithm
setting and required CUBLAS workspace `:4096:8`. BF16 GEMM reduction remains
ordinary/enabled, Flash stays the sole attention backend. No math substitution,
precision/input/optimizer change or tolerance relaxation. Compare retained
operator/precision/gradient records. If this control succeeds, report it as
**deterministic native Flash BF16**, and retain the default-backward failure
explicitly; do not call nondeterministic defaults numerically validated.

PyTorch 2.13's [native CUDA backward source](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L112-L122)
selects Flash's deterministic backward from `use_deterministic_algorithms`.
This is a hypothesis/control, not evidence that nondeterminism caused the old
unprofiled mismatch. The public
[SDPA selector](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
provides explicit Flash selection; actual operator readings remain required.

## Budget and evidence

Provider-reported charges for all six prior task-owned rentals: $2.142,
queried before this follow-up. Use the previous conservative upper $2.26
plus this new rental's fixed 45-minute compute/storage cap and $0.15 reserve,
and require the sum below the user's **$4 cumulative ceiling**. Refresh live
offers and exclude all existing instance IDs/physical machine IDs. Create
only a new labelled instance with a dedicated key and independent scoped
watchdog; never mutate existing rentals. Stop on transfer/environment/GPU
preflight failure, failed correctness, insufficient resources or time limit.

Reuse image Torch via a venv; remove the unrelated conflicting `spin` utility
with the actual global interpreter inside this disposable container before
requiring a clean `pip check`. Pass the 4 MiB binary SHA256 transfer before
training. Seal/download/hash-verify each arm and await its collector receipt
before the next arm, so bulk transfer cannot overlap timed windows. Preserve
failed campaign diagnostics and complete numeric references before cleanup.
Capture commands/configs/source SHA/versions, all rank timings/memory and
repeat-sized uncertainty. Compare only matching arms on this new same host.
Optimizer/scheduler resume, quality, 70B and 4/8 GPUs remain unmeasured.
