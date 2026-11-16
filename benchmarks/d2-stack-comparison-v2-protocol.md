# D2 stack-v2: declared before the new paid run

Question: how long does the same complete TinyLlama 1.1B LoRA training workload
take in Soup, PEFT+Liger, and Unsloth, with each stack's numerical differences
reported alongside its speed? This does **not** assert equal training quality
or attribute a whole-stack ratio exclusively to LoRA fusion.

The earlier strict external 50-step comparison remains failed. Its raw records,
thresholds and source archives are unchanged. This is a separately named
experiment, not a retrospective PASS or a larger tolerance for that experiment.
The original Soup correctness gates, and Soup-versus-PEFT full-model `.3f`/1%
gate, remain hard requirements.

## Unpaid source and CPU diagnosis

The pinned Unsloth loader routes fused CE through `accelerator_scaler`; its
Trainer attaches that object to wrapper/base-model modules. Our standalone
worker omitted the hook. The worker now invokes Unsloth's own hook on the active
GradScaler, including the new scaler created after warmup. The worker verifies
that the actual causal model holds the same object. This fixes an integration
omission; it is not a claim that every prior gradient difference is explained.

Liger's mean fused linear CE computes and stores gradients in forward before
the later upstream scale reaches backward. With FP16, scaling an already-rounded
small gradient cannot recover lost precision. The pinned kernels also fuse
SwiGLU without the intermediate low-precision SiLU cast used by the stock graph.
CPU experiments in `d2_rounding_diagnostic.py` isolate those rounding boundaries.
They execute our arithmetic emulation, not vendor GPU kernels. At the fixed
seed/shapes, late scaling changes the FP16 head gradient by about 2.5% relative
L2; moving normalization after the sum reduces that diagnostic discrepancy to
about 0.008%. Fused intermediate rounding also differs in BF16. These observations
identify mechanisms, not a complete attribution of the old full-model drift.

The pinned Unsloth LoRA kernels also cast FP32-stored A/B adapters to the active
compute dtype inside their matrix/backward paths. Ordinary PEFT uses the FP32
adapter path. FP32 storage therefore does not mean identical FP32 arithmetic
across stacks. The CPU diagnostic separately demonstrates those cast boundaries;
the full-stack report must disclose them, without calling the two numerical
paths bitwise-equivalent or silently changing Unsloth to a non-native path.

## Configuration frozen for stack-v2

Reuse the exact old pinned model revision, FP16/BF16 base parameter digests,
FP32 initial adapters, nonzero B, rank 16/alpha 32, batch 1/sequence 512 and
synthetic token tensors. Same AdamW lr 0.003/betas/eps, dropout/cache/checkpointing
off and deterministic execution control. No seed, learning-rate or input tuning.

All four stacks receive `num_items_in_batch=511`: sum over the 511 valid
next-token targets, divided by 511. This is mathematically the same mean loss,
with the normalization explicitly outside Liger's fused sum. Liger's fused CE,
RoPE, RMSNorm and SwiGLU remain enabled; Unsloth's fused loss and LoRA kernels
remain enabled. No return-logits or unpatched fallback is benchmarked.
The normalization path and Unsloth scaler hook are recorded in each ready file.

## Required validation

Before full-model timing, execute nine actual FP32 vendor operator cases in
fresh backend processes: Liger RMSNorm/SwiGLU/RoPE/fused linear CE; Unsloth
RMSNorm/fused CE/LoRA-W/QKV/MLP. Compare actual outputs and all requested input
and adapter gradients with ordinary PyTorch formulas. Every retained quantity
must be finite, shape/dtype-identical, and relative L2 <=0.001. Preserve all
reference/candidate raw pairs and independently replay them on CPU. These are
operator probes, not a complete upstream test suite or proof of FP16/BF16 equality.

Collect all eight full-model 50-step gates. Require Soup/PEFT strict equality
under the existing `.3f`/1% criterion. For external stacks preserve the same
strict comparison result as **PASS or FAIL**, plus every loss and retained tensor
comparison; do not relabel drift as equivalence. Missing/malformed/nonfinite
data, changed bases, missing custom execution, or failed operator probes block
timing. Disclosed finite cross-stack numerical drift does not answer the new
throughput question by itself and is not silently discarded.

Every timed worker must replay against its **own backend's** frozen gate: all
50 losses at `.3f`, every first/final adapter tensor within 1% relative L2,
same fixture/source/packages/mode. This tests repeatability without treating
another stack's rounding order as that backend's canonical result.

## Timing, audit and budget

Soup/Liger and Soup/Unsloth; two dtypes; three ABBA cycles per pair;
four fresh processes per cycle = 48 timing workers (`--external-only`).
PEFT remains in all numerical gates. Before any new GPU validation or timing,
the optional Soup/PEFT timing series was removed to reserve evidence-transfer
time after slow package setup. The earlier 72-worker preparation snapshot is
preserved; no observed timing result informed this scope change. Same >=20 warmup updates
before resetting adapters/optimizer/scaler; compilation/model loading are outside
step timers. Fifty complete updates per worker. Capture synchronized CUDA events,
wall time, 511-target tokens/sec, allocated/reserved VRAM, all losses and full
first/final snapshots. Report each cycle and candidate/Soup wall-time ratio
(>1 means Soup is faster), including regressions and inconclusive results.

Retain the existing strict GPU ownership/UUID/empty-before-after, clock, power,
throttle and ABBA stability requirements. Independent CPU replay recomputes
operator pairs, all gate/timing tensors and lifecycle/cycle ratios. Hash identical
raw files for lossless deduplication, not numeric rounding. Preserve all failed
or partial runs. No ratio is accepted for a missing/failed cycle.

User authorization: team-account rentals, maximum **3 hours / $1.50** in total,
including disk and traffic. Prepare fixtures/source and CPU validation first.
Install independent local/provider destruction guards before experiments; delete
the exact task instance by 2 h 50 min and verify API absence. Reserve the final
20 minutes before that guard for evidence transfer/cleanup; benchmark timeout is
calculated from the remaining provider window. Do not affect other team instances.
If the matrix cannot finish inside that window, preserve the partial result;
no extension of the total authorization is permitted. The first RTX 3090 host
failed thermal/clock stability gates at both 20 and 100 warmup updates; clock
and power controls were denied. Before obtaining any accepted timing ratio,
the operational plan changed to preserve/delete that pilot and use one A100
replacement within the same original absolute deadline and cumulative budget.
Both raw VOID attempts remain retained. Numerical/timing criteria are unchanged.
