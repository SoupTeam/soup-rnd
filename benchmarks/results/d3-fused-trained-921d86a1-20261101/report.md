# D3 packed fused public trainer: completed qualification

Measured commit: `921d86a19d96e62ecc12c7c6aef49074f3ebb14d`. One RTX 3090 24 GiB (sm_86), pinned pretrained OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`. Frozen routed experts, q/v attention LoRA, seven synthetic 512-token rows, accumulation windows 2/2/2/1, four SGD updates; one warmup and three timed windows. Ordinary/L2L and NF4/nested NF4 are separate cohorts.

All **112/112** declared processes completed: **24 correctness, 80 timing and eight CUDA profiles**, all F1 OK. All eight packed-fused resident/streamed correctness cells passed. Every repetition calculation retains five blocks: **zero additions and zero replacements**.

The original CPU analyzer verifies actual measured Git source blobs, retained native tensors, A/A, effective negative/restored controls, exact gradients/updates, real adapter save/reload, same-kernel resident/streamed pairs, timing windows and actual CUDA traces. The source-host archive audit separately verifies every ZIP payload and the complete archive SHA-256. This is CPU replay of recorded evidence, not a new GPU run.

| Recipe | Staged tok/s | Packed hybrid tok/s | Paired time change | Training peak allocated MiB, staged / packed |
|---|---:|---:|---:|---:|
| nf4/ordinary | 136.050 | 111.282 | +22.240% | 678.07 / 626.07 |
| nf4/l2l | 145.267 | 116.688 | +24.521% | 701.07 / 649.07 |
| nf4-dq/ordinary | 131.103 | 100.714 | +30.158% | 676.58 / 622.97 |
| nf4-dq/l2l | 137.424 | 105.227 | +30.617% | 699.58 / 645.97 |

Negative paired time change means faster. The frozen statistic is the mean of within-block fused/staged time ratios; throughput ratios are not substituted for it. Peaks are synchronized PyTorch training allocator measurements after warmup, not total-device memory certification. Reserved peaks and whole-child RSS remain in analysis.json.

| Profile | Backend | Actual CUDA kernels | Packed fused kernels |
|---|---|---:|---:|
| nf4/ordinary | grouped | 211622 | 0 |
| nf4/ordinary | fused | 228626 | 1968 |
| nf4/l2l | grouped | 215632 | 0 |
| nf4/l2l | fused | 232636 | 1968 |
| nf4-dq/ordinary | grouped | 212245 | 0 |
| nf4-dq/ordinary | fused | 258752 | 1958 |
| nf4-dq/l2l | grouped | 216253 | 0 |
| nf4-dq/l2l | fused | 262760 | 1958 |

## Scope and retained evidence

The staged comparator and packed candidate have explicitly different high-row dispatch profiles. Staged uses native rows 1..128 and 481..512; packed hybrid uses native rows 1..128 and fused WMMA rows 129..512. Native SiLU, routing and ordered scatter remain unchanged. This compares complete backends, including their row-policy difference; it does not isolate fusion alone. Small-row fallback is not counted as fused work.

The packed pool does not retain a complete dense MoE layer. Separate resident-expert references are correctness-only controls; they are not full-resident-model timing runs. No convergence, general speedup, full-weight training, 8 GiB fit, physical-host isolation or default change is established. Guest F1 and this pinned hardware/software recipe delimit the result.

The full raw ZIP is retained at `/root/d3-fused-trained-921d86a1-20261101.zip` on the source instance: 2942937841 bytes, SHA-256 `3221d8ec19dbb01cbb1cadce9927f5cb284250e6a8070dd98328ddb6152fe6c1`. Its 4131 payload identities are in raw-archive-manifest.json. It retains source snapshots, all raw tensors/adapters, traces, replay records and logs. Original/generated checkpoint/cache payload omissions are explicit in that manifest.

**The complete 2.94 GB archive now has an author-local verified backup outside Git.** Its full SHA-256, all ZIP CRCs and all 4,131 payload hashes match; see `backup-verification.json`. This small Git publication still does not contain the full tensor/trace archive. Author-local retention is separate from reviewer access: the operator will supply both raw ZIPs directly, with the [portable replay command](../../../docs/d3-evidence-replay.md). Artifact receipt/access must be confirmed by the reviewer. Storage verification alone is not a numerical replay.

The historical staged 234-cell study and unsuccessful development candidates remain unchanged. Current-head CI, issue-tracking decision and final owner/reviewer D3 acceptance remain separate. No additional GPU repetition is requested by this completed rule.
