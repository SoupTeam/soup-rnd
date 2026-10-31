# D1 NF4: correctness and the useful operating region

Status: **PREPARATION; CUDA NOT RUN; D1 NOT COMPLETE**. Rule dated 2026-10-15.
This rule must be committed before the first CUDA arm. Results are appended in
separate records; this decision rule is retained verbatim. CPU checks are only
harness preparation, never evidence of GPU speed or CUDA numerical acceptance.

## Question and scope

Does the shipped checkpoint-visible NF4 dispatch preserve native resident
training, and at which shapes does it improve time or memory? Does forcing the
Function at training M provide a useful memory/time trade? The forced arm is a
harness intervention, not a new product flag. A negative result is reportable.

Track sources, read 2026-10-15:
- https://github.com/MakazhanAlpamys/Soup/issues/842
- https://github.com/MakazhanAlpamys/Soup/pull/1143#issuecomment-5812578575
- https://github.com/MakazhanAlpamys/Soup/issues/842#issuecomment-5888017772
- https://github.com/MakazhanAlpamys/Soup/issues/842#issuecomment-5982384436

The code half landed as 33639fb4. #842 was released for CUDA measurement on
2026-10-04. The native unpatched bitsandbytes resident model is the reference.
The old sm_120 exception is withdrawn. The Function still dequantises in dX;
this study does not claim a large-M fused backward kernel or the full ceiling.

## Before measuring

Pin source commit, imported source path, runtime/harness/rule hashes, Python and
dependency versions, physical GPU UUID, driver, dtype, quantization, model/data
revisions, seeds, order, optimizer, steps and sequence settings. Commit this rule
and the executed harness. Record pilots and failures under fresh output paths;
no overwrite or silent retry. A change after a measured arm needs a dated rule
amendment and a new cohort. Earlier results stay attributable to their source.

Use seed 842, deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, TF32
off, zero LoRA dropout. Native CUDA bf16 must execute for bf16 acceptance.
Do not downgrade dtype, loosen equality or silently change the hardware.

## Correctness, separately from timing

1. Large-alias fixture: two frozen 32768x12288 weights, nested NF4 blocksize 64,
   M=3, fp16 and bf16. Each packed weight is 192 MiB (dense half: 768 MiB),
   above the historical approximately 165 MiB/layer threshold even if it refers
   to packed bytes. Record both byte counts and actual fused dispatch.
   Compare private-buffer and checkpoint-visible recycled-buffer outputs and
   dX by torch.equal. Save all tensors needed for independent CPU replay.
2. In the same process, a plain-ctx recycled-buffer implementation deliberately
   keeps packed weights and quantization state outside save_for_backward.
   Its finite dX must differ from the private reference; the positive fixture
   must recompute and match. A negative control that is equal or nonfinite
   means the harness has no verdict, not that production is correct.
3. Four tiny decoder layers over two slots: compare streamed logits and EVERY
   expected trainable LoRA gradient with an UNPATCHED resident model. Randomize
   B before comparison, synchronize all adapters, reject missing gradients and
   an all-zero set. Evaluate M=3 (fused) and M=2048 (training), fp16 and bf16.
   Pin CUDA quantization to both arms; report exactness per quantity.
4. Counters must prove a custom Function at fused M and the dense path at
   training M, with streamed bnb.matmul_4bit count zero. A shifted bitsandbytes
   dispatch is a failed/unsupported fixture, requiring a recorded amendment.
5. Run the existing #842 and #968 GPU regressions too. CPU replay confirms
   stored tensors, not an unobserved CUDA execution or a universal invariant.

## Projection study

Frozen-base shapes (N,K): q/o=(4096,4096), k/v=(1024,4096),
gate/up=(14336,4096), down=(4096,14336). M in {1,8,32,128,512,2048},
fp16 and bf16, nested NF4 on; non-nested NF4 is a separately labelled cohort.

- Pair 1: A=dense dequantize_4bit + F.linear; B=shipped native dispatch.
- Pair 2: A=shipped native dispatch; B=forced checkpoint-visible Function.

Both include non-reentrant checkpoint, a frozen base and dX. Explicitly report
that no LoRA optimizer or full decoder is included. Same-path comparisons must
say SAME_PATH and cannot be called a fused-kernel speedup. Different arithmetic
against dense is reported per shape; native exactness uses native reference.

One process and one model per arm. Warm up 20 steps and retain 100 synchronized
timed steps, wall times and CUDA events. Record allocated AND reserved peaks
before creating oracle tensors or the dense ceiling. Measure a same-session
dense GEMM ceiling after peak capture; retain shape/FLOPs, clock and memory-
controller utilization. Keep the backward's dequantisation explicit.

## Real-task extension and useful result

Use a pinned pretrained TinyLlama-1.1B-Chat-v1.0 and SQuAD 2.0 as a first
budget-controlled RAM-tier case, with answer-only labels and NO_ANSWER targets.
Reuse the pinned, disjoint D2 QA split only through its manifest and source
hashes; never silently mix D2 production changes into the D1 source.

Compare native versus forced Function with identical initialization, batches,
rank, optimizer and updates; a dense comparison is separately labelled. Report
full-step wall time, non-padding and supervised tokens/s, allocated/reserved
VRAM and held-out QA outputs. Loss alone is not task quality: report normalized
EM/F1, answerable and unanswerable results and false answers to unanswerables.
Record training/update count and fixed data budget beside every quality result.
The tiny/model case does not establish an 8B/70B multiplier. An 8B RAM-tier
extension needs a separately fixed model revision, storage/fit and cost plan.
The initial bounded screen uses 256 train / 128 validation / 64 test examples
from the fixed D2 order, rank 16, lr 2e-4, batch 1, length 1024, 20 warm-up
updates plus 256 timed updates. Evaluation follows 276 total updates. The test
set is evaluated once per arm, without selecting configurations from its scores.
Only expand to five ABBA blocks after the screen fits the authorized budget.

The target-card ablation uses the existing stream_probe Instruments A/C mechanism
on the same RAM-tier model at sequence 512. C uses cached zero weights and is
timing-only garbage mathematics. It is a ceiling probe, never quality evidence.
Any ratio at that stage must keep that label. The initial study does not claim
a cold-disk benefit. All cached-zero allocations count toward that arm's peak.

## Validity and repeats

Use the shipped F1 ArmWatch; required power/suspend/foreign_reads all must be
ok. Unknown sensors give NO_VERDICT. Every comparison uses fresh-process ABBA
blocks. Start with five complete valid blocks and use F1 blocks_to_add, with
one sizing top-up as METHOD.md defines. A short budget screen remains a pilot.
The 10% F1 effect-size rule cannot establish a statistically supported 1-4% win.
Ratios/ranges are descriptive; no positive speed threshold is required to report.

Additional required rows:
- same GPU UUID, source, rule and settings hash across a comparison;
- finite positive timings, complete ordered runs, no duplicate records;
- SM-clock samples DURING timed work on every arm, with max/min <=1.05 across
  the comparison; missing samples mean NO_VERDICT, retain raw samples;
- memory-controller utilization recorded; missing telemetry is not imputed;
- no foreign GPU compute/graphics processes at pre-arm, sampled execution and
  post-arm; unavailable peer visibility means NO_VERDICT;
- same power limit across arms, no suspend/battery/foreign-reader contamination;
- failed, OOM, unsupported and interrupted attempts remain in the record.

No global PASS follows from a projection subset. Full D1 requires all applicable
correctness, aliasing, dispatch and performance rows plus independent review.
Quality study scope is reported separately; a result on one dataset does not
prove absence of hallucinations generally. No cloud rental, secret transfer,
upload, public claim, merge or release is performed by this harness.

## Current acceptance ledger

| Item | Status |
|---|---|
| Harness CPU contract/regression checks | 50 passed, 13 hardware skips; full CI not run |
| CUDA fp16/bf16 correctness and large negative control | NOT RUN |
| Fresh-process projection grid and GEMM ceiling | NOT RUN |
| Target-card dequantisation ablation | NOT RUN |
| Real-task full-step speed/VRAM/QA comparison | NOT RUN |
| Independent review / final D1 acceptance | NOT REQUESTED |
