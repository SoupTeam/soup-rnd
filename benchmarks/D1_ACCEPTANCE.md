# D1 acceptance ledger

Updated 2026-10-19 against the [reviewer's remaining-work list](https://github.com/SoupTeam/soup-rnd/pull/16#issuecomment-6253790872)
and the [original #842 scope](https://github.com/MakazhanAlpamys/Soup/issues/842).
This ledger describes evidence, not an owner decision to narrow or close the task.
Latest [RTX 3090 VM F1](d1-vm-f1-result-20261019.md): five accepted ABBA blocks,
no rejected blocks, sizing_extra=0. Correctness, same-session GEMM and telemetry
pass; candidate is slower while saving VRAM. The VM and temporary access are released.
Earlier [rejected follow-ups](d1-review-followup-20261019.md) remain unchanged.

## Current decision

The scoped research result is ready for review. **Full D1 and #842 remain open.**
The [latest synthetic decoder result](d1-vm-f1-result-20261019.md) is credited:
157 MiB allocated and 140 MiB reserved saved, with exact full updates against
native streaming and separately against unpatched resident NF4. Production does
not select this backend. CUDA WMMA is acceptable; a Triton rewrite is not needed.
Formal scoped F1 is now measured: 51.568 ms native vs 83.512 ms candidate at fixed
1200 MHz. This is a negative speed result, not a portable performance guarantee.

| Requirement | Evidence/status | Remaining action |
|---|---|---|
| Changelog | `changelog.d/0.75.1/16.added.md` describes the bounded research scope; fragment validator and 141 focused D1/changelog checks pass | Done; evaluate final-head CI separately |
| Checkpoint-visible tensor state | All six frozen NF4 tensors use `save_for_backward`; no frozen tensor on plain `ctx` | Extend controls for any newly supported cohort |
| Large recycled-buffer control | Nested BF16 `(128,32768,12288)`, 192 MiB packed buffer, two recomputations; forward and dX exact, finite differing plain-ctx control | Other accepted hardware/dtype cohorts need their own evidence |
| Formal full-step performance | VM cohort: **MEASURED_SUBSET**, comparison **MEASURED**, five accepted ABBA blocks, 4000 timed steps, 20 valid same-session GEMM references, fixed1200 MHz, sizing_extra=0 | Done for stated synthetic BF16/RTX 3090 scope; earlier rejected cohorts keep NO_VERDICT |
| Broader coverage | Limited matrix below; historical native/forced-Function results are not new-kernel coverage | Complete applicable gates or obtain an explicit owner decision narrowing this deliverable and leaving exclusions open |
| Full-task acceptance | Scoped performance resolved; no owner sign-off narrowing coverage | Keep D1/#842 open; record reviewer/owner decision explicitly |

All original failed/rejected runs and manifests remain unchanged in the
[verified evidence archive](D1_EVIDENCE.md). CPU replay proves properties of saved
CUDA tensors; it does not establish a new GPU execution or performance result.

## Covered candidate scope

Descriptors below are `(M,N,K)` for `x[M,K]` and `W[N,K]`. The public research API
allows only these four descriptors for `wmma_native` / `wmma_native64`. The
observed default-math result is Linux, RTX 3090 **sm_86**, Torch 2.11.0+cu126,
bitsandbytes 0.50.2, BF16 activations/weights, nested NF4 (blocksize 64, nested
blocksize 256), TF32 off, BF16 reduced-precision reduction **on**. Tile32 and
tile64 are separately identified in `d1-native-result-20261016.md`.

| Descriptor/cohort | Forward evidence | Backward/checkpoint evidence | Boundaries |
|---|---|---|---|
| `(128,64,64)` | Exact streamed vs unpatched resident logits on the small four-layer/two-slot decoder | All 16 nonzero LoRA gradients exact, seq128 checkpointed runtime | Small synthetic fixture, not a pretrained model |
| `(2048,64,64)` | Same gate at seq2048 passes | Same gradient/checkpoint comparison passes | Does not cover seq2048 at hidden4096 |
| `(128,4096,4096)` | Projection parity; final hidden4096 decoder logits/loss exact | Three SGD updates: all 16 gradients and 16 updated parameters exact vs native streaming and separately resident | Final large decoder uses tile64; k/v projections of another shape preserve shipped dispatch |
| `(128,32768,12288)` | Exact native forward; this descriptor reuses upstream's already fused forward | Exact dX after overwriting packed weights/scales; two checkpoint calls and failing plain-ctx control in the same process | Observed native split-K BF16 rounding on this GPU/stack, not a portable cuBLAS contract |

The native acceptance seed and recycled-buffer replacement seed are recorded in
the source/archive. Repeated copies/replays of the same seed are not new coverage.
The old `wmma` backend with BF16 reduced-precision reduction **off** is a separate
diagnostic cohort; its successes do not close default-math acceptance elsewhere.

## Explicit outstanding coverage

The original projection study declares nested NF4, FP16 and BF16,
`M = {1,8,32,128,512,2048}` and these distinct `(N,K)` values:

| Projection | `(N,K)` | New default-math BF16 candidate status |
|---|---|---|
| q/o | `(4096,4096)` | Covered at M128 only; M1/8/32/512/2048 remain outside the candidate API |
| k/v | `(1024,4096)` | All six M values outside candidate coverage; shipped fallback is not candidate parity |
| gate/up | `(14336,4096)` | All six M values outside candidate coverage |
| down | `(4096,14336)` | All six M values outside candidate coverage |

Thus one of the 24 distinct BF16 projection descriptors has scoped candidate
correctness evidence; this is not completion of the old 24-point timing protocol.
The older native/forced-Function measurement grid remains a separate experiment.

- **FP16:** current WMMA public API rejects it. The earlier lower-level aligned
  screen failed forward on `(129,11008,4096)`, seed843, non-nested (509004 differing
  values). A few small passing FP16 cases do not establish FP16 training support.
- **Unaligned/tail shapes:** earlier `(133,67,96)` failed dX for FP16 non-nested
  seed843 (6 values) and BF16 nested seed1977 (1 value). They are rejected by the
  current public WMMA gate, not silently counted as passing.
- **Other M, batches and sequence lengths:** no generalization beyond the four
  descriptors. Flattened batch/sequence dimensions do not by themselves prove
  a new descriptor, layout, checkpoint lifetime or target-model correctness.
- **Non-nested scales / other block sizes:** earlier diagnostic samples exist;
  no complete default-math full-step + large-alias acceptance cohort is claimed.
- **GPU/stack:** the candidate is demonstrated on RTX 3090 sm_86 only. The
  2026-10-16 A100 sm_80 startup attempt never reached CUDA. A new 2026-10-19 A100
  SXM4 gate reached CUDA: forward was exact, but dX had 746429 differing values,
  max absolute error 0.0625; timing was not run. The recycled-buffer gate made two
  recomputations and detected the finite differing plain-ctx control. CPU replay
  independently reproduced these outcomes; A100 support is not accepted.
  Earlier A100 native/forced routes do not establish this WMMA backend.
  Other sm_80/sm_89/sm_90/sm_120 devices,
  drivers and Torch/bitsandbytes versions need separate parity and buffer controls.
  The #842/#776 resident-invariant caveat must be checked before using sm_120.
- **Real target models:** no pretrained 7B/8B full-step acceptance for this kernel;
  the measured memory delta belongs only to the stated synthetic fixture.

For each newly accepted dtype/GPU cohort, report forward and dX separately against
unpatched resident NF4, plus private vs recycled buffers and a same-process
repair-disabled/plain-ctx negative control above the empirical 165 MiB threshold.
For newly covered projection descriptors, preserve deterministic native arithmetic,
prove the candidate path actually ran and verify checkpointed forward/backward.
Only then time full steps. Do not loosen exactness, patch the reference silently,
or substitute existing fallback results for candidate evidence.

## Formal performance and resource gate

Use the existing [final F1 rule](d1-final-f1-rule-20261016.md), its
[separate A100 cohort note](d1-final-f1-a100-20261016.md) where applicable, and
`d1_final_f1.py`. Neither rule is edited after observing results. Keep the old
clock-spread block and its NO_VERDICT; new UUID/source/date is a new cohort.

1. Record booking/lease, GPU UUID, versions, executed-source hashes, rule hashes,
   device isolation and a bounded cleanup plan before running. Check target-card
   dequantization relevance and retain actual dispatch counts; the previous
   post-archive dispatch probe was not fully exported and is not acceptance evidence.
2. Pass target-shape native/candidate and unpatched-resident full-update gates.
   On a new hardware cohort also pass its applicable large-alias/negative control.
3. Run five fresh-process ABBA blocks, 20 warmup and 200 timed full steps per arm.
   Record tokens/s, wall/CUDA times, allocated/reserved peaks before the reference,
   power, suspend/foreign-read validity, peer lifecycle, SM clocks and memory-
   controller utilization. Measure the same-session dense GEMM after primary peaks
   with the preregistered warmup/duration. Apply the unchanged <=5% clock guard.
4. Stop after the first invalid block, at any index. After five valid blocks apply existing F1 sizing;
   if more are required, retain NO_VERDICT rather than automatically extending.
5. Export raw records and tensors with hashes, replay them independently and
   release only this D1 rental. Do not touch another team's/account's instance.

Billing refreshed on 2026-10-19: **$8.371 of the $10 cap**, leaving **$1.629**
before this follow-up (the earlier provider snapshot was $8.354). This is not the
team account balance and billing may lag. Follow-up bookings: A100 SXM4 lease
56604017, machine152356 (failed parity, exported and released); then RTX 3090 lease
56606044, machine55636 (startup timeout, released), capped at $0.15 / 30 minutes.
Combined new spending was capped at $0.30, within the existing $10 total. New
provider charges are $0.059; latest cumulative snapshot **$8.430/$10**, remaining
**$1.570**, billing may update. Source is `28b0b724`, unchanged rules, separate
UUID/cohorts with independent local/container cleanup guards. No GPU work remains running.
No owner acceptance of excluded coverage is inferred from API access or green CI.

Latest finish-session snapshot: **$8.502/$10**, remaining **$1.498**, billing may
update. RTX leases56610875 and56613719 are exported/deleted; failed VM request
created no lease. No GPU work remains running. Executed source was `e274b7e2`;
the later controller-only early-stop fix has CPU regression/mutation checks and
does not count as a new CUDA run or validate the rejected timing records.

Latest VM cohort, executed source `a7ee402c`: independent replay verified 609
source files, 204 exact finite full-update quantities, large recycled-buffer
forward/dX and same-process negative control, all 20 timing arms and references.
Guest driver580.95.05; accepted GPU scope remains RTX 3090 sm_86 only. Numerical
F1 rule unchanged; the time/cost allowance was amended before CUDA under the
user's raised $15 cap. VM56660413 deleted, clocks reset, persistence disabled,
temporary SSH registration revoked. Billing snapshot **$8.653/$15**, remaining
**$6.347**, new VM **$0.137**, may update. See the
[current report](d1-vm-f1-result-20261019.md) for immutable evidence and exact scope.
