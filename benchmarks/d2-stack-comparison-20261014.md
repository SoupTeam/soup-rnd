# D2 complete-step stack comparison — A100, 2026-10-14

Raw measurements, historical files and replay commands are preserved in the
[D2 evidence archive](d2-evidence-archive.md). Evidence links below pin the
pre-cleanup snapshot; use that checkout for the original replay paths.

**Collected: 48 timing workers / 12 ABBA cycles, FP16 and BF16.**
All timed trajectories pass independent CPU replay against their own backend gates.
Linux and isolated Windows CPU replay agree across 68,992 tensor comparisons.
Numerical equivalence to PEFT remains a separate FAIL for Liger/Unsloth.
These are configured-stack timings under observed default GPU Boost; full D2
requires team review of this scope, final-head full CI and D2/T00/F1 sign-off.

## Full-step times

Complete pretrained TinyLlama 1.1B LoRA, dense FP16/BF16 frozen bases, FP32
rank-16/alpha-32 adapters at all seven projections, batch 1 / 512 fixed synthetic
tokens / 511 next-token targets. Same AdamW (lr 0.003, betas 0.9/0.999, eps 1e-8,
weight decay 0, foreach/fused off), dropout/cache/checkpointing off, TF32 off,
deterministic algorithms and CUBLAS workspace `:4096:8`. Same frozen inputs
and source across processes. Each worker warms for 100 updates, restores initial
adapters and recreates optimizer/scaler, then measures 50 complete updates.

Wall time covers zero_grad, forward/loss, backward and AdamW.step. Loading,
compilation, profiling, warmup and validation copies are outside the step timers.
This excludes tokenizer/dataloader/transfer time and does not measure Soup CLI
overhead, real-data quality, convergence, full-weight training or full 8B/NF4 training.

Candidate/Soup <1 means the external stack takes less time. Soup times below
are paired with that candidate, not pooled across the two campaigns. Every
cycle is retained; median/range describes three repetitions, not a confidence interval.

| Dtype | Candidate | Soup mean ms | Candidate mean ms | All three candidate/Soup cycles | Median (range) |
|---|---|---:|---:|---|---|
| float16 | liger | 219.617 | 197.830 | 0.885584, 0.898394, 0.918642 | 0.898394 (0.885584–0.918642) |
| float16 | unsloth | 219.830 | 193.966 | 0.896150, 0.888043, 0.862545 | 0.888043 (0.862545–0.896150) |
| bfloat16 | liger | 216.344 | 192.961 | 0.915449, 0.885300, 0.875317 | 0.885300 (0.875317–0.915449) |
| bfloat16 | unsloth | 215.809 | 190.970 | 0.886838, 0.887081, 0.880765 | 0.886838 (0.880765–0.887081) |

All observed cycles favor the external stacks in this configuration.
No Soup whole-model acceleration is established. No ratio is attributed solely
to LoRA fusion or treated as evidence of equal training quality. Full-precision
wall/CUDA-event means, target tokens/sec and memory are in the [CSV](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-stack-v3-20261014/summary.csv).

## Numerical results and admission

All eight full-model gates ran before timing. The original strict rule remains
all 50 losses equal at `.3f` and every first/final adapter weight/gradient within
1% relative L2. There are 1,232 tensor quantities per gate. PEFT self-comparisons
are references, not independent correctness results.

| Against same-dtype PEFT | Losses matching .3f | Tensor failures / 1,232 | torch.equal quantities | Strict result |
|---|---:|---:|---:|---|
| float16 / soup | 50/50 | 0 | 1232 | PASS |
| float16 / liger | 1/50 | 831 | 0 | FAIL |
| float16 / unsloth | 1/50 | 860 | 0 | FAIL |
| bfloat16 / soup | 50/50 | 0 | 1232 | PASS |
| bfloat16 / liger | 1/50 | 1223 | 0 | FAIL |
| bfloat16 / unsloth | 0/50 | 1224 | 0 | FAIL |

Native FP32 vendor probes independently pass all nine cases / 37 quantities
at the predeclared relative L2 <=0.001. They test Liger RMSNorm/SwiGLU/RoPE/
fused linear CE and Unsloth RMSNorm/fused CE/LoRA-W/QKV/MLP. Complete raw pairs
are public. This is operator validation, not a full upstream suite or proof
of FP16/BF16 full-model equivalence.

The unpaid diagnosis found a missing standalone Unsloth GradScaler hook; the
worker now attaches its own vendor hook, including after the warmup reset.
All stacks use sum/511 normalization. CPU arithmetic emulation isolates fused
loss scaling, intermediate SwiGLU rounding and FP32-stored versus half-compute
adapter boundaries. It does not execute vendor GPU kernels or completely
explain the remaining model drift. Native optimized paths remain enabled.

Finite external cross-stack drift is disclosed under the separately declared
[stack-v2 numerical policy](d2-stack-comparison-v2-protocol.md). Original FAILs
are not relabelled as equivalence. Missing/nonfinite quantities, source/input
changes or failed operator checks block admission. Soup/PEFT remains a hard
strict gate. All 48 timed arms additionally pass all 50 loss comparisons and
all 1,232 saved quantities against their own backend gate.

## Hardware, memory and clock policy

NVIDIA A100-PCIE-40GB (sm80), driver 595.91.07, stock 250 W limit, Linux/
Python 3.12.3; torch 2.11.0+cu126, Triton 3.6.0, Transformers 5.17.0, PEFT
0.20.0, Liger 0.8.4, Unsloth/Unsloth Zoo 2026.10.4. The locked environment
passed dependency checks and compiled/executed a real Triton GPU kernel.

| Dtype/backend | Peak allocated GiB | Peak reserved GiB | Observed SM MHz | Temperature °C |
|---|---:|---:|---|---|
| float16/soup | 3.054 | 3.238 | 1035–1215 | 31–36 |
| float16/liger | 3.489 | 3.545 | 945–1095 | 32–35 |
| float16/unsloth | 2.906 | 2.973 | 780–900 | 30–34 |
| bfloat16/soup | 3.054 | 3.238 | 1050–1215 | 31–34 |
| bfloat16/liger | 3.489 | 3.545 | 930–1080 | 31–33 |
| bfloat16/unsloth | 2.906 | 2.973 | 810–945 | 30–31 |

Peaks are maxima over fresh workers after warmup/reset; allocated and allocator
reserved memory remain distinct. Soup has 12 timing workers per dtype, each
external stack six. Each arm has one GPU model; snapshots are copied to CPU
outside the timers. Power, frequency, pstate, temperatures, graphics/compute
peers, context birth, UUID and empty-before/after records are all retained.

Both hosts denied frequency control. The RTX 3090 pilot failed thermal/clock
checks, its longer-warmup retry failed clock stability, and the A100 strict
pilot failed the <=5% clock gate. Those attempts remain VOID. Before any
accepted timing, the user delegated the choice of a separately declared
[stack-v3 default-Boost experiment](d2-stack-comparison-v3-protocol.md).

Only frequency spread admission changes: >=3 finite positive clock observations
are still mandatory, and the strict <=5% diagnostic is preserved. Normal Boost
variation is disclosed rather than capped. No clock/power override succeeded.
Peer/UUID/lifecycle, positive power, performance state and thermal/hardware
exclusions remain hard gates. Software power-cap and idle flags are recorded
and allowed. These ratios do not establish fixed-frequency kernel efficiency.

## Reproducibility and cost

[Public evidence and CPU replay](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-stack-v3-20261014/README.md) includes
all timing/loss/telemetry records, exact source and fixtures, native operator
pairs, selected model tensor witnesses and both independent audit outputs.
The complete model snapshots remain in the separately identified local full
raw archive; the compact public package alone cannot reproduce every positive
full-model tensor match. This boundary is explicit in its replay instructions.

Executed source ZIP: `52f6b6ef22a8a5359ba57ed311168321e174059c9fcf3e3a81a42478edf0f396`.
All 1,510 text or unchanged-binary payloads match the final checkout. One unused
UI PNG was normalized by the old archiver; the executed ZIP is unchanged and
its original PNG plus both hashes are supplied separately. No benchmark/runtime
Python differs. The post-run public timing verifier is identified separately.

Full A100 raw archive SHA-256: `389b352bcd67d90b62635d512405046d73a1ae3b3a2822785969948eec8b5d23`
(1130005846 bytes; all ZIP CRCs and payload hashes independently verified).

- Instance 55871696: deleted, API absence verified; 66.48 minutes; observed provider charges $0.285.
- Instance 55879869: deleted, API absence verified; 85.26 minutes; observed provider charges $0.861.

Current authorization observed total: **$1.146**, below $1.50 including disk/traffic.
Both rentals share the original 3-hour ceiling and 08:18:06 UTC destruction
deadline. No third rental or extension was used. Provider charges may settle
later; these are exact instance-source invoice rows, not team balance subtraction.
Other team instances were not changed.

## CI and remaining acceptance

[Run 38513528907](https://github.com/SoupTeam/soup-rnd/actions/runs/38513528907)
failed one absolute boxed-parser timing bound under coverage (3.2648 s vs 3 s);
its relative growth check passed. Windows 3.10/3.12 timed out at 90 min around
96%/99%, without a prior assertion failure. Scan timing now runs real scans
in a fresh interpreter without coverage hooks; both original limits remain.
The full test-job bound is 120 min, retaining every test.

Local validation: 106 targeted D2/parser tests pass on Windows/Python 3.10
with torch 2.14.1+cpu; 68 workflow tests pass; all nine parser tests also pass
under coverage. These invocations overlap and are not a full CI result.
Source/scripts/tests/benchmarks Ruff is clean. The public timing verifier passes
the real 48-arm records and rejects eight in-memory corrupted controls.
The first workspace torch 2.13 CPU replay exited without diagnostics; its failure
is retained. Isolated torch 2.14.1+cpu replay with OMP/MKL threads set to 1
completed successfully; no numerical criterion changed.
Final-head full CI is still required; consult the [PR checks](https://github.com/SoupTeam/soup-rnd/pull/5/checks).
Final team review/sign-off is required; this report does not mark D2 Done.

## Fresh GPU reproduction

Extract the frozen source/fixtures and install its exact hashed Linux lock.
Populate and hash-check the pinned base checkpoint for both fixture directories.
Set `PYTHONPATH` to the extracted `src` and OMP/MKL threads to 1. Regenerate
native proofs on that same host before starting the configured experiment:

```bash
for backend in liger unsloth; do
  python benchmarks/harness/d2_operator_probes.py --backend "$backend" \
    --output "results/operator-proofs/$backend"
done
python benchmarks/harness/d2_training_baselines.py run \
  --fixtures fixtures/float16 fixtures/bfloat16 --output results/comparison \
  --steps 50 --warmup 100 --cycles 3 --seconds 3600 --deterministic \
  --token-sum-normalization --stack-comparison --external-only --default-boost \
  --operator-proofs results/operator-proofs
```

Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Restore exact archived inputs rather
regenerating them with another torch version. Set up an independent rental
destruction guard and evidence-transfer reserve before any paid reproduction.
