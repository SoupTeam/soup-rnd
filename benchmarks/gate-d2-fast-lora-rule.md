# D2 Fast-LoRA: decision rule before evidence runs

## Scope

Repository: SoupTeam/soup-rnd. Starting revision: 9aa43bd71afe12d3724f196202f1140e7dc409dc.
This record defines the rule, not a measured result. It covers the existing
single-projection, shared-X QKV and SiLU/SwiGLU MLP autograd paths. The local
change removes QKV input-gradient work when the input does not require gradients;
all trainable A/B gradients must remain unchanged.

The first evidence fixtures are SYNTHETIC: randomly initialized tiny models and
fixed synthetic token batches. An 8B-shaped single block is a shape fixture, not
a trained 8B checkpoint. No quality claim on real data or pretrained models follows
from these fixtures. No end-to-end upstream multiplier is claimed.

## Correctness rule

Use the same initial weights, inputs, active adapters, per-projection scaling,
dtype and device for the fast path and unpatched PEFT reference. Base weights are
frozen; A and B remain trainable; dropout is zero. Initialize B nonzero for
projection-gradient checks so A's derivative is exercised.

1. CPU float64 gradcheck must pass for each Function, with respect to X and every
   trainable A/B tensor. Reuse the existing per-path gradcheck tests.
2. Report forward and backward separately. Compare every output, requested dX,
   and every expected dA/dB. Missing or nonfinite values fail; a comparison that
   simply omits missing gradients is invalid. Frozen bases must not gain gradients.
3. For fp32 dense parity, use torch.testing.assert_close's dtype defaults. Record
   max absolute error and torch.equal independently. Passing an approximate check
   is not evidence of bit-exactness.
4. For fp16/bf16 dense parity, compare both paths against a float64 reference.
   The proposed #792 criterion is per tensor:
   fast_max_abs_error <= 2 * peft_max_abs_error + 1e-8.
   Report both errors and the ratio when the denominator is nonzero. This is a
   declared numerical criterion, not a claim that the algorithms are bit-exact.
5. NF4 references must use the same packed weights and quantization state as the
   fast path, never an independently quantized or bf16 base. Report quantization
   and compute dtype explicitly. Unrun NF4 cases remain UNVERIFIED.
6. Exercise checkpoint(use_reentrant=False) and applicable streamed paths with
   more layers than buffer slots. Check all layers' adapter gradients. CPU
   streaming evidence does not replace CUDA/NF4 streaming evidence.
7. Verify the fast path is actually taken: expected patch counts and custom
   autograd Functions, not just equal output from a silent PEFT fallback. A
   same-process deliberately changed adapter/control must be detected.
8. QKV regression: with X.requires_grad=False, no frozen-base dequantization is
   performed in backward; adapter gradients match the reference. With
   X.requires_grad=True, all required base contributions to dX remain.

## Loss-curve rule

Use a random tiny Llama and fixed synthetic token batches; record architecture,
seed, batch shape, optimizer, learning rate and dtype. Run 50 updates with ordinary
PEFT and the fast paths from identical initial states. All three fast paths must
be installed and exercised. Record every step's loss, not only the last value.

PASS: every finite reference and fast loss has the same formatting to three
decimal places (format(value, '.3f')). Report unrounded differences as well.
A shorter debugging run may be useful but is not the 50-step acceptance result.
A broken/disabled fast path that accidentally makes the two arms identical must
not pass the path-execution checks. A failure is kept, not hidden by changing the
seed, precision, step count or learning rate after observing it.

## Single-layer timing rule

Only measure performance after the corresponding correctness gate passes.
Compare identical shapes, dtype, input-gradient requirement and quantization.
Use warm-up, repeated synchronized CUDA events for forward+backward, raw per-arm
samples, and ABBA interleaving. Prefer fresh processes per arm. Capture peak
allocated VRAM after resetting peak counters, and distinguish it from reserved
memory. Record card, compute capability, driver, torch/CUDA/PEFT/bitsandbytes
versions, SM clock, power state and GPU peers per arm when available.

CPU timing is debug-only and cannot support a CUDA performance claim. A missing
clock/power/peer validity sample or interference is disclosed; contaminated arms
are VOID, not included in the headline. Keep raw data for VOID arms. If the
validity evidence is insufficient, the timing verdict is NO VERDICT.

There is no promised speedup threshold. Report time and memory for every path,
including no measurable gain or a regression. A single-layer ratio is not an
end-to-end multiplier. Full comparison with Liger/Unsloth waits for a separately
approved compatible Linux GPU (16 GB+, suitable architecture and Triton runtime).

## Hardware and completion status

Kaggle T4 is a first free CUDA validation target, not proof of compatibility with
all required regimes. Use one visible GPU. Do not silently replace required bf16
checks with fp16; report each regime separately. T4 lacks native BF16 Tensor Cores
and is outside current Triton's officially supported NVIDIA 8.0+ capability range.

CPU_PASS / CUDA_UNVERIFIED is a partial result, not complete D2 acceptance.
An infrastructure blocker includes the exact command, commit, versions and full
error. All commands and original JSON/CSV/logs belong with the resulting report.
The team reviewer, not the implementer, marks the task Done. T00's definition and
F1 approval remain organizational prerequisites until confirmed by the team.
