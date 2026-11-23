# Ticket 12: where the gradient non-determinism comes from

Model Moonlight-16B-A3B (`2a9a177f`), card RTX 3050 Laptop (sm_86, 4 GiB, one card, exclusively
used here), torch 2.14.1+cu130, CUDA 13.0, transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2.
Mode: bits. Every run was a fresh process, on AC power (`/sys/class/power_supply/ACAD/online` =
1 before each launch), inside `systemd-inhibit --what=sleep`. Each row in
`raw/*/nondet_rows.jsonl` states model, card, window, arm, mode and torch version.

Raw tensors and logs: `raw/` (git-ignored). Scripts: this folder. Comparison of the saved
tensors is by `torch.equal` and, from the final scripts, by raw bytes and dtype as well (both
agree on every pair reported here).

## Answer

The source is the **SDPA memory-efficient attention backward**: its `dQ` accumulation across key
blocks is order-dependent, and PyTorch runs the non-deterministic variant while determinism is
`warn_only`. `dK` and `dV` are bitwise stable, the forward output and the loss are bitwise stable,
and at S=64 (one key block) the whole run is bitwise stable. Strict determinism
(`torch.use_deterministic_algorithms(True)`, not `warn_only`) switches the `dQ` path to a
deterministic one without leaving the memory-efficient kernel, and two strict runs agree in all
216 gradients.

## 1. Red experiment: A0 does not reproduce itself

A0, k=1, S=512, default SDPA, deterministic (`warn_only=True`), three fresh runs
(`raw/A0-default-k1-rep{1,2,3}-s512/`, plus the earlier reproduction in
`raw/repro-default-s512/`):

| pair | loss equal | LoRA grads differing | max abs diff | max abs grad |
|---|---|---|---|---|
| rep1 vs rep2 | yes | 105 / 216 | 5.89e-3 | 1.52 |
| rep1 vs rep3 | yes | 105 / 216 | 4.62e-3 | 1.52 |
| rep2 vs rep3 | yes | 105 / 216 | 6.43e-3 | 1.52 |

Loss is bitwise equal (2.774064540863037) in every pair: nothing in the forward diverges. The
differing tensors are only `lora_B` gradients (`lora_A` gradients are exactly zero because PEFT
initialises B to zero and the model is at step 1). The per-layer shape is the key clue:

- layers 0–25: all four `lora_B` gradients differ (4 × 26 = 104);
- layer 26 (the top layer): only `q_proj.lora_B` differs (1) — 105 in total.

Layer 26 is the first layer the backward reaches, and there the only differing gradient is the
one fed by the attention backward's `dQ` output path. Layers below inherit the difference and all
four of their projections differ.

Magnitude at the source and its amplification downward (rep1 vs rep2, relative L1 difference of
the `q_proj.lora_B` gradient): layer 26 1.5e-7 (1363 of 49152 elements, max abs 6.0e-7), layer 25
1.3e-4, layers 0–24 between 1e-3 and 8e-3 (max abs at layer 0 7.1e-4). The seed is a rounding-size
difference at the top; the 4e-3-scale differences at the bottom are that seed amplified through
the stack, not a second source.

## 2. The `warn_only` list

`raw/A0-default-k1-rep1-s512/warnings.txt` contains exactly one torch warning:

```
Memory Efficient attention defaults to a non-deterministic algorithm. To explicitly enable
determinism call torch.use_deterministic_algorithms(True, warn_only=False). (Triggered internally
at .../aten/src/ATen/native/transformers/cuda/attention_backward.cu:910.)
```

The same lone warning appears at S=64 (where the result is nevertheless bitwise stable) and is
absent on the math backend and in strict mode. No other operation is flagged by
`use_deterministic_algorithms(True, warn_only=True)` on this model and card. Strict mode does not
raise anywhere in this model; it simply takes the deterministic attention-backward path.

## 3. Boundaries: window, layer, operation

**Window.** S=512 diverges; S=64 does not: `raw/A0-default-k1-rep{1,2}-s64/` agree in 0 / 216
gradients and in the loss. The synthetic probe (below) gives the same split: at S=64 `dq` is
bitwise equal, at S=512 it is not. The number of key blocks is what changes with S.

**Layer.** From the captured per-projection output gradients (`capture/.../output_grads.json`):
105 of 108 differ; layer 26 has only `q_proj` different (`kv_a_proj_with_mqa`, `kv_b_proj`,
`o_proj` bitwise equal), every layer below has all four different. So the first gradient that
differs in backward order is the top layer's `q_proj` output gradient.

**Operation.** The captured SDPA input gradients (`capture/.../sdpa_calls.json`, 81 calls = 27
layers × 3 steps; calls 0, 27, 54 are the top layer in steps 1, 2, 3):

- call 0 / 27 / 54: `dq` differs in every pair of runs and across the three steps of one run
  (three distinct digests), while `dk` and `dv` have the same single digest in all runs and all
  steps;
- call 1 (layer 25): all three differ, because its incoming gradient from above already differs.

Full `dq/dk/dv` tensors of call 0 are saved (`sdpa_call0_*.pt`). At the top layer, between two
runs, `dQ` differs in 1363 of 49152 elements (max abs 6.0e-7) while `dK` and `dV` are bitwise
equal; the unsigned sum of `dQ` is preserved to 5e-8 relative.

Synthetic probe on the exact MLA attention shape (q/k `(1,16,S,192)`, v `(1,16,S,128)`, bf16,
causal, `raw/probe/probe.md`): flash and cuDNN report "No available kernel" on this card, so the
default SDPA call goes to the memory-efficient kernel. At S=512, two backward passes on identical
inputs give bitwise equal output, `dk` and `dv`, and a `dq` that differs in 178 elements
(max abs 9.77e-4, signed-sum difference 9.8e-9 relative). At S=64 everything is bitwise equal.
Ten backward passes in one process at S=512 gave 10 distinct `dq` digests and 1 digest each for
`dk`/`dv`; at S=64, 1 digest for all three.

## 4. Hypothesis: order of atomic additions in backward

**Verdict: confirmed, in the narrowed form.** The order-dependent accumulation is the `dQ`
reduction of the memory-efficient attention backward, not "the backward" as a whole and not
cuBLAS:

- same process, same inputs, only the backward repeated: `dq` changes every pass (10/10 unique),
  `dk`/`dv` never change;
- only `dq` differs at the first layer reached; `dK`/`dV` are per-key-block results written by
  each block, `dQ` is summed over key blocks, which is where the order can change;
- elementwise bits change while the signed sum is preserved to ~1e-8 relative (rounding-order
  signature, not a different formula);
- S=64 is stable and S=512 is not, consistent with the number of key blocks in the reduction;
- the forward output and the loss are bitwise equal, so no matmul, quantisation or loader is
  involved;
- strict determinism makes `dq` bitwise equal (torch's own deterministic variant), without
  changing the loss or the kernel family.

## 5. Default against math SDPA

| runs | backend | mode | loss | LoRA grads differing |
|---|---|---|---|---|
| 3 fresh, pairs | default (memory-efficient) | deterministic (`warn_only`) | 2.774064540863037, equal | 105 / 216 in every pair |
| 2 fresh | math | deterministic (`warn_only`) | 2.7908143997192383, equal | 0 / 216 |
| 2 fresh | default | **strict** deterministic | 2.774064540863037, equal | 0 / 216 |

The default and math backends change the numbers themselves (different kernels), so their runs
must not be compared to each other — only within one backend. Strict mode keeps the default
kernel's numbers: its loss equals the `warn_only` default loss to the bit, and its gradients are
still a different set from the math backend's (108 of 216 equal), so strict determinism does not
silently fall back to math.

## 6. Case (b): the F outlier at k=1 on math

Out of scope of this investigation and left open. In the ticket-03 record, the one F run of seven
that disagreed with A0 also had a different loss, so that outlier is a **forward** divergence
(loader/routing), not the backward-only effect measured here. It was not reproduced.

## 7. Consequence for the track

- The harness's "deterministic" is `warn_only=True` and therefore does **not** guarantee A0 bits
  on the default SDPA backend at S=512. Equality requires the math backend (`--math-attention`)
  or strict determinism.
- The S=512 bits tables of the track therefore stay on the math backend. If the default backend is
  ever needed for speed, strict determinism holds the bits (verified here) at the cost of the
  deterministic `dQ` path.
- The PR note must say "strict deterministic mode or the math backend", not just "deterministic
  mode"; `warn_only` is the mode the earlier tables used and it is not sufficient at S=512.
- No code change in `src/` is justified by this finding: the choice belongs to the measurement
  mode, not to the trainer.

## Files

- `nondet_analysis.py` — bitwise pair comparison, LoRA gradient breakdown, canonical rows.
- `probe_sdpa.py` — synthetic SDPA probe on the MLA shape, per backend.
- `run_capture.py` — harness run with backend forcing, strict determinism, SDPA/projection
  gradient capture, warnings file and canonical row.
- `compare_runs.py` — pairwise comparison of saved runs and of captures.
- `run_matrix.sh` — the matrix below (refuses on battery or over existing destinations).
- `tests/test_nondet_analysis.py` — CPU tests for the pure logic and the scripts' contracts.

Matrix (`raw/matrix.log`): probe; A0 default S=512 ×3 (capture on the first two); A0 math S=512
×2; A0 default S=64 ×2; A0 strict S=512 ×1 (a second strict run, `raw/A0-strict-k1-rep2-s512/`,
was run by hand for the equality check). No failures.
