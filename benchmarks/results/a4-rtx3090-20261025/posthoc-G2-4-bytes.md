# A4 G2: the 4-byte overrun (analysis after the run, does NOT change the verdict)

The verdict stays **`SPILL NOT SAFE: A3 must not ship the spill path as built`**. As written,
rule v1's G2 (`a4_rule._cap_problems` at `e123b5ff`) bounds the **whole** growth of the CUDA
host allocator during the timed steps by the activation budget. Every arm grew by budget +
4 B, so G2 is a legitimate FAIL under that rule. No record of the session was changed; this
note and the local probe were added after it.

## What the run data shows (read from `arms/*/*.json` in this directory)

| fact | arms / blocks |
|---|---|
| out-of-store excess (allocator peak − baseline − store bytes held) = +4 B, arm A (all-pinned, `spill_layers` 0, budget `1`) | 140 / 140 |
| the same +4 B excess, arm B (spill at 20, 24, 28 or 32 layers) | 140 / 140 |
| excess difference B − A per block (both A, both B; not a raw-growth difference, which would include each arm's store) | 0 in 70 / 70 complete blocks |
| store bytes == plan bytes == budget (`cap.ok`, `allocator_matches_plan`, `held == host == budget`) | 280 / 280 |

Since held == budget in every arm, "over budget" and "outside the store" are the same 4 B here.
Beyond its store, the spill arm allocates no more host-allocator bytes than the all-pinned arm.
That does not make the overrun a measurement error: it is a real 4 B pinned allocation that both
arms make, outside the store.

## Source: confirmed locally

`a4_probe.run_timed` (lines 488-520 at `e123b5ff`): on warm-up steps `continue` (505-506) runs
before `float(x)` on the losses (510), so the losses are read on the host only on timed steps,
after `reset_host_peak()` (490-494) and after `mode_arm` took the baseline (564-565, following
`release_pinned_cache()`). The loss is float32: HF `ForCausalLMLoss` upcasts the logits with
`.float()` (transformers 5.19.0, the run's version). `_local_scalar_dense_cuda` in torch v2.14.1
(`aten/src/ATen/native/cuda/CUDAScalar.cu`) copies a CUDA scalar to the host through a pinned
`empty_cpu({1}, dtype, pin_memory=true)`, which is a `sizeof(dtype)` request to the caching
host allocator.

**Confirmed locally** (`posthoc_probe_hostalloc.py`, `posthoc-probe-result.json`): RTX 4050 Laptop, driver
616.92, Windows 11, torch 2.14.1+cu130 (same build as the run) in a throwaway venv, no rental,
about 20 min. After `torch._C._host_emptyCache()` (what `release_cached_pinned_memory` calls),
`host_memory_stats()["allocated_bytes.current"]`:

| case | delta |
|---|---|
| `float(torch.ones((), device="cuda", dtype=float32))` | **+4** |
| control: same tensor, device work, no host read | **0** |
| bf16 / fp64 scalar read | +2 / +8 (block = sizeof(dtype)) |
| k = 2, 3, 4 losses × 5 steps of `float(loss)` | +4 each (one block, reused) |
| the arm's order: baseline, warm-up without a read, reset peak, timed read | peak − baseline **+4** |
| the same with a read in warm-up | peak − baseline +4 (block is created before the reset, still counted against the baseline) |

The last row matters: reading the loss during warm-up would **not** have avoided the overrun,
because the baseline is taken before warm-up. Only taking the baseline after a host read would
hide it, and that would silently change what G2 measures. It must not be done without a rule
change.

Not shown: that the run's 4 B block is exactly this one. The probe reproduces the size,
dtype and reuse, and the code path shows no other host read inside the timed window. The run
itself did not record per-allocation host events, so the attribution is local reproduction plus
code reading, not a trace from the box.

## The rule change that followed

Protocol v2 (`../../probe-a4-l2l-ram-budget-spill.md` §3.1) bounds each spill arm's excess
outside its store by the excess both all-pinned arms of its block share. It applies to v2
sessions only; this session is never re-judged under it (`v1-under-rule-v2-ILLUSTRATION.json`
shows what v2 would read here, as an illustration, not a verdict).
