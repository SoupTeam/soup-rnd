# P3 phase 0: DeepSeek L2L integration

Working contract for three parallel implementation tracks, prepared on 2026-11-04.
Phase 0 delivers a correct, configurable tiny-DeepSeek path and a measured memory plan.
The 30B/671B throughput campaigns and a claim of 50x acceleration belong to later phases.

This foundation changes documentation only. The new configuration names and planning
interfaces below are implementation proposals; they are not accepted by the current schema.
Every actual configuration field must be implemented in
[`config/schema.py`](../src/soup_cli/config/schema.py), the single source of truth.

## Source snapshot and dependencies

- Product baseline for integration: `main` at `d272e476`, including
  [B1.3 / #47](https://github.com/SoupTeam/soup-rnd/pull/47) for internal L2L/MoE
  backward and [A4 / #40](https://github.com/SoupTeam/soup-rnd/pull/40) for activation spill.
- [G4 / #35](https://github.com/SoupTeam/soup-rnd/pull/35) and
  [C3 / #48](https://github.com/SoupTeam/soup-rnd/pull/48) are also in `main`.
  G4's compressed store remains in the benchmark harness. C3's public L2L allocation
  path is uncompressed and its partial-weight-cache feature is restricted to dense Llama.

Existing code to extend:

| Component | Entry point | Existing contract / missing connection |
|---|---|---|
| Config and launch | [`schema.py`](../src/soup_cli/config/schema.py), [`SFTTrainerWrapper._build_l2l_trainer`](../src/soup_cli/trainer/sft.py) | `l2l` already exists; `k` is `gradient_accumulation_steps`. The schema refuses `per_expert` with L2L, and #47 keeps the public Llama/Mistral guard. |
| Activation store | [`ActivationStore`](../src/soup_cli/utils/l2l_activations.py), [`_store_for`](../src/soup_cli/trainer/l2l_trainer.py) | Per-micro-batch capacities, window growth, RAM and spill are implemented. The trainer selects/creates the store in `_store_for`. |
| Compression | [`CodecActivationStore`](../benchmarks/harness/g4_activation_codec.py) | G4 has per-row FP8 and scale accounting; its wrapper does not accept the trainer's `mb_shapes`. |
| Memory accounting | [`HostMemoryBudget`](../src/soup_cli/utils/host_memory_budget.py), [`BudgetedActivationStore`](../src/soup_cli/utils/budgeted_activations.py), [`plan_c3_l2l`](../src/soup_cli/utils/c3_l2l_budget.py) | Allocation leases and an uncompressed C3 partition exist. [#53](https://github.com/SoupTeam/soup-rnd/pull/53) adds the FP32/BF16 MoE calculation; trainer integration and FP8 accounting are pending. |
| Available RAM | [`ram_budget`](../src/soup_cli/utils/l2l_spill.py) | Existing available-memory/cgroup detection, headroom, reserve and spill accounting are reusable. |

## Parallel tracks and file ownership

| Track | Owner | Work and completion evidence |
|---|---|---|
| B1.3 and FP8 store | Batyr | B1.3 is merged. Adapt G4 into product code, preserve the store contract below, and supply a shared allocation estimator. Validate round trips, variable shapes, cleanup and DeepSeek routing with compression. |
| B1.4 and correctness | Aldiyar | Own schema fields, recipe snapshots, public DeepSeek/per-expert launch gates and trainer wiring. Extend the tiny native DeepSeek comparisons to `k = 1/4/16/64`, including real optimizer updates and short final windows. |
| MoE memory planner | koko | Implement a planner using `HostMemoryBudget`, the store's allocation estimator, host/device MoE costs and available-memory detection. Select effective `k` and compare the breakdown with actual allocations. |

Start compression and planner implementation in separate new product modules. Aldiyar owns
the config/trainer integration edits; Batyr supplies store construction and allocation
estimation, and koko supplies the plan. Bring changes to shared files through the owning
track's PR rather than independently replacing `_store_for` in three branches.

Run the uncompressed DeepSeek comparison once B1.4 is wired up. FP8 correctness and
memory checks also need the new store. The planner can be reviewed on its own.

## Activation-store contract

Preserve the interface already consumed by `L2LStep`:

```text
put(layer, mb, tensor) -> None
get(layer, mb) -> tensor
begin_backward() -> None
end_step() -> None
close() -> None
```

- Keep `n_layers`, `k`, `mb_shapes`, `dtype`, `device`, `spill_layers` and `stats`
  usable by the trainer. `get` returns the original logical shape on the compute device
  in the compute dtype; `dtype` does not become the compressed payload dtype.
- Preserve per-position capacities and short final windows. Capacity growth releases
  old asynchronous users and allocations before replanning and reserving new storage.
- At the codec boundary retain logical `[batch, sequence, hidden]` rows. The current
  trainer flattens capacities; forwarding those flat capacities to G4 would change the
  meaning of its per-row scales. Adapt this explicitly and test mixed sequence lengths.
- Move reusable codec code under `src/soup_cli/utils/`; product code must not import
  `benchmarks/harness`. Keep existing G4 harness callers working through that shared code.
- Supply one pure allocation estimator used by both the planner and store constructor.
  Report payload, scales, pinned-allocation rounding, staging and transient peak separately.
  Scale slabs stay accounted for even if payload spills.
- Reserve before allocation. Release a reservation only after copies/workers have drained
  and owned storage has been dropped, including constructor failures and repeated `close`.

Phase 0's compressed route targets RAM storage. Keep existing uncompressed spill behavior
working. Compressed spilling needs its own alignment, allocation and correctness controls
before it is enabled; a RAM-only FP8 implementation must reject unsupported spill requests.

## Configuration and automatic k

The proposed additions are opt-in:

| Field | Proposed default | Meaning |
|---|---|---|
| `training.l2l_activation_codec` | `none` | `none` or `fp8_e4m3`; selects boundary storage independently of model compute precision. |
| `training.l2l_auto_k` | `false` | Select the largest RAM-fitting `k` up to the configured accumulation count. |

`training.l2l` remains a bool. With automatic selection off,
`training.gradient_accumulation_steps` retains its existing meaning and behavior.
With automatic selection on it is the requested upper bound; for a search up to 64,
the user requests `gradient_accumulation_steps: 64`. Reject new options when their
required L2L/SFT/streaming path is absent, rather than accepting ignored settings.

Plan before creating the HF trainer/optimizer schedule and before collecting its first
window. Pass the effective `k` consistently to HF training arguments, accumulation/loss
scaling, store capacity and logging. Keep the effective value fixed during a run and
persist enough resolved-plan information to validate it on resume. A smaller last window
uses the same normalization and optimizer-step rules as ordinary accumulation.

If no `k >= 1` fits the known safe RAM envelope, fail before allocating with the requested
and available bytes. If no safe envelope can be established, fail automatic selection
explicitly. Manual behavior remains unchanged. Changing a requested codec requires an
explicit configuration change; do not silently switch an FP8 request to another codec.

## Planner input, output and accounting

The planner takes model dimensions, maximum/per-micro-batch logical shapes, compute dtype,
codec, requested maximum `k`, available host memory and a measured/bounded MoE workspace
description. The return contract is a plan containing:

- requested and effective `k`, codec, compute dtype and shape envelope;
- available host bytes, the safety policy applied and its limiting source;
- a component breakdown of owned host allocations and peak transient reservations;
- separate device-workspace estimates/measurements and any unsupported or unknown costs;
- the shared host budget/reservation data needed when the store is constructed.

Use the existing cgroup-aware detection and `HostMemoryBudget`. Distinguish a total-memory
envelope from memory that is already free: subtract live allocations and safety reserves
exactly once. The budget accounts for allocations; process/cgroup memory is measured
separately. Enabling C3's dense-Llama partial-weight-cache flag is not a prerequisite for
using its budget primitive on DeepSeek.

Count each cost in the place where it actually lives:

1. Boundary payload and FP32 scales; actual pinned allocator rounding; staging and codec
   compression/restoration peaks.
2. Host routing records retained across layers/windows, reader staging and fixed host users.
3. Device hidden states, routing masks, gathered expert inputs and route weights, gradients,
   expert arena, and transient recompute workspaces. Use lifetime-aware bounds: #47 runs
   recompute/backward one micro-batch at a time, so do not assume every expert autograd
   input stays alive for all layers and all `k`.

Host and device peaks are separate constraints. A successful host reservation does not
establish the 8 GiB VRAM limit. Validate shapes against the reserved plan before growing
the store; cover insufficient budgets, resizing, exceptions and cleanup with controls.

## Two integration questions to resolve with tiny-model evidence

### FP8 and expert routing

#47's `RoutingGuard` compares the experts chosen in the original forward and recompute
using exact equality. Lossy boundary compression may change the router input and its
choices. Keep this guard and add an FP8/native-DeepSeek test that exercises it. A routing
change must stop the step. A design that replays recorded routing instead requires its
own explicit implementation and correctness evidence; deleting the guard is not acceptance.

Lossless FP32/BF16 correctness and G4's lossy FP8 gates are separate checks. G4's Mistral
result does not establish DeepSeek accuracy. Record the E1/E2/E3 error metrics and routing
outcome; use the existing G4 thresholds, and keep the BF16 path available explicitly.

### Expert weight reuse across micro-batches

#47 currently executes a layer for every micro-batch, and its expert pool refetches each
hit expert on a lease. Its test `test_b13_l2l_moe.py` asserts
`expert_fetches == 3 * total_hits`, counting hits per micro-batch across the forward,
recompute and expert backward. This provides correctness and bounded expert-weight
storage, but does not implement expert reuse across the whole accumulation window.

Add a tiny-model diagnostic at `k = 1/4/16/64` for layer loads, expert fetches and logical
expert bytes. Logical counters and physical disk reads must remain distinct. Record the
measured scaling and the mechanism needed to amortize expert reads before accepting P3's
one-read-per-window performance assumption. Route grouping inside one micro-batch alone
does not establish reuse across all `k` micro-batches. This prerequisite does not require
a giant-model rental or authorize a throughput claim.

## Branch and PR workflow

This foundation branch is `codex/p3-phase0-foundation`. Its PR targets `main` and carries
the common contract. Each implementation uses its own branch and PR targeting `main`:

| Work | Suggested branch | Dependency |
|---|---|---|
| B1.3 | `b1.3-moe-backward` / #47 | Merged. |
| FP8 store | `codex/p3-phase0-fp8` | Bring the G4 store and allocation estimator into product code. |
| B1.4 | `codex/p3-phase0-b14` | Enable the public DeepSeek path and run the tiny-model comparison. |
| Memory planner | `codex/p3-phase0-moe-planner` / [#53](https://github.com/SoupTeam/soup-rnd/pull/53) | FP32/BF16 planner in review; validate with the shared codec and trainer. |

Create work branches from current `origin/main`, and reference this contract PR. While it
is a draft the contract is readable on the foundation branch; its documentation commit
does not need to be copied into every implementation PR. After dependencies land, merge
fresh `origin/main` into each working branch and rerun affected checks. Update the source
snapshot in evidence when product code moves.

The repository's CI workflow targets PRs to `main`; PRs aimed at this foundation branch
would miss those checks. Each branch owner pushes only their own implementation branch.
Keep #47's implementation/review in #47 and integrate its accepted result through `main`.

## Phase 0 completion checklist

- [x] B1.3 / #47 is merged.
- [ ] Preserve per-micro-batch store, spill and C3 regression coverage in the combined path.
- [ ] Public config starts native `deepseek_v3` SFT with L2L and per-expert streaming;
  unsupported combinations still fail clearly and existing defaults remain unchanged.
- [ ] On tiny DeepSeek, `k = 1/4/16/64` matches ordinary accumulation for loss, trainable
  gradients and updated trainable state: FP32 bitwise, BF16 within the fixed A3 tolerance.
  Include multiple updates, masked labels, mixed lengths and a short final window.
- [ ] The FP8 store satisfies the trainer contract, G4 numerical gates, routing checks
  and allocation/cleanup controls. Preserve an explicit BF16 baseline if FP8 is blocked.
- [ ] Automatic `k` respects its requested upper bound and safe host envelope; the store
  uses the planner's same allocation estimator and reservations. Compare actual host and
  device peaks separately, including expert/routing workspaces.
- [ ] The expert-read diagnostic is recorded and its implications for the next performance
  phase are resolved. Required reuse changes and their owners are explicit if still open.
- [ ] Each implementation passes appropriate CPU tests and repository lint; CUDA checks
  are reported separately with hardware and skips. Final evidence identifies the combined
  source revision. The full required CI is evaluated on each final PR head.

Start with tests already covering the touched path: `tests/test_l2l_*.py`,
`tests/test_g4_codec.py`, `tests/test_c3_host_budget.py`,
`tests/test_c3_activation_spill_budget.py`, and `tests/test_c3_public_l2l.py`.
Include `tests/test_b13_*.py` and the new larger-`k` cases. Follow
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for supported Python versions, lint and test commands;
report CUDA skips rather than treating CPU success as GPU memory/performance evidence.
