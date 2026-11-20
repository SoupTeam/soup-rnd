# Two-read expert cache integration: CPU preparation

**Latest GPU result:** [Moonlight four-arm qualification](GPU-RESULTS.md).

Original preparation status: CPU correctness controls passed; CUDA memory, real-model quality and speed are unqualified.

Source [b9d32144](https://github.com/SoupTeam/soup-rnd/tree/b9d32144c0131f5fb261ab25df298e9308ac1506) on
`koko/two-read-cache-20261025` merges the exact upstream PR #2158 head `f2599ddc`
with cache candidate `c22e5f90`, then adds expert-level cache ownership and admission.
Upstream two-read commits and authorship are preserved. This report does not replace
the older production source in PR #5. The combined patch against `c22e5f90`, exact
executed changed files, controls and logs are in [cpu-evidence.zip](cpu-evidence.zip),
with internal byte/SHA-256 index `files.json`; [summary.json](summary.json) binds the packet.

## Implemented

- Shared `ExpertFetcher`: bounded host LRU of owned packed tensors and complete NF4
  sidecars; first-fit immutable dense device hotset. Entries use reader/file, expert,
  dtype, quantization and code-table identities. Device entries do not duplicate host entries.
- Warm hits validate open-file identity too. Entries never alias reusable staging;
  runtime cleanup releases caches/staging even if another source fails to close.
- Setup API `configure_stream_expert_cache`: complete baseline/candidate F/B probes,
  state preservation, host headroom, shared device cap and setup/optimizer reserves.
  Later decoder/activation planners hold the full lazy device ceiling; later SSD
  hotset rebuilds hold the full lazy host ceiling. Warm capacity can be deliberately
  charged twice. This is conservative admission, not optimal or speed-aware placement.
- No CLI/config/default change. Direct low-level ceilings count owned tensor payload,
  not total allocator/RSS/VRAM. Use the setup helper before training; re-admit after
  source, budget, shape or adapter changes. Existing two-read architecture, hook and
  routed-expert-adapter refusals are retained.

## Validation

598 passed, 11 CUDA skips in the retained focused
regression selection; Ruff passes. The full repository suite was not run.
All required CPU NF4 cells actually executed. Real tiny DeepSeek-V3 models cover
resident/default/two-read, FP32/BF16/autocast, host/device/both/disabled caches,
two accumulated AdamW updates, loss/logits/dX/router and every LoRA gradient,
all final adapters and optimizer moments. All adapters actually change.
The same frozen NF4 base also passes all default/two-read cache trajectories.

CPU equality is bitwise and additionally checks unchanged reference-relative
`rtol=.01, atol=1e-5`, finiteness and three-decimal loss gates. CPU device caching
is a semantic control, not evidence of CUDA copies, allocation or acceleration.
Five deliberate ownership/identity/precision/reservation mutations are killed;
each source file is restored byte-for-byte. A real CUDA NF4 trajectory test is
prepared and skipped on this CPU build.

Earlier setup controls exposed a host-availability-dependent tiny positive fixture
and a changelog baseline/reference mismatch; the fixture now fixes only synthetic
host availability while production refusal controls remain. An isolated run lacked
TRL; its failed report is retained as `before-trl.*`. Final dependencies and commands
are recorded in the packet. No numerical threshold was weakened.

## Next GPU qualification

Compare ordinary three-read, two-read, expert cache, and two-read plus expert cache
on the same supported real MoE checkpoint, adapter/input state and physical GPU.
Qualify full numerical captures before unprofiled speed/RAM/VRAM measurements.
Record real physical reads separately from logical expert visits and include setup
costs. Retained activations already remove checkpoint recompute, so the 3-to-2 saving
must not be counted twice. This preparation rented no GPU and spent no GPU budget.

## Reproducible GPU runner prepared

[Runner, commands and retained CPU verification](GPU-RUNNER.md): 172 passed, 1 CUDA skip, Ruff/pre-commit clean. This is preparation only; no GPU rental or speed result.

## Real Moonlight GPU qualification completed

[Four-arm results, exact gates, memory and cleanup](GPU-RESULTS.md).
