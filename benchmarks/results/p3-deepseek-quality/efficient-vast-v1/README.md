# Strict EFFICIENT attention smoke

Execution qualification passed. The native tiny-model numerical gate is **FAIL**, so this evidence does not admit full-model training or a quality claim. No671B weights were downloaded.

## Frozen scope

- RTX3060 12GB, native BF16, SM8.6; torch2.11.0+cu130. Full package inventory: `setup/pip-freeze.txt`.
- EFFICIENT only for both arms/layouts; MATH only for the alternative gate floor at8×64. No fallback. Deterministic algorithms enabled with `warn_only=False`; TF32 disabled.
- H128, Q/K192, V128; B1; L1024/k4 and L4096/k1; seed23 and the existing optimizer/schedule. No mandatory8GB claim.
- Attention protocol SHA256`fc887c5a996bd58536ef0325b61405cd73a2a83cf363fcc4156204ab180e1ed6` was fixed before outcomes. The real cycling pack remains unchanged; this smoke uses explicit synthetic fixtures.
- Research base commit:`364d003bfa0cc83f02a716e687b5cc7b41056be8`. Quality implementation remains in the research worktree; no full-study or PR publication occurred.

## Verification

| Snapshot | Result | Interpretation |
|---|---|---|
| v1 |432quality cases passed; full selected command465passed/1failed | Existing marker guard found six private CUDA probes in prior/frozen tests. No quality-file failure. |
| v2 |445passed/1failed,0skips | Sole failure: the endpoint CLI test still requested CPU. Production correctly refused it. |
| v3 |50passed,0skips; Ruff clean | Full scorer/timing files rerun with a real native BF16/NF4 CUDA endpoint fixture and strict runtime validation. |

Final coverage is446distinct passing cases:396unchanged cases from v2 plus50from v3. This is not one clean full-suite command. Implementation bytes are identical between v2 and v3; only the endpoint test file changed. Exact commands, environment and hashes are in `integrated-inputs-v*.json`, `integrated-checks-v*/execution.json` and `tests.xml`. All execution occurred remotely.

The four instance-loss/resume cases passed for both arms/lengths, comparing adapter digests, digest chains, loss trajectories, optimizer/scheduler and RNG. The primitive separately repeated outputs and Q/K/V gradients bitwise at both lengths. Receipts contain actual efficient forward/backward operators and `fmha_cutlassF`/`fmha_cutlassB` CUDA kernels, not just requested flags.

Review regressions cover missing/wrong backend receipts, mixed Q/K/V geometry, gate/training environment mismatch, cancelled per-microbatch loss differences, incomplete forward coverage, changed source/policy identities and preserved timing failure evidence. Earlier failures remain in `review-regressions-red-v*.log`.

## Standalone native execution

Outside pytest, `standalone-smoke-v1/driver.py.txt` executed three updates for each arm/layout and a resource-only forward. The model has8,185,608parameters and two layers; these numbers cannot establish671B fit or performance. Full loss trajectories, learning rates and initial/final parameter hashes are in `summary.json` and each run's `steps.jsonl`.

| Arm | L / k | CUDA allocated peak MiB | Boundary payload MiB | FP32 scales KiB |
|---|---:|---:|---:|---:|
| bf16_l2l | 1024 / 4 | 548.39 | 2.0000 | 0 |
| oracle_block64_e4m3 | 1024 / 4 | 548.39 | 1.0000 | 64 |
| bf16_l2l | 4096 / 1 | 1904.97 | 2.0000 | 0 |
| oracle_block64_e4m3 | 4096 / 1 | 1904.97 | 1.0000 | 64 |

CUDA peaks are whole-process allocator observations. Boundary columns are measured owned host tensor-storage capacities, not total activation memory. Oracle allocations, codec temporary upper bounds, pinned storage, RSS and cgroup observations remain separate in each step record; they are not added to manufacture an activation-only peak.

## Numerical gate: preserve the failure

The unchanged three-layer tiny gate completed with A3, E1 and E2 PASS. E3 failed:

| E3 metric | Observed | Frozen criterion | Result |
|---|---:|---:|---|
| Global relative L2 | 0.05028628354277727 | ≤0.05 | FAIL |
| Minimum tensor cosine | 0.9947999141202121 | ≥0.99 | PASS |

The int4 control failed E1/E2/E3 and passed discrimination. v1 and v2 recorded the same E3 value. No bound, scientific input or gate fixture was changed to obtain a PASS. Raw evidence is in `integrated-checks-v2/evidence/gate-bfloat16-cuda/`. A real671B verdict remains unmeasured.

## Evidence and closeout

`final-evidence-v1.tar.gz` SHA256:`a80dca2972a05cfb2c8596593e2e90c54f2fe3186052e798661094bfb0e9e4b9` matches the server archive. Three exact source ZIPs, executed launchers, prior failed attempts and the final standalone artifacts are retained.

Owned instance60113395 was destroyed and the specific-instance API confirmed absence at`2026-11-10T20:43:42.622625+00:00`. Unrelated instances were not mutated. Provider reportsUSD0.116 for this smoke andUSD0.543 across all three owned research instances. These are rounded post-teardown [instance charges](https://docs.vast.ai/api-reference/billing/show-charges), not a final invoice; details are in `provider-charges.json` and `resource-closeout.json`. No active owned rental remains.

Further rental, full-model download/pilot, protocol freeze and the paired quality experiment require separate authorization and their own numerical admission.
