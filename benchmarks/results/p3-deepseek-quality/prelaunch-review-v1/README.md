# Prelaunch verification evidence

## Final source-bound result

`review-fix-v1/prelaunch-evidence/run.json` is **COMPLETE**. The remote Kaggle CPU
run used Python3.12.3, with CUDA hidden and the native quality device set to CPU.
The exact package set is in `packages.log`; the bootstrap and commands are in
`review-fix-v1/verifier.py.txt`.

Source admission SHA256:
`064ed79ed4cb43143f9f832e190c27dd79b2184114d7ce034480bc586240e6ee`.
The runner verifies that digest, the source ZIP digest, its complete member set
and every member's SHA256 before executing. The packet takes every member's bytes
from Git commit `78a2b5e130119b59e7039ada5f38b5bbbf975903` (recorded as
`git_commit` in `execution-admission.json`), the team-lead review fixes on top of
the reviewed PR. It does not execute workstation checkout code. Documentation is
not part of the executable validation claim. Relative to `canonical-v3`, only
`p3_deepseek_quality_prepare.py`, `p3_deepseek_quality_stage.py` and their two
test files changed. [`git-source-binding-v1.json`](git-source-binding-v1.json)
binds the earlier `canonical-v3` packet.

| Check | Observed result |
| --- | --- |
| All 14 `test_p3_deepseek_quality_*.py` files | 559 passed, 46 skipped; 99.74s pytest time |
| `ruff check src/soup_cli/ scripts/ tests/ benchmarks/harness/` | All checks passed |
| Download CLI without `--approved-pilot` | Nonzero exit before creating output |
| Real process-stage success | Child output retained; exit0 |
| Real TERM-ignoring child, one-second stage deadline | Process-group cleanup; exit124; failed-stage receipt retained |
| Resource timing with absent gate and 5400-second bound | Refused with FAILED manifest; no model forward |
| Native smoke on CPU | Refused with FAILED manifest; `scientific_admission=false` |
| Live public Hugging Face metadata | Exact reviewed inventory accepted; config and first shard HEAD verified |
| Real bounded metadata download | Only the 1660-byte `config.json`; content identity checked; zero weight-payload bytes |
| Frozen scoring input admission | In-checkout pinned protocol accepted; missing default refused despite an identical sibling copy; no quality evaluation |

The final lint roots contain the complete tracked Python source for those roots
plus the required new files. Unrelated archived benchmark scripts outside
`benchmarks/harness/` are not part of that lint claim. The entire repository's
pytest suite was not run. The 46 skipped checks require native CUDA/BF16 execution;
they are not passes and are not covered by CPU substitutions.

The control-flow regressions exercise A3 and candidate refusal ordering, incomplete
observations, immutable partial evidence, source/cache/adapter/environment binding,
explicit protocol freeze, pinned metadata, HTTPS redirects, cgroup memory ownership,
required memory evidence, SIGTERM finalizers and failed process cleanup. These
checks do not establish the 671B numerical result, full-model memory or throughput.

## Retained attempts

All directories are immutable observations, not alternate candidates from which a
scientific PASS was selected. No historical tiny gate or its E3 was rerun.

| Directory | Outcome and purpose |
| --- | --- |
| `red/` | Initial verifier packet lacked the attention dependency; collection failure, not a behavioral red result |
| `red2/` | Before the early-stop fix: seven expected regression failures and three passes |
| `green/` | Intermediate implementation: 485 passed, 46 skipped; clean Ruff and real CLI smokes; superseded by review fixes |
| `canonical-v1/` | Canonical Git-byte packet: 536 passed, 46 skipped; live Hub smoke passed; two smoke-module Ruff errors retained |
| `canonical-v2/` | Corrected import ordering and line wrapping; tests, lint and command smokes passed |
| `canonical-v3/` | Removed implicit sibling-worktree scoring input; tests, lint and all command/admission smokes passed |
| `review-fix-v1/` | Final: team-lead review fixes (resumable download, stage status precedence, cache-parent refusal); 559 passed, 46 skipped; lint and all smokes passed |

The older packets used different source snapshots and verification scopes; their
manifests must not be presented as final-source validation. In
`red2/prelaunch-evidence/run.json`, COMPLETE means the verifier successfully
captured the expected nonzero regression exit, not that those tests passed.

Each `prelaunch-evidence/manifest.json` hashes that attempt's output artifacts.
`execution-admission.json` binds the executed source and input bytes. Public
receipts include logs, JUnit XML, source inventories and the final verifier text;
private provider launch metadata, authentication, repeated source ZIPs and caches
are excluded. The output manifests do not include their own digest or the
separately preserved `verifier.py.txt`.

## Not executed by this PR

- No Vast instance, paid pilot, 688.6GB weight download or NF4 conversion.
- No native smoke for the changed revision and no full-model gradient gate.
- No long step, base-model quality scoring or phase3 training/evaluation.
- No repeat, recalculation or replacement of tiny E3: it remains
  **FAIL 0.05028628354277727 >0.05** in the unchanged historical evidence.
- No retention after the pilot and no phase3 protocol approval. The user approved
  the 671B pilot itself on 2026-11-11; the offer, cap and deadline are confirmed
  at launch against the USD90 /96h proposal.

The full-model pilot must follow the reviewed runbook. CPU
success cannot bypass native Ampere/BF16/EFFICIENT admission, a failed full gate,
OOM, the spending cap or the deadline. Provider accounting, evidence upload and
verified instance plus volume deletion remain operator responsibilities; the
process wrapper is not a provider controller.
