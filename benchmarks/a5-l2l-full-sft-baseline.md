# A5 baseline: branch, dependencies, environment, gaps

Recorded 2026-10-27 for ticket 01 of `a5-l2l-full-sft`. No GPU was used.

## Base

- Branch `a5-l2l-full-sft`, worktree `../soup-rnd-a5`, created from
  `origin/a4-l2l-ram-budget-spill` at `d6f7f733`. The branch has no upstream, so a plain
  `git push` cannot reach the A4 branch.
- A4 is PR SoupTeam/soup-rnd#40, state OPEN on 2026-10-27, not merged. The base stays A4,
  not `main`.
- The A4 v3 verdict (`SPILL READY FOR A3`) names `26c54ca8`. That object does not exist in
  this repository, so the commit was rewritten after the session. The v3 protocol commit
  now in history is `5ec5b600` (`bench(l2l): protocol v3`). A5 cites the verdict by the
  record in `benchmarks/probe-a4-l2l-ram-budget-spill.md`, not by the hash.
- `origin/main` (`9d8b1d7b`) is merged into the branch. `git diff HEAD...origin/main -- src`
  was empty before the merge: `main` adds A3 tests, G1 code and records, and benchmark
  data, and nothing under `src/` that A4 lacks.

## Upstream work in MakazhanAlpamys/Soup

None of the four PRs is in this repository. `grep stream_recompute` over `src/` and `tests/`
finds nothing on this branch or on `origin/main`.

| PR | State on 2026-10-27 | Touches | Effect on the L2L path |
|---|---|---|---|
| #2152 `training.stream_recompute` | merged 2026-10-23 | `stream_setup.py`, `layer_stream.py`, `layer_stream_runtime.py`, schema | Absent here. Upstream it keeps activations of the last K streamed layers and skips their recompute forward. It changes how many times a layer is loaded and run per step, so a merge would change what tickets 03 and 04 measure. Not merged for A5. |
| #2158 two-read routed experts | open | `expert_stream.py`, `layer_stream_runtime.py` | None. Experimental, API only, DeepSeek-V3 decoder only. A5 measures a dense Mistral-7B. |
| #2159 boundary LoRA save and resume | merged 2026-10-26 | `layer_stream_runtime.py` | None while the LoRA targets exclude `embed_tokens` and `lm_head`. Matters for A5 only if a run targets them. |
| #2161 selective attention checkpoint | merged 2026-10-26 | benchmarks only, no `src/` | None. Production configuration unchanged. |

Decision: no upstream PR is brought in. A5 measures the L2L path as it is on A4.

## `stream_recompute` with `l2l`

`training.stream_recompute` is not a field of this repository's schema. A config that sets
it, with `l2l` true or false and any of `on`, `off`, `auto`, is refused at load with
`unknown config key 'training.stream_recompute'`. Pinned by
`tests/test_l2l_stream_recompute_absent.py` (6 cases). Tickets 03 and 04 therefore cannot
run a mixture of modes. If #2152 is merged later, that test fails and the combination with
`l2l` must be decided then.

## Dependencies

| Ticket dependency | Reference | Status |
|---|---|---|
| A3 | SoupTeam/soup-rnd#34 | merged, merge commit `da1fc473` |
| F1 (repeats rule, validity checks) | SoupTeam/soup-rnd#3 | merged; `benchmarks/METHOD.md`, `benchmarks/harness/arm_validity.py` |
| A4 | SoupTeam/soup-rnd#40 | open; verdict `SPILL READY FOR A3`, integration into the trainer not done |

## Open: spill is not wired into the trainer

`soup_cli.trainer.l2l_trainer._store_for` builds `ActivationStore` without `spill_layers` or
`spill_factory`, and `check_activation_budget` refuses a store that exceeds free RAM. The
`ActivationStore` spill backends exist (`utils/l2l_activations.py`). By the user's decision
of 2026-10-26, integrating the A4 budget into the trainer is not part of A5. The report
lists it as open. A5 measures the existing RAM store and the NVMe spill at the store level.

## GPU resources

`vastai show user` on 2026-10-27: balance 0, balance threshold -0.01, credit 65.43,
`can_pay` True. The ticket records a balance of 0 on 2026-10-26. Whether the 65.43 credit is
spendable on rentals is not established. Tickets 03 to 05 stay blocked until the user
confirms funds. Every GPU rental goes through `vastai`.

## Environment

Working set from earlier probes (A3 ticket 05):

| Item | Value |
|---|---|
| Image | `nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| torch | 2.14.1+cu130 |
| transformers | 5.19.0 |
| peft | 0.21.2 |
| bitsandbytes | 0.50.2 |

The local macOS machine (CPU tests only) has torch 2.14.1, transformers 5.20.2, peft 0.21.2,
bitsandbytes 0.50.2. Its transformers is newer than the probe set. GPU runs pin the
probe set.

## Local test notes

- The editable install points at `/Users/Asus/Documents/soup-rnd/src`, the main checkout.
  In the worktree, run tests with `PYTHONPATH=$PWD/src`. Without it, 36 L2L tests fail
  because the main checkout lacks A2 (dropout).
- `ruff check src/soup_cli/ tests/` is clean.
- `pytest tests/test_l2l_*.py` with `PYTHONPATH=$PWD/src`: 281 passed, 15 skipped (GPU).
  The full suite was not run: `src/` is unchanged.
