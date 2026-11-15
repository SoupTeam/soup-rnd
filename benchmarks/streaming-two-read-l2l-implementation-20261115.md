# Ticket 08: two-read L2L implementation

CPU implementation and benchmark-driver support are ready. The full-suite check, Kanana
comparison and final stage-1 verdict remain pending.
No GPU was rented or used for this work.

`L2LStep.run` supplies separate routing-record contexts for each policy (layer, micro-batch).
The no-grad forward records routing; recompute replays it and skips expert reads; expert
backward reads the weights again. Reference passes do not enter these contexts. The existing
hook guard refuses forward hooks on the L2L wrapper, native decoder, MoE block and routed
experts before they can
observe the recompute's zero output. RoutingGuard still compares routing during recompute.

The full 32-cell tiny CPU matrix covers GA and L2L, ordinary, grouping, two-read and both
CPU levers, at k = 1, 4, 16, 64. Two-read arms use two reads per hit in both modes. Losses,
LoRA gradients, updated parameters and AdamW state match ordinary bit for bit. Separate tests
cover hook refusal and recovery after failure.

The benchmark launcher preserves `l2l_complete` as arm A (three reads). The new
`l2l_two_read` arm is B (two reads). Existing stage-1 captures and verdicts are not reinterpreted.
After capturing B, `oc_arm_run.py judge --two-read --runs RUNS --k K` additionally requires
B-versus-A and B-versus-ordinary accuracy pairs; the default judge preserves the old pairs.

## Approved validity rule

`ArmWatch(foreign_read_fraction=0.05)` opts into the proposed Linux threshold
`max(1_000_000_000, 0.05 * own_read_bytes)`, permitting equality. The generic ArmWatch default retains ADR 0002's
original threshold and boundary. Evidence records the chosen fraction and effective limit.
Unknown sensor readings and ambiguous disk bounds retain their existing handling.

Tests use the ticket's seven historical void cases (foreign GB / own GB):
3.91/142, 3.90/142, 1.57/343, 3.89/343, 3.90/343, 1.79/655, 3.92/655.
All pass with the opt-in threshold; 5.01/100 remains void. These tests do not alter stage-1 data.

## Driver support

`oc_stage1_drive.py capture --two-read` omits GA complete, includes B, and requires two reads per
hit before judging
B against A and ordinary. Both capture k values (2 and 8) must pass before A/B timing.
`blocks --comparison AB` runs A,B,B,A with the existing five-block pilot and `blocks_to_add`.
`stage1 --two-read` captures accuracy first, then runs C2 k=2, A/B k=1, A/B k=2; it does not
repeat C1 or C2 k=1. `verdict --two-read --ks 1 2` reports those three comparison points.
Use a fresh root on the new host; old-host records and pilot sigma must not be copied into it.

The `--two-read` CLI profile selects the approved fraction 0.05. With a positive fraction, each verdict also includes `original_threshold` and `sensitive_to_foreign_reads`.
The original-threshold view reclassifies copies using the saved raw disk bounds, retains
other required checks, and never rewrites rows or watch files. Missing evidence remains
unknown. The series stops after two recorded runs with foreign reads above 5% of own reads,
keeping both runs for inspection. The user approved this rule on 2026-11-15;
`streaming-two-read-l2l-rule-20261115.md` is frozen by the commit adding it.

## Remaining acceptance work

- Run the full suite once at the very end of the whole ticket, after the remaining work.
  Between commits, use targeted checks only. The attempted local full run was not green.

- Explicit permission for GPU rental, then the 30-minute host check and disk qualification.
- Kanana accuracy for B against A and ordinary at k = 2 and 8, under A3, before timing.
- Five new clean C2 k = 2 blocks on the new host, then A/B k = 1 and k = 2 ABBA pilots
  and required additional blocks. Do not combine hosts or reuse the previous pilot sigma.
- Final report and PR with the measured complete-versus-ordinary and two-read contributions,
  void caveats, physical 12 GB GPU and emulated RAM-limit caveats as applicable.

The prior C2 k = 2 result remains NO VERDICT. No speed claim follows from CPU read counts.

## Validation

- Final targeted checks: 249 passed, 5 skipped (103 streaming tests and 146 benchmark tests).
- Final launcher, driver and validity checks: 145 passed, 5 skipped.
- Ruff: clean across `src/soup_cli/`, `tests/`, and `benchmarks/`.
- Mypy: no issues in `l2l_step.py`, `arm_validity.py` and `oc_stage1_drive.py`
  (the harness driver uses `--follow-imports=skip --explicit-package-bases` to avoid its
  dual script/package import paths).
- `expert_stream.py` has six existing mypy errors; the same six reproduce from starting HEAD
  (`1c5fd99e`) with no implementation changes.
- Full suite attempted with the two macOS auth-status exclusions from DECISIONS.md. It was
  stopped at 79% and is not green: failures/setup errors had occurred, and execution remained
  inside a very slow native CPU BF16 GEMM without handling SIGINT. A native macOS sample
  confirmed `at::native::DEFAULT::bf16_dot_with_fp32_arith`; the process was then terminated.
- Eleven files with earlier full-run failures passed outside the sandbox: 362 tests. These
  included localhost and distributed-process checks prohibited by the sandbox. Later
  full-run failures were not all independently resolved before stopping the run.
- The unrelated Rewind DataLoader-worker test also stalled in the full run. After terminating
  its two workers, the suite continued; the same test passed separately outside the sandbox
  in 6.34 seconds. Full-suite validation still needs a completed run.

Final code review is deferred until all ticket acceptance boxes are complete, as requested.
