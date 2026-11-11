# DeepSeek full-model pilot: prelaunch review

## Scope and authorization

This PR prepares the DeepSeek-V3 671B diagnostic. On 2026-11-11 the user approved
running the pilot on the full 671B checkpoint; no smaller substitute model is
used. That approval does not cover retention after the pilot or phase 3
training/evaluation. Paid execution starts after team-lead review and the user's
confirmation of a fresh offer, its price, spending cap and deadline.

The proposed envelope is USD 90 and 96 elapsed hours. Stop computation at USD 87
or hour 95, leaving USD 3 and one hour for evidence and teardown. The operator
tracks cumulative compute, root storage, local-volume storage and traffic. There
is no automated provider budget or rental controller. The process deadline
helper cannot enforce provider billing or delete a rental.

The recorded tiny E3 remains **FAIL: 0.05028628354277727 > 0.05**. Its artifacts,
threshold and fixture are immutable. A future full-model verdict is a separate
row in the report, not a replacement or a new calculation of the tiny result.

Operator commands: [pilot runbook](p3-deepseek-quality-pilot-runbook.md).
The [price proposal](results/p3-deepseek-quality/pilot-cost-proposal-v1/README.md)
is a quote-time record; its "changes still required" section predates this PR.

## Review stack

This PR is stacked on `feat/fp8-activation-storage` ([#57](https://github.com/SoupTeam/soup-rnd/pull/57)).
Pilot-specific work starts after merge commit
`364d003bfa0cc83f02a716e687b5cc7b41056be8`, which also incorporates the already-main
B14 work from #58/#59. Compare that commit with the PR head to isolate this pilot;
the GitHub base diff also includes those inherited main changes. Retarget to
`main` after #57 is integrated. Merge alone does not start execution.

The [CPU evidence index](results/p3-deepseek-quality/prelaunch-review-v1/README.md)
identifies the tested source snapshots, actual commands and untested native paths.

## Implementation plan and review order

1. Keep the existing shared attention, data, model, training and scoring modules.
   The exact checkpoint, source/cache identity, initial adapter, pack and frozen
   scoring inputs must be reviewable from this checkout.
2. Add `p3_deepseek_quality_prepare.py` for the missing download/conversion
   commands. Refuse an unapproved invocation, overlay/non-NVMe storage, less than
   1.4 TB free or less than 64 GiB effective host RAM before downloading. Verify
   the pinned Hub commit, shard coverage, sizes and metadata identity. Convert
   using the existing NF4 noDQ/per_expert path with eight threads, then write the
   shared initial adapter once. Preserve failed-stage evidence.
3. Change gate control flow, not the numerical rules. Judge A3 immediately after
   the ordinary/repeated GA, BF16 L2L and MATH-floor passes. Refuse before FP8 when
   A3 fails or cannot be judged. Judge E1/E2/E3 immediately after FP8. Refuse before
   INT4 when the candidate fails or lacks required evidence. A successful path
   still requires all six passes and a discriminating INT4 negative control.
4. Require raw-verified full-model gate admission for resource timing, bound to
   the live source/cache/initial adapter, attention policy and software/GPU
   environment. Allow at most 5400 seconds. Bound the process group separately;
   do not compute base-model accuracy, NLL or predictions.
5. Add CPU control-flow regressions and real subprocess deadline tests. Run CPU
   checks remotely, not on the workstation. Preserve failing-before results and
   new source-bound verification. Native Ampere qualification remains a separate
   paid-run prerequisite; no T4 retry and no claim that CPU mocks prove kernels.
6. Publish a complete runbook, exact input hashes, command order, evidence rules,
   conditional long steps, reporting requirements and the phase 3 approval gate.
   Review and commit only the required source/data/evidence closure. Exclude
   credentials, raw account responses, temporary launch packages and repeated
   large archives.

## Files to review

| Area | Files | Reviewer questions |
| --- | --- | --- |
| Strict attention | `benchmarks/harness/p3_deepseek_quality_attention.py` | Is fallback impossible? Are executed kernels, backward and MLA shapes observed? |
| Data | `p3_deepseek_quality_data.py` | Same frozen cycling, masks and token normalization for both arms? |
| Source and cache | `p3_deepseek_quality_model.py`, `p3_deepseek_quality_prepare.py` | Full pinned source? Native configuration preserved? NF4 noDQ/per_expert? |
| Technical smoke | `p3_deepseek_quality_smoke.py` | Fixed native primitive and fixture workloads; no tiny gate/E3 rerun? |
| Gradient gate | `p3_deepseek_quality_gate.py` | Early refusal without changed metrics, fabricated floors or skipped required controls? |
| Long-step measurement | `p3_deepseek_quality_train.py` | Same initial adapter, N4096, B1, correct k and schedule, no OOM resizing? |
| Timing/scoring | `p3_deepseek_quality_timing.py`, `p3_deepseek_quality_eval.py` | Resource-only pilot; trained quality scoring requires a frozen approved protocol? |
| Process lifetime | `p3_deepseek_quality_stage.py` | Deadline kills descendants and preserves output; no false cleanup or scientific admission? |
| Regressions | `tests/test_p3_deepseek_quality_*.py` | Consumer-visible refusals, cleanup, coverage and identity errors? |

Bare module names in this table are under `benchmarks/harness/`.

## Scientific invariants

- Checkpoint `deepseek-ai/DeepSeek-V3`, revision
  `e815299b0bcbac849fa540c768ef21845365c9eb`.
- All 163 safetensors shards, 688586727753 bytes. Metadata/tokenizer files are
  additional files, not part of that shard count or byte total.
- Reviewed source inventory:
  `results/p3-deepseek-quality/source-inventory-v1.json`, SHA256
  `5677f32a218d96414334a1536d66a93d0bb87ac98786585faef41329f7ae7e1c`.
  The pinned revision has four selected metadata files and no generation config.
  Hub LFS identities are not a claim of a fresh local weight-content hash pass.
- Disk NF4, no double quantization, `per_expert`; native source configuration.
- BF16 compute in both arms; BF16 boundary storage versus block64 E4M3 plus Oracle.
  Oracle replay is an additional mechanism, not measured natural route stability.
- Strict EFFICIENT_ATTENTION for main passes and both long-step layouts. MATH only
  for the independently observed A3 floor on eight 64-token microbatches.
- H128, Q/K192, V128; deterministic algorithms with `warn_only=False`.
- Gate B1/L64/k8, shared initial adapter, no optimizer update. E1 RMS <=0.05 and
  worst row <=0.25; E2 recompute RMS <=0.10; E3 global relative L2 <=0.05 and
  minimum tensor cosine >=0.99. The existing A3 floor rule remains unchanged.
- Long steps B1/L1024/k4 and B1/L4096/k1, one step per arm per layout after PASS.
  The first scheduled optimizer call has LR0. It measures execution and memory,
  not improvement from learning. Do not change the schedule to manufacture an update.
- Native model tests use Ampere+ with native BF16. CPU control-flow tests are not
  numerical qualification. No native BF16 acceptance on Kaggle T4.
- Fresh output directories, no retry for PASS, no threshold or L/B/k changes.

## Review checklist before any rental

- [ ] The team lead reviewed the complete source and the unchanged prerequisites
      (provenance: `results/p3-deepseek-quality/README.md`).
- [ ] The code is frozen at the reviewed commit; the rental runs only that commit.
- [ ] No credentials, private provider responses or unneeded binary archives enter Git.
- [ ] Frozen protocol/input bytes retain their published hashes across checkout.
- [ ] New CPU regressions and command smokes have current source-bound evidence.
- [ ] Native-only checks not run for this revision are explicitly identified.
- [ ] The user approved a still-available offer, the total price and the deadline.
- [ ] The operator can upload evidence and perform/verify both instance and volume deletion.
- [ ] Phase 3 remains blocked on the future protocol hash and its separate budget approval.

## Corrections from prelaunch review

- Gate A3/candidate refusals stop before later passes; completed pass observations
  and execution identities survive a missing final manifest. Per-pass RSS/CUDA
  peaks are observed, not inferred from the host-store budget.
- Gate source hashes require the critical imported modules to come from this
  checkout, not an unrelated installed Soup copy.
- Download metadata must match the reviewed Hub inventory. CDN transport ETags
  are not LFS hashes. Exact bounded chunked payloads are accepted; every redirect
  stays HTTPS and redirect bodies are closed without unbounded buffering.
- Hybrid cgroup memory ownership is mount-aware. Missing kernel peak counters
  are explicitly unavailable; unreadable/malformed counters and incomplete
  required RSS/CUDA evidence refuse completion.
- Preparation SIGTERM unwinds its finalizers when Python can receive it. Hard
  kills preserve only available partial evidence. Stage cleanup errors still
  produce failed receipts and never claim that an unreaped child is gone.
- Training and scoring require an explicit FROZEN study protocol, not a draft
  with a matching hash. Timing cannot bypass the full gate.
- The default frozen scoring protocol must exist in this checkout. A missing
  input cannot silently resolve through a sibling worktree; an explicitly
  supplied protocol path still requires the pinned content hash.

### Team-lead review (2026-11-11)

- Download is resumable. Each file is written via `<name>.partial`, fsynced and
  atomically renamed. Transient transport errors retry up to three attempts per
  file with a fresh signed URL; integrity errors and interrupts never retry.
  `download --resume` reuses only complete files whose local SHA256 matches the
  pinned Hub LFS hash; the stage-a time allocation stays shared.
- Code freeze before rental: `load_gate` binds `source_files_sha256` to the
  executing checkout, so a post-gate harness edit invalidates the gate. The
  runbook forbids hotfixes on the rental.
- `convert` refuses a missing cache parent with an explicit message and a FAILED
  manifest. A second signal during timeout handling no longer replaces
  `TIMED_OUT` (exit 124); it is recorded as `interrupted_after_timeout`.
- Provenance of the 17 prerequisite modules is documented with their v1/v2
  protocol bindings.

The process wrapper supports non-detaching commands in its original process
group; it is not containment for children that create a new session. Provider
instance continuity and deletion remain manual. Software/GPU-class matching is
not physical-host UUID attestation.

## Reporting boundaries

Report per-step wall time, sampled RSS, cgroup lifetime peak, CUDA allocated and
reserved peaks separately. Owned boundary-store bytes, scales, padding and Oracle
storage are not a total decoder/autograd activation peak and must not be summed
with whole-process peaks. Gate timing includes diagnostic disk I/O and is not
training throughput.

The current timing sample is the first frozen dev task for each of HellaSwag,
ARC and BoolQ, ten choice requests in total. It is not a full dev sweep, does not
cover LAMBADA or secondary NLL and cannot alone admit the complete phase 3 cost.
The report must state this and separate measured rates from extrapolated costs.

After the pilot, prepare Task 2.2 with the full-model verdict beside the unchanged
tiny FAIL, measured step costs, timing coverage, a phase 3 forecast and explicit
scope-reduction alternatives. No phase 3 training or evaluation before the user
approves both its exact protocol SHA256 and budget.
