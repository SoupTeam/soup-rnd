# D3 public SFT/L2L prospective protocol

**Status: revision 3 time-budget correction is committed at `77a6bc3`; its first formal launch ended with no verdict.**
On `76d2c42`, required current-head Ubuntu tests, lint, gate and merge freshness
passed. Its formal session stopped at the declared three-hour cap with 39 cells
complete and one timing cell without a verdict. It is retained as an incomplete
attempt; no throughput result is accepted. The original `cd6515ac` evaluation
failure is also retained in its own report.
The historical `protocol-draft.json` filename is retained; its explicit status,
switch and bound evidence determine readiness.

The new CPU regression proves that the evaluation callback created IDs on the
streamed embedding's **meta placeholder** device. Evaluation now uses the real
runtime device. The first synthetic fixture tied embedding/head and missed this
path. Revision 2 requires a pinned **four-layer untied** fixture with both
streamed boundary modules and more decoder layers than reusable buffer slots.
The failed `cd6515ac` session and the incomplete `76d2c42` session remain
distinct and unchanged. Revision 3 fixes only the prospective time envelope;
it was committed at `77a6bc3` before the failed formal launch recorded below.

The formal session at `77a6bc3` (`formal-77a6bc35-01`) started but its first
correctness child exited before model setup with `ModuleNotFoundError:
No module named 'soup_cli'`. The remote launcher supplied the checkout root in
`PYTHONPATH` but omitted `checkout_root/src`. The complete failed session,
protocol and no-verdict cell record are retained in
[`d3-public-trainer-launch-attempt-77a6bc35-20261029`](../../results/d3-public-trainer-launch-attempt-77a6bc35-20261029/).
No pretrained forward, training step or performance measurement completed.
This attempt is not deleted, relabeled or reused. The import-only correction
was verified by importing both `soup_cli.trainer.sft` and the benchmark gate
with `PYTHONPATH` set to the checkout root **and** its `src` directory.

The authoritative machine-readable draft is [protocol-draft.json](protocol-draft.json).
The source/assignment snapshot is [requirements.json](requirements.json); the
full requirement/evidence/remaining-work mapping is
[D3 acceptance audit](../../../docs/d3-task-acceptance.md).

## What this qualification must establish

Use pinned pretrained `allenai/OLMoE-1B-7B-0924` at revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`, including all sixteen trained layers.
The actual entry point is `SFTTrainerWrapper.setup()` and `.train()`, with q/v
attention LoRA, frozen routed experts and explicit auxiliary-router-loss zero.
Ordinary SFT and merged A3 L2L are distinct training strata.

Three stored formats: BF16 dense, NF4 and nested NF4. Do not pass internal `nf4`
as the public configuration spelling: the Soup quantization field is `4bit` with
the declared double-quant flag. Initial factors, final factors, gradients and
reload all preserve the actual per-codec adapter dtype; TRL's QLoRA BF16 cast is
part of the recipe, not a post-hoc excuse for mismatches.

The seven-row pre-tokenized dataset has 512 tokens per row and frozen ignored
labels. Batch one and accumulation two produce windows **2, 2, 2, 1**, four SGD
updates. First window is warmup; three measured windows contain **2,560 input
tokens** altogether. The last window is partial: never derive throughput by
pretending every step processed two rows. Freeze actual sampler order and label
denominators and require identical delivered inputs between arms.

The native-expert resident ordinary-GA oracle uses independent reconstruction
of the identical prepared bytes. The non-expert public streaming setup remains
the same. Here **resident means resident routed experts**, not an unsupported
claim that the entire model has been resident. Pair native/sequential,
grouped-resident/grouped and Triton-resident/Triton with matching expert kernels.

## Correctness before performance

Run 36 fresh-process correctness cells: three formats, two trainer modes and six
declared execution modes. Capture forward outputs/routing and all-layer input /
router derivatives separately; check every trainable parameter gradient and
every real optimizer update against the ordinary native-expert GA oracle.
Retain exact dtypes, shapes and bytes, including signed zero.

Each process must pass A/A, the fixed effective negative control, exact restored
replay and public adapter save/reload. Clone/restore initial factors, optimizer,
sampler and CPU/CUDA RNG; the negative control's +0.5 perturbation is fixed before
measurement. If it does not change logits and a gradient, report no verdict.
Do not tune the perturbation after seeing its result. No `allclose`, statistical
noise-floor relaxation, compare-and-recompute fallback or loss-only substitute
can pass this gate. A3's separately accepted Mistral tolerance is not D3's rule.

CUDA BF16/cuBLASLt, no TF32 and reduction/split-K `(False, False)` are fixed.
The Ada dispatch profile uses native OLMoE GEMM for rows 1–128 and 481+, grouped
GEMM for 129–480. Require positive executed grouped and Triton paths; do not
force routes or change cutoffs to manufacture coverage after a failed cell.

The source asks for a 16 GiB+ Linux CUDA card. Capacity discovery records
25,250,627,584 CUDA-visible bytes on this RTX4090; the corrected minimum is
16 GiB. An earlier unexecuted draft minimum exceeded the observed CUDA capacity
and was rejected by the final host check. That diagnostic is retained; neither
the numerical gate nor the physical GPU was changed to make it pass.

## Two separate timing questions

For each format/trainer stratum, use five fresh-process symmetric pilot blocks:

```text
A = sequential expert loop
B = grouped restoration/GEMM + native activation
C = same grouped pool/GEMM + Triton activation
order: A B C C B A
```

Positions 0/1/4/5 form a real A-B-B-A grouping comparison; positions 1/2/3/4
form a real B-C-C-B activation comparison. Both cancel linear drift without
mixing unrelated GPUs, routing policies or resident/streamed memory setups.

The primary arm value is synchronized measured wall seconds / 2,560 input tokens.
Retain each step, GPU event durations and token counts. Model preparation,
compilation, reference/correctness capture and reload are outside timing peaks.
Profile separately; count actual CUPTI CUDA kernel events and retain traces.
Measure allocated and reserved training peaks separately from RSS, staging,
arena/workspace bounds and device telemetry. No modeled activation saving is a
whole-step VRAM result.

The planner projects both comparisons into the unchanged F1 sigma formula and
10% effect size. The larger of the two five-block estimates selects a joint
first target. At that target, recompute once on that prefix for a joint final
target. Never request a second top-up or more than ten blocks per stratum.
This shared six-arm/joint-top-up construction is a predeclared extension of the
two-arm scheduling rule, not a claim that the F1 implementation was modified.

| Phase | Pilot/minimum cells | Maximum cells |
| --- | ---: | ---: |
| Correctness, including controls/reload inside each cell | 36 | 36 |
| Fresh-process timing, six strata | 180 | 360 |
| Separate GPU profiles | 18 | 18 |
| Total | 234 | 414 |

These are process/cell counts, not a short-run promise. The revision-2 session
at `76d2c42` reached its three-hour cap after 40 cells had started: 36 correctness
cells and three timing cells completed, and the fourth timing cell had no verdict
because only 118 seconds remained against its 900-second cell allowance. This was
a session-budget stop, not a numerical failure. The incomplete run is retained and
none of its timing cells is reused for a formal verdict.

Using the observed rate, the declared 234-cell minimum projects to about 17.6 hours
and the 414-cell maximum to about 31.1 hours. Revision 3 fixes the whole-session
cap prospectively at **36 hours**, with a 15-minute per-child cap. The cell budget,
zero replacements and one session per committed source remain unchanged. Stop with
no verdict if either budget expires; do not retry under another output name. At the
currently selected instance rate of $0.613/hour, 36 hours is about $22.07 before
bandwidth. A positive speedup or lower whole-step memory is not an acceptance
threshold.

## Completed preparation and required freeze

The new RTX4090 KVM identity/software/cuBLAS pins are recorded. The earlier
2 GiB synthetic disk fixture is distinct from the completed actual-shard controls.

- All three original checkpoint tensors and config match their pinned SHA-256.
  Public sharding/setup prepared dense/NF4/nested NF4 without a pretrained forward.
- The complete prepared manifest binds 157 files, including exact Arrow data,
  ignored labels, original row JSON, initial adapter dtypes and immutable cache bytes.
- All **48 actual expert shards** were empirically mapped; the other bound files
  use that same guest filesystem. Actual-shard baseline/own/independent-foreign
  controls passed **OK/OK/VOID**. This is not physical-host/cache certification.
- The corrective **four-layer untied random-weight** CUDA launcher self-test
  passed **36 numerical cells and 18 profiles**, including ordinary SFT/L2L,
  expert outputs, all-layer derivatives, A/A, fixed negative/restored controls
  and same-recipe reload. Both embedding/head are confirmed streamed boundaries.
- The corrective harness/planner/analyzer suite passes **74 CPU tests**.
  The historical broad runtime selection passed **999**, with eight skips and
  116 GPU deselections; production source files are unchanged by this callback
  fix. On exact head `76d2c42`, full-repository Ubuntu tests, lint, gate and
  merge freshness pass. Optional platform/smoke jobs are skipped.

Before the formal run:

1. Preserve the current preparation and exact host. A reboot, data mutation or
   software/GPU change requires rediscovery and a new prospective binding.
2. The main merge is already completed in `cd6515ac`; revision 3 is committed at
   `77a6bc3`. The operator commits/pushes this retained launch failure and the
   corrected import-path requirement as a new head before another formal session.
   No agent commit/push is performed.
3. Export only canonical committed Git blobs from a clean checkout. The remote
   runner requires `FROZEN-D3-TRAINER-SOURCE.json`, the rule verbatim and unchanged
   data/software/boot/GPU identities before any formal child allocation.

The formal prepared-manifest path is fixed in the rule to
`/root/d3-public-trainer-preparation-01/prepared-02/prepared-after-restart-r2.json`.
Keep the output inside its bound workspace. Do not copy/alias that manifest to
create a second marker directory, remove a session marker or retry under another
output name. The runner records the complete child command for the reviewer.
The remote parent and every child must receive
`PYTHONPATH=<checkout_root>:<checkout_root>/src`; the root is needed for
`benchmarks`, and `src` is needed for `soup_cli`.

Read-only local checks, from the repository root:

```cmd
python -m benchmarks.harness.d3_trainer_protocol check
python -m benchmarks.harness.d3_trainer_protocol plan --out d3-trainer-plan.json
python -m pytest tests/test_d3_trainer_protocol.py tests/test_d3_trainer_gate.py tests/test_d3_trainer_analysis.py -q --no-cov
python benchmarks/protocols/d3-public-trainer/verify_evidence.py
```

`check` refuses an uncommitted corrective checkout; the main merge is finished.
`plan` does not allocate GPU memory or start a child. `export` refuses missing or
changed bindings, failed/stale launcher tests, dirty checkout and unfinished merge.
Do not run the old `d3_training_gate.py`: it would test a different adapter fixture.

The separate synthetic development launcher is `python -m
benchmarks.harness.d3_trainer_selftest --synthetic-checkpoint PATH --library PATH
--out NEW_PATH`. It rejects all model bytes except the pinned four-layer untied
random fixture; the seed-71 generation recipe and exact software are retained
in `untied-preflight.zip`. It is not a shortcut for the pretrained gate.

## Retained evidence and preparation repairs

`setup-evidence.zip` retains setup/manifests, all per-replay records, initial and
positive saved adapters, device traces, source snapshots and preliminary failures.
The first preparation failed the public path-containment guard before model
setup; the corrected workspace obeys that guard. The first launcher exposed
Python-int conversion of Trainer's token-count tensor: CUDA loss division rounded
one bit differently. L2L now preserves the original tensor; no tolerance changed.
Output capture also handles checkpoint early-stop by recording the initial
forward output and recomputation derivatives separately.

The original full setup archive retains redundant saved adapters from every
control replay. It and the native synthetic tensor supplement are backed up
locally and identified by SHA-256; they are **not claimed as Git-published
payloads**. See `FULL-ARCHIVE-RETENTION.json` inside the published ZIP and
[native-supplement.json](native-supplement.json). These are synthetic development
artifacts, not trained OLMoE results. All exclusions/failures remain visible.
`untied-preflight.zip` retains the corrective fixture, source, actual controls,
positive adapter files and all 18 CUDA traces. `untied-full-archive.json` identifies
the complete retained archive, including native tensors and redundant replay
adapters; those extra payloads are not claimed as Git-published files.
`r2-cpu-evidence.zip` retains the failing-before/passing-after device regression,
74-case current CPU run and exact corrective source. The old preparation packets
stay unchanged. `previous-attempt.json` retains the first formal FAIL; its full
897 MB archive was downloaded and verified before preparing a new revision.
The verifier checks published and nested archive hashes/CRCs without a GPU.
Add `--backup-directory C:\ml\soup\work` to verify all four retained local full
archives too; a reviewer can verify the Git publication without those local files.

## Verdict and final task scope

The failed `cd6515ac` Linux CI also exposed publication lint, a global `os`
test double breaking pytest capture, a readiness test assuming a dirty checkout
and a missing opt-in default in the recipe snapshot. Those confirmed causes are
repaired; the historical analyzer reproduces the same analysis bytes. See
`ci-repair.json` and `ci-repair-evidence.zip` for local checks. Two integration
cases reported FAIL before the CI crash but pass locally with the CI
Transformers/PEFT versions. Linux reproduction traced both to an unfrozen CPU
thread count: the default-thread run fails both, while an explicit one-thread
CPU fixture gives **39/39 PASS**, including the full integration/security
selection. The fixture restores the previous thread count; byte-exact assertions
and CUDA math policy remain unchanged. This qualifies the declared one-thread
CPU recipe, not arbitrary parallel BLAS. The formal runner already selects one
CPU thread. The next committed head must complete full Linux CI; its later tests
were never reached by the interrupted job.

The operator restarted the server during investigation. Boot identity is now
`2142c629-3e42-4d7c-a70f-0b42b8161c4d`. All 157 prepared file hashes remain
unchanged; all 48 shards were remapped and F1 controls again passed OK/OK/VOID.
The original prepared manifest and failed-session marker remain intact. The new
manifest/rule binds the new boot; `d3-restart-evidence.zip` retains rediscovery.
No pretrained forward/training was repeated after the revision-2 session cap.
The historical 36/18 synthetic CUDA preflight remains evidence for the unchanged
numerical source on the earlier boot, not a claim of a new CUDA preflight after
this restart.

A numerical mismatch fails; OOM, interruption, missing/ineffective controls,
missing device traces, unknown/void F1, inactive required paths, identity drift or
exhausted precision/time/attempt budget gives no verdict. Preserve every result.
The revision-2 run is `incomplete_no_task_verdict`. Revision 3 is committed and
frozen at `77a6bc3`, but its first session ended before model setup with no
verdict due to the recorded launcher environment error. No pretrained
qualification or throughput verdict is claimed. Any subsequent formal session
must use a new committed head and the corrected `PYTHONPATH`; preserve this
attempt unchanged.

The current B1 interface contract is accepted under the identified owner's
delegation, with no retrospective coordination claim. The new gate does not
resolve literal packed-weight dequant/GEMM fusion or restricted repository issue
tracking. Source page 10 says SwiGLU is wrap-only; the custom Triton activation
arm remains the operator-requested research supplement, not required D3 fusion
or automatically accepted production integration. The acceptance audit retains
these distinctions. Frozen experts plus attention LoRA and four SGD
steps do not establish quality/convergence, full-weight FT, 8 GiB fit or final
owner/reviewer task acceptance. Existing measured studies and all failed attempts
stay unchanged.
