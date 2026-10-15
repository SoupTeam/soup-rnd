# F3: audit of published fine-tuning benchmarks with the `soup bench train` checks

## Status

This is the skeleton. It holds the decision rule and nothing else. The first commit of
this record must come before the `box_stamp_before.unix_s` of the first arm. A reviewer
compares the two to confirm the rule was not fitted to a result (house rule 1). No code
has run and nothing has been measured. The condition and refusal tables below are empty
on purpose and are filled after the arms run.

## Question

The Chronicals preprint (arXiv 2601.02609) says a widely quoted 46,000 tok/s fine-tuning
run had a `grad_norm` of zero, so the model was not training. `soup bench train` refuses
such a run. This record asks whether the same checks, applied to the public configs of
other frameworks, change any public number.

## Words

`GLOSSARY.md` is local and a reviewer does not see it. Every term the rule uses is
defined here.

| Term | Meaning |
|---|---|
| arm | One variant under comparison, identified by everything that distinguishes it from the other variants. Here: one framework at one version on one public config. |
| run | One execution of one arm in a fresh process. |
| block | Four runs in the order A, B, B, A. It yields one difference between the means of A and B. |
| pilot | The first 5 blocks of the speed comparison. They count toward the final test. |
| void arm | An arm with a void check. Its number is not used. Void means unknown, not slow. |
| validity row | A condition committed with the rule before the first arm, which checks whether the measurement measured what the rule reads, not whether the hypothesis holds. |
| no verdict | The outcome of a gate that depends on a void arm, on an unknown required check, or on a failed control. It is distinct from fail. |
| public config | A training setup that someone else published with a throughput number, pinned here to a repository commit and library versions, and run by this audit. A published number with no recoverable setup is not a public config. |
| public number | A throughput figure someone else published, for a public config or for a setup that cannot be recovered. |
| struck number | A public number whose arm failed a validity check in two fresh processes while the zero-gradient control passed. Its status changes from measured to struck. Its value is not compared with ours. |
| zero-gradient control | A run in which every label is masked (`labels == -100`) and `weight_decay` is 0, so nothing can train. The checks must reject it, in the native trainer and in the foreign trainer the collector is wired into. |
| not reproduced | The audit ran a public config at its pinned commit and got a valid run, or could not run it. It makes no claim about the publication. |
| our run invalid | Our run of a public config under comparable conditions failed a check in two fresh processes with the control passing. It says nothing yet about the published run. |
| published run invalid | Our run is invalid, and either the author's own log shows the same failure, or the published config and versions are exact and the cause is confirmed in code. |
| `step_count` | The number of optimizer steps the collector counted. A run that reaches fewer steps than configured fails this check. |
| box stamp | A snapshot of the machine taken before and after an arm. `box_stamp_before.unix_s` is the Unix time of the snapshot before the arm starts. |
| ArmWatch | The harness object that records three checks per arm: `suspend` (the machine slept), `power` (the machine left mains) and `foreign_reads` (other processes read the disk under test). Each check returns ok, void or unknown. |
| `gpu_clock` | The SM clock of the card recorded per arm. Read by hand, not a check. |

## Arms and conditions

- **C1.** Full fine-tuning of Qwen2.5-0.5B, batch 16, sequence length 512, bf16, on
  `yahma/alpaca-cleaned` at a pinned revision.
- **C2.** LoRA r=32, run through the `unsloth_max_benchmark` script.
- **A** is Unsloth `2026.1.2`, the release closest to the preprint date. **B** is
  Chronicals at commit `9b33605`.
- One extra arm on current Unsloth `2026.10.6` runs only if budget remains after
  everything else. The main verdict never depends on it.
- Card: one rented A100 PCIE 80 GB. Numbers from different cards are never compared
  (house rule 3).

The conditions that must match for a number to be compared: model, dataset, sequence
length 512, batch, accumulation, bf16, optimizer, LoRA parameters, step count,
checkpointing, packing and the token formula. A changed condition gets a row in the
conditions table and that number is not compared with the published one. The batch of
the publication is kept on 80 GB. On OOM the batch drops to the largest that fits and
the record names it a changed condition.

## Decision rule

Committed before the first arm. It is not edited afterwards.

### Validity verdict, per published arm

An arm is run twice, each time in a fresh process. The three training checks are
`grad_norm`, `parameters_changed` and the trainable count. `step_count` is read with them.

- **pass.** All three checks and `step_count` are ok in both processes, the two reports
  are identical, and the zero-gradient control was rejected in both paths.
- **fail.** At least one check refuses in both processes, and the control was rejected
  in both paths. The inference step reached is written next to the verdict.
- **no verdict.** Any of these holds: the control was not rejected in either path; a
  validity row is void or unknown; the arm did not start; the two reports differ.

Required `ArmWatch` checks for the validity verdict: none (ADR 0006). The three training
outcomes are deterministic and do not move with sleep, power or neighbour disk reads. A
void check still voids the arm whatever `required` says. A void block is replaced, at
most twice per config.

The control fails if any run of it reports `valid: true`, or if only one of `grad_norm`
and `parameters_changed` refuses. One working check must not hide a broken one. A failed
control turns every verdict into no verdict, with the reason written.

### Speed verdict, separate row

Run only if both arms of that config are pass. Blocks of speed are not run otherwise.

- Required `ArmWatch` checks: `suspend` and `power`.
- `foreign_reads` is not required. A container path on an overlay file system maps to no
  block device, so the check would be unknown on every arm. The record states: neighbour
  disk reads were not checked.
- `gpu_clock` is read by hand before the verdict. A card under a power cap does not pass
  as clean.
- Pilot of 5 blocks, then one top-up by the repeats rule of `benchmarks/METHOD.md`:
  N = ceil(7.85 (sigma / Delta)^2), with Delta equal to 10% of the mean of A.
- Test: means of A and B over blocks, two-sided t-test, alpha 0.05.
- Token count: the formula of the publication, `steps x batch x seq / time`, as `total`.
  The `useful` count goes in a second column. A gap between them is a note in the
  report and does not change the status of a number.

### Validity rows

| Row | Verdict it gates | Required checks |
|---|---|---|
| Control rejected in the native trainer and in the Unsloth trainer, with both checks refusing | every verdict | none |
| `weight_decay` is 0 in the control | every verdict | none |
| Two fresh processes produce the same report | validity | none |
| Arm started and reached `step_count` | validity | none |
| Both arms of the config are pass | speed | none |
| No void run in the block | speed | `suspend`, `power` |
| Neighbour disk reads | speed | not required, "not checked" |

### Inference steps

1. **Not reproduced.** The config ran at its pinned commit and gave a valid run, or did
   not run. No claim about the publication.
2. **Our run invalid.** Fail in two fresh processes, control passed, conditions comparable.
   No claim yet about the published run.
3. **Published run invalid.** Step 2, plus the author's log showing the same failure, or
   an exact config and version with the cause confirmed in code. My own run alone never
   reaches step 3. "Invalid" is never written without its step.

### The run at 46,000 tok/s

Config not found, not audited, at most step 1. The preprint names no script, version or
date for it, and a search of the Unsloth repository, issues and docs finds no such number.
Step 3 is not allowed for it.

## Out of scope

Axolotl, LLaMA-Factory, the Unsloth docs, the edit of the "2-5x faster" lines in Soup's
docs (a separate card), changes to `src/soup_cli/bench/` (`backend: unsloth` stays
refused by `soup bench train`), training loops that do not use `Trainer` (recorded out of
scope with the reason, no verdict), and any comparison across cards.

## Minimum that counts as success

- The zero-gradient control is rejected in both paths, native and Unsloth.
- Both arms of C1 reach an outcome under this rule, no verdict included, with the reason
  written.

"No public number changed" and "all arms no verdict" are finished results, not failures.

## Order of cuts when money runs short

1. The extra arm on current Unsloth `2026.10.6`.
2. Speed blocks.
3. C2.

Spending stops at the ceiling of 2.50 USD (rent 1.95, storage and traffic 0.25, margin
0.30), with 1.50 USD held in reserve and a hard limit of 3 hours on the instance. The
instance is destroyed after the last upload.

## Unresolved

- **Dataset licence mismatch.** The copy `yahma/alpaca-cleaned` is tagged `cc-by-4.0` and
  its source is `cc-by-nc-4.0`. Not resolved here and not checked legally. The dataset is
  not in git. Only the revision, the example count and the sha256 of the tokenized sample
  are.
- The preprint gives three peak memory figures for C1 (12.8, 16.8 and 22.4 GB), two
  sequence lengths, and commands without scripts. These are facts about the source, not
  a judgment about its authors.
- Facts marked as inferred in the research notes (the behavior of `Trainer` with every
  label masked, version compatibility, memory at batch 16) are confirmed by the first
  run, not assumed.

## Conditions table

| Config | Condition | Published | Ours | Match |
|---|---|---|---|---|
| | | | | |

## Refusal table

| Config | Arm | Process | Check | Outcome | Reason | Commit | Report |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Labels

Results are labelled `"RTX 3050 Laptop, control"`, `"RTX 3050 Laptop, rehearsal"` and
`"A100-PCIE-80GB vast.ai, real"`. Published numbers sit in their own column, marked
"A100-40GB, not measured by me".
