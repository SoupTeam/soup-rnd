# F3: audit of published fine-tuning benchmarks with the `soup bench train` checks

## Status

Finished. The decision rule was committed in `0e17ab6d` (Unix time 1792100938) before the first
arm, whose `box_stamp_before.unix_s` is 1792106218.86, so the rule came 5,280 s earlier. The
rule section below is unchanged since that commit (a test pins its sha256).

Outcome in one line. The zero-gradient control was rejected in both paths. C1 arm A
(Unsloth `2026.1.2`) is **pass** in two fresh processes, so no public number was struck. C1
arm B (Chronicals) is **out of scope** and has no verdict. C2 and the speed blocks did not
run. "No public number changed" is a finished result under this rule, not a failure.

Scope of this result. It is the reduced audit that fit the budget: one Unsloth arm on an
altered-condition C1 (see the conditions table). It does not verify the published
configuration and it makes no cross-framework speed comparison. The historical reports in
`benchmarks/results/f3-benchmark-audit/` are kept as they were produced. Review of the PR then
fixed three paths of the harness that the rented run did not exercise (a speed run is checked
against its own report, training runs inside the guarded lifecycle, the raw evidence uploads
before the instance is destroyed); the historical run used the earlier code.

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

## Results

Labels: laptop results are `"RTX 3050 Laptop, control"` and `"RTX 3050 Laptop, rehearsal"`,
the rented card is `"A100-PCIE-80GB vast.ai, real"`. Cards never share a column and no
number from one card is compared with a number from another (house rule 3).

Harness commit: results were committed in `aba89347`; the harness is `benchmarks/harness/f3_*.py`.
The commit checked out on the instance was not recorded, so I cannot tie the run to a single
hash more tightly than the branch `feat/f3-benchmark-audit` at `127d4ab0` to `aba89347`.
Versions: Unsloth and unsloth-zoo `2026.1.2`, torch `2.9.1+cu128`, triton `3.5.1`, transformers
`4.57.3`, trl `0.24.0`, peft `0.18.0`, Python 3.12, Chronicals at `9b33605177a98bc798387c1742b9d36e53ea9712`
(not installed, not run). Dataset `yahma/alpaca-cleaned` at revision
`12567cabf869d7c92e573c7c783905fc160e9639`, 51,760 examples, token sha256 for
`Qwen/Qwen2.5-0.5B` `f968d82c9244011d9c36d88cad0a00d8c8c1f8e69f35e53c900d89af390afed3`.
Versions and hashes: `benchmarks/results/f3-benchmark-audit/setup-notes.md`.

### Verdicts

| Row | Published (A100-40GB, not measured by me) | Ours (A100-PCIE-80GB vast.ai, real) | Ours (RTX 3050 Laptop, control) | Verdict | Inference step |
|---|---|---|---|---|---|
| Zero-gradient control, native trainer | none | not rerun on the A100 | rejected: `grad_norm` and `parameters_changed` both refused (SmolLM2-135M, 12 steps, `weight_decay` 0) | control passed | none |
| Zero-gradient control, Unsloth trainer | none | not rerun on the A100 | rejected: both checks refused | control passed | none |
| C1 arm A, Unsloth `2026.1.2` | 11,736 tok/s ("Unsloth (actual training)", figure 10 and table 2) | `pass` in two fresh processes, 290 trainable tensors, `grad_norm` finite and non-zero, `parameters_changed` true, 20 of 20 steps | none | pass | not reproduced: no number struck |
| C1 arm B, Chronicals `9b33605` | 41,184 tok/s | out of scope, no run | none | no verdict | none |
| C1 speed row | 11,736 vs 41,184, not compared by me | not run | none | no verdict | none |
| C2 | 11,699 (Chronicals LoRA+) vs 2,857 (Unsloth MAX) | not run | none | no verdict | none |
| Run at 46,482 tok/s ("Unsloth (grad_norm=0)") | 46,482 (the text rounds to 46,000) | not run | none | config not found, not audited | at most not reproduced |

The line on 46,000: **config not found, not audited**. The preprint names no script, version
or date for it, and the Unsloth repository, issues and docs hold no such number. The status of
this number is **not checked**. It is neither confirmed nor refuted. Step 3 is not allowed.

Our peak allocated memory for arm A at batch 16 was 11.18 GB (11,182,846,976 bytes). The
preprint gives 12.8, 16.8 and 22.4 GB for the same config on another card; no comparison is
made. Our `total` and `useful` token counts and the step times are in the raw JSON and are not
a speed claim. The first step includes compilation and took 15.0 s in process 1 and 4.8 s in
process 2, which is why the two reports differ in timing and agree in checks.

`gpu_clock` for arm A: the SM clock sampled 210 to 1410 MHz and the clock event reasons
included `sw_power_cap` (5 of 26 samples in process 1). The speed row did not run, so this
did not gate a verdict. Neighbour disk reads were not checked: `foreign_reads` was `unknown`
because the model path maps to no block device. `suspend` and `power` answered `ok`.

### Raw reports

- Rental: `benchmarks/results/f3-benchmark-audit/rental/f3-results/audit-c1.json`,
  `benchmarks/results/f3-benchmark-audit/rental/f3-results/c1/A-unsloth-p1.json`,
  `benchmarks/results/f3-benchmark-audit/rental/f3-results/c1/A-unsloth-p2.json`,
  `benchmarks/results/f3-benchmark-audit/rental/f3-results/lifecycle/summary.json`,
  `benchmarks/results/f3-benchmark-audit/rental/f3-results/observe-sleep60.json`,
  `benchmarks/results/f3-benchmark-audit/rental/f3-results/refusals.jsonl` (empty: no check refused),
  logs `benchmarks/results/f3-benchmark-audit/rental/c1.log` and
  `benchmarks/results/f3-benchmark-audit/rental/bootstrap.log`, notes
  `benchmarks/results/f3-benchmark-audit/rental/NOTES.md`.
- Control: `benchmarks/results/f3-benchmark-audit/control/control.json`,
  `benchmarks/results/f3-benchmark-audit/control/control-native.json`,
  `benchmarks/results/f3-benchmark-audit/control/control-unsloth.json`.
- Rehearsal: `benchmarks/results/f3-benchmark-audit/rehearsal/audit.json`, void or out of scope
  arms included. There is no void run on the A100.
- Copy: the same files are in the private Hugging Face repo `m1kado/f3-audit-results`. Private
  means a reviewer needs access from the owner; the paths above are the public reference.
- Spend: about 0.15 USD of the 2.50 USD ceiling (offer 1.00 USD per hour, instance 56167498,
  alive about 12 minutes).

## What was not done and why

- **Chronicals (arm B).** Its loop is custom (`ChronicalsTrainer`), not an HF `Trainer`, so the
  collector cannot attach. Recorded as out of scope, no verdict. No Chronicals versus Unsloth
  comparison and no "our number versus theirs" column exists.
- **C2.** Not run. `f3_arm.py` has no LoRA path and `unsloth_max_benchmark.py` is not in this
  repository (it lives in the Chronicals repository, fetched at run time).
- **Speed blocks.** The speed blocks did not run: arm B has no verdict, so the rule forbids the speed row.
- **Unsloth `2026.10.6`.** The extra arm never ran. Whether it fits transformers 5.19.0 was not tested.
- **The `destroy` path with the per-instance key.** Not verified. `CONTAINER_ID` was not set in
  the ssh shell, the on-instance destroy raised after the upload, and the instance was destroyed
  from the laptop by hand.
- **The control was not rerun on the A100.** The record holds the laptop reports,
  labelled `"RTX 3050 Laptop, control"`, copied to the instance. The control model is
  SmolLM2-135M, not Qwen2.5-0.5B.
- **Other frameworks.** Axolotl, LLaMA-Factory and the Unsloth docs have no published pair of
  number and config on a small model that fits the budget.
- **Docs of Soup.** The "2-5x faster" lines (`README.md`, `docs/models.md`,
  `docs/backends-and-ops.md`) are not edited here. A separate card covers them.

## Prepared statements

Prepared before any result, one per outcome. Use the one that matches. The Chronicals
preprint (arXiv 2601.02609) is not peer-reviewed, and it is inconsistent: three peak memory
figures for one config, two sequence lengths, two licences and commands with no script.

- **Confirmed.** "Chronicals reports that a 46,000 tok/s Unsloth run had a `grad_norm` of
  zero. We ran Unsloth `<version>` on the public config `<config>` at `<commit>` and our run
  was refused by `<checks>` in two fresh processes, with the zero-gradient control rejected.
  This is the step `<2 or 3>` result, `our run invalid` or `published run invalid`."
- **Not confirmed.** "Chronicals reports that a 46,000 tok/s Unsloth run had a `grad_norm` of
  zero. We could not find the config for that run, so we did not audit it. We ran Unsloth
  `2026.1.2` on the public C1 config and it passed in two fresh processes. This says nothing
  about the published run."

For this audit the second statement applies.

## Conditions table

A changed condition means the number is not compared with the published one. No number was
compared in this audit, because no speed row ran. The table is here so the changes stay visible.

| Config | Condition | Published | Ours | Match |
|---|---|---|---|---|
| C1 | Model | Qwen2.5-0.5B | Qwen2.5-0.5B | yes |
| C1 | Dataset | Alpaca-cleaned | `yahma/alpaca-cleaned` at `12567cab`, 51,760 examples | yes, revision pinned |
| C1 | Sequence length | 512 (the preprint also names another) | 512, truncated | yes |
| C1 | Batch per device | 16 | 16, fit on 80 GB | yes |
| C1 | Gradient accumulation | not stated for table 2 | 2 (effective batch 32) | unknown |
| C1 | Precision | bf16 | float32 weights with bf16 autocast, because the collector hashes weights with `.numpy()`, which fails on bf16 | **changed** |
| C1 | Optimizer | not stated | Trainer default, learning rate 1e-5, `weight_decay` 0.01 | unknown |
| C1 | Steps | 10 warmup and 100 measured (also 100 warmup and 5 runs in another section) | 20 steps, no separate warmup, first step includes compilation | **changed** |
| C1 | Gradient checkpointing | not stated | not enabled | unknown |
| C1 | Packing | not stated for C1 | off | unknown |
| C1 | Token formula | `steps x batch x seq / time` | collector `total` (padded tokens per batch) and `useful`; no tok/s claim | **changed** |
| C1 | Card | A100-40GB (also named A100-80GB) | A100 PCIE 80 GB | **changed**, never compared |
| C1 | Unsloth version | not stated | `2026.1.2` (closest release to the preprint date) | inferred |
| C2 | all | LoRA r32, `unsloth_max_benchmark.py` | not run | not applicable |

## Refusal table

`refusals.jsonl` of the rental run is empty: no report had `valid: false` and every arm that
started produced a report. The rows below list what refused or did not start.

| Config | Arm | Process | Check | Outcome | Reason | Commit | Report |
|---|---|---|---|---|---|---|---|
| control | native | 1 | `grad_norm` | refused (expected) | step 0 logged `grad_norm == 0.0` | `c125040c` | `benchmarks/results/f3-benchmark-audit/control/control-native.json` |
| control | native | 1 | `parameters_changed` | refused (expected) | trainable parameters bit-identical between first and last step | `c125040c` | `benchmarks/results/f3-benchmark-audit/control/control-native.json` |
| control | unsloth | 1 | `grad_norm` | refused (expected) | step 0 logged `grad_norm == 0.0` | `c125040c` | `benchmarks/results/f3-benchmark-audit/control/control-unsloth.json` |
| control | unsloth | 1 | `parameters_changed` | refused (expected) | trainable parameters bit-identical between first and last step | `c125040c` | `benchmarks/results/f3-benchmark-audit/control/control-unsloth.json` |
| C1 | A-unsloth | 1, 2 | none | no refusal | all checks ok | `aba89347` | `benchmarks/results/f3-benchmark-audit/rental/f3-results/audit-c1.json` |
| C1 | B-chronicals | none | none | did not start | custom loop, not an HF `Trainer`; out of scope | `aba89347` | `benchmarks/results/f3-benchmark-audit/rental/f3-results/lifecycle/summary.json` |

## Labels

Results are labelled `"RTX 3050 Laptop, control"`, `"RTX 3050 Laptop, rehearsal"` and
`"A100-PCIE-80GB vast.ai, real"`. Published numbers sit in their own column, marked
"A100-40GB, not measured by me".

The datasets and models above are not in git. The tokens stay outside the repository.
Results are also in the private Hugging Face repo `m1kado/f3-audit-results`.
