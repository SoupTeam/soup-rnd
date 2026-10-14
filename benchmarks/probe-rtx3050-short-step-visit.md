# G7 probe: the fixed cost of a streamed layer visit on short training steps

Card: G7. Parent ticket `.scratch/g7-short-step-overhead/issues/01-short-step-overhead.md`. Follows F1 (`benchmarks/METHOD.md`).

Status: measured 2026-10-15, recommendation No verdict (section 9). The rule below was committed before the first arm. The commit that adds this file is the rule commit. To check house rule 1, compare its commit time (`git log --format=%cI -- benchmarks/probe-rtx3050-short-step-visit.md`) with the box stamp of the first arm record in `benchmarks/results/probe-rtx3050-short-step-visit/`. The stamp must be later. After this commit no threshold below changes. A change after a number is seen is written as a new section, with the old text kept.

Every arm carries the label "RTX 3050, synthetic". Environment: Linux, driver 595.99.02, torch 2.14.1+cu130, bitsandbytes 0.50.2, transformers 5.19.0.

## 1. Question

How long is the fixed cost of one streamed layer visit at 16 to 128 tokens per step on this laptop, which of its four parts is large, and is any lever worth an engineering project: `tie_weights=False` upstream, CUDA graphs in the runtime, or neither?

No record in `benchmarks/` has measured a visit since the #841 cache merged on 2026-09-29. The figures the card quotes come from other machines and other code (section 8).

## 2. Fixture and step shape

- The synthetic `llama-8b` shape from `harness/synth_checkpoint.py`: 32 layers, hidden 4096, NF4 with double quantization, RAM tier, LoRA r=8 on q, k, v and o, the step probe's defaults.
- The checkpoint is written with the GPU hidden, because the writer runs out of GPU memory on a 4 GB card when it draws the embedding. Shapes and bytes are the same.
- Batch 1. Sequence length 16, 32, 64 and 128. Tokens per step equal the sequence length.
- The #842 dispatch is left to bitsandbytes. Each arm record holds the number of Linear layers on the fused path and on the old path.
- Steps per run. The bare and timer arms take 3 warm-up steps and 8 timed steps. The profile and control arms take 3 warm-up steps and 3 profiled steps (a trace of 8 steps of a 32-layer model is hundreds of megabytes). The experiment's bare arms take 3 warm-up steps and 8 timed steps.

## 3. The rule

### 3.1 Pass, fail and no verdict

Each row and each branch of the rule has one of three outcomes.

- pass: the condition holds on valid arms.
- fail: the condition does not hold on valid arms.
- no verdict: an arm the row reads is missing, void or unknown, a number it reads is missing, or a row it depends on has no pass. A no verdict is not a fail. It names the row and the rerun that would clear it.

### 3.2 Validity rows

Required checks on every arm, with the shard folder as the model path: `suspend`, `power`, `foreign_reads`, `gpu_power_limit` and `swap`. The arm's outcome combines them through F1's outcome rule (`harness/arm_validity.py`).

| Check | ok | void | unknown |
|---|---|---|---|
| `gpu_power_limit` | "Current Power Limit" from `nvidia-smi -q -d POWER`, read before and after the arm, is 80.00 W to 95.00 W | any other value | the field is unreadable |
| `swap` | the kernel's swap-in and swap-out page counters (`pswpin`, `pswpout`) did not grow | a counter grew | a counter is unreadable |

The SM clock stays on record and voids nothing.

| Instrument row | Condition | If not |
|---|---|---|
| Timer is free | The timer arm's median step is at most 2% longer than the bare arm's, at the same token count | fail. Every row that reads the timer arm gets no verdict |
| Profile is readable | The profile arm's host time per forward visit is at most 1.5 times the timer arm's (the inflation factor) | fail. The rows that read the parts get no verdict |

Rows are computed at each token count. A row that two rounds report differently, or that either round reports as no verdict, is a no verdict.

### 3.3 The control

The control arm runs at 16 tokens, once per round: the timer and profile instruments plus a 200 µs spin-wait in weight preparation and 50 trivial launches inside `functional_call`. It passes if all three hold, each against the profile arm of the same round:

1. weight preparation grows by 200 µs ± 20%;
2. launches per forward visit grow by exactly 50;
3. the timer's host time per forward visit grows by 200 µs ± 20% plus the cost of the 50 launches (the growth of launch time in the control's profile).

If the control fails, or the profile is not readable, the control row has no verdict, and the CPU-bound threshold has no verdict.

### 3.4 CPU-bound threshold

At 16 tokens, on the median forward visit: capturable host time is at least 2 times the GPU kernel time per visit. Capturable host time is the timer arm's host time per visit minus weight preparation from the profile arm. The launch time share is reported and decides nothing. See section 6 for the reason.

### 3.5 Experiment

Arm A is the code as shipped. Arm B has `functional_call` patched to `tie_weights=False`.

1. Bit-exact gate, before any timing: loss and every LoRA gradient equal under `torch.equal`, with seeded non-zero LoRA B matrices, and a control (one input token changed) that must differ. If the gate fails, the experiment has no verdict.
2. A-B-B-A blocks at 16 tokens by the repeats rule: a pilot of 5 blocks, then N blocks, then one recompute of N. A block with a void run is kept on record and replaced.
3. A two-sided t-test on the mean block difference at alpha 0.05.

The experiment always runs, whatever the threshold says.

### 3.6 Recommendations

1. "Continue: `tie_weights=False` upstream" if the gate passes and B is faster than A at p < 0.05, at any size. The gain is reported with its 95% interval.
2. "Continue: CUDA graphs in the runtime" if all of these hold: the threshold fired, the prototype gate passed, the replay saves at least 30% of capturable host time, payback is at most 100 steps, and 32 private pools of one layer each fit within 1.1 GB.
3. "Stop" if neither 1 nor 2 holds. A non-significant experiment is written as "effect below the detection limit, about 10% of a step".

A branch whose validity row failed has no verdict, and the record names the row and the rerun. The other branch keeps its result. The overall answer is each branch that passed. If none passed and a branch has no verdict, the answer is "No verdict".

The recommendation reads the 16-token rows, the experiment verdict and the prototype result. `visit_verdicts.recommend` computes it.

### 3.7 Prototype gate, cap and thresholds

The prototype runs only if the threshold fired. It is a standalone harness: one layer, 16 tokens, fixed weight addresses, run through `functional_call`, forward and backward as two graphs, with no checkpoint and no pool. Copies and stream waits stay outside the capture.

- Gate: replay against eager on a new input under exact equality. Forward: the layer output. Backward, stated separately: the input gradient and every LoRA gradient. Control: a replay of a graph captured on another layer must differ.
- 3 warm-up passes on a side stream. Capture cost is the wall time of warm-up plus capture, reported apart from the replay time.
- Payback is capture cost divided by the eager time minus the replay time, in visits and in steps.
- Thresholds: the replay saves at least 30% of capturable host time, payback is at most 100 steps, and 32 times one layer's private pool is at most 1.1 GB (decimal, 1.1e9 bytes).
- Cap: 1 hour of the researcher's time and 4 agent hours. If the gate is not passed by then, the delivery is "prototype blocked: reason", and the graph branch has no verdict.

### 3.8 Order of the measurement

`harness/visit_measure.py` runs the profile in two rounds. Round 1 runs the bare, timer and profile arms at 16, 32, 64 and 128 tokens, with the control arm after the 16-token arms. Round 2 runs the same list reversed. Each arm runs once per round, in a fresh process. The experiment runs after the profile. `visit_measure.py dry-run` prints this order.

## 4. What the profile reports

Each visit's host time splits into four parts that sum to it by construction: launch time, weight preparation, `functional_call` machinery, and Python with dispatcher. The GPU kernel time per visit is a separate number. The forward visit, the recompute visit and the layer backward are reported in ms, as medians over layers and timed steps. Visits, layer backwards, the work outside the layers and the final GPU wait must sum to the step time under the profiler within 5%.

## 5. Words

| Term | Meaning |
|---|---|
| visit | One pass of a streamed layer's body during a training step: wait for the layer's weights, substitute them, run the layer. Under activation checkpointing each layer has two visits per step. |
| forward visit | The visit a layer gets during the forward pass. |
| recompute visit | The second visit of a layer in a step. Checkpointing runs the body again during backward. |
| layer backward | The gradient computation over a layer's recomputed graph. It runs after the recompute visit and belongs to no visit. |
| profile | A measurement that splits time into parts without comparing two arms. Its arms are box stamped, but the repeats rule does not apply to it. |
| GPU power cap | A GPU power limit below the card's default while the machine is on mains. An arm that runs under it is void. |
| launch time | Host time spent in the calls that put GPU work in a queue: kernel launches, asynchronous copies, event records and waits. |
| weight preparation | The part of a visit that gets the layer's weights ready: waiting for the copy, starting the next layer's copy, building the substitution mapping. Launch time inside it counts as launch time. |
| stable visit | A visit of one layer with a fixed token count and fixed weight addresses, with copies and stream waits kept outside it. It is the unit a CUDA graph can capture. |
| capture cost | The wall time of the warm-up passes plus the capture of a CUDA graph, reported apart from the replay time. |
| payback | Capture cost divided by the time a replay saves per visit against eager. |
| inflation factor | A visit's host time under the profiler divided by its host time under the light timer. |
| CPU-bound visit | A visit whose capturable host time is at least twice the GPU kernel time it issues. |
| positive control | A delay of known size placed in a known part of the measured work. The instrument must find it, or its negative answer means nothing. |

The terms arm, round, run, block, pilot, box stamp, void arm, validity row, check outcome and no verdict are in `benchmarks/METHOD.md`.

## 6. Why "the launches dominate" is read as a CPU-bound visit

The card says to graph-capture the visit "if the launches dominate". The rule tests something else: the visit's capturable host time must be at least twice the GPU kernel time the visit issues (ADR 0005). A CUDA graph replay skips Python, the PyTorch dispatcher and the launch calls together, not the launch calls alone. In the graph demo on the RTX 5070 laptop, the launch calls were 5 of the 19 µs a replay saved. A test on launch time alone could miss a visit that a graph would speed up a lot, and a test at 50% launch share would almost never fire. PyTorch's own advice is to try graphs on a workload that is at least somewhat CPU-limited. The launch share is reported, and it decides nothing.

## 7. Prerequisites for a run

1. Reboot before the first arm, for an 80 W power limit and clean swap. Confirm 80.00 W in the full power query.
2. Plug in the charger. Close every other app and run from a plain terminal. Turn swap off with `sudo swapoff -a` and back on with `sudo swapon -a` afterwards.
3. Build the fixture on the CPU with `visit_profile.py fixture`, with shards on the NVMe disk. The system temp directory here is RAM-backed, so the arms run with `TMPDIR` set to the fixture root.

## 8. Numbers from other records

These figures are quoted by the card. They come from other cards, other code or other methods. No ratio or difference against this record's numbers is computed, because the conditions differ.

| Figure | Card and OS | torch | Date | Commit | Method |
|---|---|---|---|---|---|
| 0.65 ms per visit | RTX 5070 Laptop, Windows 11 Pro 26100 | 2.14.0+cu130 | measured 2026-09-15, published 2026-09-16, before the #841 cache of 2026-09-29 | PR #991 tree `3d2be4d0` | Synthetic Llama-70B shape, NF4, disk tier, seq 512 and 256. Twice the 256-token figure minus the 512-token figure, with dequantization replaced by a zero-weight `F.linear`. A two-point extrapolation to zero tokens, in `probe-rtx5070-what-bounds-streaming.md` finding 22 |
| ~10 ms per visit | RTX 3050 Laptop 4 GB, Windows 11 | 2.5.1+cu121 | not stated in the record (cited in #841, opened 2026-09-11) | Soup v0.73.0 `57c18e5` | Ablation residue: 0.626 to 0.716 s per step at 16 tokens on Llama-3.1-8B with copies and dequantization removed, over 64 visits, in `probe-v0.73.0-what-bounds-streaming.md` |
| 2.9x | RTX 5070 Laptop, Windows 11 Pro 26200 | 2.14.0+cu130 | run 2026-09-25 | criteria committed at `8b54b472` | Decode, resident Qwen2.5-1.5B FP16, no quantization, no backward pass. Slowest graph run 60.10 tok/s against fastest baseline 20.71 tok/s, in `gate-v0.76.0-cuda-graph-decode.md` |

## 9. Result

Measured 2026-10-15 on the RTX 3050 laptop. Recommendation: **No verdict**. The text above this section is unchanged except the status line.

### 9.1 Validity

Attempt 2: 26 profile arms and 20 experiment arms, all five required checks `ok` on every one. No arm is void. The first arm's box stamp is 2026-10-15T00:15:41+05:00; the rule commit `5c05c4fa` is 2026-10-14T23:55:26+05:00, so the rule came first (house rule 1).

### 9.2 Step time and instrument rows

Median step, seconds. Two values in a cell are round 1 / round 2.

| Tokens | Bare | Timer | Profile | Timer is free (limit +2%) | Profile is readable (limit 1.5) |
|---|---|---|---|---|---|
| 16 | 0.8432 / 0.8456 | 0.8427 / 0.8415 | 0.8486 / 0.8474 | pass | pass |
| 32 | 0.8477 / 0.8440 | 0.8445 / 0.8448 | 0.8501 / 0.8497 | pass | no verdict: rounds disagree, 1.07 and 1.67 |
| 64 | 0.9054 / 0.9067 | 0.9063 / 0.9091 | 0.9107 / 0.9126 | pass | pass |
| 128 | 1.0323 / 1.0341 | 1.0330 / 1.0320 | 1.0398 / 1.0386 | pass | pass |

### 9.3 Parts of a visit under the profiler

Median over visits (96 forward, 96 recompute, 93 layer backward per run), ms, round 1 / round 2. Host time is the sum of the four parts. The layer backward has no weight preparation or `functional_call`, so its parts are launch time and the rest. The inflation factor and shares for the forward visit use the timer arm's median forward visit as the base.

| Tokens | Visit | Host | Launch | Weight prep | `functional_call` | Python and dispatcher | GPU kernel | Launches |
|---|---|---|---|---|---|---|---|---|
| 16 | forward | 2.85 / 2.99 | 0.30 / 0.32 | 0.21 / 0.22 | 0.19 / 0.19 | 2.00 / 2.12 | 2.74 / 2.74 | 100 |
| 16 | recompute | 2.69 / 2.75 | 0.29 / 0.30 | 0.22 / 0.23 | 0.20 / 0.20 | 1.94 / 2.00 | 2.74 / 2.73 | 99 |
| 16 | layer backward | 7.66 / 7.58 | 4.67 / 4.51 | n/a | n/a | 3.02 / 3.08 (rest) | 4.73 / 4.73 | 121 |
| 32 | forward | 2.95 / 4.39 | 0.32 / 0.46 | 0.22 / 0.22 | 0.20 / 0.20 | 2.07 / 2.09 | 2.92 / 2.93 | 100 |
| 32 | recompute | 2.66 / 2.61 | 0.29 / 0.29 | 0.22 / 0.22 | 0.21 / 0.20 | 1.93 / 1.90 | 2.92 / 2.93 | 99 |
| 32 | layer backward | 7.65 / 7.68 | 4.70 / 4.80 | n/a | n/a | 2.98 / 2.89 (rest) | 4.93 / 4.93 | 121 |
| 64 | forward | 6.66 / 6.71 | 3.72 / 3.45 | 0.22 / 0.22 | 0.20 / 0.20 | 2.07 / 2.09 | 4.19 / 4.20 | 104 |
| 64 | recompute | 2.65 / 2.62 | 0.30 / 0.29 | 0.22 / 0.22 | 0.20 / 0.20 | 1.92 / 1.89 | 4.19 / 4.19 | 103 |
| 64 | layer backward | 9.58 / 9.61 | 6.68 / 6.76 | n/a | n/a | 2.91 / 2.86 (rest) | 7.73 / 7.73 | 123 |
| 128 | forward | 6.46 / 6.54 | 3.46 / 3.79 | 0.22 / 0.21 | 0.19 / 0.19 | 2.03 / 2.01 | 6.89 / 6.88 | 104 |
| 128 | recompute | 2.73 / 2.65 | 0.31 / 0.30 | 0.23 / 0.22 | 0.21 / 0.20 | 1.97 / 1.93 | 6.87 / 6.86 | 103 |
| 128 | layer backward | 13.24 / 13.31 | 9.32 / 9.45 | n/a | n/a | 3.92 / 3.86 (rest) | 8.73 / 8.73 | 119 |

Forward host time at 64 and 128 tokens is 6.5 to 6.7 ms against 2.8 to 3.0 ms at 16 and 32, and launch time rises from 0.3 ms to 3.4 to 3.7 ms with it. The forward visit's launch count grows only from 100 to 104. This is unexplained and the profile did not look further. The recompute visit does not show it.

Inflation factor of the forward visit, profile host time over timer host time, round 1 / round 2:

| Tokens | Inflation |
|---|---|
| 16 | 1.07 / 1.11 |
| 32 | 1.07 / 1.67 |
| 64 | 1.03 / 1.06 |
| 128 | 1.05 / 0.99 |

### 9.4 Reconciliation

Visits, layer backwards, the work outside the layers and the final GPU wait sum to the profiled step time within 5% at every token count. The gap is 0.6% to 1.1% of the step (largest at 32 tokens, round 2). The work outside the layers is 482 to 552 ms of a profiled step, 3 steps per run.

### 9.5 Control

Result: **fail**. At 16 tokens, the timer's host time per forward visit grew by 7160 µs in round 1 and 7027 µs in round 2. The rule expects 200 µs ± 20% plus the cost of the 50 launches (6799 µs in round 1, 6770 µs in round 2), so 6959 to 7039 µs in round 1. Round 1 is 121 µs over the top of its window. Weight preparation grew by 208 µs and 198 µs, and the launch count grew by exactly 50 in both rounds. The control row has no pass, so by the rule the CPU-bound threshold has no verdict. Nothing was rerun and no window moved.

### 9.6 Threshold

Result: **no verdict** (control row). The numbers the rule would have read, shown for the reader only and deciding nothing:

| Round | Timer host per forward visit | Weight prep | Capturable host | GPU kernel | Capturable over kernel (needs 2) | Launch share of host |
|---|---|---|---|---|---|---|
| 1 | 2.661 ms | 0.213 ms | 2.448 ms | 2.738 ms | 0.89 | 10.5% |
| 2 | 2.684 ms | 0.225 ms | 2.459 ms | 2.736 ms | 0.90 | 10.7% |

Both rounds sit near 0.9, a long way from 2. This does not clear the no verdict. It says a control rerun is unlikely to change the graph branch.

### 9.7 Experiment

Gate: passed. Loss and 256 LoRA gradients are bit-equal under `torch.equal`, the control (one changed input token) differs, and a repeat of the shipped arm is equal.

Blocks (A-B-B-A at 16 tokens, median step seconds), 5-block pilot, none void:

| Block | A | B | B | A |
|---|---|---|---|---|
| 1 | 0.8426 | 0.8428 | 0.8423 | 0.8428 |
| 2 | 0.8485 | 0.8413 | 0.8432 | 0.8441 |
| 3 | 0.8428 | 0.8423 | 0.8432 | 0.8437 |
| 4 | 0.8538 | 0.8411 | 0.8448 | 0.8427 |
| 5 | 0.8434 | 0.8411 | 0.8409 | 0.8412 |

Mean of A 0.8446 s. B is faster by 2.26 ms (0.27% of a step), 95% interval -0.59 to +5.11 ms, p = 0.093. The pilot did not call for more blocks. Verdict: **fail**, written as "effect below the detection limit, about 10% of a step". Block 2 and block 4 show A at 0.8485 and 0.8538 in the first position; those are kept and move the mean.

### 9.8 Prototype

None. The threshold has no verdict, so the prototype did not run (ticket 09 closed as `wontfix`, no code). The graph branch has no verdict.

### 9.9 Recommendation

**No verdict.**

- `tie_weights=False` upstream: fail. Effect below the detection limit, about 10% of a step.
- CUDA graphs in the runtime: no verdict. Row: control. The rerun that clears it is the control arm again at 16 tokens, in both rounds. The numbers in 9.6 suggest the visit is GPU-bound at 16 tokens (capturable host time about 0.9 times the kernel time), so a passing control would most likely give a threshold fail, not a pass.

### 9.10 Attempts, void arms and environment

- **Attempt 1** (`benchmarks/results/probe-rtx3050-short-step-visit-attempt1-void-swap/`). Swap was on. All 10 arms that finished (runs 1 to 10 of round 1) were void on `swap`: the first arm swapped out 234139 pages and swapped in 66855. The run was stopped during run 12. Its records and `run.log` are kept. The profile has no replacement rule, so attempt 2 is a new attempt under the same rule, with no threshold or code changed.
- **Attempt 2** (`benchmarks/results/probe-rtx3050-short-step-visit/`). Swap off with `sudo swapoff -a` for the whole run, as section 7 says. Attempt 1 reset swap with `swapoff -a && swapon -a` and left it on. That was the cause.
- Swap was on for the F1 clean-arm check, which passed (`ok`) before attempt 1.
- Environment: Linux 7.0.0-38-generic, RTX 3050 Laptop GPU 4 GB, driver 595.99.02, torch 2.14.1+cu130, bitsandbytes 0.50.2, transformers 5.19.0. Harness commit measured: `aa90c5b4`. Power limit 80.00 W, on mains.
- Withdrawn numbers: none.


### 9.11 Evidence scope

The saved arm and run JSON hold the derived per-visit numbers and the recommendation. They do not hold the original CUDA profiler traces: `run_arm_process` deletes them by default. A reader can replay the verdicts from the JSON, but cannot re-parse the measured launch and kernel attribution from this package. The control discrepancy in 9.5 therefore cannot be diagnosed from these records alone. No traces or source hashes were reconstructed after the run.
