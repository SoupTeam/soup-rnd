# A3 stage 2 - GPU check of activation spill in `training.l2l`

**Status: session 3 (protocol v3, §11), RTX 4090 VM, 2026-10-28: `SPILL CONFIRMED: equal,
within its RAM, cleaned up; overhead below 10% confirmed on a faster machine; L2L is outside
the A3 stand's GA tolerance (criterion 1b, not a spill finding)`. Criteria 1, 2, 3 and 4 PASS;
criterion 1b FAIL, reported apart, cause unestablished.** Session 2 (v2, §10) keeps its
`NO VERDICT: criterion 3 has no verdict`; session 1 (v1, §9) stopped at calibration with
`NO VERDICT: criterion 1, 2, 3 has no verdict`. Neither historical result is re-judged.

v1's criteria were fixed by commit `2a05e15f` before any rental, the machine admission (§5)
by `794c6827`; v2 (§3.6) by `6bdd0ec8` before session 2; v3 (§3.7) by
`bc2ca55038a2d85950356cfdd38d556333519ad1` before session 3, measured at
`ec795ef175c222b60ff91b9ba44f7f2bfad55dd5`. v2 added the product's chosen 512 MiB reserve,
calibrated the tight limit to it, and fails criterion 2 on a kernel OOM kill of any child at
the tight limit after its first training step. v3 restores Python's SIGINT handler in signal
children and signals them in a hold inside optimizer step 2. The rule is
`harness/a3_spill_rule.py` (`RULE`, profile `a3-spill-gpu-check-v3`, `rule_version` v3;
v1 at `bda4a00a`, v2 at `667c4a1b`), copied into every `protocol.json`. The stand is
`harness/a3_spill_stand.py`, the admission `harness/a3_admission.py`, the signal preflight
`harness/a3_signal_preflight.py`. CPU tests: `tests/test_a3_spill_rule.py`,
`tests/test_a3_spill_stand.py`, `tests/test_a3_admission.py`.

## 1. Question

Does the automatic spill of A3 stage 1 work on a GPU, through the real HF trainer path
(`make_l2l_trainer_class(transformers.Trainer)` + `attach_l2l_state`), when the product's
own RAM check under a real cgroup limit, and nothing else, turns it on?

- **Equal**: spill gives the bits of no spill.
- **Within its RAM**: no OOM, and no host memory outside the planned store.
- **Clean**: no spill file left after a normal end or Ctrl-C. A SIGKILL leaves the file,
  and the next start in the same directory removes it.
- **Cost**: step-time overhead of spill against no spill, judged against 10%.

Fixture: Mistral-7B NF4 (`unsloth/mistral-7b-instruct-v0.3` at the A4 revision
`60b99dcbc7f853b7cd18472a607e8d045b96f03e`; 32 layers, hidden 4096), RAM tier, LoRA r 8 on
q/k/v/o, k = 4, seq 512, built by `stream_probe.build`. The A3 stand
(`harness/l2l_mistral_stand.py`) supplies the deterministic setup, the rows, the run
comparison, the GA tolerance and the bnb #2034 probe. It is imported, not changed.

## 2. Arms

Each arm is a fresh process (`a3_spill_stand.py child`), each in its **own** transient
scope `systemd-run --scope -p MemoryMax=<limit> -p MemorySwapMax=0`. Each child's
`memory.peak` and `memory.events` are then its own, and criterion 2 reads them per arm.
The parent (`session`) holds no model and runs outside any limit. Two limits:
**generous** (`--generous-limit`, default 24G; nothing may spill there) and **tight** (from
the calibration, §4).

| Order | Child | Limit | What it does |
|---|---|---|---|
| 1 | `calibration` | generous | builds, trains one window; the product plans the store; working set read (§4); bnb #2034 and SDPA probes after training |
| 2 | `calibration_check_1` (`_2` only if needed) | tight | the product's own plan at the computed limit; outside [12, 20] spilled layers the limit is recomputed once from its numbers |
| 3 | `l2l_nospill_1`, `l2l_nospill_2` | generous | L2L, nothing spilled (the reference, run twice) |
| 4 | `l2l_spill` | tight | L2L, spill_layers > 0 |
| 5 | `ga_1`, `ga_2`, `ga_floor` | generous | plain Trainer, gradient accumulation; `ga_floor` on EFFICIENT_ATTENTION |
| 6 | `compare` | none (CPU) | per-step gradient tensors: hash check, max \|d\|, A3 stand `decide` |
| 7 | `sigint` | tight | ONE SIGINT while it holds inside optimizer step 2 (v3, §3.7) |
| 8 | `kill9`, then `kill9_next` | tight | SIGKILL while it holds inside step 2; a new child in the same spill directory |
| 9 | cost blocks `A B B A` | A generous, B tight | only if criterion 1 PASSES (§3.4) |

Correctness and cleanup rows: `make_rows(10, 512, vocab)` of the A3 stand, with mixed
lengths and masked labels (-100). k = 4 gives windows of 4, 4 and 2 rows, i.e. 3
optimizer steps with a short last window. SGD, lr 1e-3 linear with 1 warm-up step,
`max_grad_norm` 0, seed 1, `make_non_vacuous` LoRA B. Deterministic mode: the A3 stand's
`set_deterministic_env()` first (`CUBLAS_WORKSPACE_CONFIG=:4096:8` before torch/CUDA),
strict `torch.use_deterministic_algorithms(True)` (no warn_only), SDPA pinned to MATH.

Every child writes a marker (`markers/<child>.json`) at its first `training_step` entry,
and the parent copies it into `session.json` after the exit. A child killed without one
died in the build (§3.1, criterion 2). The signal children rewrite it at every
`training_step` entry and exit. v1 and v2: the parent waited for the marker that says one
optimizer step is done (step 2 has started), waited 0.5 s, listed the spill directory, and
sent one signal to the PID in the marker. v3 (§3.7): the child holds inside optimizer step 2
and writes a `hold` marker; the parent lists the spill directory, reads the target's SIGINT
masks from `/proc`, and sends the one signal then. After the exit it lists the directory
again.

Cost rows: 4 x (2 warm-up + 5 timed) rows, all 512 tokens, no masked label. The value of
an arm is the median of its 5 timed optimizer steps. Each cost arm runs under an F1
`ArmWatch` (METHOD.md) with `suspend`, `power` and `foreign_reads` required, watching the
spill drive.

Every child writes JSON: arm, argv, versions, GPU facts (driver, PCIe gen and width), the
scope readback at start, after the build, at the first `training_step` and at the end;
the product's `spill_plan` and `stale_removed`; the log history, lr and LoRA gradient
SHA-256/dtype/shape per optimizer step; the store's bytes written and read; the CUDA host
allocator at train begin and at each step end; the spill files before and after `train()`;
and the step times. The gradient tensors of the correctness arms go to `<out>.tensors/`
(one `.pt` per step), which is **not published**. Only `compare` reads them.

## 3. Criteria, written before the run

### 3.1 Table

| # | Criterion | PASS | FAIL | NO VERDICT |
|---|---|---|---|---|
| 1 | spill == no spill | `l2l_spill` equals `l2l_nospill_1` bit for bit: SHA-256 of the raw bytes, dtype and shape of every LoRA gradient at every optimizer step, the logged losses, the number of optimizer steps, the lr at every step; and `l2l_nospill_1` equals `l2l_nospill_2` the same way | the spill arm differs in any of these | the two no-spill runs differ (the reference does not repeat); or any V-check below fails on one of the three arms |
| 1b | L2L vs GA (reported) | the A3 stand's `decide(ga_1, ga_2, l2l_spill, floor=ga_floor)` in bf16: within 1.5 x the floor at every step | `l2l_out_of_tolerance` or `l2l_differs` | any other status, the bnb #2034 probe not passing, a tensor file not hashing to its record, a V-check. **Does not change the spill verdict.** |
| 2 | RAM | for `l2l_spill`, `l2l_nospill_1`, `l2l_nospill_2`: child exit 0, `memory.events oom_kill` not raised, scope `memory.peak` <= its MemoryMax; and excess(`l2l_spill`) <= excess(`l2l_nospill_1`), where excess = host allocator `allocated_bytes.current` at the last optimizer step - at train begin - the plan's `host_bytes` | a non-zero exit, an `oom_kill` increment, a peak above the limit, or more excess with spill. A kill or OOM is a FAIL wherever it happens; the detail says `before first training_step (build)` or `after first training_step` (from the child's first-step marker, or its own readback at that entry). After it means the product's 0.3 headroom did not cover training at this remainder: a product finding | a counter not read, a V-check |
| 3 | cleanup | all three: **normal** (no spill file after `train()` of `l2l_spill` nor after its process exit), **sigint** (the child exits, any code; no spill file after), **kill9** (the file is still there after SIGKILL; `kill9_next` lists it in `stale_removed`, it is gone, and `kill9_next` trains its 3 optimizer steps normally) | a file left after a normal end or after Ctrl-C; a child that does not exit after one SIGINT; the kill9 file gone after the kill; not removed at the next start; the next child not exiting 0 or not training normally | the scenario not reached (no spill file before the signal, the signal not inside a step, the child gone first), a V-check |
| 4 | cost | §3.2 upper bound < 10% | §3.2 lower bound >= 10% | §3.2 otherwise |

### 3.2 The cost rule

A-B-B-A blocks of fresh processes. A = L2L at the generous limit (no spill), B = L2L at
the tight limit (spill). Per block, overhead = (mean B - mean A) / mean A. Over the valid
blocks: the mean and a two-sided 95% Student t interval (`a3_spill_rule.interval`, the
`t975` table of `a4_rule`). Blocks: 5-block pilot, then `arm_validity.blocks_to_add`
until it returns 0, at most 12. A block with a void run is replaced and not counted
(METHOD.md).

- **Upper bound < 10%: PASS.** The docs may say the overhead is below 10%, confirmed on a
  faster machine (this box, named with its GPU).
- **Lower bound >= 10%: FAIL.** The fallback stays automatic. The console message and the
  docs state the measured overhead.
- **Otherwise: NO VERDICT**, recorded as such. This covers an interval that straddles
  10%, and the repeats rule still wanting blocks at the 12-block cap (V7).

### 3.3 V-checks

| V | Check | Effect when it fails |
|---|---|---|
| V1 | device cuda in every child record (calibration, checks, arms, cleanup, cost); the killed `kill9` child through its marker | a CPU session never gets a verdict: `NO VERDICT: CPU session (V1)`; a CPU `kill9` marker: kill9 NO VERDICT |
| V2 | correctness arms: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms on, warn_only off, SDPA MATH (`ga_floor`: EFFICIENT_ATTENTION) | NO VERDICT for 1 / 1b |
| V3 | the child's own cgroup v2 scope reads back `memory.max` = the limit `session.json` planned for it, and `memory.swap.max` = 0 | NO VERDICT for every criterion that reads that child |
| V4 | spill arms spilled (`spill_layers` > 0, store bytes written > 0 and read > 0); no-spill arms planned `spill_layers` 0 | the arm is invalid: NO VERDICT |
| V5 | compared arms trained on the same rows (SHA-256 per row of ids and labels) | NO VERDICT |
| V6 | cost arms: F1 outcome `ok` (`unknown` on a counted block removes criterion 4) | NO VERDICT for 4 |
| V7 | the repeats rule has no block left to add | NO VERDICT for 4 |
| V8 | trained arms: normal exit, finite losses and gradients, a nonzero LoRA gradient at every step, the expected optimizer steps | NO VERDICT (FAIL for `kill9_next` in criterion 3) |

### 3.4 Stop conditions

- **Criterion 1 FAILS: the parent stops before the cost blocks.** It prints `STOP:
  criterion 1 failed (spill != no spill) ... No new rentals: find the cause.`, writes
  `state: stopped_before_cost` and exits 4. No new rentals until the cause is found.
- Criterion 1 NO VERDICT: no cost block either (they run only after a PASS); exit 4.
- Calibration: the generous limit already spills (raise `--generous-limit`), a check
  child fails, or after 2 checks the product spills nothing: the session stops (exit 3)
  before any correctness arm.
- **Calibration OOM risk** (§4): before any child runs at a tight limit, if its **build
  margin is below 0** (or unknown), the session stops with exit 3 and `state:
  stopped_calibration_oom_risk`, printing the numbers. The limit is never raised to make
  room: that would move the spill off its target, so the owner decides. A negative
  training margin, and either margin below 256 MiB, only warn.
- A cost child exits non-zero, or a source file hashed in `protocol.json` changes during
  the session: stop, exit 3.
- `--max-hours` (default 5) stops child work. It does NOT stop or destroy the rented
  server.

### 3.5 Verdict

`verdict.json` (LF) is written by the session and is recomputed from the raw JSON alone by
`a3_spill_rule.py <out> --out verdict.json`:

- `SPILL CHANGES THE NUMBERS: no new rentals, find the cause` (criterion 1 FAIL);
- `SPILL NOT SAFE IN THE TRAINER: criterion <2/3> failed`;
- `NO VERDICT: criterion <n> has no verdict`;
- `SPILL CONFIRMED: equal, within its RAM, cleaned up; overhead below 10% confirmed on a
  faster machine` (1, 2, 3 and 4 PASS);
- `SPILL CONFIRMED, COSTLY: ...` (4 FAIL);
- `SPILL CONFIRMED, COST NO VERDICT: ...`.

A 1b FAIL is appended as a note and changes none of these. The verdict also reports
`spill_layers` of `l2l_spill` and whether it is in the target band [12, 20] (description,
not a criterion).

### 3.6 Protocol v2 (fixed after session 1, before session 2)

Session 1 (§9) stopped when the check child at the tight limit was OOM-killed by its cgroup
inside its first training step. v2 changes the product and the protocol because of it. Session
1 is not re-judged: its `verdict.json` is rule v1's output and stays NO VERDICT.

| | v1 | v2 | why |
|---|---|---|---|
| product budget | 0.7 x available | min(0.7 x available, available - 512 MiB) (`l2l_spill.budget_of`, `RAM_RESERVE_BYTES`); below 576 MiB available Mistral's run stops with the numbers | in session 1 the 0.3 headroom was about 130 MB at the plan point and the first step grew about 183 MB outside the store after it. 512 MiB is chosen, not measured |
| tight limit (§4) | working set + (target host_bytes + half a layer) / 0.7 = working set + 469 MiB | working set + `available_for`(target host_bytes + half a layer) = working set + max(328 MiB / 0.7, 328 MiB + 512 MiB) = **working set + 840 MiB** | the product's own budget, not the stand's guess, decides the spilled layers; target still 16 of 32, band still [12, 20] |
| drift the band holds | exactly 16 at -10..+11 MiB, [12, 20] at -102..+103 MiB | exactly 16 at -7..+8 MiB, [12, 20] at -71..+72 MiB | below 1.79 GB a MiB of working set moves the budget by a MiB, not 0.7 of one. Session 1's check child sat about 37-61 MB above the calibration's working set |
| criterion 2 | only `l2l_spill`, `l2l_nospill_1`, `l2l_nospill_2` | also every other child at the tight limit (calibration checks, `sigint`, `kill9`, `kill9_next`, cost B): a kernel memory-cgroup OOM kill of its pid after its first-step marker is a FAIL; before the marker (the build) no verdict; a kill the kernel log cannot tell about, no verdict | §4 already called an OOM in training a criterion 2 FAIL; v1's rule never read the check children, so session 1 came out NO VERDICT |
| OOM evidence | return code and `memory.events` | the parent reads `journalctl -k` for the child's pids over its run after a non-zero exit (again after 3 s if the line is not there yet) and keeps the `Killed process <pid>` and `CONSTRAINT_MEMCG` lines in `session.json` as `kernel_oom`. A return code of -9 alone is not an OOM, and the SIGKILL the parent sends `kill9` is not one | `--collect` removes the scope with the child, so its `memory.events` cannot be read after an OOM kill |
| stop | a failed check child stops (exit 3, `stopped`) | a kernel OOM kill of a child at the tight limit after its first step stops the session (exit 3, `stopped_oom_in_training`), and the verdict is `SPILL NOT SAFE IN THE TRAINER: criterion 2 failed` | |
| rule version | `a3-spill-gpu-check-v1`, v1 | `a3-spill-gpu-check-v2`, v2 | |

Unchanged: criteria 1, 1b, 3 and 4 and their thresholds (10%, the repeats rule, 12 blocks),
V1-V8, the arms and their order, the admission (§5, `a3_admission.py`, whose demand comes from
the A child at the generous limit and does not depend on the budget), `--max-hours 5`, the
24 GiB generous limit, swap 0, the calibration's OOM margins (build below 0 stops; training
describes), and that the limit is never raised. The training margin at v2's limit is (W - U0) +
840 - 320 MiB - training transient: the 512 MiB reserve plus the reclaimable active cache.

Any outcome of the v2 session (PASS, FAIL or NO VERDICT, cost included) ends the GPU work on
A3. No other machine is taken without the owner's decision.

### 3.7 Protocol v3 (fixed after session 2, before session 3)

Session 2 (§10) ran every scenario, but its `sigint` child never saw its SIGINT: the child
exited `normal` after 3 optimizer steps, and the rule returned NO VERDICT for criterion 3. The
cause, reproduced on Linux after the session: the session was started as `setsid nohup bash -c
'...' &` from a non-interactive shell. bash starts such a background job with SIGINT ignored,
exec keeps an ignored signal ignored, neither `subprocess.Popen` nor `systemd-run --scope`
resets it, and Python leaves an inherited SIG_IGN in place, so no KeyboardInterrupt was ever
raised. Session 2 is not re-judged: its `verdict.json` is rule v2's output and stays NO
VERDICT.

| | v2 | v3 | why |
|---|---|---|---|
| SIGINT in the children | inherited from the launch | every child installs Python's own handler (`signal.default_int_handler`) first, and records what it inherited (handler and `/proc` SigIgn/SigBlk/SigCgt) as `sigint_handling` | session 2's children inherited SIG_IGN |
| when the signal goes | 0.5 s after the step-2 entry marker | the signal child holds inside optimizer step 2 at the backward recompute of its highest spilled layer (first micro-batch; a forward pre-hook on the decoder layers, grad on) and writes a `hold` marker; the parent sends the one signal on reading it. The hold ends on the interrupt or after 120 s | the entry marker said nothing about where the step was |
| at the send | the spill directory | also the target's SIGINT pending, blocked, ignored and caught bits from `/proc/<pid>/status` (`target_at_send`) | tells an ignored or blocked signal from a missed phase |
| sigint PASS | exit `interrupted`, `in_step` true, file before, none after | also the marker at the send is the hold, 1 optimizer step done, and the hold was released by the interrupt (`at_interrupt.hold.released` = `interrupt`, backward pass) | a KeyboardInterrupt in the hold, not merely a step |
| sigint NO VERDICT | "scenario not reached" | the detail names why: SIGINT ignored (SigIgn) or blocked (SigBlk) at the send, the hold ended by its timeout, or the interrupt outside the hold | |
| kill9 | SIGKILL 0.5 s after the step-2 entry | SIGKILL at the same hold; a marker that is not the hold gives no verdict | |
| before the model runs | — | on the session's box, launched exactly as the session is, `a3_signal_preflight.py` runs the stand's `sigint`, `kill9` and `kill9_next` children on a tiny 8-layer Llama with spill, each in its own scope; any failed check stops the session before the model is downloaded | the launch, not the stand alone, broke session 2 |
| rule version | `a3-spill-gpu-check-v2`, v2 | `a3-spill-gpu-check-v3`, v3 | |

Unchanged: criteria 1, 1b, 2 and 4 and every threshold (10%, 1.5 x the floor, the repeats rule,
12 blocks), the 512 MiB reserve, the calibration of the tight limit and its OOM margins,
V1-V8, the arms and their order, the admission, `--max-hours 5`, the 24 GiB generous limit and
swap 0. The cost children do not hold and never get a signal. Criterion 3's FAIL conditions
are the same: a file left after a normal end or after Ctrl-C, a child that does not exit
after one SIGINT, the kill9 file gone after the kill, not removed at the next start, or the
next child not training normally.

Checked before the rental, on CPU (WSL 2, Ubuntu, cgroup v2, systemd 259; torch 2.14.1,
transformers 5.19.0, peft 0.21.2):

- a Python child started with SIGINT ignored and sent a real SIGINT keeps running without
  `restore_sigint` and raises KeyboardInterrupt with it (`tests/test_a3_spill_stand.py`);
- the three cleanup children of the tiny model, through the stand's own `run_child` and
  `drive_signal`, started by a launcher whose SIGINT is ignored, pass every preflight check;
  with the handler restore disabled the same test fails the sigint checks (exit `normal`,
  `target_at_send.ignored` true) while kill9 and kill9_next still pass;
- `a3_signal_preflight.py` launched as session 2 was (`setsid nohup bash -c '...' &` from a
  non-interactive root shell), each child in its own `systemd-run --scope` at 8 GiB with swap
  0: every check passes; the launcher's SIGINT read back ignored, the children's installed
  handler caught it. The same launch with the handler restore disabled fails the three sigint
  checks (exit `normal`, return code 0, `target_at_send.ignored` true, the hold released by its
  timeout) while kill9, kill9_next and both scope checks pass.

Not checked before the rental: the Mistral-7B NF4 children on CUDA.

A v3 session is a new full session in a new directory. Its criteria are judged from its own
records only; no result of session 2 is carried into it.

## 4. Calibration of the tight limit

*This section is v1's text, kept as session 1 ran it. Protocol v2 (§3.6) replaces the 0.7
share with the product's `budget_of` and the limit with working set + 840 MiB for the same
16 of 32 layers; the margins below are computed the same way at that limit.*

The product's budget is min(psutil available, min over the cgroup chain of limit - (used -
inactive_file)) x 0.7, and it plans the store at the first `training_step`. The
`calibration` child at the generous limit reads the working set at that moment: the
product's own `limiting_numbers` (used - inactive_file) when its limiting level is the
child's scope, else the stand's readback of `memory.current - inactive_file` at the first
`training_step` entry. Then:

limit = working set + (target host_bytes + half a layer) / 0.7, rounded up to a MiB,

with target host_bytes the product's `plan_spill` cost at 16 of 32 layers spilled. For
Mistral at k 4, seq 512, bf16, pinned: a chunk is 4 MiB (a 4 MiB pinned block), a layer 16
MiB, all resident 512 MiB, the two rings 2 x 4 x 8 MiB = 64 MiB (each 4 MiB + 4 KiB slot
holds an 8 MiB pinned block). Target 16 x 16 + 64 = 320 MiB, plus 8 MiB, / 0.7: **limit =
working set + 469 MiB**. At that limit the product's planner spills exactly 16 layers if
the working set moves by -10 to +11 MiB between children, and 12 to 20 layers from -102 to
+103 MiB (`plan_spill` itself; `tests/test_a3_spill_stand.py` pins these four edges).
`calibration_check_1` runs at the limit and
reports the product's own plan. If that is outside [12, 20], the limit is recomputed once
from that child's own plan-time numbers and checked again. After the second check any
spill > 0 is kept and reported as out of band. Nothing in the product is told the target.

**OOM margin (fixed before the run).** The stand sets the tight limit, not the product, and
puts it close to the steady state. The product plans the store only after the build. So an
OOM during the build would come from the stand's setup, not from spill, and must not be
scored. The calibration child (generous limit) samples its scope every 50 ms. It also
reads named samples at build end, at the first `training_step` entry and at each step end:
`memory.current`, `memory.peak`, `memory.stat` (`anon`, `shmem`, `active_file`,
`inactive_file`, `slab_reclaimable`, `file_dirty`, `file_writeback`), VmRSS/VmHWM and
torch's pinned host allocator. What reclaim cannot free, with swap 0, is

U = memory.current - (active_file + inactive_file - file_dirty - file_writeback) -
slab_reclaimable.

memcg reclaims clean file cache, active list included, before it OOM-kills. Dirty and
writeback pages cannot be dropped at once, so they count as unreclaimable. U0 is U at the
first `training_step` entry. The build transient is the largest U before that entry minus
U0. The training transient is the largest U after it, less the store's pinned growth
(torch's pinned host allocator since the entry, at most the plan's `host_bytes`), minus U0.
Floored at 0, so pinned bytes beyond the plan count. For each tight limit L, before any
child runs at it:

- **build margin = L - U0 - build transient. Below 0, or unknown: stop** (§3.4).
- training margin = L - U0 - training transient - target host_bytes: **description only**.
  The product promises its own 0.3 headroom. An OOM in training at L is a criterion 2 FAIL
  `after first training_step`, a product finding.
- Either margin below **256 MiB**: a warning, printed and recorded under
  `limits.oom_margin` in `session.json`, `oom_margin` in `calibration.json` and in
  `verdict.json`. It decides nothing. The literal limit - (working set + transient +
  target host_bytes) is recorded beside it as `margin_over_working_bytes`, also
  description only.

With W the product's working set, the training margin is (W - U0) + 469 - 320 MiB -
training transient: the product's 0.3 headroom of 149 MiB plus the reclaimable active
cache. The build margin is (W - U0) + 469 MiB - build transient. A4's v3 RAM-tier records
of this model show peak RSS 255 MiB above RSS after the build. That is a process figure,
not U. The calibration measures U on the box.

## 5. Machine

- **GPU**: an RTX 3090 / 3090 Ti VM with PCIe 4.0 is preferred. An RTX 4090 VM only if
  no 3090-class VM with PCIe 4.0 passes the admission (below). A VM, because a container
  has no `systemd-run` (KVM, systemd as PID 1). Gen and width are recorded per child (A4's
  v3 3090 VM ran at gen 3 x16).
- **Linux, cgroup v2** with the memory controller, `memory.peak` (kernel >= 5.19) and swap
  accounting (`MemorySwapMax`). The children run as root (`systemd-run --scope`), or
  with `--user-scope` in a user session with memory delegation.
- **Host RAM**: MemAvailable >= the generous limit (24 GiB); the session refuses less.
- **Spill drive**: a local NVMe directory, not tmpfs, not the container overlay, not a
  network volume (the product refuses those). One subdirectory per child group.
- **Disk**: ~15 GB source, ~4.5 GB NF4 cache, <= 256 MiB spill file (16 layers x 4 x 4
  MiB, also the admission's file), and an estimated ~0.5 GB of unpublished gradient
  tensors (6 arms x 3 steps of LoRA gradients). 40 GB free is ample.
- **Admission** (`harness/a3_admission.py`; copied from A4's `a4_admission.py` and
  `a4_box.py` at `d6f7f733`, nothing of A4 imported or changed). Does the spill volume
  keep up with this session's spill on this box? **Step 0 on any machine** measures the
  two inputs outside the session, after the NF4 cache exists: the calibration child (its
  `spill_plan`: n_layers, k, slot_bytes) and one no-spill cost A child (2 warm-up + 5
  timed steps; its `value` is the median timed step). Then:
  - cycle = one spill file of the session's step: spill_layers (16, the stand's
    `TARGET_SPILL`) x k chunks of slot_bytes, written in the forward's order and read back
    in the backward's (layers descending), through the store's own `DirectSpillFile`;
  - D = cycle_bytes / A.value (spill bytes written per step / the no-spill step);
  - V = cycle_bytes x (7 steps x 2 B arms x 12 blocks + 3 steps x 4 arms), the most the
    session's spill arms write: the cost B arms at 2 warm-up + 5 timed steps, at most 12
    blocks, and `l2l_spill`, `sigint`, `kill9`, `kill9_next` at the correctness rows' 3
    optimizer steps each, so V = 180 cycles;
  - period = cycle_bytes / (2 x D); N = max(20, ceil(V / cycle_bytes)); duration = N x
    period. No fixed duration: exactly N cycles are scheduled, cycle i starting at max(t0
    + i x period, the end of cycle i - 1).

  Admitted iff all N cycles ran, the largest lateness (cycle end minus t0 + (i + 1) x
  period) is <= 5 s, every chunk read back carries its cycle's per-sector stamp and its
  bytes, and no file is left. For Mistral (k 4, slot 4 MiB): cycle 256 MiB, N 180,
  period = A.value / 2, so the admission writes and reads the session's V in half its
  steps' time. The record (`admission.json`) carries D, V, N, the period, the cycle bytes
  and their inputs: the A record's value and SHA-256, the plan's numbers and the
  calibration file's SHA-256. The decision reads only those numbers. Description only:
  the per-second log, 10 s windows against their median, and the file's zero fill at
  creation. Exit 0 admitted, 4 refused (take the next box), 3 the volume itself refused
  (tmpfs, overlay, network, no direct I/O), 2 refused inputs. **The rule reads nothing of
  the admission.** It only decides whether to keep the box.

## 6. Runbook

```bash
set -e
mkdir -p /root/a3
git clone https://github.com/SoupTeam/soup-rnd /root/a3/soup-rnd
cd /root/a3/soup-rnd
git checkout ec795ef175c222b60ff91b9ba44f7f2bfad55dd5  # measured session-3 code; v3 fixed at bc2ca550
A0_PYTHON=python3.12 bash benchmarks/a0-bootstrap.sh   # venv, CUDA torch, pinned deps
export PYTHONPATH="$PWD/src:$PWD/benchmarks/harness" HF_HUB_OFFLINE=0
# CPU-fixture resume checks run with CUDA hidden, as declared before session 2.
RESUME='test_resume_with_spill_equals_resume_without or test_resume_does_not_shift_the_windows'
.venv/bin/python -m pytest tests/test_a3_spill_rule.py tests/test_a3_spill_stand.py \
  tests/test_a3_admission.py tests/test_l2l_spill.py tests/test_l2l_spill_budget.py \
  tests/test_l2l_spill_trainer.py tests/test_l2l_trainer_window.py \
  -q -o addopts='' -k "not ($RESUME)"
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest tests/test_l2l_spill_trainer.py \
  tests/test_l2l_trainer_window.py -q -o addopts='' -k "$RESUME"
SPILL=/root/a3/spill            # session 3 used the VM's local ext4 /dev/vda1
# 0a. Before the model download, run the tiny CPU signal preflight through the session's
#     background launch. Wait for its exit: a failed check stops this runbook (set -e).
#     The work directory must be new; its preflight.json is the retained report.
setsid --wait nohup bash -c 'exec .venv/bin/python benchmarks/harness/a3_signal_preflight.py run \
  --work /root/a3/sigpre-reproduction' > /root/a3/sigpre-reproduction.out 2>&1 < /dev/null &
sigpre_pid=$!
wait "$sigpre_pid"
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("unsloth/mistral-7b-instruct-v0.3",
                  revision="60b99dcbc7f853b7cd18472a607e8d045b96f03e",
                  local_dir=".soup/models/mistral-7b", token=False,
                  allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"])
EOF
export HF_HUB_OFFLINE=1
# 0. On ANY machine, outside the session (not evidence): the NF4 cache and the calibration
#    child's plan (writes .soup/f1-nf4), then one no-spill cost A child (2 warm-up + 5
#    timed steps) for the demand D. Both are the admission's inputs.
.venv/bin/python benchmarks/harness/a3_spill_stand.py child --arm calibration \
  --weights .soup/models/mistral-7b --shards .soup/f1-nf4 --spill-dir "$SPILL/prep" \
  --out .soup/a3-prep.json
.venv/bin/python benchmarks/harness/a3_spill_stand.py child --arm A \
  --weights .soup/models/mistral-7b --shards .soup/f1-nf4 --spill-dir "$SPILL/prep" \
  --out .soup/a3-demand-A.json
# 1. Admission: exactly N cycles (180 for Mistral) of N x A.value / 2 s in all.
#    Exit 4 = refused: delete the box, take the next one; 3 = the volume itself refused.
.venv/bin/python benchmarks/harness/a3_admission.py --spill-dir "$SPILL/admission" \
  --plan .soup/a3-prep.json --a .soup/a3-demand-A.json \
  --out .soup/a3-admission.json --cycles .soup/a3-admission-cycles.csv \
  --log .soup/a3-admission-per-second.csv
# 2. Plan: prints every child command in order with its scope, writes nothing.
.venv/bin/python benchmarks/harness/a3_spill_stand.py session \
  --weights .soup/models/mistral-7b --shards .soup/f1-nf4 --spill-dir "$SPILL" \
  --out .soup/results/a3-spill-03
# 3. Run through the same background launch as the preflight (as root); wait for its exit.
#    Every child gets its own scope; --wait also propagates a detached setsid child's status.
setsid --wait nohup bash -c 'exec .venv/bin/python benchmarks/harness/a3_spill_stand.py session \
  --weights .soup/models/mistral-7b --shards .soup/f1-nf4 --spill-dir /root/a3/spill \
  --out .soup/results/a3-spill-03 --execute' > /root/a3/session-reproduction.out 2>&1 < /dev/null &
session_pid=$!
wait "$session_pid"
# 4. Verdict, recomputed from the raw JSON alone:
.venv/bin/python benchmarks/harness/a3_spill_rule.py .soup/results/a3-spill-03 \
  --out verdict.json
```

Step 2 prints the complete commands with their actual paths and scope names; it is the
runbook's scope plan, not a template to fill in. Every child has its own transient scope,
with `MemoryMax` equal to its recorded limit and `MemorySwapMax=0`. The generous limit is
25,769,803,776 B (24 GiB); the tight limit is calibrated and recorded under `limits` in
`session.json`. Exit codes: 0 finished, 4 stopped before the cost blocks (§3.4), 3 stopped
otherwise, 2 refused arguments (an existing `--out` or sibling tensor directory, k or seq
other than 4/512, no `systemd-run`, MemAvailable below the generous limit). After the run,
copy the records and the sibling tensor directory privately, then destroy the instance
yourself. The session never does.

## 7. Files that come back

`.soup/results/a3-spill-<NN>/`: `protocol.json` (written first: rule, args, plan, git HEAD,
versions, SHA-256 of every `src/soup_cli` and harness `.py`), `session.json` (each child's
command, limit, return code, spill directory after exit; the signals sent), `verdict.json`,
`calibration.json`, `calibration_check_<n>.json`, `arms/<arm>.json`, `compare.json`,
`cleanup/<arm>.json` (none for `kill9`: it is killed; its marker is in `session.json`),
`cost/b<NN>_p<P>_<A|B>.json`, `markers/`, `logs/`. Beside it: the admission's inputs
`.soup/a3-prep.json` and `.soup/a3-demand-A.json`, and its `.soup/a3-admission.json`,
`.soup/a3-admission-cycles.csv` and `.soup/a3-admission-per-second.csv`. **Not published, but
kept privately with the raw records** (for recomputing `compare.json` later):
`.soup/results/a3-spill-<NN>.tensors/`.

## 8. What it does not show

- Model quality: a few steps on random token rows. Criterion 1 is bitwise, so it is the
  whole quality argument for spill.
- Other shapes, models, k, seq, batch sizes or dtypes; fp32; the disk tier (RAM tier
  only); multi-GPU.
- The psutil-limited or cgroup v1 budget: here the child's own cgroup v2 scope always
  limits. Those paths are CPU-tested in the product's tests only.
- A full drive (ENOSPC), a network or slow spill volume, two runs sharing a spill
  directory, resume from a checkpoint with spill: not exercised here.
- Cleanup after SIGKILL is best effort by design. The file stays until the next run in the
  **same** directory starts. A PID reused by a live process keeps the file there (the sweep
  never removes a live process's file).
- The cost on any other GPU, drive or step length; one box, one session. A PASS speaks for
  this box's class, not for slower drives.

## 9. Session 1: RTX 4090 VM, 2026-10-27. Stopped at calibration, NO VERDICT

Records: [`results/a3-spill-rtx4090-20261027/`](results/a3-spill-rtx4090-20261027/README.md).
One vast.ai KVM VM (RTX 4090 24 GB, PCIe gen 4 x16, 49 GiB RAM, one virtio disk), code at
`bda4a00a`. No PCIe 4.0 RTX 3090 / 3090 Ti VM passed the filters at rental time, so the 4090
was taken as §5 allows. The box ran 0.57 h of GPU time and cost $0.27; it is destroyed.

- **Admission: ADMITTED.** A.value 1.571 s, 180 of 180 cycles, largest lateness 0.312 s, 0
  chunks wrong, no file left.
- **Calibration** (24 GiB): exit 0. Working set 5,473,333,248 B; tight limit 5,965,348,864
  B. Build margin +1,001,061,608 B, so no stop; training margin -15,995,576 B, a warning
  only, as §3.4 fixed it.
- **`calibration_check_1`** at the tight limit: the product planned 18 of 32 layers (in the
  band), then the child's cgroup OOM-killed it inside its first `training_step` (kernel:
  `CONSTRAINT_MEMCG`, usage = limit, swap 0). The session stopped (exit 3, `stopped`) as
  §3.4 prescribes for a failed check child, before any scored child.
- **Verdict, rule v1 unchanged:** `NO VERDICT: criterion 1, 2, 3 has no verdict`; criterion
  4 refused. No equality, RAM, cleanup or cost result.

§4 says an OOM in training at a tight limit is a criterion 2 FAIL. The rule scores criterion
2 only on `l2l_spill` and the two no-spill arms, and §3.4 stops at a failed check child
before they run, so the rule as committed returns NO VERDICT. It was not changed after the
result. Read as a finding outside the rule: at this limit, with the product's 0.3 headroom
(about 130 MB at its plan point), the first step's growth outside the store did not fit.
In the generous child, about 499 MB of that growth came before the plan point (so it was
already in the working set the limit was built on) and 183 MB after it. Which allocations
these are is not established, nor whether other boxes show the same growth.

What is needed before another session, for the owner to decide: whether the product's
headroom or the calibration's margin should change (either changes the committed protocol
or the product, and needs a new commit before any rental), and the budget for another box.
Nothing has been rented since.

The owner chose the product reserve; protocol v2 (§3.6) follows from that decision.

## 10. Session 2: RTX 4090 VM, 2026-10-27 (protocol v2). NO VERDICT on cleanup

Records: [`results/a3-spill-v2-rtx4090-20261027/`](results/a3-spill-v2-rtx4090-20261027/README.md).
One vast.ai KVM VM (RTX 4090 24 GB, PCIe gen 4 x16, 47.4 GiB RAM, one virtio ext4 disk),
code at `667c4a1b` (protocol v2 at `6bdd0ec8`); every one of the 663 source files hashed in
`protocol.json` equals that commit. The only RTX 3090 VM offered was PCIe 3.0, so the PCIe 4.0
4090 was taken as §5 allows. Admission: ADMITTED (180 of 180 cycles, largest lateness -0.043 s,
0 of 11,520 chunks wrong, no file left). The session ran every child (33), exit 0, `state:
finished`. GPU 0.994 h x $0.52 and storage $0.050: **$0.567** (invoices); the instance is
destroyed and was confirmed gone through the API.

**Verdict (rule v2, unchanged; re-running it on the raw records reproduces `verdict.json`
byte for byte):** `NO VERDICT: criterion 3 has no verdict; L2L is outside the A3 stand's GA
tolerance (criterion 1b, not a spill finding)`.

| # | Criterion | Result | Numbers |
|---|---|---|---|
| 1 | spill == no spill | PASS | `l2l_spill` (16 of 32 layers spilled) bit-equal to `l2l_nospill_1` over 3 optimizer steps; the no-spill reference repeats |
| 1b | L2L vs GA (reported) | FAIL | `l2l_out_of_tolerance` at 4 checks: gradient steps 0, 1, 2 (relative error 0.0181, 0.0213, 0.1772 against limits 0.0165, 0.0206, 0.0767) and loss step 0 (0.000222 against 0.0000103). GA repeats bit for bit; the floor arm judged. Not a spill finding: spill equals no spill bit for bit. Its cause is not established |
| 2 | RAM | PASS | peaks 6,102,315,008 B (`l2l_spill`, MemoryMax 6,354,370,560 B), 6,300,880,896 and 6,400,360,448 B (no spill, 24 GiB); no OOM kill in any child; excess 36,877 B with and without spill |
| 3 | cleanup | NO VERDICT | normal: no file after `train()` nor after the exit (PASS); kill9: the file stayed after SIGKILL, the next start removed it and trained 3 steps (PASS); sigint: the scenario was not reached (below) |
| 4 | cost | PASS | overhead +0.03% [-0.02%, +0.09%] over 5 blocks (B spilled 16-22 layers) |

Why the sigint part has no verdict. The parent sent one SIGINT 0.52 s after the step-2 entry
marker (1 step done); the child wrote markers for 1.31 s after it, finished step 2 and step 3,
and exited `normal` with return code 0, its spill file removed by the normal end. The launch
(`setsid nohup bash -c '...' &` from a non-interactive shell) left SIGINT ignored in the
session and in every child: reproduced afterwards on Linux with the same launch, where such a
child stays running after a real SIGINT and raises KeyboardInterrupt only once the handler is
restored. Session 2 recorded no SigIgn reading, so for session 2 itself this is the
explanation the reproduction supports, not a reading from the box. Protocol v3 (§3.7) changes
the stand, not the verdict.

Memory after the plan point (description, no verdict): in each of 28 records with a cgroup
plan, scope `memory.peak` minus the plan's `used` minus the plan's `host_bytes` (growth
outside the store, over the run) was 183,582,720 to 297,558,016 B; the 15 children at the tight
limit kept 252,055,552 to 366,022,656 B below MemoryMax. The largest is 55% of the 512 MiB
reserve. One box, one model and shape; the reserve stays chosen, not measured.

## 11. Session 3: RTX 4090 VM, 2026-10-28 (protocol v3). SPILL CONFIRMED; 1b FAIL

Records: [`results/a3-spill-v3-rtx4090-20261028/`](results/a3-spill-v3-rtx4090-20261028/README.md).
Measured at `ec795ef175c222b60ff91b9ba44f7f2bfad55dd5`, protocol v3 fixed before the rental
at `bc2ca55038a2d85950356cfdd38d556333519ad1`. All 664 `source_sha256` entries in
`protocol.json` match the measured commit (checked against `git cat-file` bytes). The
retained private gradient tensors let the integration owner replay the measured commit's
CLI comparison on CPU; `compare.json` and the raw-JSON rule's `verdict.json` each reproduce
byte for byte, with all six tensor hash checks passing (`verification.json`).

**Verdict (rule v3, unchanged):** `SPILL CONFIRMED: equal, within its RAM, cleaned up;
overhead below 10% confirmed on a faster machine; L2L is outside the A3 stand's GA tolerance
(criterion 1b, not a spill finding)`.

| # | Criterion | Result | Numbers |
|---|---|---|---|
| 1 | spill == no spill | PASS | 13 of 32 layers spilled; bit-equal gradients, losses, learning rates and 3 optimizer steps; the no-spill reference repeats bit for bit |
| 1b | L2L vs GA (reported) | FAIL | `l2l_out_of_tolerance`: the same four failing checks as session 2 (gradient steps 0/1/2 and loss step 0). GA repeats bit for bit; the floor arm judged. Cause unestablished, not a spill finding |
| 2 | RAM | PASS | spill peak 6,154,133,504 B at MemoryMax 6,409,945,088 B; no-spill peaks 6,354,161,664 and 6,374,264,832 B at 24 GiB; no OOM kill; allocator excess 36,877 B in all three arms |
| 3 | cleanup | PASS | no file after normal `train()` or process exit; one SIGINT in the backward-recompute hold at zero-based layer 16 in optimizer step 2 raised KeyboardInterrupt, exit 130, file gone; SIGKILL left the file, the next start removed it and trained 3 steps |
| 4 | cost | PASS | +2.39% per step, 95% interval [-1.80%, +6.57%], 5 valid A-B-B-A blocks, no void blocks; B children spilled 12-18 layers |

One vast.ai KVM VM, instance 58144303, Quebec host 655981 / machine 154269 (the same host
as session 1): RTX 4090 24 GB, PCIe gen 4 x16, driver 580.95.05, 49.35 GiB RAM, local
ext4 `/dev/vda1` for weights and spill. No fitting PCIe 4.0 3090-class VM was offered, so
the 4090 was used as §5 allows. The offer rate was $0.562962962962963/h including storage;
this is not an invoice total. `gpu_frac` 0.125 is the provider's ratio, not proof of
isolation or a separate physical drive. VM checks recorded 323 passed, 1 skipped,
2 deselected; the separately CPU-run resume checks recorded 2 passed. Fixture hashes match
A4 v3 and both prior sessions.

The tiny CPU signal preflight passed **before the model download**, through the session's
background launch: the launcher read SIG_IGN, children installed `default_int_handler`,
SIGINT interrupted the hold, and kill9/restart cleanup held. Only its sanitized JSON report
is public. Admission: ADMITTED, 180/180 cycles in 139.646 s, maximum lateness 0.034849073 s,
0/11,520 mismatched chunks, no file left; D 172,318,018.5355 B/s and period 0.778895493 s.
Calibration's working set 5,528,711,168 B gave the tight limit above; its one check spilled
14 layers (in band). Build margin +1,290,664,024 B, training margin +373,282,520 B, no
warning. These margins describe this calibration; the product's 512 MiB reserve remains
chosen, not a measured guarantee for other runs.

Descriptive growth (not a new criterion): across 27 records limited by their own scope,
`memory.peak - plan.used - plan.host_bytes` was 186,515,456 to 297,218,048 B (at most 55.36%
of the chosen reserve). Fourteen tight-limit records ended 255,811,584 to 362,397,696 B below
MemoryMax. No spill files remained at session end. The sigint scenario's own plan spilled
17 layers and held at zero-based layer 16; this is distinct from the equality arm's 13.

The session ran from **20:31:59Z to 20:46:26Z** (`session-start.txt`, `session-exit.txt`),
32 children, session exit 0, `state: finished`; 30 children exited 0, sigint exited 130,
kill9 -9 as intended. The cost result is scoped to this faster 4090 VM, this model and
shape, this admitted virtual disk and these step lengths, not slower GPUs or drives.
Cleanup remains best-effort after an uncatchable termination: it depends on the next run
using the same directory and the old process no longer being alive (§8).

Invoices: GPU 0.803 h at $0.5333/h, charged $0.428; storage 0.832 h at $0.0296/h, charged
$0.025; **session 3 $0.453**. A3 total: $0.270 + $0.567 + $0.453 = **$1.290**, leaving
$8.710 of the $10 budget. The API returned `instances: null` at
2026-10-29T13:19:26.272356+00:00. Exact deletion time and mechanism were not retained; the
deleter log records only arming for 2026-10-29T08:09Z, not a deletion action. No exact
deletion-time or self-deletion claim follows from that deadline.

Publication replaces `/root/a3/soup-rnd` with `<work>`, then `/root/a3` with `<box>`.
`provenance.json` records raw and published SHA-256 for every whitelisted record and the
private archive `m3-final.tar` (212 entries, SHA-256
`19c3ada65e89ad0adaf805dbc3c530e8a5a3de5cd38984f5e7c3d8c9554819c2`). Logs, tensors,
operational tooling and private API/billing snapshots are not published. The private
archive and sibling tensors remain available for recomputation; the LF `SHA256SUMS` covers
the public evidence and provenance, and `.gitattributes` preserves their bytes.

## 12. Post-measurement audit corrections

Session 3 and its published records stay pinned to `ec795ef1`; their numbers and verdict
are unchanged. Later GPT 6.1 Sol reviews found invalid-input and error-path defects in the
stand and product. The current rule requires complete parameter/digest/loss/rate evidence
and the fixed expected step counts; matching omissions no longer count as equality. Cost
arms require seven steps, five finite positive timings and their median, and finite positive
gradient norms. Missing cleanup listings are not evidence of absence, and observed leftovers
take failure priority. Only an actually recorded intentional kill9 SIGKILL exempts an
unexplained -9. The signal driver respects the remaining session deadline, and nonfinite
timing or lateness cannot admit a machine. These enforce the existing criteria and leave
the declared rule/profile, thresholds and historical records unchanged.

Product corrections address stale-entry replacement races, non-default cgroup mounts, macOS
volume backing, non-CUDA transfer ordering, sector-aligned raw-I/O request limits, constructor
and teardown resource lifetimes, and old-store release before a larger-window RAM replan.
On older torch without public pinned-cache release, growth may still conservatively refuse
while idle cache remains charged. Cleanup is best-effort, including the native thread-launch
interrupt window before CPython publishes a worker identity.

Local CPU checks exercise real spill/no-spill equality, inherited-SIGINT interruption at the
hold, SIGKILL/next-start cleanup, and POSIX stale-entry replacement. They are not a new CUDA,
native MPS, native macOS-volume or 2 GiB Windows transfer measurement. Regression evidence for
invalid records and resource failures is separate from the frozen GPU result. Replaying the
stricter rule on session 3 still yields its byte-identical verdict.

## 13. Criterion 1b causal diagnostic: RTX 4090, 2026-10-29

Records: [`results/a3-1b-diagnostic-rtx4090-20261029/`](results/a3-1b-diagnostic-rtx4090-20261029/README.md).
This is a targeted cause investigation, **not a product fix or a new spill gate**. Sections
9–11 and their records retain their original results and then-known cause wording; §12's
audit is unchanged. The current bounded conclusion is that **two execution conditions
explain the measured discrepancy on this pinned fixture**: padding policy changes the
forward endpoint, and backward-entry autocast changes residual gradients despite matching
forwards. The contrasts are sequential, not additive error components.

One vast.ai KVM VM, instance **58293613**, RTX 4090, Australia host **370354** / machine
**137610** (the session-2 host/machine), offered at **$0.5826666667/h** including storage;
the offer is not an invoice. Baseline product/stand **`ec795ef1`**, diagnostic source
**`cf0a7490`**, the historical Mistral-7B NF4 fixture, bf16, LoRA r 8, k 4, seq 512.
Saved source, model, initial-adapter, shard-cache and dependency fingerprints are identical
across all valid arms. MATH is used except for the EFFICIENT_ATTENTION floor arms.

**The original 1b FAIL is reproduced, not relabeled:** gradient relative errors
**[0.018090733100463656, 0.02133434290248132, 0.17721940196011468]** exceed the original
limits **[0.016452461646560432, 0.020615278589764058, 0.07667679771063814]**; first logged
loss relative error **0.00022173059391030002** exceeds **0.000010254296903358378**.
Native GA's first loss is **12.834332466125488**, versus original L2L's
**12.831486701965332**. The unchanged pinned comparator reports `l2l_out_of_tolerance`.

### 13.1 Padding-policy endpoint and its localization limit

All three observer controls are `BITMATCH`. The original native-versus-padded GA trace
contrast first observes a valid-token difference at **decoder 0, window 0, microbatch 0**,
max absolute **0.001953125**, relative L2 **0.004390999502744277**. That contrast changes
padding **and mask policy**; it is not pure-padding layer localization.

The corrected **`ga_native_no_mask_verified_v3`** control removes only the native mask
and records actual effective inputs after removal. It matches native GA's loss and all
**256 gradients** bit for bit, at the same rows **[5, 6, 1, 2]**, native lengths
**[307, 512, 384, 256]** and **N=1215**. Native-no-mask versus padded-no-mask therefore
isolates a padding-policy endpoint effect: padded GA's first loss matches L2L's
**12.831486701965332**. The no-mask arm is untraced, so the original combined contrast's
decoder-0 localization cannot be transferred to pure padding. Two earlier no-mask attempts
had incorrect effective metadata or a metadata-file collision; neither is valid control
evidence or published as such.

### 13.2 Matched-input residual and the backward-scope intervention

Padded GA versus original L2L has **144 bitmatching initial-forward tensors** (140
valid-token/output tensors plus four CE numerator scalars), matching actual per-row losses,
N and independently computed CE numerators. Yet gradients differ: relative L2
**0.007499608746411462**, max absolute **0.2308349609375**.

A direct-child diagnostic wrapper disables **ambient CUDA autocast at backward entry**
only, around the original `torch.autograd.backward`, forwarding arguments unchanged and
restoring the callable in `finally`. This covers both Tensor.backward delegation and
L2L's direct autograd entrypoint, not explicit forward or reverse-layer forward
recomputation. All **144 original-versus-adaptive L2L initial trace file SHA-256 values**
remain identical; layer calls, per-microbatch losses, windows, dispatch and training-step
losses also stay identical. All **256 gradients** now bitmatch padded GA.

Three-step confirmation gives exact gradients at every step, an independent bitmatching
padded-GA repeat, and applied learning rates **[0, 0.001, 0.0005]**. Aggregate logged losses
are **not all bit-equal**: at zero-based step 1, padded GA logs **12.808198928833008** and
L2L-outside **12.808199882507324**; loss relative errors are
**[0, 7.445811247195722e-8, 0]**. The unchanged comparator with a newly matched padded
EFFICIENT floor returns **`within_tolerance`, `pass: true`** for this **new diagnostic
comparison**, not for the original gate. Backward-scope alignment alone still differs
from native GA: gradient relative errors **[0.015837362726258193, 0.014676687301150623,
0.12116253884098395]**.

Disabling ambient autocast at backward entry is sufficient for the matched-input residual
on this three-step fixture without changing L2L's schedule or accumulation order. An
accumulation-order change is therefore **not necessary** for that residual correction.
This does **not** establish that all backward work is FP32 or identify an internal cast or
CUDA kernel. Four decoder-0 **input** adjoint boundaries remain unmatched; the earliest
backward divergence is unresolved. The analyzer's residual kwargs warning is GA's extra
`num_items_in_batch` only, not an observed mask/position mismatch; actual denominators and
all remaining captured args/kwargs match.

### 13.3 Evidence scope

The new result directory publishes compact **DERIVED** arm/trace/numeric-pair metadata,
all original manifests, common source/model/cache/dependency fingerprints, the unchanged
original session, the aligned diagnostic comparator and confirmation summary. Provenance
records raw source hashes, published hashes, explicit field projections and longest-first
path substitutions; the LF manifest is byte-preserved by `.gitattributes`. The full analysis,
tensor archives, logs, operational tooling and private access/API records are not committed.
Only the retained raw layout and pinned sources permit tensor-level numeric replay; public
projections are not drop-in analyzer inputs. The result README gives the exact limits,
wrapper behavior/source hashes and retained initial-export archive hash.

No automatic product cutover, generalized L2L/GA parity, model-quality claim, or change to
the historical spill verdict follows from this investigation.
