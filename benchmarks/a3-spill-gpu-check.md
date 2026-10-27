# A3 stage 2 - GPU check of activation spill in `training.l2l`

**Status: protocol only, no GPU run yet. Criteria fixed by commit `2a05e15f` before any
rental; the machine admission (§5) by commit `794c6827`.** The rule is
`harness/a3_spill_rule.py` (`RULE`, profile `a3-spill-gpu-check-v1`, `rule_version` v1): the
same text in machine form, copied into every session's `protocol.json`. The stand is
`harness/a3_spill_stand.py`, the admission `harness/a3_admission.py`. CPU tests:
`tests/test_a3_spill_rule.py`, `tests/test_a3_spill_stand.py`, `tests/test_a3_admission.py`.

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
| 7 | `sigint` | tight | ONE SIGINT inside optimizer step 2 |
| 8 | `kill9`, then `kill9_next` | tight | SIGKILL inside step 2; a new child in the same spill directory |
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
`training_step` entry. The parent waits for the marker that says one optimizer step is
done (step 2 has started), waits 0.5 s, lists the spill directory, and sends one signal to
the PID in the marker. After the exit it lists the directory again.

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

## 4. Calibration of the tight limit

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
git clone https://github.com/SoupTeam/soup-rnd && cd soup-rnd
git checkout 794c6827   # the commit that fixed the admission (on top of 2a05e15f)
A0_PYTHON=python3.12 bash benchmarks/a0-bootstrap.sh   # venv, CUDA torch, pinned deps
.venv/bin/python -m pytest tests/test_a3_spill_rule.py tests/test_a3_spill_stand.py \
  tests/test_a3_admission.py tests/test_l2l_spill*.py -q -o addopts=''
export PYTHONPATH="$PWD/src:$PWD/benchmarks/harness" HF_HUB_OFFLINE=0
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("unsloth/mistral-7b-instruct-v0.3",
                  revision="60b99dcbc7f853b7cd18472a607e8d045b96f03e",
                  local_dir=".soup/models/mistral-7b", token=False,
                  allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"])
EOF
export HF_HUB_OFFLINE=1
SPILL=/mnt/nvme/a3-spill        # local NVMe, not overlay, not tmpfs
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
  --out .soup/results/a3-spill-01
# 3. Run (as root; --user-scope in a user session). Each child gets its own scope:
.venv/bin/python benchmarks/harness/a3_spill_stand.py session \
  --weights .soup/models/mistral-7b --shards .soup/f1-nf4 --spill-dir "$SPILL" \
  --out .soup/results/a3-spill-01 --execute
# 4. Verdict, recomputed from the raw JSON alone:
.venv/bin/python benchmarks/harness/a3_spill_rule.py .soup/results/a3-spill-01 \
  --out verdict.json
```

The scopes the session runs (as printed by step 2), calibration first, then the session's
children:

```bash
systemd-run --scope --quiet --collect -p MemoryMax=25769803776 -p MemorySwapMax=0 -- \
  <python> a3_spill_stand.py child --arm calibration ... --memory-max 25769803776
systemd-run --scope --quiet --collect -p MemoryMax=<TIGHT> -p MemorySwapMax=0 -- \
  <python> a3_spill_stand.py child --arm calibration_check_1 ... --memory-max <TIGHT>
systemd-run --scope --quiet --collect -p MemoryMax=25769803776 -p MemorySwapMax=0 -- \
  <python> a3_spill_stand.py child --arm l2l_nospill_1 ... --tensors-dir <out>.tensors
systemd-run --scope --quiet --collect -p MemoryMax=<TIGHT> -p MemorySwapMax=0 -- \
  <python> a3_spill_stand.py child --arm l2l_spill ... --tensors-dir <out>.tensors
```

`<TIGHT>` is the calibrated limit in bytes, recorded under `limits` in `session.json`.
Exit codes: 0 finished, 4 stopped before the cost blocks (§3.4), 3 stopped otherwise, 2
refused arguments (an existing `--out` or `<out>.tensors`, k or seq other than 4/512, no
`systemd-run`, MemAvailable below the generous limit). After the run, copy the results,
then destroy the instance yourself. The
session never does.

## 7. Files that come back

`.soup/results/a3-spill-01/`: `protocol.json` (written first: rule, args, plan, git HEAD,
versions, SHA-256 of every `src/soup_cli` and harness `.py`), `session.json` (each child's
command, limit, return code, spill directory after exit; the signals sent), `verdict.json`,
`calibration.json`, `calibration_check_<n>.json`, `arms/<arm>.json`, `compare.json`,
`cleanup/<arm>.json` (none for `kill9`: it is killed; its marker is in `session.json`),
`cost/b<NN>_p<P>_<A|B>.json`, `markers/`, `logs/`. Beside it: the admission's inputs
`.soup/a3-prep.json` and `.soup/a3-demand-A.json`, and its `.soup/a3-admission.json`,
`.soup/a3-admission-cycles.csv` and `.soup/a3-admission-per-second.csv`. **Not published**:
`.soup/results/a3-spill-01.tensors/`.

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
