# Ticket 17: E's bits after the arena, G's speed, and the D+segmented cell

Moonlight-16B-A3B (`2a9a177f`), RTX 3050 Laptop (sm_86, 4 GiB, one card, used exclusively here),
torch 2.14.1+cu130, CUDA 13.0, venv `~/.venvs/soup-rnd-cuda`, LoRA targets
`q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj` (rank 16). Every run is a fresh process, on AC power,
inside `systemd-inhibit --what=sleep`, wrapped in `ArmWatch` with `suspend`, `power` and
`foreign_reads` required (`speed-rule.md`). The ticket 13 health gate (`gate.py`: AC online, SM
clock at least 1500 MHz under a 10 s load, enforced power limit at least 50 W) ran **before every
single run**: 60 gate checks in the speed pass, 22 in the bits pass, zero refusals, SM 1740-1957
MHz, limit 89.7-90.0 W. All hands ran under the one memory recipe of ticket 13 phase 2:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A0_SCRATCH_PREALLOC_ROWS=4096 A0_SCRATCH_PREALLOC_HID=2048 A0_SCRATCH_PREALLOC_INTER=1408
A0_SCRATCH_WHEN=lazy A0_SCRATCH_PREALLOC_GATHER=1
```

`A0_SCRATCH_PREALLOC_ROWS=4096` is the largest visit of the campaign (`k*S = 64*64`). The arena
options are inert for C, D and segmented; they only size the `lt-fused` scratch arena.

## Answer

**Speed: no certified verdict.** The archived 60 speed rows lack `campaign_id`, so the analyzer
refuses both windows. The table below is a non-certified diagnostic of those rows.
**Bits: E matched A0 in all eight tested bit cells after the arena change; G matched A0 in its one cell.**

Non-certified speed diagnostic (legacy archive):

| window | comparison (B vs A) | ratio of means A/B | ratio speedup | mean paired A−B | mean per-block time reduction |
|---|---|---|---|---|---|
| (8,64) | G vs C | 1.0955× | +9.55% | +1.784 s | +8.60% |
| (8,64) | E vs C | 0.9887× | −1.13% | −0.233 s | −1.28% |
| (8,64) | G vs E | 1.1080× | +10.80% | +2.017 s | +9.74% |
| (8,512) | G vs C | 1.1586× | +15.86% | +3.750 s | +13.63% |
| (8,512) | E vs C | 0.9869× | −1.31% | −0.364 s | −1.40% |
| (8,512) | G vs E | 1.1739× | +17.39% | +4.114 s | +14.81% |

Ratio speedup is `100 × (mean(A)/mean(B) − 1)`. Time reduction is the mean of
`100 × (1 − B/A)` computed separately in each block. Positive values favor B in both columns.
G took less time than C in all five blocks at each window.

- **G matched A0 after the arena at k=2, S=64** (0 of 216 gradients differ, loss equal).
  Ticket 03 measured G against A0 at k=1,2,4,8 before the arena.
- **G uses the same memory as E at the measured precision**: NVML +194 MiB over C at (8,512),
  +198 MiB at (8,64) —
  same numbers E pays (ticket 16: +194/+198), and G's reserved equals E's to the MiB.
- E's mean step was higher than C's. The paired t on seconds was −0.57 and −0.95, below the
  absolute critical value 2.78 for these five blocks.
- **The D drift at k=2 was the forward.** D plain differs from A0 (108/216 gradients, max abs
  4.51e-2); D with `expert_gemm=segmented` (DS) is bit-equal to A0 (0/216, loss equal). D's own
  decoupled backward is exact in this cell; the grouped native forward was the whole drift.
  The decision on D does not change: it is slower than C in every window (ticket 06) and this
  cell only names the mechanism. Comment added to ticket 06.

## Summary: window x hand x speed x memory x bits

Speed and memory columns are non-certified diagnostics of the legacy archive (see Limitations).

| window | hand | step s | NVML peak GiB | bits against A0 up to k=8 |
|---|---|---|---|---|
| (8,64) | C | 20.46 | 1.18 | differs at k=2,4,8 (ticket 03) |
| (8,64) | E | 20.70 | 1.37 | equal at k=1,2,4,8 (this campaign) |
| (8,64) | **G** | **18.68** | 1.37 | equal at k=2 after arena; k=1,2,4,8 before arena (ticket 03) |
| (8,512) | C | 27.40 | 2.23 | differs at k=2,4,8 (ticket 03) |
| (8,512) | E | 27.77 | 2.42 | equal at k=1,2,4,8 (this campaign) |
| (8,512) | **G** | **23.65** | 2.42 | untested after arena; equal at k=1,2,4,8 before arena (ticket 03) |

## Bits

Bits use `torch.use_deterministic_algorithms(True, warn_only=True)`. The verdict comes from
`torch.equal` on the saved loss vector and all 216 expected LoRA gradients. The analyzer checks
their names and shapes against the plain A0 model manifest, and validates the run metadata.
These are observed comparisons of the saved tensors. They do not guarantee equality on every run.

| cell | arm | vs A0 | loss equal | grads differing | max abs | dataset_sha |
|---|---|---|---|---|---|---|
| k=1, S=64 | E | equal | yes | 0 / 216 | 0 | agrees (6beb56ae) |
| k=2, S=64 | E | equal | yes | 0 / 216 | 0 | agrees |
| k=4, S=64 | E | equal | yes | 0 / 216 | 0 | agrees |
| k=8, S=64 | E | equal | yes | 0 / 216 | 0 | agrees |
| k=1, S=512 math | E | equal | yes | 0 / 216 | 0 | agrees (91048d80) |
| k=2, S=512 math | E | equal | yes | 0 / 216 | 0 | agrees |
| k=4, S=512 math | E | equal | yes | 0 / 216 | 0 | agrees |
| k=8, S=512 math | E | equal | yes | 0 / 216 | 0 | agrees |
| k=2, S=64 | G | equal | yes | 0 / 216 | 0 | agrees |
| k=2, S=64 | D | **differs** | no | 108 / 216 | 4.51e-2 | agrees |
| k=2, S=64 | DS | equal | yes | 0 / 216 | 0 | agrees |

Stored-tensor cross (CPU, `torch.equal` on loss and all gradients). Each side is first bound to
its row: the 216-name manifest, the loss sum, and `grad_sha256` of the row. The ticket 03 and
ticket 06 rows supply the stored digests. All 23 crosses passed that binding (21 equal, 2 differs):

| fresh | stored | result |
|---|---|---|
| A0, every E cell | ticket 03 A0 (20261122) | 0 differences |
| E, every E cell | ticket 03 E (20261122) | 0 differences |
| A0, G cell | ticket 03 A0 | 0 differences |
| G, G cell | ticket 03 G | 0 differences |
| A0, D cell | ticket 03 A0 | 0 differences |
| D, D cell | ticket 06 D (20261124) | 0 differences |
| A0, DS cell | ticket 03 A0 | 0 differences |
| DS, DS cell | ticket 06 D (20261124) | **108 / 216**, max abs 4.51e-2 |
| DS vs D (same day) | fresh D | **108 / 216**, max abs 4.51e-2 |

The S=64 ladder stops at k=8. A0 does not run above k=8. Ticket 03's k=64 counter was E.
This campaign measures neither G speed nor a G bit cell at k=64. Its DS coverage is k=2, S=64 only.

## Memory

Peaks of the speed runs, max over the ten runs of the cell, GiB (allocated / reserved / NVML /
host RAM):

| window | C | E | G |
|---|---|---|---|
| (8,64) | 0.93 / 1.03 / 1.18 / 4.59 | 1.10 / 1.23 / 1.37 / 4.63 | 1.10 / 1.23 / 1.37 / 4.58 |
| (8,512) | 1.99 / 2.09 / 2.23 / 4.97 | 2.05 / 2.28 / 2.42 / 5.01 | 2.05 / 2.28 / 2.42 / 4.97 |

G spends E's arena and nothing besides: at (8,512) G's NVML peak is 2 MiB below E's and reserved
is equal; against C the premium is **+194 MiB NVML / +194 MiB reserved** at (8,512) and
**+198 / +198** at (8,64), the ticket 13/16 price.

## Method

- Speed: `mbr_measure.py ceg-speed`, 5 blocks of **C-E-G-G-E-C** per window, block-major, one
  fresh process per run: 60 runs, 60 ok, no void replacement needed. The order is the ticket's
  `A-B-C-C-B-A` with the three hands in the roles C, E, G, so every hand sits at the block centre
  and linear drift cancels. The arm's block number is the mean of its two runs.
  The window number is the mean across blocks. `repeats()` reads the original four run times
  for each pair in A-B-B-A order. It returned N=2 for all pairs. The five pilot blocks counted
  toward N and no top-up was due. `analysis.json` reports the paired t on seconds and retains the
  historical paired t on per-block percentages under a separate label.
- Bits: `mbr_measure.py lever-bits` (A0 and E at k=1,2,4,8 on S=64 and S=512 math; A0 and G, A0
  and D at k=2, S=64) and `ds-bits` (A0 and DS at k=2, S=64), deterministic mode. `--arms` keeps
  the reference and the one hand under test. Every run wrote its loss and gradients to
  `bits/<label>/tensors/` (git-ignored, about 28 MB each).
- Each accepted speed block has six consecutive C-E-G-G-E-C successes with matching ArmWatch
  records. The archived run-max threshold is 1000 MHz, distinct from the ticket 13 pre-run gate.
  An invalid block is preserved and replaced in full under a new ID. After the pilot, the
  launcher fills the largest N across windows and pairs, then recomputes once for any final top-up.
- A bits payload counts only when `mm.payload_report` accepts it: the complete gradient set of
  216 names and shapes (the authoritative manifest `mm.MOONLIGHT_GRAD_SHAPES`, which `analyze.py`
  `GRAD_SHAPES` is), a float64 loss of length k that sums to the row's loss, and gradients that hash
  to the row's `grad_sha256` under `mm.grad_digest`. A row's own `grad_shapes` is never the
  manifest: a partial set that is self-consistent gives `no verdict`. A missing digest, a
  mismatch, a partial or empty set gives `no verdict` with its reasons, never `equal`.
  `bits-verdict.json` written by the launcher uses the same manifest.
- A speed row counts only with its dataset hash for its sequence length, its `campaign_id`, the
  levers of its arm, every tracked tool version, `alloc_conf=expandable_segments:True`, and finite
  non-negative `peak_allocated`, `peak_reserved` and `peak_nvml`. One mixed environment (tool
  versions, allocator, `git_sha`, `ptxas`) or campaign ID across rows refuses the verdict. The
  analyzer takes the expected campaign ID from `--source` and `--cache` (defaults
  `$MOONLIGHT_ROOT/source` and `$MOONLIGHT_ROOT/cache`, the launchers' values with the harness
  defaults for steps, LoRA rank and tokenizer). Each row and arm record must carry that ID and its
  `run` number. The analyzer reads no marker file to relax these checks; `.ticket17-active` only
  gates `RESUME=1` in `run_speed.sh`.
- `campaign_id` is a digest of source, cache, tokenizer, LoRA rank and targets, and steps. The
  harness writes it into every row. `replace_void.py` and `RESUME=1` refuse rows with another ID
  before any gate, sweep or write.

## Resume contract

The resume path refuses. It never repairs evidence in place and never skips a record.

- A final record without its trailing newline (a torn record, or a complete one that was never
  terminated) or a malformed record anywhere else makes `rows.jsonl` or `arm_records.jsonl`
  unreadable. `replace_void.py` then exits nonzero before any gate, sweep or write, and the files
  keep their original bytes.
- Why refusal, not bounded recovery: a torn record means its block is incomplete. The block is
  replaced whole under a new journaled ID (`attempts.jsonl`), so the cost is one rerun. Automatic
  recovery would have to decide which bytes are evidence, which this campaign cannot verify.
- The manual recovery runs on a copy only: copy the campaign folder to a fresh `OUT`, delete the torn or
  malformed line from the copy's `rows.jsonl` or `arm_records.jsonl`, and run `replace_void.py`
  there with the same `--source`, `--cache`, `--lora-targets` and `--label`. Never edit the archive.
  The copy's block then fails its completeness check and is replaced under a new ID.

## Limitations

- Speed archive (`speed/`, 60 rows): all metadata except `campaign_id` is present. The analyzer
  gives no verdict. Its step times are diagnostics. Certified speed needs a new measurement under
  this code. The archive is not rewritten and no field is filled in.
- The archived bit rows carry no `campaign_id`. `analyze.py` refuses every bits row whose
  `campaign_id` is missing or differs from the expected one, so archived bits cells get no verdict
  and are diagnostic only. The archive is not rewritten. New bit rows carry `campaign_id`
  (`run_one`) and pass this check. Crosses against stored tensors are reported separately.
- The tests use synthetic `tmp_path` tensors only, and read no git-ignored `.pt` file. The
  launcher tests skip unless the platform is Linux with bash, flock and coreutils. Windows and
  macOS evidence is pending.
- Torn terminal JSONL records are refused, with no automatic recovery (see Resume contract).
- No GPU run was made for this change.
- DS uses D's step options and `expert_gemm="segmented"`. Only the joined expert forward reads its
  gather's capture buffer. DS R1/R2 skip that forward. E-Bwd receives its segment counts directly.

## Reproduce

```bash
export OUT="$(mktemp -d)"
bash benchmarks/mbr-moonlight-post-arena-20261125/run_bits.sh
bash benchmarks/mbr-moonlight-post-arena-20261125/run_speed.sh
"${VENV:-$HOME/.venvs/soup-rnd-cuda}/bin/python" \
    benchmarks/mbr-moonlight-post-arena-20261125/analyze.py --root "$OUT"
```

Both launchers require `systemd-inhibit`. `RESUME=1` on `run_speed.sh` repairs a speed campaign
started in that `OUT`; it never continues a partial block. Use a fresh `OUT` for another bits pass.
The analyzer writes `analysis.json` and `rows.md`, and exits nonzero when required evidence cannot
produce a verdict. Older stored-tensor crosses are reported separately from same-campaign cells.
With the archived local tensors present, regenerate without launches by running only the analyzer
with the campaign directory as `--root`. Ticket 03 tensors are read from `--stored-root` or
`${MOONLIGHT_ROOT:-$HOME/moonlight-local}/results/bits`.

## What this changes for the default

G is a candidate to compare against E. It reduced mean per-block time by 9.74% and 14.81% versus E
with the same memory at the measured precision. Coverage gaps remain before any default decision:

- G speed at (64,64) was unmeasured. Ticket 16 measured C at 35.50 s and E at 42.18 s there.
- G's only post-arena bit cell is k=2, S=64. This does not certify G at k=4 or k=8 after the arena
  or on the S=512 math backend.
