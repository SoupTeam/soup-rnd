# Ticket 03: exact GEMM bits on Moonlight-16B-A3B, RTX 3050 Laptop (sm_86, 4 GiB)

## Setup

- Card: NVIDIA GeForce RTX 3050 Laptop GPU, sm_86, 4096 MiB. The power limit read 80 W with
  `nvidia-smi -q -d POWER` before the first run. The CSV query in `environment.txt` returns N/A for
  the limit, and the SM clock in it reads 210 MHz because the GPU was idle at that moment. Bits do not
  depend on the clock.
- Model: moonshotai/Moonlight-16B-A3B, revision `2a9a177f4841649f849ba941e41d2ee73c042314`
  (per-expert NF4 cache from ticket 02).
- Software: torch 2.14.1+cu130, CUDA 13.0, the CUDA venv `~/.venvs/soup-rnd-cuda`.
- Environment row (`environment.txt`): `toolchain_error=None`, capability (8, 6). `lt-fused` builds
  here.
- LoRA targets: `q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj`, rank 16. Three steps per run; the loss
  and gradients come from the last step.
- Mode: deterministic (`use_deterministic_algorithms(True, warn_only=True)`) for every bits run.
- Arms: A0 (reference), C (ordered backward), E (`lt-fused`), F (E plus async lease),
  G (F plus fast row gather). Ladder k = 1, 2, 4, 8.
- Verdicts: `torch.equal` on the loss and on all 216 LoRA gradients, from the tensors saved per run
  (`bits-verdict.json`). The harness gradient digest agrees with `torch.equal` in every cell
  that was checked.
- Raw tensors (about 1.3 GB) stay outside git in `~/moonlight-local/results/bits/`.

## Attention backend: two windows, two backends

The default SDPA backend gave A0 different gradients from itself at S=512 (see the repeat section).
S=512 therefore runs with the math backend (`--math-attention`). S=64 runs with the default backend,
where A0 reproduced itself at k=1. The two windows use different backends, so their numbers are not
comparable with each other.

## Window S=64, default backend (valid at k=1 only)

| k | A0 loss | C | E | F | G |
|---|---|---|---|---|---|
| 1 | 1.521705 | equal | equal | equal | equal |
| 2 | 1.928908 | differs (loss 1.918482) | equal | equal | equal |
| 4 | 2.542863 | differs (loss 2.526555) | equal | equal | equal |
| 8 | 2.463173 | differs (loss 2.453490) | equal | equal | equal |

A0 repeat at k=1 (`repeats/rep3-s64.jsonl`): equal to the first A0 run in all 216 gradients
(`repeats/compare-s64-default.txt`). Not repeated at k=2, 4 or 8, so the C-differs verdicts at
k=2, 4 and 8 rest on one A0 run each.

## Window S=512, default backend (void as a comparison against A0)

The default-backend S=512 sweep (`s512/`) gives `differs` in all 16 cells, including E, F and G at
k=1 and 2 where the losses match. Three A0 runs at k=1 on the default backend do not agree with each
other (`repeats/compare-s512-default.txt`): 105 of 216 gradients differ between any two of them, with
a largest difference of 5.4e-3 against a largest gradient of 1.52. The losses are identical. A
reference that changes between runs cannot judge the other arms, so these verdicts are void.

## Window S=512, math backend (the verdicts to read)

| k | A0 loss | C | E | F | G |
|---|---|---|---|---|---|
| 1 | 2.790814 | equal | equal | equal (6 of 7 runs; one outlier) | equal |
| 2 | 2.743729 | differs (loss 2.734687) | equal | equal | equal |
| 4 | 2.774396 | differs (loss 2.783361) | equal | equal | equal |
| 8 | 2.846923 | differs (loss 2.851639) | equal | equal | equal |

Reproducibility of the reference: three A0 runs at k=1 on the math backend agree in all 216 gradients
(`repeats/compare-A0-s512-math.txt`).

F at k=1 was the one unstable cell; seven runs now settle it (next section). The sweep's F row is the
only run that differs from A0, and six further runs agree with A0. The F verdict at k=1 is therefore
`equal`, with the outlier kept in the table rather than hidden.

The math backend changes the numbers: its A0 losses differ from the default backend's at every k, so
the default and math S=512 numbers must not be compared.

## F repeats at k=1 and k=2, S=512, math backend (ticket 03 open item)

Model Moonlight-16B-A3B, card RTX 3050 Laptop sm_86 4 GiB, window S=512, arm F (E plus async lease),
mode bits (deterministic), torch 2.14.1+cu130. Each run is a fresh process; its loss and all 216 LoRA
gradients are saved. Comparisons: `repeats/compare-F-k1-s512-math-7runs.txt` (7 runs) and
`repeats/compare-F-k2-s512-math-5runs.txt` (5 runs), `torch.equal` pairwise.

k=1, seven runs (the sweep `s512-math/F`, reps 1 and 2, and new reps 3 to 6):

| result class | runs | loss | fused_forward | against A0 |
|---|---|---|---|---|
| typical | 6 (rep1, rep2, rep3, rep4, rep5, rep6) | 2.790814 | 3816 | equal in all 216 gradients |
| outlier | 1 (s512-math/F sweep) | 2.783530 | 3822 | differs in 108 gradients |

The outlier's largest difference is 0.0150202 against a largest gradient of 1.53392. The six-run
class agrees with A0 in the loss and every gradient, and it also has the same fused-visit count; the
outlier took 3822 fused forward visits against 3816, so it is a different execution trace, not the
typical one.

k=2, five runs (the sweep `s512-math/F` and new reps 1 to 4): one result. All five are equal to each
other and to A0 in the loss and all 216 gradients. k=2 is stable.

Verdict for the open item: F has a typical result at both k. At k=1 it is the six-run class, which is
`equal` to A0; the sweep row is one outlier in seven runs. At k=2 every run is `equal`. The single
k=1 flip could not be reproduced in six further runs, and its cause is not identified; the async
lease path shows it rarely, as Kanana did once at k=2.


## Fused visits (counters of the `lt-fused` path)

- Every E, F and G run in both windows took the fused path on every visit: `python_forward` = 0 and
  `python_backward` = 0 at all k. Fused forward visits ran from 3333 to 4545 per run.
- Counter run, E at k=64, S=64 (`counter-k64/rows.jsonl`, single run, default backend, deterministic):
  `python_forward` = 0, `python_backward` = 0, `fused_forward` = 4788, allocated 1918 MiB, NVML
  2346 MiB. The Python-path hypothesis does not hold for this window on Moonlight. This is one run,
  and it is not a ladder.

## Memory, C against E in the same window (deterministic runs, from `rows.jsonl`)

| window and backend | arm | alloc MiB | reserved MiB | NVML MiB |
|---|---|---|---|---|
| S=64, default, k=1 | C | 987 | 1072 | 1266 |
| S=64, default, k=1 | E | 1006 | 1072 | 1268 |
| S=64, default, k=8 | C | 1003 | 1178 | 1372 |
| S=64, default, k=8 | E | 1041 | 1196 | 1392 |
| S=512, math, k=1 | C | 1836 | 2136 | 2330 |
| S=512, math, k=1 | E | 1877 | 2134 | 2328 |
| S=512, math, k=8 | C | 2286 | 2482 | 2676 |
| S=512, math, k=8 | E | 2415 | 3456 | 3650 |

At S=512, k=8, E reserves 1.0 GiB more than C and reaches 3650 MiB NVML on a 4096 MiB card. The
S=512 default-backend sweep reached 3526 MiB at the same cell, so the figure depends on the backend.

## Repeat checks (files in `repeats/`)

| check | runs | result |
|---|---|---|
| A0, S=64, k=1, default | 2 | equal in all 216 gradients |
| A0, S=512, k=1, default | 3 | pairwise differ in 105 to 108 gradients; the loss is equal |
| A0, S=512, k=1, math | 3 | pairwise equal in all 216 gradients |
| F, S=512, k=1, math | 7 | six equal to each other and to A0, the sweep run differs from all six (108 gradients) |
| F, S=512, k=2, math | 5 | all five equal to each other and to A0 |

Scripts: `repeat_a0.sh SEQ K RUNS [math] [first]` (with `ARM=F` for another arm; `first` starts the
numbering at a later repeat, and folder names carry k), and `compare_repeats.py TENSOR.pt ...` for the
pairwise comparisons.

## Not in this run

- Speed. Bits runs do not measure speed; the step times in the rows are not speed evidence. The speed
  and memory gate of D1 stays on Vast for Kanana.
- The S=64 ladder on the math backend. S=64 uses the default backend, so its A0 is only checked at k=1.
- A0 repeats at k=2, 4 and 8, and F repeats at k=4 and 8. F at k=1 and k=2 is covered above.
- The k=64 counter run is a bare `run` call. Its row has no `ArmWatch` record.
- The k=64 ladder with C as reference. Ticket 04 covers it in `k64-bits-summary.md`.
- Cause of the default-backend gap. The PyTorch warning in the log says memory-efficient attention
  backward is not deterministic. The math backend fixes A0 at k=1, which supports this, but no test
  isolates the kernel beyond that.

## Files

- `environment.txt`: detect_environment and GPU query.
- `s64/`, `s512/`: default-backend sweeps (`s512/` void, see above). `s512-math/`: math-backend sweep.
  Each has `rows.jsonl`, `bits-verdict.json` and `arm_records.jsonl`.
- `counter-k64/rows.jsonl`: E at k=64, S=64.
- `repeats/`: repeat rows, logs and comparison outputs. `rep3-s64` was run with an earlier version of
  `repeat_a0.sh` that named runs differently; the current script names them `k<k>-rep1` to
  `k<k>-repN`. `compare-F-k1-s512-math-7runs.txt` and `compare-F-k2-s512-math-5runs.txt` are the
  seven-run and five-run F checks; `compare-F-s512-math.txt` is the earlier three-run check.
- `run_bits.sh` (default backend, both windows), `run_bits_math_s512.sh` (math backend, S=512),
  `repeat_a0.sh`, `compare_repeats.py`.
