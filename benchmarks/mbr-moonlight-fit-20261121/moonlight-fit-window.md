# Ticket 02: Moonlight-16B-A3B windows on RTX 3050 Laptop (4 GiB)

## Setup

- Card: NVIDIA GeForce RTX 3050 Laptop GPU, sm_86, 4096 MiB total. Before the runs,
  `nvidia-smi` reported 3759 MiB free.
- Model: moonshotai/Moonlight-16B-A3B, revision `2a9a177f4841649f849ba941e41d2ee73c042314`.
  The revision is pinned in `p72_prepare_moonlight_cache.py` and `p3_route_preserving_mbr_soup_bridge.py`.
  The B5 report used `476b36a4`. The two revisions differ, so memory peaks are not compared with B5 directly.
- Software: torch 2.14.1+cu130, transformers 5.20.2, peft 0.21.2, triton 3.8.0, bitsandbytes 0.50.2.
- Repo commit of the code that ran: `a7a2a16d`.
- LoRA targets: `q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj`. Raw rows: `fit.jsonl`.

## Model files

Built with `benchmarks/harness/p72_prepare_moonlight_cache.py --root <MOONLIGHT_ROOT>`, run from
`benchmarks/harness`, with `src` and the repo root on `PYTHONPATH`. Build record from
`<MOONLIGHT_ROOT>/results/cache.json`:

| item | value |
|---|---|
| `status` | PASS |
| `revision` | `2a9a177f4841649f849ba941e41d2ee73c042314` |
| `total_params` | 15,960,110,208 |
| download | 743 s |
| shard | 46 s |
| `source/` (weights and config) | 29.73 GiB |
| `cache/` (per-expert NF4) | 9.26 GiB |

The tokenizer needs `tiktoken` and `Brotli`. The first run failed without them, so both are
installed in the CUDA venv before the runs.

## Memory fit

Memory peaks are read per run from the harness row. Peaks do not depend on GPU clock.

| arm | k | S | alloc GiB | reserved GiB | NVML GiB | RAM GiB | status |
|---|---|---|---|---|---|---|---|
| C | 8 | 64 | 0.93 | 1.12 | 1.26 | 4.58 | ok |
| A0 | 8 | 64 | 0.92 | 1.06 | 1.20 | 4.59 | ok |
| C | 8 | 512 | 2.02 | 2.11 | 2.25 | 4.96 | ok |
| A0 | 8 | 512 | 1.77 | 2.07 | 2.21 | 4.96 | ok |
| C | 64 | 64 | 1.69 | 1.89 | 2.03 | 4.98 | ok |

All five windows fit in 4 GiB. The largest NVML peak is 2.25 GiB (C, k=8, S=512). The window
(64,512) was not run; it is not in ticket 02.

## Speed: first pass not valid, redo in speed-pilot-5blocks.md

The step times from the first pass (C 13.5 s, A0 68.5 s at k=8, S=64, and the others) are one run per
arm, with no `ArmWatch` record and no GPU clock record. They are not speed evidence.
`benchmarks/METHOD.md` requires ABBA blocks, `ArmWatch` on every run, and a clock record. A
redo followed that method. Its results, with the five-block ABBA pilot, are in
`speed-pilot-5blocks.md`.

A GPU check on this machine before the redo: idle at 210 MHz, and under a 25 s load the SM clock
held at 1927 to 1965 MHz at about 90 W. That is not the 210 MHz under a 10 W cap that
`METHOD.md` describes. The redo has a clock record for every arm.
