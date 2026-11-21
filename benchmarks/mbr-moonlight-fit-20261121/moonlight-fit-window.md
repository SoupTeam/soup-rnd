# Ticket 02: Moonlight-16B-A3B windows on RTX 3050 Laptop (4 GiB)

Card: NVIDIA GeForce RTX 3050 Laptop GPU, sm_86, 4096 MiB total, about 3.7 GiB free.
Model: moonshotai/Moonlight-16B-A3B, revision 2a9a177f4841649f849ba941e41d2ee73c042314
(pinned by `p72_prepare_moonlight_cache.py`; the B5 report used 476b36a4).
Software: torch 2.14.1+cu130, transformers 5.20.2, peft 0.21.2, triton 3.8.0, bitsandbytes 0.50.2.
Repo commit: a7a2a16d. Mode: normal (speed), one run per row, 4 steps, first is warm-up.
LoRA targets: q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj. Raw rows: `fit.jsonl`.

| arm | k | S | step s | alloc GiB | reserved GiB | NVML GiB | RAM GiB | status |
|---|---|---|---|---|---|---|---|---|
| C | 8 | 64 | 13.5 | 0.93 | 1.12 | 1.26 | 4.58 | ok |
| A0 | 8 | 64 | 68.5 | 0.92 | 1.06 | 1.20 | 4.59 | ok |
| C | 8 | 512 | 20.3 | 2.02 | 2.11 | 2.25 | 4.96 | ok |
| A0 | 8 | 512 | 80.8 | 1.77 | 2.07 | 2.21 | 4.96 | ok |
| C | 64 | 64 | 28.3 | 1.69 | 1.89 | 2.03 | 4.98 | ok |

Not run: (64,512), which is not in ticket 02.

Fits in 4 GiB: all five rows. The largest NVML peak is 2.25 GiB (C, k=8, S=512).
Speed: A0 is 4 to 5 times slower than C in the windows it runs. This is one run per row, so
speed is indicative only; the gate verdicts stay with the Kanana runs on Vast.
