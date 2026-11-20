# Kanana-2-30B-A3B on RTX 4060 Ti: baseline of A0, B0 and C (mbr-unified ticket 01)

Measured 2026-11-20 on a rented Vast instance (RTX 4060 Ti **16 GiB**, i7-11700K, 62 GB RAM, overlayfs disk). The ticket's 8 GiB card is emulated with the allocator limit `torch.cuda.set_per_process_memory_fraction` (8 GiB, `vram_limit_gib: 8` in every row). NVML peaks include the CUDA context, which the limit does not cover.

Raw rows are outside git (`rows.jsonl` per stage). Harness: `benchmarks/harness/mbr_measure.py`, `mbr_measure_kanana_kaggle.py` (entry), `mbr_measure_vast.sh` (remote script).

## What the arms are on this branch

- **A0** native L2L step.
- **B0** route-preserving MBR (`route_preserving_mbr=True`, group size = k).
- **C** B0 plus `mbr_ordered_backward`. The exact GEMM (`a0_exact_lt`) is not in `src` yet (ticket 02), so C here is **not** the final exact MBR. No FP8 codec, no Oracle (tickets 06, 08).

## Setup and limits

- Data: MBPP-train text (`google-research-datasets/mbpp`, full, train, 374 rows), padded to S, k rows per window.
- Speed: normal mode, 3 steps per process, median of steps after the first (warm-up). The run is **one A-B-C-C-B-A block** (two runs per arm and window), not five: the user stopped after block 0 to save time and money. No t-test is possible with one block; repeat runs inside the block are listed below. The D1 gate (>=3% and t>2.78 at 5 blocks) is not evaluated here; each lever gets its own A/B on this stand.
- Bits: deterministic mode (`use_deterministic_algorithms(True, warn_only=True)`), 3 steps, S=64.
- Software: torch {env['versions']['torch']}, triton {env['versions']['triton']}, transformers {env['versions']['transformers']}, peft {env['versions']['peft']}, bitsandbytes {env['versions']['bitsandbytes']}, CUDA toolkit 12.6 (ptxas). Checkpoint `kakaocorp/kanana-2-30b-a3b-base-2601` @ `0d31365a`, NF4 per-expert cache, LoRA r=16.
- `direct_io`: requested and actual (no fallback) in every row, although the disk is overlayfs.

## Speed, VRAM, RAM (normal mode, 8 GiB allocator limit)

| arm | k | S | step s | alloc GiB | reserved GiB | NVML GiB | RAM GiB | runs | status |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 8 | 64 | 169.6 | 0.78 | 0.87 | 1.24 | 6.75 | 2 | ok |
| B0 | 8 | 64 | 33.9 | 1.10 | 1.16 | 1.53 | 6.76 | 2 | ok |
| C | 8 | 64 | 32.9 | 1.10 | 1.16 | 1.53 | 6.75 | 2 | ok |
| A0 | 8 | 512 | 193.3 | 1.44 | 1.68 | 2.06 | 7.41 | 2 | ok |
| B0 | 8 | 512 | 35.2 | 2.03 | 2.12 | 2.50 | 7.41 | 2 | ok |
| C | 8 | 512 | 35.4 | 2.03 | 2.12 | 2.50 | 7.41 | 2 | ok |
| B0 | 64 | 64 | 55.1 | 3.72 | 4.03 | 4.40 | 7.45 | 2 | ok |
| C | 64 | 64 | 54.2 | 3.72 | 4.03 | 4.40 | 7.45 | 2 | ok |
| B0 | 64 | 512 | - | 7.98 | 8.00 | 5.70 | 9.92 | 2 | OOM |
| C | 64 | 512 | - | 7.98 | 8.00 | 7.69 | 9.92 | 2 | OOM |

Peaks are the max over the runs of a cell; RAM is the process peak RSS. Step times of the two runs of each cell (s):

| cell | run 1 | run 2 |
|---|---|---|
| A0 k=8 S=64 | 169.6 | 169.6 |
| B0 k=8 S=64 | 35.0 | 32.8 |
| C k=8 S=64 | 32.9 | 32.9 |
| A0 k=8 S=512 | 193.1 | 193.4 |
| B0 k=8 S=512 | 35.2 | 35.2 |
| C k=8 S=512 | 35.3 | 35.4 |
| B0 k=64 S=64 | 55.9 | 54.3 |
| C k=64 S=64 | 54.3 | 54.2 |

Speedup against A0 (k=8): B0 and C are about **5.0-5.2x** faster at S=64 (169.6 s against 33.9 and 32.9 s) and **5.5x** at S=512 (193.3 against 35.2 and 35.4 s). C against B0 differs by 1-3% at S=64 and by 0.6% at S=512 inside one block, so no C-over-B0 gain is shown.

**Window (64,512) is OOM for B0 and C at 8 GiB.** The four OOM rows record the allocator peak (7.98 GiB allocated, 8.00 reserved) and RAM peak 9.9 GiB; the NVML peak at the moment of the error varies from 4.2 to 8.3 GiB because the failure hits at different points. This is the case the large-window mode (D2) is for. A0 does not exist at k=64.

## Bit ladder (deterministic, S=64, vs A0)

| k | B0 | C | loss A0 | loss B0 | loss C |
|---|---|---|---|---|---|
| 1 | equal | equal | 1.940557 | 1.940557 | 1.940557 |
| 2 | differs | differs | 2.327247 | 2.340329 | 2.340329 |
| 4 | differs | differs | 3.433213 | 3.469577 | 3.469577 |
| 8 | differs | differs | 3.256432 | 3.258760 | 3.258760 |

On sm_89 the ladder is bit-equal to A0 only at k=1. From k=2 both B0 and C differ from A0 in losses and gradient digests (loss differs by 0.56% at k=2, 1.06% at k=4 and 0.07% at k=8). C equals B0 at k=2; at k=4 and k=8 the losses match and the gradient digests differ. The A0 digest at k=8 is identical between the probe, the ladder and the speed runs, so the stand is reproducible. This answers the open question about sm_89 in `decisions.md`: the exact GEMM of ticket 02 is needed for A0 bits, the Moonlight/sm_86 result does not carry over to the present B0 and C.

## Phase profile (one instrumented step, CUDA-synchronised; shares of summed phase time)

Instrumented seconds are longer than real steps, only the shares count. Top three phases per run:

| arm | k | S | top phases |
|---|---|---|---|
| A0 | 8 | 64 | expert_disk_read 57%, expert_load_h2d_and_misc 11%, expert_host_pack 8% |
| B0 | 8 | 64 | expert_disk_read 54%, expert_load_h2d_and_misc 10%, expert_host_pack 7% |
| C | 8 | 64 | expert_disk_read 54%, expert_load_h2d_and_misc 10%, expert_host_pack 7% |
| A0 | 8 | 512 | expert_disk_read 57%, expert_load_h2d_and_misc 11%, expert_host_pack 8% |
| B0 | 8 | 512 | expert_disk_read 51%, expert_load_h2d_and_misc 10%, bwd/autograd_backward 9% |
| C | 8 | 512 | expert_disk_read 54%, expert_load_h2d_and_misc 9%, bwd/autograd_backward 8% |
| B0 | 64 | 64 | expert_disk_read 36%, bwd/autograd_backward 14%, bwd/attn_router_shared_other 13% |
| C | 64 | 64 | expert_disk_read 36%, bwd/autograd_backward 14%, bwd/attn_router_shared_other 13% |
| B0 | 64 | 512 | OOM |
| C | 64 | 512 | OOM |

Expert reads from disk take 51-57% of the step at k=8 and 36% at k=64. Amdahl threshold for +10% overall from a 2x local speedup is 18.2% of the path (27.3% at 1.5x). No GEMM phase reaches it (expert backward GEMM is about 6%), so an FP8 compute GEMM on sm_89 (external report r6) is not supported by this profile.
