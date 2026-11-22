# Ticket 04: bits ladder at k=64 on Moonlight-16B-A3B, RTX 3050 Laptop (sm_86, 4 GiB)

## Setup

- Model moonshotai/Moonlight-16B-A3B, revision `2a9a177f4841649f849ba941e41d2ee73c042314`,
  per-expert NF4 cache from ticket 02.
- Card NVIDIA GeForce RTX 3050 Laptop GPU, sm_86, 4096 MiB (CUDA reports 3.68 GiB usable).
- Torch 2.14.1+cu130, CUDA 13.0, venv `~/.venvs/soup-rnd-cuda`.
- Mode bits, deterministic (`use_deterministic_algorithms(True, warn_only=True)`), 3 steps, loss and
  all 216 LoRA gradients from the last step. Default SDPA backend, like the S=64 ladder of ticket 03.
- Harness commit `6d84408f` (the ladder reference change, before the runs); repo commit of the runs
  `e8a2965e`. LoRA targets `q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj`, rank 16.
- Runner: `run_bits_k64.sh`. Rows: `k64-s64/`, `k64-s512/` (`rows.jsonl`, `bits-verdict.json`,
  `summary.md`, `sweep.log`). Raw tensors stay outside git in `~/moonlight-local/results/bits/`.

## The ladder reference at k > 8

A0 does not run above k=8, so the bits ladder now takes C as the reference there
(`mbr_measure.reference_arm`: A0 at k <= 8, C at k > 8). `--ladder-ks` and `--arms` select the k
values and the arms, so the two k=64 windows run C and E only. At k <= 8 the reference and the plan
are unchanged.

## Window (64,64): C and E

Each line names model, card, window, arm, mode and torch version.

- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (64,64), arm C, mode bits
  (deterministic), torch 2.14.1+cu130: ok, loss 2.892196, allocated 1773.4 MiB, reserved 1976.0 MiB,
  NVML 2170.0 MiB. Native path, so the `lt-fused` counters are not installed and the row carries no
  `fused_visits`.
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (64,64), arm E, mode bits
  (deterministic), torch 2.14.1+cu130: ok, loss 2.890608, allocated 1917.9 MiB, reserved 2150.0 MiB,
  NVML 2346.0 MiB, `fused_visits` `python_forward` = 0, `python_backward` = 0,
  `fused_forward` = 4788, `fused_backward` = 4788.
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (64,64), arm C, mode bits
  (deterministic), torch 2.14.1+cu130, repeat of C: ok, loss 2.892196, allocated 1773.4 MiB,
  reserved 1976.0 MiB, NVML 2170.0 MiB.

Verdict by `torch.equal` on the saved tensors (`repeats/compare-k64-s64-C-E.txt`):

| pair | loss equal | grads differing | max abs diff | max abs grad |
|---|---|---|---|---|
| C vs C repeat | yes | 0 of 216 | 0 | 0.415425 |
| C vs E | no | 108 of 216 | 0.027930 | 0.415425 |
| C repeat vs E | no | 108 of 216 | 0.027930 | 0.415425 |

Both arms are stable: C reproduces itself in all 216 gradients, and E is bit-identical to the
ticket 03 counter run at this window (`counter-k64/rows.jsonl`: same digest `5e81e2b2`, loss
2.8906077202409506, `fused_forward` 4788, same peaks). The C/E difference is therefore a real arm
difference, not run-to-run noise: at k=64 the `lt-fused` arm does not match the ordered native arm.
A0 cannot judge the two here. Up to k=8 E matched A0 at every k while C differed at k=2, 4 and 8, so
E is the arm that kept A0's bits where A0 could run; at k=64 C is the arm that differs.

## Window (64,512): OOM in both arms

- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (64,512), arm C, mode bits
  (deterministic), torch 2.14.1+cu130: OOM, allocated 3533.2 MiB, reserved 3590.0 MiB, NVML
  3758.0 MiB. Error as stored (first 200 characters): `CUDA out of memory. Tried to allocate 20.00
  MiB. GPU 0 has a total capacity of 3.68 GiB of which 14.38 MiB is free. Including non-PyTorch
  memory, this process has 3.64 GiB memory in use. Of the alloca`
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (64,512), arm E, mode bits
  (deterministic), torch 2.14.1+cu130: OOM, allocated 3015.1 MiB, reserved 3576.0 MiB, NVML
  3584.0 MiB. Error as stored (first 200 characters): `CUDA out of memory. Tried to allocate 126.00
  MiB. GPU 0 has a total capacity of 3.68 GiB of which 62.38 MiB is free. Including non-PyTorch
  memory, this process has 3.60 GiB memory in use. Of the alloc`
- No verdict for this window: C has no ok reference row, so the verdict file says `no C`. The OOM
  was not forced or retried; the row is the result.

## Fused visits

- E at (64,64): `python_forward` = 0, `python_backward` = 0, `fused_forward` = 4788,
  `fused_backward` = 4788. The Python-path hypothesis stays rejected at k=64.
- C at (64,64): native expert GEMM, no `lt-fused` plan, so no `fused_visits` counters exist for it.

## Not in this run

- F and G at k=64. Ticket 04 asks for C and E only.
- G, F and B0 at k=64; the ladder plan can run them, but no rows were taken.
- The S=512 window with the math backend. It already OOMs on the default backend.
- Gate D1. One block, one local card, preliminary verdicts. Kanana on Vast is separate.

## Files

- `k64-s64/`: C and E rows, verdict, arm records, summary and sweep log.
- `k64-s512/`: the two OOM rows and their verdict (`no C`).
- `repeats/compare-k64-s64-C-E.txt`, `repeats/C-k64-rep1-s64.jsonl`, `repeats/run-k64.log`:
  the C repeat and the run log.
