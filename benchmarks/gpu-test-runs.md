# GPU test runs

CI has no GPU, so the CUDA-dependent tests (`pytest -m gpu`, see
[CONTRIBUTING.md](../CONTRIBUTING.md#gpu-tests)) only run when someone runs them on
a card. This file records every such run, so a regression on real hardware has a
last-known-good row to compare against. One row per card and date. Figures are
copied from the record each row links, not re-derived.

**Both recorded runs happened before the `gpu` marker existed (#833)**, so neither
is a `pytest -m gpu` run. The scope column says what was actually run.

| Date | Card (compute capability) | Driver / torch / bitsandbytes | Commit | Scope | Result | Found |
|---|---|---|---|---|---|---|
| 2026-08-06 (the session's setup date; STEP 1 itself is undated) | 8x NVIDIA H100 80GB HBM3 | 590.48.01 / 2.13.0+cu130 / 0.50.0 | not recorded | `tests/test_v07200.py`, `test_v07202.py`, `test_v07203.py`, `test_v07204.py`, `--no-cov` | `9 failed, 419 passed`; with one card visible (`CUDA_VISIBLE_DEVICES=0`): `6 failed, 422 passed` | three defects: streaming vs `nn.DataParallel` on a multi-GPU box, the #328 meta leak, a laptop-calibrated GEMM-ceiling bound ([`gate-h100-validation.md`](gate-h100-validation.md), STEP 1, FINDINGS 1–3) |
| 2026-09-10 | NVIDIA GeForce RTX 5070 Laptop GPU (sm_120), 8151 MiB | 616.92 (CUDA 13.4) / 2.14.0+cu130 / 0.50.2 | `bb0ae6b6` | full suite | `20709 passed, 2 failed, 186 skipped` | streamed NF4 not bit-exact vs resident NF4 on Blackwell: `test_issue385_stream_dtype.py` `[nf4]` in float16 and bfloat16 ([#776](https://github.com/MakazhanAlpamys/Soup/issues/776)) |
| 2026-09-18 | NVIDIA GeForce RTX 5070 Laptop GPU (sm_120), 8123 MiB | 616.92 (CUDA 13.4) / 2.14.0+cu130 / 0.50.2 | `4ae040fb` | **`pytest tests/ -m gpu`** — 88 selected, 23091 deselected; the first run of the marker itself | `1 failed, 86 passed, 2 skipped` in 1:46. Reproduced twice (106.11 s and 98.90 s), same single failure. The 2 skips are both correct: `test_rewind_mlx.py` (no `mlx` on this box) and `test_stream_multi_gpu_guard.py` (needs >1 real CUDA device, `SOUP_TEST_MULTI_GPU=1`) | one **order-dependent** failure, `test_issue974_direct_range_reads.py::TestOnRealHardware::test_every_region_is_a_pinned_sector_aligned_view_of_a_power_of_two_arena` (`data_ptr % 4096`). It **passes alone** (1 passed in 2.17 s) and fails only after the `test_issue971_async_disk_source.py` tests have run in the same process — pinned-arena pressure, in the maintainer's own streaming code. Not a regression: the same failure is recorded on bare `main` and predates the marker |
| 2026-09-24 | NVIDIA GeForce RTX 5070 Laptop GPU (sm_120), 8151 MiB | 616.92 (CUDA 13.4) / 2.14.0+cu130 / 0.50.2 | `fe657585` (branch `feat/cuda-graph-decode`, off `8736def5`) | **`pytest tests/ -m gpu`** — 93 selected, 24897 deselected; includes the two new `--cuda-graphs` tests in `test_cuda_graphs.py` | `1 failed, 90 passed, 2 skipped` in 2:14. The failure is the same order-dependent test as the row above (`test_issue974_direct_range_reads.py::...power_of_two_arena`), this time on `assert source.pinned is True`. It **passes alone** (1 passed in 0.78 s), and the branch changes no streaming code. Both `--cuda-graphs` tests pass: bitwise-equal generated IDs against eager for a tiny Llama, and for a tiny Llama carrying an unmerged fp32 LoRA over an fp16 base, with at least one compile through the backend counted (the test counts `cudagraphify_impl` calls, not recordings) | nothing new. On the same card, `soup infer --cuda-graphs` on Qwen2.5-1.5B-Instruct (full, and with a synthetic LoRA) returned responses identical to the run without the flag, greedy, 3 prompts each |
| 2026-10-11 | NVIDIA GeForce RTX 3050 Laptop GPU (sm_86), 4 GB | 595.99.02 / 2.14.1+cu130 / 0.50.2 | `8634631b` plus the uncommitted arm 9 device parameter (committed as `60f1789d`) | `tests/test_b2_deepseek_v3_stream.py` and `tests/test_b2_shard_layout.py`, `--no-cov`, fp32, deterministic algorithms. All 72 tests; 29 are the `gpu` cases | `72 passed` in 13.46 s | nothing. Power limit 35 W (default 80 W); [`gate-b2-deepseek-v3.md`](gate-b2-deepseek-v3.md) holds the raw log and the compatibility table |

| 2026-10-18 | NVIDIA GeForce RTX 3090, 24576 MiB (compute capability not captured) | 595.99.02 / 2.14.1+cu130 / 0.50.2 | `a3e3fcdd` | `pytest tests/test_c2_*.py -q --no-cov --tb=short`; two marked CUDA cases plus the C2 CPU contracts, not the full GPU selection | `73 passed` in 3.51 s, exit 0 ([raw log](evidence/c2-linux-nvme-20261018/host-e/c2-final-tests-e3.log)) | both new delayed-copy lifetime cases pass at depth 1/2. Separate expanded preparation had `277 passed, 2 failed, 1 skipped`: the existing #974 `data_ptr % 4096` assertion and root-container F1 unreaped-child fixture (`ok` versus `unknown`). Full traces and scope are in the [C2 report](c2-results-20261018.md); no full-suite, gradient-parity or strict quiet-window acceptance is implied. |

| 2026-10-28 | NVIDIA GeForce RTX 3070 (sm_86), physical 8192 MiB | 580.142 / 2.8.0+cu128 / 0.50.2 | `9d8b1d7b` plus [source manifest](evidence/b6-20261028/rtx3070/source-manifest.json) | B1 frozen-expert and B6 tiny-model tests, `-m gpu`; internal installation guard bypass only in the tiny test | `9 passed, 46 deselected` | [raw log](evidence/b6-20261028/rtx3070/pytest.log); synthetic correctness only |
| 2026-10-28 | NVIDIA GeForce RTX 5060 (sm_120), physical 8151 MiB | 610.57.04 / 2.8.0+cu128 / 0.50.2 | `9d8b1d7b` plus [source manifest](evidence/b6-20261028/rtx5060/source-manifest.json) | B1 frozen-expert and B6 tiny-model tests, `-m gpu`; validated runtime installation without a bypass | `9 passed, 46 deselected` | [raw log](evidence/b6-20261028/rtx5060/pytest.log); real giant pilot recorded separately in [B6](b6-giant-moe-20261028.md) |

## What a green run here does and does not mean

**2026-09-18, the first `-m gpu` run.** Three things that are only visible once the marker
exists, recorded so the next reader does not over-read a green row:

1. **The selection is 88 tests across 18 files**, and its composition is not obvious from the
   names. The largest contributors are `test_v07204.py` (14), `test_issue971_async_disk_source.py`
   (11), `test_v07203.py` (10) and `test_issue968_bnb_nf4_path_divergence.py` (9) — i.e. layer
   streaming and the bitsandbytes 4-bit paths are most of what a card actually exercises here.
2. **A green `-m gpu` run does NOT mean [#776](https://github.com/MakazhanAlpamys/Soup/issues/776) is resolved.** The opposite: the
   nine tests in `test_issue968_bnb_nf4_path_divergence.py` PASS by asserting the divergence
   between the two bitsandbytes 4-bit compute paths is still non-zero and still within an order of
   magnitude of the recorded value. They are written to fail if the paths ever agree, which would
   be the signal to re-measure #776 — so "86 passed" includes nine tests whose passing is evidence
   that an open defect is still present.
3. **`-m gpu` runs in under two minutes**, against ~30 minutes for the full suite. That is the
   argument for running it on every card that touches this project: it is cheap, and CI can never
   run it — GitHub's runners have no GPU, so all 88 skip in all nine cells.

## Adding a run

Run `pytest tests/ -m gpu --no-cov -v` on a clean checkout, then send the details
listed in [CONTRIBUTING.md](../CONTRIBUTING.md#gpu-tests). A failure is as useful
as a pass: file it, and link the issue in the **Found** column.
