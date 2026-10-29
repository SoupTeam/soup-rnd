# G1 ticket 05: Mistral-7B, LoRA over a frozen NVFP4 base against NF4, result

**Quality verdict: fail.** Task loss passes with a wide margin. Two of the four gate suites fail the rule's drop limit. Under the rule's outcome table G1 stops and speed is not measured.

**Naming.** This gate is G1. It was called G4 when the rule was committed, so the run records under `benchmarks/results/g1-mistral/` and commit messages up to `471c09ca` still say G4 or `g4-`; raw records were not edited. The rule text (`benchmarks/gate-g1-nvfp4.md`) was relabeled G1 in a later commit: names and the `--follow` in its commit-order command only, no threshold, data or arm changed.

All numbers on this page are **real** runs: Mistral-7B-v0.1, the Dolly split of the rule, one RTX 5090 rented on Vast.ai. The rule is `benchmarks/gate-g1-nvfp4.md` (Russian text binding), committed at `4fddde39` before any run. Thresholds, data and arms did not change after the first run. The verdict is computed by `benchmarks/harness/g1_verdict.py` from the run records in `benchmarks/results/g1-mistral/runs/`, output in `benchmarks/results/g1-mistral/verdict.json`.

## Task loss

Task loss is the token-weighted NLL on the 512-row held-out panel (41,527 response tokens). The ratio is NVFP4 over NF4 with the same seed. The interval is the 95% paired bootstrap over panel examples (2000 resamples, seed 17), so it describes sampling spread of the panel, not spread of training.

| Seed | NF4 | NVFP4 | Ratio | Interval |
|---|---|---|---|---|
| 42 | 1.3984 | 1.4136 | 1.0109 | 1.0088 to 1.0131 |
| 7 | 1.3957 | 1.4138 | 1.0130 | 1.0108 to 1.0154 |
| 123 | 1.3972 | 1.4132 | 1.0115 | 1.0093 to 1.0139 |

The mean ratio is 1.0118 (limit 1.03). The highest upper bound is 1.0154 (limit 1.05). bf16 reference, seed 42: 1.3883. NVFP4 loses about 1.2% of task loss against NF4 and sits about 1.8% above bf16 at this scale. Exact parity with NF4 is not claimed.

## Gate suites

Correct answers per run. The rule fails a gating suite when NVFP4 answers on average 3 or more questions fewer than NF4.

| Suite (questions) | NF4 seeds 42 / 7 / 123 | NF4 mean | NVFP4 seeds 42 / 7 / 123 | NVFP4 mean | Mean drop | Result |
|---|---|---|---|---|---|---|
| mini_mmlu (26) | 12 / 7 / 4 | 7.67 | 8 / 1 / 3 | 4.00 | 3.67 | **fails** |
| mini_common_sense (24) | 14 / 3 / 1 | 6.00 | 2 / 0 / 2 | 1.33 | 4.67 | **fails** |
| mini_instruction (24) | 23 / 23 / 23 | 23.00 | 23 / 23 / 23 | 23.00 | 0.00 | passes |
| mini_arithmetic (40) | 28 / 25 / 29 | 27.33 | 27 / 26 / 28 | 27.00 | 0.33 | passes |

The bf16 reference scored 1, 0, 23 and 28 on the four suites. No suite is excluded as all-or-nothing (NF4 means are neither 0 nor the full count).

## Validity rows

All eight checks of `g1_verdict.py` hold: the six rows of the rule, complete gate suites and the commit order. No run was spoiled and none was repeated. A wrong storage format in any attempt gives no verdict, since the rule allows no repeat for row 4.

| Row | Result |
|---|---|
| Fingerprints of embeddings, norms and `lm_head` match in every arm | holds |
| Train split and panel hashes match (`a915bbbf…`, `b4b2ab93…`) | holds |
| Configs differ only in `base.format` (and the seed) | holds |
| 224 layers in the arm's format, none refused | holds (NVFP4Tensor, Params4bit, Parameter) |
| Every run reached step 250 with finite loss and grad norm | holds |
| First-step grad norm above 0 | holds (1.97 to 4.00) |
| Every record holds exactly the four gate suites, with their question counts and integer scores from 0 to n | holds |
| Rule commit earlier than every run's `started` | holds |

## What changed during the runs

- The first attempt used the harness default micro-batch of 4. NVFP4 ran out of GPU memory on the first forward pass (26.4 GiB allocated of 31.4 GiB). The rule fixes the effective batch (16) and not the micro-batch, and the ticket 04 report left the micro-batch for ticket 05. A micro-batch differing between arms would break the config row, and NVFP4 scales activations per tensor, so the micro-batch changes the numbers. All seven runs were therefore redone with `--micro-batch 2` (gradient accumulation 8). The two completed runs with micro-batch 4 (NF4 seeds 42 and 7) are kept in `benchmarks/results/g1-mistral/superseded-mb4/` and are not used.
- A pilot NF4 run (seed 42, micro-batch 4, about 10 minutes) showed the train split stays at 4000 rows: far below the rule's 2 hour limit. Pilot records are in `benchmarks/results/g1-mistral/pilot/`. A killed NVFP4 probe with micro-batch 2 is in the same folder; it never reached the end and is not a run.
- TransformerEngine imports only with the venv's NCCL preloaded (`LD_PRELOAD=…/nvidia/nccl/lib/libnccl.so.2`). The system `libnccl` 2.28.3 lacks `ncclCommResume`.

## Limits

- The gate suites are small and move a lot between seeds. NF4 alone ranges from 12 to 4 on mini_mmlu and from 14 to 1 on mini_common_sense. The rule averages three seeds and the verdict follows it. This page does not claim the drop is caused by FP4 itself and not by that spread.
- One model, one temporary data set, one seed per pair, three pairs.
- Decoding is greedy, 64 new tokens, no KV cache, padded to 16, the same in every arm.
- Step time and memory were not measured under `benchmarks/METHOD.md`. Ticket 06 would measure them only after a pass.

## Machine and versions

Vast.ai instance 57257627 (team soup-rnd), RTX 5090, UUID `GPU-4ec93557-ed58-e677-d3dd-52446030f53b`, driver 595.84, Python 3.12.15, torch 2.14.1+cu130, transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2, TransformerEngine 2.20.2, datasets 5.1.0. Repo commit `45a02451`, harness SHA-256 `525834570c32750d61c16b273edafcf6f8ee5ce07ae5f00988a7bd90c220db25`. Each `run.json` also holds the full config, per-step loss and grad norm, per-example panel NLL and gate suite outputs.
