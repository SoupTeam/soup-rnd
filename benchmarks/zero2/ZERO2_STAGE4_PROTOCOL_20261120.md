# ZERO-2 stage 4 protocol: the trade-off map around full LoRA on Qwen3.5-9B (fixed 2026-11-20, before the run)

Author: Nurseiit. Stage 3 (`ZERO2_STAGE3_RESULT_20261119.md`) gave the two ends: all 32 layers
(reference) and the top 8 (2.2x faster, −25 points on GSM8K, −10.5 on ARC); every k-8 method that
kept quality trained a low layer and saved no time. This stage maps the points between and on
the other side: faster variants that should cost little quality, and slower ones that might
beat the reference, so a user can pick a point.

## Code under test

Soup `research/zero2-fixes` at `b3bca25cda116bf843b4e76aa59f228e4cbbc0ed` (the stage-3 code; no new
Soup code). Existing features used: `training.lora_layers` (fixed), `lora.r` / `lora.alpha`,
`lora.rank_pattern` / `lora.alpha_pattern`, `lora.use_dora`, `training.kbit_keep_16bit`. Checked
before the run on a tiny model through Soup's `build_lora_config`: rank_pattern gives layers 0–1
rank 16 and the rest rank 4 as configured, with the same alpha/r scaling; DoRA adds a magnitude
vector per adapted module.

## Fixed setup

| Item | Value |
| --- | --- |
| Machine | Vast, one RTX 4090 (24 GB), >= 64 GB RAM, >= 150 GB disk; recorded in `env.json` |
| Model | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, NF4, `kbit_keep_16bit` |
| All arms | sft, alpaca, max_length 1024, lr 1e-4, batch 1, k = 4, gradient checkpointing, dropout 0, LoRA on the 10 projection types, no VRAM cap unless stated |
| Runner | `zero2_stage4.py`, sha256 `7762245a21583bf65516d6e6942e7e71dc7ed18571dd907ee60082b72c45d9ec`, via `run_zero2_stage4.sh`; resumable |

| Variant | Layers trained | LoRA | Side |
| --- | --- | --- | --- |
| full | 0–31 | r 16, alpha 32 | reference |
| top24 | 8–31 | r 16 | faster |
| top16 | 16–31 | r 16 | faster |
| top12 | 20–31 | r 16 | faster |
| r32 | 0–31 | r 32, alpha 64 | more capacity |
| imp | 0–31 | layers 0–7 r 64 / alpha 128, the rest r 16 / alpha 32 | more capacity where stage 3's grad pointed (layers 0–7 on all datasets) |
| dora | 0–31 | r 16, DoRA | more capacity |

### Part S: speed (GSM8K train, first 96 rows, 24 steps, seed 0)

A-B-B-A against full, one block per variant, in this order: top24, top16, top12, r32, imp, dora
(no cap); top16 at 8 GiB (stream RAM + chunked CE) and at 4 GiB (stream RAM + L2L + chunked CE),
each against full in the same setup.

### Part Q: quality (300 steps = 1 200 rows per arm)

| Dataset | Train rows | Eval | Metric | Seeds |
| --- | --- | --- | --- | --- |
| gsm8k | as stage 3 | `gsm8k`, first 200 test questions, batch 16 | strict EM | 0, 1 |
| arc | as stage 3 | `arc_challenge`, 1 172, batch 16 | acc_norm | 0, 1 |
| pubmedqa | as stage 3 | `zero2_pubmedqa`, 1 000, **batch 2** (stage 3 OOM'd at 16) | acc | 0 |

Order: dataset by dataset; the base model's eval, then for each seed every variant in the table
order, trained and evaluated. HumanEval is dropped (stage 3: fine-tuning on CodeAlpaca did not
improve it even with all layers, so it cannot separate variants). Expected ~10–11 h, about $5;
cap $9; the instance is destroyed when the run ends.

## What is decided

1. Speed per part-S block: **FASTER** / **SLOWER** if both variant arms beat / lose to both full
   arms, else **NO_CLEAR_DIFFERENCE**; speed-up = (F1 + F2) / (V1 + V2).
2. Quality per dataset and variant: delta = mean over seeds of the variant's score − mean of
   full's; se = sqrt(se_v² + se_full²) with each se the root mean square of that variant's lm-eval
   stderrs over its seeds. **BETTER** if delta > 2 se and every seed's paired delta is positive;
   **WORSE** if delta < −2 se and every seed's is negative; else **SAME**.

Reported, not gated: per-seed scores and deltas, every arm's s/step, tok/s and peaks, base vs
full. The trade-off table pairs each variant's quality verdict with its part-S speed-up.

## How the result is used

The map for the lead: per dataset, which variants sit at "faster, little loss" and which at
"slower, better", with numbers; the default to recommend in Soup's Qwen3.5 recipes for each side.
A failed arm is reported with its error and not rerun under this protocol.

## Not tested

k-8 methods again, learning rates other than 1e-4, longer training, other models, HumanEval.
