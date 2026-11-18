# ZERO-2 stage 3 protocol: which Qwen3.5-9B layers to train, on four datasets (fixed 2026-11-19, before the run)

Author: Nurseiit. The question of the task: which layers of Qwen3.5-9B matter, can an algorithm
pick the ones to train for a given dataset, and what does training only them cost and save in
the 8 GiB and 4 GiB budgets. Stages 2 and 2b (results in this directory) fixed the setups used here.

## Code under test

Soup `research/zero2-fixes` at `b3bca25cda116bf843b4e76aa59f228e4cbbc0ed`: stage 2b's code plus
`ef0516a5` (`embed_on_host` keeps the weight on the host after `model.to()`) and `b3bca25c`
(with `lora_layers`, the input-grad hook is removed so the resident backward stops at the lowest
trainable layer, as L2L's does; recorded as `backward_cut` in `lora_layers.json`). Liger Kernel
0.8.5 (fused linear CE, RMSNorm, SwiGLU for qwen3_5) through Soup's `use_liger`.

## Fixed setup

| Item | Value |
| --- | --- |
| Machine | Vast, one RTX 4090 (24 GB), >= 64 GB RAM, >= 150 GB disk; recorded in `env.json` |
| Model | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, NF4 |
| Soup config, all arms | sft, alpaca, max_length 1024, lr 1e-4, batch 1, k = 4, seed 0, gradient checkpointing, LoRA r 16 / alpha 32 / dropout 0 on all 10 projection types of every trained layer |
| Layer methods (`training.lora_layers`) | full (all 32), top8 (fixed 24–31), random (k 8, seed 0), grad (k 8, 64 probe rows), angular (k 8, 64 probe rows), lisa (k 8, re-drawn every 20 steps, seed 0) |
| Step time | `perf_counter` at each step end; s/step = mean of the intervals after step 3 |
| Runner | `zero2_stage3.py`, sha256 `a3d8ae92f75fcc7c87f53e1fd1dedebbd5069d5199f962d89b086375492a59f4`, via `run_zero2_stage3.sh`; resumable (an arm with a json is never run again) |

### Part S: speed and fit (GSM8K train, first 96 rows, 24 steps per arm, fixed order)

| Block | Arms | Question |
| --- | --- | --- |
| h8 | 8 GiB cap, plain, `kbit_keep_16bit` + `embed_on_host` | does plain QLoRA fit 8 GiB now |
| r | rF rT rT rF: no cap, plain keep16; F all layers, T top8 | resident speed-up of training the top 8 |
| l | lK lL lL lK: no cap, plain keep16, all layers; L with `use_liger` | Liger's kernels on Qwen3.5 |
| s | sF sT sT sF: 8 GiB, stream RAM + chunked CE; F vs T | the same in the 8 GiB budget |
| q | qF qT qT qF: 4 GiB, stream RAM + L2L + chunked CE; F vs T | the same in the 4 GiB budget |

### Part Q: quality (no cap, plain, `kbit_keep_16bit`, 300 steps = 1 200 rows per arm)

| Dataset | Train rows (first 1 200, file order) | Eval (lm-eval 0.4.13, NF4 model loaded by the runner) | Metric |
| --- | --- | --- | --- |
| gsm8k | `openai/gsm8k` @ `740312ad…` main train: question → worked answer | `gsm8k` (5-shot default), first 200 test questions | exact_match, strict-match |
| arc | `allenai/ai2_arc` @ `210d026f…` Challenge then Easy train: question → correct choice text | `arc_challenge`, all 1 172 test questions | acc_norm |
| pubmedqa | `qiaojin/PubMedQA` @ `9001f285…` pqa_artificial: "Abstract: … Question: …" → yes/no | the runner's `zero2_pubmedqa` task: pqa_labeled (1 000, disjoint from pqa_artificial), yes/no/maybe | acc |
| code | `sahil2801/CodeAlpaca-20k` @ `152bb5e9…` | `humaneval` (164, greedy pass@1; model code runs on the rented instance) | pass@1 |

Per dataset: the base model's eval, then each method trained and evaluated, in the order of the
methods table. Expected wall time ~8–9 h (~45 min part S, ~2 h per dataset), about $4.5 at
$0.51/h. Cap for the stage: $8; the instance is destroyed when the run ends.

## What is decided

1. `fits_8gib_plain_host`: h8 runs 24 steps without error or OOM.
2. Speed in each A-B-B-A block (r, l, s, q): **FASTER** if both B arms beat both A arms, **SLOWER**
   if both lose, else **NO_CLEAR_DIFFERENCE**; speed-up = (A1 + A2) / (B1 + B2).
3. Quality, per dataset and method: delta = score(method) − score(full), se = sqrt(se_method² +
   se_full²) from lm-eval's stderrs. **SAME** if |delta| <= 2 se, **WORSE** if delta < −2 se,
   **BETTER** if delta > 2 se.

Reported, not gated: base vs full per dataset, every arm's s/step, tok/s and peaks, the layers
each method chose (and grad / angular scores) per dataset, the `backward_cut` record, losses.
Speeds in part Q are one process each (descriptive); the layer methods' speed depends on the lowest
chosen layer, which is reported with it. One seed and 300 steps per arm: a SAME verdict means
"no difference this run can see", not "equal".

## How the result is used

The trade-off table for the lead (per dataset: method, score vs full, speed-up in each budget),
whether a data-driven pick (grad / angular) beats a fixed or random one, whether Liger goes on in
Soup's Qwen3.5 recipes, and which k and methods stage 4 checks further (k 4 / 16, more seeds).
A failed arm is reported with its error and not rerun under this protocol.

## Not tested

k other than 8, seeds other than 0, other learning rates, full fine-tuning, two GPUs, the Triton
NF4 kernel (off: no gain in stage 2b), evals beyond the four listed.
