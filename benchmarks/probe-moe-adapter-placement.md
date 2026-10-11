# E1: where LoRA has to go on a MoE model

**Status: rule written 2026-10-11, before any arm ran.** Track E1 of the research
program: Track B4 refuses routed-expert adapters on giant MoE models because of
the optimizer-state budget. This record measures what that costs in quality on
two real MoE checkpoints, and turns the result into a rule for the pre-flight
message. Nothing here is a product promise; it is quality-gated, not bit-exact
(house rule 2).

## 1. Question

At the same data, steps, rank and recipe, how much task quality and general
behaviour does an adapter lose when it skips routed experts, and does adapting
only the busiest experts recover it?

## 2. Arms

| Arm | LoRA targets | OLMoE-1B-7B | Qwen3.5-35B-A3B |
|---|---|---|---|
| `A` | attention projections only | yes | yes |
| `AS` | `A` + shared expert (`gate_proj`, `up_proj`, `down_proj`) | **N/A, no shared expert** | yes |
| `AT` | `A` (+ shared expert where it exists) + the busiest quarter of routed experts per layer | yes | yes |
| `AE` | `A` (+ shared expert where it exists) + every routed expert | yes | yes, only if it fits |

Attention means every linear projection of the token mixer: `q/k/v/o_proj`
on OLMoE; on Qwen3.5 the same four on full-attention layers and
`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` on linear-attention
layers. Routers (`mlp.gate`), the shared-expert gate, norms, embeddings and
`lm_head` are frozen in every arm.

**Traffic selection (`AT`).** The base model routes a calibration slice of the
training split (512 rows, disjoint from every evaluated row). Per layer, the
quarter of experts with the most top-k assignments (ties to the lower index)
gets adapters: 16 of 64 on OLMoE, 64 of 256 on Qwen3.5. The selected set is
written to the JSON before training starts.

## 3. Fixed recipe

- **Models.** `allenai/OLMoE-1B-7B-0125-Instruct@b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`;
  `Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754` (text tower only).
  bf16 base weights, eager attention and the `grouped_mm` expert path.
- **Data.** `meta-math/MetaMathQA@aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`,
  file `MetaMathQA-395K.json`, SHA-256
  `fb39a5d8c05c042ece92eae37dfd5ea414a5979df2bf3ad3b86411bef8205725`, MIT.
  Rows are shuffled with seed 1234; rows whose rendered prompt + response is longer
  than 512 tokens for EITHER tokenizer are dropped before splitting, so both models
  see the same rows. **At most one row per `original_question`**: MetaMathQA's
  395,000 rows are augmentations (rephrased, answer-augmented, self-verification,
  FOBAR) of 13,928 GSM8K/MATH source questions, so a row-level split would train on
  rephrasings of evaluated questions. Splits in shuffled order: 512 calibration,
  4096 train, 256 evaluation, group-disjoint; `run` and `profile` refuse a split
  that is not. Chat template of each model; Qwen3.5 with `enable_thinking=False`.
  Loss on response tokens only.
- **LoRA.** Rank 16, alpha 32 (scale 2), dropout 0, fp32 adapter weights, `A`
  uniform in ±1/sqrt(fan-in) (PEFT's Kaiming-uniform) and `B` zero. Each routed
  expert gets its own rank-16 pair on `gate_up_proj` and on `down_proj` (the shape
  PEFT's `ParamWrapper` builds; `tests/test_moe_adapter_placement.py` checks the
  shapes against peft 0.21.2 and that a zero adapter leaves the logits unchanged).
  The adapters are the harness's own modules rather than peft: one implementation
  for every arm, with no parameters on unselected experts.
- **Optimizer.** AdamW (beta 0.9/0.999, eps 1e-8, weight decay 0), batch 16 rows,
  one epoch of 4096 rows = 256 steps, linear warmup over 16 steps then constant,
  gradient clipping at 1.0, gradient checkpointing on, adapters' fp32 masters cast to
  the bf16 compute dtype in the forward.
- **Learning rate.** Swept per arm on seed 0 over {1e-4, 2e-4, 4e-4} on OLMoE; the
  lowest evaluation NLL wins. If an arm's winner is at the edge of the grid the grid
  is extended once in that direction (5e-5 or 8e-4). Seeds 1 and 2 then run at the
  winning rate. Qwen3.5 reuses each arm's OLMoE rate with seeds 0 and 1; that
  transfer is a stated limitation, not a measurement.

## 4. Measurements per run

Gate prompts are rendered exactly as `soup ship` renders them
(`live_eval._render_prompt_template`: the tokenizer's default chat template, one
user turn). On Qwen3.5 that default **opens a `<think>` block**, so its 256-token
budget may end inside the reasoning; base and every arm see the same rendering, so
the leg-2 deltas stay comparable, but the absolute Qwen3.5 suite scores describe
that rendering, not the model's best behaviour. Training and the task metric use
`enable_thinking=False`.

- **Task loss:** mean NLL per response token on the 256 evaluation rows.
- **Task accuracy:** greedy completion (384 new tokens) on the same rows, scored
  with Soup's `final_answer.parse_completion`/`answers_match` against the
  reference after `The answer is:`.
- **Gate suites:** Soup's bundled `DEFAULT_GENERAL_SUITE` (mini MMLU, common sense,
  instruction, arithmetic, tool call, JSON format, safety, over-refusal), scorer
  revision 4, greedy, 256 new tokens, through `score_bundled_suite` — the leg 2 of
  `soup ship`. The base model is scored once per model.
- **Cost:** trainable parameters; adapter persistent bytes under B4's
  `torch_adamw_fp32` profile (16 bytes per trainable fp32 parameter); peak CUDA
  memory allocated during training; wall time per step.

## 5. Decision rule

**Validity (otherwise NO VERDICT for that model).** (V1) `AE` (or the fullest arm
that fits) improves evaluation NLL over the base by at least 0.05 nat/token on every
seed: training must have learned something. (V2) No run has a non-finite loss.
(V3) The LR winner of every arm is not at the edge of the extended grid. (V4) A
zero-step adapter reproduces the base model's evaluation NLL to 1e-3 (checked in
the harness before training).

**Per arm X against the fullest arm F on the same model** (means over seeds; SD
is the sample SD over seeds):

- **Task gap** `g = NLL(X) - NLL(F)`. *Material* when `g > 0.02` nat/token AND
  `g > 2 * max(SD(X), SD(F))`.
- **Leg 2** passes when, for every suite and every seed, `base - X <= 0.05`
  (`DEFAULT_FORGETTING_THRESHOLD`).
- **ACCEPTABLE**: leg 2 passes and the task gap is not material.
  **COSTS QUALITY**: leg 2 passes and the task gap is material.
  **REGRESSES**: leg 2 fails (reported even when F also fails it).

Accuracy on the evaluation rows is reported with a bootstrap 95% interval and
does not enter the verdict: 256 items cannot resolve a few points.

**What the result changes.** If `A` or `AS` is ACCEPTABLE on a model, the
pre-flight may recommend it as a quality-preserving fallback for that architecture
class. If only `AT` is, the recommendation names the expert subset and its measured
memory. If nothing short of `AE` is ACCEPTABLE, the pre-flight message must say that
the narrower targets cost measured quality, with the number.

## 6. Hardware and fit, stated before any run

Trainable parameters per arm at rank 16, from the pinned configs [ESTIMATE; each
run JSON records the counted value]:

| Model | `A` | `AS` | `AT` | `AE` |
|---|---:|---:|---:|---:|
| OLMoE-1B-7B (64 experts, 16 layers) | 4.2M | N/A | 33.6M | 121.6M |
| Qwen3.5-35B-A3B (256 experts, 40 layers) | 16.3M | 21.2M | 251.9M | 943.9M |

AdamW fp32 adapter state is 16 bytes per trainable parameter: 0.24 / 0.32 / 3.75 /
14.07 GiB on Qwen3.5. Its bf16 base is ~65 GiB, so Qwen3.5 runs resident on one
96 GB card (`massedcompute_RTXPro6000`, USD 2.63/h) and `AE` is attempted there:
**fit** means the 256-step run completes without a CUDA out-of-memory error; a run
that does not fit is recorded as such and `AT` becomes the fullest arm. OLMoE runs
on 48 GB cards. These boxes are not the 8 GB target: E1 is a quality question, and
the memory column is the adapter budget B4's pre-flight would apply, not a claim
about an 8 GB step.

**Pilot.** One OLMoE `AE` run at 2e-4 and one Qwen3.5 `A` run first, to measure
step time and peak memory. Pilot runs are feasibility only and never enter a
verdict. If the pilot puts the full plan over the spend cap, the LR sweep shrinks
to {1e-4, 4e-4} before anything else is cut, and the change is written here before
the sweep starts.

**Correction before any counted run (2026-10-11).** The first OLMoE pilot was
launched on a row-level split from commit `8509f196`: 81 of its 253 evaluation
source questions also had a rephrasing in training, and 14 in calibration. That
split is withdrawn; the pilot it ran is feasibility-only (step time, memory, the
harness on real weights). Every counted run uses the group-disjoint split above.

## 7. Cloud protocol

As in [`probe-moe-expert-holdout.md`](probe-moe-expert-holdout.md): source bundle
from a committed revision with its SHA-256 checked on the box, an instance-side
shutdown and a workstation deadline armed before setup, output copied and its hash
verified before every delete, `brev ls` empty at the end. Spend cap for the whole
record: USD 60.
