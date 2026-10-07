<!--
Working measurement record. §2, the decision rule, was written and committed
before any engine run; results are appended below it as they arrive and the
rule is not edited afterwards.

Box for Part A: Windows 11 Pro 26200, AMD Ryzen 5 8645HS (6 cores / 12 threads),
15.3 GB RAM, CPU only. The box has an RTX 4050 Laptop GPU; this probe does not
use it.
-->

# Probe — does a MoE serving engine apply a Soup LoRA adapter to MLA attention and the shared expert?

**Status: Part A measured 2026-10-08, three runs. Run 3 gives the verdict
HOOK NEEDED: llama.cpp b11476 silently drops a LoRA on `q_a_proj`, `q_b_proj`
and `kv_a_proj_with_mqa`, and cannot convert one on `kv_b_proj`; `o_proj` and
the shared expert are applied exactly. The positive control (Qwen3.5-shaped)
passed. Runs 1 and 2 were void or without verdict, for instrument reasons
recorded in §3, and are kept. Run 2 also found that PEFT, on Soup's pinned
stack, cannot build the shared-expert adapter at all (§3.2). Part A', a draft
hook of 18 added and 13 removed lines, makes every one of those modules APPLIED
on the same tiny models: SUFFICIENT under its own rule (§4a-§4b).**

This record answers one question behind the serving-scope decision in
[`scope-moe-serving.md`](scope-moe-serving.md): if an external engine serves the
giant MoE and Soup only trains, how much has to be added to that engine so that
a Soup adapter works there, unmerged?

---

## 0. The question, and why it decides the serving scope

For a giant MoE (DeepSeek-V3 671B, Kimi K2 ~1T) Soup adapts the attention
projections and, optionally, the shared expert; the routed experts stay frozen.
Soup's `target_modules: auto` policy for `deepseek_v3`, `kimi_k2` and
`kimi_k25` is exactly the five MLA projections `q_a_proj`, `q_b_proj`,
`kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`
([`peft_wiring.py:81-87,126,139,144`](../src/soup_cli/utils/peft_wiring.py)).

Merging the adapter into the base is not a way around the engine. `soup export`
merges through a full fp16 load of the base on the CPU
([`export.py:409-421`](../src/soup_cli/commands/export.py)), about 1.3 TB for
671B parameters. So the engine has to load the adapter as an adapter.

Reading llama.cpp at tag `b11476` (commit `988190680d5a`) says it would not,
for the DeepSeek-V3 architecture:

- the MLA query and compressed-KV projections are plain `ggml_mul_mat` calls,
  not the LoRA-aware `build_lora_mm`:
  [`deepseek2.cpp` L502, L508, L518](https://github.com/ggml-org/llama.cpp/blob/b11476/src/models/deepseek2.cpp#L496-L520);
- the absorbed key projection `wk_b` is a plain `ggml_mul_mat`
  ([L561](https://github.com/ggml-org/llama.cpp/blob/b11476/src/models/deepseek2.cpp#L556-L566)),
  and so is the value projection `wv_b` inside the attention helper
  ([`llama-graph.cpp` L2744-L2754, L2811-L2812](https://github.com/ggml-org/llama.cpp/blob/b11476/src/llama-graph.cpp#L2744-L2812));
- the converter splits `kv_b_proj` into `k_b` and `v_b` and transposes `k_b`
  ([`conversion/deepseek.py` L434-L449](https://github.com/ggml-org/llama.cpp/blob/b11476/conversion/deepseek.py#L434-L449)),
  and the adapter converter's tensor wrapper supports that transpose only in 2-D
  ([`convert_lora_to_gguf.py` L168-L179](https://github.com/ggml-org/llama.cpp/blob/b11476/convert_lora_to_gguf.py#L168-L179));
- `o_proj` goes through `build_lora_mm` in the attention output helper, and the
  shared expert through `build_ffn`, which uses it
  ([`llama-graph.cpp` L1837-L1977, L2896](https://github.com/ggml-org/llama.cpp/blob/b11476/src/llama-graph.cpp#L1837-L1977)).

Code reading is not execution, and a dropped adapter is silent: the engine loads
it and answers. This probe runs the path end to end on tiny models and names,
tensor by tensor, what a hook has to add.

## 1. The instrument

**Models — SYNTHETIC weights.** Random initialisation with a fixed seed; the
shapes follow the real architectures, scaled down. No result here is a
statement about a trained model.

| label | transformers class | layers | attention | experts | vocabulary / tokenizer |
|---|---|---|---|---|---|
| `dsv3-tiny` | `DeepseekV3ForCausalLM` | 3 (layer 0 dense, 1-2 MoE) | MLA, 4 heads, `q_lora_rank` 32, `kv_lora_rank` 16, `qk_nope` 16, `qk_rope` 8, `v_head` 16 | 8 routed, top-2, 1 shared; sigmoid scoring, `n_group` 1 as in Kimi K2 | 129,280, tokenizer of `deepseek-ai/DeepSeek-V3@e815299b` |
| `qwen35moe-tiny` | `Qwen3_5MoeForCausalLM` | 4 (3 Gated DeltaNet, 1 full attention) | full: 4 heads, 2 KV heads, head dim 64, output gate | 8 routed, top-2, 1 shared with gate | 248,320, tokenizer of `Qwen/Qwen3.5-35B-A3B@59d61f3c` |

Deliberate deviations from the real configs, each orthogonal to the adapter
path: no YaRN rope scaling, no multi-token-prediction layer, and
`dsv3-tiny` routes without expert groups (DeepSeek-V3 uses `n_group` 8,
Kimi K2 uses 1).

**Adapters — SYNTHETIC.** PEFT LoRA, `r` 8, `lora_alpha` 16, dropout 0.
`lora_A` keeps PEFT's initialisation; `lora_B` is drawn from N(0, 0.02) with a
seeded generator, because PEFT's zero `lora_B` would make every adapter a no-op
and every engine trivially correct. One adapter per target set:

| model | variant | target modules |
|---|---|---|
| `dsv3-tiny` | `soup-auto` | `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` (Soup's policy) |
| `dsv3-tiny` | `all` | `soup-auto` + `shared_experts.{gate,up,down}_proj` |
| `dsv3-tiny` | `all-but-kv_b` | `all` without `kv_b_proj` |
| `dsv3-tiny` | one per module | each of the five MLA projections alone; `shared` = the three shared-expert projections |
| `qwen35moe-tiny` | `soup-auto` | `q_proj`, `v_proj`, `in_proj_qkv`, `out_proj` (Soup's policy, [`peft_wiring.py:25-30`](../src/soup_cli/utils/peft_wiring.py)) |
| `qwen35moe-tiny` | `all` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `shared_expert.{gate,up,down}_proj` |
| `qwen35moe-tiny` | one per module | each module of `all` alone; `shared` = the three shared-expert projections |

**Reference.** transformers + PEFT, fp32, CPU.

**Engine.** llama.cpp `b11476`: the official Windows CPU build
(`llama-b11476-bin-win-cpu-x64.zip`, sha256 `a23e548c…0100e5483`), base and
adapter converted to f32 GGUF by `convert_hf_to_gguf.py` and
`convert_lora_to_gguf.py` from the same tag, run in llama.cpp's own pinned
converter environment. Logits come from `llama-results`, which writes the
prompt's token ids and the logits of every position to a GGUF file
([`tools/results/results.cpp`](https://github.com/ggml-org/llama.cpp/blob/b11476/tools/results/results.cpp)).
The reference consumes the engine's token ids, so both sides see identical
tokens. Adapter off = no `--lora`; adapter on = `--lora <adapter.gguf>`.

**Strata** (`Niko1221/Strata`) is not run. At `d5ea7133` its README names one
model family, Qwen3.8-Flash-Next, and requires an NVIDIA or AMD card with at
least 12 GB of VRAM
([README L12, L58-L61](https://github.com/Niko1221/Strata/blob/d5ea7133741e67743c0e886bb426c0ce8d69cf6c/README.md#L58-L61)),
so it can load neither tiny model on this CPU-only box.

## 2. The decision rule, written before any run

For each model and adapter variant, over every prompt position and the whole
vocabulary, with `z_ref⁰`, `z_ref¹` the reference logits without and with the
adapter and `z_eng⁰`, `z_eng¹` the engine's:

- `e_base = max|z_eng⁰ − z_ref⁰| / max|z_ref⁰|`
- `Δ_ref = z_ref¹ − z_ref⁰`, `Δ_eng = z_eng¹ − z_eng⁰`
- `s_ref = ‖Δ_ref‖ / ‖z_ref⁰‖` (Frobenius norms), the adapter's effect on the reference
- `ρ = ‖Δ_eng‖ / ‖Δ_ref‖`, how much of that effect the engine reproduces
- `r = ‖Δ_eng − Δ_ref‖ / ‖Δ_ref‖`, how wrong the engine's effect is

The verdict is the first row that matches:

| condition | verdict |
|---|---|
| token ids differ, or `e_base > 1e-3` | **VOID**: the base model itself does not match, so nothing about adapters can be read. Re-run the model with the next seed, at most three seeds; every void attempt is kept |
| `s_ref < 1e-2` | **TOO WEAK**: redraw `lora_B` once with 4x the standard deviation, then report whatever it gives |
| `convert_lora_to_gguf.py` exits non-zero | **CONVERT-FAILED** |
| `llama-results` exits non-zero with the adapter but not without it | **LOAD-FAILED** |
| `r ≤ 1e-2` | **APPLIED** |
| `ρ ≤ 1e-2` | **DROPPED**: the adapter loads and changes nothing. This is the negative control: with the adapter off the engine's output must differ from its output with the adapter on |
| otherwise | **WRONG**: the adapter changes the output, but not the way the reference does |

**Why these thresholds.** Both sides run f32 on the same CPU, so base
differences come only from kernel accumulation order and should sit orders of
magnitude below `1e-3`. A dropped adapter gives `ρ = 0` and `r = 1`, and a
scale error of even 10% gives `r = 0.1`. The APPLIED and DROPPED bands are each
at least 10x from the other outcomes.

**Controls.**

- *Positive control.* On `qwen35moe-tiny` the variants `q_proj`, `v_proj` and
  `shared` go through llama.cpp's LoRA helper and must be APPLIED. If any of
  them is not, the instrument is broken and the probe has **NO VERDICT** on
  `dsv3-tiny`.
- *Reference sanity.* PEFT with the adapter disabled must reproduce `z_ref⁰`
  bit for bit.
- *Determinism.* One variant per model runs twice on each side; logits must be
  bitwise identical, otherwise that model is VOID.

**Expected from the code reading in §0, stated before the run so the run can
contradict it.**

- `dsv3-tiny`: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` DROPPED; `kv_b_proj`,
  `soup-auto` and `all` CONVERT-FAILED; `o_proj` and `shared` APPLIED;
  `all-but-kv_b` WRONG.
- `qwen35moe-tiny`: every variant APPLIED.

**What the result decides.** The hook's content is every `dsv3-tiny` module
whose single-module variant is not APPLIED, together with the converter or
graph code path that drops it. If that list is empty, Soup adapters on MLA need
no engine change at all. If the positive control fails, nothing is concluded.

## 3. Results

The raw output of every run, void ones included, is in
[`results/probe-lora-hook-engines/`](results/probe-lora-hook-engines/): one JSON
and one console log per run. The JSON carries the converter and engine stderr of
every failed step, the SHA-256 of every llama.cpp binary, both Python
environments, the harness fingerprint and a digest of every logit array. The
rule in §2 was not touched between runs; the instrument was, and each change is
below with the evidence that forced it.

### 3.1 Run 1: VOID on both models

- `qwen35moe-tiny`: the base model did not convert at any of the three seeds.
  `convert_hf_to_gguf.py` asserts that a Qwen3.5 config declares a
  multi-token-prediction block unless it is called with `--no-mtp`
  ([`conversion/qwen.py` L298-L305](https://github.com/ggml-org/llama.cpp/blob/b11476/conversion/qwen.py#L298-L305)),
  and the tiny model has none.
- `dsv3-tiny`: `e_base` = 0.0099, 0.0089 and 0.0110 at seeds 17, 18 and 19,
  above the `1e-3` line, so VOID by the rule. Diagnosed afterwards with scratch
  scripts that are not committed: the gap is already 0.4% at position 0, where
  rope is the identity, so it is not positional; an all-dense variant of the
  same model matched to 2.6e-4, so it is the MoE block. transformers 5.19 does
  not write `scoring_func` into a `DeepseekV3Config` it builds itself (its
  implementation hard-codes sigmoid), and the converter writes a gating function
  only when that key is present
  ([`conversion/base.py` L1532](https://github.com/ggml-org/llama.cpp/blob/b11476/conversion/base.py#L1532-L1540)),
  so llama.cpp routed with its default where the model routes with sigmoid.
  The real `deepseek-ai/DeepSeek-V3` config carries `"scoring_func": "sigmoid"`
  and `"topk_method": "noaux_tc"`; writing the same two keys into the tiny
  config brought `e_base` to 3.1e-4. The rest was the engine's CPU defaults,
  flash attention with an f16 KV cache: `-fa off -ctk f32 -ctv f32` brought it
  to 1e-6.

Instrument changes after run 1: the two config keys for `dsv3-tiny`, `--no-mtp`
for the Qwen3.5 base conversion, and the three engine flags, so that "both
sides run f32" in §2 is true of the engine as well.

### 3.2 Run 2: NO VERDICT, and a PEFT finding

Both bases were valid (`e_base` 4.8e-7 and 5.2e-7). The positive control failed
for an instrument reason: every `qwen35moe-tiny` adapter stopped at the same MTP
assertion, this time in `convert_lora_to_gguf.py`, which has no `--no-mtp`
option. The fix is to declare `mtp_num_hidden_layers: 1` as the real Qwen3.5
config does; transformers builds no MTP block from it, so the base checkpoint
is unchanged.

The `dsv3-tiny` variant `shared` failed to convert with "Unprocessed experts",
and its adapter file shows why: it held LoRA factors for
`model.layers.{1,2}.mlp.experts` (the routed experts) and none for
`mlp.shared_experts`, although the targets were
`shared_experts.{gate,up,down}_proj`. **This is a finding about the training
side, not the engine.** In peft 0.21.2, `_convert_peft_config_moe`
(`peft/utils/transformers_weight_conversion.py`, L444-L482) rewrites every LoRA
target that equals or ends in `gate_proj`, `up_proj` or `down_proj` into
`target_parameters` `gate_up_proj` / `down_proj`, the fused parameters of the
routed experts, for every model type whose transformers v5 conversion pattern is
`qwen2_moe`. transformers 5.19.0 maps `deepseek_v3`, `deepseek_v2`, `qwen3_moe`,
`qwen3_next`, `olmoe`, `glm4_moe` and `glm_moe_dsa` to that pattern. A target
prefix does not protect the shared expert, and a regex target is resolved to
the same leaf names first (L355-L399). So on Soup's pinned stack an "attention +
shared expert" adapter for DeepSeek-V3 silently becomes "attention + every
routed expert", the exact shape a giant-MoE adapter budget has to refuse. The
same code is in peft 0.20.0, Soup's floor (read from the published wheel, not
run). Soup's own `target_modules: auto` for `deepseek_v3` is attention only and
is not affected.

Instrument change after run 2: the harness switches that rewrite off while it
builds and loads adapters (`targets_as_named`), so the engine is tested on the
adapter Soup's policy intends, and every variant now records whether its file
touches the routed experts.

The `dsv3-tiny` single-module rows of run 2 already read as in run 3, but they
carry no verdict: the positive control had failed.

### 3.3 Run 3: the verdict

Both bases valid at seed 17: `e_base` 4.8e-7 (`qwen35moe-tiny`) and 5.2e-7
(`dsv3-tiny`), 22 prompt tokens, engine and reference bitwise deterministic.
For all 21 variants PEFT with the adapter disabled reproduced the base bit for
bit, the engine saw the same token ids, and no adapter touched a routed expert.

`qwen35moe-tiny`, the positive control:

| variant | modules adapted | `s_ref` | `ρ` | `r` | verdict |
|---|---|---|---|---|---|
| `soup-auto` | 8 | 0.226 | 1.000 | 1.8e-6 | APPLIED |
| `all` | 31 | 0.305 | 1.000 | 1.3e-6 | APPLIED |
| `q_proj` | 1 | 0.121 | 1.000 | 2.8e-6 | APPLIED |
| `k_proj` | 1 | 0.124 | 1.000 | 2.7e-6 | APPLIED |
| `v_proj` | 1 | 0.215 | 1.000 | 1.5e-6 | APPLIED |
| `o_proj` | 1 | 0.115 | 1.000 | 2.2e-6 | APPLIED |
| `in_proj_qkv` | 3 | 0.076 | 1.000 | 5.3e-6 | APPLIED |
| `in_proj_z` | 3 | 0.041 | 1.000 | 1.0e-5 | APPLIED |
| `in_proj_a` | 3 | 1.5e-4 | 1.000 | 2.6e-3 | TOO WEAK |
| `in_proj_b` | 3 | 0.013 | 1.000 | 3.1e-5 | APPLIED |
| `out_proj` | 3 | 0.042 | 1.000 | 9.4e-6 | APPLIED |
| `shared` | 12 | 0.072 | 1.000 | 5.5e-6 | APPLIED |

`in_proj_a` feeds the per-head decay of the Gated DeltaNet and moves the logits
by 1.5e-4 even with `lora_B` at 4x; the engine reproduces that effect too
(`ρ` 0.9999), but the rule calls it TOO WEAK and it stays so. The positive
control (`q_proj`, `v_proj`, `shared`) is APPLIED, so the instrument reads.

`dsv3-tiny`:

| variant | modules adapted | `s_ref` | `ρ` | `r` | verdict | expected (§2) |
|---|---|---|---|---|---|---|
| `q_a_proj` | 3 | 0.022 | 0 | 1.000 | DROPPED | DROPPED |
| `q_b_proj` | 3 | 0.025 | 0 | 1.000 | DROPPED | DROPPED |
| `kv_a_proj_with_mqa` | 3 | 0.241 | 0 | 1.000 | DROPPED | DROPPED |
| `kv_b_proj` | 3 | 0.468 | - | - | CONVERT-FAILED | CONVERT-FAILED |
| `o_proj` | 3 | 0.205 | 1.000 | 1.5e-6 | APPLIED | APPLIED |
| `shared` | 6 | 0.105 | 1.000 | 2.6e-6 | APPLIED | APPLIED |
| `soup-auto` | 15 | 0.559 | - | - | CONVERT-FAILED | CONVERT-FAILED |
| `all` | 21 | 0.450 | - | - | CONVERT-FAILED | CONVERT-FAILED |
| `all-but-kv_b` | 18 | 0.283 | 0.808 | 0.719 | WRONG | WRONG |

`ρ` is exactly 0 for the three dropped projections: the engine's logits with the
adapter are bit-identical to its logits without it, while the reference moves
by 2-24%. `kv_b_proj` fails with `NotImplementedError` in the 3-D transpose of
`k_b` (`convert_lora_to_gguf.py` L179, reached from `conversion/deepseek.py`
L446). `all-but-kv_b` is the case a user would actually hit if the converter
skipped `kv_b`: the server loads the adapter, the output changes, and 72% of
the change is wrong.

**Verdict under §2: HOOK NEEDED**, for exactly the four modules §2 predicted
from the code reading: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` (DROPPED)
and `kv_b_proj` (CONVERT-FAILED).

## 4. What the hook has to cover

llama.cpp `b11476`; Kimi K2 runs through the same `deepseek2` graph
(`DeepseekV3ForCausalLM`), so the list is the same for it.

| Soup module | GGUF tensor | today | change needed |
|---|---|---|---|
| `q_a_proj` | `blk.N.attn_q_a` | plain `ggml_mul_mat`, [`deepseek2.cpp` L502](https://github.com/ggml-org/llama.cpp/blob/b11476/src/models/deepseek2.cpp#L502); MTP path L266 | call `build_lora_mm` instead |
| `q_b_proj` | `blk.N.attn_q_b` | plain, L508; MTP L272 | same |
| `kv_a_proj_with_mqa` | `blk.N.attn_kv_a_mqa` | plain, L518; MTP L288 | same |
| `kv_b_proj` | `blk.N.attn_k_b`, `blk.N.attn_v_b` | the converter cannot transpose the per-head `k_b` factor in 3-D; the absorbed `wk_b` (L561; MTP L330) and `wv_b` (attention helper, [`llama-graph.cpp` L2744-L2754 and L2811](https://github.com/ggml-org/llama.cpp/blob/b11476/src/llama-graph.cpp#L2744-L2812), with and without flash attention) are plain matmuls | converter: split `lora_B` per head into K and V parts and swap the factor roles for the transposed `k_b`; graph: low-rank terms at `wk_b` and both `wv_b` sites |
| `o_proj` | `blk.N.attn_output` | APPLIED | none |
| `shared_experts.*` | `blk.N.ffn_{gate,up,down}_shexp` | APPLIED | none |

By reading, not by running: the adapter loader's shape check compares only the
first two dimensions and the rank
([`llama-adapter.cpp` L358-L371](https://github.com/ggml-org/llama.cpp/blob/b11476/src/llama-adapter.cpp#L358-L371)),
so per-head 3-D factors for `k_b` and `v_b` would pass it unchanged. Run 3
exercised the path without flash attention; the flash-attention `wv_b` site is
the same plain matmul by reading, not by measurement.

Outside llama.cpp, two more pieces are needed: Soup has to export an adapter on
its own (`soup export` merges today), and PEFT has to be stopped from rewriting
shared-expert targets (§3.2) before a shared-expert adapter can be trained at
all.

## 4a. Part A': is a small hook enough? Rule written before the run

A draft hook, written against `b11476` after run 3: 3 files, 18 lines added and
13 removed (`convert_lora_to_gguf.py` +8/-3, `src/models/deepseek2.cpp` +8/-8,
`src/llama-graph.cpp` +2/-2). It routes `wq_a`, `wq_b`, `wkv_a_mqa` and the
absorbed `wk_b` through `build_lora_mm` in the main and MTP graphs, does the
same for `wv_b` at both attention-helper sites, and teaches the adapter
converter the batched transpose of the last two dimensions that `k_b` needs:
`(B·A)ᵀ = Aᵀ·Bᵀ` per head, so the two factors swap roles. The patch under test
is fixed by its SHA-256,
`80851e0bcb5ac775e7918799333eb6e0210948132e9f8d18238caec0d5fe1c09`, and is not
edited between the control and the patched run.

Both builds come from the same local toolchain (MinGW-w64 GCC 15.2.0, CMake
4.4.4, Ninja, CPU only, `GGML_NATIVE=ON`) out of the same `b11476` checkout, and
run through the unchanged harness with §2's rule.

| condition | outcome |
|---|---|
| the stock local build does not reproduce run 3's verdict for every variant of both models | **NO VERDICT**: the local build changes behaviour, so nothing about the patch can be read |
| with the patched build and converter, every `dsv3-tiny` variant (the five MLA projections, `shared`, `soup-auto`, `all`, `all-but-kv_b`) is APPLIED and every `qwen35moe-tiny` verdict equals run 3's | **SUFFICIENT**: this hook alone serves the adapter shape on these tiny models |
| otherwise | **INSUFFICIENT**: the variants that are not APPLIED name what is still missing |

A SUFFICIENT draft says nothing about upstream acceptance, quantised base
tensors, the GPU backends or a real checkpoint; it measures whether the change
is a few dozen lines or a project.

## 4b. Part A': results

Both local builds report `version: 0.6.0-dev (build 1, commit 9881906)`, GNU
15.2.0; the harness is unchanged from run 3 (fingerprint `b0b17e1ea00dcbc4`).
The patch's SHA-256 was checked again before the patched run and matched.

- **Control, stock local build:** every verdict of both models equals run 3's,
  with `e_base` 4.8e-7 (`qwen35moe-tiny`) and 4.5e-7 (`dsv3-tiny`). The local
  toolchain does not change behaviour.
- **Patched build and converter:** every `dsv3-tiny` variant is APPLIED:
  `q_a_proj` r = 1.4e-5, `q_b_proj` 1.2e-5, `kv_a_proj_with_mqa` 1.4e-6,
  `kv_b_proj` 7.4e-7, `o_proj` 1.5e-6, `shared` 2.6e-6, `soup-auto` 6.6e-7,
  `all` 9.2e-7, `all-but-kv_b` 1.2e-6, all with `ρ` = 1.000. Every
  `qwen35moe-tiny` verdict equals run 3's.

**Verdict under §4a: SUFFICIENT.** On these tiny models the four failures of
run 3 are closed by 18 added and 13 removed lines in three files. The patch is
kept as [`mla-lora-hook-draft.patch`](results/probe-lora-hook-engines/mla-lora-hook-draft.patch)
(llama.cpp code, MIT; its licence travels with it in
[`LICENSE-llama.cpp.txt`](results/probe-lora-hook-engines/LICENSE-llama.cpp.txt)).
It is a measurement instrument, not a proposed upstream change: a pull request
would need llama.cpp's own tests, review of the DeepSeek-V2-Lite and legacy
`wkv_b` paths it leaves alone, and the GPU backends.

**An extra arm, decided after the verdict and not part of it.** Parts A and A'
run without flash attention, so the flash-attention `wv_b` site of the patch was
not exercised. The same patched build was run once more on `dsv3-tiny` with
llama.cpp's CPU defaults restored (flash attention, f16 KV cache;
`part-a2-hook-local-fa-on.*`). Base gap 3.0e-4, inside §2's line. `kv_b_proj`
is APPLIED (r 7.0e-4), and so are `kv_a_proj_with_mqa`, `o_proj`, `shared`,
`soup-auto`, `all` and `all-but-kv_b` (r 6.5e-4 to 1.7e-3). `q_a_proj` and
`q_b_proj` come out WRONG by §2's f32 line, at r = 0.012 and 0.011 with
`ρ` = 1.0003 and 1.0004 and a cosine of 0.99993 between the two effects. These
are the two smallest effects in the table (`s_ref` 0.022 and 0.025), and with
f16 storage in the KV cache the base gap is 600 times the f32 arm's, so an
error of about 1% of so small an effect is the size the noise predicts. That
reading is an inference, not a measurement, and the verdicts stand as printed.
The harness's overall line for this arm reads NO VERDICT only because the
positive-control model was not part of it.

## 4c. Part B: the real Qwen3.5-35B-A3B in the cloud. Rule written before the run

Part A's verdicts come from tiny random models. Part B asks whether the same
engine path holds on the real test model, with the adapter exported on its own
and the routed experts served from host RAM.

- **Model:** `Qwen/Qwen3.5-35B-A3B@59d61f3c` (bf16, 71.9 GB, Apache-2.0). For
  this config transformers builds `Qwen3_5MoeForCausalLM`, the text tower.
- **Adapters, SYNTHETIC:** Part A's recipe (PEFT LoRA, `r` 8, `lora_alpha` 16,
  `lora_B` ~ N(0, 0.02) from a seeded generator, one 4x redraw if `s_ref` <
  0.01). Two variants: `soup-auto`, Soup's Qwen3.5 policy (`q_proj`, `v_proj`,
  `in_proj_qkv`, `out_proj`), and `soup-auto+shared`, which adds
  `shared_expert.{gate,up,down}_proj`.
- **Export:** adapter only, through `convert_lora_to_gguf.py --base <checkpoint>`,
  which reads the base config and not its weights; f32.
- **Engine:** llama.cpp `b11476`, built from the tag on the instance (Linux,
  CPU). Base converted with `--outtype bf16 --no-mtp`, weights memory-mapped
  from the local disk, so the routed experts are served from host RAM;
  `-fa off -ctk f32 -ctv f32`.
- **Reference:** transformers 5.19.0 + PEFT 0.21.2 on the CPU in bf16. An fp32
  copy (about 140 GB) does not fit in the box's 128 GB.
- **Box:** NVIDIA Brev, GCP `n2d-highmem-16` (16 vCPU, 128 GB RAM, no GPU),
  disk of at least 300 GB.
- **Harness:** [`lora_hook_real_model.py`](harness/lora_hook_real_model.py),
  which reuses Part A's engine runner and metrics.

Both sides run bf16, so §2's f32 lines do not apply. The tolerance is set by
how much the two base models disagree:

- `n_base = ‖z_eng⁰ − z_ref⁰‖ / ‖z_ref⁰‖`, and top-1 agreement over positions;
- the floor `f = ‖z_eng⁰ − z_ref⁰‖ / ‖Δ_ref‖`, the base gap in units of the
  adapter's effect, and the tolerance `t = 3f + 0.02`.

| condition | verdict |
|---|---|
| top-1 agreement < 0.9, or `n_base` > 0.05, or either side not deterministic | **VOID** |
| `s_ref` < 0.01 after the redraw | **TOO WEAK** |
| the converter fails / the engine refuses the adapter | **CONVERT-FAILED** / **LOAD-FAILED** |
| `t` > 0.5 | **TOO NOISY**: the band would admit a dropped adapter (`r` = 1) |
| `r` ≤ `t` | **APPLIED** |
| `ρ` ≤ 0.1 | **DROPPED** |
| otherwise | **WRONG** |

Why `3f`: an engine that applies the adapter correctly still differs from the
reference by about one base gap with the adapter and one without, so its error
on the effect is about `2f`. `3f + 0.02` leaves room for that and for nothing
like a dropped (`r` = 1) or half-applied (`r` ≈ 0.5) adapter, as long as `t`
stays at or below 0.5. Expected from Part A: both variants APPLIED.

The stage's cloud budget is $10. The instance is deleted after the run and
the record gives the instance type, versions, wall times and the cost
(hours times list price, plus disk). If the reference cannot finish inside the
budget, Part B records how the engine takes the adapter (steps, memory,
versions) and correctness stays with Part A.

## 5. Reproducing

Two Python 3.12 environments: the reference one is Soup's
(`pip install -e ".[dev]"`; run 3 used torch 2.14.1+cpu, transformers 5.19.0,
peft 0.21.2), and the converter one is llama.cpp's own at the tag:

```bash
git clone --depth 1 --branch b11476 https://github.com/ggml-org/llama.cpp <llama-src>
python -m venv <convert-venv>
<convert-venv>/bin/python -m pip install -r <llama-src>/requirements/requirements-convert_lora_to_gguf.txt
<convert-venv>/bin/python -m pip install <llama-src>/gguf-py
```

Engine: the release asset `llama-b11476-bin-win-cpu-x64.zip` (sha256
`a23e548c6b3525c38bcfeceaff919786ae06741857043cb670279b70100e5483`) unpacked to
`<llama-bin>`, or any `b11476` build that has `llama-results`. Then, from the
repository root:

```bash
python benchmarks/harness/lora_hook_parity.py \
  --llama-bin <llama-bin> --llama-src <llama-src> \
  --convert-python <convert-venv>/bin/python \
  --work-dir <scratch> --models qwen35moe-tiny,dsv3-tiny --seed 17 --threads 4 \
  --out benchmarks/results/probe-lora-hook-engines/part-a-cpu-run3.json \
  --log benchmarks/results/probe-lora-hook-engines/part-a-cpu-run3.log
```

The tokenizers are fetched from the Hub at the revisions in §1; the models and
adapters are built from seeds, so nothing else is downloaded. Wall time on this
box: about 6.5 minutes.
