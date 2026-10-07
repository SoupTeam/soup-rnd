<!--
Working measurement record. §2, the decision rule, was written and committed
before any engine run; results are appended below it as they arrive and the
rule is not edited afterwards.

Box for Part A: Windows 11 Pro 26200, AMD Ryzen 5 8645HS (6 cores / 12 threads),
15.3 GB RAM, CPU only. The box has an RTX 4050 Laptop GPU; this probe does not
use it.
-->

# Probe — does a MoE serving engine apply a Soup LoRA adapter to MLA attention and the shared expert?

**Status: Part A rule committed, not run.**

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

Not run yet.
