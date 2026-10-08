# Scope decision: who serves the giant MoE a Soup adapter was trained on?

**Status: draft, 2026-10-08; final by 2026-10-14.** Evidence:
[`scope-moe-serving-evidence.md`](scope-moe-serving-evidence.md). Measurement:
[`probe-lora-hook-engines.md`](probe-lora-hook-engines.md). Next work:
[`scope-moe-serving-prototype.md`](scope-moe-serving-prototype.md).

## The question

Soup fine-tunes LoRA adapters on giant MoE models (DeepSeek-V3 671B, Kimi K2
~1T) on one laptop. Should Soup also serve them, with its own inference that
streams routed experts from SSD (variant 1)? Or should an existing engine serve
them, with a small hook added upstream so that it applies Soup adapters
(variant 2)?

## Decision

**Variant 2, with llama.cpp as the engine for the prototype.** Soup trains and
exports the adapter. The engine owns model execution, expert placement and
reads from SSD. The hook goes upstream through llama.cpp's own process. This
picks the scope for a prototype; it does not announce a product path.

Why:

- **The hook is small, and it is measured.** llama.cpp `b11476` already applies
  LoRA to `o_proj` and the shared expert of DeepSeek-V3. It silently drops it on
  `q_a_proj`, `q_b_proj` and `kv_a_proj_with_mqa`, and cannot convert it on
  `kv_b_proj`. A draft of 18 added and 13 removed lines closes all four on tiny
  DeepSeek-V3-shaped models: the adapter's effect on the logits matches
  transformers + PEFT to `r` ≤ 1.4e-5, and a stock build made with the same
  toolchain does not. With the draft's converter, a SYNTHETIC adapter for the
  real DeepSeek-V3 and Kimi K2 configs exports from `config.json` alone; every
  factor pair carries the adapter's rank, and the `k_b` and `v_b` pairs the
  hook writes have the outer dimensions llama.cpp's loader expects. The other
  pairs' outer dimensions were not checked.
- **Variant 1 is a new runtime.** Soup's streaming is built for training: it
  reads every layer once per step, has no KV cache, and refuses generation by
  design. Serving would need expert-granular storage and fetch, a generation
  loop, the DeepSeek-V3 architecture and quantised expert kernels, all of which
  engines already have. Soup wraps existing tools rather than re-implementing
  them; its trainers wrap TRL ([CONTRIBUTING](../CONTRIBUTING.md#trainers-as-wrappers)).
- **llama.cpp fits the target box best of the candidates.** It runs natively on
  Windows, implements DeepSeek-V3's grouped expert routing (read in its source,
  not run), memory-maps GGUF files so routed experts can stay on SSD, loads
  adapters unmerged, and is MIT-licensed.

Set aside, with the reason (details in the evidence appendix, §3):

- **Strata.** It serves one model family of its own, needs 12 GB of VRAM or
  more, and no adapter path was found in it. For Strata the "hook" would be a
  DeepSeek-V3 port.
- **ik_llama.cpp.** Its expert read-ahead is Linux-only, it refuses runtime LoRA
  with flash attention, and its routing for DeepSeek-V3 is not V3's grouped
  routing.
- **KTransformers.** Its wheels are Linux x86-64 only. Its MLA `kv_b` adapter
  correction runs only for composite routed-expert adapters; the fix is
  estimated at 100-250 lines. This is the fallback if Linux is acceptable.
- **SSD-LLaMA, Colibrì, SGLang SSD Expert Pack.** No released code, or no
  DeepSeek-V3/K2 support.
- **vLLM, SGLang, TensorRT-LLM.** No SSD tier: CPU offload is unified memory or
  layer prefetch, so the experts would have to fit in RAM. Linux or WSL only.

## User scenario

A Soup user fine-tunes an attention adapter (optionally with the shared expert)
for DeepSeek-V3 or Kimi K2 on a laptop. They export the adapter on its own and
serve the base model in llama.cpp with the adapter loaded at startup. The
routed experts stay on the NVMe drive and are paged into RAM by the engine;
attention runs on the 8 GB GPU where it fits. Qwen3.5-35B-A3B is the test model
at real scale. It has no MLA, so it checks the adapter-only export and
llama.cpp's standard adapter paths for attention and the shared expert, not the
tensors the hook changes; MLA on real weights is prototype task P4.

## Success criterion

The adapter's effect in the engine matches its effect in transformers + PEFT,
module by module, and a silently dropped module is detected:

- tiny models, f32: `r` ≤ 1e-2 for every target module; a dropped module shows
  as `ρ` ≤ 1e-2 (the record's §2);
- real model: `r` within three times the base models' own disagreement, as in
  the record's Part B. In bf16 on the real Qwen3.5-35B-A3B that disagreement
  was too large for any verdict (record §4f), so the precision of this
  comparison is still open.

Generation speed is **not** a criterion. Engine speed figures in the evidence
appendix are context from their sources.

## Licence conditions

- **Soup.** It calls llama.cpp as a separate program and produces GGUF files
  with llama.cpp's converter, so no llama.cpp code enters Soup.
- **llama.cpp.** The hook is contributed under MIT, through llama.cpp's process:
  an issue first, local CI, AI use disclosed, every line understood by its
  author, and PR text written by a human. No CLA or DCO is required by its
  pinned contribution files.
- **Strata.** It stays a source of ideas only, with no code read for
  implementation or copied. Its root MIT licence dates from `b500a81e`
  (2026-09-28), with ggml, a font and a projection vector under their own terms.
  The policy does not change with the licence.
- **Model weights.** These keep their own licences, separately from any engine.

## What is not settled

- **PEFT.** On Soup's pinned stack PEFT cannot build the shared-expert adapter
  for `deepseek_v3`: it retargets it onto the routed experts (record §3.2). Until
  that is worked around, the shape that can be trained today is attention only.
- **The real model.** The comparison on the real Qwen3.5-35B-A3B ran once, in
  bf16, and is VOID: the two bf16 base models differ by 6.6% of the logits (the
  line is 5%) and by 27-56% of the adapters' own effect, so the rule gives no
  verdict (record §4f). Both adapters did export adapter-only and load onto the
  real base. Correctness rests on the tiny models, and real MLA weights are P4.
- **Upstream acceptance.** The draft hook is not a pull request. Quantised base
  tensors, GPU backends and the review are untested.
- **Fit on the dev box.** No engine has shown DeepSeek-V3 or K2 with experts on
  SSD on 8 GB VRAM and 32 GB RAM. DeepSeek-V3's GPU-resident part alone is about
  9.6 GB at 4.5 bits [ESTIMATE].

## Assumptions to confirm

1. **Target box.** The dev box: RTX 5070 Laptop 8 GB (sm_120), 32 GB RAM, two
   NVMe drives, Windows 11.
2. **Windows.** Native Windows is required. If Linux is acceptable,
   KTransformers (about 100-250 lines to fix its `kv_b` path) and ik_llama.cpp
   (Linux-only expert read-ahead) become candidates.
3. **Adapter format.** PEFT safetensors with the HF `DeepseekV3` module names:
   `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, plus
   optionally `shared_experts.{gate,up,down}_proj`. Routed experts stay frozen.
   Soup cannot stream-train DeepSeek-V3 yet, so this is the format its
   `target_modules: auto` would produce, not an observed Soup artifact.
4. **Test model.** Qwen3.5-35B-A3B, as advised.
5. **Engine version.** llama.cpp at or after `b11476`, plus the hook. Soup's
   export pin (`b5270`) moves with it.
