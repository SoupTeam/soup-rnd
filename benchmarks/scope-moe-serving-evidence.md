<!--
Evidence appendix to scope-moe-serving.md. Observed 2026-10-08. Every external
claim is pinned to a release tag, commit or paper version.
-->

# Evidence: does Soup serve giant MoE itself, or hand serving to an engine?

**Status: draft, 2026-10-08.** The decision this supports is in
[`scope-moe-serving.md`](scope-moe-serving.md); the prototype tasks are in
[`scope-moe-serving-prototype.md`](scope-moe-serving-prototype.md).

Labels: **[CODE]** read in source at the pinned ref; **[DOC]** the project's own
documentation; **[RUN]** measured here, with a record; **[PAPER]** a paper's own
claim; **[ESTIMATE]** arithmetic; **[HYPOTHESIS]** not tested. Source reading
was partly done by delegated research passes; the claims that carry the
decision were re-read or measured directly and say so.

## 1. What Soup has today

| Area | Fact | Label |
|---|---|---|
| Serving | `soup serve` backends are transformers, vLLM, SGLang and DeepSpeed-MII ([`serve.py:504-511`](../src/soup_cli/commands/serve.py)); none loads weights or experts on demand during generation. The only offload is transformers' `device_map="auto"` spilling to the CPU ([`utils/gpu.py:145-162`](../src/soup_cli/utils/gpu.py)). | [CODE] |
| Layer streaming | Training only. Generation rollouts are refused for good: "re-read every layer once per generated token" ([`config/schema.py:6401-6408`](../src/soup_cli/config/schema.py)). The streamed skeleton sets `use_cache = False` ([`layer_stream_runtime.py:2373-2374`](../src/soup_cli/utils/layer_stream_runtime.py)). | [CODE] |
| MoE in streaming | A layer shard holds every tensor of the layer, routed experts included ([`layer_shard.py:1473-1605`](../src/soup_cli/utils/layer_shard.py)); the prefetch key is the layer index. No expert-granularity path exists. `deepseek_v3` and `kimi_k2` are not in `SUPPORTED_STREAM_ARCHS` ([`layer_stream.py:91-112`](../src/soup_cli/utils/layer_stream.py)), so Soup cannot stream-train DeepSeek-V3 today. | [CODE] |
| Adapter targets | `target_modules: auto` for `deepseek_v3`, `kimi_k2`, `kimi_k25` is the five MLA projections, no shared expert ([`peft_wiring.py:81-87,126,139,144`](../src/soup_cli/utils/peft_wiring.py)); for Qwen3.5 it is `q_proj`, `v_proj`, `in_proj_qkv`, `out_proj` (L25-30). | [CODE] |
| Export | `soup export` merges the adapter into the base after a full-precision CPU load ([`export.py:409-421`](../src/soup_cli/commands/export.py)). There is no adapter-only GGUF path, and the llama.cpp auto-clone is pinned at `b5270` ([`export.py:34`](../src/soup_cli/commands/export.py)). A merge needs the whole base in host RAM, about 1.3 TB in 16-bit for 671B parameters. | [CODE], [ESTIMATE] |

What the streaming code could give a generation path, if Soup built one:

| Component | For batch-1 generation |
|---|---|
| direct-I/O range reads, pinned arenas, event-safe staging (`safetensors_reader`, `async_disk_source`) | reusable as they are |
| `AsyncDiskSource`, `LayerBufferPool`, `StreamPrefetcher`, `StreamedDecoderLayer` | reusable only after a redesign around (layer, expert) keys, a router-driven fetch and an expert cache |
| meta skeleton and PEFT construction | needs a KV cache, loading of a trained adapter instead of a fresh one, new architectures |
| training setup, VRAM formulas, `stream_probe` | not reusable |

## 2. Variant 1, Soup's own streamed inference: what it would take

[CODE]-grounded list; no size is claimed beyond "a new runtime":

1. a generation loop with a KV cache over the streamed model (§1: the skeleton has none);
2. expert-granularity storage and runtime: per-expert byte ranges, an expert
   buffer pool, residency and eviction, a fetch driven by the router;
3. the DeepSeek-V3 architecture in the streaming path (MLA, grouped `noaux_tc`
   routing, shared expert) and an FP8 source path for the real checkpoints;
4. quantised expert kernels on the CPU or the GPU for decode;
5. an expert prefetch or cache policy, which is the research question itself.

Every item exists today in at least one external engine (§3). Soup's training
streaming reads each layer once per step; generation reads per token, which is
why Soup refuses rollouts under streaming in the first place.

## 3. Engines that could serve DeepSeek-V3 / Kimi K2 with experts on SSD

Target box: 8 GB NVIDIA laptop GPU, 32 GB RAM, NVMe, Windows 11. "SSD" means the
routed experts may exceed RAM and be read from the drive during generation.

| Engine, pinned | DSV3 / K2 architecture | SSD on 8 GB VRAM / 32 GB RAM | Windows | LoRA unmerged | What a Soup adapter needs | Licence |
|---|---|---|---|---|---|---|
| llama.cpp `b11476` (`988190680d5a`, 2026-10-07) | yes, grouped `noaux_tc` routing included [CODE] | memory-mapped GGUF, OS paging, `PrefetchVirtualMemory` on Windows; no predictive expert cache [CODE]; this exact setup not demonstrated [HYPOTHESIS] | native [DOC] | GGUF LoRA via `convert_lora_to_gguf.py`, `--lora`, per-request scale on the server [CODE] | the MLA hook, measured: a 31-line draft (§4) | MIT |
| ik_llama.cpp (`c069e89e`) | V3 grouped routing not exact (global top-k for `deepseek2`); K2 unaffected [CODE] | Linux-only expert read-ahead and deferred residency [CODE] | builds, but the SSD path is a no-op on Windows [CODE] | GGUF LoRA, but refused with flash attention; fused shared gate/up bypasses adapters [CODE] | the MLA hook plus fusion, flash-attention and routing fixes | MIT |
| KTransformers v0.7.1 (`0dce4c9b`) + `kvcache-ai/sglang` (`3424f35d`) | yes [CODE] | single-NUMA llamafile backend aliases the GGUF memory map [CODE]; 8/32 GB unproven | Linux x86-64 wheels [DOC] | PEFT safetensors [CODE]; its absorbed-MLA `kv_b` correction runs only for composite routed-expert adapters, so not for Soup's [CODE] | decouple the `kv_b` correction, about 100-250 lines [ESTIMATE, research pass] | Apache-2.0 |
| Strata v0.1.40.3 (`d5ea7133`) | no: `qwen4exp` only [CODE] | file-backed experts for its own model; minimum 12 GB VRAM [DOC] | yes [DOC] | no LoRA path found [CODE] | port the architecture first: thousands of lines [ESTIMATE] | MIT since `b500a81e` (2026-09-28); ggml MIT, font OFL-1.1, projection vector Qwen Community Licence; none before |
| SSD-LLaMA (arXiv 2609.18110v1) | evaluates DeepSeek-V4-Flash, Kimi-K2.7-Code, GLM-5.2, not V3/K2 [PAPER] | real SSD design; RTX 5090 32 GB [PAPER] | no release | no source released | unknown | paper CC-BY-NC-ND-4.0; code none |
| Colibrì (`bf244291`) | not supported (GLM-5.x, Kimi-K3, DeepSeek-V4) [CODE] | bounded SSD expert tier [CODE] | CPU/Vulkan [DOC] | no PEFT path found | model support, loader and CPU/GPU paths | Apache-2.0 |
| SGLang SSD Expert Pack (`81c9f837`) | rejects V3/K2, accepts V4 and Kimi-linear K3 [CODE] | CUDA expert cache with direct I/O [CODE] | POSIX-only (`fcntl`) [CODE] | not in the SSD path | model, pack and OS port | Apache-2.0 |
| vLLM v0.31.0, SGLang v0.5.21, TensorRT-LLM v1.3.0rc29 | yes (TRT-LLM per model list) | no SSD tier; CPU offload is UVA or deterministic layer prefetch [CODE] | WSL/Linux only [DOC] | yes, with MLA caveats | context only | Apache-2.0 |

Detail behind the table, kept in the research notes of this stage and re-checked
where marked: llama.cpp `deepseek2.cpp` L496-L520 (plain `ggml_mul_mat` for
`wq_a`, `wq_b`, `wkv_a_mqa`) [CODE, re-read]; ik `llama-build-context.cpp`
L1567-L1655 (grouped top-k gated to `BAILINGMOE2/3`); KT fork
`lora_manager.py` L413-L432 (static `kv_b` finalizer gated on
`kt_composite_lora_id`); Strata `tools/strata_inspect.py` L85-L114 (rejects any
architecture but `qwen4exp`) and README L58-L61 (12 GB minimum) [re-read].

## 4. The adapter hook, measured

From [`probe-lora-hook-engines.md`](probe-lora-hook-engines.md), tiny SYNTHETIC
DeepSeek-V3- and Qwen3.5-MoE-shaped models, f32 on CPU [RUN]:

- llama.cpp `b11476` silently drops LoRA on `q_a_proj`, `q_b_proj`,
  `kv_a_proj_with_mqa` (the engine's logits do not move at all) and fails to
  convert LoRA on `kv_b_proj`. `o_proj` and the shared expert are applied to
  `r` ≈ 2e-6. Every Qwen3.5 attention and shared-expert path is applied.
- A draft hook of 18 added and 13 removed lines in three files
  (`convert_lora_to_gguf.py`, `deepseek2.cpp`, `llama-graph.cpp`) makes every
  DeepSeek-V3 variant APPLIED at `r` ≤ 1.4e-5. A stock build from the same
  toolchain reproduces the unpatched verdicts, so the patch is what changed them.
- Not covered by the measurement: quantised base tensors, GPU backends, the
  DeepSeek-V2-Lite and legacy `wkv_b` paths, upstream review.
- Real Qwen3.5-35B-A3B (Part B of the record): not run; the cloud instances
  could not be reached over SSH (record §4e).

Two pieces outside the engine:

- **Soup has to export the adapter alone.** `convert_lora_to_gguf.py --base`
  reads only the base config ("actual model weights are not required",
  `convert_lora_to_gguf.py` `--base` help at `b11476`) [CODE]. Measured at real
  dimensions with a SYNTHETIC adapter (record §4e) [RUN]: with the hook's
  converter, adapters for the DeepSeek-V3 and Kimi K2 configs export from
  `config.json` alone and every factor pair passes llama.cpp's loader shape
  checks; the stock converter fails on `kv_b_proj`.
- **PEFT cannot build the shared-expert adapter on Soup's stack.** peft 0.21.2
  rewrites `shared_experts.{gate,up,down}_proj` targets on `deepseek_v3` into
  the routed experts' fused parameters (record §3.2) [RUN]; the same code is in
  peft 0.20.0 [CODE]. Soup's attention-only policy is not affected.

## 5. Licences and how each project takes changes

Variant 2 puts code in the engine, under the engine's licence and process.

| Project | Licence at the pin | How changes are accepted | Label |
|---|---|---|---|
| llama.cpp | MIT; bundled `stb_image` MIT or Unlicense, `subprocess.h` Unlicense | issue first for features; local CI plus performance or perplexity checks; CPU-first model PRs preferred; AI use must be disclosed, every line understood by the author, and PR text and review replies written by a human ([`CONTRIBUTING.md`](https://github.com/ggml-org/llama.cpp/blob/b11476/CONTRIBUTING.md), [`AGENTS.md`](https://github.com/ggml-org/llama.cpp/blob/b11476/AGENTS.md)); no CLA or DCO in those files | [DOC] |
| ik_llama.cpp | MIT | backend tests and local CI; AI tolerated but discouraged, disclosed; AI-authored PRs rejected ([`CONTRIBUTING.md`](https://github.com/ikawrakow/ik_llama.cpp/blob/c069e89e10202e37c7c2469cfa8ac1a6c7bc451b/CONTRIBUTING.md)) | [DOC] |
| KTransformers / its SGLang fork | Apache-2.0 / Apache-2.0 (a BSD-3-Clause flash-attention component in the fork) | formatter hooks and bracketed Conventional Commits (KT); regression tests, accuracy and speed evidence, oncall plus code-owner review (fork) | [DOC] |
| Strata | MIT since `b500a81e` (2026-09-28), with the exceptions in §3; no LICENSE at `55e2fb85` (2026-09-26) | PR template; `AGENTS.md` asks for measured claims with the machine stated; no CONTRIBUTING, CLA or DCO found | [DOC], [CODE] |
| Soup | Apache-2.0, no CLA | — | [DOC] |

For Soup itself: calling llama.cpp as a separate program, or shipping its
converter output, puts no llama.cpp code in Soup. Strata stays a source of
ideas only, as decided for this stage, whatever its current licence.

## 6. Generation speed: context, not a criterion

Published numbers, each on its own box; none is the target laptop:

| Source | Model and box | Decode | Label |
|---|---|---|---|
| KTransformers DSV3 guide | DeepSeek-V3 Q4_K_M, 382 GB DRAM, RTX 4090 | 12.4-13.5 tok/s, 8 experts | [DOC] |
| KTransformers Kimi K2 guide | K2, ~600 GB DRAM, one consumer GPU | ~10 tok/s | [DOC] |
| ik_llama.cpp PR 1634 | Qwen3.5-397B-A17B (~123 GiB) on 64 GiB RAM + RTX 4080 SUPER | ~6.6 tok/s | [DOC] |
| SSD-LLaMA §5.4 | Kimi-K2.7-Code, RTX 5090 (32 GB) + 32 GB RAM | 1.03 tok/s | [PAPER] |
| SGLang SSD Expert Pack blog | DeepSeek-V4-Flash, RTX 5090 + 32 GB RAM | 1.85-1.99 tok/s | [DOC] |
| Strata models page | its 125B model, experts read from SSD, 64 GB RAM + 12 GB GPU | 7-8.5 tok/s | [DOC] |
| Colibrì README | GLM-5.2 744B on a 25 GB-class laptop, cold | 0.05-0.1 tok/s | [DOC] |

Arithmetic for DeepSeek-V3 on the target box, at about 4.5 bits per weight
[ESTIMATE]:

- one routed expert is 3 × 7168 × 2048 = 44.0M parameters, about 24.8 MB; a
  token uses 8 experts in each of 58 MoE layers, about 11.5 GB of expert weights;
- with every expert missing from RAM, at the 7.0 GB/s the two dev-box drives
  reach together when striped (`gate-two-drive-striping.md`), that is about
  1.6 s per token; at the ~5 GB/s the two drives sustain together
  (`probe-rtx5070-two-drive-sustained.md`, ~2.5 GB/s each), about 2.3 s. This is
  the full-miss case, not a ceiling: caching and skewed routing lower it, and
  SSD-LLaMA reports above 1 tok/s at 1T on an RTX 5090 with 32 GB of RAM;
- the routed experts total about 368 GB; 32 GB of RAM can hold a few percent;
- the part an engine keeps on the GPU (attention, shared experts, router,
  embeddings) is about 17B parameters for DeepSeek-V3, about 9.6 GB at 4.5
  bits: more than 8 GB, so some of it would stay on the CPU. For Kimi K2 (64
  heads) it is about 11.5B, about 6.5 GB.

## 7. What transfers from the training-step coverage record to batch-1 decode

[`probe-moe-expert-coverage.md`](probe-moe-expert-coverage.md) measured a
training step (512-2048 tokens). At batch-1 decode a step is one token:

| Training-step result | At batch-1 decode |
|---|---|
| coverage C = 0.967-1.000 | carries over to a long prompt's prefill only; per decode token coverage is k/E by construction (25% granite, 12.5% OLMoE, 6.25% Qwen3-30B-A3B, 3.1% DeepSeek-V3, 2.1% Kimi K2) |
| "memory only, no read saving" | does not carry over: at decode, reading by expert saves reads by construction; the open quantity is reuse across tokens |
| a resident fraction f saves f of the reads | does not carry over: at decode the saving is the resident set's share of traffic, 0.41-0.68 for the busiest quartile in that record, as an in-sample upper bound |
| "caught by corpus hot-25" matches the top-25% share, read as "steps are homogeneous" | **not a result but an identity**: the hot set is computed from the same steps it is scored on, and every step carries the same number of assignments. Recomputed here from the published JSON: equal to 0.0e+00 in all 564 layer rows; the 0.005/0.013 gap in the summary comes from averaging layers 1..L-1 against 0..L-1 [RUN] |
| "caught by the previous layer's set" = 0.96-1.00 | does not carry over: at decode it measures how often layer l+1 picks an expert with the same index as layer l, which are different experts; worth measuring, not a predictor by itself |
| NF4 moves aggregate statistics by ~0.004 | carries over for aggregates; per-token flips were not measured |
| the method (router discovery by shape, packing, count identity, determinism) | carries over unchanged |

## 8. Prefetch at batch 1: what the literature says

Under variant 2 this is the engine's concern; it sets what a prototype can
expect, not what Soup builds.

| Predictor family | Retraining | Reported | Source |
|---|---|---|---|
| next layer's router on the current hidden state | no | ~96% top-1 next layer on Mixtral; ~90% two-three layers ahead | HOBBIT 2411.01433v2 §3.3 [PAPER] |
| learned linear predictor on pre-attention activations | predictor only | 93.0-97.6% exact selected set (DeepSeek-V2-Lite, Qwen3-30B, Phi-mini-MoE) | 2511.10676v1 §4 [PAPER] |
| learned multi-layer predictor with lead time | predictor only | 2.07x decode on average | ProMoE 2410.22134v3 §7 [PAPER] |
| activation-trace matching | no (traces) | 3.1-16.7x TPOT, batch 1 | MoE-Infinity 2401.14361v3 [PAPER] |
| compute misses on the CPU instead of fetching | no | 1.26x single batch | Fiddler 2402.07033v3 [PAPER] |
| profile-filled VRAM expert cache plus adaptive swaps | no | VRAM hit 0.50 with a profile, 0.72 adaptive | Strata paper §3.4 [PAPER] |

Two cautions. Prediction is not hidden latency: SSD-LLaMA (2609.18110v1)
reports that on GLM-5.2, with a one-layer lead, even a correct prefetch hides
about 5% of the expert transfer. And a profile or predictor taken on the base
model may not fit an adapted one: an attention adapter changes what the router
sees; no paper found measures that shift for an attention-only LoRA.

## 9. Strata: ideas only

`Niko1221/Strata` at `d5ea7133` (release v0.1.40.3, 2026-10-07). Ideas, in our
words, from its README and paper [DOC], [PAPER]: keep attention, routers and
shared experts resident on the GPU together with the most used routed experts;
hold all routed experts in RAM and let the CPU compute the misses while the GPU
computes the hits; fill the GPU expert cache from a profile and swap it as the
conversation goes (0.50 to 0.72 of expert uses served from VRAM); process long
prompts in chunks that stream the uncached experts of each layer to the GPU.
Its licence history is in §5. No Strata code was read for implementation or
copied.
