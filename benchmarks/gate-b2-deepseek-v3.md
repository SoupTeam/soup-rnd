# Gate B2: DeepSeek-V3 in layer streaming, CUDA record

Results are synthetic. Every model here is a tiny randomly initialised network built from a config
(hidden size 32 to 16, 2 to 3 layers, vocab 64), not a released checkpoint. They show that the
streamed model equals the resident model bit for bit on this code path. They say nothing about
quality, speed or memory on a real DeepSeek-V3, Kimi K2 or Qwen3.5-MoE checkpoint.

## What was run

The parity suite from `tests/test_b2_deepseek_v3_stream.py` (parity arms 1 to 8, 10 and 11) and
`tests/test_b2_shard_layout.py` (arm 9, Qwen3.5-MoE with fused experts), on the RTX 3050 in
fp32 under deterministic algorithms. Each arm builds the resident model with `from_pretrained`
plus `get_peft_model`, builds the streamed model from the same checkpoint and adapter, and
compares logits and every LoRA gradient with `torch.equal`. Both expert kernels (`grouped_mm` and
`eager`) run where the arm names them. Each arm with a bias control also checks that a streamed
model with a zeroed router bias does not equal the resident one.

Arm 9 only compared logits on CPU before this record. It now takes a `device` parameter and runs
on CUDA too. It still compares forward only.

Command, from the repository root:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
~/.venvs/soup-rnd-cuda/bin/python -m pytest \
  tests/test_b2_deepseek_v3_stream.py tests/test_b2_shard_layout.py \
  --no-cov -v -p no:cacheprovider
```

The `deterministic` fixture also calls `torch.use_deterministic_algorithms(True)` for every
`cuda` case.

Result: 72 passed, 0 failed, 0 skipped. The 29 cases marked `gpu` (the `cuda` parameters) also
ran alone with `-m gpu`: `29 passed, 43 deselected, 16 warnings`.

## Environment

| Item | Value |
|---|---|
| Date | 2026-10-11 |
| Commit | `8634631b` plus the arm 9 device parameter in `tests/test_b2_shard_layout.py`, run before that change was committed as `60f1789d` |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB (3.86 GB free at start), compute capability 8.6 |
| Driver | 595.99.02 |
| OS | Ubuntu, kernel 7.0.0-38-generic |
| Python | 3.12.14 |
| torch | 2.14.1+cu130 (CUDA 13.0) |
| transformers | 5.19.0 |
| peft | 0.21.2 |
| bitsandbytes | 0.50.2 |
| Dtype | fp32 (NF4 arms quantize the attention and shared-expert linears) |

## GPU power state

Checked before the run with `nvidia-smi -q -d POWER`: current power limit 35.00 W, default 80.00 W,
maximum 95.00 W. This is not the 10 W cap seen on 2026-10-09. The GPU sat at 210 MHz while idle
and ran at 1740 MHz during the suite, drawing 8 to 18 W. Throttle reason bits during load were
`0x24` (software power cap and software thermal slowdown). Equality checks do not depend on clock
speed, so the throttle state cannot change a pass or a failure. It would change any timing, and
this record reports none.

Clock log, sampled every 5 seconds across both runs (`nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv -l 5`):

```text
timestamp, clocks.current.sm [MHz], power.draw [W], clocks_event_reasons.active
2026/10/11 03:58:13.882, 210 MHz, 5.87 W, 0x0000000000000001
2026/10/11 03:58:18.889, 1740 MHz, 8.27 W, 0x0000000000000024
2026/10/11 03:58:23.890, 1740 MHz, 17.72 W, 0x0000000000000024
2026/10/11 03:58:28.890, 210 MHz, 6.11 W, 0x0000000000000001
2026/10/11 03:58:33.891, 210 MHz, 6.33 W, 0x0000000000000001
2026/10/11 03:58:38.892, 1740 MHz, 18.11 W, 0x0000000000000024
2026/10/11 03:58:43.893, 1740 MHz, 17.59 W, 0x0000000000000024
```

## Operation compatibility table

Each cell says "proven by gate", "refused: reason" or "not tested". Arms: 1 base fixture, 2 `q_lora_rank=None`, 3 `rope_interleave=False`,
4 yarn rope, 5 padded batch, 6 Kimi K2 shape, 7 MTP layer, 8 NF4, 9 Qwen3.5-MoE fused experts,
10 `moe_lora`, 11 disk tier, 12 LoRA on the shared experts, 13 a tiny checkpoint whose `config.json` says `model_type: kimi_k2`. "Proven" means the arm passed on CUDA in fp32 here. Every arm except 8 also runs on CPU in CI.

| Model type | Forward | Backward | NF4 | Disk tier | LoRA on attention | LoRA on shared experts | `moe_lora` on routed experts |
|---|---|---|---|---|---|---|---|
| `deepseek_v3` | proven by gate (arm 1) | proven by gate (arm 1) | proven by gate (arm 8; routed experts stay bf16, the base dtype of the run) | proven by gate (arms 8 and 11) | proven by gate (arms 1 to 8) | proven by gate (arm 12; `attach_adapter`) [^1] | proven by gate (arm 10) |
| `kimi_k2` (tiny checkpoint, real `model_type`) | proven by gate (arms 6 and 13) | proven by gate (arms 6 and 13) | proven by gate (arm 13) | proven by gate (arm 13) | proven by gate (arms 6 and 13) | proven by gate (arm 13) [^1] | proven by gate (arm 13) |
| `qwen3_5_moe` fused | proven by gate (arm 9, logits only) | not tested | not tested | not tested | not tested [^2] | not tested | not tested |
| `deepseek_v32` | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist |
| `deepseek_v2` | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist |
| `kimi_k25` | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist |

[^1]: peft 0.21 rewrites `gate_proj`, `up_proj` and `down_proj` on `deepseek_v3` onto the routed
    experts, so a plain `get_peft_model` never reaches the shared experts.
    `peft_wiring.attach_adapter` resolves shared-expert targets to full module paths and skips the
    rewrite for that one config (ticket 15). Arm 12 is `tests/test_b2_shared_expert_lora.py`: the
    adapters sit on exactly `mlp.shared_experts.{gate,up,down}_proj` of every MoE layer, streamed
    logits equal resident logits, and all 12 `lora_A` / `lora_B` gradients are non-zero and equal,
    on the RAM and disk tiers and with NF4, on CPU (not NF4) and on CUDA in fp32, with both expert
    implementations. Two controls show the check can fail: nudging one `lora_B` changes logits and
    gradients, and plain peft builds a model with different logits. `load_adapter` reloads a saved
    shared-expert adapter onto the shared experts, and a routed-expert adapter loads as peft loads it.
    The rewrite is pinned by a test, so a peft release that changes it fails CI. Not covered: a real
    checkpoint.

[^2]: Arm 9 uses a LoRA config on `q_proj` and `in_proj_qkv` with `lora_B` left at zero, so the
    adapter adds nothing and the logits comparison says nothing about LoRA.

Notes on the rows:

- `kimi_k2` is admitted as an alias of `deepseek_v3` (ticket 16). Arm 6 is the Kimi K2 decoder
  shape (one expert group, one dense layer, yarn factor 32, rope theta 50000) built under
  `model_type=deepseek_v3`. Arm 13 is `tests/test_b2_kimi_k2.py`: a tiny checkpoint whose
  `config.json` says `model_type: kimi_k2`, with per-expert keys, a spread router bias and the
  per-layer `self_attn.rotary_emb.inv_freq` keys K2 checkpoints carry. It loads through
  `soup_cli.utils.kimi_k2.load_stream_config` (no Hub code), and the conversion registered under
  `kimi_k2` makes the load complete with the same logits as the `deepseek_v3` twin. Without the
  registration the routed experts stay at init, which a control test shows. A save writes
  per-expert keys back. The sharder drops the `inv_freq` keys. Streamed and resident logits and
  all LoRA gradients are equal on RAM and disk tiers, with NF4, for attention LoRA, shared-expert
  LoRA and `moe_lora`, on CPU (not NF4) and on CUDA in fp32, with both expert implementations.
  A zero router bias changes the logits, so the check can fail.
- Two real-config facts the tests check. The native classes never read `aux_loss_alpha` or
  `seq_aux`, and under the frozen router changing them changes no loss and no gradient. Moonshot's
  router masks excluded groups with `0.0`; with one group it selects the same experts as the native
  `-inf` mask for biases from -3 to +2, and with four groups and a negative bias the two masks differ.
- `kimi_k2` is read from `config.json` fields recorded in `research.md`, not from a downloaded
  file. Not covered: a real K2 checkpoint (about 1 TB, 61 shards), one real MoE layer (about 17 GB
  in FP8, so a real run needs sub-layer streaming, B1), and K2-Thinking or K2.5 (INT4 `compressed-tensors`, refused by name).
  `kimi_k25` stays refused although its `text_config.model_type` is `kimi_k2`.
- The K2 tokenizer is built natively from `tiktoken.model`. Moonshot's `tokenization_kimi.py` ran
  once (2026-10-15, Kimi-K2-Instruct revision `fd1984e2b7a3`, transformers 4.55.4, tiktoken 0.14.0,
  throwaway venv, files in their own empty directory, `python -I`, no weights). The ids it gave for 59
  texts (ASCII, CJK, code, emoji, every special token) and 6 rendered chat templates (system default,
  multi-turn, tool call and tool result) are the fixture `tests/fixtures/kimi_k2_tokenizer_reference.json`,
  with the SHA-256 of the four source files. All match. The parity tests skip unless
  `SOUP_K2_TOKENIZER_DIR` holds those files, so CI does not run them. CI does pin the split pattern's
  hash, which catches an edit to it but nothing else. The native tokenizer is used for a local
  folder only; a Hub id reaches `AutoTokenizer` and fails with its Hub-code error. `soup train
  --find-lr` still loads a tokenizer and model through `AutoTokenizer` and `AutoModelForCausalLM`. Not covered: K2-Base's own tokenizer files (not
  downloaded; compare their `tiktoken.model` hash to the fixture's), and texts beyond the fixed set.
- The `kimi_k2` config reader and conversion registration run on the layer-streaming path only
  (`stream_setup` and `build_meta_skeleton`). `export`, `merge`, `serve`, `soup train` without
  `training.stream_layers` and `soup recipes verify` refuse a local `kimi_k2` folder by name before
  loading (ticket 19). A Hub id, `soup shrink` and `soup draft` are not guarded. `register_kimi_k2_conversion`
  edits a private transformers table and PEFT's import-time copy of it; a test pins both, so a release
  that moves them fails CI.
- `deepseek_v32`, `deepseek_v2` and `kimi_k25` get the ordinary message,
  `layer streaming does not support model_type=<type>. Supported: ...`. The tests pin the text.
- `qwen3_5_moe` fused compares logits only. It has no expert-coverage check, no `meta` check and no gradient comparison.
- Arm 8 (NF4) runs on CUDA only, so it has no CPU run in CI.
- NF4 leaves the routed experts, the router weight, the router bias and the norms at the base
  dtype. They are not quantized.
- Store size under NF4 (computed from the released config, not measured). A BF16 DeepSeek-V3
  store is about 1.32 TB, because the 58 MoE layers x 256 experts x 3 x 7168 x 2048 = 653.9 B
  routed parameters stay at 2 bytes each (1.308 TB). Quantizing them too, at 0.5625 byte per
  parameter, would give about 380 GB. Any projection that assumes a 380 GB store is conditional
  until packed routed experts exist and pass a parity test. The ratio between the two stores
  (about 3.47x in bytes) is not a measured slowdown. The parity and dense-70B measurements keep
  their original scope. Packed routed experts are tracked with B1. The contract needs packed weights
  and metadata, a dequant and GEMM consumer, cache identity and migration, reference forward and
  backward controls, and a preflight that uses the real source dtype and the set of quantized
  parameters. Adding suffix names alone is not enough.
- A DeepSeek-V3 skeleton built from Hub code (`trust_remote_code: true`) is refused with a message
  to drop the flag. This is tested on CPU and is not a table column.

## Rerun after merging main

2026-10-12, commit `97c36e3a` (the cache layout check, `main` with B3 FP8 decoding and B4 adapter
preflight merged), same machine, venv and versions as above. Power limit 80.00 W. The command
adds `tests/test_b2_b3_integration.py` (FP8 per-expert keys decoded then stacked, bit-exact
against B3's independent decoder; cache rebuilt on a layer count change):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 ~/.venvs/soup-rnd-cuda/bin/python -m pytest \
  tests/test_b2_deepseek_v3_stream.py tests/test_b2_shard_layout.py \
  tests/test_b2_b3_integration.py --no-cov -q -p no:cacheprovider
```

Result: 75 passed, 0 failed. With `-k cuda`: 29 passed, 46 deselected. The FP8 integration
tests run on CPU only. The raw output below is from the first run, before the merge.

## Raw pytest output

Full run, `tests/test_b2_deepseek_v3_stream.py` and `tests/test_b2_shard_layout.py`, CUDA venv:

```text
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- /home/m1kadoo/.venvs/soup-rnd-cuda/bin/python
rootdir: /home/m1kadoo/Documents/soup-rnd
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.15.1
collecting ... collected 72 items

tests/test_b2_deepseek_v3_stream.py::test_deepseek_v3_is_accepted_under_its_own_name PASSED [  1%]
tests/test_b2_deepseek_v3_stream.py::test_kimi_k2_is_refused_naming_what_is_missing PASSED [  2%]
tests/test_b2_deepseek_v3_stream.py::test_other_deepseek_and_kimi_types_get_the_ordinary_refusal[deepseek_v32] PASSED [  4%]
tests/test_b2_deepseek_v3_stream.py::test_other_deepseek_and_kimi_types_get_the_ordinary_refusal[deepseek_v2] PASSED [  5%]
tests/test_b2_deepseek_v3_stream.py::test_other_deepseek_and_kimi_types_get_the_ordinary_refusal[kimi_k25] PASSED [  6%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_matches_resident[cpu-grouped_mm] PASSED [  8%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_matches_resident[cpu-eager] PASSED [  9%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_matches_resident[cuda-grouped_mm] PASSED [ 11%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_matches_resident[cuda-eager] PASSED [ 12%]
tests/test_b2_deepseek_v3_stream.py::test_control_streamed_model_with_a_zero_router_bias_differs[cpu] PASSED [ 13%]
tests/test_b2_deepseek_v3_stream.py::test_control_streamed_model_with_a_zero_router_bias_differs[cuda] PASSED [ 15%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-grouped_mm-arm2-q-lora-rank-none] PASSED [ 16%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-grouped_mm-arm3-rope-not-interleaved] PASSED [ 18%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-grouped_mm-arm4-yarn-factor-4] PASSED [ 19%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-grouped_mm-arm5-padded-batch] PASSED [ 20%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-grouped_mm-arm6-kimi-k2-shape] PASSED [ 22%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-eager-arm2-q-lora-rank-none] PASSED [ 23%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-eager-arm3-rope-not-interleaved] PASSED [ 25%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-eager-arm4-yarn-factor-4] PASSED [ 26%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-eager-arm5-padded-batch] PASSED [ 27%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cpu-eager-arm6-kimi-k2-shape] PASSED [ 29%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-grouped_mm-arm2-q-lora-rank-none] PASSED [ 30%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-grouped_mm-arm3-rope-not-interleaved] PASSED [ 31%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-grouped_mm-arm4-yarn-factor-4] PASSED [ 33%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-grouped_mm-arm5-padded-batch] PASSED [ 34%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-grouped_mm-arm6-kimi-k2-shape] PASSED [ 36%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-eager-arm2-q-lora-rank-none] PASSED [ 37%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-eager-arm3-rope-not-interleaved] PASSED [ 38%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-eager-arm4-yarn-factor-4] PASSED [ 40%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-eager-arm5-padded-batch] PASSED [ 41%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_variant_matches_resident[cuda-eager-arm6-kimi-k2-shape] PASSED [ 43%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cpu-arm2-q-lora-rank-none] PASSED [ 44%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cpu-arm3-rope-not-interleaved] PASSED [ 45%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cpu-arm4-yarn-factor-4] PASSED [ 47%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cpu-arm5-padded-batch] PASSED [ 48%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cpu-arm6-kimi-k2-shape] PASSED [ 50%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cuda-arm2-q-lora-rank-none] PASSED [ 51%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cuda-arm3-rope-not-interleaved] PASSED [ 52%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cuda-arm4-yarn-factor-4] PASSED [ 54%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cuda-arm5-padded-batch] PASSED [ 55%]
tests/test_b2_deepseek_v3_stream.py::test_control_variant_with_a_zero_router_bias_differs[cuda-arm6-kimi-k2-shape] PASSED [ 56%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_from_the_disk_tier_matches_resident[cpu-grouped_mm] PASSED [ 58%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_from_the_disk_tier_matches_resident[cpu-eager] PASSED [ 59%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_from_the_disk_tier_matches_resident[cuda-grouped_mm] PASSED [ 61%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_from_the_disk_tier_matches_resident[cuda-eager] PASSED [ 62%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-ram-grouped_mm] PASSED [ 63%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-ram-eager] PASSED [ 65%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-disk-grouped_mm] PASSED [ 66%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-disk-eager] PASSED [ 68%]
tests/test_b2_deepseek_v3_stream.py::test_an_mtp_layer_is_dropped_and_the_streamed_model_still_matches[cpu-grouped_mm] PASSED [ 69%]
tests/test_b2_deepseek_v3_stream.py::test_an_mtp_layer_is_dropped_and_the_streamed_model_still_matches[cpu-eager] PASSED [ 70%]
tests/test_b2_deepseek_v3_stream.py::test_an_mtp_layer_is_dropped_and_the_streamed_model_still_matches[cuda-grouped_mm] PASSED [ 72%]
tests/test_b2_deepseek_v3_stream.py::test_an_mtp_layer_is_dropped_and_the_streamed_model_still_matches[cuda-eager] PASSED [ 73%]
tests/test_b2_deepseek_v3_stream.py::test_a_deepseek_v3_class_from_hub_code_is_refused[hub-class] PASSED [ 75%]
tests/test_b2_deepseek_v3_stream.py::test_a_deepseek_v3_class_from_hub_code_is_refused[hub-class-that-fails-to-import] PASSED [ 76%]
tests/test_b2_deepseek_v3_stream.py::test_the_same_fixture_without_the_hub_class_builds[auto_map-without-remote-code] PASSED [ 77%]
tests/test_b2_deepseek_v3_stream.py::test_the_same_fixture_without_the_hub_class_builds[remote-code-without-auto_map] PASSED [ 79%]
tests/test_b2_deepseek_v3_stream.py::test_the_refusal_fires_before_any_shard_is_read PASSED [ 80%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cpu-grouped_mm] PASSED [ 81%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cpu-eager] PASSED [ 83%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cuda-grouped_mm] PASSED [ 84%]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cuda-eager] PASSED [ 86%]
tests/test_b2_deepseek_v3_stream.py::test_fused_expert_targets_the_shards_carry_are_accepted[explicit-targets] PASSED [ 87%]
tests/test_b2_deepseek_v3_stream.py::test_fused_expert_targets_the_shards_carry_are_accepted[moe_lora] PASSED [ 88%]
tests/test_b2_deepseek_v3_stream.py::test_explicit_expert_targets_are_refused_when_the_shards_do_not_carry_fused_experts PASSED [ 90%]
tests/test_b2_deepseek_v3_stream.py::test_a_parameter_target_outside_the_decoder_layers_is_refused PASSED [ 91%]
tests/test_b2_deepseek_v3_stream.py::test_a_skeleton_without_a_model_type_is_left_to_peft PASSED [ 93%]
tests/test_b2_deepseek_v3_stream.py::test_the_training_setup_checks_the_lora_config_it_built PASSED [ 94%]
tests/test_b2_shard_layout.py::test_fused_qwen35_moe_from_per_expert_keys_matches_resident[cpu] PASSED [ 95%]
tests/test_b2_shard_layout.py::test_fused_qwen35_moe_from_per_expert_keys_matches_resident[cuda] PASSED [ 97%]
tests/test_b2_shard_layout.py::test_keys_beyond_the_skeletons_layers_are_dropped_and_counted PASSED [ 98%]
tests/test_b2_shard_layout.py::test_per_expert_keys_stay_when_the_skeleton_has_no_fused_experts PASSED [100%]

=============================== warnings summary ===============================
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-ram-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-ram-eager]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-disk-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-disk-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:957: UserWarning: inner dimension (32) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
    warn(

tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-ram-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-ram-eager]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-disk-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-disk-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:957: UserWarning: inner dimension (24) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
    warn(

tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-ram-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-ram-eager]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-disk-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_nf4_matches_resident_nf4[cuda-disk-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:957: UserWarning: inner dimension (16) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
    warn(

tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cpu-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cpu-eager]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cuda-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cuda-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/peft/tuners/tuners_utils.py:1123: RuntimeWarning: The following rank_pattern keys did not match any targeted module and were ignored: ['.*\\.gate_up_proj'].
    warnings.warn(

tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cpu-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cpu-eager]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cuda-grouped_mm]
tests/test_b2_deepseek_v3_stream.py::test_streamed_deepseek_v3_moe_lora_matches_resident[cuda-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/peft/tuners/tuners_utils.py:1123: RuntimeWarning: The following alpha_pattern keys did not match any targeted module and were ignored: ['.*\\.gate_up_proj'].
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 72 passed, 20 warnings in 13.46s =======================
```

## CUDA run: shared-expert LoRA and Kimi K2 modules

2026-10-15, commit `3482d9cc` (clean tree), same machine and venv as above. Power limit checked
first: current 80.00 W, default 80.00 W, so no cap. The run is the CUDA evidence for the
`tests/test_b2_shared_expert_lora.py` and `tests/test_b2_kimi_k2.py` rows. CPU CI does not count
for those rows.

Command, from the repository root:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$PWD
~/.venvs/soup-rnd-cuda/bin/python -m pytest \
  tests/test_b2_shared_expert_lora.py tests/test_b2_kimi_k2.py \
  -m "gpu or not gpu" --no-cov -v -p no:cacheprovider
```

| Item | Value |
|---|---|
| Driver | 595.99.02 |
| torch | 2.14.1+cu130 |
| transformers | 5.19.0 |
| peft | 0.21.2 |
| soup_cli import path | this checkout (`src/soup_cli/__init__.py`) |

Result: 79 passed, 0 failed, 0 skipped, 0 errors. 50 of the 79 cases are `cuda` parameters
(21 in `test_b2_shared_expert_lora.py`, 29 in `test_b2_kimi_k2.py`).

Clock log, sampled every 5 seconds. It holds 4 samples spanning 15 seconds, because the sampler process died early. The suite itself took 12.33 s. The GPU reached 1740 MHz with no throttle bits (`0x0`) under load:

```text
timestamp, clocks.current.sm [MHz], power.draw [W], clocks_event_reasons.active
2026/10/15 23:31:30.176, 210 MHz, 3.39 W, 0x0000000000000001
2026/10/15 23:31:35.185, 210 MHz, 3.51 W, 0x0000000000000001
2026/10/15 23:31:40.186, 1740 MHz, 18.61 W, 0x0000000000000000
2026/10/15 23:31:45.187, 1740 MHz, 18.55 W, 0x0000000000000000
```

### What this run does not cover

- No real checkpoint. Every model is a tiny randomly initialised network built from a config.
- No tokenizer. The Kimi K2 tokenizer parity is in `tests/test_b2_kimi_k2_tokenizer.py`, which this run did not include.
- `tests/test_b2_shared_expert_adapter_consumers.py` and `tests/test_b2_kimi_k2_outside_streaming.py` were not run. Neither file has a `cuda` or `gpu` case (`grep -cE "cuda|gpu"` returns 0 for both), so there is no CUDA evidence to collect from them.
- No quality, speed or memory result.

### Raw pytest output for this run

```text
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- /home/m1kadoo/.venvs/soup-rnd-cuda/bin/python
rootdir: /home/m1kadoo/Documents/soup-rnd
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.15.1
collecting ... collected 79 items

tests/test_b2_shared_expert_lora.py::test_plain_peft_moves_shared_expert_targets_onto_the_routed_experts PASSED [  1%]
tests/test_b2_shared_expert_lora.py::test_shared_expert_targets_attach_to_exactly_the_shared_experts PASSED [  2%]
tests/test_b2_shared_expert_lora.py::test_a_list_of_shared_expert_names_attaches_the_same_way PASSED [  3%]
tests/test_b2_shared_expert_lora.py::test_attention_and_shared_experts_attach_together_and_nothing_else PASSED [  5%]
tests/test_b2_shared_expert_lora.py::test_a_bare_projection_name_beside_shared_experts_is_refused PASSED [  6%]
tests/test_b2_shared_expert_lora.py::test_the_peft_conversion_is_restored_after_the_attach PASSED [  7%]
tests/test_b2_shared_expert_lora.py::test_a_config_with_no_shared_expert_target_goes_through_plain_peft PASSED [  8%]
tests/test_b2_shared_expert_lora.py::test_streamed_model_attaches_shared_expert_lora_to_the_shared_experts PASSED [ 10%]
tests/test_b2_shared_expert_lora.py::test_streamed_shared_expert_lora_matches_resident[cpu-grouped_mm] PASSED [ 11%]
tests/test_b2_shared_expert_lora.py::test_streamed_shared_expert_lora_matches_resident[cpu-eager] PASSED [ 12%]
tests/test_b2_shared_expert_lora.py::test_streamed_shared_expert_lora_matches_resident[cuda-grouped_mm] PASSED [ 13%]
tests/test_b2_shared_expert_lora.py::test_streamed_shared_expert_lora_matches_resident[cuda-eager] PASSED [ 15%]
tests/test_b2_shared_expert_lora.py::test_streamed_shared_expert_lora_from_the_disk_tier_matches_resident[cpu-grouped_mm] PASSED [ 16%]
tests/test_b2_shared_expert_lora.py::test_streamed_shared_expert_lora_from_the_disk_tier_matches_resident[cpu-eager] PASSED [ 17%]
tests/test_b2_shared_expert_lora.py::test_streamed_shared_expert_lora_from_the_disk_tier_matches_resident[cuda-grouped_mm] PASSED [ 18%]
tests/test_b2_shared_expert_lora.py::test_streamed_shared_expert_lora_from_the_disk_tier_matches_resident[cuda-eager] PASSED [ 20%]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-ram-grouped_mm] PASSED [ 21%]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-ram-eager] PASSED [ 22%]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-disk-grouped_mm] PASSED [ 24%]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-disk-eager] PASSED [ 25%]
tests/test_b2_shared_expert_lora.py::test_control_a_perturbed_shared_expert_adapter_changes_logits_and_gradients[cpu] PASSED [ 26%]
tests/test_b2_shared_expert_lora.py::test_control_a_perturbed_shared_expert_adapter_changes_logits_and_gradients[cuda] PASSED [ 27%]
tests/test_b2_shared_expert_lora.py::test_control_plain_peft_gives_different_logits_for_the_same_config PASSED [ 29%]
tests/test_b2_shared_expert_lora.py::test_a_saved_shared_expert_adapter_reloads_onto_the_shared_experts PASSED [ 30%]
tests/test_b2_shared_expert_lora.py::test_an_existing_routed_expert_adapter_loads_the_way_peft_loads_it PASSED [ 31%]
tests/test_b2_shared_expert_lora.py::test_the_peft_conversion_is_restored_after_a_reload PASSED [ 32%]
tests/test_b2_shared_expert_lora.py::test_the_adapter_plan_counts_the_shared_expert_adapters_not_the_routed_ones PASSED [ 34%]
tests/test_b2_shared_expert_lora.py::test_a_regex_naming_attention_and_shared_experts_keeps_both PASSED [ 35%]
tests/test_b2_shared_expert_lora.py::test_routed_expert_adapters_the_config_did_not_ask_for_fail_the_attach PASSED [ 36%]
tests/test_b2_shared_expert_lora.py::test_moe_lora_beside_explicit_shared_expert_targets_is_refused PASSED [ 37%]
tests/test_b2_shared_expert_lora.py::test_moe_lora_with_auto_targets_is_unchanged PASSED [ 39%]
tests/test_b2_shared_expert_lora.py::test_a_saved_adapter_mixing_shared_names_and_a_bare_projection_is_refused PASSED [ 40%]
tests/test_b2_shared_expert_lora.py::test_serve_loads_every_named_shared_expert_adapter_onto_the_shared_experts PASSED [ 41%]
tests/test_b2_kimi_k2.py::test_native_autoconfig_cannot_read_a_k2_config PASSED [ 43%]
tests/test_b2_kimi_k2.py::test_real_k2_config_identities_load_without_hub_code PASSED [ 44%]
tests/test_b2_kimi_k2.py::test_k2_instruct_0905_style_rope_and_context_are_read_not_defaulted PASSED [ 45%]
tests/test_b2_kimi_k2.py::test_other_model_types_still_go_through_autoconfig PASSED [ 46%]
tests/test_b2_kimi_k2.py::test_a_k2_config_with_trust_remote_code_is_refused PASSED [ 48%]
tests/test_b2_kimi_k2.py::test_the_aux_loss_fields_are_recorded_as_ignored PASSED [ 49%]
tests/test_b2_kimi_k2.py::test_registering_the_conversion_gives_kimi_k2_the_deepseek_v3_rules PASSED [ 50%]
tests/test_b2_kimi_k2.py::test_registering_twice_is_a_no_op_and_touches_no_other_model_type PASSED [ 51%]
tests/test_b2_kimi_k2.py::test_registering_does_not_replace_a_mapping_someone_else_registered PASSED [ 53%]
tests/test_b2_kimi_k2.py::test_a_kimi_k2_checkpoint_loads_complete_with_the_deepseek_v3_logits PASSED [ 54%]
tests/test_b2_kimi_k2.py::test_without_the_conversion_a_kimi_k2_load_leaves_the_experts_at_init PASSED [ 55%]
tests/test_b2_kimi_k2.py::test_saving_a_kimi_k2_model_writes_per_expert_keys_and_reloads_identically PASSED [ 56%]
tests/test_b2_kimi_k2.py::test_the_aux_loss_fields_change_nothing_under_a_frozen_router PASSED [ 58%]
tests/test_b2_kimi_k2.py::test_the_router_selects_the_same_experts_as_moonshots_zero_mask_with_one_group[0.0] PASSED [ 59%]
tests/test_b2_kimi_k2.py::test_the_router_selects_the_same_experts_as_moonshots_zero_mask_with_one_group[-3.0] PASSED [ 60%]
tests/test_b2_kimi_k2.py::test_the_router_selects_the_same_experts_as_moonshots_zero_mask_with_one_group[2.0] PASSED [ 62%]
tests/test_b2_kimi_k2.py::test_the_zero_mask_does_differ_when_groups_are_excluded PASSED [ 63%]
tests/test_b2_kimi_k2.py::test_kimi_k2_streams_as_a_deepseek_v3_alias PASSED [ 64%]
tests/test_b2_kimi_k2.py::test_the_kimi_k25_wrapper_stays_refused_although_its_text_tower_is_kimi_k2 PASSED [ 65%]
tests/test_b2_kimi_k2.py::test_an_int4_kimi_k2_checkpoint_is_refused_naming_the_remaining_limits PASSED [ 67%]
tests/test_b2_kimi_k2.py::test_a_block_fp8_kimi_k2_config_is_not_refused PASSED [ 68%]
tests/test_b2_kimi_k2.py::test_the_sharder_drops_the_rotary_keys_k2_checkpoints_carry PASSED [ 69%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_matches_resident[cpu-grouped_mm] PASSED [ 70%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_matches_resident[cpu-eager] PASSED [ 72%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_matches_resident[cuda-grouped_mm] PASSED [ 73%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_matches_resident[cuda-eager] PASSED [ 74%]
tests/test_b2_kimi_k2.py::test_control_a_kimi_k2_stream_with_a_zero_router_bias_differs[cpu] PASSED [ 75%]
tests/test_b2_kimi_k2.py::test_control_a_kimi_k2_stream_with_a_zero_router_bias_differs[cuda] PASSED [ 77%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_from_the_disk_tier_matches_resident[cpu-grouped_mm] PASSED [ 78%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_from_the_disk_tier_matches_resident[cpu-eager] PASSED [ 79%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_from_the_disk_tier_matches_resident[cuda-grouped_mm] PASSED [ 81%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_from_the_disk_tier_matches_resident[cuda-eager] PASSED [ 82%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-ram-grouped_mm] PASSED [ 83%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-ram-eager] PASSED [ 84%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-disk-grouped_mm] PASSED [ 86%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-disk-eager] PASSED [ 87%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cpu-grouped_mm] PASSED [ 88%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cpu-eager] PASSED [ 89%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cuda-grouped_mm] PASSED [ 91%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cuda-eager] PASSED [ 92%]
tests/test_b2_kimi_k2.py::test_peft_rewrites_expert_targets_onto_the_fused_experts_only_once_registered PASSED [ 93%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_shared_expert_lora_matches_resident[cpu-grouped_mm] PASSED [ 94%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_shared_expert_lora_matches_resident[cpu-eager] PASSED [ 96%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_shared_expert_lora_matches_resident[cuda-grouped_mm] PASSED [ 97%]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_shared_expert_lora_matches_resident[cuda-eager] PASSED [ 98%]
tests/test_b2_kimi_k2.py::test_an_int4_text_config_under_an_unknown_wrapper_is_refused_too PASSED [100%]

=============================== warnings summary ===============================
tests/test_b2_shared_expert_lora.py::test_plain_peft_moves_shared_expert_targets_onto_the_routed_experts
tests/test_b2_shared_expert_lora.py::test_control_plain_peft_gives_different_logits_for_the_same_config
tests/test_b2_shared_expert_lora.py::test_an_existing_routed_expert_adapter_loads_the_way_peft_loads_it
tests/test_b2_shared_expert_lora.py::test_routed_expert_adapters_the_config_did_not_ask_for_fail_the_attach
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cpu-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cpu-eager]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cuda-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cuda-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/peft/tuners/tuners_utils.py:1123: RuntimeWarning: The following rank_pattern keys did not match any targeted module and were ignored: ['.*\\.gate_up_proj'].
    warnings.warn(

tests/test_b2_shared_expert_lora.py::test_plain_peft_moves_shared_expert_targets_onto_the_routed_experts
tests/test_b2_shared_expert_lora.py::test_control_plain_peft_gives_different_logits_for_the_same_config
tests/test_b2_shared_expert_lora.py::test_an_existing_routed_expert_adapter_loads_the_way_peft_loads_it
tests/test_b2_shared_expert_lora.py::test_routed_expert_adapters_the_config_did_not_ask_for_fail_the_attach
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cpu-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cpu-eager]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cuda-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_moe_lora_matches_resident[cuda-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/peft/tuners/tuners_utils.py:1123: RuntimeWarning: The following alpha_pattern keys did not match any targeted module and were ignored: ['.*\\.gate_up_proj'].
    warnings.warn(

tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-ram-grouped_mm]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-ram-eager]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-disk-grouped_mm]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-disk-eager]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-ram-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-ram-eager]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-disk-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-disk-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:957: UserWarning: inner dimension (32) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
    warn(

tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-ram-grouped_mm]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-ram-eager]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-disk-grouped_mm]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-disk-eager]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-ram-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-ram-eager]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-disk-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-disk-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:957: UserWarning: inner dimension (24) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
    warn(

tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-ram-grouped_mm]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-ram-eager]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-disk-grouped_mm]
tests/test_b2_shared_expert_lora.py::test_streamed_nf4_shared_expert_lora_matches_resident_nf4[cuda-disk-eager]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-ram-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-ram-eager]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-disk-grouped_mm]
tests/test_b2_kimi_k2.py::test_streamed_kimi_k2_nf4_matches_resident_nf4[cuda-disk-eager]
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:957: UserWarning: inner dimension (16) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
    warn(

tests/test_b2_shared_expert_lora.py::test_serve_loads_every_named_shared_expert_adapter_onto_the_shared_experts
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/typer/__init__.py:24: DeprecationWarning: 'click.utils.get_binary_stream' is deprecated and will be removed in Click 9.0.
    from click.utils import get_binary_stream as get_binary_stream

tests/test_b2_shared_expert_lora.py::test_serve_loads_every_named_shared_expert_adapter_onto_the_shared_experts
  /home/m1kadoo/.venvs/soup-rnd-cuda/lib/python3.12/site-packages/typer/__init__.py:25: DeprecationWarning: 'click.utils.get_text_stream' is deprecated and will be removed in Click 9.0.
    from click.utils import get_text_stream as get_text_stream

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 79 passed, 42 warnings in 12.33s =======================
```
