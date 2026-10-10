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
| Commit | `8634631b` plus the arm 9 device parameter in `tests/test_b2_shard_layout.py` |
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

Each cell says "proven by gate", "refused: reason" or "not tested". Arm numbers refer to the list in
`.scratch/b2-deepseek-v3-stream/spec.md`. "Proven" means the arm passed on CUDA in fp32 here and on
CPU in CI.

| Model type | Forward | Backward | NF4 | Disk tier | LoRA on attention | LoRA on shared experts | `moe_lora` on routed experts |
|---|---|---|---|---|---|---|---|
| `deepseek_v3` | proven by gate (arm 1) | proven by gate (arm 1) | proven by gate (arm 8; routed experts stay bf16) | proven by gate (arms 8 and 11) | proven by gate (arms 1 to 7) | not tested [^1] | proven by gate (arm 10) |
| `kimi_k2` (tiny shape) | proven by gate (arm 6) | proven by gate (arm 6) | not tested | not tested | proven by gate (arm 6) | not tested [^1] | not tested |
| `qwen3_5_moe` fused | proven by gate (arm 9) | not tested | not tested | not tested | proven by gate (arm 9, forward only) | not tested | not tested |
| `deepseek_v32` | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist |
| `deepseek_v2` | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist |
| `kimi_k25` | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist | refused: not on the allowlist |

[^1]: peft 0.21 rewrites `gate_proj`, `up_proj` and `down_proj` on `deepseek_v3` onto the routed
    experts, so no LoRA config reaches the shared experts. Ticket 10 did not refuse the rewrite
    (arm 10 serves it), so the cell stays "not tested".

Notes on the rows:

- `kimi_k2 (tiny shape)` is the Kimi K2 decoder shape (one expert group, one dense layer, yarn
  factor 32, rope theta 50000) built under `model_type=deepseek_v3`. A checkpoint whose
  `config.json` says `model_type: kimi_k2` is refused by the architecture gate: transformers 5.19
  does not know the config type, the tokenizer is Hub code, nothing converts Kimi K2's per-expert
  weights into fused experts, the checkpoint is FP8, and one layer is too large to stream.
- `deepseek_v32`, `deepseek_v2` and `kimi_k25` get the ordinary message,
  `layer streaming does not support model_type=<type>. Supported: ...`. The tests pin the text.
- `qwen3_5_moe` fused uses a LoRA config that targets `q_proj` and `in_proj_qkv`.
- NF4 leaves the routed experts, the router weight, the router bias and the norms at the base
  dtype. They are not quantized.
- A DeepSeek-V3 skeleton built from Hub code (`trust_remote_code: true`) is refused with a message
  to drop the flag. This is tested on CPU and is not a table column.

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
