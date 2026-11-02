# L2L against gradient accumulation: mistralai/Mistral-7B-v0.1

**Verdict: FAIL** (l2l_differs)

L2L differs from GA in grads_equal, losses_equal; largest gradient gap 0.539062; worst [{'step': 2, 'name': 'base_model.model.model.layers.1.self_attn.v_proj.lora_A.default.weight', 'max_abs': 0.5390625}, {'step': 2, 'name': 'base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight', 'max_abs': 0.28125}, {'step': 0, 'name': 'base_model.model.model.layers.1.self_attn.v_proj.lora_A.default.weight', 'max_abs': 0.21875}]

## Command

```
/root/soup/.venv/bin/python /root/diag_dump_grads.py --out /root/runs/mistral7b-04-dump
```

## Environment

- torch 2.14.1+cu130
- transformers 5.19.0
- peft 0.21.2
- accelerate 1.15.0
- bitsandbytes 0.50.2
- numpy 2.4.6
- psutil 7.2.2
- GPU: NVIDIA GeForce RTX 3060
- Python: 3.12.3
- Platform: Linux-5.4.0-216-generic-x86_64-with-glibc2.39

## Determinism

- CUBLAS_WORKSPACE_CONFIG: :4096:8
- algorithms: strict (warn_only off)
- attn_implementation: sdpa

## SDPA backend

Pinned in both arms: MATH

- MATH: {'ok': True, 'backward_repeatable': True, 'error': None}
- FLASH_ATTENTION: {'ok': False, 'backward_repeatable': None, 'error': 'RuntimeError: No available kernel. Aborting execution.'}
- EFFICIENT_ATTENTION: {'ok': True, 'backward_repeatable': True, 'error': None}
- CUDNN_ATTENTION: {'ok': False, 'backward_repeatable': None, 'error': 'RuntimeError: No available kernel. Aborting execution.'}

## gemm_4bit path (calls by M)

- ga1: {128: {'custom': 640, 'dequant': 0}, 64: {'custom': 896, 'dequant': 0}}
- ga2: {128: {'custom': 640, 'dequant': 0}, 64: {'custom': 896, 'dequant': 0}}
- l2l: {}
- same path in GA and L2L: False
- WARNING: the arms did not use the same set of gemm_4bit paths

## RAM plan

- n_layers: 32
- hidden: 4096
- k: 4
- batch: 1
- seq: 256
- chunk_bytes: 2097152
- block_bytes: 2097152
- cache_bytes: 268435456
- need_free_for_cache: 383479222.85714287
- weights_bytes: 4124704768.0
- workspace_bytes: 1000000000
- need_total: 5508183990.857142

## bitsandbytes #2034 (recycled slot buffers)

- status: neutralized
- M=64: shipped max |d| 0.0, raw control max |d| 107.25 (neutralized)
- M=256: shipped max |d| 0.0, raw control max |d| 64.9375 (neutralized)

## Synthetic data (random tokens, real weights)

- A-vs-A (GA twice) bit-equal: True
- A-vs-A max |d|: 0.0
- L2L against GA bit-equal: False
- max |d| of LoRA gradients: 0.5390625
- optimizer steps (GA, L2L): (3, 3)

## Real data: not run
