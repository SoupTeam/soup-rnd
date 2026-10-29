# L2L against gradient accumulation: mistralai/Mistral-7B-v0.1

**Verdict: PASS** (within_tolerance)

## Command

```
/root/soup/.venv/bin/python benchmarks/harness/l2l_mistral_stand.py --dtype float32 --out /root/runs/mistral7b-07-fp32
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

- ga1: {153: {'custom': 0, 'dequant': 448}, 256: {'custom': 0, 'dequant': 896}, 192: {'custom': 0, 'dequant': 896}, 128: {'custom': 0, 'dequant': 896}, 64: {'custom': 0, 'dequant': 896}, 230: {'custom': 0, 'dequant': 448}}
- ga2: {153: {'custom': 0, 'dequant': 448}, 256: {'custom': 0, 'dequant': 896}, 192: {'custom': 0, 'dequant': 896}, 128: {'custom': 0, 'dequant': 896}, 64: {'custom': 0, 'dequant': 896}, 230: {'custom': 0, 'dequant': 448}}
- l2l: {256: {'custom': 0, 'dequant': 4704}}
- same path in GA and L2L: True

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
- M=64: shipped max |d| 0.0, raw control max |d| 108.31380462646484 (neutralized)
- M=256: shipped max |d| 0.0, raw control max |d| 65.01225280761719 (neutralized)

## Synthetic data (random tokens, real weights)

- A-vs-A (GA twice) bit-equal: True
- A-vs-A max |d|: 0.0
- L2L against GA bit-equal: False (information, not the criterion)
- max |d| of LoRA gradients: 4.458427429199219e-05
- optimizer steps (GA, L2L): (3, 3)

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step (this run)
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 2.109e-06 | None | 0.0001 | True |
| 1 | gradient | 3.134e-06 | None | 0.0001 | True |
| 2 | gradient | 8.303e-06 | None | 0.0001 | True |
| 0 | loss | 7.539e-08 | None | 0.0001 | True |
| 1 | loss | 1.554e-07 | None | 0.0001 | True |
| 2 | loss | 1.544e-07 | None | 0.0001 | True |

## Real data: not run
