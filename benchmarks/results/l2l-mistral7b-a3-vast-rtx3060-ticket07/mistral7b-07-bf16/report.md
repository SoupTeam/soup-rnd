# L2L against gradient accumulation: mistralai/Mistral-7B-v0.1

**Verdict: PASS** (within_tolerance)

## Command

```
/root/soup/.venv/bin/python benchmarks/harness/l2l_mistral_stand.py --out /root/runs/mistral7b-07-bf16
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

- ga1: {153: {'custom': 0, 'dequant': 448}, 256: {'custom': 0, 'dequant': 896}, 192: {'custom': 0, 'dequant': 896}, 128: {'custom': 640, 'dequant': 256}, 64: {'custom': 896, 'dequant': 0}, 230: {'custom': 0, 'dequant': 448}}
- ga2: {153: {'custom': 0, 'dequant': 448}, 256: {'custom': 0, 'dequant': 896}, 192: {'custom': 0, 'dequant': 896}, 128: {'custom': 640, 'dequant': 256}, 64: {'custom': 896, 'dequant': 0}, 230: {'custom': 0, 'dequant': 448}}
- l2l: {256: {'custom': 0, 'dequant': 4704}}
- ga_floor: {153: {'custom': 0, 'dequant': 448}, 256: {'custom': 0, 'dequant': 896}, 192: {'custom': 0, 'dequant': 896}, 128: {'custom': 640, 'dequant': 256}, 64: {'custom': 896, 'dequant': 0}, 230: {'custom': 0, 'dequant': 448}}
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
- L2L against GA bit-equal: False (information, not the criterion)
- max |d| of LoRA gradients: 0.5390625
- optimizer steps (GA, L2L): (3, 3)

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0.01797 | 0.0144 | 0.0216 | True |
| 1 | gradient | 0.0162 | 0.01543 | 0.02314 | True |
| 2 | gradient | 0.05304 | 0.04493 | 0.0674 | True |
| 0 | loss | 3.903e-05 | 9.779e-05 | 0.0001467 | True |
| 1 | loss | 6.841e-05 | 7.71e-05 | 0.0001157 | True |
| 2 | loss | 0.000367 | 0.0006736 | 0.00101 | True |

## Real data: not run
