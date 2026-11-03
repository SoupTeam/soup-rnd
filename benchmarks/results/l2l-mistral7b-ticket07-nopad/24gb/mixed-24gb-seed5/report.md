# L2L against gradient accumulation, forward and backward: mistralai/Mistral-7B-v0.1

Card: **24gb**. Results from another card are a separate report.

**Verdict: PASS** (within_tolerance)

## Command

```
/root/soup/.venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 24gb --data-seed 5 --out /root/runs/mixed-24gb-seed5
```

## Environment

- torch 2.14.1+cu130
- transformers 5.19.0
- peft 0.21.2
- bitsandbytes 0.50.2
- accelerate 1.15.0
- safetensors 0.8.0
- pydantic 2.13.5
- numpy 2.4.6
- psutil 7.2.2
- rich 15.0.0
- GPU: {'name': 'NVIDIA GeForce RTX 3090', 'memory_total_mib': 24576, 'driver': '580.159.03'}
- Container RAM: {'source': '/sys/fs/cgroup/memory.max', 'limit_bytes': 91749351424, 'current_bytes': 59089870848, 'peak_bytes': None, 'swap_source': '/sys/fs/cgroup/memory.swap.max', 'swap_max_bytes': 0, 'swap_total_bytes': 8589930496, 'version': 2}
- Commit: None
- Unavailable: {'commit': 'not a git checkout'}
- Quantization: nf4, dtype mode: bf16

## Synthetic rows (random tokens, real weights)

- lengths: [256, 192, 128, 64, 230, 153, 256]
- tokens labelled -100: 137
- windows by k (the last one incomplete): {1: [1, 1, 1, 1, 1, 1, 1], 2: [2, 2, 2, 1], 4: [4, 3]}
- The result is for this model on this card. No number from it carries over to larger models.

## k = 1

- verdict: PASS (within_tolerance)
- A against A bit-equal: True (max |d| 0.0)
- L2L against GA bit-equal: True (information, not the criterion); max |d| 0.0

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0 | 0.0232 | 0.03479 | True |
| 1 | gradient | 0 | 0.0163 | 0.02445 | True |
| 2 | gradient | 0 | 0.02488 | 0.03732 | True |
| 3 | gradient | 0 | 0.03569 | 0.05354 | True |
| 4 | gradient | 0 | 0.04235 | 0.06352 | True |
| 5 | gradient | 0 | 0.0194 | 0.0291 | True |
| 6 | gradient | 0 | 0.01964 | 0.02946 | True |
| 0 | loss | 0 | 0.0003041 | 0.0004561 | True |
| 1 | loss | 0 | 0.0006967 | 0.001045 | True |
| 2 | loss | 0 | 0.0002379 | 0.0003568 | True |
| 3 | loss | 0 | 0.0005766 | 0.0008649 | True |
| 4 | loss | 0 | 0.0002213 | 0.0003319 | True |
| 5 | loss | 0 | 8.52e-05 | 0.0001278 | True |
| 6 | loss | 0 | 6.375e-06 | 9.562e-06 | True |

## k = 2

- verdict: PASS (within_tolerance)
- A against A bit-equal: True (max |d| 0.0)
- L2L against GA bit-equal: True (information, not the criterion); max |d| 0.0

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0 | 0.0134 | 0.02011 | True |
| 1 | gradient | 0 | 0.02477 | 0.03715 | True |
| 2 | gradient | 0 | 0.02918 | 0.04377 | True |
| 3 | gradient | 0 | 0.01964 | 0.02946 | True |
| 0 | loss | 0 | 0.0004479 | 0.0006718 | True |
| 1 | loss | 0 | 0.0003404 | 0.0005107 | True |
| 2 | loss | 0 | 0.0001261 | 0.0001892 | True |
| 3 | loss | 0 | 6.375e-06 | 9.562e-06 | True |

## k = 4

- verdict: PASS (within_tolerance)
- A against A bit-equal: True (max |d| 0.0)
- L2L against GA bit-equal: True (information, not the criterion); max |d| 0.0

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0 | 0.01735 | 0.02602 | True |
| 1 | gradient | 0 | 0.02167 | 0.03251 | True |
| 0 | loss | 0 | 0.0004167 | 0.0006251 | True |
| 1 | loss | 0 | 6.79e-05 | 0.0001018 | True |

## Counters

- gemm_4bit path calls, ga1: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 384, 'dequant': 960}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- gemm_4bit path calls, ga2: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 384, 'dequant': 960}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- gemm_4bit path calls, l2l: {256: {'custom': 0, 'dequant': 3360}, 192: {'custom': 0, 'dequant': 2016}, 128: {'custom': 576, 'dequant': 1440}, 64: {'custom': 1440, 'dequant': 576}, 230: {'custom': 0, 'dequant': 2016}, 153: {'custom': 0, 'dequant': 2016}}
- gemm_4bit path calls, ga_floor: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 384, 'dequant': 960}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- layer loads of the first full window, L2L, by k: {1: 62, 2: 62, 4: 62}
- layer loads of the same window, GA, by k: {1: 62, 2: 122, 4: 244}
- NF4 restores in the L2L arm, by k (0 is expected after the first forward): {1: 0, 2: 0, 4: 0}
- NF4 restores over the whole run: 224

## Findings

- none
