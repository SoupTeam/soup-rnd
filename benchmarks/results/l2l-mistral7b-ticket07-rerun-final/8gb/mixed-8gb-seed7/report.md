# L2L against gradient accumulation, forward and backward: mistralai/Mistral-7B-v0.1

Card: **8gb**. Results from another card are a separate report.

**Verdict: PASS** (within_tolerance)

## Command

```
/root/soup/.venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 8gb --data-seed 7 --out /root/runs/mixed-8gb-seed7
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
- GPU: {'name': 'NVIDIA GeForce RTX 3070', 'memory_total_mib': 8192, 'driver': '610.43.02'}
- Container RAM: {'source': '/sys/fs/cgroup/memory.max', 'limit_bytes': 80060874752, 'current_bytes': 45939347456, 'peak_bytes': 46285500416, 'swap_source': '/sys/fs/cgroup/memory.swap.max', 'swap_max_bytes': 0, 'swap_total_bytes': 8589930496, 'version': 2}
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
| 0 | gradient | 0 | 0.02418 | 0.03627 | True |
| 1 | gradient | 0 | 0.03672 | 0.05508 | True |
| 2 | gradient | 0 | 0.02422 | 0.03633 | True |
| 3 | gradient | 0 | 0.03577 | 0.05365 | True |
| 4 | gradient | 0 | 0.02063 | 0.03094 | True |
| 5 | gradient | 0 | 0.02285 | 0.03428 | True |
| 6 | gradient | 0 | 0.02606 | 0.03909 | True |
| 0 | loss | 0 | 0.0001859 | 0.0002789 | True |
| 1 | loss | 0 | 0.0003074 | 0.0004611 | True |
| 2 | loss | 0 | 0.0005935 | 0.0008903 | True |
| 3 | loss | 0 | 0.0001133 | 0.0001699 | True |
| 4 | loss | 0 | 0.0001038 | 0.0001558 | True |
| 5 | loss | 0 | 0.0003215 | 0.0004823 | True |
| 6 | loss | 0 | 2.655e-05 | 3.983e-05 | True |

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
| 0 | gradient | 0 | 0.02508 | 0.03763 | True |
| 1 | gradient | 0 | 0.02694 | 0.04042 | True |
| 2 | gradient | 0 | 0.01985 | 0.02978 | True |
| 3 | gradient | 0 | 0.02606 | 0.03909 | True |
| 0 | loss | 0 | 5.822e-06 | 8.733e-06 | True |
| 1 | loss | 0 | 0.0003782 | 0.0005673 | True |
| 2 | loss | 0 | 1.922e-05 | 2.883e-05 | True |
| 3 | loss | 0 | 2.655e-05 | 3.983e-05 | True |

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
| 0 | gradient | 0 | 0.01896 | 0.02844 | True |
| 1 | gradient | 0 | 0.0155 | 0.02325 | True |
| 0 | loss | 0 | 0.0001061 | 0.0001592 | True |
| 1 | loss | 0 | 2.249e-05 | 3.374e-05 | True |

## Counters

- gemm_4bit path calls, ga1: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- gemm_4bit path calls, ga2: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- gemm_4bit path calls, l2l: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- gemm_4bit path calls, ga_floor: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- layer loads of the first full window, L2L, by k: {1: 61, 2: 61, 4: 61}
- layer loads of the same window, GA, by k: {1: 62, 2: 122, 4: 244}
- NF4 restores in the L2L arm, by k (0 is expected after the first forward): {1: 0, 2: 0, 4: 0}
- NF4 restores over the whole run: 224

## Findings

- none
