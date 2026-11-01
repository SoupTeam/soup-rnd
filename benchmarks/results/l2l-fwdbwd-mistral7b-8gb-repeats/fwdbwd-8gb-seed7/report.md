# L2L against gradient accumulation, forward and backward: mistralai/Mistral-7B-v0.1

Card: **8gb**. Results from another card are a separate report.

**Verdict: FAIL** (l2l_out_of_tolerance)

k=2: L2L is outside the bf16 tolerance at 3 check(s); first: [{'what': 'gradient', 'step': 2, 'error': 0.04264632931887024, 'limit': 0.025110404684372797}, {'what': 'loss', 'step': 0, 'error': 2.7105096253776717e-05, 'limit': 8.733427239968695e-06}, {'what': 'loss', 'step': 2, 'error': 0.0007235127723438315, 'limit': 2.8832353988367476e-05}]

## Command

```
/root/soup/.venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 8gb --data-seed 7 --out /root/runs/fwdbwd-8gb-seed7
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
- GPU: {'name': 'NVIDIA GeForce RTX 3070', 'memory_total_mib': 8192, 'driver': '580.142'}
- Container RAM: {'source': '/sys/fs/cgroup/memory/memory.limit_in_bytes', 'limit_bytes': 52922679296, 'current_bytes': 24383778816, 'peak_bytes': 31610605568, 'swap_source': '/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes', 'swap_max_bytes': 0, 'swap_total_bytes': 7614754816, 'version': 1}
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
| 0 | gradient | 0 | 0.02164 | 0.03247 | True |
| 1 | gradient | 0 | 0.03566 | 0.05349 | True |
| 2 | gradient | 0 | 0.02233 | 0.0335 | True |
| 3 | gradient | 0 | 0.03125 | 0.04687 | True |
| 4 | gradient | 0 | 0.02779 | 0.04168 | True |
| 5 | gradient | 0 | 0.02595 | 0.03893 | True |
| 6 | gradient | 0 | 0.02678 | 0.04017 | True |
| 0 | loss | 0 | 0.0001859 | 0.0002789 | True |
| 1 | loss | 0 | 0.0003074 | 0.0004611 | True |
| 2 | loss | 0 | 0.0005935 | 0.0008903 | True |
| 3 | loss | 0 | 0.0001133 | 0.0001699 | True |
| 4 | loss | 0 | 0.0001038 | 0.0001558 | True |
| 5 | loss | 0 | 0.0003215 | 0.0004823 | True |
| 6 | loss | 0 | 2.655e-05 | 3.983e-05 | True |

## k = 2

- verdict: FAIL (l2l_out_of_tolerance)
- L2L is outside the bf16 tolerance at 3 check(s); first: [{'what': 'gradient', 'step': 2, 'error': 0.04264632931887024, 'limit': 0.025110404684372797}, {'what': 'loss', 'step': 0, 'error': 2.7105096253776717e-05, 'limit': 8.733427239968695e-06}, {'what': 'loss', 'step': 2, 'error': 0.0007235127723438315, 'limit': 2.8832353988367476e-05}]
- A against A bit-equal: True (max |d| 0.0)
- L2L against GA bit-equal: False (information, not the criterion); max |d| 0.787109375

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0.02066 | 0.02151 | 0.03227 | True |
| 1 | gradient | 0.01401 | 0.02767 | 0.0415 | True |
| 2 | gradient | 0.04265 | 0.01674 | 0.02511 | False |
| 3 | gradient | 0 | 0.02678 | 0.04017 | True |
| 0 | loss | 2.711e-05 | 5.822e-06 | 8.733e-06 | False |
| 1 | loss | 0.0001549 | 0.0003782 | 0.0005673 | True |
| 2 | loss | 0.0007235 | 1.922e-05 | 2.883e-05 | False |
| 3 | loss | 0 | 2.655e-05 | 3.983e-05 | True |

## k = 4

- verdict: FAIL (l2l_out_of_tolerance)
- L2L is outside the bf16 tolerance at 2 check(s); first: [{'what': 'gradient', 'step': 1, 'error': 0.033028722175534236, 'limit': 0.02345802111366813}, {'what': 'loss', 'step': 1, 'error': 0.0004018065441869103, 'limit': 3.373617916746932e-05}]
- A against A bit-equal: True (max |d| 0.0)
- L2L against GA bit-equal: False (information, not the criterion); max |d| 0.58056640625

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0.01853 | 0.01909 | 0.02864 | True |
| 1 | gradient | 0.03303 | 0.01564 | 0.02346 | False |
| 0 | loss | 1.856e-05 | 0.0001061 | 0.0001592 | True |
| 1 | loss | 0.0004018 | 2.249e-05 | 3.374e-05 | False |

## Counters

- gemm_4bit path calls, ga1: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- gemm_4bit path calls, ga2: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- gemm_4bit path calls, l2l: {256: {'custom': 0, 'dequant': 6944}, 192: {'custom': 0, 'dequant': 672}, 128: {'custom': 1280, 'dequant': 512}, 64: {'custom': 480, 'dequant': 192}, 230: {'custom': 0, 'dequant': 672}, 153: {'custom': 0, 'dequant': 672}}
- gemm_4bit path calls, ga_floor: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- layer loads of the first full window, L2L, by k: {1: 62, 2: 62, 4: 62}
- layer loads of the same window, GA, by k: {1: 62, 2: 122, 4: 244}
- NF4 restores in the L2L arm, by k (0 is expected after the first forward): {1: 0, 2: 0, 4: 0}
- NF4 restores over the whole run: 224

## Findings

- none
