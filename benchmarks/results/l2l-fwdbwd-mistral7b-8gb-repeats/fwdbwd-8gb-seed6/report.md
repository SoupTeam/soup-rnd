# L2L against gradient accumulation, forward and backward: mistralai/Mistral-7B-v0.1

Card: **8gb**. Results from another card are a separate report.

**Verdict: FAIL** (l2l_out_of_tolerance)

k=2: L2L is outside the bf16 tolerance at 1 check(s); first: [{'what': 'gradient', 'step': 2, 'error': 0.041465009996170546, 'limit': 0.04086172201081654}]

## Command

```
/root/soup/.venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 8gb --data-seed 6 --out /root/runs/fwdbwd-8gb-seed6
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
- Container RAM: {'source': '/sys/fs/cgroup/memory/memory.limit_in_bytes', 'limit_bytes': 52922679296, 'current_bytes': 31557787648, 'peak_bytes': 31565119488, 'swap_source': '/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes', 'swap_max_bytes': 0, 'swap_total_bytes': 7614754816, 'version': 1}
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
| 0 | gradient | 0 | 0.0238 | 0.03571 | True |
| 1 | gradient | 0 | 0.01569 | 0.02354 | True |
| 2 | gradient | 0 | 0.02212 | 0.03317 | True |
| 3 | gradient | 0 | 0.02914 | 0.04371 | True |
| 4 | gradient | 0 | 0.02722 | 0.04083 | True |
| 5 | gradient | 0 | 0.03185 | 0.04777 | True |
| 6 | gradient | 0 | 0.03241 | 0.04862 | True |
| 0 | loss | 0 | 0.0001289 | 0.0001934 | True |
| 1 | loss | 0 | 6.92e-07 | 1.038e-06 | True |
| 2 | loss | 0 | 0.0001187 | 0.000178 | True |
| 3 | loss | 0 | 0.0007249 | 0.001087 | True |
| 4 | loss | 0 | 0.0004793 | 0.000719 | True |
| 5 | loss | 0 | 0.0004659 | 0.0006988 | True |
| 6 | loss | 0 | 1.835e-05 | 2.752e-05 | True |

## k = 2

- verdict: FAIL (l2l_out_of_tolerance)
- L2L is outside the bf16 tolerance at 1 check(s); first: [{'what': 'gradient', 'step': 2, 'error': 0.041465009996170546, 'limit': 0.04086172201081654}]
- A against A bit-equal: True (max |d| 0.0)
- L2L against GA bit-equal: False (information, not the criterion); max |d| 0.3515625

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0.01488 | 0.02156 | 0.03234 | True |
| 1 | gradient | 0.01597 | 0.02109 | 0.03164 | True |
| 2 | gradient | 0.04147 | 0.02724 | 0.04086 | False |
| 3 | gradient | 0 | 0.03241 | 0.04862 | True |
| 0 | loss | 2.524e-05 | 8.297e-05 | 0.0001245 | True |
| 1 | loss | 0.0001938 | 0.000138 | 0.000207 | True |
| 2 | loss | 0.0001849 | 0.0004754 | 0.0007131 | True |
| 3 | loss | 0 | 1.835e-05 | 2.752e-05 | True |

## k = 4

- verdict: PASS (within_tolerance)
- A against A bit-equal: True (max |d| 0.0)
- L2L against GA bit-equal: False (information, not the criterion); max |d| 0.2265625

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0.01476 | 0.01411 | 0.02116 | True |
| 1 | gradient | 0.02548 | 0.0243 | 0.03645 | True |
| 0 | loss | 3.305e-05 | 9.885e-05 | 0.0001483 | True |
| 1 | loss | 0.0001047 | 0.0002612 | 0.0003918 | True |

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
