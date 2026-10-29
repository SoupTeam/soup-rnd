# L2L against gradient accumulation, forward and backward: mistralai/Mistral-7B-v0.1

Card: **8gb**. Results from another card are a separate report.

**Verdict: PASS** (within_tolerance)

## Command

```
/root/soup/.venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 8gb --equal-length 230 --data-seed 5 --out /root/runs/ctl-8gb-seed5
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
- Container RAM: {'source': '/sys/fs/cgroup/memory/memory.limit_in_bytes', 'limit_bytes': 52922679296, 'current_bytes': 36681347072, 'peak_bytes': 36809744384, 'swap_source': '/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes', 'swap_max_bytes': 0, 'swap_total_bytes': 7614754816, 'version': 1}
- Commit: None
- Unavailable: {'commit': 'not a git checkout'}
- Quantization: nf4, dtype mode: bf16

## Synthetic rows (random tokens, real weights)

**Equal length control: every row has one length, the L2L window is not padded.**

- lengths: [230, 230, 230, 230, 230, 230, 230]
- tokens labelled -100: 227
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
| 0 | gradient | 0 | 0.01666 | 0.02499 | True |
| 1 | gradient | 0 | 0.02322 | 0.03483 | True |
| 2 | gradient | 0 | 0.01606 | 0.02408 | True |
| 3 | gradient | 0 | 0.04906 | 0.0736 | True |
| 4 | gradient | 0 | 0.03135 | 0.04702 | True |
| 5 | gradient | 0 | 0.03661 | 0.05492 | True |
| 6 | gradient | 0 | 0.02926 | 0.0439 | True |
| 0 | loss | 0 | 2.304e-05 | 3.456e-05 | True |
| 1 | loss | 0 | 0.0002312 | 0.0003467 | True |
| 2 | loss | 0 | 0.0006041 | 0.0009062 | True |
| 3 | loss | 0 | 0.0003759 | 0.0005639 | True |
| 4 | loss | 0 | 5.399e-06 | 8.099e-06 | True |
| 5 | loss | 0 | 0.0004399 | 0.0006598 | True |
| 6 | loss | 0 | 0.0001234 | 0.0001851 | True |

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
| 0 | gradient | 0 | 0.01981 | 0.02971 | True |
| 1 | gradient | 0 | 0.02776 | 0.04164 | True |
| 2 | gradient | 0 | 0.02892 | 0.04339 | True |
| 3 | gradient | 0 | 0.02926 | 0.0439 | True |
| 0 | loss | 0 | 0.0001128 | 0.0001692 | True |
| 1 | loss | 0 | 0.0004989 | 0.0007484 | True |
| 2 | loss | 0 | 0.0001625 | 0.0002437 | True |
| 3 | loss | 0 | 0.0001234 | 0.0001851 | True |

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
| 0 | gradient | 0 | 0.01917 | 0.02876 | True |
| 1 | gradient | 0 | 0.02814 | 0.04222 | True |
| 0 | loss | 0 | 0.0003012 | 0.0004517 | True |
| 1 | loss | 0 | 0.0001477 | 0.0002215 | True |

## Counters

- gemm_4bit path calls, ga1: {230: {'custom': 0, 'dequant': 9408}}
- gemm_4bit path calls, ga2: {230: {'custom': 0, 'dequant': 9408}}
- gemm_4bit path calls, l2l: {230: {'custom': 0, 'dequant': 10080}}
- gemm_4bit path calls, ga_floor: {230: {'custom': 0, 'dequant': 9408}}
- layer loads of the first full window, L2L, by k: {1: 62, 2: 62, 4: 62}
- layer loads of the same window, GA, by k: {1: 62, 2: 122, 4: 244}
- NF4 restores in the L2L arm, by k (0 is expected after the first forward): {1: 0, 2: 0, 4: 0}
- NF4 restores over the whole run: 224

## Findings

- none
