# L2L against gradient accumulation, forward and backward: mistralai/Mistral-7B-v0.1

Card: **24gb**. Results from another card are a separate report.

**Verdict: PASS** (within_tolerance)

## Command

```
/root/soup/.venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 24gb --equal-length 230 --data-seed 5 --out /root/runs/ctl-24gb-seed5
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
- GPU: {'name': 'NVIDIA GeForce RTX 3090', 'memory_total_mib': 24576, 'driver': '580.126.09'}
- Container RAM: {'source': '/sys/fs/cgroup/memory/memory.limit_in_bytes', 'limit_bytes': 64756908032, 'current_bytes': 46253133824, 'peak_bytes': 46253133824, 'swap_source': '/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes', 'swap_max_bytes': 0, 'swap_total_bytes': 68719472640, 'version': 1}
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
| 0 | gradient | 0 | 0.02102 | 0.03154 | True |
| 1 | gradient | 0 | 0.02825 | 0.04237 | True |
| 2 | gradient | 0 | 0.04586 | 0.06878 | True |
| 3 | gradient | 0 | 0.06622 | 0.09932 | True |
| 4 | gradient | 0 | 0.02032 | 0.03048 | True |
| 5 | gradient | 0 | 0.05054 | 0.07582 | True |
| 6 | gradient | 0 | 0.02422 | 0.03634 | True |
| 0 | loss | 0 | 0.0002803 | 0.0004204 | True |
| 1 | loss | 0 | 0.0004254 | 0.0006382 | True |
| 2 | loss | 0 | 0.0005546 | 0.0008319 | True |
| 3 | loss | 0 | 0.0004999 | 0.0007499 | True |
| 4 | loss | 0 | 0.0002538 | 0.0003807 | True |
| 5 | loss | 0 | 0.0005654 | 0.000848 | True |
| 6 | loss | 0 | 0.0002002 | 0.0003002 | True |

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
| 0 | gradient | 0 | 0.01892 | 0.02839 | True |
| 1 | gradient | 0 | 0.04339 | 0.06508 | True |
| 2 | gradient | 0 | 0.02971 | 0.04456 | True |
| 3 | gradient | 0 | 0.02422 | 0.03634 | True |
| 0 | loss | 0 | 2.382e-05 | 3.572e-05 | True |
| 1 | loss | 0 | 6.837e-05 | 0.0001025 | True |
| 2 | loss | 0 | 0.0003714 | 0.0005571 | True |
| 3 | loss | 0 | 0.0002002 | 0.0003002 | True |

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
| 0 | gradient | 0 | 0.02572 | 0.03858 | True |
| 1 | gradient | 0 | 0.02346 | 0.03519 | True |
| 0 | loss | 0 | 2.113e-05 | 3.169e-05 | True |
| 1 | loss | 0 | 0.0003066 | 0.00046 | True |

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
