# L2L against gradient accumulation, forward and backward: mistralai/Mistral-7B-v0.1

Card: **8gb**. Results from another card are a separate report.

**Verdict: FAIL** (l2l_out_of_tolerance)

k=2: L2L is outside the bf16 tolerance at 2 check(s); first: [{'what': 'gradient', 'step': 2, 'error': 0.03974482457166442, 'limit': 0.03473116419867697}, {'what': 'loss', 'step': 0, 'error': 0.000391334001129453, 'limit': 0.0002780439312122994}]

## Command

```
/root/soup/.venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 8gb --out /root/runs/fwdbwd-8gb-bf16-run2
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
- GPU: {'name': 'NVIDIA GeForce RTX 3070', 'memory_total_mib': 8192, 'driver': '595.71.05'}
- Container RAM: {'source': '/sys/fs/cgroup/memory.max', 'limit_bytes': 259506831360, 'current_bytes': 46020354048, 'peak_bytes': 46020354048, 'swap_source': '/sys/fs/cgroup/memory.swap.max', 'swap_max_bytes': 0, 'swap_total_bytes': 8589930496, 'version': 2}
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
| 0 | gradient | 0 | 0.02137 | 0.03205 | True |
| 1 | gradient | 0 | 0.01388 | 0.02082 | True |
| 2 | gradient | 0 | 0.02742 | 0.04113 | True |
| 3 | gradient | 0 | 0.0284 | 0.0426 | True |
| 4 | gradient | 0 | 0.03721 | 0.05582 | True |
| 5 | gradient | 0 | 0.02574 | 0.03861 | True |
| 6 | gradient | 0 | 0.02055 | 0.03082 | True |
| 0 | loss | 0 | 0.0004945 | 0.0007418 | True |
| 1 | loss | 0 | 0.0003486 | 0.0005228 | True |
| 2 | loss | 0 | 0.0003244 | 0.0004866 | True |
| 3 | loss | 0 | 0.0002423 | 0.0003635 | True |
| 4 | loss | 0 | 0.0006024 | 0.0009036 | True |
| 5 | loss | 0 | 0.0005341 | 0.0008011 | True |
| 6 | loss | 0 | 0.000609 | 0.0009136 | True |

## k = 2

- verdict: FAIL (l2l_out_of_tolerance)
- L2L is outside the bf16 tolerance at 2 check(s); first: [{'what': 'gradient', 'step': 2, 'error': 0.03974482457166442, 'limit': 0.03473116419867697}, {'what': 'loss', 'step': 0, 'error': 0.000391334001129453, 'limit': 0.0002780439312122994}]
- A against A bit-equal: True (max |d| 0.0)
- L2L against GA bit-equal: False (information, not the criterion); max |d| 0.46875

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0.01887 | 0.01425 | 0.02138 | True |
| 1 | gradient | 0.0188 | 0.02926 | 0.04389 | True |
| 2 | gradient | 0.03974 | 0.02315 | 0.03473 | False |
| 3 | gradient | 0 | 0.02055 | 0.03082 | True |
| 0 | loss | 0.0003913 | 0.0001854 | 0.000278 | False |
| 1 | loss | 6.818e-05 | 0.0001527 | 0.0002291 | True |
| 2 | loss | 0.0002326 | 0.0002498 | 0.0003747 | True |
| 3 | loss | 0 | 0.000609 | 0.0009136 | True |

## k = 4

- verdict: FAIL (l2l_out_of_tolerance)
- L2L is outside the bf16 tolerance at 1 check(s); first: [{'what': 'loss', 'step': 0, 'error': 0.00038818479400497486, 'limit': 0.00026387929190846345}]
- A against A bit-equal: True (max |d| 0.0)
- L2L against GA bit-equal: False (information, not the criterion); max |d| 0.205078125

## Tolerance

- fp32: relative L2 error at most 1e-04 at every step
- bf16: relative L2 error at most 1.5 x the error of GA on MATH against GA on EFFICIENT_ATTENTION (same run), at every step (this run)
- the threshold is fixed in the script; no option changes it

| step | what | L2L error | floor | limit | ok |
| --- | --- | --- | --- | --- | --- |
| 0 | gradient | 0.01361 | 0.01533 | 0.02299 | True |
| 1 | gradient | 0.01961 | 0.01834 | 0.02751 | True |
| 0 | loss | 0.0003882 | 0.0001759 | 0.0002639 | False |
| 1 | loss | 0.0001304 | 0.0001277 | 0.0001915 | True |

## Counters

- gemm_4bit path calls, ga1: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- gemm_4bit path calls, ga2: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- gemm_4bit path calls, l2l: {256: {'custom': 0, 'dequant': 6944}, 192: {'custom': 0, 'dequant': 672}, 128: {'custom': 1280, 'dequant': 512}, 64: {'custom': 480, 'dequant': 192}, 230: {'custom': 0, 'dequant': 672}, 153: {'custom': 0, 'dequant': 672}}
- gemm_4bit path calls, ga_floor: {256: {'custom': 0, 'dequant': 2688}, 192: {'custom': 0, 'dequant': 1344}, 128: {'custom': 960, 'dequant': 384}, 64: {'custom': 960, 'dequant': 384}, 230: {'custom': 0, 'dequant': 1344}, 153: {'custom': 0, 'dequant': 1344}}
- layer loads of the first full window, L2L, by k: {1: 62, 2: 62, 4: 62}
- layer loads of the same window, GA, by k: {1: 62, 2: 122, 4: 244}
- NF4 restores in the L2L arm, by k: {1: 0, 2: 0, 4: 0}

## Findings

- restores_zero: discrepancy: NF4 restores are 0 on a quantized model ({1: 0, 2: 0, 4: 0}); the restore path was not exercised, this is not a success
