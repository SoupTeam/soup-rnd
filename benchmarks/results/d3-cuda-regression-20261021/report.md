# D3 CUDA regression: 69/69 passed

All 69 selected CUDA cases passed on the existing RTX 3090 in **73.49 seconds**:
zero failures, errors or skips. The other 78 CPU cases in these three modules
were deselected by `-m gpu`. The retained JUnit cases match the exact 69 collected
GPU node IDs, with no duplicates or missing cases.

| Module | CUDA cases | Result |
| --- | ---: | --- |
| `test_d3_full_model.py` | 3 | PASS |
| `test_d3_grouped_compute.py` | 39 | PASS |
| `test_d3_grouped_expert.py` | 27 | PASS |

These cover tiny whole-model BF16/NF4/nested-NF4 execution and changed-weight /
restored controls; the explicit cutoff-128 OLMoE row-range regression; bounded
grouped compute against native and B1.2 references; activation-budget refusal;
and grouped reader/decode integration across widths 1/2/4, empty, partial and
all-expert routing. Weights and inputs here are synthetic, including the large
OLMoE-shaped arithmetic cases. These tests are distinct from the separately
published twelve-cell trained whole-model forward gate.

## Exact tested source

The CI fix removes private CUDA availability probes and adds the shared `gpu`
marker to five parametrized test functions. BF16, Linux and library capability
checks remain. The three test edits were **uncommitted when executed**; their
exact working bytes, hashes, original bytes and patch are archived. No fabricated
source commit or current-head CUDA certificate is claimed.

GitHub cloning on the server failed/stalled before any test execution. A separate
local clone of real commit `c94214d397ffbe609916221e43fb2efd8c43ad4d` was therefore used,
with the three exact edited test files applied. The operator's base was
`efeed00cd25562ae21b759959d9ca3be5e14715d`. A Git comparison verifies identical
committed `src`, selected original tests/helpers, relevant harness files and
configuration across these commits; every retained runtime/source file also
matches the current local bytes. All source identities are in the archive.
The old frozen whole-model checkout and its archives were left intact.

## Hardware and execution

GPU: NVIDIA GeForce RTX 3090, native BF16, capability 8.6,
UUID `GPU-5452dd2b-cfd5-1349-10e2-17d8ae60a428`, driver 580.159.03.
PyTorch 2.14.1+cu130, Transformers 5.19.0, bitsandbytes 0.50.2, Triton 3.8.0,
safetensors 0.8.0, pytest 9.1.1. The cuBLAS binary hash was checked before running.
No visible GPU peers were present before execution. Downloads were disabled.
The supervisor job ran once with no automatic restarts or replacements.

```bash
PYTHONPATH=src \
D3_CUBLAS_LIBRARY=/absolute/path/to/libcublas.so.13 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m pytest tests/test_d3_full_model.py tests/test_d3_grouped_compute.py \
  tests/test_d3_grouped_expert.py -m gpu -o addopts= -vv --tb=short \
  --junitxml=junit.xml
```

The shared marker must see a usable native-BF16 CUDA GPU. A CPU-only execution
skips these cases and does not reproduce this GPU result.

## Evidence and limits

`raw-results.zip` preserves the complete pytest log, JUnit, exact collected node
IDs, package/GPU metadata, input receipts, test patch and original/working test
sources, runtime package and relevant harness sources, and runner/launch scripts.
Its CRC and **582** manifest entries were verified after local download, together
with both nested source/input ZIPs. `summary.json` records the independent checks;
`SHA256SUMS.json` hashes LF-normalized published text and the immutable raw ZIP.

This supplies actual CUDA regression execution for the marker fix. It is not a
new timing benchmark, trained-model acceptance series, backward/training check,
F1 physical-host certification, full repository pytest run or final D3 sign-off.
The earlier [12/12 whole-model forward result](../d3-full-forward-c94214d3/report.md)
and [layer-performance study](../d3-grouped-layer-20261021/report.md) retain their
original source pins, artifacts and limitations.
