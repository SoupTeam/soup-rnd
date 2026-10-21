# B1.2: sequential expert forward with a bounded weight arena

This internal RND primitive adapts the forward loop from original Soup
[`d500ae0d`](https://github.com/MakazhanAlpamys/Soup/tree/d500ae0dca7c4da905f8d59c9866af2f8aaa824b).
It reuses the [B1.1 cache codec](expert-shard-format.md), preserves the native
eager arithmetic and loads one routed expert at a time. Experts without tokens
are not read. Coverage can approach 100%, so reduced disk reads are not promised.

This is forward-only. A gradient-bearing input or routing weight under enabled
autograd is refused with a B1.3 message. The public training builder still refuses
per-expert caches, including before materializing real weights/adapters. Backward
is B1.3; planner, public YAML and CLI integration are B1.4. No defaults change.

## Memory and ownership

`plan_expert_buffer(reader, dtype=...)` computes a single weight slot independent
of expert count. `ExpertBufferPool` allocates one byte arena with aligned views
for dense gate/up and down matrices, packed weights, statistics, shared NF4 code
tables and explicit nested-statistics scratch. The budget includes padding.
An insufficient budget fails before allocating the arena or reading an expert.
On CUDA, allocator rounding is checked before the first expert read as well.

CUDA NF4 decoding uses bitsandbytes `out` kernels directly into this arena.
Missing CUDA kernels fail explicitly; there is no allocating CUDA fallback.
CPU bitsandbytes 0.50.2 has no such `out` kernels, so CPU development checks use
temporary CPU dequantization results before copying into the arena. The CPU
arena bound does not bound those CPU temporaries or the entire process RSS.

Activations, routing masks, GEMM/backend workspaces, context memory and allocator
reservations are separate from the expert-weight budget. Record total CUDA peak
allocated **and** reserved memory separately; do not describe a 64 MiB weight
arena as a 64 MiB total-VRAM requirement. Host staging also has its own byte count.

One aligned CPU staging buffer is reused. Returned expert weights never alias
staging, but they alias the reusable compute arena and are valid only inside
`pool.lease(reader, expert)`. Nested/concurrent leases and closing an active lease
are refused. Copies are blocking; CUDA synchronizes before releasing a lease,
including on errors. This intentionally favors explicit lifetime safety, with
no asynchronous-prefetch or throughput claim.

Readers sharing a pool must agree on shapes, NF4 specs and stored dtype, and fit
its staging span. The pool owns its buffers; the caller owns and closes readers.
`pool.close()` releases buffer references, not PyTorch's global caching allocator.

## Forward and installation

`streamed_experts_class()` is lazy, so importing the module does not import
torch, transformers, bitsandbytes or PEFT. Its module accepts hidden states
`[tokens, H]`, int64 expert indices `[tokens, k]` and finite routing weights of the
same shape/device. The hidden dtype must match the pool. Expert IDs are visited
in ascending order; token gather, gate/up activation, down projection, route
weighting and index-add follow the resident eager implementation.

`last_trace` retains the latest call's layer, expert ID, token-route count,
logical bytes and load status. It is bounded by the number of routed experts;
it is not an unbounded in-memory training log. `pool.stats()` reports its plan,
allocation delta, cumulative fetches and logical bytes. Logical read bytes are
not a measurement of physical-host disk I/O. The gate records complete routing
indices and weights separately.

`install_expert_forward(layers, readers, pool)` validates every layer before
replacing anything. It requires the native fused `gate_up_proj`/`down_proj`
contract and frozen routed weights; extra parameters/adapters, mismatched
readers/shapes, duplicate destinations and missing expert modules are refused.
Only the routed-experts submodule is replaced. Router, correction bias, shared
experts and dense decoder layers remain with the original DeepSeek/Kimi block.
The full block and shared-expert call count are tested separately.

This helper is for forward validation, not installation into a training run or
serialization of a complete streamed model. Device placement is fixed when the
pool is created; moving an enclosing module does not move this caller-owned pool.

## Local checks

```text
python -m pytest tests/test_b1_expert_forward.py tests/test_b1_expert_forward_gate.py --no-cov -v
python -m pytest tests/test_b1_expert_forward.py -m gpu --no-cov -v
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

Tests cover dense and NF4 exactness, nested quantization on/off, varied top-k and
seeds, empty/partial/all-hit routing, autocast, a same-process different-weight
negative control, one-slot reuse, insufficient budget, exceptions, explicit
autograd refusal and full native DeepSeek/Kimi MoE blocks. CUDA checks must be
reported as skipped when CUDA is unavailable. Small gate-instrument fixtures
are synthetic; they are not trained-checkpoint acceptance results.

## Prospective trained-checkpoint gate

The [protocol](../benchmarks/protocols/b1.2-expert-forward/protocol.json) pins
OLMoE `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, full checkpoint hashes,
package versions, layers 0/7/15, 512 synthetic hidden states, a 64 MiB expert-weight
budget, exact-equality and negative-control rules, and **one attempt per cell
with zero replacements**. Its 12 cells cover CPU fp32, CUDA bf16, CUDA NF4 and
CUDA NF4 with nested statistics. A single cell's PASS is not a complete task PASS.

Commit the source and protocol before the first trained-checkpoint arm. Use a
clean checkout with results and model data outside it. The harness refuses a
dirty checkout, verifies full checkpoint bytes and refuses existing attempt
directories. Do not bypass the attempt budget by changing session directories.
Review any failure/NO VERDICT before separately proposing a new protocol; retain
every attempted cell, traceback, progress file and original decision rule.

Checkpoint preparation is approximately 14 GB of downloads. Allow about 64 GB
of disk space for checkpoint, converted layer files and logs. No complete model
is loaded onto the GPU; the harness loads one selected layer at a time. Use one
CUDA GPU that supports BF16 and has room for its resident reference. Two SSDs,
bare metal and a long performance rental are not requirements of this gate.
These are preparation estimates, not a measured memory-fit claim.

After the user commits the protocol, downloads the pinned snapshot and selects a
CUDA environment matching it, the Linux sequence is:

```bash
for layer in 0 7 15; do
  for cell in cpu-fp32 cuda-bf16 cuda-bf16-nf4 cuda-bf16-nf4-dq; do
    python benchmarks/harness/b1_expert_forward_gate.py \
      --model-dir /workspace/olmoe-pinned \
      --session /workspace/b1.2-forward-session \
      --layer "$layer" --cell "$cell" --power-source server || exit 1
  done
done
```

Each cell runs in a new process. The real checkpoint router supplies top-k
routing. For NF4, the resident reference is independently reconstructed from the
same stored packed bytes and scales. The original resident stack is released
before the streamed memory arm. Results preserve allocated/reserved baselines
and peaks, source identities, package versions, routing, fetch order and exposed
NVIDIA state. Power source is an operator declaration, not sensor certification.

The generic OLMoE expert-layer gate does not add OLMoE to the RND checkpoint
sharder's supported architectures; B1.1 conversion remains DeepSeek-V3-family.
Original Soup's earlier forward/backward gate is cited as prior evidence only.
This path makes no new convergence, throughput, giant-model, physical 8 GB fit,
physical-host isolation or backward-correctness claim.

## Local verification, 2026-10-20

On the B1.2 working tree based on B1.1 `9510096a`, Windows/Python 3.11.15,
torch 2.14.1+cpu, transformers 5.19.0, bitsandbytes 0.50.2 and safetensors 0.8.0:

- B1.2 unit/instrument checks plus B1.1/B2/Kimi, trust-remote-code and provenance:
  199 passed, 48 skipped, 8 existing PEFT pattern warnings. CUDA was unavailable.
- Full `ruff check src/soup_cli/ scripts/ tests/ benchmarks/` and
  `git diff --check` passed; the new files were checked for trailing whitespace.
- No full-repository pytest run or trained-checkpoint/CUDA gate was executed.

The borrowed CPU environment lacks pytest-cov, so this local run used
`python -m pytest -o addopts='' ... -q --tb=short`, with worktree `src` and a
separate TRL dependency directory on `PYTHONPATH`. Repository test configuration
and the borrowed environment were not changed. The B1.1 interrupted-conversion
fix and its regressions are included in this branch's base.

## Synthetic CUDA preflight, 2026-10-20

The retained [raw archive](../benchmarks/results/b1.2-cuda-preflight-20261020/raw-results.zip)
contains the exact uncommitted source bytes tested on Linux/Python 3.12.13,
RTX 3060 (12 GB, native BF16), torch 2.14.1+cu130 and the pinned package versions.
All seven selected tests passed without skips in 3.97 seconds: two B1.1 CUDA
NF4 codec cases, two complete native DeepSeek/Kimi MoE cases and three streamed
BF16/NF4/nested-NF4 forward and weight-arena cases. The last three retain their
same-process negative controls, expert traces and allocated/reserved memory.

Measured single-slot arena allocations were 3,072 / 4,608 / 7,168 bytes and
matched the corresponding plans and budgets. These are tiny synthetic fixtures,
not trained-checkpoint memory-fit or total-VRAM results. The separate
[verification](../benchmarks/results/b1.2-cuda-preflight-20261020/verification.json)
confirms ZIP CRC, all 606 source hashes, identity with the sent payload, seven
JUnit passes, negative controls, load order/count and arena allocation bounds.
Source identity is recorded by manifest and base `9510096a`; it is not presented
as a precommitted trained-checkpoint protocol.

## Trained-checkpoint gate, 2026-10-20

The prospective gate passed all 12 cells at committed source `99664b09` on the
RTX 3060: trained OLMoE layers 0/7/15, each in CPU fp32, CUDA BF16, CUDA NF4 and
CUDA nested NF4. Recorded output hashes match exactly, max absolute difference
is zero, and every different-weight negative control differs. All 64 experts
were routed and loaded once in ascending order in every cell. No replacement,
failed or unknown cell is present in this series.

CUDA arena allocation was 12 MiB for BF16 and 16 MiB for either NF4 variant,
below the frozen 64 MiB budget. Streamed peak allocated memory was approximately
28.11-32.34 MiB for this layer primitive; this is not a whole-model VRAM bound.
The [report, raw archive and replay analyzer](../benchmarks/results/b1.2-olmoe-forward-20261020/report.md)
retain all cell logs, routing, traces, memory, hashes, protocol and source.

## Acceptance status

CPU regression, synthetic CUDA preflight and the 12-cell prospective trained-layer
forward/arena gate passed. The B1.1 interrupted-layout review correction is
inherited by this branch. B1.1 dependency review, publication, required CI and
B1.2 reviewer acceptance remain; the task board is not marked Done. A new
changelog fragment must use the eventual B1.2 PR number; B1.1's #24 fragment is
not the B1.2 entry.
