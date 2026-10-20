# B1.1: independently readable MoE expert storage

This RND change implements storage only. It does not enable expert-streamed
training, reduce measured VRAM, or establish a speed or quality improvement.
GPU scheduling and backward integration belong to B1.2/B1.3.

The codec is adapted from original Soup's `utils/expert_stream.py` at
[`d500ae0d`](https://github.com/MakazhanAlpamys/Soup/tree/d500ae0dca7c4da905f8d59c9866af2f8aaa824b).
Only the writer, reader and storage validation are reused. RND's B2 skeleton
layout, Kimi K2 aliases, FP8 source decoder and adapter preflight remain in place.

## Storage contract

`shard_checkpoint(..., expert_layout="per_expert")` is an internal storage API.
It currently admits the canonical `deepseek_v3` architecture with a B2
`ShardLayout(stack_experts=True)`, and source checkpoints with separate
`mlp.experts.<id>.{gate,up,down}_proj.weight` matrices. Expert ids must start at
zero, be contiguous and agree across projections. Fused-only source tensors and
other architectures are refused. Existing/default calls keep the fused format.

For decoder layer N:

| File | Contents |
| --- | --- |
| `layer_NNN.safetensors` | Attention, norms, router/bias and shared experts |
| `experts_NNN.safetensors` | Independently addressed routed-expert matrices |
| `extras.safetensors` | Existing extras, including shared NF4 code tables |

Each routed expert E has `E.gate_up` with shape `[2I, H]` (gate rows then up
rows) and `E.down` with shape `[H, I]`. The integer expert id and these names map
back to the original gate/up/down names; the cache does not retain the original
three-key representation. All MoE layers must agree on expert count and shapes.
FP8 sources use B3's block-scale decoder before conversion/packing.

Under NF4 each expert matrix is quantized independently, storing the existing
packed tensor, `::absmax`, and, when enabled, `::nested_absmax` and
`::nested_offset`. Logical shapes, dtype, block sizes and nested flag are in
`expert_quant_specs`. Shared NF4 code tables are checked for consistency across
matrices and saved in extras. RND's FP8-source double-quant policy is unchanged.

## Index and compatibility

The existing shard format version remains 5. A separate expert-layout version
avoids invalidating unrelated fused caches:

| Field | Per-expert value |
| --- | --- |
| `expert_layout` | `"per_expert"` |
| `expert_layout_version` | `1` |
| `expert_layers` | Sorted unique decoder layer ids |
| `expert_count` | Routed experts per MoE layer |
| `expert_shapes` | Logical shapes of `gate_up` and `down` |
| `expert_quant_specs` | NF4 specs for both matrices, or empty when unquantized |

Missing expert fields mean the unchanged fused format. Unsupported versions,
inconsistent dimensions/counts and malformed NF4 metadata fail explicitly.
Changing layouts or losing a referenced expert file causes a cache miss.
B2's `stacked_experts`, `fp32_keys`, layer-count and source-identity checks remain
part of the cache key. Expert files follow their layer's stripe-root placement.
Switching back to fused can leave unused expert files; no active index references
them. This change does not delete unrelated files or old stripe roots.

Training entry points reject a per-expert cache before model construction or
installation, naming B1.2/B1.3 and the option to re-shard as fused. There is no
new YAML or CLI training option. Older RND releases do not understand this
extension: use a separate cache directory and re-shard as fused before downgrading.

## Independent reader

```python
from soup_cli.utils.expert_shard import ExpertShardReader, aligned_staging
from soup_cli.utils.layer_shard import expert_paths, read_shard_index

index = read_shard_index(cache_dir)
reader = ExpertShardReader(
    expert_paths(cache_dir, index)[layer],
    count=index.expert_count,
    shapes=index.expert_shapes,
    quant_specs=index.expert_quant_specs,
    dtype=index.dtype,
)
owner, staging = aligned_staging(reader.max_span)
try:
    stored = reader.read_host(expert_id, staging)
    # CPU views: clone values that must survive the next read into staging.
finally:
    reader.close()
```

The reader validates file identity and tensor sizes, reads only the selected
expert's aligned spans, and reuses one expert-sized staging allocation. Alignment
may physically include adjacent bytes; this is not a zero-overread claim.
`bytes_read` counts logical tensor bytes, not host/device physical I/O. Direct
I/O is attempted by default with a buffered fallback; no storage-speed claim
follows from correctness tests. The writer still retains one layer's output
expert tensors before its atomic file write: bounded to one layer, not one expert.

## Acceptance and reproduction

The focused tests cover a 12-expert synthetic MoE checkpoint, integer ordering,
dense/shared separation, exact reconstruction, FP8 inputs, old caches, layout
switches, missing files, malformed indices, replaced files, independent reads,
stripe placement, and explicit runtime refusal. NF4 tests compare packed bytes,
statistics and dequantized values with an independent bitsandbytes call, with
double quantization on and off. Tiny/synthetic fixtures do not establish trained
model quality or giant-checkpoint memory fit.

From a RND development environment with pytest-cov:

```text
python -m pytest tests/test_b1_expert_shard.py --no-cov -v
python -m pytest tests/test_b1_expert_shard.py -m gpu --no-cov -v
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

CUDA tests are additional platform checks and must be reported as skipped if
CUDA is unavailable. Existing Soup gate results are prior evidence, not newly
executed RND results. A0 proceed acceptance remains an external task dependency;
this storage PR does not approve A0 or close the complete B1 track.

The scoped storage extension is recorded in `changelog.d/0.75.1/24.added.md`.
It does not enable public expert-streamed training.

## Local verification, 2026-10-20

Working-tree implementation based on RND `ca9be3cc`, not a new committed
revision. Python 3.11.15, torch 2.14.1+cpu, transformers 5.19.0, peft 0.21.2,
bitsandbytes 0.50.2, safetensors 0.8.0 and pytest 9.1.1 on Windows.

- Focused B1.1 suite: 33 passed, 2 CUDA cases skipped.
- Combined B1.1, B2 (including Kimi), B3, stripe sharder, source-handle,
  adapter preflight/budget and licence-header checks: 211 passed, 89 skipped.
  Two existing PEFT rank/alpha-pattern warnings were reported.
- Full `ruff check src/soup_cli/ scripts/ tests/ benchmarks/` and
  `git diff --check`: passed.
- No full repository pytest run, CUDA execution, trained-checkpoint experiment
  or performance measurement was performed.

The borrowed local CPU environment did not have pytest-cov installed, so the
local command used `python -m pytest -o addopts='' <test files> -q --tb=short`,
with `PYTHONPATH` pointing at this worktree's `src`. It did not change the
repository's default coverage configuration or the borrowed environment.
