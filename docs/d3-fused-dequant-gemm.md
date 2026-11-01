# D3 packed NF4 grouped GEMM candidate

Status, 2026-11-01: the packed-only pool, fused forward/dX, lease-safe first
derivatives and explicit public SFT/L2L opt-in are implemented. Synthetic CUDA
screens pass on RTX 3090 (sm_86); pretrained qualification and performance are
pending. The existing completed OLMoE studies measured the staged restore/GEMM
path and do not qualify this new implementation.

## Existing implementations inspected

| Source | Finding and reuse decision |
| --- | --- |
| Soup upstream `c38330cfae3c91e8d5ec880e8966a387ee025a74`, `layer_stream_runtime.py` | Checkpoint-visible wrapper calls bitsandbytes `gemm_4bit` for eligible ordinary 2D projections. It has no grouped packed routed-expert API. Preserve it as a comparator; do not call a per-expert launch loop a grouped kernel. |
| RND D1 `benchmarks/harness/d1_fused_nf4.py`, current D3 base `55d5f9b7` | Tiled Triton NF4 forward/dX already decodes inside GEMM. Reused this in-repository algorithm as the candidate foundation, with D3 pointer tables and independent expert programs. D1's WMMA results are shape/card-specific, and its retained A100 dX failure prevents importing a universal exactness claim. |
| [grouped-nf4-gemm](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/6928361657ddd4fb22062b328548c93e2557fcb7), MIT | `kernel/nf4_grouped.py` supplies a useful grouped packed-GEMM design. Its documented arithmetic may feed FP32/TF32 decoded weights to dot rather than first rounding to the baseline BF16 weights; nested statistics are de-nested at repack. It is not a drop-in implementation of this D3 gate. No external source was copied or made a runtime dependency. |
| [bitsandbytes CUDA operators](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/backends/cuda/ops.py) | Ready fused NF4 forward for eligible 2D cells, without a matching grouped routed-expert contract. |
| [Unsloth kernel utilities](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py) | Existing fused ordinary-projection paths are useful comparison points; they do not replace the bounded D3 expert storage/lease contract. |
| [Triton grouped GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) | Pointer/geometry scheduling reference; not NF4 or native-bitwise qualification. |

The upstream refs were inspected in separate research checkouts; the D3 branch
was not rebased or committed. External licenses were read before the reuse
decision. Original Soup/D1 code remains the implementation source.

The [development record](../benchmarks/results/d3-fused-nf4-development-20261101/report.md)
preserves actual launch sources, unsuccessful candidates, profiler traces,
MLP derivatives and synthetic public-trainer evidence. These were development
runs from uncommitted snapshots, not prospective trained-model experiments.

## Implemented boundary

`expert_group_fused.py` supplies the descriptors, host validation and launch
policy. `expert_group_fused_wmma.py` supplies a CUDA WMMA mainloop derived from
Soup's D1 code. One launch handles one through four eligible independent expert
matrices. A descriptor table points to the original B1 packed bytes, code table
and scales; nested scales are decoded inside the mainloop. The packed-only pool
does not retain full BF16 expert projections or dequantization scratch slots.

The explicit `hybrid_wmma` policy uses native bitsandbytes restoration plus
`torch.mm` for rows 1 through 128, skips empty experts and uses fused WMMA for
129 through 512. Both OLMoE geometries, `(2048,2048)` and `(2048,1024)`, are
supported, including dX. Counters report fused and native matrices separately.
There is no comparison-driven rescue or automatic retry with a different kernel.
Pure Triton and pure WMMA candidates failed some native bit comparisons; their
results remain published. In particular, the small-row fallback is not fusion.

The fused mainloop rounds decoded weights to BF16 before FP32 accumulation and
converts the final output once. It uses no split-K, atomics or dense weight
allocation. The native small-row fallback temporarily restores one projection;
that allocation and nested-scale scratch are charged to the call budget. This
does not imply a whole-trainer VRAM reduction.

The primitive refuses autograd-bearing inputs/weights. The custom backward
reloads packed state under a fresh lease, recomputes the expert and uses packed
dX; SiLU, route derivatives and ordered scatter retain native semantics. Only
frozen routed weights and first derivatives are supported. Expired/recycled
leases fail before dispatch, and storage remains owned through CUDA completion.

Outputs, descriptors and fallback scratch have an allocation precheck. The
weight constructor also checks the actual CUDA allocator delta. Trainer
preflight reserves the full configured weight ceiling, activation and call
budgets plus the existing 4 MiB BLAS workspace allowance; compiler/context and
total measured allocator peaks remain separately reported.

## Explicit trainer opt-in

This candidate currently requires Linux, native CUDA BF16, **sm_86**, OLMoE
H=2048/I=1024 and NF4 for both routed projections. Tested software is torch
2.14.1+cu130, Triton 3.8.0 and bitsandbytes 0.50.2. Native SiLU is used;
the earlier optional Triton activation is not enabled by this backend.

```yaml
training:
  stream_layers: true
  quantization: 4bit
  moe_aux_loss_coeff: 0.0
  lora:
    target_modules: [q_proj, v_proj]
  stream_experts:
    backend: fused_nf4
    group_width: 4
    weight_budget_bytes: 67108864
    activation_budget_bytes: 268435456
    call_budget_bytes: 67108864
    native_rows_max: 128
    native_rows_from: null
    cublas_library: /absolute/path/to/libcublas.so.13
```

Supply the installed library's real absolute path. Existing sequential/grouped/
Triton defaults remain as before. A `fused_nf4` configuration with the old
`native_rows_from: 481` profile is refused rather than silently ignored.

## Development checks and remaining gate

The primitive screen is reproducible from its original source snapshot:

```sh
PYTHONPATH=src python benchmarks/harness/d3_fused_nf4_gate.py --backend hybrid_wmma --out /workspace/d3-fused-screen-new
```

Completed development checks:

- 20 arithmetic/control cells: independent native and singleton comparisons,
  both projections/codecs/directions and the declared finite row set.
- Eight actual CUDA profiles: one fused compute kernel per eligible group,
  output/descriptor allocation bound and unchanged input/NF4 state.
- Eight disk-backed MLP cases: exact outputs and input/router derivatives,
  widths one/four, mixed routes and expired-lease refusal.
- Eight synthetic public SFT/L2L children, repeated after the allocator
  reservation correction: exact sequential gradients, adapter updates and
  trained logits, with same-recipe adapter reload.
- 154 selected CPU tests; the full repository suite is not claimed by this count.

Still required for this new implementation: prepare/pin trained OLMoE inputs
and actual-shard validity controls on this host; freeze an executable prospective
rule and committed sources; run native and same-kernel resident/streamed
correctness, then paired staged/fused timing, memory and actual-launch profiles.
No pretrained run, speedup, convergence claim or final D3 acceptance follows
from this development packet. See the [qualification plan](../benchmarks/protocols/d3-fused-trainer/README.md).
