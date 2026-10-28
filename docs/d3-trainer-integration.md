# D3 opt-in streamed SFT integration

Implementation status: synthetic CPU integration and an 18-case synthetic CUDA
development preflight are validated. Pretrained CUDA trainer qualification and
final D3 acceptance remain pending. The historical
84-cell experiment used a separate adapter fixture and is preserved unchanged.
The [CPU evidence packet](../benchmarks/results/d3-trainer-integration-cpu-20261029/report.md)
retains the selected-suite log, JUnit, environment and working source snapshot.
The [CUDA development follow-up](../benchmarks/results/d3-trainer-integration-cuda-preflight-20261029/report.md)
retains 18 actual public-wrapper runs on RTX 4090, exact sequential comparisons,
adapter restoration, preliminary failures and separate guest-disk controls. Its
current CPU recheck passes 933 cases, including 24 integration tests. These
development runs use random two-layer weights, not pretrained OLMoE.

## Execution path

`training.stream_experts` selects independently readable expert shards in the
regular SFT setup. The ordinary meta model, PEFT adapter installation, streamed
decoder wrappers and SFTTrainer remain the training entry point. The expert
installer removes only the frozen routed expert parameters; attention and router
operations remain native. It validates every replacement before installing any.

The runtime owns the expert readers, one shared bounded group pool and optional
cuBLAS handle. Gathered activations, never reusable weight views, cross the
autograd boundary. Backward reloads one expert at a time and recomputes its local
first derivative. The D3 GEMM groups forward compute; backward is native eager
recomputation, not a grouped backward GEMM optimization.

The sharder accepts both separately stored projection tensors and the native
OLMoE fused tensors. A fused checkpoint is sliced one expert at a time, without
allocating a full dense expert stack. Existing fused caches/defaults are unchanged.

## Supported boundary

- Native OLMoE, frozen routed weights, first derivatives and stateless SiLU.
- LoRA on explicitly selected attention modules, e.g. `q_proj` and `v_proj`.
  Routed-expert adapters are refused. Other architectures are not silently admitted.
- CPU float32 development checks use eager GEMMs. They do not qualify grouped
  CUDA compute or Triton.
- Grouped/Triton CUDA requires Linux, native BF16, qualified OLMoE geometry
  (hidden 2048, intermediate 1024), and an explicit cuBLAS 12/13 library path.
- The runtime owns/restores the explicit cuBLASLt/full-BF16-reduction policy for
  its lifetime; two such CUDA runtimes cannot simultaneously own that policy.
- Router auxiliary-loss output is refused rather than dropped; the config must
  explicitly set `moe_aux_loss_coeff: 0.0`. Language loss
  retains native labels/ignored-token handling. This does not promise support
  for every OLMoE training objective or adapter placement.
- L2L admits OLMoE through the shared SFT accumulation-window implementation.
  Existing mask, loss-divisor and RNG replay controls still apply; auxiliary
  router loss remains refused.

## Configuration

The mode is disabled unless `stream_experts` is present. CPU development example:
OLMoE SFT streaming requires this explicit opt-in. Expert shards are always read
from disk through buffered positioned reads; `stream_source` selects the storage
tier of the remaining decoder weights. The runtime reports both stores separately.

```yaml
base: /absolute/path/to/local-olmoe-checkpoint
task: sft
data:
  train: /absolute/path/to/train.jsonl
training:
  backend: transformers
  stream_layers: true
  stream_source: disk
  quantization: none
  batch_size: 1
  gradient_accumulation_steps: 2
  moe_aux_loss_coeff: 0.0
  l2l: false
  lora:
    r: 4
    alpha: 8
    dropout: 0.0
    target_modules: [q_proj, v_proj]
  stream_experts:
    backend: grouped
    group_width: 4
    weight_budget_bytes: 67108864
    activation_budget_bytes: 268435456
    call_budget_bytes: 67108864
```

CUDA grouped/Triton additionally requires `cublas_library` under `stream_experts`;
it must name the actual absolute `libcublas.so.12` or `libcublas.so.13` on that host.
`backend: triton` selects fused SiLU/multiply, not a new backward kernel. The
declared default dispatch profile uses native rows up to 128 and from 481 onward;
rows 129–480 use grouped GEMM. These are explicit profile boundaries, not a
general exactness guarantee for arbitrary GPUs/shapes. `backend: sequential`
uses width one and native GEMMs for the reference path.

The runtime owns the process BLAS policy until close. Both BF16 reduction flags
are set/restored through the tuple setter; the split-K attribute is read-only.
The regression exercises the real torch setters without initializing CUDA.

Weight-arena, activation and call budgets are separate; none is total VRAM.
The setup charges an additional conservative device reservation and expert host
staging in its memory preflight. The actual training peak still needs a GPU check.

### Adapter dtype on checkpoint restoration

The tested TRL 0.29.1 SFTTrainer casts trainable QLoRA adapters to BF16 during
construction, including on a float32 CPU base. Exact checkpoint restoration
therefore uses the same trainer construction and saved adapter dtype. Copying
BF16 factors into a fresh FP32 adapter changes the matrix-multiplication recipe
and is not an exact-replay control. Tests retain exact tensor/logit comparisons;
no tolerance was added to admit that mismatched recipe. Preflight's FP32 adapter
reservation remains conservative when TRL chooses smaller BF16 tensors.

## Required qualification

CPU controls must exercise the public builder, native loss/gradient comparison,
LoRA updates, ignored labels, k=1/multiple micro-batches, RNG replay, real trainer
steps, adapter save/reload and failure cleanup. Small random weights are labeled
synthetic; they are not a pretrained-model experiment.

Before CUDA qualification, freeze the new trainer source, supported recipe,
reference numerical policy, F1 instruments and attempt budget. Check the actual
SFT entry point and L2L separately, against a matching sequential reference, with
an effective negative/restored control. The old fixture's PASS is not a PASS for
this new integration. Commits, pushes and rental decisions remain with the operator.

### Prepared public-trainer qualification

The [acceptance audit](d3-task-acceptance.md) maps the live Trello card, source
transcription, dependencies and owner-directed review. The separate
[prospective protocol](../benchmarks/protocols/d3-public-trainer/README.md) now
has executable public-wrapper correctness, timing/profile and offline replay
helpers, plus bound checkpoint/data/shard/initial-adapter and actual-shard F1
receipts. The final random-weight CUDA launcher check passes 36 numerical cases
and 18 profiles; the broad CPU selection passes 999 with eight skips. These are
preparation results, not a pretrained public-trainer verdict.

The new strict CUDA loss comparison exposed a Python-int conversion of Trainer's
`num_items_in_batch` in L2L. The original tensor now reaches the native loss
unchanged: scalar-tensor versus Python-int division can differ by one bit on
CUDA even with the same token count. A CPU identity regression and the repeated
CUDA controls cover the fix; no comparison tolerance was relaxed.

The current B1 contract is accepted under the operator's explicit delegation.
Literal packed-weight dequant/GEMM fusion, the source's wrap-only SwiGLU
constraint, restricted issue creation and final task acceptance remain visible
in the audit. The existing optional activation experiment is not substituted
for the assigned dequant/GEMM work. Commit/source freeze must precede formal run.
