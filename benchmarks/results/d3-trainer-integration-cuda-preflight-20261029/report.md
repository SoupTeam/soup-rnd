# D3 public trainer CUDA development preflight, 2026-10-29

The opt-in public Soup SFT wrapper completes two optimizer steps, adapter save and
same-recipe restoration for **18 synthetic CUDA cases**: dense/NF4/NF4-DQ,
ordinary SFT/L2L, and sequential/grouped/Triton. The model has random weights,
two layers, four experts, H=2048/I=1024, attention LoRA and three 512-token rows.
The two-step accumulation schedule includes a shorter final window.

All twelve grouped/Triton comparisons reproduce sequential gradient snapshots,
final adapter states and trained logits exactly. All eighteen saved adapters
restore exactly under the same Trainer/BF16-autocast recipe. The final adapter
state also matches between ordinary SFT and L2L for all nine backend/codec pairs.
Every grouped/Triton case records positive grouped GEMM coverage.

This is development evidence from uncommitted sources, **not a prospective
pretrained OLMoE gate or a new performance result**. GPU runtime source hashes
match the current working files. The CPU snapshot additionally includes two new
full-wrapper tests; those tests are not mislabeled as part of the GPU source packet.
The earlier CPU report and historical trained-step study remain separate.

## Environment and verification

Vast instance 58145733, machine 44906: RTX 4090 24 GiB, driver 580.95.05,
Ubuntu 22.04 KVM, Python 3.12.15, torch 2.14.1+cu130, Triton 3.8.0,
transformers 5.19.0, PEFT 0.20.0, TRL 0.29.1 and bitsandbytes 0.50.2.
The guest exposes about 49.3 GiB RAM. Inventory, environment logs, source packets,
per-case configurations, raw tensors and attempts are retained in raw-preflight.zip.

Current Windows CPU regressions: **933 passed, 8 skipped, 116 GPU deselected**,
including all 24 integration cases. The full repository Ruff check passes.
This is a selected regression set; no full repository pytest claim is made.
cpu-command.json retains the actual executable and existing test-file arguments.

Separate guest-disk instrument controls observe baseline OK, own complete-file
read OK, and independent foreign reader VOID. The 2 GiB synthetic fixture used
direct reads. These controls do not certify physical-host isolation or make the
CUDA development cases formal F1 measurements.

Verify the archive and regenerate analysis.json with:

```sh
python verify-preflight.py raw-preflight.zip --out regenerated-analysis.json
```

The verifier checks 548 outer payload hashes, nested source ZIP CRCs and 741
development source identities, then independently repeats tensor-file/state
comparisons and instrument checks. Original paths inside operational scripts are
retained as execution records; they are not a portable, frozen formal launcher.

## Retained preliminary failures

1. smoke-01 exposed a product bug: split-K is read-only in torch. The new math
   policy now sets/restores both BF16 flags through the supported tuple setter.
   The CPU regression uses torch's real flag setters instead of a permissive fake.
2. matrix-r2 rejected an inconsistent reload test: the already prepared Trainer
   used BF16 autocast/FP32 outputs while the fresh model had no matching autocast.
   Both evaluation paths now use the same explicit autocast and FP32 output policy.
   Tensor and dtype comparisons remain exact.
3. matrix-r3 retained six passing dense cases, then refused the NF4 configuration
   before training because the script used internal `nf4` instead of public `4bit`.
   matrix-r4 corrects that spelling and retains all twelve quantized cases.

No failure is counted as a passing case or erased. These are preliminary,
unfrozen development attempts; they are separate from any formal attempt budget.

## Remaining qualification

Freeze a separate public-trainer protocol and committed sources before testing
the pinned pretrained OLMoE checkpoint. It must state the native/sequential
reference, inputs/labels, numerical policy, effective negative/restored controls,
F1 mapping and attempt budget. Verify ordinary SFT and L2L separately. These small
random models do not establish trained-model correctness, quality, convergence,
8 GiB fit or a speed/VRAM improvement. Commit/push remain with the operator.
