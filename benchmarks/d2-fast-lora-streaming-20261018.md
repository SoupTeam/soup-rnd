# D2: Fast-LoRA in Soup layer-streamed SFT

Raw measurements, historical files and replay commands are preserved in the
[D2 evidence archive](d2-evidence-archive.md). Evidence links below pin the
pre-cleanup snapshot; use that checkout for the original replay paths.

Date: 2026-10-18. Soup trainer integration with CPU and RTX 3090 correctness evidence.

**Current result:** the final GPU source passed 8 production streamed-training
cases, 8 GPU kernel regressions and both supplemental NF4 streaming/float64
oracle checks. This is a synthetic correctness result, not a performance result.
All three leased instances are deleted. The current provider invoice totals
about **$0.073**, with possible settlement lag; conservatively measured cumulative
rental time is **16.63 minutes**, inside the chosen 1 h / $0.50 boundary.

## What changed

`training.fast_lora: true` now reaches the streamed Transformers SFT path, as
well as the existing resident path. The supported model family is Llama, with
one ordinary dropout-free LoRA adapter and a frozen base. Dense streaming and
4-bit NF4 streaming are accepted. Resident quantized training remains refused.

The installer runs after the streaming wrappers and adapters exist and before
the optional measured VRAM pre-flight. Forward operations read the current
`functional_call` weight substitution. No resident decoder weight copy is
created and the existing layer checkpoint/pool lifetime handling is retained.
Installation failure closes the streaming source, including the disk reader.
Unsupported architectures are refused before checkpoint resolution/sharding.

Liger, Cut-CE, LoRA variants and the previously refused training transformations
remain outside this integration. This is not a Liger streaming implementation.

The GPU check found and fixed a real NF4/FP16 backward failure with FP32 master
adapters. Adapter GEMM dtype is now retained under NF4 autocast too, and packed
bases are dequantized only for the needed input-gradient GEMM. Unsupported NF4
autocast inputs whose dtype differs from compute use the original PEFT forward:
their shared-cast accumulation graph is not modeled by the manual backward.
The normal streamed half-precision Llama cases execute the custom functions,
verified by profiler observations in the actual training test.

## Configuration

Use the same config and data for both comparison arms; toggle only `fast_lora`.
The input below is a local chat-format JSONL, not downloaded by this example.

```yaml
base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
task: sft
backend: transformers
output: ./outputs/fast-lora-streaming
data:
  train: ./data/squad-train.jsonl
  max_length: 512
training:
  seed: 792
  batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  quantization: 4bit
  stream_layers: true
  stream_buffers: 2
  stream_source: auto
  fast_lora: true
  lora:
    r: 16
    alpha: 16
    dropout: 0.0
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
```

`quantization: none` uses dense streamed weights. Keep a concrete batch size:
the resident automatic batch probe does not apply to streaming.

## Verification scope

Before GPU validation: **620 passed, 26 skipped** in the Fast-LoRA/streaming/config/SFT
regression group; **419 passed** in recipe snapshot, field-consumer, staged-field,
seed and licence-header checks. These are selected regression groups, not the
entire repository suite. Ruff passed across `src/soup_cli/ scripts/ tests/ benchmarks/`.
The original streaming file separately reported **11 passed, 4 skipped** (the CUDA
variants); these 11 are already included in the 620, not additional tests.

Environment: Windows, Python 3.10, torch 2.13.0+cpu, transformers 5.17.0,
peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.1. The schema recipe snapshot adds
only `training.fast_lora: false`; existing recipe settings are unchanged.
Raw new-test results: [JUnit XML](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-fast-lora-streaming-20261018/streaming-tests.xml).

After the AMP correction, the streamed CPU file passes **11 tests**, with the
expanded **8 CUDA cases** skipped locally. The AMP regression file plus the
meta-only saved-bytes check pass **53 tests**. The 12 new NF4 AMP cases cover
all three paths, FP16/BF16 and supported half inputs versus the explicit FP32
input fallback. Disabling AMP dtype capture in an isolated source copy kills
all six supported-half tests. No tolerances were relaxed to repair the crash.
The broader run found the meta-device autocast query issue (631 passed, one
failed); its guard is fixed and the saved-bytes regression now passes. A later
local process-session interruption is not counted as a completed test run.
Final CPU results are retained as separate XML invocations in the evidence folder.
On the final source, the disjoint runs report **620 passed / 30 hardware skips**
and **12 passed** (optimized-Python validity controls), totaling **632 passed /
30 skipped**. These are the selected regression group, not full repository CI.

`tests/test_fast_lora_streaming.py` builds a local random four-layer Llama and
trains it through the real Soup SFT/TRL setup, with two reusable layer slots.
Both RAM and asynchronous disk sources are exercised, with dense and nested
NF4 weights. The paired control is unpatched streamed PEFT LoRA, not a resident
model with different initialization or quantization.

The test starts with nonzero LoRA B, accumulates two micro-batches per update,
and checks logits, every adapter gradient, two optimizer updates and loss rounded
to three decimals. CPU tensor tolerances are rtol=1e-5, atol=1e-7 (1e-6 for
logits). It checks executed custom backward functions for MLP/QKV/single paths,
56 adapter tensors, frozen meta decoder placeholders, checkpoint recomputation
and more layer loads than forward-only execution. Periodic/final adapters must
contain canonical keys and load with ordinary PEFT on a resident base.

Additional checks cover configuration round-trips/refusals, installation before
the VRAM probe, reader cleanup on installation failure and early architecture
refusal. The resident Fast-LoRA tests are retained as regressions.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_fast_lora_streaming.py --no-cov -q
.venv\Scripts\python.exe -m ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

CUDA variants of the same streamed SFT comparison are hardware-gated:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_fast_lora_streaming.py -m gpu --no-cov -v
```

They explicitly exercise native BF16 and FP16 on one Ampere GPU by overriding
the store/TrainingArguments dtype policy; CUDA kernels and capability reporting
are not mocked. They retain the three-decimal
loss check, and use rtol=1e-2, atol=1e-5 for low-precision gradients/weights/logits.
The final Linux RTX 3090 (sm86) run uses torch 2.11.0+cu126, transformers 5.17.0,
peft 0.20.0, trl 0.29.1 and bitsandbytes 0.50.1. It verifies all 1,536 archived
source files before execution. The final source archive is split into two
parts below the repository file-size guard, with per-part and combined hashes.
It preserves the complete executed local snapshot, including unrelated research
files outside this publication. The ten published integration source/test files
match their archived GPU versions after LF normalization; the archive is not a
claim that the entire PR tree was rerun on GPU.

Independent CPU replay of the stored production tensors verifies **8 cases /
1,832 paired comparisons**, including all adapter tensors, both gradient steps,
updated weights and logits; all stored pairs are exact in these finite fixtures.
The two supplementary NF4 oracle snapshots verify
**116 quantities** under the unchanged twice-PEFT-error bound; fast/PEFT tensors
are exact for those finite fixtures. Neither replay is a new GPU execution.

Preserved non-final runs:

- First GPU pilot: 6 passed, 2 failed (NF4/FP16 RAM and disk, mixed-dtype backward).
  The original failure XML/log and three original kernel files are retained.
- Second GPU pilot: 16 tests passed, but the supplementary BF16 harness refused
  to start because the source archive omitted its decision-rule document. That
  provenance failure stays recorded. The document was added and byte-checked
  before the final lease; the numerical/provenance checks were not bypassed.
- One offer disappeared before creation; the rejected request created no instance.

The final execution completed all four planned jobs with exit code zero.
See [the evidence index and replay commands](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-fast-lora-streaming-20261018/README.md).

The tiny local fixture is a wiring and numerical regression test. No SQuAD task
quality, end-to-end acceleration or memory saving is claimed by this change.
The interrupted October 16 resident comparison remains incomplete; its paid
budget was not reused. The subsequent explicit authorization to test before
push funded this bounded correctness campaign. No instance remains running.

## Publication check

The integration was copied onto the published PR head `efee5e0d` in an isolated
checkout, leaving unrelated local research commits out of the source diff. With
imports pinned to that checkout, the selected regression group passes **618
tests / 30 hardware skips**, plus **14 optimized-Python controls** in a disjoint
invocation (632 passes total). Recipe snapshots, config consumers, staged fields,
seeding, license headers and changelog checks pass **457 tests**. Full-path Ruff
and changelog fragment validation pass. The three `publish-*.xml` files and
`publication-validation.json` retain these checks; full repository CI remains
a separate requirement on the pushed commit.
