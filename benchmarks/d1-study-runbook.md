# D1 NF4 study: execution and current evidence

**2026-10-15: CPU preparation passed; CUDA and pretrained training NOT RUN.**
Work is isolated on `koko/d1-nf4-study`, based on Soup R&D main, not the D2 branch.
The decision rule is [gate-d1-nf4-rule.md](gate-d1-nf4-rule.md).

Research question: where does the current NF4 dispatch buy useful speed or memory,
and does forcing its checkpoint-visible Function offer a useful training trade?
Correctness, projection timings, full-step performance and QA quality are distinct
results. The original large-M fused backward proposal is not implemented here.

## Preparation already checked

- Existing #842/#968 CPU regressions: 16 passed, 13 hardware skips.
- CPU protocol: private and checkpoint-visible recycled buffers agree exactly;
  a plain-ctx negative control produces a different finite gradient.
- Four-layer/two-slot CPU streaming agrees with an unpatched resident reference
  on logits and every expected LoRA gradient, at M=8 and M=128. This is float32
  preparation only; CUDA acceptance uses its separately declared sizes/dtypes.
- Independent replay reads stored CPU tensors and their hashes, rechecks all
  quantities and detects the deliberately broken gradient.
- Prepared real SQuAD pilot: 256 train / 128 validation / 64 test, no overlap in
  IDs, article titles or contexts. Tokenizer/source revisions and SHA-256 pinned.
  All 448 examples fit the prompt+answer and generation limits without truncation;
  prompt/padding labels are masked. This is a bounded pilot, not the full D2 study.

The data directory is outside Git. Its `dataset.json` SHA-256 is
`65025efbd8b11960995d241fb7bb164c9d448abcb2085b9bb66ba2c74dfd8d22`.
It derives from D2 split-manifest SHA-256
`8af24d39c57893d178be67ec3321d381bd43b4059c86eb6f4ed619df2a175297`.
It shares the frozen data selection, not D2's source modifications or results.

CPU artifacts: [evidence/d1-preparation-20261015/](evidence/d1-preparation-20261015/).
The first CPU smoke failed to import the repository's test fixtures; its failure
record is retained. After correcting the repository import path, the second
attempt passed. A real tokenizer audit also caught a BatchEncoding/list API
difference; a regression test was added and all 448 examples passed after the fix.

## Commands

Use Python 3.10-3.12 and a dedicated virtualenv. Install Soup's dev dependencies
and `nvidia-ml-py` on the GPU box. Record the exact resolved stack; a different
bitsandbytes dispatch is not silently accepted. Commit the rule and harness
before CUDA work; the entry point refuses dirty/untracked executed source.
Each output path must be new. There is no overwrite or automatic failure retry.

CPU preparation/replay:

```bash
python benchmarks/harness/d1_nf4_study.py doctor --device cpu --output /tmp/d1-doctor
python benchmarks/harness/d1_nf4_study.py correctness --device cpu --dtype float32 --output /tmp/d1-cpu
python benchmarks/harness/d1_nf4_study.py replay --input /tmp/d1-cpu --output /tmp/d1-cpu-replay
pytest tests/test_d1_nf4_study.py tests/test_d1_qa.py tests/test_issue842_checkpoint_visible_nf4.py tests/test_issue968_bnb_nf4_path_divergence.py --no-cov -q
```

CUDA correctness, before timing:

```bash
python benchmarks/harness/d1_nf4_study.py doctor --device cuda --output /tmp/d1-cuda-doctor
python benchmarks/harness/d1_nf4_study.py correctness --device cuda --dtype float16 --output /tmp/d1-fp16
python benchmarks/harness/d1_nf4_study.py correctness --device cuda --dtype bfloat16 --output /tmp/d1-bf16
pytest tests/test_issue842_checkpoint_visible_nf4.py tests/test_issue968_bnb_nf4_path_divergence.py -m gpu --no-cov -q
```

The CUDA alias control uses a 192 MiB PACKED projection (768 MiB dense half). Tiny-model
CUDA streaming uses M=3 and M=2048. Counters must prove the expected routes.
The larger model shape and the exact-vs-negative tensors are stored for replay.

For a one-block projection SCREEN (no verdict), then the five-block pilot:

```bash
python benchmarks/harness/d1_nf4_study.py run --kind projection --pair dense-native --m 8 --n 4096 --k 4096 --dtype bfloat16 --blocks 1 --output /tmp/d1-screen
python benchmarks/harness/d1_nf4_study.py run --kind projection --pair dense-native --m 8 --n 4096 --k 4096 --dtype bfloat16 --blocks 5 --output /tmp/d1-q8
python benchmarks/harness/d1_nf4_study.py run --kind projection --pair native-forced --m 2048 --n 4096 --k 4096 --dtype bfloat16 --blocks 5 --output /tmp/d1-q2048
```

Every arm starts a fresh process/CUDA context. F1 watches power, suspend and
foreign reads. NVML additionally records pre/during/post compute and graphics
processes, physical UUID, SM clock, power limit and memory-controller utilization.
Unknown monitoring gives no verdict. The post-timing oracle checks against
unpatched native bitsandbytes. The same-session dense GEMM ceiling runs AFTER
peak recording. A forced Function that internally falls back to dense GEMM is
labelled `function:dense_fallback`, not fused.

Target-card RAM-tier dequantisation upper-bound probe:

```bash
python benchmarks/harness/d1_nf4_study.py run --kind ablation --pair baseline-nodequant --weights "$d1_model_snapshot" --seq 512 --steps 100 --warmup 20 --blocks 5 --output /tmp/d1-ablation
```

The nodequant arm uses cached ZERO weights, exactly the existing stream-probe
mechanism. Its model math is garbage. It estimates a ceiling and provides no
training correctness, task quality or achieved-kernel speedup claim. Do this
before expanding the performance grid; a small share can be a stop result.

Real-task pilot on the pinned TinyLlama snapshot:

```bash
python benchmarks/harness/d1_nf4_study.py run --kind train --pair native-forced --weights "$d1_model_snapshot" --data "$d1_prepared_data" --seq 1024 --steps 256 --warmup 20 --dtype bfloat16 --blocks 1 --output /tmp/d1-qa-screen
```

`d1_model_snapshot` must be the local Hub snapshot at revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Model safetensors are hashed per arm;
tokenizer files must match the data manifest. The full-step timer includes input
transfer, forward, answer-only loss, backward and AdamW. All 20 warm-up updates
are counted; this command evaluates after 276 updates, not 256. Greedy held-out
generation uses 64 new tokens, zero sampling and no context truncation. Report
EM/F1 by answerability and false answers on unanswerable questions. Metrics are
descriptive for 64 examples. No general hallucination or 8B multiplier claim.

Reproduce the pilot dataset on a clean host from the retained D2 split manifest:

```bash
python benchmarks/harness/d1_qa.py --manifest "$d1_split_manifest" --output "$d1_prepared_data"
```

No credential, cloud API, rental or publication is performed by these scripts.
GPU spending and transfer of the prepared source/data require the user's budget
and destination authorization. After a budget screen, size the next cohort from
measured runtime and F1 repeats; retain short/void cohorts as such.

## Remaining before a D1 result

The original preparation ledger is retained in the immutable rule. The
[2026-10-15 RTX 3090 pilot report](d1-rtx3090-report-20261015.md) records the
subsequent CUDA runs, their exact source cohorts, failures and limits. Read the
report rather than interpreting the original NOT RUN rows as current status.

For the repeated projection subset, use amendment B's `--warmup 5000`; do not
silently pool those rows with the earlier 20-step screen. Amendment C's package
compatibility checks are separate from primary bitsandbytes 0.50.1 timings.

Replay the QA evidence on CPU, using the hash-checked held-out gold:

```bash
python benchmarks/harness/d1_qa_replay.py --input "$d1_raw_results/cohort-a/qa-screen" --data benchmarks/evidence/d1-rtx3090-20261015/data --output /tmp/d1-qa-replay.json
```

This checks saved finite LoRA tensors, initial/final equality, all 276 losses,
ordered predictions and rescored metrics. It is an offline consistency check,
not an independent reviewer or performance acceptance.

The full five-shape/six-M/two-dtype grid, five-block real-task timing and
independent review remain outside this bounded pilot. The full repository test
matrix/CI is not established by the focused local checks. No D1 closure or
achieved fused-backward speedup follows from the pilot.
