# D1 RTX 3090 evidence, 2026-10-15

**Bounded pilot complete; D1 open.** Read the
[report](../../d1-rtx3090-report-20261015.md) and
[machine-readable result index](result-index.json). CUDA timing is pinned to
bitsandbytes 0.50.1; isolated 0.50.2 checks concern correctness only.

| Check | Recorded outcome |
|---|---|
| FP16/BF16 large-alias and four-layer streaming | PASS in Initial, A and C |
| CPU correctness replay | 216 exact quantities; six broken controls detected |
| Existing primary GPU regressions | 4 passed, 8 version skips, 17 deselected |
| Existing isolated bitsandbytes 0.50.2 GPU regressions | 12 passed, zero skips, 17 deselected |
| Repeated M=8, M=2048 projection comparisons | Five accepted blocks each; no measured advantage |
| A/C cached-zero ablation | Five accepted blocks; timing-only garbage math |
| Projection tensor replay | PASS: 42 exact native arms, 14 declared dense/fused differences |
| Real-task QA replay | PASS: all 176 changed matrices, 276 losses and 64 predictions match across four arms |
| Real-task speed verdict | NO_VERDICT: one ABBA block only |
| Rental cleanup | Deleted and verified absent; provider reports $0.47, final invoice unsettled |
| Final focused Windows CPU regressions | 68 passed, 13 hardware skips; full lint clean |
| Full D1 acceptance / independent review | Outstanding |

## Complete raw witness package

`raw-proof/manifest.json` pins 21 ordered chunks and every archived file.
The gzip tar archive is **87,304,984 bytes**, SHA-256
`81332f7e82a54dd883cc1761efcebe543270da9e3709e6c0e94eef4f5643f4e1`.
It restores **622 files / 2,109,230,258 logical bytes**. Identical byte strings
are stored once and referenced by tar hard links; the materializer copies them
back to ordinary files for Windows portability. The internal `raw-manifest`
lists 621 files and excludes itself; the outer manifest covers all 622.

This package includes every attempt, every oracle tensor, all QA initial/final
adapters, individual step timings, telemetry, model/data hashes, dependency
records, command lines and controller logs. No model download, GPU, cloud
credential or hosted service is needed for offline replay. CPU PyTorch and
Rich suffice. Have at least 2.2 GB free and choose a new output directory.

From the repository root, using paths that work on Windows and Linux:

```bash
python benchmarks/harness/d1_materialize.py --manifest benchmarks/evidence/d1-rtx3090-20261015/raw-proof/manifest.json --output ../d1-review-20261015
python benchmarks/harness/d1_nf4_study.py replay --input ../d1-review-20261015/results/cohort-a/correctness-bfloat16 --output ../d1-review-bf16
python benchmarks/harness/d1_projection_replay.py --input ../d1-review-20261015/results --output ../d1-review-projections.json
python benchmarks/harness/d1_qa_replay.py --input ../d1-review-20261015/results/cohort-a/qa-screen --data benchmarks/evidence/d1-rtx3090-20261015/data --output ../d1-review-qa.json
```

Repeat correctness replay for `float16` and the Initial/C cohorts with fresh
output paths. Initial files are directly under `results/correctness-*`; A and
C are under their named subdirectories. This checkout's postprocessing tools
do not rewrite raw files or turn a prior void timing into a valid one.

All chunks and every extracted file were independently size/hash checked
during materialization. The restored package then passed all six correctness
replays, the projection replay and the QA replay. This is reproducible offline
consistency evidence by the experiment author, **not independent acceptance**.

## Records and limits

- `replay-*/result.json`: hash-verified correctness replay records.
- `projection-replay.json`: all 56 stored projection comparisons.
- `qa-full-archive-replay.json`: QA tensors, losses, predictions and pinned gold.
- `qa-replay-attempt1.json`: preserved first failure from aggregate F1 rounding;
  the [replay note](../../d1-offline-replay-note-20261015.md) explains the
  metric-only 1e-12 bound. Tensor/loss/prediction equality remains exact.
- `rental-summary.json`: sanitized instance-specific charge/cleanup record.
- `local-focused-tests-final.log` and `local-ruff-final.log`: final local checks;
  the earlier 64-test record predates the projection replay additions.
- `data/`: prepared split and original D2 manifest, with
  [SQuAD data attribution](data/NOTICE.md).

The earlier `qa-materialized-replay.json` checks a separate intermediate QA-only
archive; the delivered complete package is `raw-proof`. Source cohort hashes
remain in every worker record. Later type annotations and offline tools do not
make this a final-head CUDA validation. Full repository CI and independent
review remain separate gates. The 64-example endpoint is not evidence of a
general hallucination improvement, an 8B/70B gain or a new fused backward.
