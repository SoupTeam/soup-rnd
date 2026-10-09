# Mistral-7B E2: two-epoch quality record

Model revision: `caa1feb0e54d415e2df31207e5f4e273e33509b1`.
Scope: seed 42; 2000 train / 200 validation examples;
4000 optimizer steps; maximum sequence length 128.

| K | Baseline loss | E2 loss | Loss increase vs K=32 | Final fingerprints |
|---|---:|---:|---:|---|
| 8 | 1.027287721634 | 1.027287721634 | +17.1645% | Equal |
| 16 | 0.947365105152 | 0.947365105152 | +8.0491% | Equal |
| 24 | 0.889599442482 | 0.889599442482 | +1.4608% | Equal |
| 32 | 0.876791179180 | N/A | +0.0000% | Control |

For K=8/16/24, E2 recorded 2000 misses in epoch 1 and
2000 hits with zero misses in epoch 2.

Combined dataset SHA-256: `3a713df6c2d63eb4670ebb8ab19e6791e3001241c494bf6db28559b0838be1e1`.
Reported PyTorch version: `2.14.1+cu130`.

## Interpretation and limits

- E2 is compared with the no-cache baseline at the SAME K.
- Top-K versus K=32 is a separate quality trade-off.
- Fingerprint equality is what these long-run reports record;
  this summary does not perform a new elementwise tensor comparison.
- This is one seed and one truncated validation setup, not a gate-suite result.
- Gate suites, Small MoE and final acceptance remain outside this record.
- Diagnostic timings must not be merged with the earlier speed benchmarks.

## Recorded configurations

These YAML files are unchanged snapshots of the completed Kaggle runs.
They contain original local checkpoint/output paths, not portable templates.
For a rerun, preserve the model revision and use NEW output directories
so that a cold-cache run cannot reuse an old cache.
The completed runs used run_quality.py with --steps 4000.

## Source reports

- `benchmarks/e2/quality/results/k8_two_epochs_376d08f2/baseline.json` — SHA-256 `1a3a101f715db89b40a2d77186781cd8a80782deb14f75ce5118bb8d71dbd478`
- `benchmarks/e2/quality/results/k8_two_epochs_376d08f2/cached.json` — SHA-256 `3651562bbb6220f98b0d51281d3ddb39debf7272c455dc6603b80d83c04cd325`
- `benchmarks/e2/quality/results/k16_two_epochs_6f003b48/baseline.json` — SHA-256 `b5ad1df10016769307c8608a9fff53c9fc58aaf7d5ff95d08722220059a4169f`
- `benchmarks/e2/quality/results/k16_two_epochs_6f003b48/cached.json` — SHA-256 `bb151e6a94130c0dbb5bc94c8746df3f670ec01e88c84b0e7cc3df59561a1a41`
- `benchmarks/e2/quality/results/k24_two_epochs_a0e0cecc/baseline.json` — SHA-256 `beebc59fbfd6dc32a5445462d0f9e76d197b6496fa82094dd18b4430b9b69ccd`
- `benchmarks/e2/quality/results/k24_two_epochs_a0e0cecc/cached.json` — SHA-256 `7bf371c2de213ac8744652b9b56f18962992d9315380d431989c5a1500e42c94`
- `benchmarks/e2/quality/results/k32_two_epochs_control_d9b524ec/control.json` — SHA-256 `4344b68095957609382f08e22b810682e4f8baeecc9c3d0c0f2eaba7025e2d53`
