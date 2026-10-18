# D2 streamed Fast-LoRA: GPU evidence, 2026-10-18

Final RTX 3090 correctness run: 8 Soup SFT tests + 8 kernel regressions pass;
NF4 streaming float64-oracle checks pass for BF16 and FP16. Synthetic fixtures
only. No speed, VRAM reduction, SQuAD quality or full-D2 acceptance follows.

`gpu-evidence.zip` contains the final raw tensors, JSON reports, JUnit XML, logs,
package versions and the executed source manifest. The original two failing
FP16/NF4 pilot cases and the second pilot's pre-compute provenance refusal are
retained in named subdirectories. They are not presented as successful runs.
Provider/API/SSH/account files are excluded. `rental-summary.json` contains only
this campaign's costs, bounds and exact-instance deletion confirmations.

`archives.json` pins archive and source-part byte counts/SHA-256. The source
parts concatenate to the exact executed ZIP. Its manifest verifies 1,536 files;
Git metadata is absent in the exported snapshot and recorded as unavailable,
while the source/decision-rule/kernel hashes are retained. No fake Git head is used.
The archive preserves the full executed local snapshot, including research files
not added to the PR source tree. All ten published integration source/test files
match that snapshot after LF normalization; this does not assert whole-tree
identity between the archived snapshot and the PR commit.

Verify and extract the evidence with a standard ZIP tool, then run:

```powershell
python benchmarks/evidence/d2-fast-lora-streaming-20261018/replay.py `
  extracted/results/trainer-tensors --oracles extracted/results `
  --output replay.json
```

The reader needs CPU torch, safetensors and rich. Expected: 8 production cases,
1,832 paired comparisons and 116 oracle quantities. The committed
`gpu-trainer-replay.json` records the local replay. This rechecks saved tensors;
it does not rerun the model or measure GPU performance.

The trainer gates use the predeclared rtol=1e-2 / atol=1e-5 for low-precision
gradient/weight/logit pairs, exact initial adapters, and loss to three decimals.
The separate oracle uses fast error <= twice the unpatched PEFT error + 1e-8.
Missing cases, tensor keys, changed tolerances and nonfinite values fail replay.

`mutation-result.json` / `mutation-output.txt` show the isolated AMP-dtype
mutation rejected by all six supported-half NF4 regression cases. Historical
CPU results in `verification.json` predate GPU diagnosis; later XML files retain
the follow-up invocations. Do not sum overlapping test runs.

The replay/packaging files were produced after GPU source freeze; production
sources are pinned by the executed source archive and its manifest.
