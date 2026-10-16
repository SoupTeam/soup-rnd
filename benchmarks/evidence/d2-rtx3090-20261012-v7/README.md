# v7 NF4 rounding handoff

The [current report](../../d2-rtx3090-report-20261012-v7.md) verifies the tested
numerical matrix: 96 NF4 cases and 895 independently replayed positive quantities.
Performance remains NO VERDICT; this is not full D2 completion or a CI waiver.

Download [the v7 handoff ZIP](d2-rtx3090-20261012-v7-handoff.zip), SHA-256:

```text
3323b99c08b927f19c6bc329f0dc4aae4032beb6aff4dda474f51f2a7bdc5a0c
```

The internal and adjacent `package-manifest.json` record every payload's SHA-256
and size. Original archives are unchanged. Both nested and outer CRC/hash checks
pass, and no account API credentials or private SSH keys are included.

| Path in handoff | Contents |
|---|---|
| `sources/d2-cloud-source-20261012-v7.zip` | Executed source bytes, inventory and parent/diff provenance |
| `gpu/run2-evidence.zip` | Four layout experiments, final v7 run, all native/dense NF4 matrices, controls, logs and source manifests |
| `gpu/run2-policy-evidence.zip` | Same-host one-factor cast/add prototype: 48 passing cases |
| `audit/independent-audit-v7.json` | 116 tensor files, 895 quantities, exact/tolerance distinctions and excluded timing rounds |
| `offline/` | CPU-only replay tools and unchanged numerical criterion |
| `local-tests/` | Final selected command logs/JUnit and scalar midpoint regression |
| `rental-summary.json` | Verified cleanup and observed total balance decrease (~$0.2492) |

The [v6 handoff](../d2-rtx3090-20261012/README.md) retains the original failed
source/results. This new bundle does not rewrite or replace those failures.

Extract `gpu/run2-evidence.zip` to a fresh directory, then replay from this PR
checkout (or with `offline/d2_audit_evidence.py`). Only torch, safetensors and Rich
are needed; CUDA is not needed. Set the extracted directory as `BASE` below:

```bash
python benchmarks/harness/d2_audit_evidence.py "$BASE/run2-acceptance" \
  --aligned-dir "$BASE/run2-acceptance/nf4-dense-bf16-nf4-aligned" \
  --aligned-dir "$BASE/run2-acceptance/nf4-dense-fp16-nf4-aligned" \
  --aligned-dir "$BASE/run2-acceptance/nf4-bnb-bf16-tiny" \
  --aligned-dir "$BASE/run2-acceptance/nf4-bnb-bf16-nf4-aligned" \
  --aligned-dir "$BASE/run2-acceptance/nf4-bnb-fp16-tiny" \
  --aligned-dir "$BASE/run2-acceptance/nf4-bnb-fp16-nf4-aligned" \
  --output /path/to/new-audit.json
```

Expected: exit 0, 116 tensor files / 895 quantities / 0 positive failures. The
intentionally wrong streaming controls remain failing raw reports; their separate
`control-summary.json` files verify numerical rejection through 20 custom sites.
All final-run timing arms remain VOID; they must not be used for a speedup claim.

The executed production files match this update after CRLF/LF normalization. New
local midpoint regression coverage was added after the GPU source snapshot.
