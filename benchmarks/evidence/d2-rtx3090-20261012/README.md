# RTX 3090 D2 handoff

**INCOMPLETE: three NF4 quantities fail the unchanged rule; timing is NO VERDICT.**
Read the [measurement report](../../d2-rtx3090-report-20261012.md) before interpreting
passing subsets. The bundle contains original failed runs as well as the candidate.

Download [the handoff ZIP](d2-rtx3090-20261012-handoff.zip) and verify its SHA-256:

```text
bd3d02a9937a56cc5a0d800a676600e751bc4a4fd80875d94deaf1cb359db0db
```

`package-manifest.json`, both beside and inside the ZIP, lists every payload's
SHA-256 and size. Original nested ZIPs are unchanged and CRC/hash checked.

| Path in handoff | Contents |
|---|---|
| `sources/` | Frozen original v2/v4/v6 source archives |
| `gpu/d2-evidence-20261012.zip` | Original v2 GPU run, including failures |
| `gpu/d2-evidence-20261012-v4.zip` | Intermediate v4 GPU run |
| `gpu/d2-final-v6-complete.zip` | Final v6 run, both aligned NF4 probes, setup/Triton logs |
| `audit/independent-audit-v6.json` | 68 tensor-file replays / 567 quantities, three failures, timing exclusions |
| `offline/` | Final audit/replay scripts and the original numerical criterion helper |
| `local-tests/` | Selected command logs and JUnit, including the initial coverage-only failure |
| `rental-summary.json` | Price, observed balance delta and verified resource cleanup |

Extract the final GPU archive into a fresh directory. Use the report's replay
command, or run `offline/d2_audit_evidence.py` directly with that directory and the
two aligned directories. Replay requires torch, safetensors and Rich; CUDA is not
needed. A new output filename is required. Exit 1 with exactly three failed
quantities is expected. The audit independently requires all fixture outputs and
gradients, so removing a gradient from all comparison arms cannot produce a pass.

Executed source bytes are authoritative in `sources/` and the GPU source manifests.
Git checkouts can normalize line endings. Audit/replay scripts and their new tests
were finalized locally after the GPU run; they are not additional GPU evidence.
No API credentials, private SSH keys or cloud balance values are included.
