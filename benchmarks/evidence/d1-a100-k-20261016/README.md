# A100 K final derived evidence, 2026-10-16

Start with [the research summary](../../d1-final-research-20261016.md) or [the Russian handoff](D1_TEAMLEAD_RU.md). Global D1 acceptance remains incomplete; NO_VERDICT cells retain the original guard.

The packet files listed in packet-manifest.json are byte-identical to the validated local handoff. consumer-review-status.json records the completed local replay; handoff-bundle.json identifies the ZIP. Raw archive bytes are supplied in that handoff, not this Git directory.

The scoped `.gitattributes` disables text normalization so Git preserves the manifest's exact file bytes on Windows and Unix.

Put d1-a100-k-results.tgz next to this directory, create a separate checkout at the exact measured source from index.json, then run:

```text
python review_packet.py --repo <measured-checkout> --output <fresh-directory>
```

Dependencies: Python, CPU-capable torch, rich, numpy, matplotlib. The actual producer CPU stack is recorded in cpu-audit-environment.json. Replay does not allocate cloud GPUs or download the model. Raw restoration requires disk space and hard-link support. Independent reviewer acceptance remains pending.
