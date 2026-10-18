# D2 evidence archive

Raw measurements and superseded reports live in the immutable
[pre-cleanup snapshot](https://github.com/SoupTeam/soup-rnd/tree/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks).
The branch `koko/d2-evidence-20261018` keeps that commit reachable. Production code,
tests, benchmark programs, decision rules and current result reports remain in PR #5.
Removing raw files from the review diff does not remove the original Git history.

## Current reports

- [Soup SFT and layer-streaming integration](d2-fast-lora-streaming-20261018.md):
  synthetic FP16/BF16 correctness passes; real-data speed, memory and quality remain unmeasured.
- [RTX 3090 projection timing](d2-rtx3090-report-20261013.md): mixed results;
  no consistent acceleration or VRAM saving across the matrix.
- [TinyLlama strict baseline comparison](d2-training-baselines-20261014.md):
  Soup/PEFT gates pass; Liger/Unsloth strict cross-stack gates fail.
- [Complete-step external stack comparison](d2-stack-comparison-20261014.md):
  separate default-Boost policy, numerical drift retained, no equal-quality claim.

These are distinct experiments; the historical timing does not establish the
new Soup streaming integration's performance. NoRA and the real-data quality
study remain open. This cleanup changes no result or acceptance criterion.

## Archived measurements and replay

Each original evidence index retains its hashes, reconstruction/replay commands,
failed/VOID runs and stated limits. Follow it within the archived checkout.

| Experiment | Immutable evidence index |
|---|---|
| Soup streamed SFT, October 18 | [Raw tensors, GPU/CPU XML, source and replay](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-fast-lora-streaming-20261018/README.md) |
| Projection timing, October 13 | [Timing parts, raw samples and audit](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-rtx3090-20261013/README.md) |
| Strict full-model baselines, October 14 | [Compact evidence and its limits](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-baselines-20261014/README.md) |
| Default-Boost stacks, October 14 | [Timing, witnesses and replay](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-stack-v3-20261014/README.md) |
| v7 NF4 correctness, October 12 | [Evidence](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-rtx3090-20261012-v7/README.md), [historical report](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/d2-rtx3090-report-20261012-v7.md) |
| v6 pilot, including failures | [Evidence](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-rtx3090-20261012/README.md), [superseded report](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/d2-rtx3090-report-20261012.md) |

The generated [hashed Linux dependency lock](https://github.com/SoupTeam/soup-rnd/blob/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/d2-training-baselines.lock)
is archived too; the baseline protocol restores its exact bytes before installing.
To access the complete published evidence without changing the current checkout:

```bash
git fetch origin koko/d2-evidence-20261018
git worktree add --detach ../soup-d2-evidence b2923981b7cb938852999459b19c968e5fbeb37f
```

## Backup verification

`D2_EVIDENCE_20261018.zip` is a separate local backup of all 138 removed files,
plus `ARCHIVE_MANIFEST.json` with per-file SHA-256, byte length and original Git
blob ID. It contains exact published Git bytes, rather than CRLF working copies.
Every payload was extracted and compared byte-for-byte against the original
commit; all ZIP CRCs also pass. Already-compressed payloads are stored unchanged.

- Source commit: `b2923981b7cb938852999459b19c968e5fbeb37f`.
- Original payload: 416,969,480 bytes in 138 files.
- Backup ZIP: 417,035,082 bytes.
- ZIP SHA-256: `604bec4291b80334aa44508c4a42ad946fc0ee51b1c1fd6b958243fde3c79198`.

The local ZIP is a second copy; reviewer access relies on the immutable GitHub
snapshot above and does not require access to the author's machine. Archives
retain their original completeness limits; this copy check does not add missing
historical measurements or establish a new GPU result.
