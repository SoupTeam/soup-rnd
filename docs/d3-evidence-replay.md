# Replay the completed pretrained D3 studies

The original raw archives are available outside Git in the private
[SoupTeam/soup-evidence D3 folder](https://huggingface.co/datasets/SoupTeam/soup-evidence/tree/bacdc0556962240d2f6b29b5817b0aea78dccf68/d3).
The 2026-11-07 upload preserved both ZIPs unchanged; their remote sizes and
SHA-256 identities match the table below. README and SHA256SUMS.json are alongside
them. Reviewers need access to that private dataset. This confirms artifact
publication, not that a reviewer has already downloaded or replayed the files.
Obtain the actual ZIPs before running the command below: summaries and payload
manifests alone cannot replay native tensors, F1 records or CUDA traces.

| Study | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Staged public SFT/L2L, `f781e927` | `d3-public-trainer-results-f781e927-01.zip` | 5173245384 | `73383ef1cc1cdd9a2182dbdf9f7ed639eb7c0238074a4c46f2a68af9bf285b0a` |
| Packed-fused public SFT/L2L, `921d86a1` | `d3-fused-trained-921d86a1-20261101.zip` | 2942937841 | `3221d8ec19dbb01cbb1cadce9927f5cb284250e6a8070dd98328ddb6152fe6c1` |

Both are complete raw result archives, including their original manifests. They
retain per-cell results, native tensor oracles, negative/restored controls, F1
records, profiling traces and measured source receipts. Original checkpoint/cache
payload omissions remain explicit; the replay audits recorded evidence rather
than downloading weights or repeating training. Historical failed/incomplete
sessions remain published separately and are not counted among these results.

## One CPU-only command

Use a checkout of this PR that contains `benchmarks/harness/d3_review_replay.py`
and Git history including the two measured commits. A shallow clone without the
measured commits is insufficient. Use Python 3.12 and an isolated environment;
GPU access, a CUDA toolkit and two physical drives are unnecessary.

Install the project and CPU auditing dependencies in that environment:

```sh
python -m pip install -e .
python -m pip install torch==2.15.0 safetensors==0.8.0 ijson==3.6.0
```

Run from the repository root, substituting the paths where the reviewer received
the two ZIPs. The output directory must be outside the checkout:

```sh
python -m benchmarks.harness.d3_review_replay --staged /downloads/d3-public-trainer-results-f781e927-01.zip --fused /downloads/d3-fused-trained-921d86a1-20261101.zip --out /scratch/d3-replay
```

The same command works in Windows CMD/PowerShell with quoted Windows paths.
Either archive can be replayed separately by omitting the other argument.
Allow at least 35 GB of additional free disk space for the extracted evidence,
besides the original 8.12 GB ZIPs. `ijson` streams the large CUDA traces; the
reader preserves every ordered kernel name and does not replace the strict
profile comparison with aggregate counts.

## What a successful replay verifies

- The original full ZIP SHA-256, every manifest payload and ZIP CRC while reading.
- Safe extraction; duplicate/unsafe names, changed existing files and symlinks
  are refused. An interrupted extraction can reuse already verified files.
- Every declared source blob against the actual measured Git commit, not merely
  the source's own checksum list: 787 staged and 805 fused blobs.
- The original analyzers' native tensor/F1/control/profile and timing checks:
  4,362 native tensor payloads for staged and 2,908 for fused.
- Byte-for-byte equality of regenerated `analysis.json` with the published file.

Success writes `staged/replay-verification.json` and
`fused/replay-verification.json` below the chosen output directory, together with
the regenerated analyses. Failures exit nonzero; no tolerance is relaxed and no
new measurement is launched. Offline evidence replay can run on Apple Silicon;
it does not execute the public expert-training path whose CPU arithmetic there
is explicitly unqualified.

The staged numerical study remains 234 cells and the fused study 112 cells.
A successful replay is evidence verification, not a new GPU execution, a
convergence result or reviewer/task-completion sign-off.

## Author validation of this command

The command above was executed against both complete local ZIPs. The staged
replay verified 6,119 payloads, 787 measured Git blobs and 4,362 native tensors;
the fused replay verified 4,131 payloads, 805 measured Git blobs and 2,908 native
tensors. Both regenerated analyses match their published bytes exactly. The
receipts include the auditing source hashes:

- [Staged local replay](../benchmarks/results/d3-public-trainer-attempt-03-f781e927-20261030/local-replay-verification.json)
- [Fused local replay](../benchmarks/results/d3-fused-trained-921d86a1-20261101/local-replay-verification.json)

This is the author's replay, not an independent reviewer's execution or proof
that the reviewer has received the raw files.
