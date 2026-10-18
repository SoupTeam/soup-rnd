# D1 evidence archive (2026-10-18)

Raw evidence and superseded reports moved out of PR #16 for reviewability.
This branch is an archive, not a merge candidate. It does not change Soup code.

Source snapshot: `94bf6605a5a64d59177a019a4f837e01f56f4bd9` in
https://github.com/SoupTeam/soup-rnd.

The split standard ZIP preserves every tracked D1 evidence file and every historical
report removed during cleanup. JSON, logs, CUDA tensor witnesses, failed/rejected
runs, data/license notices, executed source archives, manifests, prior audits and
original compressed payloads are preserved byte-for-byte as Git blobs. No numeric
records, failed controls or validity thresholds have been edited. Embedded copies
inside original JSON/archives remain intact so their old checksums stay valid.
Repeated working-tree copies and superseded status reports are removed from the PR.

## Download and verify

Clone only the archive branch (no GPU, training packages or paid compute required):

```bash
git clone --depth 1 --single-branch --branch koko/d1-evidence-archive-20261018 \
  https://github.com/SoupTeam/soup-rnd.git d1-evidence-archive
cd d1-evidence-archive
python restore.py --output ../d1-evidence-restored
```

Python 3.10+ standard library only. The helper joins the numbered parts in manifest
order, verifies each part and the complete ZIP, checks the full file inventory,
then checks every original file's size, SHA-256 and Git blob ID before restoring.
The output directory must be new; existing files are never overwritten.

`manifest.json` lists all original repository-relative paths and hashes.
`SHA256SUMS` also supports manual verification. Manual concatenation of all numbered
parts in order produces `D1_EVIDENCE_20261018.zip`, a normal ZIP with original paths.
Do not use a text-mode shell concatenation command for binary parts on Windows.

Restored files appear under `../d1-evidence-restored/benchmarks/`.
Existing replay commands that expect `benchmarks/evidence/...` work after copying
the restored evidence directory into the study checkout. Reports under the same
directory are historical snapshots, not a newer verdict.

## Evidence map

- `d1-final-f1-20261016`: final decoder correctness, full-step memory and rejected
  ABBA timing, raw CUDA snapshots, GPU regressions, independent audit, costs/releases.
- `d1-native-20261016`: default BF16 arithmetic, checkpoint/negative-control gates,
  native split-K traces, projection and small decoder pilots.
- `d1-wmma-20261016`: earlier full-FP32-reduction WMMA experiments and failures.
- `d1-kernel-screen-20261016`: initial kernel screen and mismatch witnesses.
- `d1-a100-k-20261016`, `d1-k-precoverage-replay-20261016`,
  `d1-efg-replays-20261015`: earlier K study, audits and replay derivatives.
- `d1-rtx3090-20261015`: earlier projection/QA study, raw parts and data notices.
- `d1-preparation-20261015`: CPU preparation and negative-control witnesses.

Current conclusion remains: scoped BF16 exact-parity research, measured full-step
VRAM savings on the stated synthetic fixture, **NO_VERDICT for speed**.
Archiving is packaging only; it does not complete full D1 acceptance.
