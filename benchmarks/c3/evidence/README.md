# C3 CUDA evidence packets

Reviewer-accessible copies of the preserved evidence for the dense Llama 3.1 8B series.

| Packet | Recorded outcome | Archive SHA256 |
|---|---|---|
| `20261031_attempt017` | 20/20 arms, exact parity, all F1/clock/ownership gates pass; performance NO_VERDICT (5 of 20 required blocks) | `aa73755d…60ee5` |
| `20261031_attempt019` | 22/80 arms completed with exact parity; arms 14 and 19 void on F1 foreign reads; arm 22 interrupted (no records); series stopped | `7540b48f…a05f865` |

Each packet contains:

- `archive_manifest.json`: the archive's own member list (path, bytes, SHA256), unchanged.
- `evidence/`: every archive member byte-for-byte, including all per-arm `result.json` files
  (also the void and the interrupted arm), supervisor receipts, `io_calibration/receipt.json`,
  `foreign_read_control/receipt.json`, `cuda_tests.xml`, input receipts, `pipeline.json`,
  launch scripts and the frozen `source_manifest.json`.
- `PACKET.json`: what was left out and why. Only two members are omitted: the 128 MiB random
  calibration payload (its SHA256 stays in the calibration receipt) and `rental.json`, which
  holds the SSH endpoint and budget figures. `evidence/rental_public.json` is that file
  without those keys and names the removed keys and the original SHA256.
- Archived Python launch/preservation scripts are stored as `*.py.txt` (listed under
  `renamed_members`), so repository lint never touches frozen bytes.
- `frozen_gate/isolated_compare.py`: the measured copy of the comparison gate, whose SHA256
  is in `evidence/source_manifest.json`.
- `frozen_gate/{io_calibration,foreign_read_control,arm_validity}.py.txt`: exact measured
  dependency sources recovered from Git commit `79df4a84` and verified against the original
  source manifest's SHA256 and byte count. The text suffix preserves frozen source bytes.

## Replay

```bash
python benchmarks/c3/replay_evidence.py benchmarks/c3/evidence/20261031_attempt017 --summary
python benchmarks/c3/replay_evidence.py benchmarks/c3/evidence/20261031_attempt019 --summary
```

No GPU, network or private file is needed. The script checks every member against
`archive_manifest.json` and checks that all four gate sources in `frozen_gate/`
are byte-identical to the measured source before executing them. The foreign-reader gate
also imports its frozen calibration dependency. Later edits and Windows checkout line
endings in the live repository do not redefine historical gates. Then it recomputes the
numerical parity, F1/clock, GPU ownership, residency, direct I/O, calibration, foreign-reader and
repetition checks. Thresholds are the frozen ones. `tests/test_c3_replay_evidence.py` runs both
packets, modified-record and modified-gate controls, and independence from live gate edits.

## Source provenance

`PROVENANCE.json` retains the historical publication-time comparison with the C3 branch:
attempt017 matches 2909 of 2940 files at head, attempt019 3362 of 3392. Every C3 file that differs is
listed with its measured hash, its hash in the first three C3 commits (`f211051`) and at head,
and the reason: the public L2L integration that came after attempt017, the merge with main's
per-micro-batch L2L windows, the `close()` drain fix after review, harness-only changes, and
tests and docs. The remaining differences come only from merging main. Attempt019 measured
exactly the C3 production code of `f211051`. The cached-weight path, the disk reader and
residency selection are unchanged since both measurements.

Those counts describe that historical comparison, not later merged heads. Frozen-source
replay verifies the recorded measurements and does not qualify the current runtime as a
new CUDA execution. Original archive manifests and raw result records are unchanged.
