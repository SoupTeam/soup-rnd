# D3 trained OLMoE attempt 04 — no verdict

Date: 2026-10-22. This is the retained fourth formal trained-step session, not a completed comparison or timing result.

- Frozen source commit: `3492ed83c2cd122ec0ad6318a2f40af6295ebcbc`.
- Frozen protocol SHA-256: `8a253f6c6f968d49e80bd235fa6eb9c0ac8f8f137c6384d239c5517e04fe180f` (revision 4).
- Session: `bb01c3a357ee4f1aaefda153734762e5`.
- The first `correctness-dense-native` cell was rejected before model construction. F1 mapping validation found that the prepared shard's recorded `st_dev` changed from 46 to 49 after restarting the container; inode, size, `mtime_ns`, and the exposed `loop0` association matched. The stale mapping correctly failed closed.
- No F1 control or model step ran for this cell; GPU stayed idle. The parent stopped with zero accepted cells.
- Classification: `incomplete` / no verdict. The frozen protocol allows no replacements. Refreshing mapping and controls requires a new prospective protocol revision and commit.

The archive retains the session JSON, child result, empty child log, exact LF runner wrapper, supervisor configuration, and the frozen revision-4 protocol. No model weights or GPU measurement outputs were generated.
