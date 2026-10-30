# D3 trained OLMoE attempt 05 — no verdict

Date: 2026-10-22. Formal session under protocol revision 5, commit `0035af1ec02a896ca25fd5b755e8878223201198`.

- Session: `5617251ce0f14836b2eff51ecc548cbb`.
- Frozen protocol SHA-256: `5beb564816c12018f2f360035f80779db6d669653cdcb7d51a3aa4fff4296c0e`.
- The first `correctness-dense-native` cell completed successfully with F1 `ok`, exact native/restored controls, and four retained step-oracle tensors.
- The next `correctness-dense-sequential` cell stopped before model construction and before F1. The child looked for its oracle at `correctness/dense-native/result.json`, but the series had written it at `correctness-dense-native/result.json`.
- Session outcome: incomplete / no verdict. One correctness cell was retained; no timings or profiles ran. The fixed 84-cell budget has not been completed. No replacement is allowed under revision 5.

The tracked evidence archive retains the full session metadata, native and sequential result JSON, frozen source packet, setup identities and launch files. It omits only the four 85,265,708-byte native oracle tensor files because the resulting single archive would exceed GitHub's 100 MB file limit. Their exact SHA-256 values are listed in `native-oracle-excluded.json`. The full 269,056,105-byte capture, including those tensors, is preserved separately with SHA-256 `faa177763081303141a2562751e87510a2b42a5e1ce4035145f5026844c1f599`.

Revision 6 centralizes the native-oracle path used by both parent and child and adds a direct path-layout regression. A new frozen commit is required before another formal session.
