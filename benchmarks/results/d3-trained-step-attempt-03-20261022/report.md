# D3 trained OLMoE attempt 03 — no verdict

Date: 2026-10-22. This is the retained third formal trained-step session, not a completed comparison or timing result.

- Frozen source commit: `46937f49994edd5667769ba4ec4f161d2154997f`.
- Frozen protocol SHA-256: `83af5ce45389f7f1796380459bcce3b55b08ff26d7d55e617e95d50fb49ac8a1` (revision 3).
- Session: `d1238f4a055a4f189e5e69304f87b29e`.
- `correctness-dense-native` completed successfully: four native optimizer steps, exact internal consistency, and F1 `ok`. This is one of 84 cells, retained as evidence but not a study result.
- The next `correctness-dense-sequential` child was rejected by the CLI before model construction because the frozen `child` parser omitted the `--oracle` argument passed by the series runner. The parent stopped; no second cell result, F1 or measurement exists.
- Classification: `incomplete` / no verdict; no replacement was made. Fixing the parser requires a new commit and protocol revision.

The archive retains the session, native-cell result and both cell logs, runner and supervisor configuration. It omits the four 82 MiB native oracle tensor payloads; their SHA-256 values remain recorded in `result.json`. The passing native cell does not establish a comparison, speed result or D3 acceptance.
