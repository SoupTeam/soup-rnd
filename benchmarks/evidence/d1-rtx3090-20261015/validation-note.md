# Local validation provenance

The earlier focused checks reached 64, then 68 passes as the offline projection
replay was added. The final branch merges upstream main `d4e796730a3ac76d77b23fe6c703c51264540824`,
preserving the measured CUDA cohort commits. Upstream added 20 FP8 preflight
tests; the NF4 runtime file and F1 validity implementation did not change.

The first post-merge local invocation reused the original D2 virtualenv's
editable import path. All 20 new upstream cases therefore imported the older
`layer_stream.py` without `fp8_source` and failed. The log is retained as
`post-main-tests-wrong-source.log.gz`; this was a source-selection error, not a
newly diagnosed D1 regression.

The corrected invocation sets `PYTHONPATH` to this worktree's `src` before
starting Python and asserts the imported module belongs to that directory.
No package in the original D2 virtualenv is reinstalled or modified. The same
focused tests plus the 20 upstream cases then report **88 passed, 13 skipped**;
full `ruff check src/soup_cli/ scripts/ tests/ benchmarks/` is clean. Logs:
`post-main-tests-correct-source.log` and `post-main-ruff.log`.

These checks are Windows CPU regressions. They are not the full repository test
suite, cross-platform CI matrix, independent review or final-head CUDA reruns.
