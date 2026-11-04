# G5 synthetic CPU pilot evidence

G5 remains STOPPED under C5 PR #26. The mixed GPU experiment was NOT run.
These random-init granite CPU records are not OLMoE quality evidence and never
support a reportable G5 verdict. Publishing them does not reopen G5 or post the
proposed tracker comment.

## Published records

Source: `research/g5-mixed-precision-experts` at
`76b78fcd4de308d231fd353f2eb603cdb90f10de`.

- `selection/selection.json`, `selection/maps/*.json` and `selection/bytes.json`
  retain the original calibration statistics, selected maps and computed bytes.
- `mixed/*/run.json` holds twelve mixed CPU fixture runs.
- `baselines/*/run.json` holds the seven C5 CPU fixture runs repeated on that box.
- `acceptance-20261022/replay/verdict.json` is the corrected historical replay,
  not a fresh publication check. Every arm has `outcome: no_verdict`, the G5
  decision is `no_verdict` and `reportable` is false. Pipeline computations are
  retained separately and are not experiment verdicts.
- `provenance.json` lists each source/published UTF-8 text digest and the JSON
  pointers whose string values changed. Absolute interpreter and script paths
  became `python` and repository-relative paths. The temporary data directory
  became `<pilot-data-dir>`. Numeric JSON tokens, hashes, timestamps, measured
  records and historical outcomes did not change. These are sanitized copies,
  not byte-identical raw captures.

The source calls this the 2026-10-21 pilot; its records use 2026-10-22 timestamps
in UTC+05:00. Records name an uncommitted working tree based on `8ba2552e` and
older harness hashes. They do not establish that this publication checkout or
its revised validity checks passed. A new CPU pilot is required before any
future GPU experiment if committed harness hashes differ, as the proposed rule
requires.

All operational logs, local review scripts, setup/rental operations, environment
captures and pre-fix reproductions were omitted. The obsolete top-level verdict
and SVG are also omitted. That legacy synthetic verdict used `pass` and `rule`
before the non-reportability fix; it is unchanged in the source worktree and is
not a mixed-GPU result. The existing repository's C5 synthetic reference is
reused rather than copied here.

## CPU verification commands

Run from the repository root with the normal development environment. No model
weights or dataset download is needed. The replay reads saved JSON only. Use a
scratch output directory so historical evidence is never overwritten.

```bash
python -m pytest tests/test_g5_bytes.py tests/test_g5_select.py tests/test_g5_verdict.py tests/test_g5_shell.py --no-cov -q
python -m ruff check benchmarks/harness/g5_bytes.py benchmarks/harness/g5_select.py benchmarks/harness/g5_train.py benchmarks/harness/g5_verdict.py tests/test_g5_bytes.py tests/test_g5_select.py tests/test_g5_verdict.py tests/test_g5_shell.py
bash -n benchmarks/harness/g5_run_all.sh benchmarks/harness/g5_train_all.sh
python benchmarks/harness/g5_bytes.py --fixture granite --out .g5-cpu-check/bytes.json
python benchmarks/harness/g5_verdict.py benchmarks/results/g5-cpu-pilot/mixed \
  --baselines benchmarks/results/g5-cpu-pilot/baselines \
  --control benchmarks/results/g5-cpu-pilot/baselines/nf4-s42-a1/run.json \
  --control-reference benchmarks/results/c5-train-synthetic/nf4-s42-a1/run.json \
  --selection benchmarks/results/g5-cpu-pilot/selection/selection.json \
  --bytes benchmarks/results/g5-cpu-pilot/selection/bytes.json \
  --synthetic --out .g5-cpu-check/replay/verdict.json
python benchmarks/harness/g5_verdict.py plot .g5-cpu-check/replay/verdict.json \
  --out .g5-cpu-check/replay/replot.svg
```

The fixture-size command creates only a tiny random-init model on CPU. Some
numerical unit tests require the optional torch, transformers, bitsandbytes and
safetensors packages and skip when unavailable. Shell regressions require Bash
and do not train a model. Replay JSON must remain `synthetic: true`,
`reportable: false`, with `decision.g5: no_verdict` and every arm `no_verdict`.
Revised validity checks may reject historical records that lack current source
provenance. Do not turn a pipeline computation into a PASS or FAIL claim.

These commands are a handoff for the integration owner, not a claim that they
were run while preparing this publication. Remove the scratch output directory
after inspection; it is not evidence to publish.
