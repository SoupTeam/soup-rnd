# E/F/G offline audits, 2026-10-15

These are derived CPU audits of retained CUDA evidence, not new GPU executions
or independent reviewer acceptance. The full projection grid remains incomplete.

E/F archive SHA-256:
`22b1c446ea9ef9fddcc117525c39bb53540d058018af5aa71480ccdd9c47e990`.
G archive SHA-256:
`8fe55d9f08fa74b1d760674f000fbef1d8f8ddaa5f84506e3ce1dd4f723906c3`.
The local raw archives and manifests are described in `../../D1_RESUME_20261015_RU.md`.
All raw file hashes were verified on materialization; replays verify tensor bundle hashes.

- `full-{dtype}.json`: E correctness CPU replays; original source 241b6bff.
- `full-f-{dtype}.json`: F correctness CPU replays; original source ca1c5b00.
- `g-{dtype}.json`: G correctness CPU replays; original source 49e91d7d.
- `qa-e.json` / `qa-f.json`: all 20 stored QA runs per cohort, block IDs 0–4,
  exact initial/final adapters, loss and prediction agreement; all 176 adapter
  matrices changed. Both cohorts have the same held-out quality. E's timing
  remains rejected; numerical replay does not repair clock violations.
- `projections.json`: all 20 stored F projection witnesses, ten exact native
  results and ten explicitly declared dense/fused arithmetic differences.
  All of these projection timings remain rejected for clock spread.
- `index-f.json`: F's incomplete timing index, rebuilt using F's plan in
  source f7139ad8. It has 2/98 cases, 0/96 completed projections and 64 attempted
  arms; it must not be treated as the later H/I plan.
- `g-diagnostic-summary.json`: two accepted single-block clock diagnostics,
  both still NO_VERDICT for performance. No G full-grid timing claim.

## Existing F training observation

Five valid fresh-process ABBA blocks, 20 runs, 276 updates per run,
fixed TinyLlama revision/data/seed. F additionally used an independent synthetic
thermal prelude; H/I do not. Native versus forced Function median full step:
241.1833 versus 255.9017 ms; A/B 0.9424841. Both peak allocated 863,867,904 bytes,
reserved 1,042,284,544 bytes. Descriptive non-padding throughput 1076.02 versus
1006.77 tokens/s. Thus this case shows no forced-path speed or allocator benefit.

Held-out 64 examples: EM 53.125, F1 54.4642857; answerable 35 examples:
EM 34.2857143, F1 36.7346939; unanswerable 29 examples: 7 false answers.
Exact repeated seed agreement is not independent-seed quality validation.
No before-training baseline or larger-model generalization was measured.

Cached-zero F ablation A/B 1.2287771 is timing-only garbage math. It does not
establish an achievable training optimization. Historical pilot and cohort
timings are never pooled with the new physical A100 cohort.

## Reproduction

Use `d1_materialize.py` with the corresponding raw manifest and archive.
Run `d1_nf4_study.py replay --input <correctness-dir> --output <fresh-dir>`
for each stored correctness fixture. Run `d1_qa_replay.py --input <qa-dir>
--data <fixed-data-dir> --block-ids 0 1 2 3 4 --output <new-json>` for QA.
Use `d1_projection_replay.py` for stored projection witnesses. The replay
source must be committed and its exact hash recorded by the caller; GPU source
identity stays in each immutable original raw record.
