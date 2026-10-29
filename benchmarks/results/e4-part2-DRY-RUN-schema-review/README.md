# E4 part 2: DRY RUN records (schema review only)

**These are not calibration results.** They were produced by
`python benchmarks/harness/e4_part2_calibration.py calibrate --dry-run` on a Mac, CPU only, with
no GPU and no paid compute. Their purpose is to let the record schema be reviewed before any
paid calibration run. Nothing here may be copied into a rule, a report or a decision.

## What is fake

- **The box stamp.** `box_stamp_before` / `box_stamp_after` come from `e4_part2_dryrun.fake_stamp`:
  GPU name `NVIDIA A100-SXM4-40GB [dry-run fake]`, compute capability 8.0, memory.total 40960 MiB,
  driver `0.0 [dry-run fake]`, torch CUDA `dry-run (cpu)`, UUID `GPU-DRYRUN-...`, no GPU peers,
  made-up RAM and clock readings. The same keys as a real stamp (a test checks this).
- **F memory numbers.** `max_memory_reserved_bytes` (20,000,000,000) and
  `max_memory_allocated_bytes` are set by `--fake-reserved-bytes`; the result lists them in
  `dry_run_fake_fields`. `device_memory` is empty (no NVML sampler).
- **The model and data.** A tiny random OLMoE (2 layers, 4 experts) and synthetic rows;
  `dataset_revision` is `dry-run-no-dataset`. Every loss, tau, tol_W, noise floor and
  repeatability value is meaningless. N's scores come from a constant fake generator.
- **N scores** use the real scorer on a fake generator, so the scorer output fields are real
  but the numbers are not.

Every record has a `dry_run` block (card, injection, and which checks a dry run skips: the dirty
tree and library-version voids). The dry run does not void on a dirty tree: the trees these were
made on had untracked files, listed in each record's `provenance.dirty_paths`.

## What is real

The driver, the journal (`attempts.jsonl`), `execute_attempt`, the record layout, `decide`,
`report`, the stage code (`run_f`, `run_t`, `run_w`, the N scorer), and the identity block.

## Layout

| Folder | Run |
|---|---|
| `clean/` | `calibrate --dry-run --instance-id dry-clean`: F x3, decide, T, W, N x5, P, report. 11 records |
| `inject/` | the same with `--inject F_A_on=kill,T=void,W=oom`: 14 records |

Each folder: `attempts.jsonl` (start and end of every attempt), `runs/<attempt_id>.json` (one
record per attempt), `decision.json`, `report.json`, `preflight-*.json` (the card check before
the first attempt), `driver.log`, `SHA256SUMS`.

## What to look at in a record

- `identity`: GPU name, compute capability, memory.total, driver, CUDA (torch and driver), torch,
  bitsandbytes, transformers, commit hashes, dataset revision, instance id.
- `record_source`: `run` (the process wrote it) or `driver` (the child was killed before it could;
  the driver wrote it, with `killed.signal`).
- `attempt_number`, `outcome` (`ok`, `void`, `failed`), `void_reasons`, `failure_kind`
  (`oom`, `killed`, `crash`).
- In `inject/`: `F_A_on` attempt 1 is the SIGKILL (driver record), `T` attempt 1 is an injected
  void, `W` attempt 1 is an OutOfMemoryError crash. Each is rerun once, as the rule allows.

## Edits after generation

Adapter weights (`adapters/`) are left out. Absolute paths were replaced by `<out>` and `<repo>`
in the JSON and logs, and the logs have the progress bars removed. `SHA256SUMS` was recomputed
after that edit, over the files kept.
