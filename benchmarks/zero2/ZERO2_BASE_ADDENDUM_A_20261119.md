# ZERO-2 stage 2, addendum A (2026-11-19, before the second attempt)

Protocol: `ZERO2_BASE_PROTOCOL_20261116.md`. Nothing in "Fixed setup" or "What is decided" changes.

## First attempt (2026-11-17/18, Vast 61077212, RTX 4090, $4.02): no result

No arm started. The model download never finished: the HF Hub download (xet, no token) stalled
twice, at 4.5 GB and 10.8 GB; then my wrapper scripts failed (a placeholder `HF_TOKEN` with
Cyrillic text, `UnicodeEncodeError` on every request; then a 240 s per-try timeout shorter than a
whole shard, which restarted partial files instead of resuming them). The one runner start that got
past the hash check wrote `download.json status=ERROR` and no arm. The instance was destroyed.
Nothing was measured, so the second attempt is a first measurement, not a rerun.

## What changes for the second attempt

Only how the model gets onto the disk before the runner starts (`start_stage2.sh`):

1. Every file of `Qwen/Qwen3.5-9B` @ `c2022362…` matching the runner's patterns is fetched with
   `curl -C -` (resume), 4 in parallel; a connection below 100 KB/s for 60 s is dropped and
   resumed from the same byte, at most 60 times per file. Optional HF token, read from a file on
   the instance, never in a command line or the repo.
2. Then the runner's own `snapshot_download` call (same arguments) checks every LFS file
   against its sha256 before anything trains; on a mismatch the runner is not started.
3. The runner, its sha256 (`d1511826…`), Soup commit, arms, order and decisions are unchanged.
   `download.json`'s `wall_s` then measures the check, not the download; the download time is
   in `run.log` (`download_s=`).

Machine: as fixed (one 24 GB RTX 3090/4090, >= 64 GB RAM, >= 120 GB disk), chosen this time
also for download bandwidth >= 500 Mbps.
