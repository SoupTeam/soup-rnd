# A3 stage 2 GPU session: one RTX 4090 VM, 2026-10-27 (rule v1). Stopped at calibration

**`NO VERDICT: criterion 1, 2, 3 has no verdict`** by rule v1 (`verdict.json`, unchanged).
The session stopped during the calibration of the tight limit, before any scored arm: the
check child at the computed limit was OOM-killed by its cgroup inside its first training
step. Re-running `a3_spill_rule.py` on the session's raw records reproduces `verdict.json`
byte for byte; on this published copy it does too.

| # | Criterion | Result | Why |
|---|---|---|---|
| 1 | spill == no spill | NO VERDICT | `l2l_nospill_1`, `l2l_nospill_2`, `l2l_spill` never ran |
| 1b | L2L vs GA (reported) | NO VERDICT | `ga_1`, `ga_2`, `ga_floor`, `l2l_spill` never ran |
| 2 | RAM | NO VERDICT | the three scored arms never ran |
| 3 | cleanup | NO VERDICT | `l2l_spill`, `sigint`, `kill9` never ran |
| 4 | cost | NO VERDICT | refused: cost blocks run only after criterion 1 passes |

No overhead was measured. Nothing here confirms the spill on a GPU, and nothing here shows
that it fails.

- Measured at `bda4a00a` (criteria fixed at `2a05e15f`, admission at `794c6827`, both before
  the rental). Every one of the 659 source files hashed in `protocol.json` equals that
  commit. torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2.
- Box: vast.ai instance on host 655981 / machine 154269 (Quebec), RTX 4090 24 GB, PCIe gen 4
  x16 in every child record, gpu_frac 0.125 (vast's ratio of offer GPUs to system GPUs; not
  isolation, not a separate disk), KVM VM, driver 580.95.05, 49.35 GiB RAM, one virtio disk
  `vda` (ext4) for the spill and the weights. No PCIe 4.0 RTX 3090 / 3090 Ti VM passed the
  filters at rental time, so the 4090 VM was taken as §5 allows. Fixture SHA-256 equal to
  A4 v3's.
- Admission (`admission/`, `step0/`, before the session): ADMITTED. A.value 1.5708 s (no-spill
  A child, median of 5 timed steps), cycle 268,435,456 B, D 0.171 GB/s, N 180 cycles of
  0.785 s; 180 of 180 ran, largest lateness 0.312 s (limit 5 s), 0 of 11,520 chunks read back
  wrong, no file left.
- Calibration (`calibration.json`, generous limit 24 GiB): exit 0. Working set 5,473,333,248
  B at the product's plan point; tight limit = working set + (320 MiB + 8 MiB) / 0.7 =
  5,965,348,864 B, for 16 of 32 layers spilled. OOM margin, fixed before the run: build
  +1,001,061,608 B (no stop), training -15,995,576 B (below 256 MiB: a warning, description
  only, as committed in §3.4 / §4).
- Check (`markers/calibration_check_1.json`, `session.json`): at 5,965,348,864 B the product
  planned 18 of 32 layers (in the [12, 20] band; RAM budget 0.30 GB = 70% of 0.43 GB
  available), then the child was killed by SIGKILL inside its first `training_step` (marker
  `calls` 1, `steps_done` 0, `in_step` true), 19.1 s after start. The kernel journal (kept
  outside the repository with the logs; hash in `provenance.json`) records a
  `CONSTRAINT_MEMCG` OOM kill of that PID in that child's scope at 07:40:11Z: usage = limit
  5,825,536 kB, swap 0, memcg `anon` 1,320,644,608 B, `shmem` 4,615,835,648 B, `kernel`
  28,860,416 B. The stand stopped as §3.4 prescribes for a failed check child (exit 3,
  `state: stopped`), before any correctness, cleanup or cost child. The spill file of the
  killed child (301,989,888 B) stayed, as documented for SIGKILL; no next start in that
  directory ran, so the stale sweep was not exercised.
- Why there is no criterion 2 FAIL: §4 calls an OOM in training at a tight limit a criterion
  2 FAIL, but criterion 2 reads only `l2l_spill`, `l2l_nospill_1` and `l2l_nospill_2`, and
  §3.4 stops the session at a failed check child before those run. The rule was not changed
  after the result.
- What the calibration series shows (description, no verdict). In the generous child's first
  step, sampled every 50 ms, U (memory reclaim cannot free) less torch's pinned host
  allocator was 669 MB at the step's entry and rose to 1,168 MB within 0.56 s, before the
  product read its plan-time numbers (`memory.current` then equals the plan's `used`,
  5,473,341,440 B). So about 499 MB of the first step's growth is already inside the
  working set the tight limit was built on. The store's pinned blocks followed (+537 MB, the
  planned all-resident store); after the plan point the non-pinned part grew another 183 MB
  by step end (`train_transient_bytes` 681,512,864 B counts both parts from the entry). At
  the tight limit the product's 0.3 headroom was about 130 MB (0.43 GB available x 0.3).
  At the kill, memcg `shmem` was exactly 234,881,024 B below the generous child's step end:
  the difference between the two plans' host bytes (536,870,912 all resident against
  301,989,888 with 18 layers spilled), consistent with the tight child's store and rings
  being in place; `anon` was 36 MB below. Which allocations make up the non-pinned growth is
  not established; whether it is the same on other boxes is not known.
- Contents: `protocol.json`, `calibration.json`, `session.json`, `verdict.json`, both child
  markers, the admission's inputs (`step0/`) and records (`admission/`), without run logs.
  Box paths are replaced by `<box>` / `<work>`; `provenance.json` lists each file's raw and
  published SHA-256 and the archive's hash.
- Not run, so not shown: equality with spill, RAM within the plan, cleanup after Ctrl-C and
  kill -9 with the next-start sweep, and the cost, on any GPU.
