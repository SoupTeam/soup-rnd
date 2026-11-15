# GPU runner preparation: no rental or speed result

The four-arm runner is committed at [f5707314](https://github.com/SoupTeam/soup-rnd/blob/f57073146e7949143b64b381201a633a8d3b9908/benchmarks/harness/two_read_cache.py)
on `koko/two-read-cache-20261025`. The integration source and prior CPU report remain
separate from PR #5's older production diff. No runtime/config/default was changed
in this runner-only follow-up.

The [complete protocol and commands](https://github.com/SoupTeam/soup-rnd/blob/f57073146e7949143b64b381201a633a8d3b9908/benchmarks/harness/two_read_cache.md)
cover ordinary streaming, two-read, expert RAM/GPU caches and the combined arm.
Real-model CUDA numerical captures must pass before any speed block is launched.
Initial adapters/inputs/top-k indices are exact, floating gates remain reference-relative
`rtol=.01, atol=1e-5`, finite and three-decimal loss, and actual AdamW updates are required.

Physical 8 GiB GPU identity, native BF16, at most 32 GiB real RAM and disabled swap
are preflight gates. All arms share one frozen local checkpoint/token fixture and
identical cache ceilings; changed source/inputs/store/environment are refused.
Four balanced timing rounds use fresh processes, separate setup and warmup from
steps, record RSS/VRAM/read counters, and compare the combined arm with two-read
separately. Fixed-token numerical parity is not downstream model quality.

Local verification: **172 passed, 1 CUDA skipped**
in a focused seven-file selection (91.44 s).
All new runner controls executed. The new capture pipeline passed real tiny-DeepSeek
FP32, BF16 and actual CPU NF4 four-arm trajectories bitwise, with both cache tiers
actually hit. Controller regressions reject mismatched inputs/adapters/gradients,
truncated trajectories, missing cache coverage, changed hashes and invalid timing.
Full-repository Ruff and applicable staged pre-commit passed. The full repository
test suite and hosted CI were not run. This adds runner controls to the older
598-pass integration record; it does not replace or reclassify that record.

The evidence collector initially lacked the isolated dependency directory in its
parent metadata path after the commands had passed. Its path was fixed and the
completed XML/logs were re-collected, without silently rerunning or replacing them.

[Portable patch, exact executed source, logs, XML and verification scripts](gpu-runner-evidence.zip)
are byte-checked against the internal `files.json`; [compact summary](gpu-runner-summary.json)
records the outer SHA-256. Package directories are excluded.

**GPU rental/spend: $0. No CUDA speed, memory, SSD throughput or real-checkpoint
quality result is claimed.** Existing GPU-rental prohibition remains in effect.
