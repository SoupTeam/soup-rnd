# B6 evidence

The real [DeepSeek-V3 671B pilot](../../b6-giant-moe-20261028.md) completed two
backward/optimizer updates on one physical RTX 5060 8 GB and passed exact adapter
parameter/logits reload. [Raw journal](real-giant/events.jsonl),
[verdict](real-giant/verdict.json), [adapter/source bundle](real-giant/b6-adapter-and-source.tar.gz)
and [hash manifest](real-giant/artifact-manifest.json) are retained here.
All four rental instances were verified absent; the post-deletion ledger query
was $3.707 total. Larger host RAM means the 32 GB target remains unvalidated.
The two-step/one-heldout-text probe does not establish general quality.

Synthetic CUDA selections are separate: RTX 3070 and RTX 5060 each passed 9
selected cases covering frozen experts, NF4, autocast and adapter reload.
After the real trial exposed allocator rounding during reload, the corrected
path passed 69 selected CPU/CUDA cases on RTX 5060. The final local B1/B6
selection passed 132 tests with 11 skips; Ruff passed all four directories.

The complete CPU test collection attempt on Windows stopped in unchanged E4
modules importing Unix-only `resource`. Separate streaming setup regressions
passed 138 tests with 8 platform skips. Raw logs preserve their individual scope.
The first giant trial completed training but failed reload; its journal and
traceback are retained separately under real-giant/attempt-logs and
real-giant/failed-reload-trial-events.jsonl.
