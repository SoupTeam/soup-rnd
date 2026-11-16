# Combined 2026-10-22 stage verification

Both independent USD 5 studies stopped before model work because completed,
SHA-verified binary artifact transfers were too slow for their prospective
retention policies. This is an infrastructure blocker, not a numerical or
performance result. Numerical gates remain unchanged. No pretrained model,
Inductor compilation or training-memory measurement ran in either study.

| Study | Physical preflight | Transfer | Conservative new expense | Deleted instance |
| --- | --- | --- | ---: | --- |
| [Residency / compilation](../../stream8-compile-20261022.md) | RTX 3050, 8192 MiB | 0.378901 MiB/s | USD 0.144429 / 5 | 57114362 |
| [Four cache arms](../../stream8-cache-20261022/README.md) | RTX 3060 Ti, 8192 MiB | 0.433158 MiB/s | USD 0.150000 / 5 | 57114865 |

Total conservative new-stage expense: **USD 0.294429 / 10**. No historical
residual allowance is added. Provider-observed snapshots total USD 0.012 and may
settle later. Each report documents its own reserve and policy; measurements are
not pooled across GPUs. The compile admission rate stayed 2 MiB/s. The cache chat
prospectively used 1 MiB/s for a 12 GiB upper allowance inside four reserved hours;
its measured rate failed that rule too.

Integration used a separate clean worktree based on fresh PR #5 HEAD
`a3b5c71310703ff3b05195c5add9422d51f5414f`. Source commits, applied sequentially:

- `b686fac9a6705bca5d67a06c636c70e91ddd9848`
- `325ae1a513abea3cf293a75d50ae871bd9ca1bb5` (see `stage-summary.json` for the exact SHA)
- `edf7708542a9b075635674b4f12d906a015c607c`

The cache preparation rebuilt upstream
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b` plus the explicit research overlay and
verified all **602 production files** against Git blobs. The combined CPU run
explicitly imported that reconstructed runtime, rather than a globally installed
editable checkout: **30 passed, 2 CUDA skips**. The retained JUnit XML is
`cpu-validation.xml`. This remains a small CPU fixture, not GPU Inductor or full
Qwen3-8B qualification. Full repository Ruff and `git diff --check` passed.

The integration verified 14 compile-evidence hashes and 21 cache Git-blob hashes.
`combined-absence.json` is another independent fresh paginated provider-list check
after both chats finished: neither owned ID/label remained. Other instances were
untouched. The integration made no new rentals and did not wait for full CI.

No 8B/14B/24B/32B capacity, training speedup or training VRAM claim is established.
The cache candidate and its mutation tests are research preparation. Production
autoplanning and 10Cache work remain deferred pending actual measurements.
