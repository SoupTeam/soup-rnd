# Ubuntu CI capture regression follow-up (2026-10-29)

The [Ubuntu Python 3.12 job at bbf6efb2](https://github.com/SoupTeam/soup-rnd/actions/runs/40836500312/job/123322855180)
installed pytest 9.2.0 and aborted in the pre-existing
`test_post_open_toctou_swap_detected_on_windows` test. Its process-wide
`os.fstat` replacement returned `MockSwappedFstat`, which lacks `st_size`.
Pytest 9.2's descriptor capture now reads that field. The test file was unchanged
from the PR base; this was not a failed GPU training or adapter reload.

The three Windows simulations now replace only `soup_cli.utils.paths.os`
with a shallow namespace of the real OS functions. The Windows checks still
execute against real temporary files and retain their original rejection
assertions. The swap test also checks a real capture descriptor and explicitly
drains capture while the fake fstat is active, guarding against the original
test-runner interference. No production or numerical training code changes.

## Local validation

| Run | Result | Log |
|---|---|---|
| Linux, Python 3.12.15, pytest 9.2.0, before isolation with capture regression | Reproduces the same `MockSwappedFstat.st_size` abort | [negative control](linux-pytest92-before.log) |
| Linux container, Python 3.12.15, pytest 9.2.0, after isolation | **38 passed, 3 skipped** | [Linux](linux-after.log) |
| Windows, Python 3.10, existing pytest 9.1.1, after isolation | **40 passed, 1 skipped** | [Windows](windows-after.log) |
| Windows, Python 3.10, pytest 9.2.0 installed in a separate task directory | Pytest fails before collection with `NodeId._str_cache`; not a passing validation | [bootstrap failure](windows-pytest92-bootstrap.log) |

The last attempt did not execute the changed tests and is retained separately;
no dependency pin or unrelated pytest compatibility repair is included.
The original file alone passed on pytest 9.1.1 before this change; that older
capture implementation does not inspect `st_size`.

Both accepted runs use this subset:

```bash
python -m pytest \
  tests/test_issue820_windows_symlink_open_nofollow.py \
  tests/test_issue1158_run_log_directory_symlink_and_hardlink.py \
  tests/test_issue1138_run_log_symlink.py \
  tests/test_symlink_skips_are_capability_based.py \
  --no-cov -q --tb=short
```

Linux additionally uses `-p no:cacheprovider` with a read-only source mount.
It uses the existing local `soup-liger-interpreter:20261019` container image,
with pytest 9.2.0 supplied through a separate read-only `PYTHONPATH` directory.
This is Linux reproduction, not a rerun of the hosted Ubuntu image or full suite.
Full repository Ruff (`src/soup_cli/ scripts/ tests/ benchmarks/`) and
`git diff --check` also pass. Hosted CI was inspected for diagnosis, not polled
or awaited. No cloud rental, existing server change, or new GPU evidence.

## Synchronization with main

The hosted [Ubuntu/Python 3.12 check at 5fad22c2](https://github.com/SoupTeam/soup-rnd/actions/runs/40871901410/job/123440841687)
and lint subsequently passed. A later `merge-freshness` failure instead reported
conflicts with main. Merge main `38ccc7876622b0911fb514ba41269eb2cddd4f0b`
without rewriting the branch history. Retain both H3 and D1 benchmark index
entries, and use main's equivalent isolated OS fixture with the explicit
capture regression retained. No production or H3 numerical harness changes
are introduced by this synchronization.

The same four-file subset passes again on the merged tree: **38 passed,
3 skipped** on [Linux/Python 3.12.15/pytest 9.2.0](merge-linux.log), and
**40 passed, 1 skipped** on [Windows/Python 3.10/pytest 9.1.1](merge-windows.log).

Full repository Ruff and checks of the manually resolved files pass. Main's
historical raw logs include whitespace diagnostics; their captured bytes are
preserved instead of reformatted. Hosted CI for the merge is not awaited.
