# Prospective C4 concurrent request-size study

Protocol frozen after instrument controls and complete fixture SHA-256 verification,
before any timed arm. A=4 MiB and B=64 MiB, simultaneous independent reads on both
NVMe drives, ABBA, one hour each. Four attempts, no replacements. Start <=45 C,
stop >=70 C, up to 30 minutes cooldown per attempt. Sources and all rules are in
protocol.json; runner.zip contains the versioned tools and tests.

All 49 CPU tests passed locally and on Linux. Six instrument controls passed:
baseline/own/foreign = OK/OK/VOID for each request size, on both physical devices.
The own-read controls lasted 60 seconds each. These controls are not C4 results.
Full fixture hashes verified at freeze; existing 80 GiB files are reused read-only.

This tests concurrent repeatability and sequential request size. It does not repeat
single-drive conditions or manipulate cooling; thermal causality, random access,
and original CUDA Soup training remain outside this series. Two observations per
condition are descriptive evidence. All attempts and protocol deviations are retained.
The earlier stage-one results were already known and are not prospective replicates.

Commit and publish these exact bytes before launch. The Git receipt verifies object
linkage, not independently trusted publication time. No new timed results are included.

## Post-run lint maintenance

After the timed series completed, the repository copy of launch_wrapper.py had
its standard-library imports reordered for Ruff I001. No orchestration statements
or measurement parameters changed. The original frozen bytes remain available in
the freeze commit and in the published results archive
[raw-results.zip](../../results/c4-linux-pattern-abba-20261016/raw-results.zip),
entry evidence/launch_wrapper.py.

- Executed wrapper SHA256: 62a24b877b5114252c04c56a12244fa21068941d78397e709da08ee857458937.
- Lint-corrected repository wrapper SHA256: 9765bff43647835de408980afa11e19aae31014dfe88335f5068d85a8eb4727b.

SHA256SUMS.json describes the current repository files. This maintenance does
not alter the frozen protocol, Git receipt, runner bundle, or measured results.
