# Public trainer attempt 02: incomplete at the session time cap

This record covers frozen Git source `76d2c4281adae978af300ee039800b8f47735d55` on the RTX 4090 host. It does **not** establish a performance comparison or D3 acceptance.

The session ran from 2026-10-29 12:01 UTC to 15:01 UTC under the committed 10,800-second session cap. It started 40 cells: all 36 correctness cells and the first three timing cells returned `ok`. The next planned cell, `timing-dense-ordinary-01-4-triton`, had only 118.18 seconds left of the session budget against its 900-second child allowance and timed out. The series stopped at the cap. No complete ABCCBA block, throughput result, or accepted timing comparison exists. This was a budget exhaustion, not an observed numerical mismatch, OOM, or F1 failure.

The raw session directory remains on the server at:

`/root/d3-public-trainer-preparation-01/prepared-02/formal-76d2c428-01`

It contains 2,606 files (4,833,104,148 bytes). The detached archive wrapper was itself capped at roughly three hours and five minutes; it was terminated while creating `/root/d3-public-trainer-results-76d2c428.zip` (3,023,399,538 bytes). That file fails ZIP validation and is explicitly partial. Do not use it as a complete archive. A complete recovery archive must be written to a new unique path, with the raw session kept untouched.

## Prospective correction

The rule declared 234–414 cells but allowed only three hours. At the observed 40-started-cell rate, the same plan projects to roughly 17.6–31.1 hours. Revision 3 sets a 36-hour session cap and requires the detached wrapper to permit 38 hours so analysis and archive finalization can finish after the measured session. It keeps the 234–414 cell budget, 900-second per-cell cap, one session per source commit, zero replacements, numerical recipe, exactness rules, and F1 requirements unchanged.

The 39 completed cells are retained as historical evidence but are not reused as cells in a new formal session. Revision 3 must be committed, pass required CI, and be exported from a clean canonical Git checkout before one fresh run. A fresh host/data/software/boot readiness check is still required immediately before launch.

## Checks at the frozen head

The exact `76d2c42` head passed required Ubuntu/Python 3.12 tests, lint, gate, and merge freshness. Optional platform/smoke jobs were skipped. These CI checks do not turn the incomplete hardware session into an accepted result.
