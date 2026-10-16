# D1 K: complete matrix coverage after a terminal clock violation

Recorded 2026-10-15 before continuing measurements. Numeric source remains
`87b3bc1747ff78ff0fe16b8bdd2fa2eb0105d77c`; GPU and numerical settings are unchanged.
This amendment changes only the outer collection policy, not timing acceptance.

K stopped after 26 accepted comparisons. The next comparison,
FP16 M=8, N=14336, K=4096 dense/native, has five complete ABBA blocks but violates
the existing comparison-wide 5% clock bound. All five blocks remain rejected;
that cell is terminal **NO_VERDICT**, with no retry or reported timing ratio.
The original stopped controller, exit marker and archive are retained.

The previous outer controller aborted the entire matrix at the first such cell.
For a useful research report, collect the other independent cells under the
same frozen 98-case plan. An external, retained continuation controller imports
the original harness from the unchanged clean numeric checkout. It reuses only
accepted comparisons and correctness results, never relaunches a clock-invalid
cell, and retains terminal NO_VERDICT entries in a separate coverage ledger.
Valid cells still require the original minimum five ABBA blocks and F1 top-ups.
Identity changes, failed CUDA children and the measurement deadline still stop
the collection for review; the clock guard is neither loosened nor normalised.

Coverage and acceptance are separate: all 98 cells attempted does not mean all
98 accepted. `d1_full_report.py` remains unchanged and returns INCOMPLETE if any
required cell is invalid. A coverage figure, if generated, must show invalid cells
explicitly and cannot be called the original complete accepted result figure.
The final report must give both attempted and accepted counts and preserve the
reason for every unavailable cell. No global D1 PASS or independent acceptance
follows from this continuation.

The existing cumulative $10 ceiling, measurement deadline and absolute deletion
deadline remain unchanged. The local finish controller was stopped before rental
deletion, while both independent budget guards stayed active. It will restart
against the new coverage-controller exit marker and export all J/K attempts.
No new rental, dependency change, thermal change, clock lock or extra dataset
selection is introduced by this amendment.
