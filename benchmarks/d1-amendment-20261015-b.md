# D1 amendment B, 2026-10-15: bounded repeated timing cohort

Recorded before these arms. Preserve the original rule, amendment A and both
earlier cohorts. No earlier measurement is reclassified.

The 20-step projection screen showed more than 5% SM-clock spread at M=2048.
Use 5,000 untimed warm-up steps in each new projection arm, followed by the
original 100 synchronized timed steps. This is an attempt to reach a stable
thermal/power state, not a waiver of the clock guard. Keep the GPU's default
power/clock settings and retain all clock samples. A remaining spread above 5%
still gives NO_VERDICT. No selective sample or block removal is permitted.

The bounded repeated cohort consists of five fresh-process ABBA blocks each:

- BF16 nested NF4, N=K=4096, M=8, dense versus native;
- BF16 nested NF4, N=K=4096, M=2048, native versus forced Function;
- TinyLlama RAM-tier baseline versus cached-zero nodequant ablation, sequence
  512, 20 warm-up and 100 timed steps, with unchanged garbage-math labeling.

Five blocks are a starting point, not guaranteed acceptance. Apply the original
F1 sample-sizing rule and all identity, numerical, peer, disk, clock and power
guards. If the authorized pilot time does not permit a required top-up, report
NO_VERDICT and the required count. These cases are a subset of the preregistered
grid; they do not complete D1. QA remains the original one-block screen with
20 warm-up plus 256 timed updates and one held-out evaluation per arm.

The cloud controller must start only after cohort A exits, run arms sequentially
and stop at least ten minutes before the rental's hard deadline so evidence can
be downloaded and the instance released. No additional rental is authorized.
