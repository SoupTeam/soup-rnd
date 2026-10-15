# D1 amendment C, 2026-10-15: separately pinned compatibility check

Recorded before the compatibility arms. The primary cohorts retain
bitsandbytes 0.50.1 and their original source, package and rule attribution.
Eight existing #968 GPU regressions skip because they require 0.50.2.

After cohort B exits, install bitsandbytes 0.50.2 without dependencies into a
separate directory. Prepend that directory only to the compatibility subprocess
environment; do not replace the primary environment or pool the versions.
Retain pip's installation report, distribution files/hashes and imported package
paths/versions. Installation or import failure is a recorded unsupported attempt.

Run the existing #842/#968 GPU regressions with skip reasons and the original
large-alias/streaming correctness fixtures in CUDA FP16 and BF16. Preserve any
failures without changing pinned constants, equality requirements or numerical
tolerances. No performance measurement, QA retraining or timing comparison
between versions is authorized by this amendment. RTX 3090 results do not verify
the separately documented sm_120 hardware behavior.

Respect the same rental deadline and evidence/cleanup reserve. This is a
compatibility subset, not final D1 acceptance or an independent review.
