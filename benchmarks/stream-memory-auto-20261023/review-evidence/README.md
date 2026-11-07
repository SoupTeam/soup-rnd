# Native residency independent review evidence

The exact scripts, JSON and logs referenced below are stored under `review-evidence/`
inside the parent `evidence.zip`. Extract that archive into the parent report directory
before replaying these CPU checks.

CPU-only checks run on 2026-10-23. No GPU performance or memory-fit claim.

`check_pool_mutations.py.txt` reads the native runtime and executes an isolated mutated
copy of its pool class in memory. It first runs all three checks unmutated, then
requires the resident mapping, resident no-op, and failed-allocation cleanup
mutations to fail with their expected exception classes. Source bytes are checked
unchanged. JSON and log: `pool-mutations.json`, `pool-mutations.log`.

`check_recompute_compatibility.py.txt` runs from the exact upstream PR #2152 checkout,
commit `aa025afbb0adcd322c1fa72f31268b8e7af24bc2`. It preserves that pool's existing
constructor and always-present generation metadata. Only the native residency
methods and constructor residency attributes are patched in memory. Dynamic loads
are explicitly checked to increment generations exactly once. Dense/NF4 tiny real
LoRA models with nonzero B adapters compare ordinary recompute to resident weights
plus full/partial activation retention. All four combinations have bitwise-equal
losses and all nonzero adapter gradients. JSON and log:
`recompute-compatibility.json`, `recompute-compatibility.log`.

Both scripts were run with Python 3.10 and `PYTHONPATH=src;.` from their respective
checkout, with `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. Their logs are captured
from the actual script reruns; JSON includes executed runtime SHA-256 hashes.
The compatibility script contains the exact local native-runtime path. Change
that path to replay elsewhere, preserving and comparing the recorded source hash.
The `.py.txt` files preserve the exact executed script bytes as archived evidence;
Python can run these paths directly without renaming them.

Additional targeted checks in the native checkout:

- Initial native pool plus historical prototype: 27 passed, 5 CUDA skips.
- After final cleanup strengthening/new direct no-op check: 14 selected pool
  checks passed; the structural source-release consumer checks passed 2/2.
- Earlier adjacent full files produced 308 passes, 16 CUDA skips, one static
  consumer-name failure. The expected name was updated from `load_async` to the
  extracted `_copy_layer`; the AST scan's count and release requirements remain.
- Ruff passed for native runtime, new pool tests and the updated static scan.

These local checks do not replace physical-GPU numerical and full-step timing
qualification. The planner is a conservative capacity policy; maximum feasible
residency is not established to be the fastest selection.
