# Stack-v3 evidence — A100, 2026-10-14

Read [the results and their limits](../../d2-stack-comparison-20261014.md).
This package contains all 48 full-step timing/loss records, complete GPU
telemetry, eight gate records, native operator reference/candidate tensors,
exact frozen inputs/source, independent audit outputs and the preserved VOID
pilots. It excludes the large full-model adapter snapshots except for selected
raw witnesses. Public timing replay is separate from full tensor replay.

Reassemble and verify from the repository root with standard Python:

```python
import hashlib
import json
import os
import zipfile
from pathlib import Path

root = Path("benchmarks/evidence/d2-stack-v3-20261014")
index = json.loads((root / "package-manifest.json").read_text())
archive = root / "stack-v3.zip"
with archive.open("wb") as output:
    for item in index["parts"]:
        data = (root / item["file"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        output.write(data)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == index["archive_sha256"]
target = Path(".benchmarks/d2-stack-v3")
with zipfile.ZipFile(archive) as source:
    assert source.testzip() is None
    for name in source.namelist():
        assert os.path.commonpath([os.path.realpath(target),
                                   os.path.realpath(target / name)]) == os.path.realpath(target)
    source.extractall(target)
payloads = json.loads((target / "payload-manifest.json").read_text())
for name, item in payloads.items():
    data = (target / name).read_bytes()
    assert len(data) == item["bytes"]
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
```

With the project dependencies installed, independently recompute all ABBA
ratios, loss repeatability, native execution records and GPU lifecycle checks:

```bash
python benchmarks/harness/d2_audit_stack_timings.py \
  .benchmarks/d2-stack-v3/a100/results/comparison
```

Expected: `timing_records_valid=True`, 48 arms. This intentionally does **not**
claim to replay omitted full-model tensors. The eight negative controls in
`verification/tamper-checks.json` operate on in-memory copies; accepted bytes
remain untouched. The post-run timing verifier is archived under
`post-run-tools/`; it was added after the GPU source freeze.

Actual native FP32 operator pairs are complete and independently replayable:

```python
from pathlib import Path
from benchmarks.harness.d2_operator_probes import verify

root = Path(".benchmarks/d2-stack-v3/a100/results/operator-proofs")
for backend in ("liger", "unsloth"):
    verify(root / backend, backend)
```

Expected: nine cases / 37 quantities pass relative L2 <=0.001. This is limited
operator validation, not FP16/BF16 full-model equivalence or an upstream test suite.
`gate-witnesses.safetensors` preserves first/final weights and gradients for
`layers.0.mlp.down_proj.lora_A` in all eight model gates. Full failure counts
and positive tensor replay require the complete raw snapshots, not these witnesses.

The complete A100 raw archive remains in the local task workspace:
`benchmarks/results/d2-fast-lora/stack-v2-a100-20261014/download/evidence.zip`.
It is 1130005846 bytes, SHA-256 `389b352bcd67d90b62635d512405046d73a1ae3b3a2822785969948eec8b5d23`. This is a local path, not a public
download URL. Its manifest, full archive identity and local independent replay
are retained in this package. A reviewer needing every tensor must obtain that
exact full archive separately. After contained extraction:

```bash
python benchmarks/harness/d2_materialize_timing.py /path/to/raw-extraction
python benchmarks/harness/d2_audit_stack_comparison.py \
  /path/to/raw-extraction/results/comparison \
  --operator-proofs /path/to/raw-extraction/results/operator-proofs \
  --output full-replay.json
```

Expected: `complete=True` for the declared stack-v3 experiment. External gate
`passed=False` results remain FAILs; finite cross-stack drift is disclosed,
not relabelled as equivalence. Each of the 48 timed arms passes replay against
its own backend's gate. The original strict clock experiments remain VOID.

`source.zip` SHA-256 is
`52f6b6ef22a8a5359ba57ed311168321e174059c9fcf3e3a81a42478edf0f396`.
It includes the pinned Linux requirements lock and stack-v3 declaration.
`fixtures.zip` contains the exact FP32 initial adapters/tokens, configs and
base digests for both dtypes. It omits the reconstructible 2.2 GB base checkpoint;
download the exact model revision named in the report and verify its retained
`checkpoint-sha256.txt` before populating both `fixtures/*/model/model.safetensors`.
Extract nested archives with the same containment checks used above.
The report supplies the fresh-GPU commands; regenerate native operator proofs
on that host. The recorded budget authorization does not authorize a reader's
new rental.

Source packaging caveat: the original freezer applied CRLF normalization to
one unused UI PNG as well as text. The executed source ZIP remains unchanged.
All 1,510 text or unchanged-binary payloads match the final checkout; the original
PNG is supplied separately under `source-binary-originals/` with both hashes in
`verification/source-match.json`. The benchmark never loads that UI asset, and
all benchmark/runtime Python files match exactly after text normalization.
Future archiving preserves binary bytes.
