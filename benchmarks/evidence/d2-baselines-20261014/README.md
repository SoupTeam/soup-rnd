# D2 baseline evidence — 2026-10-14

Read [the result and its limits](../../d2-training-baselines-20261014.md) first.
This compact package proves failed full-model numerical comparisons; it is not
a complete public replay of every omitted tensor or an accepted timing matrix.
Gate latency fields are diagnostic only. No full-step speedup is established.

The ZIP is split into six parts below 5 MB. `package-manifest.json` identifies
the exact archive and each part. It contains two compact experiments, reference
control metadata, frozen source archives, a SHA-verified reconstructed reference
worker, CPU audit summaries and complete-raw-archive identity records.
Neither provider credentials nor aggregate team balances are included.

The exact task instance was destroyed and API absence verified after 1 h 31 min.
Observed provider charges: **$0.449**, including compute/storage/traffic, below
the authorized $1. This may settle later. See [the sanitized receipt](rental-summary.json).

Reassemble and verify from the repository root on Windows or Linux:

```python
import hashlib
import json
import zipfile
from pathlib import Path

root = Path("benchmarks/evidence/d2-baselines-20261014")
manifest = json.loads((root / "package-manifest.json").read_text())
archive = root / "baselines.zip"
with archive.open("wb") as output:
    for part in manifest["parts"]:
        data = (root / part["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        output.write(data)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
with zipfile.ZipFile(archive) as source:
    assert source.testzip() is None
    source.extractall(root / "extracted")
payloads = json.loads((root / "extracted/payload-manifest.json").read_text())
for name, entry in payloads.items():
    data = (root / "extracted" / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
for mode in ("default", "deterministic"):
    folder = root / "extracted" / mode
    with zipfile.ZipFile(folder / "compact.zip") as source:
        assert source.testzip() is None
        source.extractall(folder / "compact")
```

With the project's torch/safetensors/rich dependencies installed:

```bash
python benchmarks/harness/d2_verify_baseline_witnesses.py \
  benchmarks/evidence/d2-baselines-20261014/extracted/deterministic/compact
python benchmarks/harness/d2_verify_baseline_witnesses.py \
  benchmarks/evidence/d2-baselines-20261014/extracted/default/compact
```

Expected final deterministic replay: 81 payload hashes, eight complete 50-step
loss curves, four failed curves, eight raw failing weight/gradient pairs.
Expected default replay: 85 hashes, eight curves, six failed curves, 12 pairs.
The default-mode PEFT reference also failed repeatability, so that regime cannot
support a library-specific correctness or performance conclusion.

The package also contains the selected initial LoRA-A tensors and a CPU-only
first-update diagnostic. After extracting the deterministic compact ZIP:

```bash
python benchmarks/evidence/d2-baselines-20261014/extracted/diagnostics/replay_first_update.py \
  benchmarks/evidence/d2-baselines-20261014/extracted
```

It replays four selected AdamW updates from the raw unscaled gradients. The
captured maximum absolute difference from the saved GPU weights is <=1.87e-9.
This diagnostic is restricted to the selected first-step down-projection
LoRA-A tensors; it is not an upstream-kernel root-cause verdict.

The compact verifier recomputes loss matches and actual tensor relative L2
errors. The full matrix's positive tensor matches and the reported total failure
counts require the complete raw snapshots, not just these witness pairs. The
published full audit JSON is a record of a local independent replay, not a
substitute for those omitted bytes. The reference repeat control metadata has
the same limitation.

Complete raw archives remain in the local task workspace:

- `benchmarks/results/d2-fast-lora/baseline-prep-20261014/download/evidence.zip`
  (2,064,328,461 bytes), SHA-256
  `e464582c2930a08d734b2c51078999d99f5bb49b4dcba530835ab858e9f1a9c5`.
- `benchmarks/results/d2-fast-lora/baseline-prep-20261014/download/supplement.zip`
  (1,414,557,133 bytes), SHA-256
  `a99f43a24b5314458c0e834697943a52f341e840706b0a6b5a22078dadeaf163`.

These paths are local evidence locations, not GitHub download links. A reviewer
who needs every tensor must obtain those exact archives separately; they are
intentionally not committed as 3.5 GB of Git history. After extracting one:

```bash
python benchmarks/harness/d2_materialize_timing.py /path/to/raw-extraction
python benchmarks/harness/d2_audit_training.py \
  /path/to/raw-extraction/results/comparison-deterministic --output full-audit.json
```

For the first archive use `results/comparison-v2` with the archived corresponding
auditor source; it predates recorded deterministic settings. The current auditor
also rejects missing/mixed execution settings, so old evidence cannot silently
be relabelled as the final deterministic regime. A nonzero full-audit exit is
expected: numerical gates failed and all 72 performance arms are absent.

Use `sources/source-v5.zip` for the executed final worker and pinned dependency
lock. Current protocol/report files contain post-run conclusions and therefore
have different hashes from their frozen pre-run versions. The current worker
also replaces the removed profiler argument `use_cuda=False` with
`use_device=None` for PyTorch 2.14 compatibility. This changes the CPU preflight
trace API only; the frozen GPU worker is preserved in the archive, and the
compatibility edit has not been rerun on a paid GPU.
The archived post-run source check describes the original evidence publication,
before that compatibility edit. [The CI fix record](ci-profiler-fix.json) and
[its JUnit](ci-profiler-fix-tests.xml) record 59 actual torch 2.14.1+cpu passes.
`sources/source-v4-worker-reconstructed.py` is explicitly reconstructed
and SHA-verified against the captured ready record; the original v4 outer ZIP is
not preserved. `source-v4-reconstruction.json` describes that boundary.
