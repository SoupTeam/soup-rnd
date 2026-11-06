# Two-stage pipeline GPU evidence

Two fresh real Qwen2.5-0.5B + LoRA runs on two RTX 3090 GPUs, four optimizer steps
per run/stage, native GPipe, global8. Both final adapter reloads run in separate
processes and have zero logit error. See the [result report](../../../h3-pipeline-result-20261029.md).

- `repeat-1/`, `repeat-2/`: raw rank records, actual stage ownership/counts,
  first/control/final tensors, merged96-key adapters, eval inputs/references and
  reload metadata. All numeric references are public, not just scalar summaries.
- `metadata/`: hashed prepared inputs/config/base inventory, exact source commit,
  full package freeze and GPU/topology inventory.
- `campaign.log`, `bootstrap.log`: accepted commands and raw output.
- `validation.json`: offline tensor/hash/source/data/process revalidation.
- `archive-ledger.json`: exact full local archive paths, byte counts, SHA256 and
  member hashes for all four accepted archives and failed-attempt archives.
- `failed-system-torchrun/`, `failed-system-torchrun-support/`: failed pre-training
  entry-point attempt. It is excluded from successful GPU execution.
- `support/`: scoped rental/cleanup and charge proof, interpreter-fix test log,
  transfer/NCCL preflights and captured operations/verification sources.

Measured clean source is `55f675e17a20812d548054956d5af7edc7815d09`.
The pre-training `e74469aa` failure is kept separately. The fixed launcher keeps
workers inside the active Python environment; all numerical functions/data and
tolerances are unchanged. The earlier local CPU fixture lives in `../local/`
and is not GPU or quality evidence.

`artifact-manifest.json` hashes this public capture byte-for-byte. Both owned
rentals are deleted and all archives were verified before deletion. Frozen base
weights are not bundled; their immutable HF revision and exact file hashes are
retained. A GPU/CPU reader may download that same base independently. This is a
correctness smoke, not a throughput comparison or optimizer-resume result.

Verify the public byte ledger from the repository root (CPU only):

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
from rich.console import Console

root = Path("benchmarks/evidence/h3-pipeline-20261029/hardware")
ledger = json.loads((root / "artifact-manifest.json").read_text())
for name, expected in ledger["files"].items():
    path = root / name
    assert path.stat().st_size == expected["bytes"], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
Console().print(f"Verified {len(ledger['files'])} captured files")
PY
```

For numeric replay, each rank's `first-step.pt` contains actual gradients and
adapter values; `control.pt` contains the ordinary control's first gradients,
first adapter, initial adapter and final adapter. Load with
`torch.load(..., weights_only=True, map_location="cpu")`; compare matching keys
at atol=1e-5, rtol=0.02. Final `adapter.pt` shards must be disjoint, union to96
keys and match `adapter/adapter_model.safetensors` with atol=rtol=0. Offline
tensor comparison is distinct from rerunning the already-recorded GPU reload.
