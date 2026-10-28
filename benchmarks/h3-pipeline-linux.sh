#!/usr/bin/env bash
# Run only on a newly approved two-GPU rental. This script never rents/deletes.
set -euo pipefail
project_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$project_root"
root=${1:?usage: bash benchmarks/h3-pipeline-linux.sh /absolute/new/output}
[[ "$root" = /* ]] || { echo 'output must be an absolute path' >&2; exit 2; }
[[ ! -e "$root" ]] || { echo 'refusing to overwrite an earlier run' >&2; exit 2; }
mkdir -p "$root"
exec > >(tee "$root/campaign.log") 2>&1
set -x
export OMP_NUM_THREADS=2
export PYTHONPATH="$project_root/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
git rev-parse HEAD > "$root/source-commit.txt"
test -z "$(git status --porcelain)"
python -m pip check
python - <<'PY'
from importlib.metadata import version

assert version("torch").split("+")[0] == "2.13.0"
assert version("transformers") == "5.17.0"
assert version("peft") == "0.20.0"
PY
python -m pip freeze > "$root/pip-freeze.txt"
nvidia-smi -q > "$root/nvidia-smi.txt"
nvidia-smi topo -m > "$root/topology.txt"
timeout 600 python benchmarks/harness/h3_pipeline_smoke.py prepare \
  --prepared "$root/prepared"
for repeat in 1 2; do
  output="$root/repeat-$repeat"
  timeout 600 torchrun --standalone --nproc-per-node=2 \
    benchmarks/harness/h3_pipeline_smoke.py train \
    --prepared "$root/prepared" --output "$output" --device cuda --steps 4
  timeout 600 python benchmarks/harness/h3_pipeline_smoke.py reload \
    --prepared "$root/prepared" --output "$output" --device cuda
  tar -C "$output" -czf "$root/repeat-$repeat.tar.gz.partial" .
  mv "$root/repeat-$repeat.tar.gz.partial" "$root/repeat-$repeat.tar.gz"
  sha256sum "$root/repeat-$repeat.tar.gz" > "$root/repeat-$repeat.sha256"
  # Collector downloads/verifies this archive before writing this receipt.
  receipt="$root/repeat-$repeat.received.json"
  deadline=$((SECONDS + 180))
  until [[ -f "$receipt" ]]; do
    (( SECONDS < deadline )) || { echo 'collector receipt timed out' >&2; exit 1; }
    sleep 2
  done
  python - "$root/repeat-$repeat.sha256" "$receipt" <<'PY'
import json
import sys
from pathlib import Path

expected = Path(sys.argv[1]).read_text().split()[0]
receipt = json.loads(Path(sys.argv[2]).read_text())
assert receipt["sha256"] == expected, "collector receipt does not match sealed archive"
PY
done
tar -C "$root" -czf "$root/metadata.tar.gz.partial" \
  prepared source-commit.txt pip-freeze.txt nvidia-smi.txt topology.txt
mv "$root/metadata.tar.gz.partial" "$root/metadata.tar.gz"
sha256sum "$root/metadata.tar.gz" > "$root/metadata.sha256"
touch "$root/campaign.completed"
set +x
