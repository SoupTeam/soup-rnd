#!/usr/bin/env bash
# F3 rental bootstrap, run on the vast.ai instance from the unpacked repo root.
#   bash benchmarks/harness/f3_bootstrap.sh
# Installs uv, builds the f3-unsloth venv from the lock file, tokenizes Alpaca-cleaned with the
# Qwen2.5-0.5B tokenizer, and checks the dataset sha256 against the laptop value.
set -euo pipefail

EXPECTED_SHA=f968d82c9244011d9c36d88cad0a00d8c8c1f8e69f35e53c900d89af390afed3
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$HOME/.venvs/f3-unsloth"
DATA="$HOME/f3-data/qwen2.5-0.5b"

command -v gcc >/dev/null || { apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential; }
command -v uv >/dev/null || pip install -q uv
mkdir -p "$HOME/.venvs"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --no-deps -r "$ROOT/benchmarks/harness/f3_locks/unsloth.lock.txt"

export PYTHONPATH="$ROOT/src"
"$VENV/bin/python" "$ROOT/benchmarks/harness/f3_setup.py" verify "$VENV"
"$VENV/bin/python" "$ROOT/benchmarks/harness/f3_setup.py" data \
  --tokenizer Qwen/Qwen2.5-0.5B --out "$DATA" --max-length 512

ACTUAL=$("$VENV/bin/python" -c "import json,sys;print(json.load(open('$DATA/record.json'))['sha256'])")
if [ "$ACTUAL" != "$EXPECTED_SHA" ]; then
  echo "dataset sha256 differs: $ACTUAL" >&2
  exit 1
fi
echo "bootstrap ok, dataset sha256 matches"
