#!/usr/bin/env bash
# Install and check A0 only. This does not download models, train, or rent a server.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${A0_PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), sys.version'
if [ ! -f .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m pip install --no-cache-dir 'torch==2.14.1' \
  --index-url https://download.pytorch.org/whl/cu130
"$PYTHON" -m pip install --no-cache-dir -r benchmarks/a0-requirements.txt
export PYTHONPATH="$PWD/src"
mkdir -p .soup/bootstrap
"$PYTHON" -m pip freeze > .soup/bootstrap/packages.txt
"$PYTHON" -m pytest tests/test_l2l_probe_harness.py tests/test_l2l_session.py \
  -q -o addopts='' --basetemp "$PWD/.soup/bootstrap/test-tmp"
"$PYTHON" -m ruff check benchmarks/harness/l2l_box.py benchmarks/harness/l2l_probe.py \
  benchmarks/harness/l2l_rule.py benchmarks/harness/l2l_session.py \
  benchmarks/harness/l2l_prepare.py tests/test_l2l_session.py
