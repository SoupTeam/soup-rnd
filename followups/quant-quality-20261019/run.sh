#!/bin/bash
set -euo pipefail
cd /workspace/d1probe
mkdir -p source results
finish() { echo "${?}" > results/exit-code.txt; touch done; }
trap finish EXIT
echo 'aW1wb3J0IGpzb24sb3MscGF0aGxpYixzaWduYWwsc3lzLHRpbWUKcGFyZW50PWludChzeXMuYXJndlsxXSk7IGxpbWl0PTI0XzAwMF8wMDBfMDAwCndoaWxlIG5vdCBwYXRobGliLlBhdGgoJ2RvbmUnKS5leGlzdHMoKToKIHBhdGhzPVtwIGZvciBwIGluIHBhdGhsaWIuUGF0aCgnL3N5cy9jbGFzcy9uZXQnKS5nbG9iKCcqL3N0YXRpc3RpY3MvKl9ieXRlcycpIGlmIHAubmFtZSBpbiAoJ3J4X2J5dGVzJywndHhfYnl0ZXMnKSBhbmQgcC5wYXJlbnQucGFyZW50Lm5hbWUhPSdsbyddCiB0b3RhbD1zdW0oaW50KHAucmVhZF90ZXh0KCkpIGZvciBwIGluIHBhdGhzKQogcGF0aGxpYi5QYXRoKCdyZXN1bHRzL25ldHdvcmstZ3VhcmQuanNvbicpLndyaXRlX3RleHQoanNvbi5kdW1wcyh7J2J5dGVzJzp0b3RhbCwnbGltaXQnOmxpbWl0LCd0aW1lJzp0aW1lLnRpbWUoKX0pKQogaWYgdG90YWw+PWxpbWl0OgogIGNoaWxkcmVuPXt9CiAgZm9yIHBhdGggaW4gcGF0aGxpYi5QYXRoKCcvcHJvYycpLmdsb2IoJ1swLTldKi9zdGF0Jyk6CiAgIHRyeTogcHBpZD1pbnQocGF0aC5yZWFkX3RleHQoKS5yc3BsaXQoJyknLDEpWzFdLnNwbGl0KClbMV0pOyBjaGlsZHJlbi5zZXRkZWZhdWx0KHBwaWQsW10pLmFwcGVuZChpbnQocGF0aC5wYXJlbnQubmFtZSkpCiAgIGV4Y2VwdCAoT1NFcnJvcixWYWx1ZUVycm9yKTogcGFzcwogIGRlZiBzdG9wKHBpZCk6CiAgIGZvciBjaGlsZCBpbiBjaGlsZHJlbi5nZXQocGlkLFtdKToKICAgIGlmIGNoaWxkPT1vcy5nZXRwaWQoKTogY29udGludWUKICAgIHN0b3AoY2hpbGQpCiAgICB0cnk6IG9zLmtpbGwoY2hpbGQsc2lnbmFsLlNJR1RFUk0pCiAgICBleGNlcHQgUHJvY2Vzc0xvb2t1cEVycm9yOiBwYXNzCiAgc3RvcChwYXJlbnQpCiAgcGF0aGxpYi5QYXRoKCdyZXN1bHRzL25ldHdvcmstYnVkZ2V0LXN0b3AuanNvbicpLndyaXRlX3RleHQoanNvbi5kdW1wcyh7J2J5dGVzJzp0b3RhbCwnbGltaXQnOmxpbWl0fSkpCiAgYnJlYWsKIHRpbWUuc2xlZXAoMikK' | base64 -d > network_guard.py
python network_guard.py $$ > results/network-guard.log 2>&1 &
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv,noheader > results/gpu.txt
test "$(wc -l < results/gpu.txt)" -eq 1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/d1probe/hfcache
tar -xf source.tar -C source
BASE_PY=''
for candidate in /opt/conda/bin/python /opt/venv/bin/python /usr/local/bin/python /usr/bin/python; do
 if test -x "$candidate" && "$candidate" -c 'import torch; assert torch.cuda.is_available()' > results/python-probe.txt 2>&1; then BASE_PY="$candidate"; break; fi
done
test -n "$BASE_PY"
"$BASE_PY" -m venv --system-site-packages /workspace/d1probe/.venv
export PATH=/workspace/d1probe/.venv/bin:$PATH
python -c 'import sys,torch; print(sys.executable,torch.__version__,torch.cuda.is_available())' > results/python-ready.txt
python -m pip install --disable-pip-version-check transformers==5.17.0 bitsandbytes==0.50.2 accelerate==1.13.0 pyarrow rich > results/install.log 2>&1
python -m pip freeze > results/packages.txt
sha256sum source.tar source/benchmarks/harness/d1_quant_quality.py source/benchmarks/d1-quant-quality-rule-20261019.md > results/source-hashes.txt
python source/benchmarks/harness/d1_quant_quality.py run --output results/quality > results/quality.log 2>&1
