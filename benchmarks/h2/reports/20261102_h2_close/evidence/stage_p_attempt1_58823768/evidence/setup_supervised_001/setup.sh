#!/bin/bash
set -euo pipefail
umask 077
mkdir -p /data/h2/evidence
export DEBIAN_FRONTEND=noninteractive
export UV_HTTP_RETRIES=0 UV_HTTP_TIMEOUT=60 UV_CONCURRENT_DOWNLOADS=4
export PIP_RETRIES=0 PIP_DISABLE_PIP_VERSION_CHECK=1
uname -a
cat /proc/sys/kernel/random/boot_id
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv
export H2_BOOT_ID=$(cat /proc/sys/kernel/random/boot_id)
timeout -k 10s 300s apt-get update
timeout -k 10s 300s apt-get install -y python3-venv python3-pip git ca-certificates curl
python3 -m venv /data/h2/tools
timeout -k 10s 120s /data/h2/tools/bin/pip install --no-cache-dir uv==0.13.5
/data/h2/tools/bin/uv python install 3.11
/data/h2/tools/bin/uv venv --python 3.11 /data/h2/env
COMMIT=fc2880f914d0007f72c234b5b53d39f148d90526
timeout -k 10s 180s git clone --depth 1 --branch research/h2-kernel-comparison https://github.com/SoupTeam/soup-rnd.git /data/h2/repo
if [ "$(git -C /data/h2/repo rev-parse HEAD)" != "$COMMIT" ]; then
  # The branch tip moved past the frozen source: fetch exactly the pinned commit (depth 1 is enough for git show).
  timeout -k 10s 180s git -C /data/h2/repo fetch --depth 1 origin "$COMMIT"
  git -C /data/h2/repo checkout --detach "$COMMIT"
fi
test "$(git -C /data/h2/repo rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C /data/h2/repo status --porcelain --untracked-files=all)"
/data/h2/tools/bin/uv pip sync --python /data/h2/env/bin/python --require-hashes --no-cache /data/h2/repo/benchmarks/h2/runtime_candidate.lock
/data/h2/env/bin/python -m pip --version || true
/data/h2/tools/bin/uv pip check --python /data/h2/env/bin/python
/data/h2/env/bin/python -c 'import torch,importlib.metadata,json; print(json.dumps({"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),"packages":{x:importlib.metadata.version(x) for x in ("transformers","trl","peft","bitsandbytes","accelerate","datasets","liger-kernel","unsloth","triton")}},indent=2))'
date -u
