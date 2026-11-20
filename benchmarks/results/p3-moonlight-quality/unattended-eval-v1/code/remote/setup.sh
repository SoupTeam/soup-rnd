#!/usr/bin/env bash
# Same pinned environment as the qualified first host; review-source = packet + 3 overlays.
# The driver puts /opt/moonlight-py312/bin first on PATH, so every call here names the SYSTEM
# interpreter explicitly: once the venv exists, a bare `python3` would be the venv's (no uv).
set -euo pipefail
umask 077
export DEBIAN_FRONTEND=noninteractive
SYS=/usr/bin/python3
UV="$SYS -m uv"
ROOT=/data/p3-moonlight-quality
AUTO=$ROOT/autorun
printf '%s  %s\n' d0ff819e27bdcea710b0cdea6a4e16f3fb8121e3100d7271ecc391ce4013a1df "$AUTO/inputs/moonlight-source.zip" | sha256sum -c -
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$AUTO/logs/gpu.txt"
grep -q "RTX 4090" "$AUTO/logs/gpu.txt"
apt-get update
apt-get install -y python3 python3-pip python3-venv ca-certificates
$SYS -m pip install --break-system-packages uv==0.13.7
rm -rf "$ROOT/source" "$ROOT/review-source"
mkdir -p "$ROOT/source"
$SYS -m zipfile -e "$AUTO/inputs/moonlight-source.zip" "$ROOT/source"
if [ ! -x /opt/moonlight-py312/bin/python ]; then
  $UV venv --python 3.12.3 --seed /opt/moonlight-py312
fi
$UV pip install --python /opt/moonlight-py312/bin/python torch==2.11.0 transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 bitsandbytes==0.50.2 safetensors==0.8.0 rich==13.9.4 pydantic==2.14.1 numpy==2.1.3 ruff==0.17.4 pytest==8.4.2 psutil==7.2.2 pyyaml==6.0.3 trl==0.29.1 datasets==5.1.0 requests==2.32.5 tiktoken==0.14.0 tokenizers==0.23.3 regex==2026.11.9 blobfile==3.1.0 huggingface-hub==2.4.0 scipy==1.15.3 -e "$ROOT/source"
/opt/moonlight-py312/bin/python --version
$UV pip freeze --python /opt/moonlight-py312/bin/python > "$AUTO/logs/pip-freeze.txt"
mkdir -p "$ROOT/tokenizer" "$ROOT/sources" "$ROOT/cache" "$ROOT/exports"
rm -rf "$ROOT/pack" "$ROOT/tokenizer/476b36a473d4467f94469414bef6cee75c9c8172"
mv "$ROOT/source/operator-data/pack" "$ROOT/pack"
mv "$ROOT/source/operator-data/tokenizer/476b36a473d4467f94469414bef6cee75c9c8172" "$ROOT/tokenizer/"
cp -a "$ROOT/source" "$ROOT/review-source"
cp "$AUTO/inputs/overlay/"*.py "$ROOT/review-source/benchmarks/harness/"
printf '%s\n' SETUP_COMPLETE
