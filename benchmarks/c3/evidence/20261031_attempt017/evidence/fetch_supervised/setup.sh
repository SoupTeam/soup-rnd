#!/bin/bash
set -euo pipefail
cd /data/c3/repo
export PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
exec python3 -m benchmarks.c3.fetch_artifacts --manifest benchmarks/c3/offline_artifacts.json --directory /data/c3/offline --receipt /data/c3/evidence/artifact_fetch.json --deadline-unix-s 1793426161.1406274
