#!/bin/bash
set -euo pipefail
cd /data/c3/repo
export PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
exec /data/c3/env/bin/python -m benchmarks.c3.representative_workload --snapshot /data/c3/model --output /data/c3/repo/.c3_prepared --download-deadline-unix-s 1793426820.2809136
